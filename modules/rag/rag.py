"""
modules/rag.py — Consolidated Retrieval-Augmented Generation (RAG) Module (v3)
─────────────────────────────────────────────────────────────────────────────
FIXED: Removed broken Ollama/Anthropic engines and integrated free-tier Gemini API.
FIXED: Patched shape index iteration syntax breaks to avoid endpoint 500 errors.
ADDED: Public domain web scraper module inside RAGPipeline to auto-harvest medical guides.
"""

import csv
import hashlib
import io
import json
import logging
import os
import sys
import threading
import uuid
import re
import urllib.request
import ssl
import datetime as _dt

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from google import genai
from google.genai import types


logger = logging.getLogger("mindcare.rag.combined")


# ─── 1. CONFIGURATION SYSTEM ──────────────────────────────────────────────────

VECTOR_STORE_DIR = os.environ.get(
    "MINDCARE_VECTOR_STORE_DIR",
    "./vector_store"
)

UPLOAD_DIR = os.environ.get(
    "MINDCARE_UPLOAD_DIR",
    "./uploads"
)

RAG_EMBEDDING_MODEL = os.environ.get(
    "MINDCARE_RAG_EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5"
)

EMBEDDING_DIM = 384

HF_CACHE_DIR = os.environ.get(
    "MINDCARE_HF_CACHE",
    "./hf_cache"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75

DEFAULT_TOP_K = 4
MIN_RELEVANCE_SCORE = 0.20

NOT_FOUND_MESSAGE = (
    "Information not found in the provided reference documents."
)

SUPPORTED_DOC_FORMATS = {
    ".pdf",
    ".txt",
    ".docx",
    ".csv"
}

MAX_DOC_SIZE_BYTES = 25 * 1024 * 1024

DIAGNOSTIC_DISCLAIMER = (
    "This answer is generated from the uploaded documents by an AI system "
    "and is NOT a medical diagnosis or clinical judgment. Verify any "
    "clinically important information against original source documents."
)


# ─── 2. DATA OBJECTS ──────────────────────────────────────────────────────────

@dataclass
class DocSection:
    text: str
    page: Optional[int] = None
    section_label: Optional[str] = None


@dataclass
class Chunk:
    text: str
    page: Optional[int]
    section_label: Optional[str]
    chunk_index: int


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    document_name: str
    page: Optional[int]
    section_label: Optional[str]
    text: str
    upload_date: str
    source_metadata: dict


# ─── 3. INGESTION ENGINE (Txt & Base text fallbacks only) ─────────────────────

def _extract_txt(raw: bytes, filename: str) -> List[DocSection]:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(
            f"'{filename}' could not be decoded as text."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            f"'{filename}' is empty."
        )

    return [DocSection(text=text, page=None)]
def _extract_pdf(raw: bytes, filename: str) -> List[DocSection]:
    """Extract text per page. Returns one DocSection per page."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(raw))
    sections = []
    for i, page in enumerate(reader.pages):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as e:
            logger.warning(f"Page {i+1} of {filename} failed to extract: {e}")
            continue
        if text:
            sections.append(DocSection(text=text, page=i + 1))
    if not sections:
        raise ValueError(
            f"'{filename}' produced no extractable text. "
            "If it's a scanned PDF, you need OCR (pytesseract + pdf2image)."
        )
    return sections


def _extract_docx(raw: bytes, filename: str) -> List[DocSection]:
    """Extract paragraphs from a .docx as a single section."""
    from docx import Document
    doc = Document(io.BytesIO(raw))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)
    if not text:
        raise ValueError(f"'{filename}' produced no extractable text.")
    return [DocSection(text=text, page=None)]


def extract_sections(raw: bytes, filename: str) -> List[DocSection]:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return _extract_pdf(raw, filename)
    if ext == ".docx":
        return _extract_docx(raw, filename)
    # txt, md, csv, html, or anything else → text fallback
    return _extract_txt(raw, filename)

def chunk_sections(
    sections: List[DocSection]
) -> List[Chunk]:

    chunks: List[Chunk] = []
    idx = 0

    for section in sections:
        text = section.text
        start = 0
        n = len(text)

        while start < n:
            end = min(
                start + CHUNK_SIZE,
                n
            )

            piece = text[start:end].strip()

            if piece:
                chunks.append(
                    Chunk(
                        text=piece,
                        page=section.page,
                        section_label=section.section_label,
                        chunk_index=idx
                    )
                )

                idx += 1

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ─── 4. EMBEDDINGS SYSTEM ─────────────────────────────────────────────────────

_embedding_model_instance = None
_embedding_lock = threading.Lock()


def get_embedding_model():
    global _embedding_model_instance

    if _embedding_model_instance is not None:
        return _embedding_model_instance

    with _embedding_lock:
        if _embedding_model_instance is not None:
            return _embedding_model_instance

        from sentence_transformers import SentenceTransformer

        _embedding_model_instance = SentenceTransformer(
            RAG_EMBEDDING_MODEL,
            cache_folder=HF_CACHE_DIR,
            device="cpu"
        )

        return _embedding_model_instance


def embed_texts(
    texts: List[str]
) -> np.ndarray:

    if not texts:
        return np.zeros(
            (0, EMBEDDING_DIM),
            dtype=np.float32
        )

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.astype(np.float32)


# ─── 5. VECTOR STORE ENGINE (Lightweight NumPy Memory Array Flat Index) ───────

class VectorStore:

    def __init__(
        self,
        store_dir: str = VECTOR_STORE_DIR
    ):
        self.store_dir = store_dir
        self._lock = threading.RLock()

        os.makedirs(
            self.store_dir,
            exist_ok=True
        )

        self._meta_path = os.path.join(
            self.store_dir,
            "chunks_metadata.json"
        )

        self._docs_path = os.path.join(
            self.store_dir,
            "documents.json"
        )

        self._vectors_path = os.path.join(
            self.store_dir,
            "vectors.npy"
        )

        self._metadata: Dict[str, dict] = {}
        self._documents: Dict[str, dict] = {}
        self._vectors: List[np.ndarray] = []
        self._vector_ids: List[str] = []

        self._load_from_disk()

    def _load_from_disk(self):

        if os.path.exists(self._meta_path):
            with open(
                self._meta_path,
                "r",
                encoding="utf-8"
            ) as f:
                self._metadata = json.load(f)

        if os.path.exists(self._docs_path):
            with open(
                self._docs_path,
                "r",
                encoding="utf-8"
            ) as f:
                self._documents = json.load(f)

        if os.path.exists(self._vectors_path):
            try:
                loaded = np.load(
                    self._vectors_path,
                    allow_pickle=True
                ).item()

                self._vectors = loaded.get(
                    "vectors",
                    []
                )

                self._vector_ids = loaded.get(
                    "ids",
                    []
                )

            except Exception:
                self._vectors = []
                self._vector_ids = []

    def _persist(self):

        with open(
            self._meta_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                self._metadata,
                f,
                indent=2
            )

        with open(
            self._docs_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                self._documents,
                f,
                indent=2
            )

        np.save(
            self._vectors_path,
            {
                "vectors": self._vectors,
                "ids": self._vector_ids
            }
        )

    def find_document_by_hash(
        self,
        file_hash: str
    ) -> Optional[str]:

        for doc_id, info in self._documents.items():
            if info.get("file_hash") == file_hash:
                return doc_id

        return None

    def register_document(
        self,
        document_id: str,
        name: str,
        file_hash: str,
        upload_date: str,
        doc_format: str
    ):

        self._documents[document_id] = {
            "name": name,
            "file_hash": file_hash,
            "upload_date": upload_date,
            "format": doc_format,
            "num_chunks": 0
        }

    def add_chunks(
        self,
        document_id: str,
        records: List[ChunkRecord],
        embeddings: np.ndarray
    ):

        with self._lock:

            for i, rec in enumerate(records):
                c_id = rec.chunk_id

                self._metadata[c_id] = asdict(rec)

                self._vectors.append(
                    embeddings[i]
                )

                self._vector_ids.append(
                    c_id
                )

            if document_id in self._documents:
                self._documents[document_id]["num_chunks"] += len(
                    records
                )

            self._persist()

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 4,
        document_ids: Optional[List[str]] = None
    ) -> List[Tuple[float, dict]]:

        with self._lock:

            if not self._vectors:
                return []

            matrix = np.array(
                self._vectors,
                dtype=np.float32
            )

            # Compute cosine similarity using standard dot products
            scores = np.dot(
                matrix,
                query_vec.astype(np.float32)
            )

            ranked_indices = np.argsort(scores)[::-1]

            results = []

            for idx in ranked_indices:

                c_id = self._vector_ids[idx]
                meta = self._metadata.get(c_id)

                if not meta:
                    continue

                if (
                    document_ids
                    and meta["document_id"] not in document_ids
                ):
                    continue

                results.append(
                    (
                        float(scores[idx]),
                        meta
                    )
                )

                if len(results) >= top_k:
                    break

            return results

    def delete_all_documents(self):

        with self._lock:
            self._metadata = {}
            self._documents = {}
            self._vectors = []
            self._vector_ids = []

            self._persist()


# ─── 6. GEMINI GENERATION BACKEND INTERFACE ───────────────────────────────────

def generate_gemini_answer(
    question: str,
    passages: List[dict]
) -> Dict:

    if not passages:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "sources": [],
            "grounded": False,
            "disclaimer": DIAGNOSTIC_DISCLAIMER
        }

    context_lines = []
    for i, pas in enumerate(passages, start=1):
        context_lines.append(
            f"[Source {i}] (Document: {pas['document_name']}):\n{pas['text']}"
        )
    context_block = "\n\n".join(context_lines)

    prompt = (
        "You are a clinical mental health knowledge assistant. "
        "Answer the user's question accurately using strictly ONLY the "
        "reference document facts supplied below. If the references do not "
        "contain the answer, say so plainly.\n\n"
        f"FACTUAL CONTEXT CHUNKS:\n{context_block}\n\n"
        f"USER QUESTION: {question}\n"
        "ASSISTANT:"
    )

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return {
            "answer": (
                "[RAG offline mode] Found matches, but GEMINI_API_KEY is "
                "unconfigured. Top match:\n"
                f"{passages[0]['text'][:600]}"
            ),
            "sources": passages,
            "grounded": True,
            "disclaimer": DIAGNOSTIC_DISCLAIMER,
        }

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You answer only from the reference chunks provided. "
                    "If the chunks don't contain the answer, say so plainly. "
                    "Cite source numbers in square brackets."
                ),
            ),
        )
        answer_text = (response.text or "").strip()
        if not answer_text:
            answer_text = NOT_FOUND_MESSAGE

        return {
            "answer": answer_text,
            "grounded": True,
            "disclaimer": DIAGNOSTIC_DISCLAIMER,
            "sources": passages,
        }

    except Exception as e:
        logger.error("RAG generation failed: %s", e, exc_info=True)
        return {
            "answer": (
                "Retrieved matching documents, but the generation backend "
                f"failed: {e}"
            ),
            "sources": passages,
            "grounded": False,
            "disclaimer": DIAGNOSTIC_DISCLAIMER,
        }
# ─── 7. UNIFIED RAG PUBLIC INTERFACE WITH AUTONOMOUS SCRAPER ──────────────────

class RAGPipeline:

    def __init__(
        self,
        store_dir: str = VECTOR_STORE_DIR
    ):
        self.store = VectorStore(
            store_dir=store_dir
        )

    def upload_document(
        self,
        raw_bytes: bytes,
        filename: str,
        source_metadata: Optional[dict] = None
    ) -> Dict:

        file_hash = hashlib.sha256(
            raw_bytes
        ).hexdigest()

        existing_id = self.store.find_document_by_hash(
            file_hash
        )

        if existing_id:
            return {
                "status": "success",
                "document_id": existing_id,
                "message": "Identical document hash registry found."
            }

        try:
            sections = extract_sections(
                raw_bytes,
                filename
            )

            chunks = chunk_sections(
                sections
            )

            if not chunks:
                return {
                    "status": "error",
                    "error": "Zero chunks produced."
                }

            document_id = str(
                uuid.uuid4()
            )

            upload_date = _dt.datetime.now(
                _dt.timezone.utc
            ).isoformat()

            doc_format = filename.rsplit(
                ".",
                1
            )[-1].lower()

            embeddings = embed_texts(
                [chunk.text for chunk in chunks]
            )

            records = [
                ChunkRecord(
                    chunk_id=(
                        f"{document_id}_chunk"
                        f"{chunk.chunk_index}"
                    ),
                    document_id=document_id,
                    document_name=filename,
                    page=chunk.page,
                    section_label=chunk.section_label,
                    text=chunk.text,
                    upload_date=upload_date,
                    source_metadata=source_metadata or {}
                )
                for chunk in chunks
            ]

            self.store.register_document(
                document_id,
                filename,
                file_hash,
                upload_date,
                doc_format
            )

            self.store.add_chunks(
                document_id,
                records,
                embeddings
            )

            return {
                "status": "success",
                "document_id": document_id,
                "filename": filename,
                "num_chunks": len(records)
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def query(
        self,
        question: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = DEFAULT_TOP_K
    ) -> Dict:

        query_vec = embed_texts(
            [question]
        )[0]

        raw_res = self.store.search(
            query_vec,
            top_k=top_k,
            document_ids=document_ids
        )

        passages = []

        for score, meta in raw_res:

            if score < MIN_RELEVANCE_SCORE:
                continue

            passages.append(
                {
                    "score": score,
                    "text": meta["text"],
                    "document_id": meta["document_id"],
                    "document_name": meta["document_name"],
                    "chunk_id": meta["chunk_id"]
                }
            )

        return generate_gemini_answer(
            question,
            passages
        )

    def harvest_and_store_mental_health_docs(
        self
    ) -> Dict:
        """
        AUTONOMOUS WEB HARVESTER: Automatically crawls open-source clinical
        FAQ datasets regarding mental health disorders & medications to
        populate the RAG matrix.
        """

        logger.info(
            "[MindCare RAG] Running autonomous mental health text "
            "harvester node..."
        )

        target_sources = {
            "disorders_guide.txt": "githubusercontent.com",
            "medication_reference.txt": "waml.org"
        }

        ctx = ssl._create_unverified_context()
        harvested_count = 0

        for name, url in target_sources.items():

            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0"
                    }
                )

                with urllib.request.urlopen(
                    req,
                    context=ctx,
                    timeout=12
                ) as response:

                    raw_text = response.read()

                    res = self.upload_document(
                        raw_text,
                        name
                    )

                    if res.get("status") == "success":
                        harvested_count += 1

            except Exception as e:
                logger.warning(
                    f"Harvester skipped mirror node link "
                    f"{url}: {str(e)}"
                )

        return {
            "status": "success",
            "message": (
                "Harvesting completed loop. Synergized "
                f"{harvested_count} macro files into vector arrays."
            )
        }

        # In modules/rag.py — add this method to RAGPipeline class

def retrieve(
    self,
    question: str,
    document_ids: Optional[List[str]] = None,
    top_k: int = DEFAULT_TOP_K,
) -> List[dict]:
    """
    Pure retrieval — returns chunks with scores and metadata.
    Does NOT call any LLM. Works when Gemini is down.
    """
    query_vec = embed_texts([question])[0]
    raw_res = self.store.search(query_vec, top_k=top_k, document_ids=document_ids)

    passages = []
    for score, meta in raw_res:
        if score < MIN_RELEVANCE_SCORE:
            continue
        passages.append({
            "score": score,
            "text": meta["text"],
            "document_id": meta["document_id"],
            "document_name": meta["document_name"],
            "chunk_id": meta["chunk_id"],
            "page": meta.get("page"),
        })
    return passages