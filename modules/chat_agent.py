"""
modules/chat_agent.py — Document-first chat agent with Gemini and Groq fallbacks.

Answer priority:
  1. RAG retrieve (pure embedding search — no LLM, works offline)
  2. Gemini (using retrieved chunks as context, or standalone)
  3. Groq (using retrieved chunks as context, or standalone)
  4. Friendly error

Environment:
  GEMINI_API_KEY  (required)
  GROQ_API_KEY    (optional)
"""

import os
import logging
import re
from typing import Optional, Dict, Any, Tuple, List

from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError, APIError

logger = logging.getLogger("mindcare.chat_agent")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

GEMINI_MODEL_CHAIN = [
    "gemini-3.1-flash-lite",   # cheapest / fastest tier
    "gemini-3.8-flash",        # general-purpose fallback
    "gemini-3.7-flash",        # secondary fallback
]

GROQ_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "You are MindCare, a warm, non-judgmental mental-health companion for "
    "university students in Pakistan. You listen carefully, reflect feelings "
    "back, and offer evidence-based coping strategies (CBT grounding, "
    "breathing, journaling, sleep hygiene). You never diagnose. You never "
    "pretend to replace a licensed clinician. If the user expresses intent "
    "to harm themselves or someone else, gently encourage them to contact "
    "Umang (0317-4288665) or emergency services and stay with them in the "
    "conversation. Keep replies concise — 2 to 4 sentences unless the user "
    "explicitly asks for more.\n\n"
    "LANGUAGE RULES (strict):\n"
    "  * If the user writes in English, reply in English.\n"
    "  * If the user writes in Urdu script (اردو), reply in Urdu script.\n"
    "  * If the user writes in Roman Urdu (e.g. 'mujhe pareshani hai'), "
    "reply in Roman Urdu.\n"
    "  * If the user mixes languages, reply in the dominant language of "
    "their message.\n"
    "  * NEVER reply in Urdu or Roman Urdu if the user wrote in English.\n\n"
    "When reference excerpts are provided, base your answer ONLY on them "
    "and cite them inline as [Source N]."
)
# ──────────────────────────────────────────────────────────────────────────────
# Lazy clients
# ──────────────────────────────────────────────────────────────────────────────

_gemini_client = None
_groq_client = None
_groq_available = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _get_groq_client():
    global _groq_client, _groq_available
    if _groq_available is False:
        return None
    if _groq_client is not None:
        return _groq_client

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        _groq_available = False
        return None

    try:
        import groq  # type: ignore
    except ImportError:
        logger.info("groq package not installed — Groq fallback disabled")
        _groq_available = False
        return None

    try:
        _groq_client = groq.Groq(api_key=api_key)
        _groq_available = True
        logger.info("Groq fallback enabled (model=%s)", GROQ_MODEL)
        return _groq_client
    except Exception as e:
        logger.warning("Groq client init failed: %s", e)
        _groq_available = False
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Session store
# ──────────────────────────────────────────────────────────────────────────────

_gemini_sessions: Dict[str, Any] = {}
_groq_history: Dict[str, list] = {}


def reset_session(session_id: str) -> bool:
    removed = False
    if _gemini_sessions.pop(session_id, None) is not None:
        removed = True
    if _groq_history.pop(session_id, None) is not None:
        removed = True
    return removed


def _get_or_create_gemini_chat(session_id: str, model_name: str):
    existing = _gemini_sessions.get(session_id)
    if existing is not None and getattr(existing, "_mc_model", None) == model_name:
        return existing

    chat = _get_gemini_client().chats.create(
        model=model_name,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    try:
        chat._mc_model = model_name
    except Exception:
        pass
    _gemini_sessions[session_id] = chat
    return chat


# ──────────────────────────────────────────────────────────────────────────────
# RAG retrieval — pure embedding search, no LLM.
# Always returns (context_str, grounded_bool). Never returns None.
# ──────────────────────────────────────────────────────────────────────────────

def _query_rag(rag_pipeline, query: str) -> Tuple[str, bool]:
    """
    Retrieve document chunks for the query.

    Tries, in order:
      1. rag_pipeline.retrieve(query)  — pure retrieval, no LLM
      2. rag_pipeline.query(query)     — falls back, but may call Gemini

    ALWAYS returns a tuple (context_str, grounded_bool).
    Never returns None, never raises.
    """
    if rag_pipeline is None:
        return "", False

    # ── Path A: pure retrieval (preferred) ────────────────────────────────
    retrieve_fn = getattr(rag_pipeline, "retrieve", None)
    if callable(retrieve_fn):
        try:
            passages = retrieve_fn(query)
        except Exception as e:
            logger.warning("RAG.retrieve failed: %s", e)
            passages = None

        if isinstance(passages, list) and passages:
            blocks: List[str] = []
            for i, p in enumerate(passages, 1):
                if not isinstance(p, dict):
                    continue
                name = p.get("document_name", "document")
                page = p.get("page")
                page_str = f", p.{page}" if page else ""
                text = (p.get("text") or "").strip()
                if text:
                    blocks.append(f"[Source {i}] ({name}{page_str}):\n{text}")
            if blocks:
                logger.info("RAG retrieved %d chunks (pure)", len(blocks))
                return "\n\n".join(blocks), True

        if isinstance(passages, list) and not passages:
            # Legitimate "no documents found" — stop here, don't call query()
            logger.info("RAG retrieved 0 chunks — no documents match")
            return "", False

    # ── Path B: full query() — may call Gemini internally ─────────────────
    query_fn = getattr(rag_pipeline, "query", None)
    if not callable(query_fn):
        return "", False

    try:
        result = query_fn(query)
    except Exception as e:
        logger.warning("RAG.query failed: %s", e)
        return "", False

    # Plain string response
    if isinstance(result, str):
        text = result.strip()
        return (text, bool(text))

    if isinstance(result, dict):
        sources = result.get("sources") or []
        grounded_flag = result.get("grounded")

        has_documents = bool(sources) or grounded_flag is True

        if not has_documents:
            answer = result.get("answer") or ""
            if isinstance(answer, str) and answer and "Information not found" not in answer:
                has_documents = True

        if not has_documents:
            return "", False

        answer = result.get("answer") or result.get("context") or ""
        if isinstance(answer, str) and answer.strip():
            return answer.strip(), True

        # No answer string, but sources exist — stitch chunks
        if isinstance(sources, list) and sources:
            parts: List[str] = []
            for i, s in enumerate(sources, 1):
                if not isinstance(s, dict):
                    continue
                name = s.get("document_name", "document")
                text = (s.get("text") or "").strip()
                if text:
                    parts.append(f"[Source {i}] ({name}):\n{text}")
            joined = "\n\n".join(parts).strip()
            if joined:
                return joined, True

    return "", False


# ──────────────────────────────────────────────────────────────────────────────
# Prompt building
# ──────────────────────────────────────────────────────────────────────────────

def _build_prompt(user_message: str, rag_context: str) -> str:
    if not rag_context:
        return user_message
    return (
        "Reference excerpts from the user's uploaded documents:\n"
        "---\n"
        f"{rag_context}\n"
        "---\n\n"
        "Answer the user's question using ONLY these excerpts. "
        "Cite each fact inline as [Source N]. If the excerpts do not "
        "contain the answer, say so plainly instead of guessing.\n\n"
        f"User: {user_message}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Error classification
# ──────────────────────────────────────────────────────────────────────────────

def _is_quota_exhausted(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "resource_exhausted" in msg or "429" in msg:
        return True
    if "quota" in msg or "rate limit" in msg:
        return True
    code = getattr(exc, "code", None)
    if code == 429:
        return True
    return False


def _extract_retry_delay(exc: Exception) -> Optional[float]:
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        retry = details.get("retryDelay") or details.get("retry_delay")
        if isinstance(retry, str) and retry.endswith("s"):
            try:
                return float(retry[:-1])
            except ValueError:
                pass
    m = re.search(r"retry in ([\d.]+)s", str(exc), re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Provider calls
# ──────────────────────────────────────────────────────────────────────────────

def _call_gemini(session_id: str, prompt: str) -> str:
    last_err = None

    for model_name in GEMINI_MODEL_CHAIN:
        try:
            chat = _get_or_create_gemini_chat(session_id, model_name)
            response = chat.send_message(prompt)
            reply = (response.text or "").strip()
            if reply:
                logger.info("Gemini answered via %s", model_name)
                return reply
            logger.warning("Gemini %s returned empty text", model_name)

        except (ServerError, ClientError, APIError) as e:
            if _is_quota_exhausted(e):
                delay = _extract_retry_delay(e)
                logger.warning(
                    "Gemini %s quota exhausted (429), retry after %ss. "
                    "Falling through to Groq.",
                    model_name, delay if delay is not None else "?",
                )
                _gemini_sessions.pop(session_id, None)
                raise

            logger.warning("Gemini %s failed (transient): %s", model_name, e)
            last_err = e
            _gemini_sessions.pop(session_id, None)
            continue

        except Exception as e:
            logger.error("Unexpected error on %s: %s", model_name, e, exc_info=True)
            last_err = e
            _gemini_sessions.pop(session_id, None)
            continue

    if last_err:
        raise last_err
    raise RuntimeError("All Gemini models failed with no captured error")


def _call_groq(session_id: str, prompt: str) -> Optional[str]:
    client = _get_groq_client()
    if client is None:
        return None

    try:
        history = _groq_history.setdefault(session_id, [])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *history[-10:],
            {"role": "user", "content": prompt},
        ]

        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        reply = (completion.choices[0].message.content or "").strip()
        if not reply:
            return None

        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": reply})
        return reply
    except Exception as e:
        logger.error("Groq call failed: %s", e, exc_info=True)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Core turn logic
# ──────────────────────────────────────────────────────────────────────────────

def _run_turn(
    session_id: str,
    user_message: str,
    rag_pipeline: Optional[Any] = None,
) -> Dict[str, Any]:
    if not user_message or not user_message.strip():
        return {
            "status": "error",
            "error": "Empty message.",
            "reply": "I'm here. What's on your mind?",
        }

    # ── Step 1: Retrieve from documents (no LLM) ─────────────────────────
    rag_context, rag_grounded = _query_rag(rag_pipeline, user_message)
    logger.info(
        "RAG: grounded=%s, context_len=%d",
        rag_grounded, len(rag_context) if rag_context else 0,
    )

    prompt = _build_prompt(user_message, rag_context)

    # ── Step 2: Gemini ────────────────────────────────────────────────────
    try:
        reply = _call_gemini(session_id, prompt)
        return {
            "status": "success",
            "session_id": session_id,
            "reply": reply,
            "provider": "gemini",
            "rag_used": rag_grounded,
        }
    except Exception as gemini_err:
        logger.warning("Gemini failed, trying Groq: %s", gemini_err)

    # ── Step 3: Groq ──────────────────────────────────────────────────────
    reply = _call_groq(session_id, prompt)
    if reply:
        return {
            "status": "success",
            "session_id": session_id,
            "reply": reply,
            "provider": "groq",
            "rag_used": rag_grounded,
        }

    # ── Step 4: nothing worked ────────────────────────────────────────────
    # If we have RAG context but no LLM, return the raw chunks so the user
    # at least sees the source material.
    if rag_grounded and rag_context:
        return {
            "status": "success",
            "session_id": session_id,
            "reply": (
                "The AI models are briefly unavailable, so here are the "
                "relevant excerpts from your documents:\n\n" + rag_context
            ),
            "provider": "rag-only",
            "rag_used": True,
        }

    return {
        "status": "error",
        "session_id": session_id,
        "error": "all_providers_unavailable",
        "reply": (
            "All AI providers are currently unavailable. "
            "Please try again in a few minutes."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class MindCareGeminiAgent:
    def __init__(
        self,
        model_name: str = GEMINI_MODEL_CHAIN[0],
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt

    @staticmethod
    def execute_turn(
        session_id: str,
        user_message: str,
        rag_pipeline: Optional[Any] = None,
    ) -> Dict[str, Any]:
        return _run_turn(session_id, user_message, rag_pipeline)

    @staticmethod
    def reset_session(session_id: str) -> bool:
        return reset_session(session_id)


def execute_turn(
    session_id: str,
    user_message: str,
    rag_pipeline: Optional[Any] = None,
) -> Dict[str, Any]:
    return _run_turn(session_id, user_message, rag_pipeline)