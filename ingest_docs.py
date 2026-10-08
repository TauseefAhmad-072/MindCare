"""
ingest_docs.py — Bulk upload mental-health documents into the RAG store.
Usage:
    python ingest_docs.py C:\path\to\docs
"""
import os
import sys
import time
import requests

RAG_URL = "http://127.0.0.1:5000/api/rag/upload"
SUPPORTED = {".pdf", ".txt", ".md", ".docx", ".csv"}
REQUEST_TIMEOUT = 120   # long because embedding is CPU-bound


def ingest_folder(folder: str):
    if not os.path.isdir(folder):
        print(f"Not a folder: {folder}")
        return

    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            if os.path.splitext(n)[1].lower() in SUPPORTED:
                files.append(os.path.join(root, n))

    print(f"Found {len(files)} documents under {folder}\n")

    ok, failed, skipped = 0, 0, 0
    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        try:
            with open(path, "rb") as f:
                r = requests.post(
                    RAG_URL,
                    files={"file": (name, f)},
                    timeout=REQUEST_TIMEOUT,
                )
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}

            if r.status_code == 200 and data.get("status") == "success":
                if "Identical document hash" in data.get("message", ""):
                    skipped += 1
                    print(f"  [{i}/{len(files)}] SKIP  {name} (already ingested)")
                else:
                    ok += 1
                    print(f"  [{i}/{len(files)}] OK    {name} → {data.get('num_chunks', '?')} chunks")
            else:
                failed += 1
                print(f"  [{i}/{len(files)}] FAIL  {name} → {r.status_code} {data.get('error', r.text[:120])}")
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(files)}] ERR   {name} → {e}")

        time.sleep(0.1)   # be gentle on the embedding model

    print(f"\nDone. {ok} new, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "./docs"
    ingest_folder(folder)