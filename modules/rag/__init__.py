"""
MODULE D — RAG (Retrieval-Augmented Generation) over patient reports.

Public integration contract:

    from modules.rag import RAGPipeline
    rag = RAGPipeline()
    doc = rag.upload_document(file_bytes, "patient_report.pdf")
    answer = rag.query("What medications were mentioned?",
                        document_ids=[doc["document_id"]])

Or use the process-wide default instance:

    from modules.rag import get_default_pipeline
    rag = get_default_pipeline()

NOTE ON LAZY LOADING: `modules.rag.config`, `.ingestion`, and `.chunking`
have zero heavy ML dependencies and stay importable/testable in isolation.
`RAGPipeline` itself is imported lazily here; only calling methods that
need embeddings/FAISS/the Anthropic API triggers those imports.
"""
__all__ = ["RAGPipeline", "get_default_pipeline"]

_LAZY_ATTRS = {
    "RAGPipeline": (".pipeline", "RAGPipeline"),
    "get_default_pipeline": (".pipeline", "get_default_pipeline"),
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        import importlib
        module_path, attr_name = _LAZY_ATTRS[name]
        mod = importlib.import_module(module_path, __name__)
        value = getattr(mod, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
