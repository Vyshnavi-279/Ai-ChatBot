# rag_core/__init__.py
# Exposes the public interface of the RAG core package.
# Imports are lazy so that importing rag_core.config alone does not
# require chromadb or langchain-openai to be installed.

def retrieve(query, section_filter=None, top_k=None):
    """Lazy wrapper — delegates to retriever.retrieve()."""
    from rag_core.retriever import retrieve as _retrieve
    kwargs = {"section_filter": section_filter}
    if top_k is not None:
        kwargs["top_k"] = top_k
    return _retrieve(query, **kwargs)


def generate(query, chunks, history=None):
    """Lazy wrapper — delegates to generation.generate()."""
    from rag_core.generation import generate as _generate
    return _generate(query, chunks, history=history)


__all__ = ["retrieve", "generate"]
