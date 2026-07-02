# rag_core/retriever.py
# Retrieves the top-k most relevant chunks from ChromaDB for a given query.
# Supports an optional section filter via ChromaDB metadata filtering.

from __future__ import annotations

from typing import Optional

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings

from rag_core.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    TOP_K,
)


# ---------------------------------------------------------------------------
# Lazy singletons — initialised once per process to avoid repeated I/O.
# ---------------------------------------------------------------------------
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None
_embedder: Optional[OpenAIEmbeddings] = None


def _get_embedder() -> OpenAIEmbeddings:
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL,
        )
    return _embedder


def _get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    section_filter: Optional[str] = None,
    top_k: int = TOP_K,
) -> list[dict]:
    """Return the top-k most relevant chunks for *query*.

    Parameters
    ----------
    query:
        The user's question or search string.
    section_filter:
        If provided, restricts results to chunks whose ``section`` metadata
        field equals this value (must be one of the 8 canonical headings).
        Pass ``None`` or ``"All"`` to search the entire knowledge base.
    top_k:
        Number of results to return (defaults to ``config.TOP_K``).

    Returns
    -------
    list[dict]
        Each element has keys:
        ``text``     — the chunk content
        ``metadata`` — dict with ``source``, ``section``, ``page``
        ``score``    — cosine similarity distance (lower = more similar)
    """
    collection = _get_collection()
    embedder = _get_embedder()

    query_embedding: list[float] = embedder.embed_query(query)

    # Build optional where-clause for metadata filtering.
    where: Optional[dict] = None
    if section_filter and section_filter.lower() != "all":
        where = {"section": {"$eq": section_filter}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[dict] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        chunks.append(
            {
                "text": doc,
                "metadata": meta,
                "score": dist,
            }
        )

    return chunks


# ---------------------------------------------------------------------------
# __main__ — quick smoke-test for retrieval (3 sample BVRIT queries)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    SAMPLE_QUERIES = [
        "When was BVRIT Hyderabad established?",
        "What undergraduate programmes does BVRIT offer?",
        "Who are the placement partners at BVRIT?",
    ]

    print("=" * 60)
    print("Retriever smoke-test — 3 sample queries")
    print("=" * 60)

    for q in SAMPLE_QUERIES:
        print(f"\nQuery: {q!r}")
        try:
            chunks = retrieve(q, top_k=3)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            sys.exit(1)

        if not chunks:
            print("  No chunks returned — is the index populated? Run ingest.py first.")
        else:
            for i, chunk in enumerate(chunks, start=1):
                meta = chunk["metadata"]
                score = chunk["score"]
                preview = chunk["text"][:120].replace("\n", " ")
                print(
                    f"  [{i}] section={meta.get('section')!r}  "
                    f"page≈{meta.get('page')}  score={score:.4f}"
                )
                print(f"       {preview}…")

    print("\nDone.")
