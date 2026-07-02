# ingest.py
# One-time (idempotent) ingestion pipeline:
#   data/college_info.docx → chunks → embeddings → ChromaDB
#
# Running this script a second time will NOT duplicate the index — it
# detects an already-populated collection and exits early.
#
# Usage:
#   python ingest.py

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config must be imported first so .env is loaded before any other module
# tries to read env vars.
# ---------------------------------------------------------------------------
from rag_core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    DOCX_PATH,
    EMBEDDING_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    SECTION_HEADINGS,
)

import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Section-heading detection helpers
# ---------------------------------------------------------------------------

def _detect_section(text: str, current_section: str) -> str:
    """Return the section heading that *text* belongs to.

    Scans the text for the presence of one of the 8 canonical headings and
    returns it.  If none matches, the *current_section* is carried forward
    (chunks inherit the last-seen heading).
    """
    for heading in SECTION_HEADINGS:
        if heading.lower() in text.lower():
            return heading
    return current_section


def _approximate_page(char_offset: int, chars_per_page: int = 3000) -> int:
    """Rough page estimate based on character offset within the document."""
    return max(1, (char_offset // chars_per_page) + 1)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def load_and_chunk(docx_path: str) -> list[dict]:
    """Load the docx, split into chunks, attach metadata.

    Returns a list of dicts:
        text     — chunk text
        metadata — {source, section, page}
        id       — stable SHA-256 content hash (for idempotent upsert)
    """
    path = Path(docx_path)
    if not path.exists():
        print(f"[ingest] ERROR: document not found at {path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"[ingest] Loading {path} …")
    loader = Docx2txtLoader(str(path))
    raw_docs = loader.load()                 # returns list[Document]
    full_text = "\n".join(d.page_content for d in raw_docs)

    # Primary separators: the 8 section headings; fallback to standard splitter separators.
    heading_separators = [f"\n{h}" for h in SECTION_HEADINGS] + [
        "\n\n", "\n", ". ", " ", ""
    ]

    splitter = RecursiveCharacterTextSplitter(
        separators=heading_separators,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    raw_chunks = splitter.split_text(full_text)
    print(f"[ingest] Split into {len(raw_chunks)} raw chunks.")

    # Assign section and page metadata to each chunk.
    chunks: list[dict] = []
    current_section = "About BVRIT"   # default for content before first heading
    char_offset = 0

    for idx, chunk_text in enumerate(raw_chunks):
        current_section = _detect_section(chunk_text, current_section)
        page = _approximate_page(char_offset)
        char_offset += len(chunk_text)

        # ID = positional index + content hash prefix.
        # Using the index as the primary component guarantees uniqueness even
        # when two chunks happen to share identical text (e.g. repeated nav
        # paragraphs scraped from multiple pages).
        content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]
        chunk_id = f"{idx:05d}-{content_hash}"

        chunks.append(
            {
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": path.name,
                    "section": current_section,
                    "page": page,
                },
            }
        )

    return chunks


# ---------------------------------------------------------------------------
# Embedding + persistence
# ---------------------------------------------------------------------------

def persist_to_chroma(chunks: list[dict]) -> None:
    """Embed chunks and upsert into ChromaDB.

    Uses ``upsert`` (keyed on SHA-256 id) so re-running never duplicates.
    """
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Idempotency check: if the collection already has documents, bail out.
    existing_count = collection.count()
    if existing_count > 0:
        print(
            f"[ingest] Collection '{CHROMA_COLLECTION_NAME}' already contains "
            f"{existing_count} chunks — skipping ingestion to avoid duplicates.\n"
            "         Delete ./chroma_db/ and re-run to rebuild the index from scratch."
        )
        return

    print(f"[ingest] Embedding {len(chunks)} chunks via {EMBEDDING_MODEL} …")

    embedder = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
    )

    # Embed in batches of 64 to stay within rate limits.
    BATCH = 64
    all_ids: list[str] = []
    all_texts: list[str] = []
    all_embeddings: list[list[float]] = []
    all_metadatas: list[dict] = []

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i : i + BATCH]
        texts = [c["text"] for c in batch]
        batch_embeddings = embedder.embed_documents(texts)

        all_ids.extend(c["id"] for c in batch)
        all_texts.extend(texts)
        all_embeddings.extend(batch_embeddings)
        all_metadatas.extend(c["metadata"] for c in batch)

        print(f"[ingest]   embedded {min(i + BATCH, len(chunks))}/{len(chunks)} chunks …")

    collection.upsert(
        ids=all_ids,
        documents=all_texts,
        embeddings=all_embeddings,
        metadatas=all_metadatas,
    )

    final_count = collection.count()
    print(f"\n[ingest] ✓ Ingestion complete. Total chunks in index: {final_count}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    chunks = load_and_chunk(DOCX_PATH)
    persist_to_chroma(chunks)
