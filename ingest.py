# ingest.py
# One-time (idempotent) ingestion pipeline:
#   data/college_info.docx → chunks → embeddings → ChromaDB
#
# Section metadata is assigned by reading the document's Heading 1 styles —
# NOT by scanning body text for keywords, which mis-labels chunks whenever
# a heading word (e.g. "Faculty") appears inside paragraph content.
#
# Running this script a second time will NOT duplicate the index.
#
# Usage:
#   python ingest.py

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Config must be imported first so .env is loaded before anything else.
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
from docx import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Document loading — walk paragraphs, track Heading 1 as section boundary
# ---------------------------------------------------------------------------

def _normalise_heading(text: str) -> str:
    """Return the canonical section name if *text* matches one, else ''."""
    t = text.strip()
    for heading in SECTION_HEADINGS:
        if heading.lower() == t.lower():
            return heading
    return ""


def load_sections(docx_path: str) -> dict[str, str]:
    """Parse the docx and return {section_name: full_text_of_section}.

    Uses python-docx to walk paragraphs so Heading 1 styles are the
    authoritative section boundaries — no substring guessing.
    """
    path = Path(docx_path)
    if not path.exists():
        print(f"[ingest] ERROR: document not found at {path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print(f"[ingest] Loading {path} …")
    doc = Document(str(path))

    sections: dict[str, list[str]] = {h: [] for h in SECTION_HEADINGS}
    current_section: str = ""

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        style = para.style.name  # e.g. "Heading 1", "Normal", "List Bullet"

        if style == "Heading 1":
            canonical = _normalise_heading(text)
            if canonical:
                current_section = canonical
            # Don't add the heading itself as content
            continue

        if current_section:
            sections[current_section].append(text)

    # Join each section's paragraphs into a single text block
    return {sec: "\n\n".join(paras) for sec, paras in sections.items() if paras}


# ---------------------------------------------------------------------------
# Chunking — split each section independently so chunks never cross sections
# ---------------------------------------------------------------------------

def _approximate_page(char_offset: int, chars_per_page: int = 3000) -> int:
    return max(1, (char_offset // chars_per_page) + 1)


def chunk_sections(sections: dict[str, str], source_filename: str) -> list[dict]:
    """Split each section's text into overlapping chunks with exact metadata."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    all_chunks: list[dict] = []
    global_idx = 0

    for section in SECTION_HEADINGS:
        text = sections.get(section, "")
        if not text:
            print(f"[ingest]   {section:<25} — no content, skipping")
            continue

        raw = splitter.split_text(text)
        char_offset = 0

        for chunk_text in raw:
            page = _approximate_page(char_offset)
            char_offset += len(chunk_text)

            # ID = global index + content hash → always unique, stable on re-run
            content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()[:16]
            chunk_id = f"{global_idx:05d}-{content_hash}"
            global_idx += 1

            all_chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "source": source_filename,
                    "section": section,   # ← always the Heading 1, never guessed
                    "page": page,
                },
            })

        print(f"[ingest]   {section:<25} → {len(raw)} chunks")

    return all_chunks


# ---------------------------------------------------------------------------
# Embedding + persistence
# ---------------------------------------------------------------------------

def persist_to_chroma(chunks: list[dict]) -> None:
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    if existing_count > 0:
        print(
            f"[ingest] Collection '{CHROMA_COLLECTION_NAME}' already contains "
            f"{existing_count} chunks — skipping to avoid duplicates.\n"
            "         Delete ./chroma_db/ and re-run to rebuild from scratch."
        )
        return

    print(f"\n[ingest] Embedding {len(chunks)} chunks via {EMBEDDING_MODEL} …")

    embedder = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
    )

    BATCH = 64
    all_ids, all_texts, all_embeddings, all_metadatas = [], [], [], []

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i: i + BATCH]
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
    print(f"\n[ingest] ✓ Done. Total chunks in index: {final_count}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    path = Path(DOCX_PATH)
    sections = load_sections(DOCX_PATH)
    chunks = chunk_sections(sections, path.name)
    print(f"\n[ingest] Total chunks to index: {len(chunks)}")
    persist_to_chroma(chunks)
