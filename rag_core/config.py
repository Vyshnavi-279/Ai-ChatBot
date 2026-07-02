# rag_core/config.py
# Central configuration for the RAG pipeline.
# All values can be overridden by environment variables (loaded from .env).
# Import this module early so every component reads the same values.

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from repo root.  Fail loudly with a clear message if missing.
# ---------------------------------------------------------------------------
_env_path = Path(__file__).resolve().parent.parent / ".env"
if not _env_path.exists():
    raise FileNotFoundError(
        f"\n\n[rag_core] Could not find a .env file at {_env_path}.\n"
        "Please create one with at least:\n\n"
        "    OPENROUTER_API_KEY=your_key_here\n\n"
        "See .env.example for reference.\n"
    )

load_dotenv(_env_path)

# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        "\n\n[rag_core] OPENROUTER_API_KEY is not set in your .env file.\n"
        "Add the following line to .env and try again:\n\n"
        "    OPENROUTER_API_KEY=your_key_here\n"
    )

# OpenRouter uses an OpenAI-compatible API endpoint.
OPENROUTER_BASE_URL: str = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
)

# ---------------------------------------------------------------------------
# Model names  (override via env if you want to swap models)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.environ.get(
    "EMBEDDING_MODEL", "openai/text-embedding-3-small"
)
GENERATION_MODEL: str = os.environ.get(
    "GENERATION_MODEL", "openai/gpt-4o-mini"
)

# ---------------------------------------------------------------------------
# Chunking parameters (spec section 3: chunk_size=800, overlap=120)
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "120"))

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------
# Keep max_tokens below your available OpenRouter credit headroom.
# Free tier: set to 600. Paid tier: 1024 or higher is fine.
MAX_TOKENS: int = int(os.environ.get("MAX_TOKENS", "600"))

# ---------------------------------------------------------------------------
# Retrieval parameters
# ---------------------------------------------------------------------------
TOP_K: int = int(os.environ.get("TOP_K", "5"))

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR: str = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME: str = os.environ.get(
    "CHROMA_COLLECTION_NAME", "bvrit_knowledge"
)

# ---------------------------------------------------------------------------
# Source document
# ---------------------------------------------------------------------------
DOCX_PATH: str = os.environ.get("DOCX_PATH", "data/college_info.docx")

# ---------------------------------------------------------------------------
# The eight canonical section headings (used by retriever metadata filter)
# ---------------------------------------------------------------------------
SECTION_HEADINGS: list[str] = [
    "About BVRIT",
    "Departments",
    "Admissions",
    "Fee Structure",
    "Placements",
    "Campus & Facilities",
    "Faculty",
    "Contact",
]
