"""
embeddings.py — Shared embedding model and ChromaDB client
===========================================================
Single source of truth for the embedding model and the vector store
connection. Everything that embeds text or talks to ChromaDB imports
from here.

WHY: 14_ingest.py and 15_retrieval.py each declared their own
EMBEDDING_MODEL constant and instantiated their own SentenceTransformer
(and their own PersistentClient). Two literal copies of the same model
name is a correctness hazard, not just duplication: if the corpus and
the queries are ever embedded by different models, similarity scores
become meaningless with no error to warn you.

Both getters are lazy singletons. Importing this module must stay cheap
— loading MiniLM takes ~3-4s, and retrieval.py is imported by the eval
script, which should not pay that cost at import time.
"""

import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db"

# ── Singletons ───────────────────────────────────────────────────────────────
# Created on first use, then reused. A script that loads the model early
# (like the ingestion pipeline) just calls get_model() early.

_model: SentenceTransformer | None = None
_client: chromadb.PersistentClient | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_client() -> chromadb.PersistentClient:
    """Lazy-load the ChromaDB client on first use."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client
