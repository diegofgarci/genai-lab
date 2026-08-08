"""
retrieval.py — Retrieval component for RAG
===========================================
The reusable half of what used to be 15_retrieval.py. `retrieve()` is
imported by the eval script, by the LLM pipeline, and later by the
LangGraph agent. 15_retrieval.py is now just a CLI on top of this.

WHY the split: a module whose name starts with a digit cannot be
imported with the `import` statement, so `retrieve()` was unreachable
from any other script. Same pattern as utils.py.

This module returns data and stops there — console formatting and CLI
argument parsing live in 15_retrieval.py, so importers don't drag a
terminal formatter into a pipeline.

Architecture decisions (unchanged from the original):
  - Returns list of dicts, not raw Chroma response. Each dict has:
    chunk_id, text, metadata, similarity, rank. Pythonic and easy to
    serialize for the eval.
  - Returns `similarity` (= 1 - distance), not Chroma's raw `distance`.
    This is the convention everyone uses in RAG. "How similar?" is the
    natural question, not "how far?".
  - Model and client come from embeddings.py as lazy singletons, so
    hundreds of queries share one load.

Usage:
    from retrieval import retrieve
    results = retrieve("What is the AI Usage Index?", "aei_hybrid", k=5)
"""

from embeddings import get_client, get_model

# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_COLLECTION = "aei_hybrid"
DEFAULT_K = 5


# ── Core retrieval function ──────────────────────────────────────────────────

def retrieve(
    query: str,
    collection_name: str = DEFAULT_COLLECTION,
    k: int = DEFAULT_K,
    filters: dict | None = None,
) -> list[dict]:
    """
    Retrieve top-k chunks from a ChromaDB collection given a text query.

    Args:
        query: natural language query, will be embedded with MiniLM.
        collection_name: which collection to search (e.g. 'aei_hybrid').
        k: number of results to return.
        filters: optional metadata filters in ChromaDB's `where` syntax.
                 Examples:
                   {"page": 12}                              → exact match
                   {"page": {"$gte": 10}}                    → page >= 10
                   {"section": {"$ne": ""}}                  → has section
                   {"$and": [{"strategy": "hybrid"},
                             {"n_tokens": {"$gte": 100}}]}   → compound

    Returns:
        List of dicts ordered by similarity (highest first). Each dict:
            {
                "rank": int,            # 1-indexed position
                "chunk_id": str,
                "text": str,
                "similarity": float,    # 1 - cosine_distance, in [0, 1]
                "metadata": dict,
            }

    Returns an empty list if the collection has fewer than k chunks
    matching the filters.
    """
    # Embed the query. Same model that embedded the corpus — critical:
    # if the query and corpus were embedded by different models, similarity
    # scores would be meaningless.
    model = get_model()
    query_embedding = model.encode(query, convert_to_numpy=True).tolist()

    # Open the collection. Will raise if it doesn't exist — we let the
    # error bubble up rather than catching it. Failing loud is better
    # than silently returning [].
    client = get_client()
    collection = client.get_collection(collection_name)

    # Build query kwargs. ChromaDB only accepts the `where` key when
    # filters are non-empty, so we add it conditionally.
    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": k,
    }
    if filters:
        query_kwargs["where"] = filters

    raw = collection.query(**query_kwargs)

    # ChromaDB returns lists-of-lists because it supports batched queries.
    # We sent one query, so we index [0] to unpack.
    ids = raw["ids"][0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    # Transform into the clean output shape.
    # similarity = 1 - distance only valid for cosine. We set
    # hnsw:space="cosine" at ingestion time, so we're safe here.
    results = []
    for rank, (chunk_id, text, meta, dist) in enumerate(
        zip(ids, documents, metadatas, distances), start=1
    ):
        results.append({
            "rank": rank,
            "chunk_id": chunk_id,
            "text": text,
            "similarity": round(1.0 - dist, 4),
            "metadata": meta,
        })

    return results
