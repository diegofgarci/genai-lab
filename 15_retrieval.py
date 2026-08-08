"""
15_retrieval.py — Retrieval function for RAG
==============================================
Day 4: Reusable retrieval component. The function `retrieve()` is what
will be imported by the eval script tomorrow, by the LLM pipeline next
week, and by the LangGraph agent in week 4.

Architecture decisions:
  - Module + script: `retrieve()` is importable; the __main__ block is
    just a CLI for manual exploration.
  - Returns list of dicts, not raw Chroma response. Each dict has:
    chunk_id, text, metadata, similarity, rank. Pythonic and easy to
    serialize for the eval.
  - Returns `similarity` (= 1 - distance), not Chroma's raw `distance`.
    This is the convention everyone uses in RAG. "How similar?" is the
    natural question, not "how far?".
  - Embedding model is loaded once at module level, not per-call.
    Loading the model takes ~3-4s; doing it per query would kill latency.

Why a single embedding model singleton:
  Even though we load it at import, importing this module from the eval
  script will trigger the load only once. ChromaDB's PersistentClient is
  also kept as a module-level singleton for the same reason.

Usage as a module:
    from retrieval import retrieve
    results = retrieve("What is the AI Usage Index?", "aei_hybrid", k=5)

Usage as a script:
    python 15_retrieval.py "What is the AI Usage Index?"
    python 15_retrieval.py "What is the AUI?" --collection aei_recursive --k 3
    python 15_retrieval.py "Coding tasks" --filter section="Economic Primitives"
"""

import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────────

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "aei_hybrid"
DEFAULT_K = 5

# ── Singletons ───────────────────────────────────────────────────────────────
# Loaded once at import time. The eval script will import this module and
# benefit from a single load across hundreds of queries.

_model: SentenceTransformer | None = None
_client: chromadb.PersistentClient | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_client() -> chromadb.PersistentClient:
    """Lazy-load the ChromaDB client on first use."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


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
    model = _get_model()
    query_embedding = model.encode(query, convert_to_numpy=True).tolist()

    # Open the collection. Will raise if it doesn't exist — we let the
    # error bubble up rather than catching it. Failing loud is better
    # than silently returning [].
    client = _get_client()
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


# ── CLI for manual exploration ───────────────────────────────────────────────

def _parse_filter_arg(filter_str: str) -> dict:
    """
    Parse a CLI filter argument like 'page=12' or 'section=Economic Primitives'.
    Only supports simple key=value equality for the CLI. For complex filters,
    use the function directly.
    """
    if "=" not in filter_str:
        raise ValueError(f"Filter must be key=value, got: {filter_str}")
    key, _, value = filter_str.partition("=")
    # Try int conversion; fall back to string
    try:
        value = int(value)
    except ValueError:
        pass
    return {key.strip(): value if isinstance(value, int) else value.strip()}


def _print_result(result: dict, max_text_chars: int = 300):
    """Pretty-print one retrieval result."""
    text_preview = result["text"][:max_text_chars]
    if len(result["text"]) > max_text_chars:
        text_preview += "..."

    print(f"\n  [#{result['rank']}] similarity={result['similarity']:.4f}  "
          f"chunk_id={result['chunk_id']}")
    print(f"      page={result['metadata'].get('page', '?')}  "
          f"section={result['metadata'].get('section', '') or '(none)'}  "
          f"n_tokens={result['metadata'].get('n_tokens', '?')}")
    print(f"      text: {text_preview}")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve top-k chunks from a ChromaDB collection."
    )
    parser.add_argument("query", help="Natural language query")
    parser.add_argument(
        "--collection", default=DEFAULT_COLLECTION,
        help=f"Collection name (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help=f"Number of results (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--filter", action="append", default=[],
        help="Metadata filter as key=value. Can be passed multiple times.",
    )
    args = parser.parse_args()

    # Build the filter dict from CLI args
    filters = None
    if args.filter:
        filters = {}
        for f in args.filter:
            filters.update(_parse_filter_arg(f))

    print(f"\n{'='*70}")
    print(f"  Query:       {args.query}")
    print(f"  Collection:  {args.collection}")
    print(f"  k:           {args.k}")
    if filters:
        print(f"  Filters:     {filters}")
    print(f"{'='*70}")

    results = retrieve(
        query=args.query,
        collection_name=args.collection,
        k=args.k,
        filters=filters,
    )

    if not results:
        print("\n  No results. Possible causes:")
        print("    - Collection is empty")
        print("    - Filters excluded all chunks")
        return

    for result in results:
        _print_result(result)

    # Quick stats footer
    sims = [r["similarity"] for r in results]
    print(f"\n  Similarity range: {min(sims):.4f} → {max(sims):.4f}")
    print(f"  Mean similarity:  {sum(sims)/len(sims):.4f}\n")


if __name__ == "__main__":
    main()