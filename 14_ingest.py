"""
14_ingest.py — Embed chunks and ingest into ChromaDB
=====================================================
Day 4: Read processed JSONL chunks (recursive + hybrid strategies),
embed them with MiniLM in batch, and ingest into separate ChromaDB
collections for clean comparison.

Architecture decisions:
  - One collection per chunking strategy (aei_recursive, aei_hybrid).
    Reason: clean isolation for retrieval evaluation. k=5 means "top 5
    of THIS strategy", not "top 5 globally then filter".
  - Rich metadata per chunk: page, section, strategy, chunk_index,
    n_tokens. Reason: lets us slice failure modes later
    ("are failing chunks always at the doc tail? always short?").
  - Single batch encode (batch_size=32). Reason: GPU/CPU vectorized
    matrix ops are orders of magnitude faster than per-chunk loops.
    Same answer, ~15x faster.

Idempotency:
  - Uses get_or_create_collection + upsert by chunk_id. Re-running
    the script does not duplicate data — it overwrites by ID.

Usage:
    python 14_ingest.py                    # ingest both strategies
    python 14_ingest.py --strategy recursive
    python 14_ingest.py --reset            # delete collections first
"""

import argparse
import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from embeddings import CHROMA_DIR, EMBEDDING_MODEL, get_client, get_model

# ── Config ───────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("corpus/processed")
BATCH_SIZE = 32

# Map: strategy folder name → collection name
STRATEGIES = {
    "recursive": "aei_recursive",
    "hybrid": "aei_hybrid",
}


# ── JSONL loading ────────────────────────────────────────────────────────────

def load_chunks(strategy: str) -> list[dict]:
    """
    Load all chunks from corpus/processed/{strategy}/*.jsonl.

    Each JSONL file represents one source document. Each line is one chunk.
    Returns a flat list of chunk dicts ready for embedding.
    """
    strategy_dir = PROCESSED_DIR / strategy
    if not strategy_dir.exists():
        raise FileNotFoundError(f"Missing directory: {strategy_dir}")

    chunks = []
    for jsonl_file in sorted(strategy_dir.glob("*.jsonl")):
        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))

    if not chunks:
        raise ValueError(f"No chunks found in {strategy_dir}")

    return chunks


# ── Metadata extraction ──────────────────────────────────────────────────────

def build_metadata(chunk: dict, strategy: str, chunk_index: int) -> dict:
    """
    Build the metadata dict that ChromaDB will store alongside the embedding.

    ChromaDB metadata must be flat (str/int/float/bool only — no nested dicts
    or None values). We pull what we need from the chunk's nested metadata
    block and flatten it.

    Why these fields:
      - chunk_id: redundant with Chroma's `id` but useful when results come
        back as dicts and you want the ID inside the metadata too
      - source: filename — needed if we ever ingest multiple PDFs
      - page: from the loader, lets us cite "page 12" in answers
      - section: from hybrid loader's structural parsing. Empty for this
        corpus (the AEI PDF doesn't expose hierarchical headers), but
        kept in the schema for future corpora.
      - strategy: redundant with collection name but enables cross-collection
        analysis if we later merge collections
      - chunk_index: position in the document, lets us detect "tail bias"
      - n_tokens: from tiktoken, lets us correlate length with retrieval quality
      - n_chars: complementary length metric, useful for analysis
      - was_split: True if the chunker split this unit, False if it passed
        through. Useful for diagnostics — distinguishes "loader output"
        from "chunker output".
      - doc_title: lets us filter/group by document in multi-doc setups
      - total_pages: contextual info about the source doc
    """
    meta = chunk.get("metadata", {})

    # ChromaDB does not accept None — coerce to empty string for missing fields
    return {
        "chunk_id": chunk["chunk_id"],
        "source": meta.get("source", "unknown"),
        "page": int(meta.get("page", 0)),
        "section": meta.get("section", "") or "",
        "strategy": strategy,
        "chunk_index": chunk_index,
        "n_tokens": int(meta.get("n_tokens", 0)),
        "n_chars": int(meta.get("n_chars", 0)),
        "was_split": bool(meta.get("was_split", False)),
        "doc_title": meta.get("doc_title", "") or "",
        "total_pages": int(meta.get("total_pages", 0)),
    }


# ── Embedding ────────────────────────────────────────────────────────────────

def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """
    Encode texts in batch. Returns list of embedding vectors.

    show_progress_bar=True gives visual feedback for large batches.
    convert_to_numpy=False returns plain lists (ChromaDB accepts both,
    but lists are simpler to debug).
    """
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


# ── Ingestion ────────────────────────────────────────────────────────────────

def ingest_strategy(
    client: chromadb.PersistentClient,
    model: SentenceTransformer,
    strategy: str,
    collection_name: str,
    reset: bool = False,
) -> dict:
    """
    Ingest all chunks of one strategy into its dedicated collection.

    Returns a stats dict for the summary at the end.
    """
    print(f"\n{'='*70}")
    print(f"  Strategy: {strategy}  →  Collection: {collection_name}")
    print(f"{'='*70}")

    # Reset collection if requested — useful when iterating on metadata schema
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"  [reset] deleted existing collection")
        except Exception:
            pass  # Collection didn't exist, which is fine

    # get_or_create makes the script idempotent: re-running won't error out
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # explicit cosine similarity
    )

    chunks = load_chunks(strategy)
    print(f"  Loaded {len(chunks)} chunks from JSONL")

    # Prepare arrays in matching order — ChromaDB requires positional alignment
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [build_metadata(c, strategy, idx) for idx, c in enumerate(chunks)]

    # Embed everything in one call. This is the batch optimization.
    print(f"  Embedding {len(documents)} chunks (batch_size={BATCH_SIZE})...")
    t0 = time.time()
    embeddings = embed_texts(model, documents)
    embed_time = time.time() - t0
    print(f"  Embedded in {embed_time:.2f}s ({len(documents)/embed_time:.1f} chunks/s)")

    # upsert > add: idempotent, won't fail on duplicate IDs
    print(f"  Upserting into ChromaDB...")
    t0 = time.time()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    upsert_time = time.time() - t0
    print(f"  Upserted in {upsert_time:.2f}s")

    final_count = collection.count()
    print(f"  Collection now contains: {final_count} documents")

    return {
        "strategy": strategy,
        "collection": collection_name,
        "n_chunks": len(chunks),
        "final_count": final_count,
        "embed_time": embed_time,
        "upsert_time": upsert_time,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        help="Ingest only one strategy (default: all)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete collections before ingesting (useful when changing schema)",
    )
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  RAG INGESTION — Day 4")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print(f"  ChromaDB:        {CHROMA_DIR}/")
    print(f"{'#'*70}")

    # Load embedding model once. Reused across all strategies.
    print(f"\nLoading embedding model...")
    t0 = time.time()
    model = get_model()
    print(f"Model loaded in {time.time()-t0:.2f}s")

    # PersistentClient: data survives across script runs (vs in-memory client)
    client = get_client()

    targets = [args.strategy] if args.strategy else list(STRATEGIES.keys())

    stats = []
    for strategy in targets:
        collection_name = STRATEGIES[strategy]
        try:
            result = ingest_strategy(client, model, strategy, collection_name, args.reset)
            stats.append(result)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
        except Exception as e:
            print(f"  [ERROR] {strategy}: {type(e).__name__}: {e}")

    # ── Summary ──
    if stats:
        print(f"\n{'='*70}")
        print(f"  INGESTION SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Strategy':<12} {'Chunks':>8} {'Embed':>10} {'Upsert':>10}")
        print(f"  {'-'*44}")
        for s in stats:
            print(
                f"  {s['strategy']:<12} {s['n_chunks']:>8} "
                f"{s['embed_time']:>9.2f}s {s['upsert_time']:>9.2f}s"
            )
        print()


if __name__ == "__main__":
    main()