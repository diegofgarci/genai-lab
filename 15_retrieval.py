"""
15_retrieval.py — CLI for manual retrieval exploration
=======================================================
Day 4: thin command-line wrapper. The reusable logic lives in
retrieval.py; this file only parses arguments and prints results.

WHY the split: a module whose name starts with a digit cannot be
imported with the `import` statement, so `retrieve()` was unreachable
from the eval script. Same pattern as utils.py.

Usage:
    python 15_retrieval.py "What is the AI Usage Index?"
    python 15_retrieval.py "What is the AUI?" --collection aei_recursive --k 3
    python 15_retrieval.py "Coding tasks" --filter section="Economic Primitives"
"""

import argparse

from retrieval import DEFAULT_COLLECTION, DEFAULT_K, retrieve


# ── CLI helpers ──────────────────────────────────────────────────────────────

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
