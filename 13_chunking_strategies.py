"""
13_chunking_strategies.py — Chunking Strategies for RAG
========================================================
Week 3, Day 3 — Take the units produced by 12_document_loaders.py and split
them into retrievable chunks, comparing two strategies head-to-head.

WHY this script exists:
  The unit (a PDF page, a Markdown section) is too coarse for retrieval —
  retrieve a 5-page PDF page and you'll send the LLM a wall of mixed topics.
  The chunk is the actual atom of RAG: small enough to be specific, large
  enough to carry context. Today we decide HOW to make those chunks.

What this DOES NOT do:
  - No embeddings (10_embeddings.py)
  - No vector store ingestion (will be done in 14_*)
  - No retrieval evaluation (later this week)

Strategies compared:
  1. RECURSIVE — LangChain's RecursiveCharacterTextSplitter applied
     uniformly to every unit. Ignores existing structure (page boundary,
     section heading) and splits on natural separators (\\n\\n, \\n, ". ").

  2. HYBRID — Respect existing unit boundaries when they fit; recursively
     split only when a unit exceeds chunk_size. The "parent-child" pattern.
     Best of both worlds: structural coherence when possible, predictable
     size when not.

Output: corpus/processed/{strategy}/chunks.jsonl
        One JSON object per line, with stable content-hash chunk_ids
        for idempotent downstream processing.

Usage:
    python 13_chunking_strategies.py
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from statistics import mean, median

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

# We import the loaders directly — no duplication of file paths or logic.
# This is the whole point of yesterday's work being callable today.
from importlib import import_module
loader_module = import_module("12_document_loaders")
load_pdf = loader_module.load_pdf
load_markdown = loader_module.load_markdown
PDF_PATH = loader_module.PDF_PATH
MD_PATH = loader_module.MD_PATH

from utils import print_header


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — chunking parameters
# ═══════════════════════════════════════════════════════════════════════════════

# Chunk size in tokens, NOT characters.
# WHY tokens: embedding models consume tokens, not characters. A 512-char chunk
# can be anywhere from 80 to 200 tokens depending on language and content.
# Using tokens directly is the only way to control "how much information per chunk".
CHUNK_SIZE_TOKENS = 512

# Overlap as ~10% of chunk size — the documented sweet spot.
# Too low: context at chunk boundaries is lost.
# Too high: storage doubles, retrieval returns near-duplicates.
CHUNK_OVERLAP_TOKENS = 50

# Tokenizer used for measuring chunk size.
# cl100k_base is OpenAI's tokenizer (GPT-4 family).
# WHY this and not MiniLM's tokenizer (which is what we'll actually embed with):
#   - Character → cl100k_base differs from MiniLM by ~10-15%, tolerable
#   - cl100k_base is the de facto standard in production RAG pipelines
#   - tiktoken is C-fast vs HuggingFace tokenizers' Python overhead
#   - Chunking doesn't need exact precision — just "roughly N tokens, never explode"
TOKENIZER_NAME = "cl100k_base"

# Output paths
PROCESSED_DIR = Path("corpus/processed")


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZER — single instance, shared
# ═══════════════════════════════════════════════════════════════════════════════
# Loaded once. tiktoken's encode() is thread-safe and cheap to call repeatedly,
# but loading the encoding has nontrivial cost (~50ms). Cache it.
tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)


def count_tokens(text: str) -> int:
    """Count tokens using the configured tokenizer."""
    return len(tokenizer.encode(text))


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNK ID — content-addressed, idempotent
# ═══════════════════════════════════════════════════════════════════════════════
# WHY a content hash instead of a UUID:
#   - Re-running the pipeline produces identical IDs → ChromaDB upserts are no-ops
#   - Identical chunks across documents collapse to one ID → free dedup
#   - The ID is auditable: given the chunk text, you can re-derive its ID
#
# WHY 16 chars (64 bits) of SHA-256:
#   - 64 bits ≈ collision probability negligible at our scale (millions OK)
#   - Full 256 bits is unwieldy in logs and ChromaDB IDs
#   - SHA-256 truncated to 16 hex chars is the de facto standard in RAG tooling

def make_chunk_id(text: str) -> str:
    """Stable content-addressed ID for a chunk."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1 — RECURSIVE
# ═══════════════════════════════════════════════════════════════════════════════
# The workhorse. RecursiveCharacterTextSplitter tries to split on the most
# meaningful separator first ("\n\n" = paragraph), falling back to coarser
# ones ("\n", ". ", " ") only if a chunk is still too big.
#
# Crucially, this strategy IGNORES the unit boundaries from the loader.
# It treats the corpus as flat text within each unit and re-chunks from scratch.
# This is the wrong approach for documents with rich structure — but it's the
# right baseline to measure improvement against.

def chunk_recursive(units: list[dict]) -> list[dict]:
    """
    Apply recursive character splitting to every unit, uniformly.
    Returns a flat list of chunk dicts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        # length_function tells the splitter HOW to measure size.
        # Default is len() (chars); we override with token count.
        length_function=count_tokens,
        # The default separator list, ordered by preference.
        # Splitter walks this list trying coarser splits if finer fails.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for unit_idx, unit in enumerate(units):
        # split_text returns a list of strings — we wrap them with metadata
        text_chunks = splitter.split_text(unit["text"])

        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunks.append({
                "chunk_id": make_chunk_id(chunk_text),
                "text": chunk_text,
                "metadata": {
                    # Inherit the unit's metadata so we keep page/section context
                    **unit["metadata"],
                    # Chunker-specific metadata
                    "chunk_strategy": "recursive",
                    "chunk_index_in_unit": chunk_idx,
                    "n_chunks_in_unit": len(text_chunks),
                    "unit_index": unit_idx,
                    "n_tokens": count_tokens(chunk_text),
                    "n_chars": len(chunk_text),
                },
            })

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2 — HYBRID (respect structure, split when too big)
# ═══════════════════════════════════════════════════════════════════════════════
# The smart strategy. For each unit:
#   - If unit fits in CHUNK_SIZE_TOKENS → emit as a single chunk (no split)
#   - If unit exceeds CHUNK_SIZE_TOKENS → recursively split it, but each
#     resulting chunk inherits the unit's structural metadata (page, section_path)
#
# WHY this is better than pure recursive:
#   When a markdown section fits, it's a perfect chunk: one topic, with its
#   header attached, fully self-contained. Splitting it would create chunks
#   that lose the topical "envelope".
#
# WHY this is better than pure structural:
#   When a section is huge (e.g. a 4000-token "Methodology" section), shoving
#   it in as one chunk degrades retrieval — every query about anything in that
#   section returns the whole 4000-token blob. Subdividing recovers specificity.
#
# This is the parent-child pattern. The "parent" is the structural unit; the
# "children" are the recursive sub-chunks when subdivision is needed. We track
# `was_split` in metadata so we can distinguish them later.

def chunk_hybrid(units: list[dict]) -> list[dict]:
    """
    Respect unit boundaries when they fit; recursively split only when needed.
    Returns a flat list of chunk dicts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for unit_idx, unit in enumerate(units):
        unit_token_count = count_tokens(unit["text"])

        if unit_token_count <= CHUNK_SIZE_TOKENS:
            # Unit fits — emit as single chunk
            chunks.append({
                "chunk_id": make_chunk_id(unit["text"]),
                "text": unit["text"],
                "metadata": {
                    **unit["metadata"],
                    "chunk_strategy": "hybrid",
                    "chunk_index_in_unit": 0,
                    "n_chunks_in_unit": 1,
                    "unit_index": unit_idx,
                    "n_tokens": unit_token_count,
                    "n_chars": len(unit["text"]),
                    "was_split": False,
                },
            })
        else:
            # Unit too big — recursive split, but mark as split
            text_chunks = splitter.split_text(unit["text"])

            for chunk_idx, chunk_text in enumerate(text_chunks):
                chunks.append({
                    "chunk_id": make_chunk_id(chunk_text),
                    "text": chunk_text,
                    "metadata": {
                        **unit["metadata"],
                        "chunk_strategy": "hybrid",
                        "chunk_index_in_unit": chunk_idx,
                        "n_chunks_in_unit": len(text_chunks),
                        "unit_index": unit_idx,
                        "n_tokens": count_tokens(chunk_text),
                        "n_chars": len(chunk_text),
                        "was_split": True,
                    },
                })

    return chunks


# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENCE — JSONL, one chunk per line
# ═══════════════════════════════════════════════════════════════════════════════
# WHY JSONL over a single JSON file:
#   - Streams: each line parses independently — no need to load the whole file
#   - grep/wc/head friendly: standard unix tools work without jq
#   - Append-only: future scripts can extend without rewriting
#   - Industry standard for ML datasets (HuggingFace, OpenAI fine-tuning, etc.)

def write_jsonl(chunks: list[dict], out_path: Path) -> None:
    """Write chunks as JSONL — one JSON object per line."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            # ensure_ascii=False so accented chars survive readably
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS — the comparison is the whole point
# ═══════════════════════════════════════════════════════════════════════════════
# WHY this is so detailed: the chunking decision is invisible without metrics.
# Reading 5 chunks tells you nothing. The histogram tells you everything.

def describe_chunks(chunks: list[dict], label: str) -> dict:
    """Print stats on a chunk collection. Returns the stats dict for cross-comparison."""
    if not chunks:
        print(f"  ⚠️  {label}: no chunks produced")
        return {}

    token_counts = [c["metadata"]["n_tokens"] for c in chunks]
    char_counts = [c["metadata"]["n_chars"] for c in chunks]

    # Distribution buckets — 100-token wide
    buckets = Counter()
    for n in token_counts:
        bucket = (n // 100) * 100
        buckets[bucket] += 1

    # Source distribution
    by_source_type = Counter(c["metadata"]["source_type"] for c in chunks)

    # Hybrid-only: how many were split vs intact?
    split_stats = None
    if any("was_split" in c["metadata"] for c in chunks):
        split_count = sum(1 for c in chunks if c["metadata"].get("was_split"))
        split_stats = {
            "was_split": split_count,
            "intact": len(chunks) - split_count,
        }

    stats = {
        "label": label,
        "n_chunks": len(chunks),
        "total_tokens": sum(token_counts),
        "mean_tokens": round(mean(token_counts), 1),
        "median_tokens": int(median(token_counts)),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "p95_tokens": sorted(token_counts)[int(len(token_counts) * 0.95)],
        "by_source_type": dict(by_source_type),
        "split_stats": split_stats,
    }

    print(f"\n  {label}")
    print(f"  {'─' * 64}")
    print(f"  Total chunks:    {stats['n_chunks']}")
    print(f"  Total tokens:    {stats['total_tokens']:,}")
    print(f"  Tokens/chunk:    "
          f"mean={stats['mean_tokens']}, median={stats['median_tokens']}, "
          f"min={stats['min_tokens']}, max={stats['max_tokens']}, "
          f"p95={stats['p95_tokens']}")
    print(f"  By source type:  {stats['by_source_type']}")

    if split_stats:
        intact_pct = split_stats["intact"] / len(chunks) * 100
        print(f"  Intact (fit whole):  {split_stats['intact']} ({intact_pct:.0f}%)")
        print(f"  Was split:           {split_stats['was_split']}")

    # Distribution histogram (text-based)
    print(f"  Distribution (token buckets):")
    for bucket in sorted(buckets.keys()):
        bar = "█" * min(buckets[bucket], 50)
        print(f"    {bucket:>4}-{bucket+99:<4} | {bar} {buckets[bucket]}")

    return stats


def compare_strategies(stats_a: dict, stats_b: dict) -> None:
    """Side-by-side comparison of two strategies."""
    print("\n\n" + "=" * 70)
    print("  STRATEGY COMPARISON")
    print("=" * 70)
    print(f"\n  {'Metric':<25} {stats_a['label']:>18}  {stats_b['label']:>18}")
    print(f"  {'-' * 25} {'-' * 18}  {'-' * 18}")

    rows = [
        ("Total chunks",   stats_a["n_chunks"],     stats_b["n_chunks"]),
        ("Total tokens",   stats_a["total_tokens"], stats_b["total_tokens"]),
        ("Mean tokens",    stats_a["mean_tokens"],  stats_b["mean_tokens"]),
        ("Median tokens",  stats_a["median_tokens"], stats_b["median_tokens"]),
        ("Min tokens",     stats_a["min_tokens"],   stats_b["min_tokens"]),
        ("Max tokens",     stats_a["max_tokens"],   stats_b["max_tokens"]),
        ("P95 tokens",     stats_a["p95_tokens"],   stats_b["p95_tokens"]),
    ]
    for name, a, b in rows:
        print(f"  {name:<25} {a:>18}  {b:>18}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print_header("CHUNKING STRATEGIES — Recursive vs Hybrid")

    # ── Phase 1: Load units (reuse yesterday's loaders) ──
    print("Phase 1: Loading units from disk")
    print("─" * 70)
    pdf_units = load_pdf(PDF_PATH)
    md_units = load_markdown(MD_PATH)
    units = pdf_units + md_units
    print(f"\n  Total units loaded: {len(units)} "
          f"({len(pdf_units)} PDF pages + {len(md_units)} MD sections)")

    # ── Phase 2: Apply both strategies ──
    print("\n\nPhase 2: Applying chunking strategies")
    print("─" * 70)
    print(f"  Config: chunk_size={CHUNK_SIZE_TOKENS} tokens, "
          f"overlap={CHUNK_OVERLAP_TOKENS} tokens, tokenizer={TOKENIZER_NAME}")

    print("\n  Running RECURSIVE strategy...")
    recursive_chunks = chunk_recursive(units)

    print("  Running HYBRID strategy...")
    hybrid_chunks = chunk_hybrid(units)

    # ── Phase 3: Persist ──
    print("\n\nPhase 3: Persisting chunks to disk")
    print("─" * 70)
    recursive_path = PROCESSED_DIR / "recursive" / "chunks.jsonl"
    hybrid_path = PROCESSED_DIR / "hybrid" / "chunks.jsonl"

    write_jsonl(recursive_chunks, recursive_path)
    write_jsonl(hybrid_chunks, hybrid_path)

    print(f"  Recursive: {len(recursive_chunks)} chunks → {recursive_path}")
    print(f"  Hybrid:    {len(hybrid_chunks)} chunks → {hybrid_path}")

    # ── Phase 4: Diagnostics ──
    print("\n\nPhase 4: Diagnostics")
    print("─" * 70)
    stats_recursive = describe_chunks(recursive_chunks, "RECURSIVE")
    stats_hybrid = describe_chunks(hybrid_chunks, "HYBRID")

    compare_strategies(stats_recursive, stats_hybrid)

    # ── Phase 5: Sample inspection ──
    print("\n\n" + "=" * 70)
    print("  SAMPLE CHUNKS — eyeball test")
    print("=" * 70)

    print("\n  RECURSIVE — first chunk of the corpus:")
    sample = recursive_chunks[0]
    print(f"    chunk_id: {sample['chunk_id']}")
    print(f"    metadata: page={sample['metadata'].get('page')}, "
          f"section={sample['metadata'].get('section_path')}, "
          f"tokens={sample['metadata']['n_tokens']}")
    print(f"    text (first 250 chars):\n      {sample['text'][:250]}...")

    print("\n  HYBRID — first INTACT (was_split=False) chunk:")
    intact = next((c for c in hybrid_chunks if not c["metadata"].get("was_split")), None)
    if intact:
        print(f"    chunk_id: {intact['chunk_id']}")
        print(f"    metadata: page={intact['metadata'].get('page')}, "
              f"section={intact['metadata'].get('section_path')}, "
              f"tokens={intact['metadata']['n_tokens']}, was_split=False")
        print(f"    text (first 250 chars):\n      {intact['text'][:250]}...")

    print("\n  HYBRID — first SPLIT (was_split=True) chunk:")
    split = next((c for c in hybrid_chunks if c["metadata"].get("was_split")), None)
    if split:
        print(f"    chunk_id: {split['chunk_id']}")
        print(f"    metadata: page={split['metadata'].get('page')}, "
              f"section={split['metadata'].get('section_path')}, "
              f"tokens={split['metadata']['n_tokens']}, was_split=True")
        print(f"    text (first 250 chars):\n      {split['text'][:250]}...")

    print("""

  WHAT TO LOOK FOR IN THE OUTPUT:
    1. RECURSIVE will have MORE chunks than HYBRID — it ignores unit
       boundaries and splits everything to ~512 tokens uniformly.
    2. RECURSIVE max_tokens will be close to 512 (predictable).
       HYBRID max_tokens may be HIGHER for intact short units that didn't
       need splitting, but their distribution will be wider.
    3. The "intact %" in HYBRID tells you how well-sized your source units
       already are. High intact % → your loader's unit boundaries are good.
       Low intact % → most of your structure had to be subdivided anyway.
    4. Compare the SAME chunk index across strategies. Notice how RECURSIVE
       may cut mid-section while HYBRID respects section boundaries when it
       can. This is the qualitative win that metrics alone don't capture.

  NEXT STEP (14_*):
    Embed both chunk sets, store in two ChromaDB collections, run the same
    query against both, and see which strategy retrieves more relevant
    chunks. This is where the real comparison happens — chunk count and
    size are leading indicators; retrieval quality is the actual outcome.
""")


if __name__ == "__main__":
    main()