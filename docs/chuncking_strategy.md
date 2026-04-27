# Chunking Strategies for RAG — Conceptual Guide

> Reference document on document chunking for retrieval pipelines.
> Covers the *why* behind chunking decisions, the trade-offs between strategies,
> and the role chunking plays in the broader RAG architecture.

---

## 1. What Chunking Is and Why It Exists

Chunking is the operation that transforms a document into the **atomic units of retrieval** — the pieces that get embedded, stored in a vector database, and returned when a query matches.

The need for chunking comes from two hard constraints:

- **Embedding models have a maximum input size.** Sentence-transformer models like MiniLM cap around 256–512 tokens. Anything longer gets truncated silently, losing information.
- **Retrieval precision degrades with chunk size.** A 5-page chunk that contains 30 different topics will match almost any query with mediocre relevance. The retriever can't say "this 5-page chunk is about X specifically" — its embedding is an average of all 30 topics.

The chunk is therefore a deliberate compromise: small enough to be specific, large enough to carry context. The whole RAG pipeline depends on getting that compromise right.

## 2. The Chunking Pipeline in Context

```
RAW DOCUMENT (PDF, MD, HTML)
        │
        ▼
   [LOADER]  ──→ extracts text, detects natural boundaries (page, section)
        │       Output: list of "units" with text + metadata
        ▼
   [CHUNKER] ──→ splits units into retrievable chunks
        │       Output: list of "chunks" with text + inherited metadata + chunk-level metadata
        ▼
   [EMBEDDER] ──→ converts each chunk text into a vector
        │
        ▼
   [VECTOR STORE] ──→ stores (chunk_id, vector, metadata) for retrieval
```

Two key observations:

1. **The loader and the chunker are two different responsibilities.** The loader decides what counts as a logical unit of the source document (a page, a section). The chunker decides how to split those units into retrieval-sized pieces. Mixing them in a single component leads to unmaintainable code.

2. **Metadata flows downstream.** Whatever the loader extracts (page number, section path, source file) must be preserved through chunking, embedding, and storage — because at retrieval time, the system needs to cite "according to page 25 of report X" or filter "only chunks from section Methodology". If metadata is dropped at chunking, it's gone forever.

## 3. The Four Canonical Chunking Strategies

Listed from simplest to most sophisticated:

### 3.1. Fixed-size chunking
Split every N tokens, ignoring document structure entirely.

- **Pro**: trivial to implement, perfectly predictable size.
- **Con**: cuts mid-sentence, separates related context, places unrelated paragraphs in the same chunk.
- **When to use**: prototype-only baseline. Never production.

### 3.2. Recursive character splitting
Try to split on the most meaningful separator first (`\n\n` for paragraphs), fall back to coarser ones (`\n`, `. `, ` `) only when the chunk is still too big.

- **Pro**: respects natural boundaries when possible. Deterministic. Fast. Industry default.
- **Con**: still ignores higher-level document structure (sections, pages). Treats input as flat text.
- **When to use**: workhorse default for prose. Good baseline for any RAG project.

### 3.3. Document-structure-aware chunking
Use the document's own structure (markdown headers, HTML tags, PDF sections) as natural boundaries.

- **Pro**: chunks aligned with the author's logical units. Each chunk = one topical idea.
- **Con**: requires parsing structure first. Fails when documents have weak or inconsistent structure. Doesn't handle the case of an oversized section.
- **When to use**: well-structured corpora (technical documentation, reports with clear hierarchy).

### 3.4. Semantic chunking
Compute embeddings for individual sentences, then group sentences that are semantically close into chunks.

- **Pro**: chunks are topically coherent even when document structure is poor.
- **Con**: expensive (one embedding per sentence before chunking even starts). Slow. Results vary with the embedding model used to score similarity.
- **When to use**: rarely. Justified only when other strategies have measurably failed and structure is genuinely unrecoverable.

## 4. The Hybrid Strategy: What It Actually Is

Hybrid chunking is the combination of strategies 3.2 and 3.3:

- For each unit produced by the loader, **measure** its size in tokens.
- If the unit fits within the chunk size target, **emit it as a single chunk** — preserving its structural integrity.
- If the unit exceeds the target, **apply recursive character splitting** to it, with each resulting sub-chunk inheriting the original unit's metadata.

This is sometimes called the **parent-child pattern**: the parent is the structural unit; the children are the recursive sub-chunks generated when the parent is too large.

```
unit (e.g. one MD section, 350 tokens)
  └─ fits → 1 chunk (intact, was_split=False)

unit (e.g. one PDF page, 1200 tokens)
  └─ too big → recursive split → [chunk_a, chunk_b, chunk_c]
                                 (was_split=True, all share parent metadata)
```

The motivation: structural units are usually well-bounded topical envelopes. When they happen to be the right size, splitting them would destroy that natural coherence. When they happen to be too large, falling back to recursive is the pragmatic choice.

## 5. Sizing Decisions: Tokens, Not Characters

Three numbers govern any chunker:

- **Chunk size** — typical: 256–1024 tokens. The target size for each chunk.
- **Overlap** — typical: 10–20% of chunk size. Tokens shared between consecutive chunks.
- **Unit of measurement** — tokens, not characters.

### Why tokens, not characters

A 500-character chunk can be anywhere from 80 to 200 tokens depending on language and content. Embedding models consume tokens, so measuring in characters introduces a systematic error that varies by language:

| Language | Approx tokens per char |
|----------|------------------------|
| English  | ~0.25 (4 chars per token) |
| Spanish  | ~0.30 (3.3 chars per token) |
| Chinese  | ~0.50 (2 chars per token) |
| Code     | ~0.30 (varies wildly) |

Measuring in tokens via a tokenizer (e.g. `tiktoken` with `cl100k_base`) eliminates this error. The trade-off is a small constant cost per measurement.

### The chunk-size trade-off

| Chunk size | Retrieval precision | Context per chunk | Risk |
|------------|---------------------|-------------------|------|
| Small (256) | High — each chunk is specific | Low — fragments may lack surrounding context | LLM may not have enough context to answer |
| Medium (512) | Balanced | Balanced | The documented sweet spot for most prose |
| Large (1024) | Low — chunk averages many topics | High — full context per chunk | "Needle in a haystack" problem |

### The overlap trade-off

- **Zero overlap**: information that sits exactly on a chunk boundary may be lost — if the answer to a query happens to span a split, it's never retrieved as a coherent piece.
- **Excessive overlap (>30%)**: storage doubles, and retrieval returns near-duplicate chunks that compete for the same context window.
- **10% overlap (50 tokens on a 512-token chunk)**: empirically the documented sweet spot for prose.

## 6. The Crucial Insight: Loader Quality Determines Chunker Difficulty

The most important conclusion about chunking is non-obvious and rarely taught:

> **If the loader produces well-bounded units, the choice of chunking algorithm becomes nearly irrelevant.**

The reasoning:

- Modern recursive splitters (e.g. LangChain's `RecursiveCharacterTextSplitter`) only split when input exceeds the target size. Smaller inputs are returned unchanged.
- Therefore, when units are smaller than the target, recursive and hybrid produce identical output.
- They diverge only when units exceed the target — and even then, both apply the same underlying splitter.

The real architectural decision is **at what level the chunker operates**:

| Level of application | Result |
|----------------------|--------|
| Chunker over individual units (page, section) | Chunks respect structural boundaries. Recursive and hybrid converge. |
| Chunker over the concatenated corpus (whole document or multiple documents joined) | Chunks cross page and section boundaries, mixing unrelated context. Bad chunking. |

This is why the loader does 80% of the work in RAG quality. The chunker is a finishing operation, not the main act.

## 7. Metadata Inheritance: The `**` Spread Pattern

Every chunk should carry the full metadata of the unit it came from, plus chunker-specific additions:

```python
chunk = {
    "chunk_id": ...,
    "text": ...,
    "metadata": {
        **unit["metadata"],           # everything the loader added
        "chunk_strategy": "hybrid",   # what the chunker added
        "n_tokens": ...,
        "was_split": ...,
        ...
    }
}
```

Why this pattern matters:

- **Traceability**: at retrieval time, every chunk knows its source file, page, section, ingestion timestamp.
- **Citation**: the LLM response can reference "page 25 of the Anthropic Economic Index" because that field traveled with the chunk.
- **Filtering**: vector store queries can filter by `source_type=pdf` or `section_path contains Methodology`.
- **Decoupling**: if the loader later adds a new field (e.g. `language`), it appears in every chunk automatically. No chunker change required.

The order matters: spread the parent metadata first, then add chunk-level fields. If a key collides, the later write wins — the chunker's value overrides. This is the correct convention: the most recent producer has authority over its own fields.

## 8. Content-Addressed IDs

Each chunk needs a stable identifier for storage and retrieval. Two options:

- **Random UUID**: easy, but every pipeline run generates new IDs. Re-ingestion treats every chunk as new even when the content is identical — cache busting in the vector store.
- **Content hash** (e.g. `sha256(text)[:16]`): deterministic. Same text always yields the same ID.

Content-addressed IDs give three benefits for free:

1. **Idempotency**: re-running the pipeline produces no churn in the vector store. Upserts become no-ops when content hasn't changed.
2. **Deduplication**: identical chunks across documents collapse to one ID automatically.
3. **Auditability**: given a chunk's text, the ID can be re-derived. Useful for debugging.

Truncating SHA-256 to 16 hex characters (64 bits) gives collision probability negligible at any realistic scale, while keeping IDs short enough for logs and queries.

## 9. Persistence Format: JSONL

Chunks are typically persisted as JSON Lines — one JSON object per line:

```
{"chunk_id": "abc123...", "text": "...", "metadata": {...}}
{"chunk_id": "def456...", "text": "...", "metadata": {...}}
```

Why JSONL over a single JSON array:

- **Streamable**: each line parses independently. No need to load the whole file.
- **Tooling**: standard Unix tools (`grep`, `wc -l`, `head`, `tail`) work directly.
- **Append-only**: future processes can extend the file without rewriting it.
- **Industry standard**: the dominant format for ML datasets (HuggingFace, OpenAI fine-tuning, etc.).

## 10. Quality Signals to Track

Useful diagnostics to print when chunking a corpus:

| Metric | What it tells you |
|--------|-------------------|
| Total chunks generated | Pipeline scale |
| Mean / median tokens per chunk | Whether the target size is being respected |
| Min tokens | Detects fragments too small to be useful (often empty headers) |
| Max tokens | Detects chunks that exceed the embedder's input limit |
| P95 tokens | Tail behavior — how often you get oversized chunks |
| Distribution histogram | Visual confirmation of the strategy's behavior |
| `was_split` ratio (hybrid) | Health signal: low % means most units fit; high % means chunker is doing all the work |
| Chunks per source type | Asymmetry between document formats (PDF vs MD vs HTML) |

A min-token threshold (e.g. discard chunks <50 tokens) is a reasonable filter — fragments that small typically carry no information and pollute retrieval.

## 11. Common Pitfalls

- **Measuring in characters instead of tokens.** Silently introduces 10–50% error depending on language.
- **Ignoring document structure.** Treating a structured document as flat text discards information the author already encoded.
- **Dropping metadata at the chunk boundary.** Once lost, it's unrecoverable downstream — citations and filters become impossible.
- **Random IDs instead of content hashes.** Forces full re-embedding on every pipeline run.
- **Chunks without overlap.** Information at chunk boundaries gets lost from retrieval.
- **Excessive overlap.** Storage doubles and near-duplicate chunks crowd retrieval results.
- **Choosing a strategy without measuring.** Chunking without diagnostics is guessing.

## 12. Summary

Chunking is a finishing operation on top of the loader's work. Its job is to enforce a maximum size on already-bounded units while preserving metadata and producing stable identifiers.

The architectural priority order:

1. Get the loader right — it determines the natural boundaries.
2. Choose token-aware sizing — measure in tokens, not characters.
3. Pick recursive or hybrid based on whether structural metadata is meaningful in the corpus.
4. Persist with stable IDs and inherited metadata.
5. Measure distributions before and after — the histogram is the test.

The chunker is rarely where RAG quality is won or lost. It is, however, where it can be silently broken if metadata is dropped or sizing is wrong.