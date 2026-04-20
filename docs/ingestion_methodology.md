# Document Ingestion for RAG: Methodology

A technical reference for the ingestion stage of a Retrieval-Augmented Generation pipeline — the steps that sit between a raw file on disk and a vector store ready for retrieval. Covers document loading, text cleaning, metadata enrichment, and the unified output contract that makes downstream chunking format-agnostic.

---

## 1. Why Ingestion Dominates RAG Quality

A RAG system has four moving parts: ingestion, chunking, retrieval, and generation. It's tempting to focus on retrieval (the vector store, the embedding model, rerankers) because that's where the "AI" lives. This is wrong.

Ingestion decisions propagate irreversibly. A poorly loaded document produces noisy chunks. Noisy chunks produce noisy embeddings. Noisy embeddings produce irrelevant retrievals. No amount of reranking fixes a PDF whose page numbers were extracted as standalone text and embedded as meaningful content.

Put differently: retrieval quality is bounded by ingestion quality. Spending time optimizing chunk size while your loader leaks headers and footers into every page is solving the wrong problem.

---

## 2. The Unified Output Contract

The single most important design decision in the ingestion layer is the shape of its output. Regardless of whether the input is a PDF, a Markdown file, an HTML page, or a proprietary format, the ingestion layer must produce a uniform structure that downstream components can consume without branching on source type.

The contract:

```python
[
    {
        "text": str,           # Cleaned content, ready for embedding or chunking
        "metadata": {
            "source": str,          # File path or URL
            "source_type": str,     # "pdf" | "markdown" | "html" | "txt"
            "char_count": int,      # Size of the text field
            "ingested_at": str,     # ISO 8601 UTC timestamp
            # Format-specific fields below:
            "page": int,            # PDF only
            "total_pages": int,     # PDF only
            "section_path": str,    # Markdown only: "H1 > H2 > H3"
            "header_level": int,    # Markdown only: 1-6
        }
    },
    ...
]
```

The structure is a flat list of "logical units". The definition of *logical unit* varies by format — for PDFs it's a page, for Markdown a section, for HTML a `<div>` or article. What matters is that each unit is independently retrievable, carries sufficient metadata to cite back to source, and is small enough to be chunked further if needed.

### Why a flat list, not a nested tree

A nested structure (document → chapter → section → paragraph) is more faithful to the source, but forces every downstream consumer to traverse the tree. A flat list with a `section_path` metadata field encodes the same hierarchy without the traversal cost. Retrieval never needs the tree shape; it needs to filter by attributes.

### Why one dict per page (PDF), not one dict per document

Pages are *free metadata*. The document itself provides page boundaries — extracting and preserving them costs nothing. The benefits are concrete: citations ("according to page 12..."), navigable retrieval (fetch adjacent pages for context), and a natural upper bound on chunk size (no chunk can ever span pages, even if chunking logic fails).

---

## 3. Text Extraction Strategies by Format

### 3.1 PDF

PDFs are not text files. They are collections of glyphs positioned in a 2D coordinate space, with no inherent reading order beyond what the producer tool encoded. Extraction libraries differ significantly in how well they reconstruct reading order.

| Library | Backend | Speed | Layout handling | When to use |
|---|---|---|---|---|
| `pypdf` | Pure Python | Slow | Basic | Quick scripts, small docs, zero-dependency constraints |
| `pdfplumber` | Pure Python | Slow | Good for tables | Structured PDFs with tabular data |
| `pymupdf` (fitz) | C (MuPDF) | Fast | Very good | Production pipelines, large docs, complex layouts |
| `unstructured` | Mixed + ML | Slowest | Best semantic | When layout reconstruction matters more than speed |

`pymupdf` is the reasonable default for narrative documents. It exposes multiple extraction modes, each with a different use case:

```python
page.get_text("text")    # Reading order, plain string — for narrative docs
page.get_text("blocks")  # List of (x0, y0, x1, y1, text, block_no, ...)
page.get_text("words")   # Token-level with bounding boxes
page.get_text("dict")    # Full tree with fonts, sizes, colors
```

**Rule of thumb:** use `"text"` for prose (reports, papers, books). Use `"blocks"` or `"dict"` when spatial position matters (forms, invoices, layouts where proximity implies semantic relationship).

### 3.2 Markdown

Markdown is structurally trivial compared to PDF, but the structure is valuable. A Markdown document declares its own hierarchy through headers, and that hierarchy is perfect retrieval metadata. Two approaches:

**Manual parsing** (regex over `^#{1,6}\s+`) is sufficient for most cases, has zero dependencies, and is transparent about what it does. The core algorithm is a *header stack*: maintain a list representing the current breadcrumb path, pop entries deeper than the current level when encountering a new header, push the new title.

**Library parsing** (`markdown-it-py`, `mistune`) returns an AST with full fidelity (links, emphasis, lists, code blocks as distinct node types). Overkill when the only goal is header-based sectioning, appropriate when downstream operations need block-level precision.

### 3.3 HTML

HTML is the noisiest common format. Before any meaningful content extraction, strip: navigation, footers, sidebars, ads, scripts, styles, cookie banners. Libraries like `trafilatura` and `readability-lxml` are purpose-built for this and should be preferred over naive BeautifulSoup scraping for general web content.

### 3.4 Plain text and code

Plain text needs no parsing but often requires character set detection (UTF-8, Latin-1, CP-1252). Source code should not be chunked like prose; use language-aware splitters that respect function and class boundaries.

---

## 4. Text Cleaning: What to Do and What Not to Do

The default instinct is to aggressively normalize text. This is wrong for modern embedding-based retrieval. Classical NLP preprocessing (lowercase, stem, remove stopwords, strip punctuation) was designed for bag-of-words models where each token contributed independently. Modern embedding models are trained on natural text and leverage casing, punctuation, and function words as signal.

### Do

| Transformation | Why |
|---|---|
| Join hyphenated line breaks: `inter-\nnational` → `international` | PDF line wrapping artificially splits words; unjoined, they become invalid tokens |
| Collapse 3+ consecutive newlines to 2 | Preserves paragraph separation without leaving arbitrary whitespace |
| Strip trailing whitespace per line | Invisible whitespace noise without semantic value |
| Remove zero-width chars (`\u200b`, `\u200c`, `\u200d`, `\ufeff`) | PDFs and copy-pasted text often embed these; invisible but tokenized |
| Collapse multiple spaces/tabs within a line | Layout artifact, no semantic content |

### Do Not

| Transformation | Why avoiding it matters |
|---|---|
| Lowercase | Casing signals proper nouns, acronyms, headings; "IT" vs "it" mean different things |
| Strip punctuation | Sentence boundaries matter for sentence-based chunking; commas change meaning |
| Remove stopwords | Embedding models use function words for structural context; "a bank" vs "the bank" are distinguishable |
| Stem / lemmatize | Destroys signal for embedding models, provides no benefit |
| Remove numbers | Prices, dates, versions, quantities are often the exact thing users query for |
| Unicode normalization (NFKC/NFKD) | Can silently mangle emoji, diacritics, non-Latin scripts |

### The order of operations matters

A subtle trap: if you collapse whitespace before joining hyphenated breaks, the regex `([a-z])-\n([a-z])` no longer matches — the `\n` is gone. Cleaning transformations are not commutative. Document the order, test the order.

---

## 5. Metadata Enrichment

Metadata is what converts a similarity search into *useful* retrieval. Pure semantic search is insufficient in production because:

- Users ask scoped questions ("what does the Q3 report say about X?") → need filter by `source`
- Documents are versioned → need filter by `ingested_at`
- Context requires adjacent content → need `chunk_index` for navigation
- Answers require citation → need `page` or `section_path` for attribution

The minimum useful metadata per unit:

```
source          — file path, URL, or unique identifier
source_type     — to drive format-specific retrieval logic
char_count      — for debugging, chunk-size validation, pricing
ingested_at     — ISO 8601 UTC, for versioning and freshness filters
```

Format-specific additions that cost nothing to extract:

```
page            — PDFs (1-indexed; what humans expect)
total_pages     — PDFs; enables "first/last page" queries
section_path    — Markdown; breadcrumb as "H1 > H2 > H3"
header_level    — Markdown; 1-6, enables structural filters
section_index   — Markdown; 0-indexed position in document
```

Fields to add only when you need them (avoid premature metadata bloat):

```
author, doc_title, created_at, modified_at, language, department, 
confidentiality_level, ...
```

### Why not put everything in metadata

Vector stores have practical limits on metadata size (ChromaDB does not support nested objects; some stores charge by total storage). Metadata should describe the unit, not duplicate its content. If a field is derivable from text + other metadata, compute it at query time, not at ingestion.

---

## 6. Filtering Empty or Degenerate Units

Real documents contain units that should not be indexed: blank pages, cover pages with only a title, tables of contents, sections with only a header and no body, image-only pages in scanned PDFs.

A simple character-count threshold handles most cases:

```python
if len(cleaned_text) < MIN_CHARS:
    skip()
```

The threshold is empirical. Values around 20-50 characters filter out standalone page numbers and near-empty sections without dropping legitimately brief content. Log what's skipped — if the skip rate exceeds ~5% of units, the source document likely has a systemic issue (scanned PDF without OCR, broken parser, malformed input).

Do not lower the threshold below 20 characters blindly. A unit of "(2)" or "Figure 4.1" passes technical extraction but contributes noise to embeddings.

---

## 7. Known Failure Modes

Ingestion pipelines fail in predictable ways. Document these for every corpus you process:

| Failure mode | Detection | Mitigation |
|---|---|---|
| Repeated headers/footers leak per page | Manual inspection of first 3 pages | Detect via cross-page string comparison; strip top/bottom N pixels; accept the noise |
| Page numbers extracted as standalone text | Units with char_count < 10 | Skip threshold, or regex-strip lines matching `^\d+$` |
| Scanned PDFs produce empty pages | Total extracted text << expected | Run OCR (`ocrmypdf`, Tesseract) before extraction |
| Multi-column PDFs extracted in wrong order | Manual inspection; garbled sentences | Use `pymupdf`'s `blocks` mode + custom reading-order logic |
| Titles split across lines by layout | Titles contain `\n` in middle of word sequences | Accept; embedding models are robust to whitespace |
| Tables extracted as unstructured text | Rows collapsed into single line | Use `pdfplumber` or a dedicated table extractor (`camelot`) |
| Images with embedded text ignored | Relevant content missing from extraction | Run OCR on page images (`pymupdf` + `pytesseract`) |
| Mixed-language documents | Poor embeddings for minority language | Detect language per unit, route to appropriate embedding model |

Not every failure mode needs mitigation on day one. Document them, measure their impact on retrieval quality, fix the ones that matter for the corpus.

---

## 8. Output Validation Before Proceeding

Before passing ingested units to chunking or embedding, validate:

```
✓ Total units > 0
✓ All units have a `text` field that is a non-empty string
✓ All units have `source`, `source_type`, `char_count` in metadata
✓ Distribution of char_count is sensible (no extreme outliers indicating bugs)
✓ Spot-check 3-5 random units: does the text look like what you expected?
```

The last point is non-negotiable. Automated checks verify structure; only human inspection verifies content. A pipeline that runs without errors but extracts garbled text will silently produce a useless RAG system.

---

## 9. Reference Output Shape

Minimum viable ingestion output for a mixed-format corpus:

```python
[
    # From a PDF
    {
        "text": "The Anthropic Economic Index Report\nIntroduction\nHow is AI reshaping...",
        "metadata": {
            "source": "data/pdf/anthropic_economic_index.pdf",
            "source_type": "pdf",
            "page": 2,
            "total_pages": 55,
            "char_count": 1858,
            "doc_title": "anthropic_economic_index",
            "doc_author": "unknown",
            "ingested_at": "2026-04-20T18:45:00+00:00"
        }
    },
    # From a Markdown file
    {
        "text": "## 1. What Problem RAG Solves\n\nLLMs have three structural limitations...",
        "metadata": {
            "source": "docs/rag_fundamentals.md",
            "source_type": "markdown",
            "section_path": "RAG Fundamentals > 1. What Problem RAG Solves",
            "header_level": 2,
            "section_index": 1,
            "char_count": 650,
            "ingested_at": "2026-04-20T18:45:00+00:00"
        }
    },
]
```

This is the interface contract. Whatever the ingestion layer does internally, this is what the rest of the pipeline expects. Change the internals freely; do not change the contract without reviewing every downstream consumer.