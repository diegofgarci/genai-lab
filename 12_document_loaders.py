"""
12_document_loaders.py — Document Ingestion for RAG
=====================================================
Week 3, Day 2 — Load real documents (PDF, Markdown) into a unified structure
that downstream chunking + embedding can consume.

WHY this script exists:
  Yesterday we put handcrafted strings into ChromaDB. That's fine for
  learning the API, useless for real RAG. Real RAG starts with real
  documents — which come in messy formats (PDF layouts, HTML ruido,
  markdown with nested structure). The 80% of RAG quality is decided
  between the raw file and the embedding call. This script owns that 80%.

What this DOES NOT do:
  - No chunking (that's 13_chunking_strategies.py, tomorrow)
  - No embeddings (done yesterday in 10_embeddings.py)
  - No ChromaDB (done yesterday in 11_chromadb.py)

Output: a unified data structure — list[dict] — where each dict is one
"logical unit" (a PDF page, a Markdown section) with clean text + rich
metadata. Tomorrow's chunker will consume this.

Usage:
    python 12_document_loaders.py
"""

import re
from pathlib import Path
from datetime import datetime, timezone

import fitz  # PyMuPDF — imported as fitz for historical reasons (MuPDF heritage)

from utils import print_header


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — paths and constants
# ═══════════════════════════════════════════════════════════════════════════════

PDF_PATH = Path("data/pdf/anthropic_economic_index.pdf")
MD_PATH = Path("docs/rag_fundamentals.md")


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING — shared across loaders
# ═══════════════════════════════════════════════════════════════════════════════
# WHY a separate clean step:
#   Raw extraction from PDFs carries noise: line-break hyphens ("inter-\nnational"),
#   multiple consecutive blank lines, trailing whitespace, zero-width chars, etc.
#   Markdown is cleaner but still has edge cases (tabs, trailing spaces).
#   Keeping cleaning separate from extraction lets us A/B the effect —
#   and keeps each function single-purpose.

def clean_text(text: str) -> str:
    """
    Remove common noise from extracted text while preserving paragraph structure.

    Transformations (order matters):
      1. Join words broken across lines by hyphenation ("inter-\\nnational" → "international")
      2. Collapse 3+ consecutive newlines into exactly 2 (paragraph separator)
      3. Strip trailing whitespace from each line
      4. Remove zero-width and other invisible unicode chars
      5. Collapse multiple spaces within a line into one

    What we DELIBERATELY do NOT do here:
      - Do not strip newlines entirely: paragraph structure matters for chunking
      - Do not lowercase: casing is a signal (ALL CAPS headers, proper nouns)
      - Do not remove punctuation: needed for sentence-based chunking later
    """
    # 1. Join hyphenated line breaks: "word-\nword" → "wordword"
    #    Conservative heuristic: only joins when the char before "-" is lowercase.
    #    Avoids breaking legit hyphens like "state-of-the-art\nmodel".
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)

    # 2. Remove zero-width chars and other invisibles that PDFs love to embed
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

    # 3. Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # 4. Collapse multiple blank lines: \n\n\n+ → \n\n
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Collapse multiple spaces within a line (but not newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# PDF LOADER — PyMuPDF / fitz
# ═══════════════════════════════════════════════════════════════════════════════
# WHY PyMuPDF over pypdf:
#   - 5-10x faster on large docs (C backend vs pure Python)
#   - Better handling of complex layouts (multi-column, tables)
#   - Exposes per-page metadata (dimensions, rotation) if we ever need it
#   - Can extract images and bounding boxes (overkill today, useful later
#     when we tackle invoices in your capstone)
#
# WHY one-page-per-unit (not one-doc-per-unit):
#   - Page boundaries are "free metadata" — they come from the document itself,
#     we don't have to guess them
#   - Enables citations: "according to page 12 of the report..."
#   - Gives the chunker a natural upper bound — even if chunking fails,
#     a chunk never spans more than one page

def load_pdf(path: Path) -> list[dict]:
    """
    Load a PDF and return one dict per page.

    Each dict contains:
      - text: cleaned text of the page
      - metadata: dict with source, source_type, page, total_pages,
                  char_count, doc_title, doc_author, ingested_at

    Skips pages with empty/whitespace-only text (common in PDFs with
    image-only pages or blank separators).
    """
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(path)

    # Doc-level metadata — same for all pages, extracted once
    # PyMuPDF exposes this through doc.metadata (a dict with standard PDF fields)
    doc_metadata = doc.metadata or {}
    doc_title = doc_metadata.get("title") or path.stem
    doc_author = doc_metadata.get("author") or "unknown"
    total_pages = len(doc)

    units = []
    empty_pages = 0
    ingested_at = datetime.now(timezone.utc).isoformat()

    for page_num in range(total_pages):
        page = doc[page_num]

        # Extract text. "text" mode preserves reading order;
        # alternatives: "blocks" (structured), "words" (token-level).
        # For narrative docs, "text" is almost always the right choice.
        raw_text = page.get_text("text")

        cleaned = clean_text(raw_text)

        # Skip empty pages (covers, blank separators, image-only pages)
        if not cleaned or len(cleaned) < 20:  # 20 chars min — filters page numbers alone
            empty_pages += 1
            continue

        units.append({
            "text": cleaned,
            "metadata": {
                "source": str(path),
                "source_type": "pdf",
                "page": page_num + 1,        # 1-indexed — what humans expect
                "total_pages": total_pages,
                "char_count": len(cleaned),
                "doc_title": doc_title,
                "doc_author": doc_author,
                "ingested_at": ingested_at,
            }
        })

    doc.close()

    print(f"  [PDF] {path.name}")
    print(f"        Total pages: {total_pages}")
    print(f"        Non-empty pages extracted: {len(units)}")
    print(f"        Skipped (empty/short): {empty_pages}")
    print(f"        Doc title: {doc_title}")
    print(f"        Doc author: {doc_author}")

    return units


# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN LOADER — structural parsing by headers
# ═══════════════════════════════════════════════════════════════════════════════
# WHY parse structure manually instead of using a library:
#   - Pedagogically clearer — you see exactly what "structural chunking" means
#   - Zero dependencies beyond the standard library
#   - Library alternatives (markdown-it-py, mistune) parse to AST which is
#     overkill when we only care about H1-H6 boundaries
#   - Tomorrow we'll swap this for LangChain's MarkdownHeaderTextSplitter
#     and compare — learning by contrast.
#
# WHAT is a "unit" here:
#   - One unit = one section headed by a header (any level)
#   - The section includes everything until the next header of equal or higher level
#   - The metadata tracks the full breadcrumb path ("Chapter 1 > Section A > Subsection 1")

def load_markdown(path: Path) -> list[dict]:
    """
    Load a markdown file and return one dict per section.

    A "section" is the content under a header until the next header of
    equal or higher level. Preserves nested breadcrumb context.

    Each dict contains:
      - text: cleaned section content (including the header line)
      - metadata: dict with source, source_type, section_path,
                  header_level, char_count, ingested_at
    """
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {path}")

    content = path.read_text(encoding="utf-8")
    lines = content.split("\n")

    # Regex to detect headers: ^#{1,6}\s+(.*)
    # Example matches: "# Title", "## Subtitle", "### Sub-sub"
    header_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    # header_stack tracks the current breadcrumb path
    # Example state after seeing "# A", "## B", "### C": ["A", "B", "C"]
    # When we see "## D", we pop back to depth 1 → ["A", "D"]
    header_stack: list[str] = []

    # sections accumulate as we walk the file
    # Each section is: (section_path, header_level, list_of_lines)
    sections: list[tuple[list[str], int, list[str]]] = []
    current_lines: list[str] = []
    current_path: list[str] = []
    current_level: int = 0

    for line in lines:
        match = header_pattern.match(line)

        if match:
            # Flush the previous section before starting a new one
            if current_path and current_lines:
                sections.append((current_path.copy(), current_level, current_lines))

            level = len(match.group(1))       # 1 for #, 2 for ##, etc.
            title = match.group(2).strip()

            # Adjust the header stack to reflect the new depth
            # Trim deeper/equal-level entries, then push the new title
            header_stack = header_stack[: level - 1]
            header_stack.append(title)

            current_path = header_stack.copy()
            current_level = level
            current_lines = [line]  # Include the header line itself in the section
        else:
            if current_path:  # Only accumulate if we've seen a header
                current_lines.append(line)
            # Content before any header is intentionally discarded —
            # typically frontmatter, license headers, or empty space

    # Flush the final section
    if current_path and current_lines:
        sections.append((current_path.copy(), current_level, current_lines))

    # Build the output units with metadata
    units = []
    ingested_at = datetime.now(timezone.utc).isoformat()

    for idx, (path_parts, level, section_lines) in enumerate(sections):
        section_text = clean_text("\n".join(section_lines))

        if not section_text or len(section_text) < 20:
            continue

        section_path = " > ".join(path_parts)  # "Chapter 1 > Intro > Motivation"

        units.append({
            "text": section_text,
            "metadata": {
                "source": str(path),
                "source_type": "markdown",
                "section_path": section_path,
                "header_level": level,
                "section_index": idx,
                "char_count": len(section_text),
                "ingested_at": ingested_at,
            }
        })

    print(f"  [MD]  {path.name}")
    print(f"        Total sections extracted: {len(units)}")
    print(f"        Max header depth: {max((u['metadata']['header_level'] for u in units), default=0)}")

    return units


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS — look at what we got
# ═══════════════════════════════════════════════════════════════════════════════
# WHY this exists:
#   The whole point of today is SEEING what comes out of the loaders
#   before we chunk. You need to eyeball the text, spot the noise,
#   confirm the metadata is right. "It runs without errors" is not enough.

def describe_units(units: list[dict], label: str, preview_n: int = 2):
    """Print statistics and a preview of the first N units."""
    if not units:
        print(f"  ⚠️  {label}: no units extracted")
        return

    char_counts = [u["metadata"]["char_count"] for u in units]
    total_chars = sum(char_counts)

    print(f"\n  {label} — {len(units)} units")
    print(f"  {'─' * 60}")
    print(f"  Total chars:     {total_chars:,}")
    print(f"  Avg chars/unit:  {total_chars // len(units):,}")
    print(f"  Min chars:       {min(char_counts):,}")
    print(f"  Max chars:       {max(char_counts):,}")

    # Show the first N units with a text preview
    for i, unit in enumerate(units[:preview_n]):
        print(f"\n  ── Unit {i} ──")
        # Print metadata compactly
        meta = unit["metadata"]
        meta_summary = ", ".join(
            f"{k}={v}" for k, v in meta.items()
            if k in ("page", "section_path", "header_level", "char_count")
        )
        print(f"  Metadata: {meta_summary}")
        # Preview first 300 chars of text
        preview = unit["text"][:300].replace("\n", "\\n")
        print(f"  Text (first 300 chars):")
        print(f"    {preview}...")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print_header("DOCUMENT LOADERS — PDF + Markdown")

    # ── Load PDF ──
    print("Phase 1: Loading PDF with PyMuPDF")
    print("─" * 70)
    pdf_units = load_pdf(PDF_PATH)
    describe_units(pdf_units, "PDF units (1 per page)", preview_n=2)

    # ── Load Markdown ──
    print("\n\nPhase 2: Loading Markdown with structural parser")
    print("─" * 70)
    try:
        md_units = load_markdown(MD_PATH)
        describe_units(md_units, "Markdown units (1 per section)", preview_n=3)
    except FileNotFoundError as e:
        print(f"  ⚠️  {e}")
        print(f"      (If rag_fundamentals.md lives elsewhere, update MD_PATH)")
        md_units = []

    # ── Summary ──
    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  PDF units extracted:      {len(pdf_units)}")
    print(f"  Markdown units extracted: {len(md_units)}")
    print(f"  Total units ready for chunking: {len(pdf_units) + len(md_units)}")
    print("""
  NEXT STEP (13_chunking_strategies.py):
    Take these units and apply 4 chunking strategies:
      A) Fixed-size  — baseline, naive
      B) Recursive character — respects paragraph/sentence boundaries
      C) Sentence-based — splits on sentence terminators
      D) Structural — uses markdown headers (already partially done here)
    Compare chunk counts, sizes, semantic coherence.
""")


if __name__ == "__main__":
    main()