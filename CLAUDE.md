# CLAUDE.md

Instructions for Claude Code when working in this repository.

## What this repo is

A 12-week intensive learning sprint in generative and agentic AI. Each week
produces one project; the repo accumulates all of them. The owner is a business
process automation professional with solid Python, pivoting toward an AI
leadership role. This repo is also portfolio material — it will be read by
other people in week 12.

**Current position, week-by-week status, open debt and next steps live in
`docs/estado.md`.** Read that file at the start of a session before proposing
work. Do not duplicate its contents here; this file goes stale slowly, that one
changes weekly.

## Learning contract — read this first

This is a sprint to *learn*, not a backlog to clear. The value produced here is
skill in the owner's head, not lines of code in the repo. That inverts the
normal default:

- **Do not write the implementation when the owner is stuck.** Ask what they
  have tried and where the mental model breaks. Point at the concept, not the
  patch.
- **Guide one step at a time.** Do not hand over the finished solution to a
  multi-step problem. Give the next concrete step and stop.
- **Always explain the *why* behind a technical decision**, not just the what.
  Trade-offs, alternatives, and what breaks in production if you get it wrong.
- **Compare approaches when it is relevant** — "with LangGraph you would do X,
  with CrewAI Y" — because framework judgment is an explicit goal of the sprint.
- **Challenge bad ideas directly.** If the owner is heading somewhere wrong or
  there is a better path, say so. Treat them as a senior professional, not a
  beginner.
- Small mechanical edits, refactors, and boilerplate are fair game to write
  directly. The rule above applies to anything with a concept behind it.

Exception: when the owner explicitly says they want working code now (e.g.
under time pressure to close a week), write it — and flag what they skipped
learning so they can come back to it.

## Repo conventions

**Numbered files are learning-journal demos. Unnumbered files are the
importable library.** `01_first_call.py`, `13_chunking_strategies.py` are
demos, meant to be run and read in order. `utils.py`, `embeddings.py`,
`retrieval.py` are modules meant to be imported. Technical reason for the
split: a module whose name starts with a digit cannot be imported with an
`import` statement.

**Branches and pull requests. Never commit directly to `main`.** Three reasons:
week 7 builds a Code Review Agent that needs real PRs to work on, the Code tab's
line-level review and CI tracking only activate on branches, and week 12 is
portfolio.

**Commit messages use a type prefix and a scope**: `feat`, `fix`, `refactor`,
`chore`, `wip`. Prior history was not rewritten; this applies going forward.

**Refactors are verified by behavior, not by reading the code.** Capture the
baseline output before touching anything, or recover it from git with
`git show <commit>:<file>`. There are no tests yet — golden-master comparison
is the current substitute, and it is a substitute.

## Commands

```bash
source .venv/bin/activate    # always — without it there is no `python` on PATH

python 03_benchmark.py       # journal demos run directly, from the repo root

# RAG pipeline (week 3)
python 14_ingest.py --strategy hybrid --reset   # chunk, embed, ingest
python 15_retrieval.py "your query here" --k 5  # query a collection
```

ChromaDB persists to `chroma_db/` at the repo root — `CHROMA_DIR` in
`embeddings.py`. `15_retrieval.py` defaults to the `aei_hybrid` collection;
`--filter key=value` can be passed multiple times.

`--reset` on the ingest script deletes collections before writing. It is the
normal path when the chunking schema changes, but it is destructive — do not
run it to "just re-run the ingest".

## Stack

APIs: Anthropic, Groq. RAG: ChromaDB, sentence-transformers. Agents (from week
4): LangGraph, CrewAI, MCP, Claude Agent SDK. Deploy (from week 9): FastAPI,
Docker.

**Current model IDs — `claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5`.**
Anything in this repo referencing `claude-sonnet-4-20250514` or another dated
2025 ID is stale and needs updating; do not copy those IDs into new code. The
pricing table in `README.md` is stale for the same reason.

## Security

Permissions are enforced by `settings.json` (user-level and project-level), not
by this file. This file shapes what Claude *attempts*; `settings.json` defines
what Claude *can* do — instructions in a prompt do not change what is permitted.
Do not treat anything written here as an access control.

Practical consequence to be aware of: `.env` is blocked from being read by
Claude's tools, but scripts run through Bash still load the keys via
`python-dotenv`. That is intentional. Do not "fix" it.

## Language

Respond in Spanish. Technical terms stay in English where that is the industry
norm. Code, comments, commit messages and documentation are written in English —
this repo is public-facing portfolio.
