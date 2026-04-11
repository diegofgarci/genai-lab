# Multi-Model Benchmark Tool

Compare LLM providers (Anthropic Claude, Groq/Llama) across latency, cost, token usage, and response quality — then apply prompt engineering techniques to maximize output from each.

## What This Does

This toolkit runs the same prompts across multiple LLM APIs and measures what matters: how fast they respond, how much they cost, and how well they follow instructions. It progresses from basic API calls to advanced prompting techniques.

## Models Tested

| Model | Provider | Speed | Cost (per 1M tokens in/out) |
|-------|----------|-------|-----------------------------|
| Claude Sonnet 4 | Anthropic | ~1-3s | $3.00 / $15.00 |
| Llama 3.3 70B | Groq | ~0.3-0.8s | $0.59 / $0.79 |
| Llama 3.1 8B | Groq | ~0.1-0.3s | $0.05 / $0.08 |

## Project Structure

```
ai-sprint/
├── utils.py                    # Shared clients, model registry, routing, error handling
├── 01_first_call.py            # Day 1 — First Claude API call
├── 02_multi_model.py           # Day 1 — First Groq/Llama call
├── 03_benchmark.py             # Day 2 — Multi-model benchmark with cost tracking
├── 04_prompt_engineering.py    # Day 3 — Zero-shot, few-shot, CoT, output formatting
├── 05_advanced_prompting.py    # Day 4 — Role prompting, chaining, self-consistency
├── .env                        # API keys (not committed)
└── .gitignore
```

## Setup

**Requirements:** Python 3.12+, API keys for Anthropic and Groq.

```bash
# Clone and enter the project
cd ~/Dev/ai-sprint

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install anthropic openai groq python-dotenv

# Configure API keys
cp .env.example .env
# Edit .env with your keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   GROQ_API_KEY=gsk_...
```

## Usage

```bash
# Day 1 — Basic API calls
python 01_first_call.py "What is a transformer?"
python 02_multi_model.py "What is a transformer?"

# Day 2 — Run benchmark across all models
python 03_benchmark.py

# Day 3 — Prompt engineering experiments
python 04_prompt_engineering.py

# Day 4 — Advanced techniques (interactive menu)
python 05_advanced_prompting.py        # pick a block
python 05_advanced_prompting.py all    # run everything
python 05_advanced_prompting.py 2      # run block 2 only (prompt chaining)
```

## Key Findings

### Latency vs Quality Tradeoff

Groq (Llama) is 3-10x faster than Claude for simple tasks. For classification, extraction, and short-form responses, Llama 70B on Groq delivers comparable quality at a fraction of the latency and cost. Claude consistently outperforms on nuanced reasoning, multi-step logic, and constraint adherence.

### When to Use Each Model

**Claude Sonnet 4** — Complex reasoning, strict output formatting (JSON schemas), creative analysis, prompt chaining pipelines where quality compounds across steps. Higher cost is justified when errors are expensive.

**Llama 70B (Groq)** — High-throughput classification, first-pass extraction, latency-sensitive applications, tasks where you can validate output programmatically. Best bang-for-buck at scale.

**Llama 8B (Groq)** — Prototyping, simple completions, high-volume low-stakes tasks. Fast enough for real-time applications. Quality drops noticeably on complex instructions.

### Prompt Engineering Impact

Technique effectiveness varies by model:

| Technique | Claude | Llama 70B | Llama 8B |
|-----------|--------|-----------|----------|
| Few-shot examples | Moderate lift | High lift | Critical |
| Chain-of-thought | High lift | Moderate lift | Unreliable |
| JSON constraints | Near-perfect | Good (occasional fences) | Often breaks |
| Role prompting | Distinct personas | Follows but less nuanced | Minimal effect |
| Length constraints | Reliable | Approximate | Often ignores |

### Production Recommendations

1. **Use a model router** — match task complexity to model capability. Don't pay Claude prices for sentiment classification.
2. **JSON output needs validation** — even Claude occasionally wraps JSON in markdown. Always parse defensively (see `utils.parse_json_response`).
3. **Self-consistency works** — for ambiguous classification, running 3-5 times with high temperature and voting is more reliable than a single call at temperature=0.
4. **Prompt templates are business assets** — version them, test them, measure them. A 10% improvement in prompt quality across 100K daily calls is worth more than switching models.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Scripts (01-05)                    │
│  Each script focuses on one concept/experiment       │
└──────────────────────┬──────────────────────────────┘
                       │ import
┌──────────────────────▼──────────────────────────────┐
│                     utils.py                         │
│  ┌─────────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ Model        │  │ Unified  │  │ Error Handling │  │
│  │ Registry     │  │ Router   │  │ + Retry Logic  │  │
│  └─────────────┘  └──────────┘  └────────────────┘  │
│  ┌─────────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ Cost Calc   │  │ Display  │  │ JSON Parsing   │  │
│  │             │  │ Helpers  │  │                │  │
│  └─────────────┘  └──────────┘  └────────────────┘  │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Anthropic│ │   Groq   │ │   Groq   │
    │  Claude  │ │ Llama 70B│ │ Llama 8B │
    └──────────┘ └──────────┘ └──────────┘
```

## What I Learned

This project covered the full stack of LLM API integration: SDK differences between providers, unified abstraction patterns, error handling strategies (auth vs rate limit vs timeout), cost modeling, and the entire prompt engineering toolkit from zero-shot to multi-step chaining. The biggest insight: **prompt engineering is not about clever tricks — it's about decomposing problems into tasks that match model capabilities.**
