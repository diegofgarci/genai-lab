# GenAI Lab

Hands-on exploration of LLM APIs, prompt engineering, function calling, and agentic patterns. Progresses from basic API calls to an interactive multi-turn agent with state management.

## What This Does

This project builds up in layers: first, calling LLM APIs and comparing providers on speed, cost, and quality. Then, prompt engineering techniques to maximize output. Finally, function calling and agentic loops — teaching models to use tools and maintain conversational state across turns.

## Models Used

| Model | Provider | Speed | Cost (per 1M tokens in/out) |
|-------|----------|-------|-----------------------------|
| Claude Sonnet 4 | Anthropic | ~1-3s | $3.00 / $15.00 |
| Llama 3.3 70B | Groq | ~0.3-0.8s | $0.59 / $0.79 |
| Llama 3.1 8B | Groq | ~0.1-0.3s | $0.05 / $0.08 |

## Project Structure

```
ai-sprint/
├── utils.py                    # Shared clients, model registry, routing, error handling
├── 01_first_call.py            # First Claude API call
├── 02_multi_model.py           # First Groq/Llama call
├── 03_benchmark.py             # Multi-model benchmark with cost tracking
├── 04_prompt_engineering.py    # Zero-shot, few-shot, CoT, output formatting
├── 05_advanced_prompting.py    # Role prompting, chaining, self-consistency
├── 06_function_calling.py      # Claude tool use — complete call cycle
├── 08_complex_agent.py         # Multi-tool agent with task/contact/notification tools
├── 09_multi_turn_agent.py      # Interactive agent with conversation + session state
├── docs/
│   ├── benchmark_methodology.md
│   ├── prompt_engineering_techniques.md
│   └── function_calling_methodology.md
├── .env.example
└── .gitignore
```

## Setup

**Requirements:** Python 3.12+, API keys for Anthropic and Groq.

```bash
cd ~/Dev/ai-sprint

python3 -m venv .venv
source .venv/bin/activate

pip install anthropic openai groq python-dotenv

cp .env.example .env
# Edit .env with your keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   GROQ_API_KEY=gsk_...
```

## Usage

```bash
# --- Week 1: APIs + Prompt Engineering ---

# Basic API calls
python 01_first_call.py "What is a transformer?"
python 02_multi_model.py "What is a transformer?"

# Benchmark across all models
python 03_benchmark.py

# Prompt engineering experiments
python 04_prompt_engineering.py

# Advanced techniques (interactive menu)
python 05_advanced_prompting.py        # pick a block
python 05_advanced_prompting.py all    # run everything
python 05_advanced_prompting.py 2      # run block 2 only (prompt chaining)

# --- Week 2: Function Calling + Agents ---

# Function calling basics — tool use cycle
python 06_function_calling.py

# Complex multi-tool agent (scripted scenarios)
python 08_complex_agent.py

# Interactive multi-turn agent (REPL)
python 09_multi_turn_agent.py
```

## Key Findings

### Latency vs Quality Tradeoff

Groq (Llama) is 3-10x faster than Claude for simple tasks. For classification, extraction, and short-form responses, Llama 70B on Groq delivers comparable quality at a fraction of the latency and cost. Claude consistently outperforms on nuanced reasoning, multi-step logic, and constraint adherence.

### When to Use Each Model

**Claude Sonnet 4** — Complex reasoning, strict output formatting (JSON schemas), creative analysis, prompt chaining pipelines where quality compounds across steps. Function calling and tool use are more reliable. Higher cost is justified when errors are expensive.

**Llama 70B (Groq)** — High-throughput classification, first-pass extraction, latency-sensitive applications, tasks where you can validate output programmatically. Best bang-for-buck at scale.

**Llama 8B (Groq)** — Prototyping, simple completions, high-volume low-stakes tasks. Fast enough for real-time applications. Quality drops noticeably on complex instructions.

### Prompt Engineering Impact

| Technique | Claude | Llama 70B | Llama 8B |
|-----------|--------|-----------|----------|
| Few-shot examples | Moderate lift | High lift | Critical |
| Chain-of-thought | High lift | Moderate lift | Unreliable |
| JSON constraints | Near-perfect | Good (occasional fences) | Often breaks |
| Role prompting | Distinct personas | Follows but less nuanced | Minimal effect |
| Length constraints | Reliable | Approximate | Often ignores |

### Function Calling Insights

Tool description quality is the primary driver of model decision-making. The model matches user intent to tool descriptions via semantic similarity — it's not conditional logic, it's language understanding. Key lessons:

1. **Tool descriptions are prompts** — vague descriptions lead to wrong tool selection. Include WHEN to use each tool, not just what it does.
2. **The model outputs structured JSON; your code executes** — this separation is foundational. The model never runs tools directly.
3. **Multi-step tool chains work** — the model can call weather, then calculator, in sequence within a single turn. Each tool result feeds the next decision.
4. **Conversation state vs session state** — the messages array is ephemeral context (grows with every turn, gets expensive). Session state is persistent business data (compact, survives conversation resets). Separating them is the architecture that scales.

### Production Recommendations

1. **Use a model router** — match task complexity to model capability. Don't pay Claude prices for sentiment classification.
2. **JSON output needs validation** — even Claude occasionally wraps JSON in markdown. Always parse defensively (see `utils.parse_json_response`).
3. **Self-consistency works** — for ambiguous classification, running 3-5 times with high temperature and voting is more reliable than a single call at temperature=0.
4. **Prompt templates are business assets** — version them, test them, measure them. A 10% improvement in prompt quality across 100K daily calls is worth more than switching models.
5. **Track token accumulation** — in multi-turn agents, input tokens grow linearly with conversation length because every turn resends the full history. Monitor costs per session, implement truncation or summarization strategies.
6. **Error handling must distinguish error types** — auth errors (don't retry), rate limits and overloaded (retry with backoff), timeouts (retry), unexpected errors (log and skip). The strategy per error type is different.

## Architecture

### Week 1: Multi-Model Benchmark

```
┌─────────────────────────────────────────────────────┐
│                  Scripts (01-05)                      │
│  Each script focuses on one concept/experiment       │
└──────────────────────┬──────────────────────────────┘
                       │ import
┌──────────────────────▼──────────────────────────────┐
│                     utils.py                         │
│  Model Registry · Unified Router · Error Handling    │
│  Cost Calculation · Display Helpers · JSON Parsing   │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Anthropic│ │   Groq   │ │   Groq   │
    │  Claude  │ │ Llama 70B│ │ Llama 8B │
    └──────────┘ └──────────┘ └──────────┘
```

### Week 2: Agent Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     REPL Loop (main)                      │
│  Reads user input · Local commands (/tokens, /state)     │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────┐
│                  process_turn()                           │
│  Sends full messages[] to API · Handles tool chains      │
│                                                          │
│  ┌─────────────┐     ┌──────────────────┐                │
│  │ Claude API   │────▶│ stop_reason?      │                │
│  │ (+ tools)    │     └──┬───────────┬───┘                │
│  └─────────────┘        │           │                    │
│                  end_turn│    tool_use│                    │
│                         ▼           ▼                    │
│                  ┌──────────┐ ┌─────────────────┐        │
│                  │ Return   │ │ Execute tool     │        │
│                  │ text     │ │ Append results   │──┐     │
│                  └──────────┘ │ to messages[]    │  │     │
│                               └─────────────────┘  │     │
│                                        ▲           │     │
│                                        └───────────┘     │
└──────────────────────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Conversation │ │ Session      │ │ Token        │
│ State        │ │ State        │ │ Tracking     │
│ messages[]   │ │ notes,       │ │ input/output │
│ (sent to API)│ │ contacts,    │ │ cost calc    │
│              │ │ preferences  │ │              │
└──────────────┘ └──────────────┘ └──────────────┘
```

## Documentation

Standalone methodology docs in `docs/` — no course references, reusable by any team:

- **benchmark_methodology.md** — how to compare LLM providers, what metrics matter, cost modeling
- **prompt_engineering_techniques.md** — full toolkit from zero-shot to multi-step chaining
- **function_calling_methodology.md** — tool use patterns, SDK differences, context growth, error strategies

## What I Learned

**Week 1** covered the full stack of LLM API integration: SDK differences between providers, unified abstraction patterns, error handling strategies, cost modeling, and the entire prompt engineering toolkit from zero-shot to multi-step chaining. The biggest insight: prompt engineering is not about clever tricks — it's about decomposing problems into tasks that match model capabilities.

**Week 2** moved from "calling models" to "building with models." Function calling introduced the pattern where the model decides WHAT to do (structured JSON output) and your code decides HOW to do it (local execution). The multi-turn agent brought two critical concepts: conversation state (the messages array that grows with every turn and gives the model memory) and session state (business data that tools accumulate, invisible to the model until queried). This separation — what was said vs what data exists — is the architectural foundation for production agents.