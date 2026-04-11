# LLM Multi-Model Benchmark — Results & Methodology

> Team best practice document. Covers how we benchmark LLM providers, what we measured, and when to use each model in production.

---

## Why Benchmark

Choosing an LLM provider based on vibes or marketing is how teams end up overpaying 20x for tasks a smaller model handles fine. Benchmarking answers three questions: **Is the quality good enough?** **Is the speed acceptable?** **Is the cost justified?** The answer is always "it depends on the task," which is why we test across task categories, not just one prompt.

---

## Models Evaluated

| Model | Provider | Parameters | Hosting | Pricing (per 1M tokens) |
|-------|----------|------------|---------|------------------------|
| Claude Sonnet 4 | Anthropic | Undisclosed | Anthropic Cloud | $3.00 in / $15.00 out |
| Llama 3.3 70B | Groq | 70B | Groq Cloud (LPU) | $0.59 in / $0.79 out |
| Llama 3.1 8B | Groq | 8B | Groq Cloud (LPU) | $0.05 in / $0.08 out |

**Why these models:** Claude represents the frontier commercial tier. Llama 70B is the strongest open-source option on fast inference. Llama 8B tests whether a small model is "good enough" for simple tasks. This covers the full cost-performance spectrum.

---

## Methodology

### Test Categories

We tested across five task types that map to real production use cases:

| Category | What It Tests | Production Analogy |
|----------|--------------|-------------------|
| Short-form Q&A | Instruction following, conciseness | Chatbot, FAQ |
| Sentiment classification | Consistency, label accuracy | Ticket triage, feedback analysis |
| Data extraction (JSON) | Format adherence, completeness | ETL pipelines, form parsing |
| Multi-step reasoning | Accuracy under complexity | Financial analysis, code review |
| Creative analysis | Nuance, depth, perspective | Strategy docs, advisory |

### Controls

All tests used `temperature=0.0` for deterministic comparison (except self-consistency experiments which intentionally used `temperature=0.9`). System prompts and user prompts were identical across models. Each model got the same `max_tokens=1024`. Latency was measured wall-clock from request to response, including network overhead.

### Metrics Collected

For every call: response text, input tokens, output tokens, wall-clock latency (seconds), and calculated cost (USD). Quality was assessed manually against expected outputs — there is no automated eval in this phase (that comes in Week 8 with Ragas).

---

## Results Summary

### Latency

| Task Type | Claude Sonnet 4 | Llama 70B (Groq) | Llama 8B (Groq) |
|-----------|-----------------|-------------------|------------------|
| Short Q&A (2 sentences) | 1.2-2.5s | 0.3-0.6s | 0.1-0.3s |
| JSON extraction | 1.5-3.0s | 0.4-0.8s | 0.2-0.4s |
| Multi-step reasoning | 2.0-4.0s | 0.5-1.2s | 0.3-0.6s |
| Creative analysis (600 tokens) | 3.0-5.0s | 0.8-1.5s | 0.4-0.8s |

**Takeaway:** Groq's LPU hardware delivers 3-10x faster inference. For latency-sensitive applications (real-time chat, high-throughput pipelines), this matters more than marginal quality differences.

### Quality

| Task Type | Claude Sonnet 4 | Llama 70B (Groq) | Llama 8B (Groq) |
|-----------|-----------------|-------------------|------------------|
| Short Q&A | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| Sentiment classification | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| JSON constraint adherence | ★★★★★ | ★★★★☆ | ★★☆☆☆ |
| Multi-step reasoning | ★★★★★ | ★★★☆☆ | ★★☆☆☆ |
| Creative/analytical depth | ★★★★★ | ★★★☆☆ | ★☆☆☆☆ |

**Takeaway:** Claude leads on every category but the gap narrows on simpler tasks. Llama 70B is a viable production option for classification and extraction. Llama 8B should only be used for simple completions where you validate output programmatically.

### Cost per 1K Requests (estimated, avg 500 tokens out)

| Model | Cost per 1K requests |
|-------|---------------------|
| Claude Sonnet 4 | ~$7.80 |
| Llama 70B (Groq) | ~$0.45 |
| Llama 8B (Groq) | ~$0.05 |

**Takeaway:** Claude is 17x more expensive than Llama 70B. At scale (100K+ daily requests), the cost difference is the primary decision driver, not quality.

---

## Prompt Engineering Findings

### Techniques That Move the Needle

**Few-shot examples** had the single largest impact on smaller models. Llama 8B went from unreliable to usable on classification tasks just by adding 3 examples. Claude benefited less because it already infers the expected format from instructions alone.

**Chain-of-thought** ("think step by step") significantly improved Claude's accuracy on multi-step math and logic. Llama 70B showed moderate improvement. Llama 8B produced longer but not more accurate reasoning — the model lacks the capacity to benefit from CoT on hard problems.

**Output constraints** (JSON schema, exact length, tables) were respected most reliably by Claude. Llama 70B usually complied but occasionally wrapped JSON in markdown fences. Llama 8B frequently broke constraints. Production systems should always validate output structure regardless of model.

**Role prompting** produced dramatically different outputs from the same prompt on Claude — a VC, lawyer, and CTO gave genuinely distinct analyses. Llama 70B followed the role but with less nuance. This technique is highest-leverage for applications that need multiple perspectives.

**Self-consistency** (N runs + voting) improved classification reliability for ambiguous inputs. Running 5 times at high temperature and majority-voting gave more stable results than a single deterministic call. Cost is 5x but confidence is measurable.

### Techniques Ranked by Production Value

| Rank | Technique | Why |
|------|-----------|-----|
| 1 | Prompt templates (reusable, versioned) | Consistency at scale, measurable improvement |
| 2 | Output constraints (JSON, schema) | Machine-parseable output, pipeline integration |
| 3 | Few-shot examples | Biggest quality lift for smallest effort |
| 4 | Prompt chaining (multi-step) | Complex tasks decomposed into reliable subtasks |
| 5 | Role prompting | Perspective control, multi-stakeholder analysis |
| 6 | Chain-of-thought | Reasoning accuracy, but adds latency and cost |
| 7 | Self-consistency | Confidence scoring, but expensive at scale |

---

## Decision Framework — When to Use Each Model

```
Is the task simple classification, extraction, or short-form?
  YES → Can you validate output programmatically?
          YES → Llama 8B (cheapest, fastest)
          NO  → Llama 70B (good quality, fast)
  NO  → Does it require multi-step reasoning or strict format adherence?
          YES → Claude (highest reliability)
          NO  → Does latency matter more than depth?
                  YES → Llama 70B
                  NO  → Claude
```

### The Hybrid Approach (Recommended)

Use a model router that matches task complexity to model capability. In a production pipeline: Llama 8B for initial triage and filtering, Llama 70B for extraction and classification, Claude for final analysis and generation where quality matters. This can cut costs 80%+ versus using Claude for everything while maintaining quality where it counts.

---

## Limitations & Next Steps

**What this benchmark does NOT measure:** Long-context performance (tested only short prompts), multi-turn conversation quality, function calling / tool use (covered in Week 2), streaming latency (TTFB), or automated quality evaluation (covered in Week 8 with Ragas).

**Infrastructure gaps:** No automated eval pipeline yet. Quality was assessed manually. Building programmatic evaluation is critical before scaling any of these patterns.

**Next iteration:** Add Gemma and Mixtral models via Groq. Implement automated scoring. Test with domain-specific prompts (legal, financial, technical) instead of generic ones.
