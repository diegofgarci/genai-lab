"""
05_advanced_prompting.py — Advanced Prompting Techniques
=========================================================
Day 4: Role prompting, prompt chaining, self-consistency,
prompt templates, and output constraints.

Blocks:
  1. Role Prompting — same prompt, different roles → compare outputs
  2. Prompt Chaining — multi-step pipeline where each output feeds the next
  3. Self-Consistency — same question N times, aggregate for reliability
  4. Prompt Templates — reusable functions with variables
  5. Output Constraints — force specific structure (JSON, length, tables)

Usage:
    python 05_advanced_prompting.py        # select a block
    python 05_advanced_prompting.py all    # run everything
"""

import sys
import json
from collections import Counter
from utils import MODELS, call_model, print_header, print_result, parse_json_response


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 1 — ROLE PROMPTING
# ═══════════════════════════════════════════════════════════════════════════════
# The system prompt defines the model's "character". Same user prompt,
# different roles → completely different responses in tone, focus, and depth.
# This is THE most powerful lever to control output.

def demo_role_prompting():
    print_header("BLOCK 1 — ROLE PROMPTING")

    user_prompt = (
        "A startup wants to use AI to automate legal contract review. "
        "What do you think? Give your analysis in 3-4 paragraphs."
    )

    roles = {
        "VC Partner": (
            "You are a partner at a tier-1 venture capital fund. "
            "You evaluate startup investment opportunities. "
            "You care about TAM, defensibility, market timing, and team. "
            "You are direct and dislike ideas without clear differentiation."
        ),
        "Corporate Lawyer": (
            "You are a senior corporate lawyer with 20 years of M&A experience. "
            "You worry about regulatory risks, legal liability, "
            "contract analysis accuracy, and ethical implications. "
            "You are meticulous and skeptical of simplistic tech solutions."
        ),
        "Startup CTO": (
            "You are the CTO of a seed-stage AI startup. "
            "You think about technical architecture, which models to use, how to build "
            "the MVP as fast as possible, and what tech debt is acceptable early on. "
            "You are pragmatic and shipping-oriented."
        ),
    }

    print("User prompt (identical for all roles):")
    print(f'  "{user_prompt}"\n')

    for role_name, system in roles.items():
        print(f"{'─'*70}")
        print(f"  ROLE: {role_name}")
        print(f"{'─'*70}")
        result = call_model(
            "claude", user_prompt, system_prompt=system,
            temperature=0.7, max_tokens=600,
        )
        if result:
            print_result(result)


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2 — PROMPT CHAINING
# ═══════════════════════════════════════════════════════════════════════════════
# Decompose a complex task into sequential steps. Each step's output becomes
# the next step's input. This is the CONCEPTUAL BASIS for what you'll later
# build with agents (LangGraph, CrewAI).
#
# Pipeline: Extract data → Analyze → Generate recommendation

def demo_prompt_chaining():
    print_header("BLOCK 2 — PROMPT CHAINING")

    source_text = """
    Q4 2025 Report — CloudSync Inc.
    Revenue: $4.2M (up 35% YoY). Gross margin: 72%. Net burn: $800K/month.
    Runway: 14 months. ARR: $16.8M. NDR: 118%. Logo churn: 3.2%/quarter.
    Team: 45 people (28 engineering). Enterprise clients: 12 (up from 7).
    Key risk: 40% of revenue from top 2 clients. CAC payback: 18 months.
    Product: AI-powered data integration platform. NPS: 62.
    Raised Series A ($15M) in March 2025 at $60M pre-money valuation.
    Competitors: Fivetran, Airbyte, plus 3 AI-native startups with <$1M ARR.
    """

    print("Source text:")
    print(source_text)

    # ── Step 1: Extract key metrics as JSON ──
    print(f"\n{'─'*70}")
    print("  STEP 1 → Structured data extraction")
    print(f"{'─'*70}")

    step1_prompt = f"""Extract the key metrics from the following text and return ONLY valid JSON.
Required structure:
{{
    "company": "name",
    "revenue_q4": "value",
    "growth_yoy": "value",
    "gross_margin": "value",
    "burn_rate": "value",
    "runway_months": number,
    "arr": "value",
    "ndr": "value",
    "team_size": number,
    "enterprise_clients": number,
    "top_client_concentration": "value",
    "valuation": "value"
}}

Text:
{source_text}"""

    step1 = call_model(
        "claude", step1_prompt,
        system_prompt="You are a data analyst. Respond ONLY with valid JSON, no markdown.",
        temperature=0.0,
    )
    if not step1:
        print("  Step 1 failed. Aborting chain.")
        return

    print_result(step1, "Extraction")

    extracted_data = parse_json_response(step1["text"])
    if extracted_data:
        print("  ✅ JSON parsed successfully\n")
    else:
        print("  ⚠️  Could not parse JSON. Using raw text for next step.\n")
        extracted_data = step1["text"]

    # ── Step 2: Analyze using extracted data ──
    print(f"{'─'*70}")
    print("  STEP 2 → Analysis (input = output from step 1)")
    print(f"{'─'*70}")

    data_str = json.dumps(extracted_data, indent=2) if isinstance(extracted_data, dict) else extracted_data

    step2_prompt = f"""Analyze the following SaaS startup metrics.
Identify: 3 strengths, 3 risks, and 1 metric that needs additional context.
Be specific — use the numbers, not generalities.

Metrics:
{data_str}"""

    step2 = call_model(
        "claude", step2_prompt,
        system_prompt="You are a venture capital analyst. Be concise and direct.",
        temperature=0.5, max_tokens=600,
    )
    if not step2:
        print("  Step 2 failed. Aborting chain.")
        return

    print_result(step2, "Analysis")

    # ── Step 3: Recommendation based on analysis ──
    print(f"{'─'*70}")
    print("  STEP 3 → Recommendation (input = output from step 2)")
    print(f"{'─'*70}")

    step3_prompt = f"""Based on the following analysis of CloudSync Inc., generate an investment
recommendation in exactly this format:

VERDICT: [INVEST / PASS / WATCH]
THESIS (2 sentences): [why]
CONDITIONS: [what would have to change to flip your verdict]
NEXT STEP: [one concrete action]

Analysis:
{step2['text']}"""

    step3 = call_model(
        "claude", step3_prompt,
        system_prompt="You are a VC partner making investment decisions.",
        temperature=0.3, max_tokens=400,
    )
    if step3:
        print_result(step3, "Recommendation")

    print("  CHAIN SUMMARY:")
    print("  Raw text → Structured JSON → Analysis → Actionable recommendation")
    print("  3 model calls, each specialized in one subtask.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3 — SELF-CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════
# Ask the same question N times with temperature > 0, then aggregate.
# Useful for: classification, numeric estimation, decisions where you want
# to measure model "confidence".

def demo_self_consistency():
    print_header("BLOCK 3 — SELF-CONSISTENCY")

    prompt = """Classify the following text into exactly ONE category.
Possible categories: POSITIVE, NEGATIVE, NEUTRAL
Respond with ONLY the category, one single word.

Text: "The product works fine but the customer support experience
was frustrating. I'd probably buy again though."
"""

    n_runs = 5
    results = {"claude": [], "llama-70b": []}

    print(f"Running {n_runs} times with temperature=0.9 (high variability)...\n")

    for model_key in results:
        for i in range(n_runs):
            result = call_model(model_key, prompt, temperature=0.9, max_tokens=10)
            if not result:
                results[model_key].append("FAILED")
                continue

            classification = result["text"].strip().upper()
            # Clean — sometimes the model adds punctuation
            for cat in ["POSITIVE", "NEGATIVE", "NEUTRAL"]:
                if cat in classification:
                    classification = cat
                    break
            results[model_key].append(classification)

    # Show results and consensus
    for model_key, classifications in results.items():
        label = MODELS[model_key]["label"]
        print(f"  {label}:")
        print(f"    Responses: {classifications}")

        votes = Counter(classifications)
        winner = votes.most_common(1)[0]
        confidence = winner[1] / n_runs * 100

        print(f"    Consensus: {winner[0]} ({confidence:.0f}% — {winner[1]}/{n_runs})")
        print(f"    Distribution: {dict(votes)}")
        print()

    print("  INSIGHT: If consensus is low (< 80%), the text is genuinely")
    print("  ambiguous OR the prompt needs more context/examples.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 4 — PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════
# Reusable functions that generate consistent prompts. In production
# this becomes your "prompt library" — company assets.

def demo_prompt_templates():
    print_header("BLOCK 4 — PROMPT TEMPLATES")

    # ── Template 1: Competitive analysis ──
    def competitive_analysis_prompt(
        company: str, competitors: list[str], dimension: str
    ) -> str:
        comp_list = ", ".join(competitors)
        return f"""Analyze {company} vs its competitors ({comp_list})
on the dimension of: {dimension}.

Response format:
| Company | {dimension} | Advantage/Disadvantage |
Then 1 paragraph conclusion on who leads and why."""

    # ── Template 2: Executive summary ──
    def executive_summary_prompt(
        topic: str, audience: str, max_bullets: int = 5
    ) -> str:
        return f"""Generate an executive summary on: {topic}
Audience: {audience}
Format: maximum {max_bullets} bullet points.
Each bullet must be actionable — no generic observations.
Start directly, no introduction."""

    # ── Template 3: Risk assessment ──
    def risk_assessment_prompt(project: str, context: str) -> str:
        return f"""Assess the risks of the following project:
Project: {project}
Context: {context}

For each risk identify:
- Risk (1 line)
- Probability: HIGH / MEDIUM / LOW
- Impact: HIGH / MEDIUM / LOW
- Mitigation (1 concrete line)

Identify exactly 4 risks. Be specific, not generic."""

    templates = [
        (
            "Competitive Analysis",
            competitive_analysis_prompt(
                "Anthropic", ["OpenAI", "Google DeepMind"], "developer experience"
            ),
        ),
        (
            "Executive Summary",
            executive_summary_prompt(
                "AI agent adoption in enterprise 2026",
                "C-suite at a Fortune 500 company",
                4,
            ),
        ),
        (
            "Risk Assessment",
            risk_assessment_prompt(
                "Migrate the contract system to AI-assisted review",
                "500-person company, regulated financial sector, "
                "currently fully manual with an 8-person legal team",
            ),
        ),
    ]

    for template_name, prompt in templates:
        print(f"{'─'*70}")
        print(f"  TEMPLATE: {template_name}")
        print(f"{'─'*70}")
        print(f"  Generated prompt ({len(prompt)} chars):\n")
        for line in prompt.split("\n"):
            print(f"    {line}")
        print()

        result = call_model("claude", prompt, temperature=0.5, max_tokens=500)
        if result:
            print_result(result)

    print("  IN PRODUCTION: These templates go in a separate module,")
    print("  get versioned, and A/B tested. They are business assets.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 5 — OUTPUT CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════
# Force the model to return exactly the structure you need.
# Critical for integration with downstream systems.

def demo_output_constraints():
    print_header("BLOCK 5 — OUTPUT CONSTRAINTS")

    # ── Constraint 1: Strict JSON schema ──
    print(f"{'─'*70}")
    print("  CONSTRAINT: Strict JSON Schema")
    print(f"{'─'*70}")

    json_prompt = """Analyze the following product and return ONLY valid JSON
(no markdown, no backticks, no additional text).

Product: "Notion AI — writing and organization assistant integrated into Notion"

Required schema:
{
    "product_name": string,
    "category": "productivity" | "communication" | "analytics" | "security",
    "target_market": string,
    "strengths": [string, string, string],
    "weaknesses": [string, string],
    "threat_level_to_incumbents": "low" | "medium" | "high",
    "estimated_tam_usd": string,
    "one_line_verdict": string
}"""

    result = call_model(
        "claude", json_prompt, temperature=0.0, max_tokens=400,
        system_prompt="Respond EXCLUSIVELY with valid JSON.",
    )
    if result:
        print_result(result, "JSON Schema")

        parsed = parse_json_response(result["text"])
        if parsed:
            required_keys = ["product_name", "category", "strengths", "threat_level_to_incumbents"]
            missing = [k for k in required_keys if k not in parsed]
            if missing:
                print(f"  ⚠️  Missing keys: {missing}")
            else:
                print(f"  ✅ Valid JSON with all required keys")
                print(f"  Category: {parsed.get('category')}")
                print(f"  Threat level: {parsed.get('threat_level_to_incumbents')}")
        else:
            print(f"  ❌ Invalid JSON — model did not respect the constraint")
    print()

    # ── Constraint 2: Exact length ──
    print(f"{'─'*70}")
    print("  CONSTRAINT: Controlled length")
    print(f"{'─'*70}")

    length_prompt = """Explain what RAG (Retrieval-Augmented Generation) is in EXACTLY 3 sentences.
No more, no less. Each sentence must end with a period.
The first sentence defines what it is.
The second explains how it works.
The third says why it matters."""

    for model_key in ["claude", "llama-70b"]:
        result = call_model(model_key, length_prompt, temperature=0.3, max_tokens=200)
        if result:
            print_result(result, "3 sentences")
            sentences = [s.strip() for s in result["text"].split(".") if s.strip()]
            print(f"  Sentences detected: {len(sentences)}\n")

    # ── Constraint 3: Table format ──
    print(f"{'─'*70}")
    print("  CONSTRAINT: Tabular format")
    print(f"{'─'*70}")

    table_prompt = """Compare these 3 AI agent frameworks.
Respond ONLY with a markdown table. No text before or after.
Exact columns: Framework | Best for | Learning curve | Production-ready

Frameworks: LangGraph, CrewAI, Claude Agent SDK"""

    result = call_model("claude", table_prompt, temperature=0.3, max_tokens=300)
    if result:
        print_result(result, "Table")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

DEMOS = {
    "1": ("Role Prompting", demo_role_prompting),
    "2": ("Prompt Chaining", demo_prompt_chaining),
    "3": ("Self-Consistency", demo_self_consistency),
    "4": ("Prompt Templates", demo_prompt_templates),
    "5": ("Output Constraints", demo_output_constraints),
}


def main():
    print("\n" + "█" * 70)
    print("  05_advanced_prompting.py — Advanced Techniques")
    print("  Models: Claude Sonnet 4 + Llama 70B (Groq)")
    print("█" * 70)

    print("\nAvailable blocks:")
    for key, (name, _) in DEMOS.items():
        print(f"  {key}. {name}")
    print(f"  all. Run everything")

    # Accept CLI argument or interactive input
    choice = (
        sys.argv[1].strip().lower()
        if len(sys.argv) > 1
        else input("\nWhich block to run? (1-5 / all): ").strip().lower()
    )

    if choice == "all":
        for key, (name, func) in DEMOS.items():
            func()
    elif choice in DEMOS:
        DEMOS[choice][1]()
    else:
        print("Invalid option. Use 1-5 or 'all'.")


if __name__ == "__main__":
    main()
