"""
04_prompt_engineering.py — Core Prompt Engineering Techniques
==============================================================
Day 3: Compare prompting strategies across models.

Experiments:
  1. Zero-shot vs Few-shot — sentiment classification
  2. Chain-of-thought — multi-step math reasoning
  3. Output formatting — unstructured vs JSON extraction

Each experiment runs against all models in the registry.

Usage:
    python 04_prompt_engineering.py
"""

from utils import MODELS, call_model, print_header


def run_experiment(title: str, system: str, prompt: str, model_key: str = None):
    """Run a prompt against one model or all models."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  System: {system[:80]}...")
    print(f"  Prompt: {prompt[:80]}...")
    print(f"{'='*60}")

    targets = [model_key] if model_key else MODELS.keys()

    for name in targets:
        result = call_model(name, prompt, system_prompt=system)
        if result:
            print(f"\n  [{name.upper()}] ({result['time']:.2f}s)")
            print(f"  {result['text']}")
        else:
            print(f"\n  [{name.upper()}] FAILED")


# ── Experiment 1: Zero-shot vs Few-shot ──────────────────────────────────────
# Task: classify customer feedback as POSITIVE, NEGATIVE, or NEUTRAL.
# Few-shot adds examples that anchor the model's decision boundary.

CLASSIFIER_SYSTEM = "You are a sentiment classifier. Respond with only one word: POSITIVE, NEGATIVE, or NEUTRAL."

ZERO_SHOT_PROMPT = "Classify this feedback: 'The product works but the delivery took forever and the box was damaged.'"

FEW_SHOT_PROMPT = """Classify customer feedback as POSITIVE, NEGATIVE, or NEUTRAL.

Examples:
- "Absolutely love this product, best purchase ever!" → POSITIVE
- "Terrible quality, broke after one day." → NEGATIVE
- "It's okay, nothing special but does the job." → NEUTRAL

Now classify: "The product works but the delivery took forever and the box was damaged."
"""

# ── Experiment 2: Chain-of-thought ───────────────────────────────────────────
# Task: multi-step math problem. Adding "Think step by step" dramatically
# improves accuracy on reasoning tasks.

COT_SYSTEM = "You are a logical reasoning expert."

MATH_PROMPT = """A store sells apples for $2 each. If you buy 5 or more, you get a 20% discount on the total.
You also have a coupon for $3 off your purchase (applied after the discount).
How much do you pay for 7 apples?"""

COT_PROMPT = MATH_PROMPT + "\n\nThink step by step before giving your final answer."

# ── Experiment 3: Output formatting ──────────────────────────────────────────
# Task: extract structured data from text. Without format instructions the model
# returns prose; with JSON spec it returns machine-parseable output.

FORMAT_SYSTEM = "You are a data extraction specialist."

NO_FORMAT_PROMPT = """Extract the key information from this text:
'John Smith, aged 34, works as a Senior Engineer at Google in Mountain View.
He has 8 years of experience and specializes in distributed systems.
His email is john.smith@google.com and his team has 12 members.'"""

FORMAT_PROMPT = """Extract the key information from this text and return it as a JSON object with these exact keys:
name, age, title, company, location, years_experience, specialization, email, team_size.

Return ONLY the JSON object, no other text.

Text: 'John Smith, aged 34, works as a Senior Engineer at Google in Mountain View.
He has 8 years of experience and specializes in distributed systems.
His email is john.smith@google.com and his team has 12 members.'"""


def main():
    print("\n" + "#" * 60)
    print("  PROMPT ENGINEERING — TECHNIQUE COMPARISON")
    print("#" * 60)

    # --- Experiment 1 ---
    print("\n\n" + "★" * 60)
    print("  EXPERIMENT 1: ZERO-SHOT vs FEW-SHOT")
    print("  Task: Sentiment classification")
    print("★" * 60)

    run_experiment("1A — Zero-shot", CLASSIFIER_SYSTEM, ZERO_SHOT_PROMPT)
    run_experiment("1B — Few-shot", CLASSIFIER_SYSTEM, FEW_SHOT_PROMPT)

    # --- Experiment 2 ---
    print("\n\n" + "★" * 60)
    print("  EXPERIMENT 2: WITHOUT vs WITH CHAIN-OF-THOUGHT")
    print("  Task: Multi-step math problem")
    print("★" * 60)

    run_experiment("2A — Without CoT", COT_SYSTEM, MATH_PROMPT)
    run_experiment("2B — With CoT", COT_SYSTEM, COT_PROMPT)

    # --- Experiment 3 ---
    print("\n\n" + "★" * 60)
    print("  EXPERIMENT 3: UNFORMATTED vs JSON OUTPUT")
    print("  Task: Data extraction from text")
    print("★" * 60)

    run_experiment("3A — No format instruction", FORMAT_SYSTEM, NO_FORMAT_PROMPT)
    run_experiment("3B — JSON format requested", FORMAT_SYSTEM, FORMAT_PROMPT)

    # --- Key takeaways ---
    print("\n\n" + "#" * 60)
    print("  KEY TAKEAWAYS")
    print("#" * 60)
    print("""
  1. FEW-SHOT: Examples steer the model much better than instructions alone.
     Critical for classification, formatting, and domain-specific tasks.

  2. CHAIN-OF-THOUGHT: "Think step by step" improves accuracy on reasoning tasks.
     The model shows its work, making errors easier to spot.

  3. OUTPUT FORMATTING: Explicit format instructions (especially JSON) make
     responses parseable by code. Essential for production systems.
""")


if __name__ == "__main__":
    main()
