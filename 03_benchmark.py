"""
03_benchmark.py — Multi-Model Benchmark Tool
==============================================
Day 2: Run the same prompt across Claude + Groq models,
compare latency, token usage, and cost.

This is the first script to use utils.py — shared clients,
model registry, routing, and error handling.

Usage:
    python 03_benchmark.py
"""

from utils import MODELS, call_model, calculate_cost, print_header


def print_result(name: str, result: dict):
    """Print individual model result with cost."""
    config = MODELS[name]
    cost = calculate_cost(name, result["input_tokens"], result["output_tokens"])

    print(f"\n=== {name.upper()} ({config['model_id']}) ===")
    print(result["text"])
    print(f"Tokens in: {result['input_tokens']} | Tokens out: {result['output_tokens']}")
    print(f"Time: {result['time']:.2f}s")
    print(f"Cost: ${cost:.6f}")


def print_summary(results: dict):
    """Print comparison summary table."""
    print("\n=== SUMMARY ===")
    print(f"{'Model':<15} {'Time':>8} {'Tokens Out':>12} {'Cost':>12}")
    print("-" * 50)
    for name, result in results.items():
        cost = calculate_cost(name, result["input_tokens"], result["output_tokens"])
        print(f"{name:<15} {result['time']:>7.2f}s {result['output_tokens']:>10} ${cost:>10.6f}")


def main():
    print_header("MULTI-MODEL BENCHMARK")

    prompt = input("Enter your prompt: ")

    results = {}
    for name in MODELS:
        print(f"\nCalling {name}...")
        result = call_model(name, prompt, system_prompt="You are a concise technical explainer. Always respond in 2 sentences or less.")
        if result:
            results[name] = result

    for name, result in results.items():
        print_result(name, result)

    if results:
        print_summary(results)
    else:
        print("\n[WARNING] All models failed. Check your API keys and network connection.")


if __name__ == "__main__":
    main()
