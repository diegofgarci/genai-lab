"""
utils.py — Shared infrastructure for the Multi-Model Benchmark Tool
====================================================================
Centralizes API clients, model configuration, call routing, error handling,
and cost calculation. All scripts import from here instead of duplicating.

WHY: Scripts 03, 04, and 05 each had their own copy of clients, model configs,
and call functions. This violates DRY and makes it painful to add a new model
or change pricing. One source of truth fixes that.
"""

import os
import time
import json
from dotenv import load_dotenv
from anthropic import Anthropic, APITimeoutError, RateLimitError, AuthenticationError
from groq import Groq
import groq as groq_module  # For exception types

load_dotenv()

# ── Clients ──────────────────────────────────────────────────────────────────
# Singleton clients — created once, reused across all calls.
# timeout=30s prevents hanging on network issues.
claude_client = Anthropic(timeout=30.0)
groq_client = Groq(timeout=30.0)

# ── Model Registry ───────────────────────────────────────────────────────────
# Single source of truth for model IDs, providers, and pricing.
# Pricing: USD per 1M tokens (as of April 2026).
# To add a model: add an entry here, everything else adapts automatically.
MODELS = {
    "claude": {
        "model_id": "claude-sonnet-4-20250514",
        "provider": "anthropic",
        "label": "Claude Sonnet 4",
        "price_in": 3.00,
        "price_out": 15.00,
    },
    "llama-70b": {
        "model_id": "llama-3.3-70b-versatile",
        "provider": "groq",
        "label": "Llama 3.3 70B",
        "price_in": 0.59,
        "price_out": 0.79,
    },
    "llama-8b": {
        "model_id": "llama-3.1-8b-instant",
        "provider": "groq",
        "label": "Llama 3.1 8B",
        "price_in": 0.05,
        "price_out": 0.08,
    },
}

# ── Retry Configuration ──────────────────────────────────────────────────────
MAX_RETRIES = 2
RETRY_DELAY = 5  # seconds


# ── Core API Call ────────────────────────────────────────────────────────────

def call_model(
    model_key: str,
    user_prompt: str,
    system_prompt: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> dict | None:
    """
    Unified router — calls the right SDK based on model_key.

    Returns dict with: text, input_tokens, output_tokens, time, model_label
    Returns None on failure after retries.

    WHY a router instead of direct SDK calls:
    - Calling code doesn't need to know which SDK to use
    - Retry logic and error handling are consistent across providers
    - Adding a new provider means adding one elif branch, not touching every script
    """
    config = MODELS[model_key]
    provider = config["provider"]
    model_id = config["model_id"]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            start = time.time()

            if provider == "anthropic":
                kwargs = {
                    "model": model_id,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": user_prompt}],
                }
                if system_prompt:
                    kwargs["system"] = system_prompt
                resp = claude_client.messages.create(**kwargs)
                elapsed = time.time() - start
                return {
                    "text": resp.content[0].text,
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                    "time": round(elapsed, 2),
                    "model_label": config["label"],
                }

            elif provider == "groq":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                resp = groq_client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                elapsed = time.time() - start
                return {
                    "text": resp.choices[0].message.content,
                    "input_tokens": resp.usage.prompt_tokens,
                    "output_tokens": resp.usage.completion_tokens,
                    "time": round(elapsed, 2),
                    "model_label": config["label"],
                }

            else:
                print(f"  [ERROR] {model_key}: unknown provider '{provider}'")
                return None

        # Auth errors: no retry — the key is wrong
        except (AuthenticationError, groq_module.AuthenticationError):
            print(f"  [AUTH ERROR] {model_key}: invalid API key. Check .env")
            return None

        # Rate limits: retry with backoff
        except (RateLimitError, groq_module.RateLimitError):
            if attempt < MAX_RETRIES:
                print(f"  [RATE LIMIT] {model_key}: retrying in {RETRY_DELAY}s... ({attempt}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                print(f"  [RATE LIMIT] {model_key}: exceeded after {MAX_RETRIES} attempts.")
                return None

        # Timeouts: retry
        except (APITimeoutError, groq_module.APITimeoutError):
            if attempt < MAX_RETRIES:
                print(f"  [TIMEOUT] {model_key}: retrying... ({attempt}/{MAX_RETRIES})")
            else:
                print(f"  [TIMEOUT] {model_key}: failed after {MAX_RETRIES} attempts.")
                return None

        # Catch-all: log and skip
        except Exception as e:
            print(f"  [ERROR] {model_key}: {type(e).__name__}: {e}")
            return None

    return None


# ── Cost Calculation ─────────────────────────────────────────────────────────

def calculate_cost(model_key: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD based on model pricing."""
    config = MODELS[model_key]
    return (input_tokens * config["price_in"] / 1_000_000) + \
           (output_tokens * config["price_out"] / 1_000_000)


# ── Display Helpers ──────────────────────────────────────────────────────────

def print_header(title: str):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_result(result: dict, label: str = ""):
    """Print a model result with optional label."""
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}{result['model_label']} ({result['time']}s)")
    print(f"{'-'*50}")
    text = result["text"]
    print(text[:800])
    if len(text) > 800:
        print(f"\n... ({len(text)} chars total)")
    print()


def parse_json_response(text: str) -> dict | None:
    """
    Attempt to parse JSON from model output.
    Handles common issues: markdown fences, leading/trailing text.
    Returns parsed dict or None.
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1]  # Remove first line (```json)
    if clean.endswith("```"):
        clean = clean.rsplit("```", 1)[0]
    clean = clean.strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return None
