"""
01_first_call.py — First API Call to Claude
============================================
Day 1: Connect to Claude API, send a prompt, inspect the response.

This is intentionally simple — no shared utils, no error handling.
It's a "hello world" for LLM APIs.

Usage:
    python 01_first_call.py
    python 01_first_call.py "What is a transformer?"
"""

import sys
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

client = Anthropic()

prompt = (
    input("Enter your prompt: ").strip()
    if len(sys.argv) < 2
    else " ".join(sys.argv[1:])
)

message = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are a concise technical explainer. Always respond in 2 sentences or less.",
    temperature=1.0,
    messages=[{"role": "user", "content": prompt}],
)

print(f"\nModel: {message.model}")
print(f"Tokens: {message.usage.input_tokens} in / {message.usage.output_tokens} out")
print(f"\n{message.content[0].text}")
