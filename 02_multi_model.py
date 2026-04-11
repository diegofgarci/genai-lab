"""
02_multi_model.py — First Call to Groq (Llama)
================================================
Day 1: Connect to Groq API with Llama 3.3 70B, compare SDK differences.

Intentionally standalone (no shared utils) — the point is to see how
Groq's OpenAI-compatible SDK differs from Anthropic's SDK:
  - system prompt goes in messages array, not as a separate param
  - response structure: choices[0].message.content vs content[0].text
  - token fields: prompt_tokens/completion_tokens vs input_tokens/output_tokens

Usage:
    python 02_multi_model.py
    python 02_multi_model.py "What is a transformer?"
"""

import sys
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq()

prompt = (
    input("Enter your prompt: ").strip()
    if len(sys.argv) < 2
    else " ".join(sys.argv[1:]).strip()
)

if not prompt:
    print("No prompt provided. Exiting.")
    exit()

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    max_tokens=1024,
    temperature=0.0,
    messages=[
        {
            "role": "system",
            "content": "You are a concise technical explainer. Always respond in 2 sentences or less.",
        },
        {"role": "user", "content": prompt},
    ],
)

print(f"\nModel: {response.model}")
print(f"Tokens: {response.usage.prompt_tokens} in / {response.usage.completion_tokens} out")
print(f"\n{response.choices[0].message.content}")
