"""
06 - Function Calling with Claude API
======================================
Demonstrates the complete tool use cycle:
1. Define tools (JSON schema)
2. Send message + tools to Claude
3. Detect tool_use in response
4. Execute the tool locally
5. Send tool result back to Claude
6. Get final natural language response

Tools:
- get_weather: Simulated weather lookup by city
- calculate: Basic math operations
"""

import json
import time
from anthropic import Anthropic, APIStatusError, RateLimitError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(timeout=30.0)
MODEL = "claude-sonnet-4-20250514"

# Retry config — handles 529 (overloaded) and rate limits
MAX_RETRIES = 3
RETRY_DELAY = 5  # Base delay in seconds, doubles each attempt


# =============================================================================
# STEP 1: Define tools
# =============================================================================
# Tools are described as JSON Schema so the model knows:
# - What each tool does (description)
# - What parameters it needs (properties)
# - Which parameters are required
#
# Think of this as an API contract: you're telling the model
# "here's what you can call and how to call it"

tools = [
    {
        "name": "get_weather",
        "description": (
            "Get current weather for a city. Returns temperature in Celsius, "
            "conditions, and humidity. Use this when the user asks about "
            "weather, temperature, or if they should bring an umbrella."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'A Coruña', 'Madrid', 'London'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit. Default: celsius"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": (
            "Perform a mathematical calculation. Supports basic operations "
            "(+, -, *, /, **) and common functions. Use this when the user "
            "asks to compute something, convert units, or do any math."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '(25 * 1.8) + 32'"
                }
            },
            "required": ["expression"]
        }
    }
]


# =============================================================================
# STEP 2: Implement the actual tools
# =============================================================================
# These are YOUR functions — the model never runs them directly.
# In production, these could call real APIs, query databases, etc.

WEATHER_DATA = {
    "a coruña": {"temp_c": 17, "conditions": "Partly cloudy", "humidity": 78},
    "madrid": {"temp_c": 28, "conditions": "Sunny", "humidity": 35},
    "london": {"temp_c": 14, "conditions": "Rainy", "humidity": 85},
    "tokyo": {"temp_c": 22, "conditions": "Clear", "humidity": 60},
    "new york": {"temp_c": 20, "conditions": "Overcast", "humidity": 55},
}


def get_weather(city: str, unit: str = "celsius") -> dict:
    """Simulated weather API — in production, this would call OpenWeatherMap, etc."""
    city_lower = city.lower()
    data = WEATHER_DATA.get(city_lower)

    if not data:
        return {"error": f"No weather data available for '{city}'"}

    temp = data["temp_c"]
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32, 1)

    return {
        "city": city,
        "temperature": temp,
        "unit": unit,
        "conditions": data["conditions"],
        "humidity": data["humidity"],
    }


def calculate(expression: str) -> dict:
    """
    Evaluate a math expression safely.
    NOTE: eval() is used here for simplicity — in production, use a proper
    math parser (e.g. sympy, asteval) to prevent code injection.
    """
    # Basic safety: only allow math-related characters
    allowed_chars = set("0123456789+-*/()._ **eE")
    if not all(c in allowed_chars or c.isspace() for c in expression):
        return {"error": f"Invalid characters in expression: {expression}"}

    try:
        result = eval(expression)  # noqa: S307 — safe for demo with char filter
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": f"Calculation error: {e}"}


# Map tool names to functions for dispatch
TOOL_DISPATCH = {
    "get_weather": get_weather,
    "calculate": calculate,
}


# =============================================================================
# STEP 3: API call with retry
# =============================================================================
# Wraps client.messages.create with exponential backoff for transient errors.
# WHY separate from the tool loop: the retry is per-API-call, not per-tool-cycle.
# A single user question might need 3 API calls (ask → tool → answer),
# and each one independently might hit a 529.

def api_call_with_retry(**kwargs) -> object:
    """
    Call client.messages.create with retry for overloaded/rate-limit errors.
    Raises on auth errors or after exhausting retries.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.messages.create(**kwargs)
        except (APIStatusError, RateLimitError) as e:
            # Only retry on 529 (overloaded) and 429 (rate limit)
            if isinstance(e, APIStatusError) and not isinstance(e, RateLimitError):
                if e.status_code != 529:
                    raise  # Don't retry 400, 404, 500, etc.
                error_type = "OVERLOADED (529)"
            else:
                error_type = "RATE LIMIT"
            delay = RETRY_DELAY * (2 ** (attempt - 1))  # 5s, 10s, 20s
            if attempt < MAX_RETRIES:
                print(f"  [{error_type}] Retrying in {delay}s... ({attempt}/{MAX_RETRIES})")
                time.sleep(delay)
            else:
                print(f"  [{error_type}] Failed after {MAX_RETRIES} attempts.")
                raise
        except APITimeoutError:
            if attempt < MAX_RETRIES:
                print(f"  [TIMEOUT] Retrying... ({attempt}/{MAX_RETRIES})")
            else:
                print(f"  [TIMEOUT] Failed after {MAX_RETRIES} attempts.")
                raise


# =============================================================================
# STEP 4: The tool use loop
# =============================================================================
# This is the core pattern for function calling:
# 1. Send user message + tool definitions
# 2. If model returns tool_use → execute tool → send result back
# 3. Repeat until model returns a text response (no more tool calls)

def run_with_tools(user_message: str) -> str:
    """
    Send a message to Claude with tools available.
    Handles the full tool use loop — including multiple sequential tool calls.
    """
    print(f"\n{'='*60}")
    print(f"USER: {user_message}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": user_message}]

    # Loop: the model might need multiple tool calls to answer
    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- API Call #{iteration} ---")

        response = api_call_with_retry(
            model=MODEL,
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )

        print(f"Stop reason: {response.stop_reason}")
        print(f"Content blocks: {len(response.content)}")

        # Process each content block in the response
        # A response can contain BOTH text and tool_use blocks
        tool_results = []

        for block in response.content:
            if block.type == "text":
                print(f"TEXT: {block.text}")

            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                tool_use_id = block.id

                print(f"TOOL CALL: {tool_name}({json.dumps(tool_input)})")

                # Execute the tool locally
                if tool_name in TOOL_DISPATCH:
                    result = TOOL_DISPATCH[tool_name](**tool_input)
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                print(f"TOOL RESULT: {json.dumps(result)}")

                # Collect tool results — we'll send them all at once
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(result),
                })

        # If no tools were called, we're done — return the text response
        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text += block.text
            return final_text

        # If tools were called, add the assistant response + tool results
        # to the conversation and loop again
        if response.stop_reason == "tool_use":
            # Add the assistant's response (with tool_use blocks) as-is
            messages.append({"role": "assistant", "content": response.content})
            # Add all tool results in a single user message
            messages.append({"role": "user", "content": tool_results})
        else:
            # Unexpected stop reason
            print(f"WARNING: Unexpected stop reason: {response.stop_reason}")
            return "Unexpected response from model."


# =============================================================================
# STEP 5: Test different scenarios
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("FUNCTION CALLING DEMO — Claude API Tool Use")
    print("=" * 60)

    # Scenario 1: Simple tool call — weather
    # The model should recognize this needs get_weather
    result = run_with_tools("What's the weather like in A Coruña right now?")
    print(f"\nFINAL ANSWER: {result}")

    # Scenario 2: Simple tool call — calculator
    # The model should recognize this needs calculate
    result = run_with_tools("What's 15% tip on a bill of €47.80?")
    print(f"\nFINAL ANSWER: {result}")

    # Scenario 3: No tool needed
    # The model should answer directly without calling any tool
    result = run_with_tools("What is function calling in the context of LLMs?")
    print(f"\nFINAL ANSWER: {result}")

    # Scenario 4: Multiple tools in sequence
    # The model might call weather first, then calculate for conversion
    result = run_with_tools(
        "What's the weather in Madrid? Also, if the temperature is in Celsius, "
        "what would it be in Fahrenheit? Use the calculator to convert."
    )
    print(f"\nFINAL ANSWER: {result}")


if __name__ == "__main__":
    main()