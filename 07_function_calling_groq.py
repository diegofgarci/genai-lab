"""
07 - Function Calling with Groq API (OpenAI-compatible)
========================================================
Same tool use pattern as 06, different wire format.
Demonstrates how Groq/OpenAI handle function calling:

1. Define tools (OpenAI format: type + function wrapper)
2. Send message + tools to Groq
3. Detect finish_reason="tool_calls" in response
4. Execute the tool locally
5. Send tool result back (role="tool")
6. Get final natural language response

KEY DIFFERENCES FROM CLAUDE (06):
- Tool definitions wrapped in {type: "function", function: {...}}
- Tool schema uses "parameters" instead of "input_schema"
- Tool calls arrive in message.tool_calls[] (not content blocks)
- Arguments come as JSON STRING — must json.loads() to parse
- Results sent as role="tool" messages (not role="user" with tool_result)
- Stop signal: finish_reason="tool_calls" (not stop_reason="tool_use")

Tools:
- get_weather: Simulated weather lookup by city
- calculate: Basic math operations
"""

import json
import time
from groq import Groq
import groq as groq_module
from dotenv import load_dotenv

load_dotenv()

client = Groq(timeout=30.0)
MODEL = "llama-3.3-70b-versatile"

# Retry config
MAX_RETRIES = 3
RETRY_DELAY = 5  # Base delay in seconds, doubles each attempt


# =============================================================================
# STEP 1: Define tools — OpenAI format
# =============================================================================
# Groq uses the OpenAI tool format. Two key structural differences:
#
# CLAUDE:                           GROQ/OPENAI:
# {                                 {
#   "name": "...",                    "type": "function",    ← wrapper
#   "description": "...",             "function": {          ← wrapper
#   "input_schema": { ... }             "name": "...",
# }                                     "description": "...",
#                                       "parameters": { ... }  ← renamed
#                                     }
#                                   }
#
# The JSON Schema inside is IDENTICAL — same properties, same types,
# same required array. Only the envelope changes.

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get current weather for a city. Returns temperature in Celsius, "
                "conditions, and humidity. Use this when the user asks about "
                "weather, temperature, or if they should bring an umbrella."
            ),
            "parameters": {
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
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Perform a mathematical calculation. Supports basic operations "
                "(+, -, *, /, **) and common functions. Use this when the user "
                "asks to compute something, convert units, or do any math."
            ),
            "parameters": {
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
    }
]


# =============================================================================
# STEP 2: Implement the actual tools
# =============================================================================
# Identical to the Claude version — tools are YOUR code, provider-agnostic.

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
# Same exponential backoff pattern as Claude version, different exception types.
# Groq uses groq.RateLimitError and groq.APITimeoutError.
# Groq doesn't have a dedicated OverloadedError — overload comes as
# RateLimitError or InternalServerError (500/503).

def api_call_with_retry(**kwargs) -> object:
    """
    Call client.chat.completions.create with retry for rate-limit/server errors.
    Raises on auth errors or after exhausting retries.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except groq_module.AuthenticationError:
            print("  [AUTH ERROR] Invalid Groq API key. Check .env")
            raise
        except groq_module.BadRequestError as e:
            # Llama sometimes generates malformed tool calls (e.g. XML instead of JSON).
            # Groq catches this server-side and returns tool_use_failed.
            # Recovery: retry without tools so the model answers directly.
            if "tool_use_failed" in str(e):
                print(f"  [TOOL FORMAT ERROR] Model generated invalid tool call.")
                print(f"  Retrying without tools...")
                kwargs_no_tools = {k: v for k, v in kwargs.items() if k != "tools"}
                return client.chat.completions.create(**kwargs_no_tools)
            raise  # Re-raise if it's a different BadRequestError
        except groq_module.RateLimitError:
            delay = RETRY_DELAY * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                print(f"  [RATE LIMIT] Retrying in {delay}s... ({attempt}/{MAX_RETRIES})")
                time.sleep(delay)
            else:
                print(f"  [RATE LIMIT] Failed after {MAX_RETRIES} attempts.")
                raise
        except groq_module.APITimeoutError:
            if attempt < MAX_RETRIES:
                print(f"  [TIMEOUT] Retrying... ({attempt}/{MAX_RETRIES})")
            else:
                print(f"  [TIMEOUT] Failed after {MAX_RETRIES} attempts.")
                raise
        except groq_module.InternalServerError:
            # Groq returns 500/503 when overloaded (no dedicated 529 like Anthropic)
            delay = RETRY_DELAY * (2 ** (attempt - 1))
            if attempt < MAX_RETRIES:
                print(f"  [SERVER ERROR] Retrying in {delay}s... ({attempt}/{MAX_RETRIES})")
                time.sleep(delay)
            else:
                print(f"  [SERVER ERROR] Failed after {MAX_RETRIES} attempts.")
                raise


# =============================================================================
# STEP 4: The tool use loop
# =============================================================================
# Same pattern as Claude, different response structure.
#
# CLAUDE response:                  GROQ/OPENAI response:
# response.stop_reason              choice.finish_reason
# response.content[] blocks         choice.message.tool_calls[]
# block.type == "tool_use"          tool_call.type == "function"
# block.input (dict)                tool_call.function.arguments (STRING!)
# block.id                          tool_call.id
#
# Sending results back:
# CLAUDE: role="user", content=[{type: "tool_result", tool_use_id, content}]
# GROQ:   role="tool", tool_call_id, content (one message per tool result)

def run_with_tools(user_message: str) -> str:
    """
    Send a message to Groq with tools available.
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

        choice = response.choices[0]
        message = choice.message

        print(f"Finish reason: {choice.finish_reason}")

        # Check if the model wants to call tools
        # In OpenAI format, tool_calls is an attribute on the message
        # (not mixed into content blocks like Claude)
        if choice.finish_reason == "tool_calls" and message.tool_calls:
            # Print any text the model included alongside tool calls
            if message.content:
                print(f"TEXT: {message.content}")

            # Add the assistant's full message to history FIRST
            # This preserves the tool_calls so the API can match results
            messages.append(message)

            # Process each tool call
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name

                # KEY DIFFERENCE: arguments is a JSON string, not a dict!
                # Claude gives you block.input as a ready-to-use dict.
                # Groq/OpenAI gives you a raw string that needs parsing.
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}
                    print(f"  WARNING: Could not parse arguments: {tool_call.function.arguments}")

                print(f"TOOL CALL: {tool_name}({json.dumps(tool_args)})")

                # Execute the tool locally
                if tool_name in TOOL_DISPATCH:
                    result = TOOL_DISPATCH[tool_name](**tool_args)
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                print(f"TOOL RESULT: {json.dumps(result)}")

                # Send result back — ONE message per tool result with role="tool"
                # Claude bundles all results in a single "user" message;
                # OpenAI/Groq uses a separate "tool" message for each result.
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

        elif choice.finish_reason == "stop":
            # Model is done — return the text response
            print(f"TEXT: {message.content}")
            return message.content

        else:
            # Unexpected finish reason (e.g. "length" if max_tokens hit)
            print(f"WARNING: Unexpected finish_reason: {choice.finish_reason}")
            return message.content or "Unexpected response from model."


# =============================================================================
# STEP 5: Test scenarios — identical to Claude version
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("FUNCTION CALLING DEMO — Groq API Tool Use (Llama 3.3 70B)")
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