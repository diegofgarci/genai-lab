"""
09_multi_turn_agent.py — Interactive Multi-Turn Agent with State Management
============================================================================
Week 2, Day 4: A conversational agent that maintains two types of state:

1. Conversation state (messages[]) — the full chat history sent to the API.
   Grows with every turn. The model sees all of it. This IS the model's memory.

2. Session state (dict in memory) — business data that tools accumulate.
   Notes, contacts, preferences. The model only sees this when a tool
   includes it in a result. Compact, structured, cheap.

WHY this matters: in production, conversation state gets truncated/summarized
to manage costs and context window limits. Session state gets persisted to a
database. Separating them now builds the right mental model for later.

Tools:
- manage_notes: Create, list, search, delete notes (reads/writes session state)
- manage_contacts: Lookup and add contacts (reads/writes session state)
- get_weather: Simulated weather data (stateless)
- calculate: Math operations (stateless)
- get_session_summary: Introspect current session state + token usage

Usage:
    python 09_multi_turn_agent.py
"""

import json
import time
from datetime import datetime
from anthropic import Anthropic, APIStatusError, RateLimitError, APITimeoutError
from dotenv import load_dotenv
from utils import calculate_cost

load_dotenv()

client = Anthropic(timeout=30.0)
MODEL = "claude-sonnet-4-20250514"

MAX_RETRIES = 3
RETRY_DELAY = 5

# =============================================================================
# SESSION STATE
# =============================================================================
# This dict is the agent's "working memory" — separate from conversation history.
# Tools read and write here. The model never sees it directly; it only sees
# what tools return as results.
#
# WHY a module-level dict and not a class: simplicity. This is a single-user
# demo script. In production, this becomes a per-session record in Redis/PostgreSQL
# keyed by session_id. The PATTERN is what matters, not the storage backend.

session_state = {
    "notes": [],
    "contacts": [
        # Pre-loaded demo data so you can test immediately
        {"name": "Ana García", "email": "ana@example.com", "role": "CTO", "company": "TechFlow"},
        {"name": "Carlos López", "email": "carlos@example.com", "role": "PM", "company": "DataCo"},
        {"name": "María Santos", "email": "maria@example.com", "role": "Founder", "company": "AIStart"},
    ],
    "preferences": {},
}

# Token tracking — accumulated across the entire session
token_usage = {
    "total_input": 0,
    "total_output": 0,
    "api_calls": 0,
}


# =============================================================================
# SYSTEM PROMPT
# =============================================================================
# Tells the model who it is, what tools it has, and how to behave.
# Key design decision: we DON'T dump the session state here. The model
# discovers state through tool calls, just like a real agent would query
# a database rather than having everything preloaded in context.

SYSTEM_PROMPT = """You are a helpful personal assistant with access to tools. You can:
- Manage notes (create, list, search, delete)
- Look up and add contacts
- Check weather for cities
- Do math calculations
- Show a summary of the current session

Guidelines:
- Use tools when they're the right fit. Don't guess at data you can look up.
- When the user asks about their notes or contacts, always use the tool — don't rely on memory of previous tool results, as the data may have changed.
- Be concise but friendly. No need for lengthy preambles.
- If the user asks something outside your tools, answer from your knowledge.
- When listing items, include relevant details (not just names).
"""


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================
# JSON Schema definitions that tell the model what it can call.
# Note how descriptions include WHEN to use each tool — this is what the model
# uses for semantic matching when deciding which tool fits the user's intent.

tools = [
    {
        "name": "manage_notes",
        "description": (
            "Create, list, search, or delete notes. Notes persist across the conversation. "
            "Use this when the user wants to save information, make a reminder, "
            "check their notes, find a specific note, or remove one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "search", "delete"],
                    "description": "Action to perform on notes"
                },
                "content": {
                    "type": "string",
                    "description": "For 'add': the note text. For 'search': search query. For 'delete': note ID (number)."
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "manage_contacts",
        "description": (
            "Look up or add contacts. Use when the user asks about a person, "
            "wants to find someone's email or role, or wants to save a new contact."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "add", "list"],
                    "description": "Action to perform"
                },
                "name": {
                    "type": "string",
                    "description": "For 'search': name to find. For 'add': contact name."
                },
                "email": {
                    "type": "string",
                    "description": "For 'add': contact email"
                },
                "role": {
                    "type": "string",
                    "description": "For 'add': contact role/title"
                },
                "company": {
                    "type": "string",
                    "description": "For 'add': contact company"
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "get_weather",
        "description": (
            "Get current weather for a city. Returns temperature, conditions, "
            "and humidity. Use when the user asks about weather or temperature."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'A Coruña', 'Madrid', 'London'"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": (
            "Perform a mathematical calculation. Use when the user asks to "
            "compute something, convert units, or do any math."
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
    },
    {
        "name": "get_session_summary",
        "description": (
            "Show a summary of the current session: number of notes, contacts, "
            "preferences set, and token usage. Use when the user asks about "
            "session stats, how much they've used, or wants an overview."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
]


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================
# Each function reads/writes session_state and returns a dict.
# The dict gets JSON-serialized and sent back to the model as a tool_result.
#
# KEY PATTERN: tools are the BRIDGE between session state and the model.
# The model asks "show me the notes" → tool reads session_state["notes"] →
# returns them as a result → model sees them and responds to the user.

WEATHER_DATA = {
    "a coruña": {"temp_c": 17, "conditions": "Partly cloudy", "humidity": 78},
    "madrid": {"temp_c": 28, "conditions": "Sunny", "humidity": 35},
    "london": {"temp_c": 14, "conditions": "Rainy", "humidity": 85},
    "tokyo": {"temp_c": 22, "conditions": "Clear", "humidity": 60},
    "new york": {"temp_c": 20, "conditions": "Overcast", "humidity": 55},
    "san francisco": {"temp_c": 16, "conditions": "Foggy", "humidity": 72},
    "berlin": {"temp_c": 12, "conditions": "Cloudy", "humidity": 68},
}


def manage_notes(action: str, content: str = "") -> dict:
    """
    CRUD operations on session_state["notes"].
    Each note gets an auto-incrementing ID and a timestamp.
    """
    notes = session_state["notes"]

    if action == "add":
        if not content:
            return {"error": "Cannot add empty note. Provide content."}
        note = {
            "id": len(notes) + 1,
            "content": content,
            "created_at": datetime.now().strftime("%H:%M:%S"),
        }
        notes.append(note)
        return {"status": "created", "note": note, "total_notes": len(notes)}

    elif action == "list":
        if not notes:
            return {"status": "empty", "message": "No notes yet."}
        return {"status": "ok", "notes": notes, "total": len(notes)}

    elif action == "search":
        if not content:
            return {"error": "Provide a search query."}
        query = content.lower()
        matches = [n for n in notes if query in n["content"].lower()]
        return {
            "status": "ok",
            "query": content,
            "matches": matches,
            "total_matches": len(matches),
        }

    elif action == "delete":
        if not content:
            return {"error": "Provide the note ID to delete."}
        try:
            note_id = int(content)
        except ValueError:
            return {"error": f"Invalid note ID: '{content}'. Must be a number."}
        for i, note in enumerate(notes):
            if note["id"] == note_id:
                deleted = notes.pop(i)
                return {"status": "deleted", "deleted_note": deleted, "remaining": len(notes)}
        return {"error": f"Note with ID {note_id} not found."}

    return {"error": f"Unknown action: {action}"}


def manage_contacts(
    action: str,
    name: str = "",
    email: str = "",
    role: str = "",
    company: str = "",
) -> dict:
    """Search, list, or add contacts in session_state."""
    contacts = session_state["contacts"]

    if action == "list":
        if not contacts:
            return {"status": "empty", "message": "No contacts."}
        return {"status": "ok", "contacts": contacts, "total": len(contacts)}

    elif action == "search":
        if not name:
            return {"error": "Provide a name to search."}
        query = name.lower()
        matches = [
            c for c in contacts
            if query in c["name"].lower()
            or query in c.get("company", "").lower()
            or query in c.get("role", "").lower()
        ]
        if not matches:
            return {"status": "not_found", "query": name, "message": f"No contacts matching '{name}'."}
        return {"status": "ok", "query": name, "matches": matches}

    elif action == "add":
        if not name:
            return {"error": "Name is required to add a contact."}
        contact = {"name": name, "email": email, "role": role, "company": company}
        contacts.append(contact)
        return {"status": "created", "contact": contact, "total_contacts": len(contacts)}

    return {"error": f"Unknown action: {action}"}


def get_weather(city: str) -> dict:
    """Simulated weather lookup — same as script 06 but simplified."""
    data = WEATHER_DATA.get(city.lower())
    if not data:
        return {"error": f"No weather data for '{city}'. Available: {', '.join(WEATHER_DATA.keys())}"}
    return {"city": city, "temperature_c": data["temp_c"], "conditions": data["conditions"], "humidity": data["humidity"]}


def calculate(expression: str) -> dict:
    """Safe math evaluation with character whitelist."""
    allowed_chars = set("0123456789+-*/()._ **eE")
    if not all(c in allowed_chars or c.isspace() for c in expression):
        return {"error": f"Invalid characters in expression: {expression}"}
    try:
        result = eval(expression)  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": f"Calculation error: {e}"}


def get_session_summary() -> dict:
    """Introspect current session state and token usage."""
    cost = calculate_cost(
        "claude", token_usage["total_input"], token_usage["total_output"]
    )
    return {
        "notes_count": len(session_state["notes"]),
        "contacts_count": len(session_state["contacts"]),
        "preferences": session_state["preferences"] if session_state["preferences"] else "None set",
        "api_calls": token_usage["api_calls"],
        "total_input_tokens": token_usage["total_input"],
        "total_output_tokens": token_usage["total_output"],
        "estimated_cost_usd": f"${cost:.6f}",
    }


# Tool dispatch map — same pattern as script 06
TOOL_DISPATCH = {
    "manage_notes": manage_notes,
    "manage_contacts": manage_contacts,
    "get_weather": get_weather,
    "calculate": calculate,
    "get_session_summary": get_session_summary,
}


# =============================================================================
# API CALL WITH RETRY
# =============================================================================
# Same pattern as script 06. Exponential backoff for transient errors.
# Returns the API response object or raises after exhausting retries.

def api_call_with_retry(**kwargs):
    """Call Claude API with retry for overloaded/rate-limit/timeout errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.messages.create(**kwargs)
        except (APIStatusError, RateLimitError) as e:
            # APIStatusError with status 529 = overloaded (same retry strategy as rate limits)
            # RateLimitError (429) = too many requests
            is_overloaded = isinstance(e, APIStatusError) and e.status_code == 529
            is_retryable = is_overloaded or isinstance(e, RateLimitError)
            if not is_retryable:
                raise  # Non-retryable API error (4xx, 5xx) — bail out
            error_type = "OVERLOADED (529)" if is_overloaded else "RATE LIMIT (429)"
            delay = RETRY_DELAY * (2 ** (attempt - 1))
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
# THE AGENT LOOP
# =============================================================================
# Two nested loops:
# - OUTER (REPL): reads user input, adds to messages, prints response, repeats
# - INNER (tool loop): handles multi-step tool calls within a single turn
#
# The messages[] array grows across BOTH loops — that's conversation state.
# session_state is modified by tools within the inner loop — that's session state.

def process_turn(messages: list) -> str:
    """
    Process one conversational turn: send messages to API, handle any tool calls,
    return the final text response.

    This is the INNER loop — it may make multiple API calls if the model
    chains several tools before giving a final answer.
    """
    iteration = 0

    while True:
        iteration += 1

        response = api_call_with_retry(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        # Track token usage
        token_usage["total_input"] += response.usage.input_tokens
        token_usage["total_output"] += response.usage.output_tokens
        token_usage["api_calls"] += 1

        # Process content blocks — may contain text, tool_use, or both
        tool_results = []

        for block in response.content:
            if block.type == "text" and block.text:
                # During tool chains, the model sometimes emits thinking text
                # before a tool call. We only print the FINAL text (when stop_reason == "end_turn").
                pass

            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                tool_use_id = block.id

                print(f"  🔧 {tool_name}({json.dumps(tool_input, ensure_ascii=False)})")

                # Execute tool — dispatch by name, pass all input as kwargs
                if tool_name in TOOL_DISPATCH:
                    result = TOOL_DISPATCH[tool_name](**tool_input)
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        # If the model is done (no more tool calls), extract and return text
        if response.stop_reason == "end_turn":
            messages.append({"role": "assistant", "content": response.content})
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text += block.text
            return final_text

        # If tools were called, append assistant response + results, loop again
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop reason (max_tokens, etc.)
            messages.append({"role": "assistant", "content": response.content})
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text += block.text
            return final_text or "[Response truncated — max tokens reached]"


def main():
    print("=" * 60)
    print("  MULTI-TURN AGENT — Interactive Session")
    print("  Tools: notes, contacts, weather, calculator, session stats")
    print("=" * 60)
    print()
    print("Type your messages below. Special commands:")
    print("  /quit or /exit  — end the session")
    print("  /state          — dump raw session state (debug)")
    print("  /tokens         — show token usage so far")
    print("  /clear          — reset conversation (keep session state)")
    print()

    # The conversation history — this IS the conversation state.
    # It grows with every turn and gets sent in full to the API.
    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended.")
            break

        if not user_input:
            continue

        # --- Local commands (never sent to the model) ---
        if user_input.lower() in ("/quit", "/exit"):
            break

        if user_input.lower() == "/state":
            print("\n--- RAW SESSION STATE ---")
            print(json.dumps(session_state, indent=2, ensure_ascii=False))
            print("--- END STATE ---\n")
            continue

        if user_input.lower() == "/tokens":
            cost = calculate_cost(
                "claude", token_usage["total_input"], token_usage["total_output"]
            )
            print(f"\n  API calls: {token_usage['api_calls']}")
            print(f"  Input tokens:  {token_usage['total_input']:,}")
            print(f"  Output tokens: {token_usage['total_output']:,}")
            print(f"  Estimated cost: ${cost:.6f}")
            print(f"  Messages in history: {len(messages)}\n")
            continue

        if user_input.lower() == "/clear":
            messages.clear()
            print("  Conversation history cleared. Session state preserved.\n")
            continue

        # --- Normal turn: add to conversation and process ---
        messages.append({"role": "user", "content": user_input})

        try:
            response_text = process_turn(messages)
            print(f"\nAssistant: {response_text}\n")
        except Exception as e:
            print(f"\n  [ERROR] {type(e).__name__}: {e}")
            print("  The conversation continues — try again.\n")
            # Remove the failed user message to keep state consistent
            messages.pop()

    # --- Session summary on exit ---
    cost = calculate_cost(
        "claude", token_usage["total_input"], token_usage["total_output"]
    )
    print("\n" + "=" * 60)
    print("  SESSION SUMMARY")
    print("=" * 60)
    print(f"  API calls:     {token_usage['api_calls']}")
    print(f"  Input tokens:  {token_usage['total_input']:,}")
    print(f"  Output tokens: {token_usage['total_output']:,}")
    print(f"  Cost:          ${cost:.6f}")
    print(f"  Notes created: {len(session_state['notes'])}")
    print(f"  Contacts:      {len(session_state['contacts'])}")
    print(f"  Turns:         {len([m for m in messages if m['role'] == 'user' and isinstance(m['content'], str)])}")
    print("=" * 60)


if __name__ == "__main__":
    main()