"""
08 - Personal Automation Agent — Complex Tool Design
======================================================
Builds on 06's tool loop with production-grade tools:

1. manage_tasks   — CRUD with nested objects, enums, arrays
2. search_contacts — multi-filter search with optional params
3. send_notification — multi-recipient with channel routing and scheduling

Key concepts:
- Nested object schemas (task objects inside tool params)
- Multiplexed tools (one tool, multiple actions via enum)
- Structured error responses that the model can reason about
- Model-driven tool chaining (model decides when to combine tools)

Usage:
    python 08_agent_tools.py
"""

import json
import time
from datetime import datetime, timedelta
from anthropic import Anthropic, APIStatusError, RateLimitError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(timeout=30.0)
MODEL = "claude-sonnet-4-20250514"

MAX_RETRIES = 3
RETRY_DELAY = 5


# =============================================================================
# SIMULATED DATA STORES
# =============================================================================
# In production these would be databases, APIs, etc.
# Here they're in-memory dicts so we can see state change across tool calls.

TASKS = {
    "task-001": {
        "title": "Review Q1 budget proposal",
        "priority": "high",
        "status": "pending",
        "due_date": "2026-04-18",
        "tags": ["finance", "quarterly"],
        "created_at": "2026-04-10T09:00:00",
    },
    "task-002": {
        "title": "Prepare client demo for Acme Corp",
        "priority": "high",
        "status": "in_progress",
        "due_date": "2026-04-16",
        "tags": ["sales", "demo"],
        "created_at": "2026-04-11T14:30:00",
    },
    "task-003": {
        "title": "Update team wiki with onboarding docs",
        "priority": "low",
        "status": "pending",
        "due_date": "2026-04-25",
        "tags": ["documentation"],
        "created_at": "2026-04-12T10:00:00",
    },
    "task-004": {
        "title": "Fix authentication bug in staging",
        "priority": "medium",
        "status": "done",
        "due_date": "2026-04-14",
        "tags": ["engineering", "bug"],
        "created_at": "2026-04-08T16:00:00",
    },
}

CONTACTS = {
    "contact-001": {
        "name": "María García",
        "email": "maria@acme.com",
        "company": "Acme Corp",
        "role": "VP of Engineering",
        "phone": "+34 612 345 678",
        "tags": ["client", "technical"],
    },
    "contact-002": {
        "name": "James Chen",
        "email": "james@startup.io",
        "company": "StartupIO",
        "role": "CEO",
        "phone": "+1 415 555 0123",
        "tags": ["investor", "advisor"],
    },
    "contact-003": {
        "name": "Laura Fernández",
        "email": "laura@internal.com",
        "company": "Internal",
        "role": "Product Manager",
        "phone": "+34 698 765 432",
        "tags": ["team", "product"],
    },
    "contact-004": {
        "name": "Ahmed Patel",
        "email": "ahmed@acme.com",
        "company": "Acme Corp",
        "role": "CTO",
        "phone": "+44 20 7946 0958",
        "tags": ["client", "executive"],
    },
}

NOTIFICATIONS_LOG = []  # Append-only log of sent notifications

_task_counter = len(TASKS)  # For generating new task IDs


# =============================================================================
# TOOL DEFINITIONS (JSON SCHEMAS)
# =============================================================================
# These are what the model sees. The descriptions are critical —
# they're the model's ONLY guide for when and how to use each tool.

tools = [
    {
        "name": "manage_tasks",
        "description": (
            "Manage personal tasks: create, list, update, or delete. "
            "Use 'list' to see tasks, optionally filtered by status, priority, or tags. "
            "Use 'create' to add a new task (requires title at minimum). "
            "Use 'update' to change a task's fields (requires task_id). "
            "Use 'delete' to remove a task (requires task_id). "
            "Returns the affected task(s) or an error with suggestions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "update", "delete"],
                    "description": "The operation to perform"
                },
                "task_id": {
                    "type": "string",
                    "description": "Task ID (required for update/delete). Format: 'task-XXX'"
                },
                "task": {
                    "type": "object",
                    "description": "Task data (for create: new task; for update: fields to change)",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Task title/description"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Task priority. Default: medium"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "done"],
                            "description": "Task status. Default for new tasks: pending"
                        },
                        "due_date": {
                            "type": "string",
                            "description": "Due date in ISO format (YYYY-MM-DD)"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorization, e.g. ['finance', 'urgent']"
                        }
                    }
                },
                "filters": {
                    "type": "object",
                    "description": "Filters for the 'list' action (all optional, combined with AND)",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "done", "all"],
                            "description": "Filter by status. Default: 'all'"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Filter by priority"
                        },
                        "tag": {
                            "type": "string",
                            "description": "Filter by tag (matches if task has this tag)"
                        }
                    }
                }
            },
            "required": ["action"]
        }
    },
    {
        "name": "search_contacts",
        "description": (
            "Search the contact directory. Can search by name (partial match), "
            "company, role, or tag. All filters are optional and combined with AND. "
            "Returns matching contacts with full details. "
            "If no filters provided, returns all contacts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Partial name match (case-insensitive), e.g. 'mar' matches 'María García'"
                },
                "company": {
                    "type": "string",
                    "description": "Exact company name (case-insensitive)"
                },
                "role_contains": {
                    "type": "string",
                    "description": "Partial match on role/title, e.g. 'engineer' matches 'VP of Engineering'"
                },
                "tag": {
                    "type": "string",
                    "description": "Filter by contact tag, e.g. 'client', 'team', 'investor'"
                }
            }
        }
    },
    {
        "name": "send_notification",
        "description": (
            "Send a notification to one or more recipients. "
            "Requires at least one recipient (by email or contact name). "
            "Supports channels: 'email' (default), 'slack', 'sms'. "
            "Can schedule for later with schedule_time (ISO datetime). "
            "If a recipient name is given instead of email, the system will "
            "look up the contact — use search_contacts first if unsure about the name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recipients": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "email": {
                                "type": "string",
                                "description": "Recipient email address"
                            },
                            "name": {
                                "type": "string",
                                "description": "Recipient name (will be resolved to email via contacts)"
                            }
                        }
                    },
                    "description": "List of recipients. Each must have 'email' or 'name' (or both)."
                },
                "subject": {
                    "type": "string",
                    "description": "Notification subject line"
                },
                "body": {
                    "type": "string",
                    "description": "Notification body/message content"
                },
                "channel": {
                    "type": "string",
                    "enum": ["email", "slack", "sms"],
                    "description": "Delivery channel. Default: email"
                },
                "schedule_time": {
                    "type": "string",
                    "description": "ISO datetime to schedule delivery (e.g. '2026-04-15T09:00:00'). If omitted, sends immediately."
                }
            },
            "required": ["recipients", "subject", "body"]
        }
    }
]


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================
# Each tool returns a dict with either a "result" key (success)
# or an "error" key (failure) plus optional "suggestions" to guide the model.

def manage_tasks(action: str, task_id: str = None, task: dict = None, filters: dict = None) -> dict:
    """Task CRUD with structured error responses."""
    global _task_counter

    # ── CREATE ──
    if action == "create":
        if not task or not task.get("title"):
            return {
                "error": "Cannot create task without a title",
                "suggestions": ["Provide a 'task' object with at least a 'title' field"]
            }

        _task_counter += 1
        new_id = f"task-{_task_counter:03d}"
        new_task = {
            "title": task["title"],
            "priority": task.get("priority", "medium"),
            "status": task.get("status", "pending"),
            "due_date": task.get("due_date", ""),
            "tags": task.get("tags", []),
            "created_at": datetime.now().isoformat(),
        }
        TASKS[new_id] = new_task
        return {
            "result": "created",
            "task_id": new_id,
            "task": new_task,
        }

    # ── LIST ──
    elif action == "list":
        filters = filters or {}
        status_filter = filters.get("status", "all")
        priority_filter = filters.get("priority")
        tag_filter = filters.get("tag")

        matches = []
        for tid, t in TASKS.items():
            if status_filter != "all" and t["status"] != status_filter:
                continue
            if priority_filter and t["priority"] != priority_filter:
                continue
            if tag_filter and tag_filter not in t.get("tags", []):
                continue
            matches.append({"task_id": tid, **t})

        return {
            "result": "found",
            "count": len(matches),
            "total_tasks": len(TASKS),
            "filters_applied": {k: v for k, v in filters.items() if v},
            "tasks": matches,
        }

    # ── UPDATE ──
    elif action == "update":
        if not task_id:
            return {
                "error": "task_id is required for update",
                "suggestions": ["Use action='list' first to find the task_id"]
            }
        if task_id not in TASKS:
            available = list(TASKS.keys())
            return {
                "error": f"Task '{task_id}' not found",
                "available_task_ids": available,
                "suggestions": [f"Valid IDs are: {', '.join(available)}"]
            }
        if not task:
            return {
                "error": "No fields provided to update",
                "suggestions": ["Provide a 'task' object with the fields to change"]
            }

        updated_fields = []
        for key, value in task.items():
            if key in TASKS[task_id]:
                TASKS[task_id][key] = value
                updated_fields.append(key)

        return {
            "result": "updated",
            "task_id": task_id,
            "updated_fields": updated_fields,
            "task": TASKS[task_id],
        }

    # ── DELETE ──
    elif action == "delete":
        if not task_id:
            return {
                "error": "task_id is required for delete",
                "suggestions": ["Use action='list' first to find the task_id"]
            }
        if task_id not in TASKS:
            return {
                "error": f"Task '{task_id}' not found",
                "available_task_ids": list(TASKS.keys()),
            }

        deleted = TASKS.pop(task_id)
        return {
            "result": "deleted",
            "task_id": task_id,
            "deleted_task": deleted,
        }

    else:
        return {
            "error": f"Unknown action: '{action}'",
            "valid_actions": ["create", "list", "update", "delete"],
        }


def search_contacts(
    name: str = None,
    company: str = None,
    role_contains: str = None,
    tag: str = None,
) -> dict:
    """Search contacts with combined filters. All optional, AND logic."""
    matches = []

    for cid, c in CONTACTS.items():
        if name and name.lower() not in c["name"].lower():
            continue
        if company and company.lower() != c["company"].lower():
            continue
        if role_contains and role_contains.lower() not in c["role"].lower():
            continue
        if tag and tag not in c.get("tags", []):
            continue
        matches.append({"contact_id": cid, **c})

    if not matches:
        # Helpful error: tell the model what IS available
        all_companies = sorted(set(c["company"] for c in CONTACTS.values()))
        all_tags = sorted(set(t for c in CONTACTS.values() for t in c.get("tags", [])))
        return {
            "result": "no_matches",
            "count": 0,
            "suggestions": [
                f"Available companies: {', '.join(all_companies)}",
                f"Available tags: {', '.join(all_tags)}",
                "Try a broader search or check spelling",
            ]
        }

    return {
        "result": "found",
        "count": len(matches),
        "contacts": matches,
    }


def send_notification(
    recipients: list,
    subject: str,
    body: str,
    channel: str = "email",
    schedule_time: str = None,
) -> dict:
    """
    Send notification with recipient resolution and validation.
    Recipients can use email or name — names get resolved via contacts.
    """
    resolved = []
    errors = []

    for r in recipients:
        email = r.get("email")
        name = r.get("name")

        if email:
            # Direct email — use as-is
            resolved.append({"email": email, "name": name or email})
        elif name:
            # Resolve name to email via contacts
            found = None
            for c in CONTACTS.values():
                if name.lower() in c["name"].lower():
                    found = c
                    break
            if found:
                resolved.append({"email": found["email"], "name": found["name"]})
            else:
                errors.append({
                    "recipient": name,
                    "error": f"Contact '{name}' not found",
                    "suggestion": "Use search_contacts to find the correct name first"
                })
        else:
            errors.append({
                "recipient": r,
                "error": "Recipient must have 'email' or 'name'",
            })

    if not resolved and errors:
        return {
            "error": "No valid recipients — notification not sent",
            "recipient_errors": errors,
        }

    # Validate schedule_time if provided
    scheduled = None
    if schedule_time:
        try:
            scheduled = datetime.fromisoformat(schedule_time)
            if scheduled < datetime.now():
                return {
                    "error": f"Schedule time '{schedule_time}' is in the past",
                    "suggestion": "Use a future datetime or omit schedule_time to send immediately",
                    "current_time": datetime.now().isoformat(),
                }
        except ValueError:
            return {
                "error": f"Invalid datetime format: '{schedule_time}'",
                "suggestion": "Use ISO format: 'YYYY-MM-DDTHH:MM:SS'",
            }

    # "Send" the notification (simulated)
    notification = {
        "id": f"notif-{len(NOTIFICATIONS_LOG) + 1:03d}",
        "recipients": resolved,
        "subject": subject,
        "body": body,
        "channel": channel,
        "status": "scheduled" if scheduled else "sent",
        "scheduled_for": schedule_time if scheduled else None,
        "sent_at": None if scheduled else datetime.now().isoformat(),
    }
    NOTIFICATIONS_LOG.append(notification)

    result = {
        "result": "scheduled" if scheduled else "sent",
        "notification_id": notification["id"],
        "delivered_to": [r["email"] for r in resolved],
        "channel": channel,
    }

    if scheduled:
        result["scheduled_for"] = schedule_time

    # Include partial errors if some recipients failed
    if errors:
        result["warnings"] = errors

    return result


# Tool dispatch map
TOOL_DISPATCH = {
    "manage_tasks": manage_tasks,
    "search_contacts": search_contacts,
    "send_notification": send_notification,
}


# =============================================================================
# API CALL WITH RETRY (same pattern as 06)
# =============================================================================

def api_call_with_retry(**kwargs) -> object:
    """Call client.messages.create with retry for transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.messages.create(**kwargs)
        except (APIStatusError, RateLimitError) as e:
            is_overloaded = isinstance(e, APIStatusError) and e.status_code == 529
            is_rate_limit = isinstance(e, RateLimitError)
            if not (is_overloaded or is_rate_limit):
                raise  # Not a retryable error — re-raise immediately
            error_type = "OVERLOADED (529)" if is_overloaded else "RATE LIMIT"
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
# TOOL USE LOOP (same pattern as 06, with richer logging)
# =============================================================================

SYSTEM_PROMPT = (
    "You are a personal automation assistant. You manage tasks, contacts, "
    "and notifications. Be helpful and proactive — if a user asks to notify "
    "someone, look up their contact first. If a task operation fails, "
    "use the error details to retry or suggest alternatives. "
    "Always confirm what you did after completing an action."
)


def run_agent(user_message: str) -> str:
    """
    Run the automation agent with tool loop.
    The model decides which tools to call and in what order.
    """
    print(f"\n{'='*70}")
    print(f"  USER: {user_message}")
    print(f"{'='*70}")

    messages = [{"role": "user", "content": user_message}]

    iteration = 0
    max_iterations = 10  # Safety valve — prevent infinite loops

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Agent Turn #{iteration} ---")

        response = api_call_with_retry(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        print(f"  Stop reason: {response.stop_reason}")

        tool_results = []

        for block in response.content:
            if block.type == "text":
                print(f"  AGENT: {block.text}")

            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input

                # Pretty-print the tool call
                print(f"  TOOL CALL: {tool_name}")
                for key, value in tool_input.items():
                    print(f"    {key}: {json.dumps(value)}")

                # Execute
                if tool_name in TOOL_DISPATCH:
                    result = TOOL_DISPATCH[tool_name](**tool_input)
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                # Pretty-print result
                result_str = json.dumps(result, indent=2, ensure_ascii=False)
                print(f"  RESULT: {result_str[:500]}")
                if len(result_str) > 500:
                    print(f"  ... ({len(result_str)} chars total)")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        # Done — no more tool calls
        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text += block.text
            return final_text

        # Tools were called — feed results back and continue
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

    return "[Agent exceeded maximum iterations]"


# =============================================================================
# TEST SCENARIOS
# =============================================================================
# Each scenario tests a different capability:
# 1. Simple CRUD — model uses manage_tasks correctly
# 2. Filtered search — model applies filters
# 3. Multi-tool chain — model chains search_contacts → send_notification
# 4. Error recovery — model handles failure and adapts
# 5. Complex multi-step — model orchestrates multiple tools autonomously

def main():
    print("\n" + "=" * 70)
    print("  PERSONAL AUTOMATION AGENT — Complex Tool Design")
    print("  Tools: manage_tasks, search_contacts, send_notification")
    print("=" * 70)

    # ── Scenario 1: List + Create ──────────────────────────────────────
    # Tests: basic CRUD, nested object creation, filter usage
    print("\n\n" + "★" * 70)
    print("  SCENARIO 1: Task management (list + create)")
    print("★" * 70)

    result = run_agent(
        "Show me all my high-priority tasks. Then create a new one: "
        "'Prepare investor update deck', high priority, due April 20, "
        "tagged 'finance' and 'investor-relations'."
    )
    print(f"\n  FINAL: {result}")

    # ── Scenario 2: Contact search with filters ───────────────────────
    # Tests: multi-filter search, partial matching
    print("\n\n" + "★" * 70)
    print("  SCENARIO 2: Contact search")
    print("★" * 70)

    result = run_agent(
        "Find all contacts from Acme Corp who have a technical role."
    )
    print(f"\n  FINAL: {result}")

    # ── Scenario 3: Tool chaining (search → notify) ──────────────────
    # Tests: model decides to search contacts first, then send notification
    # This is NOT prompted explicitly — the model should chain on its own
    print("\n\n" + "★" * 70)
    print("  SCENARIO 3: Tool chaining — search then notify")
    print("★" * 70)

    result = run_agent(
        "Send an email to María from Acme about the demo next Wednesday. "
        "Subject: 'Client Demo — Schedule Confirmation'. "
        "Let her know the demo is at 3pm CET and ask if she needs "
        "anything prepared in advance."
    )
    print(f"\n  FINAL: {result}")

    # ── Scenario 4: Error recovery ────────────────────────────────────
    # Tests: model gets an error and uses suggestions to recover
    print("\n\n" + "★" * 70)
    print("  SCENARIO 4: Error recovery")
    print("★" * 70)

    result = run_agent(
        "Update task task-999 to set priority to high."
    )
    print(f"\n  FINAL: {result}")

    # ── Scenario 5: Complex multi-step ────────────────────────────────
    # Tests: model orchestrates 3+ tool calls autonomously
    print("\n\n" + "★" * 70)
    print("  SCENARIO 5: Multi-step orchestration")
    print("★" * 70)

    result = run_agent(
        "I need to prepare for the Acme Corp demo. Here's what I need: "
        "1) Show me all tasks tagged 'demo' or 'sales'. "
        "2) Create a new task 'Send pre-demo materials to Acme team', "
        "   high priority, due tomorrow, tagged 'demo' and 'sales'. "
        "3) Find all Acme Corp contacts and send them a Slack notification "
        "   saying the demo is confirmed for Wednesday at 3pm CET."
    )
    print(f"\n  FINAL: {result}")

    # ── Show final state ──────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("  FINAL STATE")
    print("=" * 70)

    print(f"\n  Tasks ({len(TASKS)}):")
    for tid, t in TASKS.items():
        print(f"    {tid}: [{t['priority'].upper()}] [{t['status']}] {t['title']}")

    print(f"\n  Notifications sent ({len(NOTIFICATIONS_LOG)}):")
    for n in NOTIFICATIONS_LOG:
        recipients = ", ".join(r["email"] for r in n["recipients"])
        print(f"    {n['id']}: [{n['channel']}] → {recipients} | {n['subject']}")


if __name__ == "__main__":
    main()