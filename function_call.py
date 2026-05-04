from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
import json
import os

# Load API key from .env
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------------------------
# PART 1 — Your actual Python functions
# -----------------------------------------------
def create_ticket(title, priority, description):
    ticket_id = "TKT-" + str(abs(hash(title)))[-4:]
    print(f"\n⚙️  FUNCTION CALLED: create_ticket()")
    print(f"   Title: {title}")
    print(f"   Priority: {priority}")
    print(f"   Description: {description}")
    return {"ticket_id": ticket_id, "status": "created", "title": title}

def check_ticket_status(ticket_id):
    print(f"\n⚙️  FUNCTION CALLED: check_ticket_status()")
    print(f"   Ticket ID: {ticket_id}")
    return {"ticket_id": ticket_id, "status": "in_progress", "assigned_to": "Dev Team"}

def send_notification(user_email, message):
    print(f"\n⚙️  FUNCTION CALLED: send_notification()")
    print(f"   To: {user_email}")
    print(f"   Message: {message}")
    return {"sent": True, "email": user_email}

# -----------------------------------------------
# PART 2 — Describe functions to AI
# -----------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Creates a support ticket when a user reports a bug or issue",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short title of the issue"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Priority level based on severity"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the issue"
                    }
                },
                "required": ["title", "priority", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_ticket_status",
            "description": "Checks the current status of an existing support ticket",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The ticket ID to check, format: TKT-XXXX"
                    }
                },
                "required": ["ticket_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_notification",
            "description": "Sends an email notification to a user",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_email": {
                        "type": "string",
                        "description": "Email address of the user"
                    },
                    "message": {
                        "type": "string",
                        "description": "The notification message to send"
                    }
                },
                "required": ["user_email", "message"]
            }
        }
    }
]

# -----------------------------------------------
# PART 3 — Map names to actual functions
# -----------------------------------------------
available_functions = {
    "create_ticket": create_ticket,
    "check_ticket_status": check_ticket_status,
    "send_notification": send_notification
}

# -----------------------------------------------
# PART 4 — The Agent Loop
# -----------------------------------------------
def run_agent(user_message):
    print(f"\nUser: {user_message}")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful support agent. Use the available tools to help users."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    # Step 1 — Send message + tools to AI
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    ai_message = response.choices[0].message

    # Step 2 — Did AI want to call a function?
    if ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n🤖 AI decided to call: {function_name}")
            print(f"   With arguments: {function_args}")

            # Step 3 — Run the actual function
            function_result = available_functions[function_name](**function_args)

            # Step 4 — Send result back to AI
            messages.append({"role": "assistant", "tool_calls": [tool_call]})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(function_result)
            })

        # Step 5 — Get final reply from AI
        final_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        print(f"\nAI: {final_response.choices[0].message.content}")

    else:
        # No tool needed — AI just replied
        print(f"\nAI: {ai_message.content}")

# -----------------------------------------------
# PART 5 — Test it
# -----------------------------------------------
run_agent("My app keeps crashing when I try to login.")
print("\n" + "="*60)

run_agent("Can you check the status of ticket TKT-4829?")
print("\n" + "="*60)

run_agent("Send a notification to john@example.com that his issue has been resolved")
print("\n" + "="*60)

run_agent("What is 2 + 2?")