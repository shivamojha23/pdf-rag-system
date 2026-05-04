from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from pathlib import Path
import os

# Load API key
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=os.getenv("GROQ_API_KEY")
)

# -----------------------------------------------
# PART 1 — Define Tools
# -----------------------------------------------
@tool
def create_ticket(title: str, priority: str, description: str) -> dict:
    """Creates a support ticket when a user reports a bug or issue.
    Use this when user describes a problem, bug, or error they are facing."""
    ticket_id = "TKT-" + str(abs(hash(title)))[-4:]
    print(f"\n⚙️  Creating ticket: {title} [{priority}]")
    return {"ticket_id": ticket_id, "status": "created", "title": title}

@tool
def check_ticket_status(ticket_id: str) -> dict:
    """Checks the current status of an existing support ticket.
    Use this when user asks about status of a ticket."""
    print(f"\n⚙️  Checking ticket: {ticket_id}")
    return {"ticket_id": ticket_id, "status": "in_progress", "assigned_to": "Dev Team"}

@tool
def search_knowledge_base(query: str) -> str:
    """Searches knowledge base for SmartTravel policies and information.
    Use this when user asks about refunds, cities, app, or support hours."""
    print(f"\n⚙️  Searching knowledge base for: {query}")
    knowledge = {
        "refund": "Full refund if cancelled 24 hours before travel. 50% within 24 hours.",
        "cities": "We cover 50+ cities including Mumbai, Delhi, Pune, Bangalore.",
        "support": "Support available Monday-Saturday 9am-6pm IST.",
        "app": "Available on Android 8.0+ and iOS 13+."
    }
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    return "No specific information found. Please contact support."

# -----------------------------------------------
# PART 2 — Bind tools to LLM (newer way)
# -----------------------------------------------
tools = [create_ticket, check_ticket_status, search_knowledge_base]

# This binds the tools directly to the LLM
llm_with_tools = llm.bind_tools(tools)

# Map tool names to actual functions
tool_map = {t.name: t for t in tools}

# -----------------------------------------------
# PART 3 — The Agent Loop (manual but clean)
# -----------------------------------------------
def run_agent(user_message):
    print(f"\n{'='*60}")
    print(f"User: {user_message}")
    print('='*60)

    messages = [
        SystemMessage(content="""You are a helpful SmartTravel support agent.
Use the available tools to help users.
Be concise and friendly."""),
        HumanMessage(content=user_message)
    ]

    # Agent loop — keeps running until no more tool calls
    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # If no tool calls — agent is done
        if not response.tool_calls:
            print(f"\n✅ Final Answer: {response.content}")
            break

        # Run each tool the AI requested
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"\n🤖 AI calling: {tool_name}({tool_args})")

            # Execute the tool
            result = tool_map[tool_name].invoke(tool_args)

            # Add result back to conversation
            from langchain_core.messages import ToolMessage
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            ))

# -----------------------------------------------
# PART 4 — Test it
# -----------------------------------------------
run_agent("My app keeps crashing on login, please create a ticket")
run_agent("What is your refund policy?")
run_agent("Check status of ticket TKT-4829")