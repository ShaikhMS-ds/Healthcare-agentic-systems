"""
Single-agent healthcare assistant built with LangGraph — RAW ReAct loop edition.

PATTERN: identical to single_agent.py (one LLM + a fixed set of tools + a system
prompt, looping reason -> act -> observe until done). The only difference is HOW
the tool-calling mechanics are wired: this file hand-writes the three pieces that
`llm.bind_tools()`, `ToolNode`, and `tools_condition` normally do for you, so you
can see exactly what those helpers are doing under the hood. Read this file next
to single_agent.py and compare section "4. THE GRAPH".

MODEL: Gemini, called through its OpenAI-compatible endpoint, so the standard
LangChain `ChatOpenAI` wrapper works without changes.

SETUP:
  pip install langgraph langchain-openai python-dotenv
  add GOOGLE_API_KEY="your_key_from_aistudio.google.com" to a .env file
  python single_agent_raw.py
"""

import os
import json
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()


# ---------------------------------------------------------------------------
# 1. THE MODEL  —  Gemini through the OpenAI-compatible endpoint
# ---------------------------------------------------------------------------
llm = ChatOpenAI(
    model="gemini-2.5-flash",  # swap for a newer one (e.g. "gemini-3.5-flash") if available
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0,
)


# ---------------------------------------------------------------------------
# 2. THE TOOLS  —  the agent's "external data / actions"
#    (mock in-memory data stands in for a real clinic database/API)
# ---------------------------------------------------------------------------
_PATIENTS = {
    "P001": {
        "name": "Asha Patil",
        "age": 58,
        "medications": ["lisinopril", "warfarin", "ibuprofen"],
        "allergies": ["penicillin"],
        "next_appointment": None,
    },
    "P002": {
        "name": "Ravi Kumar",
        "age": 34,
        "medications": ["metformin"],
        "allergies": [],
        "next_appointment": "2026-07-02",
    },
}

# Tiny, illustrative interaction table — NOT a real clinical reference.
_INTERACTIONS = {
    frozenset({"warfarin", "ibuprofen"}):
        "Increased risk of bleeding when taken together.",
    frozenset({"lisinopril", "ibuprofen"}):
        "Ibuprofen may reduce lisinopril's effect and stress the kidneys.",
}


@tool
def get_patient_record(patient_id: str) -> str:
    """Look up a patient by ID. Returns name, age, current medications and allergies."""
    patient = _PATIENTS.get(patient_id.upper())
    if not patient:
        return f"No patient found with ID {patient_id}."
    return json.dumps(patient)


@tool
def check_drug_interactions(medications: list[str]) -> str:
    """Check a list of medication names for known interactions between any pair."""
    meds = [m.lower().strip() for m in medications]
    found = []
    for i in range(len(meds)):
        for j in range(i + 1, len(meds)):
            note = _INTERACTIONS.get(frozenset({meds[i], meds[j]}))
            if note:
                found.append(f"- {meds[i]} + {meds[j]}: {note}")
    if not found:
        return "No known interactions found among the provided medications."
    return "Potential interactions found:\n" + "\n".join(found)


@tool
def book_appointment(patient_id: str, preferred_date: str, reason: str) -> str:
    """Book a clinic appointment. preferred_date must be YYYY-MM-DD. Returns a confirmation."""
    patient = _PATIENTS.get(patient_id.upper())
    if not patient:
        return f"Cannot book: no patient found with ID {patient_id}."
    patient["next_appointment"] = preferred_date
    return f"Appointment confirmed for {patient['name']} on {preferred_date}. Reason: {reason}."


tools = [get_patient_record, check_drug_interactions, book_appointment]

# Hand-written equivalent of llm.bind_tools(tools): bind_tools() runs each tool
# through convert_to_openai_tool() to build the OpenAI function-calling schema,
# then curries tools=<that list> into every future .invoke() call. We convert
# once here and pass the result explicitly on each invoke below instead.
tool_schemas = [convert_to_openai_tool(t) for t in tools]
tools_by_name = {t.name: t for t in tools}


# ---------------------------------------------------------------------------
# 3. THE SYSTEM PROMPT  —  defines task, persona, tool conditions, guardrails
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are ClinicConcierge, an administrative assistant for a primary-care clinic.

What you do:
- Help patients look up their record, review current medications, and book appointments.
- Always call get_patient_record FIRST to load a patient's real data before reasoning
  about their medications or booking anything.
- For interaction questions, call check_drug_interactions with the patient's actual
  current medications (not ones the patient guesses).

Safety rules (non-negotiable):
- You are NOT a doctor. Never diagnose, and never tell a patient to start or stop a
  medication. Report what the tools return and tell them to confirm with their physician.
- If there is any sign of an emergency (chest pain, trouble breathing, severe bleeding,
  thoughts of self-harm), stop and tell them to contact emergency services immediately.

Be concise and clear."""


# ---------------------------------------------------------------------------
# 4. THE GRAPH  —  the single-agent ReAct loop, wired by hand
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def agent_node(state: AgentState) -> dict:
    """LLM node: prepend the system prompt, pass tool schemas explicitly
    (no bind_tools(), so there's no curried tools=... kwarg on llm)."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    print("*********this is message", messages, "********************")
    response = llm.invoke(messages, tools=tool_schemas)
    # print("agent response:", response)
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Hand-written equivalent of ToolNode(tools): for each tool call on the
    last AIMessage, look the tool up by name, run it, and wrap the result in
    a ToolMessage tagged with that call's id. Real ToolNode also runs calls
    concurrently and has configurable error handling for bad names/args;
    this version is sequential with no try/except, on purpose — see the
    note in the module docstring."""
    last_message = state["messages"][-1]
    results = []
    for call in last_message.tool_calls:
        selected_tool = tools_by_name[call["name"]]
        observation = selected_tool.invoke(call["args"])
        results.append(
            ToolMessage(content=str(observation), name=call["name"], tool_call_id=call["id"])
        )
    return {"messages": results}


def route_after_agent(state: AgentState) -> str:
    """Hand-written equivalent of tools_condition: tool calls -> run them,
    otherwise the agent gave its final answer -> end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    return END


builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
# After the agent speaks: if it requested tools -> "tools" node, else -> END.
builder.add_conditional_edges(
    "agent",
    route_after_agent,
    {"tools": "tools", END: END},   # <-- tells the drawer the possible branches
)
# After tools run, loop back so the agent can read the results and continue.
builder.add_edge("tools", "agent")

graph = builder.compile()

GRAPH_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "single_agent_raw_graph.png")


def save_graph_image(path: str = GRAPH_IMAGE_PATH) -> None:
    """Render the compiled graph topology to a PNG (network call to mermaid.ink)."""
    try:
        graph.get_graph().draw_mermaid_png(output_file_path=path)
        print(f"Graph diagram saved to {path}")
    except Exception as e:
        print(f"Could not save graph diagram: {e}")


# ---------------------------------------------------------------------------
# 5. RUN IT
# ---------------------------------------------------------------------------
def run(user_message: str) -> None:
    """Stream every step so you can watch the reason -> act -> observe loop."""
    for event in graph.stream(
        {"messages": [{"role": "user", "content": user_message}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()


if __name__ == "__main__":
    save_graph_image()
    # input_text = input("query:")
    run(
        "Hi, I'm patient P001. Can you check whether my current medications "
        "interact, and book me a follow-up for 2026-06-29 about my blood pressure?"
    )
    # run ("what is machine learning")
    # run(input_text)
