"""
Single-agent healthcare assistant — Google ADK edition.

PATTERN: identical to single_agent_raw.py (one LLM + a fixed set of tools + a system
prompt, looping reason -> act -> observe until done). The difference is HOW the
agent loop is wired: ADK's Agent class handles the ReAct loop natively — no manual
StateGraph, agent_node, tool_node, or route_after_agent needed. Read this file next
to single_agent_raw.py and compare section "4. THE AGENT" with that file's
"4. THE GRAPH".

LangGraph → ADK mapping:
  ChatOpenAI + OpenAI base_url      →  Agent(model="gemini-2.5-flash")  [native Gemini]
  @tool + convert_to_openai_tool()  →  plain Python functions            [ADK reads type hints]
  tool_schemas / tools_by_name      →  not needed                        [ADK resolves internally]
  agent_node (SystemMessage + llm)  →  Agent(instruction=...)            [built-in]
  tool_node (loop tool_calls)       →  Agent(tools=[...])                [built-in]
  route_after_agent (cond. edge)    →  built-in ReAct loop in LlmAgent
  StateGraph + add_edge + compile() →  one Agent(...) declaration
  graph.stream(...)                 →  Runner.run_async(...)  via asyncio

MODEL: Gemini, called natively through google-genai (no OpenAI-compatible wrapper).

SETUP:
  pip install google-adk python-dotenv
  add GOOGLE_API_KEY="your_key_from_aistudio.google.com" to a .env file
  python single_agent_raw_adk.py
"""

import asyncio
import json

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()


# ---------------------------------------------------------------------------
# 1. THE TOOLS  —  plain Python functions; ADK generates OpenAI-style schemas
#    from type hints + docstrings automatically.  No @tool decorator,
#    no convert_to_openai_tool(), no tools_by_name dict needed.
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


def get_patient_record(patient_id: str) -> str:
    """Look up a patient by ID. Returns name, age, current medications and allergies."""
    patient = _PATIENTS.get(patient_id.upper())
    if not patient:
        return f"No patient found with ID {patient_id}."
    return json.dumps(patient)


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


def book_appointment(patient_id: str, preferred_date: str, reason: str) -> str:
    """Book a clinic appointment. preferred_date must be YYYY-MM-DD. Returns a confirmation."""
    patient = _PATIENTS.get(patient_id.upper())
    if not patient:
        return f"Cannot book: no patient found with ID {patient_id}."
    patient["next_appointment"] = preferred_date
    return f"Appointment confirmed for {patient['name']} on {preferred_date}. Reason: {reason}."


# ---------------------------------------------------------------------------
# 2. THE SYSTEM PROMPT  —  defines task, persona, tool conditions, guardrails
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
# 3. THE AGENT  —  one declaration replaces the entire LangGraph graph section:
#    StateGraph + agent_node + tool_node + route_after_agent + all add_edge calls.
#    ADK's LlmAgent runs the ReAct loop (reason -> act -> observe) internally.
# ---------------------------------------------------------------------------
agent = Agent(
    name="clinic_concierge",
    model="gemini-2.5-flash",
    instruction=SYSTEM_PROMPT,
    tools=[get_patient_record, check_drug_interactions, book_appointment],
)


# ---------------------------------------------------------------------------
# 4. RUN IT  —  Runner + InMemorySessionService replaces graph.stream().
#    Must be async because runner.run_async() is a coroutine.
# ---------------------------------------------------------------------------
async def run(user_message: str) -> None:
    """Stream every event so you can watch the reason -> act -> observe loop."""
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="clinic_concierge", user_id="user", session_id="s1"
    )
    runner = Runner(
        agent=agent,
        app_name="clinic_concierge",
        session_service=session_service,
    )

    async for event in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=types.Content(
            role="user", parts=[types.Part.from_text(text=user_message)]
        ),
    ):
        if not event.content or not event.content.parts:
            continue
        for part in event.content.parts:
            if part.function_call:
                # Agent is requesting a tool — mirrors AIMessage with tool_calls in LangGraph
                print(f"\n[Tool Call]  {part.function_call.name}({json.dumps(dict(part.function_call.args), indent=2)})")
            elif part.function_response:
                # Tool returned a result — mirrors ToolMessage in LangGraph
                print(f"[Tool Result] {part.function_response.name}: {part.function_response.response}")
            elif part.text:
                # LLM text: label as Final Answer on the last event, agent name otherwise
                label = "Final Answer" if event.is_final_response() else event.author
                print(f"\n[{label}]: {part.text}")


if __name__ == "__main__":
    asyncio.run(
        run(
            "Hi, I'm patient P001. Can you check whether my current medications "
            "interact, and book me a follow-up for 2026-06-29 about my blood pressure?"
        )
    )
