"""
Healthcare Demo — Sequential Multi-Agent Pipeline (Google ADK)
=================================================================

Same problem as sequence.py: Patient Intake & Triage Assistant.
Three specialized agents run in a FIXED linear order, each agent's
output feeding the next — but wired using ADK primitives instead of
a LangGraph StateGraph.

    [Symptom Extractor]  ->  [Triage Assessor]  ->  [Care Recommender]

LangGraph → ADK mapping:
  TriageState TypedDict              →  Session state + output_key per agent
  StateGraph + add_edge (linear)     →  SequentialAgent(sub_agents=[...])
  START→extract→assess→recommend→END →  SequentialAgent runs sub_agents in order
  _ask_json(system, user)            →  Agent(output_schema=PydanticModel)
  return {"symptoms": data} in node  →  Agent(output_key="symptoms")
  state["symptoms"] in next node     →  {symptoms} injected in next instruction
  graph.compile().invoke({...})      →  Runner.run_async(...) via asyncio
  ChatOpenAI via OpenAI-compat URL   →  Agent(model="gemini-2.5-flash") native

NOTE: This is a demo for routing/triage education only. It does not
diagnose and is not a substitute for a clinician.

Run:
    pip install google-adk python-dotenv
    export GOOGLE_API_KEY=...
    python sequence_adk.py
"""

import asyncio
import json
from typing import Literal

from dotenv import load_dotenv
from google.adk.agents import Agent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Structured output schemas  (replace _ask_json + manual JSON parsing)
#    output_schema forces the LLM to return valid JSON — no code-fence
#    stripping or json.loads() needed. ADK handles it internally.
# ---------------------------------------------------------------------------
class SymptomsData(BaseModel):
    symptoms: list[str]
    duration: str
    red_flags: list[str]


class UrgencyData(BaseModel):
    level: Literal["SELF_CARE", "SEE_DOCTOR", "URGENT", "EMERGENCY"]
    reason: str


# ---------------------------------------------------------------------------
# 2. The three agents  (each replaces one node function from sequence.py)
#
#    output_key  — stores the agent's final text in session.state[key]
#    {key}       — injects that state value into the next agent's instruction
#    output_schema — constrains the model's reply to the Pydantic shape
# ---------------------------------------------------------------------------
extractor = Agent(
    name="extractor",
    model="gemini-2.5-flash",
    instruction=(
        "You are a clinical intake assistant. "
        "Extract structured symptom data from the patient's complaint. "
        "Return symptoms (list of strings), duration (e.g. '2 days' or 'unknown'), "
        "and red_flags (list of any alarming symptoms like chest pain, "
        "trouble breathing, confusion, severe bleeding)."
    ),
    output_schema=SymptomsData,  # forces JSON — mirrors _ask_json in sequence.py
    output_key="symptoms",       # saved to state["symptoms"] for assessor
)

assessor = Agent(
    name="assessor",
    model="gemini-2.5-flash",
    instruction=(
        "You are a triage nurse. Given these structured symptoms:\n{symptoms}\n\n"
        "Assign an urgency level. Be conservative: when in doubt, escalate. "
        "Return level (one of: SELF_CARE, SEE_DOCTOR, URGENT, EMERGENCY) "
        "and reason (one short sentence)."
    ),
    output_schema=UrgencyData,  # mirrors _ask_json in assess_urgency()
    output_key="urgency",       # saved to state["urgency"] for recommender
)

recommender = Agent(
    name="recommender",
    model="gemini-2.5-flash",
    instruction=(
        "You are a patient-facing care advisor. "
        "Symptoms: {symptoms}\n"
        "Triage assessment: {urgency}\n\n"
        "Write calm, clear next-step guidance in 2-3 sentences. "
        "Always remind the patient this is not a diagnosis and to seek "
        "professional care if symptoms worsen."
    ),
    output_key="recommendation",  # plain text — no output_schema needed
)


# ---------------------------------------------------------------------------
# 3. Wire the sequential pipeline
#    SequentialAgent replaces the entire StateGraph + add_edge chain.
#    The list order IS the orchestration — no LLM decides what runs next.
#    (ADK 2.3 emits a deprecation warning suggesting "Workflow" — but that
#    import doesn't exist yet in this version; SequentialAgent is correct.)
# ---------------------------------------------------------------------------
pipeline = SequentialAgent(
    name="triage_pipeline",
    sub_agents=[extractor, assessor, recommender],
)


# ---------------------------------------------------------------------------
# 4. Run  (replaces pipeline.invoke() + in-node print statements)
#    Runner.run_async() streams events from all three sub-agents in order.
# ---------------------------------------------------------------------------
_LABELS = {
    "extractor":   "🩺  [Extractor] ",
    "assessor":    "🚦  [Assessor]  ",
    "recommender": "💬  [Recommender]",
}


async def run(complaint: str) -> None:
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="triage_pipeline", user_id="user", session_id="s1"
    )
    runner = Runner(
        agent=pipeline,
        app_name="triage_pipeline",
        session_service=session_service,
    )

    print("=" * 60)
    print("PATIENT:", complaint)
    print("=" * 60)

    final_urgency_level = None

    async for event in runner.run_async(
        user_id="user",
        session_id=session.id,
        new_message=types.Content(
            role="user", parts=[types.Part.from_text(text=complaint)]
        ),
    ):
        if event.author not in _LABELS:
            continue
        if not event.is_final_response():
            continue
        if not event.content or not event.content.parts:
            continue
        for part in event.content.parts:
            if part.text:
                print(f"\n{_LABELS[event.author]}\n{part.text}")
                if event.author == "assessor":
                    try:
                        final_urgency_level = json.loads(part.text).get("level")
                    except (json.JSONDecodeError, AttributeError):
                        pass

    if final_urgency_level:
        print("\n" + "=" * 60)
        print("FINAL TRIAGE:", final_urgency_level)
        print("=" * 60)


if __name__ == "__main__":
    complaint = "i have sore throat from today morning"
    asyncio.run(run(complaint))
