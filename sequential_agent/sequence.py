
"""
Healthcare Demo — Sequential Multi-Agent Pipeline (raw LangGraph)
=================================================================

Problem: Patient Intake & Triage Assistant
A patient describes their complaint in plain language. Three specialized
agents run in a FIXED linear order, each agent's output feeding the next:

    [Symptom Extractor]  ->  [Triage Assessor]  ->  [Care Recommender]

This is the "sequential pattern": the order is hard-coded in the graph
edges. No LLM decides what runs next, which keeps it cheap, fast, and
predictable.

NOTE: This is a demo for routing/triage education only. It does not
diagnose and is not a substitute for a clinician.

Run:
    pip install langgraph langchain-anthropic
    export ANTHROPIC_API_KEY=sk-...
    python triage_pipeline.py
"""

import json
from typing import TypedDict, List, Literal

from langgraph.graph import StateGraph, START, END
# from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

load_dotenv()

# One shared model instance. Swap for ChatOpenAI etc. if you prefer.
llm = ChatOpenAI(
    model="gemini-2.5-flash",  # swap for a newer one (e.g. "gemini-3.5-flash") if available
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0,
)

# ---------------------------------------------------------------------------
# 1. Shared State
# ---------------------------------------------------------------------------
# Every node reads from and writes to this dict. It is the "conveyor belt"
# that carries data from one agent to the next.
class TriageState(TypedDict):
    complaint: str                 # raw patient input
    symptoms: dict                 # filled by extractor
    urgency: dict                  # filled by assessor
    recommendation: str            # filled by recommender


def _ask_json(system: str, user: str) -> dict:
    """Call the LLM and parse a JSON object out of its reply."""
    raw = llm.invoke(
        [("system", system + " Respond with ONLY a JSON object, no prose."),
         ("human", user)]
    ).content
    # Strip code fences if the model added them.
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# 2. The three agents (each is just a node function)
# ---------------------------------------------------------------------------
def extract_symptoms(state: TriageState) -> dict:
    """Agent 1: free text -> structured symptoms."""
    data = _ask_json(
        system=(
            "You are a clinical intake assistant. Extract structured symptom "
            "data from a patient's complaint."
        ),
        user=(
            f"Patient complaint: \"{state['complaint']}\"\n\n"
            "Return keys: "
            "symptoms (list of strings), "
            "duration (string, e.g. '2 days' or 'unknown'), "
            "red_flags (list of any alarming symptoms like chest pain, "
            "trouble breathing, confusion, severe bleeding)."
        ),
    )
    print("🩺  [Extractor] ", json.dumps(data, indent=2))
    return {"symptoms": data}


def assess_urgency(state: TriageState) -> dict:
    """Agent 2: structured symptoms -> urgency level."""
    data = _ask_json(
        system=(
            "You are a triage nurse. Given structured symptoms, assign an "
            "urgency level. Be conservative: when in doubt, escalate."
        ),
        user=(
            f"Structured symptoms:\n{json.dumps(state['symptoms'], indent=2)}\n\n"
            "Return keys: "
            "level (one of: SELF_CARE, SEE_DOCTOR, URGENT, EMERGENCY), "
            "reason (one short sentence)."
        ),
    )
    print("🚦  [Assessor]  ", json.dumps(data, indent=2))
    return {"urgency": data}


def recommend_care(state: TriageState) -> dict:
    """Agent 3: urgency level -> patient-friendly next steps."""
    msg = llm.invoke([
        ("system",
         "You are a patient-facing care advisor. Write calm, clear next-step "
         "guidance in 2-3 sentences. Always remind the patient this is not a "
         "diagnosis and to seek professional care if symptoms worsen."),
        ("human",
         f"Symptoms: {json.dumps(state['symptoms'])}\n"
         f"Triage level: {state['urgency']['level']} "
         f"({state['urgency']['reason']})\n\n"
         "Give the patient their recommended next steps."),
    ]).content
    print("💬  [Recommender]\n", msg)
    return {"recommendation": msg}


# ---------------------------------------------------------------------------
# 3. Wire the sequential graph
# ---------------------------------------------------------------------------
# The linear edges ARE the orchestration. No model decides the path.
def build_pipeline():
    g = StateGraph(TriageState)

    g.add_node("extract", extract_symptoms)
    g.add_node("assess", assess_urgency)
    g.add_node("recommend", recommend_care)

    g.add_edge(START, "extract")
    g.add_edge("extract", "assess")
    g.add_edge("assess", "recommend")
    g.add_edge("recommend", END)

    return g.compile()


def show_graph(app):
    # (a) ASCII art straight to the terminal — needs `pip install grandalf`
    # print(app.get_graph().draw_ascii())
 
    # (b) Mermaid text — paste into any mermaid viewer or a markdown file
    # print(app.get_graph().draw_mermaid())
 
    # (c) PNG file — needs internet (uses mermaid.ink) OR a local renderer
    try:
        png = app.get_graph().draw_mermaid_png()
        with open("triage_graph.png", "wb") as f:
            f.write(png)
        print("Saved triage_graph.png")
    except Exception as e:
        print(f"(PNG skipped: {e})")

# ---------------------------------------------------------------------------
# 4. Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pipeline = build_pipeline()
    show_graph(pipeline)
    # complaint = (
    #     "I've had a fever around 101 for two days, a sore throat, "
    #     "and a mild headache. No trouble breathing."
    # )
    complaint = (
    "i have sore throat from today morning " )
    print("=" * 60)
    print("PATIENT:", complaint)
    print("=" * 60)

    final = pipeline.invoke({"complaint": complaint})

    print("\n" + "=" * 60)
    print("FINAL TRIAGE:", final["urgency"]["level"])
    print("=" * 60)