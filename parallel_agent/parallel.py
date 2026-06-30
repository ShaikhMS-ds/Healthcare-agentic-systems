

import asyncio
import os

from dotenv import load_dotenv
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

APP_NAME = "lab_interpreter"
USER_ID = "patient_demo"
SESSION_ID = "labreport_001"


# ---------------------------------------------------------------------------
# 1) THE PARALLEL PANEL SPECIALISTS
#    Independent LlmAgents. Each pulls ITS panel out of the same report and
#    writes its interpretation to a DISTINCT output_key in shared state
#    (distinct keys = no race conditions when writing concurrently).
# ---------------------------------------------------------------------------

cbc_agent = LlmAgent(
    name="CBCInterpreter",
    model=MODEL,
    description="Interprets the complete blood count panel.",
    instruction=(
        "You interpret the Complete Blood Count (CBC) panel. From the lab "
        "report, look ONLY at CBC values (e.g., hemoglobin, hematocrit, WBC, "
        "RBC, platelets). For each value, say whether it is normal, high, or "
        "low, and in one simple sentence what that can suggest (e.g., low "
        "hemoglobin can suggest anemia). Use plain words, no jargon. "
        "If a value is missing, skip it."
    ),
    output_key="cbc",
)

metabolic_agent = LlmAgent(
    name="MetabolicInterpreter",
    model=MODEL,
    description="Interprets the basic/comprehensive metabolic panel.",
    instruction=(
        "You interpret the Metabolic panel. From the lab report, look ONLY at "
        "metabolic values (e.g., glucose, HbA1c, creatinine, BUN, sodium, "
        "potassium, ALT, AST). For each, say normal/high/low and in one simple "
        "sentence what it can indicate (e.g., high glucose/HbA1c relates to "
        "blood sugar control). Plain words only. Skip missing values."
    ),
    output_key="metabolic",
)

lipid_agent = LlmAgent(
    name="LipidInterpreter",
    model=MODEL,
    description="Interprets the lipid (cholesterol) panel.",
    instruction=(
        "You interpret the Lipid panel. From the lab report, look ONLY at "
        "cholesterol values (total cholesterol, LDL, HDL, triglycerides). "
        "For each, say normal/high/low and in one simple sentence why it "
        "matters for heart health (e.g., high LDL is the 'bad' cholesterol). "
        "Plain words only. Skip missing values."
    ),
    output_key="lipid",
)

thyroid_agent = LlmAgent(
    name="ThyroidInterpreter",
    model=MODEL,
    description="Interprets the thyroid panel.",
    instruction=(
        "You interpret the Thyroid panel. From the lab report, look ONLY at "
        "thyroid values (e.g., TSH, Free T4, Free T3). For each, say "
        "normal/high/low and in one simple sentence what it can suggest "
        "(e.g., high TSH can suggest an underactive thyroid). Plain words "
        "only. Skip missing values."
    ),
    output_key="thyroid",
)

# Deterministic fan-out: ADK runs all four panels concurrently, each in its
# own branch, and waits for all to finish. No LLM orchestrates this.
parallel_panels = ParallelAgent(
    name="ParallelPanelReview",
    sub_agents=[cbc_agent, metabolic_agent, lipid_agent, thyroid_agent],
    description="Interprets all four lab panels at the same time.",
)


# ---------------------------------------------------------------------------
# 2) THE GATHER / SYNTHESIZER (fan-in)
#    Reads all four panel interpretations via {curly} placeholders and writes
#    ONE friendly summary. It groups the abnormal results so the patient sees
#    what actually needs attention first.
# ---------------------------------------------------------------------------

synthesizer_agent = LlmAgent(
    name="ReportSynthesizer",
    model=MODEL,
    description="Combines the four panel interpretations into one patient summary.",
    instruction=(
        "You are writing a friendly, plain-language summary of a patient's "
        "blood test using the four panel interpretations below. Do NOT add new "
        "medical claims; only summarize what the panels say.\n\n"
        "CBC:\n{cbc}\n\n"
        "METABOLIC:\n{metabolic}\n\n"
        "LIPID:\n{lipid}\n\n"
        "THYROID:\n{thyroid}\n\n"
        "Write the summary in this order:\n"
        "1. OVERALL: one warm sentence on the general picture.\n"
        "2. WHAT LOOKS GOOD: values that are normal.\n"
        "3. WORTH A CLOSER LOOK: any high/low values, grouped simply, in calm "
        "non-alarming language.\n"
        "4. QUESTIONS TO ASK YOUR DOCTOR: 2-3 specific questions.\n"
        "End with: 'This is a plain-language summary, not a diagnosis. Please "
        "review your results with your doctor.'"
    ),
)


# ---------------------------------------------------------------------------
# 3) ROOT WORKFLOW — panels finish, THEN the synthesizer runs.
# ---------------------------------------------------------------------------

root_agent = SequentialAgent(
    name="LabReportPipeline",
    sub_agents=[parallel_panels, synthesizer_agent],
    description="Parallel panel interpretation followed by a synthesized summary.",
)


# ---------------------------------------------------------------------------
# 4) RUNNER — drives one lab report through the pipeline
# ---------------------------------------------------------------------------

SAMPLE_LAB_REPORT = """
LAB REPORT — Patient: 45F

Complete Blood Count (CBC):
  Hemoglobin .......... 10.8 g/dL   (normal 12.0-15.5)
  Hematocrit .......... 33%         (normal 36-46)
  WBC ................. 6.2 x10^9/L (normal 4.0-11.0)
  Platelets ........... 250 x10^9/L (normal 150-400)

Metabolic Panel:
  Fasting Glucose ..... 118 mg/dL   (normal 70-99)
  HbA1c ............... 6.1%        (normal < 5.7)
  Creatinine .......... 0.9 mg/dL   (normal 0.6-1.1)
  ALT ................. 22 U/L      (normal 7-56)

Lipid Panel:
  Total Cholesterol ... 232 mg/dL   (desirable < 200)
  LDL ................. 158 mg/dL   (optimal < 100)
  HDL ................. 45 mg/dL    (normal > 40)
  Triglycerides ....... 180 mg/dL   (normal < 150)

Thyroid Panel:
  TSH ................. 2.1 mIU/L   (normal 0.4-4.0)
  Free T4 ............. 1.1 ng/dL   (normal 0.8-1.8)
"""


async def main():
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    message = types.Content(role="user", parts=[types.Part(text=SAMPLE_LAB_REPORT)])

    print("=" * 70)
    print("RUNNING PARALLEL LAB-RESULT INTERPRETER")
    print("=" * 70)

    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=message
    ):
        if event.is_final_response() and event.content:
            print("\n--- PATIENT-FRIENDLY SUMMARY ---\n")
            print(event.content.parts[0].text)

    # Inspect each panel's raw interpretation from shared state.
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    print("\n--- RAW PARALLEL OUTPUTS (from session.state) ---")
    for key in ("cbc", "metabolic", "lipid", "thyroid"):
        print(f"\n[{key}]\n{session.state.get(key, '(missing)')}")


if __name__ == "__main__":
    asyncio.run(main())