"""
APEX — Live-Map Risk Analyst Agent
======================================
An interactive ADK agent for the Live Map. It answers an operator's questions
about the current geopolitical risk picture — corridors, supplier exposure,
sanctions, inbound tanker traffic — by calling tools over the latest Risk-Agent
assessment, and returns structured `map directives` the Live Map applies (fly to
a corridor, highlight exposed suppliers, show vessels).

Same design as the Twin Operator: the agent reasons and decides; the numbers come
from the real assessment (passed in per request) and the live AIS feed. A
deterministic, data-grounded fallback keeps the chat working without the LLM.
"""

import os
import json
import asyncio
import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool

from agents import ais_feed

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

APP_NAME = "apex"
USER_ID = "apex_analyst"
MODEL = "gemini-flash-lite-latest"
_session_service = InMemorySessionService()

# Current assessment the tools read (set per request; chat is low-volume/serialized)
_ctx = {"risk": {}}

CORRIDOR_NAMES = {"hormuz": "Strait of Hormuz", "red_sea": "Red Sea / Bab-el-Mandeb",
                  "opec_policy": "OPEC+ Supply Policy"}


# ─────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────
def analyst_get_assessment() -> dict:
    """Get the current APEX risk assessment headline: alert level, overall
    disruption probability, agent confidence, per-corridor risk scores, and the
    top active risks. Use this first to understand the current picture."""
    r = _ctx["risk"]
    cor = r.get("corridors", {})
    return {
        "alert_level": r.get("alert_level"),
        "overall_disruption_probability": r.get("overall_disruption_probability"),
        "agent_confidence": r.get("agent_confidence"),
        "corridor_scores": {k: cor.get(k, {}).get("risk_score") for k in CORRIDOR_NAMES},
        "top_3_risks": r.get("top_3_risks", []),
        "recommended_action": r.get("recommended_action"),
        "brent_price": r.get("commodity_signals", {}).get("brent_price"),
    }


def analyst_corridor(corridor: str) -> dict:
    """Get detail on one shipping corridor ('hormuz', 'red_sea', or 'opec_policy'):
    its risk score, trend, primary threat, analysis and key signal."""
    c = _ctx["risk"].get("corridors", {}).get(corridor, {})
    return {"corridor": corridor, "name": CORRIDOR_NAMES.get(corridor, corridor), **c}


def analyst_supplier_exposure() -> dict:
    """Get India's crude supplier exposure: each supplier's import share, country
    risk score, and weighted risk contribution — i.e. which suppliers drive India's
    concentration risk. Use when the user asks about suppliers or diversification."""
    r = _ctx["risk"]
    return {
        "supplier_shares": r.get("supplier_shares", {}),
        "exposure": r.get("supplier_risk_exposure", {}),
        "supplier_country_risk": r.get("supplier_country_risk", {}),
    }


def analyst_sanctions() -> dict:
    """Get the current sanctions-pressure read (Iran / Russia levels and combined
    impact) from the OFAC-driven assessment."""
    return _ctx["risk"].get("sanctions_pressure", {})


def analyst_vessel_traffic() -> dict:
    """Get the current count of tankers by region (Strait of Hormuz, Red Sea, India
    west/east coasts) from the live aisstream.io AIS feed. Fewer inbound tankers on a
    corridor is a near-term supply-risk signal."""
    v = ais_feed.get_vessels()
    return {"by_region": v["by_region"], "total": v["count"], "is_live": v["is_live"], "source": v["source"]}


def analyst_historical_precedent(query: str) -> dict:
    """Semantic search (RAG) over 16 real historical disruption events and econometric/policy
    reference documents. Use when the operator asks 'has this happened before' or wants grounding in
    past precedent rather than just the current live picture."""
    from agents import rag_engine
    return {"query": query, "retrieved": rag_engine.search(query, k=4)}


def analyst_relationship_context(entity: str) -> dict:
    """Query the supplier-route-refinery knowledge graph for a named entity (e.g. 'Iraq' or
    'Jamnagar'). Returns its real traced relationships to corridors, ports, refineries, and sectors."""
    from agents import knowledge_graph
    info = knowledge_graph.node_neighbors(entity)
    if "error" in info:
        return info
    if info.get("type") == "supplier":
        return {**info, "exposure_chain": knowledge_graph.get_exposure_chain(entity)}
    return info


_TOOLS = [analyst_get_assessment, analyst_corridor, analyst_supplier_exposure,
          analyst_sanctions, analyst_vessel_traffic, analyst_historical_precedent,
          analyst_relationship_context]

RESPONSE_SCHEMA = """
When you have what you need, reply with ONLY valid JSON (no markdown), in this shape:

{
  "narrative": "<2-4 sentences answering the question, in plain analyst language>",
  "assessment": {"alert_level": "LOW|MEDIUM|HIGH|CRITICAL", "disruption_pct": <num>, "confidence_pct": <num>},
  "actions": [
    {"type": "focus_corridor",     "target": "hormuz|red_sea|opec_policy", "rationale": "<why>"},
    {"type": "highlight_suppliers","targets": ["Russia", "Iraq", ...],     "rationale": "<why>"},
    {"type": "show_vessels",       "rationale": "<why>"},
    {"type": "clear"}
  ]
}

Only include actions that help the operator see the answer on the map. An empty
actions list is fine. Ground every number in tool results — never invent figures.
"""


async def _run_adk(message: str, session_id: str) -> dict:
    a = analyst_get_assessment()
    instruction = f"""You are APEX's Risk Analyst for India's energy supply-chain security,
answering an operator's questions on the live geopolitical risk map. You have tools over the
current assessment (corridors, suppliers, sanctions), the live AIS tanker feed, a semantic search
over real historical disruption precedent (analyst_historical_precedent), and a knowledge-graph
query over real supplier-corridor-refinery-sector relationships (analyst_relationship_context).
Decide which tools to call, investigate, then answer concisely and point the operator's map at
the evidence. Use the historical and relationship tools when the operator asks "why" or "has this
happened before" rather than just "what is the current score."

Current headline (for grounding): alert={a['alert_level']}, disruption={a['overall_disruption_probability']}%,
corridor scores={a['corridor_scores']}.

India context: 88% crude import dependence, ~42% via Hormuz, ~18% via Red Sea, SPR ≈ 9.5 days.

{RESPONSE_SCHEMA}"""

    agent = Agent(name="apex_risk_analyst", model=MODEL, instruction=instruction,
                  tools=[FunctionTool(t) for t in _TOOLS])
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=_session_service)
    existing = await _session_service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    if existing is None:
        await _session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    final = None
    async for event in runner.run_async(
        user_id=USER_ID, session_id=session_id,
        new_message=genai_types.Content(role="user", parts=[genai_types.Part(text=message)]),
    ):
        if event.is_final_response() and event.content:
            final = event.content.parts[0].text
    if not final:
        raise RuntimeError("analyst produced no response")
    raw = final.strip().replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    result["agent_mode"] = "agentic (ADK Risk Analyst — tools + multi-turn)"
    return result


# ─────────────────────────────────────────────────────────────
# DETERMINISTIC FALLBACK
# ─────────────────────────────────────────────────────────────
def _deterministic(message: str) -> dict:
    t = message.lower()
    r = _ctx["risk"]
    cor = r.get("corridors", {})
    a = analyst_get_assessment()
    actions, bits = [], []

    focus = None
    if "hormuz" in t:
        focus = "hormuz"
    elif "red sea" in t or "bab" in t or "suez" in t:
        focus = "red_sea"
    elif "opec" in t:
        focus = "opec_policy"
    if focus:
        c = cor.get(focus, {})
        actions.append({"type": "focus_corridor", "target": focus,
                        "rationale": f"{CORRIDOR_NAMES[focus]} risk {c.get('risk_score','—')}/100."})
        bits.append(f"{CORRIDOR_NAMES[focus]} is at {c.get('risk_score','—')}/100 "
                    f"({c.get('trend','')}). {c.get('analysis','')}".strip())

    if any(w in t for w in ["supplier", "diversif", "russia", "iraq", "import", "exposure", "concentration"]):
        exp = r.get("supplier_risk_exposure", {})
        top = (exp.get("breakdown") or [])[:3]
        if top:
            actions.append({"type": "highlight_suppliers", "targets": [x["name"] for x in top],
                            "rationale": "Highest weighted risk contributors to India's crude supply."})
            bits.append("Top exposure: " + ", ".join(f"{x['name']} ({x['share_pct']}% share, risk {x['risk_score']})" for x in top) + ".")

    if any(w in t for w in ["vessel", "tanker", "ship", "traffic", "ais"]):
        v = analyst_vessel_traffic()
        actions.append({"type": "show_vessels", "rationale": "Show live tanker positions."})
        bits.append(f"Live AIS: {v['total']} tankers tracked ({'live feed' if v['is_live'] else 'awaiting coverage'}).")

    if not bits:
        bits.append(f"Current alert is {a['alert_level']} with {a['overall_disruption_probability']}% disruption "
                    f"probability. Highest corridor: " +
                    max(a["corridor_scores"].items(), key=lambda x: x[1] or 0)[0].replace('_', ' ') + ".")
        tr = a.get("top_3_risks") or []
        if tr:
            bits.append("Top risk: " + tr[0] + ".")

    return {
        "narrative": " ".join(bits),
        "assessment": {"alert_level": a["alert_level"], "disruption_pct": a["overall_disruption_probability"],
                       "confidence_pct": a["agent_confidence"]},
        "actions": actions,
        "agent_mode": "deterministic analyst (LLM unavailable — assessment-grounded)",
    }


def run_risk_analyst(message: str, session_id: str, risk_data: dict) -> dict:
    _ctx["risk"] = risk_data or {}
    try:
        return asyncio.run(_run_adk(message, session_id))
    except Exception as e:
        print(f"  Risk Analyst ADK failed ({str(e)[:80]}) — deterministic fallback")
        return _deterministic(message)


if __name__ == "__main__":
    _ctx["risk"] = {
        "alert_level": "MEDIUM", "overall_disruption_probability": 35, "agent_confidence": 90,
        "corridors": {"hormuz": {"risk_score": 45, "trend": "stable", "analysis": "Elevated Gulf tension."},
                      "red_sea": {"risk_score": 25}, "opec_policy": {"risk_score": 30}},
        "top_3_risks": ["Hormuz escalation", "Russia sanctions", "Libya instability"],
        "supplier_risk_exposure": {"breakdown": [{"name": "Russia", "share_pct": 31, "risk_score": 80},
                                                  {"name": "Iraq", "share_pct": 18, "risk_score": 50}]},
    }
    for m in ["Why is Hormuz risky?", "What's our supplier exposure?", "Show me the tankers", "Give me the overall picture"]:
        r = _deterministic(m)
        print(f"\nQ: {m}\n   {r['narrative'][:150]}\n   actions={[a['type'] for a in r['actions']]}")
