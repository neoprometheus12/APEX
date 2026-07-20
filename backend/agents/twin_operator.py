"""
APEX — Digital Twin Operator Agent
======================================
A genuinely agentic operator for the supply-chain twin. It is an ADK Agent that:
  • PERCEIVES the network via deterministic tools (flow simulation + live vessels),
  • REASONS about cascading failure and demand shortfall,
  • DECIDES a mitigation plan (offline nodes, port stress, SPR drawdown, reroute),
  • ACTS by returning structured `map_directives` the frontend applies live,
and does this in a multi-turn chat so an operator can converse with the twin.

The agent owns the *judgement*; the exact physics stays in twin_sim.py (the agent
calls it as a tool, it never invents flow numbers). A deterministic fallback keeps
the chat working when the LLM is rate-limited.
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

from agents import twin_sim
from agents import ais_feed

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

APP_NAME = "apex"
USER_ID = "apex_operator"
MODEL = "gemini-flash-lite-latest"
_session_service = InMemorySessionService()


# ─────────────────────────────────────────────────────────────
# TOOLS — deterministic, exact. The agent decides when to call them.
# ─────────────────────────────────────────────────────────────
def twin_get_state() -> dict:
    """Get the current baseline state of India's crude→product supply network:
    national demand vs supply, per-city coverage, and overall network health,
    with no outages applied. Use this to understand the healthy baseline."""
    return twin_sim.simulate()


def twin_simulate(offline_nodes: list, stressed_ports: list) -> dict:
    """Run the flow simulation with the given refineries taken offline and the
    given ports stressed (50% capacity). Returns exact per-city unmet demand,
    national shortfall, served %, and network health after rerouting. Use this to
    test 'what if X fails' and to check whether a mitigation actually helps.
    Pass node/port ids from twin_list_nodes (e.g. ['jamnagar'], ['mundra'])."""
    stress = {p: 50 for p in (stressed_ports or [])}
    return twin_sim.simulate(offline=offline_nodes or [], port_stress=stress)


def twin_rank_critical() -> dict:
    """Rank refineries by how much national demand shortfall their loss would cause
    (an N-1 single-point-of-failure analysis). Use this to identify the network's
    most critical assets."""
    return {"critical_refineries": twin_sim.node_criticality(8)}


def twin_vessel_inflow() -> dict:
    """Get the current count of tankers approaching by region (Strait of Hormuz,
    Red Sea, India west/east coasts) from the live AIS feed. More vessels inbound =
    healthier near-term crude supply; a drop near Hormuz signals inflow risk."""
    v = ais_feed.get_vessels()
    return {"by_region": v["by_region"], "total": v["count"], "is_live": v["is_live"], "source": v["source"]}


def twin_scenario_stress(scenario: str) -> dict:
    """Translate a geopolitical corridor scenario ('hormuz', 'red_sea', or
    'opec_policy') into which ports it stresses, then simulate the network impact.
    Use this when the user asks about a corridor closure."""
    _, port_stress = twin_sim.corridor_to_offline(scenario)
    sim = twin_sim.simulate(port_stress=port_stress)
    return {"scenario": scenario, "ports_stressed": list(port_stress), "impact": sim}


def twin_list_nodes() -> dict:
    """List the valid node ids and names for refineries, ports, wellheads, SPR, and
    demand centres — so actions reference real ids."""
    geo = twin_sim.all_nodes_geo()
    return {role: [{"id": n["id"], "name": n["name"]} for n in nodes] for role, nodes in geo.items()}


_TOOLS = [twin_get_state, twin_simulate, twin_rank_critical, twin_vessel_inflow, twin_scenario_stress, twin_list_nodes]

RESPONSE_SCHEMA = """
When you have gathered enough from the tools, reply with ONLY valid JSON (no
markdown, no prose outside it), in exactly this shape:

{
  "narrative": "<2-4 sentences: the situation and your plan, in plain operator language>",
  "assessment": {"network_health": "NORMAL|ELEVATED|CRITICAL", "served_pct": <num>, "cities_at_risk": ["<city>", ...]},
  "actions": [
    {"type": "set_offline",   "targets": ["<refinery_id>", ...], "rationale": "<why>"},
    {"type": "stress_ports",  "targets": ["<port_id>", ...],     "rationale": "<why>"},
    {"type": "drawdown_spr",  "value": <days 0-9.5>,             "rationale": "<why>"},
    {"type": "highlight",     "targets": ["<node_id>", ...],     "rationale": "<what to watch>"},
    {"type": "clear"}
  ],
  "projected_served_pct": <num or null>
}

Only include actions that are actually warranted — an empty actions list is fine
if nothing needs to change. `set_offline`/`stress_ports` should reflect the shock
the user described; `drawdown_spr`/`reroute`/`highlight` are your mitigations.
Always ground numbers in tool results — never invent flow figures.
"""


async def _run_adk(message: str, session_id: str, context: dict) -> dict:
    ctx_txt = ""
    if context:
        off = context.get("offline") or []
        scn = context.get("scenario")
        if off:
            ctx_txt += f"\nCurrently OFFLINE on the operator's map: {off}."
        if scn:
            ctx_txt += f"\nActive corridor scenario: {scn}."

    instruction = f"""You are APEX's Digital Twin Operator — an autonomous agent supervising
India's crude-oil supply network (18 refineries, 8 import ports, 5 domestic wellheads,
3 strategic reserves, feeding 6 major demand centres).

You have tools to inspect the live network, simulate outages, rank critical assets, and
read inbound tanker traffic. Decide which tools to call and in what order. Investigate
the user's question with the tools, reason about cascading impact and demand shortfall,
then produce a concrete mitigation plan.

India context: 88% crude import dependence, ~42% via Strait of Hormuz, ~18% via Red Sea,
SPR ≈ 9.5 days national cover. Your objective: keep every demand centre supplied while
minimising SPR burn and import cost.{ctx_txt}

{RESPONSE_SCHEMA}"""

    agent = Agent(name="apex_twin_operator", model=MODEL, instruction=instruction,
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
        raise RuntimeError("operator produced no response")
    raw = final.strip().replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    result["agent_mode"] = "agentic (ADK Twin Operator — tools + multi-turn)"
    return result


# ─────────────────────────────────────────────────────────────
# DETERMINISTIC FALLBACK — keyword-driven, still data-grounded
# ─────────────────────────────────────────────────────────────
_REF_BY_NAME = {r["name"].lower(): r["id"] for r in twin_sim.REFINERIES}
_REF_BY_ID = {r["id"]: r for r in twin_sim.REFINERIES}


def _match_refineries(text: str) -> list:
    t = text.lower()
    hits = []
    for name, rid in _REF_BY_NAME.items():
        base = name.split()[0]
        if base in t and rid not in hits:
            hits.append(rid)
    return hits


def _deterministic(message: str, context: dict) -> dict:
    t = message.lower()
    offline = list((context or {}).get("offline") or [])
    port_stress = {}
    scenario = None

    if "hormuz" in t:
        scenario = "hormuz"
        _, port_stress = twin_sim.corridor_to_offline("hormuz")
    elif "red sea" in t or "red_sea" in t or "bab" in t or "suez" in t:
        scenario = "red_sea"
        _, port_stress = twin_sim.corridor_to_offline("red_sea")

    mentioned = _match_refineries(t)
    if any(w in t for w in ["offline", "down", "fails", "lost", "attack", "fire", "shut"]):
        for rid in mentioned:
            if rid not in offline:
                offline.append(rid)

    sim = twin_sim.simulate(offline=offline, port_stress=port_stress)
    at_risk = [d["name"] for d in sim["demand_centres"] if d["status"] in ("UNMET", "TIGHT")]

    actions = []
    if offline:
        actions.append({"type": "set_offline", "targets": offline,
                        "rationale": "Assets flagged offline in the described disruption."})
    if port_stress:
        actions.append({"type": "stress_ports", "targets": list(port_stress),
                        "rationale": f"{scenario} corridor disruption throttles these import ports."})
    if sim["national_shortfall_mmtpa"] > 1:
        spr_days = min(9.5, round(sim["national_shortfall_mmtpa"] / twin_sim.TOTAL_REFINING * 9.5 * 3, 1))
        actions.append({"type": "drawdown_spr", "value": spr_days,
                        "rationale": f"Bridge the {sim['national_shortfall_mmtpa']} MMTPA shortfall while supply reroutes."})
        crit = twin_sim.node_criticality(3)
        actions.append({"type": "highlight", "targets": [c["id"] for c in crit],
                        "rationale": "Most critical remaining refineries — protect these first."})
    if not offline and not port_stress:
        actions.append({"type": "clear"})

    if at_risk:
        narr = (f"Simulated the disruption on the live flow network: national supply falls to "
                f"{sim['served_pct']}% ({sim['network_health']}). At-risk cities: {', '.join(at_risk)}. "
                f"Recommend SPR drawdown to bridge the {sim['national_shortfall_mmtpa']} MMTPA gap and "
                f"protecting the critical refineries highlighted.")
    else:
        narr = (f"Ran the flow simulation — the network holds at {sim['served_pct']}% served "
                f"({sim['network_health']}); no demand centre is short under this condition.")

    return {
        "narrative": narr,
        "assessment": {"network_health": sim["network_health"], "served_pct": sim["served_pct"],
                       "cities_at_risk": at_risk},
        "actions": actions,
        "projected_served_pct": sim["served_pct"],
        "sim": sim,
        "agent_mode": "deterministic operator (LLM unavailable — sim-grounded heuristic)",
    }


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY
# ─────────────────────────────────────────────────────────────
def run_twin_operator(message: str, session_id: str = "twin_default", context: dict = None) -> dict:
    context = context or {}
    try:
        result = asyncio.run(_run_adk(message, session_id, context))
        # attach the exact sim of the agent's plan so the frontend has ground truth
        offline, ports = _extract_plan(result, context)
        result["sim"] = twin_sim.simulate(offline=offline, port_stress={p: 50 for p in ports})
        return result
    except Exception as e:
        print(f"  Twin Operator ADK failed ({str(e)[:80]}) — deterministic fallback")
        return _deterministic(message, context)


def _extract_plan(result: dict, context: dict) -> tuple:
    offline = list((context or {}).get("offline") or [])
    ports = []
    for a in result.get("actions", []):
        if a.get("type") == "set_offline":
            for x in a.get("targets", []):
                if x in _REF_BY_ID and x not in offline:
                    offline.append(x)
        elif a.get("type") == "stress_ports":
            ports.extend(a.get("targets", []))
        elif a.get("type") == "clear":
            offline, ports = [], []
    return offline, ports


if __name__ == "__main__":
    print("=== deterministic fallback tests ===")
    for msg in ["What happens if the Strait of Hormuz closes?",
                "Jamnagar refinery is on fire and offline, what do we do?",
                "Is the network healthy right now?"]:
        r = _deterministic(msg, {})
        print(f"\nQ: {msg}")
        print(f"   health={r['assessment']['network_health']} served={r['assessment']['served_pct']}% "
              f"at_risk={r['assessment']['cities_at_risk']}")
        print(f"   actions={[a['type'] for a in r['actions']]}")
        print(f"   {r['narrative'][:120]}...")
