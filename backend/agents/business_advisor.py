"""
APEX — Business Impact Advisor Agent
=========================================
The 6th genuine ADK agent — and the one built to be fully agentic end-to-end.

Earlier version: the backend always pre-computed the financial impact and
only asked the agent to narrate a fixed result. That's decoration, not
agency. Here, the agent receives the raw business profile and the live
market context (cascade shocks, FX, commodity references, macro data) and
DECIDES for itself to call the calculator before reasoning — the actual
arithmetic call is a tool invocation the model chooses to make, not a
precomputed input it's handed. It can also independently check live
commodity reference prices or India's real macro data mid-conversation
without being told to.

The arithmetic itself (business_impact.calculate_impact) stays deterministic
and auditable — the agent's autonomy is over WHEN and WHY to call it and
what to do with the result, never over the numbers themselves.
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

from agents import business_impact, market_data

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

APP_NAME = "apex"
USER_ID = "apex_business_owner"
MODEL = "gemini-flash-lite-latest"
_session_service = InMemorySessionService()

# Per-session context: {session_id: {"profile","cascade_sectors","fx","impact","scenario_name"}}
# "impact" starts as None — it only gets populated once the agent (or the
# deterministic fallback) actually calls the calculator.
_ctx = {}


# ─────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────
def business_calculate_impact() -> dict:
    """Run APEX's exact financial model for this business against the live
    cascade shock and FX data already loaded for this session. This performs
    the real computation — call it before reasoning about cost or margin
    impact; nothing is computed until you call this."""
    print("  [Business Advisor] Tool call: business_calculate_impact")
    active = _ctx.get("_active", {})
    profile, cascade_sectors, fx = active.get("profile"), active.get("cascade_sectors"), active.get("fx")
    if not profile:
        return {"error": "no business profile loaded for this session"}
    impact = business_impact.calculate_impact(profile, cascade_sectors, fx)
    active["impact"] = impact
    sid = active.get("_session_id")
    if sid and sid in _ctx:
        _ctx[sid]["impact"] = impact
    return impact


def business_get_impact_summary() -> dict:
    """Get the already-computed financial impact for this business (call
    business_calculate_impact first if you haven't yet): current vs projected
    cost/revenue/profit/margin, total cost delta, and the price increase
    needed to stay whole."""
    imp = _ctx.get("_active", {}).get("impact")
    if not imp:
        return {"error": "impact not yet computed — call business_calculate_impact first"}
    return {"current": imp.get("current"), "projected": imp.get("projected"),
            "delta": imp.get("delta"), "logistics": imp.get("logistics"),
            "fx": imp.get("fx"), "buffer_days": imp.get("buffer_days")}


def business_get_material_breakdown() -> dict:
    """Get the per-raw-material breakdown (requires business_calculate_impact
    to have been called first): which materials are driving the cost
    increase, by how much, whether currency risk is a factor, and which are
    protected by fixed-price contracts (deferred, not eliminated)."""
    print("  [Business Advisor] Tool call: business_get_material_breakdown")
    imp = _ctx.get("_active", {}).get("impact")
    if not imp:
        return {"error": "impact not yet computed — call business_calculate_impact first"}
    return {"most_exposed": imp.get("most_exposed"),
            "all_materials": [{"label": m["label"], "delta_inr": m["delta_inr"],
                               "deferred_delta_inr": m["deferred_delta_inr"],
                               "combined_shock_pct": m["combined_shock_pct"],
                               "contract_locked": m["contract_locked"]}
                               for m in imp.get("materials", [])]}


def business_recalculate(overrides_json: str) -> dict:
    """Re-run the exact financial model with modified assumptions to answer a
    'what if' question — e.g. changing logistics mode, pass-through capacity,
    or a material's contract type. overrides_json is a JSON object merged
    into the business profile before recalculating. Example:
    '{"logistics": {"mode": "rail"}}' or '{"product": {"pass_through_capacity_pct": 70}}'.
    Returns the new projected numbers so you can compare against the original."""
    print(f"  [Business Advisor] Tool call: business_recalculate({overrides_json})")
    active = _ctx.get("_active", {})
    profile = json.loads(json.dumps(active.get("profile", {})))  # deep copy
    try:
        overrides = json.loads(overrides_json)
    except Exception:
        return {"error": "overrides_json must be valid JSON"}
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(profile.get(k), dict):
            profile[k].update(v)
        else:
            profile[k] = v
    new_impact = business_impact.calculate_impact(profile, active["cascade_sectors"], active["fx"])
    return {"projected": new_impact["projected"], "delta": new_impact["delta"], "logistics": new_impact["logistics"]}


def business_market_context() -> dict:
    """Get the live geopolitical risk headline and currency context driving
    this projection: active scenario name, and INR/USD trend."""
    active = _ctx.get("_active", {})
    return {"cascade_scenario": active.get("scenario_name"), "fx": active.get("fx")}


def business_get_commodity_reference(material_key: str) -> dict:
    """Look up the live global reference price and 1-month trend for a raw
    material (steel, aluminium, synthetic_rubber, edible_oil, fertiliser_urea,
    naphtha, pet_bottles) from real FRED/World-Bank commodity data — use this
    when you want an independent check on a specific material beyond the
    cascade model's own passthrough estimate."""
    print(f"  [Business Advisor] Tool call: business_get_commodity_reference({material_key})")
    ref = market_data.get_commodity_reference(material_key)
    return ref or {"note": f"no live reference series mapped for '{material_key}'"}


def business_get_india_macro_context() -> dict:
    """Get India's real annual CPI inflation and GDP growth (World Bank, live)
    — an independent real-data check against the cascade model's fixed
    macro passthrough assumptions. Use when reasoning about broader economic
    context beyond this one business."""
    print("  [Business Advisor] Tool call: business_get_india_macro_context")
    return market_data.get_india_macro_context()


_TOOLS = [business_calculate_impact, business_get_impact_summary, business_get_material_breakdown,
          business_recalculate, business_market_context, business_get_commodity_reference,
          business_get_india_macro_context]

RESPONSE_SCHEMA = """
Reply with ONLY valid JSON (no markdown), in exactly this shape:

{
  "narrative": "<3-5 sentences: what this shock means for THIS business specifically, in plain owner language>",
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "recommendations": [
    {"action": "<short imperative, e.g. 'Lock a 6-month forward contract on edible oil'>", "rationale": "<why, grounded in the numbers>", "priority": "HIGH|MEDIUM|LOW"}
  ]
}

Give 2-4 recommendations, ranked by priority. Ground every claim in the tool
results — never invent a number that didn't come from a tool call.
"""

FIRST_ANALYSIS_INSTRUCTION = """You are APEX's Business Impact Advisor — an autonomous agent helping an
Indian business owner understand how geopolitical energy-supply risk hits THEIR specific costs and margin.

You have NOT yet computed anything for this business. A profile has been loaded for you (materials, logistics,
product economics) but nothing has been calculated. Decide for yourself what to do:
1. Call business_calculate_impact to run the real financial model — do this first.
2. Call business_get_material_breakdown to see which materials drive the exposure.
3. If a material's exposure looks significant, consider business_get_commodity_reference on it for an
   independent live-price check.
4. Consider business_get_india_macro_context if broader economic framing would help your recommendation.
5. Only then reason and answer.

India context: 88% crude import dependence, ~42% via Hormuz, ~18% via Red Sea, SPR ≈ 9.5 days, INR import
costs are FX-sensitive.

{schema}"""

FOLLOWUP_INSTRUCTION = """You are APEX's Business Impact Advisor, continuing a conversation with an
Indian business owner. The financial model has already been run for this session — use
business_get_impact_summary / business_get_material_breakdown to recall it, business_recalculate for "what
if" questions (always re-run the real model, never guess a number), and business_get_commodity_reference or
business_get_india_macro_context if the owner's question calls for independent live-data context.

{schema}"""


async def _run_adk(message: str, session_id: str, first_turn: bool) -> dict:
    instruction = (FIRST_ANALYSIS_INSTRUCTION if first_turn else FOLLOWUP_INSTRUCTION).format(schema=RESPONSE_SCHEMA)

    agent = Agent(name="apex_business_advisor", model=MODEL, instruction=instruction,
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
        raise RuntimeError("advisor produced no response")
    raw = final.strip().replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    result["agent_mode"] = "agentic (ADK Business Advisor — agent-invoked calculation + tools + multi-turn)"
    return result


# ─────────────────────────────────────────────────────────────
# DETERMINISTIC FALLBACK — used only if the ADK path itself fails
# ─────────────────────────────────────────────────────────────
def _deterministic(impact: dict) -> dict:
    d = impact["delta"]
    pct = d["total_cost_pct"]
    risk = "CRITICAL" if pct >= 8 else "HIGH" if pct >= 4 else "MEDIUM" if pct >= 1.5 else "LOW"
    exposed = impact.get("most_exposed", [])
    recs = []
    if exposed:
        top = exposed[0]
        recs.append({"action": f"Prioritise hedging {top['label']}", "priority": "HIGH",
                    "rationale": f"Largest exposure at ₹{top['delta_inr']:,.0f}/month ({top['combined_shock_pct']:+.1f}% combined shock)."})
    if impact["fx"]["drift_pct_30d"] and abs(impact["fx"]["drift_pct_30d"]) > 1:
        recs.append({"action": "Consider forward FX cover on USD-denominated inputs", "priority": "MEDIUM",
                    "rationale": f"INR has moved {impact['fx']['drift_pct_30d']:+.2f}% vs USD in 30 days, compounding commodity shocks."})
    if d["deferred_by_fixed_contracts_inr"] > 0:
        recs.append({"action": "Plan for cost normalization when fixed contracts expire", "priority": "MEDIUM",
                    "rationale": f"₹{d['deferred_by_fixed_contracts_inr']:,.0f}/month is currently deferred by fixed-price contracts, not eliminated."})
    if impact["pass_through_capacity_pct"] < 50 and pct > 2:
        recs.append({"action": "Reassess pricing power or find margin elsewhere", "priority": "HIGH",
                    "rationale": f"Only {impact['pass_through_capacity_pct']:.0f}% of the cost increase can be passed to customers — the rest compresses margin directly."})
    if not recs:
        recs.append({"action": "No urgent action needed", "priority": "LOW", "rationale": "Projected cost impact is within normal operating variance."})

    narrative = (f"Projected monthly cost impact is ₹{d['total_cost_inr']:,.0f} ({pct:+.1f}%), "
                f"moving margin by {d['margin_pct_points']:+.1f} points. "
                f"{'Largest driver: ' + exposed[0]['label'] + '.' if exposed else ''}")

    return {"narrative": narrative, "risk_level": risk, "recommendations": recs,
            "agent_mode": "deterministic advisor (LLM unavailable — model-grounded heuristic)"}


def run_advisor(message: str, session_id: str, first_turn: bool = False) -> dict:
    if session_id not in _ctx:
        return {"error": "No business profile loaded for this session — call /api/business/analyze first."}
    active = _ctx[session_id]
    active["_session_id"] = session_id
    _ctx["_active"] = active
    try:
        result = asyncio.run(_run_adk(message, session_id, first_turn))
        result["impact"] = _ctx[session_id].get("impact") or business_impact.calculate_impact(
            active["profile"], active["cascade_sectors"], active["fx"])
        return result
    except Exception as e:
        print(f"  Business Advisor ADK failed ({str(e)[:80]}) — deterministic fallback")
        impact = _ctx[session_id].get("impact") or business_impact.calculate_impact(
            active["profile"], active["cascade_sectors"], active["fx"])
        _ctx[session_id]["impact"] = impact
        result = _deterministic(impact)
        result["impact"] = impact
        return result


def set_session_context(session_id: str, profile: dict, cascade_sectors: dict, fx: dict, scenario_name: str = None):
    _ctx[session_id] = {"profile": profile, "cascade_sectors": cascade_sectors, "fx": fx,
                        "impact": None, "scenario_name": scenario_name, "_session_id": session_id}


def analyze_business(session_id: str, profile: dict, cascade_sectors: dict, fx: dict, scenario_name: str = None) -> dict:
    """First-pass analysis: loads the session's live market context (NOT the
    financial impact — that's left for the agent to decide to compute), then
    asks the advisor to analyze. The agent itself calls business_calculate_impact."""
    set_session_context(session_id, profile, cascade_sectors, fx, scenario_name)
    profile_summary = json.dumps({k: v for k, v in profile.items() if k != "business_name"})
    message = f"Here is the business profile that has been loaded for you:\n{profile_summary}\n\nAnalyze this business's exposure to the current risk assessment and give recommendations."
    return run_advisor(message, session_id, first_turn=True)


if __name__ == "__main__":
    from agents.business_impact import calculate_impact
    sample_cascade = {"fuels": {"diesel": {"change_pct": 6.5}}, "industry": {"plastics_pvc": {"change_pct": 3.5}},
                      "transport": {"road_freight": {"change_pct": 5.8}}, "consumer_cpg": {"edible_oil": {"change_pct": 1.9}}}
    profile = {"materials": [{"key": "plastics_pvc", "monthly_volume": 5000, "unit_cost": 120, "currency": "INR", "contract_type": "spot"}],
              "logistics": {"mode": "road", "monthly_cost": 180000},
              "product": {"current_price": 20, "monthly_units_sold": 500000, "pass_through_capacity_pct": 40},
              "inventory": {"buffer_days": 21}}
    fx = market_data.get_fx_rate()
    imp = calculate_impact(profile, sample_cascade, fx)
    r = _deterministic(imp)
    print(json.dumps(r, indent=2))
