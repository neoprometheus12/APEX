import os
import json
import asyncio
import datetime
import numpy as np
import joblib
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

APP_NAME = "apex"
USER_ID = "apex_system"
_scenario_session_service = InMemorySessionService()

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

model    = joblib.load(os.path.join(DATA_PATH, "scenario_model.pkl"))
scaler   = joblib.load(os.path.join(DATA_PATH, "scenario_scaler.pkl"))
FEATURES = joblib.load(os.path.join(DATA_PATH, "scenario_features.pkl"))
TARGETS  = joblib.load(os.path.join(DATA_PATH, "scenario_targets.pkl"))
ECON     = joblib.load(os.path.join(DATA_PATH, "econometric.pkl"))

SCENARIO_CONFIG = {
    "hormuz": {
        "name": "Strait of Hormuz Partial Closure",
        "description": "50% flow disruption for 30 days",
        "type_hormuz": 1, "type_redsea": 0, "type_opec": 0,
        "severity": 0.50, "duration_days": 30,
        "india_exposure": 0.42,
        "most_stressed_refineries": ["Jamnagar (Reliance)", "Kochi (BPCL)", "Mumbai (BPCL)"],
    },
    "red_sea": {
        "name": "Red Sea Full Suspension",
        "description": "All traffic rerouted via Cape of Good Hope (+12-14 days)",
        "type_hormuz": 0, "type_redsea": 1, "type_opec": 0,
        "severity": 0.80, "duration_days": 90,
        "india_exposure": 0.18,
        "most_stressed_refineries": ["Paradip (IOCL)", "Vizag (HPCL)", "Chennai (CPCL)"],
    },
    "opec_policy": {
        "name": "OPEC+ Emergency Cut",
        "description": "2 million barrels/day coordinated cut, immediate effect",
        "type_hormuz": 0, "type_redsea": 0, "type_opec": 1,
        "severity": 0.20, "duration_days": 180,
        "india_exposure": 0.65,
        "most_stressed_refineries": ["All refineries (price impact)", "Jamnagar (volume)", "Paradip (grade)"],
    },
}

HISTORICAL_VALIDATION = [
    {"event": "2019 Abqaiq Attack", "type": "hormuz",
     "predicted_d7": 11.5, "actual_d7": 9.7, "accuracy": 81.4},
    {"event": "2021 Suez Blockage", "type": "red_sea",
     "predicted_d7": 6.6, "actual_d7": 4.0, "accuracy": 35.0},
    {"event": "2022 Russia Sanctions", "type": "opec",
     "predicted_d7": 7.7, "actual_d7": 15.0, "accuracy": 51.3},
]

def predict_cascade(scenario_key: str, baseline_brent: float, risk_score: int = 50):
    cfg = SCENARIO_CONFIG[scenario_key]
    adjusted_severity = cfg["severity"] * (0.7 + (risk_score / 100) * 0.6)
    adjusted_severity = min(adjusted_severity, 1.0)

    X = np.array([[
        adjusted_severity,
        cfg["duration_days"],
        cfg["type_hormuz"],
        cfg["type_redsea"],
        cfg["type_opec"],
        baseline_brent,
        baseline_brent * 0.98,
        baseline_brent * 0.05,
        2.0,
    ]])

    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    pred_dict = dict(zip(TARGETS, pred))

    brent_d1     = round(float(pred_dict["brent_shock_d1"]), 2)
    brent_d7     = round(float(pred_dict["brent_shock_d7"]), 2)
    brent_d30    = round(float(pred_dict["brent_shock_d30"]), 2)
    refinery_chg = round(float(pred_dict["refinery_rate_chg"]), 2)
    gdp_d30      = round(float(pred_dict["gdp_impact"]), 3)
    power_stress = round(float(pred_dict["power_stress"]))
    spr_draw     = round(float(pred_dict["spr_draw"]), 2)

    petrol_d7    = round(brent_d7  * ECON["crude_to_petrol_passthrough"], 2)
    petrol_d30   = round(brent_d30 * ECON["crude_to_petrol_passthrough"], 2)
    diesel_d7    = round(brent_d7  * ECON["crude_to_diesel_passthrough"], 2)
    diesel_d30   = round(brent_d30 * ECON["crude_to_diesel_passthrough"], 2)
    cpi_impact   = round((brent_d30 / 10) * ECON["brent_10usd_to_india_cpi"], 3)
    gdp_econ     = round((brent_d30 / 10) * ECON["brent_10usd_to_india_gdp"], 3)
    gdp_combined = round((gdp_d30 + gdp_econ) / 2, 3)

    spr_exhaustion_day = round(ECON["india_spr_days"] / max(spr_draw / 30, 0.01)) if spr_draw > 0 else 999

    brent_price_d30 = baseline_brent * (1 + brent_d30 / 100)
    daily_cost_increase = (brent_price_d30 - baseline_brent) * ECON["india_daily_crude_mbd"] * 1_000_000
    import_bill_30d = round(daily_cost_increase * 30 / 1_000_000_000, 2)

    time_series = []
    for day in [0, 1, 3, 7, 14, 21, 30, 45, 60, 90]:
        if day == 0:
            shock = 0.0
        elif day <= 1:
            shock = brent_d1
        elif day <= 7:
            shock = brent_d1 + (brent_d7 - brent_d1) * (day - 1) / 6
        elif day <= 30:
            shock = brent_d7 + (brent_d30 - brent_d7) * (day - 7) / 23
        elif day <= 60:
            shock = brent_d30 + (brent_d30 * 0.5 - brent_d30) * (day - 30) / 30
        else:
            shock = brent_d30 * 0.3
        time_series.append({
            "day": day,
            "brent_price": round(baseline_brent * (1 + shock / 100), 2),
            "shock_pct": round(shock, 2)
        })

    return {
        "scenario": scenario_key,
        "scenario_name": cfg["name"],
        "scenario_description": cfg["description"],
        "baseline_brent": baseline_brent,
        "adjusted_severity": round(adjusted_severity, 2),
        "model_type": "ML (Gradient Boosting) + Econometric (IMF/RBI/World Bank)",
        "day_1": {
            "crude_price_change_pct": brent_d1,
            "brent_price_usd": round(baseline_brent * (1 + brent_d1/100), 2),
            "refinery_run_rate_change_pct": round(refinery_chg * 0.3, 2),
            "power_sector_stress_index": round(power_stress * 0.4),
            "spr_drawdown_days": round(spr_draw * 0.05, 2),
            "gdp_impact_pct": round(gdp_combined * 0.05, 4),
            "petrol_price_change_pct": round(petrol_d7 * 0.3, 2),
            "diesel_price_change_pct": round(diesel_d7 * 0.3, 2),
            "narrative": f"Markets react instantly to {cfg['name']}. Spot prices spike as traders price in supply uncertainty."
        },
        "day_7": {
            "crude_price_change_pct": brent_d7,
            "brent_price_usd": round(baseline_brent * (1 + brent_d7/100), 2),
            "refinery_run_rate_change_pct": round(refinery_chg * 0.6, 2),
            "power_sector_stress_index": round(power_stress * 0.7),
            "spr_drawdown_days": round(spr_draw * 0.25, 2),
            "gdp_impact_pct": round(gdp_combined * 0.2, 4),
            "petrol_price_change_pct": petrol_d7,
            "diesel_price_change_pct": diesel_d7,
            "narrative": f"Supply tightening becomes physical. Indian refineries begin adjusting run rates. SPR drawdown accelerates."
        },
        "day_30": {
            "crude_price_change_pct": brent_d30,
            "brent_price_usd": round(baseline_brent * (1 + brent_d30/100), 2),
            "refinery_run_rate_change_pct": refinery_chg,
            "power_sector_stress_index": power_stress,
            "spr_drawdown_days": spr_draw,
            "gdp_impact_pct": gdp_combined,
            "petrol_price_change_pct": petrol_d30,
            "diesel_price_change_pct": diesel_d30,
            "cpi_impact_pct": cpi_impact,
            "narrative": f"Full cascade realized. India's import bill rises ${import_bill_30d}B. Policy intervention likely required."
        },
        "most_stressed_refineries": cfg["most_stressed_refineries"],
        "spr_exhaustion_day": spr_exhaustion_day,
        "india_import_bill_30d_bn": import_bill_30d,
        "intervention_trigger": f"SPR cover below 3 days (projected Day {min(spr_exhaustion_day, 30)})",
        "spr_recommendation": f"Begin controlled drawdown at {round(spr_draw/3, 2)} days/week to bridge {cfg['duration_days']}-day disruption",
        "cascade_to_sectors": {
            "aviation": f"ATF costs +{round(brent_d30 * 0.85, 1)}% — IndiGo, Air India hedging costs spike",
            "transport": f"Diesel +{diesel_d30}% — freight rates rise, e-commerce last mile costs up",
            "power": f"Gas-linked generation stressed — {power_stress}/100 stress index",
            "fertiliser": f"Urea input costs +{round(brent_d30 * 0.4, 1)}% — kharif season risk"
        },
        "time_series": time_series,
        "historical_validation": [v for v in HISTORICAL_VALIDATION if v["type"] == scenario_key],
        "model_confidence": 81 if scenario_key == "hormuz" else 45 if scenario_key == "red_sea" else 51,
        "econometric_sources": ["IMF WP/22/58", "RBI MPR 2023", "World Bank Commodity Outlook", "PPAC India"],
    }

def _run_cascade_model(scenario_key: str, baseline_brent: float, risk_score: int = 50) -> dict:
    """Run APEX's trained ML + econometric cascade model for a given disruption scenario
    (hormuz, red_sea, or opec_policy), baseline Brent crude price, and current corridor risk
    score (0-100). Returns Day 1/7/30 crude price shock, refinery run-rate, GDP, and SPR
    drawdown projections."""
    return predict_cascade(scenario_key, baseline_brent, risk_score)

async def _run_scenario_adk_agent(scenario_key: str, baseline_brent: float, risk_score: int, result: dict) -> str:
    instruction = f"""You are APEX's Scenario Analysis Agent for India's energy supply chain.

Call run_cascade_model with scenario_key="{scenario_key}", baseline_brent={baseline_brent},
risk_score={risk_score} to get the modelled impact, then read its Day 30 fields and write
ONLY a 2-sentence executive summary for Indian energy policymakers. Be specific and direct,
citing real numbers from the tool result. Max 50 words. Plain text only — no JSON, no markdown."""

    agent = Agent(
        name="apex_scenario_agent",
        model="gemini-flash-lite-latest",
        instruction=instruction,
        tools=[FunctionTool(_run_cascade_model)],
    )
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=_scenario_session_service)

    session_id = f"scenario_{scenario_key}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    await _scenario_session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    final_text = None
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=genai_types.Content(role="user", parts=[genai_types.Part(text="Analyze this scenario now.")]),
    ):
        if event.is_final_response() and event.content:
            final_text = event.content.parts[0].text

    if not final_text:
        raise RuntimeError("ADK scenario agent produced no final response")
    return final_text.strip()

def run_scenario_agent(risk_data: dict, scenario: str = None):
    if not scenario:
        corridors = risk_data.get("corridors", {})
        scenario = max(corridors.items(), key=lambda x: x[1].get("risk_score", 0))[0]
        if scenario not in SCENARIO_CONFIG:
            scenario = "hormuz"

    brent_price = risk_data.get("commodity_signals", {}).get("brent_price", 78.89)
    risk_score = risk_data.get("corridors", {}).get(scenario, {}).get("risk_score", 50)

    print(f"  Running ML+Econometric model for: {SCENARIO_CONFIG[scenario]['name']}")
    print(f"  Baseline Brent: ${brent_price} | Risk score: {risk_score}/100")

    # The cascade math is computed directly, once, deterministically — we don't
    # want an LLM "deciding" refinery stress or GDP numbers. The ADK agent below
    # independently calls the same model as a tool and only owns the narrative.
    result = predict_cascade(scenario, brent_price, risk_score)

    try:
        result["executive_summary"] = asyncio.run(_run_scenario_adk_agent(scenario, brent_price, risk_score, result))
        result["agent_mode"] = "agentic (Google ADK — Agent + Runner + FunctionTool)"
    except Exception as e:
        print(f"  ADK scenario agent failed ({e}) — falling back to direct model call")
        try:
            prompt = f"""You are APEX's scenario analysis agent. A disruption scenario has been modelled.

Scenario: {result['scenario_name']}
Baseline Brent: ${brent_price}/barrel
Day 30 Brent impact: {result['day_30']['crude_price_change_pct']:+.1f}%
Refinery run rate: {result['day_30']['refinery_run_rate_change_pct']:+.1f}%
India GDP impact: {result['day_30']['gdp_impact_pct']:+.3f}%
SPR exhaustion day: {result['spr_exhaustion_day']}
Import bill increase: ${result['india_import_bill_30d_bn']}B

Write a 2-sentence executive summary for Indian energy policymakers. Be specific and direct. Max 50 words total."""

            response = client.models.generate_content(model="gemini-flash-lite-latest", contents=prompt)
            result["executive_summary"] = response.text.strip()
            result["agent_mode"] = "sequential fallback (direct model call, no ADK)"
        except Exception as e2:
            result["executive_summary"] = f"{result['scenario_name']} could increase India's crude import bill by ${result['india_import_bill_30d_bn']}B over 30 days, requiring immediate SPR activation and procurement diversification."
            result["agent_mode"] = "deterministic fallback (no LLM available)"

    return result

if __name__ == "__main__":
    sample_risk = {
        "corridors": {
            "hormuz": {"risk_score": 90, "trend": "rising"},
            "red_sea": {"risk_score": 20, "trend": "stable"},
            "opec_policy": {"risk_score": 45, "trend": "stable"}
        },
        "commodity_signals": {"brent_price": 69.56},
        "alert_level": "CRITICAL",
        "overall_disruption_probability": 85
    }
    print("="*55)
    print("APEX — Scenario Modelling Agent v2 (ML + Econometric)")
    print("="*55)
    result = run_scenario_agent(sample_risk, "hormuz")
    print(json.dumps(result, indent=2))