import os
import sys
import time
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Fix import path when running from agents/ folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# ─────────────────────────────────────────
# APEX STATE SCHEMA
# Shared across all 6 LangGraph nodes
# ─────────────────────────────────────────
class ApexState(TypedDict):
    risk_data: Optional[dict]
    alert_level: Optional[str]
    brent_price: Optional[float]
    hormuz_score: Optional[int]
    redsea_score: Optional[int]
    opec_score: Optional[int]
    overall_disruption_probability: Optional[int]
    context_corridor: Optional[str]
    rag_context: Optional[list]
    kg_context: Optional[dict]
    reflection: Optional[dict]
    active_scenario: Optional[str]
    scenario_data: Optional[dict]
    auto_triggered: Optional[bool]
    twin_state: Optional[dict]
    stressed_refineries: Optional[list]
    spr_status: Optional[dict]
    cascade_data: Optional[dict]
    sector_impacts: Optional[dict]
    pipeline_run_time: Optional[float]
    pipeline_triggered_by: Optional[str]
    errors: Optional[list]

# ─────────────────────────────────────────
# NODE 1 — Risk Intelligence Agent
# ─────────────────────────────────────────
def risk_agent_node(state: ApexState) -> ApexState:
    print("[LangGraph] Node 1: Risk Intelligence Agent...")
    try:
        from agents.risk_agent import run_risk_agent
        risk_data = run_risk_agent()
        return {
            **state,
            "risk_data": risk_data,
            "alert_level": risk_data.get("alert_level","LOW"),
            "brent_price": risk_data.get("commodity_signals",{}).get("brent_price",69.56),
            "hormuz_score": risk_data.get("corridors",{}).get("hormuz",{}).get("risk_score",0),
            "redsea_score": risk_data.get("corridors",{}).get("red_sea",{}).get("risk_score",0),
            "opec_score": risk_data.get("corridors",{}).get("opec_policy",{}).get("risk_score",0),
            "overall_disruption_probability": risk_data.get("overall_disruption_probability",0),
            "errors": state.get("errors") or [],
        }
    except Exception as e:
        print(f"[LangGraph] Node 1 error: {e}")
        errors = state.get("errors") or []
        errors.append(f"risk_agent: {str(e)[:100]}")
        return {**state, "errors": errors}

# ─────────────────────────────────────────
# NODE 2 — Context Enrichment (RAG + Knowledge Graph)
# Runs after the Risk Agent so it knows which corridor is actually the
# highest-risk one to enrich against, rather than guessing beforehand.
# Grounds the pipeline in real historical precedent (semantic search over
# 16 past disruption events + policy sources) and real supplier-corridor-
# refinery-sector relationships (the knowledge graph), not just the live
# snapshot from Node 1.
# ─────────────────────────────────────────
def context_enrichment_node(state: ApexState) -> ApexState:
    print("[LangGraph] Node 2: Context Enrichment (RAG + Knowledge Graph)...")
    try:
        from agents import rag_engine, knowledge_graph
        scores = {
            "hormuz":      state.get("hormuz_score", 0) or 0,
            "red_sea":     state.get("redsea_score", 0) or 0,
            "opec_policy": state.get("opec_score", 0) or 0,
        }
        top_corridor = max(scores.items(), key=lambda x: x[1])[0]

        rag_query = f"{top_corridor.replace('_', ' ')} disruption precedent and historical price impact"
        rag_results = rag_engine.search(rag_query, k=3)
        print(f"[LangGraph] RAG retrieved {len(rag_results)} docs for '{rag_query}'")

        kg_context = None
        if top_corridor in ("hormuz", "red_sea"):
            suppliers = knowledge_graph.get_suppliers_via(top_corridor)
            refineries = knowledge_graph.get_refineries_exposed_to(top_corridor)
            kg_context = {"corridor": top_corridor, "suppliers_via_corridor": suppliers,
                          "refineries_exposed": [r["name"] for r in refineries]}
            print(f"[LangGraph] KG: {len(suppliers)} suppliers, {len(refineries)} refineries exposed via {top_corridor}")

        return {**state, "context_corridor": top_corridor, "rag_context": rag_results, "kg_context": kg_context}
    except Exception as e:
        print(f"[LangGraph] Node 2 error: {e}")
        errors = state.get("errors") or []
        errors.append(f"context_enrichment: {str(e)[:100]}")
        return {**state, "errors": errors, "rag_context": None, "kg_context": None}

# ─────────────────────────────────────────
# NODE 3 — Reflection / Quality-Control
# The QC agent (risk_agent.py's apex_qc_agent) already runs inside
# run_risk_agent() and its verdict is carried in risk_data["reflection"].
# This node does not re-invoke the LLM (that would double-spend quota for
# no new information) — it surfaces that already-computed verdict as its
# own explicit, visible pipeline stage, so the graph topology honestly
# reflects that reflection is a distinct step or the orchestrator, not a
# hidden implementation detail of Node 1.
# ─────────────────────────────────────────
def reflection_node(state: ApexState) -> ApexState:
    print("[LangGraph] Node 3: Reflection / QC review...")
    try:
        risk_data = state.get("risk_data") or {}
        reflection = risk_data.get("reflection") or {"verdict": "UNKNOWN", "notes": "no reflection recorded on risk_data"}
        print(f"[LangGraph] QC verdict: {reflection.get('verdict')} — {reflection.get('notes', '')[:80]}")
        return {**state, "reflection": reflection}
    except Exception as e:
        print(f"[LangGraph] Node 3 error: {e}")
        errors = state.get("errors") or []
        errors.append(f"reflection: {str(e)[:100]}")
        return {**state, "errors": errors}

# ─────────────────────────────────────────
# CONDITIONAL ROUTER
# ─────────────────────────────────────────
def route_after_risk(state: ApexState) -> str:
    hormuz = state.get("hormuz_score", 0) or 0
    redsea = state.get("redsea_score", 0) or 0
    opec   = state.get("opec_score", 0) or 0
    alert  = state.get("alert_level", "LOW")
    print(f"[LangGraph] Router: Hormuz={hormuz} RedSea={redsea} OPEC={opec} Alert={alert}")
    if hormuz >= 60 or redsea >= 60 or opec >= 60 or alert in ["HIGH","CRITICAL"]:
        print("[LangGraph] Router -> HIGH RISK -> triggering scenario agent")
        return "high_risk"
    print("[LangGraph] Router -> LOW RISK -> skipping scenario")
    return "low_risk"

# ─────────────────────────────────────────
# NODE 4 — Scenario Modelling Agent
# ─────────────────────────────────────────
def scenario_agent_node(state: ApexState) -> ApexState:
    print("[LangGraph] Node 4: Scenario Modelling Agent...")
    try:
        from agents.scenario_agent import run_scenario_agent
        risk_data = state.get("risk_data", {})
        scores = {
            "hormuz":      state.get("hormuz_score", 0) or 0,
            "red_sea":     state.get("redsea_score", 0) or 0,
            "opec_policy": state.get("opec_score", 0) or 0,
        }
        active_scenario = max(scores.items(), key=lambda x: x[1])[0]
        scenario_data = run_scenario_agent(risk_data, active_scenario)
        return {
            **state,
            "active_scenario": active_scenario,
            "scenario_data": scenario_data,
            "auto_triggered": True,
        }
    except Exception as e:
        print(f"[LangGraph] Node 4 error: {e}")
        errors = state.get("errors") or []
        errors.append(f"scenario_agent: {str(e)[:100]}")
        return {**state, "errors": errors}

# ─────────────────────────────────────────
# NODE 5 — Digital Twin Agent
# ─────────────────────────────────────────
def twin_agent_node(state: ApexState) -> ApexState:
    print("[LangGraph] Node 5: Digital Twin Agent...")
    try:
        hormuz   = state.get("hormuz_score", 0) or 0
        redsea   = state.get("redsea_score", 0) or 0
        brent    = state.get("brent_price", 69.56) or 69.56
        scenario = state.get("active_scenario")
        d30      = (state.get("scenario_data") or {}).get("day_30", {})

        REFINERIES = [
            {"id":"jamnagar",     "name":"Jamnagar",     "company":"Reliance",    "crude_source":"Gulf",           "mmtpa":62},
            {"id":"jamnagar_sez", "name":"Jamnagar SEZ", "company":"Reliance",    "crude_source":"Gulf",           "mmtpa":35},
            {"id":"kochi",        "name":"Kochi",         "company":"BPCL",        "crude_source":"Gulf",           "mmtpa":15.5},
            {"id":"panipat",      "name":"Panipat",       "company":"IOCL",        "crude_source":"Mixed",          "mmtpa":15},
            {"id":"paradip",      "name":"Paradip",       "company":"IOCL",        "crude_source":"Mixed",          "mmtpa":15},
            {"id":"mathura",      "name":"Mathura",       "company":"IOCL",        "crude_source":"Mixed",          "mmtpa":8},
            {"id":"mumbai_bpcl",  "name":"Mumbai BPCL",   "company":"BPCL",        "crude_source":"Mixed",          "mmtpa":12},
            {"id":"bina",         "name":"Bina",          "company":"BPCL",        "crude_source":"Mixed",          "mmtpa":7.8},
            {"id":"vizag",        "name":"Vizag",         "company":"HPCL",        "crude_source":"Import",         "mmtpa":8.3},
            {"id":"mumbai_hpcl",  "name":"Mumbai HPCL",   "company":"HPCL",        "crude_source":"Domestic+Import","mmtpa":6.5},
            {"id":"bathinda",     "name":"Bathinda",      "company":"HPCL-Mittal", "crude_source":"Mixed",          "mmtpa":9},
            {"id":"chennai",      "name":"Chennai",       "company":"CPCL",        "crude_source":"Import",         "mmtpa":10.5},
            {"id":"haldia",       "name":"Haldia",        "company":"IOCL",        "crude_source":"Import",         "mmtpa":7.5},
            {"id":"barauni",      "name":"Barauni",       "company":"IOCL",        "crude_source":"Domestic+Import","mmtpa":6},
            {"id":"numaligarh",   "name":"Numaligarh",    "company":"NRL",         "crude_source":"Domestic",       "mmtpa":3},
            {"id":"digboi",       "name":"Digboi",        "company":"IOCL",        "crude_source":"Domestic",       "mmtpa":0.65},
            {"id":"guwahati",     "name":"Guwahati",      "company":"IOCL",        "crude_source":"Domestic",       "mmtpa":1},
            {"id":"tatipaka",     "name":"Tatipaka",      "company":"ONGC",        "crude_source":"Domestic",       "mmtpa":0.083},
        ]

        stressed = []
        for r in REFINERIES:
            stress = 0
            if "Gulf" in r["crude_source"]:
                stress += hormuz * 0.4
            if scenario == "hormuz" and "Gulf" in r["crude_source"]:
                stress += abs(d30.get("refinery_run_rate_change_pct", 0))
            if scenario == "red_sea" and r["name"] in ["Paradip","Vizag","Chennai","Haldia"]:
                stress += 20
            if scenario == "opec_policy":
                stress += abs(d30.get("refinery_run_rate_change_pct", 0)) * 0.5
            stress = min(round(stress), 100)
            if stress >= 20:
                stressed.append({
                    "name":         r["name"],
                    "company":      r["company"],
                    "stress":       stress,
                    "mmtpa":        r["mmtpa"],
                    "crude_source": r["crude_source"],
                })

        stressed.sort(key=lambda x: x["stress"], reverse=True)

        spr_draw = d30.get("spr_drawdown_days", 0) if d30 else 0
        spr_status = {
            "total_days":    9.5,
            "drawn":         round(spr_draw, 2),
            "remaining":     round(max(0, 9.5 - spr_draw), 2),
            "pct_remaining": round(max(0, (9.5 - spr_draw) / 9.5 * 100), 1),
            "locations": [
                {"name":"Visakhapatnam", "capacity_mmt":1.33, "days":2.4},
                {"name":"Mangalore",     "capacity_mmt":1.5,  "days":2.7},
                {"name":"Padur",         "capacity_mmt":2.5,  "days":4.4},
            ]
        }

        twin_state = {
            "alert_level":             state.get("alert_level","LOW"),
            "brent_price":             brent,
            "hormuz_risk":             hormuz,
            "redsea_risk":             redsea,
            "active_scenario":         scenario,
            "stressed_refineries":     stressed,
            "spr_status":              spr_status,
            "spr_drawdown":            spr_draw,
            "scenario_day30":          d30,
            "network_health":          "CRITICAL" if len(stressed)>=5 else "ELEVATED" if len(stressed)>=2 else "NORMAL",
            "total_stressed_capacity": round(sum(r["mmtpa"] for r in stressed), 1),
        }

        return {
            **state,
            "twin_state":          twin_state,
            "stressed_refineries": stressed,
            "spr_status":          spr_status,
        }
    except Exception as e:
        print(f"[LangGraph] Node 5 error: {e}")
        errors = state.get("errors") or []
        errors.append(f"twin_agent: {str(e)[:100]}")
        return {**state, "errors": errors}

# ─────────────────────────────────────────
# NODE 6 — Economic Cascade Agent
# 7 sectors, 45+ commodities
# ─────────────────────────────────────────
def cascade_agent_node(state: ApexState) -> ApexState:
    print("[LangGraph] Node 6: Economic Cascade Agent...")
    try:
        brent     = state.get("brent_price", 69.56) or 69.56
        d30       = (state.get("scenario_data") or {}).get("day_30", {})
        brent_chg = d30.get("crude_price_change_pct", 0) if d30 else 0

        ECON = {
            "petrol_passthrough":    0.65,
            "diesel_passthrough":    0.58,
            "lpg_passthrough":       0.45,
            "aviation_passthrough":  0.85,
            "fertiliser_sensitivity":0.40,
            "plastic_sensitivity":   0.35,
            "transport_sensitivity": 0.58,
            "food_sensitivity":      0.18,
            "power_sensitivity":     0.12,
            "steel_sensitivity":     0.08,
            "pharma_sensitivity":    0.15,
            "brent_to_cpi":          0.015,
            "brent_to_gdp":         -0.012,
        }

        def impact(coeff):
            val = round(brent_chg * coeff, 2)
            return {"change_pct": val, "direction": "up" if val > 0 else "down"}

        sectors = {
            "fuels": {
                "petrol":            impact(ECON["petrol_passthrough"]),
                "diesel":            impact(ECON["diesel_passthrough"]),
                "lpg":               impact(ECON["lpg_passthrough"]),
                "aviation_fuel_atf": impact(ECON["aviation_passthrough"]),
                "kerosene":          impact(0.40),
                "naphtha":           impact(0.70),
            },
            "industry": {
                "plastics_pvc":      impact(ECON["plastic_sensitivity"]),
                "fertiliser_urea":   impact(ECON["fertiliser_sensitivity"]),
                "steel":             impact(ECON["steel_sensitivity"]),
                "cement":            impact(0.10),
                "chemicals":         impact(0.30),
                "synthetic_rubber":  impact(0.28),
                "paints_coatings":   impact(0.22),
                "aluminium":         impact(0.15),
            },
            "transport": {
                "road_freight":         impact(ECON["transport_sensitivity"]),
                "aviation":             impact(ECON["aviation_passthrough"]),
                "railways":             impact(0.20),
                "coastal_shipping":     impact(0.35),
                "last_mile_ecommerce":  impact(0.45),
                "cold_chain_logistics": impact(0.50),
            },
            "agriculture": {
                "fertiliser_cost":   impact(ECON["fertiliser_sensitivity"]),
                "irrigation_diesel": impact(0.30),
                "food_transport":    impact(ECON["food_sensitivity"]),
                "pesticides":        impact(0.25),
                "tractor_fuel":      impact(0.50),
                "crop_cold_storage": impact(0.15),
            },
            "consumer_cpg": {
                "fmcg_packaging":    impact(0.20),
                "edible_oil":        impact(ECON["food_sensitivity"]),
                "dairy_products":    impact(0.15),
                "synthetic_textiles":impact(0.18),
                "pharma":            impact(ECON["pharma_sensitivity"]),
                "personal_care":     impact(0.18),
                "pet_bottles":       impact(0.20),
            },
            "services": {
                "airlines":          impact(ECON["aviation_passthrough"]),
                "hotels":            impact(0.12),
                "restaurants_lpg":   impact(0.25),
                "retail_logistics":  impact(0.30),
                "it_datacentres":    impact(0.08),
                "healthcare_power":  impact(0.10),
            },
            "macro": {
                "cpi_inflation":     impact(ECON["brent_to_cpi"]),
                "gdp_impact":        impact(ECON["brent_to_gdp"]),
                "trade_deficit":     impact(0.80),
                "inr_usd_pressure":  impact(0.30),
                "fiscal_deficit":    impact(0.25),
                "rbi_rate_pressure": impact(0.20),
            }
        }

        return {
            **state,
            "cascade_data": {
                "brent_change_pct":    brent_chg,
                "brent_baseline":      brent,
                "brent_new":           round(brent * (1 + brent_chg/100), 2),
                "sectors":             sectors,
                "total_commodities":   sum(len(v) for v in sectors.values()),
                "econometric_sources": [
                    "IMF WP/22/58",
                    "RBI MPR 2023",
                    "World Bank Commodity Outlook",
                    "PPAC India Statistics"
                ],
            },
            "sector_impacts": sectors,
        }
    except Exception as e:
        print(f"[LangGraph] Node 6 error: {e}")
        errors = state.get("errors") or []
        errors.append(f"cascade_agent: {str(e)[:100]}")
        return {**state, "errors": errors}

# ─────────────────────────────────────────
# BUILD LANGGRAPH STATE GRAPH
# ─────────────────────────────────────────
def build_pipeline():
    graph = StateGraph(ApexState)

    graph.add_node("risk_agent",         risk_agent_node)
    graph.add_node("context_enrichment", context_enrichment_node)
    graph.add_node("reflection",         reflection_node)
    graph.add_node("scenario_agent",     scenario_agent_node)
    graph.add_node("twin_agent",         twin_agent_node)
    graph.add_node("cascade_agent",      cascade_agent_node)

    graph.set_entry_point("risk_agent")

    graph.add_edge("risk_agent",         "context_enrichment")
    graph.add_edge("context_enrichment", "reflection")

    graph.add_conditional_edges(
        "reflection",
        route_after_risk,
        {
            "high_risk": "scenario_agent",
            "low_risk":  "twin_agent",
        }
    )

    graph.add_edge("scenario_agent", "twin_agent")
    graph.add_edge("twin_agent",     "cascade_agent")
    graph.add_edge("cascade_agent",  END)

    return graph.compile()

PIPELINE = build_pipeline()

# ─────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────
def run_pipeline(trigger: str = "scheduled") -> dict:
    print(f"\n{'='*55}")
    print(f"APEX LangGraph Pipeline — trigger: {trigger}")
    print(f"{'='*55}")

    start = time.time()

    initial_state: ApexState = {
        "risk_data":                     None,
        "alert_level":                   None,
        "brent_price":                   None,
        "hormuz_score":                  None,
        "redsea_score":                  None,
        "opec_score":                    None,
        "overall_disruption_probability":None,
        "active_scenario":               None,
        "scenario_data":                 None,
        "auto_triggered":                False,
        "twin_state":                    None,
        "stressed_refineries":           None,
        "spr_status":                    None,
        "cascade_data":                  None,
        "sector_impacts":                None,
        "context_corridor":              None,
        "rag_context":                   None,
        "kg_context":                    None,
        "reflection":                    None,
        "pipeline_run_time":             None,
        "pipeline_triggered_by":         trigger,
        "errors":                        [],
    }

    result = PIPELINE.invoke(initial_state)
    result["pipeline_run_time"] = round(time.time() - start, 2)

    print(f"\n[LangGraph] Pipeline complete in {result['pipeline_run_time']}s")
    print(f"[LangGraph] Alert: {result.get('alert_level')}")
    print(f"[LangGraph] Context corridor: {result.get('context_corridor')} "
          f"(RAG docs: {len(result.get('rag_context') or [])}, KG: {'yes' if result.get('kg_context') else 'no'})")
    print(f"[LangGraph] Reflection verdict: {(result.get('reflection') or {}).get('verdict')}")
    print(f"[LangGraph] Active scenario: {result.get('active_scenario')}")
    print(f"[LangGraph] Auto-triggered: {result.get('auto_triggered')}")
    print(f"[LangGraph] Stressed refineries: {len(result.get('stressed_refineries') or [])}")
    print(f"[LangGraph] Commodities modelled: {(result.get('cascade_data') or {}).get('total_commodities')}")
    print(f"[LangGraph] Errors: {result.get('errors')}")

    return result

if __name__ == "__main__":
    import json
    result = run_pipeline("test")
    summary = {
        "alert_level":          result.get("alert_level"),
        "brent_price":          result.get("brent_price"),
        "hormuz_score":         result.get("hormuz_score"),
        "redsea_score":         result.get("redsea_score"),
        "opec_score":           result.get("opec_score"),
        "active_scenario":      result.get("active_scenario"),
        "auto_triggered":       result.get("auto_triggered"),
        "stressed_refineries":  [r["name"] for r in (result.get("stressed_refineries") or [])],
        "network_health":       (result.get("twin_state") or {}).get("network_health"),
        "spr_remaining":        (result.get("spr_status") or {}).get("remaining"),
        "total_commodities":    (result.get("cascade_data") or {}).get("total_commodities"),
        "pipeline_run_time":    result.get("pipeline_run_time"),
        "errors":               result.get("errors"),
    }
    print("\nPIPELINE SUMMARY:")
    print(json.dumps(summary, indent=2))