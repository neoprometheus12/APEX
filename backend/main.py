import os
import threading
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import time

app = FastAPI(title="APEX API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cache = {
    "risk": None,
    "risk_time": 0,
    "scenarios": {},
    "pipeline_result": None,
    "pipeline_time": 0,
}
CACHE_TTL = 300

# The Risk Agent now reuses one fixed, DB-persisted ADK session across
# calls (for genuine cross-restart memory). Two requests computing a
# fresh assessment at the same time would race to append events to that
# same session row and throw a storage-conflict error, so overlapping
# calls are serialized here rather than run concurrently.
_risk_lock = threading.Lock()

# Anchored to this file's location so routes work regardless of the
# process's current working directory (previously "../frontend/..." broke
# whenever uvicorn wasn't launched from inside backend/).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
DATA_DIR = os.path.join(BASE_DIR, "..", "Data")

@app.on_event("startup")
def _start_ais():
    # Begin streaming real AIS at boot so tanker positions accumulate (15-min window)
    try:
        from agents import ais_feed
        ais_feed.start_collector()
    except Exception as e:
        print(f"AIS collector not started: {e}")

def _background_refresh_loop():
    """Keeps the risk/pipeline cache continuously warm by proactively re-running
    the full LangGraph pipeline on a fixed interval — the same always-on pattern
    already used for the AIS feed. Previously this data was only ever fetched
    lazily, in-request, when a caller happened to land after the cache had gone
    stale; that meant the platform could sit on data up to CACHE_TTL seconds old
    with nothing refreshing it until someone asked. Runs forever in a daemon
    thread; a failed cycle is logged and the previous good cache is kept rather
    than the thread dying."""
    while True:
        try:
            with _risk_lock:
                from agents.pipeline import run_pipeline
                result = run_pipeline("background_refresh")
                cache["pipeline_result"] = result
                cache["pipeline_time"] = time.time()
                if result.get("risk_data"):
                    cache["risk"] = result["risk_data"]
                    cache["risk_time"] = time.time()
                if result.get("scenario_data") and result.get("active_scenario"):
                    cache["scenarios"][result["active_scenario"]] = {
                        "data": result["scenario_data"],
                        "time": time.time(),
                    }
            print(f"[BackgroundRefresh] cache refreshed — alert={result.get('alert_level')}")
        except Exception as e:
            print(f"[BackgroundRefresh] cycle failed, keeping previous cache: {str(e)[:150]}")
        time.sleep(CACHE_TTL)

@app.on_event("startup")
def _start_background_refresh():
    threading.Thread(target=_background_refresh_loop, daemon=True, name="pipeline-refresh").start()

def frontend_file(name: str, media_type: str = None):
    path = os.path.join(FRONTEND_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"{name} not found")
    return FileResponse(path, media_type=media_type)

# ── FRONTEND ROUTES ──
@app.get("/")
def serve_index():
    return frontend_file("index.html")

@app.get("/index.html")
def serve_index_html():
    return frontend_file("index.html")

@app.get("/scenario.html")
def serve_scenario():
    return frontend_file("scenario.html")

@app.get("/twin.html")
def serve_twin():
    return frontend_file("twin.html")

@app.get("/business.html")
def serve_business():
    return frontend_file("business.html")

@app.get("/cascade.html")
def serve_cascade():
    return frontend_file("cascade.html")

@app.get("/india_state.geojson")
def serve_geojson():
    path = os.path.join(DATA_DIR, "india_state.geojson")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="india_state.geojson not found")
    return FileResponse(path, media_type="application/json")

# ── API ROUTES ──
@app.get("/api/risk")
def get_risk():
    # The background refresh loop (_background_refresh_loop) keeps this cache
    # continuously warm on its own timer, so a request never triggers a fetch
    # itself anymore — it just serves whatever the background cycle last
    # computed. The only exception is a genuine cold start, in the brief
    # window before that loop's first cycle has completed.
    if cache["risk"] is not None:
        age = int(time.time() - cache["risk_time"])
        result = dict(cache["risk"])
        result["cache_age_seconds"] = age
        result["next_refresh_seconds"] = max(0, CACHE_TTL - age)
        return result
    with _risk_lock:
        if cache["risk"] is not None:
            age = int(time.time() - cache["risk_time"])
            result = dict(cache["risk"])
            result["cache_age_seconds"] = age
            result["next_refresh_seconds"] = max(0, CACHE_TTL - age)
            return result
        try:
            from agents.risk_agent import run_risk_agent
            data = run_risk_agent()
            cache["risk"] = data
            cache["risk_time"] = time.time()
            result = dict(data)
            result["cache_age_seconds"] = 0
            result["next_refresh_seconds"] = CACHE_TTL
            return result
        except Exception as e:
            return JSONResponse(status_code=503, content={"error": str(e)[:200]})

@app.get("/api/scenario/{scenario}")
def get_scenario(scenario: str):
    if scenario in cache["scenarios"]:
        cached = cache["scenarios"][scenario]
        if time.time() - cached["time"] < 600:
            return cached["data"]
    try:
        from agents.scenario_agent import run_scenario_agent
        risk = cache["risk"]
        if not risk:
            from agents.risk_agent import run_risk_agent
            risk = run_risk_agent()
            cache["risk"] = risk
            cache["risk_time"] = time.time()
        result = run_scenario_agent(risk, scenario)
        cache["scenarios"][scenario] = {"data": result, "time": time.time()}
        return result
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e)[:200]})

@app.get("/api/pipeline")
def get_pipeline():
    # Same continuous-refresh model as /api/risk: the background loop keeps
    # this warm, so a request just reads the cache. Only a genuine cold start
    # (no cycle has completed yet) falls through to computing it here.
    if cache["pipeline_result"] is not None:
        age = int(time.time() - cache["pipeline_time"])
        result = dict(cache["pipeline_result"])
        result["cache_age_seconds"] = age
        return result
    try:
        from agents.pipeline import run_pipeline
        result = run_pipeline("api_request")
        cache["pipeline_result"] = result
        cache["pipeline_time"] = time.time()
        if result.get("risk_data"):
            cache["risk"] = result["risk_data"]
            cache["risk_time"] = time.time()
        if result.get("scenario_data") and result.get("active_scenario"):
            cache["scenarios"][result["active_scenario"]] = {
                "data": result["scenario_data"],
                "time": time.time()
            }
        return result
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e)[:200]})

@app.get("/api/twin/state")
def get_twin_state():
    # Try pipeline result first
    if cache.get("pipeline_result") and cache["pipeline_result"].get("twin_state"):
        return cache["pipeline_result"]["twin_state"]

    # Fallback to constructing from risk cache
    risk = cache.get("risk") or {}
    corridors = risk.get("corridors", {})
    hormuz_score = corridors.get("hormuz", {}).get("risk_score", 0)
    redsea_score = corridors.get("red_sea", {}).get("risk_score", 0)
    brent = risk.get("commodity_signals", {}).get("brent_price", 69.56)
    alert = risk.get("alert_level", "LOW")

    active_scenario = None
    scenario_data = None
    if cache["scenarios"]:
        latest = max(cache["scenarios"].items(), key=lambda x: x[1]["time"])
        active_scenario = latest[0]
        scenario_data = latest[1]["data"]

    d30 = scenario_data.get("day_30", {}) if scenario_data else {}

    return {
        "alert_level": alert,
        "brent_price": brent,
        "hormuz_risk": hormuz_score,
        "redsea_risk": redsea_score,
        "active_scenario": active_scenario,
        "stressed_refineries": [],
        "spr_status": {"total_days": 9.5, "drawn": 0, "remaining": 9.5},
        "spr_drawdown": d30.get("spr_drawdown_days", 0),
        "scenario_day30": d30,
        "network_health": "NORMAL",
    }

# ── AGENTIC DIGITAL TWIN ──
def _parse_ids(csv):
    return [x for x in (csv or "").split(",") if x]

@app.get("/api/twin/network")
def get_twin_network(offline: str = "", ports: str = ""):
    """Full network geography + a flow simulation. Optional ?offline=id,id and
    ?ports=id,id re-solve the network with those assets down/stressed."""
    from agents import twin_sim
    off = _parse_ids(offline)
    port_stress = {p: 50 for p in _parse_ids(ports)}
    sim = twin_sim.simulate(offline=off, port_stress=port_stress)
    return {
        "geo": twin_sim.all_nodes_geo(),
        "edges": {"crude": twin_sim.CRUDE_EDGES, "product": twin_sim.PRODUCT_EDGES},
        "totals": {
            "refining_mmtpa": twin_sim.TOTAL_REFINING,
            "demand_mmtpa": twin_sim.TOTAL_DEMAND,
            "spr_mmt": twin_sim.TOTAL_SPR_MMT,
            "spr_days": twin_sim.TOTAL_SPR_DAYS,
        },
        "simulation": sim,
    }

@app.get("/api/twin/vessels")
def get_twin_vessels(region: str = None):
    from agents import ais_feed
    ais_feed.start_collector()
    return ais_feed.get_vessels(region)

@app.get("/api/twin/critical")
def get_twin_critical():
    from agents import twin_sim
    return {"critical_refineries": twin_sim.node_criticality(8)}

@app.get("/api/vessels")
def get_vessels(region: str = None):
    """Live AIS tanker positions (aisstream.io) across Hormuz, Red Sea, and India
    coasts. Shared by the Live Map and the Digital Twin."""
    from agents import ais_feed
    ais_feed.start_collector()
    return ais_feed.get_vessels(region)

@app.post("/api/risk/chat")
def post_risk_chat(body: dict):
    """Converse with the Risk Analyst agent over the current live risk assessment.
    Body: {message, session_id?}. Returns narrative + map directives."""
    message = (body or {}).get("message", "").strip()
    if not message:
        return JSONResponse(status_code=400, content={"error": "message required"})
    session_id = (body or {}).get("session_id") or "analyst_default"
    try:
        from agents.risk_analyst import run_risk_analyst
        return run_risk_analyst(message, session_id, cache.get("risk") or {})
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e)[:200]})

@app.post("/api/twin/chat")
def post_twin_chat(body: dict):
    """Converse with the Twin Operator Agent. Body: {message, session_id?, context?}.
    Returns the agent's narrative, structured map actions, and the exact flow sim of
    its plan."""
    message = (body or {}).get("message", "").strip()
    if not message:
        return JSONResponse(status_code=400, content={"error": "message required"})
    session_id = (body or {}).get("session_id") or "twin_default"
    context = (body or {}).get("context") or {}
    try:
        from agents.twin_operator import run_twin_operator
        return run_twin_operator(message, session_id, context)
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e)[:200]})


# ── BUSINESS IMPACT ADVISOR ──
def _get_live_business_context(scenario_override: str = None):
    """Shared by every entry point into the Business Advisor — the wizard's
    own /api/business/analyze and the ERP integration endpoints below — so an
    ERP-submitted request is reasoned over with exactly the same live risk,
    scenario, and cascade numbers a person sees in the web UI, never a
    separate or lesser computation."""
    import time as _time
    from agents.risk_agent import run_risk_agent
    from agents.scenario_agent import run_scenario_agent
    from agents.pipeline import cascade_agent_node
    from agents import market_data

    if cache["risk"] is not None:
        risk = cache["risk"]
    else:
        risk = run_risk_agent()
        cache["risk"] = risk
        cache["risk_time"] = _time.time()

    scenario = scenario_override
    if not scenario:
        corridors = risk.get("corridors", {})
        scenario = max(corridors.items(), key=lambda x: x[1].get("risk_score", 0))[0] if corridors else "hormuz"

    if scenario in cache["scenarios"] and _time.time() - cache["scenarios"][scenario]["time"] < 600:
        scenario_data = cache["scenarios"][scenario]["data"]
    else:
        scenario_data = run_scenario_agent(risk, scenario)
        cache["scenarios"][scenario] = {"data": scenario_data, "time": _time.time()}

    brent = risk.get("commodity_signals", {}).get("brent_price", 69.56)
    cascade_state = cascade_agent_node({"brent_price": brent, "scenario_data": scenario_data})
    cascade_sectors = cascade_state["cascade_data"]["sectors"]
    fx = market_data.get_fx_rate()
    return risk, scenario, scenario_data, cascade_sectors, fx


@app.get("/api/business/materials")
def get_business_materials():
    from agents.business_impact import list_materials
    return {"materials": list_materials()}

@app.post("/api/business/analyze")
def post_business_analyze(body: dict):
    """Run a business profile through the live cascade model. Body: {profile, scenario?, session_id?}.
    Returns the exact financial impact plus the advisor agent's narrative and recommendations."""
    import uuid
    from agents.business_advisor import analyze_business
    profile = (body or {}).get("profile")
    if not profile:
        return JSONResponse(status_code=400, content={"error": "profile required"})
    session_id = (body or {}).get("session_id") or f"biz_{uuid.uuid4().hex[:12]}"

    try:
        risk, scenario, scenario_data, cascade_sectors, fx = _get_live_business_context((body or {}).get("scenario"))
        result = analyze_business(session_id, profile, cascade_sectors, fx, scenario_data.get("scenario_name"))
        result["session_id"] = session_id
        result["scenario"] = scenario
        result["scenario_name"] = scenario_data.get("scenario_name")
        return result
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e)[:300]})


# ── ERP INTEGRATION ──
# A generic REST surface any ERP — SAP via its HTTP/OData/CPI outbound
# adapters, Oracle, Dynamics, or a custom system — can call directly, so a
# business's procurement/finance data already sitting in their ERP doesn't
# need to be re-typed into the web wizard by hand. Authenticated by a shared
# API key (ERP_API_KEY in .env); every endpoint lives under /api/erp/v1.
def _check_erp_api_key(x_api_key: str = None):
    expected = os.getenv("ERP_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="ERP integration not configured — set ERP_API_KEY in the server's .env")
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")

@app.get("/api/erp/v1/health")
def erp_health(x_api_key: str = Header(default=None, alias="X-API-Key")):
    """Connectivity + credential check an integrator calls once while wiring
    up the connection, before sending real business data."""
    _check_erp_api_key(x_api_key)
    return {"status": "ok", "service": "APEX ERP Integration", "version": "v1"}

@app.get("/api/erp/v1/materials")
def erp_materials(x_api_key: str = Header(default=None, alias="X-API-Key")):
    """The full 31-item material taxonomy with ERP-facing aliases and a SAP
    material-group hint per item, so an integrator can build a one-time
    mapping table on their side rather than relying on runtime fuzzy-matching
    for every request."""
    _check_erp_api_key(x_api_key)
    from agents.erp_integration import materials_reference
    return {"materials": materials_reference()}

@app.post("/api/erp/v1/impact-analysis")
def erp_impact_analysis(body: dict, x_api_key: str = Header(default=None, alias="X-API-Key")):
    """The main integration endpoint. Body is an ERP-shaped payload — see
    agents/erp_integration.py for the exact field mapping and material
    matching rules. Runs through the identical live Business Advisor Agent
    pipeline as the web wizard (agents.business_advisor.analyze_business) —
    an ERP-submitted request gets the same genuine, agent-invoked
    calculation, not a simplified stand-in. Unmapped materials are reported
    back explicitly rather than silently dropped or guessed at."""
    import uuid
    from agents.erp_integration import build_profile_from_erp_payload
    from agents.business_advisor import analyze_business
    _check_erp_api_key(x_api_key)

    if not body or not body.get("materials"):
        return JSONResponse(status_code=400, content={"error": "at least one material is required"})

    profile, unmapped, mapping_report = build_profile_from_erp_payload(body)
    if not profile["materials"]:
        return JSONResponse(status_code=422, content={
            "error": "none of the submitted materials could be matched to APEX's taxonomy",
            "unmapped_materials": unmapped,
            "hint": "GET /api/erp/v1/materials for the full taxonomy and accepted aliases",
        })

    session_id = f"erp_{uuid.uuid4().hex[:12]}"
    try:
        risk, scenario, scenario_data, cascade_sectors, fx = _get_live_business_context(body.get("scenario"))
        result = analyze_business(session_id, profile, cascade_sectors, fx, scenario_data.get("scenario_name"))
        result["session_id"] = session_id
        result["scenario"] = scenario
        result["scenario_name"] = scenario_data.get("scenario_name")
        result["material_mapping"] = mapping_report
        if unmapped:
            result["unmapped_materials"] = unmapped
        return result
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e)[:300]})

@app.post("/api/business/chat")
def post_business_chat(body: dict):
    message = (body or {}).get("message", "").strip()
    session_id = (body or {}).get("session_id")
    if not message or not session_id:
        return JSONResponse(status_code=400, content={"error": "message and session_id required"})
    try:
        from agents.business_advisor import run_advisor
        return run_advisor(message, session_id)
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e)[:300]})

@app.get("/api/cascade")
def get_cascade():
    if cache.get("pipeline_result") and cache["pipeline_result"].get("cascade_data"):
        return cache["pipeline_result"]["cascade_data"]
    return {"error": "Run /api/pipeline first to generate cascade data"}

@app.get("/api/health")
def health():
    return {
        "status": "live",
        "alert_level": cache["risk"].get("alert_level") if cache["risk"] else None,
        "pipeline_cached": cache["pipeline_result"] is not None,
        "risk_cached": cache["risk"] is not None,
        "scenarios_cached": list(cache["scenarios"].keys()),
    }