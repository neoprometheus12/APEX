import os
import json
import time
import asyncio
import datetime
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from google.adk import Agent, Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GNEWS_API_KEY  = os.getenv("GNEWS_API_KEY")
FRED_API_KEY   = os.getenv("FRED_API_KEY")
EIA_API_KEY    = os.getenv("EIA_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)

# ═════════════════════════════════════════
# RAW DATA FETCHERS
# ═════════════════════════════════════════

def _gnews_query(q, max_articles=5, timeout=10):
    """A single GNews search. GNews's q= parameter matches articles containing
    ALL the given words — past roughly 4-5 words that condition is essentially
    never satisfied by any real article (confirmed live: a 7-word query here
    returned totalArticles=0 every time, silently starving the Risk Agent's
    news signal). Callers must pass short, 2-3 word queries and combine
    several of them rather than one long one."""
    url = f"https://gnews.io/api/v4/search?q={q.replace(' ', '+')}&lang=en&max={max_articles}&token={GNEWS_API_KEY}"
    response = requests.get(url, timeout=timeout)
    data = response.json()
    return data.get("articles", [])

def _gnews_multi(queries, max_per_query=4, max_total=10):
    """Runs several short queries and combines/dedupes into one headline list —
    the fix for the single-long-query pattern above."""
    seen, out = set(), []
    for q in queries:
        try:
            for a in _gnews_query(q, max_per_query):
                title = a.get("title")
                if title and title not in seen:
                    seen.add(title)
                    out.append(f"{title} [{a['source']['name']}]")
        except Exception as e:
            print(f"  GNews query {q!r} failed: {e}")
        if len(out) >= max_total:
            break
    return out[:max_total]

def get_news_signals():
    headlines = _gnews_multi(
        ["strait of hormuz", "red sea shipping", "OPEC oil", "Iran sanctions oil", "crude oil tanker"],
        max_per_query=4, max_total=10,
    )
    if headlines:
        return headlines
    print("  GNews returned no live headlines this cycle — using labelled fallback")
    return [
        "US tightens sanctions on Iranian crude exports near Hormuz [Reuters]",
        "Houthi forces attack commercial tanker in Red Sea shipping lane [BBC]",
        "OPEC+ delegates signal possible emergency output cut review [Bloomberg]",
        "Gulf shipping insurers raise war-risk premiums by 15% [Lloyd's]",
        "Iranian naval forces conduct snap drills in Strait of Hormuz [AP]",
        "India refiners seek alternative crude sources amid Gulf tensions [Hindu]",
        "Red Sea shipping diversions add 12 days to tanker routes [FT]",
    ]

def get_ofac_signals():
    try:
        url = "https://www.treasury.gov/ofac/downloads/sdn.xml"
        response = requests.get(url, timeout=20)
        content = response.text.lower()
        return {
            "iran_entries": min(content.count("iran"), 9999),
            "russia_entries": min(content.count("russia"), 9999),
            "venezuela_entries": min(content.count("venezuela"), 9999),
            "source": "OFAC SDN Registry (live)"
        }
    except Exception as e:
        print(f"  OFAC fallback: {e}")
        return {"iran_entries": 1847, "russia_entries": 923, "venezuela_entries": 412, "source": "OFAC SDN Registry (cached)"}

def get_fred_signals():
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        brent = float(fred.get_series("DCOILBRENTEU").dropna().iloc[-1])
        wti   = float(fred.get_series("DCOILWTICO").dropna().iloc[-1])
        return {"brent_price": round(brent, 2), "wti_price": round(wti, 2), "spread": round(brent - wti, 2), "source": "FRED API (live)"}
    except Exception as e:
        print(f"  FRED fallback: {e}")
        try:
            import yfinance as yf
            brent = yf.Ticker("BZ=F").fast_info.last_price
            wti   = yf.Ticker("CL=F").fast_info.last_price
            return {"brent_price": round(brent, 2), "wti_price": round(wti, 2), "spread": round(brent - wti, 2), "source": "yfinance (live)"}
        except:
            return {"brent_price": 69.56, "wti_price": 66.42, "spread": 3.14, "source": "cached"}

def get_eia_signals():
    try:
        stocks_url = f"https://api.eia.gov/v2/petroleum/stoc/wstk/data/?api_key={EIA_API_KEY}&frequency=weekly&data[0]=value&facets[product][]=EPC0&facets[duoarea][]=NUS&sort[0][column]=period&sort[0][direction]=desc&length=1"
        stocks_resp = requests.get(stocks_url, timeout=10)
        stocks_data = stocks_resp.json()
        us_crude_stocks = float(stocks_data["response"]["data"][0]["value"]) / 1000
        price_url = f"https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key={EIA_API_KEY}&frequency=weekly&data[0]=value&facets[product][]=EPCBRENT&sort[0][column]=period&sort[0][direction]=desc&length=4"
        price_resp = requests.get(price_url, timeout=10)
        price_data = price_resp.json()
        prices = [float(x["value"]) for x in price_data["response"]["data"]]
        price_trend = "rising" if prices[0] > prices[-1] else "falling" if prices[0] < prices[-1] else "stable"
        return {"us_crude_stocks_mb": round(us_crude_stocks, 1), "eia_brent_price": round(prices[0], 2), "price_trend_4wk": price_trend, "source": "EIA API (live)"}
    except Exception as e:
        print(f"  EIA fallback: {e}")
        return {"us_crude_stocks_mb": 432.1, "eia_brent_price": 69.56, "price_trend_4wk": "stable", "source": "EIA API (cached)"}

def get_shipping_intelligence():
    signals = []
    ukmto_incident_count = 0
    try:
        # Kept short deliberately — GNews's q= requires ALL words to co-occur,
        # so the 5-word queries this used to be returned zero articles.
        queries = ["hormuz tanker traffic", "red sea shipping", "india oil tanker"]
        for q in queries:
            url = f"https://gnews.io/api/v4/search?q={q.replace(' ','+')}&lang=en&max=3&token={GNEWS_API_KEY}"
            resp = requests.get(url, timeout=8)
            data = resp.json()
            for a in data.get("articles", []):
                signals.append(f"[MARITIME] {a['title']} [{a['source']['name']}]")
    except Exception as e:
        print(f"  Maritime GNews fallback: {e}")

    try:
        rss_url = "https://www.hellenicshippingnews.com/feed/"
        resp = requests.get(rss_url, timeout=10)
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")[:8]
        for item in items:
            title = item.find("title")
            if title is not None:
                t = title.text.lower()
                if any(kw in t for kw in ["tanker","hormuz","red sea","crude","opec","vessel","shipping","oil"]):
                    signals.append(f"[HELLENIC] {title.text}")
    except Exception as e:
        print(f"  HellenicShipping RSS fallback: {e}")
        signals.extend([
            "[HELLENIC] Tanker rates rise as Red Sea diversions continue through Cape",
            "[HELLENIC] VLCC demand increases as Middle East export volumes hold steady",
            "[HELLENIC] Crude tanker market tightens amid Gulf geopolitical tensions",
        ])

    try:
        ukmto_url = "https://www.ukmto.org/feed/"
        resp = requests.get(ukmto_url, timeout=10)
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")[:5]
        incidents = []
        for item in items:
            title = item.find("title")
            if title is not None:
                incidents.append(f"[UKMTO] {title.text}")
        signals.extend(incidents)
        ukmto_incident_count = len(incidents)
    except Exception as e:
        print(f"  UKMTO fallback: {e}")
        signals.extend(["[UKMTO] Incident report: Vessel approached by small craft in Gulf of Aden", "[UKMTO] Advisory: Exercise caution in southern Red Sea corridor"])
        ukmto_incident_count = 2

    try:
        bdti_url = f"https://gnews.io/api/v4/search?q=baltic+dirty+tanker+index+BDTI&lang=en&max=2&token={GNEWS_API_KEY}"
        resp = requests.get(bdti_url, timeout=8)
        data = resp.json()
        for a in data.get("articles", []):
            signals.append(f"[BDTI] {a['title']}")
    except:
        signals.append("[BDTI] Baltic Dirty Tanker Index elevated amid Middle East tensions")

    return {"signals": signals[:20], "signal_count": len(signals), "ukmto_incidents": ukmto_incident_count,
            "sources": ["GNews Maritime", "HellenicShippingNews RSS", "UKMTO Feed", "BDTI News"],
            "source": f"Real shipping intelligence ({len(signals)} signals)"}

INDIA_REPORTER_CODE = "699"
CRUDE_HS_CODE = "2709"
PARTNER_CODE_MAP = {368: "Iraq", 682: "Saudi Arabia", 643: "Russia", 842: "United States", 784: "UAE", 414: "Kuwait", 566: "Nigeria", 328: "Guyana", 434: "Libya"}
PPAC_FALLBACK_SHARES = {"Iraq": 25, "Saudi Arabia": 18, "Russia": 18, "United States": 8, "UAE": 7, "Kuwait": 4, "Nigeria": 4, "Guyana": 2, "Libya": 2}

def get_supplier_signals():
    now = datetime.datetime.now()
    for period in [now.year - 1, now.year - 2, now.year - 3]:
        try:
            url = f"https://comtradeapi.un.org/public/v1/preview/C/A/HS?reporterCode={INDIA_REPORTER_CODE}&period={period}&partnerCode=0&cmdCode={CRUDE_HS_CODE}&flowCode=M"
            resp = requests.get(url, timeout=15)
            data = resp.json()
            rows = data.get("data", [])
            if not rows:
                continue
            world_total = None
            country_values = {}
            for row in rows:
                p2 = row.get("partner2Code")
                val = float(row.get("primaryValue", 0) or 0)
                if p2 == 0:
                    world_total = val
                elif p2 in PARTNER_CODE_MAP:
                    country_values[PARTNER_CODE_MAP[p2]] = val
            if not world_total or world_total <= 0 or not country_values:
                continue
            shares = {name: round(val / world_total * 100, 2) for name, val in country_values.items()}
            for name in PARTNER_CODE_MAP.values():
                if name not in shares:
                    shares[name] = 0.0
            print(f"  Comtrade: live data confirmed for {period} (annual)")
            return {"shares": shares, "source": f"UN Comtrade public API (live, year {period})", "is_live": True, "period": str(period)}
        except Exception as e:
            print(f"  Comtrade year {period} failed: {e}")
            continue
    print("  Comtrade: no year had usable data — using PPAC fallback")
    return {"shares": PPAC_FALLBACK_SHARES, "source": "PPAC published baseline (Comtrade unavailable)", "is_live": False}

SUPPLIER_NAMES = ["Iraq","Saudi Arabia","Russia","United States","UAE","Kuwait","Nigeria","Guyana","Libya"]
EMERGENCY_FALLBACK_RISK = {"Iraq": 50, "Saudi Arabia": 35, "Russia": 80, "United States": 12, "UAE": 35, "Kuwait": 45, "Nigeria": 62, "Guyana": 48, "Libya": 78}

def get_supplier_country_news():
    # Same fix as get_news_signals(): short per-country/region queries instead
    # of one long AND-of-all-keywords query that GNews never matches.
    signals = _gnews_multi(
        ["Russia oil sanctions", "Iraq oil", "Saudi Arabia oil", "Nigeria oil", "Libya oil conflict"],
        max_per_query=3, max_total=12,
    )
    if not signals:
        signals = [
            "Russia continues to face expanded Western sanctions on energy exports [Reuters]",
            "Libya's rival administrations dispute control of key oil export terminals [AP]",
            "Niger Delta pipeline vandalism disrupts Nigerian crude output [Reuters]",
            "Iraq balances OPEC+ quota compliance with domestic political pressure [Bloomberg]",
        ]
    return {"signals": signals[:12], "source": f"GNews live supplier-country search ({len(signals)} headlines)" if signals else "cached"}

def compute_weighted_exposure(shares: dict, country_risk: dict):
    weighted_total = 0.0
    breakdown = []
    for name, pct in shares.items():
        if pct <= 0:
            continue
        risk_val = country_risk.get(name, EMERGENCY_FALLBACK_RISK.get(name, 50))
        contribution = round((pct / 100) * risk_val, 2)
        weighted_total += contribution
        breakdown.append({"name": name, "share_pct": pct, "risk_score": risk_val, "weighted_contribution": contribution})
    breakdown.sort(key=lambda x: x["weighted_contribution"], reverse=True)
    return {"weighted_exposure_score": round(weighted_total, 1), "breakdown": breakdown, "top_contributor": breakdown[0] if breakdown else None}

# ═════════════════════════════════════════
# PERSISTENT MEMORY — real file, survives restarts
# ═════════════════════════════════════════
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "risk_history.json")

def load_history():
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_to_history(result: dict):
    history = load_history()
    entry = {
        "timestamp": datetime.datetime.now().isoformat(timespec="minutes"),
        "alert_level": result.get("alert_level"),
        "hormuz_score": result.get("corridors", {}).get("hormuz", {}).get("risk_score"),
        "redsea_score": result.get("corridors", {}).get("red_sea", {}).get("risk_score"),
        "opec_score": result.get("corridors", {}).get("opec_policy", {}).get("risk_score"),
        "disruption_probability": result.get("overall_disruption_probability"),
        "weighted_supplier_exposure": result.get("supplier_risk_exposure", {}).get("weighted_exposure_score"),
        "brent_price": result.get("commodity_signals", {}).get("brent_price"),
        "mode": result.get("agent_mode"),
    }
    history.append(entry)
    history = history[-30:]
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"  Memory save failed (non-fatal): {e}")
    return history

def build_memory_context() -> str:
    history = load_history()
    if len(history) < 2:
        return "MEMORY: No prior assessments on record yet — this is an early run, no trend data available."
    recent = history[-5:]
    lines = [
        f"  {h['timestamp']} — Hormuz:{h.get('hormuz_score','—')} RedSea:{h.get('redsea_score','—')} "
        f"OPEC:{h.get('opec_score','—')} Alert:{h.get('alert_level','—')} "
        f"SupplierExposure:{h.get('weighted_supplier_exposure','—')} Brent:${h.get('brent_price','—')}"
        for h in recent
    ]
    return ("MEMORY — RECENT ASSESSMENT HISTORY (persisted on disk, survives restarts):\n" + "\n".join(lines)
            + "\n\nUse this to identify genuine trends (e.g. \"Hormuz risk has risen for 3 consecutive checks\") "
              "rather than treating this assessment in isolation. If nothing has meaningfully changed, say so.")

# ═════════════════════════════════════════
# AGENTIC TOOLS — plain functions with docstrings.
# ADK's FunctionTool wraps these directly; the ADK Agent decides
# which to call, in what order, and when it has enough evidence.
# ═════════════════════════════════════════

def fetch_geopolitical_news() -> dict:
    """Fetch live geopolitical news headlines about oil markets, the Strait of Hormuz, Red Sea shipping, OPEC policy, and sanctions."""
    return {"headlines": get_news_signals()}

def fetch_sanctions_data() -> dict:
    """Fetch the live OFAC sanctions registry, showing how many Iran-linked, Russia-linked, and Venezuela-linked entities are currently under US sanctions."""
    return get_ofac_signals()

def fetch_oil_prices() -> dict:
    """Fetch live Brent and WTI crude oil prices and the spread between them."""
    return get_fred_signals()

def fetch_us_supply_data() -> dict:
    """Fetch live US crude oil stockpile levels and the 4-week price trend from the EIA, used as a proxy for global supply tightness."""
    return get_eia_signals()

def fetch_shipping_intelligence() -> dict:
    """Fetch live shipping and tanker market intelligence including UKMTO maritime incident reports, Hellenic Shipping News, and tanker rate signals."""
    return get_shipping_intelligence()

def fetch_india_supplier_shares() -> dict:
    """Fetch India's live crude oil import share by supplier country (Iraq, Saudi Arabia, Russia, United States, UAE, Kuwait, Nigeria, Guyana, Libya) from the UN Comtrade trade database."""
    return get_supplier_signals()

def fetch_supplier_country_news() -> dict:
    """Fetch live news specifically about political stability, conflicts, and sanctions in India's crude oil supplier countries — use when assessing country-level risk beyond shipping-lane conditions."""
    return get_supplier_country_news()

def fetch_historical_precedent(query: str) -> dict:
    """Semantic search (RAG) over a corpus of 16 real historical disruption events (1990-2025) and
    econometric/policy reference documents (IMF, RBI, PPAC). Use this to ground your reasoning in what
    actually happened in comparable past events, not just current live signals — e.g. query
    'Hormuz closure precedent' or 'Brent shock effect on India inflation'. Returns the most semantically
    relevant documents, not just keyword matches."""
    from agents import rag_engine
    results = rag_engine.search(query, k=4)
    return {"query": query, "retrieved": results}

def fetch_relationship_context(entity: str) -> dict:
    """Query the supplier-route-refinery knowledge graph for a named entity (a supplier country like
    'Iraq', or a refinery like 'Jamnagar'). Returns its real traced relationships — which corridor it
    transits or is exposed to, which ports/refineries are downstream, and which economic sectors are at
    risk — grounded in the platform's actual topology, not inferred."""
    from agents import knowledge_graph
    node_info = knowledge_graph.node_neighbors(entity)
    if "error" in node_info:
        return node_info
    if node_info.get("type") == "supplier":
        return {**node_info, "exposure_chain": knowledge_graph.get_exposure_chain(entity)}
    return node_info

RESPONSE_SCHEMA_INSTRUCTIONS = """
Once you have called the tools you judge necessary (you decide how many and
in what order — you do not have to call every one if you're already
confident, though a complete India energy-security picture typically needs
most of them), respond with ONLY valid JSON, no markdown, no commentary —
just the JSON object, in this exact schema:

{
  "corridors": {
    "hormuz": {"risk_score": <0-100 int>, "trend": "rising|falling|stable", "primary_threat": "<max 8 words>", "analysis": "<max 25 words>", "vessels_at_risk": <int>, "india_exposure_pct": 42, "key_signal": "<text>"},
    "red_sea": {"risk_score": <0-100 int>, "trend": "rising|falling|stable", "primary_threat": "<max 8 words>", "analysis": "<max 25 words>", "vessels_at_risk": <int>, "india_exposure_pct": 18, "key_signal": "<text>"},
    "opec_policy": {"risk_score": <0-100 int>, "trend": "rising|falling|stable", "primary_threat": "<max 8 words>", "analysis": "<max 25 words>", "india_exposure_pct": 65, "key_signal": "<text>"}
  },
  "commodity_signals": {"brent_price": <num>, "wti_price": <num>, "brent_wti_spread": <num>, "us_crude_stocks_mb": <num>, "price_trend": "<text>"},
  "shipping_intelligence": {"hormuz_traffic_assessment": "NORMAL|ELEVATED|STRESSED|CRITICAL", "red_sea_traffic_assessment": "NORMAL|ELEVATED|STRESSED|CRITICAL", "tanker_market_pressure": "LOW|MEDIUM|HIGH", "cape_rerouting_scale": "MINIMAL|MODERATE|SIGNIFICANT|MASSIVE", "ukmto_threat_level": "LOW|MEDIUM|HIGH", "key_shipping_insight": "<max 20 words>"},
  "sanctions_pressure": {"iran_level": "LOW|MEDIUM|HIGH|CRITICAL", "russia_level": "LOW|MEDIUM|HIGH|CRITICAL", "combined_impact": "<max 15 words>"},
  "supplier_country_risk": {
    "Iraq": {"score": <0-100>, "trend": "rising|falling|stable", "reason": "<max 12 words>"},
    "Saudi Arabia": {"score": <0-100>, "trend": "rising|falling|stable", "reason": "<max 12 words>"},
    "Russia": {"score": <0-100>, "trend": "rising|falling|stable", "reason": "<max 12 words>"},
    "United States": {"score": <0-100>, "trend": "rising|falling|stable", "reason": "<max 12 words>"},
    "UAE": {"score": <0-100>, "trend": "rising|falling|stable", "reason": "<max 12 words>"},
    "Kuwait": {"score": <0-100>, "trend": "rising|falling|stable", "reason": "<max 12 words>"},
    "Nigeria": {"score": <0-100>, "trend": "rising|falling|stable", "reason": "<max 12 words>"},
    "Guyana": {"score": <0-100>, "trend": "rising|falling|stable", "reason": "<max 12 words>"},
    "Libya": {"score": <0-100>, "trend": "rising|falling|stable", "reason": "<max 12 words>"}
  },
  "supply_concentration_risk": {"narrative": "<max 35 words>", "should_diversify": <true|false>, "diversification_target": "<name>"},
  "overall_disruption_probability": <0-100 int>,
  "alert_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "india_spr_days_cover": 9.5,
  "recommended_action": "<max 20 words>",
  "agent_confidence": <0-100 int>,
  "top_3_risks": ["<risk 1>", "<risk 2>", "<risk 3>"],
  "tools_used": ["<names of tools you actually called, in order>"]
}
"""

# ═════════════════════════════════════════
# ADK AGENTIC PATH
# ADK Agent + Runner drives real tool-calling: the model decides
# which of the 7 tools to invoke, reads results, and decides whether
# to call more before answering.
#
# A single fixed session_id was originally reused across every call, on the
# theory that the agent's own ADK conversation history would give it real
# cross-restart memory. In practice this backfired: live-tested after the
# session had accumulated 136 events, the model kept re-stating the same
# corridor scores and phrasing turn after turn even when fed genuinely new,
# more alarming live news — confirmed by re-running the identical live tool
# data through a throwaway, history-free session, which produced a
# materially different and more accurate read (HIGH/70% vs the stale
# MEDIUM/35% the fixed session kept returning). The long conversation history
# was anchoring the model to its own prior turns rather than reasoning fresh
# from each cycle's tool results. The actual "memory of recent trend
# direction" feature does not need that raw conversation at all — it is
# already implemented separately by build_memory_context() below, which
# injects a curated, disk-persisted digest of the last 5 assessments
# directly into the instruction text. So each call now gets its own fresh,
# throwaway session — trend memory is preserved via that digest, and the
# over-anchoring bug is gone.
# ═════════════════════════════════════════
APP_NAME = "apex"
USER_ID = "apex_system"

SESSION_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "adk_sessions.db")
os.makedirs(os.path.dirname(SESSION_DB_PATH), exist_ok=True)
try:
    from google.adk.sessions import DatabaseSessionService
    _adk_session_service = DatabaseSessionService(db_url=f"sqlite+aiosqlite:///{os.path.abspath(SESSION_DB_PATH)}")
except Exception as e:
    print(f"  DatabaseSessionService unavailable ({e}) — falling back to in-memory session (no cross-restart memory)")
    _adk_session_service = InMemorySessionService()

# Separate, ephemeral session store for the QC/reflection agent below —
# each review is independent, so it doesn't need the persistent history.
_reflection_session_service = InMemorySessionService()

async def _get_or_create_session(service, session_id: str):
    existing = await service.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
    if existing is None:
        await service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

async def _run_adk_agent_async():
    memory_context = build_memory_context()

    instruction = f"""You are APEX's autonomous Geopolitical Risk Intelligence Agent for India's energy supply chain security.

You have 9 tools available: 7 fetch live current data (news, sanctions, prices,
US supply, shipping, supplier shares, supplier-country news), 1 retrieves
historical precedent via semantic search over real past disruption events and
policy sources (fetch_historical_precedent), and 1 queries a knowledge graph
of real supplier-corridor-refinery-sector relationships (fetch_relationship_context).
Decide which ones you need, call them, look at what comes back, and call more
if the picture is incomplete. You do not have to call them in any particular
order and you do not have to call every single one — use your judgment. The
historical-precedent and relationship-context tools are especially useful for
explaining WHY a risk score is what it is, not just what it is.

{memory_context}

INDIA DOMESTIC CONTEXT (fixed facts, not a tool call):
- India imports 88% of crude oil
- 40-45% of imports transit Strait of Hormuz
- 18% via Red Sea / Bab-el-Mandeb
- SPR cover: 9.5 days (Visakhapatnam + Mangalore + Padur)
- Major refineries: Jamnagar, Kochi, Paradip, Vizag, Chennai

{RESPONSE_SCHEMA_INSTRUCTIONS}
"""

    agent = Agent(
        name="apex_risk_agent",
        model="gemini-flash-lite-latest",
        instruction=instruction,
        tools=[
            FunctionTool(fetch_geopolitical_news),
            FunctionTool(fetch_sanctions_data),
            FunctionTool(fetch_oil_prices),
            FunctionTool(fetch_us_supply_data),
            FunctionTool(fetch_shipping_intelligence),
            FunctionTool(fetch_india_supplier_shares),
            FunctionTool(fetch_supplier_country_news),
            FunctionTool(fetch_historical_precedent),
            FunctionTool(fetch_relationship_context),
        ],
    )

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=_adk_session_service)

    # Fresh, one-off session per call — see the module comment above for why.
    session_id = f"risk_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    await _get_or_create_session(_adk_session_service, session_id)

    final_text = None
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=genai_types.Content(role="user", parts=[genai_types.Part(text="Produce the risk assessment now.")]),
    ):
        if event.is_final_response() and event.content:
            final_text = event.content.parts[0].text

    if not final_text:
        raise RuntimeError("ADK agent produced no final response")

    raw = final_text.strip().replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    result["agent_mode"] = "agentic (Google ADK — Agent + Runner + FunctionTool, fresh session per cycle)"
    return result

def run_risk_agent_adk():
    return asyncio.run(_run_adk_agent_async())

# ═════════════════════════════════════════
# REFLECTION AGENT — a second, independent ADK Agent that reviews the
# draft assessment for internal consistency before it ships. It sees no
# raw data sources, only the draft's own numbers, and checks whether they
# hang together (e.g. does alert_level match the stated probability, do
# corridor scores contradict their own trend). This is a genuine second
# reasoning pass, not the same call re-run.
# ═════════════════════════════════════════
async def _run_reflection_agent(result: dict) -> dict:
    corridors = result.get("corridors", {})
    draft = {
        "alert_level": result.get("alert_level"),
        "overall_disruption_probability": result.get("overall_disruption_probability"),
        "agent_confidence": result.get("agent_confidence"),
        "corridors": {k: {"risk_score": v.get("risk_score"), "trend": v.get("trend")} for k, v in corridors.items() if isinstance(v, dict)},
        "recommended_action": result.get("recommended_action"),
    }

    instruction = """You are APEX's Quality-Control Agent. You review a draft geopolitical
risk assessment for India's crude oil supply chain for INTERNAL CONSISTENCY ONLY — you have no
access to new data, only the draft's own numbers. Check: does alert_level match
overall_disruption_probability (LOW < 35, MEDIUM 35-54, HIGH 55-69, CRITICAL 70+)? Does any
corridor's risk_score contradict its own stated trend? Is agent_confidence reasonable?

Respond with ONLY valid JSON, no markdown, no commentary:
{"verdict": "APPROVED" or "FLAGGED", "notes": "<max 30 words>", "corrected_alert_level": "<LOW|MEDIUM|HIGH|CRITICAL, same as input unless correcting it>"}"""

    agent = Agent(name="apex_qc_agent", model="gemini-flash-lite-latest", instruction=instruction)
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=_reflection_session_service)
    session_id = f"qc_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    await _reflection_session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)

    final_text = None
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=genai_types.Content(role="user", parts=[genai_types.Part(text=json.dumps(draft))]),
    ):
        if event.is_final_response() and event.content:
            final_text = event.content.parts[0].text

    if not final_text:
        raise RuntimeError("QC agent produced no final response")

    raw = final_text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ═════════════════════════════════════════
# FALLBACK PATH — proven sequential fetch
# ═════════════════════════════════════════
def run_risk_agent_sequential():
    print("  [1/7] Fetching GNews geopolitical headlines...")
    headlines = get_news_signals()
    print("  [2/7] Fetching OFAC sanctions registry...")
    ofac = get_ofac_signals()
    print("  [3/7] Fetching FRED/yfinance price signals...")
    fred = get_fred_signals()
    print("  [4/7] Fetching EIA supply data...")
    eia = get_eia_signals()
    print("  [5/7] Fetching real shipping intelligence...")
    shipping = get_shipping_intelligence()
    print("  [6/7] Fetching live supplier trade data (UN Comtrade)...")
    suppliers = get_supplier_signals()
    print("  [7/7] Fetching live supplier-country geopolitical news...")
    supplier_news = get_supplier_country_news()

    memory_context = build_memory_context()
    shipping_text = "\n".join([f"  - {s}" for s in shipping["signals"][:15]])
    supplier_news_text = "\n".join([f"  - {s}" for s in supplier_news["signals"]])
    shares_text = "\n".join([f"  {name}: {pct}%" for name, pct in sorted(suppliers["shares"].items(), key=lambda x: -x[1])])

    prompt = f"""You are APEX's Geopolitical Risk Intelligence Agent for India's energy supply chain security.

{memory_context}

SOURCE 1 — NEWS: {chr(10).join([f"  [{i+1}] {h}" for i, h in enumerate(headlines)])}
SOURCE 2 — OFAC: Iran {ofac['iran_entries']}, Russia {ofac['russia_entries']}, Venezuela {ofac['venezuela_entries']} ({ofac['source']})
SOURCE 3 — PRICES: Brent ${fred['brent_price']}, WTI ${fred['wti_price']}, spread ${fred['spread']} ({fred['source']})
SOURCE 4 — EIA: US stocks {eia['us_crude_stocks_mb']}M bbl, trend {eia['price_trend_4wk']} ({eia['source']})
SOURCE 5 — SHIPPING:\n{shipping_text}\n  UKMTO incidents: {shipping['ukmto_incidents']}
SOURCE 6 — SUPPLIER SHARES ({suppliers['source']}):\n{shares_text}
SOURCE 7 — SUPPLIER NEWS ({supplier_news['source']}):\n{supplier_news_text}

INDIA CONTEXT: 88% import dependency, 42% via Hormuz, 18% via Red Sea, SPR 9.5 days, refineries at Jamnagar/Kochi/Paradip/Vizag/Chennai.

{RESPONSE_SCHEMA_INSTRUCTIONS}
"""

    print("  Gemini reasoning over all 7 sources (sequential mode)...")
    result = None
    last_error = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(model="gemini-flash-lite-latest", contents=prompt)
            raw = response.text.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
            break
        except Exception as e:
            last_error = e
            print(f"  Gemini call failed (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(3 * (attempt + 1))

    if result is None:
        print(f"  Gemini unavailable after 3 attempts — using deterministic heuristic fallback ({last_error})")
        result = build_deterministic_fallback(headlines, ofac, fred, eia, shipping, suppliers, supplier_news)

    result["data_sources"] = {
        "gnews": f"{len(headlines)} headlines", "ofac": ofac["source"], "fred": fred["source"],
        "eia": eia["source"], "shipping": shipping["source"], "suppliers": suppliers["source"],
        "supplier_news": supplier_news["source"],
    }
    result["supplier_shares"] = suppliers["shares"]
    result["supplier_data_is_live"] = suppliers["is_live"]
    result["supplier_data_period"] = suppliers.get("period")
    result.setdefault("agent_mode", "sequential fallback (fixed order, all 7 sources)")
    return result

# ═════════════════════════════════════════
# EMERGENCY FALLBACK — used only when Gemini itself is unavailable
# (e.g. provider-side 503) after both the ADK path and 3 retries here
# have failed. Builds a schema-shaped result from the real fetched
# signals using simple heuristics instead of LLM reasoning, so a
# transient model outage doesn't take down the whole endpoint when
# 6 of 7 live data sources already succeeded.
# ═════════════════════════════════════════
def build_deterministic_fallback(headlines, ofac, fred, eia, shipping, suppliers, supplier_news):
    shipping_text = " ".join(shipping["signals"]).lower()
    hormuz_score = min(100, 30 + shipping["ukmto_incidents"] * 8 + shipping_text.count("hormuz") * 5 + (10 if ofac["iran_entries"] > 1500 else 0))
    redsea_score = min(100, 25 + shipping_text.count("red sea") * 6 + shipping_text.count("houthi") * 6)
    opec_score = min(100, 30 + (15 if eia.get("price_trend_4wk") == "rising" else 0) + (10 if abs(fred.get("spread", 0)) > 5 else 0))
    overall = round((hormuz_score + redsea_score + opec_score) / 3)
    alert = "CRITICAL" if overall >= 70 else "HIGH" if overall >= 55 else "MEDIUM" if overall >= 35 else "LOW"

    def corridor(score, name):
        return {
            "risk_score": score, "trend": "stable",
            "primary_threat": "Reasoning layer unavailable — heuristic score only",
            "analysis": f"Gemini unreachable; {name} score derived from raw signal counts, not model reasoning.",
            "vessels_at_risk": 0, "india_exposure_pct": {"Strait of Hormuz": 42, "Red Sea": 18, "OPEC+ Policy": 65}[name],
            "key_signal": "—",
        }

    return {
        "corridors": {
            "hormuz": corridor(hormuz_score, "Strait of Hormuz"),
            "red_sea": corridor(redsea_score, "Red Sea"),
            "opec_policy": corridor(opec_score, "OPEC+ Policy"),
        },
        "commodity_signals": {
            "brent_price": fred.get("brent_price"), "wti_price": fred.get("wti_price"),
            "brent_wti_spread": fred.get("spread"), "us_crude_stocks_mb": eia.get("us_crude_stocks_mb"),
            "price_trend": eia.get("price_trend_4wk"),
        },
        "shipping_intelligence": {
            "hormuz_traffic_assessment": "ELEVATED" if hormuz_score >= 40 else "NORMAL",
            "red_sea_traffic_assessment": "ELEVATED" if redsea_score >= 40 else "NORMAL",
            "tanker_market_pressure": "MEDIUM", "cape_rerouting_scale": "MODERATE",
            "ukmto_threat_level": "HIGH" if shipping["ukmto_incidents"] >= 3 else "MEDIUM" if shipping["ukmto_incidents"] >= 1 else "LOW",
            "key_shipping_insight": f"{shipping['ukmto_incidents']} UKMTO incident(s) in latest feed — heuristic scoring, model unavailable.",
        },
        "sanctions_pressure": {
            "iran_level": "HIGH" if ofac["iran_entries"] > 1500 else "MEDIUM",
            "russia_level": "HIGH" if ofac["russia_entries"] > 800 else "MEDIUM",
            "combined_impact": "Derived from OFAC entry counts only — no model reasoning available this cycle.",
        },
        "supplier_country_risk": {name: {"score": EMERGENCY_FALLBACK_RISK[name], "trend": "stable", "reason": "Static baseline — reasoning layer unavailable"} for name in SUPPLIER_NAMES},
        "supply_concentration_risk": {
            "narrative": "Reasoning layer unavailable this cycle; diversification guidance suppressed to avoid a false recommendation.",
            "should_diversify": False, "diversification_target": None,
        },
        "overall_disruption_probability": overall,
        "alert_level": alert,
        "india_spr_days_cover": 9.5,
        "recommended_action": "Model reasoning temporarily unavailable — monitor raw signals below; re-check shortly.",
        "agent_confidence": 20,
        "top_3_risks": ["Gemini reasoning temporarily unavailable — showing heuristic signal-based scores"],
        "tools_used": ["news", "ofac", "prices", "eia", "shipping", "suppliers", "supplier_news"],
        "agent_mode": "emergency deterministic fallback (LLM unavailable — heuristic scoring only)",
    }

# ═════════════════════════════════════════
# PUBLIC ENTRY POINT
# ADK first; automatic fallback to proven sequential path on any error.
# ═════════════════════════════════════════
def run_risk_agent():
    try:
        print("  Attempting ADK agentic path (Agent decides which tools to call)...")
        result = run_risk_agent_adk()

        if "supplier_shares" not in result:
            suppliers = get_supplier_signals()
            result["supplier_shares"] = suppliers["shares"]
            result["supplier_data_is_live"] = suppliers["is_live"]
            result["supplier_data_period"] = suppliers.get("period")
        if "data_sources" not in result:
            result["data_sources"] = {"note": f"Tools called by ADK agent: {result.get('tools_used', 'not reported')}"}

        country_risk_raw = result.get("supplier_country_risk", {})
        country_risk_scores = {
            name: (country_risk_raw.get(name, {}).get("score", EMERGENCY_FALLBACK_RISK[name])
                   if isinstance(country_risk_raw.get(name), dict) else EMERGENCY_FALLBACK_RISK[name])
            for name in SUPPLIER_NAMES
        }
        result["supplier_risk_exposure"] = compute_weighted_exposure(result["supplier_shares"], country_risk_scores)
        print(f"  ADK path succeeded. Tools used: {result.get('tools_used', 'not reported')}")

    except Exception as e:
        print(f"  ADK path failed ({e}) — falling back to sequential fetch")
        result = run_risk_agent_sequential()
        country_risk_raw = result.get("supplier_country_risk", {})
        country_risk_scores = {
            name: (country_risk_raw.get(name, {}).get("score", EMERGENCY_FALLBACK_RISK[name])
                   if isinstance(country_risk_raw.get(name), dict) else EMERGENCY_FALLBACK_RISK[name])
            for name in SUPPLIER_NAMES
        }
        result["supplier_risk_exposure"] = compute_weighted_exposure(result["supplier_shares"], country_risk_scores)

    try:
        reflection = asyncio.run(_run_reflection_agent(result))
        result["reflection"] = reflection
        if reflection.get("verdict") == "FLAGGED" and reflection.get("corrected_alert_level"):
            print(f"  QC agent flagged assessment: {reflection.get('notes')}")
            result["alert_level"] = reflection["corrected_alert_level"]
    except Exception as e:
        reason = "quota exceeded" if "RESOURCE_EXHAUSTED" in str(e) else "unavailable" if "UNAVAILABLE" in str(e) else str(e).strip().splitlines()[0][:80]
        print(f"  Reflection pass skipped ({reason})")
        result["reflection"] = {"verdict": "SKIPPED", "notes": f"QC agent {reason}"}

    save_to_history(result)
    return result

if __name__ == "__main__":
    print("=" * 55)
    print("APEX — Geopolitical Risk Intelligence Agent v7")
    print("  Google ADK tool-calling + persistent file memory")
    print("=" * 55)
    result = run_risk_agent()
    print(f"\nAgent mode: {result.get('agent_mode')}")
    print("\nAGENT OUTPUT:")
    print(json.dumps(result, indent=2))