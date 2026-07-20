"""
APEX — Digital Twin Flow-Network Simulation Engine
=======================================================
Deterministic physics for India's crude→product supply network, modelled as a
capacitated directed max-flow graph:

    super-source ─▶ import ports + domestic wellheads  (crude supply)
                 ─▶ refineries  (throughput-limited: crude in → product out)
                 ─▶ demand centres  ─▶ super-sink

Taking a node offline (or stressing it) changes edge capacities and the graph is
re-solved with a real max-flow, so downstream starvation, rerouting via alternate
paths, and unmet demand are *computed*, not assumed. This is the deterministic
tool layer the Twin Operator Agent calls — the agent reasons and decides; this
module does the exact arithmetic so the numbers stay trustworthy.

Units: everything in MMTPA (million metric tonnes per annum). Crude→product is
modelled 1:1 (refinery yield loss ignored — fine for a resilience twin).
"""

import networkx as nx

# 1 million barrels/day of crude ≈ 49.6 MMT/year
MBD_TO_MMTPA = 49.6

# ─────────────────────────────────────────────────────────────
# NETWORK TOPOLOGY — nodes carry real capacities + coordinates
# ─────────────────────────────────────────────────────────────
REFINERIES = [
    {"id": "jamnagar",     "name": "Jamnagar",     "company": "Reliance",    "cap": 62.0,  "lat": 22.47, "lng": 70.06, "crude": "Gulf"},
    {"id": "jamnagar_sez", "name": "Jamnagar SEZ", "company": "Reliance",    "cap": 35.0,  "lat": 22.45, "lng": 70.04, "crude": "Gulf"},
    {"id": "panipat",      "name": "Panipat",      "company": "IOCL",        "cap": 15.0,  "lat": 29.39, "lng": 76.97, "crude": "Mixed"},
    {"id": "mathura",      "name": "Mathura",      "company": "IOCL",        "cap": 8.0,   "lat": 27.49, "lng": 77.67, "crude": "Mixed"},
    {"id": "barauni",      "name": "Barauni",      "company": "IOCL",        "cap": 6.0,   "lat": 25.47, "lng": 86.00, "crude": "Domestic+Import"},
    {"id": "paradip",      "name": "Paradip",      "company": "IOCL",        "cap": 15.0,  "lat": 20.32, "lng": 86.60, "crude": "Mixed"},
    {"id": "haldia",       "name": "Haldia",       "company": "IOCL",        "cap": 7.5,   "lat": 22.03, "lng": 88.07, "crude": "Import"},
    {"id": "digboi",       "name": "Digboi",       "company": "IOCL",        "cap": 0.65,  "lat": 27.39, "lng": 95.62, "crude": "Domestic"},
    {"id": "guwahati",     "name": "Guwahati",     "company": "IOCL",        "cap": 1.0,   "lat": 26.14, "lng": 91.74, "crude": "Domestic"},
    {"id": "mumbai_bpcl",  "name": "Mumbai BPCL",  "company": "BPCL",        "cap": 12.0,  "lat": 19.01, "lng": 72.85, "crude": "Mixed"},
    {"id": "kochi",        "name": "Kochi",        "company": "BPCL",        "cap": 15.5,  "lat": 9.93,  "lng": 76.26, "crude": "Gulf"},
    {"id": "bina",         "name": "Bina",         "company": "BPCL",        "cap": 7.8,   "lat": 23.99, "lng": 78.17, "crude": "Mixed"},
    {"id": "vizag",        "name": "Vizag",        "company": "HPCL",        "cap": 8.3,   "lat": 17.69, "lng": 83.22, "crude": "Import"},
    {"id": "mumbai_hpcl",  "name": "Mumbai HPCL",  "company": "HPCL",        "cap": 6.5,   "lat": 19.05, "lng": 72.88, "crude": "Domestic+Import"},
    {"id": "bathinda",     "name": "Bathinda",     "company": "HPCL-Mittal", "cap": 9.0,   "lat": 30.21, "lng": 74.95, "crude": "Mixed"},
    {"id": "chennai",      "name": "Chennai",      "company": "CPCL",        "cap": 10.5,  "lat": 13.07, "lng": 80.27, "crude": "Import"},
    {"id": "numaligarh",   "name": "Numaligarh",   "company": "NRL",         "cap": 3.0,   "lat": 26.66, "lng": 93.68, "crude": "Domestic"},
    {"id": "tatipaka",     "name": "Tatipaka",     "company": "ONGC",        "cap": 0.083, "lat": 16.81, "lng": 81.73, "crude": "Domestic"},
]

# Import ports — crude entry points (capacity = crude throughput MMTPA)
PORTS = [
    {"id": "mundra",         "name": "Mundra Port",   "cap": 70.0, "lat": 22.84, "lng": 69.72, "corridor": "hormuz"},
    {"id": "jnpt",           "name": "JNPT Mumbai",   "cap": 20.0, "lat": 18.95, "lng": 72.95, "corridor": "hormuz"},
    {"id": "kandla",         "name": "Kandla Port",   "cap": 30.0, "lat": 23.00, "lng": 70.22, "corridor": "hormuz"},
    {"id": "kochi_port",     "name": "Kochi Port",    "cap": 15.0, "lat": 9.96,  "lng": 76.24, "corridor": "red_sea"},
    {"id": "mangalore_port", "name": "Mangalore Port","cap": 12.0, "lat": 12.87, "lng": 74.84, "corridor": "hormuz"},
    {"id": "vizag_port",     "name": "Vizag Port",    "cap": 20.0, "lat": 17.69, "lng": 83.28, "corridor": "red_sea"},
    {"id": "chennai_port",   "name": "Chennai Port",  "cap": 18.0, "lat": 13.09, "lng": 80.29, "corridor": "red_sea"},
    {"id": "paradip_port",   "name": "Paradip Port",  "cap": 25.0, "lat": 20.25, "lng": 86.68, "corridor": "red_sea"},
]

# Domestic wellheads — indigenous crude (production mbd → MMTPA)
WELLHEADS = [
    {"id": "mumbai_high",     "name": "Mumbai High",     "cap": 0.28 * MBD_TO_MMTPA, "lat": 19.10, "lng": 71.50},
    {"id": "kg_basin",        "name": "KG Basin",        "cap": 0.08 * MBD_TO_MMTPA, "lat": 16.00, "lng": 82.00},
    {"id": "rajasthan",       "name": "Rajasthan Fields","cap": 0.17 * MBD_TO_MMTPA, "lat": 27.20, "lng": 71.50},
    {"id": "assam",           "name": "Assam Fields",    "cap": 0.07 * MBD_TO_MMTPA, "lat": 27.10, "lng": 94.60},
    {"id": "gujarat_onshore", "name": "Gujarat Onshore", "cap": 0.05 * MBD_TO_MMTPA, "lat": 22.30, "lng": 72.50},
]

# Strategic reserves — emergency crude buffer (MMT stored, days of national cover)
SPR = [
    {"id": "spr_vizag",     "name": "Vizag SPR",     "mmt": 1.33, "days": 2.4, "lat": 17.72, "lng": 83.30},
    {"id": "spr_mangalore", "name": "Mangalore SPR", "mmt": 1.5,  "days": 2.7, "lat": 12.92, "lng": 74.82},
    {"id": "spr_padur",     "name": "Padur SPR",     "mmt": 2.5,  "days": 4.4, "lat": 13.10, "lng": 75.02},
]

# Demand centres — raw_weight is relative consumption; absolute demand is scaled
# at graph-build time so total demand ≈ 90% of national refining capacity, giving
# a realistically *tight* network where a major outage actually causes shortfall.
DEMAND = [
    {"id": "delhi",      "name": "Delhi NCR",    "raw": 0.42, "lat": 28.61, "lng": 77.21, "pop_m": 32},
    {"id": "mumbai_d",   "name": "Mumbai Metro", "raw": 0.28, "lat": 19.08, "lng": 72.88, "pop_m": 21},
    {"id": "chennai_d",  "name": "Chennai",      "raw": 0.18, "lat": 13.08, "lng": 80.27, "pop_m": 11},
    {"id": "bengaluru",  "name": "Bengaluru",    "raw": 0.15, "lat": 12.97, "lng": 77.59, "pop_m": 13},
    {"id": "kolkata",    "name": "Kolkata",      "raw": 0.14, "lat": 22.57, "lng": 88.36, "pop_m": 15},
    {"id": "hyderabad",  "name": "Hyderabad",    "raw": 0.12, "lat": 17.38, "lng": 78.49, "pop_m": 10},
]

# CRUDE edges: (supply_node, refinery) — which ports/wells can feed which refineries
CRUDE_EDGES = [
    ("mundra", "jamnagar"), ("mundra", "jamnagar_sez"), ("mundra", "panipat"), ("mundra", "bathinda"),
    ("kandla", "bathinda"), ("kandla", "panipat"), ("kandla", "mathura"),
    ("jnpt", "mumbai_bpcl"), ("jnpt", "mumbai_hpcl"), ("jnpt", "jamnagar"),
    ("kochi_port", "kochi"), ("mangalore_port", "kochi"),
    ("vizag_port", "vizag"), ("chennai_port", "chennai"),
    ("paradip_port", "paradip"), ("paradip_port", "haldia"),
    # domestic wellheads
    ("mumbai_high", "jamnagar"), ("mumbai_high", "mumbai_bpcl"),
    ("kg_basin", "vizag"), ("kg_basin", "tatipaka"),
    ("rajasthan", "panipat"), ("rajasthan", "bathinda"),
    ("assam", "numaligarh"), ("assam", "digboi"), ("assam", "guwahati"), ("assam", "barauni"),
    ("gujarat_onshore", "jamnagar"), ("gujarat_onshore", "bina"),
]

# PRODUCT edges: (refinery, demand_centre) — which refineries serve which cities
PRODUCT_EDGES = [
    ("jamnagar", "delhi"), ("jamnagar", "mumbai_d"), ("jamnagar_sez", "delhi"), ("jamnagar_sez", "mumbai_d"),
    ("panipat", "delhi"), ("mathura", "delhi"), ("bina", "delhi"), ("bathinda", "delhi"),
    ("mumbai_bpcl", "mumbai_d"), ("mumbai_hpcl", "mumbai_d"),
    ("kochi", "bengaluru"), ("kochi", "chennai_d"),
    ("chennai", "chennai_d"), ("chennai", "bengaluru"),
    ("vizag", "hyderabad"), ("paradip", "hyderabad"), ("paradip", "kolkata"),
    ("haldia", "kolkata"), ("barauni", "kolkata"), ("numaligarh", "kolkata"),
    ("digboi", "kolkata"), ("guwahati", "kolkata"), ("tatipaka", "hyderabad"),
    # redundancy links — give each city multiple suppliers so the baseline
    # network is fully served and rerouting has alternate paths after an outage
    ("panipat", "mumbai_d"), ("bina", "mumbai_d"), ("bina", "delhi"),
    ("vizag", "chennai_d"), ("chennai", "hyderabad"), ("kochi", "hyderabad"),
    ("mumbai_bpcl", "delhi"), ("mathura", "mumbai_d"),
    # long-haul coastal product movement (Reliance Jamnagar ships nationwide) —
    # lets the surplus NW capacity reach the supply-short south/east
    ("jamnagar", "bengaluru"), ("jamnagar", "hyderabad"), ("jamnagar_sez", "chennai_d"),
    ("jamnagar_sez", "kolkata"), ("bina", "hyderabad"),
]

# Fast lookups
_ALL_NODES = {n["id"]: n for n in (REFINERIES + PORTS + WELLHEADS + DEMAND)}
_REF_IDS = {r["id"] for r in REFINERIES}
_PORT_IDS = {p["id"] for p in PORTS}
_WELL_IDS = {w["id"] for w in WELLHEADS}
_DEMAND_IDS = {d["id"] for d in DEMAND}

TOTAL_REFINING = round(sum(r["cap"] for r in REFINERIES), 1)
_RAW_DEMAND_SUM = sum(d["raw"] for d in DEMAND)
# Scale demand so total ≈ 72% of national refining capacity: the baseline network
# is comfortably served (healthy start), leaving ~28% headroom so that a major
# outage — losing Jamnagar, or a Hormuz closure — produces a real, visible shortfall.
DEMAND_SCALE = (0.72 * TOTAL_REFINING) / _RAW_DEMAND_SUM
TOTAL_DEMAND = round(sum(d["raw"] * DEMAND_SCALE for d in DEMAND), 1)
TOTAL_SPR_MMT = round(sum(s["mmt"] for s in SPR), 2)
TOTAL_SPR_DAYS = round(sum(s["days"] for s in SPR), 1)


def demand_mmtpa(d):
    return d["raw"] * DEMAND_SCALE


# ─────────────────────────────────────────────────────────────
# CORE SIMULATION
# ─────────────────────────────────────────────────────────────
def _build_flow_graph(offline=None, stress=None, port_stress=None):
    """Build the capacitated max-flow DiGraph. Capacitated nodes are split into
    <id>__in → <id>__out with the throughput on the connecting edge; offline nodes
    get 0 capacity, stressed nodes get proportionally reduced capacity."""
    offline = set(offline or [])
    stress = stress or {}       # {refinery_id: stress_pct 0..100}
    port_stress = port_stress or {}  # {port_id: 0..100}

    G = nx.DiGraph()
    SRC, SINK = "__SOURCE__", "__SINK__"

    def cap_after(node_id, base, stress_map):
        if node_id in offline:
            return 0.0
        frac = max(0.0, 1.0 - stress_map.get(node_id, 0) / 100.0)
        return base * frac

    # Supply: source → port/well (crude availability), split node for its cap
    for p in PORTS:
        c = cap_after(p["id"], p["cap"], port_stress)
        G.add_edge(SRC, p["id"] + "__in", capacity=c)
        G.add_edge(p["id"] + "__in", p["id"] + "__out", capacity=c)
    for w in WELLHEADS:
        c = cap_after(w["id"], w["cap"], {})
        G.add_edge(SRC, w["id"] + "__in", capacity=c)
        G.add_edge(w["id"] + "__in", w["id"] + "__out", capacity=c)

    # Refineries: split node, edge cap = throughput (crude in → product out)
    for r in REFINERIES:
        c = cap_after(r["id"], r["cap"], stress)
        G.add_edge(r["id"] + "__in", r["id"] + "__out", capacity=c)

    # Crude edges: supply_out → refinery_in
    for s, r in CRUDE_EDGES:
        if s in _ALL_NODES and r in _REF_IDS:
            # generous pipeline cap — refinery throughput is the binding constraint
            G.add_edge(s + "__out", r + "__in", capacity=1e6)

    # Product edges: refinery_out → demand_in
    for r, d in PRODUCT_EDGES:
        if r in _REF_IDS and d in _DEMAND_IDS:
            G.add_edge(r + "__out", d + "__in", capacity=1e6)

    # Demand: demand_in → demand_out → sink (cap = demand)
    for d in DEMAND:
        dm = demand_mmtpa(d)
        G.add_edge(d["id"] + "__in", d["id"] + "__out", capacity=dm)
        G.add_edge(d["id"] + "__out", SINK, capacity=dm)

    return G, SRC, SINK


def simulate(offline=None, stress=None, port_stress=None):
    """Run the flow simulation. Returns per-demand satisfaction, unmet demand,
    network health, and the binding bottleneck. All numbers are exact max-flow
    results, not heuristics."""
    G, SRC, SINK = _build_flow_graph(offline, stress, port_stress)
    flow_value, flow_dict = nx.maximum_flow(G, SRC, SINK)

    demand_status = []
    total_unmet = 0.0
    for d in DEMAND:
        want = demand_mmtpa(d)
        got = flow_dict.get(d["id"] + "__out", {}).get(SINK, 0.0)
        unmet = round(max(0.0, want - got), 2)
        total_unmet += unmet
        demand_status.append({
            "id": d["id"], "name": d["name"], "pop_m": d["pop_m"],
            "demand_mmtpa": round(want, 1), "supplied_mmtpa": round(got, 1),
            "unmet_mmtpa": unmet,
            "coverage_pct": round(got / want * 100, 1) if want else 100.0,
            "status": "UNMET" if unmet > 0.5 else "TIGHT" if got < want * 0.98 else "OK",
        })

    total_demand = round(sum(demand_mmtpa(d) for d in DEMAND), 1)
    served_pct = round(flow_value / total_demand * 100, 1) if total_demand else 100.0
    shortfall = round(total_demand - flow_value, 1)

    # SPR can bridge a shortfall for however many days its crude lasts at that rate
    spr_days_at_shortfall = round(TOTAL_SPR_MMT / (shortfall / 365), 1) if shortfall > 0.1 else None

    n_unmet = sum(1 for d in demand_status if d["status"] == "UNMET")
    n_tight = sum(1 for d in demand_status if d["status"] == "TIGHT")
    if n_unmet >= 2 or served_pct < 85:
        health = "CRITICAL"
    elif n_unmet >= 1 or n_tight >= 2 or served_pct < 96:
        health = "ELEVATED"
    else:
        health = "NORMAL"

    return {
        "national_demand_mmtpa": total_demand,
        "national_supplied_mmtpa": round(flow_value, 1),
        "national_shortfall_mmtpa": shortfall,
        "served_pct": served_pct,
        "network_health": health,
        "demand_centres": sorted(demand_status, key=lambda x: -x["unmet_mmtpa"]),
        "offline_nodes": sorted(offline or []),
        "stressed_nodes": {k: v for k, v in (stress or {}).items() if v},
        "spr_bridge_days_at_current_shortfall": spr_days_at_shortfall,
        "total_spr_mmt": TOTAL_SPR_MMT,
    }


def node_criticality(top_n=6):
    """Rank refineries by how much national shortfall their loss would cause —
    an N-1 contingency analysis. This is what an operator uses to know which
    assets are single points of failure."""
    base = simulate()
    base_short = base["national_shortfall_mmtpa"]
    rows = []
    for r in REFINERIES:
        sim = simulate(offline=[r["id"]])
        delta = round(sim["national_shortfall_mmtpa"] - base_short, 1)
        rows.append({
            "id": r["id"], "name": r["name"], "company": r["company"],
            "capacity_mmtpa": r["cap"],
            "shortfall_if_lost_mmtpa": max(0.0, delta),
            "cities_hit": [d["name"] for d in sim["demand_centres"] if d["status"] == "UNMET"],
        })
    rows.sort(key=lambda x: -x["shortfall_if_lost_mmtpa"])
    return rows[:top_n]


def corridor_to_offline(scenario, hormuz_score=0, redsea_score=0):
    """Translate a geopolitical corridor scenario into which ports go offline /
    stressed, so the flow sim can be driven directly from Agent-1 risk output."""
    port_stress = {}
    offline = []
    if scenario == "hormuz" or hormuz_score >= 60:
        sev = 100 if scenario == "hormuz" else min(90, hormuz_score)
        for p in PORTS:
            if p["corridor"] == "hormuz":
                port_stress[p["id"]] = sev
    if scenario == "red_sea" or redsea_score >= 60:
        sev = 80 if scenario == "red_sea" else min(80, redsea_score)
        for p in PORTS:
            if p["corridor"] == "red_sea":
                port_stress[p["id"]] = max(port_stress.get(p["id"], 0), sev)
    return offline, port_stress


def all_nodes_geo():
    """Expose node coordinates + roles for the frontend map overlay."""
    def pack(lst, role):
        return [{"id": n["id"], "name": n["name"], "lat": n["lat"], "lng": n["lng"],
                 "role": role, "cap": round(n.get("cap", n.get("mmt", 0)), 2)} for n in lst]
    return {
        "refineries": pack(REFINERIES, "refinery"),
        "ports": pack(PORTS, "port"),
        "wellheads": pack(WELLHEADS, "wellhead"),
        "spr": pack(SPR, "spr"),
        "demand": pack(DEMAND, "demand"),
    }


if __name__ == "__main__":
    import json
    print("=" * 60)
    print("TWIN FLOW SIM — self test")
    print("=" * 60)
    print(f"Total refining: {TOTAL_REFINING} MMTPA | scaled demand: {TOTAL_DEMAND} MMTPA")
    base = simulate()
    print(f"\nBASELINE: served {base['served_pct']}%  health={base['network_health']}  shortfall={base['national_shortfall_mmtpa']}")
    jam = simulate(offline=["jamnagar", "jamnagar_sez"])
    print(f"\nJAMNAGAR COMPLEX OFFLINE: served {jam['served_pct']}%  health={jam['network_health']}  shortfall={jam['national_shortfall_mmtpa']}")
    for d in jam["demand_centres"][:4]:
        print(f"   {d['name']:14} {d['coverage_pct']:5}%  {d['status']}")
    off, ps = corridor_to_offline("hormuz")
    hz = simulate(port_stress=ps)
    print(f"\nHORMUZ CLOSURE (ports stressed {list(ps)}): served {hz['served_pct']}%  health={hz['network_health']}")
    print("\nTOP CRITICAL REFINERIES (N-1):")
    for r in node_criticality(5):
        print(f"   {r['name']:14} loss -> +{r['shortfall_if_lost_mmtpa']} MMTPA shortfall  hits {r['cities_hit']}")
