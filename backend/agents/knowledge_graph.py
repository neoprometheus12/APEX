"""
APEX — Supplier-Route-Risk-Refinery Knowledge Graph
=========================================================
A genuine entity-relationship knowledge graph, distinct in kind from the
Digital Twin's flow-capacity graph (twin_sim.py). The Twin's graph answers
"how many MMTPA can flow through this network" via max-flow; this graph
answers "what is X connected to, and through what kind of relationship" via
typed nodes and typed, labelled edges — the queryable ontology the platform
brief specifically asks for (supplier–route–risk–refinery relationships).

Built with networkx (already a dependency) as a MultiDiGraph, but used here
as a semantic graph, not a capacity-flow graph: edges carry a `relation`
label (TRANSITS_VIA, FEEDS_PORT, SUPPLIES, EXPOSED_TO, DRIVES_RISK_IN) rather
than a numeric capacity, and queries traverse relationships rather than
solving a flow problem.
"""

import networkx as nx
from agents import twin_sim

SUPPLIER_CORRIDOR = {
    "Iraq": "hormuz", "Saudi Arabia": "hormuz", "UAE": "hormuz", "Kuwait": "hormuz",
    "Russia": None, "United States": None, "Guyana": None,
    "Nigeria": "red_sea", "Libya": "red_sea",
}
CORRIDOR_LABELS = {
    "hormuz": "Strait of Hormuz",
    "red_sea": "Red Sea / Bab-el-Mandeb",
    "opec_policy": "OPEC+ Policy",
}
# Same red-sea-exposed refinery set already used in twin_sim / pipeline stress
# logic — the KG is built to agree with the rest of the platform, not invent
# a second, inconsistent notion of exposure.
RED_SEA_REFINERIES = {"Paradip", "Vizag", "Chennai", "Haldia"}
SECTORS = ["fuels", "industry", "transport", "agriculture", "consumer_cpg", "services", "macro"]

_GRAPH = None


def build_graph():
    global _GRAPH
    G = nx.MultiDiGraph()

    for name, corridor in SUPPLIER_CORRIDOR.items():
        G.add_node(f"supplier:{name}", type="supplier", label=name)
    for key, label in CORRIDOR_LABELS.items():
        G.add_node(f"corridor:{key}", type="corridor", label=label)
    for name, corridor in SUPPLIER_CORRIDOR.items():
        if corridor:
            G.add_edge(f"supplier:{name}", f"corridor:{corridor}", relation="TRANSITS_VIA")

    for p in twin_sim.PORTS:
        G.add_node(f"port:{p['id']}", type="port", label=p["name"])
        if p.get("corridor"):
            G.add_edge(f"corridor:{p['corridor']}", f"port:{p['id']}", relation="FEEDS_PORT")

    for w in twin_sim.WELLHEADS:
        G.add_node(f"wellhead:{w['id']}", type="wellhead", label=w["name"])

    for r in twin_sim.REFINERIES:
        G.add_node(f"refinery:{r['id']}", type="refinery", label=r["name"],
                   company=r["company"], mmtpa=r["cap"])

    port_ids = {p["id"] for p in twin_sim.PORTS}
    for src, dst in twin_sim.CRUDE_EDGES:
        src_node = f"port:{src}" if src in port_ids else f"wellhead:{src}"
        dst_node = f"refinery:{dst}"
        if G.has_node(src_node) and G.has_node(dst_node):
            G.add_edge(src_node, dst_node, relation="SUPPLIES")

    for r in twin_sim.REFINERIES:
        if "Gulf" in r["crude"]:
            G.add_edge(f"refinery:{r['id']}", "corridor:hormuz", relation="EXPOSED_TO")
        if r["name"] in RED_SEA_REFINERIES:
            G.add_edge(f"refinery:{r['id']}", "corridor:red_sea", relation="EXPOSED_TO")

    for s in SECTORS:
        G.add_node(f"sector:{s}", type="sector", label=s.replace("_", " ").title())
        for c in CORRIDOR_LABELS:
            G.add_edge(f"corridor:{c}", f"sector:{s}", relation="DRIVES_RISK_IN")

    _GRAPH = G
    return G


def _graph():
    return _GRAPH or build_graph()


def _find_node(name: str):
    """Case-insensitive label lookup so agents can query by natural-language name."""
    G = _graph()
    name_lo = name.strip().lower()
    for n, data in G.nodes(data=True):
        if data.get("label", "").lower() == name_lo:
            return n
    for n, data in G.nodes(data=True):
        if name_lo in data.get("label", "").lower():
            return n
    return None


def stats() -> dict:
    G = _graph()
    by_type = {}
    for _, data in G.nodes(data=True):
        by_type[data["type"]] = by_type.get(data["type"], 0) + 1
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "node_types": by_type}


def get_suppliers_via(corridor_key: str) -> list:
    """All suppliers whose crude transits the given corridor."""
    G = _graph()
    target = f"corridor:{corridor_key}"
    if not G.has_node(target):
        return []
    return [G.nodes[u]["label"] for u, v, d in G.in_edges(target, data=True) if d["relation"] == "TRANSITS_VIA"]


def get_refineries_exposed_to(corridor_key: str) -> list:
    """All refineries directly exposed to a given corridor's risk."""
    G = _graph()
    target = f"corridor:{corridor_key}"
    if not G.has_node(target):
        return []
    return [{"name": G.nodes[u]["label"], "company": G.nodes[u]["company"], "mmtpa": G.nodes[u]["mmtpa"]}
            for u, v, d in G.in_edges(target, data=True) if d["relation"] == "EXPOSED_TO"]


def get_exposure_chain(supplier_name: str) -> dict:
    """Trace a supplier's full exposure chain: supplier -> corridor -> ports fed
    by that corridor -> refineries supplied by those ports -> sectors the
    corridor drives risk in. This is the direct 'supplier-route-risk-refinery'
    relationship chain the platform is meant to expose."""
    G = _graph()
    node = _find_node(supplier_name)
    if not node or G.nodes[node]["type"] != "supplier":
        return {"error": f"no supplier found matching '{supplier_name}'"}

    corridors = [v for u, v, d in G.out_edges(node, data=True) if d["relation"] == "TRANSITS_VIA"]
    if not corridors:
        return {"supplier": G.nodes[node]["label"], "corridor": None,
                "note": "This supplier does not transit a monitored chokepoint corridor."}

    corridor = corridors[0]
    ports = [v for u, v, d in G.out_edges(corridor, data=True) if d["relation"] == "FEEDS_PORT"]
    refineries = set()
    for p in ports:
        for u, v, d in G.out_edges(p, data=True):
            if d["relation"] == "SUPPLIES":
                refineries.add(v)
    sectors = [v for u, v, d in G.out_edges(corridor, data=True) if d["relation"] == "DRIVES_RISK_IN"]

    return {
        "supplier": G.nodes[node]["label"],
        "corridor": G.nodes[corridor]["label"],
        "ports": [G.nodes[p]["label"] for p in ports],
        "refineries_reachable": sorted(G.nodes[r]["label"] for r in refineries),
        "sectors_at_risk": [G.nodes[s]["label"] for s in sectors],
    }


def explain_path(entity_a: str, entity_b: str) -> dict:
    """Find and narrate the relationship path between any two named entities
    in the graph — e.g. 'Jamnagar' and 'Iraq' — so an agent can answer
    'why is X connected to Y' with an actual traced path, not a guess."""
    G = _graph()
    na, nb = _find_node(entity_a), _find_node(entity_b)
    if not na or not nb:
        missing = entity_a if not na else entity_b
        return {"error": f"no entity found matching '{missing}'"}
    try:
        path = nx.shortest_path(G.to_undirected(), na, nb)
    except nx.NetworkXNoPath:
        return {"entity_a": G.nodes[na]["label"], "entity_b": G.nodes[nb]["label"],
                "connected": False, "note": "No relationship path found between these entities."}

    steps = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        rel = None
        if G.has_edge(u, v):
            rel = list(G.get_edge_data(u, v).values())[0]["relation"]
        elif G.has_edge(v, u):
            rel = list(G.get_edge_data(v, u).values())[0]["relation"] + " (reverse)"
        steps.append(f"{G.nodes[u]['label']} --[{rel}]--> {G.nodes[v]['label']}")

    return {"entity_a": G.nodes[na]["label"], "entity_b": G.nodes[nb]["label"],
            "connected": True, "hops": len(path) - 1, "path": steps}


def node_neighbors(entity: str) -> dict:
    """All direct relationships (incoming and outgoing) for a named entity."""
    G = _graph()
    n = _find_node(entity)
    if not n:
        return {"error": f"no entity found matching '{entity}'"}
    out = [{"to": G.nodes[v]["label"], "relation": d["relation"]} for u, v, d in G.out_edges(n, data=True)]
    inn = [{"from": G.nodes[u]["label"], "relation": d["relation"]} for u, v, d in G.in_edges(n, data=True)]
    return {"entity": G.nodes[n]["label"], "type": G.nodes[n]["type"], "outgoing": out, "incoming": inn}


if __name__ == "__main__":
    build_graph()
    print("Graph stats:", stats())
    print("\nSuppliers via Hormuz:", get_suppliers_via("hormuz"))
    print("\nRefineries exposed to Hormuz:", [r["name"] for r in get_refineries_exposed_to("hormuz")])
    print("\nExposure chain for Iraq:")
    import json
    print(json.dumps(get_exposure_chain("Iraq"), indent=2))
    print("\nExplain path: Jamnagar <-> Saudi Arabia")
    r = explain_path("Jamnagar", "Saudi Arabia")
    print(json.dumps(r, indent=2))
