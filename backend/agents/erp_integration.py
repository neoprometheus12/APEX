"""
APEX — ERP Integration Layer
===============================
Lets an external ERP (SAP or otherwise) call APEX's Business Impact Advisor
directly over HTTP, instead of a person re-typing data that already lives in
their ERP into the web wizard. This module does the one thing a generic
integration actually needs: translate whatever material names/codes the
ERP sends into APEX's internal 31-item taxonomy (business_impact.py's
BUSINESS_MATERIALS), and assemble the exact profile shape the existing,
already-live Business Advisor Agent pipeline expects.

Deliberately does NOT reimplement the financial model or bypass the agent —
an ERP-submitted request gets exactly the same live agentic computation
(agents/business_advisor.py's analyze_business) as a request typed into the
wizard by hand. This module is purely an input adapter.
"""
import difflib
import re
from agents.business_impact import BUSINESS_MATERIALS, MATERIAL_BY_KEY

# ─────────────────────────────────────────────────────────────
# MATERIAL ALIASES
# Common ERP / SAP material-master terminology an integrator's system is
# realistically likely to send, mapped onto APEX's internal material key.
# SAP material master data (MM01/MM02) typically carries a short description
# and a material group (Warengruppe) — both are matched against here, so an
# integrator can send either without pre-building their own mapping table.
# ─────────────────────────────────────────────────────────────
MATERIAL_ALIASES = {
    "petrol":               ["petrol", "motor spirit", "gasoline"],
    "diesel":                ["diesel", "hsd", "high speed diesel"],
    "lpg":                   ["lpg", "liquefied petroleum gas", "cooking gas"],
    "aviation_fuel_atf":     ["atf", "aviation turbine fuel", "jet fuel", "jet a1", "jet a-1"],
    "kerosene":              ["kerosene", "sko", "superior kerosene oil"],
    "naphtha":               ["naphtha"],
    "plastics_pvc":          ["pvc", "pvc resin", "polyvinyl chloride", "plastic resin", "plastics"],
    "fertiliser_urea":       ["urea", "urea fertiliser", "urea fertilizer", "n-fertiliser"],
    "steel":                  ["steel", "ms steel", "mild steel", "tmt bar", "tmt", "hr coil", "cr coil"],
    "cement":                 ["cement", "opc", "ppc"],
    "chemicals":              ["chemicals", "industrial chemicals", "specialty chemicals"],
    "synthetic_rubber":       ["synthetic rubber", "rubber", "sbr", "nbr"],
    "paints_coatings":        ["paints", "coatings", "paints & coatings", "paint"],
    "aluminium":              ["aluminium", "aluminum", "al ingot"],
    "road_freight":           ["road freight", "trucking", "road transport", "gta", "lorry freight"],
    "aviation":               ["air freight", "air cargo", "airfreight"],
    "railways":               ["rail freight", "railways", "rail transport"],
    "coastal_shipping":       ["coastal shipping", "sea freight", "ocean freight", "coastal/sea freight"],
    "last_mile_ecommerce":    ["last mile", "last-mile delivery", "last mile delivery"],
    "cold_chain_logistics":   ["cold chain", "cold-chain logistics", "reefer"],
    "irrigation_diesel":      ["irrigation diesel", "farm diesel"],
    "pesticides":             ["pesticides", "agrochemicals", "crop protection"],
    "tractor_fuel":           ["tractor fuel", "tractor diesel"],
    "crop_cold_storage":      ["crop cold storage", "cold storage"],
    "fmcg_packaging":         ["fmcg packaging", "packaging", "pkg material"],
    "edible_oil":             ["edible oil", "cooking oil", "vegetable oil"],
    "dairy_products":         ["dairy", "dairy inputs", "milk inputs"],
    "synthetic_textiles":     ["synthetic textiles", "textiles", "polyester fabric"],
    "pharma":                 ["pharma", "api", "excipients", "pharma inputs"],
    "personal_care":          ["personal care", "personal-care inputs"],
    "pet_bottles":            ["pet bottles", "pet packaging", "pet preform", "pet"],
}

# A plausible SAP Material Group (Warengruppe) label per material — published
# so an integrator building a one-time mapping table on the SAP side has a
# concrete, familiar reference point rather than guessing at APEX's own keys.
SAP_MATERIAL_GROUP_HINTS = {
    "petrol": "ROH-FUEL-01", "diesel": "ROH-FUEL-02", "lpg": "ROH-FUEL-03",
    "aviation_fuel_atf": "ROH-FUEL-04", "kerosene": "ROH-FUEL-05", "naphtha": "ROH-FUEL-06",
    "plastics_pvc": "ROH-POLY-01", "fertiliser_urea": "ROH-AGRI-01", "steel": "ROH-METAL-01",
    "cement": "ROH-CONST-01", "chemicals": "ROH-CHEM-01", "synthetic_rubber": "ROH-POLY-02",
    "paints_coatings": "ROH-CHEM-02", "aluminium": "ROH-METAL-02",
    "road_freight": "DIEN-LOG-01", "aviation": "DIEN-LOG-02", "railways": "DIEN-LOG-03",
    "coastal_shipping": "DIEN-LOG-04", "last_mile_ecommerce": "DIEN-LOG-05",
    "cold_chain_logistics": "DIEN-LOG-06", "irrigation_diesel": "ROH-AGRI-02",
    "pesticides": "ROH-AGRI-03", "tractor_fuel": "ROH-AGRI-04", "crop_cold_storage": "DIEN-AGRI-01",
    "fmcg_packaging": "VERP-01", "edible_oil": "ROH-FOOD-01", "dairy_products": "ROH-FOOD-02",
    "synthetic_textiles": "ROH-TEX-01", "pharma": "ROH-PHARM-01", "personal_care": "ROH-CPG-01",
    "pet_bottles": "VERP-02",
}

_ALIAS_LOOKUP = {}
for _key, _aliases in MATERIAL_ALIASES.items():
    for _a in _aliases:
        _ALIAS_LOOKUP[_a] = _key
for _k, _label, _s, _f, _u in BUSINESS_MATERIALS:
    _ALIAS_LOOKUP.setdefault(_label.lower(), _k)
    _ALIAS_LOOKUP.setdefault(_k.replace("_", " "), _k)


def match_material(text: str):
    """Resolves free-text ERP material identifiers (name, description, or a
    string that happens to already be an APEX key) to an internal material
    key. Tries, in order: exact key, exact alias, substring containment
    (either direction), then a fuzzy close-match as a last resort — always
    returns a confidence level so a low-confidence guess is never silently
    treated as certain."""
    if not text:
        return None
    q = text.strip().lower()

    if q in MATERIAL_BY_KEY:
        return {"key": q, "confidence": "exact", "matched_on": q}
    if q in _ALIAS_LOOKUP:
        return {"key": _ALIAS_LOOKUP[q], "confidence": "exact", "matched_on": q}

    # Whole-word containment only, and only for aliases of a meaningful
    # length — a short, unqualified alias substring-matching against
    # arbitrary text is how "al" (aluminium) previously matched inside
    # "materi-AL" and similar false positives.
    for alias, key in _ALIAS_LOOKUP.items():
        if len(alias) < 4:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", q) or re.search(rf"\b{re.escape(q)}\b", alias):
            return {"key": key, "confidence": "partial", "matched_on": alias}

    close = difflib.get_close_matches(q, _ALIAS_LOOKUP.keys(), n=1, cutoff=0.72)
    if close:
        return {"key": _ALIAS_LOOKUP[close[0]], "confidence": "fuzzy", "matched_on": close[0]}

    return None


def materials_reference():
    """The full taxonomy plus ERP-facing aliases and a SAP material-group
    hint per item — published so an integrator can build a one-time mapping
    table on their side instead of relying on runtime fuzzy-matching."""
    return [
        {
            "key": k, "label": label, "sector": sector, "unit_hint": unit,
            "aliases": MATERIAL_ALIASES.get(k, []),
            "sap_material_group_hint": SAP_MATERIAL_GROUP_HINTS.get(k),
        }
        for k, label, sector, _field, unit in BUSINESS_MATERIALS
    ]


def build_profile_from_erp_payload(payload: dict):
    """Maps an ERP-shaped request body onto the exact profile dict shape
    agents/business_impact.py's calculate_impact() and business_advisor.py's
    analyze_business() already expect. Returns (profile, unmapped, mapping_report) —
    unmapped materials are never silently dropped; the caller decides whether
    to proceed with a partial profile or reject the request."""
    unmapped = []
    mapping_report = []
    materials_out = []

    for m in payload.get("materials", []):
        raw_name = m.get("material") or m.get("material_name") or m.get("name") or m.get("key") or ""
        match = match_material(raw_name)
        if not match:
            unmapped.append({"submitted": raw_name})
            continue
        mapping_report.append({"submitted": raw_name, "resolved_key": match["key"],
                                "resolved_label": MATERIAL_BY_KEY[match["key"]][1],
                                "confidence": match["confidence"]})
        materials_out.append({
            "key": match["key"],
            "monthly_volume": m.get("monthly_quantity") or m.get("monthly_volume") or 0,
            "unit_cost": m.get("unit_cost") or m.get("price_per_unit") or 0,
            "currency": (m.get("currency") or "INR").upper(),
            "contract_type": (m.get("contract_type") or "spot").lower(),
            "contract_months_remaining": m.get("contract_months_remaining") or 0,
        })

    product = payload.get("product") or {}
    logistics = payload.get("logistics") or {}

    profile = {
        "business_name": payload.get("company_name") or payload.get("business_name") or "ERP-submitted business",
        "sector": payload.get("sector") or "industry",
        "materials": materials_out,
        "logistics": {
            "mode": (logistics.get("mode") or "road").lower(),
            "monthly_cost": logistics.get("monthly_cost") or 0,
        },
        "product": {
            "name": product.get("name") or payload.get("product_name") or "Product",
            "current_price": product.get("unit_price") or product.get("current_price") or 0,
            "monthly_units_sold": product.get("monthly_units") or product.get("monthly_units_sold") or 0,
            "pass_through_capacity_pct": product.get("pass_through_pct") or product.get("pass_through_capacity_pct") or 40,
        },
        "inventory": {"buffer_days": payload.get("inventory_buffer_days") or 15},
    }
    return profile, unmapped, mapping_report
