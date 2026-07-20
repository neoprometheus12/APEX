"""
APEX — Business Impact Calculator
======================================
Deterministic engine that translates APEX's existing macro cascade
projection (from the trained ML+econometric Scenario model) into a specific
business's rupee impact: input cost delta, logistics cost delta, currency
risk, margin compression, and the price increase needed to stay whole.

This is the same design principle as the Digital Twin's flow simulation: the
agent that talks to the user reasons and recommends, but the arithmetic lives
here, deterministic and auditable, never invented by an LLM.
"""

from agents import market_data

# ─────────────────────────────────────────────────────────────
# MATERIAL TAXONOMY — mirrors the Economic Cascade Agent's own sector/commodity
# model exactly (pipeline.py's cascade_agent_node), so every % shock used here
# is the same number already shown on the Cascade page — never a second,
# inconsistent estimate.
# ─────────────────────────────────────────────────────────────
BUSINESS_MATERIALS = [
    # key, label, sector, cascade_field, suggested unit
    ("petrol",             "Petrol",                        "fuels",        "petrol",              "litres"),
    ("diesel",              "Diesel",                        "fuels",        "diesel",              "litres"),
    ("lpg",                 "LPG",                           "fuels",        "lpg",                 "kg"),
    ("aviation_fuel_atf",   "Aviation Turbine Fuel (ATF)",    "fuels",        "aviation_fuel_atf",   "litres"),
    ("kerosene",            "Kerosene",                       "fuels",        "kerosene",            "litres"),
    ("naphtha",             "Naphtha",                        "fuels",        "naphtha",             "MT"),
    ("plastics_pvc",        "Plastics / PVC resin",           "industry",     "plastics_pvc",        "kg"),
    ("fertiliser_urea",     "Urea fertiliser",                "industry",     "fertiliser_urea",     "MT"),
    ("steel",               "Steel",                          "industry",     "steel",               "MT"),
    ("cement",               "Cement",                         "industry",     "cement",              "MT"),
    ("chemicals",            "Industrial chemicals",           "industry",     "chemicals",           "kg"),
    ("synthetic_rubber",     "Synthetic rubber",               "industry",     "synthetic_rubber",    "kg"),
    ("paints_coatings",      "Paints & coatings",              "industry",     "paints_coatings",     "litres"),
    ("aluminium",            "Aluminium",                      "industry",     "aluminium",           "kg"),
    ("road_freight",         "Road freight (logistics)",       "transport",    "road_freight",        "trip"),
    ("aviation",             "Air freight (logistics)",        "transport",    "aviation",            "kg"),
    ("railways",             "Rail freight (logistics)",       "transport",    "railways",            "tonne-km"),
    ("coastal_shipping",     "Coastal/sea freight (logistics)","transport",    "coastal_shipping",    "TEU"),
    ("last_mile_ecommerce",  "Last-mile delivery",             "transport",    "last_mile_ecommerce", "delivery"),
    ("cold_chain_logistics", "Cold-chain logistics",           "transport",    "cold_chain_logistics","pallet"),
    ("irrigation_diesel",    "Irrigation diesel",              "agriculture",  "irrigation_diesel",   "litres"),
    ("pesticides",           "Pesticides",                     "agriculture",  "pesticides",          "litres"),
    ("tractor_fuel",         "Tractor fuel",                   "agriculture",  "tractor_fuel",        "litres"),
    ("crop_cold_storage",    "Crop cold storage",              "agriculture",  "crop_cold_storage",   "MT/month"),
    ("fmcg_packaging",       "FMCG packaging",                 "consumer_cpg", "fmcg_packaging",      "unit"),
    ("edible_oil",           "Edible oil",                     "consumer_cpg", "edible_oil",          "litres"),
    ("dairy_products",       "Dairy inputs",                   "consumer_cpg", "dairy_products",      "litres"),
    ("synthetic_textiles",   "Synthetic textiles",             "consumer_cpg", "synthetic_textiles",  "metre"),
    ("pharma",               "Pharma inputs (API/excipients)", "consumer_cpg", "pharma",              "kg"),
    ("personal_care",        "Personal-care inputs",           "consumer_cpg", "personal_care",       "unit"),
    ("pet_bottles",          "PET bottles/packaging",          "consumer_cpg", "pet_bottles",         "unit"),
]
MATERIAL_BY_KEY = {m[0]: m for m in BUSINESS_MATERIALS}

LOGISTICS_MODE_TO_FIELD = {
    "road": "road_freight", "air": "aviation", "rail": "railways",
    "sea": "coastal_shipping", "last_mile": "last_mile_ecommerce",
}


def list_materials():
    return [{"key": k, "label": l, "sector": s, "unit_hint": u,
             "has_live_reference": k in market_data.COMMODITY_FRED_SERIES}
            for k, l, s, _f, u in BUSINESS_MATERIALS]


def _sector_shock_pct(cascade_sectors, sector, field):
    try:
        return float(cascade_sectors[sector][field]["change_pct"])
    except (KeyError, TypeError):
        return 0.0


def calculate_impact(profile: dict, cascade_sectors: dict, fx: dict) -> dict:
    """
    profile = {
      "business_name": str, "sector": str,
      "materials": [{"key","monthly_volume","unit_cost","currency"("INR"/"USD"),
                     "contract_type"("spot"/"fixed"), "contract_months_remaining"}],
      "logistics": {"mode", "monthly_cost", "currency", "route"},
      "product": {"name","current_price","monthly_units_sold","currency",
                  "pass_through_capacity_pct"},
      "inventory": {"buffer_days": int}
    }
    cascade_sectors = the live cascade_data["sectors"] dict (day-30 % shocks per commodity)
    fx = market_data.get_fx_rate() result
    """
    fx_rate, fx_drift = fx["rate"], fx["drift_pct_30d"]

    materials_out = []
    total_cost_now_inr = 0.0
    total_cost_new_inr = 0.0
    total_delta_inr = 0.0
    deferred_delta_inr = 0.0  # cost increase locked out by a fixed contract, for now

    for m in profile.get("materials", []):
        key = m.get("key")
        info = MATERIAL_BY_KEY.get(key)
        if not info:
            continue
        _k, label, sector, field, unit = info
        vol = float(m.get("monthly_volume") or 0)
        unit_cost = float(m.get("unit_cost") or 0)
        currency = (m.get("currency") or "INR").upper()
        contract = (m.get("contract_type") or "spot").lower()
        months_left = int(m.get("contract_months_remaining") or 0)

        cost_now_native = vol * unit_cost
        cost_now_inr = cost_now_native * (fx_rate if currency == "USD" else 1.0)

        shock_pct = _sector_shock_pct(cascade_sectors, sector, field)
        # USD-denominated inputs also carry currency drift on top of the commodity shock
        currency_pct = fx_drift if currency == "USD" else 0.0
        combined_pct = round((1 + shock_pct / 100) * (1 + currency_pct / 100) * 100 - 100, 2)

        cost_new_native = vol * unit_cost * (1 + shock_pct / 100)
        cost_new_inr = cost_new_native * (fx_rate if currency == "USD" else 1.0)
        delta_inr = round(cost_new_inr - cost_now_inr, 2)

        locked = contract == "fixed" and months_left > 0
        ref = market_data.get_commodity_reference(key)

        materials_out.append({
            "key": key, "label": label, "sector": sector, "unit": unit,
            "monthly_volume": vol, "unit_cost": unit_cost, "currency": currency,
            "cost_now_inr": round(cost_now_inr, 2),
            "cascade_shock_pct": shock_pct, "currency_drift_pct": currency_pct,
            "combined_shock_pct": combined_pct,
            "cost_new_inr": round(cost_new_inr, 2) if not locked else round(cost_now_inr, 2),
            "delta_inr": 0.0 if locked else delta_inr,
            "deferred_delta_inr": delta_inr if locked else 0.0,
            "contract_locked": locked, "contract_months_remaining": months_left,
            "live_reference": ref,
        })

        total_cost_now_inr += cost_now_inr
        if locked:
            deferred_delta_inr += delta_inr
            total_cost_new_inr += cost_now_inr
        else:
            total_cost_new_inr += cost_new_inr
            total_delta_inr += delta_inr

    materials_out.sort(key=lambda x: -(x["delta_inr"] + x["deferred_delta_inr"]))

    # ── Logistics ──
    log = profile.get("logistics", {}) or {}
    mode = (log.get("mode") or "road").lower()
    log_field = LOGISTICS_MODE_TO_FIELD.get(mode, "road_freight")
    log_cost_now = float(log.get("monthly_cost") or 0)
    log_shock_pct = _sector_shock_pct(cascade_sectors, "transport", log_field)
    log_cost_new = round(log_cost_now * (1 + log_shock_pct / 100), 2)
    log_delta = round(log_cost_new - log_cost_now, 2)

    total_cost_now_inr += log_cost_now
    total_cost_new_inr += log_cost_new
    total_delta_inr += log_delta

    # ── Product economics ──
    prod = profile.get("product", {}) or {}
    price = float(prod.get("current_price") or 0)
    units = float(prod.get("monthly_units_sold") or 0)
    pass_through_pct = max(0.0, min(100.0, float(prod.get("pass_through_capacity_pct") or 0)))

    revenue_now = price * units
    profit_now = revenue_now - total_cost_now_inr
    margin_now_pct = round(profit_now / revenue_now * 100, 2) if revenue_now else None

    price_increase_per_unit = round((total_delta_inr * pass_through_pct / 100) / units, 4) if units else 0.0
    revenue_new = revenue_now + total_delta_inr * pass_through_pct / 100
    absorbed_cost = total_delta_inr * (1 - pass_through_pct / 100)
    profit_new = profit_now - absorbed_cost
    margin_new_pct = round(profit_new / revenue_new * 100, 2) if revenue_new else None

    # ── Inventory buffer: days before the shock actually hits COGS ──
    buffer_days = int((profile.get("inventory") or {}).get("buffer_days") or 0)

    total_cost_increase_pct = round(total_delta_inr / total_cost_now_inr * 100, 2) if total_cost_now_inr else 0.0

    exposed = [m for m in materials_out if (m["delta_inr"] + m["deferred_delta_inr"]) > 0][:3]
    deferred_total = round(deferred_delta_inr, 2)

    return {
        "fx": {"rate": fx_rate, "drift_pct_30d": fx_drift, "source": fx["source"], "is_live": fx["is_live"]},
        "current": {"total_cost_inr": round(total_cost_now_inr, 2), "revenue_inr": round(revenue_now, 2),
                    "profit_inr": round(profit_now, 2), "margin_pct": margin_now_pct},
        "projected": {"total_cost_inr": round(total_cost_new_inr, 2), "revenue_inr": round(revenue_new, 2),
                      "profit_inr": round(profit_new, 2), "margin_pct": margin_new_pct},
        "delta": {"total_cost_inr": round(total_delta_inr, 2), "total_cost_pct": total_cost_increase_pct,
                  "profit_inr": round(profit_new - profit_now, 2),
                  "margin_pct_points": round((margin_new_pct - margin_now_pct), 2) if (margin_new_pct is not None and margin_now_pct is not None) else None,
                  "price_increase_needed_per_unit": price_increase_per_unit,
                  "deferred_by_fixed_contracts_inr": deferred_total},
        "logistics": {"mode": mode, "cost_now_inr": log_cost_now, "cost_new_inr": log_cost_new,
                      "shock_pct": log_shock_pct, "delta_inr": log_delta},
        "materials": materials_out,
        "most_exposed": [{"label": m["label"], "delta_inr": round(m["delta_inr"] + m["deferred_delta_inr"], 2),
                          "combined_shock_pct": m["combined_shock_pct"], "locked": m["contract_locked"]} for m in exposed],
        "buffer_days": buffer_days,
        "pass_through_capacity_pct": pass_through_pct,
    }


if __name__ == "__main__":
    sample_cascade_sectors = {
        "fuels": {"diesel": {"change_pct": 6.5}, "petrol": {"change_pct": 7.2}},
        "industry": {"plastics_pvc": {"change_pct": 3.5}, "steel": {"change_pct": 0.8}},
        "transport": {"road_freight": {"change_pct": 5.8}},
        "consumer_cpg": {"edible_oil": {"change_pct": 1.9}},
    }
    profile = {
        "business_name": "Test FMCG Co", "sector": "consumer_cpg",
        "materials": [
            {"key": "plastics_pvc", "monthly_volume": 5000, "unit_cost": 120, "currency": "INR", "contract_type": "spot"},
            {"key": "edible_oil", "monthly_volume": 2000, "unit_cost": 95, "currency": "USD", "contract_type": "spot"},
            {"key": "diesel", "monthly_volume": 1500, "unit_cost": 92, "currency": "INR", "contract_type": "fixed", "contract_months_remaining": 3},
        ],
        "logistics": {"mode": "road", "monthly_cost": 180000},
        "product": {"name": "Snack packs", "current_price": 20, "monthly_units_sold": 500000, "pass_through_capacity_pct": 40},
        "inventory": {"buffer_days": 21},
    }
    fx = market_data.get_fx_rate()
    r = calculate_impact(profile, sample_cascade_sectors, fx)
    import json
    print(json.dumps(r, indent=2))
