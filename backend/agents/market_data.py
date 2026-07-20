"""
APEX — Live Market Data (FX + Commodity Reference Prices)
===============================================================
Two data sources feeding the Business Impact Advisor:

1. INR/USD exchange rate — Frankfurter.app (free, keyless). India's import
   costs are USD-denominated but paid in rupees, so landed cost is hit by BOTH
   the commodity shock AND currency depreciation — APEX otherwise ignores
   this entirely.

2. Commodity reference prices — FRED's "Global price of X" series. These are
   IMF/World Bank Primary Commodity Price (Pink Sheet) data, re-published on
   FRED with a stable JSON API we already hold a key for, rather than scraping
   the World Bank's monthly Excel release. Verified series (no fertilizer
   series exists on this mirror — flagged honestly where absent).
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

FRANKFURTER_URL = "https://api.frankfurter.dev/v1"

# material_key -> (FRED series id, human label, unit)
COMMODITY_FRED_SERIES = {
    "steel":            ("PIORECRUSDM", "Iron Ore (steel input proxy)", "USD/dry metric ton"),
    "aluminium":        ("PALUMUSDM",   "Aluminum",                     "USD/metric ton"),
    "synthetic_rubber": ("PRUBBUSDM",   "Rubber",                       "USD/kg"),
    "edible_oil":       ("PPOILUSDM",   "Palm Oil (edible oil proxy)",  "USD/metric ton"),
    "fertiliser_urea":  ("PNGASEUUSDM", "Natural Gas, EU (urea is gas-intensive — proxy only)", "USD/MMBtu"),
    "fertiliser_cost":  ("PNGASEUUSDM", "Natural Gas, EU (urea is gas-intensive — proxy only)", "USD/MMBtu"),
    "naphtha":          ("PNRGINDEXM",  "Global Energy Index",         "index"),
    "pet_bottles":      ("PALUMUSDM",   "Aluminum (packaging-input proxy)", "USD/metric ton"),
}

_cache = {"fx": {"ts": 0, "data": None}, "commodity": {}}
_CACHE_TTL = 3600  # 1 hour — this is monthly/daily reference data, no need to hammer it


def get_fx_rate(base="USD", quote="INR"):
    """Current and ~30-day-ago FX rate, so callers can compute currency drift.
    Falls back to a fixed reference rate if the API is unreachable."""
    cache_key = f"{base}_{quote}"
    cached = _cache["fx"].get(cache_key)
    if cached and time.time() - cached["ts"] < _CACHE_TTL:
        return cached["data"]

    try:
        latest = requests.get(f"{FRANKFURTER_URL}/latest", params={"base": base, "symbols": quote}, timeout=8).json()
        current = latest["rates"][quote]
        current_date = latest["date"]

        hist = requests.get(f"{FRANKFURTER_URL}/{current_date}", params={"base": base, "symbols": quote,
                             "amount": 1}, timeout=8)
        # 30 days prior, for a drift baseline
        import datetime
        d = datetime.date.fromisoformat(current_date) - datetime.timedelta(days=30)
        hist30 = requests.get(f"{FRANKFURTER_URL}/{d.isoformat()}", params={"base": base, "symbols": quote}, timeout=8).json()
        baseline = hist30["rates"][quote]

        drift_pct = round((current - baseline) / baseline * 100, 2)
        result = {"base": base, "quote": quote, "rate": current, "rate_30d_ago": baseline,
                  "drift_pct_30d": drift_pct, "date": current_date, "source": "Frankfurter.app (live)", "is_live": True}
    except Exception as e:
        print(f"  FX fallback: {e}")
        result = {"base": base, "quote": quote, "rate": 83.5, "rate_30d_ago": 83.5,
                  "drift_pct_30d": 0.0, "date": None, "source": "cached reference rate", "is_live": False}

    _cache["fx"].setdefault(cache_key, {})
    _cache["fx"][cache_key] = {"ts": time.time(), "data": result}
    _cache["fx"] = {cache_key: _cache["fx"][cache_key]}  # keep cache tiny; single pair in practice
    return result


WORLD_BANK_URL = "https://api.worldbank.org/v2/country/IN/indicator"
_wb_cache = {"ts": 0, "data": None}


def get_india_macro_context():
    """India's real annual CPI inflation and GDP growth from the World Bank
    Indicators API (free, keyless) — an independent real-data cross-check
    against the model's fixed 0.015 CPI / -0.012 GDP passthrough assumptions."""
    if _wb_cache["data"] and time.time() - _wb_cache["ts"] < _CACHE_TTL:
        return _wb_cache["data"]
    try:
        cpi = requests.get(f"{WORLD_BANK_URL}/FP.CPI.TOTL.ZG", params={"format": "json", "per_page": 1}, timeout=8).json()
        gdp = requests.get(f"{WORLD_BANK_URL}/NY.GDP.MKTP.KD.ZG", params={"format": "json", "per_page": 1}, timeout=8).json()
        cpi_row, gdp_row = cpi[1][0], gdp[1][0]
        result = {"cpi_inflation_pct": round(cpi_row["value"], 2), "cpi_year": cpi_row["date"],
                  "gdp_growth_pct": round(gdp_row["value"], 2), "gdp_year": gdp_row["date"],
                  "source": "World Bank Indicators API (live)", "is_live": True}
    except Exception as e:
        print(f"  World Bank macro fallback: {e}")
        result = {"cpi_inflation_pct": None, "gdp_growth_pct": None, "source": "unavailable", "is_live": False}
    _wb_cache["ts"], _wb_cache["data"] = time.time(), result
    return result


def get_commodity_reference(material_key):
    """Live reference price + 30-day trend for a material, if FRED has a mapped
    series. Returns None (not an error) if this material has no live reference —
    the caller should fall back to the cascade model's own % projection."""
    if material_key not in COMMODITY_FRED_SERIES:
        return None

    cached = _cache["commodity"].get(material_key)
    if cached and time.time() - cached["ts"] < _CACHE_TTL:
        return cached["data"]

    series_id, label, unit = COMMODITY_FRED_SERIES[material_key]
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        s = fred.get_series(series_id).dropna()
        latest = float(s.iloc[-1])
        month_ago = float(s.iloc[-2]) if len(s) > 1 else latest
        trend_pct = round((latest - month_ago) / month_ago * 100, 2) if month_ago else 0.0
        result = {"material": material_key, "label": label, "unit": unit, "value": round(latest, 2),
                   "trend_pct_1m": trend_pct, "as_of": str(s.index[-1].date()),
                   "source": f"FRED {series_id} (IMF/World Bank commodity price data)", "is_live": True}
    except Exception as e:
        print(f"  Commodity reference fallback for {material_key}: {e}")
        result = {"material": material_key, "label": label, "unit": unit, "value": None,
                  "trend_pct_1m": 0.0, "as_of": None, "source": "unavailable", "is_live": False}

    _cache["commodity"][material_key] = {"ts": time.time(), "data": result}
    return result


if __name__ == "__main__":
    fx = get_fx_rate()
    print(f"FX: 1 {fx['base']} = {fx['rate']} {fx['quote']}  (30d drift {fx['drift_pct_30d']}%)  [{fx['source']}]")
    print()
    for mat in COMMODITY_FRED_SERIES:
        r = get_commodity_reference(mat)
        print(f"{mat:20} {r['value']} {r['unit']:28} 1m trend {r['trend_pct_1m']:+}%  [{r['source']}]")
