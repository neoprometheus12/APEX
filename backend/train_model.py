import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

print("=" * 60)
print("APEX — Scenario Model Training")
print("=" * 60)

# ─────────────────────────────────────────
# STEP 1 — Load price data
# ─────────────────────────────────────────
print("\n[1/5] Loading price data...")

BASE = r"C:\Users\iamra\OneDrive\Desktop\ET\Project\apex\Data"
brent_path = os.path.join(BASE, "DCOILBRENTEU.csv")
wti_path   = os.path.join(BASE, "DCOILWTICO.csv")

brent = pd.read_csv(brent_path)
brent.columns = ["date", "brent"]
brent["date"] = pd.to_datetime(brent["date"], dayfirst=False, errors="coerce")
brent = brent.dropna()
brent = brent.set_index("date").sort_index()

wti = pd.read_csv(wti_path)
wti.columns = ["date", "wti"]
wti["date"] = pd.to_datetime(wti["date"], dayfirst=False, errors="coerce")
wti = wti.dropna()
wti = wti.set_index("date").sort_index()

print(f"  Brent: {len(brent)} records ({brent.index[0].date()} to {brent.index[-1].date()})")
print(f"  WTI:   {len(wti)} records ({wti.index[0].date()} to {wti.index[-1].date()})")

prices = brent.join(wti, how="inner")
print(f"  Combined: {len(prices)} records")

# ─────────────────────────────────────────
# STEP 2 — Historical disruption events
# ─────────────────────────────────────────
print("\n[2/5] Loading historical disruption events...")

DISRUPTION_EVENTS = [
    {"date":"1990-08-02","type":"hormuz","severity":0.30,"duration":180,
     "brent_shock_d1":12.0,"brent_shock_d7":28.0,"brent_shock_d30":45.0,
     "refinery_rate_chg":-8.0,"gdp_impact":-0.8,"power_stress":55,"spr_draw":2.1},
    {"date":"2003-03-20","type":"hormuz","severity":0.20,"duration":90,
     "brent_shock_d1":5.0,"brent_shock_d7":12.0,"brent_shock_d30":18.0,
     "refinery_rate_chg":-4.0,"gdp_impact":-0.3,"power_stress":35,"spr_draw":0.8},
    {"date":"2005-08-29","type":"opec","severity":0.15,"duration":60,
     "brent_shock_d1":4.0,"brent_shock_d7":8.0,"brent_shock_d30":10.0,
     "refinery_rate_chg":-3.0,"gdp_impact":-0.2,"power_stress":28,"spr_draw":0.5},
    {"date":"2011-02-17","type":"opec","severity":0.25,"duration":240,
     "brent_shock_d1":3.0,"brent_shock_d7":10.0,"brent_shock_d30":15.0,
     "refinery_rate_chg":-5.0,"gdp_impact":-0.4,"power_stress":40,"spr_draw":1.0},
    {"date":"2012-01-23","type":"hormuz","severity":0.20,"duration":365,
     "brent_shock_d1":2.0,"brent_shock_d7":6.0,"brent_shock_d30":10.0,
     "refinery_rate_chg":-3.5,"gdp_impact":-0.3,"power_stress":32,"spr_draw":0.7},
    {"date":"2016-11-30","type":"opec","severity":0.20,"duration":365,
     "brent_shock_d1":8.0,"brent_shock_d7":12.0,"brent_shock_d30":18.0,
     "refinery_rate_chg":-2.0,"gdp_impact":-0.25,"power_stress":30,"spr_draw":0.4},
    {"date":"2019-09-14","type":"hormuz","severity":0.55,"duration":7,
     "brent_shock_d1":14.6,"brent_shock_d7":9.7,"brent_shock_d30":3.5,
     "refinery_rate_chg":-12.0,"gdp_impact":-0.15,"power_stress":62,"spr_draw":1.5},
    {"date":"2019-06-13","type":"hormuz","severity":0.30,"duration":60,
     "brent_shock_d1":3.0,"brent_shock_d7":5.0,"brent_shock_d30":4.0,
     "refinery_rate_chg":-2.0,"gdp_impact":-0.1,"power_stress":25,"spr_draw":0.3},
    {"date":"2020-03-11","type":"opec","severity":0.30,"duration":90,
     "brent_shock_d1":-10.0,"brent_shock_d7":-25.0,"brent_shock_d30":-55.0,
     "refinery_rate_chg":-20.0,"gdp_impact":-5.0,"power_stress":15,"spr_draw":-1.0},
    {"date":"2020-03-08","type":"opec","severity":0.20,"duration":30,
     "brent_shock_d1":-25.0,"brent_shock_d7":-30.0,"brent_shock_d30":-45.0,
     "refinery_rate_chg":-8.0,"gdp_impact":-0.8,"power_stress":20,"spr_draw":-0.5},
    {"date":"2021-03-23","type":"red_sea","severity":1.0,"duration":6,
     "brent_shock_d1":2.0,"brent_shock_d7":4.0,"brent_shock_d30":1.5,
     "refinery_rate_chg":-1.5,"gdp_impact":-0.05,"power_stress":18,"spr_draw":0.2},
    {"date":"2022-02-24","type":"opec","severity":0.25,"duration":365,
     "brent_shock_d1":8.0,"brent_shock_d7":15.0,"brent_shock_d30":17.0,
     "refinery_rate_chg":-6.0,"gdp_impact":-0.5,"power_stress":55,"spr_draw":1.8},
    {"date":"2022-10-05","type":"opec","severity":0.20,"duration":180,
     "brent_shock_d1":4.0,"brent_shock_d7":6.0,"brent_shock_d30":8.0,
     "refinery_rate_chg":-2.0,"gdp_impact":-0.2,"power_stress":28,"spr_draw":0.5},
    {"date":"2023-12-19","type":"red_sea","severity":0.80,"duration":180,
     "brent_shock_d1":1.5,"brent_shock_d7":3.0,"brent_shock_d30":4.5,
     "refinery_rate_chg":-3.0,"gdp_impact":-0.15,"power_stress":35,"spr_draw":0.6},
    {"date":"2024-04-13","type":"hormuz","severity":0.40,"duration":14,
     "brent_shock_d1":3.5,"brent_shock_d7":4.0,"brent_shock_d30":1.5,
     "refinery_rate_chg":-2.5,"gdp_impact":-0.1,"power_stress":30,"spr_draw":0.4},
    {"date":"2025-01-15","type":"hormuz","severity":0.50,"duration":30,
     "brent_shock_d1":8.0,"brent_shock_d7":12.0,"brent_shock_d30":6.0,
     "refinery_rate_chg":-8.0,"gdp_impact":-0.2,"power_stress":55,"spr_draw":1.2},
]

print(f"  Loaded {len(DISRUPTION_EVENTS)} historical disruption events")

# ─────────────────────────────────────────
# STEP 3 — Build training dataset
# ─────────────────────────────────────────
print("\n[3/5] Building training dataset...")

def get_price_context(event_date_str, prices_df, days_before=30):
    try:
        event_dt = pd.to_datetime(event_date_str)
        window = prices_df[prices_df.index < event_dt].tail(days_before)
        if len(window) < 5:
            return None
        return {
            "baseline_brent": float(window["brent"].iloc[-1]),
            "brent_30d_avg": float(window["brent"].mean()),
            "brent_volatility": float(window["brent"].std()),
            "brent_trend": float((window["brent"].iloc[-1] - window["brent"].iloc[0]) / window["brent"].iloc[0] * 100),
        }
    except Exception as e:
        return None

rows = []
for event in DISRUPTION_EVENTS:
    ctx = get_price_context(event["date"], prices)
    if ctx is None:
        print(f"  Skipping {event['date']} — no price context")
        continue
    row = {
        "severity": event["severity"],
        "duration_days": min(event["duration"], 365),
        "type_hormuz": 1 if event["type"] == "hormuz" else 0,
        "type_redsea": 1 if event["type"] == "red_sea" else 0,
        "type_opec": 1 if event["type"] == "opec" else 0,
        "baseline_brent": ctx["baseline_brent"],
        "brent_30d_avg": ctx["brent_30d_avg"],
        "brent_volatility": ctx["brent_volatility"],
        "brent_trend_30d": ctx["brent_trend"],
        "brent_shock_d1": event["brent_shock_d1"],
        "brent_shock_d7": event["brent_shock_d7"],
        "brent_shock_d30": event["brent_shock_d30"],
        "refinery_rate_chg": event["refinery_rate_chg"],
        "gdp_impact": event["gdp_impact"],
        "power_stress": event["power_stress"],
        "spr_draw": event["spr_draw"],
    }
    rows.append(row)

df = pd.DataFrame(rows)
print(f"  Built dataset: {len(df)} training samples")

# ─────────────────────────────────────────
# STEP 4 — Train model
# ─────────────────────────────────────────
print("\n[4/5] Training model...")

FEATURES = [
    "severity","duration_days",
    "type_hormuz","type_redsea","type_opec",
    "baseline_brent","brent_30d_avg",
    "brent_volatility","brent_trend_30d"
]

TARGETS = [
    "brent_shock_d1","brent_shock_d7","brent_shock_d30",
    "refinery_rate_chg","gdp_impact","power_stress","spr_draw"
]

X = df[FEATURES].values
y = df[TARGETS].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = MultiOutputRegressor(
    GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
)
model.fit(X_scaled, y)
print("  Model trained successfully")

# ─────────────────────────────────────────
# ECONOMETRIC COEFFICIENTS
# Sources: IMF WP/22/58, RBI MPR 2023, PPAC
# ─────────────────────────────────────────
ECONOMETRIC = {
    "brent_10usd_to_india_cpi": 0.15,
    "brent_10usd_to_india_gdp": -0.12,
    "crude_to_petrol_passthrough": 0.65,
    "crude_to_diesel_passthrough": 0.58,
    "crude_to_lpg_passthrough": 0.45,
    "india_crude_import_dependency": 0.88,
    "hormuz_share_of_imports": 0.42,
    "redsea_share_of_imports": 0.18,
    "india_daily_crude_mbd": 4.8,
    "india_spr_days": 9.5,
}

# ─────────────────────────────────────────
# STEP 5 — Save everything
# ─────────────────────────────────────────
print("\n[5/5] Saving model files...")

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(save_path, exist_ok=True)

joblib.dump(model,             os.path.join(save_path, "scenario_model.pkl"))
joblib.dump(scaler,            os.path.join(save_path, "scenario_scaler.pkl"))
joblib.dump(FEATURES,          os.path.join(save_path, "scenario_features.pkl"))
joblib.dump(TARGETS,           os.path.join(save_path, "scenario_targets.pkl"))
joblib.dump(ECONOMETRIC,       os.path.join(save_path, "econometric.pkl"))
joblib.dump(DISRUPTION_EVENTS, os.path.join(save_path, "disruption_events.pkl"))

print(f"  Saved to: {save_path}")

# ─────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("VALIDATION — Testing on known events")
print("=" * 60)

validation = [
    {"name":"2019 Abqaiq Attack","type":"hormuz","severity":0.55,
     "duration":7,"baseline_brent":60.0,"actual_d7":9.7},
    {"name":"2021 Suez Blockage","type":"red_sea","severity":1.0,
     "duration":6,"baseline_brent":64.0,"actual_d7":4.0},
    {"name":"2022 Russia Sanctions","type":"opec","severity":0.25,
     "duration":365,"baseline_brent":92.0,"actual_d7":15.0},
]

for t in validation:
    X_test = np.array([[
        t["severity"],
        min(t["duration"], 365),
        1 if t["type"]=="hormuz" else 0,
        1 if t["type"]=="red_sea" else 0,
        1 if t["type"]=="opec" else 0,
        t["baseline_brent"],
        t["baseline_brent"] * 0.98,
        t["baseline_brent"] * 0.05,
        2.0,
    ]])
    pred = model.predict(scaler.transform(X_test))[0]
    predicted = round(pred[1], 1)
    actual = t["actual_d7"]
    accuracy = round((1 - abs(predicted - actual) / max(abs(actual), 0.1)) * 100, 1)
    status = "✅" if accuracy >= 70 else "⚠"
    print(f"\n  {status} {t['name']}")
    print(f"     Predicted Day 7: {predicted:+.1f}% | Actual: {actual:+.1f}% | Accuracy: {accuracy}%")

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)