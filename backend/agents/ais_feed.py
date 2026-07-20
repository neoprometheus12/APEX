"""
APEX — Live Vessel Feed (aisstream.io)
==========================================
Streams real tanker positions from aisstream.io over a WebSocket and keeps a
rolling in-memory snapshot the Digital Twin can query. Regions watched: Strait of
Hormuz, Red Sea / Bab-el-Mandeb, and India's west + east coasts — i.e. the crude
inflow to India's ports.

Free API key required — set AISSTREAM_API_KEY in .env (sign up at aisstream.io).
Without a key, or if the stream is unreachable, a clearly-labelled *simulated*
tanker set is returned so the twin still functions. The `is_live` flag says which.

A daemon thread runs its own asyncio loop; requests just read the latest snapshot
(atomic dict swap, no lock needed on read).
"""

import os
import json
import time
import threading
from dotenv import load_dotenv

load_dotenv()
AISSTREAM_API_KEY = os.getenv("AISSTREAM_API_KEY")
AISSTREAM_URL = "wss://stream.aisstream.io/v0/stream"

# aisstream bounding boxes are [[min_lat, min_lon], [max_lat, max_lon]] (SW corner
# then NE corner). Getting the corner order wrong makes aisstream return nothing.
REGIONS = {
    "hormuz":     {"box": [[24.0, 55.0], [27.5, 58.8]], "label": "Strait of Hormuz"},
    "red_sea":    {"box": [[11.0, 39.5], [16.5, 44.8]], "label": "Red Sea / Bab-el-Mandeb"},
    "india_west": {"box": [[8.0, 66.0],  [24.5, 76.5]], "label": "India West Coast"},
    "india_east": {"box": [[9.0, 79.5],  [22.5, 87.5]], "label": "India East Coast"},
}

# snapshot: {mmsi: {lat,lng,name,sog,cog,region,ts}} — replaced atomically
_snapshot = {}
_state = {"is_live": False, "source": "not started", "last_update": 0, "count": 0}
_started = False


def _region_of(lat, lng):
    for key, r in REGIONS.items():
        (a_lat, a_lng), (c_lat, c_lng) = r["box"]
        if min(a_lat, c_lat) <= lat <= max(a_lat, c_lat) and min(a_lng, c_lng) <= lng <= max(a_lng, c_lng):
            return key
    return None


# ─────────────────────────────────────────────────────────────
# SIMULATED FALLBACK — plausible tankers so the twin works keyless
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
# LIVE COLLECTOR — aisstream websocket in a daemon thread
# (real vessels only — no simulated fallback)
# ─────────────────────────────────────────────────────────────
async def _stream_loop():
    import websockets
    global _snapshot, _state
    # Subscribe to all position-bearing report types (tankers broadcast as Class A
    # PositionReport *and* Class B / long-range types). Coordinates are in MetaData
    # on every message, so we read those rather than the type-specific inner payload.
    sub = {
        "APIKey": AISSTREAM_API_KEY,
        "BoundingBoxes": [r["box"] for r in REGIONS.values()],
        "FilterMessageTypes": [
            "PositionReport", "StandardClassBPositionReport",
            "ExtendedClassBPositionReport", "LongRangeAisBroadcastMessage",
        ],
    }
    while True:
        try:
            async with websockets.connect(AISSTREAM_URL, ping_interval=20, close_timeout=5) as ws:
                await ws.send(json.dumps(sub))
                _state.update(is_live=True, source="aisstream.io (live)")
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        meta = msg.get("MetaData", {})
                        lat = meta.get("latitude"); lng = meta.get("longitude")
                        if lat is None or lng is None:
                            continue
                        region = _region_of(lat, lng)
                        if region is None:
                            continue
                        # Sog/Cog live under a type-specific inner key — grab whichever is present
                        inner = next(iter((msg.get("Message") or {}).values()), {}) or {}
                        mmsi = meta.get("MMSI")
                        _snapshot[mmsi] = {
                            "mmsi": mmsi,
                            "name": (meta.get("ShipName") or "").strip() or f"MMSI {mmsi}",
                            "lat": round(lat, 4), "lng": round(lng, 4),
                            "sog": inner.get("Sog"), "cog": inner.get("Cog"),
                            "region": region, "ts": time.time(),
                        }
                        _state.update(last_update=time.time(), count=len(_snapshot))
                    except Exception:
                        continue
        except Exception as e:
            _state.update(is_live=False, source=f"aisstream unreachable ({str(e)[:60]})")
            await _async_sleep(15)


async def _async_sleep(s):
    import asyncio
    await asyncio.sleep(s)


def _prune(d, max_age=900):
    """Drop vessels not seen in the last 15 min so the map reflects live traffic."""
    now = time.time()
    return {k: v for k, v in d.items() if now - v.get("ts", 0) < max_age}


def _run_thread():
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_stream_loop())


def start_collector():
    """Start the live collector once, if a key is configured. Idempotent."""
    global _started
    if _started:
        return
    _started = True
    if AISSTREAM_API_KEY:
        threading.Thread(target=_run_thread, daemon=True, name="ais-collector").start()
        _state.update(source="aisstream.io (connecting)")
    else:
        _state.update(is_live=False, source="AISSTREAM_API_KEY not set — add a free key from aisstream.io")


def get_vessels(region=None):
    """Return the current REAL vessel snapshot from aisstream.io (empty if no key
    is configured or the stream hasn't delivered positions yet). No simulated
    data — the map shows genuine AIS traffic or nothing."""
    data = _prune(_snapshot)
    vessels = list(data.values())
    if region:
        vessels = [v for v in vessels if v["region"] == region]

    by_region = {}
    for v in vessels:
        by_region[v["region"]] = by_region.get(v["region"], 0) + 1

    is_live = bool(vessels)
    if is_live:
        source = _state["source"]
    elif not AISSTREAM_API_KEY:
        source = "No live vessels — set AISSTREAM_API_KEY (free at aisstream.io) to stream real AIS traffic"
    else:
        source = _state.get("source", "aisstream.io connecting — awaiting first positions")

    return {
        "vessels": vessels,
        "by_region": by_region,
        "regions": {k: r["label"] for k, r in REGIONS.items()},
        "is_live": is_live,
        "source": source,
        "count": len(vessels),
        "key_configured": bool(AISSTREAM_API_KEY),
    }


if __name__ == "__main__":
    start_collector()
    time.sleep(2)
    snap = get_vessels()
    print(f"is_live={snap['is_live']}  source={snap['source']}  count={snap['count']}")
    print("by_region:", snap["by_region"])
    for v in snap["vessels"][:5]:
        print(f"  {v['name']:20} {v['lat']:.2f},{v['lng']:.2f}  {v['region']}  sog={v['sog']}")
