"""
APEX — Retrieval-Augmented Generation Engine
==================================================
Genuine RAG over a geopolitical/commodity intelligence corpus: documents are
embedded with Gemini's embedding model and retrieved by semantic (cosine)
similarity, not by keyword match or "most recent N" — the gap this closes is
that APEX's live API tools (Section 3.2/3.3 of the platform docs) give an
agent the CURRENT picture, but nothing that lets it answer "has this happened
before, and how did it play out?" grounded in retrieved historical precedent.

Corpus (static, versioned):
  - The 16 real historical disruption events already used to train the
    scenario-prediction ML model (train_model.py), turned into prose documents
    an LLM can retrieve and reason over directly.
  - Econometric source summaries (IMF WP/22/58, RBI MPR 2023, PPAC) — the
    citations already used numerically elsewhere, now retrievable as text.
  - Corridor and India-context background documents.

Embeddings are computed once and cached to disk (data/rag_index.json), keyed
by a content hash, so a restart doesn't re-embed (and re-spend quota on)
unchanged documents.
"""

import os
import json
import hashlib
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL = "gemini-embedding-001"

INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "rag_index.json")


# ─────────────────────────────────────────────────────────────
# STATIC CORPUS
# ─────────────────────────────────────────────────────────────
def _historical_event_docs():
    """Turn the same 16 events used to train the ML model into retrievable prose.
    Loaded from the already-persisted disruption_events.pkl (saved by
    train_model.py) rather than importing train_model.py itself, which would
    re-run its full training script as an import side effect."""
    import joblib
    events_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "disruption_events.pkl")
    DISRUPTION_EVENTS = joblib.load(events_path)
    docs = []
    for e in DISRUPTION_EVENTS:
        text = (
            f"On {e['date']}, a {e['type'].replace('_', ' ')}-type disruption occurred with severity "
            f"{e['severity']} lasting {e['duration']} days. Brent crude moved {e['brent_shock_d1']:+.1f}% on "
            f"Day 1, {e['brent_shock_d7']:+.1f}% by Day 7, and {e['brent_shock_d30']:+.1f}% by Day 30. "
            f"Refinery run rates changed {e['refinery_rate_chg']:+.1f}%, India's GDP impact was estimated at "
            f"{e['gdp_impact']:+.2f}%, power-sector stress reached {e['power_stress']}/100, and SPR drawdown "
            f"was {e['spr_draw']:+.2f} days-equivalent."
        )
        docs.append({"id": f"event_{e['date']}", "type": "historical_event", "date": e["date"],
                     "corridor": e["type"], "text": text})
    return docs


def _reference_docs():
    return [
        {"id": "ref_imf_wp2258", "type": "econometric_source", "text":
            "IMF Working Paper 22/58 estimates that a 10 USD/barrel rise in Brent crude raises India's "
            "consumer price inflation by approximately 0.15 percentage points and reduces India's GDP growth "
            "by approximately 0.12 percentage points, reflecting India's high crude import dependency and "
            "the pass-through of energy costs into transport, manufacturing input costs, and fertiliser "
            "production."},
        {"id": "ref_rbi_mpr2023", "type": "econometric_source", "text":
            "The Reserve Bank of India's 2023 Monetary Policy Report identifies fuel and transport cost "
            "pass-through as a primary channel through which crude oil price shocks propagate into headline "
            "inflation in India, with petrol and diesel retail pricing carrying an estimated 0.65 and 0.58 "
            "passthrough coefficient respectively against international crude price movements."},
        {"id": "ref_ppac", "type": "econometric_source", "text":
            "PPAC (Petroleum Planning and Analysis Cell) publishes India's monthly crude import statistics, "
            "showing approximately 88% import dependency, with the Strait of Hormuz corridor carrying "
            "roughly 40-45% of import volume from Gulf suppliers including Iraq, Saudi Arabia, UAE, and "
            "Kuwait, and the Red Sea / Bab-el-Mandeb corridor carrying approximately 18% via Suez-side and "
            "East African routes."},
        {"id": "ref_hormuz_geo", "type": "corridor_background", "text":
            "The Strait of Hormuz is a narrow chokepoint between Iran and Oman connecting the Persian Gulf "
            "to the Arabian Sea. It is the world's most important oil chokepoint by volume and has "
            "repeatedly been the site of tanker seizures, naval posturing, and direct attacks on Gulf "
            "energy infrastructure, most notably the 2019 Abqaiq-Khurais attack on Saudi processing "
            "facilities. India sources a large share of its Gulf crude — from Iraq, Saudi Arabia, UAE, and "
            "Kuwait — through this corridor."},
        {"id": "ref_redsea_geo", "type": "corridor_background", "text":
            "The Red Sea and Bab-el-Mandeb corridor connects the Gulf of Aden to the Suez Canal and "
            "Mediterranean, providing the shortest sea route between Asia and Europe. Sustained Houthi "
            "attacks on commercial shipping from late 2023 forced a large share of global tanker traffic to "
            "reroute around the Cape of Good Hope, adding 12-14 days of transit time and raising freight "
            "and insurance costs on this route."},
        {"id": "ref_spr_context", "type": "policy_background", "text":
            "India's Strategic Petroleum Reserve is held at three underground sites — Visakhapatnam, "
            "Mangalore, and Padur — with a combined capacity of approximately 5.33 million metric tonnes, "
            "providing roughly 9.5 days of national crude consumption cover in an emergency, intended to "
            "bridge short-term supply disruptions while alternative sourcing or diplomatic resolution is "
            "pursued."},
    ]


def _content_hash(docs):
    blob = json.dumps([d["text"] for d in docs], sort_keys=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _embed_batch(texts):
    embeddings = []
    for t in texts:
        r = client.models.embed_content(model=EMBED_MODEL, contents=t)
        embeddings.append(r.embeddings[0].values)
    return embeddings


def _load_index():
    if os.path.isfile(INDEX_PATH):
        try:
            with open(INDEX_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_index(index):
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f)


_INDEX = None  # {"hash":..., "docs":[{...,"embedding":[...]}]}


def build_or_load_index(force=False):
    """Embed the static corpus once, cache to disk, and skip re-embedding on
    every restart if the corpus content hasn't changed."""
    global _INDEX
    docs = _historical_event_docs() + _reference_docs()
    h = _content_hash(docs)

    cached = None if force else _load_index()
    if cached and cached.get("hash") == h:
        _INDEX = cached
        return _INDEX

    print(f"  [RAG] Embedding {len(docs)} corpus documents (model={EMBED_MODEL})...")
    texts = [d["text"] for d in docs]
    embeddings = _embed_batch(texts)
    for d, emb in zip(docs, embeddings):
        d["embedding"] = emb
    _INDEX = {"hash": h, "docs": docs}
    _save_index(_INDEX)
    print(f"  [RAG] Index built and cached ({len(docs)} docs, {len(embeddings[0])} dims).")
    return _INDEX


def _cosine(a, b):
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def search(query: str, k: int = 4, doc_type: str = None) -> list:
    """Semantic search over the corpus. Returns the top-k most relevant
    documents by cosine similarity, not keyword match or recency."""
    index = _INDEX or build_or_load_index()
    q_emb = client.models.embed_content(model=EMBED_MODEL, contents=query).embeddings[0].values

    candidates = index["docs"]
    if doc_type:
        candidates = [d for d in candidates if d["type"] == doc_type]

    scored = [(_cosine(q_emb, d["embedding"]), d) for d in candidates]
    scored.sort(key=lambda x: -x[0])
    return [{"score": round(s, 3), "id": d["id"], "type": d["type"], "text": d["text"]}
            for s, d in scored[:k]]


if __name__ == "__main__":
    build_or_load_index()
    for q in ["Has Hormuz been closed before and what happened to prices?",
              "How does a Brent price shock affect India's inflation?",
              "What is India's emergency oil reserve capacity?"]:
        print(f"\nQuery: {q}")
        for r in search(q, k=3):
            print(f"  [{r['score']}] ({r['type']}) {r['text'][:110]}...")
