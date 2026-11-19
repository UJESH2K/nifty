"""Headline -> (direction in [-1, +1], cruciality 1..10, category).

Three interchangeable scorers, all free:
  lexicon : built-in India-market rules. Instant, offline, transparent. Default.
  finbert : ProsusAI/finbert from Hugging Face (pip install transformers torch). Better language
            understanding of direction; cruciality/category still from the lexicon.
  llm     : any OpenAI-compatible endpoint. Default = Ollama on your own machine (free, local).
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache

CATEGORIES = [  # (category, base cruciality, regex)
    ("budget", 9, r"\bunion budget|budget\b|finance minister"),
    ("rbi", 9, r"\brbi\b|reserve bank|repo rate|monetary policy|\bmpc\b"),
    ("geopolitics", 8, r"war|border|tension|attack|missile|sanction|terror|trade deal|tariff|ceasefire"),
    ("global", 7, r"\bfed\b|federal reserve|wall street|us markets|global (stocks|markets)|asian markets|dow|nasdaq"),
    ("macro", 7, r"gdp|inflation|\bcpi\b|\bwpi\b|iip|fiscal deficit|pmi|unemployment"),
    ("election", 8, r"election|exit poll|poll results|lok sabha"),
    ("flows", 6, r"\bfiis?\b|\bfpis?\b|\bdiis?\b|foreign (investors|inflows|outflows)"),
    ("commodities", 6, r"crude|oil price|brent|gold"),
    ("regulation", 6, r"\bsebi\b|regulat|ban on|tighten"),
    ("policy", 7, r"government|infrastructure|capex|\bpli\b|cabinet approves|gst"),
    ("fx", 5, r"rupee|dollar index|forex"),
    ("earnings", 5, r"earnings|results|profit|revenue|guidance|q[1-4]\b|quarterly"),
    ("market", 3, r"nifty|sensex|market|stocks|shares"),
]

# Specific phrase rules first (weight 2) – these encode India-market logic
SPECIFIC = [
    (r"\bcut[s]?\b[^.,;]{0,25}\b(repo|interest|policy)?\s*rates?\b|\brate cuts?\b", +2),
    (r"\bhike[s]?\b[^.,;]{0,25}\brates?\b|\brate hikes?\b|higher (rates|for longer)", -2),
    (r"crude[^.,;]{0,25}\b(slump|fall|drop|plunge|ease|slide|tumble)", +2),
    (r"crude[^.,;]{0,25}\b(spike|jump|surge|rise|soar|rall)", -2),
    (r"inflation[^.,;]{0,25}\b(eases?|cools?|falls?|slows?|below)", +2),
    (r"inflation[^.,;]{0,25}\b(jumps?|rises?|spikes?|surges?|above|hot)", -2),
    (r"\b(fiis?|fpis?|foreign investors)\b[^.,;]{0,30}\b(net buyers|buy|pump|inflows?|invest)", +2),
    (r"\b(fiis?|fpis?|foreign investors)\b[^.,;]{0,30}\b(net sellers|sell|dump|outflows?|pull out)", -2),
    (r"\bbeats?\b[^.,;]{0,20}(estimates|expectations)|earnings beat|strong (quarterly|q[1-4])", +2),
    (r"\bmiss(es|ed)?\b[^.,;]{0,20}(estimates|expectations)|earnings miss|cut[s]? guidance", -2),
    (r"deal signed|tariffs? (slashed|cut|removed)|ceasefire|peace talks", +2),
    (r"tensions? escalat|war fears|attack|sanctions? imposed|tariffs? (hiked|imposed|raised)", -2),
]
GENERIC_POS = r"\b(rall(y|ies|ied)|surge[sd]?|soar(s|ed)?|jump(s|ed)?|gain(s|ed)?|record high|boost|upgrade[sd]?|bullish|rebound(s|ed)?|recover(s|y|ed)?)\b"
GENERIC_NEG = r"\b(fall(s|en)?|fell|slump(s|ed)?|tumble[sd]?|crash(es|ed)?|plunge[sd]?|drag(s|ged)?|lower|flee|selloff|sell-off|downgrade[sd]?|bearish|slide[sd]?|sink(s)?|weak)\b"
INTENSIFIERS = r"unexpected|surprise|shock|sharply|biggest|record|crash|emergency|historic|plunge"


def _category(h: str) -> tuple[str, int]:
    for cat, base, rx in CATEGORIES:
        if re.search(rx, h):
            return cat, base
    return "other", 2


def lexicon_score(headline: str) -> dict:
    h = headline.lower()
    s = 0.0
    for rx, w in SPECIFIC:
        if re.search(rx, h):
            s += w
    s += 1.0 * len(re.findall(GENERIC_POS, h))
    s -= 1.0 * len(re.findall(GENERIC_NEG, h))
    direction = math.tanh(s / 2.0)
    cat, base = _category(h)
    cru = base + (1 if re.search(INTENSIFIERS, h) else 0)
    if abs(direction) < 0.25:
        cru = min(cru, 3)  # a headline with no clear direction rarely moves the index
    return {"direction": round(direction, 3), "cruciality": int(max(1, min(10, cru))), "category": cat}


@lru_cache(maxsize=1)
def _finbert():
    from transformers import pipeline  # pip install transformers torch
    return pipeline("text-classification", model="ProsusAI/finbert", top_k=None)


def finbert_score(headline: str) -> dict:
    base = lexicon_score(headline)
    probs = {d["label"].lower(): d["score"] for d in _finbert()(headline[:512])[0]}
    base["direction"] = round(probs.get("positive", 0) - probs.get("negative", 0), 3)
    return base


LLM_PROMPT = """You score Indian stock-market headlines for their likely effect on the NIFTY 50 index
in the next 30 minutes. Reply ONLY with JSON: {"direction": float -1..1 (negative = index likely down),
"cruciality": int 1..10 (10 = market-wide shock like a surprise RBI move), "category": short word}.
Headline: """


def llm_score(headline: str, base_url: str, model: str) -> dict:
    import requests

    r = requests.post(f"{base_url.rstrip('/')}/chat/completions", timeout=20, json={
        "model": model, "temperature": 0,
        "messages": [{"role": "user", "content": LLM_PROMPT + headline}],
        "response_format": {"type": "json_object"},
    })
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    d = json.loads(m.group(0) if m else txt)
    return {"direction": float(max(-1, min(1, d.get("direction", 0)))),
            "cruciality": int(max(1, min(10, d.get("cruciality", 3)))),
            "category": str(d.get("category", "other"))[:20]}


def make_scorer(cfg: dict):
    kind = cfg["news"].get("scorer", "lexicon")
    if kind == "finbert":
        return finbert_score
    if kind == "llm":
        return lambda h: llm_score(h, cfg["news"]["llm_base_url"], cfg["news"]["llm_model"])
    return lexicon_score


def score_frame(df, scorer):
    import pandas as pd
    if df.empty:
        return df.assign(direction=[], cruciality=[], category=[])
    res = pd.DataFrame([scorer(h) for h in df["headline"].astype(str)], index=df.index)
    return pd.concat([df.drop(columns=[c for c in res.columns if c in df.columns]), res], axis=1)
