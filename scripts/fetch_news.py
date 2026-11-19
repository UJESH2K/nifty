"""Collect timestamped headlines into data/news.parquet (for training the 'with news' model).

    python scripts/fetch_news.py --gdelt-days 90     # free historical backfill (GDELT, no key)
    python scripts/fetch_news.py                     # one pass over live RSS / Finnhub / Marketaux
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from niftysim.config import load_config, data_dir
from niftysim.news.sources import NewsPoller, backfill_gdelt
from niftysim.news.scorer import make_scorer, score_frame
from niftysim.storage import save_news

ap = argparse.ArgumentParser()
ap.add_argument("--gdelt-days", type=int, default=0)
a = ap.parse_args()
cfg = load_config()
frames = [NewsPoller(cfg).poll_once()]
if a.gdelt_days:
    frames.append(backfill_gdelt(cfg["news"]["gdelt_query"], a.gdelt_days))
df = pd.concat(frames, ignore_index=True).dropna(subset=["headline"])
df = df[df["headline"].str.len() > 15]
save_news(df[["ts", "source", "headline"]], data_dir(cfg) / "news.parquet")
print(f"saved {len(df)} headlines. Sample scores:")
print(score_frame(df.tail(10).copy(), make_scorer(cfg))[["ts", "headline", "direction", "cruciality"]].to_string())
