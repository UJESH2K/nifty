"""One-shot switch from the demo market to REAL NIFTY data. Run tonight; takes a few minutes.

    python scripts/setup_real_data.py                         # Kaggle history (auto-download) + last ~29 days from Yahoo
    python scripts/setup_real_data.py --no-kaggle             # skip Kaggle
    python scripts/setup_real_data.py --kaggle my_file.csv    # use a CSV you downloaded yourself
    python scripts/setup_real_data.py --angel 60              # + Angel One history (needs .env)

What it does:
  1. moves any existing data files to data/backup_<time>/ (so demo data never mixes with real data)
  2. downloads real 1-minute NIFTY bars (and optional Kaggle / broker history)
  3. collects real headlines (RSS now + GDELT for the past days)
  4. trains naive, ridge and lgbm (with-news and price-only)
"""
import argparse, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from niftysim import history
from niftysim.config import load_config, data_dir, ROOT
from niftysim.storage import save_bars, save_news

ap = argparse.ArgumentParser()
ap.add_argument("--kaggle", help="path to a NIFTY 50 minute CSV you already have")
ap.add_argument("--no-kaggle", action="store_true", help="don't auto-download the Kaggle history")
ap.add_argument("--kaggle-years", type=float, default=20, help="keep only the last N years (free 1-min data starts 2015)")
ap.add_argument("--angel", type=int, default=0, help="days of Angel One history to add")
ap.add_argument("--upstox", type=int, default=0, help="days of Upstox history to add")
ap.add_argument("--news-days", type=int, default=14, help="days of GDELT headlines to backfill (0 = skip)")
ap.add_argument("--models", nargs="*", default=["naive", "ridge"],
                help="add lgbm for the slower non-linear model (can take 30+ min on 10 years)")
a = ap.parse_args()
cfg = load_config()
d = data_dir(cfg)

# 1 ---- back up
olds = [p for p in (d / "bars_1min.parquet", d / "bars_1s.parquet", d / "news.parquet") if p.exists()]
if olds:
    bk = d / f"backup_{pd.Timestamp.now():%Y%m%d_%H%M}"
    bk.mkdir()
    for p in olds:
        shutil.move(str(p), bk / p.name)
    print(f"[1/4] moved old data files to {bk}")
if not (d / "events.csv").exists() and (d / "events.example.csv").exists():
    shutil.copy(d / "events.example.csv", d / "events.csv")

# 2 ---- prices
out = d / "bars_1min.parquet"
got = 0
k = None
if a.kaggle:
    k = history.load_kaggle_csv(a.kaggle)
elif not a.no_kaggle:
    print("[2/4] downloading the Kaggle NIFTY 50 1-minute file (~50 MB, only that one file)…")
    try:
        k = history.download_kaggle_nifty()
    except Exception as ex:
        print(f"[2/4] Kaggle download failed ({ex}). Continuing with Yahoo only.")
if k is not None and len(k):
    k = k[k.index >= k.index[-1] - pd.Timedelta(days=365 * a.kaggle_years)]
    save_bars(k, out); got += len(k)
    print(f"[2/4] Kaggle: {len(k):,} bars {k.index[0]:%d %b %Y} -> {k.index[-1]:%d %b %Y}")
for name, fn, days in (("Angel One", lambda n: history.download_angel(n, cfg["angel"]["token"]), a.angel),
                       ("Upstox", lambda n: history.download_upstox(n, cfg["upstox"]["instrument_key"]), a.upstox)):
    if days:
        try:
            b = fn(days); save_bars(b, out); got += len(b)
            print(f"[2/4] {name}: {len(b):,} bars")
        except Exception as ex:
            print(f"[2/4] {name} failed: {ex}")
try:
    y = history.download_yahoo(29); save_bars(y, out); got += len(y)
    print(f"[2/4] Yahoo: {len(y):,} bars {y.index[0]:%d %b} -> {y.index[-1]:%d %b %H:%M}")
except Exception as ex:
    print(f"[2/4] Yahoo failed: {ex}")
if not out.exists():
    sys.exit("No price data could be downloaded. Check your internet connection, or pass --kaggle with the CSV.")

# 3 ---- news
try:
    from niftysim.news.sources import NewsPoller, backfill_gdelt
    frames = [NewsPoller(cfg).poll_once()]
    if a.news_days:
        frames.append(backfill_gdelt(cfg["news"]["gdelt_query"], a.news_days))
    frames = [f for f in frames if len(f)]
    n = pd.concat(frames, ignore_index=True).dropna(subset=["headline"]) if frames else pd.DataFrame(columns=["ts", "source", "headline"])
    n = n[n["headline"].str.len() > 15]
    if len(n):
        save_news(n[["ts", "source", "headline"]], d / "news.parquet")
    print(f"[3/4] headlines saved: {len(n)}")
except Exception as ex:
    print(f"[3/4] news collection failed ({ex}); the with-news model will learn from price only for now")

# 4 ---- train
from niftysim.train import train_all
print(f"[4/4] training {', '.join(a.models)} on all {pd.read_parquet(out).shape[0]:,} bars…")
train_all(cfg, "1min", a.models, do_cv=False)
print("\nOptional, slower: python scripts/train_models.py --no-cv --models lgbm")
print("\nDone. Start the app with:  python scripts/run_dashboard.py   (or double-click start_niftysim.bat)")
