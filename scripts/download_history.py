"""Download free 1-minute NIFTY history into data/bars_1min.parquet (appends + de-duplicates).

    python scripts/download_history.py kaggle --file nifty_50_minute.csv   # years of data (download CSV from Kaggle first)
    python scripts/download_history.py yfinance                            # last ~30 days, no account
    python scripts/download_history.py upstox --days 120                   # needs UPSTOX_ACCESS_TOKEN in .env
    python scripts/download_history.py angel  --days 60                    # needs ANGEL_* in .env

Tip: if data/bars_1min.parquet holds synthetic data, delete it first so you don't mix the two.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from niftysim import history
from niftysim.config import load_config, data_dir
from niftysim.storage import save_bars

ap = argparse.ArgumentParser()
ap.add_argument("source", choices=["kaggle", "yfinance", "upstox", "angel"])
ap.add_argument("--file")
ap.add_argument("--days", type=int, default=30)
a = ap.parse_args()
cfg = load_config()
if a.source == "kaggle":
    df = history.load_kaggle_csv(a.file)
elif a.source == "yfinance":
    df = history.download_yfinance(min(a.days, 29))
elif a.source == "upstox":
    df = history.download_upstox(a.days, cfg["upstox"]["instrument_key"])
else:
    df = history.download_angel(a.days, cfg["angel"]["token"])
out = data_dir(cfg) / "bars_1min.parquet"
save_bars(df, out)
print(f"{len(df):,} bars {df.index[0]} -> {df.index[-1]} saved to {out}")
