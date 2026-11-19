"""Generate a synthetic market (1s ticks -> 1s + 1min bars) and headlines, for offline testing.

    python scripts/make_synthetic.py --days 60
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from niftysim.config import load_config, data_dir
from niftysim.feeds.synthetic import generate_history
from niftysim.storage import resample_bars

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=int, default=60)
ap.add_argument("--keep-1s-days", type=int, default=5, help="1s bars are big; keep only the last N days")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

cfg = load_config()
d = data_dir(cfg)
t = time.time()
bars1s, news = generate_history(a.days, 1, a.seed)
m1 = resample_bars(bars1s, "1min")
m1.to_parquet(d / "bars_1min.parquet")
keep = bars1s.index.normalize().unique()[-a.keep_1s_days:]
bars1s[bars1s.index.normalize() >= keep[0]].to_parquet(d / "bars_1s.parquet")
news.to_parquet(d / "news.parquet", index=False)
print(f"{len(m1):,} 1-min bars, {len(news)} headlines, {time.time()-t:.0f}s -> {d}")
