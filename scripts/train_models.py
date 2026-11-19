"""Slow layer: (re)train all models with walk-forward validation.

    python scripts/train_models.py                       # all models, config resolution
    python scripts/train_models.py --models lgbm ridge   # pick models
    python scripts/train_models.py --res 1s --last-days 20
    python scripts/train_models.py --holdout-days 5      # keep last 5 days unseen for honest time-travel
Schedule it nightly after 15:30 IST (Windows Task Scheduler / cron) to keep models current.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from niftysim.config import load_config
from niftysim.models import REGISTRY
from niftysim.train import train_all

ap = argparse.ArgumentParser()
ap.add_argument("--res", default=None)
ap.add_argument("--models", nargs="*", default=None, choices=list(REGISTRY))
ap.add_argument("--no-cv", action="store_true")
ap.add_argument("--last-days", type=int, default=None)
ap.add_argument("--holdout-days", type=int, default=0)
a = ap.parse_args()
cfg = load_config()
rep = train_all(cfg, a.res, a.models, not a.no_cv, last_days=a.last_days, holdout_days=a.holdout_days)
print(f"\nrows={rep['rows']:,} days={rep['days']}  (full report: models_store/{rep['resolution']}/report.json)")
print("skill = 1 - MAE/MAE(predict zero)  (>0 beats 'no change');  hit% = direction accuracy on the 20% most confident calls")
print(f"\n{'model':14s} {'h':>4s} {'skill':>7s} {'hit%':>6s} {'cover':>6s} {'mae':>7s} | after-news: {'skill':>7s} {'hit%':>6s}")
for k, v in rep["models"].items():
    if v["cv"]:
        for h, m in v["cv"].items():
            print(f"{k:14s} {h:>4} {m['skill_vs_zero']:+7.3f} {100*m['dir_hit_top20pct']:6.1f} {m['coverage_p10_p90']:6.2f} {m['mae_bps']:7.2f} |             {m['after_news']['skill_vs_zero']:+7.3f} {100*m['after_news']['dir_hit_top20pct']:6.1f}")
