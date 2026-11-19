"""Headless recorder: run this every market day (09:10 -> 15:35 IST) to build your OWN per-second
history, which no free source provides. Saves raw ticks, 1s bars, 1m bars and headlines.

    python scripts/record_ticks.py --source upstox
    python scripts/record_ticks.py --source angel

(The dashboard records too whenever it runs on upstox/angel — this is for when you don't need the UI.)
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from niftysim.config import load_config
from niftysim.engine import Engine

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, choices=["upstox", "angel"])
ap.add_argument("--model", default="naive", help="cheapest model is fine for pure recording")
a = ap.parse_args()
cfg = load_config()
eng = Engine(cfg, "1min", a.model, history_days=2)
eng.start(a.source)
print("recording… Ctrl+C to stop (auto-stops after 15:35 IST)")
try:
    while pd.Timestamp.now(tz="Asia/Kolkata").time() < pd.Timestamp("15:35").time():
        time.sleep(30)
        print(pd.Timestamp.now(tz="Asia/Kolkata").strftime("%H:%M:%S"), "last", eng.last_price, "bars", len(eng.bars))
finally:
    eng.stop()
    print("flushed to data/")
