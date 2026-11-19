"""Autopilot: runs NiftySim every trading day with nobody touching it.

Daily loop
  1. wait for the open (holidays and weekends are skipped by the market clock)
  2. go live: backfill today's bars, predict every minute, save every tick / bar / headline
  3. after the close: score today's predictions (saved to data/daily_scores.csv so you can see
     whether it is really getting better over time)
  4. retrain the model on all data including today (the "slow layer"), then reload it
  5. sleep until the next open and repeat

During the day the forecast ranges also self-correct minute by minute (the "fast layer").
"""
from __future__ import annotations

import threading
import time
import traceback

import pandas as pd

from .config import data_dir

TZ = "Asia/Kolkata"


class Autopilot:
    def __init__(self, cfg, engine, source="auto", train_days=None, log=print):
        self.cfg = cfg
        self.eng = engine
        self.source = source
        self.train_days = train_days
        self.log = log
        self.state = "starting"
        self.last_retrain = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ the loop
    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()
        self.eng.disarm()

    def _run(self):
        while not self._stop.is_set():
            try:
                st = self.eng.clock.status()
                if not st["open"]:
                    self.state = f"waiting for the open ({st['next_open']:%a %d %b %H:%M})"
                self.eng.go_live(self.source)          # waits by itself if the market is closed
                self._wait_for_close()
                if self._stop.is_set():
                    return
                self.state = "session over: saving and scoring today"
                time.sleep(90)                          # let the last bars complete
                self.eng.stop()
                self.eng.armed = False
                day = pd.Timestamp.now(tz=TZ).normalize()
                self.score_day(day)
                self.retrain()
            except Exception as ex:
                self.state = f"error: {ex} (retrying in 5 min)"
                self.log(traceback.format_exc())
                self.eng.armed = False
                self._stop.wait(300)

    def _wait_for_close(self):
        # wait until the market has opened (go_live is waiting on its own) ...
        while not self._stop.is_set() and not self.eng.status.startswith("live"):
            if not self.eng.clock.status()["open"]:
                st = self.eng.clock.status()
                self.state = f"waiting for the open ({st['next_open']:%a %d %b %H:%M})"
            self._stop.wait(10)
        self.state = "live"
        # ... then until it closes
        while not self._stop.is_set() and self.eng.clock.status()["open"]:
            self._stop.wait(10)

    # ------------------------------------------------------------------ after the close
    def score_day(self, day):
        """Append today's live accuracy per horizon to data/daily_scores.csv."""
        rows = []
        for v in ("price", "news"):
            for h in self.eng.H:
                lc = self.eng.live_check(v, h, since=day)
                if lc.get("n"):
                    rows.append({"date": day.date(), "model": self.eng.model_name, "variant": v, "horizon": h,
                                 "checked": lc["n"], "miss_pts": round(lc["miss_pts"], 2),
                                 "nochange_miss_pts": round(lc["nochange_miss_pts"], 2),
                                 "dir_right": None if pd.isna(lc["dir_right"]) else round(lc["dir_right"], 3),
                                 "in_range": round(lc["in_range"], 3)})
        if rows:
            p = data_dir(self.cfg) / "daily_scores.csv"
            df = pd.DataFrame(rows)
            df.to_csv(p, mode="a", header=not p.exists(), index=False)
            self.log(f"scored {day:%d %b}: {len(rows)} rows -> {p}")

    def retrain(self):
        from .train import train_all
        self.state = "retraining on all data including today"
        self.log(self.state)
        train_all(self.cfg, self.eng.res, [self.eng.model_name], do_cv=False, log=self.log,
                  last_days=self.train_days)
        with self.eng.lock:
            self.eng.message = ""
            self.eng.signals_log = []
            self.eng.pending.clear()
            self.eng._load_history(10)
            self.eng.set_model(self.eng.model_name)
        self.last_retrain = pd.Timestamp.now(tz=TZ)
        self.state = f"retrained at {self.last_retrain:%H:%M}; waiting for the next open"
        self.log(self.state)


def load_daily_scores(cfg) -> pd.DataFrame:
    p = data_dir(cfg) / "daily_scores.csv"
    return pd.read_csv(p, parse_dates=["date"]) if p.exists() else pd.DataFrame()
