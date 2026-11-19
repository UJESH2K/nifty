"""Blind replay: train a model that has never seen day D, then replay D minute-by-minute and watch
how its live predictions compare with what really happened.
"""
from __future__ import annotations

import threading

import pandas as pd

from .config import data_dir
from .engine import Engine
from .feeds.replay import ReplayFeed
from .market_clock import MarketClock
from .storage import load_bars, load_news
from .train import train_all

TZ = "Asia/Kolkata"


def available_days(cfg) -> list[pd.Timestamp]:
    p = data_dir(cfg) / "bars_1min.parquet"
    if not p.exists():
        return []
    b = load_bars(p)
    counts = b.groupby(b.index.normalize()).size()
    days = list(counts[counts >= 300].index)       # full sessions only
    return days[10:]                                # need some history before the day


class BlindSession:
    """state: idle -> training -> replaying -> done | error"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = "idle"
        self.msg = ""
        self.engine: Engine | None = None
        self.feed: ReplayFeed | None = None
        self.day = None

    def start(self, day, model="ridge", speed=60.0):
        self.stop()
        self.day = pd.Timestamp(day).tz_localize(TZ) if pd.Timestamp(day).tzinfo is None else pd.Timestamp(day)
        o, c = MarketClock(self.cfg).session(self.day)
        self.state, self.msg = "training", f"Training {model} only on data before {self.day:%a %d %b}…"

        def job():
            try:
                train_all(self.cfg, "1min", [model], do_cv=False, log=lambda *_: None, until=o, subdir="blind")
                eng = Engine(self.cfg, "1min", model, history_days=10, history_until=o, models_subdir="blind")
                d = data_dir(self.cfg)
                bars = load_bars(d / "bars_1min.parquet")
                day_bars = bars[(bars.index >= o) & (bars.index < c)]
                news = load_news(d / "news.parquet")
                day_news = news[(news["ts"] >= o) & (news["ts"] < c)] if len(news) else None
                self.feed = ReplayFeed(day_bars, speed=speed, news=day_news)
                eng.feed = self.feed
                eng.live_source = "replay"
                eng.status = "replaying"
                self.engine = eng
                self.state, self.msg = "replaying", f"Replaying {self.day:%a %d %b} — the model has never seen this day"
                self.feed.start(eng.on_tick, eng.on_news)
                threading.Thread(target=self._watch, daemon=True).start()
            except Exception as ex:
                self.state, self.msg = "error", f"{type(ex).__name__}: {ex}"
        threading.Thread(target=job, daemon=True).start()

    def _watch(self):
        import time
        while self.feed and not self.feed.done and self.state == "replaying":
            time.sleep(0.5)
        if self.state == "replaying":
            self.state, self.msg = "done", f"Finished replaying {self.day:%a %d %b}"

    def set_speed(self, speed):
        if self.feed:
            self.feed.set_speed(speed)

    def stop(self):
        if self.feed:
            self.feed.stop()
        self.state = "idle"
