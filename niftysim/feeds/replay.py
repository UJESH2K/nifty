"""Replay stored bars (and that day's headlines) as if they were live.

Each 1-min bar is expanded into 60 one-second ticks O -> (H/L) -> (L/H) -> C so the tick view moves
realistically. With your own recorded 1s bars the replay is exact.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pandas as pd

from ..storage import load_bars


class ReplayFeed:
    def __init__(self, path_or_df, speed: float = 60, start: str | None = None, news: pd.DataFrame | None = None):
        self.bars = path_or_df if isinstance(path_or_df, pd.DataFrame) else load_bars(path_or_df)
        if start:
            self.bars = self.bars[self.bars.index >= pd.Timestamp(start, tz="Asia/Kolkata")]
        self.news = news.sort_values("ts").to_dict("records") if news is not None and len(news) else []
        self.speed = speed
        self._stop = threading.Event()
        self.name = "replay"
        self.done = False
        self.progress = 0.0

    def set_speed(self, speed: float):
        self.speed = speed

    def _ticks_for_bar(self, ts, row, bar_s):
        if bar_s <= 1:
            return [(ts, row.close)]
        o, h, l, c = row.open, row.high, row.low, row.close
        path = [o, l, h, c] if c >= o else [o, h, l, c]
        n = int(bar_s)
        rng = np.random.default_rng(int(ts.value % 2**32))
        x = np.linspace(0, 3, n)
        prices = np.interp(x, [0, 1, 2, 3], path)
        jitter = rng.normal(0, max(h - l, 1e-6) * 0.04, n)
        prices = np.clip(prices + jitter, l, h)
        prices[0], prices[-1] = o, c
        return [(ts + pd.Timedelta(seconds=i), float(p)) for i, p in enumerate(prices)]

    def start(self, on_tick, on_news):
        bar_s = int(pd.Series(self.bars.index).diff().median().total_seconds() or 60) if len(self.bars) > 1 else 60

        def run():
            k = 0
            total = len(self.bars)
            for b, (ts, row) in enumerate(self.bars.iterrows()):
                for t, p in self._ticks_for_bar(ts, row, bar_s):
                    if self._stop.is_set():
                        return
                    while k < len(self.news) and self.news[k]["ts"] <= t:
                        on_news(self.news[k])
                        k += 1
                    on_tick(t, p)
                    time.sleep(1.0 / max(self.speed, 1e-6))
                self.progress = (b + 1) / max(total, 1)
            # push one tick past the last bar so the final bar completes
            last = self.bars.index[-1] + pd.Timedelta(seconds=bar_s)
            on_tick(last, float(self.bars["close"].iloc[-1]))
            self.done = True
        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
