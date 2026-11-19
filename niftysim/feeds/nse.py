"""NSE website feed — the exchange's own public numbers. No account, no demat, no key.

Polls the same JSON the nseindia.com pages use. NSE expects browser-like requests and cookies, so
we open the homepage first and refresh cookies when they expire. Personal use only: we poll every
few seconds, like a browser tab would. If NSE blocks the requests, the feed switches to Yahoo
automatically (see AutoFeed).
NOT tested from the build environment (nseindia.com was unreachable there).
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd

TZ = "Asia/Kolkata"
BASE = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/live-market-indices",
}


class NSESession:
    def __init__(self):
        import requests
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self._warm = False

    def warm(self):
        self.s.get(BASE + "/", timeout=10)
        self.s.get(BASE + "/market-data/live-market-indices", timeout=10)
        self._warm = True

    def get_json(self, path, params=None):
        if not self._warm:
            self.warm()
        r = self.s.get(BASE + path, params=params, timeout=10)
        if r.status_code in (401, 403):
            self.warm()
            r = self.s.get(BASE + path, params=params, timeout=10)
        r.raise_for_status()
        return r.json()

    def nifty_last(self) -> float:
        js = self.get_json("/api/allIndices")
        for row in js.get("data", []):
            if str(row.get("index", "")).upper() == "NIFTY 50":
                return float(row["last"])
        raise RuntimeError("NIFTY 50 not found in NSE response")

    def intraday_points(self) -> pd.Series:
        js = self.get_json("/api/chart-databyindex", {"index": "NIFTY 50", "indices": "true"})
        pts = js.get("grapthData") or js.get("graphData") or []
        return parse_points(pts)


def parse_points(pts) -> pd.Series:
    """NSE intraday chart points [[epoch_ms, price], ...]. NSE sometimes encodes IST wall-clock time as
    if it were UTC; pick whichever reading puts the points inside market hours."""
    if not pts:
        return pd.Series(dtype=float)
    a = np.array([[p[0], p[1]] for p in pts if p and p[1] is not None], dtype=float)
    t_utc = pd.to_datetime(a[:, 0], unit="ms", utc=True).tz_convert(TZ)
    t_ist = pd.to_datetime(a[:, 0], unit="ms").tz_localize(TZ)

    def dist_from_open(t):  # the series starts at (or just after) the 09:15 open
        m = t[0].hour * 60 + t[0].minute
        return abs(m - 555)

    t = t_utc if dist_from_open(t_utc) <= dist_from_open(t_ist) else t_ist
    return pd.Series(a[:, 1], index=t).sort_index()


def nse_today_bars() -> pd.DataFrame:
    s = NSESession().intraday_points()
    if s.empty:
        return pd.DataFrame()
    b = s.resample("1min", label="left", closed="left").agg(["first", "max", "min", "last", "count"]).dropna()
    b.columns = ["open", "high", "low", "close", "ticks"]
    b.index.name = "ts"
    return b


class NSEFeed:
    def __init__(self, poll_seconds: float = 3.0):
        self.poll = poll_seconds
        self.name = "nse"
        self._stop = threading.Event()
        self.last_error = None
        self.lag_seconds = 0.0
        self.fail_streak = 0

    def start(self, on_tick, on_news):
        self.sess = NSESession()

        def run():
            while not self._stop.is_set():
                try:
                    px = self.sess.nifty_last()
                    on_tick(pd.Timestamp.now(tz=TZ), px)
                    self.last_error, self.fail_streak = None, 0
                except Exception as ex:
                    self.last_error = str(ex)[:120]
                    self.fail_streak += 1
                self._stop.wait(self.poll)
        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        self._stop.set()


class AutoFeed:
    """NSE website first; if it fails 5 times in a row, switch to Yahoo for the rest of the session."""

    def __init__(self, poll_seconds: float = 3.0):
        from .yahoo import YahooFeed
        self.nse = NSEFeed(poll_seconds)
        self.yahoo = YahooFeed(poll_seconds)
        self.active = self.nse
        self.name = "auto"
        self._stop = threading.Event()

    @property
    def last_error(self):
        return self.active.last_error

    @property
    def lag_seconds(self):
        return getattr(self.active, "lag_seconds", 0.0)

    @property
    def using(self):
        return "NSE website" if self.active is self.nse else "Yahoo (NSE was blocked)"

    def start(self, on_tick, on_news):
        self.nse.start(on_tick, on_news)

        def watch():
            while not self._stop.is_set():
                if self.active is self.nse and self.nse.fail_streak >= 5:
                    self.nse.stop()
                    self.active = self.yahoo
                    self.yahoo.start(on_tick, on_news)
                self._stop.wait(2)
        threading.Thread(target=watch, daemon=True).start()

    def stop(self):
        self._stop.set()
        self.nse.stop()
        self.yahoo.stop()
