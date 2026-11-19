"""Yahoo Finance polling feed — zero setup, no account, no key.

Polls Yahoo's public chart endpoint for ^NSEI every few seconds. Good enough to watch your
forecasts against the market on day one. Caveats:
  * unofficial endpoint, can change or rate-limit without notice
  * index quotes may lag the exchange by a few seconds to minutes — compare with Kite/your broker app
    for the first few minutes; switch to Angel One / Upstox for true tick-by-tick data
NOT tested from the build environment (Yahoo was unreachable there).
"""
from __future__ import annotations

import threading

import pandas as pd

TZ = "Asia/Kolkata"
URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def yahoo_chart(range_: str = "5d", interval: str = "1m") -> tuple[pd.DataFrame, dict]:
    """Returns (1-min OHLC bars in IST, meta dict)."""
    import requests

    r = requests.get(URL, params={"range": range_, "interval": interval, "includePrePost": "false"},
                     headers=HEADERS, timeout=15)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    meta = res.get("meta", {})
    ts = res.get("timestamp") or []
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]},
                      index=pd.to_datetime(ts, unit="s", utc=True).tz_convert(TZ))
    df.index.name = "ts"
    df = df.dropna()
    df["ticks"] = 1.0
    return df, meta


class YahooFeed:
    def __init__(self, poll_seconds: float = 3.0):
        self.poll = poll_seconds
        self.name = "yahoo"
        self._stop = threading.Event()
        self.last_error = None
        self.lag_seconds = 0.0

    def start(self, on_tick, on_news):
        import requests

        def run():
            while not self._stop.is_set():
                try:
                    r = requests.get(URL, params={"range": "1d", "interval": "1m"}, headers=HEADERS, timeout=10)
                    r.raise_for_status()
                    meta = r.json()["chart"]["result"][0]["meta"]
                    px = float(meta["regularMarketPrice"])
                    t = pd.Timestamp(int(meta["regularMarketTime"]), unit="s", tz="UTC").tz_convert(TZ)
                    # stamp with wall clock so bars close on time; remember Yahoo's own time to detect lag
                    now = pd.Timestamp.now(tz=TZ)
                    self.lag_seconds = max(0.0, (now - t).total_seconds())
                    on_tick(now, px)
                    self.last_error = None
                except Exception as ex:  # keep polling through hiccups
                    self.last_error = str(ex)[:120]
                self._stop.wait(self.poll)
        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        self._stop.set()
