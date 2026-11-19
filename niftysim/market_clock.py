"""NSE market clock: is the market open, and when does it next open?"""
from __future__ import annotations

import pandas as pd

TZ = "Asia/Kolkata"


class MarketClock:
    def __init__(self, cfg: dict):
        m = cfg["market"]
        self.open_t = pd.Timestamp(m["open"]).time()
        self.close_t = pd.Timestamp(m["close"]).time()
        self.holidays = {pd.Timestamp(d).date() for d in m.get("holidays", [])}
        self.special = {pd.Timestamp(d).date() for d in m.get("special_sessions", [])}

    @staticmethod
    def now() -> pd.Timestamp:
        return pd.Timestamp.now(tz=TZ)

    def is_trading_day(self, d) -> bool:
        d = pd.Timestamp(d).date()
        if d in self.special:
            return True
        return pd.Timestamp(d).dayofweek < 5 and d not in self.holidays

    def session(self, d):
        d = pd.Timestamp(d).date()
        return (pd.Timestamp.combine(d, self.open_t).tz_localize(TZ),
                pd.Timestamp.combine(d, self.close_t).tz_localize(TZ))

    def is_open(self, t: pd.Timestamp | None = None) -> bool:
        t = t or self.now()
        if not self.is_trading_day(t):
            return False
        o, c = self.session(t)
        return o <= t < c

    def next_open(self, t: pd.Timestamp | None = None) -> pd.Timestamp:
        t = t or self.now()
        d = t.normalize()
        for _ in range(15):
            if self.is_trading_day(d):
                o, _c = self.session(d)
                if t < o:
                    return o
            d = d + pd.Timedelta(days=1)
        raise RuntimeError("no trading day in the next 2 weeks?")

    def last_session_day(self, t: pd.Timestamp | None = None) -> pd.Timestamp:
        t = t or self.now()
        d = t.normalize()
        o, c = self.session(d)
        if self.is_trading_day(d) and t >= c:
            return d
        for _ in range(15):
            d = d - pd.Timedelta(days=1)
            if self.is_trading_day(d):
                return d
        return d

    def status(self, t: pd.Timestamp | None = None) -> dict:
        t = t or self.now()
        if self.is_open(t):
            _o, c = self.session(t)
            return {"open": True, "label": "Market open", "closes": c, "secs_to_close": (c - t).total_seconds()}
        nxt = self.next_open(t)
        reason = ""
        if t.date() in self.holidays:
            reason = "Trading holiday"
        elif t.dayofweek >= 5:
            reason = "Weekend"
        elif t.time() < self.open_t:
            reason = "Pre-market"
        else:
            reason = "Market closed"
        return {"open": False, "label": reason, "next_open": nxt, "secs_to_open": (nxt - t).total_seconds()}


def fmt_countdown(secs: float) -> str:
    secs = max(0, int(secs))
    d, r = divmod(secs, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return (f"{d}d " if d else "") + f"{h:02d}:{m:02d}:{s:02d}"
