"""Synthetic NIFTY-like market with news shocks.

Purpose: lets you run, train and test the WHOLE system offline (nights, weekends, no API keys).
Properties built in on purpose:
  * U-shaped intraday volatility (busy open/close, quiet lunch)
  * volatility clustering (GARCH-like regimes)
  * fat tails (Student-t shocks)
  * random headlines whose text carries their direction & importance, which push price
    for 5-25 minutes afterwards  -> a news-aware model SHOULD beat a price-only one here.
  * no hidden directional pattern in price alone (like the real market at 1-min scale).

Never treat results on synthetic data as evidence about the real market.
"""
from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TZ = "Asia/Kolkata"

POS = [
    ("RBI unexpectedly cuts repo rate by 25 bps, banks rally", "rbi", 9),
    ("FIIs turn net buyers, pump Rs {n} crore into Indian equities", "flows", 6),
    ("India GDP growth beats estimates at {x}%", "macro", 7),
    ("Crude oil slumps {x}% easing inflation worries for India", "commodities", 6),
    ("US Fed signals rate cuts, Asian markets surge", "global", 7),
    ("Government announces Rs {n} lakh crore infrastructure boost", "policy", 7),
    ("HDFC Bank, Reliance lead rally after strong quarterly earnings beat", "earnings", 5),
    ("India-US trade deal signed, tariffs slashed; exporters gain", "geopolitics", 8),
]
NEG = [
    ("RBI hikes repo rate by 25 bps in surprise move, markets slump", "rbi", 9),
    ("FIIs dump Rs {n} crore of Indian shares amid global selloff", "flows", 6),
    ("India CPI inflation jumps to {x}%, above RBI target", "macro", 7),
    ("Crude oil spikes {x}% on Middle East war fears", "commodities", 6),
    ("US Fed warns of higher rates for longer, global stocks tumble", "global", 7),
    ("Border tensions escalate, investors flee risky assets", "geopolitics", 8),
    ("SEBI tightens F&O rules; brokers fall sharply", "regulation", 6),
    ("IT majors cut guidance, earnings miss drags Nifty lower", "earnings", 5),
]
NEU = [
    ("Sensex, Nifty open flat ahead of weekly expiry", "market", 2),
    ("Rupee steady against dollar in early trade", "fx", 2),
    ("Markets trade rangebound as traders await cues", "market", 2),
    ("Analysts await quarterly results season", "earnings", 2),
]


def _fmt(t: str, rng: random.Random) -> str:
    return t.format(n=rng.randint(800, 9000), x=round(rng.uniform(1.5, 8.5), 1))


@dataclass
class _Impact:
    t0: float          # seconds since session start
    drift_pts: float   # total points it will move price
    tau: float         # seconds (decay constant)
    vol_mult: float


@dataclass
class SyntheticMarket:
    start_price: float = 25500.0
    seed: int = 7
    news_per_day: float = 7.0
    base_vol_per_min_bps: float = 2.6     # ~ typical 1-min NIFTY std in bps
    rng: random.Random = field(init=False)
    np_rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.price = self.start_price
        self.regime = 1.0
        self.impacts: list[_Impact] = []

    # --- intraday U-shape: 1.8x at open, 0.7x at lunch, 1.4x near close
    @staticmethod
    def _u(sec_of_session: float) -> float:
        x = sec_of_session / (375 * 60)
        return 0.7 + 1.1 * math.exp(-x * 12) + 0.7 * math.exp(-(1 - x) * 10)

    def _maybe_news(self, ts: pd.Timestamp, s: float, dt: float, out: list):
        lam = self.news_per_day / (375 * 60) * dt
        if self.rng.random() < lam:
            r = self.rng.random()
            if r < 0.4:
                tpl, cat, cru = self.rng.choice(POS); sign = 1
            elif r < 0.8:
                tpl, cat, cru = self.rng.choice(NEG); sign = -1
            else:
                tpl, cat, cru = self.rng.choice(NEU); sign = 0
            cru = max(1, min(10, cru + self.rng.choice([-1, 0, 0, 1])))
            if sign:
                pts = sign * cru * self.rng.uniform(5, 11)          # 30..110 pts
                self.impacts.append(_Impact(s, pts, self.rng.uniform(300, 1200), 1 + cru / 3))
            out.append({"ts": ts, "source": "synthetic", "headline": _fmt(tpl, self.rng)})

    def step(self, ts: pd.Timestamp, sec_of_session: float, dt: float, news_out: list) -> float:
        """Advance dt seconds. Returns new price."""
        self._maybe_news(ts, sec_of_session, dt, news_out)
        # regime (vol clustering), mean-reverting log-AR(1) updated continuously
        k = dt / 1800.0
        self.regime = math.exp((1 - k) * math.log(self.regime) + math.sqrt(2 * k) * 0.25 * self.np_rng.standard_normal())
        drift, vm = 0.0, 1.0
        alive = []
        for im in self.impacts:
            age = sec_of_session - im.t0
            if age < 4 * im.tau:
                # fraction of the total move delivered in this dt
                drift += im.drift_pts * (math.exp(-age / im.tau) - math.exp(-(age + dt) / im.tau))
                vm += (im.vol_mult - 1) * math.exp(-age / im.tau)
                alive.append(im)
        self.impacts = alive
        sigma = self.price * self.base_vol_per_min_bps * 1e-4 * math.sqrt(dt / 60) * self._u(sec_of_session) * self.regime * vm
        shock = self.np_rng.standard_t(4) / math.sqrt(2)  # unit variance t(4)
        self.price = max(1.0, self.price + drift + sigma * shock)
        return self.price

    def new_day(self):
        self.impacts = []
        gap = self.np_rng.normal(0, 0.004) * self.price
        self.price += gap


def session_seconds(day: pd.Timestamp, step: float = 1.0):
    start = pd.Timestamp(f"{day.date()} 09:15", tz=TZ)
    n = int(375 * 60 / step)
    for i in range(n):
        yield start + pd.Timedelta(seconds=i * step), i * step


def generate_history(days: int = 60, resolution_s: int = 1, seed: int = 7,
                     end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate `days` trading days of 1-second ticks -> returns (bars_1s, news)."""
    mkt = SyntheticMarket(seed=seed)
    end_day = pd.Timestamp(end or pd.Timestamp.now(tz=TZ).normalize()) - pd.Timedelta(days=1)
    bdays = pd.bdate_range(end=end_day.tz_localize(None) if end_day.tz else end_day, periods=days)
    rows, news = [], []
    for d in bdays:
        mkt.new_day()
        prev = mkt.price
        for ts, s in session_seconds(pd.Timestamp(d), resolution_s):
            o = prev
            p = mkt.step(ts, s, resolution_s, news)
            rows.append((ts, o, max(o, p), min(o, p), p, 1.0))
            prev = p
    bars = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "ticks"]).set_index("ts")
    return bars, pd.DataFrame(news)


class SyntheticFeed:
    """Live tick stream from the synthetic market, accelerated by `speed`."""

    def __init__(self, speed: float = 60, seed: int = 99, start_price: float | None = None,
                 tick_seconds: float = 1.0):
        self.mkt = SyntheticMarket(seed=seed, start_price=start_price or 25500.0)
        self.speed = speed
        self.tick_seconds = tick_seconds
        self._stop = threading.Event()
        self.name = "synthetic"

    def start(self, on_tick, on_news, start_day: pd.Timestamp | None = None):
        def run():
            day = start_day or pd.Timestamp.now(tz=TZ).normalize()
            while not self._stop.is_set():
                if day.dayofweek < 5:
                    self.mkt.new_day()
                    for ts, s in session_seconds(day, self.tick_seconds):
                        if self._stop.is_set():
                            return
                        news = []
                        p = self.mkt.step(ts, s, self.tick_seconds, news)
                        for n in news:
                            on_news(n)
                        on_tick(ts, p)
                        time.sleep(self.tick_seconds / max(self.speed, 1e-6))
                day = day + pd.Timedelta(days=1)
        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop.set()
