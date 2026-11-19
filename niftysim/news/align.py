"""Turn scored headlines + the event calendar into per-bar features.

A headline stamped at 10:03:20 belongs to the 10:03 bar and is first usable at that bar's close,
so the model can never peek at news before it was published.

Scheduled events go in data/events.csv (you maintain it):
    date,time,name,importance
    2026-12-05,10:00,RBI MPC decision,9
    2027-02-01,11:00,Union Budget,10
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

NEWS_FEATURES = ["news_impulse", "news_dir_decay", "news_cru_decay", "news_count_30",
                 "news_bars_since_crucial", "event_window", "event_importance"]


def _decay(x: np.ndarray, half_life: float) -> np.ndarray:
    a = 0.5 ** (1.0 / max(half_life, 1e-6))
    return lfilter([1.0], [1.0, -a], x)


def news_features(bars: pd.DataFrame, news: pd.DataFrame, bar_seconds: int,
                  half_life_bars: float = 15, events: pd.DataFrame | None = None) -> pd.DataFrame:
    idx = bars.index
    out = pd.DataFrame(index=idx)
    imp_dir = np.zeros(len(idx))
    imp_cru = np.zeros(len(idx))
    cnt = np.zeros(len(idx))
    if news is not None and len(news) and "direction" in news:
        n = news.dropna(subset=["direction"]).copy()
        ts = pd.to_datetime(n["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
        bucket = ts.dt.floor(f"{bar_seconds}s")
        pos = idx.get_indexer(bucket)
        # headlines outside session -> attach to the next bar that exists
        miss = pos < 0
        if miss.any():
            pos[miss] = np.searchsorted(idx.values, bucket[miss].values)
        ok = (pos >= 0) & (pos < len(idx))
        w = (n["direction"].values * n["cruciality"].values / 10.0)[ok]
        np.add.at(imp_dir, pos[ok], w)
        np.add.at(imp_cru, pos[ok], n["cruciality"].values[ok] / 10.0)
        np.add.at(cnt, pos[ok], 1)
    out["news_impulse"] = imp_dir
    out["news_dir_decay"] = _decay(imp_dir, half_life_bars)
    out["news_cru_decay"] = _decay(imp_cru, half_life_bars)
    out["news_count_30"] = pd.Series(cnt, index=idx).rolling(30, min_periods=1).sum().values
    crucial = np.where(imp_cru >= 0.7, 1, 0)
    last = -10_000
    since = np.empty(len(idx))
    for i, c in enumerate(crucial):
        if c:
            last = i
        since[i] = min(i - last, 500)
    out["news_bars_since_crucial"] = since / 500.0
    ew, ei = event_features(idx, events, bar_seconds)
    out["event_window"] = ew
    out["event_importance"] = ei
    return out


def load_events(path: Path) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    ev = pd.read_csv(path)
    ev["ts"] = pd.to_datetime(ev["date"].astype(str) + " " + ev["time"].astype(str)).dt.tz_localize("Asia/Kolkata")
    return ev


def event_features(idx: pd.DatetimeIndex, events: pd.DataFrame | None, bar_seconds: int,
                   before_min: int = 15, after_min: int = 45):
    """event_window: -1..+1 ramp around a scheduled event (negative before, positive after)."""
    w = np.zeros(len(idx))
    imp = np.zeros(len(idx))
    if events is None or not len(events):
        return w, imp
    t = idx.asi8 / 1e9
    for _, e in events.iterrows():
        te = e["ts"].value / 1e9
        dt = (t - te) / 60.0
        m = (dt >= -before_min) & (dt <= after_min)
        w[m] = np.where(dt[m] < 0, dt[m] / before_min, 1 - dt[m] / after_min)
        imp[m] = np.maximum(imp[m], float(e.get("importance", 5)) / 10.0)
    return w, imp
