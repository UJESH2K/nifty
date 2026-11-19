"""Stationary price features + targets.

Everything is expressed as returns / ratios / z-scores, never raw index levels, so a model trained
when NIFTY was 8,000 still works at 25,500.
Targets are forward log-returns in basis points over each horizon, and are masked (NaN) when the
horizon would cross the session close (no overnight leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_FEATURES = [
    "r1", "r2", "r3", "r5", "r10", "r20", "r60",
    "vol5", "vol20", "vol60", "vol_ratio", "atr14_bps", "range_bps",
    "z20", "z60", "bb_width", "rsi14", "pos_in_range60", "body_frac",
    "ticks_z", "gap_bps", "tod_sin", "tod_cos", "bars_since_open", "bars_to_close", "is_expiry",
]


def price_features(bars: pd.DataFrame, bar_seconds: int, expiry_weekday: int = 1) -> pd.DataFrame:
    c = bars["close"]
    lc = np.log(c)
    r = lc.diff() * 1e4
    day = bars.index.normalize()
    first = pd.Series(day, index=bars.index).ne(pd.Series(day, index=bars.index).shift()).values
    r[first] = np.nan          # the overnight gap is not a 1-minute move; it goes to gap_bps instead
    f = pd.DataFrame(index=bars.index)
    for k in (1, 2, 3, 5, 10, 20, 60):
        f[f"r{k}"] = (lc - lc.shift(k)) * 1e4
    for w in (5, 20, 60):
        f[f"vol{w}"] = r.rolling(w, min_periods=2).std()
    f["vol_ratio"] = f["vol5"] / (f["vol60"] + 1e-9)
    pc = c.shift().where(~first)   # no previous close across the overnight gap
    tr = pd.concat([bars["high"] - bars["low"],
                    (bars["high"] - pc).abs(),
                    (bars["low"] - pc).abs()], axis=1).max(axis=1)
    f["atr14_bps"] = tr.rolling(14, min_periods=1).mean() / c * 1e4
    f["range_bps"] = (bars["high"] - bars["low"]) / c * 1e4
    for w in (20, 60):
        m = c.rolling(w, min_periods=2).mean()
        s = c.rolling(w, min_periods=2).std()
        f[f"z{w}"] = (c - m) / (s + 1e-9)
    f["bb_width"] = 4 * c.rolling(20, min_periods=2).std() / c * 1e4
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi14"] = 100 - 100 / (1 + up / (dn + 1e-9))
    hi = bars["high"].rolling(60, min_periods=1).max()
    lo = bars["low"].rolling(60, min_periods=1).min()
    f["pos_in_range60"] = (c - lo) / (hi - lo + 1e-9)
    f["body_frac"] = (c - bars["open"]) / (bars["high"] - bars["low"] + 1e-9)
    t = bars["ticks"]
    f["ticks_z"] = (t - t.rolling(60, min_periods=2).mean()) / (t.rolling(60, min_periods=2).std() + 1e-9)

    new_day = pd.Series(first, index=bars.index)
    gap = (bars["open"] / c.shift() - 1) * 1e4
    f["gap_bps"] = gap.where(new_day).ffill().fillna(0)
    sec = (bars.index.hour * 3600 + bars.index.minute * 60 + bars.index.second) - (9 * 3600 + 15 * 60)
    frac = np.clip(sec / (375 * 60), 0, 1)
    f["tod_sin"] = np.sin(2 * np.pi * frac)
    f["tod_cos"] = np.cos(2 * np.pi * frac)
    f["bars_since_open"] = sec / bar_seconds / (375 * 60 / bar_seconds)
    f["bars_to_close"] = 1 - f["bars_since_open"]
    f["is_expiry"] = (bars.index.dayofweek == expiry_weekday).astype(float)
    return f.replace([np.inf, -np.inf], np.nan)


def targets(bars: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    c = bars["close"]
    lc = np.log(c)
    day = pd.Series(bars.index.normalize(), index=bars.index)
    out = pd.DataFrame(index=bars.index)
    for h in horizons:
        y = (lc.shift(-h) - lc) * 1e4
        same = day.shift(-h) == day
        out[f"y{h}"] = y.where(same)
        # max favourable excursions (for the "did it reach 1/3 of the projection?" scorecard)
        fh = bars["high"][::-1].rolling(h, min_periods=1).max()[::-1].shift(-1)
        fl = bars["low"][::-1].rolling(h, min_periods=1).min()[::-1].shift(-1)
        out[f"mfe_up{h}"] = ((fh - c)).where(same)
        out[f"mfe_dn{h}"] = ((c - fl)).where(same)
    return out


def build_dataset(bars, news_feats, horizons, bar_seconds, expiry_weekday=1):
    pf = price_features(bars, bar_seconds, expiry_weekday)
    X = pf if news_feats is None else pd.concat([pf, news_feats], axis=1)
    Y = targets(bars, horizons)
    return X, Y
