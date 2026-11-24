"""Path risk: how far the market is likely to RISE and to DIP along the way within the next h minutes,
and how often a target is reached before a stop.

Why: a forecast of "+50 in 3 hours" says nothing about the -30 dip that often comes first. Unlike
direction, the size of these swings IS predictable, because volatility clusters (a jumpy morning
tends to stay jumpy). Everything here is learned from your own minute history.

  ExcursionModel.predict(features, h) -> {"up50","up80","dn50","dn80"} in bps
      up50 = the typical (median) best rise within h minutes
      dn80 = the dip that 80% of the time is NOT exceeded within h minutes (a stop-loss candidate)
  first_passage(kT, kS, h, side) -> share of past cases where a +T target was hit before a -S stop
      (T, S in units of the h-minute volatility), measured on the real path minute by minute.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EX_H = [5, 15, 30, 60, 120, 180]
EX_FEATURES = ["vol5", "vol20", "vol60", "atr14_bps", "range_bps", "vol_ratio", "tod_sin", "tod_cos",
               "bars_to_close", "gap_bps", "is_expiry"]
GRID = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])


def _sig(X, h):
    v = X["vol20"].fillna(X["vol20"].median()).clip(lower=0.5).values
    return v * np.sqrt(h)


class ExcursionModel:
    def __init__(self, horizons=EX_H):
        self.H = list(horizons)

    # ---------------------------------------------------------------- fit
    def fit(self, X: pd.DataFrame, bars: pd.DataFrame, max_rows: int = 400_000, seed: int = 0):
        from sklearn.linear_model import Ridge
        X = X[X["vol60"].notna()]
        if len(X) > max_rows:
            X = X.sample(max_rows, random_state=seed).sort_index()
        close = bars["close"].reindex(X.index).values
        hi, lo = bars["high"].values, bars["low"].values
        pos = bars.index.get_indexer(X.index)
        day = bars.index.normalize().values
        self.mu, self.sd = X[EX_FEATURES].mean(), X[EX_FEATURES].std().replace(0, 1)
        Z = ((X[EX_FEATURES] - self.mu) / self.sd).fillna(0).clip(-8, 8).values
        self.m, self.rq, self.fp = {}, {}, {}
        n = len(bars)
        for h in self.H:
            end = np.minimum(pos + h, n - 1)
            same = day[end] == day[pos]
            up = np.full(len(pos), np.nan)
            dn = np.full(len(pos), np.nan)
            # running max/min over the next h bars, vectorised by stepping through the horizon
            run_hi = np.full(len(pos), -np.inf)
            run_lo = np.full(len(pos), np.inf)
            for k in range(1, h + 1):
                j = np.minimum(pos + k, n - 1)
                run_hi = np.maximum(run_hi, hi[j])
                run_lo = np.minimum(run_lo, lo[j])
            up[same] = np.log(run_hi[same] / close[same]) * 1e4
            dn[same] = np.log(close[same] / run_lo[same]) * 1e4
            s = _sig(X, h)
            for side, y in (("up", up), ("dn", dn)):
                ok = np.isfinite(y) & (y > -1)
                t = np.log(np.maximum(y[ok], 0.25) / s[ok])          # size in volatility units, log scale
                r = Ridge(alpha=10.0).fit(Z[ok], t)
                res = t - r.predict(Z[ok])
                self.m[(h, side)] = r
                self.rq[(h, side)] = np.quantile(res, [0.5, 0.8])
            self.fp[h] = self._first_passage_table(bars, h, seed)
        return self

    def _first_passage_table(self, bars, h, seed, n_starts=6000):
        """P(target first) on a grid of target/stop sizes, from real minute paths."""
        rng = np.random.default_rng(seed + h)
        c, hi, lo = bars["close"].values, bars["high"].values, bars["low"].values
        day = bars.index.normalize().values
        r = np.diff(np.log(c), prepend=np.log(c[0])) * 1e4
        vol20 = pd.Series(r).rolling(20, min_periods=10).std().values
        n = len(c)
        cand = np.where((np.arange(n) + h < n) & np.isfinite(vol20) & (vol20 > 0))[0]
        cand = cand[day[np.minimum(cand + h, n - 1)] == day[cand]]
        starts = rng.choice(cand, size=min(n_starts, len(cand)), replace=False)
        sig = vol20[starts] * np.sqrt(h)                                     # bps
        idx = starts[:, None] + np.arange(1, h + 1)[None, :]
        up = np.log(hi[idx] / c[starts][:, None]) * 1e4 / sig[:, None]        # in sigma units
        dn = np.log(c[starts][:, None] / lo[idx]) * 1e4 / sig[:, None]
        tab = {}
        for side in ("long", "short"):
            fav, adv = (up, dn) if side == "long" else (dn, up)
            T = np.zeros((len(GRID), len(GRID)))
            S = np.zeros((len(GRID), len(GRID)))
            for a, kT in enumerate(GRID):
                hitT = fav >= kT
                tT = np.where(hitT.any(1), hitT.argmax(1), h + 1)
                for b_, kS in enumerate(GRID):
                    hitS = adv >= kS
                    tS = np.where(hitS.any(1), hitS.argmax(1), h + 1)
                    T[a, b_] = np.mean(tT < tS)                 # target strictly first
                    S[a, b_] = np.mean((tS <= tT) & (tS <= h))  # stop first (same minute = stop: cautious)
            tab[side] = T
            tab[side + "_stop"] = S
        return tab

    # ---------------------------------------------------------------- predict
    def _hk(self, h):
        return min(self.H, key=lambda x: abs(np.log(x) - np.log(max(h, 1))))

    def predict(self, Xrow: pd.DataFrame, h: float) -> dict:
        hk = self._hk(h)
        Z = ((Xrow[EX_FEATURES] - self.mu) / self.sd).fillna(0).clip(-8, 8).values
        s = _sig(Xrow, h)[0]
        out = {}
        for side in ("up", "dn"):
            base = self.m[(hk, side)].predict(Z)[0]
            q50, q80 = self.rq[(hk, side)]
            # scale from the fitted horizon to the asked one by sqrt(time)
            out[f"{side}50"] = float(np.exp(base + q50) * s)
            out[f"{side}80"] = float(np.exp(base + q80) * s)
        out["sigma"] = float(s)
        return out

    def outcomes(self, h, target_bps, stop_bps, sigma_bps, side="long") -> dict:
        """Share of past cases: target first / stop first / neither within h."""
        t = self.p_target_first(h, target_bps, stop_bps, sigma_bps, side)
        s = self.p_target_first(h, target_bps, stop_bps, sigma_bps, side + "_stop")
        return {"target": t, "stop": s, "neither": max(0.0, 1 - t - s)}

    def p_target_first(self, h, target_bps, stop_bps, sigma_bps, side="long"):
        tab = self.fp[self._hk(h)][side]
        kT, kS = target_bps / sigma_bps, stop_bps / sigma_bps
        a = np.interp(kT, GRID, np.arange(len(GRID)))
        b = np.interp(kS, GRID, np.arange(len(GRID)))
        a0, b0 = int(np.floor(a)), int(np.floor(b))
        a1, b1 = min(a0 + 1, len(GRID) - 1), min(b0 + 1, len(GRID) - 1)
        fa, fb = a - a0, b - b0
        v = (tab[a0, b0] * (1 - fa) * (1 - fb) + tab[a1, b0] * fa * (1 - fb) +
             tab[a0, b1] * (1 - fa) * fb + tab[a1, b1] * fa * fb)
        return float(v)
