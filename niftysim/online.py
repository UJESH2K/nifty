"""Fast layer: self-correction every bar WITHOUT touching model weights.

Why not retrain the model every minute? Online weight updates on 1-min data cause catastrophic
forgetting (30 minutes of chop erases what 10 years of data taught). Instead we keep the trained
model frozen intraday and learn two small things from every prediction that matures:

  bias  : EWMA of (actual - predicted median)   -> OFF by default: its feedback arrives h bars late,
                                                   so at 30m-3h it ends up chasing stale trends
  scale : adaptive conformal inference          -> widens/narrows the p10..p90 band so it really
                                                   contains ~80% of outcomes in the CURRENT regime
The weights themselves are refreshed by the slow layer (scripts/train_models.py, nightly).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np


class OnlineCalibrator:
    def __init__(self, target_coverage=0.8, gamma=0.02, bias_halflife=60, window=300, use_bias=False):
        self.use_bias = use_bias
        self.target = target_coverage
        self.gamma = gamma
        self.alpha = 1 - 0.5 ** (1 / bias_halflife)
        self.bias = 0.0
        self.scale = 1.0
        self.hist = deque(maxlen=window)   # (abs_err_raw, abs_err_corr, hit_corr, covered, strong)

    def apply(self, q: np.ndarray) -> np.ndarray:
        """q = [p10, p50, p90] raw (bps) -> corrected."""
        lo, med, hi = q
        m2 = med + (self.bias if self.use_bias else 0.0)
        return np.array([m2 - self.scale * (med - lo), m2, m2 + self.scale * (hi - med)])

    def update(self, raw: np.ndarray, corr: np.ndarray, y: float):
        covered = corr[0] <= y <= corr[2]
        self.scale *= math.exp(self.gamma * ((0 if covered else 1) - (1 - self.target)))
        self.scale = float(min(max(self.scale, 0.5), 2.5))
        self.bias = (1 - self.alpha) * self.bias + self.alpha * (y - raw[1])
        self.hist.append((abs(y - raw[1]), abs(y - corr[1]), float(np.sign(corr[1]) == np.sign(y)),
                          float(covered), abs(y)))

    def new_session(self):
        """Halfway back to neutral at each open: yesterday's regime shouldn't fully carry over."""
        self.scale = 1.0 + (self.scale - 1.0) * 0.5

    def stats(self) -> dict:
        if not self.hist:
            return {"n": 0}
        a = np.array(self.hist)
        return {"n": len(a), "mae_raw": a[:, 0].mean(), "mae_corr": a[:, 1].mean(),
                "mae_zero": a[:, 4].mean(), "dir_hit": a[:, 2].mean(), "coverage": a[:, 3].mean(),
                "bias": self.bias, "scale": self.scale}
