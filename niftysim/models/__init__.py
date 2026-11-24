"""Model registry. Every model predicts, for each horizon h, the quantiles (p10, p50, p90) of the
forward log-return in basis points. The dashboard converts that to index points anchored on the
live price, so the projection can never drift away from the real market.

Add your own model: subclass ForecastModel, implement _fit/_predict, add it to REGISTRY.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd


# Bump when the model maths changes, so servers retrain instead of loading stale pickles.
CODE_VERSION = 3


class ForecastModel:
    name = "base"
    description = ""

    def __init__(self, horizons, quantiles=(0.1, 0.5, 0.9), features=None):
        self.horizons = list(horizons)
        self.quantiles = list(quantiles)
        self.features = list(features) if features is not None else None

    shrink = True   # learn how much to trust the directional call, per horizon, on unseen days

    # ---- public API
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        """Two passes. (1) fit on the first 75% of days, predict the last 25%, and per horizon find the
        factor s in [0, 1] that best scales the predicted median (0 = 'no change', 1 = trust fully).
        (2) refit on everything. Long horizons with too little data automatically fall back to ~0."""
        self.features = self.features or list(X.columns)
        Xf = X[self.features]
        self.shrink_ = {h: 1.0 for h in self.horizons}
        days = Xf.index.normalize()
        ud = days.unique()
        if self.shrink and len(ud) >= 8:
            cut = ud[int(len(ud) * 0.75)]
            tr, te = days < cut, days >= cut
            self._fit(Xf[tr], Y[tr])
            P = self._predict(Xf[te])
            grid = np.linspace(0, 1, 21)
            for h in self.horizons:
                y = Y.loc[te, f"y{h}"].values
                ok = ~np.isnan(y)
                if ok.sum() < 50:
                    continue
                med = P[h][ok, 1]
                errs = [np.mean(np.abs(y[ok] - g * med)) for g in grid]
                self.shrink_[h] = float(grid[int(np.argmin(errs))])
        self._fit(Xf, Y)
        return self

    def predict(self, X: pd.DataFrame) -> dict[int, np.ndarray]:
        out = self._predict(X[self.features])
        for h in out:
            s = getattr(self, "shrink_", {}).get(h, 1.0)
            if s != 1.0:
                out[h] = out[h] + ((s - 1.0) * out[h][:, 1])[:, None]
            out[h] = np.sort(out[h], axis=1)  # quantiles must not cross
        return out

    def save(self, path):
        self.code_version = CODE_VERSION
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "ForecastModel":
        return joblib.load(path)

    # ---- helpers for mean models: heteroscedastic residual quantiles
    # Time-of-day buckets (share of the session elapsed): the open is far jumpier than the afternoon,
    # and recent volatility alone (often measured on yesterday's quiet close) misses that.
    TOD_EDGES = np.array([0, 15, 30, 60, 120, 240, 376]) / 375.0

    def _scale(self, X, h):
        v = X["vol20"].fillna(X["vol20"].median() if X["vol20"].notna().any() else 5.0).clip(lower=0.5)
        return v.values * np.sqrt(h)

    def _bucket(self, X):
        f = X["bars_since_open"].fillna(0.5).clip(0, 1).values if "bars_since_open" in X else np.full(len(X), 0.5)
        return np.clip(np.searchsorted(self.TOD_EDGES, f, side="right") - 1, 0, len(self.TOD_EDGES) - 2)

    def _fit_resid_q(self, X, y, mean, h):
        z = (y - mean) / self._scale(X, h)
        self._rq = getattr(self, "_rq", {})
        self._rqb = getattr(self, "_rqb", {})
        mid = self.quantiles.index(0.5) if 0.5 in self.quantiles else None

        def qz(zz):
            q = np.nanquantile(zz, self.quantiles)
            return q - q[mid] if mid is not None else q   # centred: the middle line is the model's own call
        self._rq[h] = qz(z)
        b = self._bucket(X)
        self._rqb[h] = {k: (qz(z[b == k]) if (b == k).sum() > 300 else self._rq[h]) for k in range(len(self.TOD_EDGES) - 1)}

    def _resid_pred(self, X, mean, h):
        s = self._scale(X, h)
        rqb = getattr(self, "_rqb", {}).get(h)
        if rqb is None:
            return mean[:, None] + s[:, None] * self._rq[h][None, :]
        Q = np.vstack([rqb[k] for k in self._bucket(X)])
        return mean[:, None] + s[:, None] * Q


class NaiveVol(ForecastModel):
    shrink = False
    name = "naive"
    description = "Baseline: no direction (random walk) + volatility-scaled bands. Every model must beat this."

    def _fit(self, X, Y):
        self._rq = {}
        for h in self.horizons:
            y = Y[f"y{h}"].values
            m = ~np.isnan(y)
            self._fit_resid_q(X[m], y[m], np.zeros(m.sum()), h)

    def _predict(self, X):
        return {h: self._resid_pred(X, np.zeros(len(X)), h) for h in self.horizons}


class _Standardizer:
    def fit(self, X):
        self.mu = X.mean()
        self.sd = X.std().replace(0, 1).fillna(1)
        return self

    def __call__(self, X):
        return ((X - self.mu) / self.sd).fillna(0).clip(-8, 8).values


class RidgeQuant(ForecastModel):
    name = "ridge"
    description = "Linear (Ridge) mean + vol-scaled residual bands. Fast, stable, hard to overfit."

    def _fit(self, X, Y):
        from sklearn.linear_model import Ridge
        self.std = _Standardizer().fit(X)
        Z = self.std(X)
        self.m = {}
        for h in self.horizons:
            y = Y[f"y{h}"].values
            ok = ~np.isnan(y)
            # de-meaned target: no built-in up (or down) lean from the period it was trained on
            self.m[h] = Ridge(alpha=200.0, fit_intercept=False).fit(Z[ok], y[ok] - np.nanmean(y[ok]))
            self._fit_resid_q(X[ok], y[ok], self.m[h].predict(Z[ok]), h)

    def _predict(self, X):
        Z = self.std(X)
        return {h: self._resid_pred(X, self.m[h].predict(Z), h) for h in self.horizons}


class LGBMQuantile(ForecastModel):
    name = "lgbm"
    description = "LightGBM, one quantile model per horizon x quantile. Best default for tabular features."

    params = dict(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=400,
                  subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=20.0, verbose=-1)

    def _fit(self, X, Y):
        import lightgbm as lgb
        self.m = {}
        for h in self.horizons:
            y = Y[f"y{h}"].values
            ok = ~np.isnan(y)
            for q in self.quantiles:
                self.m[(h, q)] = lgb.LGBMRegressor(objective="quantile", alpha=q, **self.params).fit(X[ok], y[ok])

    def _predict(self, X):
        return {h: np.column_stack([self.m[(h, q)].predict(X) for q in self.quantiles]) for h in self.horizons}

    def importance(self, h=None, q=0.5) -> pd.Series:
        h = h or self.horizons[-1]
        return pd.Series(self.m[(h, q)].feature_importances_, index=self.features).sort_values(ascending=False)


class MLPQuant(ForecastModel):
    name = "mlp"
    description = "Small neural net (sklearn MLP), all horizons at once + vol-scaled bands. Neural baseline without PyTorch."

    def _fit(self, X, Y):
        import warnings
        from sklearn.neural_network import MLPRegressor
        warnings.filterwarnings("ignore", module="sklearn.neural_network")
        self.std = _Standardizer().fit(X)
        Z = self.std(X)
        ycols = [f"y{h}" for h in self.horizons]
        ok = Y[ycols].notna().all(axis=1).values
        self.ys = Y.loc[ok, ycols].std().values
        self.net = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=200,
                                early_stopping=True, random_state=0).fit(Z[ok], Y.loc[ok, ycols].values / self.ys)
        pred = self.net.predict(Z[ok]) * self.ys
        for i, h in enumerate(self.horizons):
            self._fit_resid_q(X[ok], Y.loc[ok, f"y{h}"].values, pred[:, i], h)

    def _predict(self, X):
        pred = self.net.predict(self.std(X)) * self.ys
        if pred.ndim == 1:
            pred = pred[:, None]
        return {h: self._resid_pred(X, pred[:, i], h) for i, h in enumerate(self.horizons)}


REGISTRY = {c.name: c for c in (NaiveVol, RidgeQuant, LGBMQuantile, MLPQuant)}
