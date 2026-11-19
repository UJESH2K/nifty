"""Walk-forward training & evaluation ("slow layer").

Always trains two variants of each model on identical data:
    price : price features only
    news  : price features + news/calendar features
so the dashboard can show both side by side and you can see whether news actually helps.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import bar_seconds, data_dir, horizons, models_dir
from .features import PRICE_FEATURES, build_dataset
from .models import REGISTRY
from .news.align import NEWS_FEATURES, load_events, news_features
from .news.scorer import make_scorer, score_frame
from .storage import load_bars, load_news


def load_training_data(cfg, resolution):
    d = data_dir(cfg)
    bars = load_bars(d / f"bars_{resolution}.parquet")
    news = load_news(d / "news.parquet")
    if len(news) and ("direction" not in news or news["direction"].isna().any()):
        news = score_frame(news.drop(columns=[c for c in ("direction", "cruciality", "category") if c in news]),
                           make_scorer(cfg))
    return bars, news


def make_xy(cfg, bars, news, resolution):
    bs = bar_seconds(resolution)
    events = load_events(data_dir(cfg) / "events.csv")
    nf = news_features(bars, news, bs, cfg["news"]["half_life_bars"], events)
    X, Y = build_dataset(bars, nf, horizons(cfg, resolution), bs, cfg["market"]["weekly_expiry_weekday"])
    return X, Y


def evaluate(pred: dict, Y: pd.DataFrame, price, cfg, mask=None) -> dict:
    """mask: optional boolean array -> metrics only on those rows (e.g. bars right after news)."""
    out = {}
    qs = cfg["quantiles"]
    for h, P in pred.items():
        y = Y[f"y{h}"].values
        ok = ~np.isnan(y)
        if mask is not None:
            ok &= mask
        p = P[ok]; yy = y[ok]
        med = p[:, 1]
        mae = np.mean(np.abs(yy - med))
        mae0 = np.mean(np.abs(yy))
        strong = np.abs(med) > np.quantile(np.abs(med), 0.8)
        hit = np.mean(np.sign(med[strong]) == np.sign(yy[strong])) if strong.any() and med[strong].any() else 0.5
        cov = np.mean((yy >= p[:, 0]) & (yy <= p[:, -1]))
        pin = np.mean([np.mean(np.maximum(q * (yy - p[:, i]), (q - 1) * (yy - p[:, i]))) for i, q in enumerate(qs)])
        out[h] = {"mae_bps": round(float(mae), 3), "skill_vs_zero": round(float(1 - mae / mae0), 4),
                  "dir_hit_top20pct": round(float(hit), 4), "coverage_p10_p90": round(float(cov), 3),
                  "pinball": round(float(pin), 4), "n": int(ok.sum())}
    return out


def walk_forward(cfg, X, Y, model_name, feats, n_folds=4, min_train_days=10):
    days = np.array(sorted(set(X.index.normalize())))
    if len(days) < min_train_days + n_folds:
        n_folds = max(1, len(days) - min_train_days)
    test_blocks = np.array_split(days[min_train_days:], n_folds)
    res = []
    for blk in test_blocks:
        if not len(blk):
            continue
        tr = X.index.normalize() < blk[0]
        te = X.index.normalize().isin(blk)
        m = REGISTRY[model_name](horizons=[int(c[1:]) for c in Y.columns if c.startswith("y")],
                                 quantiles=cfg["quantiles"], features=feats)
        m.fit(X[tr], Y[tr])
        P = m.predict(X[te])
        news_mask = (X.loc[te, "news_cru_decay"].values > 0.15) if "news_cru_decay" in X else None
        res.append((evaluate(P, Y[te], None, cfg), evaluate(P, Y[te], None, cfg, news_mask)))
    agg = {}
    for h in res[0][0]:
        agg[h] = {k: round(float(np.nanmean([r[0][h][k] for r in res])), 4) for k in res[0][0][h] if k != "n"}
        agg[h]["n"] = int(sum(r[0][h]["n"] for r in res))
        agg[h]["after_news"] = {k: round(float(np.nanmean([r[1][h][k] for r in res])), 4)
                                for k in ("skill_vs_zero", "dir_hit_top20pct")}
    return agg


def train_all(cfg, resolution=None, model_names=None, do_cv=True, log=print, last_days=None,
              holdout_days=0, until=None, subdir=None):
    """holdout_days > 0 keeps the final N days out of the saved models, so you can 'time-travel'
    over them in the dashboard and see honest, never-seen-before predictions."""
    import os
    resolution = resolution or cfg["resolution"]
    if last_days is None and os.environ.get("NIFTYSIM_MAX_TRAIN_DAYS"):
        last_days = int(os.environ["NIFTYSIM_MAX_TRAIN_DAYS"])   # e.g. 250 on a 512 MB free server
    bars, news = load_training_data(cfg, resolution)
    if until is not None:  # blind mode: the model never sees anything at/after `until`
        bars = bars[bars.index < until]
        news = news[news["ts"] < until] if len(news) else news
    if last_days:  # rolling window retrain
        days_all = bars.index.normalize().unique()
        cutoff = days_all[-min(last_days, len(days_all))]
        bars = bars[bars.index >= cutoff]
    X, Y = make_xy(cfg, bars, news, resolution)
    ycols = [c for c in Y.columns if c.startswith("y")]
    Y = Y[ycols]
    warm = X["vol60"].notna()
    X, Y = X[warm], Y[warm]
    out_dir = models_dir(cfg) / resolution
    if subdir:
        out_dir = out_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"resolution": resolution, "rows": int(len(X)), "days": int(X.index.normalize().nunique()),
              "trained_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(), "holdout_days": holdout_days,
              "models": {}}
    for name in model_names or list(REGISTRY):
        for variant, feats in (("price", PRICE_FEATURES), ("news", PRICE_FEATURES + NEWS_FEATURES)):
            t0 = time.time()
            cv = walk_forward(cfg, X, Y, name, feats) if do_cv else None
            m = REGISTRY[name](horizons=horizons(cfg, resolution), quantiles=cfg["quantiles"], features=feats)
            days = X.index.normalize().unique()
            tr = X.index.normalize() < days[-holdout_days] if holdout_days else np.ones(len(X), bool)
            m.fit(X[tr], Y[tr])
            m.train_end = X.index[tr][-1]
            m.save(out_dir / f"{name}_{variant}.joblib")
            report["models"][f"{name}_{variant}"] = {"cv": cv, "seconds": round(time.time() - t0, 1)}
            log(f"[{resolution}] {name:6s} {variant:5s} trained in {time.time()-t0:5.1f}s"
                + (f" | h={list(cv)[-1]} skill={cv[list(cv)[-1]]['skill_vs_zero']:+.3f} "
                   f"hit={cv[list(cv)[-1]]['dir_hit_top20pct']:.3f} cov={cv[list(cv)[-1]]['coverage_p10_p90']:.2f}"
                   if cv else ""))
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    return report
