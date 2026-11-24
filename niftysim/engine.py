"""The live engine: feed -> ticks -> bars -> features -> two forecasts (price-only / with-news)
-> online self-correction -> signals.  Thread-safe; the dashboard only reads `snapshot()`.
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import pandas as pd

from .config import bar_seconds, data_dir, horizons, models_dir
from .features import price_features
from .models import ForecastModel
from .news.align import load_events, news_features
from .news.scorer import make_scorer, score_frame
from .online import OnlineCalibrator
from .storage import BarBuilder, TickRecorder, load_bars, load_news, save_bars, save_news
from .train import make_xy

TZ = "Asia/Kolkata"
VARIANTS = ("price", "news")


class Engine:
    def __init__(self, cfg, resolution=None, model_name="lgbm", history_days=10,
                 history_until=None, models_subdir=None):
        """history_until: only load bars/news strictly before this time (blind replay).
        models_subdir: load models from models_store/<res>/<subdir>/ (blind replay models)."""
        self.cfg = cfg
        self.history_until = history_until
        self.models_subdir = models_subdir
        from .market_clock import MarketClock
        self.clock = MarketClock(cfg)
        self.live_source = None
        self.armed = False
        self.message = ""
        self.res = resolution or cfg["resolution"]
        self.bs = bar_seconds(self.res)
        self.H = horizons(cfg, self.res)
        self.Q = cfg["quantiles"]
        self.lock = threading.RLock()
        self.scorer = make_scorer(cfg)
        self.events = load_events(data_dir(cfg) / "events.csv")
        self.ticks = deque(maxlen=3600)
        self.status = "idle"
        self.feed = None
        self.news_poller = None
        self.recorder = None
        self.builder = BarBuilder(self.bs)
        self.builder_1s = BarBuilder(1)
        self.bars_1s_buf: list[dict] = []
        self.last_price = np.nan
        self.last_ts = None
        self.signals_log: list[dict] = []
        self.pending: deque = deque()
        al = cfg.get("alerts", {})
        self.refresh_s = float(al.get("refresh_seconds", 5))
        self.live_q: dict = {}
        self.live_ts = None
        self.live_anchor = None
        self._last_refresh_wall = 0.0
        self._last_refresh_sim = None
        self.alerts: list[dict] = []
        self.fast: dict = {}            # intraday re-tuned models (used 50/50 with the main model)
        self.live_X = None              # latest live feature row (for path-risk)
        self.ex = None                  # ExcursionModel: typical rise / dip within h minutes
        self.nextday = None             # NextDayModel: next session's open / close
        self.next_fc = None
        self._next_fc_at = 0.0
        self.alarms: list[dict] = []    # stop / target levels the user armed
        self.alarm_events: list[dict] = []
        self.refit_info = ""
        self._refitting = False
        self._load_history(history_days)
        self.set_model(model_name)
        threading.Thread(target=self._fit_aux, daemon=True).start()

    def _fit_aux(self):
        """Fit the path-risk and next-session models from the full bar file (a few seconds)."""
        try:
            from .daily import NextDayModel
            from .excursion import ExcursionModel
            p = data_dir(self.cfg) / f"bars_{self.res}.parquet"
            full = load_bars(p) if p.exists() else self.bars
            if self.history_until is not None:
                full = full[full.index < self.history_until]
            days = full.index.normalize().unique()
            # a blind replay only needs a shorter window, and memory on the free server is tight
            n_ex = 250 if self.history_until is None else 60
            recent = full[full.index >= days[-min(n_ex, len(days))]] if len(days) else full
            if self.res == "1min":
                self.nextday = NextDayModel().fit(full)
            del full
            X = price_features(recent, self.bs, self.cfg["market"]["weekly_expiry_weekday"])
            self.ex = ExcursionModel().fit(X, recent)
            del recent, X
            import gc
            gc.collect()
        except Exception as ex:
            self.message = f"path-risk model unavailable: {ex}"

    # ------------------------------------------------------------------ setup
    def _load_history(self, days):
        d = data_dir(self.cfg)
        p = d / f"bars_{self.res}.parquet"
        bars = load_bars(p) if p.exists() else pd.DataFrame(columns=["open", "high", "low", "close", "ticks"])
        if self.history_until is not None and len(bars):
            bars = bars[bars.index < self.history_until]
        self.saved_until = bars.index[-1] if len(bars) else None
        if len(bars):
            keep = bars.index.normalize().unique()[-days:]
            bars = bars[bars.index.normalize() >= keep[0]]
        self.bars = bars
        news = load_news(d / "news.parquet")
        if self.history_until is not None and len(news):
            news = news[news["ts"] < self.history_until].reset_index(drop=True)
        if len(news) and ("direction" not in news or news["direction"].isna().any()):
            news = score_frame(news.drop(columns=[c for c in ("direction", "cruciality", "category") if c in news]),
                               self.scorer)
        self.news = news
        self.n_history = len(bars)

    def set_model(self, name):
        with self.lock:
            md = models_dir(self.cfg) / self.res
            if self.models_subdir:
                md = md / self.models_subdir
            self.model_name = name
            models = {v: ForecastModel.load(md / f"{name}_{v}.joblib") for v in VARIANTS}
            if any(list(m.horizons) != list(self.H) for m in models.values()):
                raise FileNotFoundError(f"{name} was trained for old horizons; retrain it "
                                        f"(python scripts/train_models.py --no-cv --models {name})")
            self.models = models
            self.train_end = getattr(self.models["price"], "train_end", None)
            self.cal = {v: {h: OnlineCalibrator(self.cfg["online"]["target_coverage"], self.cfg["online"]["gamma"],
                                                self.cfg["online"]["bias_halflife"],
                                                use_bias=self.cfg["online"].get("use_bias", False)) for h in self.H} for v in VARIANTS}
            self._predict_history()

    def _tail(self):
        """Recent bars for live features: at least 150, and always back to yesterday's last bars so the
        overnight gap and day-open features are computed exactly as in training."""
        n = len(self.bars)
        if n == 0:
            return self.bars
        day = self.bars.index[-1].normalize()
        first_today = int(np.searchsorted(self.bars.index.values, day.to_datetime64()))
        start = max(0, min(n - 150, first_today - 70))
        return self.bars.iloc[start:]

    def _predict(self, v, X):
        P = self.models[v].predict(X)
        f = self.fast.get(v)
        if f is not None:
            F = f.predict(X)
            P = {h: (P[h] + F[h]) / 2.0 for h in P}
        return P

    def _predict_history(self):
        """Batch-predict every historical bar (powers time-travel + the past-forecast trail), and replay
        the online calibrator through it in time order so the bands start calibrated."""
        self.raw = {v: {h: np.full((len(self.bars), 3), np.nan) for h in self.H} for v in VARIANTS}
        self.corr = {v: {h: np.full((len(self.bars), 3), np.nan) for h in self.H} for v in VARIANTS}
        if len(self.bars) < 80:
            return
        X, Y = make_xy(self.cfg, self.bars, self.news, self.res)
        for v in VARIANTS:
            P = self.models[v].predict(X)
            for h in self.H:
                self.raw[v][h][:] = P[h]
                y = Y[f"y{h}"].values
                cal = self.cal[v][h]
                for i in range(len(y)):
                    c = cal.apply(P[h][i])
                    self.corr[v][h][i] = c
                    j = i - h  # prediction made h bars ago matures now
                    if j >= 0 and not np.isnan(y[j]):
                        cal.update(self.raw[v][h][j], self.corr[v][h][j], y[j])
        self.signals_log = []
        atr = price_features(self.bars, self.bs, self.cfg["market"]["weekly_expiry_weekday"])["atr14_bps"].values
        for i in range(60, len(self.bars)):
            self._signals(i, float(atr[i]))
        self._mature(len(self.bars) - 1, calibrate=False)
        self.last_price = float(self.bars["close"].iloc[-1])
        self.last_ts = self.bars.index[-1]

    # ------------------------------------------------------------------ live input
    REAL = ("auto", "nse", "yahoo", "upstox", "angel")

    def go_live(self, source):
        """Arm the engine: starts now if the market is open, otherwise waits for the next open."""
        if self.armed and self.live_source == source:
            return
        self.stop()
        self.armed = True
        self.live_source = source
        if source not in self.REAL:
            self.start(source)
            return
        self._arm_stop = threading.Event()

        def waiter():
            while not self._arm_stop.is_set():
                st = self.clock.status()
                if st["open"]:
                    try:
                        self.message = "Loading today's bars so far…"
                        self.backfill(source)
                    except Exception as ex:
                        self.message = f"Backfill failed ({str(ex)[:80]}) — continuing with live ticks only"
                    self.start(source)
                    return
                self.status = "waiting"
                self.message = ""
                self._arm_stop.wait(min(5, max(0.5, st["secs_to_open"] - 1)))
        threading.Thread(target=waiter, daemon=True).start()

    def disarm(self):
        self.armed = False
        if getattr(self, "_arm_stop", None):
            self._arm_stop.set()
        self.stop()

    def backfill(self, source):
        from .history import recent_bars
        df = recent_bars(source, self.cfg, 5)
        if not len(df):
            return
        with self.lock:
            last = self.bars.index[-1] if len(self.bars) else None
            new = df[df.index > last] if last is not None else df
            now_bucket = pd.Timestamp.now(tz=TZ).floor(f"{self.bs}s")
            new = new[new.index < now_bucket]          # never include the still-forming bar
            if len(new):
                self.bars = pd.concat([self.bars, new[["open", "high", "low", "close", "ticks"]]])
                self.n_history = len(self.bars)
                self._predict_history()
            self.message = f"Backfilled {len(new)} bars"

    def start(self, source=None):
        from .feeds import make_feed

        src = source or self.cfg["feed"]["source"]
        for x in (self.feed, self.news_poller):
            if x:
                try:
                    x.stop()
                except Exception:
                    pass
        start_price = self.last_price if np.isfinite(self.last_price) else None
        self.fast, self.refit_info, self.live_q = {}, "", {}
        self.feed = make_feed(self.cfg, src, start_price=start_price)
        self.live_source = src
        if src in self.REAL:
            self.recorder = TickRecorder(data_dir(self.cfg))
            from .news.sources import NewsPoller
            self.news_poller = NewsPoller(self.cfg)
            self.news_poller.start(self.on_news)
        self.feed.start(self.on_tick, self.on_news)
        self.status = f"live:{src}"

    def stop(self):
        if getattr(self, "_arm_stop", None):
            self._arm_stop.set()
        for x in (self.feed, self.news_poller):
            if x:
                try:
                    x.stop()
                except Exception:
                    pass
        self.flush()
        self.status = "stopped"

    def on_news(self, item: dict, backfill: bool = False):
        try:
            s = self.scorer(item["headline"])
            row = {**item, **s}
            ts = pd.Timestamp(row["ts"])
            row["ts"] = ts.tz_localize(TZ) if ts.tzinfo is None else ts.tz_convert(TZ)
            with self.lock:
                new = pd.DataFrame([row])
                self.news = new if self.news is None or self.news.empty else pd.concat([self.news, new], ignore_index=True)
                self.news["ts"] = pd.to_datetime(self.news["ts"], utc=True).dt.tz_convert(TZ)
        except Exception as ex:
            self.message = f"headline skipped: {ex}"

    def on_tick(self, ts: pd.Timestamp, price: float):
        if ts.tzinfo is None:
            ts = ts.tz_localize(TZ)
        if self.live_source in self.REAL and not self.clock.is_open(ts):
            self.last_price = price
            if self.status.startswith("live"):
                self.status = "session over"
            return
        with self.lock:
            self.last_price, self.last_ts = price, ts
            self.ticks.append((ts, price))
            if self.alarms:
                self._check_alarms(ts, price)
            if self.recorder:
                self.recorder.add(ts, price)
                b1 = self.builder_1s.update(ts, price)
                if b1 and self.res != "1s":
                    self.bars_1s_buf.append(b1)
            done = self.builder.update(ts, price)
            if done:
                try:
                    self._on_bar(done)
                except Exception as ex:   # a bad bar must never stop the price feed
                    self.message = f"bar skipped: {ex}"
            self.refresh_live(ts)

    # ------------------------------------------------------------------ per bar
    def _on_bar(self, b: dict):
        ts = b.pop("ts")
        if len(self.bars) and ts <= self.bars.index[-1]:
            return
        if len(self.bars) and ts.normalize() != self.bars.index[-1].normalize():
            for d in self.cal.values():
                for c in d.values():
                    c.new_session()
        self.bars = pd.concat([self.bars, pd.DataFrame([b], index=pd.DatetimeIndex([ts], name="ts"))])
        i = len(self.bars) - 1
        tail = self._tail()
        pf = price_features(tail, self.bs, self.cfg["market"]["weekly_expiry_weekday"]).iloc[[-1]]
        nws = self.news[self.news["ts"] >= tail.index[0]] if len(self.news) else self.news
        nf = news_features(tail, nws, self.bs, self.cfg["news"]["half_life_bars"], self.events).iloc[[-1]]
        X = pd.concat([pf, nf], axis=1)
        for v in VARIANTS:
            P = self._predict(v, X)
            for h in self.H:
                raw = P[h][0]
                corr = self.cal[v][h].apply(raw)
                self.raw[v][h] = np.vstack([self.raw[v][h], raw])
                self.corr[v][h] = np.vstack([self.corr[v][h], corr])
                self.pending.append((i, v, h))
        self._mature(i)
        self._signals(i, float(pf["atr14_bps"].iloc[0]))
        if self.recorder and i % 5 == 0:
            self.flush()

    def _mature(self, i_now: int, calibrate: bool = True):
        close = self.bars["close"].values
        day = self.bars.index.normalize()
        keep = deque()
        while calibrate and self.pending:
            j, v, h = self.pending.popleft()
            if j + h > i_now:
                keep.append((j, v, h))
                continue
            if day[j + h] != day[j]:
                continue
            y = np.log(close[j + h] / close[j]) * 1e4
            self.cal[v][h].update(self.raw[v][h][j], self.corr[v][h][j], y)
        if calibrate:
            self.pending = keep
        # capture-rule outcomes
        hi, lo = self.bars["high"].values, self.bars["low"].values
        for s in self.signals_log:
            if s.get("done") or s["i"] + s["h"] > i_now:
                continue
            if day[s["i"] + s["h"]] != day[s["i"]]:
                s.update(done=True, best_pts=0.0, success=False, cut_by_close=True)
                continue
            a, b = s["i"] + 1, s["i"] + s["h"] + 1
            up, dn = hi[a:b].max() - s["anchor"], s["anchor"] - lo[a:b].min()
            exc, adv = (up, dn) if s["side"] > 0 else (dn, up)
            s.update(done=True, best_pts=float(exc), adverse_pts=float(adv),
                     success=bool(exc >= s["target_pts"]),
                     opp_success=bool(adv >= s["target_pts"]))   # coin-flip baseline

    # ------------------------------------------------------------------ signals
    def _signals(self, i, atr_bps):
        sc = self.cfg["signals"]
        h = self.capture_h()
        anchor = float(self.bars["close"].iloc[i])
        for v in VARIANTS:
            lo, med, hi = self.corr[v][h][i]
            med_pts, atr_pts = anchor * med / 1e4, anchor * atr_bps / 1e4
            side = int(np.sign(med))
            cons = lo if side > 0 else hi
            if (side and abs(med_pts) >= sc["min_projection_atr"] * atr_pts
                    and -side * cons <= sc.get("max_adverse_ratio", 1.0) * abs(med)):
                recent = [s for s in self.signals_log if s["variant"] == v and s["side"] == side and i - s["i"] < h]
                if not recent:
                    self.signals_log.append({"i": i, "ts": self.bars.index[i], "variant": v, "side": side, "h": h,
                                             "anchor": anchor, "proj_pts": med_pts,
                                             "target_pts": sc["capture_fraction"] * abs(med_pts)})

    def capture_h(self) -> int:
        want = self.cfg["signals"].get("capture_horizon", 30)
        return min(self.H, key=lambda h: abs(h - want))

    def stand_aside(self, v, i=None) -> tuple[bool, list[str]]:
        sc = self.cfg["signals"]
        i = len(self.bars) - 1 if i is None else i
        h = self.capture_h()
        w = self.corr[v][h][max(0, i - 300): i + 1]
        w = w[:, 2] - w[:, 0]
        w = w[~np.isnan(w)]
        reasons = []
        if len(w) > 30 and w[-1] >= np.quantile(w, sc["stand_aside_width_pctile"]):
            reasons.append("the expected range is unusually wide right now, so the market is choppy")
        if v == "news" and len(self.news):
            t = self.bars.index[i]
            fresh = self.news[(self.news["ts"] <= t + pd.Timedelta(seconds=self.bs)) &
                              (self.news["ts"] >= t - pd.Timedelta(minutes=10))]
            if len(fresh) and fresh["cruciality"].max() >= sc["stand_aside_cruciality"]:
                reasons.append(f"a high-importance headline just landed: {fresh.sort_values('cruciality').iloc[-1]['headline'][:70]}")
        if self.events is not None:
            t = self.bars.index[i]
            near = self.events[(self.events["ts"] - t).abs() <= pd.Timedelta(minutes=15)]
            if len(near):
                reasons.append(f"scheduled event within 15 min: {near.iloc[0]['name']}")
        return bool(reasons), reasons

    # ------------------------------------------------------------------ persistence
    def flush(self):
        if not self.recorder:
            return
        d = data_dir(self.cfg)
        self.recorder.flush()
        live = self.bars if self.saved_until is None else self.bars[self.bars.index > self.saved_until]
        if len(live):
            save_bars(live, d / f"bars_{self.res}.parquet")
            self.saved_until = live.index[-1]
        if self.bars_1s_buf:
            df = pd.DataFrame(self.bars_1s_buf).set_index("ts")
            save_bars(df, d / "bars_1s.parquet")
            self.bars_1s_buf = []
        if len(self.news):
            save_news(self.news[["ts", "source", "headline"]], d / "news.parquet")

    # ------------------------------------------------------------------ any-horizon forecasts
    def interp_live(self, v, h: float):
        """Same as interp() but for the 5-second live forecast (None if not available yet)."""
        lq = self.live_q.get(v)
        if not lq:
            return None
        return self._interp_A({k: np.asarray(lq[k])[None, :] for k in self.H}, h)[0]

    def interp(self, v, h: float, rows=None) -> np.ndarray:
        """Quantiles (bps) for an arbitrary horizon h (in bars), for the given bar rows (default all).
        Median: linear in h. Band half-widths: linear in sqrt(h) (uncertainty grows like sqrt(time))."""
        return self._interp_A({k: (self.corr[v][k] if rows is None else self.corr[v][k][rows]) for k in self.H}, h)

    def _interp_A(self, A, h):
        H = self.H
        if h <= H[0]:
            a = A[H[0]]
            f = h / H[0]
            med = a[:, 1] * f
            g = np.sqrt(f)
            return np.column_stack([med - (a[:, 1] - a[:, 0]) * g, med, med + (a[:, 2] - a[:, 1]) * g])
        if h >= H[-1]:
            return A[H[-1]].copy()
        k = int(np.searchsorted(H, h))
        h0, h1 = H[k - 1], H[k]
        a0, a1 = A[h0], A[h1]
        wl = (h - h0) / (h1 - h0)
        ws = (np.sqrt(h) - np.sqrt(h0)) / (np.sqrt(h1) - np.sqrt(h0))
        med = a0[:, 1] * (1 - wl) + a1[:, 1] * wl
        lo = (a0[:, 1] - a0[:, 0]) * (1 - ws) + (a1[:, 1] - a1[:, 0]) * ws
        hi = (a0[:, 2] - a0[:, 1]) * (1 - ws) + (a1[:, 2] - a1[:, 1]) * ws
        return np.column_stack([med - lo, med, med + hi])

    def trajectory(self, v, i, max_h, anchor_px, step=1, live=False):
        """Per-bar path from bar i (or, live=True, from the forming bar using the 5-second forecast)
        out to max_h bars, in index points, capped at the session close."""
        base = self.live_ts if (live and self.live_q.get(v) and self.live_ts is not None) else self.bars.index[i]
        use_live = base is not self.bars.index[i] and live
        close_t = self.clock.session(base)[1] - pd.Timedelta(seconds=self.bs)
        hs, rows = [0.0], [[0.0, 0.0, 0.0]]
        h = step
        while h <= max_h:
            t = base + pd.Timedelta(seconds=self.bs * h)
            if t > close_t:
                break
            rows.append(self.interp_live(v, h) if use_live else self.interp(v, h, rows=[i])[0])
            hs.append(h)
            h += step
        q = np.array(rows)
        px = anchor_px * np.exp(q / 1e4)
        # candles are drawn at their start time, so the close of bar i+h sits at index[i] + h bars
        ts = [base + pd.Timedelta(seconds=self.bs * x) for x in hs]
        return pd.DataFrame({"ts": ts, "h": hs, "p10": px[:, 0], "p50": px[:, 1], "p90": px[:, 2]})

    # ------------------------------------------------------------------ 5-second live layer
    def refresh_live(self, ts=None, force=False):
        """Every `refresh_seconds` (5 s): rebuild features with the still-forming candle, predict all
        horizons with both models, then check whether this is a critical moment."""
        now = time.time()
        if not force:
            if now - self._last_refresh_wall < 1.0:
                return
            if self._last_refresh_sim is not None and ts is not None and \
                    (ts - self._last_refresh_sim).total_seconds() < self.refresh_s:
                return
        cur = self.builder.current()
        if cur is None or len(self.bars) < 60:
            return
        self._last_refresh_wall, self._last_refresh_sim = now, ts
        try:
            tail = self._tail()
            if cur["ts"] > tail.index[-1]:
                row = pd.DataFrame([{k: cur[k] for k in ("open", "high", "low", "close", "ticks")}],
                                   index=pd.DatetimeIndex([cur["ts"]], name="ts"))
                tail = pd.concat([tail, row])
            pf = price_features(tail, self.bs, self.cfg["market"]["weekly_expiry_weekday"]).iloc[[-1]]
            nws = self.news[self.news["ts"] >= tail.index[0]] if len(self.news) else self.news
            nf = news_features(tail, nws, self.bs, self.cfg["news"]["half_life_bars"], self.events).iloc[[-1]]
            X = pd.concat([pf, nf], axis=1)
            lq = {}
            for v in VARIANTS:
                P = self._predict(v, X)
                lq[v] = {h: self.cal[v][h].apply(P[h][0]) for h in self.H}
            self.live_q, self.live_ts, self.live_anchor = lq, tail.index[-1], float(cur["close"])
            self.live_X = pf
            self._detect(pf.iloc[0], tail)
        except Exception as ex:  # never let a refresh kill the feed
            self.message = f"live refresh error: {ex}"

    def _detect(self, f, tail):
        """Critical-moment detector. Fires an alert (and an intraday re-tune) when something unusual
        is happening right now."""
        al = self.cfg.get("alerts", {})
        t = self.live_ts
        if self.alerts and (t - self.alerts[-1]["ts"]).total_seconds() < 60 * al.get("cooldown_minutes", 15):
            return
        v, h = "news", self.capture_h()
        q = self.live_q[v][h]
        anchor = self.live_anchor
        to_pts = lambda b: anchor * (np.exp(b / 1e4) - 1)
        med, lo, hi = to_pts(q[1]), to_pts(q[0]), to_pts(q[2])
        atr_pts = anchor * float(f["atr14_bps"]) / 1e4
        reasons = []
        w = self.corr[v][h][-300:]
        w = (w[:, 2] - w[:, 0])[~np.isnan(w[:, 0])]
        mins_in = (t - self.clock.session(t)[0]).total_seconds() / 60
        if (len(w) > 60 and mins_in >= 20 and (q[2] - q[0]) > np.quantile(w, al.get("range_jump_pctile", 0.97))
                and (q[2] - q[0]) > 1.5 * np.median(w)):
            reasons.append(f"the expected {h}-min range just widened to {hi - lo:.0f} pts (volatility breakout)")
        r5, vol60 = float(f["r5"]), float(f["vol60"])
        if mins_in >= 6 and np.isfinite(r5) and np.isfinite(vol60) and vol60 > 0:
            z = abs(r5) / (vol60 * np.sqrt(5))
            if z >= al.get("fast_move_sigma", 3.5):
                reasons.append(f"price moved {anchor * r5 / 1e4:+.0f} pts in 5 min ({z:.1f}× the usual speed)")
        if len(self.news):
            fresh = self.news[(self.news["ts"] <= t + pd.Timedelta(seconds=self.bs)) &
                              (self.news["ts"] >= t - pd.Timedelta(minutes=5))]
            if len(fresh) and fresh["cruciality"].max() >= self.cfg["signals"]["stand_aside_cruciality"]:
                reasons.append("high-importance headline: " + fresh.sort_values("cruciality").iloc[-1]["headline"][:90])
        big = abs(med) >= self.cfg["signals"]["min_projection_atr"] * atr_pts and abs(med) >= al.get("min_points", 15)
        if big:
            reasons.append(f"model projects {med:+.0f} pts in {h} min")
        if not reasons:
            return
        side = int(np.sign(med)) if abs(med) >= 0.5 * atr_pts else 0
        frac = self.cfg["signals"]["capture_fraction"]
        a = {"id": len(self.alerts) + 1, "ts": t, "side": side, "h": h, "proj_pts": med, "lo_pts": lo, "hi_pts": hi,
             "big": bool(big), "target_pts": frac * med if big else 0.0, "anchor": anchor, "reasons": reasons}
        self.alerts.append(a)
        if al.get("refit_on_alert", True):
            threading.Thread(target=self._refit, daemon=True).start()

    def _refit(self):
        """Intraday re-tune: fit a fresh model on the last N days before today plus everything seen
        today so far, and blend it 50/50 with the main model for the rest of the session."""
        if self._refitting:
            return
        self._refitting = True
        try:
            from .models import REGISTRY
            days = int(self.cfg.get("alerts", {}).get("refit_days", 60))
            with self.lock:
                today0 = self.bars.index[-1].normalize()
                today_bars = self.bars[self.bars.index >= today0].copy()
                news = self.news.copy()
            p = data_dir(self.cfg) / f"bars_{self.res}.parquet"
            hist = load_bars(p) if p.exists() else today_bars.iloc[:0]
            hist = hist[hist.index < today0]                       # never peeks past today's open
            if len(hist):
                keep = hist.index.normalize().unique()[-days:]
                hist = hist[hist.index.normalize() >= keep[0]]
            bars = pd.concat([hist, today_bars])
            bars = bars[~bars.index.duplicated(keep="last")].sort_index()
            X, Y = make_xy(self.cfg, bars, news[news["ts"] < bars.index[-1]] if len(news) else news, self.res)
            Y = Y[[c for c in Y.columns if c.startswith("y")]]
            ok = X["vol60"].notna()
            fast = {}
            for v in VARIANTS:
                m = REGISTRY[self.model_name](horizons=self.H, quantiles=self.Q, features=self.models[v].features)
                fast[v] = m.fit(X[ok], Y[ok])
            with self.lock:
                self.fast = fast
                self.refit_info = (f"Re-tuned at {pd.Timestamp.now(tz=TZ):%H:%M:%S} on the last {days} days "
                                   f"+ {len(today_bars)} bars from today")
                for v in VARIANTS:           # widen bands a little right after a critical moment
                    for c in self.cal[v].values():
                        c.scale = max(c.scale, 1.15)
        except Exception as ex:
            self.refit_info = f"Re-tune failed: {ex}"
        finally:
            self._refitting = False

    @staticmethod
    def p_up(q) -> float:
        """Chance the move is > 0 from (p10, p50, p90) via a piecewise-linear CDF."""
        lo, med, hi = q
        if 0 <= lo:
            return float(min(0.99, 0.9 + 0.1 * (1 - np.exp(-(lo) / max(hi - lo, 1e-9)))))
        if 0 >= hi:
            return float(max(0.01, 0.1 * np.exp(-(-hi) / max(hi - lo, 1e-9))))
        if 0 <= med:
            cdf0 = 0.1 + 0.4 * (0 - lo) / max(med - lo, 1e-9)
        else:
            cdf0 = 0.5 + 0.4 * (0 - med) / max(hi - med, 1e-9)
        return float(1 - cdf0)

    def live_check(self, v, h: float, since=None) -> dict:
        """How did forecasts for horizon h actually do? Uses every matured forecast since `since`."""
        h_int = int(round(h))
        n = len(self.bars)
        if n <= h_int + 1:
            return {"n": 0}
        idx = self.bars.index
        js = np.arange(0, n - h_int)
        if since is not None:
            js = js[idx[js] >= since]
        day = idx.normalize()
        js = js[day[js + h_int] == day[js]]
        if not len(js):
            return {"n": 0}
        close = self.bars["close"].values
        q = self.interp(v, h_int, rows=js)
        ok = ~np.isnan(q).any(axis=1)
        js, q = js[ok], q[ok]
        if not len(js):
            return {"n": 0}
        y = np.log(close[js + h_int] / close[js]) * 1e4
        px = close[js]
        miss = np.abs(y - q[:, 1]) * px / 1e4
        miss0 = np.abs(y) * px / 1e4
        moved = np.abs(q[:, 1]) > 0.5
        return {"n": int(len(js)),
                "dir_right": float(np.mean(np.sign(q[moved, 1]) == np.sign(y[moved]))) if moved.any() else np.nan,
                "in_range": float(np.mean((y >= q[:, 0]) & (y <= q[:, 2]))),
                "miss_pts": float(miss.mean()), "nochange_miss_pts": float(miss0.mean())}

    # ------------------------------------------------------------------ path risk, plans, alarms
    def features_at(self, i):
        """Price-feature row for bar i (history / replay), same window rule as live."""
        day = self.bars.index[i].normalize()
        first = int(np.searchsorted(self.bars.index.values, day.to_datetime64()))
        start = max(0, min(i + 1 - 150, first - 70))
        return price_features(self.bars.iloc[start:i + 1], self.bs,
                              self.cfg["market"]["weekly_expiry_weekday"]).iloc[[-1]]

    def minutes_left(self, t):
        return max(1.0, (self.clock.session(t)[1] - t).total_seconds() / self.bs - 1)

    def plan(self, h, i=None, live=True, anchor_px=None):
        """Long and short plans for the next h bars: target = typical best move, stop = the adverse move
        that 80% of the time is NOT exceeded; odds measured on real history in the same volatility."""
        if self.ex is None:
            return None
        if live and self.live_X is not None and self.live_ts is not None:
            X, t = self.live_X, self.live_ts
        else:
            i = len(self.bars) - 1 if i is None else i
            X, t = self.features_at(i), self.bars.index[i]
        px = float(anchor_px if anchor_px is not None else self.last_price)
        h_eff = min(h, self.minutes_left(t))
        r = self.ex.predict(X, h_eff)
        to_pts = lambda b: px * b / 1e4
        out = {"h": h_eff, "anchor": px, "t": t, "sigma_pts": to_pts(r["sigma"]),
               "rise50": to_pts(r["up50"]), "rise80": to_pts(r["up80"]),
               "dip50": to_pts(r["dn50"]), "dip80": to_pts(r["dn80"])}
        for side, tgt, stp in (("long", r["up50"], r["dn80"]), ("short", r["dn50"], r["up80"])):
            o = self.ex.outcomes(h_eff, tgt, stp, r["sigma"], side)
            sgn = 1 if side == "long" else -1
            out[side] = {"target_px": px * np.exp(sgn * tgt / 1e4), "stop_px": px * np.exp(-sgn * stp / 1e4),
                         "target_pts": to_pts(tgt), "stop_pts": to_pts(stp), **o}
        return out

    def set_alarm(self, side, plan):
        p = plan[side]
        a = {"id": len(self.alarms) + len(self.alarm_events) + 1, "side": side, "entry": plan["anchor"],
             "target_px": p["target_px"], "stop_px": p["stop_px"], "set_at": self.last_ts,
             "until": self.last_ts + pd.Timedelta(seconds=self.bs * plan["h"]), "h": plan["h"]}
        with self.lock:
            self.alarms.append(a)
        return a

    def cancel_alarms(self):
        with self.lock:
            self.alarms = []

    def _check_alarms(self, ts, price):
        keep = []
        for a in self.alarms:
            up = a["side"] == "long"
            hit_t = price >= a["target_px"] if up else price <= a["target_px"]
            hit_s = price <= a["stop_px"] if up else price >= a["stop_px"]
            kind = "target" if hit_t else "stop" if hit_s else ("expired" if ts >= a["until"] else None)
            if kind:
                self.alarm_events.append({**a, "kind": kind, "ts": ts, "price": price,
                                          "pts": (price - a["entry"]) * (1 if up else -1)})
            else:
                keep.append(a)
        self.alarms = keep

    def next_session(self, force=False):
        """Next session forecast (refreshed at most every 15 min; US cue fetched when available)."""
        if self.nextday is None or not len(self.bars):
            return None
        now = time.time()
        if self.next_fc is not None and not force and now - self._next_fc_at < 900:
            return self.next_fc
        t_now = pd.Timestamp.now(tz=TZ)
        st = self.clock.status(t_now)
        last_close = float(self.last_price if np.isfinite(self.last_price) else self.bars["close"].iloc[-1])
        last_t = self.clock.session(self.clock.last_session_day(t_now))[1] if not st["open"] else t_now
        us = self.nextday.us_move_since(last_t) if not st["open"] else None
        fc = self.nextday.forecast(last_close, us)
        nxt = self.clock.next_open(t_now + pd.Timedelta(minutes=1)) if st["open"] else st["next_open"]
        fc.update(next_open=nxt, based_on="today's last price (session still open)" if st["open"]
                  else "the last close", last_close=last_close)
        self.next_fc, self._next_fc_at = fc, now
        return fc

    # ------------------------------------------------------------------ read API for UI
    def snapshot(self):
        with self.lock:
            return {
                "bars": self.bars, "news": self.news, "ticks": list(self.ticks),
                "corr": {v: {h: a.copy() for h, a in d.items()} for v, d in self.corr.items()},
                "last_price": self.last_price, "last_ts": self.last_ts,
                "status": self.status, "model": self.model_name, "n_history": self.n_history,
                "cal": {v: {h: c.stats() for h, c in d.items()} for v, d in self.cal.items()},
                "signals": list(self.signals_log), "train_end": self.train_end,
                "forming": self.builder.current() if self.status.startswith(("live", "replay")) else None,
                "live_q": {v: dict(d) for v, d in self.live_q.items()}, "live_ts": self.live_ts,
                "alerts": list(self.alerts), "refit_info": self.refit_info, "refitting": self._refitting,
                "armed": self.armed, "source": self.live_source, "message": self.message,
                "alarms": list(self.alarms), "alarm_events": list(self.alarm_events),
                "using": getattr(self.feed, "using", None) if self.feed else None,
                "lag": getattr(self.feed, "lag_seconds", 0.0) if self.feed else 0.0,
                "feed_error": getattr(self.feed, "last_error", None) if self.feed else None,
            }
