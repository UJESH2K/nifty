"""Tick -> bar aggregation and on-disk storage (Parquet).

Layout under data/:
    ticks/YYYY-MM-DD.parquet   raw ticks  (ts, price)           <- every tick you receive
    bars_1s.parquet            1-second OHLC bars + tick count
    bars_1min.parquet          1-minute OHLC bars + tick count
    news.parquet               headlines (ts, source, headline, [direction, cruciality, category])
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Asia/Kolkata"
BAR_COLS = ["open", "high", "low", "close", "ticks"]


# ----------------------------------------------------------------- bars
def ensure_bar_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise any OHLC frame: tz-aware IST DatetimeIndex, lowercase cols, sorted, de-duplicated."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        for cand in ("ts", "datetime", "date", "timestamp", "time"):
            if cand in df.columns:
                df.index = pd.to_datetime(df.pop(cand))
                break
    if df.index.tz is None:
        df.index = df.index.tz_localize(TZ)
    else:
        df.index = df.index.tz_convert(TZ)
    df.index.name = "ts"
    if "ticks" not in df.columns:
        df["ticks"] = df["volume"] if "volume" in df.columns else 1
    df = df[BAR_COLS].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def filter_session(df: pd.DataFrame, open_="09:15", close="15:30") -> pd.DataFrame:
    """Drop pre-open / post-close / weekend rows (Muhurat sessions are dropped too)."""
    t = df.index.time
    o = pd.Timestamp(open_).time()
    c = pd.Timestamp(close).time()
    mask = (t >= o) & (t < c) & (df.index.dayofweek < 5)
    return df[mask]


def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample finer bars to coarser bars (e.g. 1s -> 1min)."""
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "ticks": "sum"}
    )
    return out.dropna(subset=["close"])


def load_bars(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return ensure_bar_frame(df)


def save_bars(df: pd.DataFrame, path: str | Path, append: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists():
        df = pd.concat([pd.read_parquet(path), df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(path)


# ----------------------------------------------------------------- live aggregation
class BarBuilder:
    """Feeds ticks in, emits completed bars out. One builder per resolution."""

    def __init__(self, seconds: int):
        self.seconds = seconds
        self.cur_start: pd.Timestamp | None = None
        self.o = self.h = self.l = self.c = np.nan
        self.n = 0

    def _bucket(self, ts: pd.Timestamp) -> pd.Timestamp:
        return ts.floor(f"{self.seconds}s")

    def update(self, ts: pd.Timestamp, price: float) -> dict | None:
        """Returns the *completed* previous bar when a new bucket starts, else None."""
        b = self._bucket(ts)
        done = None
        if self.cur_start is None:
            self._reset(b, price)
        elif b > self.cur_start:
            done = self.current()
            self._reset(b, price)
        else:
            self.h = max(self.h, price)
            self.l = min(self.l, price)
            self.c = price
            self.n += 1
        return done

    def _reset(self, b, price):
        self.cur_start = b
        self.o = self.h = self.l = self.c = price
        self.n = 1

    def current(self) -> dict | None:
        if self.cur_start is None:
            return None
        return {"ts": self.cur_start, "open": self.o, "high": self.h, "low": self.l,
                "close": self.c, "ticks": float(self.n)}


class TickRecorder:
    """Buffers raw ticks and flushes them to data/ticks/<date>.parquet."""

    def __init__(self, data_dir: Path, flush_every: int = 500):
        self.dir = Path(data_dir) / "ticks"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.buf: list[tuple] = []
        self.flush_every = flush_every

    def add(self, ts: pd.Timestamp, price: float):
        self.buf.append((ts, price))
        if len(self.buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        df = pd.DataFrame(self.buf, columns=["ts", "price"])
        self.buf = []
        for day, g in df.groupby(df["ts"].dt.date):
            p = self.dir / f"{day}.parquet"
            if p.exists():
                g = pd.concat([pd.read_parquet(p), g])
            g.to_parquet(p, index=False)


# ----------------------------------------------------------------- news
NEWS_COLS = ["ts", "source", "headline"]


def load_news(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=NEWS_COLS + ["direction", "cruciality", "category"])
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(TZ)
    else:
        df["ts"] = df["ts"].dt.tz_convert(TZ)
    return df.sort_values("ts").reset_index(drop=True)


def save_news(df: pd.DataFrame, path: str | Path, append: bool = True) -> None:
    path = Path(path)
    if append and path.exists():
        df = pd.concat([load_news(path), df])
    df = df.drop_duplicates(subset=["headline"], keep="first").sort_values("ts")
    df.to_parquet(path, index=False)
