"""Free historical 1-minute data loaders.

What exists for free (Nov 2026):
  * Kaggle "NSE - Nifty 50 Index Minute data (2015 to 2026)"  -> years of 1-min bars (download CSV manually)
  * Upstox historical API          -> recent months of 1-min bars (free account + token)
  * Angel One SmartAPI             -> ~30 days of 1-min bars per request window (free account)
  * Yahoo Finance via yfinance     -> last ~30 days of 1-min bars (no account, unofficial, can break)
  * 1-SECOND history               -> not free anywhere. Run scripts/record_ticks.py daily to build your own.
"""
from __future__ import annotations

import os
import time

import pandas as pd

from .storage import ensure_bar_frame, filter_session

TZ = "Asia/Kolkata"


def load_kaggle_csv(path: str) -> pd.DataFrame:
    """Loads any OHLC CSV (Kaggle, GitHub dumps, broker exports). Column names are auto-detected."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns and "time" in df.columns and not str(df["date"].iloc[0]).count(":"):
        df["ts"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str))
        df = df.drop(columns=["date", "time"])
    df = filter_session(ensure_bar_frame(df))
    df["ticks"] = 1.0          # index "volume" is always 0; keep it consistent with other sources
    return df


def find_nifty_minute_csv(folder) -> str | None:
    """In a Kaggle dataset folder, pick the NIFTY 50 *1-minute* file (not 3/5/15-minute or other indices)."""
    import re
    from pathlib import Path
    best = None
    for f in Path(folder).rglob("*.csv"):
        n = f.name.lower().replace("_", " ")
        if "nifty" not in n or "50" not in n or "minute" not in n:
            continue
        if re.search(r"\d+\s*minute", n.replace("50", "", 1)):   # 3minute, 5 minute, 15minute...
            continue
        if any(x in n for x in ("bank", "next", "alpha", "midcap", "it ", "fin")):
            continue
        if best is None or f.stat().st_size > best.stat().st_size:
            best = f
    return str(best) if best else None


KAGGLE_DATASET = "debashis74017/nifty-50-minute-data"
KAGGLE_FILES = ["NIFTY 50_minute.csv", "NIFTY50_minute.csv", "nifty_50_minute.csv", "NIFTY 50_minute_data.csv"]


def clear_kaggle_cache(dataset: str = KAGGLE_DATASET) -> int:
    """Delete kagglehub's cached copy of the full (834 MB) archive to free disk space. Returns bytes freed."""
    import shutil
    from pathlib import Path
    d = Path.home() / ".cache" / "kagglehub" / "datasets" / Path(dataset)
    if not d.exists():
        return 0
    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    shutil.rmtree(d, ignore_errors=True)
    return size


def download_kaggle_nifty(dataset: str = KAGGLE_DATASET) -> pd.DataFrame:
    """Downloads ONLY the NIFTY 50 1-minute CSV (~50 MB) from the free Kaggle dataset, not the whole
    834 MB archive with every index and interval."""
    import kagglehub
    freed = clear_kaggle_cache(dataset)
    if freed:
        print(f"      freed {freed / 1e6:,.0f} MB left over from the earlier full-archive download")
    last = None
    for name in KAGGLE_FILES:
        try:
            path = kagglehub.dataset_download(dataset, path=name)
            print(f"      using {path}")
            return load_kaggle_csv(path)
        except Exception as ex:
            last = ex
    raise RuntimeError(f"couldn't fetch the single NIFTY 50 file ({last}). Download 'NIFTY 50_minute.csv' "
                       f"by hand from kaggle.com/datasets/{dataset}, save it in this folder as "
                       f"nifty_50_minute.csv and run setup again")


def download_yahoo(days: int = 29) -> pd.DataFrame:
    """Yahoo chart API directly (no yfinance needed). 1-min data exists for ~30 days, 7 days per call."""
    import requests
    from .feeds.yahoo import HEADERS, URL

    end = pd.Timestamp.now(tz=TZ)
    start = end - pd.Timedelta(days=min(days, 29))
    frames = []
    while start < end:
        stop = min(start + pd.Timedelta(days=7), end)
        r = requests.get(URL, params={"period1": int(start.timestamp()), "period2": int(stop.timestamp()),
                                      "interval": "1m"}, headers=HEADERS, timeout=20)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
        if res.get("timestamp"):
            q = res["indicators"]["quote"][0]
            frames.append(pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"]},
                                       index=pd.to_datetime(res["timestamp"], unit="s", utc=True)))
        start = stop
        time.sleep(0.5)
    if not frames:
        raise RuntimeError("Yahoo returned no 1-min data.")
    df = pd.concat(frames).dropna()
    df.index.name = "ts"
    return filter_session(ensure_bar_frame(df))


def recent_bars(source: str, cfg: dict, days: int = 5) -> pd.DataFrame:
    """Best-effort backfill of the last few sessions incl. today-so-far, used when going live
    (so the models have the warm-up bars they need even if you start at 11:00)."""
    if source in ("yahoo", "auto", "nse"):
        frames = []
        try:
            from .feeds.yahoo import yahoo_chart
            df, _ = yahoo_chart("5d", "1m")
            frames.append(df)
        except Exception as ex:
            print("yahoo backfill failed:", ex)
        if source in ("auto", "nse"):
            try:  # NSE's own intraday points for today (overrides Yahoo for today)
                from .feeds.nse import nse_today_bars
                nb = nse_today_bars()
                if len(nb):
                    frames.append(nb)
            except Exception as ex:
                print("nse backfill failed:", ex)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return filter_session(ensure_bar_frame(df))
    if source == "upstox":
        return download_upstox(days, cfg["upstox"]["instrument_key"], include_today=True)
    if source == "angel":
        return download_angel(days, cfg["angel"]["token"])
    return pd.DataFrame()


def download_yfinance(days: int = 29) -> pd.DataFrame:
    import yfinance as yf  # pip install yfinance

    end = pd.Timestamp.now(tz=TZ)
    frames = []
    start = end - pd.Timedelta(days=days)
    while start < end:
        stop = min(start + pd.Timedelta(days=7), end)
        df = yf.download("^NSEI", start=start.tz_convert("UTC").tz_localize(None),
                         end=stop.tz_convert("UTC").tz_localize(None), interval="1m",
                         progress=False, auto_adjust=False)
        if len(df):
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            frames.append(df)
        start = stop
    if not frames:
        raise RuntimeError("yfinance returned nothing (Yahoo limits 1m data to ~30 days).")
    df = pd.concat(frames)
    df.index = pd.to_datetime(df.index)
    return filter_session(ensure_bar_frame(df))


def download_upstox(days: int = 60, instrument_key: str = "NSE_INDEX|Nifty 50", include_today: bool = True) -> pd.DataFrame:
    """Upstox v3 historical candles, 1-minute, fetched in monthly chunks."""
    import requests

    token = os.environ.get("UPSTOX_ACCESS_TOKEN", "")
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    key = requests.utils.quote(instrument_key, safe="")
    end = pd.Timestamp.now(tz=TZ).normalize()
    frames = []
    while days > 0:
        chunk = min(days, 28)
        frm = end - pd.Timedelta(days=chunk)
        url = (f"https://api.upstox.com/v3/historical-candle/{key}/minutes/1/"
               f"{end:%Y-%m-%d}/{frm:%Y-%m-%d}")
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        candles = r.json().get("data", {}).get("candles", [])
        if candles:
            df = pd.DataFrame(candles).iloc[:, :6]
            df.columns = ["ts", "open", "high", "low", "close", "volume"]
            frames.append(df)
        end, days = frm - pd.Timedelta(days=1), days - chunk
        time.sleep(0.3)
    if include_today:  # historical endpoint excludes the current session; intraday endpoint has it
        try:
            r = requests.get(f"https://api.upstox.com/v3/historical-candle/intraday/{key}/minutes/1",
                             headers=headers, timeout=30)
            r.raise_for_status()
            candles = r.json().get("data", {}).get("candles", [])
            if candles:
                df = pd.DataFrame(candles).iloc[:, :6]
                df.columns = ["ts", "open", "high", "low", "close", "volume"]
                frames.append(df)
        except Exception as ex:
            print("upstox intraday backfill failed:", ex)
    if not frames:
        raise RuntimeError("Upstox returned no candles.")
    return filter_session(ensure_bar_frame(pd.concat(frames)))


def download_angel(days: int = 30, token: str = "99926000") -> pd.DataFrame:
    from .feeds.angel import angel_login

    obj, _, _ = angel_login()
    end = pd.Timestamp.now(tz=TZ)
    frames = []
    while days > 0:
        chunk = min(days, 25)
        frm = end - pd.Timedelta(days=chunk)
        res = obj.getCandleData({"exchange": "NSE", "symboltoken": token, "interval": "ONE_MINUTE",
                                 "fromdate": f"{frm:%Y-%m-%d %H:%M}", "todate": f"{end:%Y-%m-%d %H:%M}"})
        data = (res or {}).get("data") or []
        if data:
            df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume"])
            frames.append(df)
        end, days = frm, days - chunk
        time.sleep(0.5)
    if not frames:
        raise RuntimeError("Angel returned no candles.")
    return filter_session(ensure_bar_frame(pd.concat(frames)))
