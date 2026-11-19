"""Free news sources with timestamps.

Latency warning: RSS and GDELT typically lag the actual event by 1-15 minutes. Institutions get
machine-readable news in milliseconds, so the first move after a headline is usually already done
when it reaches us. The model therefore learns the *continuation* (5-30 min) after a headline,
which is the realistic edge for a retail setup.
"""
from __future__ import annotations

import os
import threading
import time
from email.utils import parsedate_to_datetime

import pandas as pd

TZ = "Asia/Kolkata"


def fetch_rss(urls: list[str]) -> pd.DataFrame:
    import feedparser

    rows = []
    for u in urls:
        try:
            f = feedparser.parse(u)
        except Exception:
            continue
        for e in f.entries:
            ts = None
            for key in ("published", "updated"):
                if e.get(key):
                    try:
                        ts = pd.Timestamp(parsedate_to_datetime(e[key]))
                        break
                    except Exception:
                        pass
            if ts is None:
                continue
            ts = ts.tz_localize(TZ) if ts.tz is None else ts.tz_convert(TZ)
            rows.append({"ts": ts, "source": u.split("/")[2], "headline": e.get("title", "").strip()})
    return pd.DataFrame(rows, columns=["ts", "source", "headline"])


def fetch_gdelt(query: str, start: pd.Timestamp, end: pd.Timestamp, max_records: int = 250) -> pd.DataFrame:
    """GDELT DOC 2.0 API — free, no key, history back to 2017. Use it to backfill news for training.
    Call it day by day: each call returns at most 250 articles."""
    import requests

    params = {"query": query, "mode": "artlist", "format": "json", "maxrecords": max_records,
              "sort": "datedesc",
              "startdatetime": start.tz_convert("UTC").strftime("%Y%m%d%H%M%S"),
              "enddatetime": end.tz_convert("UTC").strftime("%Y%m%d%H%M%S")}
    r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=30)
    r.raise_for_status()
    try:
        arts = r.json().get("articles", [])
    except ValueError:
        return pd.DataFrame(columns=["ts", "source", "headline"])
    rows = [{"ts": pd.Timestamp(a["seendate"], tz="UTC").tz_convert(TZ),
             "source": a.get("domain", "gdelt"), "headline": a.get("title", "")} for a in arts]
    return pd.DataFrame(rows, columns=["ts", "source", "headline"])


def backfill_gdelt(query: str, days: int, sleep: float = 6.0) -> pd.DataFrame:
    """GDELT allows roughly one request every 5 seconds; on 'Too Many Requests' we wait and retry,
    and give up after 3 days in a row fail (the rest of setup carries on)."""
    end = pd.Timestamp.now(tz=TZ)
    frames, fails = [], 0
    for i in range(days):
        e = end - pd.Timedelta(days=i)
        s = e - pd.Timedelta(days=1)
        got = False
        for attempt in range(3):
            try:
                df = fetch_gdelt(query, s, e)
                if len(df):
                    frames.append(df)
                got = True
                break
            except Exception as ex:
                if "429" in str(ex):
                    time.sleep(15 * (attempt + 1))
                else:
                    print("gdelt", s.date(), str(ex)[:80])
                    break
        fails = 0 if got else fails + 1
        if fails >= 3:
            print("gdelt is rate-limiting; skipping the rest of the news backfill (live headlines still work)")
            break
        print(f"      news {s:%d %b}: {len(frames[-1]) if got and frames else 0} headlines")
        time.sleep(sleep)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ts", "source", "headline"])


def fetch_finnhub() -> pd.DataFrame:
    import requests

    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        return pd.DataFrame(columns=["ts", "source", "headline"])
    r = requests.get("https://finnhub.io/api/v1/news", params={"category": "general", "token": key}, timeout=20)
    r.raise_for_status()
    return pd.DataFrame([{"ts": pd.Timestamp(a["datetime"], unit="s", tz="UTC").tz_convert(TZ),
                          "source": "finnhub:" + a.get("source", ""), "headline": a.get("headline", "")}
                         for a in r.json()])


def fetch_marketaux() -> pd.DataFrame:
    import requests

    key = os.environ.get("MARKETAUX_API_KEY")
    if not key:
        return pd.DataFrame(columns=["ts", "source", "headline"])
    r = requests.get("https://api.marketaux.com/v1/news/all",
                     params={"countries": "in", "language": "en", "api_token": key}, timeout=20)
    r.raise_for_status()
    return pd.DataFrame([{"ts": pd.Timestamp(a["published_at"]).tz_convert(TZ),
                          "source": "marketaux:" + a.get("source", ""), "headline": a.get("title", "")}
                         for a in r.json().get("data", [])])


class NewsPoller:
    """Polls all configured live sources every N seconds and hands NEW headlines to on_news."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.seen: set[str] = set()
        self._stop = threading.Event()

    def poll_once(self) -> pd.DataFrame:
        frames = [fetch_rss(self.cfg["news"]["rss"])]
        for f in (fetch_finnhub, fetch_marketaux):
            try:
                frames.append(f())
            except Exception as ex:
                print("news source error", f.__name__, ex)
        frames = [f for f in frames if f is not None and len(f)]
        if not frames:
            return pd.DataFrame(columns=["ts", "source", "headline"])
        df = pd.concat(frames, ignore_index=True)
        if df.empty:
            return df
        df = df[~df["headline"].isin(self.seen)]
        self.seen.update(df["headline"])
        return df.sort_values("ts")

    def start(self, on_news):
        def run():
            first = True
            while not self._stop.is_set():
                try:
                    df = self.poll_once()
                    for r in df.to_dict("records"):
                        on_news(r, backfill=first)
                    first = False
                except Exception as ex:
                    print("news poll error", ex)
                self._stop.wait(self.cfg["news"].get("poll_seconds", 30))
        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        self._stop.set()
