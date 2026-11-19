"""Upstox v3 market-data WebSocket (free with an Upstox account).

Setup (once):
  1. Upstox account -> https://account.upstox.com/developer/apps -> create an app (free).
  2. Do the daily OAuth login to get an access token (tokens expire every day ~3:30 AM IST).
     Put it in .env as UPSTOX_ACCESS_TOKEN=...
  3. pip install upstox-python-sdk

Note: the index has no volume; we use tick count per bar as an activity proxy.
NOT tested against the live API from the build environment — check field names on first run
(set NIFTYSIM_DEBUG=1 to print raw messages).
"""
from __future__ import annotations

import os

import pandas as pd

TZ = "Asia/Kolkata"


def _extract_ltp(feed: dict):
    """Handle ltpc / full / option_greeks payload shapes."""
    if "ltpc" in feed:
        return feed["ltpc"]
    ff = feed.get("fullFeed") or feed.get("full_feed") or {}
    for k in ("indexFF", "marketFF"):
        if k in ff and "ltpc" in ff[k]:
            return ff[k]["ltpc"]
    return None


class UpstoxFeed:
    def __init__(self, instrument_key: str = "NSE_INDEX|Nifty 50", mode: str = "ltpc"):
        self.key = instrument_key
        self.mode = mode
        self.name = "upstox"
        self.streamer = None

    def start(self, on_tick, on_news):
        import upstox_client  # pip install upstox-python-sdk

        token = os.environ.get("UPSTOX_ACCESS_TOKEN")
        if not token:
            raise RuntimeError("Set UPSTOX_ACCESS_TOKEN in .env (daily OAuth token).")
        conf = upstox_client.Configuration()
        conf.access_token = token
        self.streamer = upstox_client.MarketDataStreamerV3(
            upstox_client.ApiClient(conf), [self.key], self.mode)
        debug = os.environ.get("NIFTYSIM_DEBUG")

        def on_message(msg):
            if debug:
                print("UPSTOX", msg)
            feeds = (msg or {}).get("feeds", {})
            f = feeds.get(self.key)
            if not f:
                return
            l = _extract_ltp(f)
            if not l or "ltp" not in l:
                return
            ltt = l.get("ltt")
            ts = (pd.Timestamp(int(ltt), unit="ms", tz="UTC").tz_convert(TZ)
                  if ltt else pd.Timestamp.now(tz=TZ))
            on_tick(ts, float(l["ltp"]))

        self.streamer.on("message", on_message)
        self.streamer.auto_reconnect(True, 5, 20)
        self.streamer.connect()

    def stop(self):
        if self.streamer:
            try:
                self.streamer.disconnect()
            except Exception:
                pass
