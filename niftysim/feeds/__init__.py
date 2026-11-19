"""Feed factory. Every feed exposes start(on_tick(ts, price), on_news(dict)) and stop()."""
from __future__ import annotations

from ..config import ROOT


def make_feed(cfg: dict, source: str | None = None, start_price: float | None = None):
    src = source or cfg["feed"]["source"]
    speed = cfg["feed"].get("speed", 60)
    if src == "synthetic":
        from .synthetic import SyntheticFeed
        return SyntheticFeed(speed=speed, start_price=start_price)
    if src == "replay":
        from .replay import ReplayFeed
        return ReplayFeed(str(ROOT / cfg["feed"]["replay_file"]), speed=speed)
    if src in ("auto", "nse"):
        from .nse import AutoFeed, NSEFeed
        poll = cfg["feed"].get("yahoo_poll_seconds", 3)
        return AutoFeed(poll) if src == "auto" else NSEFeed(poll)
    if src == "yahoo":
        from .yahoo import YahooFeed
        return YahooFeed(cfg["feed"].get("yahoo_poll_seconds", 3))
    if src == "upstox":
        from .upstox import UpstoxFeed
        return UpstoxFeed(cfg["upstox"]["instrument_key"])
    if src == "angel":
        from .angel import AngelFeed
        return AngelFeed(cfg["angel"]["token"], cfg["angel"]["exchange_type"])
    raise ValueError(f"unknown feed source {src}")
