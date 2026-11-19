"""Angel One SmartAPI WebSocket v2 (free, tick-by-tick).

Setup (once):
  1. Angel One account -> https://smartapi.angelone.in/ -> create an app (free) -> API key.
  2. Enable TOTP for your account; keep the TOTP secret (the text behind the QR code).
  3. .env: ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_PIN, ANGEL_TOTP_SECRET
  4. pip install smartapi-python pyotp logzero websocket-client

SmartAPI sends prices in paise for NSE -> divide by 100.
NOT tested against the live API from the build environment — set NIFTYSIM_DEBUG=1 on first run.
"""
from __future__ import annotations

import os
import threading

import pandas as pd

TZ = "Asia/Kolkata"


def angel_login():
    import pyotp
    from SmartApi import SmartConnect

    api_key = os.environ["ANGEL_API_KEY"]
    obj = SmartConnect(api_key=api_key)
    data = obj.generateSession(os.environ["ANGEL_CLIENT_CODE"], os.environ["ANGEL_PIN"],
                               pyotp.TOTP(os.environ["ANGEL_TOTP_SECRET"]).now())
    if not data.get("status"):
        raise RuntimeError(f"Angel login failed: {data}")
    return obj, data["data"]["jwtToken"], obj.getfeedToken()


class AngelFeed:
    def __init__(self, token: str = "99926000", exchange_type: int = 1):
        self.token = token
        self.exchange_type = exchange_type
        self.name = "angel"
        self.ws = None

    def start(self, on_tick, on_news):
        from SmartApi.smartWebSocketV2 import SmartWebSocketV2

        _, jwt, feed_token = angel_login()
        self.ws = SmartWebSocketV2(jwt, os.environ["ANGEL_API_KEY"],
                                   os.environ["ANGEL_CLIENT_CODE"], feed_token)
        debug = os.environ.get("NIFTYSIM_DEBUG")
        tokens = [{"exchangeType": self.exchange_type, "tokens": [self.token]}]

        def on_data(wsapp, msg):
            if debug:
                print("ANGEL", msg)
            if not isinstance(msg, dict) or "last_traded_price" not in msg:
                return
            ets = msg.get("exchange_timestamp")
            ts = (pd.Timestamp(int(ets), unit="ms", tz="UTC").tz_convert(TZ)
                  if ets else pd.Timestamp.now(tz=TZ))
            on_tick(ts, float(msg["last_traded_price"]) / 100.0)

        def on_open(wsapp):
            self.ws.subscribe("niftysim", 1, tokens)   # mode 1 = LTP

        self.ws.on_data = on_data
        self.ws.on_open = on_open
        self.ws.on_error = lambda *a: print("ANGEL ws error", a)
        threading.Thread(target=self.ws.connect, daemon=True).start()

    def stop(self):
        if self.ws:
            try:
                self.ws.close_connection()
            except Exception:
                pass
