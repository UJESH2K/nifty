"""Next-session forecast: where NIFTY is likely to OPEN (the overnight gap) and CLOSE next session.

What 11 years of real data say (tested year by year):
  * gap SIZE is predictable from recent volatility (corr ~0.4)
  * gap DIRECTION from NIFTY's own day adds nothing beyond the recent base rate
    (NIFTY opened higher ~67% of days 2015-2026, but much less in 2026), so we use the last ~year's rate
  * US markets overnight are the strongest known driver of India's open. When Yahoo data is
    reachable (it is on Render), the S&P 500's move since India closed is added with a weight
    learned from history. If not reachable, the forecast simply leaves it out and says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "Asia/Kolkata"


def daily_table(bars: pd.DataFrame) -> pd.DataFrame:
    d = bars.index.normalize()
    g = bars.groupby(d)
    D = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                      "close": g["close"].last(), "n": g.size()})
    D = D[D["n"] > 300].copy()
    D["gap"] = np.log(D["open"] / D["close"].shift()) * 1e4
    D["cc"] = np.log(D["close"] / D["close"].shift()) * 1e4
    D["gapvol10"] = D["gap"].rolling(10, min_periods=5).std()
    D["vol5"] = D["cc"].rolling(5, min_periods=3).std()
    return D


def _yahoo_daily(symbol: str, rng: str = "5y") -> pd.Series:
    import requests
    from .feeds.yahoo import HEADERS
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                     params={"range": rng, "interval": "1d"}, headers=HEADERS, timeout=15)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    s = pd.Series(res["indicators"]["quote"][0]["close"],
                  index=pd.to_datetime(res["timestamp"], unit="s", utc=True)).dropna()
    return s


def _yahoo_last(symbol: str) -> tuple[float, pd.Timestamp]:
    import requests
    from .feeds.yahoo import HEADERS
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                     params={"range": "1d", "interval": "5m"}, headers=HEADERS, timeout=15)
    r.raise_for_status()
    m = r.json()["chart"]["result"][0]["meta"]
    return float(m["regularMarketPrice"]), pd.Timestamp(int(m["regularMarketTime"]), unit="s", tz="UTC")


class NextDayModel:
    def fit(self, bars: pd.DataFrame, lookback_days: int = 250):
        D = daily_table(bars)
        self.D = D
        recent = D.dropna(subset=["gap"]).iloc[-lookback_days:]
        self.p_gap_up = float((recent["gap"] > 0).mean())
        self.mean_gap = float(recent["gap"].median())
        z = (D["gap"] / D["gapvol10"].shift()).dropna().iloc[-750:]
        self.gap_zq = np.quantile(z - z.median(), [0.1, 0.5, 0.9])
        zc = (D["cc"] / D["vol5"].shift()).dropna().iloc[-750:]
        self.cc_zq = np.quantile(zc - zc.median(), [0.1, 0.5, 0.9])
        self.beta_us, self.us_note = None, "US cue not used (no data)"
        try:  # learn how much of the S&P 500's overnight move shows up in India's open
            spx = _yahoo_daily("^GSPC", "5y")
            spx_ret = np.log(spx / spx.shift()) * 1e4
            # US session dated t (closes ~01:30 IST on t+1) -> India's open on the next Indian session after t
            us = pd.Series(spx_ret.values, index=spx_ret.index.tz_convert(TZ).normalize())
            rows = []
            dates = D.index
            for i in range(1, len(dates)):
                prev, cur = dates[i - 1], dates[i]
                m = us[(us.index >= prev) & (us.index < cur)]
                if len(m):
                    rows.append((m.sum(), D["gap"].iloc[i]))
            if len(rows) > 200:
                a = np.array(rows[-750:])
                x, y = a[:, 0], a[:, 1]
                self.beta_us = float(np.cov(x, y)[0, 1] / np.var(x))
                self.us_note = f"US cue weight {self.beta_us:.2f} (learned from {len(a)} nights)"
        except Exception as ex:
            self.us_note = f"US cue unavailable ({type(ex).__name__})"
        return self

    def forecast(self, last_close: float, us_move_bps: float | None = None) -> dict:
        D = self.D
        gv = float(D["gapvol10"].dropna().iloc[-1]) if D["gapvol10"].notna().any() else 60.0
        cv = float(D["vol5"].dropna().iloc[-1]) if D["vol5"].notna().any() else 80.0
        center = self.mean_gap
        used_us = us_move_bps is not None and self.beta_us is not None
        if used_us:
            center = self.mean_gap + self.beta_us * us_move_bps
        q = center + self.gap_zq * gv
        # chance of a gap up: base rate, shifted by the US cue when present
        if used_us:
            from math import erf, sqrt
            p_up = 0.5 * (1 + erf((center / max(gv, 1)) / sqrt(2)))
        else:
            p_up = self.p_gap_up
        qc = self.cc_zq * cv
        px = lambda bps: last_close * np.exp(bps / 1e4)
        return {"open_mid": px(center), "open_lo": px(q[0]), "open_hi": px(q[2]), "p_gap_up": float(p_up),
                "close_lo": px(qc[0]), "close_hi": px(qc[2]), "close_mid": px(0.0),
                "gap_mid_pts": px(center) - last_close, "used_us": used_us,
                "us_move_bps": us_move_bps, "note": self.us_note}

    @staticmethod
    def us_move_since(t_from: pd.Timestamp) -> float | None:
        """S&P 500 futures move (bps) since India's last close, if Yahoo is reachable."""
        try:
            es = _yahoo_daily("ES=F", "5d")
            now_px, _ = _yahoo_last("ES=F")
            ref = es[es.index <= t_from.tz_convert("UTC")]
            if not len(ref):
                return None
            return float(np.log(now_px / ref.iloc[-1]) * 1e4)
        except Exception:
            return None
