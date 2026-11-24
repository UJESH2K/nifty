"""Shared dashboard helpers (kept from v2).

Live         : press Go live; it waits for 09:15 IST on its own, then overlays the model's path
               for any horizon (1 min to 3 h, or custom) on the real market and checks itself.
Blind replay : train a model that has never seen a chosen day, then replay that day at any speed.
Lab          : time-travel through history, full scorecard, retrain.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from niftysim.blind import BlindSession, available_days  # noqa: E402
from niftysim.engine import VARIANTS, Engine  # noqa: E402
from niftysim.market_clock import fmt_countdown  # noqa: E402
from niftysim.models import REGISTRY  # noqa: E402

TZ = "Asia/Kolkata"
C = {"news": "#ab9ff2", "price": "#f5b451", "up": "#4cd9a0", "down": "#ff7a8a", "bg": "#211e2c",
     "grid": "#2e2a3c", "ink": "#c9c3dd", "muted": "#9a94b3", "ghost": "#6d6782"}
NAME = {"news": "With news", "price": "Price only"}
HORIZONS = [("1m", "1"), ("5m", "5"), ("10m", "10"), ("15m", "15"), ("30m", "30"),
            ("1h", "60"), ("2h", "120"), ("3h", "180"), ("Custom", "c")]
SOURCES = [("NSE + Yahoo (no account)", "auto"), ("Yahoo only", "yahoo"), ("Angel One (demat)", "angel"),
           ("Upstox (demat)", "upstox"), ("Demo market", "synthetic")]

LIVE: Engine | None = None
AUTO = None   # Autopilot instance when running unattended
BLIND: BlindSession | None = None
CFG: dict = {}
RETRAIN = {"state": ""}


def _rgba(hex_, a):
    h = hex_.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _nv(x):
    """tz-naive local times for plotly"""
    if isinstance(x, pd.Timestamp):
        return x.tz_localize(None) if x.tzinfo else x
    idx = pd.DatetimeIndex(x)
    return idx.tz_localize(None) if idx.tz is not None else idx


def _hval(chip, custom):
    return float(custom) if chip == "c" else float(chip)


def _hlabel(h):
    h = int(round(h))
    if h < 60:
        return f"{h} min"
    return f"{h // 60} h" + (f" {h % 60} min" if h % 60 else "")


# ======================================================================================== chart

def tick_chart(ticks):
    fig = go.Figure()
    if ticks:
        t, p = zip(*ticks[-900:])
        fig.add_trace(go.Scatter(x=_nv(pd.DatetimeIndex(t)), y=p, mode="lines", line=dict(color="#f2effa", width=1.2),
                                 hovertemplate="%{x|%H:%M:%S}  %{y:,.2f}<extra></extra>"))
    fig.update_layout(height=190, margin=dict(l=6, r=6, t=6, b=6), paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
                      font=dict(family="Manrope", color=C["ink"], size=10), showlegend=False, uirevision="ticks",
                      xaxis=dict(gridcolor=C["grid"]), yaxis=dict(gridcolor=C["grid"], side="right", tickformat=",.0f"))
    return fig


# ======================================================================================== panels
def alert_view(snap, within_min=30):
    """Big 'be prepared' banner for the latest critical moment, plus what to send as a notification."""
    al = snap.get("alerts") or []
    now = snap.get("live_ts") or (snap["bars"].index[-1] if len(snap["bars"]) else None)
    info = snap.get("refit_info") or ""
    if not al or now is None or (now - al[-1]["ts"]).total_seconds() > within_min * 60:
        return (html.Div(info, className="note") if info else None), None
    a = al[-1]
    # Only call a direction when the whole likely range is on one side; otherwise say "either way".
    big = a.get("big") and ((a["side"] > 0 and a["lo_pts"] > 0) or (a["side"] < 0 and a["hi_pts"] < 0))
    if big:
        head = f"Be prepared: a big move {'UP' if a['side'] > 0 else 'DOWN'} looks likely in the next {a['h']} min"
    else:
        lean = {1: ", leaning up", -1: ", leaning down"}.get(a["side"], "")
        head = (f"Be prepared: something important just happened. It could move sharply either way in "
                f"the next {a['h']} min{lean}")
    nums = (f"Expected {a['proj_pts']:+.0f} pts, likely range {a['lo_pts']:+.0f} to {a['hi_pts']:+.0f}. "
            + (f"Your one-third target: {a['target_pts']:+.0f} pts." if big else
               "No one-third target: the expected move is too small to call a direction."))
    body = [html.Div([html.B(f"{pd.Timestamp(a['ts']):%H:%M}  "), head], style={"fontSize": "16px", "fontWeight": 800}),
            html.Div(nums, style={"marginTop": "4px"}),
            html.Div("Why: " + "; ".join(a["reasons"]), className="note", style={"marginTop": "4px"})]
    if info:
        body.append(html.Div(("Re-tuning now…" if snap.get("refitting") else info), className="note"))
    banner = html.Div(body, className="alert " + (("up" if a["side"] > 0 else "down") if big else "either"))
    note = {"id": f"{pd.Timestamp(a['ts']).isoformat()}", "title": "NiftySim: " + head, "body": nums}
    return banner, note


def check_box(eng, v, h, since, title="How it's doing today"):
    lc = eng.live_check(v, h, since=since)
    if not lc.get("n"):
        return html.Div([html.H2(title), html.Div(f"Checks start {_hlabel(h)} after the first prediction.", className="note")])
    dr = lc["dir_right"]
    better = lc["miss_pts"] < lc["nochange_miss_pts"] * 0.98
    worse = lc["miss_pts"] > lc["nochange_miss_pts"] * 1.02
    verdict = ("Beating a plain 'no change' guess so far." if better else
               "Doing worse than a plain 'no change' guess so far." if worse else
               "About the same as a plain 'no change' guess, which is normal for most of the day.")
    return html.Div([
        html.H2(title + f" ({_hlabel(h)} ahead, {NAME[v].lower()})"),
        html.Div([
            html.Div([html.Div("Predictions checked", className="k"), html.Div(f"{lc['n']}", className="v")], className="mini"),
            html.Div([html.Div("Direction right", className="k"),
                      html.Div("—" if np.isnan(dr) else f"{dr * 100:.0f}%", className="v")], className="mini"),
            html.Div([html.Div("Price landed inside the range", className="k"),
                      html.Div(f"{lc['in_range'] * 100:.0f}%", className="v")], className="mini"),
            html.Div([html.Div("Average miss vs 'no change'", className="k"),
                      html.Div(f"{lc['miss_pts']:.1f} vs {lc['nochange_miss_pts']:.1f}", className="v")], className="mini"),
        ], className="two"),
        html.Div(verdict, className="verdict " + ("good" if better else "bad" if worse else "")),
    ])


def news_list(snap, until, n=12):
    news = snap["news"]
    if not len(news):
        return html.Div("No headlines yet. They appear here as they arrive, scored for direction and importance.",
                        className="note")
    nw = news[news["ts"] <= until].tail(n).iloc[::-1]
    out = []
    for r in nw.itertuples():
        d = float(r.direction)
        w = abs(d) * 50
        out.append(html.Div([
            html.Div(pd.Timestamp(r.ts).strftime("%H:%M") if pd.Timestamp(r.ts).normalize() == until.normalize()
                     else pd.Timestamp(r.ts).strftime("%d %b"), className="ts"),
            html.Div([html.Div(r.headline), html.Div(f"{r.category}, {r.source}", className="tag")]),
            html.Div([html.Div(html.Span(style={"left": f"{50 if d >= 0 else 50 - w}%", "width": f"{w}%",
                                                "background": C["up"] if d >= 0 else C["down"]}), className="dirbar"),
                      html.Div(f"importance {int(r.cruciality)}", className="tag")])], className="news-item"))
    return html.Div(out)


def score_table(eng, since, variants=VARIANTS):
    rows = []
    for v in variants:
        for h in eng.H:
            lc = eng.live_check(v, h, since=since)
            if not lc.get("n"):
                continue
            r = lc["miss_pts"] / max(lc["nochange_miss_pts"], 1e-9)
            better, same = r < 0.99, abs(r - 1) <= 0.01
            rows.append(html.Tr([
                html.Td([html.Span(className="dot", style={"background": C[v], "marginRight": "8px"}), f"{NAME[v]}, {_hlabel(h)}"]),
                html.Td(lc["n"]), html.Td(f"{lc['miss_pts']:.1f}"), html.Td(f"{lc['nochange_miss_pts']:.1f}"),
                html.Td("same" if same else "better" if better else "worse",
                        className="" if same else "good-t" if better else "bad-t"),
                html.Td("—" if np.isnan(lc["dir_right"]) else f"{lc['dir_right'] * 100:.0f}%"),
                html.Td(f"{lc['in_range'] * 100:.0f}%")]))
    return html.Div([html.Table([html.Thead(html.Tr([html.Th(x) for x in (
        "Model and horizon", "Checked", "Avg miss (pts)", "'No change' miss", "Verdict", "Direction right",
        "Inside 80% range")]))] + [html.Tbody(rows)], className="t"), html.P(
        "A dash under 'Direction right' means the model made no directional call at that horizon: on days it hadn't "
        "seen, its calls there didn't help, so it learned to predict 'no change' and only give a range.",
        className="note", style={"marginTop": "8px"})])


# ======================================================================================== layout
def improvement_chart(cfg):
    from niftysim.autopilot import load_daily_scores
    df = load_daily_scores(cfg)
    fig = go.Figure()
    if len(df):
        df = df[df["horizon"].isin([5, 30, 60])]
        for (v, h), g in df.groupby(["variant", "horizon"]):
            g = g.sort_values("date")
            fig.add_trace(go.Scatter(
                x=g["date"], y=(g["miss_pts"] / g["nochange_miss_pts"]).round(3), mode="lines+markers",
                name=f"{NAME[v]}, {_hlabel(h)}", line=dict(color=C[v], width=2,
                                                          dash={5: "dot", 30: "solid", 60: "dash"}[h]),
                hovertemplate="%{x|%d %b}: %{y:.3f}<extra></extra>"))
    fig.add_hline(y=1, line=dict(color=C["muted"], width=1, dash="dash"))
    fig.update_layout(height=300, margin=dict(l=6, r=6, t=6, b=6), paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
                      font=dict(family="Manrope", color=C["ink"], size=11),
                      yaxis=dict(gridcolor=C["grid"], title="miss ÷ 'no change' miss (below 1 = better)"),
                      xaxis=dict(gridcolor=C["grid"], type="date", tickformat="%d %b", dtick=86400000),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    return fig


def _ring_svg(frac, circ):
    """Countdown ring around the orb, drawn as an inline SVG via an <img> data URI (Dash has no svg tags)."""
    import base64
    off = circ * (1 - frac)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 300" width="300" height="300">'
           f'<circle cx="150" cy="150" r="140" fill="none" stroke="#2b2738" stroke-width="6"/>'
           f'<circle cx="150" cy="150" r="140" fill="none" stroke="#ab9ff2" stroke-width="6" stroke-linecap="round" '
           f'stroke-dasharray="{circ:.1f}" stroke-dashoffset="{off:.1f}" transform="rotate(-90 150 150)"/></svg>')
    return html.Img(src="data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode(),
                    style={"position": "absolute", "inset": 0, "width": "300px", "height": "300px"}, alt="")
