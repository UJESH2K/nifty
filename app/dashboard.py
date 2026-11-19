"""NiftySim dashboard.

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
def main_chart(eng: Engine, snap, i, h, variants, mode, height=560, trail=True):
    bars = snap["bars"]
    fig = go.Figure()
    n = len(bars)
    if n == 0:
        return fig
    i = int(min(max(i, 0), n - 1))
    bs = eng.bs
    day = bars.index.normalize()
    day_start = int(np.searchsorted(day.values, day[i].to_datetime64()))
    lo = day_start
    win = bars.iloc[lo:i + 1]
    fig.add_trace(go.Candlestick(
        x=_nv(win.index), open=win.open, high=win.high, low=win.low, close=win.close, name="NIFTY 50",
        increasing=dict(line=dict(color=C["up"], width=1), fillcolor=C["up"]),
        decreasing=dict(line=dict(color=C["down"], width=1), fillcolor=C["down"]), whiskerwidth=0.2,
        showlegend=False))
    anchor_px = float(bars["close"].iloc[i])
    forming = snap.get("forming")
    if mode in ("live", "replay") and forming:
        fig.add_trace(go.Candlestick(
            x=[_nv(forming["ts"])], open=[forming["open"]], high=[forming["high"]], low=[forming["low"]],
            close=[forming["close"]], name="Forming", opacity=0.6, showlegend=False,
            increasing=dict(line=dict(color=C["up"], width=1), fillcolor=C["up"]),
            decreasing=dict(line=dict(color=C["down"], width=1), fillcolor=C["down"])))
        anchor_px = float(snap["last_price"])
    session_close = eng.clock.session(bars.index[i])[1]
    hi_t = min(bars.index[i] + pd.Timedelta(seconds=bs * h), session_close - pd.Timedelta(seconds=bs))

    if mode == "travel":
        j_end = int(min(n - 1, i + h))
        fut = bars.iloc[i + 1:j_end + 1]
        fut = fut[fut.index.normalize() == day[i]]
        if len(fut):
            fig.add_trace(go.Candlestick(
                x=_nv(fut.index), open=fut.open, high=fut.high, low=fut.low, close=fut.close,
                name="What really happened", opacity=0.5,
                increasing=dict(line=dict(color=C["ghost"], width=1), fillcolor=C["ghost"]),
                decreasing=dict(line=dict(color="#4b465c", width=1), fillcolor="#4b465c")))

    close = bars["close"].values
    for v in variants:
        col = C[v]
        if trail:
            hi_ = int(round(h))
            js = np.arange(lo, i + 1)
            js = js[day[js] == day[i]]
            if len(js):
                q = eng.interp(v, hi_, rows=js)
                tx = bars.index[js] + pd.Timedelta(seconds=bs * hi_)
                keep = (tx <= hi_t) & (tx.normalize() == bars.index[js].normalize()) & ~np.isnan(q[:, 1])
                if keep.any():
                    fig.add_trace(go.Scatter(
                        x=_nv(tx[keep]), y=close[js][keep] * np.exp(q[keep, 1] / 1e4), mode="lines",
                        line=dict(color=col, width=1.6, dash="dot"), opacity=0.85,
                        name=f"{NAME[v]}: what it said {_hlabel(hi_)} earlier",
                        hovertemplate="%{y:,.1f}<extra>said " + _hlabel(hi_) + " earlier</extra>"))
        tr = eng.trajectory(v, i, h, anchor_px, step=max(1, int(h // 90)), live=mode in ("live", "replay"))
        if len(tr) > 1 and not tr[["p10", "p50", "p90"]].isna().any().any():
            x = _nv(tr["ts"])
            fig.add_trace(go.Scatter(x=x, y=tr.p90, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=tr.p10, mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor=_rgba(col, 0.16), name=f"{NAME[v]}: likely range (80%)", hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=x, y=tr.p50, mode="lines", line=dict(color=col, width=2.4), name=f"{NAME[v]}: predicted path",
                customdata=np.column_stack([tr.p10, tr.p90]),
                hovertemplate="%{x|%H:%M}  %{y:,.1f}<br>range %{customdata[0]:,.0f} – %{customdata[1]:,.0f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[x[-1]], y=[tr.p50.iloc[-1]], mode="markers", showlegend=False, hoverinfo="skip",
                                     marker=dict(size=10, color=col, line=dict(color=C["bg"], width=2))))
    # headlines
    news = snap["news"]
    if "news" in variants and len(news):
        nw = news[(news["ts"] >= win.index[0]) & (news["ts"] <= bars.index[i] + pd.Timedelta(seconds=bs))
                  & (news["cruciality"] >= 4)]
        if len(nw):
            fig.add_trace(go.Scatter(
                x=_nv(nw["ts"].dt.floor(f"{bs}s")), y=[win["high"].max()] * len(nw), mode="markers", name="Headline",
                marker=dict(symbol="diamond", size=7 + nw["cruciality"].values,
                            color=[C["up"] if d > .2 else C["down"] if d < -.2 else C["muted"] for d in nw["direction"]],
                            line=dict(color=C["bg"], width=2)),
                text=[f"{t}<br>direction {d:+.2f}, importance {c}/10" for t, d, c in
                      zip(nw["headline"], nw["direction"], nw["cruciality"])], hovertemplate="%{text}<extra></extra>"))
    now_x = _nv(bars.index[i])
    fig.add_vline(x=now_x, line=dict(color=C["muted"], width=1, dash="dash"))
    rb = [dict(bounds=["sat", "mon"])]
    if bs >= 60:
        rb.append(dict(bounds=[15.5, 9.25], pattern="hour"))
    fig.update_layout(
        height=height, margin=dict(l=6, r=6, t=6, b=6), paper_bgcolor=C["bg"], plot_bgcolor=C["bg"],
        font=dict(family="Manrope, system-ui", color=C["ink"], size=11), hovermode="x unified",
        uirevision=f"{mode}-{'-'.join(variants)}-{h}", hoverlabel=dict(bgcolor="#2b2738", bordercolor="#36314a"),
        xaxis=dict(rangeslider=dict(visible=False), gridcolor=C["grid"], rangebreaks=rb,
                   range=[_nv(win.index[0]), _nv(max(hi_t, bars.index[i]) + pd.Timedelta(seconds=bs * 2))]),
        yaxis=dict(gridcolor=C["grid"], side="right", tickformat=",.0f"),
        legend=dict(orientation="h", y=1.0, x=0.01, yanchor="top", bgcolor="rgba(33,30,44,.7)", font=dict(size=11)))
    return fig


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
def readout(eng, snap, i, h, v, mode, compact=False):
    bars = snap["bars"]
    live = mode in ("live", "replay") and snap.get("live_q", {}).get(v) and snap.get("live_ts") is not None
    anchor_px = float(snap["last_price"]) if mode in ("live", "replay") and np.isfinite(snap["last_price"]) \
        else float(bars["close"].iloc[i])
    t_now = snap["live_ts"] if live else bars.index[i]
    close_t = eng.clock.session(t_now)[1]
    h_eff = min(h, max(1, (close_t - t_now).total_seconds() / eng.bs - 1))
    q = eng.interp_live(v, h_eff) if live else eng.interp(v, h_eff, rows=[i])[0]
    if np.isnan(q).any():
        return html.Div("Not enough data yet for this horizon.", className="note")
    px = anchor_px * np.exp(q / 1e4)
    d = px[1] - anchor_px
    pu = eng.p_up(q)
    target_t = t_now + pd.Timedelta(seconds=eng.bs * h_eff)
    capped = h_eff < h
    when = f"In {_hlabel(h_eff)}, at {target_t:%H:%M}" + (" (market close)" if capped else "")
    kids = [html.Div([html.Span(className="dot", style={"background": C[v], "marginRight": "8px"}), NAME[v], ": ", when],
                     className="when"),
            html.Div(f"{px[1]:,.1f}", className="big"),
            *([html.Div(f"updated {pd.Timestamp(snap['last_ts']):%H:%M:%S}, refreshes every "
                        f"{int(eng.refresh_s)} s", className="note")] if live and snap.get("last_ts") is not None else []),
            html.Div(f"{d:+,.1f} pts ({d / anchor_px * 100:+.2f}%)", className="d",
                     style={"color": C["up"] if d > 0.05 else C["down"] if d < -0.05 else C["muted"]})]
    if not compact:
        kids += [html.Div([html.Span("Likely range (80%)"), html.B(f"{px[0]:,.0f} – {px[2]:,.0f}")], className="row"),
                 html.Div([html.Span("Chance it ends higher"), html.B(f"{pu * 100:.0f}%")], className="row")]
    return html.Div(kids, className="readout")


def alert_view(snap, within_min=30):
    """Big 'be prepared' banner for the latest critical moment, plus what to send as a notification."""
    al = snap.get("alerts") or []
    now = snap.get("live_ts") or (snap["bars"].index[-1] if len(snap["bars"]) else None)
    info = snap.get("refit_info") or ""
    if not al or now is None or (now - al[-1]["ts"]).total_seconds() > within_min * 60:
        return (html.Div(info, className="note") if info else None), None
    a = al[-1]
    big = a.get("big") and a["side"] != 0
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


def signal_box(eng, snap, i, v):
    aside, reasons = eng.stand_aside(v, i)
    sigs = [s for s in snap["signals"] if s["variant"] == v and i - s["h"] < s["i"] <= i]
    if aside:
        return html.Div("Stand aside: " + "; ".join(reasons), className="sig warn")
    if sigs:
        s = sigs[-1]
        return html.Div(f"Big move expected {'up' if s['side'] > 0 else 'down'}: about {abs(s['proj_pts']):.0f} pts in "
                        f"{_hlabel(s['h'])}. Your rule aims for {s['target_pts']:.0f}.", className="sig good")
    return html.Div("No strong call right now: the expected move is small next to normal noise.", className="sig")


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


def horizon_controls(prefix, default="30"):
    return html.Div([
        html.Div([html.Span("Predict ahead", className="lab"),
                  dcc.RadioItems(id=f"{prefix}-h", value=default, className="chips", inline=True,
                                 options=[{"label": a, "value": b} for a, b in HORIZONS])], className="field"),
        html.Div([dcc.Slider(id=f"{prefix}-hc", min=1, max=180, step=1, value=45,
                             marks={1: "1m", 30: "30m", 60: "1h", 120: "2h", 180: "3h"},
                             tooltip={"placement": "bottom", "always_visible": False})],
                 id=f"{prefix}-hc-wrap", className="field", style={"display": "none"}),
        html.Div([html.Span("Model", className="lab"),
                  dcc.RadioItems(id=f"{prefix}-v", value="news", className="seg", inline=True, options=[
                      {"label": "With news", "value": "news"}, {"label": "Price only", "value": "price"},
                      {"label": "Compare", "value": "both"}])], className="field"),
    ])


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


def build_app(cfg, engine: Engine, autopilot=None):
    global LIVE, BLIND, CFG, AUTO
    LIVE, CFG, AUTO = engine, cfg, autopilot
    BLIND = BlindSession(cfg)
    kw = dict(title="NiftySim", assets_folder=str(Path(__file__).parent / "assets"),
              external_stylesheets=["https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Sora:wght@600;700&display=swap"],
              suppress_callback_exceptions=True)
    try:   # all callbacks here are normal functions; async mode breaks on some Flask installs
        app = Dash(__name__, use_async=False, **kw)
    except TypeError:
        app = Dash(__name__, **kw)
    days = available_days(cfg)
    day_opts = [{"label": d.strftime("%a %d %b %Y"), "value": d.strftime("%Y-%m-%d")} for d in days[::-1]]
    models = list(REGISTRY)

    live = html.Div([
        html.Div([
            html.Div([
                html.Div([html.Div(id="orb-ring"),
                          html.Button(id="go", className="orb", children=[html.Span("Go live", className="big")])],
                         className="orb-wrap"),
                html.H1(id="hero-title"), html.P(id="hero-text"),
                dcc.RadioItems(id="src", value=cfg["feed"]["source"] if cfg["feed"]["source"] in dict((b, a) for a, b in SOURCES) else "auto",
                               className="sources", inline=True, options=[{"label": a, "value": b} for a, b in SOURCES]),
                html.Div(id="src-hint", className="hint"),
                html.Div([html.Button("Cancel", id="cancel", className="linkbtn"),
                          html.Button("Meanwhile, replay the last session", id="to-blind", className="linkbtn")]),
            ], className="hero"),
        ], id="live-hero"),
        html.Div([
            html.Div(id="live-head", className="pricehead"),
            html.Div(id="live-alert"),
            html.Div([
                html.Div(dcc.Graph(id="live-chart", config={"displaylogo": False, "scrollZoom": True}), className="panel chart"),
                html.Div([horizon_controls("l"), html.Div(id="live-readout"), html.Div(id="live-signal")], className="panel"),
            ], className="live-grid"),
            html.Div([
                html.Div(id="live-check", className="panel"),
                html.Div([html.H2("Every tick, last 15 minutes"), dcc.Graph(id="live-ticks", config={"displayModeBar": False})],
                         className="panel"),
            ], className="sub-grid"),
            html.Div([html.Div([html.H2("Headlines"), html.Div(id="live-news")], className="panel"),
                      html.Div([html.H2("Session"), html.Div(id="live-session"),
                                html.Button("Turn on pop-up alerts", id="enable-alerts", className="btn primary",
                                            style={"marginTop": "12px", "marginRight": "8px"}),
                                html.Button("Stop live", id="stop-live", className="btn",
                                            style={"marginTop": "12px", "display": "none" if autopilot else None})],
                               className="panel")], className="sub-grid"),
        ], id="live-main", style={"display": "none"}),
    ], id="sec-live")

    blind = html.Div([
        html.Div([
            html.H2("Blind replay"),
            html.P("Pick a past session. A model is trained only on the days before it, so it has never seen that day. "
                   "The day then plays back minute by minute with its headlines, and every prediction is checked "
                   "against what really happened.", className="note", style={"maxWidth": "80ch"}),
            html.Div([
                html.Div([html.Span("Session", className="lab"),
                          dcc.Dropdown(id="b-day", options=day_opts, value=day_opts[0]["value"] if day_opts else None,
                                       clearable=False, style={"width": "210px"})], className="field"),
                html.Div([html.Span("Model", className="lab"),
                          dcc.Dropdown(id="b-model", value="ridge", clearable=False, style={"width": "170px"},
                                       options=[{"label": m, "value": m} for m in models if m != "naive"])], className="field"),
                html.Div([html.Span("Speed", className="lab"),
                          dcc.RadioItems(id="b-speed", value="60", className="chips", inline=True, options=[
                              {"label": "Real time", "value": "1"}, {"label": "10×", "value": "10"},
                              {"label": "60×", "value": "60"}, {"label": "300×", "value": "300"},
                              {"label": "Max", "value": "5000"}])], className="field"),
                html.Div([html.Button("Start replay", id="b-start", className="btn primary")], className="field"),
            ], className="row-controls"),
            html.Div(id="b-status", className="note"), html.Div(id="b-progress"),
        ], className="panel", style={"marginBottom": "14px"}),
        html.Div([
            html.Div(id="b-head", className="pricehead"),
            html.Div(id="b-alert"),
            html.Div([
                html.Div(dcc.Graph(id="b-chart", config={"displaylogo": False, "scrollZoom": True}), className="panel chart"),
                html.Div([horizon_controls("b"), html.Div(id="b-readout"), html.Div(id="b-signal")], className="panel"),
            ], className="live-grid"),
            html.Div([html.Div(id="b-check", className="panel"),
                      html.Div([html.H2("Headlines that day"), html.Div(id="b-news")], className="panel")], className="sub-grid"),
            html.Div([html.H2("Every horizon for this session"), html.Div(id="b-score")], className="panel",
                     style={"marginTop": "14px"}),
        ], id="b-main", style={"display": "none"}),
    ], id="sec-blind", style={"display": "none"})

    lab = html.Div([
        html.Div([
            html.Div([html.Span("Move through history. The model only uses data up to the point you choose; dimmed "
                                "candles show what happened next.", className="lab"),
                      dcc.Slider(id="lab-tt", min=0, max=max(len(engine.bars) - 1, 1), step=1,
                                 value=max(len(engine.bars) - 1, 0), marks=None)], style={"flex": 1}),
            html.Button("◀", id="lab-back", className="btn"), html.Button("▶", id="lab-fwd", className="btn"),
            html.Div([dcc.Dropdown(id="lab-model", value=engine.model_name, clearable=False, style={"width": "150px"},
                                   options=[{"label": m, "value": m} for m in models])],
                     style={"display": "none"} if autopilot else {}),
            html.Button("Retrain now", id="lab-retrain", className="btn",
                        style={"display": "none"} if autopilot else {}),
            html.Span(id="lab-rt", className="note"),
        ], className="panel row-controls", style={"marginBottom": "14px", "alignItems": "center"}),
        html.Div(id="lab-banner"),
        html.Div([
            html.Div(dcc.Graph(id="lab-chart", config={"displaylogo": False, "scrollZoom": True}), className="panel chart"),
            html.Div([horizon_controls("x"), html.Div(id="lab-readout"), html.Div(id="lab-signal")], className="panel"),
        ], className="live-grid"),
        html.Div([html.H2("Scorecard over the last 5 sessions in memory"), html.Div(id="lab-score")], className="panel",
                 style={"marginTop": "14px"}),
        html.Div([html.H2("Is it getting better? One point per trading day, after each nightly retrain"),
                  dcc.Graph(id="lab-improve", config={"displayModeBar": False}),
                  html.P("Below the dashed line means the model missed by less than a plain 'no change' guess that day. "
                         "Points appear after each session once autopilot is running.", className="note")],
                 className="panel", style={"marginTop": "14px"}),
    ], id="sec-lab", style={"display": "none"})

    app.layout = html.Div([
        html.Div([
            html.Div([html.I(), "NiftySim"], className="logo"),
            dcc.RadioItems(id="tab", value="live", className="nav", inline=True, options=[
                {"label": "Live", "value": "live"}, {"label": "Blind replay", "value": "blind"}, {"label": "Lab", "value": "lab"}]),
            html.Div(id="status", className="status"),
        ], className="bar"),
        live, blind, lab,
        dcc.Interval(id="tick", interval=int(__import__("os").environ.get("NIFTYSIM_UI_REFRESH_MS", 1000))),
        dcc.Store(id="alert-store"),
        html.Div(id="sink", style={"display": "none"}),
        html.Div(id="alert-sink", style={"display": "none"}),
        html.Div(id="perm-sink", style={"display": "none"}),
    ], className="app")

    # ---------------------------------------------------------------- browser notifications
    app.clientside_callback(
        """
        function(a) {
          if (!a || !a.id) { return window.dash_clientside.no_update; }
          if (window._nsLastAlert === a.id) { return window.dash_clientside.no_update; }
          window._nsLastAlert = a.id;
          try { if ("Notification" in window && Notification.permission === "granted") {
                  new Notification(a.title, {body: a.body}); } } catch (e) {}
          try { const C = window.AudioContext || window.webkitAudioContext; const c = new C();
                const o = c.createOscillator(); const g = c.createGain(); o.connect(g); g.connect(c.destination);
                g.gain.value = 0.08; o.frequency.value = 880; o.start();
                setTimeout(() => { o.frequency.value = 660; }, 180);
                setTimeout(() => { o.stop(); c.close(); }, 420); } catch (e) {}
          return "";
        }
        """, Output("alert-sink", "children"), Input("alert-store", "data"))
    app.clientside_callback(
        """
        function(n) {
          if (!n) { return window.dash_clientside.no_update; }
          try { Notification.requestPermission(); } catch (e) {}
          return "";
        }
        """, Output("perm-sink", "children"), Input("enable-alerts", "n_clicks"))

    # ---------------------------------------------------------------- navigation
    @app.callback(Output("sec-live", "style"), Output("sec-blind", "style"), Output("sec-lab", "style"),
                  Input("tab", "value"))
    def _nav(tab):
        show, hide = {}, {"display": "none"}
        return (show if tab == "live" else hide, show if tab == "blind" else hide, show if tab == "lab" else hide)

    @app.callback(Output("tab", "value"), Input("to-blind", "n_clicks"), prevent_initial_call=True)
    def _toblind(_):
        return "blind"

    for p in ("l", "b", "x"):
        @app.callback(Output(f"{p}-hc-wrap", "style"), Input(f"{p}-h", "value"))
        def _hc(v):
            return {} if v == "c" else {"display": "none"}

    # ---------------------------------------------------------------- live
    @app.callback(Output("sink", "children"), Input("go", "n_clicks"), Input("cancel", "n_clicks"),
                  Input("stop-live", "n_clicks"), State("src", "value"), prevent_initial_call=True)
    def _go(_a, _b, _c, src):
        if AUTO is not None:
            return ""
        if ctx.triggered_id == "go" and not LIVE.armed:
            LIVE.go_live(src)
        elif ctx.triggered_id in ("cancel", "stop-live"):
            LIVE.disarm()
        return ""

    @app.callback(Output("src-hint", "children"), Input("src", "value"))
    def _hint(src):
        return {"auto": "No account needed. Reads NSE's own website; switches to Yahoo by itself if NSE blocks it.",
                "yahoo": "No account needed. Prices can lag the exchange a little.",
                "angel": "Needs an Angel One demat account and its keys in the .env file.",
                "upstox": "Needs an Upstox demat account and today's access token in the .env file.",
                "synthetic": "A simulated market that starts right away, for trying the screen out."}[src]

    @app.callback(Output("live-hero", "style"), Output("live-main", "style"), Output("orb-ring", "children"),
                  Output("go", "className"), Output("go", "children"), Output("hero-title", "children"),
                  Output("hero-text", "children"), Output("cancel", "style"), Output("src", "style"),
                  Output("status", "children"),
                  Output("live-head", "children"), Output("live-chart", "figure"), Output("live-readout", "children"),
                  Output("live-signal", "children"), Output("live-check", "children"), Output("live-ticks", "figure"),
                  Output("live-news", "children"), Output("live-session", "children"),
                  Output("live-alert", "children"), Output("alert-store", "data"),
                  Input("tick", "n_intervals"), Input("tab", "value"), Input("l-h", "value"), Input("l-hc", "value"),
                  Input("l-v", "value"))
    def _live(_n, tab, chip, custom, var):
        snap = LIVE.snapshot()
        st = LIVE.clock.status()
        running = LIVE.status.startswith("live") or LIVE.status == "session over"
        # top-right status
        if st["open"]:
            stat = [html.Span(className="dot on"), html.Span(f"Market open, closes in {fmt_countdown(st['secs_to_close'])}")]
        else:
            stat = [html.Span(className="dot"), html.Span(f"{st['label']}. Opens {st['next_open']:%a %d %b, %H:%M}")]
        if tab != "live":
            return [no_update] * 9 + [stat] + [no_update] * 10
        hide, show = {"display": "none"}, {}
        if not running:
            total = 17.75 * 3600
            if not st["open"]:
                total = max(3600.0, (st["next_open"] - LIVE.clock.session(LIVE.clock.last_session_day())[1]).total_seconds())
            frac = 1.0 if st["open"] else max(0.0, min(1.0, 1 - st.get("secs_to_open", 0) / total))
            circ = 2 * np.pi * 140
            ring = _ring_svg(frac, circ)
            if AUTO is not None:
                nxt = st.get("next_open")
                orb_cls = "orb armed"
                orb_kids = ([html.Span(fmt_countdown(st.get("secs_to_open", 0)), className="big"), html.Small("until the open")]
                            if not st["open"] else [html.Span("Starting…", className="big pulse"), html.Small(snap.get("message") or "")])
                title = (f"Starts by itself at {nxt:%H:%M} on {nxt:%A}" if nxt is not None else "Starting now")
                text = ("Nothing to press. Every trading day it predicts the market live, checks itself, and "
                        f"retrains overnight on what it saw. Status: {AUTO.state}.")
                return ([show, hide, ring, orb_cls, orb_kids, title, text, hide, hide, stat] + [no_update] * 10)
            if snap["armed"] and LIVE.status == "waiting":
                orb_cls = "orb armed"
                orb_kids = [html.Span(fmt_countdown(st.get("secs_to_open", 0)), className="big"), html.Small("until the open")]
                title = f"Ready. Analysis starts at {st['next_open']:%H:%M} on {st['next_open']:%A}"
                text = ("Keep this tab open. When trading begins, it loads today's prices and starts predicting "
                        "on its own.")
                cancel_style, src_style = {}, hide
            elif snap["armed"]:
                orb_cls, orb_kids = "orb armed", [html.Span("Starting…", className="big pulse"),
                                                  html.Small(snap.get("message") or "connecting")]
                title, text, cancel_style, src_style = "Connecting to the market", snap.get("message") or "", {}, hide
            else:
                orb_cls = "orb"
                orb_kids = [html.Span("Go live", className="big"),
                            html.Small("starts now" if st["open"] else f"starts {st['next_open']:%a %H:%M}")]
                if st["open"]:
                    title, text = "The market is open", "Press Go live to start predicting from the next price update."
                else:
                    title = f"The market opens {st['next_open']:%A} at {st['next_open']:%H:%M}"
                    text = ("Press Go live now and leave this tab open. Analysis starts by itself when trading begins. "
                            "Pick where prices come from below.")
                cancel_style, src_style = hide, {}
            return ([show, hide, ring, orb_cls, orb_kids, title, text, cancel_style, src_style, stat]
                    + [no_update] * 10)

        # ---- running
        bars = snap["bars"]
        i = len(bars) - 1
        h = _hval(chip, custom)
        vs = list(VARIANTS)[::-1] if var == "both" else [var]
        fig = main_chart(LIVE, snap, i, h, vs, "live")
        lp = snap["last_price"]
        today = bars[bars.index.normalize() == bars.index[i].normalize()]
        prev_close = bars[bars.index.normalize() < bars.index[i].normalize()]["close"]
        ref = float(prev_close.iloc[-1]) if len(prev_close) else float(today["open"].iloc[0])
        chg = lp - ref
        src_name = snap.get("using") or dict((b, a) for a, b in SOURCES).get(snap["source"], snap["source"])
        lag = snap.get("lag") or 0
        head = [html.Span("NIFTY 50", className="name"), html.Span(f"{lp:,.2f}", className="price"),
                html.Span(f"{chg:+,.2f} ({chg / ref * 100:+.2f}%)", className="chg " + ("up" if chg >= 0 else "down")),
                html.Span([html.Span(className="dot on" if LIVE.status.startswith("live") else "dot",
                                     style={"marginRight": "8px"}),
                           f"{src_name}" + (f", about {lag:.0f}s behind" if lag > 20 else "")
                           + (f". Feed error: {snap['feed_error']}" if snap.get("feed_error") else "")], className="meta")]
        if LIVE.status == "session over":
            head.append(html.Span("Session over. Everything below is today's record.", className="meta"))
        ro = ([readout(LIVE, snap, i, h, v, "live", compact=(var == "both")) for v in vs])
        sig = signal_box(LIVE, snap, i, vs[0])
        since = bars.index[i].normalize()
        chk = check_box(LIVE, vs[0], h, since)
        sess = html.Div([
            html.Div(f"Autopilot: {AUTO.state}" if AUTO is not None else "", className="note"),
            html.Div(f"Bars today: {len(today)}. Headlines today: "
                     f"{int((snap['news']['ts'] >= since).sum()) if len(snap['news']) else 0}.", className="note"),
            html.Div(f"Model: {snap['model']}. Every tick, 1-second and 1-minute bar is saved to the data folder, "
                     "so tonight's retrain learns from today.", className="note"),
        ])
        banner, note = alert_view(snap)
        return ([hide, show, no_update, no_update, no_update, no_update, no_update, no_update, no_update, stat,
                 head, fig, ro, sig, chk, tick_chart(snap["ticks"]),
                 news_list(snap, bars.index[i] + pd.Timedelta(minutes=1)), sess, banner, note or no_update])

    # ---------------------------------------------------------------- blind
    @app.callback(Output("b-status", "children", allow_duplicate=True), Input("b-start", "n_clicks"),
                  State("b-day", "value"), State("b-model", "value"), State("b-speed", "value"), prevent_initial_call=True)
    def _bstart(_, day, model, speed):
        if not day:
            return "No full sessions found in the data folder yet."
        BLIND.start(day, model, float(speed))
        return BLIND.msg

    @app.callback(Output("sink", "title"), Input("b-speed", "value"), prevent_initial_call=True)
    def _bspeed(s):
        BLIND.set_speed(float(s))
        return ""

    @app.callback(Output("b-status", "children"), Output("b-progress", "children"), Output("b-main", "style"),
                  Output("b-head", "children"), Output("b-chart", "figure"), Output("b-readout", "children"),
                  Output("b-signal", "children"), Output("b-check", "children"), Output("b-news", "children"),
                  Output("b-score", "children"), Output("b-alert", "children"),
                  Input("tick", "n_intervals"), Input("tab", "value"), Input("b-h", "value"), Input("b-hc", "value"),
                  Input("b-v", "value"))
    def _blind(_n, tab, chip, custom, var):
        if tab != "blind":
            return [no_update] * 11
        e = BLIND.engine
        prog = BLIND.feed.progress if BLIND.feed else 0
        pbar = html.Div(html.Span(style={"width": f"{prog * 100:.1f}%"}), className="progress") if BLIND.state != "idle" else ""
        if e is None or BLIND.state in ("idle", "training", "error"):
            return [BLIND.msg, pbar, {"display": "none"}] + [no_update] * 8
        snap = e.snapshot()
        bars = snap["bars"]
        if not len(bars) or bars.index[-1] < BLIND.day:
            return [BLIND.msg + " Waiting for the first minute…", pbar, {"display": "none"}] + [no_update] * 8
        i = len(bars) - 1
        h = _hval(chip, custom)
        vs = list(VARIANTS)[::-1] if var == "both" else [var]
        lp = float(snap["last_price"])
        prev = bars[bars.index < BLIND.day]["close"]
        ref = float(prev.iloc[-1]) if len(prev) else float(bars["open"].iloc[0])
        chg = lp - ref
        head = [html.Span(f"NIFTY 50, {BLIND.day:%a %d %b}", className="name"),
                html.Span(f"{lp:,.2f}", className="price"),
                html.Span(f"{chg:+,.2f} ({chg / ref * 100:+.2f}%)", className="chg " + ("up" if chg >= 0 else "down")),
                html.Span(f"replay clock {pd.Timestamp(snap['last_ts']):%H:%M:%S}" if snap["last_ts"] is not None else "",
                          className="meta")]
        fig = main_chart(e, snap, i, h, vs, "replay")
        ro = [readout(e, snap, i, h, v, "replay", compact=(var == "both")) for v in vs]
        return [BLIND.msg, pbar, {}, head, fig, ro, signal_box(e, snap, i, vs[0]),
                check_box(e, vs[0], h, BLIND.day, "How it's doing this session"),
                news_list(snap, bars.index[i] + pd.Timedelta(minutes=1)),
                score_table(e, BLIND.day), alert_view(snap)[0]]

    # ---------------------------------------------------------------- lab
    @app.callback(Output("lab-tt", "max"), Output("lab-tt", "value"), Input("tab", "value"),
                  Input("lab-back", "n_clicks"), Input("lab-fwd", "n_clicks"), State("lab-tt", "value"))
    def _labtt(tab, _b, _f, val):
        mx = max(len(LIVE.bars) - 1, 1)
        if ctx.triggered_id == "lab-back":
            return mx, max(0, (val or mx) - 1)
        if ctx.triggered_id == "lab-fwd":
            return mx, min(mx, (val or 0) + 1)
        if ctx.triggered_id == "tab" and tab == "lab":
            day = LIVE.bars.index.normalize()
            u = day.unique()
            d = u[-2] if len(u) > 1 else u[-1]
            idx = np.where(day == d)[0]
            return mx, int(idx[len(idx) // 3]) if len(idx) else mx
        return mx, no_update

    @app.callback(Output("lab-rt", "children"), Input("lab-model", "value"), Input("lab-retrain", "n_clicks"),
                  prevent_initial_call=True)
    def _labmodel(m, _r):
        if ctx.triggered_id == "lab-retrain":
            if RETRAIN["state"] == "running":
                return "Already retraining…"

            def job():
                from niftysim.train import train_all
                RETRAIN["state"] = "running"
                try:
                    LIVE.flush()
                    train_all(CFG, LIVE.res, [LIVE.model_name], do_cv=False, log=lambda *_: None)
                    LIVE.set_model(LIVE.model_name)
                    RETRAIN["state"] = "Retrained at " + pd.Timestamp.now().strftime("%H:%M")
                except Exception as ex:
                    RETRAIN["state"] = f"Retrain failed: {ex}"
            threading.Thread(target=job, daemon=True).start()
            return "Retraining in the background…"
        try:
            LIVE.set_model(m)
            return f"Using {m}"
        except FileNotFoundError:
            return f"{m} isn't trained yet. Run: python scripts/train_models.py --models {m}"

    @app.callback(Output("lab-improve", "figure"), Input("tab", "value"))
    def _improve(tab):
        return improvement_chart(CFG) if tab == "lab" else no_update

    @app.callback(Output("lab-banner", "children"), Output("lab-chart", "figure"), Output("lab-readout", "children"),
                  Output("lab-signal", "children"), Output("lab-score", "children"),
                  Input("tab", "value"), Input("lab-tt", "value"), Input("x-h", "value"), Input("x-hc", "value"),
                  Input("x-v", "value"), Input("lab-rt", "children"))
    def _lab(tab, tt, chip, custom, var, _rt):
        if tab != "lab":
            return [no_update] * 5
        snap = LIVE.snapshot()
        bars = snap["bars"]
        if not len(bars):
            return ["No data yet."] + [no_update] * 4
        i = int(min(tt or len(bars) - 1, len(bars) - 1))
        h = _hval(chip, custom)
        vs = list(VARIANTS)[::-1] if var == "both" else [var]
        te = snap["train_end"]
        t = bars.index[i]
        if te is not None and t <= te:
            ban = html.Div("The model was trained on this period, so it will look better here than it really is. "
                           "Use Blind replay for an honest test.", className="banner ins")
        else:
            ban = html.Div("The model has never seen this period. This is an honest test.", className="banner oos")
        snap["forming"] = None
        fig = main_chart(LIVE, snap, i, h, vs, "travel")
        ro = [readout(LIVE, snap, i, h, v, "travel", compact=(var == "both")) for v in vs]
        days = bars.index.normalize().unique()
        since = days[-5] if len(days) >= 5 else days[0]
        return ban, fig, ro, signal_box(LIVE, snap, i, vs[0]), score_table(LIVE, since)

    return app


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
