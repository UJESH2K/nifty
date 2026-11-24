"""v3 views: a clear single-day chart, the prediction ladder, the stop/target plan, the prediction log
(what it said vs what happened) and the next-session card."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from app.parts import C, NAME, _hlabel, _nv, _rgba

WINDOWS = [("Day", "day"), ("2 h", "2h"), ("1 h", "1h"), ("30 min", "30m")]
LADDER_H = [1, 5, 10, 30, 60, 180]


def _pts(x):
    return f"{x:+,.0f}" if abs(x) >= 1 else f"{x:+.1f}"


def _q_at(eng, v, h, i, live):
    q = eng.interp_live(v, h) if live else None
    if q is None:
        q = eng.interp(v, h, rows=[i])[0]
    return q


# ============================================================================================ chart
def day_chart(eng, snap, i, h, variants, mode, window="day", show_log=True):
    """mode: live | replay | travel. One trading day, a labelled 'now', the predicted path with its
    80% range, and (show_log) the checkpoints: what the model said earlier vs what really happened."""
    bars = snap["bars"]
    fig = go.Figure()
    if not len(bars):
        return fig
    i = int(min(max(i, 0), len(bars) - 1))
    bs = eng.bs
    t_i = bars.index[i]
    day = t_i.normalize()
    o_t, c_t = eng.clock.session(t_i)
    first = int(np.searchsorted(bars.index.values, day.to_datetime64()))
    today = bars.iloc[first:i + 1]
    live = mode in ("live", "replay")
    anchor = float(snap["last_price"]) if live and np.isfinite(snap["last_price"]) else float(bars["close"].iloc[i])
    now_t = snap.get("live_ts") if live and snap.get("live_ts") is not None else t_i
    up, dn = C["up"], C["down"]

    # ---- x window (constant for 'day' so zoom/pan survive every refresh)
    if window == "day":
        x0, x1 = o_t - pd.Timedelta(minutes=3), c_t + pd.Timedelta(minutes=3)
    else:
        back, fwd = {"2h": (80, 45), "1h": (40, 25), "30m": (20, 12)}[window]
        fwd = max(fwd, min(h, {"2h": 60, "1h": 30, "30m": 15}[window]) + 4)
        q = pd.Timedelta(minutes=5)
        x0 = (now_t - pd.Timedelta(minutes=back)).floor("5min")
        x1 = min(c_t + pd.Timedelta(minutes=3), (now_t + pd.Timedelta(minutes=fwd)).ceil("5min") + q * 0)
        x0 = max(x0, o_t - pd.Timedelta(minutes=3))

    # future shading
    fig.add_vrect(x0=_nv(now_t), x1=_nv(c_t), fillcolor=_rgba(C["news"], 0.05), line_width=0, layer="below")

    fig.add_trace(go.Candlestick(
        x=_nv(today.index), open=today.open, high=today.high, low=today.low, close=today.close, name="NIFTY 50",
        increasing=dict(line=dict(color=up, width=1), fillcolor=up),
        decreasing=dict(line=dict(color=dn, width=1), fillcolor=dn), whiskerwidth=0.2, showlegend=False,
        hoverinfo="x+y"))
    forming = snap.get("forming")
    if live and forming:
        fig.add_trace(go.Candlestick(
            x=[_nv(forming["ts"])], open=[forming["open"]], high=[forming["high"]], low=[forming["low"]],
            close=[forming["close"]], opacity=0.55, showlegend=False, hoverinfo="skip",
            increasing=dict(line=dict(color=up, width=1), fillcolor=up),
            decreasing=dict(line=dict(color=dn, width=1), fillcolor=dn)))
    if mode == "travel":
        fut = bars.iloc[i + 1:]
        fut = fut[fut.index.normalize() == day]
        if len(fut):
            fig.add_trace(go.Candlestick(
                x=_nv(fut.index), open=fut.open, high=fut.high, low=fut.low, close=fut.close,
                name="What really happened next", opacity=0.45,
                increasing=dict(line=dict(color=C["ghost"], width=1), fillcolor=C["ghost"]),
                decreasing=dict(line=dict(color="#4b465c", width=1), fillcolor="#4b465c")))

    v0 = variants[0]
    # ---- prediction log on the chart: checkpoints every 15/30 min
    if show_log:
        cps = checkpoints(eng, bars, i, h, v0, first)
        if cps:
            cx, cy = [], []
            for c in cps:
                cx += [_nv(c["t0"]), _nv(c["t1"]), None]
                cy += [c["p0"], c["pred"], None]
            fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", line=dict(color=C["muted"], width=1, dash="dot"),
                                     hoverinfo="skip", showlegend=False))
            for status, col, name in (("pending", C["news"], "Earlier call, not due yet"),
                                      ("hit", C["up"], "Earlier call: landed in range"),
                                      ("miss", C["down"], "Earlier call: missed")):
                sel = [c for c in cps if c["status"] == status]
                if not sel:
                    continue
                fig.add_trace(go.Scatter(
                    x=[_nv(c["t1"]) for c in sel], y=[c["pred"] for c in sel], mode="markers", name=name,
                    marker=dict(size=9, color=_rgba(col, 0.25), line=dict(color=col, width=2)),
                    error_y=dict(type="data", symmetric=False, array=[c["hi"] - c["pred"] for c in sel],
                                 arrayminus=[c["pred"] - c["lo"] for c in sel], color=_rgba(col, 0.6),
                                 thickness=1.2, width=4),
                    text=[f"Said at {c['t0']:%H:%M}: {c['t1']:%H:%M} → {c['pred']:,.0f} "
                          f"(range {c['lo']:,.0f}–{c['hi']:,.0f})"
                          + (f"<br>Actual {c['actual']:,.0f} ({c['actual'] - c['pred']:+.0f} off)" if c["actual"] else "")
                          for c in sel], hovertemplate="%{text}<extra></extra>"))

    # ---- current forecast path + range
    end_note = None
    for v in variants:
        col = C[v]
        tr = eng.trajectory(v, i, h, anchor, step=max(1, int(h // 90)), live=live)
        if len(tr) < 2 or tr[["p10", "p50", "p90"]].isna().any().any():
            continue
        if live:  # trajectory starts at the forming bar; anchor its first point at 'now'
            tr.loc[tr.index[0], "ts"] = now_t
        x = _nv(pd.DatetimeIndex(tr["ts"]))
        fig.add_trace(go.Scatter(x=x, y=tr.p90, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=tr.p10, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=_rgba(col, 0.15), name=f"{NAME[v]}: 80% range", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=tr.p50, mode="lines", line=dict(color=col, width=2.6),
                                 name=f"{NAME[v]}: prediction",
                                 customdata=np.column_stack([tr.p10, tr.p90]),
                                 hovertemplate="%{x|%H:%M} predicted %{y:,.0f}<br>range %{customdata[0]:,.0f} – "
                                               "%{customdata[1]:,.0f}<extra></extra>"))
        last = tr.iloc[-1]
        fig.add_trace(go.Scatter(x=[x[-1]], y=[last.p50], mode="markers", showlegend=False, hoverinfo="skip",
                                 marker=dict(size=11, color=col, line=dict(color=C["bg"], width=2))))
        if v == v0:
            end_note = (x[-1], last.p50, f"{pd.Timestamp(tr['ts'].iloc[-1]):%H:%M} → {last.p50:,.0f} "
                                         f"({_pts(last.p50 - anchor)})")

    # ---- alarms (stop / target lines)
    for a in snap.get("alarms") or []:
        for lvl, col, lab in ((a["target_px"], C["up"], "Target"), (a["stop_px"], C["down"], "Stop")):
            fig.add_hline(y=lvl, line=dict(color=col, width=1.4, dash="dash"),
                          annotation_text=f"{lab} {lvl:,.0f}", annotation_position="top left",
                          annotation_font=dict(color=col, size=11))
        fig.add_hline(y=a["entry"], line=dict(color=C["muted"], width=1, dash="dot"))

    # ---- headlines
    news = snap["news"]
    if "news" in variants and len(news):
        nw = news[(news["ts"] >= today.index[0]) & (news["ts"] <= t_i + pd.Timedelta(seconds=bs))
                  & (news["cruciality"] >= 4)] if len(today) else news.iloc[:0]
        if len(nw):
            fig.add_trace(go.Scatter(
                x=_nv(nw["ts"].dt.floor(f"{bs}s")), y=[today["high"].max()] * len(nw), mode="markers", name="Headline",
                marker=dict(symbol="diamond", size=7 + nw["cruciality"].values,
                            color=[C["up"] if d > .2 else C["down"] if d < -.2 else C["muted"] for d in nw["direction"]],
                            line=dict(color=C["bg"], width=2)),
                text=[f"{t}<br>direction {d:+.2f}, importance {c}/10" for t, d, c in
                      zip(nw["headline"], nw["direction"], nw["cruciality"])], hovertemplate="%{text}<extra></extra>"))

    # ---- now marker + end label
    fig.add_vline(x=_nv(now_t), line=dict(color=C["ink"], width=1, dash="dash"))
    fig.add_annotation(x=_nv(now_t), y=1, yref="paper", yanchor="bottom", showarrow=False,
                       text=f"<b>Now {now_t:%H:%M}</b> · {anchor:,.0f}", font=dict(color=C["ink"], size=12),
                       bgcolor="#2b2738", borderpad=4)
    if end_note:
        fig.add_annotation(x=end_note[0], y=end_note[1], text=f"<b>{end_note[2]}</b>", showarrow=True, arrowhead=0,
                           ax=-10, ay=-34, font=dict(color=C["ink"], size=12), bgcolor="#2b2738",
                           bordercolor=C[v0], borderwidth=1, borderpad=4)
    fig.add_vline(x=_nv(c_t), line=dict(color=C["muted"], width=1))
    # Invisible anchors fix the x extent without sending xaxis.range: an explicit range in every
    # refresh overrides the user's pan/zoom, an autoranged axis lets uirevision keep it.
    if window == "day":
        fig.add_trace(go.Scatter(x=[_nv(x0), _nv(x1)], y=[anchor, anchor], mode="markers", opacity=0,
                                 hoverinfo="skip", showlegend=False, name="_anchor"))
    fig.update_layout(
        margin=dict(l=4, r=4, t=34, b=8), paper_bgcolor=C["bg"], plot_bgcolor=C["bg"], autosize=True,
        font=dict(family="Manrope, system-ui", color=C["ink"], size=11), hovermode="closest", dragmode="pan",
        uirevision=f"{mode}-{window}-{day:%Y%m%d}",
        hoverlabel=dict(bgcolor="#2b2738", bordercolor="#36314a"),
        xaxis=dict(type="date", tickformat="%H:%M", gridcolor=C["grid"], autorange=True,
                   rangeslider=dict(visible=False), fixedrange=False),
        yaxis=dict(gridcolor=C["grid"], side="right", tickformat=",.0f", fixedrange=False),
        legend=dict(orientation="h", y=-0.07, x=0, yanchor="top", bgcolor="rgba(0,0,0,0)", font=dict(size=10.5)))
    if window != "day":
        fig.update_xaxes(range=[_nv(x0), _nv(x1)], autorange=False)
    return fig


def checkpoints(eng, bars, i, h, v, first):
    """Predictions made earlier today at regular times, for horizon h, and how they turned out."""
    step = 15 if h <= 30 else 30
    hi_ = int(round(h))
    idx = np.arange(first, i + 1)
    t = bars.index[idx]
    idx = idx[(t.minute % step == 0)]
    if not len(idx):
        return []
    close_t = eng.clock.session(bars.index[i])[1]
    q = eng.interp(v, hi_, rows=idx)
    c = bars["close"].values
    out = []
    for k, j in enumerate(idx):
        t0 = bars.index[j]
        t1 = t0 + pd.Timedelta(seconds=eng.bs * hi_)
        if t1 >= close_t or np.isnan(q[k]).any():
            continue
        p0 = float(c[j])
        lo, pred, hi = (p0 * np.exp(q[k] / 1e4)).tolist()
        actual = float(c[j + hi_]) if j + hi_ <= i else None
        status = "pending" if actual is None else ("hit" if lo <= actual <= hi else "miss")
        out.append({"t0": t0, "t1": t1, "p0": p0, "pred": pred, "lo": lo, "hi": hi, "actual": actual,
                    "status": status})
    return out


# ============================================================================================ panels
def summary(eng, snap, i, h, v, mode):
    """The selected horizon in plain words: expected, down case, up case, chance higher."""
    bars = snap["bars"]
    live = mode in ("live", "replay")
    anchor = float(snap["last_price"]) if live and np.isfinite(snap["last_price"]) else float(bars["close"].iloc[i])
    t_now = snap["live_ts"] if live and snap.get("live_ts") is not None else bars.index[i]
    h_eff = min(h, eng.minutes_left(t_now))
    q = _q_at(eng, v, h_eff, i, live)
    if q is None or np.isnan(q).any():
        return html.Div("Not enough data yet.", className="note")
    lo, mid, hi = anchor * np.exp(np.asarray(q) / 1e4)
    pu = eng.p_up(q)
    lean = ("leans up" if pu >= 0.56 else "leans down" if pu <= 0.44 else "no clear direction")
    t1 = t_now + pd.Timedelta(seconds=eng.bs * h_eff)
    return html.Div([
        html.Div([html.Span(f"In {_hlabel(h_eff)}"), html.Span(f" at {t1:%H:%M}", className="muted"),
                  html.Span(" (market close)", className="muted") if h_eff < h else None], className="when"),
        html.Div([html.Div([html.Div("Down case", className="k"), html.Div(f"{lo:,.0f}", className="v down"),
                            html.Div(_pts(lo - anchor) + " pts", className="d down")], className="case"),
                  html.Div([html.Div("Middle", className="k"), html.Div(f"{mid:,.0f}", className="v"),
                            html.Div(_pts(mid - anchor) + " pts", className="d")], className="case mid"),
                  html.Div([html.Div("Up case", className="k"), html.Div(f"{hi:,.0f}", className="v up"),
                            html.Div(_pts(hi - anchor) + " pts", className="d up")], className="case")],
                 className="cases"),
        html.Div([html.Div(html.Span(style={"width": f"{(1 - pu) * 100:.0f}%"}), className="oddsbar"),
                  html.Div([html.Span(f"Down {100 * (1 - pu):.0f}%", className="down"),
                            html.Span(lean, className="muted"),
                            html.Span(f"Up {100 * pu:.0f}%", className="up")], className="oddslab")]),
        html.Div("Down and up cases: 1 in 10 times it ends below the down case, 1 in 10 above the up case.",
                 className="note"),
    ], className="summary")


def ladder(eng, snap, i, v, mode, next_fc=None):
    bars = snap["bars"]
    live = mode in ("live", "replay")
    anchor = float(snap["last_price"]) if live and np.isfinite(snap["last_price"]) else float(bars["close"].iloc[i])
    t_now = snap["live_ts"] if live and snap.get("live_ts") is not None else bars.index[i]
    left = eng.minutes_left(t_now)
    rows = []
    hs = [(f"{_hlabel(h)}", h) for h in LADDER_H if h < left - 0.5] + [("At close", left)]
    for name, h in hs:
        q = _q_at(eng, v, h, i, live)
        if q is None or np.isnan(q).any():
            continue
        lo, mid, hi = anchor * np.exp(np.asarray(q) / 1e4)
        pu = eng.p_up(q)
        t1 = t_now + pd.Timedelta(seconds=eng.bs * h)
        rows.append(html.Tr([html.Td(name), html.Td(f"{t1:%H:%M}", className="muted"),
                             html.Td(f"{mid:,.0f}"),
                             html.Td(f"{lo:,.0f} – {hi:,.0f}", className="muted"),
                             html.Td(f"{100 * pu:.0f}%", className="up" if pu >= .56 else "down" if pu <= .44 else "")]))
    if next_fc:
        rows.append(html.Tr([html.Td("Next open"), html.Td(f"{next_fc['next_open']:%a %H:%M}", className="muted"),
                             html.Td(f"{next_fc['open_mid']:,.0f}"),
                             html.Td(f"{next_fc['open_lo']:,.0f} – {next_fc['open_hi']:,.0f}", className="muted"),
                             html.Td(f"{100 * next_fc['p_gap_up']:.0f}%")], className="next"))
    return html.Div([html.Table([html.Thead(html.Tr([html.Th("Ahead"), html.Th("Time"), html.Th("Middle"),
                                                     html.Th("80% range"), html.Th("Up chance")]))]
                                + [html.Tbody(rows)], className="t ladder")], className="scrollx")


def plan_card(eng, snap, h, i, mode, with_buttons):
    live = mode in ("live", "replay")
    anchor = float(snap["last_price"]) if live and np.isfinite(snap["last_price"]) else float(snap["bars"]["close"].iloc[i])
    pl = eng.plan(h, i=i, live=live, anchor_px=anchor)
    if pl is None:
        return html.Div("Learning how far the market typically swings… (a few seconds after start)", className="note")

    def side(name, key, sign):
        p = pl[key]
        seg = lambda w, cls: html.Span(style={"width": f"{100 * w:.0f}%"}, className=cls)
        return html.Div([
            html.Div(name, className="side-name"),
            html.Div([html.Span("Target", className="k"), html.B(f"{p['target_px']:,.0f}"),
                      html.Span(f" {sign}{p['target_pts']:.0f}", className="up")], className="lvl"),
            html.Div([html.Span("Stop", className="k"), html.B(f"{p['stop_px']:,.0f}"),
                      html.Span(f" {'-' if sign == '+' else '+'}{p['stop_pts']:.0f}", className="down")], className="lvl"),
            html.Div([seg(p["target"], "g"), seg(p["neither"], "n"), seg(p["stop"], "r")], className="tribar"),
            html.Div(f"In similar past moments: target first {100 * p['target']:.0f}%, stop first "
                     f"{100 * p['stop']:.0f}%, neither within {_hlabel(pl['h'])} {100 * p['neither']:.0f}%.",
                     className="note"),
            html.Button(f"Set alarm ({name.split()[0].lower()})", id=f"alarm-{key}", className="btn",
                        style={} if with_buttons else {"display": "none"}),
        ], className="plan-side")

    alarms = snap.get("alarms") or []
    act = []
    for a in alarms:
        act.append(html.Div(f"Watching {a['side']} from {a['entry']:,.0f}: target {a['target_px']:,.0f}, "
                            f"stop {a['stop_px']:,.0f}, until {a['until']:%H:%M}", className="note"))
    ev = (snap.get("alarm_events") or [])[-3:][::-1]
    for e in ev:
        act.append(html.Div(f"{e['ts']:%H:%M} {e['kind'].upper()}: {e['side']} from {e['entry']:,.0f} at "
                            f"{e['price']:,.0f} ({e['pts']:+.0f} pts)",
                            className="note " + ("up" if e["kind"] == "target" else "down" if e["kind"] == "stop" else "")))
    return html.Div([
        html.Div(f"Next {_hlabel(pl['h'])}: typical best rise +{pl['rise50']:.0f}, typical worst dip "
                 f"−{pl['dip50']:.0f}. 4 times in 5 the dip stays within −{pl['dip80']:.0f}.", className="swing"),
        html.Div([side("Buy (expect up)", "long", "+"), side("Sell (expect down)", "short", "-")], className="plan"),
        html.Div(act + ([html.Button("Cancel alarms", id="alarm-cancel", className="linkbtn")]
                        if (alarms and with_buttons) else
                        [html.Button("Cancel alarms", id="alarm-cancel", className="linkbtn",
                                     style={"display": "none"})] if with_buttons else []), className="alarms"),
        html.Div("Levels come from how far NIFTY really moved in similar volatility over 11 years. They are not "
                 "advice; the direction itself is close to a coin flip.", className="note"),
    ])


def pred_log(eng, snap, i, h, v):
    bars = snap["bars"]
    if not len(bars):
        return html.Div()
    first = int(np.searchsorted(bars.index.values, bars.index[i].normalize().to_datetime64()))
    cps = checkpoints(eng, bars, i, h, v, first)[::-1][:12]
    if not cps:
        return html.Div(f"The first {_hlabel(h)} prediction is checked {_hlabel(h)} after the open.", className="note")
    rows = []
    for c in cps:
        verdict = {"pending": "not due yet", "hit": "in range", "miss": "outside range"}[c["status"]]
        cls = {"pending": "muted", "hit": "up", "miss": "down"}[c["status"]]
        rows.append(html.Tr([html.Td(f"{c['t0']:%H:%M}"), html.Td(f"{c['t1']:%H:%M}"), html.Td(f"{c['pred']:,.0f}"),
                             html.Td(f"{c['lo']:,.0f}–{c['hi']:,.0f}", className="muted"),
                             html.Td("—" if c["actual"] is None else f"{c['actual']:,.0f}"),
                             html.Td("—" if c["actual"] is None else f"{c['actual'] - c['pred']:+.0f}"),
                             html.Td(verdict, className=cls)]))
    done = [c for c in cps if c["actual"] is not None]
    hit = sum(c["status"] == "hit" for c in done)
    return html.Div([
        html.Div(f"{hit} of {len(done)} checked calls landed inside their range." if done else "", className="note"),
        html.Div(html.Table([html.Thead(html.Tr([html.Th(x) for x in ("Said at", "For", "Predicted", "80% range",
                                                                       "Actual", "Off by", "Result")]))]
                            + [html.Tbody(rows)], className="t"), className="scrollx")])


def next_card(fc):
    if not fc:
        return html.Div("Learning from the daily history… (a few seconds after start)", className="note")
    us = ("Includes the US market move since India closed "
          f"({fc['us_move_bps'] / 100:+.2f}%)." if fc.get("used_us") else
          "US markets not included yet (needs Yahoo access; usually works on the server).")
    pu = fc["p_gap_up"]
    return html.Div([
        html.Div([html.Span(f"{fc['next_open']:%A %d %b}, 09:15"), html.Span(f" · based on {fc['based_on']}",
                                                                          className="muted")], className="when"),
        html.Div([html.Div([html.Div("Opens around", className="k"), html.Div(f"{fc['open_mid']:,.0f}", className="v"),
                            html.Div(f"{_pts(fc['gap_mid_pts'])} pts", className="d")], className="case mid"),
                  html.Div([html.Div("Likely open range", className="k"),
                            html.Div(f"{fc['open_lo']:,.0f} – {fc['open_hi']:,.0f}", className="v sm")], className="case"),
                  html.Div([html.Div("Likely close range", className="k"),
                            html.Div(f"{fc['close_lo']:,.0f} – {fc['close_hi']:,.0f}", className="v sm")],
                           className="case")], className="cases"),
        html.Div([html.Div(html.Span(style={"width": f"{(1 - pu) * 100:.0f}%"}), className="oddsbar"),
                  html.Div([html.Span(f"Opens lower {100 * (1 - pu):.0f}%", className="down"),
                            html.Span(f"Opens higher {100 * pu:.0f}%", className="up")], className="oddslab")]),
        html.Div(us + " Opening direction from India's own data is close to a coin flip; the range is the "
                 "reliable part.", className="note"),
    ], className="summary")


def alarm_view(snap, within_min=20):
    ev = snap.get("alarm_events") or []
    now = snap.get("last_ts")
    if not ev or now is None or (now - ev[-1]["ts"]).total_seconds() > within_min * 60:
        return None, None
    e = ev[-1]
    word = {"target": "TARGET HIT", "stop": "STOP HIT", "expired": "Plan expired"}[e["kind"]]
    msg = (f"{word} at {e['ts']:%H:%M}: {e['side']} from {e['entry']:,.0f}, now {e['price']:,.0f} "
           f"({e['pts']:+.0f} pts)")
    cls = "up" if e["kind"] == "target" else "down" if e["kind"] == "stop" else "either"
    return (html.Div(html.B(msg), className=f"alert {cls}"),
            {"id": f"alarm-{e['id']}-{e['kind']}", "title": "NiftySim: " + word, "body": msg})


def window_chips(pid, value="day"):
    return dcc.RadioItems(id=pid, value=value, className="chips small", inline=True,
                          options=[{"label": a, "value": b} for a, b in WINDOWS])
