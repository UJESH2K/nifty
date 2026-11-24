"""NiftySim dashboard (v3).

Live         : Go live waits for 09:15 by itself, then shows the real market (today only, with a clear
               'now'), the predicted path with its down/up cases, a prediction ladder (1 min to 3 h, the
               close and tomorrow's open), stop/target levels with alarms, and a log of what it said vs
               what happened.
Blind replay : a model that never saw a chosen day predicts it live, minute by minute.
Lab          : move through history, scorecards, retrain.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import parts  # noqa: E402
from app.parts import (C, HORIZONS, NAME, SOURCES, _hlabel, _hval, _ring_svg, alert_view, check_box,  # noqa: E402
                       improvement_chart, news_list, score_table, tick_chart)
from app.views import (alarm_view, day_chart, ladder, next_card, plan_card, pred_log, summary,  # noqa: E402
                       window_chips)
from niftysim.blind import BlindSession, available_days  # noqa: E402
from niftysim.engine import VARIANTS, Engine  # noqa: E402
from niftysim.market_clock import fmt_countdown  # noqa: E402
from niftysim.models import REGISTRY  # noqa: E402

LIVE: Engine | None = None
AUTO = None
BLIND: BlindSession | None = None
CFG: dict = {}
RETRAIN = {"state": ""}
GRAPH_CFG = dict(displaylogo=False, scrollZoom=True, responsive=True,
                 modeBarButtonsToRemove=["select2d", "lasso2d", "autoScale2d", "toggleSpikelines",
                                         "hoverClosestCartesian", "hoverCompareCartesian"])


def horizon_controls(p, default="60"):
    return html.Div([
        html.Div([html.Span("Predict ahead", className="lab"),
                  dcc.RadioItems(id=f"{p}-h", value=default, className="chips", inline=True,
                                 options=[{"label": a, "value": b} for a, b in HORIZONS])], className="field"),
        html.Div(dcc.Slider(id=f"{p}-hc", min=1, max=180, step=1, value=45,
                            marks={1: "1m", 30: "30m", 60: "1h", 120: "2h", 180: "3h"},
                            tooltip={"placement": "bottom", "always_visible": True}),
                 id=f"{p}-hc-wrap", className="field", style={"display": "none"}),
        html.Div([
            html.Div([html.Span("Model", className="lab"),
                      dcc.RadioItems(id=f"{p}-v", value="news", className="seg", inline=True, options=[
                          {"label": "With news", "value": "news"}, {"label": "Price only", "value": "price"},
                          {"label": "Compare", "value": "both"}])]),
            html.Div([html.Span("Chart window", className="lab"), window_chips(f"{p}-win")]),
        ], className="row2"),
    ], className="controls-card")


def main_block(p, live_extras=False):
    return html.Div([
        html.Div(id=f"{p}-head", className="pricehead"),
        html.Div(id=f"{p}-alert", className="mb-alert"),
        html.Div(horizon_controls(p), className="mb-controls"),
        dcc.Store(id=f"{p}-fig"),
        html.Div(dcc.Graph(id=f"{p}-chart", config=GRAPH_CFG, className="chartbox",
                           style={"height": "100%", "width": "100%"}), className="panel chart chartwrap"),
        html.Div([
            html.Div([html.H2("Selected horizon"), html.Div(id=f"{p}-sum")], className="panel"),
            html.Div([html.H2("Prediction ladder"), html.Div(id=f"{p}-ladder")], className="panel"),
        ], className="grid2"),
        html.Div([html.H2("Stop-loss and target"), html.Div(id=f"{p}-plan")], className="panel mt"),
        html.Div([
            html.Div([html.H2("Prediction log: what it said vs what happened"), html.Div(id=f"{p}-log")],
                     className="panel"),
            html.Div(id=f"{p}-check", className="panel"),
        ], className="grid2"),
        html.Div([
            html.Div([html.H2("Next session"), html.Div(id=f"{p}-next")], className="panel"),
            html.Div([html.H2("Headlines"), html.Div(id=f"{p}-news", className="newsbox")], className="panel"),
        ], className="grid2"),
        html.Div([
            html.Div([html.H2("Every tick, last 15 minutes"), dcc.Graph(id="live-ticks", config={"displayModeBar": False},
                                                                      className="tickbox")], className="panel"),
            html.Div([html.H2("Session"), html.Div(id="live-session"),
                      html.Button("Turn on pop-up alerts", id="enable-alerts", className="btn primary mt"),
                      html.Button("Stop live", id="stop-live", className="btn mt")], className="panel"),
        ], className="grid2") if live_extras else None,
    ])


def render(eng, snap, p, i, mode, chip, custom, var, win, with_buttons):
    """All per-tab panels at once."""
    bars = snap["bars"]
    h = _hval(chip, custom)
    vs = list(VARIANTS)[::-1] if var == "both" else [var]
    v0 = vs[0]
    fig = day_chart(eng, snap, i, h, vs, mode, win or "day")
    sm = html.Div([summary(eng, snap, i, h, v, mode) for v in vs])
    nfc = eng.next_session() if mode == "live" else None
    lad = ladder(eng, snap, i, v0, mode, nfc)
    pl = plan_card(eng, snap, h, i, mode, with_buttons)
    lg = pred_log(eng, snap, i, h, v0)
    since = bars.index[i].normalize()
    chk = check_box(eng, v0, h, since)
    nx = next_card(nfc) if mode == "live" else html.Div("Shown in Live mode.", className="note")
    nw = news_list(snap, bars.index[i] + pd.Timedelta(minutes=1))
    return fig, sm, lad, pl, lg, chk, nx, nw


def price_head(snap, i, label="NIFTY 50", extra=None):
    bars = snap["bars"]
    lp = float(snap["last_price"]) if np.isfinite(snap["last_price"]) else float(bars["close"].iloc[i])
    d = bars.index[i].normalize()
    prev = bars[bars.index < d]["close"]
    ref = float(prev.iloc[-1]) if len(prev) else float(bars["open"].iloc[0])
    chg = lp - ref
    asof = snap.get("last_ts") or bars.index[i]
    kids = [html.Span(label, className="name"), html.Span(f"{lp:,.2f}", className="price"),
            html.Span(f"{chg:+,.2f} ({chg / ref * 100:+.2f}%)", className="chg " + ("up" if chg >= 0 else "down")),
            html.Span(f"as of {pd.Timestamp(asof):%a %d %b, %H:%M:%S}", className="meta")]
    if extra:
        kids.append(html.Span(extra, className="meta"))
    return kids


# ============================================================================================ app
def build_app(cfg, engine: Engine, autopilot=None):
    global LIVE, BLIND, CFG, AUTO
    LIVE, CFG, AUTO = engine, cfg, autopilot
    parts.LIVE, parts.CFG = engine, cfg
    BLIND = BlindSession(cfg)
    app = Dash(__name__, title="NiftySim", assets_folder=str(Path(__file__).parent / "assets"),
               external_stylesheets=["https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Sora:wght@600;700&display=swap"],
               meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1, maximum-scale=5"},
                          {"name": "theme-color", "content": "#17151f"}],
               suppress_callback_exceptions=True, **({"use_async": False} if "use_async" in Dash.__init__.__code__.co_varnames else {}))
    days = available_days(cfg)
    day_opts = [{"label": d.strftime("%a %d %b %Y"), "value": d.strftime("%Y-%m-%d")} for d in days[::-1]]
    models = list(REGISTRY)
    refresh = int(os.environ.get("NIFTYSIM_UI_REFRESH_MS", 1000))

    live = html.Div([
        html.Div([
            html.Div([html.Div(id="orb-ring"),
                      html.Button(id="go", className="orb", children=[html.Span("Go live", className="big")])],
                     className="orb-wrap"),
            html.H1(id="hero-title"), html.P(id="hero-text"),
            dcc.RadioItems(id="src", value=cfg["feed"]["source"] if cfg["feed"]["source"] in dict((b, a) for a, b in SOURCES) else "auto",
                           className="sources", inline=True, options=[{"label": a, "value": b} for a, b in SOURCES]),
            html.Div(id="src-hint", className="hint"),
            html.Div([html.Button("Cancel", id="cancel", className="linkbtn"),
                      html.Button("Meanwhile, replay a past session", id="to-blind", className="linkbtn")]),
            html.Div([html.H2("Next session"), html.Div(id="hero-next")], className="panel hero-next"),
        ], id="live-hero", className="hero"),
        html.Div(main_block("l", live_extras=True), id="live-main", style={"display": "none"}),
    ], id="sec-live")

    blind = html.Div([
        html.Div([
            html.H2("Blind replay"),
            html.P("Pick a past session. A model is trained only on the days before it, so it has never seen that day. "
                   "The day then plays back minute by minute with its headlines, and every prediction is checked "
                   "against what really happened.", className="note"),
            html.Div([
                html.Div([html.Span("Session", className="lab"),
                          dcc.Dropdown(id="b-day", options=day_opts, value=day_opts[0]["value"] if day_opts else None,
                                       clearable=False, style={"width": "210px"})], className="field"),
                html.Div([html.Span("Speed", className="lab"),
                          dcc.RadioItems(id="b-speed", value="60", className="chips", inline=True, options=[
                              {"label": "Real time", "value": "1"}, {"label": "10×", "value": "10"},
                              {"label": "60×", "value": "60"}, {"label": "300×", "value": "300"},
                              {"label": "Max", "value": "5000"}])], className="field"),
                html.Div(html.Button("Start replay", id="b-start", className="btn primary"), className="field"),
            ], className="row-controls"),
            html.Div(id="b-status", className="note"), html.Div(id="b-progress"),
        ], className="panel mb"),
        html.Div([main_block("b"),
                  html.Div([html.H2("Every horizon for this session"), html.Div(id="b-score")], className="panel mt")],
                 id="b-main", style={"display": "none"}),
    ], id="sec-blind", style={"display": "none"})

    lab = html.Div([
        html.Div([
            html.Div([html.Span("Move through history. The model only uses data up to the point you choose; dimmed "
                                "candles show what happened next.", className="lab"),
                      dcc.Slider(id="lab-tt", min=0, max=max(len(engine.bars) - 1, 1), step=1,
                                 value=max(len(engine.bars) - 1, 0), marks=None)], style={"flex": 1, "minWidth": "240px"}),
            html.Button("◀", id="lab-back", className="btn"), html.Button("▶", id="lab-fwd", className="btn"),
            html.Div(dcc.Dropdown(id="lab-model", value=engine.model_name, clearable=False, style={"width": "150px"},
                                  options=[{"label": m, "value": m} for m in models]),
                     style={"display": "none"} if autopilot else {}),
            html.Button("Retrain now", id="lab-retrain", className="btn", style={"display": "none"} if autopilot else {}),
            html.Span(id="lab-rt", className="note"),
        ], className="panel row-controls mb"),
        html.Div(id="lab-banner"),
        main_block("x"),
        html.Div([html.H2("Scorecard over the last 5 sessions in memory"), html.Div(id="lab-score")], className="panel mt"),
        html.Div([html.H2("Is it getting better? One point per trading day"),
                  dcc.Graph(id="lab-improve", config={"displayModeBar": False}),
                  html.P("Below the dashed line = it missed by less than a plain 'no change' guess that day.",
                         className="note")], className="panel mt"),
    ], id="sec-lab", style={"display": "none"})

    app.layout = html.Div([
        html.Div([
            html.Div([html.I(), "NiftySim"], className="logo"),
            dcc.RadioItems(id="tab", value="live", className="nav", inline=True, options=[
                {"label": "Live", "value": "live"}, {"label": "Blind replay", "value": "blind"}, {"label": "Lab", "value": "lab"}]),
            html.Div(id="status", className="status"),
        ], className="bar"),
        live, blind, lab,
        dcc.Interval(id="tick", interval=refresh),
        dcc.Store(id="alert-store"),
        html.Div(id="sink", style={"display": "none"}), html.Div(id="sink2", style={"display": "none"}),
        html.Div(id="alert-sink", style={"display": "none"}), html.Div(id="perm-sink", style={"display": "none"}),
    ], className="app")

    # ---------------------------------------------------------------- notifications (browser side)
    # The server sends each fresh figure to a Store; this keeps whatever pan/zoom the person
    # has on screen before handing it to the graph. (Dash writes GUI edits back into the
    # figure prop, which defeats Plotly's uirevision, so the view is carried over by hand.)
    for p in ("l", "b", "x"):
        app.clientside_callback(
            """function(fig, id) {
                if (!fig) { return window.dash_clientside.no_update; }
                var el = document.querySelector('#' + id + ' .js-plotly-plot');
                var num = function (v) { return typeof v === 'string' ? Date.parse(v.replace(' ', 'T')) : v; };
                var key = function (r) { return r ? r.map(function (v) { return Math.round(num(v) / 1000); }).join(',') : 'auto'; };
                var srv = {};
                ['xaxis', 'yaxis'].forEach(function (ax) {
                    var A = (fig.layout && fig.layout[ax]) || {};
                    srv[ax] = A.autorange === false && A.range ? key(A.range) : 'auto';
                });
                if (el && el.layout && fig.layout && el.layout.uirevision === fig.layout.uirevision && el.__srv) {
                    ['xaxis', 'yaxis'].forEach(function (ax) {
                        var L = el.layout[ax];
                        // Keep the on-screen view only if the person moved it away from what we last sent.
                        if (L && L.autorange === false && L.range && key(L.range) !== el.__srv[ax]) {
                            fig.layout[ax] = Object.assign({}, fig.layout[ax] || {},
                                {range: L.range.slice(), autorange: false});
                            srv[ax] = el.__srv[ax];
                        }
                    });
                }
                if (el) { el.__srv = srv; }
                return fig;
            }""",
            Output(f"{p}-chart", "figure"), Input(f"{p}-fig", "data"), State(f"{p}-chart", "id"))

    app.clientside_callback(
        """
        function(a) {
          if (!a || !a.id) { return window.dash_clientside.no_update; }
          if (window._nsLastAlert === a.id) { return window.dash_clientside.no_update; }
          window._nsLastAlert = a.id;
          try { if ("Notification" in window && Notification.permission === "granted") {
                  new Notification(a.title, {body: a.body}); } } catch (e) {}
          try { if (navigator.vibrate) { navigator.vibrate([200, 100, 200]); } } catch (e) {}
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
        function(n) { if (!n) { return window.dash_clientside.no_update; }
          try { Notification.requestPermission(); } catch (e) {} return ""; }
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

    # ---------------------------------------------------------------- live controls
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

    @app.callback(Output("sink2", "children"), Input("alarm-long", "n_clicks"), Input("alarm-short", "n_clicks"),
                  Input("alarm-cancel", "n_clicks"), State("l-h", "value"), State("l-hc", "value"),
                  prevent_initial_call=True)
    def _alarm(_l, _s, _c, chip, custom):
        if ctx.triggered_id == "alarm-cancel":
            LIVE.cancel_alarms()
            return ""
        if not ctx.triggered or not ctx.triggered[0].get("value"):
            return no_update
        pl = LIVE.plan(_hval(chip, custom), live=True)
        if pl:
            LIVE.set_alarm("long" if ctx.triggered_id == "alarm-long" else "short", pl)
        return ""

    @app.callback(Output("src-hint", "children"), Input("src", "value"))
    def _hint(src):
        return {"auto": "No account needed. Reads NSE's own website; switches to Yahoo by itself if NSE blocks it.",
                "yahoo": "No account needed. Prices can lag the exchange a little.",
                "angel": "Needs an Angel One demat account and its keys in the .env file.",
                "upstox": "Needs an Upstox demat account and today's access token in the .env file.",
                "synthetic": "A simulated market that starts right away, for trying the screen out."}[src]

    LIVE_OUT = [Output("live-hero", "style"), Output("live-main", "style"), Output("orb-ring", "children"),
                Output("go", "className"), Output("go", "children"), Output("hero-title", "children"),
                Output("hero-text", "children"), Output("cancel", "style"), Output("src", "style"),
                Output("hero-next", "children"), Output("status", "children"),
                Output("l-head", "children"), Output("l-alert", "children"), Output("l-fig", "data"),
                Output("l-sum", "children"), Output("l-ladder", "children"), Output("l-plan", "children"),
                Output("l-log", "children"), Output("l-check", "children"), Output("l-next", "children"),
                Output("l-news", "children"), Output("live-ticks", "figure"), Output("live-session", "children"),
                Output("alert-store", "data")]
    N_MAIN = 13

    @app.callback(*LIVE_OUT, Input("tick", "n_intervals"), Input("tab", "value"), Input("l-h", "value"),
                  Input("l-hc", "value"), Input("l-v", "value"), Input("l-win", "value"))
    def _live(_n, tab, chip, custom, var, win):
        snap = LIVE.snapshot()
        st = LIVE.clock.status()
        running = LIVE.status.startswith("live") or LIVE.status == "session over"
        if st["open"]:
            stat = [html.Span(className="dot on"), html.Span(f"Market open · closes in {fmt_countdown(st['secs_to_close'])}")]
        else:
            stat = [html.Span(className="dot"), html.Span(f"{st['label']} · opens {st['next_open']:%a %d %b, %H:%M}")]
        if tab != "live":
            return [no_update] * 10 + [stat] + [no_update] * N_MAIN
        hide, show = {"display": "none"}, {}
        if not running:
            total = 17.75 * 3600
            if not st["open"]:
                total = max(3600.0, (st["next_open"] - LIVE.clock.session(LIVE.clock.last_session_day())[1]).total_seconds())
            frac = 1.0 if st["open"] else max(0.0, min(1.0, 1 - st.get("secs_to_open", 0) / total))
            ring = _ring_svg(frac, 2 * np.pi * 140)
            why = st["label"] if not st["open"] else ""
            nxt = st.get("next_open")
            nfc = next_card(LIVE.next_session())
            if AUTO is not None:
                orb_cls = "orb armed"
                orb_kids = ([html.Span(fmt_countdown(st.get("secs_to_open", 0)), className="big"), html.Small("until the open")]
                            if not st["open"] else [html.Span("Starting…", className="big pulse"), html.Small(snap.get("message") or "")])
                title = (f"{why}. Starts by itself {nxt:%A} at {nxt:%H:%M}" if nxt is not None else "Starting now")
                text = ("Nothing to press. Every trading day it predicts the market live, checks itself, and "
                        f"retrains overnight on what it saw. Status: {AUTO.state}.")
                return [show, hide, ring, orb_cls, orb_kids, title, text, hide, hide, nfc, stat] + [no_update] * N_MAIN
            if snap["armed"] and LIVE.status == "waiting":
                orb_cls = "orb armed"
                orb_kids = [html.Span(fmt_countdown(st.get("secs_to_open", 0)), className="big"), html.Small("until the open")]
                title = f"{why}. Analysis starts {nxt:%A} at {nxt:%H:%M}"
                text = "Keep this tab open. When trading begins it loads today's prices and starts predicting on its own."
                cancel_style, src_style = {}, hide
            elif snap["armed"]:
                orb_cls, orb_kids = "orb armed", [html.Span("Starting…", className="big pulse"),
                                                  html.Small(snap.get("message") or "connecting")]
                title, text, cancel_style, src_style = "Connecting to the market", snap.get("message") or "", {}, hide
            else:
                orb_cls = "orb"
                orb_kids = [html.Span("Go live", className="big"),
                            html.Small("starts now" if st["open"] else f"starts {nxt:%a %H:%M}")]
                if st["open"]:
                    title, text = "The market is open", "Press Go live to start predicting from the next price update."
                else:
                    title = f"{why}. The market opens {nxt:%A} at {nxt:%H:%M}"
                    text = "Press Go live now and leave this tab open. Analysis starts by itself when trading begins."
                cancel_style, src_style = hide, {}
            return [show, hide, ring, orb_cls, orb_kids, title, text, cancel_style, src_style, nfc, stat] + [no_update] * N_MAIN

        bars = snap["bars"]
        i = len(bars) - 1
        src_name = snap.get("using") or dict((b, a) for a, b in SOURCES).get(snap["source"], snap["source"])
        lag = snap.get("lag") or 0
        extra = f"{src_name}" + (f", about {lag:.0f}s behind" if lag > 20 else "") + \
                (f". Feed error: {snap['feed_error']}" if snap.get("feed_error") else "")
        if LIVE.status == "session over":
            extra += ". Session over: everything below is today's record."
        head = price_head(snap, i, extra=extra)
        fig, sm, lad, pl, lg, chk, nx, nw = render(LIVE, snap, "l", i, "live", chip, custom, var, win, True)
        banner, note = alert_view(snap)
        ab, anote = alarm_view(snap)
        today = bars[bars.index.normalize() == bars.index[i].normalize()]
        sess = html.Div([
            html.Div(f"Autopilot: {AUTO.state}" if AUTO is not None else "", className="note"),
            html.Div(f"Model: {snap['model']}. Bars today: {len(today)}. "
                     f"{snap.get('refit_info') or ''}", className="note"),
        ])
        return ([hide, show] + [no_update] * 8 + [stat, head, [x for x in (ab, banner) if x is not None], fig, sm, lad, pl,
                                                  lg, chk, nx, nw, tick_chart(snap["ticks"]), sess,
                                                  anote or note or no_update])

    # ---------------------------------------------------------------- blind
    @app.callback(Output("b-status", "children", allow_duplicate=True), Input("b-start", "n_clicks"),
                  State("b-day", "value"), State("b-speed", "value"), prevent_initial_call=True)
    def _bstart(_, day, speed):
        if not day:
            return "No full sessions found in the data folder yet."
        BLIND.start(day, LIVE.model_name, float(speed))
        return BLIND.msg

    @app.callback(Output("sink", "title"), Input("b-speed", "value"), prevent_initial_call=True)
    def _bspeed(s):
        BLIND.set_speed(float(s))
        return ""

    @app.callback(Output("b-status", "children"), Output("b-progress", "children"), Output("b-main", "style"),
                  Output("b-head", "children"), Output("b-alert", "children"), Output("b-fig", "data"),
                  Output("b-sum", "children"), Output("b-ladder", "children"), Output("b-plan", "children"),
                  Output("b-log", "children"), Output("b-check", "children"), Output("b-next", "children"),
                  Output("b-news", "children"), Output("b-score", "children"),
                  Input("tick", "n_intervals"), Input("tab", "value"), Input("b-h", "value"), Input("b-hc", "value"),
                  Input("b-v", "value"), Input("b-win", "value"))
    def _blind(_n, tab, chip, custom, var, win):
        if tab != "blind":
            return [no_update] * 14
        e = BLIND.engine
        prog = BLIND.feed.progress if BLIND.feed else 0
        pbar = html.Div(html.Span(style={"width": f"{prog * 100:.1f}%"}), className="progress") if BLIND.state != "idle" else ""
        msg = BLIND.msg
        if BLIND.state == "training":
            msg += " (about a minute on the free server)"
        if e is None or BLIND.state in ("idle", "training", "error"):
            return [msg, pbar, {"display": "none"}] + [no_update] * 11
        snap = e.snapshot()
        bars = snap["bars"]
        if not len(bars) or bars.index[-1] < BLIND.day:
            return [msg + " Waiting for the first minute…", pbar, {"display": "none"}] + [no_update] * 11
        i = len(bars) - 1
        head = price_head(snap, i, label=f"NIFTY 50 · replay of {BLIND.day:%a %d %b}")
        fig, sm, lad, pl, lg, chk, nx, nw = render(e, snap, "b", i, "replay", chip, custom, var, win, False)
        return [msg, pbar, {}, head, alert_view(snap)[0], fig, sm, lad, pl, lg, chk, nx, nw, score_table(e, BLIND.day)]

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
        except FileNotFoundError as ex:
            return str(ex)

    @app.callback(Output("lab-improve", "figure"), Input("tab", "value"))
    def _improve(tab):
        return improvement_chart(CFG) if tab == "lab" else no_update

    @app.callback(Output("lab-banner", "children"), Output("x-head", "children"), Output("x-fig", "data"),
                  Output("x-sum", "children"), Output("x-ladder", "children"), Output("x-plan", "children"),
                  Output("x-log", "children"), Output("x-check", "children"), Output("x-next", "children"),
                  Output("x-news", "children"), Output("lab-score", "children"),
                  Input("tab", "value"), Input("lab-tt", "value"), Input("x-h", "value"), Input("x-hc", "value"),
                  Input("x-v", "value"), Input("x-win", "value"), Input("lab-rt", "children"))
    def _lab(tab, tt, chip, custom, var, win, _rt):
        if tab != "lab":
            return [no_update] * 11
        snap = LIVE.snapshot()
        bars = snap["bars"]
        if not len(bars):
            return ["No data yet."] + [no_update] * 10
        i = int(min(tt if tt is not None else len(bars) - 1, len(bars) - 1))
        te = snap["train_end"]
        t = bars.index[i]
        ban = (html.Div("The model was trained on this period, so it looks better here than it really is. "
                        "Use Blind replay for an honest test.", className="banner ins")
               if te is not None and t <= pd.Timestamp(te) else
               html.Div("The model has never seen this period. This is an honest test.", className="banner oos"))
        snap["forming"] = None
        snap["last_price"] = float(bars["close"].iloc[i])
        snap["last_ts"] = t
        head = price_head(snap, i, label="NIFTY 50 · history")
        fig, sm, lad, pl, lg, chk, nx, nw = render(LIVE, snap, "x", i, "travel", chip, custom, var, win, False)
        days = bars.index.normalize().unique()
        since = days[-5] if len(days) >= 5 else days[0]
        return ban, head, fig, sm, lad, pl, lg, chk, nx, nw, score_table(LIVE, since)

    return app
