"""Start NiftySim.   python scripts/run_dashboard.py   then open http://127.0.0.1:8050

Options:
  --model ridge|lgbm|mlp|naive   which trained model to use (default ridge)
  --autopilot auto|nse|yahoo|angel|upstox
                                 run unattended: goes live every trading day, saves everything,
                                 scores itself and retrains every night. No buttons needed.
  --autostart SOURCE             just arm Go live once at startup (no nightly retrain)
  --host 0.0.0.0                 listen for other computers too (use with a password!)
  Password: set NIFTYSIM_PASSWORD in .env (and optionally NIFTYSIM_USER, default "family")
"""
import argparse, os, sys, webbrowser, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from niftysim.config import load_config, models_dir, horizons
from niftysim.engine import Engine
from niftysim.models import ForecastModel
from app.dashboard import build_app

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="ridge")
ap.add_argument("--res", default=None)
ap.add_argument("--history-days", type=int, default=10)
ap.add_argument("--speed", type=float, default=None, help="demo market speed (simulated seconds per real second)")
ap.add_argument("--autostart", default=None)
ap.add_argument("--autopilot", default=None)
ap.add_argument("--train-days", type=int, default=None, help="autopilot: retrain on the last N days only (default all)")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8050)))
ap.add_argument("--no-browser", action="store_true")
a = ap.parse_args()

cfg = load_config()
if a.speed:
    cfg["feed"]["speed"] = a.speed
res = a.res or cfg["resolution"]
mp = models_dir(cfg) / res / f"{a.model}_price.joblib"
if not mp.exists():
    sys.exit(f"No trained '{a.model}' model yet. Run first:\n  python scripts/setup_real_data.py")
if list(ForecastModel.load(mp).horizons) != horizons(cfg, res):
    sys.exit("Your saved models were trained for different horizons than config.yaml. Retrain:\n"
             "  python scripts/train_models.py --no-cv")
eng = Engine(cfg, res, a.model, a.history_days)

auto = None
if a.autopilot:
    from niftysim.autopilot import Autopilot
    auto = Autopilot(cfg, eng, a.autopilot, a.train_days)
    auto.start()
elif a.autostart:
    eng.go_live(a.autostart)

app = build_app(cfg, eng, autopilot=auto)


@app.server.route("/healthz")
def _healthz():
    st = eng.clock.status()
    return {"ok": True, "status": eng.status, "market_open": st["open"],
            "autopilot": auto.state if auto else None, "bars": int(len(eng.bars))}


# Keep-alive for free hosts that sleep after 15 idle minutes (Render sets RENDER_EXTERNAL_URL).
# The request goes out and back in through the public URL, so it counts as inbound traffic.
_ext = os.environ.get("NIFTYSIM_PUBLIC_URL") or os.environ.get("RENDER_EXTERNAL_URL")
if _ext:
    def _keepalive():
        import time, requests
        while True:
            time.sleep(600)
            try:
                requests.get(_ext.rstrip("/") + "/healthz", timeout=20)
            except Exception as ex:
                print("keep-alive ping failed:", ex)
    threading.Thread(target=_keepalive, daemon=True).start()
    print(f"keep-alive: pinging {_ext}/healthz every 10 minutes")

pw = os.environ.get("NIFTYSIM_PASSWORD")
if pw:
    from flask import Response, request
    user = os.environ.get("NIFTYSIM_USER", "family")

    @app.server.before_request
    def _auth():
        if request.path == "/healthz":
            return None
        au = request.authorization
        if not au or au.username != user or au.password != pw:
            return Response("Login required", 401, {"WWW-Authenticate": 'Basic realm="NiftySim"'})
elif a.host != "127.0.0.1":
    print("note: no password set, so anyone with the link can view (add NIFTYSIM_PASSWORD to .env for a login)")

url = f"http://127.0.0.1:{a.port}"
print(f"NiftySim running at {url}" + ("  [autopilot]" if auto else "") + "  (Ctrl+C to stop)")
if not a.no_browser:
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
app.run(debug=False, port=a.port, host=a.host)
