"""Build a slim copy of NiftySim for Render's free plan (512 MB RAM, no lasting disk).

    python scripts/make_render_bundle.py

Creates ../niftysim_render/ next to this project folder with:
  * the code + render.yaml
  * only the last 320 trading days of 1-minute data (your PC keeps all 11 years)
  * headlines, events, and freshly trained naive + ridge models that match this code
Then push that folder to GitHub and connect it on Render (see deploy/RENDER.md).
Run it again whenever you want the cloud copy to start from your PC's latest data.
"""
import os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from niftysim.storage import load_bars

OUT = ROOT.parent / "niftysim_render"
DAYS = 320

OUT.mkdir(exist_ok=True)
for d in ("niftysim", "app", "scripts", "deploy"):
    if (OUT / d).exists():
        shutil.rmtree(OUT / d)
    shutil.copytree(ROOT / d, OUT / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
for f in ("config.yaml", "render.yaml", "requirements-render.txt", "README.md", ".env.example"):
    if (ROOT / f).exists():
        shutil.copy(ROOT / f, OUT / f)
shutil.copy(ROOT / "requirements-render.txt", OUT / "requirements.txt")
(OUT / ".gitignore").write_text(".env\n__pycache__/\ndata/ticks/\ndata/backup_*/\n")

# data: last DAYS sessions only
(OUT / "data").mkdir(exist_ok=True)
bars = load_bars(ROOT / "data" / "bars_1min.parquet")
cut = bars.index.normalize().unique()[-DAYS]
bars = bars[bars.index >= cut]
bars.to_parquet(OUT / "data" / "bars_1min.parquet")
for f in ("news.parquet", "events.csv", "events.example.csv"):
    if (ROOT / "data" / f).exists():
        shutil.copy(ROOT / "data" / f, OUT / "data" / f)
print(f"data: {len(bars):,} bars, {bars.index[0]:%d %b %Y} -> {bars.index[-1]:%d %b %Y}")

# models trained on exactly what the server will have
if (OUT / "models_store").exists():
    shutil.rmtree(OUT / "models_store")
env = dict(os.environ, NIFTYSIM_MAX_TRAIN_DAYS="250")
subprocess.run([sys.executable, "scripts/train_models.py", "--no-cv", "--models", "naive", "ridge"],
               cwd=OUT, env=env, check=True)
size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
print(f"\nReady: {OUT}  ({size:.0f} MB). Next: push it to GitHub and connect it on Render (deploy/RENDER.md).")
