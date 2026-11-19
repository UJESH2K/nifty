"""Config loading + small shared helpers."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _load_dotenv(ROOT / ".env")
    cfg["_root"] = str(ROOT)
    return cfg


def _load_dotenv(p: Path) -> None:
    """Minimal .env loader (no extra dependency)."""
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def data_dir(cfg: dict) -> Path:
    d = ROOT / cfg["paths"]["data_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir(cfg: dict) -> Path:
    d = ROOT / cfg["paths"]["models_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def horizons(cfg: dict, resolution: str | None = None) -> list[int]:
    return list(cfg["horizons"][resolution or cfg["resolution"]])


def bar_seconds(resolution: str) -> int:
    return {"1min": 60, "1s": 1}[resolution]
