#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path

# Raíz de la app (dos niveles arriba de este archivo)
APP_DIR: Path = Path(__file__).resolve().parents[1]

CFG_DIR: Path = APP_DIR / "config"
DATA_DIR: Path = APP_DIR / "data"
CAPTURES_DIR: Path = DATA_DIR / "captures"
REPORTS_DIR: Path = DATA_DIR / "reports"
ASSETS_DIR: Path = APP_DIR / "assets"

def ensure_dirs() -> None:
    for d in (CFG_DIR, DATA_DIR, CAPTURES_DIR, REPORTS_DIR, ASSETS_DIR):
        d.mkdir(parents=True, exist_ok=True)
