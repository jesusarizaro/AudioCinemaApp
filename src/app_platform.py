#!/usr/bin/env python3
from pathlib import Path

# APP_DIR: raíz del proyecto (carpeta "AudioCinema"), relativo al file actual
# src/ está justo debajo de la raíz, así que parent es el proyecto.
APP_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = (APP_DIR / "assets").absolute()
DATA_DIR = (APP_DIR / "data").absolute()
CAPTURES_DIR = (DATA_DIR / "captures").absolute()
REPORTS_DIR = (DATA_DIR / "reports").absolute()
CONFIG_DIR = (APP_DIR / "config").absolute()
CONFIG_PATH = (CONFIG_DIR / "config.json").absolute()

def ensure_dirs():
    for p in (ASSETS_DIR, CAPTURES_DIR, REPORTS_DIR, CONFIG_DIR):
        p.mkdir(parents=True, exist_ok=True)
