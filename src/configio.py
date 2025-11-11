#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import yaml

# Importa rutas de la app
from app_platform import CFG_DIR, ASSETS_DIR

CFG_PATH = CFG_DIR / "config.yaml"

# valores por defecto
DEFAULTS = {
    "general": {
        "oncalendar": "*-*-* 02:00:00",
    },
    "audio": {
        "fs": 48000,
        "duration_s": 10.0,
        "prefer_input_name": "",
    },
    "thingsboard": {
        "host": "thingsboard.cloud",
        "port": 1883,
        "use_tls": False,
        "token": "",
    },
    # NUEVO: referencia fija en assets
    "reference": {
        # por defecto usaremos assets/reference_master.wav
        "file": str((ASSETS_DIR / "reference_master.wav").resolve())
    }
}

def _ensure_dirs():
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def load_config() -> dict:
    _ensure_dirs()
    if CFG_PATH.exists():
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    # mezcla con defaults (shallow merge suficiente aquí)
    out = DEFAULTS | cfg
    # asegura sub-secciones
    for k in ("general","audio","thingsboard","reference"):
        out.setdefault(k, {}) 
        out[k] = DEFAULTS[k] | out[k]
    return out

def save_config(cfg: dict) -> None:
    _ensure_dirs()
    # garantizamos claves
    for k in ("general","audio","thingsboard","reference"):
        cfg.setdefault(k, {})
        cfg[k] = DEFAULTS[k] | cfg[k]
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
