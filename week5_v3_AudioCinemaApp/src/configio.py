#!/usr/bin/env python3
from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict
from app_platform import CFG_DIR, ensure_dirs, APP_DIR

CFG_FILE = CFG_DIR / "config.yaml"

_DEFAULTS: Dict[str, Any] = {
    "reference": {
        # apunta a un wav dentro de assets por defecto
        "wav_path": str((APP_DIR / "assets" / "reference.wav").absolute())
    },
    "audio": {
        "fs": 48000,
        "duration_s": 10.0,
        "preferred_input_name": ""
    },
    "thingsboard": {
        "host": "thingsboard.cloud",
        "port": 1883,
        "use_tls": False,
        "token": "REEMPLAZA_TU_TOKEN"
    },
    # systemd OnCalendar (ejemplo: todos los días 02:00)
    "oncalendar": "*-*-* 02:00:00"
}

def load_config() -> Dict[str, Any]:
    ensure_dirs()
    if not CFG_FILE.exists():
        save_config(_DEFAULTS.copy())
        return _DEFAULTS.copy()
    with open(CFG_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # merge simple (nivel 1/2)
    cfg = _DEFAULTS.copy()
    for k, v in data.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg

def save_config(cfg: Dict[str, Any]) -> None:
    ensure_dirs()
    with open(CFG_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
