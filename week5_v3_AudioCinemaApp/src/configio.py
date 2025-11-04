#!/usr/bin/env python3
from pathlib import Path
import yaml
from app_platform import CFG_DIR, ASSETS_DIR

CFG_PATH = CFG_DIR / "config.yaml"

DEFAULTS = {
    "oncalendar": "*-*-* 02:00:00",
    "thingsboard": {
        "host": "thingsboard.cloud",
        "port": 1883,
        "use_tls": False,
        "token": "REEMPLAZA_TU_TOKEN"
    },
    "audio": {
        "fs": 48000,
        "duration_s": 10.0,
        "preferred_input_name": ""
    },
    "reference": {
        "wav_path": str((ASSETS_DIR / "reference.wav").resolve())
    },
    "ui": {
        "last_test_name": "name"
    }
}

def load_config() -> dict:
    if not CFG_PATH.exists():
        save_config(DEFAULTS)
        return DEFAULTS.copy()
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    out = DEFAULTS.copy()
    for k, v in data.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            tmp = out[k].copy(); tmp.update(v); out[k] = tmp
        else:
            out[k] = v
    return out

def save_config(cfg: dict):
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CFG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
