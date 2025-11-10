#!/usr/bin/env python3
import json, re, time
from pathlib import Path
from typing import Any, Dict
from app_platform import CONFIG_PATH, ensure_dirs

_DEFAULT = {
    "reference": {"wav_path": "assets/reference.wav"},
    "audio": {"fs": 44100, "duration_s": 8.0, "preferred_input_name": ""},
    "thingsboard": {"host":"thingsboard.cloud","port":1883,"use_tls":False,"token":"REEMPLAZA_TU_TOKEN"},
    "schedule": {"mode":"daemon","interval":"15m"},
    "oncalendar": "every 15m"
}

def _parse_interval_to_seconds(text: str) -> int:
    """
    Acepta '10m', '15min', '1h', '2h', '900s', 'every 10m', 'cada 10m', etc.
    Devuelve segundos (mínimo 60s).
    """
    if not text: return 900
    s = text.strip().lower()
    s = s.replace("every","").replace("cada","").strip()
    m = re.match(r"^(\d+)\s*(s|sec|secs|second|seconds)$", s)
    if m: return max(60, int(m.group(1)))
    m = re.match(r"^(\d+)\s*(m|min|mins|minute|minutes)$", s)
    if m: return max(60, int(m.group(1))*60)
    m = re.match(r"^(\d+)\s*(h|hr|hour|hours)$", s)
    if m: return max(60, int(m.group(1))*3600)
    # casos '10m', '1h' sin sufijos plurales
    if s.endswith('m') and s[:-1].isdigit(): return max(60, int(s[:-1])*60)
    if s.endswith('h') and s[:-1].isdigit(): return max(60, int(s[:-1])*3600)
    if s.endswith('s') and s[:-1].isdigit(): return max(60, int(s[:-1]))
    # fallback
    return 900

def load_config() -> Dict[str,Any]:
    ensure_dirs()
    if not CONFIG_PATH.exists():
        save_config(_DEFAULT)
        return json.loads(json.dumps(_DEFAULT))
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = json.loads(json.dumps(_DEFAULT))
    # sane defaults merge
    def merge(a,b):
        for k,v in b.items():
            if k not in a: a[k]=v
            elif isinstance(v,dict): merge(a[k],v)
    merge(cfg, _DEFAULT)
    return cfg

def save_config(cfg: Dict[str,Any]) -> None:
    ensure_dirs()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def read_interval_seconds(cfg: Dict[str,Any]) -> int:
    # prioridad: schedule.interval (nuevo) > oncalendar (compat GUI)
    if "schedule" in cfg and "interval" in cfg["schedule"]:
        return _parse_interval_to_seconds(str(cfg["schedule"]["interval"]))
    return _parse_interval_to_seconds(str(cfg.get("oncalendar","15m")))
