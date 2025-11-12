#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from app_platform import APP_DIR, ASSETS_DIR, ensure_dirs
from configio import load_config
from analyzer import (
    normalize_mono, record_audio, analyze_pair,
    detect_beeps, build_segments, build_json_payload
)
from iot_tb import send_json_to_thingsboard

APP_NAME = "AudioCinema (headless)"

def _read_reference(ref_path: Path, fs_target: int):
    x, fs = sf.read(str(ref_path), dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = normalize_mono(x)
    if fs != fs_target:
        # re-muestreo simple por interpolación (suficiente para runner)
        n_new = int(round(len(x) * fs_target / fs))
        idx_old = np.linspace(0, 1, len(x))
        idx_new = np.linspace(0, 1, n_new)
        x = np.interp(idx_new, idx_old, x).astype(np.float32)
    return x

def main():
    ensure_dirs()
    cfg = load_config()

    fs  = int(cfg["audio"]["fs"])
    dur = float(cfg["audio"]["duration_s"])

    # referencia (si no hay en config, usar assets/reference_master.wav)
    ref_path = Path(cfg.get("reference",{}).get("file", str(ASSETS_DIR / "reference_master.wav")))
    if not ref_path.exists():
        raise FileNotFoundError(f"Referencia no encontrada: {ref_path}")

    x_ref = _read_reference(ref_path, fs)
    x_cur = record_audio(dur, fs=fs, channels=1)  # mic por defecto

    res = analyze_pair(x_ref, x_cur, fs)

    ref_markers = detect_beeps(x_ref, fs)
    cur_markers = detect_beeps(x_cur, fs)
    ref_segments = build_segments(x_ref, fs, ref_markers)
    cur_segments = build_segments(x_cur, fs, cur_markers)

    payload = build_json_payload(
        fs, res, [],
        ref_markers, cur_markers,
        ref_segments, cur_segments,
        None, None
    )

    out_dir = (APP_DIR / "data" / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    tb = cfg["thingsboard"]
    sent = False
    if tb.get("token") and tb["token"] != "REEMPLAZA_TU_TOKEN":
        sent = send_json_to_thingsboard(payload, tb["host"], int(tb["port"]), tb["token"], bool(tb["use_tls"]))

    print(f"[{APP_NAME}] Resultado: {res['overall']}  JSON: {out}  EnviadoTB: {sent}")

if __name__ == "__main__":
    main()
