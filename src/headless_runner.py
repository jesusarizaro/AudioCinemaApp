#!/usr/bin/env python3
import os, json, time, argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import soundfile as sf

from app_platform import APP_DIR, REPORTS_DIR
from configio import load_config, save_config, read_interval_seconds
from analyzer import (
    normalize_mono, record_audio, analyze_pair,
    detect_beeps, build_segments, build_json_payload, BANDS
)
from iot_tb import send_json_to_thingsboard

APP_NAME = "AudioCinema"

def _safe_resample(x: np.ndarray, fs_src: int, fs_dst: int) -> np.ndarray:
    if fs_src == fs_dst: return x
    n_new = int(round(len(x) * fs_dst / fs_src))
    idx_src = np.linspace(0, 1, len(x), dtype=np.float32)
    idx_new = np.linspace(0, 1, n_new, dtype=np.float32)
    return np.interp(idx_new, idx_src, x).astype(np.float32)

def run_once() -> bool:
    cfg = load_config()
    fs = int(cfg["audio"]["fs"]); dur = float(cfg["audio"]["duration_s"])

    # 1) referencia
    ref_path = Path(APP_DIR / cfg["reference"]["wav_path"]).resolve()
    if not ref_path.exists():
        print(f"[{APP_NAME}] No existe referencia: {ref_path}")
        return False
    x_ref, fs_ref = sf.read(ref_path, dtype="float32", always_2d=False)
    if x_ref.ndim==2: x_ref = x_ref.mean(axis=1)
    x_ref = normalize_mono(x_ref)
    x_ref = _safe_resample(x_ref, fs_ref, fs)

    # 2) grabar actual
    x_cur = record_audio(dur, fs=fs, channels=1, device=None)

    # 3) análisis
    res = analyze_pair(x_ref, x_cur, fs)

    # 4) markers/segments
    ref_markers = detect_beeps(x_ref, fs)
    cur_markers = detect_beeps(x_cur, fs)
    ref_segments = build_segments(x_ref, fs, ref_markers)
    cur_segments = build_segments(x_cur, fs, cur_markers)

    # 5) export JSON
    payload = build_json_payload(
        fs, res, list(BANDS), ref_markers, cur_markers, ref_segments, cur_segments, None, None
    )
    out = REPORTS_DIR / f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 6) enviar a ThingsBoard (si token válido)
    tb = cfg["thingsboard"]
    sent = False
    if tb.get("token") and tb["token"] != "REEMPLAZA_TU_TOKEN":
        sent = send_json_to_thingsboard(payload, tb["host"], int(tb["port"]), tb["token"], bool(tb["use_tls"]))

    print(f"[{APP_NAME}] overall={res['overall']} json={out} sentTB={'yes' if sent else 'no'}")
    return True

def _touch_heartbeat():
    hb = APP_DIR / "data" / "audiocinema_heartbeat.txt"
    hb.parent.mkdir(parents=True, exist_ok=True)
    with open(hb, "w") as f:
        f.write(datetime.utcnow().isoformat() + "Z\n")

def daemon_loop():
    print(f"[{APP_NAME}] Daemon iniciado.")
    # Ciclo infinito: lee intervalos del config en cada iteración
    while True:
        try:
            _touch_heartbeat()
            run_once()
        except Exception as e:
            print(f"[{APP_NAME}] Error en run_once(): {e}")
        # lee config otra vez (por si cambió el intervalo en GUI)
        try:
            cfg = load_config()
            sec = max(60, read_interval_seconds(cfg))
        except Exception:
            sec = 900
        print(f"[{APP_NAME}] Próxima ejecución en {sec} s.")
        time.sleep(sec)

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--once", action="store_true", help="Ejecuta una sola prueba")
    g.add_argument("--daemon", action="store_true", help="Corre servicio en loop")
    args = ap.parse_args()
    if args.once: 
        ok = run_once()
        raise SystemExit(0 if ok else 1)
    else:
        daemon_loop()

if __name__ == "__main__":
    main()
