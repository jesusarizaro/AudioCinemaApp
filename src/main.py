#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from datetime import datetime
import json
import sys

# Rutas / utilidades de plataforma y configuración
from app_platform import ensure_dirs, CFG_DIR, REP_DIR, APP_DIR
from configio import load_config, save_config

# Análisis y envío a ThingsBoard
import numpy as np
import soundfile as sf
from analyzer import (
    normalize_mono, record_audio, analyze_pair,
    detect_beeps, build_segments, build_json_payload
)
from iot_tb import send_json_to_thingsboard

# Doctor (re-uso del chequeo existente)
try:
    from doctor import main as doctor_main
except Exception:
    doctor_main = None


# ==========================
# Utilidades internas
# ==========================
def _resample_mono_for_run(x: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """
    Re-muestreo lineal simple + conversión a mono + normalización.
    Evita dependencias adicionales (p. ej. SciPy) para correr en Raspberry sin problemas.
    """
    if fs_in == fs_out:
        return normalize_mono(x)
    x = normalize_mono(x)
    n_new = int(round(len(x) * fs_out / fs_in))
    if n_new <= 1 or len(x) <= 1:
        # Evita casos degenerados
        return normalize_mono(x.astype(np.float32))
    idx = np.linspace(0.0, 1.0, len(x), dtype=np.float64)
    new_idx = np.linspace(0.0, 1.0, n_new, dtype=np.float64)
    y = np.interp(new_idx, idx, x).astype(np.float32)
    return y


# ==========================
# Comandos existentes
# ==========================
def cmd_setup() -> int:
    """
    Crea la estructura de carpetas y genera/actualiza el YAML de configuración.
    """
    ensure_dirs()
    cfg = load_config()
    save_config(cfg)
    print(f"✅ Config lista en {CFG_DIR / 'config.yaml'}")
    return 0


def cmd_doctor() -> int:
    """
    Ejecuta el 'doctor' existente si está disponible; si no, hace un chequeo mínimo.
    """
    ensure_dirs()
    if doctor_main is not None:
        return int(doctor_main() or 0)

    # Chequeo mínimo si no existe doctor.main
    print("ℹ️  Ejecutando chequeo mínimo...")
    cfg = load_config()
    print(f"✅ Config encontrada: {CFG_DIR / 'config.yaml'}")
    REP_DIR.mkdir(parents=True, exist_ok=True)
    print("✅ Directorios de datos creados.")

    ref_path = Path(cfg["reference"]["wav_path"])
    if not ref_path.exists():
        print("⚠️  Falta el archivo de referencia definido en config.reference.wav_path")
        return 1
    print("✅ Archivo de referencia OK.")
    print("✅ Config OK.")
    return 0


# ==========================
# NUEVO: modo headless (run once)
# ==========================
def run_once_headless() -> int:
    """
    Ejecuta UNA medición sin abrir la GUI:
    - Carga referencia
    - Graba muestra actual
    - Analiza (RMS, bandas, crest, etc.)
    - Construye JSON y lo guarda en data/reports/
    - (Opcional) Envia a ThingsBoard si hay token válido

    Retorna código de salida tipo proceso:
      0 = PASSED
      1 = FAILED
      2 = Error referencia
      3 = Error grabación
      4 = Error análisis u otro
    """
    try:
        ensure_dirs()
        cfg = load_config()

        fs = int(cfg["audio"]["fs"])
        dur = float(cfg["audio"]["duration_s"])
        preferred_input_name = cfg["audio"].get("preferred_input_name")
        device_index = None

        # Selección opcional de dispositivo por nombre parcial
        if preferred_input_name:
            import sounddevice as sd
            try:
                for i, d in enumerate(sd.query_devices()):
                    if d.get("max_input_channels", 0) > 0 and preferred_input_name.lower() in str(d.get("name", "")).lower():
                        device_index = i
                        break
            except Exception:
                # Si falla la búsqueda de dispositivos, seguimos con el predeterminado
                pass

        # 1) Cargar referencia
        ref_path = Path(cfg["reference"]["wav_path"])
        if not ref_path.exists():
            print(f"[ERR] No se encuentra la referencia WAV: {ref_path}")
            return 2
        x_ref, fs_ref = sf.read(ref_path, dtype="float32", always_2d=False)
        if x_ref.ndim == 2:
            x_ref = x_ref.mean(axis=1)
        x_ref = _resample_mono_for_run(x_ref, fs_ref, fs)

        # 2) Grabar actual
        try:
            x_cur = record_audio(dur, fs=fs, channels=1, device=device_index)
        except Exception as e:
            print(f"[ERR] Falló la grabación del micrófono: {e}")
            return 3

        # 3) Analizar
        res = analyze_pair(x_ref, x_cur, fs)

        # 4) Detección de beeps y segmentos (si aplica en tu pipeline)
        ref_markers = detect_beeps(x_ref, fs)
        cur_markers = detect_beeps(x_cur, fs)
        ref_segments = build_segments(x_ref, fs, ref_markers)
        cur_segments = build_segments(x_cur, fs, cur_markers)

        # 5) JSON: construir y guardar en data/reports
        payload = build_json_payload(
            fs, res, [],
            ref_markers, cur_markers,
            ref_segments, cur_segments,
            None, None
        )
        REP_DIR.mkdir(parents=True, exist_ok=True)
        out = REP_DIR / f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # 6) Enviar a ThingsBoard (si hay token válido)
        tb = cfg["thingsboard"]
        sent = False
        if tb.get("token") and tb["token"] != "REEMPLAZA_TU_TOKEN":
            sent = send_json_to_thingsboard(payload, tb["host"], int(tb["port"]), tb["token"], bool(tb["use_tls"]))

        # 7) Salida y código de retorno
        print(f"[OK] overall={res.get('overall','?')} | JSON={out} | TB={'enviado' if sent else 'no-enviado'}")
        return 0 if res.get("overall") == "PASSED" else 1

    except KeyboardInterrupt:
        print("\n[INT] Cancelado por el usuario.")
        return 130
    except SystemExit as e:
        return int(getattr(e, "code", 4) or 0)
    except Exception as e:
        print(f"[ERR] Error no controlado en run-once: {e}")
        return 4


# ==========================
# Punto de entrada
# ==========================
def main():
    ap = argparse.ArgumentParser(
        description="AudioCinema - utilidades CLI (setup, doctor, run-once)"
    )
    ap.add_argument("--setup", action="store_true", help="Crea carpetas base y genera/actualiza config.yaml")
    ap.add_argument("--doctor", action="store_true", help="Verifica configuración y prerequisitos")
    ap.add_argument("--run-once", action="store_true", help="Ejecuta una corrida headless y termina (para systemd timer)")
    args = ap.parse_args()

    if args.setup:
        sys.exit(cmd_setup())

    if args.doctor:
        sys.exit(cmd_doctor())

    if args.run-once:
        sys.exit(run_once_headless())

    # Si no se especificó nada, muestra ayuda (la GUI se lanza por su propio launcher/desktop entry).
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
