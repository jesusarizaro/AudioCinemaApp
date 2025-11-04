#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from app_platform import APP_DIR, CFG_DIR, ASSETS_DIR, ensure_dirs
from configio import load_config
from pathlib import Path

def main():
    ensure_dirs()
    cfg = load_config()
    print(f"✅ Config cargada de: {Path(CFG_DIR, 'config.yaml')}")
    if not (ASSETS_DIR / "audiocinema.png").exists():
        print("⚠ Falta assets/audiocinema.png (icono)")
    ref = Path(cfg["reference"]["wav_path"])
    if not ref.exists():
        print(f"⚠ Referencia no encontrada: {ref}")
    print("✅ Directorios listos:", (APP_DIR/"data").absolute())

if __name__ == "__main__":
    main()
