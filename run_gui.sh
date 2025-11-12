#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/audio/AudioCinemaApp"
PY="${APP_DIR}/venv/bin/python"   # ajusta si tu venv tiene otro nombre
GUI="${APP_DIR}/src/gui_app.py"

cd "$APP_DIR"
exec "$PY" "$GUI"
