#!/usr/bin/env bash
set -euo pipefail

# Detecta ruta absoluta del proyecto (independiente del usuario/sitio)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[1/6] Creando venv..."
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[2/6] Instalando dependencias..."
pip install --upgrade pip wheel
pip install -r "$PROJECT_DIR/requirements.txt"

echo "[3/6] Creando carpetas de datos..."
mkdir -p "$PROJECT_DIR/data/captures" "$PROJECT_DIR/data/reports" "$PROJECT_DIR/config"

# Config por defecto si no existe
CFG="$PROJECT_DIR/config/config.json"
if [ ! -f "$CFG" ]; then
  cat > "$CFG" <<'JSON'
{
  "reference": {
    "wav_path": "assets/reference.wav"
  },
  "audio": {
    "fs": 44100,
    "duration_s": 8.0,
    "preferred_input_name": ""
  },
  "thingsboard": {
    "host": "thingsboard.cloud",
    "port": 1883,
    "use_tls": false,
    "token": "REEMPLAZA_TU_TOKEN"
  },
  "schedule": {
    "mode": "daemon",          // "daemon" recomendado; "timer" opcional
    "interval": "15m"          // 10m, 15m, 1h, 2h, etc.
  },
  "oncalendar": "every 15m"    // por compatibilidad con tu GUI actual
}
JSON
  echo "[cfg] Escrito config.json por defecto."
fi

# Icono y assets básicos
mkdir -p "$PROJECT_DIR/assets"
if [ ! -f "$PROJECT_DIR/assets/audiocinema.png" ]; then
  # icono placeholder
  convert -size 256x256 xc:#0d6efd -gravity center -pointsize 72 -fill white \
    -annotate 0 "AC" "$PROJECT_DIR/assets/audiocinema.png" 2>/dev/null || true
fi
if [ ! -f "$PROJECT_DIR/assets/reference.wav" ]; then
  # referencia placeholder: 1kHz seno 8s @ -12 dBFS
  python3 - <<'PY'
import numpy as np, soundfile as sf, os
sr=44100; t=8.0
n=int(sr*t)
x=0.25*np.sin(2*np.pi*1000*np.arange(n)/sr).astype(np.float32)
os.makedirs("assets", exist_ok=True)
sf.write("assets/reference.wav", x, sr)
PY
fi

echo "[4/6] Instalando systemd service..."
# Genera servicio con rutas correctas
SERVICE_SRC="$PROJECT_DIR/systemd/audiocinema.service"
SERVICE_DST="/etc/systemd/system/audiocinema.service"

# Sustituye la variable %%PROJECT_DIR%% por la ruta real
sudo bash -c "sed 's|%%PROJECT_DIR%%|$PROJECT_DIR|g' '$SERVICE_SRC' > '$SERVICE_DST'"

# (Opcional) instala el timer con un intervalo base (no se usará si mode=daemon)
TIMER_SRC="$PROJECT_DIR/systemd/audiocinema.timer"
TIMER_DST="/etc/systemd/system/audiocinema.timer"
if [ -f "$TIMER_SRC" ]; then
  sudo bash -c "sed 's|%%PROJECT_DIR%%|$PROJECT_DIR|g' '$TIMER_SRC' > '$TIMER_DST'"
fi

echo "[5/6] Recargando systemd y habilitando servicio..."
sudo systemctl daemon-reload
sudo systemctl enable audiocinema.service
sudo systemctl restart audiocinema.service

# (Opcional) habilitar timer si quieres modo timer en el futuro
# sudo systemctl enable --now audiocinema.timer

echo "[6/6] Listo ✅"
echo "Servicio activo:   sudo systemctl status audiocinema.service"
echo "Ejecutar 1 vez:    $VENV_DIR/bin/python -m src.headless_runner --once"
echo "Logs en vivo:      journalctl -u audiocinema.service -f"
