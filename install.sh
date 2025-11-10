#!/usr/bin/env bash
set -euo pipefail

# === DETECTAR DIRECTORIOS ===
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USER_HOME=$(eval echo "~$USER")
DESKTOP_DIR="$USER_HOME/Desktop"
LOCAL_APPS_DIR="$USER_HOME/.local/share/applications"

echo "=== [1/6] Creando entorno virtual ==="
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "=== [2/6] Instalando dependencias ==="
pip install --upgrade pip wheel
pip install -r "$PROJECT_DIR/requirements.txt"

echo "=== [3/6] Creando carpetas de datos/config ==="
mkdir -p "$PROJECT_DIR/data/captures" "$PROJECT_DIR/data/reports" "$PROJECT_DIR/config"

CFG="$PROJECT_DIR/config/config.json"
if [ ! -f "$CFG" ]; then
cat > "$CFG" <<'JSON'
{
  "reference": {"wav_path": "assets/reference.wav"},
  "audio": {"fs": 44100, "duration_s": 8.0, "preferred_input_name": ""},
  "thingsboard": {"host":"thingsboard.cloud","port":1883,"use_tls":false,"token":"REEMPLAZA_TU_TOKEN"},
  "schedule": {"mode":"daemon","interval":"15m"},
  "oncalendar": "every 15m"
}
JSON
  echo "[cfg] Archivo config.json creado."
fi

echo "=== [4/6] Verificando assets provistos por el usuario ==="
if [ ! -d "$PROJECT_DIR/assets" ]; then
  echo "❌ No existe la carpeta '$PROJECT_DIR/assets'. Debes proveerla antes de instalar."
  exit 1
fi
if [ ! -f "$PROJECT_DIR/assets/audiocinema.png" ]; then
  echo "❌ Falta '$PROJECT_DIR/assets/audiocinema.png'. Debes proveer el icono."
  exit 1
fi
if [ ! -f "$PROJECT_DIR/assets/reference.wav" ]; then
  echo "❌ Falta '$PROJECT_DIR/assets/reference.wav'. Debes proveer la referencia de audio."
  exit 1
fi
echo "✅ Assets verificados."

echo "=== [5/6] Instalando servicio systemd (daemon headless) ==="
SERVICE_SRC="$PROJECT_DIR/systemd/audiocinema.service"
SERVICE_DST="/etc/systemd/system/audiocinema.service"

if [ ! -f "$SERVICE_SRC" ]; then
  echo "❌ Falta '$SERVICE_SRC'. Asegúrate de tener systemd/audiocinema.service en el proyecto."
  exit 1
fi

sudo bash -c "sed 's|%%PROJECT_DIR%%|$PROJECT_DIR|g' '$SERVICE_SRC' > '$SERVICE_DST'"
sudo systemctl daemon-reload
sudo systemctl enable audiocinema.service
sudo systemctl restart audiocinema.service
echo "[systemd] Servicio activo. Usa: sudo systemctl status audiocinema.service"

echo "=== [6/6] Creando acceso directo (.desktop) ==="
mkdir -p "$LOCAL_APPS_DIR" "$DESKTOP_DIR"

DESKTOP_FILE="$LOCAL_APPS_DIR/audiocinema.desktop"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=AudioCinema
Comment=Analizador de audio IoT
Exec=$PROJECT_DIR/.venv/bin/python -m src.gui_app
Icon=$PROJECT_DIR/assets/audiocinema.png
Path=$PROJECT_DIR
Terminal=false
Categories=Audio;Utility;
StartupNotify=true
EOF

cp "$DESKTOP_FILE" "$DESKTOP_DIR/audiocinema.desktop" 2>/dev/null || true
chmod +x "$DESKTOP_FILE" "$DESKTOP_DIR/audiocinema.desktop" 2>/dev/null || true

echo "✅ Instalación completa."
echo "Abre la GUI desde el menú (Sonido y video → AudioCinema) o el icono del escritorio."
echo "El servicio headless se ejecuta en segundo plano y enviará datos automáticamente."
