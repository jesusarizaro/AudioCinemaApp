#!/usr/bin/env bash
set -euo pipefail

# === DETECTAR DIRECTORIOS ===
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USER_HOME=$(eval echo "~$USER")
DESKTOP_DIR="$USER_HOME/Desktop"
LOCAL_APPS_DIR="$USER_HOME/.local/share/applications"

echo "=== [1/7] Creando entorno virtual ==="
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "=== [2/7] Instalando dependencias ==="
pip install --upgrade pip wheel
pip install -r "$PROJECT_DIR/requirements.txt"

echo "=== [3/7] Creando carpetas necesarias ==="
mkdir -p "$PROJECT_DIR/data/captures" "$PROJECT_DIR/data/reports" "$PROJECT_DIR/config" "$PROJECT_DIR/assets"

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

# === [4/7] Crear referencia e icono si faltan ===
if [ ! -f "$PROJECT_DIR/assets/audiocinema.png" ]; then
  echo "[icon] Creando icono por defecto..."
  python3 - <<'PY'
import numpy as np, soundfile as sf, os, matplotlib.pyplot as plt
os.makedirs("assets", exist_ok=True)
fig, ax = plt.subplots(figsize=(2.56,2.56))
ax.axis("off")
ax.text(0.5,0.5,"AC",fontsize=120,va="center",ha="center",color="white",weight="bold")
fig.patch.set_facecolor("#0d6efd")
plt.savefig("assets/audiocinema.png", dpi=100, bbox_inches="tight", pad_inches=0)
PY
fi

if [ ! -f "$PROJECT_DIR/assets/reference.wav" ]; then
  echo "[ref] Creando referencia 1kHz.wav..."
  python3 - <<'PY'
import numpy as np, soundfile as sf, os
sr=44100; t=8.0
x=0.25*np.sin(2*np.pi*1000*np.arange(int(sr*t))/sr).astype(np.float32)
sf.write("assets/reference.wav",x,sr)
PY
fi

# === [5/7] Instalar servicio systemd ===
SERVICE_SRC="$PROJECT_DIR/systemd/audiocinema.service"
SERVICE_DST="/etc/systemd/system/audiocinema.service"

echo "[systemd] Instalando servicio..."
sudo bash -c "sed 's|%%PROJECT_DIR%%|$PROJECT_DIR|g' '$SERVICE_SRC' > '$SERVICE_DST'"
sudo systemctl daemon-reload
sudo systemctl enable audiocinema.service
sudo systemctl restart audiocinema.service
echo "[systemd] Servicio activo. Usa: sudo systemctl status audiocinema.service"

# === [6/7] Crear acceso directo (.desktop) ===
echo "[desktop] Creando acceso directo de AudioCinema..."

mkdir -p "$LOCAL_APPS_DIR"
mkdir -p "$DESKTOP_DIR"

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

# Copiar al escritorio también
cp "$DESKTOP_FILE" "$DESKTOP_DIR/audiocinema.desktop" 2>/dev/null || true
chmod +x "$DESKTOP_FILE" "$DESKTOP_DIR/audiocinema.desktop" 2>/dev/null || true

echo "[desktop] Acceso directo creado:"
echo " - Menú principal → Sonido y video → AudioCinema"
echo " - Escritorio → icono AudioCinema"

# === [7/7] Final ===
echo "✅ Instalación completa."
echo "Puedes abrir el programa desde el menú o el icono del escritorio."
echo "El servicio se ejecuta en segundo plano y enviará datos automáticamente."
