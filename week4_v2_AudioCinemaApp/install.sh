#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
REQ="${APP_DIR}/requirements.txt"
PY="${APP_DIR}/venv/bin/python"
PIP="${APP_DIR}/venv/bin/pip"
USER_NAME="${USER}"
PNG_SRC="${APP_DIR}/assets/audiocinema.png"
ICO_DST="${APP_DIR}/assets/audiocinema.ico"

echo "[1/3] Creando entorno Python…"
command -v python3 >/dev/null || { echo "Python3 no encontrado"; exit 1; }
python3 -m venv "${APP_DIR}/venv"
"${PIP}" install --upgrade pip wheel
"${PIP}" install -r "${REQ}"

echo "[2/3] Generando instalador (iconos, menú y systemd)…"
if ! command -v convert >/dev/null 2>&1; then
  echo "  (Opcional) Instala ImageMagick para .ico: sudo apt install -y imagemagick"
else
  if [ -f "${PNG_SRC}" ]; then
    convert "${PNG_SRC}" "${ICO_DST}" || true
    echo "  ICO generado en ${ICO_DST}"
  fi
fi

sudo mkdir -p /usr/share/pixmaps
if [ -f "${PNG_SRC}" ]; then
  sudo cp "${PNG_SRC}" /usr/share/pixmaps/audiocinema.png
fi

mkdir -p "${HOME}/.local/share/icons/hicolor/256x256/apps"
if [ -f "${PNG_SRC}" ]; then
  cp "${PNG_SRC}" "${HOME}/.local/share/icons/hicolor/256x256/apps/audiocinema.png"
fi
update-icon-theme 2>/dev/null || true
sudo update-icon-caches /usr/share/icons/hicolor 2>/dev/null || true

mkdir -p "${HOME}/.local/share/applications"
cat > "${HOME}/.local/share/applications/audiocinema.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=AudioCinema
Comment=Analizador de sonido para salas de cine
Exec=${PY} ${APP_DIR}/src/gui_app.py
Icon=audiocinema
Terminal=false
Categories=AudioVideo;Utility;
StartupNotify=true
StartupWMClass=AudioCinema
EOF
update-desktop-database "${HOME}/.local/share/applications" 2>/dev/null || true

"${PY}" "${APP_DIR}/src/main.py" --setup || true

echo "[3/3] Verificación…"
"${PY}" "${APP_DIR}/src/doctor.py" || true

echo "
✅ Instalación lista.
Siga la siguiente ruta para abrir el programa AudioCinema:
• Menú → Sonido y Video → AudioCinema
"
