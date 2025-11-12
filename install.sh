#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${APP_DIR}/venv"
PY="${VENV}/bin/python"
PIP="${VENV}/bin/pip"
PNG_SRC="${APP_DIR}/assets/audiocinema.png"

echo "[1/3] Preparando entorno…"
command -v python3 >/dev/null || { echo "Python3 no encontrado"; exit 1; }
python3 -m venv "$VENV"
"$PIP" -q install --upgrade pip wheel
[ -f "${APP_DIR}/requirements.txt" ] && "$PIP" -q install -r "${APP_DIR}/requirements.txt" || true

echo "[2/3] Lanzador…"
# Copia de iconos (usuario y sistema si hay sudo)
mkdir -p "${HOME}/.local/share/icons/hicolor/256x256/apps"
[ -f "$PNG_SRC" ] && cp "$PNG_SRC" "${HOME}/.local/share/icons/hicolor/256x256/apps/audiocinema.png" || true
if command -v sudo >/dev/null 2>&1; then
  sudo mkdir -p /usr/share/pixmaps
  [ -f "$PNG_SRC" ] && sudo cp "$PNG_SRC" /usr/share/pixmaps/audiocinema.png || true
fi

# Wrapper ejecutable que arranca la app (activa venv y lanza GUI)
cat > "${APP_DIR}/run_gui.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${APP_DIR}/venv/bin/python" "${APP_DIR}/src/gui_app.py"
EOF
chmod +x "${APP_DIR}/run_gui.sh"

# .desktop para el usuario (si hay sudo se instala también a nivel sistema)
USR_DESKTOP="${HOME}/.local/share/applications/AudioCinema.desktop"
mkdir -p "$(dirname "$USR_DESKTOP")"
cat > "$USR_DESKTOP" <<EOF
[Desktop Entry]
Type=Application
Name=AudioCinema
Comment=Analizador de audio (grabación, evaluación y envío a ThingsBoard)
TryExec=${APP_DIR}/run_gui.sh
Exec=${APP_DIR}/run_gui.sh
Icon=audiocinema
Terminal=false
Categories=AudioVideo;Utility;
StartupWMClass=AudioCinema
X-AppImage-Integrate=false
EOF

# Lanzador global opcional (requiere sudo)
if command -v sudo >/dev/null 2>&1; then
  SYS_DESKTOP="/usr/share/applications/audiocinema.desktop"
  sudo bash -c "cat > '$SYS_DESKTOP' <<EOF
[Desktop Entry]
Type=Application
Name=AudioCinema
Comment=Analizador de audio (grabación, evaluación y envío a ThingsBoard)
TryExec=${APP_DIR}/run_gui.sh
Exec=${APP_DIR}/run_gui.sh
Icon=audiocinema
Terminal=false
Categories=AudioVideo;Utility;
StartupWMClass=AudioCinema
X-AppImage-Integrate=false
EOF"
fi

# Actualiza cachés de iconos/menú (ignorar si no existen)
update-desktop-database "${HOME}/.local/share/applications" >/dev/null 2>&1 || true
sudo update-desktop-database /usr/share/applications >/dev/null 2>&1 || true
update-icon-caches "${HOME}/.local/share/icons/hicolor" >/dev/null 2>&1 || true
sudo update-icon-caches /usr/share/icons/hicolor >/dev/null 2>&1 || true

echo "[3/3] Verificación…"
"${PY}" - <<'PY'
print("✓ Python OK")
PY

echo "
✅ Instalación lista.
Siga la siguiente ruta para abrir el programa AudioCinema:
• Menú → Sonido y Video → AudioCinema
"
