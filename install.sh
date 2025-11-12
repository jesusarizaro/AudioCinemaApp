#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
CFG="${APP_DIR}/config/config.yaml"

# Detectar venv python real
if [[ -x "${APP_DIR}/venv/bin/python" ]]; then
  PY="${APP_DIR}/venv/bin/python"
else
  # fallback al python3 del sistema
  PY="$(command -v python3)"
fi

USER_NAME="${USER}"

# Leer OnCalendar del YAML (línea 'oncalendar:' a la derecha)
if [[ -f "$CFG" ]]; then
  ONCAL=$(awk -F: '/^oncalendar:/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}' "$CFG")
fi
# Valor por defecto si viene vacío
if [[ -z "${ONCAL:-}" ]]; then
  ONCAL="*-*-* 02:00:00"
fi

echo "[1/3] Generando unidades…"
mkdir -p "${APP_DIR}/systemd"

sed -e "s#__APP_DIR__#${APP_DIR}#g" \
    -e "s#__PY__#${PY}#g" \
    -e "s#__USER__#${USER_NAME}#g" \
    "${APP_DIR}/systemd/audiocinema.service" > "${APP_DIR}/systemd/.gen.audiocinema.service"

sed -e "s#__ONCALENDAR__#${ONCAL}#g" \
    "${APP_DIR}/systemd/audiocinema.timer" > "${APP_DIR}/systemd/.gen.audiocinema.timer"

echo "[2/3] Instalando y habilitando…"
sudo cp "${APP_DIR}/systemd/.gen.audiocinema.service" /etc/systemd/system/audiocinema.service
sudo cp "${APP_DIR}/systemd/.gen.audiocinema.timer"   /etc/systemd/system/audiocinema.timer
sudo systemctl daemon-reload
sudo systemctl enable audiocinema.timer
sudo systemctl restart audiocinema.timer

echo "[3/3] Estado:"
systemctl status audiocinema.timer --no-pager || true
echo
echo "Timers:"
systemctl list-timers audiocinema.timer --all || true

echo
echo "✅ Listo. Si cambias la hora en Configuración, vuelve a ejecutar:"
echo "   bash install_systemd.sh"
