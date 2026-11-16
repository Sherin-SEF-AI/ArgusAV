#!/usr/bin/env bash
# Argus Labs desktop shortcut and app-menu entry for the current user (no root needed).
#   bash setup/install-desktop.sh            install
#   bash setup/install-desktop.sh --remove   remove
set -euo pipefail
ROOT="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
APPS="$HOME/.local/share/applications"
DESK="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
ICONS="$HOME/.local/share/icons/hicolor/scalable/apps"

if [ "${1:-}" = "--remove" ]; then
    rm -f "$APPS/argus.desktop" "$DESK/argus.desktop" "$ICONS/argus.svg"
    echo "removed"
    exit 0
fi

mkdir -p "$APPS" "$DESK" "$ICONS"
cp "$ROOT/avrig/assets/argus.svg" "$ICONS/argus.svg"
cat > "$APPS/argus.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Argus Labs
GenericName=AV research data collection
Comment=Six-camera hardware-synced data collection rig (avrig gui)
Exec=$ROOT/bin/argus-launch
Icon=argus
Terminal=false
Categories=Science;Engineering;
Keywords=camera;rig;recording;GMSL;IMU;GNSS;data collection;
StartupWMClass=Argus Labs
StartupNotify=true
DESKTOP
cp "$APPS/argus.desktop" "$DESK/argus.desktop"
chmod +x "$APPS/argus.desktop" "$DESK/argus.desktop"
# GNOME only launches desktop files that are marked trusted
gio set "$DESK/argus.desktop" metadata::trusted true 2>/dev/null || true
update-desktop-database "$APPS" 2>/dev/null || true
gtk-update-icon-cache -q "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
echo "installed: $DESK/argus.desktop and the app menu entry"
