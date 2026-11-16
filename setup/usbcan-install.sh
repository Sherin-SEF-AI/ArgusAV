#!/usr/bin/env bash
# USB CAN adapters:
#  - Waveshare USB-CAN-A (CH340): builds the ch341 USB-serial driver for the running kernel
#    (the Jetson kernel ships without it); udev: /dev/usbcan0, group dialout, and
#    ModemManager told to leave the port alone (it would probe it with AT commands).
#  - CANalyst-II (04d8:0053, userspace driver over libusb): udev gives the plugdev group
#    access to the USB device.
#   sudo bash setup/usbcan-install.sh             install (re-run after a kernel update)
#   sudo bash setup/usbcan-install.sh --status    what is installed / plugged in
#   sudo bash setup/usbcan-install.sh --uninstall
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
KREL="$(uname -r)"
KO="/lib/modules/$KREL/updates/ch341.ko"
RULE=/etc/udev/rules.d/99-avrig-usbcan.rules

canalyst() {   # "plugged, <mode> <owner>:<group>" of the CANalyst-II USB device node
    local d
    for d in /sys/bus/usb/devices/*; do
        if [ "$(cat "$d/idVendor" 2>/dev/null)" = 04d8 ] && [ "$(cat "$d/idProduct")" = 0053 ]; then
            stat -c 'plugged, %A %U:%G' "/dev/bus/usb/$(printf %03d "$(cat "$d/busnum")")/$(printf %03d "$(cat "$d/devnum")")"
            return
        fi
    done
    echo "not plugged"
}

status() {
    echo "kernel:  $KREL"
    echo "driver:  $([ -f "$KO" ] && echo "installed ($KO)" || echo "not installed")," \
         "$(grep -q '^ch341 ' /proc/modules && echo loaded || echo 'not loaded')"
    echo "rule:    $([ -f "$RULE" ] && echo installed || echo 'not installed')"
    echo "USB-CAN-A: $(grep -l 1a86 /sys/bus/usb/devices/*/idVendor 2>/dev/null | while read -r f; do
        d=$(dirname "$f"); [ "$(cat "$d/idProduct")" = 7523 ] && echo plugged; done | head -1 | grep . || echo 'not plugged')"
    echo "CANalyst-II: $(canalyst)"
    echo "port:    $(ls -l /dev/usbcan0 2>/dev/null | awk '{print $NF}' | sed 's#^#/dev/usbcan0 -> #' | grep . || echo none)"
}

[ "${1:-}" = "--status" ] && { status; exit 0; }
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

if [ "${1:-}" = "--uninstall" ]; then
    rm -f "$RULE" "$KO"
    depmod -a "$KREL"
    modprobe -r ch341 2>/dev/null || true
    udevadm control --reload-rules
    echo "removed"
    exit 0
fi

[ -f "/lib/modules/$KREL/build/Makefile" ] || { echo "kernel headers for $KREL are missing"; exit 1; }
BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
cp "$HERE/usbcan/ch341.c" "$HERE/usbcan/Makefile" "$BUILD/"
echo "== building ch341 for $KREL"
make -s -C "/lib/modules/$KREL/build" M="$BUILD" modules
install -D -m 0644 "$BUILD/ch341.ko" "$KO"
depmod -a "$KREL"
modprobe ch341
cat > "$RULE" <<'RULES'
# avrig: Waveshare USB-CAN-A (CH340). Stable name, operator access, and no ModemManager probing.
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="usbcan0", GROUP="dialout", MODE="0660", ENV{ID_MM_DEVICE_IGNORE}="1"
# avrig: CANalyst-II (libusb, userspace). Operator access to the USB device.
SUBSYSTEM=="usb", ATTR{idVendor}=="04d8", ATTR{idProduct}=="0053", GROUP="plugdev", MODE="0660", ENV{ID_MM_DEVICE_IGNORE}="1"
RULES
udevadm control --reload-rules
udevadm trigger --subsystem-match=tty --action=add
udevadm trigger --subsystem-match=usb --attr-match=idVendor=04d8 --action=add
udevadm settle
status
