#!/bin/bash
# Root-side setup for avrig. Each part is optional and reversible; nothing here touches
# rigcap or blucap.
#
#   sudo bash avrig-install.sh --autostart          frame sync at boot (warm start, verified)
#   sudo bash avrig-install.sh --collection-mode    max clocks + pause listed load while recording
#        [--pause-containers "name1 name2"] [--stop-services "ollama"]
#   sudo bash avrig-install.sh --can [--bitrate 500000] [--ifaces "can0 can1"]
#   sudo bash avrig-install.sh --auto               hands-free collection service (avrig auto)
#   sudo bash avrig-install.sh --depot              offload at the depot every 15 min (timer)
#   sudo bash avrig-install.sh --status
#   sudo bash avrig-install.sh --uninstall          remove everything this script installed
#
# GNSS as master clock is a separate step: setup/gnss-clock/install.sh
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"
AVRIG="$(cd "$HERE/.." && pwd)/bin/avrig"
RIG_USER="${SUDO_USER:-blurabbit}"
RIG_HOME="$(getent passwd "$RIG_USER" | cut -d: -f6)"
HELPER=/usr/local/sbin/avrig-collection-mode
CONF=/etc/avrig/collection-mode.conf
SUDOERS=/etc/sudoers.d/avrig-collection-mode
BOOT_UNIT=/etc/systemd/system/avrig-boot.service
CAN_UNIT=/etc/systemd/system/avrig-can.service
AUTO_UNIT=/etc/systemd/system/avrig-auto.service
DEPOT_UNIT=/etc/systemd/system/avrig-depot.service
DEPOT_TIMER=/etc/systemd/system/avrig-depot.timer
NAME_RE='^[A-Za-z0-9_.@-]+$'

do_autostart=0 do_collect=0 do_can=0 do_status=0 do_uninstall=0 do_auto=0 do_depot=0
containers="" services="" bitrate=500000 ifaces="can0 can1"
while [ $# -gt 0 ]; do
    case "$1" in
        --autostart) do_autostart=1 ;;
        --collection-mode) do_collect=1 ;;
        --pause-containers) containers="$2"; shift ;;
        --stop-services) services="$2"; shift ;;
        --can) do_can=1 ;;
        --auto) do_auto=1 ;;
        --depot) do_depot=1 ;;
        --bitrate) bitrate="$2"; shift ;;
        --ifaces) ifaces="$2"; shift ;;
        --status) do_status=1 ;;
        --uninstall) do_uninstall=1 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option $1 (see --help)"; exit 2 ;;
    esac
    shift
done
[ $((do_autostart + do_collect + do_can + do_status + do_uninstall + do_auto + do_depot)) -gt 0 ] || { sed -n '2,15p' "$0"; exit 2; }

check_names() {   # refuse anything that is not a plain unit / container name
    for n in $1; do [[ "$n" =~ $NAME_RE ]] || { echo "invalid name: $n"; exit 2; }; done
}

# ------------------------------------------------------------------ autostart
install_autostart() {
    echo "== autostart: avrig-boot.service (runs as $RIG_USER)"
    if systemctl is-enabled -q blucap-framesync.service 2>/dev/null; then
        echo "   blucap-framesync.service is ENABLED - it sets frame sync COLD, which is the"
        echo "   known boot failure. Disable it first: sudo systemctl disable blucap-framesync"
        exit 1
    fi
    cat > "$BOOT_UNIT" <<EOF
# avrig: hardware frame sync at boot.
# Waits (up to 120 s) for all cameras to enumerate, streams them free-running first
# (warm start - setting the mode cold has left cameras dead), enables frame_sync_mode,
# and verifies the lock. One retry; if it still does not lock, the rig is left
# FREE-RUNNING and preflight blocks recording until an operator runs 'avrig sync on'.
# Every attempt is in ~/.local/state/avrig/audit.log.
[Unit]
Description=avrig: enable and verify hardware frame sync on all cameras
After=systemd-udev-settle.service gpsd.service
Wants=systemd-udev-settle.service
Conflicts=blucap-framesync.service

[Service]
Type=oneshot
User=$RIG_USER
SupplementaryGroups=video
Environment=HOME=$RIG_HOME
ExecStart=$AVRIG -q sync on --boot
TimeoutStartSec=300
Nice=5

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable avrig-boot.service
    echo "   enabled; takes effect at next boot. Result: journalctl -u avrig-boot"
    if systemctl is-active -q blucap-engine.service 2>/dev/null; then
        echo "   note: blucap-engine.service is active. If it opens the cameras at boot,"
        echo "   avrig-boot will report 'busy' - stop / disable it when avrig is used."
    fi
}

# ------------------------------------------------------------ collection mode
install_collect() {
    echo "== collection mode: $HELPER"
    check_names "$containers"
    check_names "$services"
    install -d -m 0755 /etc/avrig
    if [ ! -f "$CONF" ] || [ -n "$containers$services" ]; then
        cat > "$CONF" <<EOF
# avrig collection mode: what is paused while recording (root-owned; the helper only
# ever acts on these names). Docker containers are paused (frozen, state kept), not
# stopped. Services are stopped and started again afterwards.
CONTAINERS="$containers"
SERVICES="$services"
EOF
        chmod 0644 "$CONF"
    fi
    cat > "$HELPER" <<'EOF'
#!/bin/bash
# avrig collection mode helper: on | off | status. Installed by avrig-install.sh.
# on : save clocks + jetson_clocks (max), pause listed containers, stop listed services
# off: restore exactly what 'on' changed (recorded in /run), nothing else
set -u
CONF=/etc/avrig/collection-mode.conf
STATE=/run/avrig-collection-mode
RE='^[A-Za-z0-9_.@-]+$'
CONTAINERS="" SERVICES=""
[ -f "$CONF" ] && . "$CONF"
ok() { [[ "$1" =~ $RE ]]; }
case "${1:-}" in
on)
    if [ -e "$STATE/on" ]; then echo "already on"; exit 0; fi
    mkdir -p "$STATE"; : > "$STATE/paused"; : > "$STATE/stopped"
    jetson_clocks --store "$STATE/clocks.conf" >/dev/null 2>&1 && jetson_clocks >/dev/null 2>&1 \
        && echo "clocks: max" || echo "clocks: jetson_clocks failed"
    for c in $CONTAINERS; do
        ok "$c" || continue
        if [ "$(docker inspect -f '{{.State.Running}} {{.State.Paused}}' "$c" 2>/dev/null)" = "true false" ] \
            && docker pause "$c" >/dev/null 2>&1; then echo "$c" >> "$STATE/paused"; echo "paused $c"; fi
    done
    for s in $SERVICES; do
        ok "$s" || continue
        if systemctl is-active -q "$s" && systemctl stop "$s"; then echo "$s" >> "$STATE/stopped"; echo "stopped $s"; fi
    done
    touch "$STATE/on" ;;
off)
    [ -d "$STATE" ] || { echo "already off"; exit 0; }
    [ -f "$STATE/clocks.conf" ] && jetson_clocks --restore "$STATE/clocks.conf" >/dev/null 2>&1 && echo "clocks: restored"
    while read -r c; do ok "$c" && docker unpause "$c" >/dev/null 2>&1 && echo "unpaused $c"; done < "$STATE/paused"
    while read -r s; do ok "$s" && systemctl start "$s" && echo "started $s"; done < "$STATE/stopped"
    rm -rf "$STATE" ;;
status)
    [ -e "$STATE/on" ] && echo "on" || echo "off"
    echo "containers: ${CONTAINERS:-none}"; echo "services: ${SERVICES:-none}" ;;
*) echo "usage: $0 on|off|status"; exit 2 ;;
esac
EOF
    chmod 0755 "$HELPER"
    chown root:root "$HELPER"
    local tmp
    tmp="$(mktemp)"
    printf '%s ALL=(root) NOPASSWD: %s on, %s off, %s status\n' "$RIG_USER" "$HELPER" "$HELPER" "$HELPER" > "$tmp"
    visudo -cf "$tmp" >/dev/null || { rm -f "$tmp"; echo "sudoers rule invalid - not installed"; exit 1; }
    install -m 0440 "$tmp" "$SUDOERS"
    rm -f "$tmp"
    . "$CONF"
    echo "   pauses containers: ${CONTAINERS:-none}   stops services: ${SERVICES:-none}"
    echo "   edit $CONF (or re-run with --pause-containers / --stop-services) to change"
    echo "   then set 'collection_mode: true' in ~/.config/avrig.yaml"
    echo "   running now, for reference:"
    docker ps --format '      container {{.Names}}' 2>/dev/null || true
    for s in ollama; do systemctl is-active -q "$s" 2>/dev/null && echo "      service   $s"; done
}

# ------------------------------------------------------------------------ CAN
install_can() {
    echo "== CAN: $ifaces at $bitrate bit/s at boot"
    check_names "$ifaces"
    [[ "$bitrate" =~ ^[0-9]+$ ]] || { echo "invalid bitrate"; exit 2; }
    local cmds=""
    for i in $ifaces; do
        cmds+="ExecStart=/usr/sbin/ip link set $i down
ExecStart=/usr/sbin/ip link set $i type can bitrate $bitrate restart-ms 100
ExecStart=/usr/sbin/ip link set $i up
"
    done
    cat > "$CAN_UNIT" <<EOF
# avrig: bring up the Orin CAN interfaces (mttcan) for passive logging.
# restart-ms recovers from bus-off by itself (e.g. vehicle off / no termination).
[Unit]
Description=avrig: CAN interfaces up ($ifaces @ $bitrate)
After=network-pre.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStartPre=/usr/sbin/modprobe mttcan
$cmds
[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now avrig-can.service || echo "   failed - see: journalctl -u avrig-can"
    for i in $ifaces; do echo "   $i: $(cat /sys/class/net/$i/operstate 2>/dev/null || echo missing)"; done
    echo "   then set 'can_ifaces: [${ifaces// /, }]' in ~/.config/avrig.yaml"
}

# ------------------------------------------------------------------ hands-free
install_auto() {
    echo "== hands-free collection: avrig-auto.service (runs as $RIG_USER)"
    if systemctl is-enabled -q avrig-boot.service 2>/dev/null; then :; else
        echo "   note: avrig-boot.service (frame sync at boot) is not enabled - consider --autostart too"
    fi
    cat > "$AUTO_UNIT" <<UNIT
# avrig: hands-free collection. Records while the vehicle drives (GNSS speed), stops when
# parked; preflight still gates every start. The LAN status page is its UI (remote_port).
# While this runs, the GUI cannot open the cameras: systemctl stop avrig-auto first.
[Unit]
Description=avrig: hands-free collection
After=avrig-boot.service gpsd.service network-online.target
Wants=avrig-boot.service

[Service]
Type=simple
User=$RIG_USER
SupplementaryGroups=video dialout gpio
Environment=HOME=$RIG_HOME
ExecStart=$AVRIG auto
Restart=on-failure
RestartSec=10
KillSignal=SIGTERM
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
UNIT
    systemctl daemon-reload
    systemctl enable avrig-auto.service
    echo "   enabled; starts at next boot (or now: sudo systemctl start avrig-auto)"
    echo "   set remote_token (and remote_markers: true for the phone page) in ~/.config/avrig.yaml"
}

install_depot() {
    echo "== depot offload: avrig-depot.timer every 15 min (runs as $RIG_USER)"
    cat > "$DEPOT_UNIT" <<UNIT
# avrig: offload finished sessions when offload_dest is reachable; never while recording.
[Unit]
Description=avrig: depot offload
After=network-online.target remote-fs.target

[Service]
Type=oneshot
User=$RIG_USER
Environment=HOME=$RIG_HOME
ExecStart=$AVRIG -q depot
Nice=15
IOSchedulingClass=idle
UNIT
    cat > "$DEPOT_TIMER" <<UNIT
[Unit]
Description=avrig: depot offload every 15 min

[Timer]
OnBootSec=5min
OnUnitActiveSec=15min

[Install]
WantedBy=timers.target
UNIT
    systemctl daemon-reload
    systemctl enable --now avrig-depot.timer
    echo "   set offload_dest (and depot_prune_days if wanted) in ~/.config/avrig.yaml"
}

# --------------------------------------------------------------------- status
status() {
    echo "avrig-boot.service : $(systemctl is-enabled avrig-boot.service 2>/dev/null || echo not installed)"
    echo "avrig-can.service  : $(systemctl is-enabled avrig-can.service 2>/dev/null || echo not installed)"
    echo "avrig-auto.service : $(systemctl is-enabled avrig-auto.service 2>/dev/null || echo not installed)"
    echo "avrig-depot.timer  : $(systemctl is-enabled avrig-depot.timer 2>/dev/null || echo not installed)"
    echo "collection helper  : $([ -x "$HELPER" ] && "$HELPER" status | head -1 || echo not installed)"
    echo "sudoers rule       : $([ -f "$SUDOERS" ] && echo installed || echo not installed)"
    echo "blucap-framesync   : $(systemctl is-enabled blucap-framesync.service 2>/dev/null || echo -)"
    echo "chrony GNSS refclock: $([ -f /etc/chrony/conf.d/gnss-master.conf ] && echo installed || echo 'not installed (setup/gnss-clock/install.sh)')"
}

uninstall() {
    [ -x "$HELPER" ] && "$HELPER" off || true
    systemctl disable --now avrig-boot.service 2>/dev/null || true
    systemctl disable avrig-can.service 2>/dev/null || true
    systemctl disable --now avrig-auto.service avrig-depot.timer 2>/dev/null || true
    rm -f "$BOOT_UNIT" "$CAN_UNIT" "$HELPER" "$SUDOERS" "$AUTO_UNIT" "$DEPOT_UNIT" "$DEPOT_TIMER"
    systemctl daemon-reload
    echo "removed (kept $CONF; CAN interfaces stay as they are until reboot)"
}

[ $do_uninstall -eq 1 ] && { uninstall; exit 0; }
[ $do_autostart -eq 1 ] && install_autostart
[ $do_collect -eq 1 ] && install_collect
[ $do_can -eq 1 ] && install_can
[ $do_auto -eq 1 ] && install_auto
[ $do_depot -eq 1 ] && install_depot
[ $do_status -eq 1 ] && status
exit 0
