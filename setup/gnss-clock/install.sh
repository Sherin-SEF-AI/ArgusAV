#!/bin/bash
# Make the u-blox NEO-F9P the MASTER CLOCK of this host, from its time messages over
# USB - no PPS. gpsd owns the receiver and publishes every epoch to chrony's SHM 0
# refclock; chrony disciplines CLOCK_REALTIME to it. Network NTP remains only as a
# fallback for when the receiver has no fix.
#
#   sudo bash install.sh              install, then calibrate if the receiver has a fix
#   sudo bash install.sh calibrate    (re)measure the latency offset and promote GNSS
#   sudo bash install.sh calibrate 0.078   promote GNSS with a known offset (no measuring)
#   sudo bash install.sh uninstall    remove the GNSS refclock and restore gpsd defaults
#
# WHY CALIBRATION: without PPS the time arrives inside a USB message, some tens of ms
# after the epoch it describes. That latency is roughly constant, so it is measured
# against the current (NTP-disciplined) clock with GNSS as a monitor-only source, and
# written into the refclock "offset". Only then is GNSS promoted to master.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
CONF=/etc/chrony/conf.d/gnss-master.conf
SECS="${CALIBRATE_SECONDS:-180}"

refclock() {  # $1 = offset seconds, $2 = extra flags
    cat > "$CONF" <<CONFEOF
# GNSS master clock (u-blox NEO-F9P via gpsd SHM 0, no PPS). Managed by
# DataCollectionRig/setup/gnss-clock/install.sh - re-run "calibrate" to update.
refclock SHM 0 refid GNSS poll 2 filter 8 precision 1e-3 delay 0.1 offset $1 $2
CONFEOF
    systemctl restart chrony
}

gnss_fix() {   # true if gpsd reports a 2D/3D fix within 10 s
    # Capture first, then match: under pipefail, "gpspipe | grep -q" fails whenever there
    # IS a fix, because grep exits at the first match and gpspipe then dies of SIGPIPE.
    # gpsd may also have just been restarted, so give it the full 10 s to report.
    local out
    out="$(timeout 10 gpspipe -w 2>/dev/null || true)"
    grep -q '"class":"TPV".*"mode":[23]' <<<"$out"
}

calibrate() {   # $1 (optional) = a known latency offset in seconds: promote it, no measuring
    if ! gnss_fix; then
        echo "!! no GNSS fix - put the antenna under open sky, then: sudo bash $0 calibrate"
        return 1
    fi
    if [ -n "${1:-}" ]; then
        awk -v o="$1" 'BEGIN{exit !(o ~ /^-?[0-9]*\.?[0-9]+$/ && o > -0.5 && o < 0.5 && o != 0)}' \
            || { echo "!! offset must be seconds, e.g. 0.078191452"; return 1; }
        echo "-> promoting GNSS to master with the given offset ${1}s"
        refclock "$1" "prefer trust"
        sleep 30
        chronyc sources | sed -n '1,2p;/GNSS/p'
        chronyc tracking | sed -n '1,4p'
        return 0
    fi
    # The measurement is only as good as the clock it is measured against. Keep the
    # current setting so a failed or rejected run does not demote a working GNSS master.
    local prev
    prev="$(cat "$CONF" 2>/dev/null || true)"
    restore() {
        if [ -n "$prev" ]; then printf '%s\n' "$prev" > "$CONF"; systemctl restart chrony
            echo "   previous clock setting restored"; fi
    }
    echo "-> measuring GNSS latency against the current clock for ${SECS}s (GNSS monitor-only)"
    refclock 0.0 noselect
    sleep "$SECS"
    # A mobile hotspot can move network time by 100+ ms; calibrating against that bakes the
    # error into the GNSS clock. chrony's RMS offset says how steady the network clock was.
    rms=$(chronyc -c tracking | cut -d, -f7)
    if awk -v r="${rms:-1}" 'BEGIN{r=(r<0?-r:r); exit !(r > 0.005)}'; then
        echo "!! network time is too unsteady to calibrate against (RMS offset ${rms}s, limit 0.005s)"
        echo "   use a steady network (wired or office WiFi), or re-apply a known offset:"
        echo "   sudo bash $0 calibrate <offset-seconds>"
        restore; return 1
    fi
    row=$(chronyc -c sourcestats | awk -F, '$1=="GNSS"')
    [ -n "$row" ] || { echo "!! chrony has no GNSS samples - is gpsd feeding SHM 0?"; restore; return 1; }
    off=$(echo "$row" | cut -d, -f7); sd=$(echo "$row" | cut -d, -f8); np=$(echo "$row" | cut -d, -f2)
    echo "   GNSS vs clock: offset ${off}s  std-dev ${sd}s  over ${np} samples"
    if awk -v n="$np" 'BEGIN{exit !(n < 8)}'; then echo "!! too few samples"; restore; return 1; fi
    echo "-> promoting GNSS to master with offset ${off}s"
    refclock "$off" "prefer trust"
    # Once GNSS is the reference its own offset reads ~0 whatever the sign, so verify
    # against the network sources: with the right sign they agree within a few ms;
    # with the wrong one they sit ~2x the latency away. Flip it if so.
    sleep 90
    ntp=$(chronyc -c sources | awk -F, '$1=="^"{print ($8<0?-$8:$8)}' | sort -g | awk '{a[NR]=$1}END{if(NR)print a[int((NR+1)/2)]; else print ""}')
    lim=$(awk -v o="$off" 'BEGIN{o=(o<0?-o:o); print (o>0.005?o:0.005)}')
    if [ -n "$ntp" ] && awk -v n="$ntp" -v l="$lim" 'BEGIN{exit !(n > l)}'; then
        neg=$(awk -v o="$off" 'BEGIN{printf "%.9f", -o}')
        echo "   NTP sources now ${ntp}s away - offset sign was inverted, using ${neg}s"
        refclock "$neg" "prefer trust"
        sleep 60
    fi
    chronyc sources | sed -n '1,2p;/GNSS/p'
    chronyc tracking | sed -n '1,4p'
}

case "${1:-install}" in
install)
    install -m 0644 "$HERE/99-rig-serial.rules" /etc/udev/rules.d/99-rig-serial.rules
    udevadm control --reload-rules && udevadm trigger --subsystem-match=tty --action=add && udevadm settle
    [ -e /dev/gnss0 ] || { echo "!! /dev/gnss0 missing - is the receiver plugged in?"; exit 1; }
    cp -a /etc/default/gpsd "/etc/default/gpsd.bak.$(date +%Y%m%d-%H%M%S)"
    cat > /etc/default/gpsd <<'GEOF'
# gpsd polls the receiver continuously (-n) so chrony's SHM refclock is fed even with
# no client attached. Managed by DataCollectionRig/setup/gnss-clock/install.sh.
DEVICES="/dev/gnss0"
GPSD_OPTIONS="-n"
USBAUTO="false"
GEOF
    systemctl enable gpsd.service gpsd.socket
    systemctl restart gpsd.socket gpsd.service
    refclock 0.0 noselect
    echo "-> gpsd: $(systemctl is-active gpsd)  chrony GNSS refclock installed (monitor-only)"
    calibrate || echo "   GNSS stays monitor-only until calibrated; NTP remains the reference"
    ;;
calibrate) calibrate "${2:-}" ;;
uninstall)
    rm -f "$CONF" /etc/udev/rules.d/99-rig-serial.rules
    last=$(ls -t /etc/default/gpsd.bak.* 2>/dev/null | head -1 || true)
    [ -n "$last" ] && cp -a "$last" /etc/default/gpsd
    systemctl restart chrony gpsd.socket gpsd.service
    echo "removed GNSS master clock"
    ;;
*) echo "usage: $0 [install|calibrate|uninstall]"; exit 2 ;;
esac
