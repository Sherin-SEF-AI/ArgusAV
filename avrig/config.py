"""Rig configuration. Defaults match the target rig; override in ~/.config/avrig.yaml.
Every value is type- and range-checked; an unknown or invalid key is a hard error."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields

import yaml

from .errors import ConfigError

DEFAULT_PATH = "~/.config/avrig.yaml"

# Sensor I2C identity -> (logical id, lens role). The bus selects the deserializer
# (9 = deser 1, 10 = deser 2) and the address selects the port (0x43..0x46 = A..D),
# so this identity survives reboots, link re-trains and /dev/videoN renumbering.
# Lens role is read from the sensor by the driver (FQ = wide, FS = narrow).
# Mounting POSITION in the ids follows the shipped udev names and is unverified.
CAMERAS = {
    "9-0043": ("front_wide", "wide"),
    "9-0044": ("left", "wide"),
    "9-0045": ("right", "wide"),
    "9-0046": ("rear_wide", "wide"),
    "10-0044": ("front_narrow", "narrow"),
    "10-0046": ("rear_narrow", "narrow"),
}

# key -> (min, max) for numeric fields
RANGES = {
    "width": (64, 8192), "height": (64, 8192), "fps": (1, 60), "capture_buffers": (4, 32),
    "bitrate": (1_000_000, 100_000_000), "segment_s": (10, 3600),
    "preview_w": (64, 1920), "preview_h": (64, 1536), "preview_fps": (1, 30),
    "imu_baud": (9600, 4_000_000), "gpsd_port": (1, 65535), "gnss_latency_ms": (-500, 500),
    "expected_cameras": (1, 8), "min_free_gb": (1, 10_000), "stall_s": (0.5, 30),
    "temp_warn_c": (40, 110), "temp_fail_c": (40, 110), "preflight_s": (1, 30),
    "lock_warn_us": (5, 10_000), "lock_fail_us": (5, 10_000), "quality_fps": (0.2, 5),
    "calib_max_age_days": (1, 3650), "calib_move_px": (0.5, 50), "remote_port": (1024, 65535),
    "event_brake_mps2": (1.0, 10.0), "event_accel_mps2": (1.0, 10.0), "event_turn_dps": (5.0, 120.0),
    "event_bump_g": (0.1, 3.0),
    "auto_start_speed_mps": (0.5, 20.0), "auto_start_hold_s": (0.5, 60.0), "auto_stop_speed_mps": (0.05, 5.0),
    "auto_stop_after_s": (10.0, 7200.0), "auto_max_session_s": (60.0, 86400.0), "auto_retry_s": (5.0, 3600.0),
    "depot_prune_days": (0.0, 3650.0),
    "vehicle_ground_z_m": (-3.0, 3.0), "foxglove_port": (1024, 65535), "mount_guard_deg": (0.3, 10.0), "mount_guard_fps": (1.0, 10.0), "mount_guard_rate_hz": (0.5, 20.0),
}
LIST_KEYS = ("marker_labels", "can_ifaces", "fleet_rigs", "library_dirs")
VECTOR_KEYS = {"imu_mount_rpy_deg": 3, "gnss_lever_arm_m": 3, "ref_camera_pose": 6}   # float lists


@dataclass
class Config:
    # capture
    width: int = 1920
    height: int = 1536
    fps: int = 30                       # FSIN readout-trigger sync is supported at 30 only
    capture_buffers: int = 32           # frames the driver holds per camera while the app is busy (32 = 1 s)
    bitrate: int = 20_000_000           # per camera, H.265
    segment_s: int = 300                # keyframe-aligned mkv segments, no frame gaps
    preview_w: int = 320
    preview_h: int = 256
    preview_fps: int = 10
    # sensors
    imu_dev: str = field(default_factory=lambda: "/dev/imu0" if os.path.exists("/dev/imu0")
                         else "/dev/ttyUSB0")   # /dev/imu0 from setup/gnss-clock udev rules
    imu_baud: int = 115200
    gpsd_host: str = "127.0.0.1"
    gpsd_port: int = 2947
    gnss_latency_ms: float = 0.0        # written by `avrig clock --calibrate`
    # storage and supervision
    out_dir: str = "~/avrig-sessions"
    min_free_gb: float = 20.0           # preflight floor; recording auto-stops at half
    stall_s: float = 1.0                # no frames this long while streaming = stalled
    temp_warn_c: float = 85.0
    temp_fail_c: float = 97.0
    preflight_s: float = 3.0            # streaming time before evaluating preflight
    # preflight policy: what blocks a recording (FAIL) rather than warning
    expected_cameras: int = 6
    require_frame_sync: bool = True
    require_imu: bool = True
    require_gnss_fix: bool = False
    require_gnss_clock: bool = False
    # hardware-sync quality: measured lock on this rig is ~30 us median, ~40 us p95
    lock_warn_us: float = 100.0
    lock_fail_us: float = 300.0
    # data quality: 1 Hz grey thumbnails from a side branch; never touches recording
    quality_enable: bool = True
    quality_fps: float = 1.0
    # live camera-mount guard while recording (see mountguard.py): warns when a camera
    # has moved on its mount by more than mount_guard_deg since its learned reference
    mount_guard: bool = True
    mount_guard_deg: float = 1.0
    mount_guard_fps: float = 4.0        # analysis frames per camera (640x512 grey, leaky)
    mount_guard_rate_hz: float = 4.0    # global cap on analysed pairs (CPU bound)
    # privacy: own-vehicle mask for `avrig anonymize` ({camera: [[x, y], ...]} as frame fractions;
    # check it with `avrig egomask SESSION`). Empty = no mask unless --ego-mask is given.
    ego_mask_path: str = ""
    # research library: more folders of sessions to show in the Lab overview, the pipeline and
    # the Python API (e.g. an offload disk, or rigcap's folder). Never pruned or offloaded.
    library_dirs: list = field(default_factory=list)
    # operator workflow
    marker_labels: list = field(default_factory=lambda: ["interesting", "bad data",
                                                         "near miss", "note"])
    presets_path: str = "~/.config/avrig/presets.yaml"
    offload_dest: str = ""              # default destination for `avrig offload`
    # calibration
    calibration_dir: str = "~/.local/share/avrig/calibration"
    calibration_id: str = ""            # the active calibration, recorded in every session
    calib_max_age_days: int = 30
    calib_move_px: float = 2.0          # camera-moved threshold, at 320 px preview width
    # read-only remote status (LAN)
    remote_enable: bool = False
    remote_bind: str = "0.0.0.0"
    remote_port: int = 8088
    remote_token: str = ""
    remote_markers: bool = False        # phone page marker buttons (needs remote_token)
    # Foxglove live view (WebSocket, foxglove.websocket.v1): previews, IMU, GNSS, status;
    # work only for subscribed topics. On the LAN: foxglove_bind 0.0.0.0 + remote_token.
    foxglove_live: bool = False
    foxglove_bind: str = "127.0.0.1"
    foxglove_port: int = 8765
    # collection mode: max clocks + pause listed services while recording. Needs the
    # root helper from setup/avrig-install.sh --collection-mode; off unless enabled.
    collection_mode: bool = False
    # phase 4 (hardware-dependent)
    can_ifaces: list = field(default_factory=list)   # e.g. ["can0"]; bitrate set by setup
    # USB CAN adapter: "" = not used; "auto" = whichever is plugged in; "canalyst" (CANalyst-II);
    # "waveshare" or a serial port ("/dev/usbcan0") for the Waveshare USB-CAN-A.
    # Listen-only by default: the adapter never transmits or acknowledges on the vehicle bus.
    can_usb: str = ""
    can_bitrate: int = 500000
    can_listen_only: bool = True
    can_channels: str = "1"             # CANalyst-II: "1", "2" or "1,2" (CAN1 / CAN2 on the box)
    gnss_timemarks: bool = False        # log UBX-TIM-TM2 (camera trigger on F9P EXTINT)
    # GNSS ground truth: raw receiver log for PPK (gnss_raw.ubx; `avrig gnss --raw-enable`
    # once per power cycle) and live RTK corrections from an NTRIP caster
    gnss_raw: bool = False
    ntrip_url: str = ""                 # ntrip://user:password@caster:2101/MOUNTPOINT
    ntrip_gga: bool = True              # send our position (VRS / nearest-base mount points)
    gnss_device: str = ""               # receiver path for corrections (default: from gpsd)
    # receiver interference watch: jamming / spoofing / antenna state (UBX-MON-RF, SEC-SIG);
    # `avrig gnss --rf-enable` switches the messages on in the receiver
    gnss_interference: bool = True
    # automatic event markers (IMU + GNSS), written to markers.csv with source "auto"
    events_enable: bool = True
    event_brake_mps2: float = 3.0
    event_accel_mps2: float = 2.5
    event_turn_dps: float = 20.0
    event_bump_g: float = 0.45
    # trajectory (offline GNSS + IMU fusion). Empty = estimated from the drive.
    imu_mount_rpy_deg: list = field(default_factory=list)   # IMU body -> vehicle [roll, pitch, yaw]
    gnss_lever_arm_m: list = field(default_factory=list)    # antenna in the vehicle frame [x fwd, y left, z up]
    ref_camera_pose: list = field(default_factory=list)     # reference camera in the vehicle frame
                                                            # [x, y, z, roll, pitch, yaw(deg)], measured
    vehicle_ground_z_m: float = 0.0     # ground height in the vehicle frame (origin on the ground: 0)
    # hands-free collection (`avrig auto`, or the avrig-auto service)
    auto_start_speed_mps: float = 2.5
    auto_start_hold_s: float = 3.0
    auto_stop_speed_mps: float = 0.5
    auto_stop_after_s: float = 300.0
    auto_max_session_s: float = 3600.0
    auto_retry_s: float = 60.0
    auto_label: str = "auto"
    auto_operator: str = ""
    auto_vehicle: str = ""
    auto_status_gpio: str = ""          # optional status LED, e.g. "gpiochip0:106"
    # depot offload (`avrig depot`, timer from avrig-install.sh --depot); 0 = never prune
    depot_prune_days: float = 0.0
    # fleet dashboard: ["name=http://host:8088/?token=SECRET", ...]
    fleet_rigs: list = field(default_factory=list)
    sim: bool = False
    cameras: dict = field(default_factory=lambda: dict(CAMERAS))

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        cfg = cls()
        p = os.path.expanduser(path or DEFAULT_PATH)
        if path and not os.path.exists(p):
            raise ConfigError(f"config file not found: {p}")
        if os.path.exists(p):
            try:
                with open(p) as f:
                    data = yaml.safe_load(f) or {}
            except (OSError, yaml.YAMLError) as e:
                raise ConfigError(f"cannot read {p}: {e}") from None
            if not isinstance(data, dict):
                raise ConfigError(f"{p}: top level must be a mapping")
            cfg.update(data, source=p)
        for key in ("out_dir", "presets_path", "calibration_dir", "ego_mask_path"):
            setattr(cfg, key, os.path.expanduser(getattr(cfg, key)))
        cfg.validate()
        return cfg

    def update(self, data: dict, source: str = "config"):
        types = {f.name: f.type for f in fields(self)}
        for key, val in data.items():
            if key not in types:
                raise ConfigError(f"{source}: unknown key {key!r}",
                                  "check spelling; unknown keys are never silently ignored")
            if key == "cameras":
                val = self._cameras(val, source)
            elif key in VECTOR_KEYS:
                n = VECTOR_KEYS[key]
                if not (isinstance(val, list) and (len(val) in (0, n)) and
                        all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in val)):
                    raise ConfigError(f"{source}: {key} must be a list of {n} numbers (or empty)")
                val = [float(x) for x in val]
            elif key in LIST_KEYS:
                if not (isinstance(val, list) and all(isinstance(x, str) and x.strip() for x in val)):
                    raise ConfigError(f"{source}: {key} must be a list of non-empty strings")
            else:
                want = type(getattr(self, key))
                if want is float and isinstance(val, int) and not isinstance(val, bool):
                    val = float(val)
                if not isinstance(val, want) or (want is int and isinstance(val, bool)):
                    raise ConfigError(f"{source}: {key} must be {want.__name__}, "
                                      f"got {type(val).__name__} {val!r}")
            setattr(self, key, val)

    @staticmethod
    def _cameras(val, source) -> dict:
        if not isinstance(val, dict) or not val:
            raise ConfigError(f"{source}: cameras must be a non-empty mapping")
        out = {}
        for k, v in val.items():
            if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, str) for x in v)):
                raise ConfigError(f"{source}: cameras[{k!r}] must be [id, role]")
            out[str(k)] = tuple(v)
        ids = [v[0] for v in out.values()]
        if len(set(ids)) != len(ids):
            raise ConfigError(f"{source}: camera ids must be unique")
        return out

    def validate(self):
        for key, (lo, hi) in RANGES.items():
            v = getattr(self, key)
            if not lo <= v <= hi:
                raise ConfigError(f"{key}={v} out of range [{lo}, {hi}]")
        if self.temp_warn_c >= self.temp_fail_c:
            raise ConfigError("temp_warn_c must be below temp_fail_c")
        if self.width % 2 or self.height % 2:
            raise ConfigError("width and height must be even")
        if self.lock_warn_us >= self.lock_fail_us:
            raise ConfigError("lock_warn_us must be below lock_fail_us")
        if not 1 <= len(self.marker_labels) <= 9 or len(set(self.marker_labels)) != len(self.marker_labels):
            raise ConfigError("marker_labels: 1..9 unique labels (keys 1..9 / F1..F9)")
        import re
        bad = [i for i in self.can_ifaces if not re.fullmatch(r"v?can[0-9]+", i)]
        if bad:
            raise ConfigError(f"can_ifaces: invalid interface name(s) {bad}")
        if self.can_usb:
            from .can import usb_bitrates
            ok = usb_bitrates(self.can_usb)
            if self.can_bitrate not in ok:
                raise ConfigError(f"can_bitrate {self.can_bitrate}: one of {sorted(ok)}")
            if not re.fullmatch(r"[12](,[12])?", self.can_channels.replace(" ", "")):
                raise ConfigError(f"can_channels {self.can_channels!r}: \"1\", \"2\" or \"1,2\"")
            if self.can_ifaces:
                raise ConfigError("use either can_ifaces (SocketCAN) or can_usb (USB-CAN-A), not both")
        if self.ntrip_url:
            from .ntrip import parse_url
            parse_url(self.ntrip_url)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["cameras"] = {k: list(v) for k, v in self.cameras.items()}
        return d
