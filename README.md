<p align="center"><img src="avrig/assets/argus.svg" width="96" alt="Argus icon"></p>

# Argus

**Documentation website: https://sherin-sef-ai.github.io/ArgusAV/**

Argus is the data-collection application for the six-camera GMSL2 rig on the Jetson AGX Orin 64 GB. It records cameras with hardware frame sync, plus the IMU and GNSS. It is built for a data-collection team and an AV lab: operator workflow, QA, offload, export and replay, plus ground truth (raw GNSS, RTK, a smoothed 6-DoF trajectory, visual speed through GNSS outages), a calibration studio, live guards (camera-mount and GNSS interference), automatic event markers, on-device tagging, search by meaning, collection curation, 3D pre-labels with a CVAT review loop, map layers across drives, privacy blurring, dataset and reconstruction exporters, Foxglove live view and replay, and fleet operation (hands-free recording, depot offload, a fleet dashboard).

The command, Python package, config file and data folders are named `avrig`; Argus is the display name. avrig is a separate application, derived from **rigcap 2.0**, an earlier recorder for the same rig, which is kept unchanged. Both use the same capture core: direct V4L2 capture with driver (SoC TSC) timestamps, NVENC H.265 and per-camera supervision. Do not run both at once, because a camera can only be opened by one process.

## Measured on this rig (2026-11-12)

The rig was on the bench. Sync was the deserializer's internal trigger (SW2 = internal). The GNSS antenna was indoors (no fix).

| What | Result |
|---|---|
| `avrig check` (15 s) | 29.99 fps on all six, jitter 16–17 µs, spread **median 29 µs / p95 36 µs**, LOCKED, 0 drops |
| 61 s recording + `verify` | frames.csv == video frames on all six, **0 drops**, alignment **29 µs median / 38 µs p95** |
| 4 min soak | LOCKED for 234 of 235 s, 1 s spread median 29 µs (worst 37 µs), 0 drops, 0 restarts, CPU 28 %, 54 → 57 °C |
| CPU, recording 6 cameras (30 s) | avrig **180 %** of one core, rigcap **177 %** (out of 12 cores): the added features cost ~3 % of one core |
| Stall recovery (3 injected stalls) | frames back **2.1–2.4 s** after the last good frame (rigcap: 4.6 s); the other five cameras untouched; re-locked at ~29 µs |
| Camera-moved check, real scene | 0.0 px on all six unchanged views; a simulated 6 px bump was reported as moved at 5.98 px |
| Replay of the real session | seek ~0.5 s; all six views within ±30 µs of each other; 1× playback keeps up |
| MCAP export of a 5.6 min session | 27 s; 5.03 GB for 5.02 GB of video (no base64 overhead); opens and seeks from the index in ~0.02 s; video byte-identical to the recording |
| New live features (automatic events + raw GNSS log), 2 × 30 s A/B | **165 % → 166 %** of one core: ~1 % of one core; recording LOCKED ~28 µs, 0 drops |
| Object tagging, 5.6 min × 6 cameras at 2 fps | 118 s (3× faster than real time), TensorRT FP16 on the GPU |
| Privacy blurring, 61 s × 6 cameras (10,876 frames) | 494 s (22 fps, 3 workers); every frame kept (verify: frames.csv == video on all six) |
| Detector latency on the Orin GPU | objects (YOLOX-s) 3.2 ms, plates 5.5 ms, faces ~3 ms per 640 px inference |
| 10 s MCAP clip from a search hit | 6.8 s; 315 frames per camera (300 + lead-in from the keyframe before); decodes on NVDEC from the first frame |
| Camera-mount guard, 4 × 60 s A/B (2026-11-13) | **+7–8 % of one core** (the 640×512 frame branch; the guard itself 0.0 % while parked); LOCKED 58/58 s, **0 drops** |
| Foxglove live view, 60 s recording with a viewer on every topic | previews ~5 fps × 6, IMU 100 Hz, status 1 Hz; recorder CPU unchanged (164 %), LOCKED, 0 drops |
| Search embeddings (CLIP, keyframes only) | 61 s session: 4 s (cached) / 21 s (cold disk); 5.6 min session: 91 s cold |
| 3D pre-labels, 61 s × 6 cameras at 10 Hz | 33 s (3 camera workers, one GPU lock); 11 people tracked in the office footage |
| Offline mount check, 61 s × 6 cameras | 25 s (parallel NVDEC) |
| Model latency (TensorRT FP16) | depth (Depth Anything V2 S, 518×420) 5.6 ms/frame in batches of 6; CLIP image 0.7 ms/frame; YOLOX 3.4 ms |
| GNSS interference on this receiver (NEO-F9P, HPGL1L5 1.40, protocol 27.40) | MON-RF (indicator, antenna, AGC) and NAV-STATUS (spoofing) work; the jamming detector (CFG-ITFM) and SEC-SIG are not in this firmware, so the jamming *state* reads "unknown" |
| Tests | **102 passed** (unit + simulated cameras on real GStreamer/NVENC + GPU) |

Accuracy on simulated data with exact truth (no real drive or printed board yet):

| What | Result |
|---|---|
| Trajectory, 200 s drive, 1 Hz GNSS with 1.5 m RMS error | horizontal **0.37–0.42 m RMS**, vehicle yaw **0.06–0.08°** median, roll/pitch ~0.2°; the reported 1-sigma matched the real error |
| IMU mounting estimated from the drive (4 mountings incl. 90°, -150° yaw) | yaw within **0.1°** |
| Camera calibration, pinhole + fisheye rendered at known poses | pinhole focal length within 0.03 %, centre within 0.6 px; fisheye model within 0.33 px median over the whole image |
| Camera-to-camera extrinsics (pinhole + fisheye, 35° apart) | **0.03°** and **1.3 mm** |
| Mount guard, three cameras (wide, side, narrow rear), IMU mounted upside down | learned up / forward directions within 1°; a **2.5° knock** on one camera flagged (1.5–3.5° measured), the other cameras stayed "ok", no false alarm while evidence accumulated |
| Visual speed on a rendered road, neighbouring frames | **±1.5 %** from 5 to 45 m/s (162 km/h) with the road-plane warp; yaw rate within 0.3°/s |
| 120 s GNSS outage (1.27 km) with a drifting accelerometer bias | max position error **5.2 m (IMU) → 0.57 m (IMU + visual speed)**; the 2 % speed-scale error recovered from GNSS within 0.07 % |
| 3D lift from exact boxes (3 calibrated cameras, cars at 5–60 m) | median **0.2 m** front cameras, 0.5 m side camera (cars seen at an angle); 4 objects, merged across two overlapping cameras, moving car's speed within 1 m/s |
| Map layers | rough road ranked > 3× smooth (braking excluded); pothole found at its second; landmarks triangulated within 0.6 m; a removed sign reported after 2 passes |

## Start

```bash
~/DataCollectionRig/avrig/bin/avrig --help      # also linked as ~/.local/bin/avrig
avrig check                                     # stream 10 s: sync, drops, sensors, health
avrig gui                                       # operator application
bash setup/install-desktop.sh                   # Argus desktop shortcut + app-menu entry (no root)
avrig record -t 600 --operator Ann --vehicle MG-07 --route "test loop" --weather clear --light day
```

Configuration lives in `~/.config/avrig.yaml`. Every key is validated, and unknown keys are rejected. For the full list with defaults and comments, see `avrig/config.py`. Sessions are written to `~/avrig-sessions/` by default.

## Operator workflow

1. **Before the drive:** run `avrig gui` (or `avrig check`). The chip bar shows SYNC, CAMS, QUALITY, IMU, GNSS, CLOCK, REC, DISK and ISSUES. A preflight with any **FAIL** blocks the recording; for example frame sync off, a missing camera, low disk or overheating. Recording past it needs an explicit force, and the session is then marked as forced.
2. **Metadata:** set operator, vehicle, route, weather, light, road, tags and notes with *Info…* (GUI) or the `record` flags. The GUI requires an operator and a vehicle. The values are stored in `meta.json` and in the manifest. Pick lists come from `~/.config/avrig/presets.yaml` (`operators`, `vehicles`, `routes`, `weather`, `light`, `road`, `tags`).
3. **While recording:** add event markers with keys **1–9 / F1–F9** or the marker buttons. On the CLI, press Enter for the first label, a digit for a label, or type text to add a note. Markers go to `markers.csv` with the exact capture-clock time.
4. **After the drive:** use *Sessions* (GUI) or the CLI to verify, open the HTML report, replay, export, offload and prune. These jobs run at low CPU and I/O priority and are **refused while recording**.

## Commands

| Command | Purpose |
|---|---|
| `status`, `check`, `monitor` | Health without streaming / with streaming / live terminal dashboard (`--remote`) |
| `sync on\|off\|status` | Hardware frame sync. `on` streams the cameras free-running first (warm start), then enables sync and verifies the lock. `--boot` is for autostart. |
| `record` | Preflight-gated recording, with metadata flags and stdin markers |
| `verify SESSION… [--report]` | Offline QA: frame counts vs video, drops, alignment, IMU/GNSS/clock, quality, markers. Writes `verify.json`. |
| `report SESSION` | Self-contained `report.html`: sync, offsets, host load, temperatures, quality timeline, route, markers |
| `sessions` | Index of sessions: duration, size, status, verify result, offloaded, marker count |
| `offload SESSION… --dest DIR` | Copy with SHA-256 while copying, re-hash on read-back, and write a receipt on both sides. Resumable. |
| `prune --older-than DAYS [--yes]` | Deletes only sessions with a verified offload whose copy still exists and re-hashes correctly. Dry run by default. |
| `export mcap SESSION` | Indexed MCAP (chunk + message indexes, CRCs), ROS 2 CDR encoding: `foxglove_msgs/CompressedVideo` (H.265, copied verbatim), `sensor_msgs/Imu`, `sensor_msgs/NavSatFix`, `foxglove_msgs/Log` markers, session metadata. UTC log times. Opens in Foxglove; `ros2 bag play` also needs `foxglove_msgs` installed for the video topics. |
| `export frames SESSION --markers\|--every S\|--at NS` | Timestamp-matched frame sets from all cameras, with `index.csv` (capture time, offset from the target) |
| `soak 2h` | Long recording + verify + report, with a thermal / load / sync trend |
| `calib register\|list\|reference\|check` | Calibration registry (immutable, hashed) and the camera-moved check |
| `wizard` | Camera-position verification (cover / uncover one lens at a time); writes udev rules for `/dev/cam_<position>` |
| `perf on\|off\|status` | Collection mode (below) |
| `clock [--calibrate]` | GNSS master-clock state; calibrates the GNSS message latency |
| `gnss`, `imu`, `can` | Raw GNSS (`--raw-enable`), RTK caster test (`--ntrip`), time marks, IMU SyncIn, CAN (below) |
| `events SESSION… [--list] [--write]` | Automatic events from IMU + GNSS (re-scan / retune old sessions) |
| `trends [--html]` | Rig health across sessions: sync, drops, restarts, offsets, temperature |
| `catalog`, `search QUERY [--export DIR]`, `coverage` | Scenario catalog, search (below) with clip export, coverage map |
| `calibrate board\|solve\|kalibr` | Calibration studio: printable ChArUco board, intrinsics + extrinsics from a recording, Kalibr export |
| `trajectory SESSION…` | 6-DoF vehicle trajectory (GNSS + IMU, smoothed) -> `trajectory.csv` |
| `models [--fetch] [--build]` | Download the perception models (checksums pinned; ~750 MB) and compile TensorRT engines (~30 min once) |
| `tag SESSION…` | Objects in every camera (2 fps): `tags.csv` for search, `detections.csv` pre-annotations |
| `anonymize SESSION` | Copy `<session>_anon` with faces and number plates blurred; same frames |
| `dataset SESSION --format nuscenes\|kitti [--annotate]` | Dataset with ego poses, camera calibration and COCO pre-annotations |
| `retain SESSION [--replace --yes]` | Event-only copy `<session>_events` (video around events), hard-linked |
| `ledsync --gpio LINE` / `--analyze SESSION` | Optical sync check with an LED (records, then measures each camera's exposure window) |
| `auto` | Hands-free collection (normally the `avrig-auto` service) |
| `depot` | Offload finished sessions when the destination is reachable (normally a timer) |
| `fleet --rigs name=URL,...` | Dashboard of several rigs' status pages |
| `mount [status\|reset\|check SESSION]` | Camera-mount guard: learned references, forget one after a re-mount, run the guard over a recorded drive |
| `gnss --rf-enable` | Receiver interference reports (MON-RF, NAV-STATUS; RAM layer, `--persist` to keep) |
| `embed SESSION…` | CLIP embeddings + zero-shot scene labels (lighting, weather, road, traffic) of one keyframe per second per camera |
| `find "TEXT" \| --like S:CAM:NS \| --image FILE` | Frames by meaning, across every embedded session |
| `rare [--html]`, `curate --budget N` | Rarest frames, visual clusters and scene-condition hours; a diverse, rarity-weighted keyframe selection (`dataset --select`) |
| `label3d SESSION…` | 3D pre-labels: tracked 3D boxes of road users -> `labels3d.csv` (nuScenes annotations via `dataset`) |
| `cvat export\|import` | Review the 2D tracks in CVAT; reviewed tracks drive the next `label3d` |
| `trajectory --visual` | Trajectory with visual speed from the road (GNSS outages) |
| `maps update\|html\|changes` | Road roughness, potholes, traffic lights / stop signs and their changes across drives |
| `recon SESSION` | Nerfstudio dataset (poses, intrinsics, moving-object masks) for neural reconstruction |
| `foxglove SESSION [--speed --start --loop]` | Replay a session live into Foxglove over WebSocket (no export) |

## Session folder

```
<YYYYMMDD-HHMMSS>_<label>/
  manifest.json      status, config, timing model, per-camera stats, preflight, meta, markers,
                     calibration id, sync verdict seconds (checkpointed during recording)
  meta.json          operator / vehicle / route / conditions / tags / notes
  <camera>/          <camera>_rNN_NNN.mkv segments + frames.csv (frame,mono_ns,tsc_ns,seq)
  segments.csv       segment file -> first/last frame time
  imu.csv  gnss.csv  clock.csv (CLOCK_MONOTONIC <-> MONOTONIC_RAW <-> REALTIME every 2 s)
  markers.csv        operator markers
  sync.csv           1 Hz: verdict, spread median/p95/max, per-camera fps and offset
  host.csv           5 s: CPU, GPU, memory, load, temperatures, disk free, write rate
  quality.csv        ~1 Hz per camera: brightness, clipping, sharpness, flat area, flags
  can.csv  timemarks.csv   only when enabled
  gnss_raw.ubx       complete receiver log incl. RXM-RAWX/SFRBX, for PPK (gnss_raw: true)
  gnss_rf.csv        1 Hz receiver interference / antenna state (gnss_interference + --rf-enable)
  events.log  verify.json  report.html  SHA256SUMS  offload.json
  -- written by the lab tools after the drive --
  auto_events.csv    automatic events re-computed by `avrig events --write`
  trajectory.csv/.json   6-DoF pose per IMU sample (`avrig trajectory`)
  tags.csv  detections.csv   objects per camera (`avrig tag`)
  ledsync.csv        LED flash times (`avrig ledsync`)
  embeddings.npy/.csv  scene.csv   search embeddings + zero-shot scene labels (`avrig embed`)
  tracks2d.csv  labels3d.csv/.json   2D tracks and 3D pre-labels (`avrig label3d`)
  reviewed_tracks.csv  CVAT-reviewed tracks (`avrig cvat import`)
  visual_speed.csv   road speed from a forward camera (`avrig trajectory --visual`)
<session>_anon/      blurred copy (`avrig anonymize`);  <session>_events/  event-only copy (`avrig retain`)
```

Files added after an offload make `prune` refuse that session until it is offloaded again (by design).

Timing: `frames.csv` is the timing authority. Matroska rounds timestamps to 1 ms. `manifest.json → timebase` documents the mapping.

## Features and their cost

The capture path is the same as in rigcap. Everything added runs off the capture threads, at low rate, or offline:

- **Sync log and alarm.** The verdict is LOCKED below 100 µs (`lock_warn_us`), DEGRADED below 300 µs (`lock_fail_us`), then PARTIAL / FREE-RUN. DEGRADED is a WARN; a lost lock with sync on is a FAIL. `sync.csv` records the lock every second.
- **Faster recovery.** `stall_s` is 1.0 s (was 2.0). A stalled camera is restarted alone, with back-off.
- **Image quality.** One 160×128 grey thumbnail per camera per second, taken from a leaky side branch (<1 ms of numpy). It flags black, dark, over-exposed, blocked (large featureless area) and blurred images (relative to that camera's own recent sharpness). A flag must persist 3 samples. Flags are advisory (WARN).
- **Calibration link.** `calib register FILE --name N --activate` stores an immutable, hashed copy, and every session records the active id. Health warns when there is no calibration, it is older than `calib_max_age_days`, or the last check found a moved camera. `calib reference` (in the calibration bay, right after calibrating) and `calib check` (same bay, later) compare ORB features to detect a camera that moved or rotated. This is deliberately an explicit check, not a live alarm while driving.
- **Remote status / phone page** (`--remote`, or `remote_enable`): HTTP on the LAN. Pages: `/`, `/api/status`, `/api/health`, `/thumb/<cam>.jpg`. Thumbnails are at most 1 per second and the status snapshot at most 1 per second. Set `remote_token` to require `?token=` or the `X-Avrig-Token` header. With `remote_markers: true` **and** a token, the page shows big marker buttons for a co-driver (`POST /api/mark`, configured labels only). There are no other write endpoints: nothing remote can start, stop or reconfigure the rig.
- **Audit log.** `~/.local/state/avrig/audit.log` gets one JSON line per operator or lifecycle action: record start/stop/refused, sync, offload, prune, calibration, wizard.
- **Replay** (GUI → Sessions → Replay). Up to six hardware decoders at preview size, all driven by one clock. Each view shows the last frame captured at or before the shared time.

- **Automatic events** (`events_enable`, default on): hard brake / acceleration (GNSS speed change), sharp turn (yaw rate), bump / pothole (vertical shock), stop / start, GNSS lost / regained, RTK fixed / lost. Detected live from the IMU and GNSS samples (a deque append on the sensor threads, analysis by the supervisor at 2 Hz) and written to `markers.csv` with source `auto` at the event's own time. No mounting calibration needed. Thresholds: `event_brake_mps2`, `event_accel_mps2`, `event_turn_dps`, `event_bump_g`.
- **Raw GNSS log** (`gnss_raw: true`): every UBX frame from the F9P through gpsd's raw watch into `gnss_raw.ubx` (~2–5 KB/s). With `avrig gnss --raw-enable` it contains carrier phase and pseudoranges (RXM-RAWX) and navigation data (RXM-SFRBX): post-process with a base station's data in RTKLIB / Emlid Studio for cm-level positions.
- **Live RTK** (`ntrip_url: ntrip://user:pass@caster:2101/MOUNT`): RTCM 3 from an NTRIP caster, CRC-checked, written to the receiver **through gpsd** (`?DEVICE` hexdata, so gpsd stays the only owner of the port); GGA sent back for VRS mount points. The GNSS chip shows RTK / RTKf, `gnss.csv` records the fix status. Corrections improve position, not timing.

- **Camera-mount guard** (`mount_guard`, default on; warning above `mount_guard_deg`, 1°). While driving, each camera measures two directions that belong to the vehicle: the axis it turns about when the vehicle turns (checked against the gyro), and the direction it moves on straight road. Together they fix the camera's orientation in the vehicle. The first drive with enough turns and straight road learns a reference per camera (`~/.local/share/avrig/mount_ref.json`); later drives are compared with it, and a significant change (beyond 2 standard errors, three evaluations in a row) raises an automatic `camera_moved` marker, a WARN and the MOUNT chip. Frames come from a leaky 640×512 grey branch at 4 Hz, rate-limited before its queue; a pair is analysed only while turning, or every 4 s on straight road at a known GNSS speed; parked or with no speed and no rotation, frames are not even copied. Measured: +7–8 % of one core, 0 drops. A valve to stop the branch while parked was tried and stalled the pipeline when closed; it is not used.
- **GNSS interference watch** (`gnss_interference`, default on, after `avrig gnss --rf-enable`): MON-RF jamming indicator, antenna open / short and AGC, NAV-STATUS spoofing state -> RF chip, health WARNs, `gnss_rf.csv`.
- **Foxglove live view** (`foxglove_live: true`): previews, IMU, GNSS fix and rig status over the Foxglove WebSocket protocol (`ws://localhost:8765`; LAN: `foxglove_bind: 0.0.0.0` + `remote_token`, `?token=`). Work only for subscribed topics; each viewer has a bounded queue (a slow viewer loses messages, the rig never waits). The FOX chip shows the viewers.

Nothing heavy runs during a recording. Neural networks (tagging, blurring, embeddings, 3D labels), decoding and dataset work run after the drive, at the lowest CPU / IO priority, refuse to start while recording and stop themselves if one starts. On the Jetson, CPU code must not touch CUDA unified memory while any kernel of the same process runs, so every TensorRT call goes through one GPU lock (`Engine.infer`).

## AV-lab tools

All are in the GUI (**Lab** button: Explore / Scenarios / Collection / Labels & maps / Rig health / Calibration / Processing) and the CLI. Jobs show a progress bar, elapsed time and a Stop button.

- **Search by meaning.** `avrig embed` (after the drive; ~1/30 of the video is decoded: only keyframes) gives every second of every camera a CLIP ViT-B/32 embedding (MIT) and zero-shot scene labels. Then: free text (`avrig find "cyclist crossing at night"`, or Lab -> Explore with thumbnails; double-click replays the moment), "more like this", search with an image, and catalog terms (`scene:rain`, `"wet road" night person>=2`). Results are de-duplicated (a static view seen in many sessions shows once). CLIP B/32 is coarse: it finds scene types and objects well, fine attributes (a shirt pattern) less so.
- **Collection and curation.** `avrig rare [--html]` / Lab -> Collection: hours per scene condition (what is missing), the rarest frames (1 - similarity to their 10 nearest frames outside their own 30 s, as a percentile), visual clusters smallest first. `avrig curate --budget N`: a diverse selection (farthest-point sampling, rarity-weighted) -> `avrig dataset SESSION --select selection.csv`.
- **3D pre-labels.** `avrig label3d SESSION`: YOLOX at 10 Hz in every camera on a common time grid, SORT-style tracking per camera, each box lifted to the vehicle frame where its bottom edge meets the ground (calibrated camera height and angle; the centre is placed where the ray enters a road-aligned footprint), or by the depth model (Depth Anything V2 Small, Apache-2.0, scaled to metres by the ground in the same frame) when the box is cut by the image edge; observations moved to the world with the trajectory, merged across cameras, smoothed; heading from motion. -> `labels3d.csv` at 2 Hz, and nuScenes `sample_annotation` / `instance` / `category` / `attribute` tables in `avrig dataset` (marked `prelabel`). Lab -> Labels & maps shows them top-down with a time slider. Review loop: `avrig cvat export SESSION --cam X` (CVAT for video 1.1), correct in CVAT, `avrig cvat import SESSION FOLDER`, re-run `label3d`. These are monocular pre-labels (no LiDAR): they save labelling time, they are not ground truth. Without a calibration, `--nominal-geometry` gives rough labels marked as such.
- **Visual speed (GNSS outages).** `avrig trajectory SESSION --visual`: road features 3–15 m ahead in a forward camera, tracked between neighbouring frames; the next frame is first warped onto the current one along the road plane for the previous motion, so tracking holds at any speed; a RANSAC 2D rigid fit of the ground points gives speed and yaw rate. The speed's scale is calibrated against GNSS where both exist, then the filter uses it as a speed measurement: through tunnels and under flyovers the error grows with distance, not with time cubed.
- **Map layers.** `avrig maps update` (incremental; `~/.local/share/avrig/maps.db`), `avrig maps html`, `avrig maps changes`: road roughness (IMU vertical vibration, 0.5–20 Hz, speed-normalised) on 10 m cells, potholes / bumps over 0.35 g, traffic lights and stop signs triangulated from their tracks (needs a trajectory and a calibration), merged within 4 m across drives, and changes (a landmark not seen on the last passes with it in view; a new one where earlier passes saw nothing).
- **Reconstruction export.** `avrig recon SESSION`: Nerfstudio `transforms.json` (camera-to-world from trajectory × calibration, OpenGL axes, scene centred; OPENCV / OPENCV_FISHEYE intrinsics), full-resolution images, masks over moving objects. Train elsewhere: `ns-train splatfacto --data DIR`. ~35 GB per hour of driving at 2 Hz × 6 cameras.
- **Foxglove replay.** `avrig foxglove SESSION`: the MCAP export's messages (H.265 video, IMU, GNSS, markers, /tf, annotations, calibration) streamed at real time (or `--speed`, `--start`, `--loop`) to `ws://localhost:8765`, no export step.

- **Scenario search.** `avrig catalog` indexes every session into 10 s windows (SQLite, incremental): position and speed, sun elevation (day / twilight / night from UTC and position), image brightness, events, detected objects, and the session metadata. `avrig search` takes terms, ANDed: `night person>=3 road:highway`, `hard_brake speed>40`, `traffic_light twilight`, `near:LAT,LON,KM`, `vehicle:X route:X weather:X tag:X since:2026-11-01`. Hits are merged into clips; `--export DIR` writes each as an MCAP clip (from the keyframe before, so it decodes from the first frame) or frame sets. The GUI can replay a session at a hit.
- **Coverage map.** `avrig coverage` writes one HTML page: every drive on OpenStreetMap (offline: a plain track plot), coloured day / twilight / night, plus hours and km per light / weather / road / vehicle / route, and event counts.
- **Rig health trends.** `avrig trends` summarizes every session (cached outside the sessions) and flags wear: a camera restarting or dropping in 2 of the last 5 sessions, a camera offset or the inter-camera spread creeping up, time not LOCKED, temperatures rising, IMU packet loss.
- **Calibration studio.**
  1. `avrig calibrate board` writes a printable ChArUco board at true size; print at 100 %, mount flat, **measure** a square.
  2. Record a session moving the board through every camera's whole image and through the overlap of neighbouring cameras. The GUI shows live detection and coverage, which works only when the board is large in the 320 px preview.
  3. `avrig calibrate solve SESSION --square 0.0597 --register NAME --activate`: full-resolution frames on NVDEC, fisheye model for the wide lenses, pinhole for the narrow, outlier-view rejection, extrinsics from hardware-synced frame pairs chained to the reference camera (with their spread). Writes `calibration.yaml`, ROS `camera_info`, a Kalibr `camchain.yaml` and a report.
  4. Camera-IMU: `avrig calibrate kalibr SESSION` exports images + `imu0.csv` for `kalibr_bagcreater` (Kalibr needs an AprilGrid target, not ChArUco).
- **Trajectory.** `avrig trajectory SESSION`: error-state Kalman filter (15 states) on the IMU, GNSS position (at the fix time, with antenna lever arm) and velocity, zero-velocity and no-sideslip constraints, then an RTS smoother. IMU mounting is estimated from the drive (or `imu_mount_rpy_deg`). Output: pose per IMU sample with its 1-sigma (`trajectory.csv`), summary (`trajectory.json`). Needs driving with a fix; with RTK fixes it uses their cm accuracy.
- **Perception.** `avrig models --fetch --build` once (YOLOX-s Apache-2.0, YuNet MIT, YOLOv9-t plate detector MIT, CLIP ViT-B/32 MIT, Depth Anything V2 Small Apache-2.0 - its Base/Large variants are non-commercial and not used; checksums and revisions pinned; models are not in this repository). `avrig tag` adds tags and pre-annotations. `avrig anonymize` blurs faces (plus the head of anyone whose face was not found) and number plates in a copy of the session, with the same frames. Review blurred data before release; detection is never perfect.
- **Datasets.** `avrig dataset SESSION --format nuscenes|kitti [--annotate]`: keyframes at 2 Hz from all cameras, ego pose from the trajectory, camera intrinsics and poses on the vehicle (needs a calibration and `ref_camera_pose`), IMU, COCO pre-annotations. nuScenes tables load with the devkit's schema (annotation tables empty, to be labelled).
- **MCAP extras.** With `trajectory.csv`: `/tf` (map -> base_link, 50 Hz) and `/trajectory/fix`; with a calibration: `/tf_static` (base_link -> cameras) and `/camera/<id>/calibration`; with `detections.csv`: `/camera/<id>/annotations` (boxes over the video in Foxglove).
- **Event-only storage.** `avrig retain SESSION --pre 20 --post 10`: keeps the video segments around every event (hard links, no extra space), rewrites the frame index, keeps all sensor logs, verifies. `--replace --yes` then deletes the original, only if the copy verified. Use `segment_s: 30` on rigs that rely on it.
- **LED sync check.** An LED on a 40-pin GPIO flashes 0.5 ms every 200.7 ms; the flashes sweep the 33.3 ms frame cycle. From the recorded LED brightness, `avrig ledsync` measures each camera's exposure window relative to its own timestamp: that proves the images are aligned, not only the timestamps.
- **Hands-free collection.** `avrig auto` (or the `avrig-auto` service): records when GNSS speed >= 2.5 m/s for 3 s, stops after 5 min parked, starts a new session every hour; preflight still gates every start and is retried. Cameras are off while parked. The status page is the UI; an optional GPIO LED shows waiting / recording / fault (`auto_status_gpio`). SIGTERM stops a recording cleanly.
- **Depot offload.** `avrig depot` (or the 15-minute timer): when `offload_dest` is reachable and nothing records, verify and offload every finished session with the SHA-256-verified offload; with `depot_prune_days`, prune the ones whose copy re-hashes correctly.
- **Fleet dashboard.** `avrig fleet --rigs car1=http://10.0.0.11:8088/?token=X,car2=...`: one page for all rigs (health, recording, sync, issues), polled server-side, read-only.

## Root setup (optional, reversible)

```bash
sudo bash setup/avrig-install.sh --status
sudo bash setup/avrig-install.sh --autostart
sudo bash setup/avrig-install.sh --collection-mode --pause-containers "sentigon-grafana-1" --stop-services "ollama"
sudo bash setup/avrig-install.sh --can --bitrate 500000 --ifaces "can0 can1"
sudo bash setup/avrig-install.sh --auto           # hands-free service (needs --autostart too)
sudo bash setup/avrig-install.sh --depot          # offload timer
sudo bash setup/avrig-install.sh --uninstall
```

- **`--autostart`** installs `avrig-boot.service`: `avrig sync on --boot` as your user. It waits up to 120 s for all cameras, warms them, enables sync and verifies the lock. If the lock fails, it retries once. If it still fails, the rig is left **free-running**, and preflight then blocks recording until someone runs `avrig sync on`. A camera left dead by setting sync cold is worse than an unsynced rig, and free-run is never silent. The installer refuses to run while `blucap-framesync.service` (the known cold-boot failure) is enabled. The cold path itself has not been re-tested on this hardware. After installing, reboot once and check `journalctl -u avrig-boot`.
- **`--collection-mode`** installs `/usr/local/sbin/avrig-collection-mode on|off|status` and a sudoers rule for exactly those three commands.
  - While recording it saves the clocks and runs `jetson_clocks`, pauses the Docker containers listed in `/etc/avrig/collection-mode.conf`, and stops the listed services. When the recording stops it restores exactly what it changed.
  - Nothing is listed by default: choose what may be paused. During the measurements above, Grafana alone used ~125 % of one core.
  - Then set `collection_mode: true` in `~/.config/avrig.yaml`.
- **`--can`** installs `avrig-can.service`, which brings the mttcan interfaces up with a bitrate and `restart-ms 100` at boot. Then set `can_ifaces: [can0]`.
- **`--auto`** installs `avrig-auto.service` (`avrig auto` as your user, restarts on failure). The GUI cannot open the cameras while it runs: `sudo systemctl stop avrig-auto` first.
- **`--depot`** installs `avrig-depot.timer` (every 15 min, lowest priority; `avrig -q depot`).
- **GNSS master clock:** `setup/gnss-clock/install.sh` (chrony SHM refclock from gpsd, no PPS). Needs a fix.

## Not yet verified on hardware

The code paths are implemented and covered by tests (framing, parsing, simulated data with exact truth), but each needs wiring or conditions this bench does not have yet:

| Feature | Needs | Use |
|---|---|---|
| GNSS master clock | Antenna with sky view (fix), then `setup/gnss-clock/install.sh` | CLOCK chip turns green; `clock.csv` realtime becomes GNSS time |
| GNSS time marks of the camera trigger | Deserializer trigger (1.8 V) → level shifter → F9P **EXTINT**; a fix | `avrig gnss --enable`, `gnss_timemarks: true`; `verify` fits TSC → GNSS time per frame |
| IMU SyncIn | Trigger → level shifter → MTi **SYNC_IN** | `avrig imu` shows the settings; `avrig imu --set FUNCTION LINE --yes` writes and reads back. Check the codes against the Xsens Low-Level Communication Protocol for your MTi before applying. |
| CAN logging | Vehicle CAN-H/L + transceiver, correct bitrate | `--can` setup, `can_ifaces: [can0]`, `avrig can -t 10`; raw frames with kernel timestamps in `can.csv`, decode offline with the vehicle DBC |
| Live RTK | An NTRIP caster account (or own base) and a fix | `ntrip_url`, then `avrig gnss --ntrip -t 60`; tested against a local fake caster and fake gpsd, and the write path to the receiver through gpsd was checked on the rig (UBX poll answered) |
| PPK | `avrig gnss --raw-enable` + base-station data | RAWX/SFRBX logging checked on the rig (indoors: epochs with 0 measurements) |
| Trajectory | A drive with a fix | Validated on simulated drives only |
| Calibration studio | A printed, measured ChArUco board | Validated on rendered views only; the full-resolution pipeline was run on a real session |
| Plate blurring | Road traffic | Plate detector checked on a sample car photo; faces / heads checked on real office footage |
| LED sync check, status LED | An LED + resistor on a 40-pin GPIO | GPIO structures match the kernel ABI; analysis validated on synthetic data; no pin was driven |
| Hands-free mode | Driving | Policy and the full start / split / stop cycle tested with simulated cameras and speed |
| Depot / fleet | A NAS / several rigs | Tested with a local directory / local HTTP servers |
| Camera-mount guard | A drive with turns and straight road | Detection validated on rendered scenes; on the rig it ran in recordings (stationary: nothing to learn) and offline on real sessions |
| Visual speed | A drive with a calibrated forward camera | Rendered road only; real asphalt, glare, wet roads and traffic will lower the rate of valid measurements |
| 3D pre-labels | A calibration and a drive | Pipeline run on real office footage (people found and tracked, nominal geometry); accuracy measured with exact synthetic boxes only |
| Map layers | Repeated drives with GNSS (landmarks: + calibration) | Synthetic signals / rays / passes |
| Reconstruction export | A drive with a trajectory and a calibration; a desktop GPU to train | Export structure tested on a simulated drive; no reconstruction trained |
| GNSS jamming state | A receiver firmware with the interference detector (CFG-ITFM) | This NEO-F9P (HPGL1L5 1.40) rejects it; indicator, antenna and spoofing state work |

## Tests

```bash
cd ~/DataCollectionRig/avrig && python3 -m pytest -q tests
```

- Unit tests: sync verdicts, quality metrics and persistence, metadata, offload/prune safety (including same-size corruption and files added after offload), MCAP index/CRC structure and CDR layout, IMU sample-clock back-fill, UBX/TIM-TM2 parsing and the time-mark fit, Xsens framing, CAN frames, the calibration registry and moved check, the position wizard, config, and health rules.
- Lab unit tests: UBX raw tap, RTCM framing / CRC-24Q / GGA, the NTRIP client against a fake caster (ICY and chunked HTTP) and fake gpsd, bad credentials, automatic events on a synthetic drive, trend findings, catalog / search / coverage (incl. a power-cut-truncated CSV line), calibration of rendered pinhole and fisheye lenses and of extrinsics, trajectory accuracy and IMU mounting on simulated drives, ENU round trip, LED-sync analysis, GPIO ABI, the hands-free policy, phone markers (token, labels), the fleet dashboard, GPU detectors and blurring.
- Lab end-to-end (simulated cameras): live auto events + raw GNSS log in a recording, trajectory -> nuScenes / KITTI datasets and MCAP with /tf, /tf_static, calibration, event-only retention (hard links, verify), catalog search -> clip export, hands-free start / split / park-stop, depot offload, tagging + annotations + anonymization keeping every frame.
- Simulated-camera tests (real GStreamer + NVENC): exact frame accounting across segments, stall and pipeline-error isolation, write-error safe stop, SIGKILL mid-recording, markers / logs / verify.json / report, replay across a segment boundary, MCAP export (every frame at its capture time), and remote status (token, read-only). These recordings run with the mount-guard branch on, so a pipeline stall in it fails a test.
- v3 unit tests: mount guard on rendered scenes (learning, a 2.5° knock, no false alarm, nothing analysed while parked), gyro integration, MON-RF / SEC-SIG / NAV-STATUS parsing, the CLIP tokenizer against reference ids and image-text agreement, index search / rarity / curation / de-duplication, catalog text and scene terms and the schema migration, zero-shot classification, 3D lift + tracking + cross-camera merge on projected boxes, the depth-scale fit, visual speed on a rendered road up to 45 m/s, a 120 s GNSS outage bridged by visual speed, road roughness / potholes, landmark triangulation, merge and change detection, the Foxglove WebSocket server (protocol, token, slow viewers).
- v3 end-to-end (simulated cameras): nuScenes 3D annotation tables (linked chains per instance), the Nerfstudio export with masks, Foxglove replay received by a WebSocket client, map indexing of a drive, the offline mount check.
- The tests write the audit log and recording flag to a temporary directory, never the operator's.
