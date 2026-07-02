# Stray Dog Monitoring System

A real-time computer-vision system that watches a scene and **warns before a stray
dog attacks a person**. It detects people and dogs with **YOLO26**, tracks every
dog across frames, and continuously scores how dangerous each dog is to nearby
people — raising an alert (on screen, by sound, and optionally a hardware buzzer
distance trigger) the moment a dog starts behaving aggressively toward someone.

The system runs on ordinary video, a laptop webcam, or a low-cost **ESP32-CAM +
HC-SR04 ultrasonic** node mounted in the field.

---

## Why geometry instead of a "aggressive-dog" classifier

Labelled "aggressive dog" datasets are tiny, noisy and hard to trust. What
actually makes a stray dog dangerous is **geometry over time**, and that can be
measured directly from stock detections:

- **How close** the dog is to the nearest person
- **Whether it is closing in** (velocity toward the person)
- **Whether it is lunging / rearing** (sudden box shape change)
- **Whether a nearby human is reacting defensively** (arms up, crouching)
- **Whether it is part of a pack** (multiple dogs)

These signals are combined into one interpretable **risk score in [0, 1]**, then
smoothed and confirmed over several frames so a single noisy frame never triggers
a false alarm. No custom training required — YOLO26's COCO weights already know
`person` and `dog`.

---

## Architecture

```
                 ┌──────────────────────────────────────────────┐
  Video / Webcam │                                              │
  / ESP32-CAM ──▶│  1. Detector (YOLO26)                        │
                 │     person + dog boxes  (+ YOLO11 pose)       │
                 │                     │                         │
                 │                     ▼                         │
                 │  2. IoU Tracker — one persistent id per dog   │
                 │     keeps box + risk history per track        │
                 │                     │                         │
                 │                     ▼                         │
                 │  3. Risk Engine (geometric + human pose)      │
                 │     distance · velocity · posture · pose      │
                 │                     │                         │
                 │                     ▼                         │
                 │  4. Temporal logic                            │
                 │     EMA smoothing → sustain N frames →        │
                 │     cooldown → ALERT                          │
                 └──────────────────────┬───────────────────────┘
                                        ▼
                     On-screen box + risk sparkline,
                     sound / flash alert, JSON alert log
        (ESP32 ultrasonic distance adds an independent proximity trigger)
```

---

## Project structure

```
stray-dog-monitering/
├── run.py                       # launch the desktop app
├── config/
│   └── config.yaml              # every threshold / model / hardware setting
├── src/
│   ├── config.py                # config loader
│   ├── detection/detector.py    # YOLO26 / YOLO11 person+dog (+ pose) wrapper
│   ├── tracking/tracker.py      # greedy IoU tracker + per-dog memory
│   ├── risk/
│   │   ├── risk_engine.py       # geometric risk scoring
│   │   └── pose_features.py     # human posture features
│   ├── sources/mjpeg.py         # robust ESP32-CAM MJPEG reader
│   ├── analytics/
│   │   ├── recorder.py          # per-session ML/CV stats recording
│   │   └── dashboard.py         # offline HTML analytics dashboard
│   ├── pipeline.py              # end-to-end monitor (also a CLI)
│   └── app/gui.py               # desktop control-room application
├── firmware/
│   └── esp32_cam_ultrasonic/    # ESP32-CAM + HC-SR04 sketch
└── requirements.txt
```

---

## Quick start

### 1. Install

```bash
git clone https://github.com/tedo001/stray-dog-monitering.git
cd stray-dog-monitering
pip install -r requirements.txt
```

A CUDA GPU is recommended but not required — the `yolo26n` model runs on CPU.

### 2. Launch the app

```bash
python run.py
```

Then, in the control panel on the left:

1. **Select a source** — Video File, Laptop Webcam, or ESP32-CAM.
2. **Pick a model** — `yolo26n` (fastest) … `yolo26x` (most accurate), or the
   YOLO11 family (`yolo11n` / `yolo11m` / `yolo11x`). Weights auto-download on
   first run.
3. **Choose an alert type** — *Normal* or *HR* (high-risk: lower threshold,
   red screen flash + triple beep).
4. Press **Start Monitoring**.

You'll see live bounding boxes coloured by risk (green → amber → red), a per-dog
risk sparkline, a running dashboard (frames / persons / dogs / alerts / FPS /
distance), and a timestamped alert log you can export to JSON.

### 3. Analytics dashboard

Every monitoring run is recorded as a session (`data/sessions/*.json`): a
downsampled risk timeline, per-frame dog/person counts, every alert with its
feature breakdown, and the model + settings used. Press **Open Analytics
Dashboard** in the app (section 7) — or generate it headless:

```bash
python -m src.analytics.dashboard
```

This writes `outputs/dashboard.html`, a fully offline single-file dashboard
(no CDN, works without internet) with:

- **KPI row** — alerts fired, dog/person detections, frames processed, peak risk
- **Risk over time** — per-session risk timeline with the alert threshold and
  every alert marked (or peak-risk-per-session when viewing all sessions)
- **Detections** — dogs vs persons across the session
- **Alerts by hour of day** — when dogs actually get dangerous
- **Risk distribution** — histogram of observed risk scores
- **Alert signals** — which risk features (distance / velocity / posture /
  human pose) actually drive alerts
- **Sessions table** — every run with its model, source, alert mode and stats

A session filter scopes every chart, each chart has a data-table fallback, and
the page supports light/dark mode. Recording is fail-safe: analytics only
observes the pipeline and can never affect detection or alerting. Disable it
with `analytics.enabled: false` in `config/config.yaml`.

### 4. Command-line inference (headless)

```bash
# Single image
python -m src.pipeline --source path/to/image.jpg --output result.jpg

# Video file
python -m src.pipeline --source path/to/video.mp4 --output annotated.mp4

# Webcam
python -m src.pipeline --source webcam
```

---

## Hardware node (ESP32-CAM + HC-SR04)

Flash `firmware/esp32_cam_ultrasonic/esp32_cam_ultrasonic.ino` to an AI-Thinker
ESP32-CAM (set your WiFi SSID/password at the top). On boot it prints its IP and
serves:

| Endpoint | Purpose |
|----------|---------|
| `http://<IP>:81/stream` | MJPEG video stream (consumed by the app) |
| `http://<IP>/distance`  | `{"distance_cm": 42.5}` ultrasonic reading |
| `http://<IP>/status`    | device health (uptime, WiFi RSSI, PSRAM) |

In the app, enter that IP under **ESP Sensor**, press **Connect Sensor**, and the
ultrasonic distance is polled live. If a dog comes closer than the
**proximity alert threshold**, an independent hardware-side alert fires — a
second line of defence that works even if the camera view is momentarily blocked.

Wiring: HC-SR04 `TRIG → GPIO14`, `ECHO → GPIO13` (use a voltage divider on ECHO).

---

## Configuration

All behaviour is driven by `config/config.yaml` — there are no magic numbers in
the code. The most useful knobs:

```yaml
detector:
  model: yolo26n.pt        # yolo26n / s / m / l / x
  conf: 0.35               # detection confidence

risk:
  threshold: 0.35          # alert when smoothed risk crosses this
  smoothing_alpha: 0.6     # lower = smoother, slower to react
  sustain_frames: 2        # must stay above threshold for N frames
  cooldown_frames: 30      # suppress repeat alerts for the same dog
  weights:                 # contribution of each risk signal
    distance: 0.55
    velocity: 0.20
    posture: 0.10
    human_pose: 0.05

hardware:
  proximity_alert_cm: 100  # ultrasonic "too close" distance

alerts:
  default_type: normal     # normal | hr
  hr_threshold_drop: 0.10  # HR mode lowers the risk threshold by this
```

The GUI sliders are seeded from these values and override them per run.

---

## How the risk score works

For each tracked dog, per frame:

| Signal | Weight | Intuition |
|--------|:------:|-----------|
| Distance to nearest person | 0.55 | The single strongest danger cue |
| Velocity toward that person | 0.20 | A dog *charging* is far worse than one nearby but still |
| Posture change | 0.10 | Aspect-ratio spikes catch lunging / rearing |
| Human defensive pose | 0.05 | Arms up / crouching corroborates a real threat |
| Pack bonus | +0.10 | Multiple dogs escalate risk |

The raw score is EMA-smoothed per dog, must stay above the threshold for
`sustain_frames` consecutive frames, and then enters a `cooldown_frames` window
so one event produces one alert — not a storm of them.

**Safety priority: recall over precision.** Missing a genuinely aggressive dog is
worse than an occasional false alarm, so the defaults lean toward catching more.

---

## Tech stack

- **Detection**: YOLO26 (Ultralytics) — person + dog
- **Pose** (optional): YOLO11-Pose — human keypoints for defensive-posture cues
- **Tracking**: custom greedy IoU tracker with per-dog temporal memory
- **Risk**: interpretable geometric + temporal scoring (no black-box classifier)
- **GUI**: Tkinter + OpenCV + Pillow
- **Edge hardware**: ESP32-CAM (MJPEG) + HC-SR04 ultrasonic

---

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA optional but recommended)
- Ultralytics (YOLO26 support), OpenCV, NumPy, Pillow, PyYAML
