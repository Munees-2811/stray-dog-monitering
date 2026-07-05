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
├── run.py                       # launch the desktop app (Tkinter)
├── app2.py                      # launch the web app (Streamlit) + CCTV module
├── app3.py                      # launch the native desktop app (PyQt6) + CCTV
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

### 2b. Web app (Streamlit) — with CCTV module

```bash
streamlit run app2.py
```

Everything the desktop app does, in the browser, plus a **CCTV module**:

- **Live Monitor tab** — Video File (upload or path), Webcam, ESP32-CAM, or
  **CCTV (RTSP)** sources; the same YOLO26/YOLO11 model picker, Normal/HR alert
  modes and threshold sliders; live annotated frames, stats row, alert log,
  browser alert sound, annotated-video saving, alert JSON export, and a +10 s
  skip for video files.
- **Analytics Dashboard tab** — the full dashboard rendered in-page (sessions
  from the desktop and web apps aggregate together), with an HTML download.
- **CCTV Cameras tab** — register named RTSP/HTTP cameras
  (`rtsp://user:pass@ip:554/stream1`), test each one, view a snapshot wall of
  every camera, and pick any of them as a Live Monitor source. Cameras are
  stored in `data/cameras.json`.
- **ESP32 sensor** — distance readout in the sidebar plus proximity toasts
  while monitoring.

### 2c. Native desktop app (PyQt6) — fully local, fastest

```bash
python app3.py
```

The same features as the Streamlit app, but as a native window with **no web
server and no browser** — the simplest thing to run locally, and the fastest
(no websocket between the model and the display). Recommended for on-site /
MSME deployments.

- **Live Monitor** — Video file / Webcam / ESP32-CAM / CCTV (RTSP) sources; the
  YOLO26 + YOLO11 model picker and custom-weights field; Normal/HR alerts;
  live annotated video, a stats row, alert log + JSON export, annotated-video
  saving, +10 s skip. A **compute line** under the video shows GPU vs CPU.
- **CCTV Cameras tab** — register/test/remove RTSP cameras (shared
  `data/cameras.json` with the web app); any camera is selectable as a source.
- **Analytics** — one click builds the dashboard and opens it in your browser.
- Detection runs in a background thread (the window never freezes) and live
  sources use the threaded latest-frame reader, so the feed can't lag behind.

> Run it with the **same Python that has CUDA torch** (e.g.
> `& .\venv_gpu\Scripts\python.exe app3.py`) — the compute line tells you
> whether you got the GPU.

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
- PyTorch 2.0+ (CUDA strongly recommended for real-time — see below)
- Ultralytics (YOLO26 support), OpenCV, NumPy, Pillow, PyYAML

## GPU acceleration (recommended)

On an NVIDIA GPU the detector runs many times faster than on CPU (the
difference between smooth real-time and a slideshow). The app selects the GPU
automatically (`project.device: auto` in `config/config.yaml`) — you only have
to install a **CUDA build of PyTorch**.

> On Windows, a plain `pip install torch` installs the **CPU-only** build. You
> must install from PyTorch's CUDA index to use the GPU.

```bash
# 1. confirm the GPU is visible to the OS
nvidia-smi

# 2. install matched CUDA torch + torchvision TOGETHER (one command, one index)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. verify — you want "cuda True" and your GPU name
python -c "import torch, torchvision; print(torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Notes:
- **Install `torch` and `torchvision` in the same command from the same index** —
  they are version-locked; installing/removing them separately mismatches them
  (`torch 2.6.0+cu124` pairs with `torchvision 0.21.0+cu124`, etc.).
- If the `cu124` index has no wheel for your torch version, use the official
  selector at <https://pytorch.org/get-started/locally> (Windows · Pip · CUDA)
  for the exact command.
- No GPU? The app still runs on CPU — keep it light: model `yolo26n`, pose off,
  and raise **Skip frames** to 3–5.

### If FPS is low

When monitoring starts, both apps show a **compute line** (web: above the
preview; desktop: in the alert log). If it says `CPU — no CUDA torch in this
environment`, inference is running on the CPU — that is the cause of ~1 FPS,
and the fix is the CUDA install above **in the same environment the app runs
from**. Launch through the environment's own interpreter so there is no
ambiguity:

```powershell
& .\venv_gpu\Scripts\python.exe -m streamlit run app2.py
```

Other speed levers, in order of impact:

1. **Pose model off** (sidebar toggle) — pose is a second full inference per
   frame but contributes only 5% of the risk score. The default pose model is
   the nano (`yolo11n-pose.pt`) for this reason.
2. **Skip frames = 2–4** — process every Nth frame; the risk engine's temporal
   logic tolerates this well.
3. **Smaller detector** — `yolo26n`/`yolo26s` for live sources.

### Picking a model for your hardware

| Use case | Model | Notes |
|----------|-------|-------|
| Live webcam / CCTV / ESP32-CAM | `yolo26n` or `yolo26s` | fastest; turn pose off for max FPS |
| Saved-video analysis (accuracy first) | `yolo26m` / `yolo26x` | practical on GPU; slow on CPU |

You do **not** need to train or fine-tune anything: the YOLO26/YOLO11 weights
are already COCO-pretrained and detect people and dogs out of the box. Only
fine-tune (on a *custom* stray-dog dataset, never on COCO itself) if you need
better accuracy on your specific scenes — see the next section.

## Fine-tuning on a project-oriented dataset (optional)

COCO is a *general* 80-class dataset (dog is class 16, person is class 0) —
that is exactly why stock weights already detect dogs and people, and why
re-training on COCO is pointless: you'd spend days reproducing the weights you
downloaded. **Never run `yolo detect train data=coco.yaml ...`** — full COCO is
a ~20 GB download and 118k images per epoch, which also OOM-kills most laptops
(`yolo26m` at the default `batch=16` needs more than 4 GB VRAM).

Fine-tune only on a small **stray-dog-specific** dataset:

**Where to get one (pick either):**

1. **Roboflow Universe** (recommended) — go to <https://universe.roboflow.com>
   and search **"stray dog"** or **"dog detection"**. Pick a dataset with
   ~500–3,000 images, click *Download Dataset* → format **YOLOv11** — you get a
   zip with `data.yaml`, `train/`, `valid/`, `test/`. Free account required.
2. **Your own footage** (best accuracy for your exact scenes) — the app already
   saves annotated/raw videos to `outputs/`. Extract a few hundred frames,
   upload them to Roboflow, draw dog/person boxes (its assisted labeling makes
   this ~1–2 hours for 300–500 images), then export as YOLOv11 like above.

**Sanity check first** (~7 MB built-in mini-dataset, finishes in minutes and
proves your training setup works — this is the "simple dataset" to test with):

```bash
python finetune.py --data coco128.yaml --model yolo26n.pt --epochs 5
```

**Real fine-tune** (defaults sized for a 4 GB GPU: `batch=8`; use `--batch 4`
or `--batch -1` if you hit out-of-memory):

```bash
python finetune.py --data path/to/my_dataset/data.yaml --model yolo26n.pt --epochs 60
```

It prints the path to `best.pt` when done (typically
`runs/detect/straydog_finetune/weights/best.pt`). ~500 images × 60 epochs takes
roughly 30–90 minutes on an RTX 3050 — not days.

**Use the fine-tuned model** — paste the `best.pt` path into:

- *Web app*: sidebar → **Custom weights (.pt)** (overrides the model picker)
- *Desktop app*: section 4 → **Custom weights (.pt)** entry
- or set `detector.model: runs/detect/straydog_finetune/weights/best.pt`
  in `config/config.yaml`

One caveat: the risk engine expects both `person` and `dog` detections. If your
custom dataset contains only dogs, keep using a stock model — or make sure your
dataset labels people too, with class ids matching `data.yaml`.
