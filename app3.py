"""
Stray Dog Monitoring System — native desktop app (PyQt6).

A fully local desktop version of app2.py (the Streamlit web app), with the
same features but no browser and no web server:

  - Live Monitor ....... Video File / Webcam / ESP32-CAM / CCTV (RTSP) sources,
                         YOLO26 + YOLO11 model picker + custom fine-tuned
                         weights, Normal / HR alert modes, live annotated
                         video, stats, alert log + JSON export, annotated-video
                         saving, +10 s skip for video files
  - CCTV Cameras ....... register named RTSP/HTTP cameras, test them, and use
                         any of them as a monitoring source
  - Analytics .......... one-click offline HTML dashboard (opens in browser)
  - ESP32 sensor ....... HC-SR04 distance readout + proximity alerts

Detection runs in a background QThread so the UI never freezes, and live
sources use the threaded latest-frame reader so the feed can never lag behind.

Run locally:
    python app3.py
"""

import os
import sys
import json
import time
import threading
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;4000000")
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2
try:
    cv2.setLogLevel(0)
except AttributeError:
    pass

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QSlider, QCheckBox, QRadioButton, QLineEdit, QTextEdit, QSpinBox,
    QFileDialog, QMessageBox, QTabWidget, QScrollArea, QVBoxLayout,
    QHBoxLayout, QGridLayout, QGroupBox, QFrame, QSizePolicy, QButtonGroup,
)

# In-app dashboard rendering. Imported at module level on purpose: Qt requires
# WebEngine to be initialised before the QApplication is created.
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except Exception:                     # PyQt6-WebEngine not installed
    HAS_WEBENGINE = False

from src.config import load_config, PROJECT_ROOT

if sys.platform == "win32":
    import winsound as _winsound
else:
    _winsound = None


MODEL_VARIANTS = {
    "yolo26n.pt": "YOLO26 Nano — fastest (CPU friendly)",
    "yolo26s.pt": "YOLO26 Small — fast",
    "yolo26m.pt": "YOLO26 Medium — balanced",
    "yolo26l.pt": "YOLO26 Large — more accurate",
    "yolo26x.pt": "YOLO26 XLarge — best accuracy",
    "yolo11n.pt": "YOLO11 Nano — fastest",
    "yolo11m.pt": "YOLO11 Medium — balanced",
    "yolo11x.pt": "YOLO11 XLarge — best accuracy",
}

SRC_VIDEO, SRC_WEBCAM, SRC_ESP, SRC_CCTV = "video", "webcam", "espcam", "cctv"
CAMERAS_FILE = PROJECT_ROOT / "data" / "cameras.json"

# The embedded WebEngine view defaults to a light color-scheme; the dashboard
# reads its colors from CSS variables at render time, so applying the dark
# palette inline and re-rendering gives the designed dark theme (not an
# auto-inverted approximation).
_DASH_DARK_JS = """
(function () {
  const v = {'--surface-1':'#1a1a19','--page':'#0d0d0d','--text-primary':'#ffffff',
    '--text-secondary':'#c3c2b7','--text-muted':'#898781','--grid':'#2c2c2a',
    '--baseline':'#383835','--border':'rgba(255,255,255,0.10)',
    '--series-1':'#3987e5','--series-2':'#199e70','--critical':'#d03b3b'};
  for (const k in v) document.body.style.setProperty(k, v[k]);
  if (typeof render === 'function') render();
})();
"""

DARK_QSS = """
QWidget { background: #1e1e1e; color: #e8e8e8; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
QScrollArea { background: #232323; border: none; }
QScrollArea > QWidget > QWidget { background: #232323; }
QGroupBox { background: #262626; border: 1px solid #383838; border-radius: 8px;
            margin-top: 12px; padding: 10px 8px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px;
                   color: #f59e0b; font-weight: 600; }
QGroupBox QWidget { background: #262626; }
QPushButton { background: #3a3a3a; border: none; border-radius: 6px; padding: 8px 12px; }
QPushButton:hover { background: #454545; }
QPushButton:disabled { background: #2a2a2a; color: #666; }
QPushButton#primary { background: #16a34a; color: white; font-weight: 600; padding: 10px; }
QPushButton#primary:hover { background: #18b352; }
QPushButton#danger { background: #dc2626; color: white; }
QPushButton#accent { background: #7c3aed; color: white; }
QLineEdit, QComboBox, QSpinBox, QTextEdit { background: #1a1a1a; border: 1px solid #3a3a3a;
                                            border-radius: 5px; padding: 6px; }
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView { background: #1a1a1a; selection-background-color: #0078d4; }
QSlider { min-height: 24px; }
QSlider::groove:horizontal { height: 5px; background: #3a3a3a; border-radius: 2px; }
QSlider::handle:horizontal { background: #f59e0b; width: 16px; margin: -6px 0; border-radius: 8px; }
QSlider::sub-page:horizontal { background: #f59e0b; border-radius: 2px; }
QTabBar::tab { background: #252525; padding: 9px 18px; border-top-left-radius: 6px;
               border-top-right-radius: 6px; margin-right: 2px; }
QTabBar::tab:selected { background: #1e1e1e; color: #f59e0b; }
QTabWidget::pane { border: none; }
QScrollBar:vertical { background: #1e1e1e; width: 12px; border-radius: 6px; }
QScrollBar::handle:vertical { background: #4a4a4a; min-height: 40px; border-radius: 5px; margin: 2px; }
QScrollBar::handle:vertical:hover { background: #5c5c5c; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #1e1e1e; height: 12px; }
QScrollBar::handle:horizontal { background: #4a4a4a; min-width: 40px; border-radius: 5px; margin: 2px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QLabel#stat { font-size: 24px; font-weight: 600; }
QLabel#statlabel { color: #888; font-size: 11px; }
QLabel#section { color: #f59e0b; font-weight: 600; font-size: 14px; }
QLabel#hint { color: #777; font-size: 11px; }
"""


# ── helpers ───────────────────────────────────────────────────────────

def load_cameras():
    try:
        with open(CAMERAS_FILE, "r", encoding="utf-8") as fh:
            cams = json.load(fh)
        return cams if isinstance(cams, list) else []
    except Exception:
        return []


def save_cameras(cams):
    CAMERAS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CAMERAS_FILE, "w", encoding="utf-8") as fh:
        json.dump(cams, fh, indent=2)


def esp_get(ip, endpoint, timeout=2):
    with urllib.request.urlopen(f"http://{ip}/{endpoint}", timeout=timeout) as r:
        return json.loads(r.read())


def bgr_to_qimage(bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


def grab_one_frame(url_or_index, timeout_s=6):
    cap = cv2.VideoCapture(url_or_index)
    try:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
        return None
    finally:
        cap.release()


# ── background workers ────────────────────────────────────────────────

class GrabThread(QThread):
    """One-off frame grab for CCTV Test / snapshot (keeps the UI responsive)."""
    done = pyqtSignal(bool, object, str)

    def __init__(self, url, name):
        super().__init__()
        self.url, self.name = url, name

    def run(self):
        frame = grab_one_frame(self.url)
        if frame is None:
            self.done.emit(False, None, self.name)
        else:
            self.done.emit(True, bgr_to_qimage(frame), self.name)


class EspReadThread(QThread):
    done = pyqtSignal(str)

    def __init__(self, ip):
        super().__init__()
        self.ip = ip

    def run(self):
        try:
            d = esp_get(self.ip, "distance").get("distance_cm")
            self.done.emit("Out of range" if d is None else f"{d:.1f} cm")
        except Exception as e:
            self.done.emit(f"unreachable: {e}")


class MonitorThread(QThread):
    """Runs the capture + inference loop off the UI thread."""
    frameReady = pyqtSignal(object)      # QImage
    statsReady = pyqtSignal(dict)
    logMsg = pyqtSignal(str)
    alertMsg = pyqtSignal(dict)
    computeMsg = pyqtSignal(str)
    finishedRun = pyqtSignal(dict)
    errorMsg = pyqtSignal(str)

    def __init__(self, meta, cfg):
        super().__init__()
        self.m = meta
        self.cfg = cfg
        self._stop = False
        self._seek_lock = threading.Lock()
        self._seek_seconds = 0

    def request_skip(self):
        with self._seek_lock:
            self._seek_seconds += 10

    def stop(self):
        self._stop = True

    def run(self):
        m, cfg = self.m, self.cfg
        pipeline = writer = recorder = cap = None
        try:
            from src.pipeline import StrayDogMonitor
            self.logMsg.emit(f"Loading {m['model']} …")
            pipeline = StrayDogMonitor(
                detector_path=m["model"], pose_model=m["pose"],
                det_conf=m["det_conf"], risk_threshold=m["eff_risk"],
                sustain_frames=m["sustain"], config=cfg)
            self.computeMsg.emit(getattr(pipeline, "compute", "unknown"))

            is_live = m["is_live"]
            if m["src_type"] == SRC_ESP:
                from src.sources import MJPEGCapture, LatestFrameCapture
                ip = m["esp_ip"]
                try:
                    info = esp_get(ip, "status", timeout=3)
                    self.logMsg.emit(f"[ESP] online RSSI {info.get('wifi_rssi','?')} dBm")
                except Exception as e:
                    raise RuntimeError(f"ESP32 not reachable at {ip}: {e}")
                cap = LatestFrameCapture(MJPEGCapture(f"http://{ip}:81/stream", timeout=5))
            else:
                raw = cv2.VideoCapture(m["open_args"])
                if is_live:
                    raw.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    from src.sources import LatestFrameCapture
                    cap = LatestFrameCapture(raw)
                else:
                    cap = raw
            if not cap.isOpened():
                raise RuntimeError("Could not open the video source.")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if fps <= 0:
                fps = 30.0
            total_frames = 0 if is_live else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

            acfg = cfg.get("analytics", {}) or {}
            if acfg.get("enabled", True):
                try:
                    from src.analytics import SessionRecorder
                    recorder = SessionRecorder(
                        model=m["model"], source=m["src_type"],
                        alert_type=m["alert_type"], risk_threshold=m["eff_risk"],
                        det_conf=m["det_conf"],
                        sessions_dir=acfg.get("sessions_dir", "data/sessions"),
                        timeline_max_points=acfg.get("timeline_max_points", 600))
                except Exception:
                    recorder = None

            output_path = None
            if m["save"] and not is_live:
                out_dir = PROJECT_ROOT / cfg["inference"].get("output_dir", "outputs")
                out_dir.mkdir(parents=True, exist_ok=True)
                tag = "hr" if m["alert_type"] == "hr" else "norm"
                output_path = str(out_dir / (
                    f"desktop_{Path(m['model']).stem}_{tag}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"))
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (width, height))

            skip = m["skip"]
            frame_count = processed = total_alerts = 0
            peak_dogs = peak_persons = 0
            alerts = []
            start_time = time.time()
            last_emit = last_esp = 0.0
            dist_txt = "–"

            while not self._stop:
                if not is_live:
                    with self._seek_lock:
                        if self._seek_seconds > 0:
                            frame_count += int(self._seek_seconds * fps)
                            self._seek_seconds = 0
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                if (frame_count - 1) % skip != 0:
                    if writer:
                        writer.write(frame)
                    continue

                results = pipeline.process_frame(frame)
                annotated = pipeline.draw_results(frame, results)
                persons_now = len(pipeline._last_persons)
                dogs_now = len(results)
                peak_persons = max(peak_persons, persons_now)
                peak_dogs = max(peak_dogs, dogs_now)

                if recorder:
                    recorder.record_frame(results, persons_now, frame_count)

                new_alerts = [r for r in results if r.get("new_alert")]
                if new_alerts:
                    total_alerts += len(new_alerts)
                    ts = frame_count / fps
                    for a in new_alerts:
                        entry = {
                            "time": f"{int(ts//60):02d}:{int(ts%60):02d}",
                            "frame": frame_count, "track_id": a["track_id"],
                            "risk": round(a["risk"], 3), "model": m["model"],
                            "alert_type": m["alert_type"],
                            "features": a.get("features", {})}
                        alerts.append(entry)
                        self.alertMsg.emit(entry)

                if m["esp_poll"] and time.time() - last_esp > 0.5:
                    last_esp = time.time()
                    try:
                        d = esp_get(m["esp_ip"], "distance", timeout=1).get("distance_cm")
                        dist_txt = "–" if d is None else f"{d:.0f}"
                    except Exception:
                        dist_txt = "×"

                if writer:
                    writer.write(annotated)

                processed += 1
                elapsed = time.time() - start_time
                cur_fps = processed / elapsed if elapsed > 0 else 0.0

                now = time.time()
                if now - last_emit >= 0.033 or new_alerts:
                    last_emit = now
                    self.frameReady.emit(bgr_to_qimage(annotated))
                    self.statsReady.emit({
                        "frames": frame_count, "persons": persons_now,
                        "dogs": dogs_now, "alerts": total_alerts,
                        "fps": cur_fps, "dist": dist_txt,
                        "progress": (frame_count / total_frames) if total_frames else 0.0})

            if recorder:
                recorder.finalize(save=True)

            self.finishedRun.emit({
                "stopped": self._stop, "alerts": alerts, "output_path": output_path,
                "frames": processed, "peak_dogs": peak_dogs, "model": m["model"],
                "alert_type": m["alert_type"]})

        except Exception as e:
            self.errorMsg.emit(str(e))
        finally:
            try:
                if cap:
                    cap.release()
            except Exception:
                pass
            if writer:
                writer.release()


# ── main window ───────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = load_config()
        self.setWindowTitle("Stray Dog Monitoring System — Desktop")
        self.resize(1360, 900)

        self.thread = None
        self.alerts = []
        self._last_qimg = None

        d, r, hw, al, inf = (self.cfg["detector"], self.cfg["risk"],
                             self.cfg["hardware"], self.cfg["alerts"],
                             self.cfg["inference"])
        self._pose_model = d.get("pose_model")
        self._hr_drop = al.get("hr_threshold_drop", 0.10)
        self._defaults = dict(d=d, r=r, hw=hw, al=al, inf=inf)

        tabs = QTabWidget()
        tabs.addTab(self._build_monitor_tab(), "🎥  Live Monitor")
        tabs.addTab(self._build_analytics_tab(), "📊  Analytics")
        tabs.addTab(self._build_cctv_tab(), "🎦  CCTV Cameras")
        self.tabs = tabs
        self.TAB_ANALYTICS = 1
        tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(tabs)

    # ---- monitor tab ----

    def _build_monitor_tab(self):
        d, r, hw, inf = (self._defaults["d"], self._defaults["r"],
                         self._defaults["hw"], self._defaults["inf"])
        page = QWidget()
        root = QHBoxLayout(page)

        # left: scrollable controls
        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setSpacing(6)
        pl.setContentsMargins(12, 8, 18, 8)   # right margin clears the scrollbar

        # 1 · Source
        g = self._group(pl, "1 · Source")
        self.src_group = QButtonGroup(self)
        src_row = QHBoxLayout(); src_row.setSpacing(10)
        for key, label in [(SRC_VIDEO, "Video"), (SRC_WEBCAM, "Webcam"),
                           (SRC_ESP, "ESP32"), (SRC_CCTV, "CCTV")]:
            rb = QRadioButton(label)
            rb.setProperty("srckey", key)
            if key == SRC_VIDEO:
                rb.setChecked(True)
            rb.toggled.connect(self._on_source_change)
            self.src_group.addButton(rb)
            src_row.addWidget(rb)
        src_row.addStretch()
        srw = QWidget(); srw.setLayout(src_row); g.addWidget(srw)

        # video sub-panel
        self.video_box = QWidget(); vlay = QVBoxLayout(self.video_box)
        vlay.setContentsMargins(0, 4, 0, 0); vlay.setSpacing(6)
        browse = QPushButton("Browse local video…"); browse.clicked.connect(self._browse_video)
        vlay.addWidget(browse)
        self.path_edit = QLineEdit(); self.path_edit.setPlaceholderText(r"…or paste C:\videos\clip.mp4")
        vlay.addWidget(self.path_edit)
        self.path_label = QLabel("No video selected"); self.path_label.setObjectName("hint")
        vlay.addWidget(self.path_label)
        g.addWidget(self.video_box)

        # webcam sub-panel
        self.webcam_box = QWidget(); wlay = QHBoxLayout(self.webcam_box)
        wlay.setContentsMargins(0, 4, 0, 0)
        wlay.addWidget(QLabel("Camera index:"))
        self.cam_spin = QSpinBox(); self.cam_spin.setRange(0, 10)
        wlay.addWidget(self.cam_spin); wlay.addStretch()
        g.addWidget(self.webcam_box)

        # cctv sub-panel
        self.cctv_box = QWidget(); clay = QVBoxLayout(self.cctv_box)
        clay.setContentsMargins(0, 4, 0, 0); clay.setSpacing(6)
        clay.addWidget(QLabel("Registered camera:"))
        self.cctv_combo = QComboBox(); clay.addWidget(self.cctv_combo)
        self.cctv_manual = QLineEdit(); self.cctv_manual.setPlaceholderText("…or rtsp://user:pass@ip:554/stream1")
        clay.addWidget(self.cctv_manual)
        g.addWidget(self.cctv_box)

        # 2 · Detection model
        g = self._group(pl, "2 · Detection model")
        self.model_combo = QComboBox()
        for k, v in MODEL_VARIANTS.items():
            self.model_combo.addItem(f"{k} — {v}", k)
        default = d["model"] if d["model"] in MODEL_VARIANTS else "yolo26n.pt"
        self.model_combo.setCurrentIndex(list(MODEL_VARIANTS).index(default))
        g.addWidget(self.model_combo)
        self.custom_edit = QLineEdit()
        self.custom_edit.setPlaceholderText(r"Custom weights .pt (overrides above)")
        g.addWidget(self.custom_edit)
        self.pose_check = QCheckBox("Use pose model (human skeleton)")
        self.pose_check.setChecked(bool(d.get("pose_model")))
        g.addWidget(self.pose_check)

        # 3 · Alert type
        g = self._group(pl, "3 · Alert type")
        self.alert_group = QButtonGroup(self)
        row = QHBoxLayout()
        self.rb_normal = QRadioButton("Normal"); self.rb_normal.setChecked(True)
        self.rb_hr = QRadioButton("HR (high-risk)")
        self.alert_group.addButton(self.rb_normal); self.alert_group.addButton(self.rb_hr)
        row.addWidget(self.rb_normal); row.addWidget(self.rb_hr); row.addStretch()
        rw = QWidget(); rw.setLayout(row); g.addWidget(rw)
        self.sound_check = QCheckBox("Alert sound"); self.sound_check.setChecked(True)
        g.addWidget(self.sound_check)

        # 4 · Thresholds
        g = self._group(pl, "4 · Thresholds")
        self.conf_slider = self._slider(g, "Detection confidence", 10, 95,
                                        int(d["conf"] * 100), factor=100)
        self.risk_slider = self._slider(g, "Risk threshold", 10, 95,
                                        int(r["threshold"] * 100), factor=100)
        self.sustain_slider = self._slider(g, "Sustain frames (N)", 1, 20,
                                           int(r["sustain_frames"]))
        self.skip_slider = self._slider(g, "Skip frames", 1, 30,
                                        int(inf.get("skip_frames", 1)))
        hint = QLabel("Skip 2–4 speeds up videos; above ~5 degrades motion signals.")
        hint.setObjectName("hint"); hint.setWordWrap(True)
        g.addWidget(hint)
        self.save_check = QCheckBox("Save annotated output video")
        self.save_check.setChecked(bool(inf.get("save_output", True)))
        g.addWidget(self.save_check)

        # 5 · ESP32 sensor
        g = self._group(pl, "5 · ESP32 sensor (HC-SR04)")
        erow = QHBoxLayout()
        erow.addWidget(QLabel("IP:"))
        self.esp_edit = QLineEdit(hw.get("esp_ip", "")); erow.addWidget(self.esp_edit)
        ew = QWidget(); ew.setLayout(erow); g.addWidget(ew)
        self.prox_slider = self._slider(g, "Proximity alert (cm)", 10, 400,
                                        int(hw.get("proximity_alert_cm", 100)))
        self.esp_poll_check = QCheckBox("Poll distance while monitoring")
        self.esp_poll_check.setChecked(bool(hw.get("esp_ip")))
        g.addWidget(self.esp_poll_check)
        read_btn = QPushButton("Read distance now"); read_btn.clicked.connect(self._read_distance)
        g.addWidget(read_btn)

        # 6 · Run
        g = self._group(pl, "6 · Run")
        self.start_btn = QPushButton("▶  Start monitoring"); self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self._start)
        g.addWidget(self.start_btn)
        run_row = QHBoxLayout()
        self.stop_btn = QPushButton("■  Stop"); self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False); self.stop_btn.clicked.connect(self._stop)
        self.fwd_btn = QPushButton("⏩  +10 s"); self.fwd_btn.setEnabled(False)
        self.fwd_btn.clicked.connect(lambda: self.thread and self.thread.request_skip())
        run_row.addWidget(self.stop_btn); run_row.addWidget(self.fwd_btn)
        rr = QWidget(); rr.setLayout(run_row); g.addWidget(rr)
        dash_btn = QPushButton("📊  Analytics dashboard"); dash_btn.setObjectName("accent")
        dash_btn.clicked.connect(self._open_dashboard)
        g.addWidget(dash_btn)
        pl.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setFixedWidth(370)
        # vertical-only scrolling: content always fits the width, wheel scrolls
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        root.addWidget(scroll)

        # right: display + stats + log
        right = QVBoxLayout()
        self.video_label = QLabel("Video preview will appear here")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background:#000; border:3px solid #1a1a1a; color:#555;")
        self.video_label.setMinimumHeight(430)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right.addWidget(self.video_label, stretch=1)

        self.compute_label = QLabel("")
        self.compute_label.setStyleSheet("color:#888;")
        right.addWidget(self.compute_label)

        stats_row = QHBoxLayout()
        self.stat_widgets = {}
        for key, label in [("frames", "Frames"), ("persons", "Persons now"),
                           ("dogs", "Dogs now"), ("alerts", "Alerts"),
                           ("fps", "FPS"), ("dist", "Dist (cm)")]:
            box = QVBoxLayout()
            val = QLabel("0"); val.setObjectName("stat")
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if key == "alerts":
                val.setStyleSheet("color:#dc2626;")
            elif key == "fps":
                val.setStyleSheet("color:#22c55e;")
            cap = QLabel(label); cap.setObjectName("statlabel")
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.addWidget(val); box.addWidget(cap)
            w = QWidget(); w.setLayout(box); stats_row.addWidget(w)
            self.stat_widgets[key] = val
        sr = QWidget(); sr.setLayout(stats_row); right.addWidget(sr)

        self.status_label = QLabel("Ready"); self.status_label.setStyleSheet("color:#888;")
        right.addWidget(self.status_label)

        log_box = QGroupBox("Alert log")
        lv = QVBoxLayout(log_box)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9)); self.log_text.setFixedHeight(160)
        lv.addWidget(self.log_text)
        self.export_btn = QPushButton("Export alerts (JSON)"); self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_alerts)
        lv.addWidget(self.export_btn)
        right.addWidget(log_box)

        rw2 = QWidget(); rw2.setLayout(right)
        root.addWidget(rw2, stretch=1)

        self._on_source_change()
        return page

    # ---- analytics tab ----

    def _build_analytics_tab(self):
        page = QWidget(); v = QVBoxLayout(page)
        v.setContentsMargins(10, 10, 10, 10); v.setSpacing(8)

        bar = QHBoxLayout()
        title = QLabel("Analytics dashboard"); title.setObjectName("section")
        bar.addWidget(title)
        self.analytics_info = QLabel(""); self.analytics_info.setObjectName("hint")
        bar.addWidget(self.analytics_info)
        bar.addStretch()
        refresh_btn = QPushButton("↻  Refresh")
        refresh_btn.clicked.connect(self._refresh_analytics)
        bar.addWidget(refresh_btn)
        ext_btn = QPushButton("Open in browser")
        ext_btn.clicked.connect(self._open_dashboard_browser)
        bar.addWidget(ext_btn)
        bw = QWidget(); bw.setLayout(bar); v.addWidget(bw)

        if HAS_WEBENGINE:
            self.dash_view = QWebEngineView()
            self.dash_view.setStyleSheet("background:#0d0d0d;")
            self.dash_view.loadFinished.connect(
                lambda ok: ok and self.dash_view.page().runJavaScript(_DASH_DARK_JS))
            v.addWidget(self.dash_view, stretch=1)
        else:
            self.dash_view = None
            missing = QLabel(
                "In-app dashboard needs the PyQt6-WebEngine package:\n\n"
                "    pip install PyQt6-WebEngine\n\n"
                "Until then, use 'Open in browser' above.")
            missing.setAlignment(Qt.AlignmentFlag.AlignCenter)
            missing.setStyleSheet("color:#888; font-family:Consolas;")
            v.addWidget(missing, stretch=1)
        return page

    def _on_tab_changed(self, idx):
        self._refresh_cctv_combo()
        if idx == self.TAB_ANALYTICS:
            self._refresh_analytics()

    def _generate_dashboard(self):
        """Build the dashboard HTML; returns (path, session_count) or None."""
        try:
            from src.analytics import generate_dashboard
            from src.analytics.recorder import load_sessions
            n = len(load_sessions())
            path = generate_dashboard(open_browser=False)
            return path, n
        except Exception as e:
            QMessageBox.critical(self, "Analytics", f"Could not build dashboard:\n{e}")
            return None

    def _refresh_analytics(self):
        if not self.dash_view:
            return
        built = self._generate_dashboard()
        if not built:
            return
        path, n = built
        self.analytics_info.setText(
            f"{n} session(s) · regenerated {datetime.now():%H:%M:%S}")
        self.dash_view.load(QUrl.fromLocalFile(str(Path(path).resolve())))

    def _open_dashboard(self):
        """Monitor-tab button: show the dashboard inside the app when possible."""
        if self.dash_view:
            self.tabs.setCurrentIndex(self.TAB_ANALYTICS)  # triggers refresh
        else:
            self._open_dashboard_browser()

    def _open_dashboard_browser(self):
        built = self._generate_dashboard()
        if built:
            webbrowser.open(Path(built[0]).as_uri())

    # ---- cctv tab ----

    def _build_cctv_tab(self):
        page = QWidget(); v = QVBoxLayout(page)
        v.addWidget(self._section("CCTV camera manager"))
        v.addWidget(QLabel("Register RTSP/HTTP cameras once — they become Live Monitor sources.\n"
                           "Typical URL: rtsp://user:pass@192.168.1.64:554/stream1"))
        add_row = QHBoxLayout()
        self.cam_name_edit = QLineEdit(); self.cam_name_edit.setPlaceholderText("Camera name")
        self.cam_url_edit = QLineEdit(); self.cam_url_edit.setPlaceholderText("Stream URL")
        add_btn = QPushButton("Add"); add_btn.clicked.connect(self._add_camera)
        add_row.addWidget(self.cam_name_edit, 2); add_row.addWidget(self.cam_url_edit, 4)
        add_row.addWidget(add_btn)
        aw = QWidget(); aw.setLayout(add_row); v.addWidget(aw)

        self.cctv_list = QVBoxLayout()
        list_wrap = QWidget(); list_wrap.setLayout(self.cctv_list)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(list_wrap)
        v.addWidget(scroll, stretch=1)

        self.cctv_preview = QLabel(); self.cctv_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cctv_preview.setFixedHeight(260)
        self.cctv_preview.setStyleSheet("background:#000; border:1px solid #3a3a3a;")
        v.addWidget(self.cctv_preview)
        self._refresh_camera_list()
        return page

    # ---- small UI builders ----

    def _section(self, text):
        lbl = QLabel(text); lbl.setObjectName("section")
        return lbl

    def _group(self, parent_layout, title):
        """Add a titled group box and return its inner layout."""
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.setSpacing(8)
        parent_layout.addWidget(box)
        return lay

    def _slider(self, parent_layout, label, lo, hi, init, factor=1):
        head = QHBoxLayout()
        name = QLabel(label)
        value = QLabel(str(init / factor if factor != 1 else init))
        value.setStyleSheet("color:#f59e0b; font-weight:600;")
        head.addWidget(name); head.addStretch(); head.addWidget(value)
        hw = QWidget(); hw.setLayout(head); parent_layout.addWidget(hw)
        s = QSlider(Qt.Orientation.Horizontal); s.setRange(lo, hi); s.setValue(init)
        s.valueChanged.connect(
            lambda v: value.setText(str(round(v / factor, 2) if factor != 1 else v)))
        parent_layout.addWidget(s)
        s._factor = factor
        return s

    def _sval(self, slider):
        return slider.value() / slider._factor if slider._factor != 1 else slider.value()

    # ---- source switching ----

    def _current_source(self):
        for b in self.src_group.buttons():
            if b.isChecked():
                return b.property("srckey")
        return SRC_VIDEO

    def _on_source_change(self):
        src = self._current_source()
        self.video_box.setVisible(src == SRC_VIDEO)
        self.webcam_box.setVisible(src == SRC_WEBCAM)
        self.cctv_box.setVisible(src == SRC_CCTV)
        if src == SRC_CCTV:
            self._refresh_cctv_combo()

    def _refresh_cctv_combo(self):
        if not hasattr(self, "cctv_combo"):
            return
        cur = self.cctv_combo.currentText()
        self.cctv_combo.clear()
        cams = load_cameras()
        self.cctv_combo.addItem("— none —", None)
        for c in cams:
            self.cctv_combo.addItem(c["name"], c["url"])
        idx = self.cctv_combo.findText(cur)
        if idx >= 0:
            self.cctv_combo.setCurrentIndex(idx)

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.webm *.wmv);;All files (*.*)")
        if path:
            self.path_edit.setText(path)
            self.path_label.setText(Path(path).name)
            self.path_label.setStyleSheet("color:#22c55e;")

    # ---- ESP read ----

    def _read_distance(self):
        ip = self.esp_edit.text().strip()
        if not ip:
            QMessageBox.warning(self, "ESP32", "Enter the ESP32 IP first.")
            return
        self.status_label.setText("Reading ESP32 distance…")
        self._esp_thread = EspReadThread(ip)
        self._esp_thread.done.connect(lambda t: self.status_label.setText(f"ESP32 distance: {t}"))
        self._esp_thread.start()

    # ---- CCTV management ----

    def _add_camera(self):
        name, url = self.cam_name_edit.text().strip(), self.cam_url_edit.text().strip()
        if not name or not url:
            QMessageBox.warning(self, "CCTV", "Both a name and a URL are required.")
            return
        cams = load_cameras()
        if any(c["name"] == name for c in cams):
            QMessageBox.warning(self, "CCTV", "A camera with that name already exists.")
            return
        cams.append({"name": name, "url": url}); save_cameras(cams)
        self.cam_name_edit.clear(); self.cam_url_edit.clear()
        self._refresh_camera_list(); self._refresh_cctv_combo()

    def _refresh_camera_list(self):
        while self.cctv_list.count():
            item = self.cctv_list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        cams = load_cameras()
        if not cams:
            self.cctv_list.addWidget(QLabel("No cameras registered yet."))
            return
        for i, cam in enumerate(cams):
            row = QHBoxLayout()
            row.addWidget(QLabel(f"<b>{cam['name']}</b>"), 2)
            url = QLabel(cam["url"]); url.setStyleSheet("color:#0ea5e9; font-family:Consolas;")
            row.addWidget(url, 4)
            test = QPushButton("Test"); test.clicked.connect(lambda _, c=cam: self._test_camera(c))
            rm = QPushButton("Remove"); rm.clicked.connect(lambda _, idx=i: self._remove_camera(idx))
            row.addWidget(test); row.addWidget(rm)
            w = QWidget(); w.setLayout(row)
            self.cctv_list.addWidget(w)

    def _test_camera(self, cam):
        self.cctv_preview.setText(f"Connecting to {cam['name']}…")
        self._grab_thread = GrabThread(cam["url"], cam["name"])
        self._grab_thread.done.connect(self._on_camera_tested)
        self._grab_thread.start()

    def _on_camera_tested(self, ok, qimg, name):
        if not ok:
            self.cctv_preview.setText(f"“{name}” offline — check URL / credentials / network")
        else:
            pix = QPixmap.fromImage(qimg).scaled(
                self.cctv_preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.cctv_preview.setPixmap(pix)

    def _remove_camera(self, idx):
        cams = load_cameras()
        if 0 <= idx < len(cams):
            cams.pop(idx); save_cameras(cams)
            self._refresh_camera_list(); self._refresh_cctv_combo()

    # ---- run / stop ----

    def _resolve_source(self):
        src = self._current_source()
        if src == SRC_VIDEO:
            path = self.path_edit.text().strip().strip('"')
            if not path or not Path(path).exists():
                QMessageBox.warning(self, "Video", "Select or paste a valid video path.")
                return None
            return src, path, False, Path(path).name
        if src == SRC_WEBCAM:
            return src, int(self.cam_spin.value()), True, f"webcam #{self.cam_spin.value()}"
        if src == SRC_ESP:
            ip = self.esp_edit.text().strip()
            if not ip:
                QMessageBox.warning(self, "ESP32-CAM", "Enter the ESP32 IP in section 5.")
                return None
            return src, None, True, f"ESP32-CAM @ {ip}"
        # cctv
        url = self.cctv_combo.currentData() or self.cctv_manual.text().strip()
        if not url:
            QMessageBox.warning(self, "CCTV", "Pick a registered camera or paste a stream URL.")
            return None
        return src, url, True, "CCTV stream"

    def _start(self):
        resolved = self._resolve_source()
        if not resolved:
            return
        src, open_args, is_live, desc = resolved

        model = self.model_combo.currentData()
        cw = self.custom_edit.text().strip().strip('"')
        if cw:
            if not Path(cw).exists():
                QMessageBox.critical(self, "Weights", f"Custom weights not found:\n{cw}")
                return
            model = cw

        risk = self._sval(self.risk_slider)
        alert_type = "hr" if self.rb_hr.isChecked() else "normal"
        eff_risk = max(0.10, risk - self._hr_drop) if alert_type == "hr" else risk

        meta = {
            "src_type": src, "open_args": open_args, "is_live": is_live,
            "source_desc": desc, "model": model,
            "pose": self._pose_model if self.pose_check.isChecked() else None,
            "alert_type": alert_type, "eff_risk": eff_risk,
            "det_conf": self._sval(self.conf_slider),
            "sustain": int(self._sval(self.sustain_slider)),
            "skip": int(self._sval(self.skip_slider)),
            "save": self.save_check.isChecked() and not is_live,
            "esp_ip": self.esp_edit.text().strip(),
            "esp_poll": self.esp_poll_check.isChecked() and bool(self.esp_edit.text().strip()),
        }

        self.alerts = []
        self.log_text.clear()
        self.export_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.fwd_btn.setEnabled(src == SRC_VIDEO)
        self.status_label.setText(f"Monitoring — {desc} · {Path(model).name} · {alert_type.upper()}")

        self.thread = MonitorThread(meta, self.cfg)
        self.thread.frameReady.connect(self._on_frame)
        self.thread.statsReady.connect(self._on_stats)
        self.thread.logMsg.connect(self._log)
        self.thread.alertMsg.connect(self._on_alert)
        self.thread.computeMsg.connect(self._on_compute)
        self.thread.finishedRun.connect(self._on_finished)
        self.thread.errorMsg.connect(self._on_error)
        self.thread.start()

    def _stop(self):
        if self.thread:
            self.thread.stop()
        self.stop_btn.setEnabled(False)
        self.fwd_btn.setEnabled(False)

    # ---- worker signal slots ----

    def _on_frame(self, qimg):
        self._last_qimg = qimg
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def _on_stats(self, s):
        self.stat_widgets["frames"].setText(f"{s['frames']:,}")
        self.stat_widgets["persons"].setText(str(s["persons"]))
        self.stat_widgets["dogs"].setText(str(s["dogs"]))
        self.stat_widgets["alerts"].setText(str(s["alerts"]))
        self.stat_widgets["fps"].setText(f"{s['fps']:.1f}")
        self.stat_widgets["dist"].setText(s["dist"])

    def _on_compute(self, text):
        warn = text.startswith("CPU")
        self.compute_label.setText(("⚠ " if warn else "⚡ ") + text)
        self.compute_label.setStyleSheet("color:#f87171;" if warn else "color:#22c55e;")

    def _on_alert(self, entry):
        self.alerts.append(entry)
        prefix = "[HR ALERT]" if entry["alert_type"] == "hr" else "[ALERT]"
        self._log(f"[{entry['time']}] {prefix} dog#{entry['track_id']} "
                  f"risk={entry['risk']:.2f}  frame {entry['frame']}")
        if self.sound_check.isChecked():
            self._play_sound(entry["alert_type"] == "hr")
        if entry["alert_type"] == "hr":
            self.video_label.setStyleSheet("background:#000; border:3px solid #dc2626;")
            QTimer.singleShot(500, lambda: self.video_label.setStyleSheet(
                "background:#000; border:3px solid #1a1a1a;"))

    def _on_finished(self, summary):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fwd_btn.setEnabled(False)
        if summary["alerts"]:
            self.export_btn.setEnabled(True)
        verb = "Stopped" if summary["stopped"] else "Complete"
        self.status_label.setText(
            f"{verb} — {summary['frames']:,} frames · "
            f"{len(summary['alerts'])} alert(s) · peak {summary['peak_dogs']} dogs")
        if summary.get("output_path"):
            self._log(f"Saved: {summary['output_path']}")

    def _on_error(self, msg):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fwd_btn.setEnabled(False)
        self._log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Monitoring failed", msg)

    # ---- misc ----

    def _log(self, msg):
        self.log_text.append(msg)

    def _play_sound(self, hr):
        def _beep():
            if _winsound:
                if hr:
                    for _ in range(3):
                        _winsound.Beep(1500, 250); time.sleep(0.05)
                else:
                    _winsound.Beep(1000, 600)
        if _winsound:
            threading.Thread(target=_beep, daemon=True).start()
        else:
            QApplication.beep()

    def _export_alerts(self):
        if not self.alerts:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export alerts", f"straydog_alerts_{datetime.now():%Y%m%d_%H%M%S}.json",
            "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.alerts, f, indent=2)
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(2000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
