"""
Stray Dog Monitoring System — native desktop app (PyQt6).

Fully local: no browser, no web server, no page loads.

  - Live Monitor ....... Video / Webcam / ESP32-CAM / CCTV (RTSP) sources,
                         YOLO26 + YOLO11 + custom fine-tuned weights,
                         Normal / HR alerts, live annotated video with a
                         per-dog BEHAVIOR label (idle / roaming / approaching /
                         charging / lunging / pack …), stats, alert log,
                         annotated-video saving, +10 s skip
  - Dashboard .......... native PyQt charts (no web view): updates in real
                         time while monitoring, with a live risk panel fed
                         straight from the running pipeline
  - CCTV Cameras ....... register / test / remove RTSP cameras
  - Logs ............... timestamped application log, mirrored to
                         data/logs/app_YYYYMMDD.log

The left control panel uses collapsible sections (chevron headers). Detection
runs in a background QThread; live sources use the threaded latest-frame
reader so the feed never lags behind reality.

Run locally:
    python app3.py
"""

import os
import sys
import json
import time
import logging
import threading
import urllib.request
import webbrowser
from collections import deque
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

from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, QTimer, QRectF, QPointF
from PyQt6.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen, QBrush, QPainterPath
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QSlider, QCheckBox, QRadioButton, QLineEdit, QTextEdit, QSpinBox,
    QFileDialog, QMessageBox, QTabWidget, QScrollArea, QVBoxLayout,
    QHBoxLayout, QGridLayout, QGroupBox, QFrame, QSizePolicy, QButtonGroup,
    QToolButton, QTableWidget, QTableWidgetItem, QHeaderView,
)

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

IMGSZ_OPTIONS = [
    (640, "640 — accurate (default)"),
    (512, "512 — faster"),
    (416, "416 — fast"),
    (320, "320 — fastest (CPU rescue)"),
]

SRC_VIDEO, SRC_WEBCAM, SRC_ESP, SRC_CCTV = "video", "webcam", "espcam", "cctv"
CAMERAS_FILE = PROJECT_ROOT / "data" / "cameras.json"
LOG_DIR = PROJECT_ROOT / "data" / "logs"

# chart palette (dark)
C_BLUE = QColor("#3987e5")
C_AQUA = QColor("#199e70")
C_RED = QColor("#d03b3b")
C_GRID = QColor("#2f2f2d")
C_BASE = QColor("#3f3f3c")
C_TXT = QColor("#c3c2b7")
C_MUTED = QColor("#898781")
C_CARD = QColor("#262626")

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
QToolButton#secthead { background: transparent; border: none; color: #f59e0b;
                       font-weight: 600; font-size: 13px; padding: 8px 4px; text-align: left; }
QToolButton#secthead:hover { color: #ffc14d; }
QFrame#sectcard { background: #262626; border: 1px solid #383838; border-radius: 8px; }
QFrame#sectcard QWidget { background: #262626; }
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
QTableWidget { background: #1f1f1f; border: 1px solid #383838; border-radius: 6px;
               gridline-color: #2f2f2d; }
QHeaderView::section { background: #262626; color: #c3c2b7; border: none;
                       border-bottom: 1px solid #3f3f3c; padding: 6px; font-weight: 600; }
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
QLabel#kpival { font-size: 22px; font-weight: 700; }
QLabel#kpilabel { color: #999; font-size: 11px; }
QLabel#kpisub { color: #777; font-size: 10px; }
QFrame#kpitile { background: #262626; border: 1px solid #383838; border-radius: 8px; }
QFrame#kpitile QLabel { background: #262626; }
"""


# ── logging ───────────────────────────────────────────────────────────

class QtLogBridge(QObject):
    message = pyqtSignal(str)


class _GuiLogHandler(logging.Handler):
    def __init__(self, bridge):
        super().__init__()
        self.bridge = bridge

    def emit(self, record):
        try:
            self.bridge.message.emit(self.format(record))
        except Exception:
            pass


def setup_logging(bridge):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("straydog")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                                "%H:%M:%S")
        fh = logging.FileHandler(
            LOG_DIR / f"app_{datetime.now():%Y%m%d}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(message)s", "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)
        gh = _GuiLogHandler(bridge)
        gh.setFormatter(fmt)
        logger.addHandler(gh)
    return logger


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


def sess_label(sid):
    """'20260702_120000' -> '07-02 12:00'"""
    try:
        return f"{sid[4:6]}-{sid[6:8]} {sid[9:11]}:{sid[11:13]}"
    except Exception:
        return sid


def mmss(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def peak_dogs(s):
    return max((p["dogs"] for p in s.get("timeline", [])), default=0)


def peak_persons(s):
    return max((p["persons"] for p in s.get("timeline", [])), default=0)


# ── collapsible section (chevron header, like a proper sidebar) ───────

class CollapsibleSection(QWidget):
    def __init__(self, title, expanded=True):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.header = QToolButton()
        self.header.setObjectName("secthead")
        self.header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self._title = title
        self.header.clicked.connect(self._toggle)
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Fixed)
        outer.addWidget(self.header)

        self.card = QFrame()
        self.card.setObjectName("sectcard")
        self.body = QVBoxLayout(self.card)
        self.body.setSpacing(8)
        self.body.setContentsMargins(10, 10, 10, 10)
        outer.addWidget(self.card)

        self._apply()

    def _toggle(self):
        self._apply()

    def _apply(self):
        open_ = self.header.isChecked()
        self.card.setVisible(open_)
        arrow = "▾" if open_ else "▸"
        self.header.setText(f"{arrow}  {self._title}")

    def layout_(self):
        return self.body


# ── native chart widget (QPainter — no web view) ─────────────────────

class ChartWidget(QWidget):
    """Line / grouped-bar / horizontal-bar chart painted natively."""

    ML, MR, MT, MB = 46, 14, 12, 24

    def __init__(self, height=210):
        super().__init__()
        self.setMinimumHeight(height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._mode = None
        self._empty_text = "No data yet"

    # -- data setters --------------------------------------------------
    def set_line(self, xs, series, threshold=None, markers=None,
                 y_max=None, xfmt=None, pct=False):
        self._mode = "line"
        self._xs, self._series = list(xs), series
        self._threshold, self._markers = threshold, markers or []
        self._ymax = y_max
        self._xfmt = xfmt or (lambda v: str(v))
        self._pct = pct
        self.update()

    def set_bars(self, cats, series, pct=False, int_ticks=False,
                 catfmt=None, tick_every=1):
        self._mode = "bars"
        self._cats, self._series = list(cats), series
        self._pct, self._int = pct, int_ticks
        self._catfmt = catfmt or (lambda v: str(v))
        self._tick_every = max(1, tick_every)
        self.update()

    def set_hbars(self, labels, values, vmax=1.0, color=C_BLUE, fmt=None):
        self._mode = "hbars"
        self._labels, self._values = list(labels), list(values)
        self._vmax = max(vmax, 1e-6)
        self._color = color
        self._fmt = fmt or (lambda v: f"{v:.2f}")
        self.setMinimumHeight(28 * max(1, len(self._labels)) + 20)
        self.update()

    def clear(self, text="No data yet"):
        self._mode = None
        self._empty_text = text
        self.update()

    # -- painting ------------------------------------------------------
    def _ytick(self, v, ymax):
        if self._pct if hasattr(self, "_pct") else False:
            return f"{int(round(v * 100))}%"
        if ymax <= 2:
            return f"{v:.2f}"
        return f"{int(round(v)):,}"

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), C_CARD)
        w, h = self.width(), self.height()

        if self._mode is None:
            p.setPen(C_MUTED)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._empty_text)
            return
        if self._mode == "hbars":
            self._paint_hbars(p, w, h)
            return

        iw = w - self.ML - self.MR
        ih = h - self.MT - self.MB
        if iw < 40 or ih < 30:
            return

        if self._mode == "line":
            vals = [v for s in self._series for v in s["values"]] or [0]
        else:
            vals = [v for s in self._series for v in s["values"]] or [0]
        ymax = self._ymax if getattr(self, "_ymax", None) else None
        if ymax is None:
            m = max(vals + [1e-6])
            if getattr(self, "_pct", False):
                ymax = 1.0
            elif getattr(self, "_int", False):
                ymax = max(4, int(np.ceil(m / 4) * 4))
            else:
                ymax = m * 1.15

        # grid + y labels
        font = QFont(self.font()); font.setPointSize(8)
        p.setFont(font)
        for i in range(5):
            y = self.MT + ih - ih * i / 4
            p.setPen(QPen(C_GRID if i else C_BASE, 1))
            p.drawLine(QPointF(self.ML, y), QPointF(self.ML + iw, y))
            p.setPen(C_MUTED)
            p.drawText(QRectF(0, y - 8, self.ML - 6, 16),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       self._ytick(ymax * i / 4, ymax))

        if self._mode == "line":
            self._paint_line(p, iw, ih, ymax)
        else:
            self._paint_bars(p, iw, ih, ymax)

        # legend for 2+ series
        if len(self._series) > 1:
            lx = self.ML + iw
            for s in reversed(self._series):
                name = s["name"]
                tw = p.fontMetrics().horizontalAdvance(name)
                lx -= tw + 26
                p.setPen(C_TXT)
                p.drawText(QPointF(lx + 16, self.MT + 4), name)
                p.fillRect(QRectF(lx, self.MT - 3, 10, 10), s["color"])

    def _paint_line(self, p, iw, ih, ymax):
        n = len(self._xs)
        if n < 2:
            return
        X = lambda i: self.ML + iw * i / (n - 1)
        Y = lambda v: self.MT + ih - min(1.0, v / ymax) * ih

        if self._threshold is not None and self._threshold <= ymax:
            ty = Y(self._threshold)
            p.setPen(QPen(C_RED, 1, Qt.PenStyle.SolidLine))
            p.setOpacity(0.65)
            p.drawLine(QPointF(self.ML, ty), QPointF(self.ML + iw, ty))
            p.setOpacity(1.0)

        for s in self._series:
            pts = [QPointF(X(i), Y(v)) for i, v in enumerate(s["values"])]
            if len(self._series) == 1:      # area wash for a single series
                path = QPainterPath()
                path.moveTo(pts[0])
                for pt in pts[1:]:
                    path.lineTo(pt)
                path.lineTo(QPointF(X(n - 1), self.MT + ih))
                path.lineTo(QPointF(X(0), self.MT + ih))
                wash = QColor(s["color"]); wash.setAlpha(26)
                p.fillPath(path, wash)
            p.setPen(QPen(s["color"], 2, Qt.PenStyle.SolidLine,
                          Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            for i in range(1, len(pts)):
                p.drawLine(pts[i - 1], pts[i])
            # end dot with card-colored ring
            end = pts[-1]
            p.setBrush(QBrush(s["color"]))
            p.setPen(QPen(C_CARD, 2))
            p.drawEllipse(end, 5, 5)
            p.setBrush(Qt.BrushStyle.NoBrush)

        for (mi, mv) in self._markers:
            if 0 <= mi < n:
                p.setBrush(QBrush(C_RED))
                p.setPen(QPen(C_CARD, 2))
                p.drawEllipse(QPointF(X(mi), Y(mv)), 4.5, 4.5)
                p.setBrush(Qt.BrushStyle.NoBrush)

        # x tick labels (up to 6)
        p.setPen(C_MUTED)
        ticks = min(6, n)
        for t in range(ticks):
            i = round(t * (n - 1) / max(1, ticks - 1))
            p.drawText(QRectF(X(i) - 34, self.MT + ih + 4, 68, 16),
                       Qt.AlignmentFlag.AlignHCenter,
                       self._xfmt(self._xs[i]))

    def _paint_bars(self, p, iw, ih, ymax):
        n = len(self._cats)
        if n == 0:
            return
        ns = len(self._series)
        band = iw / n
        bw = min(24.0, max(3.0, (band - 8) / ns - 2))
        gw = bw * ns + 2 * (ns - 1)
        for ci in range(n):
            gx = self.ML + band * ci + (band - gw) / 2
            for si, s in enumerate(self._series):
                v = s["values"][ci]
                bh = min(1.0, v / ymax) * ih if ymax > 0 else 0
                if bh <= 0:
                    continue
                x = gx + si * (bw + 2)
                y = self.MT + ih - bh
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QBrush(s["color"]))
                p.drawRoundedRect(QRectF(x, y, bw, bh), 3, 3)
                if bh > 5:   # square the baseline end
                    p.drawRect(QRectF(x, self.MT + ih - min(3.0, bh), bw,
                                      min(3.0, bh)))
            if ci % self._tick_every == 0:
                p.setPen(C_MUTED)
                p.drawText(QRectF(self.ML + band * ci, self.MT + ih + 4,
                                  band, 16),
                           Qt.AlignmentFlag.AlignHCenter,
                           self._catfmt(self._cats[ci]))

    def _paint_hbars(self, p, w, h):
        rows = len(self._labels)
        if rows == 0:
            return
        label_w = 150
        row_h = min(30.0, (h - 10) / rows)
        iw = w - label_w - 64
        font = QFont(self.font()); font.setPointSize(8)
        p.setFont(font)
        for i, (lab, val) in enumerate(zip(self._labels, self._values)):
            y = 8 + i * row_h
            p.setPen(C_TXT)
            p.drawText(QRectF(0, y, label_w - 8, row_h),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       str(lab)[:24])
            bw = max(2.0, iw * min(1.0, val / self._vmax))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(self._color))
            p.drawRoundedRect(QRectF(label_w, y + 4, bw, row_h - 10), 3, 3)
            p.setPen(QColor("#e8e8e8"))
            p.drawText(QRectF(label_w + bw + 6, y, 56, row_h),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       self._fmt(val))
        p.setPen(QPen(C_BASE, 1))
        p.drawLine(QPointF(label_w, 4), QPointF(label_w, 8 + rows * row_h - 4))


# ── background workers ────────────────────────────────────────────────

class GrabThread(QThread):
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
    """Capture + inference loop, off the UI thread."""
    frameReady = pyqtSignal(object)      # QImage (already display-sized)
    statsReady = pyqtSignal(dict)
    logMsg = pyqtSignal(str)
    alertMsg = pyqtSignal(dict)
    computeMsg = pyqtSignal(str)
    finishedRun = pyqtSignal(dict)
    errorMsg = pyqtSignal(str)

    PREVIEW_W = 1100    # downscale frames before crossing the thread boundary

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
            self.logMsg.emit(f"Loading {m['model']} (imgsz={m['imgsz']}) …")
            pipeline = StrayDogMonitor(
                detector_path=m["model"], pose_model=m["pose"],
                det_conf=m["det_conf"], risk_threshold=m["eff_risk"],
                sustain_frames=m["sustain"], imgsz=m["imgsz"], config=cfg)
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
                    f"{datetime.now():%Y%m%d_%H%M%S}.mp4"))
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (width, height))

            skip = m["skip"]
            # Real-time playback (video files only): pace to the source FPS and
            # DROP frames the pipeline can't keep up with, so the video always
            # plays at true speed instead of crawling frame-by-frame. Live
            # sources already do this via LatestFrameCapture. When off, every
            # frame is processed (thorough analysis, but as fast as hardware).
            realtime = bool(m.get("realtime", True)) and not is_live
            frame_count = processed = total_alerts = 0
            peak_dogs_seen = peak_persons_seen = 0
            alerts = []
            start_time = time.time()
            rt0 = time.time()          # wall-clock origin for real-time pacing
            last_emit = last_esp = 0.0
            dist_txt = "–"

            while not self._stop:
                if not is_live:
                    with self._seek_lock:
                        if self._seek_seconds > 0:
                            if realtime:
                                rt0 -= self._seek_seconds   # shift timeline; pacing seeks
                            else:
                                frame_count += int(self._seek_seconds * fps)
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                            self._seek_seconds = 0

                if realtime:
                    target = int((time.time() - rt0) * fps)
                    if target < frame_count:
                        time.sleep(0.003)      # ahead of schedule — wait for the clock
                        continue
                    if target > frame_count:   # behind — jump ahead, dropping frames
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                        frame_count = target
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                else:
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
                peak_persons_seen = max(peak_persons_seen, persons_now)
                peak_dogs_seen = max(peak_dogs_seen, dogs_now)
                cur_risk = max((r["risk"] for r in results), default=0.0)
                top = max(results, key=lambda r: r["risk"], default=None)
                top_behavior = top.get("behavior_label", "") if top else ""

                if recorder:
                    recorder.record_frame(results, persons_now, frame_count)

                new_alerts = [r for r in results if r.get("new_alert")]
                if new_alerts:
                    total_alerts += len(new_alerts)
                    ts = frame_count / fps
                    for a in new_alerts:
                        entry = {
                            "time": mmss(ts), "frame": frame_count,
                            "track_id": a["track_id"],
                            "risk": round(a["risk"], 3),
                            "behavior": a.get("behavior_label", ""),
                            "model": m["model"],
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
                    disp = annotated
                    dh, dw = disp.shape[:2]
                    if dw > self.PREVIEW_W:      # display copy only; writer got full-res
                        disp = cv2.resize(
                            disp, (self.PREVIEW_W, int(dh * self.PREVIEW_W / dw)),
                            interpolation=cv2.INTER_AREA)
                    self.frameReady.emit(bgr_to_qimage(disp))
                    self.statsReady.emit({
                        "frames": frame_count, "persons": persons_now,
                        "dogs": dogs_now, "alerts": total_alerts,
                        "fps": cur_fps, "dist": dist_txt,
                        "risk": cur_risk, "behavior": top_behavior,
                        "t": elapsed,
                        "progress": (frame_count / total_frames) if total_frames else 0.0})

            if recorder:
                recorder.finalize(save=True)

            self.finishedRun.emit({
                "stopped": self._stop, "alerts": alerts, "output_path": output_path,
                "frames": processed, "peak_dogs": peak_dogs_seen,
                "model": m["model"], "alert_type": m["alert_type"]})

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
        self.resize(1400, 920)

        self.thread = None
        self.alerts = []
        self._live = deque(maxlen=240)   # (t, risk) from the running monitor
        self._sess_cache = []
        self._sess_mtime = -1.0

        self.log_bridge = QtLogBridge()
        self.logger = setup_logging(self.log_bridge)

        d, r, hw, al, inf = (self.cfg["detector"], self.cfg["risk"],
                             self.cfg["hardware"], self.cfg["alerts"],
                             self.cfg["inference"])
        self._pose_model = d.get("pose_model")
        self._hr_drop = al.get("hr_threshold_drop", 0.10)
        self._defaults = dict(d=d, r=r, hw=hw, al=al, inf=inf)

        tabs = QTabWidget()
        tabs.addTab(self._build_monitor_tab(), "🎥  Live Monitor")
        tabs.addTab(self._build_dashboard_tab(), "📊  Dashboard")
        tabs.addTab(self._build_cctv_tab(), "🎦  CCTV Cameras")
        tabs.addTab(self._build_logs_tab(), "🧾  Logs")
        self.tabs = tabs
        self.TAB_DASH = 1
        tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(tabs)

        self.log_bridge.message.connect(self._append_log_line)

        # real-time dashboard sync: refresh while the dashboard tab is open
        self._dash_timer = QTimer(self)
        self._dash_timer.setInterval(2000)
        self._dash_timer.timeout.connect(self._dash_tick)
        self._dash_timer.start()

        self.logger.info("Application started")

    # ══════════════════════ monitor tab ══════════════════════

    def _build_monitor_tab(self):
        d, r, hw, inf = (self._defaults["d"], self._defaults["r"],
                         self._defaults["hw"], self._defaults["inf"])
        page = QWidget()
        root = QHBoxLayout(page)

        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setSpacing(4)
        pl.setContentsMargins(10, 6, 14, 8)

        # — Source —
        sec = CollapsibleSection("1 · Source", expanded=True)
        g = sec.layout_()
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

        self.video_box = QWidget(); vlay = QVBoxLayout(self.video_box)
        vlay.setContentsMargins(0, 4, 0, 0); vlay.setSpacing(6)
        browse = QPushButton("Browse local video…"); browse.clicked.connect(self._browse_video)
        vlay.addWidget(browse)
        self.path_edit = QLineEdit(); self.path_edit.setPlaceholderText(r"…or paste C:\videos\clip.mp4")
        vlay.addWidget(self.path_edit)
        self.path_label = QLabel("No video selected"); self.path_label.setObjectName("hint")
        vlay.addWidget(self.path_label)
        g.addWidget(self.video_box)

        self.webcam_box = QWidget(); wlay = QHBoxLayout(self.webcam_box)
        wlay.setContentsMargins(0, 4, 0, 0)
        wlay.addWidget(QLabel("Camera index:"))
        self.cam_spin = QSpinBox(); self.cam_spin.setRange(0, 10)
        wlay.addWidget(self.cam_spin); wlay.addStretch()
        g.addWidget(self.webcam_box)

        self.cctv_box = QWidget(); clay = QVBoxLayout(self.cctv_box)
        clay.setContentsMargins(0, 4, 0, 0); clay.setSpacing(6)
        clay.addWidget(QLabel("Registered camera:"))
        self.cctv_combo = QComboBox(); clay.addWidget(self.cctv_combo)
        self.cctv_manual = QLineEdit(); self.cctv_manual.setPlaceholderText("…or rtsp://user:pass@ip:554/stream1")
        clay.addWidget(self.cctv_manual)
        g.addWidget(self.cctv_box)
        pl.addWidget(sec)

        # — Model —
        sec = CollapsibleSection("2 · Detection model", expanded=True)
        g = sec.layout_()
        self.model_combo = QComboBox()
        for k, v in MODEL_VARIANTS.items():
            self.model_combo.addItem(f"{k} — {v}", k)
        default = d["model"] if d["model"] in MODEL_VARIANTS else "yolo26n.pt"
        self.model_combo.setCurrentIndex(list(MODEL_VARIANTS).index(default))
        g.addWidget(self.model_combo)
        self.custom_edit = QLineEdit()
        self.custom_edit.setPlaceholderText(r"Custom weights .pt (overrides above)")
        g.addWidget(self.custom_edit)
        irow = QHBoxLayout()
        irow.addWidget(QLabel("Inference size:"))
        self.imgsz_combo = QComboBox()
        for val, lab in IMGSZ_OPTIONS:
            self.imgsz_combo.addItem(lab, val)
        cfg_imgsz = int(d.get("imgsz", 640))
        idx = next((i for i, (v, _) in enumerate(IMGSZ_OPTIONS) if v == cfg_imgsz), 0)
        self.imgsz_combo.setCurrentIndex(idx)
        irow.addWidget(self.imgsz_combo, 1)
        iw = QWidget(); iw.setLayout(irow); g.addWidget(iw)
        self.pose_check = QCheckBox("Use pose model (human skeleton)")
        self.pose_check.setChecked(bool(d.get("pose_model")))
        g.addWidget(self.pose_check)
        hint = QLabel("Slow? Use a smaller size + pose off. The compute line "
                      "under the video shows GPU vs CPU.")
        hint.setObjectName("hint"); hint.setWordWrap(True)
        g.addWidget(hint)
        pl.addWidget(sec)

        # — Alert type —
        sec = CollapsibleSection("3 · Alert type", expanded=False)
        g = sec.layout_()
        self.alert_group = QButtonGroup(self)
        row = QHBoxLayout()
        self.rb_normal = QRadioButton("Normal"); self.rb_normal.setChecked(True)
        self.rb_hr = QRadioButton("HR (high-risk)")
        self.alert_group.addButton(self.rb_normal); self.alert_group.addButton(self.rb_hr)
        row.addWidget(self.rb_normal); row.addWidget(self.rb_hr); row.addStretch()
        rw = QWidget(); rw.setLayout(row); g.addWidget(rw)
        self.sound_check = QCheckBox("Alert sound"); self.sound_check.setChecked(True)
        g.addWidget(self.sound_check)
        pl.addWidget(sec)

        # — Thresholds —
        sec = CollapsibleSection("4 · Thresholds", expanded=False)
        g = sec.layout_()
        self.conf_slider = self._slider(g, "Detection confidence", 10, 95,
                                        int(d["conf"] * 100), factor=100)
        self.risk_slider = self._slider(g, "Risk threshold", 10, 95,
                                        int(r["threshold"] * 100), factor=100)
        self.sustain_slider = self._slider(g, "Sustain frames (N)", 1, 20,
                                           int(r["sustain_frames"]))
        self.skip_slider = self._slider(g, "Skip frames", 1, 30,
                                        int(inf.get("skip_frames", 1)))
        hint = QLabel("Skip 2–4 speeds up videos; above ~5 degrades motion signals. "
                      "(Ignored in real-time mode — it drops frames automatically.)")
        hint.setObjectName("hint"); hint.setWordWrap(True)
        g.addWidget(hint)
        self.realtime_check = QCheckBox("Real-time playback (play at true speed, drop frames)")
        self.realtime_check.setChecked(True)
        g.addWidget(self.realtime_check)
        rt_hint = QLabel("On: video plays at actual speed even on a slow PC "
                         "(analyses fewer frames). Off: analyse every frame.")
        rt_hint.setObjectName("hint"); rt_hint.setWordWrap(True)
        g.addWidget(rt_hint)
        self.save_check = QCheckBox("Save annotated output video")
        self.save_check.setChecked(bool(inf.get("save_output", True)))
        g.addWidget(self.save_check)
        pl.addWidget(sec)

        # — ESP32 —
        sec = CollapsibleSection("5 · ESP32 sensor (HC-SR04)", expanded=False)
        g = sec.layout_()
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
        pl.addWidget(sec)

        # — Run —
        sec = CollapsibleSection("6 · Run", expanded=True)
        g = sec.layout_()
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
        pl.addWidget(sec)
        pl.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setFixedWidth(372)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        root.addWidget(scroll)

        # right: display + stats + alert log
        right = QVBoxLayout()
        self.video_label = QLabel("Video preview will appear here")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background:#000; border:3px solid #1a1a1a; color:#555;")
        self.video_label.setMinimumHeight(420)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right.addWidget(self.video_label, stretch=1)

        crow = QHBoxLayout()
        self.compute_label = QLabel(""); self.compute_label.setObjectName("hint")
        crow.addWidget(self.compute_label)
        crow.addStretch()
        self.behavior_label = QLabel(""); self.behavior_label.setStyleSheet(
            "color:#f59e0b; font-weight:600;")
        crow.addWidget(self.behavior_label)
        cw = QWidget(); cw.setLayout(crow); right.addWidget(cw)

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

        self.status_label = QLabel("Ready"); self.status_label.setObjectName("hint")
        right.addWidget(self.status_label)

        log_box = QGroupBox("Alert log")
        lv = QVBoxLayout(log_box)
        self.alert_text = QTextEdit(); self.alert_text.setReadOnly(True)
        self.alert_text.setFont(QFont("Consolas", 9)); self.alert_text.setFixedHeight(140)
        lv.addWidget(self.alert_text)
        self.export_btn = QPushButton("Export alerts (JSON)"); self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_alerts)
        lv.addWidget(self.export_btn)
        right.addWidget(log_box)

        rw2 = QWidget(); rw2.setLayout(right)
        root.addWidget(rw2, stretch=1)

        self._on_source_change()
        return page

    # ══════════════════════ dashboard tab (native) ══════════════════════

    def _build_dashboard_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(10, 10, 10, 6)
        outer.setSpacing(8)

        bar = QHBoxLayout()
        title = QLabel("Analytics"); title.setObjectName("section")
        bar.addWidget(title)
        self.dash_info = QLabel(""); self.dash_info.setObjectName("hint")
        bar.addWidget(self.dash_info)
        bar.addStretch()
        bar.addWidget(QLabel("Session:"))
        self.sess_combo = QComboBox()
        self.sess_combo.setMinimumWidth(230)
        self.sess_combo.currentIndexChanged.connect(
            lambda _: self._refresh_dashboard())
        bar.addWidget(self.sess_combo)
        refresh_btn = QPushButton("↻  Refresh")
        refresh_btn.clicked.connect(lambda: self._refresh_dashboard(force=True))
        bar.addWidget(refresh_btn)
        html_btn = QPushButton("Export HTML")
        html_btn.clicked.connect(self._export_html_dashboard)
        bar.addWidget(html_btn)
        bw = QWidget(); bw.setLayout(bar); outer.addWidget(bw)

        content = QWidget()
        cv_ = QVBoxLayout(content)
        cv_.setSpacing(10)
        cv_.setContentsMargins(2, 2, 14, 10)

        # KPI row
        kpi_row = QHBoxLayout(); kpi_row.setSpacing(10)
        self.kpi = {}
        for key, label in [("alerts", "Alerts fired"),
                           ("pdogs", "Peak dogs in frame"),
                           ("ppersons", "Peak persons in frame"),
                           ("frames", "Frames processed"),
                           ("prisk", "Peak risk")]:
            tile = QFrame(); tile.setObjectName("kpitile")
            tl = QVBoxLayout(tile); tl.setContentsMargins(12, 10, 12, 10); tl.setSpacing(2)
            lab = QLabel(label); lab.setObjectName("kpilabel")
            val = QLabel("0"); val.setObjectName("kpival")
            sub = QLabel(""); sub.setObjectName("kpisub")
            tl.addWidget(lab); tl.addWidget(val); tl.addWidget(sub)
            kpi_row.addWidget(tile, 1)
            self.kpi[key] = (val, sub)
        kw = QWidget(); kw.setLayout(kpi_row); cv_.addWidget(kw)

        # Live panel (real-time from the running monitor)
        live_box = QGroupBox("Live — current run (real time)")
        ll = QVBoxLayout(live_box)
        self.live_status = QLabel("Not monitoring"); self.live_status.setObjectName("hint")
        ll.addWidget(self.live_status)
        self.live_chart = ChartWidget(height=150)
        self.live_chart.clear("Start monitoring to see live risk")
        ll.addWidget(self.live_chart)
        cv_.addWidget(live_box)

        # Risk over time / per session
        self.risk_box = QGroupBox("Risk over time")
        rl = QVBoxLayout(self.risk_box)
        self.risk_sub = QLabel(""); self.risk_sub.setObjectName("hint")
        rl.addWidget(self.risk_sub)
        self.risk_chart = ChartWidget(height=230)
        rl.addWidget(self.risk_chart)
        cv_.addWidget(self.risk_box)

        # two-column charts
        grid = QGridLayout(); grid.setSpacing(10)
        self.det_chart = ChartWidget(height=200)
        self.hour_chart = ChartWidget(height=200)
        self.hist_chart = ChartWidget(height=200)
        self.sig_chart = ChartWidget(height=160)
        self.beh_chart = ChartWidget(height=160)
        for (title2, chart, pos) in [
            ("Detections — dogs vs persons", self.det_chart, (0, 0)),
            ("Alerts by hour of day", self.hour_chart, (0, 1)),
            ("Risk distribution (share of frames)", self.hist_chart, (1, 0)),
            ("Alert signals (avg at alert)", self.sig_chart, (1, 1)),
            ("Alert behaviors", self.beh_chart, (2, 0)),
        ]:
            box = QGroupBox(title2)
            bl = QVBoxLayout(box)
            bl.addWidget(chart)
            grid.addWidget(box, *pos)
        gw = QWidget(); gw.setLayout(grid); cv_.addWidget(gw)

        # sessions table
        tbox = QGroupBox("Sessions")
        tl2 = QVBoxLayout(tbox)
        self.sess_table = QTableWidget(0, 10)
        self.sess_table.setHorizontalHeaderLabels(
            ["Session", "Started", "Source", "Model", "Mode", "Frames",
             "Peak dogs", "Alerts", "Peak risk", "Avg FPS"])
        self.sess_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)
        self.sess_table.verticalHeader().setVisible(False)
        self.sess_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.sess_table.setMinimumHeight(220)
        tl2.addWidget(self.sess_table)
        cv_.addWidget(tbox)
        cv_.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll, stretch=1)
        return page

    # -- dashboard data ------------------------------------------------

    def _sessions(self, force=False):
        d = PROJECT_ROOT / "data" / "sessions"
        try:
            mt = max((p.stat().st_mtime for p in d.glob("session_*.json")),
                     default=0.0)
        except Exception:
            mt = 0.0
        if force or mt != self._sess_mtime:
            self._sess_mtime = mt
            from src.analytics.recorder import load_sessions
            self._sess_cache = load_sessions()
            self._repopulate_sess_combo()
        return self._sess_cache

    def _repopulate_sess_combo(self):
        cur = self.sess_combo.currentData()
        self.sess_combo.blockSignals(True)
        self.sess_combo.clear()
        self.sess_combo.addItem("All sessions", None)
        for s in reversed(self._sess_cache):
            self.sess_combo.addItem(
                f"{sess_label(s['session_id'])} · {Path(s['model']).stem} · "
                f"{s['alerts_total']} alert(s)", s["session_id"])
        if cur:
            i = self.sess_combo.findData(cur)
            if i >= 0:
                self.sess_combo.setCurrentIndex(i)
        self.sess_combo.blockSignals(False)

    def _refresh_dashboard(self, force=False):
        sessions = self._sessions(force)
        sel = self.sess_combo.currentData()
        S = [s for s in sessions if s["session_id"] == sel] if sel else sessions
        single = len(S) == 1 and sel is not None

        self.dash_info.setText(
            f"{len(S)} session(s) in view · updated {datetime.now():%H:%M:%S}")

        # KPIs
        alerts_total = sum(s["alerts_total"] for s in S)
        pk_d = max((peak_dogs(s) for s in S), default=0)
        pk_p = max((peak_persons(s) for s in S), default=0)
        frames = sum(s["frames_processed"] for s in S)
        dur = sum(s["duration_s"] for s in S)
        pk_r = max((s["peak_risk"] for s in S), default=0.0)
        avg_fps = (sum(s["avg_fps"] for s in S) / len(S)) if S else 0.0
        self.kpi["alerts"][0].setText(f"{alerts_total:,}")
        self.kpi["alerts"][1].setText("aggression alerts")
        self.kpi["pdogs"][0].setText(str(pk_d))
        self.kpi["pdogs"][1].setText("most seen at once")
        self.kpi["ppersons"][0].setText(str(pk_p))
        self.kpi["ppersons"][1].setText("most seen at once")
        self.kpi["frames"][0].setText(f"{frames:,}")
        self.kpi["frames"][1].setText(f"{mmss(dur)} monitored")
        self.kpi["prisk"][0].setText(f"{pk_r:.2f}")
        self.kpi["prisk"][1].setText(f"avg FPS {avg_fps:.1f}")

        # Risk chart
        if not S:
            self.risk_chart.clear("No sessions recorded yet — run the monitor")
            self.risk_sub.setText("")
        elif single:
            s = S[0]
            tl = s["timeline"]
            self.risk_box.setTitle("Risk over time")
            self.risk_sub.setText(
                f"Highest per-dog risk — session {s['session_id']} ({s['model']})")
            markers = []
            for a in s["alerts"]:
                idx = min(range(len(tl)), key=lambda i: abs(tl[i]["t"] - a["t"]))
                markers.append((idx, a["risk"]))
            self.risk_chart.set_line(
                [p["t"] for p in tl],
                [{"name": "risk", "color": C_BLUE,
                  "values": [p["risk"] for p in tl]}],
                threshold=s["risk_threshold"], markers=markers,
                y_max=1.0, xfmt=mmss)
        else:
            self.risk_box.setTitle("Peak risk per session")
            self.risk_sub.setText("Pick a single session above for its full timeline")
            self.risk_chart.set_bars(
                [s["session_id"] for s in S],
                [{"name": "peak risk", "color": C_BLUE,
                  "values": [s["peak_risk"] for s in S]}],
                pct=True, catfmt=sess_label,
                tick_every=max(1, len(S) // 8))

        # Detections
        if single:
            tl = S[0]["timeline"]
            self.det_chart.set_line(
                [p["t"] for p in tl],
                [{"name": "dogs", "color": C_BLUE,
                  "values": [p["dogs"] for p in tl]},
                 {"name": "persons", "color": C_AQUA,
                  "values": [p["persons"] for p in tl]}],
                xfmt=mmss)
        elif S:
            self.det_chart.set_bars(
                [s["session_id"] for s in S],
                [{"name": "peak dogs", "color": C_BLUE,
                  "values": [peak_dogs(s) for s in S]},
                 {"name": "peak persons", "color": C_AQUA,
                  "values": [peak_persons(s) for s in S]}],
                int_ticks=True, catfmt=sess_label,
                tick_every=max(1, len(S) // 6))
        else:
            self.det_chart.clear()

        # Alerts by hour
        hours = [0] * 24
        for s in S:
            try:
                h0 = int(s["started"][11:13])
                m0 = int(s["started"][14:16])
            except Exception:
                h0, m0 = 0, 0
            for a in s["alerts"]:
                hours[(h0 + int((m0 * 60 + a["t"]) // 3600)) % 24] += 1
        self.hour_chart.set_bars(
            list(range(24)),
            [{"name": "alerts", "color": C_BLUE, "values": hours}],
            int_ticks=True, tick_every=4, catfmt=lambda hh: f"{hh:02d}")

        # Risk distribution
        bins = [0] * 10
        total = 0
        for s in S:
            for pnt in s["timeline"]:
                bins[min(9, int(pnt["risk"] * 10))] += 1
                total += 1
        shares = [(b / total if total else 0.0) for b in bins]
        self.hist_chart.set_bars(
            list(range(10)),
            [{"name": "share", "color": C_BLUE, "values": shares}],
            pct=True, tick_every=2,
            catfmt=lambda i: f"{i/10:.1f}–{(i+1)/10:.1f}")

        # Alert signals
        all_alerts = [a for s in S for a in s["alerts"]]
        names = ["distance", "velocity", "posture", "human_pose"]
        labels = ["Distance", "Velocity", "Posture", "Human pose"]
        if all_alerts:
            avgs = [sum((a.get("features") or {}).get(n, 0) for a in all_alerts)
                    / len(all_alerts) for n in names]
            self.sig_chart.set_hbars(labels, avgs, vmax=1.0)
        else:
            self.sig_chart.clear("No alerts in scope")

        # Alert behaviors (from the behavior engine)
        counts = {}
        for a in all_alerts:
            b = a.get("behavior") or "unlabelled"
            counts[b] = counts.get(b, 0) + 1
        if counts:
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
            self.beh_chart.set_hbars(
                [k for k, _ in top], [v for _, v in top],
                vmax=max(v for _, v in top), color=C_RED,
                fmt=lambda v: str(int(v)))
        else:
            self.beh_chart.clear("No alerts in scope")

        # sessions table
        self.sess_table.setRowCount(len(sessions))
        for row, s in enumerate(reversed(sessions)):
            cells = [s["session_id"], s["started"].replace("T", " "),
                     s["source"], Path(s["model"]).name,
                     s["alert_type"].upper(), f"{s['frames_processed']:,}",
                     str(peak_dogs(s)), str(s["alerts_total"]),
                     f"{s['peak_risk']:.2f}", f"{s['avg_fps']:.1f}"]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.sess_table.setItem(row, col, item)

    def _dash_tick(self):
        if self.tabs.currentIndex() != self.TAB_DASH:
            return
        # live chart from the running monitor
        if self.thread and self.thread.isRunning() and self._live:
            xs = [t for t, _ in self._live]
            self.live_chart.set_line(
                xs, [{"name": "risk", "color": C_RED,
                      "values": [r for _, r in self._live]}],
                threshold=None, y_max=1.0, xfmt=mmss)
        self._refresh_dashboard()   # cheap: reloads files only when changed

    def _export_html_dashboard(self):
        try:
            from src.analytics import generate_dashboard
            path = generate_dashboard(open_browser=False)
            webbrowser.open(Path(path).as_uri())
            self.logger.info(f"HTML dashboard exported: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Analytics", f"Export failed:\n{e}")

    # ══════════════════════ cctv tab ══════════════════════

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

    # ══════════════════════ logs tab ══════════════════════

    def _build_logs_tab(self):
        page = QWidget(); v = QVBoxLayout(page)
        v.setContentsMargins(10, 10, 10, 10); v.setSpacing(8)
        bar = QHBoxLayout()
        title = QLabel("Application log"); title.setObjectName("section")
        bar.addWidget(title)
        self.log_path_label = QLabel(
            str(LOG_DIR / f"app_{datetime.now():%Y%m%d}.log"))
        self.log_path_label.setObjectName("hint")
        bar.addWidget(self.log_path_label)
        bar.addStretch()
        open_btn = QPushButton("Open log folder")
        open_btn.clicked.connect(lambda: webbrowser.open(LOG_DIR.as_uri()))
        bar.addWidget(open_btn)
        clear_btn = QPushButton("Clear view")
        clear_btn.clicked.connect(lambda: self.logs_text.clear())
        bar.addWidget(clear_btn)
        bw = QWidget(); bw.setLayout(bar); v.addWidget(bw)

        self.logs_text = QTextEdit(); self.logs_text.setReadOnly(True)
        self.logs_text.setFont(QFont("Consolas", 9))
        v.addWidget(self.logs_text, stretch=1)
        return page

    def _append_log_line(self, line):
        low = line.lower()
        if "error" in low:
            color = "#f87171"
        elif "warning" in low:
            color = "#fbbf24"
        elif "alert" in low:
            color = "#f59e0b"
        else:
            color = "#c3c2b7"
        self.logs_text.append(f'<span style="color:{color}">{line}</span>')

    # ══════════════════════ small UI builders ══════════════════════

    def _section(self, text):
        lbl = QLabel(text); lbl.setObjectName("section")
        return lbl

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

    # ══════════════════════ source switching ══════════════════════

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

    # ══════════════════════ ESP / CCTV utility ══════════════════════

    def _read_distance(self):
        ip = self.esp_edit.text().strip()
        if not ip:
            QMessageBox.warning(self, "ESP32", "Enter the ESP32 IP first.")
            return
        self.status_label.setText("Reading ESP32 distance…")
        self._esp_thread = EspReadThread(ip)
        self._esp_thread.done.connect(
            lambda t: (self.status_label.setText(f"ESP32 distance: {t}"),
                       self.logger.info(f"ESP32 distance: {t}")))
        self._esp_thread.start()

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
        self.logger.info(f"CCTV camera registered: {name}")

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
            self.logger.warning(f"CCTV test failed: {name}")
        else:
            pix = QPixmap.fromImage(qimg).scaled(
                self.cctv_preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.cctv_preview.setPixmap(pix)
            self.logger.info(f"CCTV test OK: {name}")

    def _remove_camera(self, idx):
        cams = load_cameras()
        if 0 <= idx < len(cams):
            removed = cams.pop(idx); save_cameras(cams)
            self._refresh_camera_list(); self._refresh_cctv_combo()
            self.logger.info(f"CCTV camera removed: {removed['name']}")

    # ══════════════════════ run / stop ══════════════════════

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
            "imgsz": self.imgsz_combo.currentData(),
            "realtime": self.realtime_check.isChecked(),
            "save": self.save_check.isChecked() and not is_live,
            "esp_ip": self.esp_edit.text().strip(),
            "esp_poll": self.esp_poll_check.isChecked() and bool(self.esp_edit.text().strip()),
        }

        self.alerts = []
        self._live.clear()
        self.alert_text.clear()
        self.export_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.fwd_btn.setEnabled(src == SRC_VIDEO)
        self.status_label.setText(
            f"Monitoring — {desc} · {Path(model).name} · {alert_type.upper()}")
        self.logger.info(f"Monitoring started: {desc} | model={Path(model).name} "
                         f"imgsz={meta['imgsz']} | mode={alert_type.upper()} "
                         f"risk>={eff_risk:.2f}")

        self.thread = MonitorThread(meta, self.cfg)
        self.thread.frameReady.connect(self._on_frame)
        self.thread.statsReady.connect(self._on_stats)
        self.thread.logMsg.connect(self.logger.info)
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
        self.logger.info("Stop requested")

    # ══════════════════════ worker slots ══════════════════════

    def _on_frame(self, qimg):
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation)   # speed over smoothing
        self.video_label.setPixmap(pix)

    def _on_stats(self, s):
        self.stat_widgets["frames"].setText(f"{s['frames']:,}")
        self.stat_widgets["persons"].setText(str(s["persons"]))
        self.stat_widgets["dogs"].setText(str(s["dogs"]))
        self.stat_widgets["alerts"].setText(str(s["alerts"]))
        self.stat_widgets["fps"].setText(f"{s['fps']:.1f}")
        self.stat_widgets["dist"].setText(s["dist"])
        self.behavior_label.setText(s.get("behavior", ""))
        self._live.append((s.get("t", 0.0), s.get("risk", 0.0)))
        self.live_status.setText(
            f"risk {s.get('risk', 0):.2f} · {s.get('behavior') or '—'} · "
            f"{s['dogs']} dog(s), {s['persons']} person(s) · {s['fps']:.1f} FPS")

    def _on_compute(self, text):
        warn = text.startswith("CPU")
        self.compute_label.setText(("⚠ " if warn else "⚡ ") + text)
        self.compute_label.setStyleSheet(
            "color:#f87171; font-weight:600;" if warn else "color:#22c55e;")
        (self.logger.warning if warn else self.logger.info)(f"Compute: {text}")

    def _on_alert(self, entry):
        self.alerts.append(entry)
        prefix = "[HR ALERT]" if entry["alert_type"] == "hr" else "[ALERT]"
        line = (f"[{entry['time']}] {prefix} dog#{entry['track_id']} "
                f"{entry.get('behavior','')} risk={entry['risk']:.2f} "
                f"frame {entry['frame']}")
        self.alert_text.append(line)
        self.logger.warning(f"ALERT dog#{entry['track_id']} "
                            f"{entry.get('behavior','')} risk={entry['risk']:.2f}")
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
        msg = (f"{verb} — {summary['frames']:,} frames · "
               f"{len(summary['alerts'])} alert(s) · peak {summary['peak_dogs']} dogs")
        self.status_label.setText(msg)
        self.logger.info(f"Monitoring finished: {msg}")
        if summary.get("output_path"):
            self.logger.info(f"Annotated video saved: {summary['output_path']}")
        self.live_status.setText("Not monitoring")
        self._refresh_dashboard(force=True)   # new session file — sync now

    def _on_error(self, msg):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fwd_btn.setEnabled(False)
        self.logger.error(f"Monitoring failed: {msg}")
        QMessageBox.critical(self, "Monitoring failed", msg)

    def _on_tab_changed(self, idx):
        self._refresh_cctv_combo()
        if idx == self.TAB_DASH:
            self._refresh_dashboard(force=True)

    # ══════════════════════ misc ══════════════════════

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
            self.logger.info(f"Alerts exported: {path}")
            QMessageBox.information(self, "Exported", f"Saved to:\n{path}")

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.thread.wait(2000)
        self.logger.info("Application closed")
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
