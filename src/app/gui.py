"""
Stray Dog Monitoring System — desktop application.

A single-window control room for watching a scene and getting warned before a
stray dog attack happens. Features:

  - Source selection ....... Video File / Laptop Webcam / ESP32-CAM stream
  - Model picker ........... YOLO26 (n/s/m/l/x) + YOLO11 (n/m/x), auto-downloaded
  - ESP32-CAM + HC-SR04 .... live MJPEG video and ultrasonic proximity alerts
  - Alert type ............. Normal or HR (high-risk: lower threshold, flash+beep)
  - Live stats bar ......... frames, persons, dogs, alerts, FPS, distance
  - Alert log + JSON export
  - Analytics .............. sessions recorded to data/sessions/, one-click
                             offline HTML dashboard (outputs/dashboard.html)

Run with:  python run.py      (or)   python -m src.app.gui
"""

import os
import sys
import time
import json
import threading
import urllib.request
from pathlib import Path
from datetime import datetime

import numpy as np

# Windows-only loud beep via winsound; fall back to the Tk bell elsewhere.
if sys.platform == "win32":
    import winsound as _winsound
else:
    _winsound = None

# Silence FFmpeg/OpenCV internal stream noise before cv2 import.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
try:
    cv2.setLogLevel(0)   # silence the OpenCV C++ logger (not in every build)
except AttributeError:
    pass
from PIL import Image, ImageTk

from src.config import load_config
from src.sources import MJPEGCapture


YOLO26_VARIANTS = {
    "yolo26n.pt": "Nano   - fastest (CPU friendly)",
    "yolo26s.pt": "Small  - fast",
    "yolo26m.pt": "Medium - balanced",
    "yolo26l.pt": "Large  - more accurate",
    "yolo26x.pt": "XLarge - best accuracy",
}

YOLO11_VARIANTS = {
    "yolo11n.pt": "Nano   - fastest (CPU friendly)",
    "yolo11m.pt": "Medium - balanced",
    "yolo11x.pt": "XLarge - best accuracy",
}

# Every model selectable in the app (all COCO-pretrained, auto-downloaded)
MODEL_VARIANTS = {**YOLO26_VARIANTS, **YOLO11_VARIANTS}

SOURCE_VIDEO = "video"
SOURCE_WEBCAM = "webcam"
SOURCE_ESP_CAM = "espcam"


class StrayDogMonitorApp:
    """Tkinter control room for the stray-dog monitoring pipeline."""

    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv",
                  ".m4v", ".ts", ".3gp", ".webm", ".mpeg", ".mpg"}
    MAX_RECENT = 10

    def __init__(self, root, config=None):
        self.cfg = config or load_config()
        self.root = root
        self.root.title("Stray Dog Monitoring System")
        self.root.geometry("1280x900")
        self.root.configure(bg="#1e1e1e")

        d = self.cfg["detector"]
        r = self.cfg["risk"]
        hw = self.cfg["hardware"]
        al = self.cfg["alerts"]
        inf = self.cfg["inference"]

        # source
        self.source_type = tk.StringVar(value=SOURCE_VIDEO)
        self.video_path = None
        self.cam_index = tk.IntVar(value=0)
        self._folder_files = []

        # ESP32
        self.esp_ip = tk.StringVar(value=hw.get("esp_ip", "192.168.1.100"))
        self.esp_connected = False
        self.esp_dist_var = tk.StringVar(value="-- cm")
        self.esp_status_var = tk.StringVar(value="Not connected")
        self._esp_poll_stop = threading.Event()
        self._esp_poll_thread = None
        self._proximity_last_ts = 0.0
        self._flash_last_ts = 0.0
        self._poll_interval = hw.get("poll_interval_s", 0.5)

        # alert type
        self.alert_type = tk.StringVar(value=al.get("default_type", "normal"))
        self.alert_sound_on = tk.BooleanVar(value=al.get("sound_enabled", True))
        self._hr_drop = al.get("hr_threshold_drop", 0.10)

        # processing state
        self.is_processing = False
        self.stop_requested = False
        self._fwd_lock = threading.Lock()
        self.forward_count = 0
        self.alerts = []

        # model + inference settings (seeded from config)
        default_variant = d["model"] if d["model"] in MODEL_VARIANTS else "yolo26n.pt"
        self.yolo26_variant = tk.StringVar(value=default_variant)
        self.pose_enabled = tk.BooleanVar(value=bool(d.get("pose_model")))
        self.det_conf = tk.DoubleVar(value=d["conf"])
        self.risk_threshold = tk.DoubleVar(value=r["threshold"])
        self.sustain_frames = tk.IntVar(value=r["sustain_frames"])
        self.skip_frames = tk.IntVar(value=inf.get("skip_frames", 1))
        self.save_output = tk.BooleanVar(value=inf.get("save_output", True))
        self.dist_alert_cm = tk.DoubleVar(value=hw.get("proximity_alert_cm", 100.0))
        self._output_dir = inf.get("output_dir", "outputs")
        self._pose_model = d.get("pose_model")
        self.custom_weights = tk.StringVar(value="")

        self._build_ui()

    # ── UI BUILD ──────────────────────────────────────────────────────

    def _build_ui(self):
        title_frame = tk.Frame(self.root, bg="#2d2d2d", height=60)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)
        tk.Label(title_frame, text="Stray Dog Monitoring System",
                 font=("Segoe UI", 16, "bold"), bg="#2d2d2d", fg="#ffffff",
                 ).pack(side="left", padx=20, pady=15)
        tk.Label(title_frame, text="YOLO26",
                 font=("Segoe UI", 11, "bold"), bg="#2d2d2d", fg="#f59e0b",
                 ).pack(side="left", pady=15)

        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Scrollable left control panel
        left_outer = tk.Frame(main_frame, bg="#252525", width=390)
        left_outer.pack(side="left", fill="y", padx=(0, 10))
        left_outer.pack_propagate(False)

        canvas = tk.Canvas(left_outer, bg="#252525", highlightthickness=0, width=370)
        scrollbar = ttk.Scrollbar(left_outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        left_frame = tk.Frame(canvas, bg="#252525")
        canvas_window = canvas.create_window((0, 0), window=left_frame, anchor="nw")
        left_frame.bind("<Configure>",
                        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        right_frame = tk.Frame(main_frame, bg="#1e1e1e")
        right_frame.pack(side="right", fill="both", expand=True)

        self._build_controls(left_frame)
        self._build_display(right_frame)

    def _build_controls(self, parent):
        # 1. Source
        self._section_header(parent, "1. Select Source")
        for val, lbl in [
            (SOURCE_VIDEO, "Video File"),
            (SOURCE_WEBCAM, "Laptop Webcam"),
            (SOURCE_ESP_CAM, "ESP32-CAM"),
        ]:
            tk.Radiobutton(
                parent, text=lbl, variable=self.source_type, value=val,
                command=self._on_source_change,
                bg="#252525", fg="#cccccc", selectcolor="#1e1e1e",
                activebackground="#252525", activeforeground="#f59e0b",
                font=("Segoe UI", 10), anchor="w",
            ).pack(fill="x", padx=15, pady=2)

        # Video sub-panel
        self.video_panel = tk.Frame(parent, bg="#2a2a2a", padx=10, pady=8)
        tk.Button(self.video_panel, text="Browse Local Video",
                  command=self.browse_video, bg="#0078d4", fg="white",
                  font=("Segoe UI", 11, "bold"), relief="flat", pady=10,
                  cursor="hand2").pack(fill="x", pady=(0, 6))

        file_box = tk.Frame(self.video_panel, bg="#1e1e1e", relief="solid", bd=1)
        file_box.pack(fill="x", pady=(0, 5))
        tk.Label(file_box, text="Selected:", bg="#1e1e1e", fg="#555555",
                 font=("Segoe UI", 8)).pack(anchor="w", padx=6, pady=(4, 0))
        self.path_label = tk.Label(file_box, text="No video selected",
                                   bg="#1e1e1e", fg="#888888", font=("Segoe UI", 9),
                                   wraplength=300, justify="left", anchor="w")
        self.path_label.pack(fill="x", padx=6, pady=(0, 6))

        tk.Label(self.video_panel, text="Recent videos:", bg="#2a2a2a",
                 fg="#666666", font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 2))
        list_frame = tk.Frame(self.video_panel, bg="#1e1e1e", relief="solid", bd=1)
        list_frame.pack(fill="x", pady=(0, 6))
        self.file_listbox = tk.Listbox(
            list_frame, bg="#1e1e1e", fg="#cccccc",
            selectbackground="#0078d4", selectforeground="#ffffff",
            font=("Segoe UI", 9), height=5, relief="flat",
            activestyle="none", cursor="hand2")
        list_sb = ttk.Scrollbar(list_frame, orient="vertical",
                                command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=list_sb.set)
        list_sb.pack(side="right", fill="y")
        self.file_listbox.pack(side="left", fill="both", expand=True)
        self.file_listbox.bind("<<ListboxSelect>>", self._on_file_select)
        self.file_listbox.bind("<Double-Button-1>", self._on_file_double_click)

        vbtn_row = tk.Frame(self.video_panel, bg="#2a2a2a")
        vbtn_row.pack(fill="x")
        tk.Button(vbtn_row, text="Enter Path", command=self.enter_path,
                  bg="#3a3a3a", fg="white", font=("Segoe UI", 9), relief="flat",
                  padx=10, pady=4, cursor="hand2").pack(side="left", padx=(0, 4))
        self.forward_btn = tk.Button(
            vbtn_row, text=">> +10s", command=self.video_forward,
            bg="#7c3aed", fg="white", font=("Segoe UI", 9, "bold"),
            relief="flat", padx=10, pady=4, cursor="hand2", state="disabled")
        self.forward_btn.pack(side="left")

        # Webcam sub-panel
        self.webcam_panel = tk.Frame(parent, bg="#2a2a2a", padx=10, pady=6)
        cam_row = tk.Frame(self.webcam_panel, bg="#2a2a2a")
        cam_row.pack(fill="x")
        tk.Label(cam_row, text="Camera index:", bg="#2a2a2a", fg="#cccccc",
                 font=("Segoe UI", 9)).pack(side="left")
        ttk.Spinbox(cam_row, from_=0, to=10, textvariable=self.cam_index,
                    width=4, font=("Segoe UI", 9)).pack(side="left", padx=8)
        tk.Label(self.webcam_panel, text="Uses laptop/USB camera via OpenCV index.",
                 bg="#2a2a2a", fg="#555555", font=("Segoe UI", 8),
                 ).pack(anchor="w", pady=(4, 0))

        # ESP32-CAM sub-panel
        self.esp_cam_panel = tk.Frame(parent, bg="#2a2a2a", padx=10, pady=6)
        tk.Label(self.esp_cam_panel, text="Stream URL:  http://<ESP-IP>:81/stream",
                 bg="#2a2a2a", fg="#0ea5e9", font=("Consolas", 8)).pack(anchor="w")
        tk.Label(self.esp_cam_panel, text="Set the ESP IP in the ESP Sensor section below.",
                 bg="#2a2a2a", fg="#555555", font=("Segoe UI", 8),
                 ).pack(anchor="w", pady=(2, 0))

        # 2. ESP sensor
        self._section_header(parent, "2. ESP Sensor (HC-SR04 + CAM)")
        ip_row = tk.Frame(parent, bg="#252525")
        ip_row.pack(fill="x", padx=15, pady=(0, 6))
        tk.Label(ip_row, text="ESP32 IP:", bg="#252525", fg="#cccccc",
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Entry(ip_row, textvariable=self.esp_ip, width=16, bg="#1e1e1e",
                 fg="#ffffff", insertbackground="white",
                 font=("Segoe UI", 9)).pack(side="left", padx=8)

        esp_btn_row = tk.Frame(parent, bg="#252525")
        esp_btn_row.pack(fill="x", padx=15, pady=(0, 6))
        self.esp_connect_btn = tk.Button(
            esp_btn_row, text="Connect Sensor", command=self.esp_connect,
            bg="#0284c7", fg="white", font=("Segoe UI", 9), relief="flat",
            padx=10, pady=4, cursor="hand2")
        self.esp_connect_btn.pack(side="left", padx=(0, 6))
        self.esp_disconnect_btn = tk.Button(
            esp_btn_row, text="Disconnect", command=self.esp_disconnect,
            bg="#3a3a3a", fg="white", font=("Segoe UI", 9), relief="flat",
            padx=10, pady=4, cursor="hand2", state="disabled")
        self.esp_disconnect_btn.pack(side="left")

        dist_box = tk.Frame(parent, bg="#1a1a1a", relief="solid", bd=1)
        dist_box.pack(fill="x", padx=15, pady=(0, 4))
        st_row = tk.Frame(dist_box, bg="#1a1a1a")
        st_row.pack(fill="x", padx=8, pady=(6, 2))
        tk.Label(st_row, text="Status:", bg="#1a1a1a", fg="#888888",
                 font=("Segoe UI", 9)).pack(side="left")
        self.esp_status_lbl = tk.Label(st_row, textvariable=self.esp_status_var,
                                       bg="#1a1a1a", fg="#f59e0b",
                                       font=("Segoe UI", 9, "bold"))
        self.esp_status_lbl.pack(side="left", padx=6)
        d_row = tk.Frame(dist_box, bg="#1a1a1a")
        d_row.pack(fill="x", padx=8, pady=(0, 8))
        tk.Label(d_row, text="Distance:", bg="#1a1a1a", fg="#888888",
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Label(d_row, textvariable=self.esp_dist_var, bg="#1a1a1a",
                 fg="#22c55e", font=("Segoe UI", 16, "bold")).pack(side="left", padx=8)

        self._slider(parent, "Proximity alert threshold (cm)",
                     self.dist_alert_cm, 10, 400, is_int=True)

        # 3. Alert type
        self._section_header(parent, "3. Alert Type")
        for val, desc in [
            ("normal", "Normal Alert  -  standard sensitivity"),
            ("hr", "HR Alert       -  high-risk, lower threshold"),
        ]:
            tk.Radiobutton(
                parent, text=desc, variable=self.alert_type, value=val,
                command=self._on_alert_type_change,
                bg="#252525", fg="#cccccc", selectcolor="#1e1e1e",
                activebackground="#252525", activeforeground="#f59e0b",
                font=("Segoe UI", 9), anchor="w").pack(fill="x", padx=15, pady=2)

        self.alert_hint = tk.Label(parent, text="", bg="#252525", fg="#666666",
                                   font=("Segoe UI", 8), wraplength=330,
                                   justify="left")
        self.alert_hint.pack(anchor="w", padx=15, pady=(2, 6))

        sound_row = tk.Frame(parent, bg="#252525")
        sound_row.pack(fill="x", padx=15, pady=(0, 4))
        tk.Checkbutton(sound_row, text="Beep sound on alert",
                       variable=self.alert_sound_on,
                       bg="#252525", fg="#cccccc", selectcolor="#1e1e1e",
                       activebackground="#252525", activeforeground="#f59e0b",
                       font=("Segoe UI", 9), anchor="w").pack(side="left")
        tk.Button(sound_row, text="Test beep",
                  command=lambda: self._play_alert_sound(self.alert_type.get()),
                  bg="#374151", fg="#f59e0b", relief="flat",
                  font=("Segoe UI", 8), padx=8, pady=2,
                  cursor="hand2").pack(side="left", padx=(10, 0))
        self._on_alert_type_change()

        # 4. Model
        self._section_header(parent, "4. Detection Model")
        tk.Label(parent, text="YOLO26", bg="#252525", fg="#888888",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=15)
        for variant, desc in YOLO26_VARIANTS.items():
            tk.Radiobutton(
                parent, text=f"{variant}  -  {desc}",
                variable=self.yolo26_variant, value=variant,
                bg="#252525", fg="#cccccc", selectcolor="#1e1e1e",
                activebackground="#252525", activeforeground="#f59e0b",
                font=("Consolas", 9), anchor="w").pack(fill="x", padx=15, pady=1)
        tk.Label(parent, text="YOLO11", bg="#252525", fg="#888888",
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=15, pady=(6, 0))
        for variant, desc in YOLO11_VARIANTS.items():
            tk.Radiobutton(
                parent, text=f"{variant}  -  {desc}",
                variable=self.yolo26_variant, value=variant,
                bg="#252525", fg="#cccccc", selectcolor="#1e1e1e",
                activebackground="#252525", activeforeground="#f59e0b",
                font=("Consolas", 9), anchor="w").pack(fill="x", padx=15, pady=1)
        tk.Label(parent, text="(auto-downloads on first run)", bg="#252525",
                 fg="#666666", font=("Segoe UI", 8)).pack(anchor="w", padx=15, pady=(4, 4))
        tk.Label(parent, text="Custom weights (.pt) — overrides the choice above:",
                 bg="#252525", fg="#888888",
                 font=("Segoe UI", 8)).pack(anchor="w", padx=15, pady=(4, 0))
        tk.Entry(parent, textvariable=self.custom_weights,
                 bg="#1e1e1e", fg="#ffffff", insertbackground="white",
                 font=("Segoe UI", 9)).pack(fill="x", padx=15, pady=(2, 4))
        tk.Checkbutton(parent, text="Use pose model (human skeleton)",
                       variable=self.pose_enabled,
                       bg="#252525", fg="#cccccc", selectcolor="#1e1e1e",
                       activebackground="#252525", activeforeground="white",
                       font=("Segoe UI", 9)).pack(anchor="w", padx=15, pady=(0, 10))

        # 5. Settings
        self._section_header(parent, "5. Settings")
        self._slider(parent, "Detection confidence", self.det_conf, 0.1, 0.95)
        self._slider(parent, "Risk threshold", self.risk_threshold, 0.1, 0.95)
        self._slider(parent, "Sustain frames (N)", self.sustain_frames, 1, 20, is_int=True)
        self._slider(parent, "Skip frames", self.skip_frames, 1, 30, is_int=True)
        tk.Checkbutton(parent, text="Save annotated output video",
                       variable=self.save_output,
                       bg="#252525", fg="#cccccc", selectcolor="#1e1e1e",
                       activebackground="#252525", activeforeground="white",
                       font=("Segoe UI", 9)).pack(anchor="w", padx=15, pady=(5, 10))

        # 6. Run
        self._section_header(parent, "6. Run")
        self.start_btn = tk.Button(parent, text="Start Monitoring",
                                   command=self.start_detection, bg="#16a34a",
                                   fg="white", font=("Segoe UI", 11, "bold"),
                                   relief="flat", padx=15, pady=10, cursor="hand2")
        self.start_btn.pack(fill="x", padx=15, pady=(0, 5))
        self.stop_btn = tk.Button(parent, text="Stop", command=self.stop_detection,
                                  bg="#dc2626", fg="white", font=("Segoe UI", 10),
                                  relief="flat", padx=15, pady=5, state="disabled",
                                  cursor="hand2")
        self.stop_btn.pack(fill="x", padx=15, pady=(0, 10))
        self.status_label = tk.Label(parent, text="Ready", bg="#252525",
                                     fg="#888888", font=("Segoe UI", 9), anchor="w")
        self.status_label.pack(fill="x", padx=15, pady=5)
        self.progress = ttk.Progressbar(parent, mode="determinate")
        self.progress.pack(fill="x", padx=15, pady=5)

        # 7. Analytics
        self._section_header(parent, "7. Analytics")
        tk.Button(parent, text="Open Analytics Dashboard",
                  command=self.open_dashboard, bg="#7c3aed", fg="white",
                  font=("Segoe UI", 10, "bold"), relief="flat", padx=15,
                  pady=8, cursor="hand2").pack(fill="x", padx=15, pady=(0, 4))
        tk.Label(parent,
                 text="Aggregates every monitoring session into charts:\n"
                      "risk timelines, alerts by hour, detections, model stats.",
                 bg="#252525", fg="#666666", font=("Segoe UI", 8),
                 justify="left").pack(anchor="w", padx=15, pady=(0, 10))

        self._on_source_change()

    def _build_display(self, parent):
        self.video_frame = tk.Frame(parent, bg="#1a1a1a", relief="solid", bd=4)
        self.video_frame.pack(fill="both", expand=True, pady=(0, 10))
        self.video_label = tk.Label(self.video_frame,
                                    text="Video preview will appear here",
                                    bg="#000000", fg="#555555", font=("Segoe UI", 12))
        self.video_label.pack(fill="both", expand=True)

        stats_frame = tk.Frame(parent, bg="#252525", height=80)
        stats_frame.pack(fill="x", pady=(0, 10))
        stats_frame.pack_propagate(False)
        self.stats_vars = {
            "frames": tk.StringVar(value="0"),
            "persons": tk.StringVar(value="0"),
            "dogs": tk.StringVar(value="0"),
            "alerts": tk.StringVar(value="0"),
            "fps": tk.StringVar(value="0.0"),
            "distance": tk.StringVar(value="--"),
        }
        for key, label in [
            ("frames", "Frames"), ("persons", "Persons now"), ("dogs", "Dogs now"),
            ("alerts", "ALERTS"), ("fps", "FPS"), ("distance", "DIST(cm)"),
        ]:
            col = tk.Frame(stats_frame, bg="#252525")
            col.pack(side="left", fill="both", expand=True, padx=5, pady=10)
            color = {"alerts": "#dc2626", "fps": "#22c55e",
                     "distance": "#0ea5e9"}.get(key, "#ffffff")
            tk.Label(col, textvariable=self.stats_vars[key], bg="#252525",
                     fg=color, font=("Segoe UI", 18, "bold")).pack()
            tk.Label(col, text=label, bg="#252525", fg="#888888",
                     font=("Segoe UI", 9)).pack()

        self.model_status = tk.StringVar(value="Idle")
        model_frame = tk.Frame(parent, bg="#1a1a1a", pady=4)
        model_frame.pack(fill="x", pady=(0, 5))
        tk.Label(model_frame, text="Model: ", bg="#1a1a1a", fg="#888888",
                 font=("Segoe UI", 9)).pack(side="left", padx=10)
        tk.Label(model_frame, textvariable=self.model_status, bg="#1a1a1a",
                 fg="#f59e0b", font=("Segoe UI", 9, "bold")).pack(side="left")

        log_frame = tk.LabelFrame(parent, text="  Alert Log  ", bg="#252525",
                                  fg="#ffffff", font=("Segoe UI", 10, "bold"))
        log_frame.pack(fill="x")
        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=7, bg="#1a1a1a", fg="#ffffff",
            insertbackground="white", font=("Consolas", 9), relief="flat", wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        self.export_btn = tk.Button(log_frame, text="Export Alerts (JSON)",
                                    command=self.export_alerts, bg="#3a3a3a",
                                    fg="white", font=("Segoe UI", 9), relief="flat",
                                    padx=10, pady=3, state="disabled")
        self.export_btn.pack(side="right", padx=5, pady=(0, 5))

    # ── source / alert callbacks ─────────────────────────────────────

    def _on_source_change(self):
        self.video_panel.pack_forget()
        self.webcam_panel.pack_forget()
        self.esp_cam_panel.pack_forget()
        src = self.source_type.get()
        if src == SOURCE_VIDEO:
            self.video_panel.pack(fill="x", padx=15, pady=(4, 10))
        elif src == SOURCE_WEBCAM:
            self.webcam_panel.pack(fill="x", padx=15, pady=(4, 10))
        elif src == SOURCE_ESP_CAM:
            self.esp_cam_panel.pack(fill="x", padx=15, pady=(4, 10))

    def _on_alert_type_change(self):
        if self.alert_type.get() == "hr":
            self.alert_hint.config(
                text=f"HR mode: threshold -{self._hr_drop:.2f}, red flash + 3x beep "
                     f"on every alert.", fg="#f87171")
        else:
            self.alert_hint.config(
                text="Normal mode: standard thresholds, single beep on detection.",
                fg="#666666")

    # ── ESP sensor ───────────────────────────────────────────────────

    def esp_connect(self):
        ip = self.esp_ip.get().strip()
        if not ip:
            messagebox.showerror("Error", "Enter the ESP32 IP address first.")
            return
        self._esp_poll_stop.clear()
        self.esp_connected = True
        self.esp_status_var.set("Connecting...")
        self.esp_status_lbl.config(fg="#f59e0b")
        self.esp_connect_btn.config(state="disabled")
        self.esp_disconnect_btn.config(state="normal")
        self._esp_poll_thread = threading.Thread(target=self._esp_poll_loop, daemon=True)
        self._esp_poll_thread.start()
        self.log(f"[ESP] Connecting to {ip} ...")

    def esp_disconnect(self):
        self._esp_poll_stop.set()
        self.esp_connected = False
        self.esp_status_var.set("Disconnected")
        self.esp_status_lbl.config(fg="#888888")
        self.esp_dist_var.set("-- cm")
        self.stats_vars["distance"].set("--")
        self.esp_connect_btn.config(state="normal")
        self.esp_disconnect_btn.config(state="disabled")
        self.log("[ESP] Disconnected.")

    def _esp_reset_ui(self, msg, color="#f87171"):
        self.esp_connected = False
        self.esp_status_var.set(msg)
        self.esp_status_lbl.config(fg=color)
        self.esp_dist_var.set("-- cm")
        self.stats_vars["distance"].set("--")
        self.esp_connect_btn.config(state="normal")
        self.esp_disconnect_btn.config(state="disabled")

    def _esp_poll_loop(self):
        ip = self.esp_ip.get().strip()
        url = f"http://{ip}/distance"
        failures = 0
        MAX_FAIL = 6

        while not self._esp_poll_stop.is_set():
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    data = json.loads(resp.read())
                dist = data.get("distance_cm")
                if dist is None:
                    dist_str, dist_stat = "Out of range", "--"
                else:
                    dist_str = f"{dist:.1f} cm"
                    dist_stat = f"{dist:.0f}"
                self.root.after(0, self.esp_dist_var.set, dist_str)
                self.root.after(0, self.stats_vars["distance"].set, dist_stat)
                self.root.after(0, self.esp_status_var.set, f"Online  {ip}")
                self.root.after(0, self.esp_status_lbl.config, {"fg": "#22c55e"})
                failures = 0

                if dist is not None and dist < self.dist_alert_cm.get():
                    now = time.time()
                    if now - self._proximity_last_ts >= 10.0:
                        self._proximity_last_ts = now
                        self.root.after(0, self._proximity_alert, dist)

            except Exception as exc:
                failures += 1
                err_msg = str(exc).replace("<", "").replace(">", "")
                if failures < MAX_FAIL:
                    self.root.after(0, self.esp_status_var.set,
                                    f"Retrying... ({failures}/{MAX_FAIL})  {err_msg}")
                    self.root.after(0, self.esp_status_lbl.config, {"fg": "#f59e0b"})
                else:
                    self.root.after(0, self._esp_reset_ui,
                                    f"Cannot reach {ip}  ({err_msg})")
                    self.root.after(0, self.log,
                                    f"[ESP] Failed after {MAX_FAIL} attempts. "
                                    f"Check IP/WiFi and try again.")
                    return

            self._esp_poll_stop.wait(self._poll_interval)

    def _proximity_alert(self, dist):
        alert_t = self.alert_type.get()
        prefix = "[ESP][HR ALERT]" if alert_t == "hr" else "[ESP]"
        self.log(f"{prefix} Proximity: {dist:.1f} cm")
        if alert_t == "hr" and self.is_processing:
            self.root.bell()
            self._flash_alert()

    # ── video file controls ──────────────────────────────────────────

    def _add_to_recent(self, path):
        name = Path(path).name
        for i in range(self.file_listbox.size()):
            if self.file_listbox.get(i) == name:
                self.file_listbox.delete(i)
                self._folder_files.pop(i)
                break
        self.file_listbox.insert(0, name)
        self._folder_files.insert(0, Path(path))
        while self.file_listbox.size() > self.MAX_RECENT:
            self.file_listbox.delete(self.MAX_RECENT)
            self._folder_files = self._folder_files[:self.MAX_RECENT]

    def _on_file_select(self, _event=None):
        sel = self.file_listbox.curselection()
        if not sel or not self._folder_files:
            return
        idx = sel[0]
        if idx < len(self._folder_files):
            self._set_video(str(self._folder_files[idx]))

    def _on_file_double_click(self, _event=None):
        self._on_file_select()
        if self.video_path and not self.is_processing:
            self.start_detection()

    def browse_video(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files",
                 "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.m4v *.ts *.3gp *.webm *.mpeg *.mpg"),
                ("All files", "*.*"),
            ])
        if path:
            self._set_video(path)

    def enter_path(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Enter Video Path")
        dialog.geometry("500x120")
        dialog.configure(bg="#252525")
        dialog.transient(self.root)
        dialog.grab_set()
        tk.Label(dialog, text="Enter full local video path:", bg="#252525",
                 fg="#ffffff", font=("Segoe UI", 10)).pack(pady=(15, 5))
        path_var = tk.StringVar()
        entry = tk.Entry(dialog, textvariable=path_var, width=60, bg="#1e1e1e",
                         fg="#ffffff", insertbackground="white", font=("Segoe UI", 10))
        entry.pack(padx=15, pady=5, fill="x")
        entry.focus()

        def submit():
            p = path_var.get().strip().strip('"')
            if p and Path(p).exists():
                self._set_video(p)
                dialog.destroy()
            else:
                messagebox.showerror("Error", f"File not found:\n{p}")

        tk.Button(dialog, text="Load", command=submit, bg="#0078d4", fg="white",
                  relief="flat", font=("Segoe UI", 10), padx=20, pady=5).pack(pady=10)
        entry.bind("<Return>", lambda e: submit())

    def _set_video(self, path):
        self.video_path = path
        self.path_label.config(text=Path(path).name, fg="#22c55e")
        self._add_to_recent(path)
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        if ret:
            self._display_frame(frame)
        cap.release()
        self.log(f"Video loaded: {path}")

    def video_forward(self):
        if self.is_processing and self.source_type.get() == SOURCE_VIDEO:
            with self._fwd_lock:
                self.forward_count += 1

    # ── display ──────────────────────────────────────────────────────

    def _display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()
        if label_w < 10 or label_h < 10:
            label_w, label_h = 800, 500
        h, w = rgb.shape[:2]
        scale = min(label_w / w, label_h / h)
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk, text="")

    def _clear_display(self):
        self.video_label.imgtk = None
        self.video_label.config(image="", text="Video preview will appear here",
                                bg="#000000", fg="#555555")
        self.video_frame.config(bg="#1a1a1a")

    def _play_alert_sound(self, alert_type):
        def _beep():
            if _winsound:
                if alert_type == "hr":
                    for _ in range(3):
                        _winsound.Beep(1500, 250)
                        time.sleep(0.05)
                else:
                    _winsound.Beep(1000, 600)
            else:
                self.root.after(0, self.root.bell)
        threading.Thread(target=_beep, daemon=True).start()

    def _flash_alert(self):
        now = time.time()
        if now - self._flash_last_ts < 2.0:
            return
        self._flash_last_ts = now
        self.video_frame.config(bg="#dc2626")
        self.root.after(600, lambda: self.video_frame.config(bg="#1a1a1a"))

    # ── detection ────────────────────────────────────────────────────

    def start_detection(self):
        src = self.source_type.get()
        if src == SOURCE_VIDEO and not self.video_path:
            messagebox.showwarning("No Video", "Please select a video file first.")
            return
        if src == SOURCE_ESP_CAM and not self.esp_ip.get().strip():
            messagebox.showwarning("No IP", "Enter the ESP32-CAM IP address.")
            return

        self.is_processing = True
        self.stop_requested = False
        self.forward_count = 0
        self.alerts = []
        self.log_text.delete(1.0, "end")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.export_btn.config(state="disabled")
        if src == SOURCE_VIDEO:
            self.forward_btn.config(state="normal")
        for key, var in self.stats_vars.items():
            var.set("--" if key == "distance" else "0")

        if self.esp_ip.get().strip() and not self.esp_connected:
            self.esp_connect()

        threading.Thread(target=self._detection_worker, daemon=True).start()

    def stop_detection(self):
        self.stop_requested = True
        self.forward_btn.config(state="disabled")
        self.log("Stop requested...")

    def _detection_worker(self):
        pipeline = None
        try:
            src = self.source_type.get()
            model_name = self.yolo26_variant.get()
            # custom fine-tuned weights override the picker when provided
            cw = self.custom_weights.get().strip().strip('"')
            if cw:
                if not Path(cw).exists():
                    raise RuntimeError(f"Custom weights not found: {cw}")
                model_name = cw
            alert_t = self.alert_type.get()

            self._update_status(f"Loading {model_name}...")
            self.model_status.set(f"Loading {model_name}")
            self.log(f"Source: {src.upper()} | Alert type: {alert_t.upper()}")
            self.log(f"Initialising model: {model_name}")

            from src.pipeline import StrayDogMonitor

            pose_model = self._pose_model if self.pose_enabled.get() else None

            eff_risk = self.risk_threshold.get()
            if alert_t == "hr":
                eff_risk = max(0.10, eff_risk - self._hr_drop)
                self.log(f"[HR] Effective risk threshold: {eff_risk:.2f}")

            pipeline = StrayDogMonitor(
                detector_path=model_name,
                pose_model=pose_model,
                det_conf=self.det_conf.get(),
                risk_threshold=eff_risk,
                sustain_frames=self.sustain_frames.get(),
                config=self.cfg,
            )
            self.model_status.set(f"{model_name} (active)")
            self.log(f"[compute] {getattr(pipeline, 'compute', 'unknown')}")

            cap, is_live, stream_url = self._open_source(src)

            if not cap.isOpened():
                raise RuntimeError("Could not open video source.")

            if src == SOURCE_ESP_CAM:
                self._update_status("Waiting for first frame ...")
                deadline = time.time() + 6
                ret_test = False
                while time.time() < deadline:
                    ret_test, _ = cap.read()
                    if ret_test:
                        break
                if not ret_test:
                    cap.release()
                    raise RuntimeError(
                        f"ESP32-CAM connected but no frames received within 6 s.\n\n"
                        f"Stream URL: {stream_url}\n"
                        f"Make sure the MJPEG server on the ESP32 is running.")

            self._update_status("Processing...")
            self._run_loop(cap, pipeline, src, is_live, model_name, alert_t)

        except Exception as e:
            self.log(f"ERROR: {e}")
            self.model_status.set("Error")
            messagebox.showerror("Error", f"Monitoring failed:\n{e}")
        finally:
            del pipeline
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            self.is_processing = False
            self.model_status.set("Idle")
            self.progress["value"] = 0
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.forward_btn.config(state="disabled")
            self.root.after(0, self._clear_display)

    def _open_source(self, src):
        """Return (capture, is_live, stream_url). Live sources are wrapped in
        a threaded latest-frame reader so slow inference never builds lag."""
        from src.sources import LatestFrameCapture
        if src == SOURCE_VIDEO:
            return cv2.VideoCapture(self.video_path), False, None
        if src == SOURCE_WEBCAM:
            cap = cv2.VideoCapture(int(self.cam_index.get()))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return LatestFrameCapture(cap), True, None

        # ESP32-CAM
        ip = self.esp_ip.get().strip()
        stream_url = f"http://{ip}:81/stream"
        self.log(f"[ESP-CAM] Pinging {ip} ...")
        self._update_status(f"Checking ESP32 at {ip} ...")
        try:
            with urllib.request.urlopen(f"http://{ip}/status", timeout=3) as r:
                info = json.loads(r.read())
            self.log(f"[ESP-CAM] Device online  RSSI={info.get('wifi_rssi', '?')} dBm")
        except Exception as ping_err:
            clean = str(ping_err).replace("<", "").replace(">", "")
            raise RuntimeError(
                f"ESP32 not reachable at {ip}\n\nDetails: {clean}\n\n"
                f"Check:\n  1. ESP32 is powered and WiFi connected\n"
                f"  2. IP address is correct\n"
                f"  3. Laptop and ESP32 are on the same WiFi")
        self.log(f"[ESP-CAM] Opening MJPEG stream: {stream_url}")
        self._update_status("Connecting to ESP32-CAM stream ...")
        return (LatestFrameCapture(MJPEGCapture(stream_url, timeout=5)),
                True, stream_url)

    def _run_loop(self, cap, pipeline, src, is_live, model_name, alert_t):
        # Analytics recorder — observes only; failures never affect monitoring
        recorder = None
        acfg = self.cfg.get("analytics", {}) or {}
        if acfg.get("enabled", True):
            try:
                from src.analytics import SessionRecorder
                recorder = SessionRecorder(
                    model=model_name, source=src, alert_type=alert_t,
                    risk_threshold=pipeline.risk_threshold,
                    det_conf=self.det_conf.get(),
                    sessions_dir=acfg.get("sessions_dir", "data/sessions"),
                    timeline_max_points=acfg.get("timeline_max_points", 600),
                )
            except Exception:
                recorder = None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer, output_path = None, None
        if self.save_output.get() and not is_live:
            out_dir = Path(self._output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = "hr" if alert_t == "hr" else "norm"
            output_path = str(out_dir / (
                f"{Path(model_name).stem}_{tag}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        skip = self.skip_frames.get()
        frame_count = processed = 0
        total_alerts = 0
        # counts are instantaneous (how many are in THIS frame), not a running
        # sum — summing per frame made 4 dogs read as hundreds. Peaks track the
        # most seen at once across the run.
        cur_persons = cur_dogs = peak_persons = peak_dogs = 0
        start_time = time.time()

        while not self.stop_requested:
            if not is_live:
                with self._fwd_lock:
                    fwd = self.forward_count
                    if fwd > 0:
                        new_pos = frame_count + int(fwd * fps * 10)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        self.log(f"[FWD] +{fwd*10}s  ->  frame {frame_count}")
                        self.forward_count = 0

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

            cur_persons = len(pipeline._last_persons)
            cur_dogs = len(results)
            peak_persons = max(peak_persons, cur_persons)
            peak_dogs = max(peak_dogs, cur_dogs)

            if recorder:
                recorder.record_frame(results, len(pipeline._last_persons),
                                      frame_count)

            new_alerts_now = [r for r in results if r.get("new_alert")]
            total_alerts += len(new_alerts_now)

            if new_alerts_now:
                timestamp_s = frame_count / fps
                for a in new_alerts_now:
                    entry = {
                        "time": f"{int(timestamp_s//60):02d}:{int(timestamp_s%60):02d}",
                        "frame": frame_count,
                        "track_id": a["track_id"],
                        "risk": round(a["risk"], 3),
                        "model": model_name,
                        "alert_type": alert_t,
                        "features": a.get("features", {}),
                    }
                    self.alerts.append(entry)
                    prefix = "[HR ALERT]" if alert_t == "hr" else "[ALERT]"
                    self.log(f"[{entry['time']}] {prefix} dog#{a['track_id']} "
                             f"risk={a['risk']:.2f}  frame {frame_count}")
                if self.alert_sound_on.get():
                    self._play_alert_sound(alert_t)
                if alert_t == "hr":
                    self.root.after(0, self._flash_alert)

            if writer:
                writer.write(annotated)

            processed += 1
            elapsed = time.time() - start_time
            current_fps = processed / elapsed if elapsed > 0 else 0
            self.stats_vars["frames"].set(f"{frame_count:,}")
            self.stats_vars["persons"].set(f"{cur_persons}")
            self.stats_vars["dogs"].set(f"{cur_dogs}")
            self.stats_vars["alerts"].set(f"{total_alerts:,}")
            self.stats_vars["fps"].set(f"{current_fps:.1f}")
            if total_frames > 0:
                self.progress["value"] = (frame_count / total_frames) * 100
            self.root.after(0, self._display_frame, annotated)

        cap.release()
        if writer:
            writer.release()

        if recorder:
            saved = recorder.finalize(save=True)
            if saved and saved.get("_path"):
                self.log(f"[analytics] Session saved: {saved['_path']}")

        total_time = time.time() - start_time
        if self.stop_requested:
            self._update_status(f"Stopped after {processed:,} frames")
        else:
            self._update_status(f"Complete - {processed:,} frames in {total_time:.1f}s "
                                f"({total_alerts} alerts)")
        if output_path:
            self.log(f"Output saved: {output_path}")

        if not is_live:
            model_line = f"Model: {model_name}  |  Alert: {alert_t.upper()}\n\n"
            if self.alerts:
                self.export_btn.config(state="normal")
                messagebox.showinfo("Analysis Complete", model_line +
                                    f"Found {len(self.alerts)} alert(s)!\n"
                                    f"Frames: {processed:,}  |  "
                                    f"Peak dogs in frame: {peak_dogs}")
            else:
                messagebox.showinfo("Analysis Complete", model_line +
                                    f"No aggression risk detected.\n"
                                    f"Frames: {processed:,}  |  "
                                    f"Peak dogs in frame: {peak_dogs}")

    # ── analytics ────────────────────────────────────────────────────

    def open_dashboard(self):
        """Generate the analytics dashboard from saved sessions and open it."""
        try:
            from src.analytics import generate_dashboard
            path = generate_dashboard(open_browser=True)
            self.log(f"[analytics] Dashboard opened: {path}")
        except Exception as e:
            messagebox.showerror("Analytics",
                                 f"Could not generate the dashboard:\n{e}")

    # ── helpers ──────────────────────────────────────────────────────

    def log(self, msg):
        def _append():
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
        self.root.after(0, _append)

    def _update_status(self, msg):
        self.root.after(0, lambda: self.status_label.config(text=msg))

    def _section_header(self, parent, text):
        tk.Label(parent, text=text, bg="#252525", fg="#f59e0b",
                 font=("Segoe UI", 11, "bold"), anchor="w",
                 ).pack(fill="x", padx=15, pady=(10, 8))

    def _slider(self, parent, label, variable, from_, to, is_int=False):
        container = tk.Frame(parent, bg="#252525")
        container.pack(fill="x", padx=15, pady=(3, 8))
        header = tk.Frame(container, bg="#252525")
        header.pack(fill="x")
        tk.Label(header, text=label, bg="#252525", fg="#cccccc",
                 font=("Segoe UI", 9)).pack(side="left")
        value_label = tk.Label(header, text="", bg="#252525", fg="#f59e0b",
                               font=("Segoe UI", 9, "bold"))
        value_label.pack(side="right")

        def update_label(*_):
            val = variable.get()
            value_label.config(text=f"{int(val) if is_int else round(val, 2)}")
        variable.trace_add("write", update_label)
        update_label()

        slider = ttk.Scale(container, from_=from_, to=to, variable=variable,
                           orient="horizontal")
        slider.pack(fill="x")
        if is_int:
            slider.config(command=lambda v: variable.set(int(float(v))))

    def export_alerts(self):
        if not self.alerts:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"straydog_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        if path:
            with open(path, "w") as f:
                json.dump(self.alerts, f, indent=2)
            messagebox.showinfo("Exported", f"Alerts saved to:\n{path}")


def main():
    root = tk.Tk()
    StrayDogMonitorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
