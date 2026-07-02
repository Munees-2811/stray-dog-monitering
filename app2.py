"""
Stray Dog Monitoring System — Streamlit web app.

Browser-based control room with everything the desktop app does, plus a CCTV
module and the analytics dashboard embedded in-page:

  - Live Monitor ....... Video File / Webcam / ESP32-CAM / CCTV (RTSP) sources,
                         YOLO26 + YOLO11 model picker, Normal / HR alert modes,
                         live annotated frames, stats, alert log + JSON export,
                         annotated-video saving, +10 s skip for video files
  - Analytics .......... the offline HTML dashboard rendered in-page
                         (sessions are recorded exactly like the desktop app)
  - CCTV Cameras ....... register named RTSP cameras, test them, snapshot wall;
                         any registered camera becomes a Live Monitor source
  - ESP32 sensor ....... HC-SR04 distance readout + proximity alerts while
                         monitoring

Run with:
    streamlit run app2.py
"""

import io
import json
import os
import time
import wave
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

# Make RTSP capture fail fast instead of hanging (must be set before cv2 use)
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;4000000"
)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2

from src.config import load_config, PROJECT_ROOT

# ── static data ───────────────────────────────────────────────────────

MODEL_VARIANTS = {
    "yolo26n.pt": "YOLO26 Nano — fastest (CPU friendly)",
    "yolo26s.pt": "YOLO26 Small — fast",
    "yolo26m.pt": "YOLO26 Medium — balanced",
    "yolo26l.pt": "YOLO26 Large — more accurate",
    "yolo26x.pt": "YOLO26 XLarge — best accuracy",
    "yolo11n.pt": "YOLO11 Nano — fastest (CPU friendly)",
    "yolo11m.pt": "YOLO11 Medium — balanced",
    "yolo11x.pt": "YOLO11 XLarge — best accuracy",
}

SOURCE_VIDEO = "Video file"
SOURCE_WEBCAM = "Webcam"
SOURCE_ESP = "ESP32-CAM"
SOURCE_CCTV = "CCTV (RTSP)"

CAMERAS_FILE = PROJECT_ROOT / "data" / "cameras.json"

CFG = load_config()


# ── small helpers ─────────────────────────────────────────────────────

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


def grab_frame(url_or_index, timeout_s=6):
    """Grab a single frame from any cv2-openable source. Returns frame or None."""
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


def esp_get(ip, endpoint, timeout=2):
    with urllib.request.urlopen(f"http://{ip}/{endpoint}", timeout=timeout) as r:
        return json.loads(r.read())


@st.cache_data
def beep_wav(times=1, freq=1000, dur=0.45):
    """Small in-memory WAV used for browser alert sounds."""
    sr = 22050
    t = np.linspace(0, dur, int(sr * dur), False)
    tone = np.sin(2 * np.pi * freq * t) * 0.6
    fade = np.minimum(1, np.linspace(0, 8, tone.size))          # click-free attack
    tone = tone * fade * fade[::-1]
    gap = np.zeros(int(sr * 0.08))
    sig = np.concatenate([np.concatenate([tone, gap]) for _ in range(times)])
    pcm = (sig * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def play_alert_sound(placeholder, hr):
    try:
        placeholder.audio(beep_wav(times=3 if hr else 1,
                                   freq=1500 if hr else 1000),
                          format="audio/wav", autoplay=True)
    except TypeError:      # older Streamlit without autoplay
        placeholder.audio(beep_wav(times=3 if hr else 1), format="audio/wav")


@st.cache_resource(max_entries=2, show_spinner="Loading detection model …")
def get_pipeline(model, pose_model, det_conf, risk_threshold, sustain_frames):
    from src.pipeline import StrayDogMonitor
    return StrayDogMonitor(
        detector_path=model, pose_model=pose_model, det_conf=det_conf,
        risk_threshold=risk_threshold, sustain_frames=sustain_frames,
    )


def ss_init():
    d = CFG["detector"]; r = CFG["risk"]
    defaults = {
        "monitoring": False,
        "pos_frame": 0,          # resume point for video files across reruns
        "seek_seconds": 0,       # pending +10s skips
        "video_fps": 30.0,
        "alerts": [],
        # counts are instantaneous (in THIS frame) + peak concurrent, never a
        # per-frame running sum (that made 4 dogs read as hundreds).
        "totals": {"frames": 0, "cur_persons": 0, "cur_dogs": 0,
                   "peak_persons": 0, "peak_dogs": 0, "alerts": 0},
        "run_meta": None,        # dict while a run is active (model, source, …)
        "last_summary": None,
        "esp_last": None,
        "uploaded_path": None,
        "uploaded_key": None,
        "proximity_log_ts": 0.0,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("s_model",
        d["model"] if d["model"] in MODEL_VARIANTS else "yolo26n.pt")
    st.session_state.setdefault("s_risk", float(r["threshold"]))
    st.session_state.setdefault("s_conf", float(d["conf"]))


def stop_monitoring():
    st.session_state.monitoring = False


def request_skip():
    st.session_state.seek_seconds += 10


def start_monitoring(meta):
    st.session_state.monitoring = True
    st.session_state.pos_frame = 0
    st.session_state.seek_seconds = 0
    st.session_state.alerts = []
    st.session_state.totals = {"frames": 0, "cur_persons": 0, "cur_dogs": 0,
                               "peak_persons": 0, "peak_dogs": 0, "alerts": 0}
    st.session_state.run_meta = meta
    st.session_state.last_summary = None
    get_pipeline.clear()          # fresh tracker state for a fresh run


# ── page setup ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stray Dog Monitoring System",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)
ss_init()

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Stray Dog Monitoring System")
st.caption("Real-time stray-dog aggression detection — YOLO26 · geometric risk "
           "engine · ESP32 + CCTV inputs · session analytics")


# ── sidebar: settings (mirrors the desktop app) ──────────────────────

with st.sidebar:
    st.header("Settings")

    st.subheader("1 · Detection model")
    model_name = st.selectbox(
        "Model", list(MODEL_VARIANTS.keys()),
        format_func=lambda m: f"{m} — {MODEL_VARIANTS[m]}",
        key="s_model",
        help="COCO-pretrained; auto-downloads on first use.")
    custom_weights = st.text_input(
        "Custom weights (.pt) — optional",
        placeholder=r"runs\detect\straydog_finetune\weights\best.pt",
        help="Path to fine-tuned weights (see finetune.py). "
             "When set, this overrides the model selected above.")
    pose_enabled = st.toggle("Use pose model (human skeleton)",
                             value=bool(CFG["detector"].get("pose_model")))

    st.subheader("2 · Alert type")
    alert_type = st.radio(
        "Alert mode", ["normal", "hr"], horizontal=True,
        format_func=lambda v: "Normal" if v == "normal" else "HR (high-risk)")
    hr_drop = CFG["alerts"].get("hr_threshold_drop", 0.10)
    if alert_type == "hr":
        st.caption(f"HR mode: risk threshold −{hr_drop:.2f}, triple beep.")
    sound_on = st.toggle("Alert sound", value=True)

    st.subheader("3 · Thresholds")
    det_conf = st.slider("Detection confidence", 0.10, 0.95, key="s_conf")
    risk_threshold = st.slider("Risk threshold", 0.10, 0.95, key="s_risk")
    sustain_frames = st.slider("Sustain frames (N)", 1, 20,
                               int(CFG["risk"]["sustain_frames"]))
    skip_frames = st.slider("Skip frames", 1, 30,
                            int(CFG["inference"].get("skip_frames", 1)))
    save_output = st.toggle("Save annotated output video",
                            value=bool(CFG["inference"].get("save_output", True)))

    st.subheader("4 · ESP32 sensor (HC-SR04)")
    esp_ip = st.text_input("ESP32 IP", value=CFG["hardware"].get("esp_ip", ""))
    dist_alert_cm = st.slider("Proximity alert (cm)", 10, 400,
                              int(CFG["hardware"].get("proximity_alert_cm", 100)))
    esp_poll = st.toggle("Poll distance while monitoring", value=bool(esp_ip))
    if st.button("Read distance now", use_container_width=True):
        try:
            data = esp_get(esp_ip, "distance")
            d = data.get("distance_cm")
            st.session_state.esp_last = d
            st.success("Out of range" if d is None else f"{d:.1f} cm")
        except Exception as e:
            st.error(f"ESP32 not reachable: {e}")


# ── tabs ──────────────────────────────────────────────────────────────

tab_monitor, tab_dash, tab_cctv = st.tabs(
    ["🎥  Live Monitor", "📊  Analytics Dashboard", "🎦  CCTV Cameras"])


# ═══════════════════════════════ CCTV TAB ═════════════════════════════
with tab_cctv:
    st.subheader("CCTV camera manager")
    st.caption("Register RTSP/HTTP cameras once — they appear as Live Monitor "
               "sources. Typical URL: rtsp://user:pass@192.168.1.64:554/stream1")

    cams = load_cameras()

    with st.form("add_cam", clear_on_submit=True):
        c1, c2, c3 = st.columns([2, 4, 1])
        cam_name = c1.text_input("Camera name", placeholder="Gate camera")
        cam_url = c2.text_input("Stream URL",
                                placeholder="rtsp://user:pass@ip:554/stream1")
        add = c3.form_submit_button("Add", use_container_width=True)
    if add:
        if not cam_name.strip() or not cam_url.strip():
            st.warning("Both a name and a URL are required.")
        elif any(c["name"] == cam_name.strip() for c in cams):
            st.warning("A camera with that name already exists.")
        else:
            cams.append({"name": cam_name.strip(), "url": cam_url.strip()})
            save_cameras(cams)
            st.success(f"Added “{cam_name.strip()}”.")
            st.rerun()

    if not cams:
        st.info("No cameras registered yet.")
    else:
        st.markdown("##### Registered cameras")
        for i, cam in enumerate(cams):
            c1, c2, c3, c4 = st.columns([2, 4, 1, 1])
            c1.write(f"**{cam['name']}**")
            c2.code(cam["url"], language=None)
            if c3.button("Test", key=f"test_{i}", use_container_width=True):
                with st.spinner(f"Connecting to {cam['name']} …"):
                    frame = grab_frame(cam["url"])
                if frame is None:
                    st.error(f"“{cam['name']}” did not return a frame — check "
                             f"the URL, credentials and network.")
                else:
                    st.success(f"“{cam['name']}” online — "
                               f"{frame.shape[1]}×{frame.shape[0]}")
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=480)
            if c4.button("Remove", key=f"del_{i}", use_container_width=True):
                cams.pop(i)
                save_cameras(cams)
                st.rerun()

        st.markdown("##### Snapshot wall")
        st.caption("One live frame from every registered camera.")
        if st.button("Refresh snapshots"):
            cols = st.columns(min(3, max(1, len(cams))))
            for i, cam in enumerate(cams):
                with cols[i % len(cols)]:
                    with st.spinner(cam["name"]):
                        frame = grab_frame(cam["url"])
                    if frame is None:
                        st.error(f"{cam['name']}: offline")
                    else:
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                 caption=cam["name"],
                                 use_container_width=True)


# ═══════════════════════════ DASHBOARD TAB ════════════════════════════
with tab_dash:
    from src.analytics import generate_dashboard
    from src.analytics.recorder import load_sessions

    sessions = load_sessions()
    c1, c2 = st.columns([3, 1])
    c1.subheader("Analytics dashboard")
    c1.caption(f"{len(sessions)} recorded session(s) in data/sessions/ — every "
               f"monitoring run (desktop or web) is aggregated here.")
    if not sessions:
        st.info("No sessions recorded yet — run the Live Monitor first; every "
                "run is saved automatically.")
    else:
        dash_path = generate_dashboard(open_browser=False)
        html = Path(dash_path).read_text(encoding="utf-8")
        c2.download_button("Download dashboard (HTML)", data=html,
                           file_name="dashboard.html", mime="text/html",
                           use_container_width=True)
        st.components.v1.html(html, height=2350, scrolling=True)


# ═══════════════════════════ MONITOR TAB ══════════════════════════════
with tab_monitor:
    src_type = st.radio(
        "Source", [SOURCE_VIDEO, SOURCE_WEBCAM, SOURCE_ESP, SOURCE_CCTV],
        horizontal=True, label_visibility="collapsed",
        disabled=st.session_state.monitoring)

    source_ready, source_desc, open_args = False, "", None

    if src_type == SOURCE_VIDEO:
        c1, c2 = st.columns(2)
        up = c1.file_uploader("Upload a video",
                              type=["mp4", "avi", "mov", "mkv", "webm", "wmv"])
        path_in = c2.text_input("… or enter a local video path",
                                placeholder=r"C:\videos\street.mp4")
        if up is not None:
            key = f"{up.name}:{up.size}"
            if st.session_state.uploaded_key != key:
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(up.name).suffix)
                tmp.write(up.getbuffer())
                tmp.close()
                st.session_state.uploaded_path = tmp.name
                st.session_state.uploaded_key = key
            open_args = st.session_state.uploaded_path
            source_ready, source_desc = True, up.name
        elif path_in.strip():
            p = path_in.strip().strip('"')
            if Path(p).exists():
                open_args, source_ready, source_desc = p, True, Path(p).name
            else:
                st.warning("File not found at that path.")

    elif src_type == SOURCE_WEBCAM:
        cam_idx = st.number_input("Camera index", 0, 10, 0)
        open_args, source_ready = int(cam_idx), True
        source_desc = f"webcam #{cam_idx}"

    elif src_type == SOURCE_ESP:
        if esp_ip.strip():
            open_args = f"esp://{esp_ip.strip()}"
            source_ready, source_desc = True, f"ESP32-CAM @ {esp_ip.strip()}"
            st.caption(f"Stream: http://{esp_ip.strip()}:81/stream — set the IP "
                       f"in the sidebar (ESP32 sensor section).")
        else:
            st.warning("Enter the ESP32 IP in the sidebar first.")

    else:  # CCTV
        cams = load_cameras()
        manual = st.text_input("RTSP/HTTP stream URL",
                               placeholder="rtsp://user:pass@ip:554/stream1")
        if cams:
            pick = st.selectbox(
                "… or pick a registered camera",
                ["—"] + [c["name"] for c in cams])
            if pick != "—":
                cam = next(c for c in cams if c["name"] == pick)
                open_args, source_ready = cam["url"], True
                source_desc = f"CCTV “{cam['name']}”"
        if not source_ready and manual.strip():
            open_args, source_ready = manual.strip(), True
            source_desc = "CCTV stream"
        if not cams and not manual.strip():
            st.info("Register cameras in the CCTV Cameras tab, or paste a "
                    "stream URL above.")

    is_live_src = src_type != SOURCE_VIDEO

    # ── controls row ──────────────────────────────────────────────────
    b1, b2, b3, b4 = st.columns([1.4, 1, 1, 3])
    if not st.session_state.monitoring:
        if b1.button("▶  Start monitoring", type="primary",
                     disabled=not source_ready, use_container_width=True):
            eff_risk = risk_threshold
            if alert_type == "hr":
                eff_risk = max(0.10, risk_threshold - hr_drop)
            # custom fine-tuned weights override the picker when provided
            run_model = model_name
            cw = custom_weights.strip().strip('"')
            if cw:
                if not Path(cw).exists():
                    st.error(f"Custom weights not found: {cw}")
                    st.stop()
                run_model = cw
            start_monitoring({
                "src_type": src_type, "open_args": open_args,
                "source_desc": source_desc, "model": run_model,
                "pose": CFG["detector"].get("pose_model") if pose_enabled else None,
                "alert_type": alert_type, "eff_risk": eff_risk,
                "det_conf": det_conf, "sustain": sustain_frames,
                "skip": skip_frames, "save": save_output and not is_live_src,
                "is_live": is_live_src, "esp_ip": esp_ip.strip(),
                "esp_poll": esp_poll and bool(esp_ip.strip()),
            })
            st.rerun()
    else:
        b1.button("■  Stop", type="primary", on_click=stop_monitoring,
                  use_container_width=True)
        if st.session_state.run_meta and not st.session_state.run_meta["is_live"]:
            b2.button("⏩  +10 s", on_click=request_skip,
                      use_container_width=True)
    if st.session_state.monitoring and st.session_state.run_meta:
        m = st.session_state.run_meta
        b4.caption(f"**{m['source_desc']}** · {m['model']} · "
                   f"{m['alert_type'].upper()} mode · risk ≥ {m['eff_risk']:.2f}")

    # ── live placeholders ─────────────────────────────────────────────
    frame_ph = st.empty()
    mc = st.columns(6)
    metric_phs = {k: mc[i].empty() for i, k in enumerate(
        ["frames", "persons", "dogs", "alerts", "fps", "distance"])}
    progress_ph = st.empty()
    sound_ph = st.empty()
    st.markdown("##### Alert log")
    alerts_ph = st.empty()

    def show_metrics(fps_val="–", dist_val="–"):
        t = st.session_state.totals
        metric_phs["frames"].metric("Frames", f"{t['frames']:,}")
        metric_phs["persons"].metric("Persons now", f"{t['cur_persons']}",
                                     help=f"Peak concurrent: {t['peak_persons']}")
        metric_phs["dogs"].metric("Dogs now", f"{t['cur_dogs']}",
                                  help=f"Peak concurrent: {t['peak_dogs']}")
        metric_phs["alerts"].metric("Alerts", f"{t['alerts']:,}")
        metric_phs["fps"].metric("FPS", fps_val)
        metric_phs["distance"].metric("Dist (cm)", dist_val)

    def show_alerts():
        rows = st.session_state.alerts
        if rows:
            alerts_ph.dataframe(
                [{k: a[k] for k in
                  ("time", "frame", "track_id", "risk", "alert_type")}
                 for a in reversed(rows[-50:])],
                use_container_width=True, height=220)
        else:
            alerts_ph.caption("No alerts yet.")

    show_metrics()
    show_alerts()

    # export row (available after a finished run)
    if st.session_state.last_summary and not st.session_state.monitoring:
        s = st.session_state.last_summary
        if s.get("stopped"):
            st.info(s["text"])
        elif s.get("alerts"):
            st.error(s["text"])
        else:
            st.success(s["text"])
        if s.get("output_path"):
            st.caption(f"Annotated video saved: `{s['output_path']}`")
        if st.session_state.alerts:
            st.download_button(
                "Export alerts (JSON)",
                data=json.dumps(st.session_state.alerts, indent=2),
                file_name=f"straydog_alerts_"
                          f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json")

    # ── the monitoring loop ───────────────────────────────────────────
    if st.session_state.monitoring and st.session_state.run_meta:
        m = st.session_state.run_meta

        pipeline = get_pipeline(m["model"], m["pose"], m["det_conf"],
                                m["eff_risk"], m["sustain"])

        # open the source (rerun-safe: video files resume from pos_frame)
        if m["src_type"] == SOURCE_ESP:
            from src.sources import MJPEGCapture
            ip = m["esp_ip"]
            try:
                info = esp_get(ip, "status", timeout=3)
                st.toast(f"ESP32 online — RSSI {info.get('wifi_rssi', '?')} dBm")
            except Exception as e:
                st.session_state.monitoring = False
                st.error(f"ESP32 not reachable at {ip}: {e}")
                st.stop()
            cap = MJPEGCapture(f"http://{ip}:81/stream", timeout=5)
        else:
            cap = cv2.VideoCapture(m["open_args"])

        if not cap.isOpened():
            st.session_state.monitoring = False
            st.error("Could not open the video source.")
            st.stop()

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0:
            fps = 30.0
        st.session_state.video_fps = fps
        total_frames = (0 if m["is_live"]
                        else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

        frame_count = st.session_state.pos_frame
        if not m["is_live"] and frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        # analytics recorder (same fail-safe layer as the desktop app);
        # a run resumed after a +10s click records as a fresh stretch
        recorder = None
        acfg = CFG.get("analytics", {}) or {}
        if acfg.get("enabled", True):
            try:
                from src.analytics import SessionRecorder
                recorder = SessionRecorder(
                    model=m["model"],
                    source={SOURCE_VIDEO: "video", SOURCE_WEBCAM: "webcam",
                            SOURCE_ESP: "espcam",
                            SOURCE_CCTV: "cctv"}[m["src_type"]],
                    alert_type=m["alert_type"],
                    risk_threshold=m["eff_risk"], det_conf=m["det_conf"],
                    sessions_dir=acfg.get("sessions_dir", "data/sessions"),
                    timeline_max_points=acfg.get("timeline_max_points", 600),
                )
            except Exception:
                recorder = None

        writer, output_path = None, None
        if m["save"]:
            out_dir = PROJECT_ROOT / CFG["inference"].get("output_dir", "outputs")
            out_dir.mkdir(parents=True, exist_ok=True)
            tag = "hr" if m["alert_type"] == "hr" else "norm"
            output_path = str(out_dir / (
                f"web_{Path(m['model']).stem}_{tag}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"))
            writer = cv2.VideoWriter(output_path,
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (width, height))

        processed = 0
        start_time = time.time()
        last_esp_poll = 0.0
        dist_display = "–"
        ended_naturally = False

        # try/finally is load-bearing: Streamlit interrupts this script at the
        # next UI call when Stop is clicked (or the tab closes), so cleanup
        # must run on that interrupt too — otherwise the annotated video is
        # left unplayable and the analytics session is never saved.
        try:
            while st.session_state.monitoring:
                # pending +10 s skips (video files only)
                if not m["is_live"] and st.session_state.seek_seconds > 0:
                    frame_count += int(st.session_state.seek_seconds * fps)
                    st.session_state.seek_seconds = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

                ret, frame = cap.read()
                if not ret:
                    ended_naturally = True
                    break
                frame_count += 1
                st.session_state.pos_frame = frame_count

                if (frame_count - 1) % m["skip"] != 0:
                    if writer:
                        writer.write(frame)
                    continue

                results = pipeline.process_frame(frame)
                annotated = pipeline.draw_results(frame, results)
                persons_now = len(pipeline._last_persons)

                if recorder:
                    recorder.record_frame(results, persons_now, frame_count)

                t = st.session_state.totals
                t["frames"] = frame_count
                t["cur_persons"] = persons_now
                t["cur_dogs"] = len(results)
                t["peak_persons"] = max(t["peak_persons"], persons_now)
                t["peak_dogs"] = max(t["peak_dogs"], len(results))

                new_alerts = [r for r in results if r.get("new_alert")]
                if new_alerts:
                    t["alerts"] += len(new_alerts)
                    ts = frame_count / fps
                    for a in new_alerts:
                        st.session_state.alerts.append({
                            "time": f"{int(ts // 60):02d}:{int(ts % 60):02d}",
                            "frame": frame_count,
                            "track_id": a["track_id"],
                            "risk": round(a["risk"], 3),
                            "model": m["model"],
                            "alert_type": m["alert_type"],
                            "features": a.get("features", {}),
                        })
                    if sound_on:
                        play_alert_sound(sound_ph, hr=m["alert_type"] == "hr")
                    show_alerts()

                # ESP ultrasonic poll (~2× per second, never blocks the loop long)
                if m["esp_poll"] and time.time() - last_esp_poll > 0.5:
                    last_esp_poll = time.time()
                    try:
                        d = esp_get(m["esp_ip"], "distance", timeout=1).get(
                            "distance_cm")
                        dist_display = "–" if d is None else f"{d:.0f}"
                        if (d is not None and d < dist_alert_cm and
                                time.time() - st.session_state.proximity_log_ts > 10):
                            st.session_state.proximity_log_ts = time.time()
                            st.toast(f"⚠ Proximity: {d:.1f} cm", icon="⚠️")
                    except Exception:
                        dist_display = "×"

                if writer:
                    writer.write(annotated)

                processed += 1
                elapsed = time.time() - start_time
                cur_fps = processed / elapsed if elapsed > 0 else 0.0

                frame_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                               use_container_width=True)
                if processed % 3 == 1:
                    show_metrics(f"{cur_fps:.1f}", dist_display)
                    if total_frames > 0:
                        progress_ph.progress(
                            min(1.0, frame_count / total_frames),
                            text=f"{frame_count:,} / {total_frames:,} frames")

        finally:
            # runs on natural end, Stop click AND Streamlit's script interrupt
            cap.release()
            if writer:
                writer.release()
            if recorder:
                recorder.finalize(save=True)

        # ── wrap up (reached only when the loop exited normally) ────────
        t = st.session_state.totals
        if ended_naturally:
            st.session_state.monitoring = False
            st.session_state.last_summary = {
                "alerts": t["alerts"] > 0,
                "output_path": output_path,
                "text": (f"Analysis complete — {t['alerts']} alert(s), "
                         f"{t['frames']:,} frames, peak {t['peak_dogs']} dog(s) "
                         f"in frame ({m['model']}, "
                         f"{m['alert_type'].upper()} mode)."),
            }
            st.rerun()
        else:
            # Stop was clicked between loop iterations
            st.session_state.last_summary = {
                "stopped": True, "output_path": output_path,
                "text": f"Stopped after {t['frames']:,} frames "
                        f"({t['alerts']} alert(s)).",
            }
