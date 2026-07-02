"""
Latest-frame capture for live sources (webcam / RTSP CCTV / ESP32-CAM).

Problem: a plain ``cap.read()`` loop consumes frames *sequentially*. When
inference is slower than the camera (e.g. 5 FPS pipeline on a 25 FPS CCTV
stream), unread frames pile up in the driver/network buffer and the preview
drifts seconds — then minutes — behind reality. For a safety system, alerting
on the past is useless.

Fix: a reader thread drains the source continuously and keeps ONLY the newest
frame. The pipeline always analyses "now" and stale frames are dropped, so the
system stays live at whatever FPS the hardware can actually sustain.

Only use this for live sources — video files must be read sequentially.
"""

import threading
import time


class LatestFrameCapture:
    """Threaded wrapper serving the newest frame from a VideoCapture-like source."""

    def __init__(self, cap):
        self.cap = cap
        self._lock = threading.Lock()
        self._frame = None
        self._alive = True
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self._alive = False
                time.sleep(0.05)
                continue
            self._alive = True
            with self._lock:
                self._frame = frame   # overwrite: older frames are dropped

    # ── VideoCapture-compatible surface ──────────────────────────────

    def read(self, timeout=5.0):
        """Return (True, newest_frame), waiting up to ``timeout`` for one."""
        deadline = time.time() + timeout
        while time.time() < deadline and not self._stop.is_set():
            with self._lock:
                frame, self._frame = self._frame, None
            if frame is not None:
                return True, frame
            time.sleep(0.005)
        return False, None

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop):
        return self.cap.get(prop)

    def set(self, prop, value):
        return self.cap.set(prop, value)

    def release(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()
