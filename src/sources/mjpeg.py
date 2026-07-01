"""
MJPEG stream reader for the ESP32-CAM.

``cv2.VideoCapture`` is unreliable (and on Windows outright fails) with
``multipart/x-mixed-replace`` HTTP streams. This reader instead pulls the raw
byte stream and slices out JPEG frames by scanning for SOI/EOI markers,
exposing the small subset of the ``cv2.VideoCapture`` API the pipeline uses.
"""

import urllib.request

import cv2
import numpy as np


class MJPEGCapture:
    """Drop-in ``VideoCapture``-like reader for ESP32-CAM MJPEG over HTTP."""

    def __init__(self, url, timeout=5):
        self.url = url
        self._stream = None
        self._buf = b""
        self._opened = False
        try:
            req = urllib.request.Request(url, headers={"Connection": "keep-alive"})
            self._stream = urllib.request.urlopen(req, timeout=timeout)
            self._opened = True
        except Exception:
            self._opened = False

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened:
            return False, None
        try:
            while True:
                self._buf += self._stream.read(4096)
                start = self._buf.find(b"\xff\xd8")   # JPEG SOI
                end = self._buf.find(b"\xff\xd9")     # JPEG EOI
                if start != -1 and end != -1 and end > start:
                    jpg = self._buf[start:end + 2]
                    self._buf = self._buf[end + 2:]
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        return True, frame
        except Exception:
            self._opened = False
            return False, None

    def get(self, prop):
        defaults = {
            cv2.CAP_PROP_FPS:          15.0,
            cv2.CAP_PROP_FRAME_COUNT:   0,
            cv2.CAP_PROP_FRAME_WIDTH:  640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }
        return defaults.get(prop, 0)

    def set(self, prop, value):
        return False   # live stream — seeking is not supported

    def release(self):
        self._opened = False
        if self._stream:
            try:
                self._stream.close()
            except Exception:
                pass
