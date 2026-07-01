"""
Greedy IoU tracker.

Assigns a persistent id to each dog across frames and holds the per-dog
temporal memory (box + risk history) that the risk engine needs to reason
about velocity, posture change and sustained aggression.
"""

from collections import deque


def iou(a, b):
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def box_center(b):
    return ((b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0)


def box_aspect_ratio(b):
    w = max(1, b["x2"] - b["x1"])
    h = max(1, b["y2"] - b["y1"])
    return w / h


class Track:
    """Per-dog state with temporal risk memory."""

    _next_id = 1

    def __init__(self, box, history_len=30):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.box_history = deque([box], maxlen=history_len)
        self.risk_history = deque(maxlen=history_len)
        self.missed = 0
        self.risk_ema = 0.0
        self.sustained_frames = 0     # consecutive frames above threshold
        self.last_alert_frame = -10_000
        self.age = 0                  # total frames since birth

    def update(self, box):
        self.box = box
        self.box_history.append(box)
        self.missed = 0
        self.age += 1

    def mark_missed(self):
        self.missed += 1
        self.age += 1

    @classmethod
    def reset_ids(cls):
        cls._next_id = 1


class IoUTracker:
    """Greedy IoU association — sufficient for dog-scale object counts."""

    def __init__(self, iou_threshold=0.2, max_missed=15, history_len=30):
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.history_len = history_len

    def update(self, detections):
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                pairs.append((iou(track.box, det), ti, di))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        for score, ti, di in pairs:
            if score < self.iou_threshold:
                break
            if ti in used_t or di in used_d:
                continue
            self.tracks[ti].update(detections[di])
            used_t.add(ti)
            used_d.add(di)

        for di, det in enumerate(detections):
            if di not in used_d:
                self.tracks.append(Track(det, history_len=self.history_len))

        for ti, track in enumerate(self.tracks):
            if ti not in used_t:
                track.mark_missed()
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        return self.tracks
