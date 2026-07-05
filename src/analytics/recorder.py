"""
Session recorder — the ML/CV analytics data layer.

While the monitor runs, a ``SessionRecorder`` collects a lightweight sample per
processed frame (risk, dog/person counts) plus every alert, and writes one JSON
file per monitoring session to ``data/sessions/``. The dashboard generator
(``src/analytics/dashboard.py``) aggregates those files into charts.

The recorder is deliberately fail-safe: it only ever *observes* the pipeline,
every public method swallows its own errors, and nothing in detection or
alerting depends on it — analytics can never break monitoring.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from src.config import PROJECT_ROOT

SCHEMA_VERSION = 1


class SessionRecorder:
    """Collects per-frame samples + alerts for one monitoring session."""

    def __init__(self, model, source, alert_type, risk_threshold, det_conf,
                 sessions_dir="data/sessions", timeline_max_points=600):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.started = datetime.now().isoformat(timespec="seconds")
        self.model = model
        self.source = source
        self.alert_type = alert_type
        self.risk_threshold = risk_threshold
        self.det_conf = det_conf
        self.timeline_max_points = timeline_max_points

        self.sessions_dir = Path(sessions_dir)
        if not self.sessions_dir.is_absolute():
            self.sessions_dir = PROJECT_ROOT / self.sessions_dir

        self._t0 = time.time()
        self._samples = []     # (t, max_risk, dogs, persons)
        self._alerts = []
        self._frames_read = 0

    # ── collection (called from the monitoring loop) ─────────────────

    def record_frame(self, results, persons_count, frame_number):
        """Record one processed frame. ``results`` is the pipeline output."""
        try:
            self._frames_read = frame_number
            t = time.time() - self._t0
            max_risk = max((r["risk"] for r in results), default=0.0)
            self._samples.append(
                (round(t, 2), round(max_risk, 3), len(results), persons_count)
            )
            for r in results:
                if r.get("new_alert"):
                    self._alerts.append({
                        "t": round(t, 2),
                        "frame": frame_number,
                        "track_id": r["track_id"],
                        "risk": round(r["risk"], 3),
                        "behavior": r.get("behavior_label", ""),
                        "features": r.get("features", {}),
                        "alert_type": self.alert_type,
                    })
        except Exception:
            pass  # analytics must never break monitoring

    # ── persistence ──────────────────────────────────────────────────

    def _downsampled_timeline(self):
        """Bucket samples so the saved timeline stays <= timeline_max_points.
        Each bucket keeps its MAX risk (never smooth away a spike) and the max
        dog/person counts seen inside it."""
        n = len(self._samples)
        limit = self.timeline_max_points
        if n <= limit:
            return [
                {"t": t, "risk": r, "dogs": d, "persons": p}
                for t, r, d, p in self._samples
            ]
        out = []
        bucket = max(1, n // limit)
        for i in range(0, n, bucket):
            chunk = self._samples[i:i + bucket]
            out.append({
                "t": chunk[0][0],
                "risk": max(c[1] for c in chunk),
                "dogs": max(c[2] for c in chunk),
                "persons": max(c[3] for c in chunk),
            })
        return out

    def finalize(self, save=True):
        """Build the session record and (optionally) write it to disk.
        Returns the record dict, or None if nothing was captured / save failed."""
        try:
            if not self._samples:
                return None
            duration = time.time() - self._t0
            processed = len(self._samples)
            record = {
                "schema": SCHEMA_VERSION,
                "session_id": self.session_id,
                "started": self.started,
                "ended": datetime.now().isoformat(timespec="seconds"),
                "duration_s": round(duration, 1),
                "source": self.source,
                "model": self.model,
                "alert_type": self.alert_type,
                "risk_threshold": round(float(self.risk_threshold), 3),
                "det_conf": round(float(self.det_conf), 3),
                "frames_read": self._frames_read,
                "frames_processed": processed,
                "avg_fps": round(processed / duration, 1) if duration > 0 else 0.0,
                "dogs_total": sum(s[2] for s in self._samples),
                "persons_total": sum(s[3] for s in self._samples),
                "peak_risk": max(s[1] for s in self._samples),
                "alerts_total": len(self._alerts),
                "alerts": self._alerts,
                "timeline": self._downsampled_timeline(),
            }
            if save:
                self.sessions_dir.mkdir(parents=True, exist_ok=True)
                path = self.sessions_dir / f"session_{self.session_id}.json"
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(record, fh)
                record["_path"] = str(path)
            return record
        except Exception:
            return None


def load_sessions(sessions_dir="data/sessions"):
    """Load all saved session records, oldest first. Bad files are skipped."""
    d = Path(sessions_dir)
    if not d.is_absolute():
        d = PROJECT_ROOT / d
    sessions = []
    if not d.exists():
        return sessions
    for path in sorted(d.glob("session_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                rec = json.load(fh)
            if isinstance(rec, dict) and rec.get("timeline"):
                sessions.append(rec)
        except Exception:
            continue
    return sessions
