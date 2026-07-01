"""
End-to-end stray-dog monitoring pipeline.

Per frame:
    1. YOLO26 detects persons + dogs (YOLO11-pose adds human keypoints).
    2. A greedy IoU tracker follows each dog, keeping temporal history.
    3. The risk engine scores each dog from geometry + human posture.
    4. Temporal intelligence turns noisy per-frame risk into stable alerts:
         - EMA smoothing per track (removes single-frame spikes)
         - sustained alert: risk must stay above threshold for N frames
         - cooldown: suppress repeat alerts for the same dog

The result objects and drawing are UI-agnostic so the same pipeline powers
the desktop app, video/file runs and the webcam / ESP32-CAM live modes.
"""

from pathlib import Path

import cv2

from src.config import load_config
from src.detection import Detector
from src.tracking import IoUTracker
from src.risk import RiskEngine


class StrayDogMonitor:
    """Detection + tracking + temporally-smoothed risk scoring."""

    COLORS = {
        "person": (255, 180, 60),
        "dog_safe": (0, 220, 90),
        "dog_caution": (40, 200, 240),
        "dog_alert": (60, 60, 255),
        "skeleton": (220, 220, 220),
    }

    # COCO-Pose skeleton edges for drawing
    SKELETON = [
        (5, 7), (7, 9), (6, 8), (8, 10),         # arms
        (5, 6), (5, 11), (6, 12), (11, 12),      # torso
        (11, 13), (13, 15), (12, 14), (14, 16),  # legs
        (0, 5), (0, 6),                          # head to shoulders
    ]

    def __init__(self, detector_path=None, pose_model="__default__",
                 det_conf=None, risk_threshold=None, smoothing_alpha=None,
                 sustain_frames=None, cooldown_frames=None, device=None,
                 config=None):
        cfg = config or load_config()
        d, t, r = cfg["detector"], cfg["tracker"], cfg["risk"]

        detector_path = detector_path or d["model"]
        if pose_model == "__default__":
            pose_model = d.get("pose_model")
        det_conf = d["conf"] if det_conf is None else det_conf
        device = device or cfg.get("project", {}).get("device", "auto")
        if device == "auto":
            device = None  # let ultralytics choose

        self.detector = Detector(
            model_path=detector_path,
            pose_model=pose_model,
            conf=det_conf,
            iou=d["iou"],
            device=device,
        )
        self.risk_engine = RiskEngine(
            weights=r.get("weights"),
            pack_bonus=r.get("pack_bonus", 0.10),
        )
        self.tracker = IoUTracker(
            iou_threshold=t["iou_threshold"],
            max_missed=t["max_missed"],
            history_len=t["history_len"],
        )

        self.risk_threshold = r["threshold"] if risk_threshold is None else risk_threshold
        self.smoothing_alpha = r["smoothing_alpha"] if smoothing_alpha is None else smoothing_alpha
        self.sustain_frames = r["sustain_frames"] if sustain_frames is None else sustain_frames
        self.cooldown_frames = r["cooldown_frames"] if cooldown_frames is None else cooldown_frames

        self.frame_idx = 0
        self._last_persons = []

        print(f"[monitor] detector={detector_path} pose={pose_model} "
              f"conf={det_conf} risk_threshold={self.risk_threshold} "
              f"sustain={self.sustain_frames} cooldown={self.cooldown_frames}")

    # ── per-frame ────────────────────────────────────────────────────

    def process_frame(self, frame):
        self.frame_idx += 1

        det = self.detector.detect(frame)
        persons, dogs = det["persons"], det["dogs"]
        self._last_persons = persons

        tracks = self.tracker.update(dogs)

        results = []
        for t in tracks:
            if t.missed > 0:
                continue

            raw_risk, features = self.risk_engine.score(
                t, persons, frame.shape, len(dogs)
            )

            # EMA smoothing (first frame initialises to the raw value)
            if len(t.risk_history) == 0:
                t.risk_ema = raw_risk
            else:
                a = self.smoothing_alpha
                t.risk_ema = (1 - a) * t.risk_ema + a * raw_risk
            t.risk_history.append(t.risk_ema)

            # Sustained-alert counter
            if t.risk_ema >= self.risk_threshold:
                t.sustained_frames += 1
            else:
                t.sustained_frames = 0

            cooldown_ok = (self.frame_idx - t.last_alert_frame) > self.cooldown_frames
            alert = (t.sustained_frames >= self.sustain_frames) and cooldown_ok
            new_alert = alert and (t.sustained_frames == self.sustain_frames)
            if new_alert:
                t.last_alert_frame = self.frame_idx

            results.append({
                "x1": t.box["x1"], "y1": t.box["y1"],
                "x2": t.box["x2"], "y2": t.box["y2"],
                "track_id": t.id,
                "dog_confidence": t.box["confidence"],
                "risk_raw": raw_risk,
                "risk": t.risk_ema,
                "features": features,
                "sustained": t.sustained_frames,
                "sustain_target": self.sustain_frames,
                "alert": alert,
                "new_alert": new_alert,
                "history": list(t.risk_history),
                "behavior": "aggressive" if alert else "non_aggressive",
            })
        return results

    # ── drawing ──────────────────────────────────────────────────────

    def _dog_color(self, risk):
        if risk >= self.risk_threshold:
            return self.COLORS["dog_alert"]
        if risk >= self.risk_threshold * 0.6:
            return self.COLORS["dog_caution"]
        return self.COLORS["dog_safe"]

    def _draw_box(self, img, x1, y1, x2, y2, color, thickness=2, label=None):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_skeleton(self, img, keypoints):
        if keypoints is None:
            return
        for a, b in self.SKELETON:
            xa, ya, va = keypoints[a]
            xb, yb, vb = keypoints[b]
            if va < 0.3 or vb < 0.3:
                continue
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)),
                     self.COLORS["skeleton"], 2)
        for x, y, v in keypoints:
            if v < 0.3:
                continue
            cv2.circle(img, (int(x), int(y)), 3, self.COLORS["skeleton"], -1)

    def _draw_risk_bar(self, img, x, y, w, h, history):
        """Miniature sparkline of risk over the last N frames."""
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
        if len(history) < 2:
            return
        step = w / (len(history) - 1)
        pts = []
        for i, r in enumerate(history):
            px = int(x + i * step)
            py = int(y + h - r * h)
            pts.append((px, py))
        for i in range(1, len(pts)):
            color = (60, 60, 255) if history[i] >= self.risk_threshold else (0, 220, 90)
            cv2.line(img, pts[i - 1], pts[i], color, 2)
        ty = int(y + h - self.risk_threshold * h)   # threshold line
        cv2.line(img, (x, ty), (x + w, ty), (100, 100, 100), 1)

    def draw_results(self, frame, results):
        annotated = frame.copy()

        for p in self._last_persons:
            self._draw_box(
                annotated, p["x1"], p["y1"], p["x2"], p["y2"],
                self.COLORS["person"], thickness=2,
                label=f"person {p['confidence']:.2f}",
            )
            if "keypoints" in p:
                self._draw_skeleton(annotated, p["keypoints"])

        for r in results:
            color = self._dog_color(r["risk"])
            thickness = 3 if r["alert"] else 2
            if r["alert"]:
                label = f"ALERT dog#{r['track_id']} risk {r['risk']:.2f}"
            else:
                label = (f"dog#{r['track_id']} risk {r['risk']:.2f} "
                         f"[{r['sustained']}/{r['sustain_target']}]")
            self._draw_box(
                annotated, r["x1"], r["y1"], r["x2"], r["y2"],
                color, thickness=thickness, label=label,
            )
            bar_w, bar_h = 120, 30
            bx = r["x1"]
            by = max(0, r["y1"] - bar_h - 30)
            self._draw_risk_bar(annotated, bx, by, bar_w, bar_h, r["history"])

        return annotated

    # ── file / video helpers (headless CLI use) ──────────────────────

    def predict_image(self, image_path, save_path=None):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        results = self.process_frame(frame)
        annotated = self.draw_results(frame, results)
        if save_path:
            cv2.imwrite(str(save_path), annotated)
            print(f"[monitor] Saved to {save_path}")
        return results, annotated

    def predict_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        frame_count, new_alerts = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.process_frame(frame)
            annotated = self.draw_results(frame, results)
            for r in results:
                if r.get("new_alert"):
                    new_alerts += 1
                    print(f"  Frame {frame_count}: SUSTAINED ALERT dog#{r['track_id']} "
                          f"risk={r['risk']:.2f}")
            if writer:
                writer.write(annotated)
            frame_count += 1

        cap.release()
        if writer:
            writer.release()
        print(f"[monitor] {frame_count} frames, {new_alerts} sustained alert(s)")
        return new_alerts

    def predict_webcam(self, cam_index=0):
        cap = cv2.VideoCapture(cam_index)
        print("[monitor] Press 'q' to quit webcam")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.process_frame(frame)
            annotated = self.draw_results(frame, results)
            cv2.imshow("Stray Dog Monitoring System", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Stray Dog Monitoring — CLI inference")
    p.add_argument("--source", required=True,
                   help="image/video path, or 'webcam'")
    p.add_argument("--detector", default=None, help="YOLO26 weights")
    p.add_argument("--pose", default="__default__",
                   help="pose weights, or 'none' to disable")
    p.add_argument("--output", default=None)
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--risk", type=float, default=None)
    args = p.parse_args()

    pose = None if str(args.pose).lower() in ("none", "off", "") else args.pose
    monitor = StrayDogMonitor(
        detector_path=args.detector,
        pose_model=pose,
        det_conf=args.conf,
        risk_threshold=args.risk,
    )
    if args.source == "webcam":
        monitor.predict_webcam()
    elif str(args.source).lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        monitor.predict_video(args.source, output_path=args.output)
    else:
        monitor.predict_image(args.source, save_path=args.output)
