"""
Detector — YOLO26 person + dog boxes, with optional human pose.

Two models run in a single ``detect()`` call:
    - yolo26<n/s/m/l/x>.pt   -> person + dog boxes (COCO classes 0 and 16)
    - yolo11n-pose.pt        -> 17 human keypoints for posture features

The stray-dog risk engine never needs a custom-trained model: stock
COCO weights already know "person" and "dog", and the aggression signal is
derived geometrically (see ``src/risk``). Pose is optional — pass
``pose_model=None`` to skip it and save compute.
"""

from ultralytics import YOLO

# COCO detection class ids
COCO_PERSON = 0
COCO_DOG = 16

# COCO keypoint indices (17-joint standard)
KP_NOSE = 0
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16


def _iou(a, b):
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


def _resolve_class_ids(names):
    """
    Map a model's own class names to (person_ids, dog_ids).

    Stock COCO weights resolve to ({0}, {16}) as before. Fine-tuned custom
    models often re-number classes (e.g. dog=0), so ids are matched by NAME:
    any class containing "dog" (except hot dog) counts as dog, and
    person/people/human/pedestrian count as person.
    """
    person_ids, dog_ids = set(), set()
    for i, raw in (names or {}).items():
        n = str(raw).lower().replace("_", " ").strip()
        if n in ("person", "people", "human", "pedestrian"):
            person_ids.add(int(i))
        elif "dog" in n and n not in ("hot dog", "hotdog"):
            dog_ids.add(int(i))
    return person_ids, dog_ids


class Detector:
    """Ultralytics YOLO26 wrapper returning persons (w/ optional pose) + dogs."""

    def __init__(self, model_path="yolo26n.pt", pose_model="yolo11n-pose.pt",
                 conf=0.35, iou=0.45, device=None, imgsz=640):
        if model_path in (None, "", "None"):
            model_path = "yolo26n.pt"
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = int(imgsz) if imgsz else 640
        self.pose = YOLO(pose_model) if pose_model else None

        self.person_ids, self.dog_ids = _resolve_class_ids(
            getattr(self.model, "names", None))
        if not self.person_ids and not self.dog_ids:
            # names unavailable — assume the COCO convention
            self.person_ids, self.dog_ids = {COCO_PERSON}, {COCO_DOG}

    def detect(self, frame):
        """
        Run detection (+ pose when enabled) on a single BGR frame.

        Returns a dict:
            persons: list of {x1, y1, x2, y2, confidence, keypoints?}
            dogs:    list of {x1, y1, x2, y2, confidence}
        ``keypoints``, when present, is a list of (x, y, visibility) tuples in
        COCO joint order.
        """
        det_results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=sorted(self.person_ids | self.dog_ids),
            device=self.device,
            verbose=False,
        )

        persons, dogs = [], []
        for result in det_results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                det = {
                    "x1": int(box.xyxy[0][0]),
                    "y1": int(box.xyxy[0][1]),
                    "x2": int(box.xyxy[0][2]),
                    "y2": int(box.xyxy[0][3]),
                    "confidence": float(box.conf[0]),
                }
                if cls_id in self.person_ids:
                    persons.append(det)
                elif cls_id in self.dog_ids:
                    dogs.append(det)

        # Pose pass (persons only). Attach keypoints to the matching person box
        # by IoU so we keep one canonical person list.
        if self.pose is not None and persons:
            pose_results = self.pose.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            for result in pose_results:
                if result.boxes is None or result.keypoints is None:
                    continue
                kp_data = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                for i, pbox in enumerate(boxes_xyxy):
                    pose_box = {
                        "x1": int(pbox[0]), "y1": int(pbox[1]),
                        "x2": int(pbox[2]), "y2": int(pbox[3]),
                    }
                    best = max(persons, key=lambda p: _iou(p, pose_box),
                               default=None)
                    if best is not None and _iou(best, pose_box) > 0.3:
                        best["keypoints"] = [
                            (float(x), float(y), float(v))
                            for x, y, v in kp_data[i]
                        ]

        return {"persons": persons, "dogs": dogs}

    def get_dog_boxes(self, frame):
        return self.detect(frame)["dogs"]
