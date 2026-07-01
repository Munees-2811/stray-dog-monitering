"""
Geometric + temporal risk engine.

The core insight of the system: you do not need a labelled "aggressive dog"
dataset to flag a dangerous situation. What makes a stray dog dangerous to a
person is *geometry over time* — how close it is, whether it is closing in,
whether it is lunging, and whether a nearby human is reacting defensively.

``RiskEngine.score`` turns one dog track + the current people into an
interpretable risk value in [0, 1]. Weights are configurable so the behaviour
can be tuned without touching code.
"""

import math

from src.tracking import box_center, box_aspect_ratio
from src.risk.pose_features import compute_pose_features


class RiskEngine:
    """Interpretable per-dog risk scoring from geometry + human pose."""

    def __init__(self, weights=None, pack_bonus=0.10):
        self.weights = weights or {
            "distance": 0.55,
            "velocity": 0.20,
            "posture": 0.10,
            "human_pose": 0.05,
        }
        self.pack_bonus = pack_bonus

    def score(self, track, persons, frame_shape, num_dogs):
        """
        Return ``(risk, features)`` for one dog track.

        Contributions (default weights):
            0.55 distance   - closer dog-to-person (dominant signal)
            0.20 velocity   - dog moving toward the nearest person
            0.10 posture    - dog box aspect-ratio change (lunging / rearing)
            0.05 human_pose - nearest person defensive (arms up / crouched)
            +pack_bonus     - when more than one dog is present
        """
        features = {"distance": 0, "velocity": 0, "posture": 0, "human_pose": 0}
        if not persons:
            return 0.0, features

        h, w = frame_shape[:2]
        diag = math.hypot(w, h)
        dog_cx, dog_cy = box_center(track.box)

        # nearest person to this dog
        nearest = min(
            persons,
            key=lambda p: math.hypot(
                box_center(p)[0] - dog_cx, box_center(p)[1] - dog_cy
            ),
        )
        p_cx, p_cy = box_center(nearest)

        # 1. Distance — normalised so anything within 50% of the diagonal scores.
        distance = math.hypot(dog_cx - p_cx, dog_cy - p_cy) / diag
        distance_risk = max(0.0, 1.0 - distance / 0.50)

        # 2. Velocity toward the person (over the last 3 frames)
        velocity_risk = 0.0
        if len(track.box_history) >= 3:
            old_cx, old_cy = box_center(track.box_history[-3])
            dx, dy = dog_cx - old_cx, dog_cy - old_cy
            pdx, pdy = p_cx - old_cx, p_cy - old_cy
            p_len = math.hypot(pdx, pdy) + 1e-6
            pdx, pdy = pdx / p_len, pdy / p_len
            velocity_toward = (dx * pdx + dy * pdy) / diag
            velocity_risk = min(1.0, max(0.0, velocity_toward / 0.015))

        # 3. Posture — aspect-ratio variance over the last 5 frames (lunging)
        posture_risk = 0.0
        if len(track.box_history) >= 3:
            recent = [box_aspect_ratio(b) for b in list(track.box_history)[-5:]]
            posture_risk = min(1.0, (max(recent) - min(recent)) / 0.4)

        # 4. Human pose — defensive/bracing from the nearest person
        pose = compute_pose_features(nearest)
        pose_risk = max(pose["arms_raised"], pose["crouched"])

        # 5. Pack bonus
        pack_bonus = self.pack_bonus if num_dogs > 1 else 0.0

        w_ = self.weights
        risk = (
            w_["distance"] * distance_risk
            + w_["velocity"] * velocity_risk
            + w_["posture"] * posture_risk
            + w_["human_pose"] * pose_risk
            + pack_bonus
        )

        features = {
            "distance": round(distance_risk, 2),
            "velocity": round(velocity_risk, 2),
            "posture": round(posture_risk, 2),
            "human_pose": round(pose_risk, 2),
        }
        return min(1.0, risk), features
