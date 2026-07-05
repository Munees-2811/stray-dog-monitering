"""
Behavior engine — names WHAT the dog is doing, not just how risky it is.

The risk score answers "how dangerous is this moment" as a single number.
This layer turns the same temporal-geometric evidence into a human-readable
behavior state per dog, so operators see *why* a dog is scored the way it is:

    idle / resting        no meaningful movement
    roaming               moving, no person interaction
    running               fast movement, not directed at a person
    close to person       stationary but inside personal space
    approaching person    closing the distance
    lunging / agitated    body-shape spikes (rearing, snapping posture)
    charging at person    fast + directed + near
    ATTACK RISK           sustained above the alert threshold while charging
    pack (N) — …          any active state escalated when 3+ dogs are present

Severity: 0 calm · 1 watch · 2 warning · 3 danger.

Purely additive: it consumes the features the risk engine already computes
plus track history, and never feeds back into risk scoring or alerting.
"""

import math

from src.tracking import box_center


def _speed(track, span=5):
    """Movement over the last `span` frames, in box-widths per frame
    (scale-invariant: a far dog and a near dog running read the same)."""
    hist = list(track.box_history)
    if len(hist) < 2:
        return 0.0
    span = min(span, len(hist))
    x0, y0 = box_center(hist[-span])
    x1, y1 = box_center(hist[-1])
    w = max(1.0, hist[-1]["x2"] - hist[-1]["x1"])
    return math.hypot(x1 - x0, y1 - y0) / (w * (span - 1) if span > 1 else w)


def classify_behavior(track, features, risk, risk_threshold, num_dogs):
    """
    Return ``(label, severity)`` for one dog track.

    ``features`` is the dict the risk engine produced for this frame
    (distance / velocity / posture / human_pose, each 0..1).
    """
    d = features.get("distance", 0.0)
    v = features.get("velocity", 0.0)
    p = features.get("posture", 0.0)
    hp = features.get("human_pose", 0.0)
    spd = _speed(track)

    if track.sustained_frames >= 1 and risk >= risk_threshold and v >= 0.4:
        label, sev = "ATTACK RISK", 3
    elif v >= 0.5 and d >= 0.4:
        label, sev = "charging at person", 3
    elif p >= 0.5 and d >= 0.35:
        label, sev = "lunging / agitated", 2
    elif v >= 0.25 and d >= 0.2:
        label, sev = "approaching person", 2
    elif d >= 0.7:
        label, sev = ("close to person (defensive human)", 2) if hp >= 0.3 \
            else ("close to person", 1)
    elif spd >= 0.09:
        label, sev = "running", 1
    elif spd >= 0.015:
        label, sev = "roaming", 0
    else:
        label, sev = "idle / resting", 0

    # A pack escalates anything that is already active behavior.
    if num_dogs >= 3 and sev >= 1:
        label = f"pack ({num_dogs}) — {label}"
        sev = min(3, sev + 1)

    return label, sev
