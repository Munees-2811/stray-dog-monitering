"""
Human posture features from COCO-Pose keypoints.

A nearby person reacting defensively (arms thrown up, crouching/bracing) is a
weak-but-useful corroborating signal that a dog is behaving threateningly.
These features feed the ``human_pose`` term of the risk score.
"""

from src.detection import (
    KP_L_SHOULDER, KP_R_SHOULDER,
    KP_L_WRIST, KP_R_WRIST,
    KP_L_HIP, KP_R_HIP,
    KP_L_KNEE, KP_R_KNEE,
)


def _kp_xy(keypoints, idx, visibility_threshold=0.3):
    """Return (x, y) if the joint is visible, else None."""
    x, y, v = keypoints[idx]
    if v < visibility_threshold:
        return None
    return x, y


def compute_pose_features(person):
    """
    Return behavioural features derived from a person's keypoints:
        arms_raised: wrists above shoulders (defensive / warding off)  -> 0/.5/1
        crouched:    hips close to knees vertically (bracing)          -> 0..1
    """
    features = {"arms_raised": 0.0, "crouched": 0.0}
    kps = person.get("keypoints")
    if kps is None:
        return features

    l_sh = _kp_xy(kps, KP_L_SHOULDER)
    r_sh = _kp_xy(kps, KP_R_SHOULDER)
    l_wr = _kp_xy(kps, KP_L_WRIST)
    r_wr = _kp_xy(kps, KP_R_WRIST)
    l_hip = _kp_xy(kps, KP_L_HIP)
    l_knee = _kp_xy(kps, KP_L_KNEE)

    # Arms raised: wrist y < shoulder y (image y grows downward)
    raised = 0.0
    if l_wr and l_sh and l_wr[1] < l_sh[1]:
        raised += 1
    if r_wr and r_sh and r_wr[1] < r_sh[1]:
        raised += 1
    features["arms_raised"] = raised / 2.0  # 0.0, 0.5, or 1.0

    # Crouched: small hip->knee vertical gap relative to shoulder->hip torso
    if l_hip and l_knee and l_sh:
        torso = max(1, abs(l_sh[1] - l_hip[1]))
        thigh = abs(l_hip[1] - l_knee[1])
        ratio = thigh / torso
        if ratio < 0.6:   # standing: thigh >= torso; crouched: thigh < 0.6*torso
            features["crouched"] = max(features["crouched"], 1.0 - ratio / 0.6)

    return features
