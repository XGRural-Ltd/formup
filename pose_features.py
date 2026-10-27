"""Shared pose feature definitions used by training and runtime inference."""

import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


POSE_LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "nose": 0,
}

ANGLE_DEFINITIONS = {
    "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
    "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
    "shoulder_angle": ("left_elbow", "left_shoulder", "right_shoulder"),
    "hip_angle": ("left_knee", "left_hip", "right_hip"),
    "torso_angle": ("left_shoulder", "left_hip", "left_knee"),
}

DISTANCE_DEFINITIONS = {
    "right_hand_to_right_hip_dist": ("right_wrist", "right_hip"),
    "left_hand_to_left_hip_dist": ("left_wrist", "left_hip"),
    "feet_distance": ("left_ankle", "right_ankle"),
}

EXTRA_FEATURES = [
    "elbow_symmetry",
    "knee_symmetry",
    "left_hand_to_shoulder_vertical",
    "right_hand_to_shoulder_vertical",
    "left_elbow_to_hip_vertical",
    "right_elbow_to_hip_vertical",
    "left_ankle_to_knee_horizontal",
    "right_ankle_to_knee_horizontal",
    "torso_lean",
    "stance_width_ratio",
]

FEATURE_NAMES = (
    list(ANGLE_DEFINITIONS)
    + list(DISTANCE_DEFINITIONS)
    + EXTRA_FEATURES
)


def calculate_angle(
    point_a: Sequence[float],
    point_b: Sequence[float],
    point_c: Sequence[float],
) -> float:
    """Return the angle at point_b in degrees."""
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    c = np.asarray(point_c, dtype=float)
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cosine_angle = np.dot(ba, bc) / denominator
    return float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))


def calculate_distance(
    point_a: Sequence[float], point_b: Sequence[float]
) -> float:
    return math.hypot(point_b[0] - point_a[0], point_b[1] - point_a[1])


def calculate_vertical_angle(
    point_a: Sequence[float], point_b: Sequence[float]
) -> float:
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    if dy == 0:
        return 90.0
    return math.degrees(math.atan(abs(dx) / abs(dy)))


def calculate_symmetry(left_value: float, right_value: float) -> float:
    if left_value == 0 and right_value == 0:
        return 1.0
    return 1.0 - abs(left_value - right_value) / max(left_value, right_value, 1e-6)


def landmarks_from_row(row: Mapping[str, float]) -> Dict[str, Tuple[float, float]]:
    return {
        name: (float(row[f"x{index}"]), float(row[f"y{index}"]))
        for name, index in POSE_LANDMARKS.items()
    }


def build_features(
    landmarks: Mapping[str, Sequence[float]],
) -> Dict[str, float]:
    """Build the exact 21-feature vector used by the GRU."""
    features: Dict[str, float] = {}

    for feature_name, (a, b, c) in ANGLE_DEFINITIONS.items():
        features[feature_name] = calculate_angle(
            landmarks[a], landmarks[b], landmarks[c]
        )

    for feature_name, (point_a, point_b) in DISTANCE_DEFINITIONS.items():
        features[feature_name] = calculate_distance(
            landmarks[point_a], landmarks[point_b]
        )

    features["elbow_symmetry"] = calculate_symmetry(
        features["left_elbow_angle"], features["right_elbow_angle"]
    )
    features["knee_symmetry"] = calculate_symmetry(
        features["left_knee_angle"], features["right_knee_angle"]
    )
    features["left_hand_to_shoulder_vertical"] = (
        landmarks["left_shoulder"][1] - landmarks["left_wrist"][1]
    )
    features["right_hand_to_shoulder_vertical"] = (
        landmarks["right_shoulder"][1] - landmarks["right_wrist"][1]
    )
    features["left_elbow_to_hip_vertical"] = (
        landmarks["left_hip"][1] - landmarks["left_elbow"][1]
    )
    features["right_elbow_to_hip_vertical"] = (
        landmarks["right_hip"][1] - landmarks["right_elbow"][1]
    )
    features["left_ankle_to_knee_horizontal"] = abs(
        landmarks["left_ankle"][0] - landmarks["left_knee"][0]
    )
    features["right_ankle_to_knee_horizontal"] = abs(
        landmarks["right_ankle"][0] - landmarks["right_knee"][0]
    )
    features["torso_lean"] = calculate_vertical_angle(
        landmarks["left_shoulder"], landmarks["left_hip"]
    )
    shoulder_width = calculate_distance(
        landmarks["left_shoulder"], landmarks["right_shoulder"]
    )
    features["stance_width_ratio"] = (
        features["feet_distance"] / shoulder_width if shoulder_width > 0 else 0.0
    )
    return features


def build_rule_angles(
    landmarks: Mapping[str, Sequence[float]],
    features: Mapping[str, float],
) -> Dict[str, float]:
    """Build the angles consumed by the deadlift form analyzer."""
    return {
        "left_hip_angle": calculate_angle(
            landmarks["left_shoulder"],
            landmarks["left_hip"],
            landmarks["left_knee"],
        ),
        "right_hip_angle": features["right_hip_angle"],
        "left_knee_angle": features["left_knee_angle"],
        "right_knee_angle": features["right_knee_angle"],
        "torso_angle": features["torso_angle"],
    }
