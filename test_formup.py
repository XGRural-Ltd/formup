import math

from checkRules import (
    DeadliftState,
    DeadliftVariant,
    analyze_deadlift,
    detect_deadlift_variant,
)
from pose_features import FEATURE_NAMES, build_features


def test_feature_builder_preserves_model_width():
    points = {
        "left_shoulder": (0.4, 0.2),
        "right_shoulder": (0.6, 0.2),
        "left_elbow": (0.35, 0.35),
        "right_elbow": (0.65, 0.35),
        "left_wrist": (0.3, 0.5),
        "right_wrist": (0.7, 0.5),
        "left_hip": (0.43, 0.5),
        "right_hip": (0.57, 0.5),
        "left_knee": (0.43, 0.7),
        "right_knee": (0.57, 0.7),
        "left_ankle": (0.4, 0.9),
        "right_ankle": (0.6, 0.9),
        "nose": (0.5, 0.1),
    }

    features = build_features(points)

    assert len(FEATURE_NAMES) == 21
    assert list(features) == FEATURE_NAMES
    assert all(math.isfinite(value) for value in features.values())


def test_deadlift_completes_one_rep():
    angles = {
        "left_hip_angle": 130.0,
        "right_hip_angle": 130.0,
        "left_knee_angle": 140.0,
        "right_knee_angle": 140.0,
        "torso_angle": 160.0,
    }

    descending = analyze_deadlift(angles)
    ascending = analyze_deadlift(
        {**angles, "left_hip_angle": 90.0, "right_hip_angle": 90.0},
        descending.state,
        descending.reps,
    )
    lockout = analyze_deadlift(
        {
            "left_hip_angle": 170.0,
            "right_hip_angle": 170.0,
            "left_knee_angle": 175.0,
            "right_knee_angle": 175.0,
            "torso_angle": 175.0,
        },
        ascending.state,
        ascending.reps,
    )
    completed = analyze_deadlift(
        {
            "left_hip_angle": 175.0,
            "right_hip_angle": 175.0,
            "left_knee_angle": 175.0,
            "right_knee_angle": 175.0,
            "torso_angle": 175.0,
        },
        lockout.state,
        lockout.reps,
    )

    assert descending.state == DeadliftState.DESCENDING
    assert ascending.state == DeadliftState.ASCENDING
    assert lockout.state == DeadliftState.LOCKOUT
    assert completed.reps == 1
    assert completed.state == DeadliftState.READY


def test_deadlift_reports_neutral_back_feedback():
    result = analyze_deadlift(
        {
            "left_hip_angle": 100.0,
            "right_hip_angle": 100.0,
            "left_knee_angle": 140.0,
            "right_knee_angle": 140.0,
            "torso_angle": 120.0,
        }
    )

    assert not result.posture_correct
    assert result.feedback == "Keep your back more neutral"


def test_deadlift_variations_are_detected_from_pose_geometry():
    assert (
        detect_deadlift_variant(
            {
                "stance_width_ratio": 1.6,
                "left_knee_angle": 140.0,
                "right_knee_angle": 140.0,
                "left_hip_angle": 120.0,
                "right_hip_angle": 120.0,
            }
        )
        == DeadliftVariant.SUMO
    )
    assert (
        detect_deadlift_variant(
            {
                "stance_width_ratio": 1.0,
                "left_knee_angle": 155.0,
                "right_knee_angle": 155.0,
                "left_hip_angle": 130.0,
                "right_hip_angle": 130.0,
            }
        )
        == DeadliftVariant.ROMANIAN
    )
    assert (
        detect_deadlift_variant(
            {
                "stance_width_ratio": 1.0,
                "left_knee_angle": 135.0,
                "right_knee_angle": 135.0,
                "left_hip_angle": 100.0,
                "right_hip_angle": 100.0,
            }
        )
        == DeadliftVariant.CONVENTIONAL
    )
