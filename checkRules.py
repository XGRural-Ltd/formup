"""Deadlift-specific form analysis for the FormUp MVP."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class DeadliftState(str, Enum):
    READY = "ready"
    DESCENDING = "descending"
    ASCENDING = "ascending"
    LOCKOUT = "lockout"


class DeadliftVariant(str, Enum):
    CONVENTIONAL = "Conventional"
    SUMO = "Sumo"
    ROMANIAN = "Romanian (RDL)"


@dataclass(frozen=True)
class DeadliftAnalysis:
    posture_correct: bool
    state: DeadliftState
    reps: int
    feedback: str
    variant: DeadliftVariant


def is_between(value: float, low: float, high: float) -> bool:
    return low <= value <= high


def _average_angles(angles: Dict[str, float], left: str, right: str) -> float:
    return (angles.get(left, 180.0) + angles.get(right, 180.0)) / 2


def detect_deadlift_variant(features: Dict[str, float]) -> DeadliftVariant:
    """Classify the three MVP variations using pose geometry heuristics.

    The current GRU identifies deadlift, not its variation. Stance width is
    the strongest available signal for sumo; a shallow knee bend identifies
    Romanian deadlifts. Conventional is the fallback.
    """
    stance_width_ratio = features.get("stance_width_ratio", 1.0)
    knee_angle = _average_angles(
        features, "left_knee_angle", "right_knee_angle"
    )
    hip_angle = _average_angles(features, "left_hip_angle", "right_hip_angle")

    if stance_width_ratio >= 1.45:
        return DeadliftVariant.SUMO
    if knee_angle >= 145 and hip_angle < 155:
        return DeadliftVariant.ROMANIAN
    return DeadliftVariant.CONVENTIONAL


def analyze_deadlift(
    angles: Dict[str, float],
    state: DeadliftState = DeadliftState.READY,
    reps: int = 0,
    variant: DeadliftVariant = DeadliftVariant.CONVENTIONAL,
) -> DeadliftAnalysis:
    """Advance the deadlift state machine by one valid pose frame."""
    hip_angle = _average_angles(angles, "left_hip_angle", "right_hip_angle")
    knee_angle = _average_angles(angles, "left_knee_angle", "right_knee_angle")
    torso_angle = angles.get("torso_angle", 180.0)

    if torso_angle < 145:
        return DeadliftAnalysis(
            False, state, reps, "Keep your back more neutral", variant
        )

    if not is_between(hip_angle, 45, 180):
        return DeadliftAnalysis(
            False, state, reps, "Adjust your hip position", variant
        )

    posture_correct = True
    next_state = state
    next_reps = reps

    if variant == DeadliftVariant.ROMANIAN:
        bottom_hip_threshold = 125
        bottom_knee_threshold = 170
    elif variant == DeadliftVariant.SUMO:
        bottom_hip_threshold = 120
        bottom_knee_threshold = 165
    else:
        bottom_hip_threshold = 115
        bottom_knee_threshold = 160

    if state == DeadliftState.READY:
        if hip_angle < bottom_hip_threshold and knee_angle < bottom_knee_threshold:
            next_state = DeadliftState.ASCENDING
        elif hip_angle < 135:
            next_state = DeadliftState.DESCENDING
    elif state == DeadliftState.DESCENDING:
        if hip_angle < 115:
            next_state = DeadliftState.ASCENDING
    elif state == DeadliftState.ASCENDING:
        if hip_angle > 160 and torso_angle > 160:
            next_state = DeadliftState.LOCKOUT
    elif state == DeadliftState.LOCKOUT:
        next_reps += 1
        next_state = DeadliftState.READY

    return DeadliftAnalysis(
        posture_correct, next_state, next_reps, "Form OK", variant
    )


EXERCISE_RULES = {"deadlift": analyze_deadlift}
