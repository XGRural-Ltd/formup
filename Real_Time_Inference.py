"""Real-time FormUp MVP for deadlift recognition and form feedback."""

import argparse
import json
import time
from pathlib import Path
from typing import Any, List

import cv2
import joblib
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

from checkRules import (
    DeadliftAnalysis,
    DeadliftState,
    DeadliftVariant,
    analyze_deadlift,
    detect_deadlift_variant,
)
from pose_features import (
    FEATURE_NAMES,
    POSE_LANDMARKS,
    build_features,
    build_rule_angles,
)


MODEL_DIRECTORY = Path(__file__).parent / "files" / "GRU_model_7"
SEQUENCE_LENGTH = 25
PREDICTION_INTERVAL_SECONDS = 0.5
CONFIDENCE_THRESHOLD = 0.7
NO_EXERCISE_CLASSES = {"Sitting", "Walking", "Standing Still"}


def load_artifacts():
    try:
        model = load_model(MODEL_DIRECTORY / "GRU_model_7.h5")
        scaler = joblib.load(MODEL_DIRECTORY / "real_time_scaler.pkl")
        with (MODEL_DIRECTORY / "encoded_labels.json").open(
            "r", encoding="utf-8"
        ) as labels_file:
            labels = {int(key): value for key, value in json.load(labels_file).items()}
    except (FileNotFoundError, OSError, ValueError) as error:
        raise RuntimeError(
            f"Could not load FormUp model artifacts from {MODEL_DIRECTORY}"
        ) from error
    return model, scaler, labels


def stabilize_landmarks(previous: Any, current: Any):
    if previous is None:
        return [
            {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility,
            }
            for landmark in current
        ]

    for index, landmark in enumerate(current):
        previous[index]["x"] = 0.7 * previous[index]["x"] + 0.3 * landmark.x
        previous[index]["y"] = 0.7 * previous[index]["y"] + 0.3 * landmark.y
        previous[index]["z"] = 0.7 * previous[index]["z"] + 0.3 * landmark.z
        previous[index]["visibility"] = (
            0.7 * previous[index]["visibility"] + 0.3 * landmark.visibility
        )
    return previous


def features_from_landmarks(landmarks: List[dict]):
    points = {
        name: [landmarks[index]["x"], landmarks[index]["y"]]
        for name, index in POSE_LANDMARKS.items()
    }
    features = build_features(points)
    return features, build_rule_angles(points, features)


def draw_status(
    image,
    prediction_label: str,
    confidence: float,
    analysis: DeadliftAnalysis,
):
    height, width, _ = image.shape
    cv2.rectangle(image, (0, 0), (width, 160), (30, 30, 30), -1)
    color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 0, 255)
    cv2.putText(
        image,
        f"Activity: {prediction_label}",
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )
    cv2.putText(
        image,
        f"Confidence: {confidence:.2f}",
        (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )
    if prediction_label == "Deadlift":
        cv2.putText(
            image,
            f"Variation: {analysis.variant.value}",
            (15, 102),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            image,
            f"Reps: {analysis.reps} | {analysis.feedback}",
            (15, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if analysis.posture_correct else (0, 0, 255),
            2,
        )


def run(source: int | str = 0):
    model, scaler, labels = load_artifacts()
    pose_module = mp.solutions.pose
    pose = pose_module.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        model_complexity=1,
    )
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        pose.close()
        raise RuntimeError(f"Could not open video source: {source}")

    frame_sequence = []
    stable_landmarks = None
    last_prediction_time = 0.0
    prediction_label = "No Person Detected"
    confidence = 0.0
    analysis = DeadliftAnalysis(
        True,
        DeadliftState.READY,
        0,
        "Waiting",
        DeadliftVariant.CONVENTIONAL,
    )

    try:
        while capture.isOpened():
            received, frame = capture.read()
            if not received:
                break
            frame = cv2.flip(frame, 1) if isinstance(source, int) else frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if not results.pose_landmarks:
                prediction_label = "No Person Detected"
                confidence = 0.0
                frame_sequence.clear()
                stable_landmarks = None
                draw_status(frame, prediction_label, confidence, analysis)
                cv2.imshow("FormUp - Deadlift", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            stable_landmarks = stabilize_landmarks(
                stable_landmarks, results.pose_landmarks.landmark
            )
            try:
                current_features, rule_angles = features_from_landmarks(
                    stable_landmarks
                )
            except (KeyError, ValueError, ZeroDivisionError) as error:
                print(f"Skipping invalid pose frame: {error}")
                frame_sequence.clear()
                continue

            frame_sequence.append(
                np.asarray(
                    [current_features[name] for name in FEATURE_NAMES],
                    dtype=np.float32,
                )
            )
            del frame_sequence[:-SEQUENCE_LENGTH]

            now = time.monotonic()
            if (
                len(frame_sequence) == SEQUENCE_LENGTH
                and now - last_prediction_time >= PREDICTION_INTERVAL_SECONDS
            ):
                sequence = np.asarray(frame_sequence)
                scaled = scaler.transform(sequence).reshape(
                    1, SEQUENCE_LENGTH, len(FEATURE_NAMES)
                )
                probabilities = model.predict(scaled, verbose=0)[0]
                class_index = int(np.argmax(probabilities))
                confidence = float(probabilities[class_index])
                predicted_label = labels[class_index]
                last_prediction_time = now

                if confidence < CONFIDENCE_THRESHOLD:
                    prediction_label = "Uncertain"
                elif predicted_label == "deadlift":
                    prediction_label = "Deadlift"
                    variant = detect_deadlift_variant(
                        {**current_features, **rule_angles}
                    )
                    analysis = analyze_deadlift(
                        rule_angles,
                        analysis.state,
                        analysis.reps,
                        variant,
                    )
                elif predicted_label in NO_EXERCISE_CLASSES:
                    prediction_label = "No Exercise"
                else:
                    prediction_label = "Not Deadlift"

            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, pose_module.POSE_CONNECTIONS
            )
            draw_status(frame, prediction_label, confidence, analysis)
            cv2.imshow("FormUp - Deadlift", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        pose.close()
        cv2.destroyAllWindows()


def parse_source(value: str):
    return int(value) if value.isdigit() else value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FormUp deadlift MVP.")
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index or path to a video file.",
    )
    args = parser.parse_args()
    run(parse_source(args.source))
