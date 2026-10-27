import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from pose_features import POSE_LANDMARKS, build_features, landmarks_from_row


def process_row_slim(row_dict):
    features = build_features(landmarks_from_row(row_dict))
    features["video_id"] = row_dict["video_id"]
    features["label"] = row_dict["label"]
    return features


def process_csv_revised(input_csv, output_csv):
    print(f"Processing {input_csv} -> {output_csv}")
    df = pd.read_csv(input_csv)
    left_hip_visibility = f"v{POSE_LANDMARKS['left_hip']}"
    right_hip_visibility = f"v{POSE_LANDMARKS['right_hip']}"
    df = df[
        (df[left_hip_visibility] > 0.5)
        & (df[right_hip_visibility] > 0.5)
    ].copy()

    rows = df.to_dict(orient="records")
    feature_rows = []
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(process_row_slim, row) for row in rows]
        for future in tqdm(as_completed(futures), total=len(futures)):
            feature_rows.append(future.result())

    final_df = pd.DataFrame(feature_rows)
    output_directory = os.path.dirname(output_csv)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    final_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}, shape={final_df.shape}")


if __name__ == "__main__":
    process_csv_revised(
        "dataset/keypoints/dataset_train.csv",
        "dataset/angles/angles_train.csv",
    )
    process_csv_revised(
        "dataset/keypoints/dataset_val.csv",
        "dataset/angles/angles_val.csv",
    )
    process_csv_revised(
        "dataset/keypoints/dataset_test.csv",
        "dataset/angles/angles_test.csv",
    )
