# client/auto_recorder.py

import cv2
import os
import numpy as np
from collections import deque

from pose_utils import PoseEstimator

# ---------------- CONFIG ----------------

WINDOW_SIZE = 30
DATASET_DIR = "dataset"

EXERCISES = {
    "1": "bicep_curl",
    "2": "squat",
    "3": "pushup",
    "4": "lunge",
    "5": "press",
    "6": "mountain_climber",
}

# ----------------------------------------


def choose_exercise():
    print("\nSelect exercise to record:")
    for k, v in EXERCISES.items():
        print(f"  {k}. {v}")
    choice = input("Enter number: ").strip()
    return EXERCISES.get(choice)


def get_next_window_index(folder):
    if not os.path.exists(folder):
        return 0
    files = [f for f in os.listdir(folder) if f.startswith("window_")]
    if not files:
        return 0
    indices = [int(f.split("_")[1].split(".")[0]) for f in files]
    return max(indices) + 1


def draw_overlay(frame, exercise, window_idx, buffer_len):
    cv2.rectangle(frame, (0, 0), (420, 110), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Exercise: {exercise}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Window: {window_idx}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        frame,
        f"Frames: {buffer_len}/{WINDOW_SIZE}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )


def main():
    exercise = choose_exercise()
    if exercise is None:
        print("Invalid selection.")
        return

    out_dir = os.path.join(DATASET_DIR, exercise)
    os.makedirs(out_dir, exist_ok=True)

    start_idx = get_next_window_index(out_dir)
    print(f"Recording '{exercise}' starting from window_{start_idx:05d}.npz")

    video_path = input(
        "Enter video path OR press Enter to use webcam: "
    ).strip()

    cap = cv2.VideoCapture(video_path if video_path else 0)
    if not cap.isOpened():
        print("Error opening video/camera.")
        return

    pose = PoseEstimator()
    buffer = deque(maxlen=WINDOW_SIZE)

    window_idx = start_idx

    print("Recording... Press Q to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        features, _ = pose.process(frame)

        if features:
            feature_vector = np.array([
                # -------- Angles (9) --------
                features["left_knee"],
                features["right_knee"],
                features["left_elbow"],
                features["right_elbow"],
                features["left_shoulder"],
                features["right_shoulder"],
                features["hip_angle"],
                features["knee_min"],
                features["elbow_min"],

                # -------- Velocities (4) --------
                features["left_knee_vel"],
                features["right_knee_vel"],
                features["left_elbow_vel"],
                features["right_elbow_vel"],

                # -------- Orientation (3) --------
                features["torso_dx"],
                features["torso_dy"],
                features["torso_vertical_angle"],

                # -------- Relative posture (2) --------
                features["wrist_shoulder_y"],
                features["shoulder_hip_y"],

                # -------- Ratios (2) --------
                features["hip_ankle_ratio"],
                features["shoulder_ankle_ratio"],

                # -------- Symmetry (2) --------
                features["knee_asym"],
                features["elbow_asym"],

                # -------- Ground relation (2) --------
                features["hip_ground_ratio"],
                features["knee_ground_ratio"],
            ], dtype=np.float32)

            buffer.append(feature_vector)

            if len(buffer) == WINDOW_SIZE:
                window = np.array(buffer)

                out_path = os.path.join(
                    out_dir, f"window_{window_idx:05d}.npz"
                )

                np.savez_compressed(
                    out_path,
                    sequence=window,
                    exercise=exercise,
                )

                print(f"Saved {out_path}")
                window_idx += 1
                buffer.clear()

        draw_overlay(frame, exercise, window_idx, len(buffer))
        cv2.imshow("Auto Recorder", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Recording finished.")


if __name__ == "__main__":
    main()
