# client/build_exercise_dataset.py

import os
import numpy as np
from glob import glob

DATASET_DIR = "dataset"
WINDOW_SIZE = 30  # must match recorder
FEATURE_DIM = 24  # new feature count


def build_dataset():
    exercise_names = sorted([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])

    label_map = {name: i for i, name in enumerate(exercise_names)}
    print("Labels:", label_map)

    X = []
    y = []

    total_files = 0

    for exercise in exercise_names:
        exercise_dir = os.path.join(DATASET_DIR, exercise)
        files = glob(os.path.join(exercise_dir, "window_*.npz"))

        for f in files:
            data = np.load(f)

            if "sequence" not in data:
                continue

            seq = data["sequence"]

            # ensure correct shape
            if seq.shape != (WINDOW_SIZE, FEATURE_DIM):
                continue

            X.append(seq)
            y.append(label_map[exercise])
            total_files += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"✅ Exercise dataset built")
    print("X:", X.shape)
    print("y:", y.shape)

    np.save("exercise_X.npy", X)
    np.save("exercise_y.npy", y)
    np.save("exercise_labels.npy", label_map)

    print("Saved:")
    print("- exercise_X.npy")
    print("- exercise_y.npy")
    print("- exercise_labels.npy")


if __name__ == "__main__":
    build_dataset()
