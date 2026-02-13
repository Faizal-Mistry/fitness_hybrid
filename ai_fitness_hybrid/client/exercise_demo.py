import cv2
import numpy as np
import tensorflow as tf
from collections import deque

from pose_utils import PoseEstimator


class ExercisePredictor:
    def __init__(self, model_path="../models/exercise_classifier.keras",
                 labels_path="../models/exercise_labels.npy",
                 window_size=30):

        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.labels = np.load(labels_path, allow_pickle=True).item()
        self.id_to_label = {v: k for k, v in self.labels.items()}

        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def add_frame(self, feature_vector):
        self.buffer.append(feature_vector)

        if len(self.buffer) < self.window_size:
            return None, 0.0

        x = np.array(self.buffer, dtype=np.float32)
        x = np.expand_dims(x, axis=0)  # (1, 30, 24)

        probs = self.model.predict(x, verbose=0)[0]
        class_id = int(np.argmax(probs))
        confidence = float(np.max(probs))

        label = self.id_to_label[class_id]
        return label, confidence


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera error")
        return

    pose = PoseEstimator()
    predictor = ExercisePredictor()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        features, landmarks = pose.process(frame)

        if features:
            feature_vector = np.array([
                # Angles (9)
                features["left_knee"],
                features["right_knee"],
                features["left_elbow"],
                features["right_elbow"],
                features["left_shoulder"],
                features["right_shoulder"],
                features["hip_angle"],
                features["knee_min"],
                features["elbow_min"],

                # Velocities (4)
                features["left_knee_vel"],
                features["right_knee_vel"],
                features["left_elbow_vel"],
                features["right_elbow_vel"],

                # Orientation (3)
                features["torso_dx"],
                features["torso_dy"],
                features["torso_vertical_angle"],

                # Relative posture (2)
                features["wrist_shoulder_y"],
                features["shoulder_hip_y"],

                # Ratios (2)
                features["hip_ankle_ratio"],
                features["shoulder_ankle_ratio"],

                # Symmetry (2)
                features["knee_asym"],
                features["elbow_asym"],

                # Ground (2)
                features["hip_ground_ratio"],
                features["knee_ground_ratio"],
            ], dtype=np.float32)

            exercise, conf = predictor.add_frame(feature_vector)

            if exercise:
                text = f"{exercise} ({conf:.2f})"
                cv2.putText(display, text, (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 255, 0), 3)

        cv2.imshow("Exercise Detection", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
