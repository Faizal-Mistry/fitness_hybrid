import numpy as np
import tensorflow as tf
from collections import deque
import json
import os


class ExercisePredictor:
    def __init__(self,
                 model_path="../models/exercise_classifier.keras",
                 window_size=30,
                 confidence_threshold=0.85):

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

        self.confidence_threshold = confidence_threshold

        # Load label mapping
        labels_path = "../models/exercise_labels.json"

        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                self.label_map = json.load(f)
        else:
            raise FileNotFoundError(
                "exercise_labels.json not found in models folder"
            )

        self.idx_to_name = {v: k for k, v in self.label_map.items()}

        # Stability filter
        self.last_prediction = None
        self.stable_count = 0
        self.required_stable_frames = 5

    def predict(self, feature_vector):
        self.buffer.append(feature_vector)

        if len(self.buffer) < self.window_size:
            return None

        x = np.array(self.buffer, dtype=np.float32)
        x = np.expand_dims(x, axis=0)

        probs = self.model.predict(x, verbose=0)[0]
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])

        # 🚨 Reject low confidence predictions
        if confidence < self.confidence_threshold:
            self.last_prediction = None
            self.stable_count = 0
            return None

        predicted_name = self.idx_to_name[class_id]

        # 🚨 Stability check (avoid flickering)
        if predicted_name == self.last_prediction:
            self.stable_count += 1
        else:
            self.stable_count = 1
            self.last_prediction = predicted_name

        if self.stable_count >= self.required_stable_frames:
            return predicted_name

        return None
