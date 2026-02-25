# client/exercise_inference.py

import numpy as np
import tensorflow as tf
from collections import deque
import json
import os


class ExercisePredictor:
    def __init__(self,
                 model_path="../models/exercise_classifier.keras",
                 window_size=30):

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

        # Load label mapping
        labels_path = "../models/exercise_labels.json"

        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                self.label_map = json.load(f)
        else:
            raise FileNotFoundError(
                "exercise_labels.json not found in models folder"
            )

        # Reverse mapping index → name
        self.idx_to_name = {v: k for k, v in self.label_map.items()}

    def predict(self, feature_vector):
        """
        feature_vector: shape (24,)
        """

        self.buffer.append(feature_vector)

        # Wait until window full
        if len(self.buffer) < self.window_size:
            return None

        x = np.array(self.buffer, dtype=np.float32)
        x = np.expand_dims(x, axis=0)  # (1, 30, 24)

        probs = self.model.predict(x, verbose=0)[0]
        class_id = int(np.argmax(probs))

        return self.idx_to_name[class_id]
