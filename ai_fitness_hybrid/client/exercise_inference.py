import numpy as np
import tensorflow as tf
from collections import deque

class ExerciseClassifier:
    def __init__(self, model_path="../client/models/exercise_classifier.keras",
                 window_size=30,
                 confidence_threshold=0.85):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.confidence_threshold = confidence_threshold

        # MUST match training label order
        self.labels = [
            "bicep_curl",
            "lunge",
            "mountain_climber",
            "press",
            "pushup",
            "squat",
        ]

    def add_frame(self, feature_vector):
        """
        feature_vector: shape (8,)
        Returns:
          (exercise_name | None, confidence)
        """
        self.buffer.append(feature_vector)

        if len(self.buffer) < self.window_size:
            return None, 0.0

        x = np.expand_dims(np.array(self.buffer, dtype=np.float32), axis=0)
        probs = self.model.predict(x, verbose=0)[0]

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])

        if confidence < self.confidence_threshold:
            return None, confidence

        return self.labels[idx], confidence
