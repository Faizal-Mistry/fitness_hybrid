# client/rep_demo.py

import cv2
import numpy as np
from collections import defaultdict

from pose_utils import PoseEstimator
from exercise_inference import ExercisePredictor
from rule_engine import get_rule


# ===============================
# Motion Detection Helper
# ===============================

def compute_motion(f):
    """
    Motion score based on joint angular velocities.
    """
    return (
        abs(f["left_knee_vel"]) +
        abs(f["right_knee_vel"]) +
        abs(f["left_elbow_vel"]) +
        abs(f["right_elbow_vel"])
    )


# ===============================
# Main
# ===============================

def main():
    cap = cv2.VideoCapture(0)

    pose = PoseEstimator()
    classifier = ExercisePredictor()

    # Store rule object per exercise (so memory persists)
    exercise_rules = {}

    current_exercise = None

    # Motion control
    STILL_THRESHOLD = 8        # Adjust if needed (5–12 range)
    MAX_STILL_FRAMES = 15      # Frames before switching to None
    still_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        features, landmarks = pose.process(frame)

        if features:
            feature_vector = np.array(list(features.values()), dtype=np.float32)

            # -------------------------------
            # Motion detection
            # -------------------------------
            motion = compute_motion(features)

            if motion < STILL_THRESHOLD:
                still_counter += 1
            else:
                still_counter = 0

            # -------------------------------
            # If still → show None
            # -------------------------------
            if still_counter > MAX_STILL_FRAMES:
                current_exercise = None

            else:
                exercise_name = classifier.predict(feature_vector)

                if exercise_name:

                    # Create rule once per exercise
                    if exercise_name not in exercise_rules:
                        exercise_rules[exercise_name] = get_rule(exercise_name)

                    current_exercise = exercise_name
                    rule = exercise_rules[current_exercise]

                    rep_done = rule.update(features)

                    if rep_done:
                        print(f"{current_exercise} Rep {rule.rep_count}")

        # ===============================
        # UI
        # ===============================

        if current_exercise:
            cv2.putText(display,
                        f"Exercise: {current_exercise}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2)

            rule = exercise_rules[current_exercise]

            cv2.putText(display,
                        f"Reps: {rule.rep_count}",
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3)
        else:
            cv2.putText(display,
                        "Exercise: None",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2)

        cv2.imshow("Hybrid AI Fitness", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
