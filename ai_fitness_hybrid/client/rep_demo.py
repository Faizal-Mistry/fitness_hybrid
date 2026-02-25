import cv2
import numpy as np

from pose_utils import PoseEstimator
from exercise_inference import ExercisePredictor
from rule_engine import get_rule


def main():
    cap = cv2.VideoCapture(0)

    pose = PoseEstimator()
    classifier = ExercisePredictor()

    current_exercise = None
    rule = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        features, landmarks = pose.process(frame)

        if features:
            feature_vector = np.array(list(features.values()), dtype=np.float32)

            exercise_name = classifier.predict(feature_vector)

            if exercise_name != current_exercise:
                current_exercise = exercise_name
                rule = get_rule(current_exercise)

            if rule:
                rep_done = rule.update(features)

                if rep_done:
                    print(f"{current_exercise} Rep {rule.rep_count}")

        # UI
        cv2.putText(display, f"Exercise: {current_exercise}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        if rule:
            cv2.putText(display, f"Reps: {rule.rep_count}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Hybrid AI Fitness", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()