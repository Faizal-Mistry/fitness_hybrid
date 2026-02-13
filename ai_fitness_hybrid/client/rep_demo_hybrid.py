import cv2
import numpy as np
import mediapipe as mp

from pose_utils import PoseEstimator
from exercise_inference import ExerciseClassifier
from rep_counter import RepCounter

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Thresholds per exercise (easy to scale)
EXERCISE_RULES = {
    "bicep_curl": {"joint": "elbow", "flex": 90, "extend": 20},
    "pushup": {"joint": "elbow", "flex": 40, "extend": 20},
    "press": {"joint": "elbow", "flex": 80, "extend": 25},
    "squat": {"joint": "knee", "flex": 35, "extend": 15},
    "lunge": {"joint": "knee", "flex": 45, "extend": 20},
    "mountain_climber": {"joint": "knee", "flex": 50, "extend": 25},
}


def main():
    cap = cv2.VideoCapture(0)
    pose = PoseEstimator()
    classifier = ExerciseClassifier()
    rep_counter = RepCounter()

    current_exercise = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        features, landmarks = pose.process(frame)

        if landmarks:
            mp_draw.draw_landmarks(display, landmarks, mp_pose.POSE_CONNECTIONS)

        if features:
            feature_vec = np.array([
                features["knee_min_angle"],
                features["elbow_min_angle"],
                features["left_knee_angle"],
                features["right_knee_angle"],
                features["left_elbow_angle"],
                features["right_elbow_angle"],
                features["center_hip_y"],
                features["torso_dev"],
            ], dtype=np.float32)

            exercise, conf = classifier.add_frame(feature_vec)

            if exercise != current_exercise:
                rep_counter.reset()
                current_exercise = exercise

            if exercise:
                rules = EXERCISE_RULES[exercise]

                if rules["joint"] == "knee":
                    flex_amount = 180 - features["knee_min_angle"]
                else:
                    flex_amount = 180 - features["elbow_min_angle"]

                rep_done = rep_counter.update(
                    flex_amount,
                    rules["flex"],
                    rules["extend"]
                )

                cv2.putText(display, f"{exercise} ({conf:.2f})",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 2)

                if rep_done:
                    print(f"✅ Rep {rep_counter.rep_count}")

        cv2.putText(display, f"Reps: {rep_counter.rep_count}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 0), 3)

        cv2.imshow("Hybrid AI Fitness", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
