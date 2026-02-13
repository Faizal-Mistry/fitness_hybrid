import cv2
import mediapipe as mp
from pose_utils import PoseEstimator

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
pose = PoseEstimator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    features, landmarks = pose.process(frame)

    if landmarks:
        mp_drawing.draw_landmarks(
            frame,
            landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    if features:
        y = 30
        for k, v in features.items():
            cv2.putText(frame, f"{k}: {v:.1f}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            y += 18

    cv2.imshow("Pose + Features", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
