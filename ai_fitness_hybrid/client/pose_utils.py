# client/pose_utils.py

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

mp_pose = mp.solutions.pose


def angle_between(a, b, c):
    """
    Angle at point b formed by a-b-c
    """
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class PoseEstimator:
    def __init__(self, vel_window=3):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.prev_angles = deque(maxlen=vel_window)

    def process(self, frame_bgr):
        h, w, _ = frame_bgr.shape
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None, None

        lm = results.pose_landmarks.landmark

        def pt(idx):
            p = lm[idx]
            return (p.x * w, p.y * h)

        # ---- Joints ----
        l_hip, r_hip = pt(23), pt(24)
        l_knee, r_knee = pt(25), pt(26)
        l_ankle, r_ankle = pt(27), pt(28)

        l_sh, r_sh = pt(11), pt(12)
        l_el, r_el = pt(13), pt(14)
        l_wr, r_wr = pt(15), pt(16)

        # ---- Angles ----
        left_knee = angle_between(l_hip, l_knee, l_ankle)
        right_knee = angle_between(r_hip, r_knee, r_ankle)
        left_elbow = angle_between(l_sh, l_el, l_wr)
        right_elbow = angle_between(r_sh, r_el, r_wr)
        left_shoulder = angle_between(l_el, l_sh, l_hip)
        right_shoulder = angle_between(r_el, r_sh, r_hip)
        hip_angle = angle_between(l_sh, l_hip, l_knee)

        knee_min = min(left_knee, right_knee)
        elbow_min = min(left_elbow, right_elbow)

        # ---- Velocity (angular) ----
        current_angles = np.array([
            left_knee, right_knee,
            left_elbow, right_elbow
        ])

        if len(self.prev_angles) > 0:
            vel = current_angles - self.prev_angles[-1]
        else:
            vel = np.zeros(4)

        self.prev_angles.append(current_angles)

        # ---- Torso orientation ----
        mid_sh = np.mean([l_sh, r_sh], axis=0)
        mid_hip = np.mean([l_hip, r_hip], axis=0)
        torso_vec = mid_sh - mid_hip
        torso_len = np.linalg.norm(torso_vec) + 1e-6
        torso_dx, torso_dy = torso_vec / torso_len

        torso_vertical_angle = angle_between(
            (mid_hip[0], mid_hip[1] - 100),
            mid_hip,
            mid_sh
        )

        # ---- Relative posture ----
        wrist_shoulder_y = ((l_wr[1] + r_wr[1]) / 2 - mid_sh[1]) / h
        shoulder_hip_y = (mid_sh[1] - mid_hip[1]) / h

        # ---- Ratios (scale invariant) ----
        hip_ankle_ratio = (mid_hip[1] - np.mean([l_ankle[1], r_ankle[1]])) / h
        shoulder_ankle_ratio = (mid_sh[1] - np.mean([l_ankle[1], r_ankle[1]])) / h

        # ---- Symmetry ----
        knee_asym = abs(left_knee - right_knee)
        elbow_asym = abs(left_elbow - right_elbow)

        # ---- Ground relation ----
        hip_ground_ratio = mid_hip[1] / h
        knee_ground_ratio = np.mean([l_knee[1], r_knee[1]]) / h

        features = {
            # Angles (9)
            "left_knee": left_knee,
            "right_knee": right_knee,
            "left_elbow": left_elbow,
            "right_elbow": right_elbow,
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "hip_angle": hip_angle,
            "knee_min": knee_min,
            "elbow_min": elbow_min,

            # Velocities (4)
            "left_knee_vel": vel[0],
            "right_knee_vel": vel[1],
            "left_elbow_vel": vel[2],
            "right_elbow_vel": vel[3],

            # Orientation (3)
            "torso_dx": torso_dx,
            "torso_dy": torso_dy,
            "torso_vertical_angle": torso_vertical_angle,

            # Relative posture (2)
            "wrist_shoulder_y": wrist_shoulder_y,
            "shoulder_hip_y": shoulder_hip_y,

            # Ratios (2)
            "hip_ankle_ratio": hip_ankle_ratio,
            "shoulder_ankle_ratio": shoulder_ankle_ratio,

            # Symmetry (2)
            "knee_asym": knee_asym,
            "elbow_asym": elbow_asym,

            # Ground (2)
            "hip_ground_ratio": hip_ground_ratio,
            "knee_ground_ratio": knee_ground_ratio,
        }

        return features, results.pose_landmarks
