from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import face_recognition
import mediapipe as mp
import numpy as np


@dataclass
class FaceMatchResult:
    is_owner: bool
    best_distance: float


class BiometricsEngine:
    def __init__(self) -> None:
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract_face_encodings(self, frame_bgr: np.ndarray) -> List[np.ndarray]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return face_recognition.face_encodings(rgb)

    def match_owner_face(
        self,
        face_encoding: np.ndarray,
        owner_face_encodings: np.ndarray,
        threshold: float,
    ) -> FaceMatchResult:
        distances = face_recognition.face_distance(owner_face_encodings, face_encoding)
        if distances.size == 0:
            return FaceMatchResult(is_owner=False, best_distance=999.0)

        best = float(np.min(distances))
        return FaceMatchResult(is_owner=best <= threshold, best_distance=best)

    def extract_body_signature(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks.landmark

        # Shoulders/hips are used for normalization to stay robust against camera distance.
        l_sh = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z], dtype=np.float32)
        r_sh = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z], dtype=np.float32)
        l_hp = np.array([landmarks[23].x, landmarks[23].y, landmarks[23].z], dtype=np.float32)
        r_hp = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z], dtype=np.float32)

        shoulder_width = float(np.linalg.norm(l_sh - r_sh))
        torso_center = (l_sh + r_sh) / 2.0
        hip_center = (l_hp + r_hp) / 2.0
        torso_height = float(np.linalg.norm(torso_center - hip_center))
        scale = shoulder_width + torso_height

        if scale <= 1e-6:
            return None

        tracked = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
        points = []
        for index in tracked:
            lm = landmarks[index]
            points.append([lm.x, lm.y, lm.z, lm.visibility])

        points_arr = np.asarray(points, dtype=np.float32)
        coords = points_arr[:, :3]
        visibility = points_arr[:, 3]

        normalized = (coords - hip_center) / scale

        signature = np.concatenate(
            [normalized.flatten(), visibility],
            axis=0,
        )
        return signature.astype(np.float32)

    @staticmethod
    def body_distance(current: np.ndarray, reference: np.ndarray) -> float:
        return float(np.linalg.norm(current - reference))
