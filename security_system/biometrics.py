from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import face_recognition
import numpy as np


@dataclass
class FaceMatchResult:
    is_owner: bool
    best_distance: float


class BiometricsEngine:
    def __init__(self) -> None:
        self._person_detector = cv2.HOGDescriptor()
        self._person_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._roi_hog = cv2.HOGDescriptor(
            (64, 128),
            (16, 16),
            (8, 8),
            (8, 8),
            9,
        )

    def extract_face_encodings(self, frame_bgr: np.ndarray, scale: float = 1.0) -> List[np.ndarray]:
        if scale <= 0 or scale > 1.0:
            scale = 1.0

        if scale < 1.0:
            frame_bgr = cv2.resize(
                frame_bgr,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model="hog")
        if not face_locations:
            return []

        return face_recognition.face_encodings(
            rgb,
            known_face_locations=face_locations,
            num_jitters=1,
            model="small",
        )

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

    def extract_body_signature(
        self,
        frame_bgr: np.ndarray,
        person_box: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[np.ndarray]:
        frame_h, frame_w = frame_bgr.shape[:2]
        if frame_h < 64 or frame_w < 64:
            return None

        if person_box is None:
            detect_scale = 0.55
            small = cv2.resize(
                frame_bgr,
                None,
                fx=detect_scale,
                fy=detect_scale,
                interpolation=cv2.INTER_AREA,
            )

            boxes, weights = self._person_detector.detectMultiScale(
                small,
                winStride=(8, 8),
                padding=(8, 8),
                scale=1.08,
            )
            if len(boxes) == 0:
                return None

            weights_arr = np.asarray(weights).reshape(-1) if len(weights) else np.zeros(len(boxes))

            best_index = 0
            best_score = -1.0
            for index, (x, y, w, h) in enumerate(boxes):
                area_score = float(w * h)
                confidence = float(weights_arr[index]) if index < len(weights_arr) else 0.0
                score = area_score * (1.0 + max(confidence, 0.0))
                if score > best_score:
                    best_score = score
                    best_index = index

            sx, sy, sw, sh = [int(v) for v in boxes[best_index]]
            x = int(sx / detect_scale)
            y = int(sy / detect_scale)
            w = int(sw / detect_scale)
            h = int(sh / detect_scale)
        else:
            x, y, w, h = [int(v) for v in person_box]
            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)

        pad_x = int(0.08 * w)
        pad_y = int(0.06 * h)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_w, x + w + pad_x)
        y2 = min(frame_h, y + h + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_roi, (64, 128), interpolation=cv2.INTER_AREA)

        descriptor = self._roi_hog.compute(resized)
        if descriptor is None:
            return None

        hog_vec = descriptor.flatten().astype(np.float32)

        hist = cv2.calcHist([resized], [0], None, [16], [0, 256]).flatten().astype(np.float32)
        hist_norm = np.linalg.norm(hist)
        if hist_norm > 1e-6:
            hist = hist / hist_norm

        geom = np.array(
            [
                w / max(frame_w, 1),
                h / max(frame_h, 1),
                (x + (w / 2.0)) / max(frame_w, 1),
                (y + (h / 2.0)) / max(frame_h, 1),
                w / max(h, 1),
            ],
            dtype=np.float32,
        )

        signature = np.concatenate([hog_vec, hist, geom], axis=0).astype(np.float32)
        signature_norm = np.linalg.norm(signature)
        if signature_norm > 1e-6:
            signature /= signature_norm

        return signature

    @staticmethod
    
    def body_distance(current: np.ndarray, reference: np.ndarray) -> float:
        return float(np.linalg.norm(current - reference))
    