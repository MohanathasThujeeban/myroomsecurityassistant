from __future__ import annotations

import time
from typing import Callable, Optional

import cv2
import numpy as np

from .biometrics import BiometricsEngine
from .models import OwnerProfile
from .storage import owner_profile_exists, save_owner_profile


def enroll_owner(
    owner_name: str,
    camera_index: int = 0,
    min_face_samples: int = 20,
    max_duration_sec: int = 120,
    logger: Optional[Callable[[str], None]] = None,
) -> OwnerProfile:
    if owner_profile_exists():
        raise RuntimeError(
            "Only one owner profile is supported. "
            "Delete existing owner data before running enrollment again."
        )

    def log(message: str) -> None:
        if logger:
            logger(message)

    biometrics = BiometricsEngine()
    capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        raise RuntimeError("Cannot access camera for enrollment.")

    face_samples = []
    frame_index = 0
    started = time.time()

    log("Enrollment started. Face the camera clearly.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Unable to read camera frame during enrollment.")

            display_frame = cv2.flip(frame, 1)

            frame_index += 1

            elapsed = time.time() - started
            if elapsed >= max_duration_sec:
                log("Enrollment timeout reached.")
                break

            face_encodings = biometrics.extract_face_encodings(frame)

            # Sample every few frames to reduce near-duplicate captures.
            if face_encodings and frame_index % 4 == 0:
                face_samples.append(face_encodings[0])

            message = f"Face samples: {len(face_samples)}/{min_face_samples}"
            cv2.putText(
                display_frame,
                message,
                (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                display_frame,
                "Press Q to stop once both counts are complete.",
                (18, 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (180, 255, 180),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Owner Enrollment", display_frame)

            enough_samples = len(face_samples) >= min_face_samples

            pressed = cv2.waitKey(1) & 0xFF
            if pressed == ord("q") and enough_samples:
                break
            if enough_samples:
                # Auto-finish once enough data has been gathered.
                break
    finally:
        capture.release()
        cv2.destroyWindow("Owner Enrollment")

    if len(face_samples) < min_face_samples:
        raise RuntimeError(
            "Enrollment failed: not enough face samples captured. "
            "Try again in better lighting."
        )

    face_matrix = np.vstack(face_samples).astype(np.float32)

    profile = OwnerProfile(
        owner_name=owner_name,
        face_encodings=face_matrix,
    )

    save_owner_profile(profile)
    log("Enrollment completed and profile saved.")
    return profile
