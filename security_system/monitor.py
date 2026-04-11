from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np

from .alerts import build_alert_image_path, send_email_alert
from .biometrics import BiometricsEngine
from .config import AppConfig
from .models import OwnerProfile
from .voice import VoiceNotifier


class SecurityMonitor:
    def __init__(
        self,
        config: AppConfig,
        owner_profile: OwnerProfile,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config
        self.owner_profile = owner_profile
        self.logger = logger

        self._voice = VoiceNotifier()
        self._biometrics = BiometricsEngine()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._last_alert_time = 0.0
        self._last_warning_time = 0.0
        self._last_greeting_time = 0.0
        self._previous_body_signature: Optional[np.ndarray] = None

        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=16,
            detectShadows=True,
        )
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _log(self, text: str) -> None:
        if self.logger:
            stamp = datetime.now().strftime("%H:%M:%S")
            self.logger(f"[{stamp}] {text}")

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _motion_pixels(self, frame_bgr: np.ndarray) -> int:
        mask = self._bg_subtractor.apply(frame_bgr)
        _, thresholded = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)
        thresholded = cv2.medianBlur(thresholded, 5)
        return int(cv2.countNonZero(thresholded))

    def _person_present(self, frame_bgr: np.ndarray) -> bool:
        small = cv2.resize(frame_bgr, None, fx=0.65, fy=0.65)
        _, weights = self._hog.detectMultiScale(
            small,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05,
        )

        if len(weights) == 0:
            return False

        confidence_values = np.asarray(weights).reshape(-1)
        return bool(np.any(confidence_values >= self.config.people_confidence_threshold))

    def _is_owner_by_face(self, frame_bgr: np.ndarray) -> bool:
        face_encodings = self._biometrics.extract_face_encodings(frame_bgr)
        if not face_encodings:
            return False

        for face_encoding in face_encodings:
            result = self._biometrics.match_owner_face(
                face_encoding=face_encoding,
                owner_face_encodings=self.owner_profile.face_encodings,
                threshold=self.config.face_match_threshold,
            )
            if result.is_owner:
                return True

        return False

    def _is_owner_by_body(self, frame_bgr: np.ndarray) -> bool:
        current_signature = self._biometrics.extract_body_signature(frame_bgr)
        if current_signature is None:
            return False

        distance = self._biometrics.body_distance(
            current=current_signature,
            reference=self.owner_profile.body_signature,
        )
        return distance <= self.config.body_match_threshold

    def _track_body_activity(self, frame_bgr: np.ndarray) -> tuple[str, Optional[np.ndarray]]:
        body_signature = self._biometrics.extract_body_signature(frame_bgr)
        if body_signature is None:
            return "pose not visible", None

        if self._previous_body_signature is None:
            self._previous_body_signature = body_signature
            return "body detected", body_signature

        delta = self._biometrics.body_distance(body_signature, self._previous_body_signature)
        self._previous_body_signature = body_signature

        if delta >= 0.26:
            return "high movement", body_signature
        if delta >= 0.12:
            return "moderate movement", body_signature
        return "low movement", body_signature

    def _maybe_greet_owner(self) -> None:
        now = time.time()
        if now - self._last_greeting_time < self.config.greeting_cooldown_sec:
            return

        text = self.config.owner_greeting_text or f"Welcome back {self.owner_profile.owner_name}"
        self._voice.speak(text)
        self._log("Owner authenticated. Greeting voice played.")
        self._last_greeting_time = now

    def _handle_intruder(
        self,
        frame_bgr: np.ndarray,
        body_signature: Optional[np.ndarray],
        body_activity: str,
    ) -> None:
        now = time.time()

        if now - self._last_warning_time >= self.config.warning_cooldown_sec:
            self._voice.speak(self.config.warning_text)
            self._log("Warning voice played for unauthorized person.")
            self._last_warning_time = now

        if now - self._last_alert_time >= self.config.alert_cooldown_sec:
            image_path = build_alert_image_path()
            cv2.imwrite(str(image_path), frame_bgr)

            if body_signature is not None:
                np.save(str(image_path.with_suffix(".npy")), body_signature.astype(np.float32))

            subject = "Room Security Alert: Unauthorized Person Detected"
            body = (
                "An unauthorized person was detected near your laptop/room.\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Detected body activity: {body_activity}\n"
                "Attached image captured by the security system."
            )
            success, result = send_email_alert(
                config=self.config,
                image_path=image_path,
                subject=subject,
                body=body,
            )

            self._last_alert_time = now
            if success:
                self._log(f"Email alert sent with image: {image_path}")
            else:
                self._log(f"Email alert failed: {result}")

    def _run_loop(self) -> None:
        capture = cv2.VideoCapture(self.config.camera_index)
        if not capture.isOpened():
            self._log("Camera could not be opened. Monitoring stopped.")
            self._running = False
            return

        self._log("Monitoring started. Press Q in video window to stop.")

        try:
            while self._running:
                ok, frame = capture.read()
                if not ok:
                    self._log("Camera frame read failed.")
                    continue

                motion_pixels = self._motion_pixels(frame)
                person_detected = self._person_present(frame)
                motion_detected = motion_pixels >= self.config.motion_pixel_threshold
                body_activity = "not available"
                body_signature: Optional[np.ndarray] = None

                if person_detected:
                    body_activity, body_signature = self._track_body_activity(frame)
                else:
                    self._previous_body_signature = None

                status_text = "No person"
                status_color = (140, 140, 140)

                if person_detected and motion_detected:
                    face_owner = self._is_owner_by_face(frame)

                    if face_owner:
                        status_text = "Owner verified by face"
                        status_color = (0, 220, 0)
                        self._maybe_greet_owner()
                    else:
                        body_owner = False
                        if body_signature is not None:
                            distance = self._biometrics.body_distance(
                                current=body_signature,
                                reference=self.owner_profile.body_signature,
                            )
                            body_owner = distance <= self.config.body_match_threshold

                        if body_owner:
                            status_text = "Owner verified by body"
                            status_color = (0, 220, 0)
                            self._maybe_greet_owner()
                        else:
                            status_text = "Unauthorized person detected"
                            status_color = (0, 0, 255)
                            self._handle_intruder(
                                frame_bgr=frame,
                                body_signature=body_signature,
                                body_activity=body_activity,
                            )
                elif person_detected:
                    status_text = "Person detected"
                    status_color = (0, 180, 255)

                cv2.putText(
                    frame,
                    f"Status: {status_text}",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Motion pixels: {motion_pixels}",
                    (16, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    f"Body activity: {body_activity}",
                    (16, 92),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (230, 230, 230),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "Press Q to stop monitoring",
                    (16, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Room Security Monitor", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self._running = False
                    break
        finally:
            capture.release()
            cv2.destroyWindow("Room Security Monitor")
            self._running = False
            self._log("Monitoring stopped.")
