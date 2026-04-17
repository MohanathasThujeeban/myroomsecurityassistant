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
        self._owner_seen_last_frame = False
        self._intruder_seen_last_frame = False
        self._frame_index = 0

        self._face_check_every_n_frames = 2
        self._cached_face_owner = False
        self._cached_face_present = False
        self._cached_face_count = 0
        self._last_face_check_frame = -999

        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300,
            varThreshold=16,
            detectShadows=True,
        )

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
        if not self._voice.available:
            self._log("Voice engine unavailable. Alerts will continue without spoken warnings.")
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _motion_pixels(
        self,
        frame_bgr: np.ndarray,
    ) -> int:
        mask = self._bg_subtractor.apply(frame_bgr)
        _, thresholded = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)
        thresholded = cv2.medianBlur(thresholded, 5)
        return int(cv2.countNonZero(thresholded))

    def _get_face_state(self, frame_bgr: np.ndarray) -> tuple[bool, bool, int]:
        face_encodings = self._biometrics.extract_face_encodings(frame_bgr, scale=0.5)
        if not face_encodings:
            face_encodings = self._biometrics.extract_face_encodings(frame_bgr, scale=1.0)
        if not face_encodings:
            return False, False, 0

        face_count = len(face_encodings)
        for face_encoding in face_encodings:
            result = self._biometrics.match_owner_face(
                face_encoding=face_encoding,
                owner_face_encodings=self.owner_profile.face_encodings,
                threshold=self.config.face_match_threshold,
            )
            if result.is_owner:
                return True, True, face_count

        return True, False, face_count

    def _maybe_greet_owner(self) -> None:
        text = self.config.owner_greeting_text or f"Welcome back {self.owner_profile.owner_name}"
        if self._voice.speak(text):
            self._log("Owner authenticated. Greeting voice played.")
        else:
            self._log("Owner authenticated, but voice output is unavailable.")

    def _handle_intruder(
        self,
        frame_bgr: np.ndarray,
        face_count: int,
        force_warning: bool = False,
    ) -> None:
        now = time.time()

        if force_warning or (now - self._last_warning_time >= self.config.warning_cooldown_sec):
            if self._voice.speak(self.config.warning_text):
                self._log("Warning voice played for unauthorized person.")
            else:
                self._log("Unauthorized person detected, but voice output is unavailable.")
            self._last_warning_time = now

        if now - self._last_alert_time >= self.config.alert_cooldown_sec:
            image_path = build_alert_image_path()
            image_saved = cv2.imwrite(str(image_path), frame_bgr)
            if not image_saved:
                self._log(
                    f"Alert snapshot could not be written at {image_path}. "
                    "Email will be sent without attachment."
                )

            subject = "Room Security Alert: Unauthorized Person Detected"
            body = (
                "An unauthorized person was detected near your laptop/room.\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Detected faces: {face_count}\n"
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
                self._log(f"Email alert success: {result} ({image_path})")
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

                display_frame = cv2.flip(frame, 1)

                self._frame_index += 1
                motion_pixels = self._motion_pixels(frame)

                if self._frame_index - self._last_face_check_frame >= self._face_check_every_n_frames:
                    face_present, face_owner, face_count = self._get_face_state(frame)
                    self._cached_face_present = face_present
                    self._cached_face_owner = face_owner
                    self._cached_face_count = face_count
                    self._last_face_check_frame = self._frame_index

                face_present = self._cached_face_present
                face_owner = self._cached_face_owner
                face_count = self._cached_face_count

                status_text = "No face"
                status_color = (140, 140, 140)

                if face_owner:
                    status_text = "Owner verified by face"
                    status_color = (0, 220, 0)
                    if not self._owner_seen_last_frame:
                        self._maybe_greet_owner()
                    self._owner_seen_last_frame = True
                    self._intruder_seen_last_frame = False
                elif face_present:
                    self._owner_seen_last_frame = False
                    is_new_intruder_event = not self._intruder_seen_last_frame
                    self._intruder_seen_last_frame = True
                    status_text = "Unauthorized face detected"
                    status_color = (0, 0, 255)
                    self._handle_intruder(
                        frame_bgr=frame,
                        face_count=face_count,
                        force_warning=is_new_intruder_event,
                    )
                else:
                    self._owner_seen_last_frame = False
                    self._intruder_seen_last_frame = False

                cv2.putText(
                    display_frame,
                    f"Status: {status_text}",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display_frame,
                    f"Motion pixels: {motion_pixels}",
                    (16, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display_frame,
                    f"Faces detected: {face_count}",
                    (16, 92),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (230, 230, 230),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display_frame,
                    "Press Q to stop monitoring",
                    (16, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("Room Security Monitor", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self._running = False
                    break
        finally:
            capture.release()
            cv2.destroyWindow("Room Security Monitor")
            self._running = False
            self._log("Monitoring stopped.")
