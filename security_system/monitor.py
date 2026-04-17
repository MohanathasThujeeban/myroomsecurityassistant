from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Callable, Optional, Tuple

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
        self._previous_body_signature: Optional[np.ndarray] = None
        self._cached_body_signature: Optional[np.ndarray] = None
        self._cached_body_activity = "not available"
        self._last_body_signature_frame = -999
        self._frame_index = 0

        self._person_cache_frames = 3
        self._body_signature_every_n_frames = 2
        self._global_person_scan_every_n_frames = 12
        self._face_check_every_n_frames = 2
        self._cached_person_detected = False
        self._cached_person_box: Optional[Tuple[int, int, int, int]] = None
        self._cached_person_confidence = 0.0
        self._cached_face_owner = False
        self._last_face_check_frame = -999

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
        if not self._voice.available:
            self._log("Voice engine unavailable. Alerts will continue without spoken warnings.")
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _analyze_motion(
        self,
        frame_bgr: np.ndarray,
    ) -> tuple[int, Optional[Tuple[int, int, int, int]]]:
        mask = self._bg_subtractor.apply(frame_bgr)
        _, thresholded = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)
        thresholded = cv2.medianBlur(thresholded, 5)
        motion_pixels = int(cv2.countNonZero(thresholded))

        min_motion_for_roi = max(260, int(self.config.motion_pixel_threshold * 0.22))
        if motion_pixels < min_motion_for_roi:
            return motion_pixels, None

        points = cv2.findNonZero(thresholded)
        if points is None:
            return motion_pixels, None

        frame_h, frame_w = frame_bgr.shape[:2]
        x, y, w, h = cv2.boundingRect(points)
        pad_x = max(24, int(w * 0.2))
        pad_y = max(24, int(h * 0.2))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_w, x + w + pad_x)
        y2 = min(frame_h, y + h + pad_y)

        if x2 <= x1 or y2 <= y1:
            return motion_pixels, None

        return motion_pixels, (x1, y1, x2 - x1, y2 - y1)

    def _expand_box(
        self,
        box: Tuple[int, int, int, int],
        frame_shape: tuple[int, ...],
        scale: float = 1.26,
    ) -> Tuple[int, int, int, int]:
        frame_h, frame_w = frame_shape[:2]
        x, y, w, h = [int(v) for v in box]

        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        nw = int(max(w * scale, w + 24))
        nh = int(max(h * scale, h + 24))

        x1 = max(0, int(cx - (nw / 2.0)))
        y1 = max(0, int(cy - (nh / 2.0)))
        x2 = min(frame_w, x1 + nw)
        y2 = min(frame_h, y1 + nh)

        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    def _detect_person_in_region(
        self,
        frame_bgr: np.ndarray,
        region: Tuple[int, int, int, int],
        threshold_scale: float = 1.0,
    ) -> tuple[bool, Optional[Tuple[int, int, int, int]], float]:
        rx, ry, rw, rh = [int(v) for v in region]
        if rw < 40 or rh < 60:
            return False, None, 0.0

        crop = frame_bgr[ry : ry + rh, rx : rx + rw]
        if crop.size == 0:
            return False, None, 0.0

        detect_scale = 0.5
        small = cv2.resize(
            crop,
            None,
            fx=detect_scale,
            fy=detect_scale,
            interpolation=cv2.INTER_AREA,
        )
        boxes, weights = self._hog.detectMultiScale(
            small,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.08,
        )

        best_confidence = 0.0
        if len(boxes) == 0:
            return False, None, best_confidence

        confidence_values = (
            np.asarray(weights).reshape(-1)
            if len(weights)
            else np.zeros(len(boxes), dtype=np.float32)
        )

        best_index = 0
        best_score = -1.0
        for index, (_, _, w, h) in enumerate(boxes):
            area_score = float(w * h)
            confidence = float(confidence_values[index]) if index < len(confidence_values) else 0.0
            score = area_score * (1.0 + max(confidence, 0.0))
            if score > best_score:
                best_score = score
                best_index = index
                best_confidence = confidence

        threshold = self.config.people_confidence_threshold * threshold_scale
        if best_confidence >= threshold:
            sx, sy, sw, sh = [int(v) for v in boxes[best_index]]
            x = int(sx / detect_scale) + rx
            y = int(sy / detect_scale) + ry
            w = int(sw / detect_scale)
            h = int(sh / detect_scale)
            return True, (x, y, w, h), best_confidence

        return False, None, best_confidence

    def _detect_person(
        self,
        frame_bgr: np.ndarray,
        motion_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> tuple[bool, Optional[Tuple[int, int, int, int]], float]:
        frame_h, frame_w = frame_bgr.shape[:2]
        full_region = (0, 0, frame_w, frame_h)

        candidate_regions: list[Tuple[int, int, int, int]] = []
        if self._cached_person_box is not None:
            candidate_regions.append(self._expand_box(self._cached_person_box, frame_bgr.shape))
        if motion_roi is not None:
            candidate_regions.append(motion_roi)

        unique_regions: list[Tuple[int, int, int, int]] = []
        seen: set[Tuple[int, int, int, int]] = set()
        for region in candidate_regions:
            normalized = tuple(int(v) for v in region)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_regions.append(normalized)

        best_confidence = 0.0
        for region in unique_regions:
            detected, person_box, confidence = self._detect_person_in_region(
                frame_bgr,
                region,
                threshold_scale=0.92,
            )
            best_confidence = max(best_confidence, confidence)
            if detected:
                return True, person_box, confidence

        should_run_full_scan = (
            self._frame_index <= 1
            or not unique_regions
            or self._frame_index % self._global_person_scan_every_n_frames == 0
        )
        if should_run_full_scan:
            detected, person_box, confidence = self._detect_person_in_region(
                frame_bgr,
                full_region,
            )
            best_confidence = max(best_confidence, confidence)
            if detected:
                return True, person_box, confidence

        fallback_box = self._infer_person_box_from_face(frame_bgr)
        if fallback_box is not None:
            fallback_confidence = max(0.05, self.config.people_confidence_threshold * 0.5)
            return True, fallback_box, fallback_confidence

        return False, None, best_confidence

    def _infer_person_box_from_face(
        self,
        frame_bgr: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        face_box = self._biometrics.detect_primary_face_box(frame_bgr, scale=0.75)
        if face_box is None:
            return None

        frame_h, frame_w = frame_bgr.shape[:2]
        fx, fy, fw, fh = face_box

        x = max(0, fx - int(0.9 * fw))
        y = max(0, fy - int(0.7 * fh))
        w = int(2.8 * fw)
        h = int(4.6 * fh)

        if x + w > frame_w:
            w = frame_w - x
        if y + h > frame_h:
            h = frame_h - y

        if w < 40 or h < 60:
            return None

        return x, y, w, h

    def _get_person_state(
        self,
        frame_bgr: np.ndarray,
        motion_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> tuple[bool, Optional[Tuple[int, int, int, int]], float]:
        should_refresh = (
            self._frame_index <= 1
            or self._frame_index % self._person_cache_frames == 0
        )
        if should_refresh:
            detected, person_box, confidence = self._detect_person(
                frame_bgr,
                motion_roi=motion_roi,
            )
            self._cached_person_detected = detected
            self._cached_person_box = person_box
            self._cached_person_confidence = confidence

        return (
            self._cached_person_detected,
            self._cached_person_box,
            self._cached_person_confidence,
        )

    def _is_owner_by_face(self, frame_bgr: np.ndarray) -> bool:
        face_encodings = self._biometrics.extract_face_encodings(frame_bgr, scale=0.5)
        if not face_encodings:
            face_encodings = self._biometrics.extract_face_encodings(frame_bgr, scale=1.0)
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

    def _track_body_activity(
        self,
        frame_bgr: np.ndarray,
        person_box: Optional[Tuple[int, int, int, int]],
    ) -> tuple[str, Optional[np.ndarray]]:
        should_refresh_signature = (
            self._cached_body_signature is None
            or self._frame_index - self._last_body_signature_frame >= self._body_signature_every_n_frames
        )
        if not should_refresh_signature:
            return self._cached_body_activity, self._cached_body_signature

        body_signature = self._biometrics.extract_body_signature(
            frame_bgr,
            person_box=person_box,
        )
        self._last_body_signature_frame = self._frame_index

        if body_signature is None:
            self._cached_body_signature = None
            self._cached_body_activity = "pose not visible"
            return self._cached_body_activity, None

        if self._previous_body_signature is None:
            self._previous_body_signature = body_signature
            self._cached_body_signature = body_signature
            self._cached_body_activity = "body detected"
            return self._cached_body_activity, body_signature

        delta = self._biometrics.body_distance(body_signature, self._previous_body_signature)
        self._previous_body_signature = body_signature
        self._cached_body_signature = body_signature

        if delta >= 0.26:
            self._cached_body_activity = "high movement"
            return self._cached_body_activity, body_signature
        if delta >= 0.12:
            self._cached_body_activity = "moderate movement"
            return self._cached_body_activity, body_signature

        self._cached_body_activity = "low movement"
        return self._cached_body_activity, body_signature

    def _maybe_greet_owner(self) -> None:
        text = self.config.owner_greeting_text or f"Welcome back {self.owner_profile.owner_name}"
        if self._voice.speak(text):
            self._log("Owner authenticated. Greeting voice played.")
        else:
            self._log("Owner authenticated, but voice output is unavailable.")

    def _handle_intruder(
        self,
        frame_bgr: np.ndarray,
        body_signature: Optional[np.ndarray],
        body_activity: str,
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
                motion_pixels, motion_roi = self._analyze_motion(frame)
                person_detected, person_box, person_confidence = self._get_person_state(
                    frame,
                    motion_roi=motion_roi,
                )
                body_activity = "not available"
                body_signature: Optional[np.ndarray] = None

                if person_detected:
                    body_activity, body_signature = self._track_body_activity(frame, person_box)
                else:
                    self._previous_body_signature = None
                    self._cached_body_signature = None
                    self._cached_body_activity = "not available"
                    self._last_body_signature_frame = -999

                if self._frame_index - self._last_face_check_frame >= self._face_check_every_n_frames:
                    self._cached_face_owner = self._is_owner_by_face(frame)
                    self._last_face_check_frame = self._frame_index

                face_owner = self._cached_face_owner

                status_text = "No person"
                status_color = (140, 140, 140)

                if face_owner:
                    status_text = "Owner verified by face"
                    status_color = (0, 220, 0)
                    if not self._owner_seen_last_frame:
                        self._maybe_greet_owner()
                    self._owner_seen_last_frame = True
                    self._intruder_seen_last_frame = False
                elif person_detected:
                    self._owner_seen_last_frame = False
                    is_new_intruder_event = not self._intruder_seen_last_frame
                    self._intruder_seen_last_frame = True
                    status_text = "Unauthorized person detected"
                    status_color = (0, 0, 255)
                    self._handle_intruder(
                        frame_bgr=frame,
                        body_signature=body_signature,
                        body_activity=body_activity,
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
                    f"Body activity: {body_activity}",
                    (16, 92),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (230, 230, 230),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display_frame,
                    f"Person confidence: {person_confidence:.2f}",
                    (16, 122),
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
