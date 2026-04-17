from __future__ import annotations

import os
import queue
import subprocess
import threading
from typing import Optional

import pyttsx3


class VoiceNotifier:
    def __init__(self) -> None:
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._engine: Optional[object] = None
        self._engine_ready = threading.Event()
        self._init_failed = False
        self._windows_fallback = os.name == "nt"

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._engine_ready.wait(timeout=1.5)

    @property
    def available(self) -> bool:
        return self._pyttsx_available() or self._windows_fallback

    def speak(self, text: str) -> bool:
        payload = text.strip()
        if not payload:
            return False

        if self._pyttsx_available() or not self._engine_ready.is_set():
            self._queue.put(payload)
            return True

        if self._windows_fallback:
            threading.Thread(target=self._speak_with_powershell, args=(payload,), daemon=True).start()
            return True

        return False

    def _pyttsx_available(self) -> bool:
        return self._engine_ready.is_set() and (not self._init_failed) and self._engine is not None

    @staticmethod
    def _escape_for_single_quote_ps(text: str) -> str:
        return text.replace("'", "''")

    def _speak_with_powershell(self, text: str) -> None:
        escaped_text = self._escape_for_single_quote_ps(text)
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$voice.Speak('{escaped_text}')"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
        except Exception:
            return

    def _worker_loop(self) -> None:
        try:
            try:
                engine = pyttsx3.init(driverName="sapi5")
            except Exception:
                engine = pyttsx3.init()
            engine.setProperty("rate", 168)
            self._engine = engine
        except Exception:
            self._engine = None
            self._init_failed = True
            self._engine_ready.set()
            return

        self._engine_ready.set()

        while True:
            utterance = self._queue.get()
            try:
                self._engine.say(utterance)
                self._engine.runAndWait()
            except Exception:
                # Recreate the engine once if the runtime state gets corrupted.
                try:
                    replacement = pyttsx3.init()
                    replacement.setProperty("rate", 168)
                    self._engine = replacement
                    self._engine.say(utterance)
                    self._engine.runAndWait()
                except Exception:
                    self._init_failed = True
                    self._engine = None
                    return
            finally:
                self._queue.task_done()
