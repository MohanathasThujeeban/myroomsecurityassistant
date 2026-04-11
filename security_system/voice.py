from __future__ import annotations

import threading

import pyttsx3


class VoiceNotifier:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._engine = None

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 168)
            self._engine = engine
        except Exception:
            self._engine = None

    @property
    def available(self) -> bool:
        return self._engine is not None

    def speak(self, text: str) -> None:
        if not self._engine:
            return

        def _run() -> None:
            with self._lock:
                self._engine.say(text)
                self._engine.runAndWait()

        threading.Thread(target=_run, daemon=True).start()
