from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config/settings.json")


@dataclass
class AppConfig:
    owner_name: str = "Thujee"
    camera_index: int = 0

    sender_email: str = "thujeeforearn@gmail.com"
    sender_app_password: str = "dwbs acdr xiub dwkn"
    receiver_email: str = "thujeeforearn@gmail.com"
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587

    face_match_threshold: float = 0.48
    body_match_threshold: float = 0.45

    motion_pixel_threshold: int = 1800
    people_confidence_threshold: float = 0.6

    alert_cooldown_sec: int = 45
    warning_cooldown_sec: int = 20
    greeting_cooldown_sec: int = 25

    owner_greeting_text: str = "Welcome back Thujee"
    warning_text: str = (
        "This is restricted area. If you come in or do any unauthorized things, "
        "you will face consequences."
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        defaults = cls()
        payload = defaults.to_dict()
        payload.update(data or {})
        return cls(**payload)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    if not path.exists():
        return AppConfig()

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return AppConfig.from_dict(raw)


def save_config(config: AppConfig, path: Path = DEFAULT_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
