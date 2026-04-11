from __future__ import annotations

import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Tuple

from .config import AppConfig


def _image_subtype(path: Path) -> str:
    suffix = path.suffix.lower().replace(".", "")
    if suffix == "jpg":
        return "jpeg"
    return suffix or "jpeg"


def build_alert_image_path(base_dir: Path = Path("data/alerts")) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"intruder_{timestamp}.jpg"


def send_email_alert(
    config: AppConfig,
    image_path: Path,
    subject: str,
    body: str,
) -> Tuple[bool, str]:
    if not config.sender_email or not config.sender_app_password or not config.receiver_email:
        return False, "Email settings are incomplete."

    if not image_path.exists():
        return False, f"Image does not exist: {image_path}"

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config.sender_email
    message["To"] = config.receiver_email
    message.set_content(body)

    with image_path.open("rb") as image_file:
        image_bytes = image_file.read()

    message.add_attachment(
        image_bytes,
        maintype="image",
        subtype=_image_subtype(image_path),
        filename=image_path.name,
    )

    try:
        with smtplib.SMTP(config.smtp_server, config.smtp_port, timeout=30) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(config.sender_email, config.sender_app_password)
            smtp.send_message(message)
    except Exception as exc:
        return False, f"Email sending failed: {exc}"

    return True, "Alert email sent successfully."
