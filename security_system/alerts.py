from __future__ import annotations

import smtplib
import ssl
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


def _normalize_email(value: str) -> str:
    return (value or "").strip()


def _normalize_password(value: str) -> str:
    return (value or "").replace(" ", "").strip()


def build_alert_image_path(base_dir: Path = Path("data/alerts")) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir / f"intruder_{timestamp}.png"


def send_email_alert(
    config: AppConfig,
    image_path: Path,
    subject: str,
    body: str,
) -> Tuple[bool, str]:
    sender_email = _normalize_email(config.sender_email)
    sender_password = _normalize_password(config.sender_app_password)
    receiver_email = _normalize_email(config.receiver_email)

    if not sender_email or not sender_password or not receiver_email:
        return False, "Email settings are incomplete."

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = receiver_email

    attachment_available = image_path.exists()
    if attachment_available:
        with image_path.open("rb") as image_file:
            image_bytes = image_file.read()

        message.set_content(body)
        message.add_attachment(
            image_bytes,
            maintype="image",
            subtype=_image_subtype(image_path),
            filename=image_path.name,
        )
    else:
        message.set_content(
            body
            + "\n\n"
            + f"Snapshot attachment was unavailable at send time: {image_path}"
        )

    smtp_server = _normalize_email(config.smtp_server)
    smtp_port = int(config.smtp_port)

    preferred_attempts = []
    if smtp_port == 465:
        preferred_attempts = [("ssl", 465), ("starttls", 587)]
    elif smtp_port == 587:
        preferred_attempts = [("starttls", 587), ("ssl", 465)]
    else:
        preferred_attempts = [("starttls", smtp_port), ("ssl", smtp_port)]

    attempts = []
    for mode, port in preferred_attempts:
        candidate = (mode, int(port))
        if candidate not in attempts:
            attempts.append(candidate)

    last_error = "Unknown email error"
    for mode, port in attempts:
        try:
            if mode == "starttls":
                with smtplib.SMTP(smtp_server, port, timeout=30) as smtp:
                    smtp.ehlo()
                    smtp.starttls(context=ssl.create_default_context())
                    smtp.ehlo()
                    smtp.login(sender_email, sender_password)
                    smtp.send_message(message)
            else:
                with smtplib.SMTP_SSL(
                    smtp_server,
                    port,
                    timeout=30,
                    context=ssl.create_default_context(),
                ) as smtp:
                    smtp.ehlo()
                    smtp.login(sender_email, sender_password)
                    smtp.send_message(message)
            if attachment_available:
                return True, "Alert email sent successfully."
            return True, "Alert email sent without image attachment."
        except Exception as exc:
            last_error = f"{mode.upper()} on port {port} failed: {exc}"

    return False, f"Email sending failed: {last_error}"
