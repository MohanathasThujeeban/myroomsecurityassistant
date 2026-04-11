# Thujee's Room sentinal (Python)

This desktop app provides:

- Motion + person detection using webcam
- Owner recognition by face (primary)
- Owner recognition by body signature (fallback for masked/partial face)
- Body activity classification (low/moderate/high movement)
- Voice greeting for owner: "Welcome back Thujee" (customizable)
- Voice warning for unauthorized person (customizable)
- Email alert with intruder photo attachment

## Important Notes

- This is a practical starter system, not military-grade security.
- Body signature matching is a fallback heuristic and can produce false positives/negatives.
- For best accuracy, enroll in good lighting and keep a consistent camera angle.

## 1) Requirements

- Windows 10/11
- Python 3.10 or 3.11 recommended
- Webcam
- Internet for sending email alerts
- Gmail app password (16 characters) if using Gmail SMTP

Install dependencies:

```powershell
pip install -r requirements.txt
```

## 2) Run the App

```powershell
python app.py
```

## 3) Configure Settings in UI

Fill these values in the app window:

- Owner Name
- Camera Index (usually 0)
- Sender Email (your email)
- 16-digit App Password (email app password, not your normal login password)
- Receiver Email (where alerts will be sent)
- Threshold values (keep defaults first)

Click **Save Settings**.

## 4) Enroll Owner Profile

1. Click **Enroll Owner**.
2. Look at the camera with clear face and upper body visible.
3. The enrollment window collects multiple face and body samples.
4. It saves profile data to `data/owner_profile.npz`.

## 5) Start Monitoring

1. Click **Start Monitoring**.
2. A live camera window opens.
3. Behavior:
   - Owner face recognized -> greeting voice
   - Owner face not clear/masked but body matches -> greeting voice
  - Unauthorized person -> warning voice + email alert with captured image
  - Unauthorized events also save body-signature snapshot as a `.npy` file beside the image
4. Press **Q** in video window or click **Stop Monitoring** in UI.

## 6) Files

- `app.py`: desktop UI
- `security_system/enrollment.py`: owner capture/enrollment
- `security_system/monitor.py`: real-time detection + decision logic
- `security_system/alerts.py`: email alert sender
- `security_system/voice.py`: voice output
- `security_system/config.py`: settings file handling

## Troubleshooting

- If webcam does not open, try camera index 1 or 2.
- If face recognition package fails to install on Windows, install Visual C++ Build Tools and retry.
- If email fails:
  - verify sender email, receiver email, SMTP server/port
  - verify 16-character app password
  - check sender mailbox security settings

## Ethical and Legal Use

Use this system only in your own room/property and where local laws allow camera monitoring.
