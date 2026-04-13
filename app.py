from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from security_system import (
    delete_owner_profile,
    load_config,
    load_owner_profile,
    owner_profile_exists,
    save_config,
)
from security_system.config import AppConfig
from security_system.enrollment import enroll_owner
from security_system.monitor import SecurityMonitor


class SecurityApp(tk.Tk):
    COLOR_BG = "#0a1120"
    COLOR_SURFACE = "#111d34"
    COLOR_CARD = "#162745"
    COLOR_BORDER = "#2f4368"
    COLOR_TEXT = "#eef4ff"
    COLOR_MUTED = "#a9bad8"
    COLOR_ACCENT = "#4a8dff"
    COLOR_SUCCESS = "#1fbf9f"
    COLOR_DANGER = "#ff6f7d"
    COLOR_WARNING = "#f3b56a"

    def __init__(self) -> None:
        super().__init__()
        self.title("Thujee's Room sentinal")
        self.geometry("1100x760")
        self.minsize(980, 680)
        self.configure(bg=self.COLOR_BG)

        self.monitor = None
        self._log_queue: queue.Queue[str] = queue.Queue()
        self._status_queue: queue.Queue[tuple[str, str]] = queue.Queue()

        self.owner_name_var = tk.StringVar()
        self.camera_index_var = tk.StringVar()
        self.sender_email_var = tk.StringVar()
        self.sender_password_var = tk.StringVar()
        self.receiver_email_var = tk.StringVar()
        self.face_threshold_var = tk.StringVar()
        self.body_threshold_var = tk.StringVar()
        self.greeting_text_var = tk.StringVar()
        self.warning_text_var = tk.StringVar()
        self.status_var = tk.StringVar(value="System idle")
        self.owner_profile_var = tk.StringVar(value="No owner profile found.")
        self._pending_profile_refresh = False

        self._setup_theme()
        self._build_ui()
        self._load_initial_config()
        self._refresh_owner_profile_view()
        self._set_status("System ready", tone="ok")
        self.after(150, self._flush_logs)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_theme(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Root.TFrame", background=self.COLOR_BG)
        style.configure(
            "Surface.TFrame",
            background=self.COLOR_SURFACE,
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "Card.TLabelframe",
            background=self.COLOR_CARD,
            borderwidth=1,
            relief="solid",
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=self.COLOR_CARD,
            foreground=self.COLOR_TEXT,
            font=("Segoe UI Semibold", 11),
        )
        style.configure("Title.TLabel", background=self.COLOR_SURFACE, foreground=self.COLOR_TEXT)
        style.configure(
            "SubTitle.TLabel",
            background=self.COLOR_SURFACE,
            foreground=self.COLOR_MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "SectionTitle.TLabel",
            background=self.COLOR_CARD,
            foreground=self.COLOR_TEXT,
            font=("Segoe UI Semibold", 10),
        )
        style.configure(
            "Field.TLabel",
            background=self.COLOR_CARD,
            foreground=self.COLOR_MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Hint.TLabel",
            background=self.COLOR_CARD,
            foreground=self.COLOR_MUTED,
            font=("Segoe UI", 9),
        )

        style.configure(
            "Form.TEntry",
            fieldbackground="#0f1a32",
            foreground=self.COLOR_TEXT,
            bordercolor=self.COLOR_BORDER,
            lightcolor=self.COLOR_BORDER,
            darkcolor=self.COLOR_BORDER,
            padding=(7, 6),
            relief="flat",
        )
        style.map(
            "Form.TEntry",
            bordercolor=[("focus", self.COLOR_ACCENT)],
            lightcolor=[("focus", self.COLOR_ACCENT)],
            darkcolor=[("focus", self.COLOR_ACCENT)],
        )

        style.configure(
            "Primary.TButton",
            background=self.COLOR_ACCENT,
            foreground=self.COLOR_TEXT,
            borderwidth=0,
            focusthickness=0,
            padding=(12, 9),
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#3f7de3"), ("pressed", "#3367b7")],
        )

        style.configure(
            "Success.TButton",
            background=self.COLOR_SUCCESS,
            foreground="#08121e",
            borderwidth=0,
            focusthickness=0,
            padding=(12, 9),
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Success.TButton",
            background=[("active", "#19aa8d"), ("pressed", "#158a73")],
        )

        style.configure(
            "Neutral.TButton",
            background="#273a5f",
            foreground=self.COLOR_TEXT,
            borderwidth=0,
            focusthickness=0,
            padding=(12, 9),
            font=("Segoe UI", 10),
        )
        style.map(
            "Neutral.TButton",
            background=[("active", "#2f466f"), ("pressed", "#243655")],
        )

        style.configure(
            "Danger.TButton",
            background=self.COLOR_DANGER,
            foreground="#2a0f14",
            borderwidth=0,
            focusthickness=0,
            padding=(12, 9),
            font=("Segoe UI Semibold", 10),
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#eb6270"), ("pressed", "#cd535f")],
        )

    def _build_ui(self) -> None:
        container = ttk.Frame(self, style="Root.TFrame", padding=(20, 18, 20, 18))
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        header = ttk.Frame(container, style="Surface.TFrame", padding=(20, 16, 20, 16))
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="Thujee's Room sentinal",
            style="Title.TLabel",
            font=("Bahnschrift SemiBold", 24),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Premium control room for face-body identity, voice defense, and email intelligence.",
            style="SubTitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))

        self.status_chip = tk.Label(
            header,
            textvariable=self.status_var,
            bg="#203555",
            fg="#d9e8ff",
            font=("Segoe UI Semibold", 10),
            padx=12,
            pady=6,
            relief=tk.FLAT,
        )
        self.status_chip.grid(row=0, column=1, rowspan=2, sticky="e")

        content = ttk.Frame(container, style="Root.TFrame")
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(1, weight=1)

        settings_card = ttk.LabelFrame(
            content,
            text="Configuration",
            style="Card.TLabelframe",
            padding=(18, 16, 18, 16),
        )
        settings_card.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 12))
        settings_card.columnconfigure(1, weight=1)
        settings_card.columnconfigure(3, weight=1)

        ttk.Label(settings_card, text="Identity", style="SectionTitle.TLabel").grid(
            row=0, column=0, columnspan=4, sticky="w"
        )
        ttk.Label(settings_card, text="Owner Name", style="Field.TLabel").grid(
            row=1, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Entry(
            settings_card,
            textvariable=self.owner_name_var,
            style="Form.TEntry",
            width=30,
        ).grid(row=1, column=1, sticky="ew", padx=(8, 14), pady=(10, 0))

        ttk.Label(settings_card, text="Camera Index", style="Field.TLabel").grid(
            row=1, column=2, sticky="w", pady=(10, 0)
        )
        ttk.Entry(
            settings_card,
            textvariable=self.camera_index_var,
            style="Form.TEntry",
            width=10,
        ).grid(row=1, column=3, sticky="ew", padx=(8, 0), pady=(10, 0))

        ttk.Separator(settings_card).grid(row=2, column=0, columnspan=4, sticky="ew", pady=14)

        ttk.Label(settings_card, text="Email Alert Pipeline", style="SectionTitle.TLabel").grid(
            row=3, column=0, columnspan=4, sticky="w"
        )
        ttk.Label(settings_card, text="Sender Email", style="Field.TLabel").grid(
            row=4, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Entry(
            settings_card,
            textvariable=self.sender_email_var,
            style="Form.TEntry",
            width=30,
        ).grid(row=4, column=1, sticky="ew", padx=(8, 14), pady=(10, 0))

        ttk.Label(settings_card, text="16-char App Password", style="Field.TLabel").grid(
            row=4, column=2, sticky="w", pady=(10, 0)
        )
        ttk.Entry(
            settings_card,
            textvariable=self.sender_password_var,
            style="Form.TEntry",
            width=20,
            show="*",
        ).grid(row=4, column=3, sticky="ew", padx=(8, 0), pady=(10, 0))

        ttk.Label(settings_card, text="Receiver Email", style="Field.TLabel").grid(
            row=5, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Entry(
            settings_card,
            textvariable=self.receiver_email_var,
            style="Form.TEntry",
            width=30,
        ).grid(row=5, column=1, sticky="ew", padx=(8, 14), pady=(10, 0))

        ttk.Separator(settings_card).grid(row=6, column=0, columnspan=4, sticky="ew", pady=14)

        ttk.Label(settings_card, text="Recognition", style="SectionTitle.TLabel").grid(
            row=7, column=0, columnspan=4, sticky="w"
        )
        ttk.Label(settings_card, text="Face Threshold", style="Field.TLabel").grid(
            row=8, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Entry(
            settings_card,
            textvariable=self.face_threshold_var,
            style="Form.TEntry",
            width=10,
        ).grid(row=8, column=1, sticky="ew", padx=(8, 14), pady=(10, 0))

        ttk.Label(settings_card, text="Body Threshold", style="Field.TLabel").grid(
            row=8, column=2, sticky="w", pady=(10, 0)
        )
        ttk.Entry(
            settings_card,
            textvariable=self.body_threshold_var,
            style="Form.TEntry",
            width=10,
        ).grid(row=8, column=3, sticky="ew", padx=(8, 0), pady=(10, 0))

        ttk.Separator(settings_card).grid(row=9, column=0, columnspan=4, sticky="ew", pady=14)

        ttk.Label(settings_card, text="Voice Messages", style="SectionTitle.TLabel").grid(
            row=10, column=0, columnspan=4, sticky="w"
        )
        ttk.Label(settings_card, text="Owner Greeting", style="Field.TLabel").grid(
            row=11, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Entry(
            settings_card,
            textvariable=self.greeting_text_var,
            style="Form.TEntry",
            width=30,
        ).grid(row=11, column=1, sticky="ew", padx=(8, 14), pady=(10, 0))

        ttk.Label(settings_card, text="Intruder Warning", style="Field.TLabel").grid(
            row=12, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Entry(
            settings_card,
            textvariable=self.warning_text_var,
            style="Form.TEntry",
            width=70,
        ).grid(
            row=12,
            column=1,
            columnspan=3,
            sticky="ew",
            padx=(8, 0),
            pady=(10, 0),
        )

        ttk.Label(
            settings_card,
            text="Tip: keep defaults first, then tune thresholds after real tests.",
            style="Hint.TLabel",
        ).grid(row=13, column=0, columnspan=4, sticky="w", pady=(12, 0))

        action_card = ttk.LabelFrame(
            content,
            text="Operations",
            style="Card.TLabelframe",
            padding=(16, 14, 16, 14),
        )
        action_card.grid(row=0, column=1, sticky="nsew")
        action_card.columnconfigure(0, weight=1)
        action_card.columnconfigure(1, weight=1)

        ttk.Button(
            action_card,
            text="Save Settings",
            style="Neutral.TButton",
            command=self._save_settings,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=(0, 8))

        ttk.Button(
            action_card,
            text="Enroll Owner",
            style="Primary.TButton",
            command=self._enroll_owner,
        ).grid(row=0, column=1, sticky="ew", pady=(0, 8))

        ttk.Button(
            action_card,
            text="Start Monitoring",
            style="Success.TButton",
            command=self._start_monitoring,
        ).grid(row=1, column=0, sticky="ew", padx=(0, 8))

        ttk.Button(
            action_card,
            text="Stop Monitoring",
            style="Danger.TButton",
            command=self._stop_monitoring,
        ).grid(row=1, column=1, sticky="ew")

        ttk.Label(
            action_card,
            text="Launch the monitor and watch live status in the stream below.",
            style="Hint.TLabel",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))

        ttk.Separator(action_card).grid(row=3, column=0, columnspan=2, sticky="ew", pady=12)

        ttk.Label(action_card, text="Single Owner Profile", style="SectionTitle.TLabel").grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="w",
        )
        ttk.Label(
            action_card,
            textvariable=self.owner_profile_var,
            style="Hint.TLabel",
            justify=tk.LEFT,
            wraplength=350,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 10))

        ttk.Button(
            action_card,
            text="Refresh Owner Data",
            style="Neutral.TButton",
            command=self._refresh_owner_profile_view,
        ).grid(row=6, column=0, sticky="ew", padx=(0, 8))

        ttk.Button(
            action_card,
            text="Delete Owner Data",
            style="Danger.TButton",
            command=self._delete_owner_profile,
        ).grid(row=6, column=1, sticky="ew")

        log_card = ttk.LabelFrame(
            content,
            text="Activity Stream",
            style="Card.TLabelframe",
            padding=(12, 10, 12, 12),
        )
        log_card.grid(row=1, column=1, sticky="nsew", pady=(12, 0))
        log_card.columnconfigure(0, weight=1)
        log_card.rowconfigure(0, weight=1)

        self.log_box = tk.Text(
            log_card,
            height=18,
            state=tk.DISABLED,
            wrap=tk.WORD,
            bg="#0b162d",
            fg="#d8e6ff",
            insertbackground="#d8e6ff",
            selectbackground="#355488",
            relief=tk.FLAT,
            borderwidth=0,
            padx=10,
            pady=10,
            font=("Cascadia Mono", 10),
        )
        self.log_box.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(log_card, orient=tk.VERTICAL, command=self.log_box.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_box["yscrollcommand"] = scrollbar.set

    def _load_initial_config(self) -> None:
        cfg = load_config()
        self.owner_name_var.set(cfg.owner_name)
        self.camera_index_var.set(str(cfg.camera_index))
        self.sender_email_var.set(cfg.sender_email)
        self.sender_password_var.set(cfg.sender_app_password)
        self.receiver_email_var.set(cfg.receiver_email)
        self.face_threshold_var.set(str(cfg.face_match_threshold))
        self.body_threshold_var.set(str(cfg.body_match_threshold))
        self.greeting_text_var.set(cfg.owner_greeting_text)
        self.warning_text_var.set(cfg.warning_text)

    def _set_status(self, text: str, tone: str = "neutral") -> None:
        palette = {
            "neutral": ("#213a5f", "#d9e8ff"),
            "ok": ("#1a5448", "#b8ffe9"),
            "warn": ("#5f4621", "#ffe9c5"),
            "alert": ("#5a2330", "#ffd2da"),
        }
        bg, fg = palette.get(tone, palette["neutral"])
        self.status_var.set(text)
        self.status_chip.configure(bg=bg, fg=fg)

    def _queue_status(self, text: str, tone: str = "neutral") -> None:
        self._status_queue.put((text, tone))

    def _build_config_from_form(self) -> AppConfig:
        owner_name = self.owner_name_var.get().strip() or "Owner"

        try:
            camera_index = int(self.camera_index_var.get().strip())
        except ValueError:
            raise ValueError("Camera Index must be an integer like 0 or 1.")

        try:
            face_threshold = float(self.face_threshold_var.get().strip())
            body_threshold = float(self.body_threshold_var.get().strip())
        except ValueError:
            raise ValueError("Face and Body Threshold values must be valid numbers.")

        sender_password = self.sender_password_var.get().replace(" ", "").strip()
        greeting = self.greeting_text_var.get().strip() or f"Welcome back {owner_name}"

        warning = self.warning_text_var.get().strip()
        if not warning:
            warning = (
                "This is restricted area. If you come in or do any unauthorized things, "
                "you will face consequences."
            )

        return AppConfig(
            owner_name=owner_name,
            camera_index=camera_index,
            sender_email=self.sender_email_var.get().strip(),
            sender_app_password=sender_password,
            receiver_email=self.receiver_email_var.get().strip(),
            face_match_threshold=face_threshold,
            body_match_threshold=body_threshold,
            owner_greeting_text=greeting,
            warning_text=warning,
        )

    def _enqueue_log(self, text: str) -> None:
        self._log_queue.put(text)

    def _refresh_owner_profile_view(self) -> None:
        if not owner_profile_exists():
            self.owner_profile_var.set(
                "No owner enrolled.\n"
                "This app supports one owner only. Enroll one owner to create data."
            )
            return

        try:
            profile = load_owner_profile()
        except Exception as exc:
            self.owner_profile_var.set(f"Owner data could not be read: {exc}")
            return

        face_count = int(profile.face_encodings.shape[0])
        body_vector_size = int(profile.body_signature.shape[0])
        self.owner_profile_var.set(
            f"Owner Name: {profile.owner_name}\n"
            f"Created At: {profile.created_at}\n"
            f"Face Samples: {face_count}\n"
            f"Body Signature Size: {body_vector_size}"
        )

    def _delete_owner_profile(self) -> None:
        if self.monitor and self.monitor.is_running:
            messagebox.showwarning("Monitoring Active", "Stop monitoring before deleting owner data.")
            self._set_status("Stop monitoring first", tone="warn")
            return

        if not owner_profile_exists():
            messagebox.showinfo("No Owner Data", "Owner data was not found.")
            self._set_status("Owner profile missing", tone="warn")
            self._refresh_owner_profile_view()
            return

        confirmed = messagebox.askyesno(
            "Delete Owner Data",
            "This system supports one owner only.\n\n"
            "Delete current owner name and biometric data?",
        )
        if not confirmed:
            return

        try:
            deleted = delete_owner_profile()
        except Exception as exc:
            messagebox.showerror("Delete Failed", f"Could not delete owner data: {exc}")
            self._enqueue_log(f"Delete owner data failed: {exc}")
            self._set_status("Delete failed", tone="alert")
            return

        if deleted:
            self._enqueue_log("Owner profile deleted.")
            self._set_status("Owner profile deleted", tone="ok")
        else:
            self._enqueue_log("Owner profile already missing.")
            self._set_status("Owner profile missing", tone="warn")

        self._refresh_owner_profile_view()

    def _flush_logs(self) -> None:
        while not self._status_queue.empty():
            text, tone = self._status_queue.get_nowait()
            self._set_status(text, tone)

        if self._pending_profile_refresh:
            self._pending_profile_refresh = False
            self._refresh_owner_profile_view()

        while not self._log_queue.empty():
            line = self._log_queue.get_nowait()
            self.log_box.config(state=tk.NORMAL)
            self.log_box.insert(tk.END, line + "\n")
            self.log_box.see(tk.END)
            self.log_box.config(state=tk.DISABLED)
        self.after(150, self._flush_logs)

    def _save_settings(self) -> None:
        try:
            cfg = self._build_config_from_form()
        except ValueError as exc:
            messagebox.showerror("Invalid Settings", str(exc))
            self._set_status("Invalid settings", tone="alert")
            return

        save_config(cfg)
        self._enqueue_log("Settings saved.")
        self._set_status("Settings saved", tone="ok")
        if cfg.sender_app_password and len(cfg.sender_app_password) != 16:
            self._enqueue_log("Warning: app password should usually be 16 characters.")
            self._set_status("Password length warning", tone="warn")

    def _enroll_owner(self) -> None:
        if self.monitor and self.monitor.is_running:
            messagebox.showwarning("Monitoring Active", "Stop monitoring before enrollment.")
            self._set_status("Stop monitoring first", tone="warn")
            return

        if owner_profile_exists():
            messagebox.showwarning(
                "Single Owner Mode",
                "Only one owner profile is supported.\n\n"
                "Delete current owner data from Operations -> Single Owner Profile, then enroll again.",
            )
            self._set_status("Delete existing owner first", tone="warn")
            return

        try:
            cfg = self._build_config_from_form()
        except ValueError as exc:
            messagebox.showerror("Invalid Settings", str(exc))
            self._set_status("Invalid settings", tone="alert")
            return

        save_config(cfg)
        self._enqueue_log("Starting owner enrollment...")
        self._set_status("Enrollment in progress", tone="warn")

        def task() -> None:
            try:
                enroll_owner(
                    owner_name=cfg.owner_name,
                    camera_index=cfg.camera_index,
                    logger=self._enqueue_log,
                )
                self._enqueue_log("Owner enrollment successful.")
                self._pending_profile_refresh = True
                self._queue_status("Owner profile ready", tone="ok")
            except Exception as exc:
                self._enqueue_log(f"Enrollment failed: {exc}")
                self._queue_status("Enrollment failed", tone="alert")

        threading.Thread(target=task, daemon=True).start()

    def _start_monitoring(self) -> None:
        if self.monitor and self.monitor.is_running:
            self._enqueue_log("Monitoring already running.")
            self._set_status("Monitoring already running", tone="warn")
            return

        try:
            cfg = self._build_config_from_form()
        except ValueError as exc:
            messagebox.showerror("Invalid Settings", str(exc))
            self._set_status("Invalid settings", tone="alert")
            return

        if not owner_profile_exists():
            messagebox.showerror(
                "No Owner Profile",
                "Owner profile not found. Please run Enroll Owner first.",
            )
            self._set_status("Owner profile missing", tone="alert")
            return

        save_config(cfg)

        try:
            profile = load_owner_profile()
        except Exception as exc:
            messagebox.showerror("Profile Error", f"Failed to load owner profile: {exc}")
            self._set_status("Profile load failed", tone="alert")
            return

        self.monitor = SecurityMonitor(
            config=cfg,
            owner_profile=profile,
            logger=self._enqueue_log,
        )
        self.monitor.start()
        self._enqueue_log("Monitoring thread launched.")
        self._set_status("Monitoring live", tone="ok")

    def _stop_monitoring(self) -> None:
        if not self.monitor or not self.monitor.is_running:
            self._enqueue_log("Monitoring is not running.")
            self._set_status("Monitoring already stopped", tone="warn")
            return

        self.monitor.stop()
        self._enqueue_log("Stop signal sent to monitoring.")
        self._set_status("Monitoring paused", tone="neutral")

    def _on_close(self) -> None:
        if self.monitor and self.monitor.is_running:
            self.monitor.stop()
        self.destroy()


def main() -> None:
    app = SecurityApp()
    app.mainloop()


if __name__ == "__main__":
    main()
