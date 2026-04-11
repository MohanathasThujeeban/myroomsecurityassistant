from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from security_system import load_config, load_owner_profile, owner_profile_exists, save_config
from security_system.config import AppConfig
from security_system.enrollment import enroll_owner
from security_system.monitor import SecurityMonitor


class SecurityApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Smart Room Alert System")
        self.geometry("860x640")
        self.minsize(760, 560)

        self.monitor = None
        self._log_queue: queue.Queue[str] = queue.Queue()

        self.owner_name_var = tk.StringVar()
        self.camera_index_var = tk.StringVar()
        self.sender_email_var = tk.StringVar()
        self.sender_password_var = tk.StringVar()
        self.receiver_email_var = tk.StringVar()
        self.face_threshold_var = tk.StringVar()
        self.body_threshold_var = tk.StringVar()
        self.greeting_text_var = tk.StringVar()
        self.warning_text_var = tk.StringVar()

        self._build_ui()
        self._load_initial_config()
        self.after(150, self._flush_logs)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=14)
        container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            container,
            text="Owner Recognition + Intruder Alert",
            font=("Segoe UI", 16, "bold"),
        )
        title.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))

        ttk.Label(container, text="Owner Name").grid(row=1, column=0, sticky="w")
        ttk.Entry(container, textvariable=self.owner_name_var, width=28).grid(
            row=1, column=1, sticky="ew", padx=(6, 14)
        )

        ttk.Label(container, text="Camera Index").grid(row=1, column=2, sticky="w")
        ttk.Entry(container, textvariable=self.camera_index_var, width=10).grid(
            row=1, column=3, sticky="ew", padx=(6, 0)
        )

        ttk.Label(container, text="Sender Email").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(container, textvariable=self.sender_email_var, width=28).grid(
            row=2, column=1, sticky="ew", padx=(6, 14), pady=(8, 0)
        )

        ttk.Label(container, text="16-digit App Password").grid(
            row=2, column=2, sticky="w", pady=(8, 0)
        )
        ttk.Entry(
            container,
            textvariable=self.sender_password_var,
            width=20,
            show="*",
        ).grid(row=2, column=3, sticky="ew", padx=(6, 0), pady=(8, 0))

        ttk.Label(container, text="Receiver Email").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(container, textvariable=self.receiver_email_var, width=28).grid(
            row=3, column=1, sticky="ew", padx=(6, 14), pady=(8, 0)
        )

        ttk.Label(container, text="Face Threshold").grid(row=3, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(container, textvariable=self.face_threshold_var, width=10).grid(
            row=3, column=3, sticky="ew", padx=(6, 0), pady=(8, 0)
        )

        ttk.Label(container, text="Body Threshold").grid(row=4, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(container, textvariable=self.body_threshold_var, width=10).grid(
            row=4, column=3, sticky="ew", padx=(6, 0), pady=(8, 0)
        )

        ttk.Label(container, text="Owner Greeting Voice").grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(container, textvariable=self.greeting_text_var, width=28).grid(
            row=4, column=1, sticky="ew", padx=(6, 14), pady=(8, 0)
        )

        ttk.Label(container, text="Intruder Warning Voice").grid(
            row=5, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(container, textvariable=self.warning_text_var, width=80).grid(
            row=5, column=1, columnspan=3, sticky="ew", padx=(6, 0), pady=(8, 0)
        )

        button_frame = ttk.Frame(container)
        button_frame.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(14, 8))

        ttk.Button(button_frame, text="Save Settings", command=self._save_settings).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(button_frame, text="Enroll Owner", command=self._enroll_owner).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(button_frame, text="Start Monitoring", command=self._start_monitoring).pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Button(button_frame, text="Stop Monitoring", command=self._stop_monitoring).pack(
            side=tk.LEFT
        )

        ttk.Label(container, text="Activity Log").grid(row=7, column=0, sticky="w", pady=(8, 4))
        self.log_box = tk.Text(container, height=18, state=tk.DISABLED, wrap=tk.WORD)
        self.log_box.grid(row=8, column=0, columnspan=4, sticky="nsew")

        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.log_box.yview)
        scrollbar.grid(row=8, column=4, sticky="ns")
        self.log_box["yscrollcommand"] = scrollbar.set

        container.columnconfigure(1, weight=1)
        container.columnconfigure(3, weight=1)
        container.rowconfigure(8, weight=1)

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

    def _flush_logs(self) -> None:
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
            return

        save_config(cfg)
        self._enqueue_log("Settings saved.")
        if cfg.sender_app_password and len(cfg.sender_app_password) != 16:
            self._enqueue_log("Warning: app password should usually be 16 characters.")

    def _enroll_owner(self) -> None:
        if self.monitor and self.monitor.is_running:
            messagebox.showwarning("Monitoring Active", "Stop monitoring before enrollment.")
            return

        try:
            cfg = self._build_config_from_form()
        except ValueError as exc:
            messagebox.showerror("Invalid Settings", str(exc))
            return

        save_config(cfg)
        self._enqueue_log("Starting owner enrollment...")

        def task() -> None:
            try:
                enroll_owner(
                    owner_name=cfg.owner_name,
                    camera_index=cfg.camera_index,
                    logger=self._enqueue_log,
                )
                self._enqueue_log("Owner enrollment successful.")
            except Exception as exc:
                self._enqueue_log(f"Enrollment failed: {exc}")

        threading.Thread(target=task, daemon=True).start()

    def _start_monitoring(self) -> None:
        if self.monitor and self.monitor.is_running:
            self._enqueue_log("Monitoring already running.")
            return

        try:
            cfg = self._build_config_from_form()
        except ValueError as exc:
            messagebox.showerror("Invalid Settings", str(exc))
            return

        if not owner_profile_exists():
            messagebox.showerror(
                "No Owner Profile",
                "Owner profile not found. Please run Enroll Owner first.",
            )
            return

        save_config(cfg)

        try:
            profile = load_owner_profile()
        except Exception as exc:
            messagebox.showerror("Profile Error", f"Failed to load owner profile: {exc}")
            return

        self.monitor = SecurityMonitor(
            config=cfg,
            owner_profile=profile,
            logger=self._enqueue_log,
        )
        self.monitor.start()
        self._enqueue_log("Monitoring thread launched.")

    def _stop_monitoring(self) -> None:
        if not self.monitor or not self.monitor.is_running:
            self._enqueue_log("Monitoring is not running.")
            return

        self.monitor.stop()
        self._enqueue_log("Stop signal sent to monitoring.")

    def _on_close(self) -> None:
        if self.monitor and self.monitor.is_running:
            self.monitor.stop()
        self.destroy()


def main() -> None:
    app = SecurityApp()
    app.mainloop()


if __name__ == "__main__":
    main()
