#!/usr/bin/env python3
# ai_violin_master_08_realtime_monitor.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683


import customtkinter as ctk
from tkinter import messagebox
import threading
import queue
import time
import numpy as np
import os
import configparser

# Color constants
COLOR_BG = "#5A381E"
COLOR_DARK = "#3B2413"
COLOR_GOLD = "#FFD770"
COLOR_FRAME = "#4B2E18"
COLOR_TEXT = "#FFF2CF"
COLOR_RED = "#FF5555"
COLOR_GREEN = "#55FF55"
COLOR_ORANGE = "#FFAA00"
COLOR_YELLOW = "#FFFF00"
COLOR_MEDIUM = "#4B2E18"

WINDOW_SIZE = "640x1050"


class RealtimeMonitor:
    """Real-time violin performance analyzer."""

    def __init__(self, analyzer=None, sample_rate=44100, frame_size=2048, hop_size=512):
        self.analyzer = analyzer
        self.sr = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.stream = None
        self.running = False
        self.pitch_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.kalman_pitch = None
        self.kalman_P = 1.0
        self.vibrato_window = []
        self.max_vibrato_frames = 40
        self.last_onset_time = None
        self.onset_times = []
        self.callback = None
        self.midi_onsets = None
        self.midi_index = 0

        import sounddevice as sd
        self.sd = sd
        self.sounddevice_available = True

    def midi_to_note(self, midi):
        if midi is None:
            return "--"
        note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        return f"{note_names[midi % 12]}{midi//12 - 1}"

    def kalman_update(self, measurement, R=8.0, Q=0.01):
        if self.kalman_pitch is None:
            self.kalman_pitch = measurement
            self.kalman_P = 1.0
            return measurement
        pred = self.kalman_pitch
        P = self.kalman_P + Q
        K = P / (P + R)
        self.kalman_pitch = pred + K * (measurement - pred)
        self.kalman_P = (1 - K) * P
        return self.kalman_pitch

    def yin_pitch(self, frame):
        x = frame.astype(np.float32)
        N = len(x)
        diff = np.zeros(N//2)
        for tau in range(1, N//2):
            diff[tau] = np.sum((x[:-tau] - x[tau:])**2)
        cmnd = np.copy(diff)
        for i in range(1, len(cmnd)):
            cmnd[i] *= i / sum(diff[1:i+1])
        threshold = 0.1
        candidates = np.where(cmnd < threshold)[0]
        if len(candidates) == 0:
            return None
        tau = candidates[0]
        if tau == 0:
            return None
        return self.sr / tau

    def estimate_vibrato(self):
        if len(self.vibrato_window) < 10:
            return None
        times = np.linspace(0, len(self.vibrato_window) / (self.sr / self.hop_size),
                            len(self.vibrato_window))
        pitches = np.array(self.vibrato_window)
        detrend = pitches - np.mean(pitches)
        fft = np.fft.rfft(detrend)
        freqs = np.fft.rfftfreq(len(detrend), d=self.hop_size / self.sr)
        idx = np.argmax(np.abs(fft[1:])) + 1
        vibrato_rate = freqs[idx]
        if np.max(pitches) == 0:
            return None
        depth_cents = (np.max(pitches) - np.min(pitches)) * 100
        return {"rate_hz": float(vibrato_rate), "depth_cents": float(depth_cents)}

    def bow_noise_level(self, frame):
        spectrum = np.abs(np.fft.rfft(frame))
        total = np.sum(spectrum)
        if total <= 0:
            return 0.0
        high = np.sum(spectrum[int(len(spectrum)*0.6):])
        return float(high / total)

    def detect_onset(self, frame):
        energy = np.sum(frame**2)
        threshold = 0.02
        return energy > threshold

    def load_midi_reference(self, pm):
        onsets = []
        for inst in pm.instruments:
            onsets.extend([n.start for n in inst.notes])
        self.midi_onsets = sorted(onsets)
        self.midi_index = 0

    def timing_deviation(self):
        if self.midi_onsets is None:
            return None
        if self.midi_index >= len(self.midi_onsets):
            return None
        live_time = time.time()
        expected = self.midi_onsets[self.midi_index]
        deviation = live_time - expected
        self.midi_index += 1
        return deviation

    def _audio_callback(self, indata, frames, time_info, status):
        if status or not self.running:
            return
        frame = indata[:, 0].copy()
        self.audio_queue.put(frame)
        pitch = self.yin_pitch(frame)
        if pitch:
            smooth_pitch = self.kalman_update(pitch)
            self.pitch_queue.put(smooth_pitch)
        if pitch:
            self.vibrato_window.append(pitch)
            if len(self.vibrato_window) > self.max_vibrato_frames:
                self.vibrato_window.pop(0)

    def start(self, callback=None):
        if not self.sounddevice_available:
            raise ImportError("sounddevice not available.")
        self.callback = callback
        self.running = True
        self.stream = self.sd.InputStream(
            channels=1,
            samplerate=self.sr,
            blocksize=self.hop_size,
            callback=self._audio_callback
        )
        self.stream.start()
        threading.Thread(target=self._poll_thread, daemon=True).start()

    def stop(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.stream = None

    def _poll_thread(self):
        last_vibrato = None
        while self.running:
            try:
                frame = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            bow_noise = self.bow_noise_level(frame)
            pitch = None
            if not self.pitch_queue.empty():
                pitch = self.pitch_queue.get()
            note_name = "--"
            intonation = None
            if pitch:
                midi = int(round(69 + 12 * np.log2(pitch / 440.0)))
                note_name = self.midi_to_note(midi)
                ideal = 440 * 2**((midi - 69)/12)
                intonation = 1200 * np.log2(pitch / ideal)
            vib = self.estimate_vibrato()
            if vib:
                last_vibrato = vib
            vibrato_rate = last_vibrato["rate_hz"] if last_vibrato else None
            vibrato_depth = last_vibrato["depth_cents"] if last_vibrato else None
            tdev = None
            if self.detect_onset(frame):
                tdev = self.timing_deviation()
            data = {
                "pitch": pitch,
                "note": note_name,
                "intonation": intonation,
                "vibrato_rate": vibrato_rate,
                "vibrato_depth": vibrato_depth,
                "bow_noise": bow_noise,
                "timing_deviation": tdev
            }
            if self.callback:
                self.callback(data)


class RealtimeMonitorGUI:
    """GUI for Realtime Monitor using CustomTkinter."""

    def __init__(self, parent=None):
        self.ini_file = "ai_violin_master_08_realtime_monitor.ini"
        self.config = configparser.ConfigParser()

        if parent is None:
            self.root = ctk.CTk()
            self.is_standalone = True
            self.root.title("Realtime Monitor")
            self.root.geometry(WINDOW_SIZE)
            self.root.resizable(False, False)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        else:
            self.root = parent
            self.is_standalone = False

        self.monitor = RealtimeMonitor()
        self.monitoring = False

        self.current_data = {
            "pitch": None,
            "note": "--",
            "intonation": 0.0,
            "vibrato_rate": 0.0,
            "vibrato_depth": 0.0,
            "bow_noise": 0.0,
            "timing_deviation": 0.0
        }

        self._load_state()
        ctk.set_appearance_mode("dark")
        self.root.configure(fg_color=COLOR_BG)

        self._create_widgets()
        self._update_display()

        if self.is_standalone:
            self._restore_position()
            self.root.mainloop()

    def _load_state(self):
        if os.path.exists(self.ini_file):
            self.config.read(self.ini_file)
        if not self.config.has_section("window"):
            self.config.add_section("window")

    def _save_state(self):
        self.config.set("window", "x", str(self.root.winfo_x()))
        self.config.set("window", "y", str(self.root.winfo_y()))
        with open(self.ini_file, "w") as f:
            self.config.write(f)

    def _restore_position(self):
        x = self.config.getint("window", "x", fallback=100)
        y = self.config.getint("window", "y", fallback=100)
        self.root.geometry(f"+{x}+{y}")

    def _create_widgets(self):
        # Main container
        main_frame = ctk.CTkFrame(self.root, fg_color=COLOR_BG)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        header_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            header_frame,
            text="REALTIME PERFORMANCE MONITOR",
            font=("Georgia", 16, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            header_frame,
            text="Live analysis of pitch, intonation, vibrato, and tone quality",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=(0, 10))

        # Controls
        control_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        control_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            control_frame,
            text="MONITORING CONTROLS",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        btn_row = ctk.CTkFrame(control_frame, fg_color=COLOR_DARK)
        btn_row.pack(fill="x", padx=20, pady=10)

        self.start_stop_btn = ctk.CTkButton(
            btn_row,
            text="START MONITORING",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_FRAME,
            command=self._toggle_monitoring,
        )
        self.start_stop_btn.pack(side="left", padx=5)

        self.status_indicator = ctk.CTkLabel(
            btn_row,
            text="STOPPED",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_RED,
            fg_color=COLOR_DARK,
        )
        self.status_indicator.pack(side="left", padx=20)

        # Metrics display
        metrics_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        metrics_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            metrics_frame,
            text="LIVE METRICS",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        # Metrics grid
        grid_frame = ctk.CTkFrame(metrics_frame, fg_color=COLOR_DARK)
        grid_frame.pack(fill="x", padx=20, pady=10)

        self.metric_labels = {}

        metrics = [
            ("Current Note", "note", "--"),
            ("Pitch (Hz)", "pitch", "--"),
            ("Intonation", "intonation", "--"),
            ("Vibrato Rate", "vibrato_rate", "--"),
            ("Vibrato Depth", "vibrato_depth", "--"),
            ("Bow Noise", "bow_noise", "--"),
        ]

        for i, (label, key, default) in enumerate(metrics):
            row = i // 2
            col = i % 2

            item_frame = ctk.CTkFrame(grid_frame, fg_color=COLOR_FRAME, border_width=2, border_color=COLOR_GOLD)
            item_frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
            grid_frame.columnconfigure(col, weight=1)

            ctk.CTkLabel(
                item_frame,
                text=label,
                font=("Georgia", 12),
                text_color=COLOR_TEXT,
                fg_color=COLOR_FRAME,
            ).pack(pady=(5, 0))

            value_label = ctk.CTkLabel(
                item_frame,
                text=default,
                font=("Courier New", 14, "bold"),
                text_color=COLOR_GOLD,
                fg_color=COLOR_FRAME,
            )
            value_label.pack(pady=(0, 5))

            self.metric_labels[key] = value_label

        # Status bar
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Ready",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            height=30,
        )
        self.status_label.pack(fill="x", pady=(10, 0))

    def _update_display(self):
        self.metric_labels["note"].configure(text=self.current_data["note"])

        if self.current_data["pitch"]:
            self.metric_labels["pitch"].configure(text=f"{self.current_data['pitch']:.1f}")
        else:
            self.metric_labels["pitch"].configure(text="--")

        if self.current_data["intonation"] is not None:
            self.metric_labels["intonation"].configure(text=f"{self.current_data['intonation']:+.1f}c")
        else:
            self.metric_labels["intonation"].configure(text="--")

        if self.current_data["vibrato_rate"]:
            self.metric_labels["vibrato_rate"].configure(text=f"{self.current_data['vibrato_rate']:.1f} Hz")
        else:
            self.metric_labels["vibrato_rate"].configure(text="--")

        if self.current_data["vibrato_depth"]:
            self.metric_labels["vibrato_depth"].configure(text=f"{self.current_data['vibrato_depth']:.0f}c")
        else:
            self.metric_labels["vibrato_depth"].configure(text="--")

        self.metric_labels["bow_noise"].configure(text=f"{self.current_data['bow_noise']:.3f}")

        self.root.after(100, self._update_display)

    def _toggle_monitoring(self):
        if not self.monitoring:
            self.monitor.start(callback=self._on_monitor_data)
            self.monitoring = True
            self.start_stop_btn.configure(text="STOP MONITORING")
            self.status_indicator.configure(text="ACTIVE", text_color=COLOR_GREEN)
            self.status_label.configure(text="Live monitoring")
        else:
            self.monitor.stop()
            self.monitoring = False
            self.start_stop_btn.configure(text="START MONITORING")
            self.status_indicator.configure(text="STOPPED", text_color=COLOR_RED)
            self.status_label.configure(text="Monitoring stopped")

    def _on_monitor_data(self, data):
        self.current_data.update(data)
        if self.current_data["intonation"] is None:
            self.current_data["intonation"] = 0.0
        if self.current_data["vibrato_rate"] is None:
            self.current_data["vibrato_rate"] = 0.0
        if self.current_data["vibrato_depth"] is None:
            self.current_data["vibrato_depth"] = 0.0
        if self.current_data["bow_noise"] is None:
            self.current_data["bow_noise"] = 0.0

    def _on_close(self):
        if self.monitoring:
            self.monitor.stop()
        self._save_state()
        self.root.destroy()


def get_content():
    """Return module description for display in GUI."""
    return [
        "REALTIME MONITOR MODULE",
        "",
        "PURPOSE: Live microphone monitoring with instant analysis",
        "",
        "LIVE METRICS:",
        "  - Current note and pitch (Hz)",
        "  - Intonation deviation in cents",
        "  - Vibrato rate (Hz) and depth (cents)",
        "  - Bow noise level (clean vs scratchy)",
        "  - Timing deviation vs reference",
        "",
        "VISUAL DISPLAYS:",
        "  - Intonation meter with colored zones",
        "  - Bow noise level meter",
        "  - Pitch history graph",
        "  - Real-time status indicators",
        "",
        "TECHNICAL FEATURES:",
        "  - Kalman-filter stabilized pitch tracking",
        "  - Real-time FFT analysis",
        "  - Thread-safe audio processing",
        "",
        "DEPENDENCIES:",
        "  - Requires: pip install sounddevice",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = RealtimeMonitorGUI(parent)
    return gui.root


def main():
    """Standalone entry - launches a full window with mainloop."""
    RealtimeMonitorGUI()


if __name__ == "__main__":
    main()