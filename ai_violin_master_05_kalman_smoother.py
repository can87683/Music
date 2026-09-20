#!/usr/bin/env python3
# ai_violin_master_05_kalman_smoother.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683


import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
import json
import os
from datetime import datetime
import configparser
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Color constants
COLOR_BG = "#5A381E"
COLOR_DARK = "#3B2413"
COLOR_GOLD = "#FFD770"
COLOR_FRAME = "#4B2E18"
COLOR_MEDIUM = "#4B2E18"
COLOR_TEXT = "#FFF2CF"

WINDOW_SIZE = "640x1050"

class KalmanSmoother:
    """Lightweight 1D Kalman filter for smoothing pitch curves."""

    def __init__(self, process_var: float = 1e-3, measurement_var: float = 1e-2):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.estimate = None
        self.estimate_var = None

    def reset(self):
        self.estimate = None
        self.estimate_var = None

    def update(self, measurement: float) -> float:
        if self.estimate is None:
            self.estimate = measurement
            self.estimate_var = 1.0
            return measurement
        pred_estimate = self.estimate
        pred_var = self.estimate_var + self.process_var
        K = pred_var / (pred_var + self.measurement_var)
        self.estimate = pred_estimate + K * (measurement - pred_estimate)
        self.estimate_var = (1 - K) * pred_var
        return self.estimate

    def smooth_array(self, data: list) -> list:
        output = []
        for x in data:
            output.append(self.update(float(x)))
        return output


class KalmanSmootherGUI:
    """GUI for Kalman Smoother using CustomTkinter."""

    def __init__(self, parent=None):
        self.ini_file = "ai_violin_master_05_kalman_smoother.ini"
        self.config = configparser.ConfigParser()

        if parent is None:
            self.root = ctk.CTk()
            self.is_standalone = True
            self.root.title("Kalman Pitch Smoother")
            self.root.geometry(WINDOW_SIZE)
            self.root.resizable(False, False)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        else:
            self.root = parent
            self.is_standalone = False

        self.kalman = KalmanSmoother()
        self.sample_data = None
        self.original_data = None
        self.smoothed_data = None
        self.process_var = 1e-3
        self.measurement_var = 1e-2

        self._load_state()
        ctk.set_appearance_mode("dark")

        if self.is_standalone:
            self.root.configure(fg_color=COLOR_BG)

        self._create_widgets()

        if self.is_standalone:
            self._restore_position()
            self.root.mainloop()

    def _load_state(self):
        if os.path.exists(self.ini_file):
            self.config.read(self.ini_file)
        if not self.config.has_section("window"):
            self.config.add_section("window")
        if not self.config.has_section("state"):
            self.config.add_section("state")

    def _save_state(self):
        self.config.set("window", "x", str(self.root.winfo_x()))
        self.config.set("window", "y", str(self.root.winfo_y()))
        self.config.set("state", "process_var", self.process_var_var.get())
        self.config.set("state", "measurement_var", self.measurement_var_var.get())
        with open(self.ini_file, "w") as f:
            self.config.write(f)

    def _restore_position(self):
        x = self.config.getint("window", "x", fallback=100)
        y = self.config.getint("window", "y", fallback=100)
        self.root.geometry(f"+{x}+{y}")

    def _generate_sample_data(self):
        t = np.linspace(0, 10, 500)
        true_pitch = 440 + 50 * np.sin(2 * np.pi * 0.5 * t) + 20 * np.sin(2 * np.pi * 2 * t)
        noise = np.random.normal(0, 20, len(t))
        noisy_pitch = true_pitch + noise
        outlier_indices = np.random.choice(len(t), 10, replace=False)
        noisy_pitch[outlier_indices] += np.random.uniform(-50, 50, 10)

        self.sample_data = {
            "time": t.tolist(),
            "pitch": noisy_pitch.tolist(),
            "true_pitch": true_pitch.tolist()
        }
        self.original_data = noisy_pitch.tolist()

    def _create_widgets(self):
        # Main container
        main_frame = ctk.CTkFrame(self.root, fg_color=COLOR_BG)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        header_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            header_frame,
            text="KALMAN PITCH SMOOTHER",
            font=("Georgia", 16, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            header_frame,
            text="Real-time smoothing of pitch curves using Kalman filtering",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=(0, 10))

        # Parameters
        params_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        params_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            params_frame,
            text="FILTER PARAMETERS",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        proc_row = ctk.CTkFrame(params_frame, fg_color=COLOR_DARK)
        proc_row.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            proc_row,
            text="Process Variance (Q):",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            width=180,
        ).pack(side="left")

        self.process_var_var = ctk.StringVar(value="0.001")
        ctk.CTkEntry(
            proc_row,
            textvariable=self.process_var_var,
            font=("Georgia", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
            width=100,
        ).pack(side="left", padx=5)

        meas_row = ctk.CTkFrame(params_frame, fg_color=COLOR_DARK)
        meas_row.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            meas_row,
            text="Measurement Variance (R):",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            width=180,
        ).pack(side="left")

        self.measurement_var_var = ctk.StringVar(value="0.01")
        ctk.CTkEntry(
            meas_row,
            textvariable=self.measurement_var_var,
            font=("Georgia", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
            width=100,
        ).pack(side="left", padx=5)

        # Apply button
        self.apply_btn = ctk.CTkButton(
            main_frame,
            text="APPLY SMOOTHING",
            font=("Georgia", 14, "bold"),
            fg_color=COLOR_DARK,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_MEDIUM,
            height=50,
            command=self._apply_smoothing,
        )
        self.apply_btn.pack(pady=20)

        # Visualization
        viz_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        viz_frame.pack(fill="both", expand=True, pady=10)

        ctk.CTkLabel(
            viz_frame,
            text="PITCH CURVE VISUALIZATION",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        self.figure = plt.Figure(figsize=(6, 3), dpi=80, facecolor=COLOR_DARK)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(COLOR_FRAME)
        self.ax.tick_params(colors=COLOR_TEXT)
        self.ax.xaxis.label.set_color(COLOR_TEXT)
        self.ax.yaxis.label.set_color(COLOR_TEXT)
        self.ax.title.set_color(COLOR_GOLD)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Pitch (Hz)")
        self.ax.set_title("Original vs Smoothed Pitch")

        self.canvas = FigureCanvasTkAgg(self.figure, viz_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        self._update_plot()

        # Statistics
        stats_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        stats_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            stats_frame,
            text="FILTER STATISTICS",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        self.stats_label = ctk.CTkLabel(
            stats_frame,
            text="Apply smoothing to see statistics",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        )
        self.stats_label.pack(pady=10)

        # Status bar
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Ready to smooth pitch data",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            height=30,
        )
        self.status_label.pack(fill="x", pady=(10, 0))

    def _update_plot(self):
        self.ax.clear()
        self.ax.set_facecolor(COLOR_FRAME)
        self.ax.tick_params(colors=COLOR_TEXT)
        self.ax.xaxis.label.set_color(COLOR_TEXT)
        self.ax.yaxis.label.set_color(COLOR_TEXT)
        self.ax.title.set_color(COLOR_GOLD)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Pitch (Hz)")
        self.ax.set_title("Original vs Smoothed Pitch")

        if self.sample_data is not None:
            time = self.sample_data["time"]
            original = self.sample_data["pitch"]
            true_pitch = self.sample_data["true_pitch"]

            if len(time) == len(true_pitch):
                self.ax.plot(time, true_pitch, 'g-', linewidth=1, alpha=0.5, label='True Pitch')

            self.ax.plot(time, original, 'b.', markersize=3, alpha=0.5, label='Original')

            if self.smoothed_data is not None and len(self.smoothed_data) == len(time):
                self.ax.plot(time, self.smoothed_data, 'r-', linewidth=2, label='Smoothed')

            self.ax.legend(facecolor=COLOR_FRAME, edgecolor=COLOR_TEXT, labelcolor=COLOR_TEXT)

        self.canvas.draw()

    def _apply_smoothing(self):
        self.status_label.configure(text="Applying Kalman smoothing...")
        self.root.update()

        self.process_var = float(self.process_var_var.get())
        self.measurement_var = float(self.measurement_var_var.get())

        self.kalman.process_var = self.process_var
        self.kalman.measurement_var = self.measurement_var
        self.kalman.reset()

        pitch_data = self.sample_data["pitch"]
        self.smoothed_data = self.kalman.smooth_array(pitch_data)

        self._update_plot()

        # Calculate statistics
        original = np.array(self.sample_data["pitch"])
        smoothed = np.array(self.smoothed_data)
        true_pitch = np.array(self.sample_data.get("true_pitch", original))

        original_error = np.sqrt(np.mean((original - true_pitch) ** 2))
        smoothed_error = np.sqrt(np.mean((smoothed - true_pitch) ** 2))

        if original_error > 0:
            noise_reduction = (1 - (smoothed_error / original_error)) * 100
        else:
            noise_reduction = 0

        stats_text = (
            f"Original RMS Error: {original_error:.1f} Hz\n"
            f"Smoothed RMS Error: {smoothed_error:.1f} Hz\n"
            f"Noise Reduction: {noise_reduction:.1f}%"
        )
        self.stats_label.configure(text=stats_text)

        self.status_label.configure(text=f"Smoothing applied (Q={self.process_var}, R={self.measurement_var})")

    def _on_close(self):
        self._save_state()
        plt.close(self.figure)
        self.root.destroy()


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = KalmanSmootherGUI(parent)
    return gui.root


def main():
    """Standalone entry - launches a full window with mainloop."""
    KalmanSmootherGUI()


if __name__ == "__main__":
    main()