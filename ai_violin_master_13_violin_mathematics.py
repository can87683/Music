#!/usr/bin/env python3
# ai_violin_master_13_violin_mathematics.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683


import customtkinter as ctk
from tkinter import messagebox
import numpy as np
import math
import os
from datetime import datetime
import configparser
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Color constants
COLOR_BG = "#5A381E"
COLOR_DARK = "#3B2413"
COLOR_GOLD = "#FFD770"
COLOR_FRAME = "#4B2E18"
COLOR_TEXT = "#FFF2CF"
COLOR_MEDIUM = "#4B2E18"

WINDOW_SIZE = "640x1050"


class ViolinMathematics:
    """Comprehensive mathematical modeling of violin performance."""

    def __init__(self):
        self.constants = {
            "string_tension": 55,
            "string_density": 0.0005,
            "string_length": 0.33,
            "air_density": 1.225,
            "sound_speed": 343,
            "bow_hair_friction": 0.4,
            "violin_body_resonance": 440
        }

    def string_vibration(self, x, t, mode=1):
        c = math.sqrt(self.constants["string_tension"] / self.constants["string_density"])
        L = self.constants["string_length"]
        A = 0.001
        n = mode
        omega = n * math.pi * c / L
        return A * math.sin(n * math.pi * x / L) * math.cos(omega * t)

    def bow_flow(self, v_bow, pressure, angle_deg):
        angle = math.radians(angle_deg)
        hair_diameter = 0.0002
        kinematic_viscosity = 1.5e-5
        Re = v_bow * hair_diameter / kinematic_viscosity
        if Re > 2000:
            separation_factor = 0.3 + 0.2 * math.sin(angle)
        else:
            separation_factor = 0.1 + 0.05 * math.sin(angle)
        tone_quality = 0.8 - separation_factor * 0.4
        return {
            "reynolds": Re,
            "flow_state": "turbulent" if Re > 2000 else "laminar",
            "separation_factor": separation_factor,
            "tone_quality": tone_quality,
            "energy_efficiency": 0.9 - separation_factor * 0.3
        }

    def pitch_accuracy(self, target_freq, actual_freq, duration):
        cents_error = 1200 * math.log2(actual_freq / target_freq)
        time_points = np.linspace(0, duration, 100)
        if abs(cents_error) < 5:
            drift = 0.1 * np.sin(2 * np.pi * 0.5 * time_points)
            stability = 0.95
        elif abs(cents_error) < 20:
            drift = 0.5 * np.sin(2 * np.pi * 2 * time_points)
            stability = 0.75
        else:
            drift = 2.0 * np.random.randn(len(time_points))
            stability = 0.4
        return {
            "cents_error": cents_error,
            "stability_score": stability,
            "drift_pattern": drift.tolist(),
            "intonation_grade": self._grade_intonation(abs(cents_error))
        }

    def vibrato_profile(self, rate_hz, width_cents, phase=0):
        t = np.linspace(0, 2, 200)
        f0 = 440
        modulation = width_cents / 1200
        f_t = f0 * (2 ** (modulation * np.sin(2 * np.pi * rate_hz * t + phase)))
        amplitude_envelope = np.exp(-0.5 * t)
        return {
            "time": t.tolist(),
            "frequency": f_t.tolist(),
            "modulation_depth": width_cents,
            "modulation_rate": rate_hz,
            "envelope": amplitude_envelope.tolist(),
            "phase_consistency": 0.85 + 0.1 * np.random.random()
        }

    def energy_analysis(self, bow_force, bow_velocity, contact_point):
        mechanical_energy = bow_force * bow_velocity
        string_damping = 0.15
        bridge_loss = 0.25
        body_loss = 0.35
        radiation_loss = 0.15
        position_factor = abs(contact_point - 0.2) / 0.2
        total_efficiency = (1 - string_damping) * (1 - bridge_loss) * (1 - body_loss) * (1 - radiation_loss) * (1 - position_factor * 0.3)
        sound_energy = mechanical_energy * total_efficiency
        fundamental_energy = sound_energy * 0.6
        harmonic_energy = sound_energy * 0.4
        return {
            "mechanical_input": mechanical_energy,
            "sound_output": sound_energy,
            "efficiency": total_efficiency,
            "fundamental_ratio": fundamental_energy / sound_energy,
            "harmonic_richness": harmonic_energy / fundamental_energy,
            "position_penalty": position_factor * 0.3
        }

    def _grade_intonation(self, cents_error):
        if cents_error < 5:
            return "A+ (Perfect)"
        elif cents_error < 10:
            return "A (Excellent)"
        elif cents_error < 20:
            return "B (Good)"
        elif cents_error < 30:
            return "C (Fair)"
        elif cents_error < 40:
            return "D (Poor)"
        else:
            return "F (Unacceptable)"


class ViolinMathematicsGUI:
    """GUI for Violin Mathematics using CustomTkinter."""

    def __init__(self, parent=None):
        self.ini_file = "ai_violin_master_13_violin_mathematics.ini"
        self.config = configparser.ConfigParser()

        if parent is None:
            self.root = ctk.CTk()
            self.is_standalone = True
            self.root.title("Violin Mathematics Visualizer")
            self.root.geometry(WINDOW_SIZE)
            self.root.resizable(False, False)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        else:
            self.root = parent
            self.is_standalone = False

        self.mathematics = ViolinMathematics()
        self.current_figures = []

        self._load_state()
        ctk.set_appearance_mode("dark")
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
        # Main container with scroll
        main_frame = ctk.CTkScrollableFrame(self.root, fg_color=COLOR_BG)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        header_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            header_frame,
            text="VIOLIN MATHEMATICS VISUALIZER",
            font=("Georgia", 16, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            header_frame,
            text="Visualizing the Calculus, Physics, and Harmony of Violin Performance",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=(0, 10))

        # String vibration section
        self._create_string_vibration_section(main_frame)

        # Bow fluid section
        self._create_bow_fluid_section(main_frame)

        # Intonation section
        self._create_intonation_section(main_frame)

        # Vibrato section
        self._create_vibrato_section(main_frame)

        # Energy section
        self._create_energy_section(main_frame)

    def _create_string_vibration_section(self, parent):
        section_frame = ctk.CTkFrame(parent, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        section_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            section_frame,
            text="STRING VIBRATION WAVE EQUATION",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            section_frame,
            text="d2y/dt2 = c2 * d2y/dx2",
            font=("Courier New", 14),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=5)

        # Matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))
        fig.patch.set_facecolor(COLOR_DARK)

        x = np.linspace(0, 1, 100)
        times = [0, 0.002, 0.004, 0.006, 0.008]
        colors = ['#FF6B6B', '#FFD700', '#90EE90', '#87CEEB', '#DDA0DD']
        for t, color in zip(times, colors):
            y = np.sin(np.pi * x) * np.cos(2 * np.pi * 440 * t)
            ax1.plot(x, y, color=color, linewidth=1.5, label=f't={t*1000:.1f}ms')
        ax1.set_xlabel('Position', color=COLOR_TEXT, fontsize=8)
        ax1.set_ylabel('Displacement', color=COLOR_TEXT, fontsize=8)
        ax1.set_title('String Shape Over Time', color=COLOR_GOLD, fontsize=9)
        ax1.legend(fontsize=7, facecolor=COLOR_DARK, edgecolor=COLOR_TEXT)
        ax1.set_facecolor(COLOR_FRAME)
        ax1.tick_params(colors=COLOR_TEXT, labelsize=7)

        modes = [1, 2, 3, 4]
        for n in modes:
            y = np.sin(n * np.pi * x)
            ax2.plot(x, y, linewidth=1.2, label=f'n={n}')
        ax2.set_xlabel('Position', color=COLOR_TEXT, fontsize=8)
        ax2.set_ylabel('Amplitude', color=COLOR_TEXT, fontsize=8)
        ax2.set_title('Standing Wave Modes', color=COLOR_GOLD, fontsize=9)
        ax2.legend(fontsize=7, facecolor=COLOR_DARK, edgecolor=COLOR_TEXT)
        ax2.set_facecolor(COLOR_FRAME)
        ax2.tick_params(colors=COLOR_TEXT, labelsize=7)

        fig.tight_layout(pad=1.5)
        canvas = FigureCanvasTkAgg(fig, section_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True, padx=10, pady=10)
        self.current_figures.append(fig)

    def _create_bow_fluid_section(self, parent):
        section_frame = ctk.CTkFrame(parent, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        section_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            section_frame,
            text="BOW FLUID DYNAMICS",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            section_frame,
            text="Re = v*d/nu, Flow Separation = f(theta, Re)",
            font=("Courier New", 14),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=5)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))
        fig.patch.set_facecolor(COLOR_DARK)

        y = np.linspace(-0.001, 0.001, 100)
        u_laminar = 1 - (y/0.001)**2
        u_turbulent = 1 - abs(y/0.001)**(1/7)
        ax1.plot(u_laminar, y, 'b-', linewidth=1.5, label='Laminar')
        ax1.plot(u_turbulent, y, 'r--', linewidth=1.5, label='Turbulent')
        ax1.set_xlabel('Flow Velocity', color=COLOR_TEXT, fontsize=8)
        ax1.set_ylabel('Distance from hair', color=COLOR_TEXT, fontsize=8)
        ax1.set_title('Boundary Layer Profiles', color=COLOR_GOLD, fontsize=9)
        ax1.legend(fontsize=7, facecolor=COLOR_DARK, edgecolor=COLOR_TEXT)
        ax1.set_facecolor(COLOR_FRAME)
        ax1.tick_params(colors=COLOR_TEXT, labelsize=7)

        Re = np.linspace(100, 5000, 50)
        tone_quality = 0.8 - 0.4 * np.tanh((Re - 2000) / 1000)
        ax2.plot(Re, tone_quality, 'g-', linewidth=1.5)
        ax2.axvline(x=2000, color='yellow', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Reynolds Number', color=COLOR_TEXT, fontsize=8)
        ax2.set_ylabel('Tone Quality', color=COLOR_TEXT, fontsize=8)
        ax2.set_title('Flow Quality vs Re', color=COLOR_GOLD, fontsize=9)
        ax2.set_facecolor(COLOR_FRAME)
        ax2.tick_params(colors=COLOR_TEXT, labelsize=7)

        fig.tight_layout(pad=1.5)
        canvas = FigureCanvasTkAgg(fig, section_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True, padx=10, pady=10)
        self.current_figures.append(fig)

    def _create_intonation_section(self, parent):
        section_frame = ctk.CTkFrame(parent, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        section_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            section_frame,
            text="INTONATION CALCULUS",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            section_frame,
            text="dtheta/dt -> 0, Delta cents = 1200*log2(f/f0)",
            font=("Courier New", 14),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=5)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))
        fig.patch.set_facecolor(COLOR_DARK)

        t = np.linspace(0, 2, 200)
        perfect = 440 + 0.1 * np.sin(2*np.pi*0.5*t)
        good = 440 + 0.5 * np.sin(2*np.pi*2*t)
        poor = 440 + 2.0 * np.random.randn(len(t))
        ax1.plot(t, perfect, 'g-', linewidth=1.5, label='Perfect')
        ax1.plot(t, good, 'y-', linewidth=1.5, label='Good')
        ax1.plot(t, poor, 'r-', linewidth=1.5, label='Poor')
        ax1.axhline(y=440, color='white', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Time (s)', color=COLOR_TEXT, fontsize=8)
        ax1.set_ylabel('Frequency (Hz)', color=COLOR_TEXT, fontsize=8)
        ax1.set_title('Pitch Stability', color=COLOR_GOLD, fontsize=9)
        ax1.legend(fontsize=7, facecolor=COLOR_DARK, edgecolor=COLOR_TEXT)
        ax1.set_facecolor(COLOR_FRAME)
        ax1.tick_params(colors=COLOR_TEXT, labelsize=7)

        cents_errors = np.random.normal(0, 10, 1000)
        ax2.hist(cents_errors, bins=20, color='skyblue', edgecolor='white', alpha=0.7)
        ax2.axvline(x=0, color='yellow', linestyle='--', linewidth=1.5)
        ax2.set_xlabel('Cents Error', color=COLOR_TEXT, fontsize=8)
        ax2.set_ylabel('Count', color=COLOR_TEXT, fontsize=8)
        ax2.set_title('Intonation Distribution', color=COLOR_GOLD, fontsize=9)
        ax2.set_facecolor(COLOR_FRAME)
        ax2.tick_params(colors=COLOR_TEXT, labelsize=7)

        fig.tight_layout(pad=1.5)
        canvas = FigureCanvasTkAgg(fig, section_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True, padx=10, pady=10)
        self.current_figures.append(fig)

    def _create_vibrato_section(self, parent):
        section_frame = ctk.CTkFrame(parent, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        section_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            section_frame,
            text="VIBRATO WAVE MODEL",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            section_frame,
            text="f(t) = f0 * 2^(Delta * sin(2*pi*nu*t + phi))",
            font=("Courier New", 14),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=5)

        fig, ax = plt.subplots(figsize=(6, 2.5))
        fig.patch.set_facecolor(COLOR_DARK)

        t = np.linspace(0, 2, 200)
        f0 = 440
        rate = 6
        width = 30
        modulation = width / 1200
        f_t = f0 * (2 ** (modulation * np.sin(2 * np.pi * rate * t)))
        ax.plot(t, f_t, 'c-', linewidth=1.5)
        ax.fill_between(t, f_t, f0, alpha=0.3, color='cyan')
        ax.axhline(y=f0, color='white', linestyle='--', alpha=0.7)
        ax.set_xlabel('Time (s)', color=COLOR_TEXT, fontsize=8)
        ax.set_ylabel('Frequency (Hz)', color=COLOR_TEXT, fontsize=8)
        ax.set_title(f'Vibrato: {rate}Hz, {width} cents', color=COLOR_GOLD, fontsize=9)
        ax.set_facecolor(COLOR_FRAME)
        ax.tick_params(colors=COLOR_TEXT, labelsize=7)

        fig.tight_layout(pad=1.5)
        canvas = FigureCanvasTkAgg(fig, section_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True, padx=10, pady=10)
        self.current_figures.append(fig)

    def _create_energy_section(self, parent):
        section_frame = ctk.CTkFrame(parent, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        section_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            section_frame,
            text="ENERGY TRANSFER MODEL",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            section_frame,
            text="eta = (1-alpha)*(1-beta)*(1-gamma)*(1-delta)*(1-epsilon)",
            font=("Courier New", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=5)

        fig, ax = plt.subplots(figsize=(6, 2.5))
        fig.patch.set_facecolor(COLOR_DARK)

        categories = ['Input', 'String', 'Bridge', 'Body', 'Radiation', 'Output']
        values = [100, 85, 60, 40, 25, 20]
        colors = ['#FF6B6B', '#FFA500', '#FFD700', '#90EE90', '#87CEEB', '#DDA0DD']
        bars = ax.bar(categories, values, color=colors, edgecolor='white')
        ax.set_ylabel('Energy (%)', color=COLOR_TEXT, fontsize=8)
        ax.set_title('Energy Loss Through Violin', color=COLOR_GOLD, fontsize=9)
        ax.set_facecolor(COLOR_FRAME)
        ax.tick_params(colors=COLOR_TEXT, labelsize=7, rotation=45)

        fig.tight_layout(pad=1.5)
        canvas = FigureCanvasTkAgg(fig, section_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True, padx=10, pady=10)
        self.current_figures.append(fig)

    def _on_close(self):
        self._save_state()
        for fig in self.current_figures:
            plt.close(fig)
        self.root.destroy()


def get_content():
    """Return module description for display in GUI."""
    return [
        "VIOLIN MATHEMATICS VISUALIZER",
        "",
        "PURPOSE: Comprehensive visualization of violin performance mathematics",
        "",
        "MATHEMATICAL MODELS:",
        "  1. String Vibration: d2y/dt2 = c2*d2y/dx2",
        "  2. Bow Fluid Dynamics: Reynolds number, flow separation",
        "  3. Intonation Calculus: dtheta/dt -> 0",
        "  4. Vibrato Waves: frequency modulation",
        "  5. Energy Transfer: conservation through system",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = ViolinMathematicsGUI(parent)
    return gui.root


def main():
    """Standalone entry - launches a full window with mainloop."""
    ViolinMathematicsGUI()


if __name__ == "__main__":
    main()