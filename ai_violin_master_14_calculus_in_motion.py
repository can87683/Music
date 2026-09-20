#!/usr/bin/env python3
# ai_violin_master_14_calculus_in_motion.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683


import customtkinter as ctk
import numpy as np
import math
from datetime import datetime
import os
import json
import configparser
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
from matplotlib.collections import LineCollection

# Color constants
COLOR_BG = "#2A1B0A"
COLOR_DARK = "#1A0B00"
COLOR_GOLD = "#FFE8A0"
COLOR_FRAME = "#4A2B1A"
COLOR_TEXT = "#FFE8C8"
COLOR_ENERGY = "#FF6B6B"
COLOR_FLOW = "#4ECDC4"
COLOR_RESONANCE = "#FFD166"
COLOR_HARMONY = "#9D65C9"
COLOR_MEDIUM = "#4B2E18"

WINDOW_SIZE = "640x1050"


class ViolinCalculus:
    """Multivariate calculus visualization of violin performance."""

    def __init__(self):
        self.state = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.params = {
            'm_bow': 0.06,
            'k_bow': 50.0,
            'c_bow': 0.3,
            'm_string': 0.002,
            'k_string': 1000.0,
            'c_string': 0.02,
            'coupling': 0.8,
            'energy_eff': 0.15,
            'resonance_freq': 440.0
        }
        self.time = 0.0
        self.dt = 0.001
        self.history = []

    def system_equations(self, t, state):
        x_bow, v_bow, x_finger, v_finger, theta_string, omega_string, E_sound = state
        F_arm = 2.0 * math.sin(2 * math.pi * 0.5 * t)
        F_coupling = self.params['coupling'] * (x_finger - x_bow)
        a_bow = (-self.params['k_bow'] * x_bow - self.params['c_bow'] * v_bow + F_arm + F_coupling) / self.params['m_bow']
        a_finger = (-self.params['k_string'] * x_finger - self.params['c_string'] * v_finger - F_coupling) / self.params['m_string']
        omega0 = 2 * math.pi * self.params['resonance_freq']
        gamma = 0.1
        excitation = 0.5 * v_bow * math.cos(omega_string * t)
        alpha_string = -omega0**2 * theta_string - gamma * omega_string + excitation
        kinetic = 0.5 * self.params['m_bow'] * v_bow**2
        lambda_decay = 0.2
        dE_sound = self.params['energy_eff'] * kinetic - lambda_decay * E_sound
        return [v_bow, a_bow, v_finger, a_finger, omega_string, alpha_string, dE_sound]

    def update(self, time_step=0.01):
        self.history.append({
            'time': self.time,
            'state': self.state.copy(),
            'energy_kinetic': 0.5 * self.params['m_bow'] * self.state[1]**2,
            'energy_potential': 0.5 * self.params['k_bow'] * self.state[0]**2,
            'energy_sound': self.state[6],
            'pitch': self.params['resonance_freq'] * (1 + 0.1 * math.sin(self.state[4]))
        })
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        self.time += time_step
        sol = solve_ivp(
            self.system_equations,
            [self.time - time_step, self.time],
            self.state,
            t_eval=[self.time],
            method='RK45'
        )
        if sol.y.shape[1] > 0:
            self.state = sol.y[:, -1]
        return self.state

    def get_phase_portrait(self, variable1=1, variable2=4):
        if len(self.history) < 10:
            return None
        v1 = [h['state'][variable1] for h in self.history[-100:]]
        v2 = [h['state'][variable2] for h in self.history[-100:]]
        return {
            'x': v1,
            'y': v2,
            'dx': np.gradient(v1).tolist(),
            'dy': np.gradient(v2).tolist()
        }

    def get_energy_flow(self):
        if len(self.history) < 2:
            return None
        recent = self.history[-50:]
        dEk_dt = np.gradient([h['energy_kinetic'] for h in recent])
        dEp_dt = np.gradient([h['energy_potential'] for h in recent])
        dEs_dt = np.gradient([h['energy_sound'] for h in recent])
        power_mech = dEk_dt + dEp_dt
        power_sound = dEs_dt
        efficiency = []
        for i in range(len(power_mech)):
            if abs(power_mech[i]) > 1e-6:
                eff = power_sound[i] / power_mech[i] if power_mech[i] > 0 else 0
                efficiency.append(max(0, min(1, eff)))
            else:
                efficiency.append(0)
        return {
            'time': [h['time'] for h in recent],
            'power_mech': power_mech.tolist(),
            'power_sound': power_sound.tolist(),
            'efficiency': efficiency,
            'energy_total': [h['energy_kinetic'] + h['energy_potential'] + h['energy_sound'] for h in recent]
        }

    def get_mathematical_insights(self):
        if len(self.history) < 10:
            return []
        insights = []
        phase = self.get_phase_portrait(1, 4)
        if phase:
            div = sum(phase['dx'][i] + phase['dy'][i] for i in range(len(phase['dx']))) / len(phase['dx'])
            if div < 0:
                insights.append("Phase space contracting: System approaching stable limit cycle")
            else:
                insights.append("Phase space expanding: Rich, expressive dynamics")
        energy_flow = self.get_energy_flow()
        if energy_flow and len(energy_flow['energy_total']) > 5:
            energy_change = abs(energy_flow['energy_total'][-1] - energy_flow['energy_total'][0])
            if energy_change < 0.1:
                insights.append("Energy conserved: Perfect mechanical-acoustic coupling")
            else:
                insights.append(f"Energy flowing: {energy_change:.2f} J transferred")
        recent_pitch = [h['pitch'] for h in self.history[-20:]]
        pitch_variance = np.var(recent_pitch)
        if pitch_variance < 1.0:
            insights.append("Pure harmonic: d2theta/dt2 approaching zero")
        else:
            insights.append(f"Rich harmonics: Pitch variance = {pitch_variance:.2f} Hz2")
        return insights

    def reset(self):
        self.state = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.time = 0.0
        self.history = []

    def set_resonance(self, frequency):
        self.params['resonance_freq'] = frequency

    def apply_vibrato(self, rate_hz=6.0, depth_cents=30.0):
        modulation = depth_cents / 1200.0
        self.params['resonance_freq'] = 440.0 * (2 ** (modulation * math.sin(2 * math.pi * rate_hz * self.time)))


class CalculusInMotionGUI:
    """GUI visualizing violin as multivariate calculus in motion."""

    def __init__(self, parent=None):
        self.ini_file = "ai_violin_master_14_calculus_in_motion.ini"
        self.config = configparser.ConfigParser()

        if parent is None:
            self.root = ctk.CTk()
            self.is_standalone = True
            self.root.title("Calculus in Motion")
            self.root.geometry(WINDOW_SIZE)
            self.root.resizable(False, False)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        else:
            self.root = parent
            self.is_standalone = False

        self.calculus = ViolinCalculus()
        self.is_playing = False
        self.playback_speed = 1.0
        self.visualization_mode = "energy_flow"
        self.animation_id = None
        self.is_closing = False
        self.figures = []

        self._load_state()
        ctk.set_appearance_mode("dark")
        self.root.configure(fg_color=COLOR_BG)

        self._create_widgets()
        self._start_animation()

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
            text="CALCULUS IN MOTION",
            font=("Georgia", 16, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            header_frame,
            text="The violin is not an instrument, but a differential equation.",
            font=("Georgia", 12, "italic"),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=(0, 10))

        # Control panel
        self._create_control_panel(main_frame)

        # Main visualization
        self._create_main_visualization(main_frame)

        # Mathematical projections
        self._create_math_visualizations(main_frame)

        # Insights panel
        self._create_insights_panel(main_frame)

        # Action buttons
        self._create_action_buttons(main_frame)

    def _create_control_panel(self, parent):
        control_frame = ctk.CTkFrame(parent, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        control_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            control_frame,
            text="CONTROL THE DIFFERENTIAL EQUATIONS",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        play_row = ctk.CTkFrame(control_frame, fg_color=COLOR_DARK)
        play_row.pack(fill="x", padx=20, pady=5)

        self.play_btn = ctk.CTkButton(
            play_row,
            text="PLAY CALCULUS",
            font=("Georgia", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_DARK,
            command=self._toggle_play,
        )
        self.play_btn.pack(side="left", padx=5, expand=True, fill="x")

        ctk.CTkButton(
            play_row,
            text="RESET SYSTEM",
            font=("Georgia", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_DARK,
            command=self._reset_system,
        ).pack(side="left", padx=5, expand=True, fill="x")

        # Speed control
        speed_row = ctk.CTkFrame(control_frame, fg_color=COLOR_DARK)
        speed_row.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            speed_row,
            text="Time Scale:",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            width=100,
        ).pack(side="left")

        self.speed_var = ctk.DoubleVar(value=1.0)
        ctk.CTkSlider(
            speed_row,
            from_=0.1,
            to=5.0,
            variable=self.speed_var,
            fg_color=COLOR_FRAME,
            progress_color=COLOR_GOLD,
            button_color=COLOR_GOLD,
            button_hover_color=COLOR_TEXT,
            command=self._update_speed,
        ).pack(side="left", fill="x", expand=True, padx=5)

        # Visualization mode
        viz_row = ctk.CTkFrame(control_frame, fg_color=COLOR_DARK)
        viz_row.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            viz_row,
            text="View:",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            width=100,
        ).pack(side="left")

        self.viz_var = ctk.StringVar(value="energy_flow")
        ctk.CTkOptionMenu(
            viz_row,
            variable=self.viz_var,
            values=["energy_flow", "phase_space", "all"],
            font=("Georgia", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            button_color=COLOR_FRAME,
            button_hover_color=COLOR_DARK,
            dropdown_fg_color=COLOR_FRAME,
            dropdown_text_color=COLOR_TEXT,
            dropdown_hover_color=COLOR_DARK,
            command=self._update_visualization_mode,
        ).pack(side="left", padx=5)

    def _create_main_visualization(self, parent):
        viz_frame = ctk.CTkFrame(parent, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        viz_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            viz_frame,
            text="REAL-TIME CALCULUS VISUALIZATION",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        self.main_fig, self.main_ax = plt.subplots(figsize=(6, 3))
        self.main_fig.patch.set_facecolor(COLOR_DARK)
        self.main_ax.set_facecolor(COLOR_FRAME)

        self.main_canvas = FigureCanvasTkAgg(self.main_fig, viz_frame)
        self.main_canvas.get_tk_widget().pack(fill="x", expand=True, padx=10, pady=10)

        self.figures.append(self.main_fig)
        self._update_main_visualization()

    def _create_math_visualizations(self, parent):
        math_frame = ctk.CTkFrame(parent, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        math_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            math_frame,
            text="MATHEMATICAL PROJECTIONS",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        self.math_fig, self.math_axes = plt.subplots(2, 2, figsize=(6, 4.5))
        self.math_fig.patch.set_facecolor(COLOR_DARK)

        for ax in self.math_axes.flatten():
            ax.set_facecolor(COLOR_FRAME)
            ax.tick_params(colors=COLOR_TEXT, labelsize=7)
            ax.xaxis.label.set_color(COLOR_TEXT)
            ax.yaxis.label.set_color(COLOR_TEXT)
            ax.title.set_color(COLOR_GOLD)
            ax.title.set_fontsize(9)

        self.math_axes[0, 0].set_title("Energy Conservation")
        self.math_axes[0, 1].set_title("Phase Space")
        self.math_axes[1, 0].set_title("Power Flow")
        self.math_axes[1, 1].set_title("Harmonic Spectrum")

        self.math_canvas = FigureCanvasTkAgg(self.math_fig, math_frame)
        self.math_canvas.get_tk_widget().pack(fill="x", expand=True, padx=10, pady=10)

        self.figures.append(self.math_fig)

    def _create_insights_panel(self, parent):
        insights_frame = ctk.CTkFrame(parent, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        insights_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            insights_frame,
            text="MATHEMATICAL INSIGHTS",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        self.insights_text = ctk.CTkTextbox(
            insights_frame,
            font=("Georgia", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
            height=120,
            wrap="word",
        )
        self.insights_text.pack(fill="x", padx=10, pady=10)
        self.insights_text.insert("1.0", "Initializing differential equations...\n")
        self.insights_text.insert("end", "System ready for multivariate calculus simulation.\n\n")
        self.insights_text.insert("end", "Click 'PLAY CALCULUS' to begin solving the equations of motion.")

    def _create_action_buttons(self, parent):
        actions_frame = ctk.CTkFrame(parent, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        actions_frame.pack(fill="x", pady=10)

        ctk.CTkButton(
            actions_frame,
            text="APPLY VIBRATO",
            font=("Georgia", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_DARK,
            command=self._apply_vibrato,
        ).pack(side="left", padx=5, pady=10, expand=True, fill="x")

        ctk.CTkButton(
            actions_frame,
            text="CHAOTIC BOWING",
            font=("Georgia", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_DARK,
            command=self._chaotic_bowing,
        ).pack(side="left", padx=5, pady=10, expand=True, fill="x")

        ctk.CTkButton(
            actions_frame,
            text="SAVE STATE",
            font=("Georgia", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_DARK,
            command=self._save_state_file,
        ).pack(side="left", padx=5, pady=10, expand=True, fill="x")

        self.status_label = ctk.CTkLabel(
            parent,
            text="System initialized: 7 differential equations ready",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            height=30,
        )
        self.status_label.pack(fill="x", pady=(10, 0))

    def _toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.configure(text="PAUSE CALCULUS")
            self.status_label.configure(text="Solving differential equations...")
        else:
            self.play_btn.configure(text="PLAY CALCULUS")
            self.status_label.configure(text="Calculus paused")

    def _reset_system(self):
        self.calculus.reset()
        self.is_playing = False
        self.play_btn.configure(text="PLAY CALCULUS")
        self._update_main_visualization()
        self._update_math_visualizations()
        self.insights_text.delete("1.0", "end")
        self.insights_text.insert("1.0", "System reset to initial conditions.\n")
        self.insights_text.insert("end", "All differential equations set to t=0.\n\n")
        self.insights_text.insert("end", "Click 'PLAY CALCULUS' to begin solving.")
        self.status_label.configure(text="System reset: Ready for new initial conditions")

    def _update_speed(self, value):
        self.playback_speed = float(value)

    def _update_visualization_mode(self, value):
        self.visualization_mode = value

    def _apply_vibrato(self):
        self.calculus.apply_vibrato()
        self.insights_text.insert("end", "\nApplied vibrato: Frequency modulation\n")
        self.insights_text.see("end")
        self.status_label.configure(text="Vibrato: d2theta/dt2 modulation active")

    def _chaotic_bowing(self):
        self.calculus.params['coupling'] = 1.2
        self.calculus.params['c_bow'] = 0.1
        self.insights_text.insert("end", "\nChaotic bowing: Sensitivity to initial conditions\n")
        self.insights_text.see("end")
        self.status_label.configure(text="Chaotic regime: Nonlinear dynamics active")

    def _save_state_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"violin_calculus_state_{timestamp}.json"
        state_data = {
            'timestamp': timestamp,
            'system_state': self.calculus.state.tolist(),
            'parameters': self.calculus.params,
            'history_length': len(self.calculus.history),
            'mathematical_insights': self.calculus.get_mathematical_insights()
        }
        with open(filename, 'w') as f:
            json.dump(state_data, f, indent=2)
        self.insights_text.insert("end", f"\nState saved: {filename}\n")
        self.insights_text.see("end")
        self.status_label.configure(text=f"State saved to {filename}")

    def _start_animation(self):
        if not self.is_closing:
            self._animate()
            self.animation_id = self.root.after(50, self._start_animation)

    def _animate(self):
        if self.is_playing and not self.is_closing:
            for _ in range(int(10 * self.playback_speed)):
                self.calculus.update(time_step=0.01 * self.playback_speed)
            self._update_main_visualization()
            self._update_math_visualizations()
            self._update_insights()

    def _update_main_visualization(self):
        self.main_ax.clear()
        mode = self.visualization_mode
        if mode == "energy_flow" or mode == "all":
            self._plot_energy_flow(self.main_ax)
        elif mode == "phase_space":
            self._plot_phase_space(self.main_ax)

        self.main_ax.set_facecolor(COLOR_FRAME)
        self.main_ax.tick_params(colors=COLOR_TEXT, labelsize=8)
        self.main_ax.xaxis.label.set_color(COLOR_TEXT)
        self.main_ax.yaxis.label.set_color(COLOR_TEXT)
        self.main_ax.title.set_color(COLOR_GOLD)
        self.main_fig.tight_layout(pad=2)
        self.main_canvas.draw()

    def _plot_energy_flow(self, ax):
        if len(self.calculus.history) < 10:
            ax.text(0.5, 0.5, "Solving equations...",
                   ha='center', va='center', transform=ax.transAxes,
                   color=COLOR_TEXT, fontsize=10)
            ax.set_title("Energy Flow", color=COLOR_GOLD)
            return
        energy_flow = self.calculus.get_energy_flow()
        if not energy_flow:
            return
        times = energy_flow['time']
        ax.plot(times, energy_flow['power_mech'], 'b-', linewidth=1.5, label='Mechanical')
        ax.plot(times, energy_flow['power_sound'], 'r-', linewidth=1.5, label='Sound')
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Power (W)', fontsize=8)
        ax.set_title("Energy Flow", color=COLOR_GOLD, fontsize=10)
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3, color=COLOR_TEXT)

    def _plot_phase_space(self, ax):
        phase = self.calculus.get_phase_portrait(1, 4)
        if not phase:
            ax.text(0.5, 0.5, "Computing phase space...",
                   ha='center', va='center', transform=ax.transAxes,
                   color=COLOR_TEXT, fontsize=10)
            ax.set_title("Phase Space", color=COLOR_GOLD)
            return
        ax.plot(phase['x'], phase['y'], 'w-', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Bow Velocity', fontsize=8)
        ax.set_ylabel('String Angle', fontsize=8)
        ax.set_title("Phase Space", color=COLOR_GOLD, fontsize=10)
        ax.grid(True, alpha=0.3, color=COLOR_TEXT)

    def _update_math_visualizations(self):
        if len(self.calculus.history) < 10:
            return
        for ax in self.math_axes.flatten():
            ax.clear()
            ax.set_facecolor(COLOR_FRAME)
            ax.tick_params(colors=COLOR_TEXT, labelsize=6)
            ax.xaxis.label.set_color(COLOR_TEXT)
            ax.yaxis.label.set_color(COLOR_TEXT)
            ax.title.set_color(COLOR_GOLD)
            ax.title.set_fontsize(9)

        # Energy conservation
        recent = self.calculus.history[-100:] if len(self.calculus.history) >= 100 else self.calculus.history
        times = [h['time'] for h in recent]
        kinetic = [h['energy_kinetic'] for h in recent]
        potential = [h['energy_potential'] for h in recent]
        sound = [h['energy_sound'] for h in recent]
        self.math_axes[0, 0].stackplot(times, kinetic, potential, sound,
                                      colors=[COLOR_ENERGY, COLOR_RESONANCE, COLOR_HARMONY],
                                      alpha=0.7)
        self.math_axes[0, 0].set_title("Energy Conservation", color=COLOR_GOLD, fontsize=9)

        # Phase space
        phase = self.calculus.get_phase_portrait(1, 4)
        if phase:
            self.math_axes[0, 1].plot(phase['x'], phase['y'], color=COLOR_FLOW, linewidth=1.5)
            self.math_axes[0, 1].set_title("Phase Space", color=COLOR_GOLD, fontsize=9)

        # Power flow
        energy_flow = self.calculus.get_energy_flow()
        if energy_flow:
            times = energy_flow['time']
            self.math_axes[1, 0].plot(times, energy_flow['power_mech'], color=COLOR_ENERGY, linewidth=1.5)
            self.math_axes[1, 0].plot(times, energy_flow['power_sound'], color=COLOR_HARMONY, linewidth=1.5)
            self.math_axes[1, 0].set_title("Power Flow", color=COLOR_GOLD, fontsize=9)

        # Harmonic spectrum
        if len(self.calculus.history) >= 20:
            pitches = [h['pitch'] for h in self.calculus.history[-100:]]
            N = len(pitches)
            yf = fft(pitches)
            xf = fftfreq(N, 0.01)[:N//2]
            self.math_axes[1, 1].semilogy(xf[:30], 2.0/N * np.abs(yf[:30]),
                                        color=COLOR_RESONANCE, linewidth=1.5)
            self.math_axes[1, 1].set_title("Harmonic Spectrum", color=COLOR_GOLD, fontsize=9)

        self.math_fig.tight_layout(pad=1.5)
        self.math_canvas.draw()

    def _update_insights(self):
        if not self.is_playing:
            return
        insights = self.calculus.get_mathematical_insights()
        if insights and len(insights) > 0:
            current_text = self.insights_text.get("1.0", "end")
            lines = current_text.split('\n')
            if len(lines) > 15:
                self.insights_text.delete("1.0", "end")
                self.insights_text.insert("end", '\n'.join(lines[-15:]))
            new_insight = insights[-1]
            self.insights_text.insert("end", f"\n{new_insight}")
            self.insights_text.see("end")

    def _on_close(self):
        self.is_closing = True
        if self.animation_id:
            self.root.after_cancel(self.animation_id)
        for fig in self.figures:
            plt.close(fig)
        self._save_state()
        self.root.destroy()


def get_content():
    """Return module description for display in GUI."""
    return [
        "CALCULUS IN MOTION",
        "The Mathematics of Violin",
        "",
        "PHILOSOPHY:",
        "  - The violin is not played, it is solved",
        "  - Each bow stroke integrates a differential equation",
        "  - Every finger placement sets a boundary condition",
        "  - What you hear is the convergence of infinite series",
        "",
        "MATHEMATICAL ESSENCE:",
        "  - Bow position: x(t) -> dx/dt -> d2x/dt2",
        "  - String displacement: y(x,t) satisfying wave equation",
        "  - Energy flow: E(t) with dE/dt = eta*P_mech - lambda*E",
        "  - Phase space: (velocity, position) trajectories",
        "",
        "INTERACTIVE CONTROLS:",
        "  - Play/Pause the calculus simulation",
        "  - Adjust bow force and time scale",
        "  - Apply vibrato (frequency modulation)",
        "  - Trigger chaotic bowing regimes",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = CalculusInMotionGUI(parent)
    return gui.root


def main():
    """Standalone entry - launches a full window with mainloop."""
    CalculusInMotionGUI()


if __name__ == "__main__":
    main()