#!/usr/bin/env python3
# ai_violin_master_14_calculus_in_motion.py

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os
import math
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, Arrow, Polygon
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('TkAgg')

# Color constants - more vibrant for energy visualization
COLOR_BG = "#2A1B0A"       # Deep espresso
COLOR_DARK = "#1A0B00"    # Near black
COLOR_GOLD = "#FFE8A0"    # Warm gold
COLOR_FRAME = "#4A2B1A"   # Rich mahogany
COLOR_TEXT = "#FFE8C8"    # Cream
COLOR_ENERGY = "#FF6B6B"  # Energy red
COLOR_FLOW = "#4ECDC4"    # Flow cyan
COLOR_RESONANCE = "#FFD166"  # Resonance yellow
COLOR_HARMONY = "#9D65C9"    # Harmony purple

# Font constants (adjusted for 640px width)
FONT_HEADER = ("Georgia", 14, "bold")  # Reduced from 16
FONT_SECTION = ("Georgia", 11, "italic")  # Reduced from 13
FONT_TEXT = ("Georgia", 10)  # Reduced from 11
FONT_SMALL = ("Georgia", 8)  # Reduced from 9
FONT_MATH = ("Cambria Math", 10)  # Reduced from 12

# Window size
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 1020
WINDOW_SIZE = f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"

class ViolinCalculus:
    """Multivariate calculus visualization of violin performance."""

    def __init__(self):
        # State space: [x_bow, v_bow, x_finger, v_finger, θ_string, ω_string, E_sound]
        self.state = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Physical parameters
        self.params = {
            'm_bow': 0.06,      # kg (bow mass)
            'k_bow': 50.0,      # N/m (bow "spring" from arm)
            'c_bow': 0.3,       # N·s/m (bow damping)
            'm_string': 0.002,  # kg (string segment)
            'k_string': 1000.0, # N/m (string tension)
            'c_string': 0.02,   # N·s/m (string damping)
            'coupling': 0.8,    # Bow-string coupling
            'energy_eff': 0.15, # Efficiency: mechanical → sound
            'resonance_freq': 440.0  # Hz
        }

        # Time tracking
        self.time = 0.0
        self.dt = 0.001
        self.history = []

    def system_equations(self, t, state):
        """Differential equations for violin system."""
        x_bow, v_bow, x_finger, v_finger, θ_string, ω_string, E_sound = state

        # 1. Bow dynamics (2nd order ODE)
        # m_bow·a_bow = -k_bow·x_bow - c_bow·v_bow + F_arm + F_coupling
        F_arm = 2.0 * math.sin(2 * math.pi * 0.5 * t)  # Arm force (simulated)
        F_coupling = self.params['coupling'] * (x_finger - x_bow)

        a_bow = (-self.params['k_bow'] * x_bow -
                 self.params['c_bow'] * v_bow +
                 F_arm + F_coupling) / self.params['m_bow']

        # 2. Finger dynamics (simple harmonic oscillator)
        a_finger = (-self.params['k_string'] * x_finger -
                    self.params['c_string'] * v_finger -
                    F_coupling) / self.params['m_string']

        # 3. String angular motion (pitch)
        # d²θ/dt² = -ω₀²·θ - γ·dθ/dt + excitation
        ω0 = 2 * math.pi * self.params['resonance_freq']
        γ = 0.1
        excitation = 0.5 * v_bow * math.cos(ω_string * t)

        α_string = -ω0**2 * θ_string - γ * ω_string + excitation

        # 4. Sound energy accumulation
        # dE/dt = η·(kinetic_energy) - λ·E
        kinetic = 0.5 * self.params['m_bow'] * v_bow**2
        λ = 0.2  # Sound decay rate
        dE_sound = self.params['energy_eff'] * kinetic - λ * E_sound

        return [v_bow, a_bow, v_finger, a_finger, ω_string, α_string, dE_sound]

    def update(self, bow_force=0.0, finger_position=0.0, time_step=0.01):
        """Update system state."""
        # Store current state
        self.history.append({
            'time': self.time,
            'state': self.state.copy(),
            'energy_kinetic': 0.5 * self.params['m_bow'] * self.state[1]**2,
            'energy_potential': 0.5 * self.params['k_bow'] * self.state[0]**2,
            'energy_sound': self.state[6],
            'pitch': self.params['resonance_freq'] * (1 + 0.1 * math.sin(self.state[4]))
        })

        # Keep history manageable
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        # Update time
        self.time += time_step

        # Solve ODE for next state
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
        """Generate phase portrait between two variables."""
        if len(self.history) < 10:
            return None

        v1 = [h['state'][variable1] for h in self.history[-100:]]
        v2 = [h['state'][variable2] for h in self.history[-100:]]

        return {
            'x': v1,
            'y': v2,
            'dx': np.gradient(v1),
            'dy': np.gradient(v2)
        }

    def get_energy_flow(self):
        """Calculate energy flow through system."""
        if len(self.history) < 2:
            return None

        recent = self.history[-50:]

        # Energy derivatives (power)
        dEk_dt = np.gradient([h['energy_kinetic'] for h in recent])
        dEp_dt = np.gradient([h['energy_potential'] for h in recent])
        dEs_dt = np.gradient([h['energy_sound'] for h in recent])

        # Total power flow
        power_mech = dEk_dt + dEp_dt
        power_sound = dEs_dt

        # Energy efficiency
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
            'energy_total': [h['energy_kinetic'] + h['energy_potential'] + h['energy_sound']
                           for h in recent]
        }

    def get_attractor(self):
        """Calculate system attractor (strange attractor for chaotic bowing)."""
        # Lorenz-like attractor for bowing dynamics
        σ = 10.0
        ρ = 28.0
        β = 8.0/3.0

        if len(self.history) < 100:
            return None

        # Use state variables to seed attractor
        x = [h['state'][1] for h in self.history[-100:]]  # Bow velocity
        y = [h['state'][4] for h in self.history[-100:]]  # String angle
        z = [h['state'][6] for h in self.history[-100:]]  # Sound energy

        # Simple chaotic mapping
        attractor_x = []
        attractor_y = []
        attractor_z = []

        for i in range(len(x)-1):
            dx = σ * (y[i] - x[i])
            dy = x[i] * (ρ - z[i]) - y[i]
            dz = x[i] * y[i] - β * z[i]

            attractor_x.append(x[i] + dx * 0.01)
            attractor_y.append(y[i] + dy * 0.01)
            attractor_z.append(z[i] + dz * 0.01)

        return {
            'x': attractor_x,
            'y': attractor_y,
            'z': attractor_z
        }

    def reset(self):
        """Reset system to initial state."""
        self.state = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.time = 0.0
        self.history = []

    def set_resonance(self, frequency):
        """Set string resonance frequency."""
        self.params['resonance_freq'] = frequency

    def apply_vibrato(self, rate_hz=6.0, depth_cents=30.0):
        """Apply vibrato to string resonance."""
        modulation = depth_cents / 1200.0
        self.params['resonance_freq'] = 440.0 * (2 ** (modulation * math.sin(2 * math.pi * rate_hz * self.time)))

    def get_mathematical_insights(self):
        """Generate mathematical insights about current state."""
        if len(self.history) < 10:
            return []

        insights = []

        # 1. Phase space insight
        phase = self.get_phase_portrait(1, 4)  # Bow velocity vs string angle
        if phase:
            # Calculate divergence in phase space
            div = sum(phase['dx'][i] + phase['dy'][i] for i in range(len(phase['dx']))) / len(phase['dx'])
            if div < 0:
                insights.append("🎯 Phase space contracting: System approaching stable limit cycle")
            else:
                insights.append("🌀 Phase space expanding: Rich, expressive dynamics")

        # 2. Energy conservation insight
        energy_flow = self.get_energy_flow()
        if energy_flow and len(energy_flow['energy_total']) > 5:
            energy_change = abs(energy_flow['energy_total'][-1] - energy_flow['energy_total'][0])
            if energy_change < 0.1:
                insights.append("⚖️ Energy conserved: Perfect mechanical-acoustic coupling")
            else:
                insights.append(f"⚡ Energy flowing: {energy_change:.2f} J transferred")

        # 3. Harmonic insight
        recent_pitch = [h['pitch'] for h in self.history[-20:]]
        pitch_variance = np.var(recent_pitch)
        if pitch_variance < 1.0:
            insights.append("🎵 Pure harmonic: d²θ/dt² approaching zero")
        else:
            insights.append(f"🎶 Rich harmonics: Pitch variance = {pitch_variance:.2f} Hz²")

        # 4. Chaotic insight
        attractor = self.get_attractor()
        if attractor:
            # Calculate fractal dimension approximation
            x_range = max(attractor['x']) - min(attractor['x'])
            if x_range > 10:
                insights.append("🌪️ Chaotic attractor: Sensitive dependence on initial conditions")

        return insights

class CalculusInMotionGUI:
    """GUI visualizing violin as multivariate calculus in motion."""

    def __init__(self, parent):
        self.parent = parent
        self.calculus = ViolinCalculus()
        self.is_playing = False
        self.playback_speed = 1.0
        self.visualization_mode = "energy_flow"

        # Store animation ID for cleanup
        self.animation_id = None
        self.is_closing = False

        # Set window properties
        parent.geometry(WINDOW_SIZE)
        parent.resizable(False, True)  # Fixed width, scrollable height
        parent.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Create main frame with scrollbar
        main_container = tk.Frame(parent, bg=COLOR_BG)
        main_container.pack(fill="both", expand=True)

        # Create canvas and scrollbar
        self.canvas = tk.Canvas(main_container, bg=COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=self.canvas.yview)

        self.scrollable_frame = tk.Frame(self.canvas, bg=COLOR_BG, width=WINDOW_WIDTH-20)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((10, 0), window=self.scrollable_frame, anchor="nw", width=WINDOW_WIDTH-20)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel for scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Initialize visualization
        self.figures = []
        self.create_widgets()
        self.start_animation()

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_widgets(self):
        """Create all GUI widgets."""
        # Header with poetic description
        header_frame = tk.Frame(self.scrollable_frame, bg=COLOR_BG)
        header_frame.pack(fill="x", pady=(0, 10), padx=10)

        tk.Label(header_frame,
                text="🎻 CALCULUS IN MOTION",
                font=FONT_HEADER,
                fg=COLOR_GOLD,
                bg=COLOR_BG).pack()

        poetic_text = """The violin is not an instrument, but a differential equation.
Each bow stroke solves for beauty. Each finger placement integrates time.
What you hear is not music, but the convergence of infinite series."""

        tk.Label(header_frame,
                text=poetic_text,
                font=FONT_SECTION,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                wraplength=WINDOW_WIDTH-40,  # Adjusted for width
                justify=tk.CENTER).pack(pady=10)

        # Separator
        sep = tk.Frame(self.scrollable_frame, height=2, bg=COLOR_GOLD)
        sep.pack(fill="x", padx=10, pady=5)

        # Mathematical essence
        essence_frame = tk.Frame(self.scrollable_frame, bg=COLOR_BG)
        essence_frame.pack(fill="x", padx=10, pady=10)

        essence_text = """🧮 THE ESSENCE:

Bow position: x(t)        → Velocity: dx/dt        → Acceleration: d²x/dt²
Finger placement: θ(t)    → Pitch change: dθ/dt   → Intonation stability: d²θ/dt²
String displacement: y(x,t) → Wave equation: ∂²y/∂t² = c²∂²y/∂x²
Energy flow: E(t)         → Power: dE/dt          → Efficiency: η = ∫sound/∫mechanical

Every master violinist is solving these equations in real-time,
with their body as the computer, their soul as the initial condition."""

        tk.Label(essence_frame,
                text=essence_text,
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_DARK,
                relief=tk.SUNKEN,
                borderwidth=2,
                padx=15,
                pady=10,
                justify=tk.LEFT,
                wraplength=WINDOW_WIDTH-50).pack(fill="x")  # Adjusted for width

        # Control Panel
        self.create_control_panel()

        # Main Visualization
        self.create_main_visualization()

        # Mathematical Visualizations
        self.create_mathematical_visualizations()

        # Insights Panel
        self.create_insights_panel()

        # Action buttons
        self.create_action_buttons()

    def create_control_panel(self):
        """Create the control panel."""
        control_frame = tk.LabelFrame(self.scrollable_frame,
                                    text="🎛️ CONTROL THE DIFFERENTIAL EQUATIONS",
                                    font=FONT_SECTION,
                                    fg=COLOR_GOLD,
                                    bg=COLOR_BG,
                                    relief=tk.RIDGE,
                                    borderwidth=2)
        control_frame.pack(fill="x", padx=10, pady=10, ipadx=5, ipady=5)

        # Play/Pause controls
        play_frame = tk.Frame(control_frame, bg=COLOR_BG)
        play_frame.pack(fill="x", padx=5, pady=5)

        self.play_btn = tk.Button(play_frame,
                                text="▶️ PLAY CALCULUS",
                                font=FONT_TEXT,
                                fg=COLOR_GOLD,
                                bg=COLOR_DARK,
                                command=self.toggle_play)
        self.play_btn.pack(side="left", padx=2)

        reset_btn = tk.Button(play_frame,
                            text="🔄 RESET SYSTEM",
                            font=FONT_TEXT,
                            fg=COLOR_GOLD,
                            bg=COLOR_DARK,
                            command=self.reset_system)
        reset_btn.pack(side="left", padx=2)

        # Speed control
        speed_frame = tk.Frame(control_frame, bg=COLOR_BG)
        speed_frame.pack(fill="x", padx=5, pady=5)

        tk.Label(speed_frame,
                text="Time Scale:",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG).pack(side="left")

        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = tk.Scale(speed_frame,
                              from_=0.1,
                              to=5.0,
                              variable=self.speed_var,
                              orient=tk.HORIZONTAL,
                              resolution=0.1,
                              length=180,  # Reduced from 200
                              bg=COLOR_BG,
                              fg=COLOR_TEXT,
                              troughcolor=COLOR_DARK,
                              highlightbackground=COLOR_BG,
                              command=self.update_speed)
        speed_scale.pack(side="left", padx=5, fill="x", expand=True)

        # Bow force control
        force_frame = tk.Frame(control_frame, bg=COLOR_BG)
        force_frame.pack(fill="x", padx=5, pady=5)

        tk.Label(force_frame,
                text="Bow Force (F):",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG).pack(side="left")

        self.force_var = tk.DoubleVar(value=0.0)

        def update_force(val):
            self.calculus.params['coupling'] = float(val) / 10.0

        force_scale = tk.Scale(force_frame,
                              from_=0.0,
                              to=10.0,
                              variable=self.force_var,
                              orient=tk.HORIZONTAL,
                              resolution=0.1,
                              length=180,  # Reduced from 200
                              bg=COLOR_BG,
                              fg=COLOR_TEXT,
                              troughcolor=COLOR_DARK,
                              highlightbackground=COLOR_BG,
                              command=update_force)
        force_scale.pack(side="left", padx=5, fill="x", expand=True)

        # Visualization mode
        viz_frame = tk.Frame(control_frame, bg=COLOR_BG)
        viz_frame.pack(fill="x", padx=5, pady=5)

        tk.Label(viz_frame,
                text="View:",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG).pack(side="left")

        self.viz_var = tk.StringVar(value="energy_flow")

        modes = [
            ("energy_flow", "⚡ Energy"),
            ("phase_space", "🌀 Phase"),
            ("attractor", "🌪️ Attractor"),
            ("all", "🎭 All")
        ]

        for mode, label in modes:
            tk.Radiobutton(viz_frame,
                          text=label,
                          variable=self.viz_var,
                          value=mode,
                          font=FONT_SMALL,
                          fg=COLOR_TEXT,
                          bg=COLOR_BG,
                          selectcolor=COLOR_DARK,
                          command=self.update_visualization_mode).pack(side="left", padx=5)

    def create_main_visualization(self):
        """Create the main visualization canvas."""
        viz_frame = tk.LabelFrame(self.scrollable_frame,
                                text="🎨 REAL-TIME CALCULUS VISUALIZATION",
                                font=FONT_SECTION,
                                fg=COLOR_GOLD,
                                bg=COLOR_BG,
                                relief=tk.RIDGE,
                                borderwidth=2)
        viz_frame.pack(fill="x", padx=10, pady=10, ipadx=5, ipady=5)

        # Create matplotlib figure for main visualization
        self.main_fig, self.main_ax = plt.subplots(figsize=(6, 3))  # Reduced from (8, 4)
        self.main_fig.patch.set_facecolor(COLOR_DARK)
        self.main_ax.set_facecolor(COLOR_DARK)

        self.main_canvas = FigureCanvasTkAgg(self.main_fig, viz_frame)
        self.main_canvas.get_tk_widget().pack(fill="x", expand=True, padx=5, pady=5)

        # Initial plot
        self.update_main_visualization()

    def create_mathematical_visualizations(self):
        """Create additional mathematical visualizations."""
        math_frame = tk.LabelFrame(self.scrollable_frame,
                                 text="📐 MATHEMATICAL PROJECTIONS",
                                 font=FONT_SECTION,
                                 fg=COLOR_GOLD,
                                 bg=COLOR_BG,
                                 relief=tk.RIDGE,
                                 borderwidth=2)
        math_frame.pack(fill="x", padx=10, pady=10, ipadx=5, ipady=5)

        # Create subplots for different views
        self.math_fig, self.math_axes = plt.subplots(2, 2, figsize=(6, 4.5))  # Reduced from (8, 6)
        self.math_fig.patch.set_facecolor(COLOR_DARK)

        for ax in self.math_axes.flatten():
            ax.set_facecolor(COLOR_DARK)
            ax.tick_params(colors=COLOR_TEXT)
            ax.xaxis.label.set_color(COLOR_TEXT)
            ax.yaxis.label.set_color(COLOR_TEXT)
            ax.title.set_color(COLOR_GOLD)
            ax.title.set_fontsize(9)  # Smaller font for titles

        self.math_canvas = FigureCanvasTkAgg(self.math_fig, math_frame)
        self.math_canvas.get_tk_widget().pack(fill="x", expand=True, padx=5, pady=5)

        # Set titles for subplots
        self.math_axes[0, 0].set_title("Energy Conservation")
        self.math_axes[0, 1].set_title("Phase Space")
        self.math_axes[1, 0].set_title("Power Flow")
        self.math_axes[1, 1].set_title("Harmonic Spectrum")

    def create_insights_panel(self):
        """Create the mathematical insights panel."""
        insights_frame = tk.LabelFrame(self.scrollable_frame,
                                     text="💡 MATHEMATICAL INSIGHTS",
                                     font=FONT_SECTION,
                                     fg=COLOR_GOLD,
                                     bg=COLOR_BG,
                                     relief=tk.RIDGE,
                                     borderwidth=2)
        insights_frame.pack(fill="x", padx=10, pady=10, ipadx=5, ipady=5)

        # Text widget for insights
        self.insights_text = tk.Text(insights_frame,
                                   font=FONT_TEXT,
                                   bg=COLOR_DARK,
                                   fg=COLOR_TEXT,
                                   insertbackground=COLOR_TEXT,
                                   relief=tk.SUNKEN,
                                   borderwidth=1,
                                   height=6,
                                   wrap=tk.WORD)
        self.insights_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Add scrollbar
        scrollbar = tk.Scrollbar(self.insights_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.insights_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.insights_text.yview)

        # Initial insight
        self.insights_text.insert(tk.END, "Initializing differential equations...\n")
        self.insights_text.insert(tk.END, "System ready for multivariate calculus simulation.\n\n")
        self.insights_text.insert(tk.END, "Click 'PLAY CALCULUS' to begin solving the equations of motion.")

    def create_action_buttons(self):
        """Create action buttons."""
        actions_frame = tk.Frame(self.scrollable_frame, bg=COLOR_BG)
        actions_frame.pack(fill="x", padx=10, pady=10)

        # Apply vibrato button
        vibrato_btn = tk.Button(actions_frame,
                              text="🎵 APPLY VIBRATO",
                              font=FONT_TEXT,
                              fg=COLOR_GOLD,
                              bg=COLOR_DARK,
                              command=self.apply_vibrato)
        vibrato_btn.pack(side="left", padx=2, pady=2)

        # Chaotic bowing button
        chaos_btn = tk.Button(actions_frame,
                            text="🌪️ CHAOTIC BOWING",
                            font=FONT_TEXT,
                            fg=COLOR_GOLD,
                            bg=COLOR_DARK,
                            command=self.chaotic_bowing)
        chaos_btn.pack(side="left", padx=2, pady=2)

        # Save state button
        save_btn = tk.Button(actions_frame,
                           text="💾 SAVE STATE",
                           font=FONT_TEXT,
                           fg=COLOR_GOLD,
                           bg=COLOR_DARK,
                           command=self.save_state)
        save_btn.pack(side="left", padx=2, pady=2)

        # Status label
        self.status_var = tk.StringVar(value="✅ System initialized: 7 differential equations ready")
        status_label = tk.Label(actions_frame,
                              textvariable=self.status_var,
                              font=FONT_SMALL,
                              fg=COLOR_TEXT,
                              bg=COLOR_BG)
        status_label.pack(side="right", padx=5)

    def toggle_play(self):
        """Toggle playing/pausing the calculus simulation."""
        self.is_playing = not self.is_playing

        if self.is_playing:
            self.play_btn.config(text="⏸️ PAUSE CALCULUS")
            self.status_var.set("▶️ Solving differential equations...")
        else:
            self.play_btn.config(text="▶️ PLAY CALCULUS")
            self.status_var.set("⏸️ Calculus paused")

    def reset_system(self):
        """Reset the calculus system."""
        self.calculus.reset()
        self.is_playing = False
        self.play_btn.config(text="▶️ PLAY CALCULUS")
        self.update_main_visualization()
        self.update_mathematical_visualizations()

        # Clear insights
        self.insights_text.delete(1.0, tk.END)
        self.insights_text.insert(tk.END, "System reset to initial conditions.\n")
        self.insights_text.insert(tk.END, "All differential equations set to t=0.\n\n")
        self.insights_text.insert(tk.END, "Click 'PLAY CALCULUS' to begin solving.")

        self.status_var.set("✅ System reset: Ready for new initial conditions")

    def update_speed(self, value):
        """Update playback speed."""
        self.playback_speed = float(value)

    def update_visualization_mode(self):
        """Update visualization mode."""
        self.visualization_mode = self.viz_var.get()

    def apply_vibrato(self):
        """Apply vibrato to the system."""
        self.calculus.apply_vibrato()
        self.insights_text.insert(tk.END, "\n🎵 Applied vibrato: Frequency modulation\n")
        self.insights_text.see(tk.END)
        self.status_var.set("🎵 Vibrato: d²θ/dt² modulation active")

    def chaotic_bowing(self):
        """Apply chaotic bowing pattern."""
        # Modify parameters for chaotic behavior
        self.calculus.params['coupling'] = 1.2  # Strong coupling
        self.calculus.params['c_bow'] = 0.1     # Reduced damping

        self.insights_text.insert(tk.END, "\n🌪️ Chaotic bowing: Sensitivity to initial conditions\n")
        self.insights_text.see(tk.END)
        self.status_var.set("🌪️ Chaotic regime: ∇×v ≠ 0")

    def save_state(self):
        """Save current system state."""
        try:
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

            self.insights_text.insert(tk.END, f"\n💾 State saved: {filename}\n")
            self.insights_text.see(tk.END)
            self.status_var.set(f"💾 State saved to {filename}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving state:\n{str(e)}")

    def start_animation(self):
        """Start the animation loop."""
        if not self.is_closing:
            self.animate()
            self.animation_id = self.parent.after(50, self.start_animation)

    def animate(self):
        """Animation update loop."""
        if self.is_playing and not self.is_closing:
            # Update calculus system
            for _ in range(int(10 * self.playback_speed)):
                self.calculus.update(time_step=0.01 * self.playback_speed)

            # Update visualizations
            self.update_main_visualization()
            self.update_mathematical_visualizations()

            # Update insights
            self.update_insights()

    def update_main_visualization(self):
        """Update the main visualization based on current mode."""
        self.main_ax.clear()

        mode = self.visualization_mode

        if mode == "energy_flow" or mode == "all":
            self.plot_energy_flow(self.main_ax)
        elif mode == "phase_space":
            self.plot_phase_space(self.main_ax)
        elif mode == "attractor":
            self.plot_attractor(self.main_ax)

        self.main_ax.set_facecolor(COLOR_DARK)
        self.main_ax.tick_params(colors=COLOR_TEXT, labelsize=8)
        self.main_ax.xaxis.label.set_color(COLOR_TEXT)
        self.main_ax.yaxis.label.set_color(COLOR_TEXT)
        self.main_ax.title.set_color(COLOR_GOLD)
        self.main_ax.title.set_fontsize(10)

        self.main_fig.tight_layout(pad=2)
        self.main_canvas.draw()

    def plot_energy_flow(self, ax):
        """Plot energy flow visualization."""
        if len(self.calculus.history) < 10:
            ax.text(0.5, 0.5, "Solving equations...\nEnergy flow computing",
                   ha='center', va='center', transform=ax.transAxes,
                   color=COLOR_TEXT, fontsize=10)
            ax.set_title("Energy Flow: dE/dt = η·P_mech - λ·E", color=COLOR_GOLD)
            return

        energy_flow = self.calculus.get_energy_flow()
        if not energy_flow:
            return

        times = energy_flow['time']

        # Plot energy components
        ax.plot(times, energy_flow['power_mech'], 'b-', linewidth=1.5, label='Mechanical')
        ax.plot(times, energy_flow['power_sound'], 'r-', linewidth=1.5, label='Sound')
        ax.fill_between(times, 0, energy_flow['power_mech'], alpha=0.3, color='blue')
        ax.fill_between(times, 0, energy_flow['power_sound'], alpha=0.3, color='red')

        # Plot efficiency
        ax2 = ax.twinx()
        ax2.plot(times, energy_flow['efficiency'], 'g--', linewidth=1, label='Efficiency η')
        ax2.set_ylabel('Efficiency', color='green', fontsize=8)
        ax2.tick_params(colors='green', labelsize=7)
        ax2.set_ylim(0, 1)

        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Power (W)', fontsize=8)
        ax.set_title("Energy Flow: dE/dt = η·P_mech - λ·E", color=COLOR_GOLD, fontsize=10)
        ax.legend(loc='upper left', fontsize=7)
        ax2.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3, color=COLOR_TEXT)

    def plot_phase_space(self, ax):
        """Plot phase space visualization."""
        phase = self.calculus.get_phase_portrait(1, 4)  # v_bow vs θ_string
        if not phase:
            ax.text(0.5, 0.5, "Computing phase space...",
                   ha='center', va='center', transform=ax.transAxes,
                   color=COLOR_TEXT, fontsize=10)
            ax.set_title("Phase Space: (v, θ) Trajectory", color=COLOR_GOLD)
            return

        # Create quiver plot for phase space flow
        X, Y = np.meshgrid(np.linspace(min(phase['x']), max(phase['x']), 8),  # Reduced from 10
                          np.linspace(min(phase['y']), max(phase['y']), 8))

        # Simple flow field (for visualization)
        U = -Y  # Simple harmonic oscillator dx/dt = -y
        V = X   # dy/dt = x

        ax.quiver(X, Y, U, V, color=COLOR_FLOW, alpha=0.6, scale=20, width=0.005)

        # Plot actual trajectory
        ax.plot(phase['x'], phase['y'], 'w-', linewidth=1.5, alpha=0.8)
        ax.scatter(phase['x'][-1], phase['y'][-1], c=COLOR_ENERGY, s=30,  # Reduced from 100
                  edgecolors='white', zorder=5)

        ax.set_xlabel('Bow Velocity (v)', fontsize=8)
        ax.set_ylabel('String Angle (θ)', fontsize=8)
        ax.set_title("Phase Space: dv/dt vs dθ/dt", color=COLOR_GOLD, fontsize=10)
        ax.grid(True, alpha=0.3, color=COLOR_TEXT)

    def plot_attractor(self, ax):
        """Plot strange attractor visualization."""
        attractor = self.calculus.get_attractor()
        if not attractor:
            ax.text(0.5, 0.5, "Generating attractor...",
                   ha='center', va='center', transform=ax.transAxes,
                   color=COLOR_TEXT, fontsize=10)
            ax.set_title("Strange Attractor: Chaotic Bowing", color=COLOR_GOLD)
            return

        try:
            from mpl_toolkits.mplot3d import Axes3D
            # Create 3D plot if possible
            ax.remove()
            ax = self.main_fig.add_subplot(111, projection='3d')

            # Plot attractor
            ax.plot(attractor['x'], attractor['y'], attractor['z'],
                   color=COLOR_HARMONY, linewidth=0.8, alpha=0.8)
            ax.scatter(attractor['x'][-1], attractor['y'][-1], attractor['z'][-1],
                      c=COLOR_ENERGY, s=30, edgecolors='white')  # Reduced from 100

            ax.set_xlabel('X (Bow)', fontsize=7)
            ax.set_ylabel('Y (String)', fontsize=7)
            ax.set_zlabel('Z (Energy)', fontsize=7)
            ax.set_title("Strange Attractor: ∇×F ≠ 0", color=COLOR_GOLD, fontsize=10)

            # Set 3D axis colors
            ax.xaxis.label.set_color(COLOR_TEXT)
            ax.yaxis.label.set_color(COLOR_TEXT)
            ax.zaxis.label.set_color(COLOR_TEXT)
            ax.tick_params(colors=COLOR_TEXT, labelsize=6)

            self.main_ax = ax
        except Exception as e:
            # Fallback to 2D plot if 3D fails
            ax.plot(attractor['x'], attractor['y'], color=COLOR_HARMONY, linewidth=1)
            ax.scatter(attractor['x'][-1], attractor['y'][-1], c=COLOR_ENERGY, s=30)
            ax.set_title("Attractor Projection", color=COLOR_GOLD, fontsize=10)

    def update_mathematical_visualizations(self):
        """Update all mathematical visualizations."""
        if len(self.calculus.history) < 10:
            return

        # Clear all axes
        for ax in self.math_axes.flatten():
            ax.clear()
            ax.set_facecolor(COLOR_DARK)
            ax.tick_params(colors=COLOR_TEXT, labelsize=6)
            ax.xaxis.label.set_color(COLOR_TEXT)
            ax.yaxis.label.set_color(COLOR_TEXT)
            ax.title.set_color(COLOR_GOLD)
            ax.title.set_fontsize(9)

        # Plot 1: Energy conservation
        self.plot_energy_conservation(self.math_axes[0, 0])

        # Plot 2: Phase space
        self.plot_detailed_phase_space(self.math_axes[0, 1])

        # Plot 3: Power flow
        self.plot_power_flow(self.math_axes[1, 0])

        # Plot 4: Harmonic spectrum
        self.plot_harmonic_spectrum(self.math_axes[1, 1])

        # Adjust layout and draw
        self.math_fig.tight_layout(pad=1.5)
        self.math_canvas.draw()

    def plot_energy_conservation(self, ax):
        """Plot energy conservation visualization."""
        recent = self.calculus.history[-100:] if len(self.calculus.history) >= 100 else self.calculus.history

        times = [h['time'] for h in recent]
        kinetic = [h['energy_kinetic'] for h in recent]
        potential = [h['energy_potential'] for h in recent]
        sound = [h['energy_sound'] for h in recent]
        total = [k + p + s for k, p, s in zip(kinetic, potential, sound)]

        ax.stackplot(times, kinetic, potential, sound,
                    colors=[COLOR_ENERGY, COLOR_RESONANCE, COLOR_HARMONY],
                    alpha=0.7, labels=['Kinetic', 'Potential', 'Sound'])

        ax.plot(times, total, 'w-', linewidth=1.5, label='Total')

        ax.set_xlabel('Time (s)', fontsize=7)
        ax.set_ylabel('Energy (J)', fontsize=7)
        ax.set_title("Energy Conservation", color=COLOR_GOLD, fontsize=9)
        ax.legend(loc='upper right', fontsize=6)
        ax.grid(True, alpha=0.3, color=COLOR_TEXT)

    def plot_detailed_phase_space(self, ax):
        """Plot detailed phase space visualization."""
        phase = self.calculus.get_phase_portrait(1, 4)
        if not phase:
            return

        # Create streamplot for phase space flow
        x_min, x_max = min(phase['x']), max(phase['x'])
        y_min, y_max = min(phase['y']), max(phase['y'])

        # Extend range slightly
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range

        # Create grid
        Y, X = np.meshgrid(np.linspace(y_min, y_max, 15),  # Reduced from 20
                          np.linspace(x_min, x_max, 15))

        # Simple harmonic oscillator flow (for visualization)
        U = -Y  # dx/dt = -y
        V = X   # dy/dt = x

        ax.streamplot(X, Y, U, V, color=COLOR_FLOW, linewidth=0.5, arrowsize=0.8, density=1)

        # Plot actual trajectory with color indicating time
        colors = np.linspace(0, 1, len(phase['x']))
        points = np.array([phase['x'], phase['y']]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap=plt.cm.viridis, linewidth=1.5)
        lc.set_array(colors)
        ax.add_collection(lc)

        # Mark current state
        ax.scatter(phase['x'][-1], phase['y'][-1], c='white', s=20,  # Reduced from 50
                  edgecolors=COLOR_ENERGY, linewidth=1, zorder=5)

        ax.set_xlabel('Bow Velocity', fontsize=7)
        ax.set_ylabel('String Angle', fontsize=7)
        ax.set_title("Phase Space Trajectory", color=COLOR_GOLD, fontsize=9)
        ax.grid(True, alpha=0.3, color=COLOR_TEXT)

    def plot_power_flow(self, ax):
        """Plot power flow visualization."""
        energy_flow = self.calculus.get_energy_flow()
        if not energy_flow:
            return

        times = energy_flow['time']

        # Create waterfall plot of power flow
        width = (times[1] - times[0]) * 0.8 if len(times) > 1 else 0.01

        # Plot mechanical and sound power as bars
        ax.bar(times, energy_flow['power_mech'], width=width,
              color=COLOR_ENERGY, alpha=0.6, label='Mechanical')
        ax.bar(times, energy_flow['power_sound'], width=width,
              color=COLOR_HARMONY, alpha=0.6, label='Sound',
              bottom=energy_flow['power_mech'])

        ax.set_xlabel('Time (s)', fontsize=7)
        ax.set_ylabel('Power (W)', fontsize=7)
        ax.set_title("Power Flow", color=COLOR_GOLD, fontsize=9)
        ax.legend(loc='upper left', fontsize=6)
        ax.grid(True, alpha=0.3, color=COLOR_TEXT)

    def plot_harmonic_spectrum(self, ax):
        """Plot harmonic spectrum visualization."""
        if len(self.calculus.history) < 20:
            return

        # Get pitch history
        pitches = [h['pitch'] for h in self.calculus.history[-100:]]

        # Compute FFT
        from scipy.fft import fft, fftfreq

        N = len(pitches)
        T = (self.calculus.history[-1]['time'] - self.calculus.history[-100]['time']) / 100

        yf = fft(pitches)
        xf = fftfreq(N, T)[:N//2]

        # Plot spectrum
        ax.semilogy(xf[:30], 2.0/N * np.abs(yf[:30]),  # Reduced from 50
                   color=COLOR_RESONANCE, linewidth=1.5)
        ax.fill_between(xf[:30], 0, 2.0/N * np.abs(yf[:30]),
                       alpha=0.3, color=COLOR_RESONANCE)

        # Mark fundamental frequency
        fundamental_idx = np.argmax(2.0/N * np.abs(yf[:30]))
        fundamental_freq = xf[fundamental_idx]

        ax.axvline(x=fundamental_freq, color='white', linestyle='--', alpha=0.7,
                  label=f'Fund: {fundamental_freq:.0f} Hz')

        ax.set_xlabel('Frequency (Hz)', fontsize=7)
        ax.set_ylabel('Amplitude', fontsize=7)
        ax.set_title("Harmonic Spectrum", color=COLOR_GOLD, fontsize=9)
        ax.legend(loc='upper right', fontsize=6)
        ax.grid(True, alpha=0.3, color=COLOR_TEXT)

    def update_insights(self):
        """Update mathematical insights."""
        if not self.is_playing:
            return

        insights = self.calculus.get_mathematical_insights()

        if insights and len(insights) > 0:
            # Clear old insights (keep first few lines)
            current_text = self.insights_text.get(1.0, tk.END)
            lines = current_text.split('\n')

            # Keep only recent insights (last 10 lines)
            if len(lines) > 15:
                self.insights_text.delete(1.0, tk.END)
                self.insights_text.insert(tk.END, '\n'.join(lines[-15:]))

            # Add new insight
            new_insight = insights[-1]  # Get most recent insight
            self.insights_text.insert(tk.END, f"\n{new_insight}")
            self.insights_text.see(tk.END)

    def on_closing(self):
        """Clean up on closing."""
        self.is_closing = True

        # Cancel any pending animation
        if self.animation_id:
            self.parent.after_cancel(self.animation_id)

        # Close matplotlib figures
        plt.close('all')

        # Destroy the window
        self.parent.destroy()

def get_content():
    """Return module description for display in GUI."""
    return [
        "╔══════════════════════════════════════╗",
        "║       CALCULUS IN MOTION             ║",
        "║       The Mathematics of Violin      ║",
        "╚══════════════════════════════════════╝",
        "",
        "🎯 PHILOSOPHY:",
        "  • The violin is not played, it is solved",
        "  • Each bow stroke integrates a differential equation",
        "  • Every finger placement sets a boundary condition",
        "  • What you hear is the convergence of infinite series",
        "",
        "🧮 MATHEMATICAL ESSENCE:",
        "  • Bow position: x(t) → dx/dt → d²x/dt²",
        "  • String displacement: y(x,t) satisfying ∂²y/∂t² = c²∂²y/∂x²",
        "  • Energy flow: E(t) with dE/dt = η·P_mech - λ·E",
        "  • Phase space: (velocity, position) trajectories",
        "  • Strange attractors for expressive playing",
        "",
        "🎨 VISUALIZATION:",
        "  • Real-time solving of 7 coupled differential equations",
        "  • Energy flow from arm to sound waves",
        "  • Phase space portraits of bow-string interaction",
        "  • Strange attractors for chaotic bowing",
        "  • Harmonic spectrum from Fourier analysis",
        "",
        "⚡ INTERACTIVE CONTROLS:",
        "  • Play/Pause the calculus simulation",
        "  • Adjust bow force and time scale",
        "  • Apply vibrato (frequency modulation)",
        "  • Trigger chaotic bowing regimes",
        "  • View different mathematical projections",
        "",
        "💡 INSIGHTS GENERATED:",
        "  • Real-time analysis of mathematical properties",
        "  • Phase space contraction/expansion",
        "  • Energy conservation violations",
        "  • Harmonic richness assessment",
        "  • Chaotic vs regular dynamics",
        "",
        "🚀 HOW TO EXPERIENCE:",
        "  1. Click 'PLAY CALCULUS' to start solving equations",
        "  2. Adjust bow force to change coupling strength",
        "  3. Switch visualization modes for different perspectives",
        "  4. Apply vibrato or chaotic bowing for complex dynamics",
        "  5. Watch the mathematical insights appear in real-time",
        "",
        "🎻 FOR VIOLINISTS:",
        "  This is what happens in your brain and body when you play.",
        "  The equations are being solved subconsciously.",
        "  Mastery is when these solutions become elegant and efficient.",
    ]

def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    # Create the calculus in motion GUI
    gui = CalculusInMotionGUI(parent)

    # Set close handler
    parent.protocol("WM_DELETE_WINDOW", gui.on_closing)

    return gui.scrollable_frame

def test_standalone():
    """Test the module GUI standalone."""
    root = tk.Tk()
    root.title("Calculus in Motion: The Mathematics of Violin - Standalone Test")
    root.configure(bg=COLOR_BG)

    # Create the GUI
    gui = create_gui(root)

    root.mainloop()

if __name__ == "__main__":
    test_standalone()