#!/usr/bin/env python3
# ai_violin_master_13_violin_mathematics.py

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import math
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')

# Color constants matching main script
COLOR_BG = "#5A381E"       # deep violin-wood brown
COLOR_DARK = "#3B2413"    # darker wood
COLOR_GOLD = "#FFD770"    # gold text
COLOR_FRAME = "#4B2E18"   # mid-tone wood
COLOR_TEXT = "#FFF2CF"    # parchment text

# Font constants (reduced for 640px width)
FONT_HEADER = ("Georgia", 12, "bold")  # Reduced from 14
FONT_SECTION = ("Georgia", 10, "italic")  # Reduced from 12
FONT_TEXT = ("Georgia", 9)  # Reduced from 11
FONT_SMALL = ("Georgia", 8)  # Reduced from 9

# Window size - fixed width, scrollable height
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 1020
WINDOW_SIZE = f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}"

class ViolinMathematics:
    """Comprehensive mathematical modeling of violin performance."""

    def __init__(self):
        # Mathematical models for each aspect
        self.models = {
            "string_vibration": self.string_vibration_model(),
            "bow_fluid_dynamics": self.bow_fluid_model(),
            "intonation_calculus": self.intonation_calculus_model(),
            "vibrato_wave": self.vibrato_wave_model(),
            "energy_transfer": self.energy_transfer_model(),
            "harmonic_analysis": self.harmonic_analysis_model()
        }

        # Physical constants
        self.constants = {
            "string_tension": 55,      # Newtons (typical violin)
            "string_density": 0.0005,  # kg/m (steel core)
            "string_length": 0.33,     # meters
            "air_density": 1.225,      # kg/m³
            "sound_speed": 343,        # m/s
            "bow_hair_friction": 0.4,  # coefficient
            "violin_body_resonance": 440  # Hz (A4)
        }

    def string_vibration_model(self):
        """Wave equation for string vibration."""
        def wave_equation(x, t, mode=1):
            # ∂²y/∂t² = c² ∂²y/∂x²
            c = math.sqrt(self.constants["string_tension"] /
                         self.constants["string_density"])
            L = self.constants["string_length"]

            # Standing wave solution: y(x,t) = A sin(nπx/L) cos(ωt + φ)
            A = 0.001  # Amplitude (1mm)
            n = mode
            ω = n * math.pi * c / L  # Angular frequency

            return A * math.sin(n * math.pi * x / L) * math.cos(ω * t)

        return {
            "name": "String Vibration Wave Equation",
            "equation": "∂²y/∂t² = c² ∂²y/∂x²",
            "description": "Describes transverse waves on violin string",
            "parameters": ["x (position)", "t (time)", "mode (harmonic)"],
            "function": wave_equation
        }

    def bow_fluid_model(self):
        """Fluid dynamics model of bow movement."""
        def bow_flow(v_bow, pressure, angle_deg):
            # Simplified fluid model: Flow separation and vortex shedding
            angle = math.radians(angle_deg)

            # Reynolds number for bow hair interaction
            hair_diameter = 0.0002  # 0.2mm hair diameter
            kinematic_viscosity = 1.5e-5  # m²/s for air
            Re = v_bow * hair_diameter / kinematic_viscosity

            # Flow separation angle (simplified)
            if Re > 2000:
                # Turbulent flow
                separation_factor = 0.3 + 0.2 * math.sin(angle)
            else:
                # Laminar flow
                separation_factor = 0.1 + 0.05 * math.sin(angle)

            # Tone quality based on flow characteristics
            tone_quality = 0.8 - separation_factor * 0.4

            return {
                "reynolds": Re,
                "flow_state": "turbulent" if Re > 2000 else "laminar",
                "separation_factor": separation_factor,
                "tone_quality": tone_quality,
                "energy_efficiency": 0.9 - separation_factor * 0.3
            }

        return {
            "name": "Bow Fluid Dynamics",
            "equation": "Re = v·d/ν, Flow Separation = f(θ,Re)",
            "description": "Models air flow around bow hair and rosin interaction",
            "parameters": ["v_bow (m/s)", "pressure (N)", "angle (deg)"],
            "function": bow_flow
        }

    def intonation_calculus_model(self):
        """Calculus model for pitch accuracy."""
        def pitch_accuracy(target_freq, actual_freq, duration):
            # Rate of change of pitch error
            error = actual_freq - target_freq

            # Convert to cents
            cents_error = 1200 * math.log2(actual_freq / target_freq)

            # First derivative (rate of change)
            # Simulate typical pitch drift
            time_points = np.linspace(0, duration, 100)

            # Model different types of intonation issues
            if abs(cents_error) < 5:
                # Good intonation: minimal drift
                drift = 0.1 * np.sin(2 * np.pi * 0.5 * time_points)
                stability = 0.95
            elif abs(cents_error) < 20:
                # Moderate: some drift
                drift = 0.5 * np.sin(2 * np.pi * 2 * time_points)
                stability = 0.75
            else:
                # Poor: significant drift
                drift = 2.0 * np.random.randn(len(time_points))
                stability = 0.4

            return {
                "cents_error": cents_error,
                "stability_score": stability,
                "drift_pattern": drift.tolist(),
                "intonation_grade": self._grade_intonation(abs(cents_error))
            }

        return {
            "name": "Intonation Calculus",
            "equation": "dθ/dt → 0, Δcents = 1200·log₂(f_actual/f_target)",
            "description": "Models rate of pitch change and stability",
            "parameters": ["target_freq", "actual_freq", "duration"],
            "function": pitch_accuracy
        }

    def vibrato_wave_model(self):
        """Wave model for vibrato."""
        def vibrato_profile(rate_hz, width_cents, phase=0):
            # Vibrato as frequency modulation
            t = np.linspace(0, 2, 200)  # 2 seconds

            # Base frequency (A4)
            f0 = 440

            # Frequency modulation
            modulation = width_cents / 1200  # Convert cents to factor
            f_t = f0 * (2 ** (modulation * np.sin(2 * np.pi * rate_hz * t + phase)))

            # Calculate wave parameters
            amplitude_envelope = np.exp(-0.5 * t)  # Decay envelope

            return {
                "time": t.tolist(),
                "frequency": f_t.tolist(),
                "modulation_depth": width_cents,
                "modulation_rate": rate_hz,
                "envelope": amplitude_envelope.tolist(),
                "phase_consistency": 0.85 + 0.1 * np.random.random()
            }

        return {
            "name": "Vibrato Wave Model",
            "equation": "f(t) = f₀·2^(Δ·sin(2π·ν·t + φ))",
            "description": "Frequency modulation model for vibrato",
            "parameters": ["rate_hz", "width_cents", "phase"],
            "function": vibrato_profile
        }

    def energy_transfer_model(self):
        """Energy transfer from bow to sound."""
        def energy_analysis(bow_force, bow_velocity, contact_point):
            # Total mechanical energy input
            mechanical_energy = bow_force * bow_velocity

            # Loss mechanisms
            string_damping = 0.15
            bridge_loss = 0.25
            body_loss = 0.35
            radiation_loss = 0.15

            # Efficiency based on contact point (sweet spot near bridge)
            position_factor = abs(contact_point - 0.2) / 0.2  # 0-1, 0=optimal

            # Total efficiency
            total_efficiency = (1 - string_damping) * (1 - bridge_loss) * \
                              (1 - body_loss) * (1 - radiation_loss) * \
                              (1 - position_factor * 0.3)

            # Output sound energy
            sound_energy = mechanical_energy * total_efficiency

            # Frequency distribution (simplified)
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

        return {
            "name": "Energy Transfer Model",
            "equation": "η = (1-α)·(1-β)·(1-γ)·(1-δ)·(1-ε)",
            "description": "Models energy conservation through violin system",
            "parameters": ["bow_force", "bow_velocity", "contact_point"],
            "function": energy_analysis
        }

    def harmonic_analysis_model(self):
        """Fourier analysis of violin tone."""
        def harmonic_spectrum(fundamental, overtone_ratios):
            # Generate harmonic series
            harmonics = []
            amplitudes = []
            phases = []

            for i, ratio in enumerate(overtone_ratios[:8]):  # First 8 harmonics
                harmonic_num = i + 1
                frequency = fundamental * harmonic_num
                amplitude = ratio * (1.0 / harmonic_num)  # Natural decay
                phase = np.random.uniform(0, 2*np.pi)  # Random phase

                harmonics.append(frequency)
                amplitudes.append(amplitude)
                phases.append(phase)

            # Create waveform as sum of harmonics
            t = np.linspace(0, 0.01, 500)  # 10ms window
            waveform = np.zeros_like(t)

            for f, a, p in zip(harmonics, amplitudes, phases):
                waveform += a * np.sin(2 * np.pi * f * t + p)

            # Calculate spectral properties
            total_power = sum(a**2 for a in amplitudes)
            even_harmonics = sum(amplitudes[i]**2 for i in range(0, len(amplitudes), 2))
            odd_harmonics = sum(amplitudes[i]**2 for i in range(1, len(amplitudes), 2))

            return {
                "harmonics": harmonics,
                "amplitudes": amplitudes,
                "waveform": waveform.tolist(),
                "time": t.tolist(),
                "spectral_balance": even_harmonics / total_power,
                "brightness": sum(amplitudes[4:]) / sum(amplitudes[:4]),
                "richness_factor": len([a for a in amplitudes if a > 0.1]) / 8
            }

        return {
            "name": "Harmonic Analysis",
            "equation": "f(t) = Σ Aₙ·sin(2π·n·f₀·t + φₙ)",
            "description": "Fourier decomposition of violin tone",
            "parameters": ["fundamental", "overtone_ratios"],
            "function": harmonic_spectrum
        }

    def _grade_intonation(self, cents_error):
        """Convert cents error to letter grade."""
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

    def calculate_resonance_modes(self):
        """Calculate violin body resonance frequencies."""
        # Simplified plate resonance model
        # Based on Chladni patterns
        modes = []

        # Fundamental modes (simplified)
        base_freq = self.constants["violin_body_resonance"]

        # Mode (m,n) frequencies
        for m in range(1, 4):
            for n in range(1, 4):
                freq = base_freq * math.sqrt(m**2 + n**2)
                modes.append({
                    "mode": f"({m},{n})",
                    "frequency": freq,
                    "chladni_pattern": f"{m}×{n} nodal lines",
                    "description": f"{'Top plate' if (m+n)%2==0 else 'Back plate'} resonance"
                })

        return sorted(modes, key=lambda x: x["frequency"])

    def simulate_bowing_cycle(self, bow_speed_profile, pressure_profile):
        """Simulate complete bowing cycle."""
        time_points = np.linspace(0, 2, 100)  # 2-second bow stroke

        results = []
        for t in time_points:
            # Interpolate parameters
            v = np.interp(t, bow_speed_profile["time"], bow_speed_profile["speed"])
            p = np.interp(t, pressure_profile["time"], pressure_profile["pressure"])

            # Calculate string response
            string_response = self.models["string_vibration"]["function"](0.5, t, 1)

            # Calculate energy transfer
            energy = self.models["energy_transfer"]["function"](p, v, 0.2)

            results.append({
                "time": t,
                "bow_speed": v,
                "bow_pressure": p,
                "string_displacement": string_response,
                "sound_pressure": energy["sound_output"] * 1000  # Scale for visualization
            })

        return results

    def generate_performance_report(self, parameters):
        """Generate comprehensive performance analysis."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "mathematical_models": {},
            "performance_metrics": {},
            "visualization_data": {}
        }

        # Run all models
        for name, model in self.models.items():
            try:
                # Use default parameters for demonstration
                if name == "string_vibration":
                    result = model["function"](0.5, 0.1, 1)
                elif name == "bow_fluid_dynamics":
                    result = model["function"](1.0, 10, 45)
                elif name == "intonation_calculus":
                    result = model["function"](440, 442, 2)
                elif name == "vibrato_wave":
                    result = model["function"](6, 30, 0)
                elif name == "energy_transfer":
                    result = model["function"](10, 0.5, 0.2)
                elif name == "harmonic_analysis":
                    result = model["function"](440, [1.0, 0.5, 0.3, 0.2, 0.1])

                report["mathematical_models"][name] = {
                    "name": model["name"],
                    "equation": model["equation"],
                    "result": result
                }
            except Exception as e:
                report["mathematical_models"][name] = {
                    "error": str(e),
                    "name": model["name"],
                    "equation": model["equation"]
                }

        # Calculate overall performance score
        report["performance_metrics"] = {
            "technical_score": self._calculate_technical_score(report),
            "musicality_score": self._calculate_musicality_score(report),
            "efficiency_score": self._calculate_efficiency_score(report)
        }

        # Generate visualization data
        report["visualization_data"] = self._generate_visualization_data(report)

        return report

    def _calculate_technical_score(self, report):
        """Calculate technical performance score."""
        # Based on model results
        scores = []

        if "intonation_calculus" in report["mathematical_models"]:
            intonation = report["mathematical_models"]["intonation_calculus"]["result"]
            if "stability_score" in intonation:
                scores.append(intonation["stability_score"] * 100)

        if "bow_fluid_dynamics" in report["mathematical_models"]:
            bow = report["mathematical_models"]["bow_fluid_dynamics"]["result"]
            if "tone_quality" in bow:
                scores.append(bow["tone_quality"] * 100)

        return np.mean(scores) if scores else 75.0

    def _calculate_musicality_score(self, report):
        """Calculate musicality score."""
        scores = []

        if "vibrato_wave" in report["mathematical_models"]:
            vibrato = report["mathematical_models"]["vibrato_wave"]["result"]
            if "phase_consistency" in vibrato:
                scores.append(vibrato["phase_consistency"] * 100)

        if "harmonic_analysis" in report["mathematical_models"]:
            harmonics = report["mathematical_models"]["harmonic_analysis"]["result"]
            if "richness_factor" in harmonics:
                scores.append(harmonics["richness_factor"] * 100)

        return np.mean(scores) if scores else 80.0

    def _calculate_efficiency_score(self, report):
        """Calculate energy efficiency score."""
        if "energy_transfer" in report["mathematical_models"]:
            energy = report["mathematical_models"]["energy_transfer"]["result"]
            if "efficiency" in energy:
                return energy["efficiency"] * 100

        return 85.0

    def _generate_visualization_data(self, report):
        """Generate data for visualizations."""
        data = {}

        # Generate time series for string vibration
        t = np.linspace(0, 0.01, 100)
        x = np.linspace(0, 1, 50)

        # String displacement over time and space
        string_displacement = np.zeros((len(t), len(x)))
        for i, time_val in enumerate(t):
            for j, pos_val in enumerate(x):
                string_displacement[i, j] = self.models["string_vibration"]["function"](
                    pos_val, time_val, 1
                )

        data["string_displacement"] = {
            "time": t.tolist(),
            "position": x.tolist(),
            "displacement": string_displacement.tolist()
        }

        return data

class ViolinMathematicsGUI:
    """Comprehensive GUI for Violin Mathematics visualization."""

    def __init__(self, parent):
        self.parent = parent
        self.mathematics = ViolinMathematics()
        self.current_figures = []
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

        self.create_widgets()

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def create_widgets(self):
        """Create all GUI widgets."""
        # Header
        header_frame = tk.Frame(self.scrollable_frame, bg=COLOR_BG)
        header_frame.pack(fill="x", pady=(0, 10), padx=10)

        tk.Label(header_frame,
                text="🎻 VIOLIN MATHEMATICS VISUALIZER",
                font=FONT_HEADER,
                fg=COLOR_GOLD,
                bg=COLOR_BG).pack()

        tk.Label(header_frame,
                text="Visualizing the Calculus, Physics, and Harmony of Violin Performance",
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                wraplength=WINDOW_WIDTH-40).pack()  # Adjusted for width

        # Separator
        sep = tk.Frame(self.scrollable_frame, height=2, bg=COLOR_GOLD)
        sep.pack(fill="x", padx=10, pady=5)

        # Introduction text
        intro_text = """🎯 The violin is a mathematical instrument:

1. CALCULUS: Pitch = θ(t), where dθ/dt → 0 for perfect intonation
2. FLUID DYNAMICS: Bow motion = Energy flow with laminar/turbulent transitions
3. WAVE PHYSICS: String vibration follows ∂²y/∂t² = c²∂²y/∂x²
4. FOURIER ANALYSIS: Tone = Σ Aₙ·sin(2π·n·f₀·t + φₙ)
5. ENERGY CONSERVATION: Mechanical → Acoustic with efficiency η

This visualizer shows these principles in action."""

        intro_frame = tk.Frame(self.scrollable_frame, bg=COLOR_BG)
        intro_frame.pack(fill="x", padx=10, pady=10)

        intro_label = tk.Label(intro_frame,
                             text=intro_text,
                             font=FONT_SMALL,
                             fg=COLOR_TEXT,
                             bg=COLOR_BG,
                             justify=tk.LEFT,
                             wraplength=WINDOW_WIDTH-40)
        intro_label.pack()

        # Create visualization sections
        self.create_string_vibration_section()
        self.create_bow_fluid_section()
        self.create_intonation_calculus_section()
        self.create_vibrato_wave_section()
        self.create_energy_transfer_section()
        self.create_harmonic_analysis_section()
        self.create_resonance_modes_section()
        self.create_bowing_cycle_section()

        # Action buttons
        self.create_action_buttons()

    def create_string_vibration_section(self):
        """Create string vibration visualization section."""
        section_frame = tk.LabelFrame(self.scrollable_frame,
                                    text="1. String Vibration Wave Equation",
                                    font=FONT_SECTION,
                                    fg=COLOR_GOLD,
                                    bg=COLOR_BG,
                                    relief=tk.RIDGE,
                                    borderwidth=2)
        section_frame.pack(fill="x", padx=10, pady=10, ipadx=5, ipady=5)

        # Equation display
        eq_frame = tk.Frame(section_frame, bg=COLOR_BG)
        eq_frame.pack(fill="x", padx=5, pady=5)

        eq_label = tk.Label(eq_frame,
                          text="∂²y/∂t² = c² ∂²y/∂x²",
                          font=("Courier", 12, "bold"),
                          fg=COLOR_GOLD,
                          bg=COLOR_BG)
        eq_label.pack()

        # Description
        desc_label = tk.Label(section_frame,
                            text="Transverse waves on a violin string. The solution gives standing waves with nodes and antinodes.",
                            font=FONT_SMALL,
                            fg=COLOR_TEXT,
                            bg=COLOR_BG,
                            wraplength=WINDOW_WIDTH-60,
                            justify=tk.LEFT)
        desc_label.pack(padx=5, pady=5)

        # Visualization frame
        viz_frame = tk.Frame(section_frame, bg=COLOR_DARK)
        viz_frame.pack(fill="x", padx=5, pady=5, ipadx=2, ipady=2)

        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))  # Reduced from (8, 3)
        fig.patch.set_facecolor(COLOR_DARK)

        # Plot 1: String shape at different times
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
        ax1.set_facecolor(COLOR_DARK)
        ax1.tick_params(colors=COLOR_TEXT, labelsize=7)
        ax1.grid(True, alpha=0.3, color=COLOR_TEXT)

        # Plot 2: Standing wave modes
        modes = [1, 2, 3, 4]
        for n in modes:
            y = np.sin(n * np.pi * x)
            ax2.plot(x, y, linewidth=1.2, label=f'n={n}')

        ax2.set_xlabel('Position', color=COLOR_TEXT, fontsize=8)
        ax2.set_ylabel('Amplitude', color=COLOR_TEXT, fontsize=8)
        ax2.set_title('Standing Wave Modes', color=COLOR_GOLD, fontsize=9)
        ax2.legend(fontsize=7, facecolor=COLOR_DARK, edgecolor=COLOR_TEXT)
        ax2.set_facecolor(COLOR_DARK)
        ax2.tick_params(colors=COLOR_TEXT, labelsize=7)
        ax2.grid(True, alpha=0.3, color=COLOR_TEXT)

        fig.tight_layout(pad=1.5)

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True)

        self.current_figures.append(fig)

    def create_bow_fluid_section(self):
        """Create bow fluid dynamics visualization section."""
        section_frame = tk.LabelFrame(self.scrollable_frame,
                                    text="2. Bow Fluid Dynamics",
                                    font=FONT_SECTION,
                                    fg=COLOR_GOLD,
                                    bg=COLOR_BG,
                                    relief=tk.RIDGE,
                                    borderwidth=2)
        section_frame.pack(fill="x", padx=10, pady=10, ipadx=5, ipady=5)

        # Equation display
        eq_frame = tk.Frame(section_frame, bg=COLOR_BG)
        eq_frame.pack(fill="x", padx=5, pady=5)

        eq_label = tk.Label(eq_frame,
                          text="Re = v·d/ν, Flow Separation = f(θ,Re)",
                          font=("Courier", 10, "bold"),  # Reduced from 12
                          fg=COLOR_GOLD,
                          bg=COLOR_BG)
        eq_label.pack()

        # Description
        desc_label = tk.Label(section_frame,
                            text="Air flow around bow hair creates laminar or turbulent flow affecting tone quality.",
                            font=FONT_SMALL,
                            fg=COLOR_TEXT,
                            bg=COLOR_BG,
                            wraplength=WINDOW_WIDTH-60,
                            justify=tk.LEFT)
        desc_label.pack(padx=5, pady=5)

        # Visualization frame
        viz_frame = tk.Frame(section_frame, bg=COLOR_DARK)
        viz_frame.pack(fill="x", padx=5, pady=5, ipadx=2, ipady=2)

        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))  # Reduced from (8, 3)
        fig.patch.set_facecolor(COLOR_DARK)

        # Plot 1: Flow velocity profile
        y = np.linspace(-0.001, 0.001, 100)  # ±1mm around bow hair
        u_laminar = 1 - (y/0.001)**2
        u_turbulent = 1 - abs(y/0.001)**(1/7)

        ax1.plot(u_laminar, y, 'b-', linewidth=1.5, label='Laminar')
        ax1.plot(u_turbulent, y, 'r--', linewidth=1.5, label='Turbulent')
        ax1.fill_betweenx(y, u_laminar, alpha=0.3, color='blue')
        ax1.fill_betweenx(y, u_turbulent, alpha=0.3, color='red')

        ax1.set_xlabel('Flow Velocity', color=COLOR_TEXT, fontsize=8)
        ax1.set_ylabel('Distance from hair', color=COLOR_TEXT, fontsize=8)
        ax1.set_title('Boundary Layer Profiles', color=COLOR_GOLD, fontsize=9)
        ax1.legend(fontsize=7, facecolor=COLOR_DARK, edgecolor=COLOR_TEXT)
        ax1.set_facecolor(COLOR_DARK)
        ax1.tick_params(colors=COLOR_TEXT, labelsize=7)
        ax1.grid(True, alpha=0.3, color=COLOR_TEXT)

        # Plot 2: Reynolds number vs tone quality
        Re = np.linspace(100, 5000, 50)
        tone_quality = 0.8 - 0.4 * np.tanh((Re - 2000) / 1000)

        ax2.plot(Re, tone_quality, 'g-', linewidth=1.5)
        ax2.axvline(x=2000, color='yellow', linestyle='--', alpha=0.7,
                   label='Transition Re=2000')
        ax2.fill_between(Re, tone_quality, alpha=0.3, color='green')

        ax2.set_xlabel('Reynolds Number', color=COLOR_TEXT, fontsize=8)
        ax2.set_ylabel('Tone Quality', color=COLOR_TEXT, fontsize=8)
        ax2.set_title('Flow Quality vs Re', color=COLOR_GOLD, fontsize=9)
        ax2.legend(fontsize=7, facecolor=COLOR_DARK, edgecolor=COLOR_TEXT)
        ax2.set_facecolor(COLOR_DARK)
        ax2.tick_params(colors=COLOR_TEXT, labelsize=7)
        ax2.grid(True, alpha=0.3, color=COLOR_TEXT)

        fig.tight_layout(pad=1.5)

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True)

        self.current_figures.append(fig)

    def create_intonation_calculus_section(self):
        """Create intonation calculus visualization section."""
        section_frame = tk.LabelFrame(self.scrollable_frame,
                                    text="3. Intonation Calculus",
                                    font=FONT_SECTION,
                                    fg=COLOR_GOLD,
                                    bg=COLOR_BG,
                                    relief=tk.RIDGE,
                                    borderwidth=2)
        section_frame.pack(fill="x", padx=10, pady=10, ipadx=5, ipady=5)

        # Equation display
        eq_frame = tk.Frame(section_frame, bg=COLOR_BG)
        eq_frame.pack(fill="x", padx=5, pady=5)

        eq_label = tk.Label(eq_frame,
                          text="dθ/dt → 0, Δcents = 1200·log₂(f/f₀)",
                          font=("Courier", 10, "bold"),  # Reduced from 12
                          fg=COLOR_GOLD,
                          bg=COLOR_BG)
        eq_label.pack()

        # Description
        desc_label = tk.Label(section_frame,
                            text="Perfect intonation requires the rate of pitch change to approach zero.",
                            font=FONT_SMALL,
                            fg=COLOR_TEXT,
                            bg=COLOR_BG,
                            wraplength=WINDOW_WIDTH-60,
                            justify=tk.LEFT)
        desc_label.pack(padx=5, pady=5)

        # Visualization frame
        viz_frame = tk.Frame(section_frame, bg=COLOR_DARK)
        viz_frame.pack(fill="x", padx=5, pady=5, ipadx=2, ipady=2)

        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))  # Reduced from (8, 3)
        fig.patch.set_facecolor(COLOR_DARK)

        # Plot 1: Pitch stability comparison
        t = np.linspace(0, 2, 200)

        # Three types of pitch stability
        perfect = 440 + 0.1 * np.sin(2*np.pi*0.5*t)  # Minimal oscillation
        good = 440 + 0.5 * np.sin(2*np.pi*2*t)  # Some oscillation
        poor = 440 + 2.0 * np.random.randn(len(t))  # Random drift

        ax1.plot(t, perfect, 'g-', linewidth=1.5, label='Perfect (A+)')
        ax1.plot(t, good, 'y-', linewidth=1.5, label='Good (B)')
        ax1.plot(t, poor, 'r-', linewidth=1.5, label='Poor (D)')
        ax1.axhline(y=440, color='white', linestyle='--', alpha=0.5, label='Target A4')

        ax1.set_xlabel('Time (s)', color=COLOR_TEXT, fontsize=8)
        ax1.set_ylabel('Frequency (Hz)', color=COLOR_TEXT, fontsize=8)
        ax1.set_title('Pitch Stability Comparison', color=COLOR_GOLD, fontsize=9)
        ax1.legend(fontsize=7, facecolor=COLOR_DARK, edgecolor=COLOR_TEXT)
        ax1.set_facecolor(COLOR_DARK)
        ax1.tick_params(colors=COLOR_TEXT, labelsize=7)
        ax1.grid(True, alpha=0.3, color=COLOR_TEXT)

        # Plot 2: Cents error distribution
        cents_errors = np.random.normal(0, 10, 1000)  # Normal distribution

        ax2.hist(cents_errors, bins=20, color='skyblue', edgecolor='white', alpha=0.7)  # Reduced bins
        ax2.axvline(x=0, color='yellow', linestyle='--', linewidth=1.5, label='Perfect')
        ax2.axvline(x=-5, color='green', linestyle=':', alpha=0.7)
        ax2.axvline(x=5, color='green', linestyle=':', alpha=0.7, label='±5 cents')
        ax2.axvline(x=-20, color='orange', linestyle=':', alpha=0.7)
        ax2.axvline(x=20, color='orange', linestyle=':', alpha=0.7, label='±20 cents')

        ax2.set_xlabel('Cents Error', color=COLOR_TEXT, fontsize=8)
        ax2.set_ylabel('Count', color=COLOR_TEXT, fontsize=8)
        ax2.set_title('Intonation Error Distribution', color=COLOR_GOLD, fontsize=9)
        ax2.legend(fontsize=7, facecolor=COLOR_DARK, edgecolor=COLOR_TEXT)
        ax2.set_facecolor(COLOR_DARK)
        ax2.tick_params(colors=COLOR_TEXT, labelsize=7)
        ax2.grid(True, alpha=0.3, color=COLOR_TEXT)

        fig.tight_layout(pad=1.5)

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True)

        self.current_figures.append(fig)

    def create_vibrato_wave_section(self):
        """Create vibrato wave visualization section."""
        section_frame = tk.LabelFrame(self.scrollable_frame,
                                    text="4. Vibrato Wave Model",
                                    font=FONT_SECTION,
                                    fg=COLOR_GOLD,
                                    bg=COLOR_BG,
                                    relief=tk.RIDGE,
                                    borderwidth=2)
        section_frame.pack(fill="x", padx=10, pady=10, ipadx=5, ipady=5)

        # Equation display
        eq_frame = tk.Frame(section_frame, bg=COLOR_BG)
        eq_frame.pack(fill="x", padx=5, pady=5)

        eq_label = tk.Label(eq_frame,
                          text="f(t) = f₀·2^(Δ·sin(2π·ν·t + φ))",
                          font=("Courier", 10, "bold"),  # Reduced from 12
                          fg=COLOR_GOLD,
                          bg=COLOR_BG)
        eq_label.pack()

        # Description
        desc_label = tk.Label(section_frame,
                            text="Vibrato is frequency modulation of the base pitch.",
                            font=FONT_SMALL,
                            fg=COLOR_TEXT,
                            bg=COLOR_BG,
                            wraplength=WINDOW_WIDTH-60,
                            justify=tk.LEFT)
        desc_label.pack(padx=5, pady=5)

        # Visualization frame
        viz_frame = tk.Frame(section_frame, bg=COLOR_DARK)
        viz_frame.pack(fill="x", padx=5, pady=5, ipadx=2, ipady=2)

        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))  # Reduced from (8, 3)
        fig.patch.set_facecolor(COLOR_DARK)

        # Plot 1: Frequency modulation
        t = np.linspace(0, 2, 200)
        f0 = 440
        rate = 6  # Hz
        width = 30  # cents

        modulation = width / 1200
        f_t = f0 * (2 ** (modulation * np.sin(2 * np.pi * rate * t)))

        ax1.plot(t, f_t, 'c-', linewidth=1.5)
        ax1.fill_between(t, f_t, f0, alpha=0.3, color='cyan')
        ax1.axhline(y=f0, color='white', linestyle='--', alpha=0.7, label='Base A4')

        ax1.set_xlabel('Time (s)', color=COLOR_TEXT, fontsize=8)
        ax1.set_ylabel('Frequency (Hz)', color=COLOR_TEXT, fontsize=8)
        ax1.set_title(f'Vibrato: {rate}Hz, {width} cents', color=COLOR_GOLD, fontsize=9)
        ax1.legend(fontsize=7, facecolor=COLOR_DARK, edgecolor=COLOR_TEXT)
        ax1.set_facecolor(COLOR_DARK)
        ax1.tick_params(colors=COLOR_TEXT, labelsize=7)
        ax1.grid(True, alpha=0.3, color=COLOR_TEXT)

        # Plot 2: Different vibrato styles
        rates = [4, 6, 8]
        widths = [20, 30, 40]
        colors = ['red', 'green', 'blue']

        for rate, width, color in zip(rates, widths, colors):
            modulation = width / 1200
            f_t = f0 * (2 ** (modulation * np.sin(2 * np.pi * rate * t[:100])))
            ax2.plot(t[:100], f_t, color=color, linewidth=1.2,
                   label=f'{rate}Hz, {width}c')

        ax2.set_xlabel('Time (s)', color=COLOR_TEXT, fontsize=8)
        ax2.set_ylabel('Frequency (Hz)', color=COLOR_TEXT, fontsize=8)
        ax2.set_title('Different Vibrato Styles', color=COLOR_GOLD, fontsize=9)
        ax2.legend(fontsize=7, facecolor=COLOR_DARK, edgecolor=COLOR_TEXT)
        ax2.set_facecolor(COLOR_DARK)
        ax2.tick_params(colors=COLOR_TEXT, labelsize=7)
        ax2.grid(True, alpha=0.3, color=COLOR_TEXT)

        fig.tight_layout(pad=1.5)

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True)

        self.current_figures.append(fig)

    def create_energy_transfer_section(self):
        """Create energy transfer visualization section."""
        section_frame = tk.LabelFrame(self.scrollable_frame,
                                    text="5. Energy Transfer Model",
                                    font=FONT_SECTION,
                                    fg=COLOR_GOLD,
                                    bg=COLOR_BG,
                                    relief=tk.RIDGE,
                                    borderwidth=2)
        section_frame.pack(fill="x", padx=10, pady=10, ipadx=5, ipady=5)

        # Equation display
        eq_frame = tk.Frame(section_frame, bg=COLOR_BG)
        eq_frame.pack(fill="x", padx=5, pady=5)

        eq_label = tk.Label(eq_frame,
                          text="η = (1-α)·(1-β)·(1-γ)·(1-δ)·(1-ε)",
                          font=("Courier", 10, "bold"),  # Reduced from 12
                          fg=COLOR_GOLD,
                          bg=COLOR_BG)
        eq_label.pack()

        # Description
        desc_label = tk.Label(section_frame,
                            text="Energy conservation from bow arm to sound waves.",
                            font=FONT_SMALL,
                            fg=COLOR_TEXT,
                            bg=COLOR_BG,
                            wraplength=WINDOW_WIDTH-60,
                            justify=tk.LEFT)
        desc_label.pack(padx=5, pady=5)

        # Visualization frame
        viz_frame = tk.Frame(section_frame, bg=COLOR_DARK)
        viz_frame.pack(fill="x", padx=5, pady=5, ipadx=2, ipady=2)

        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))  # Reduced from (8, 3)
        fig.patch.set_facecolor(COLOR_DARK)

        # Plot 1: Energy flow diagram
        categories = ['Input', 'String', 'Bridge', 'Body', 'Radiation', 'Output']
        values = [100, 85, 60, 40, 25, 20]
        colors = ['#FF6B6B', '#FFA500', '#FFD700', '#90EE90', '#87CEEB', '#DDA0DD']

        bars = ax1.bar(categories, values, color=colors, edgecolor='white')
        ax1.set_ylabel('Energy (%)', color=COLOR_TEXT, fontsize=8)
        ax1.set_title('Energy Loss Through Violin', color=COLOR_GOLD, fontsize=9)
        ax1.set_facecolor(COLOR_DARK)
        ax1.tick_params(colors=COLOR_TEXT, labelsize=7, rotation=45)
        ax1.grid(True, alpha=0.3, color=COLOR_TEXT, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{val}%', ha='center', va='bottom',
                    color=COLOR_TEXT, fontsize=7)

        # Plot 2: Efficiency vs bow position
        positions = np.linspace(0, 1, 50)  # From fingerboard to bridge
        efficiency = 0.8 - 0.4 * abs(positions - 0.2)  # Sweet spot at 0.2

        ax2.plot(positions, efficiency, 'm-', linewidth=1.5)
        ax2.axvline(x=0.2, color='yellow', linestyle='--', alpha=0.7,
                   label='Sweet spot')
        ax2.fill_between(positions, efficiency, alpha=0.3, color='magenta')

        ax2.set_xlabel('Bow Position', color=COLOR_TEXT, fontsize=8)
        ax2.set_ylabel('Efficiency', color=COLOR_TEXT, fontsize=8)
        ax2.set_title('Efficiency vs Bow Position', color=COLOR_GOLD, fontsize=9)
        ax2.legend(fontsize=7, facecolor=COLOR_DARK, edgecolor=COLOR_TEXT)
        ax2.set_facecolor(COLOR_DARK)
        ax2.tick_params(colors=COLOR_TEXT, labelsize=7)
        ax2.grid(True, alpha=0.3, color=COLOR_TEXT)

        fig.tight_layout(pad=1.5)

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True)

        self.current_figures.append(fig)

    def create_harmonic_analysis_section(self):
        """Create harmonic analysis visualization section."""
        section_frame = tk.LabelFrame(self.scrollable_frame,
                                    text="6. Harmonic Analysis",
                                    font=FONT_SECTION,
                                    fg=COLOR_GOLD,
                                    bg=COLOR_BG,
                                    relief=tk.RIDGE,
                                    borderwidth=2)
        section_frame.pack(fill="x", padx=10, pady=10, ipadx=5, ipady=5)

        # Equation display
        eq_frame = tk.Frame(section_frame, bg=COLOR_BG)
        eq_frame.pack(fill="x", padx=5, pady=5)

        eq_label = tk.Label(eq_frame,
                          text="f(t) = Σ Aₙ·sin(2π·n·f₀·t + φₙ)",
                          font=("Courier", 10, "bold"),  # Reduced from 12
                          fg=COLOR_GOLD,
                          bg=COLOR_BG)
        eq_label.pack()

        # Description
        desc_label = tk.Label(section_frame,
                            text="Fourier decomposition of violin tone into harmonic series.",
                            font=FONT_SMALL,
                            fg=COLOR_TEXT,
                            bg=COLOR_BG,
                            wraplength=WINDOW_WIDTH-60,
                            justify=tk.LEFT)
        desc_label.pack(padx=5, pady=5)

        # Visualization frame
        viz_frame = tk.Frame(section_frame, bg=COLOR_DARK)
        viz_frame.pack(fill="x", padx=5, pady=5, ipadx=2, ipady=2)

        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 2.5))  # Reduced from (8, 3)
        fig.patch.set_facecolor(COLOR_DARK)

        # Plot 1: Harmonic spectrum
        harmonics = np.arange(1, 9)  # Reduced from 10
        amplitudes = [1.0, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08]

        bars = ax1.bar(harmonics, amplitudes, color='skyblue', edgecolor='white')
        ax1.set_xlabel('Harmonic', color=COLOR_TEXT, fontsize=8)
        ax1.set_ylabel('Amplitude', color=COLOR_TEXT, fontsize=8)
        ax1.set_title('Violin Harmonic Spectrum', color=COLOR_GOLD, fontsize=9)
        ax1.set_facecolor(COLOR_DARK)
        ax1.tick_params(colors=COLOR_TEXT, labelsize=7)
        ax1.grid(True, alpha=0.3, color=COLOR_TEXT, axis='y')

        # Plot 2: Waveform synthesis
        t = np.linspace(0, 0.01, 500)
        waveform = np.zeros_like(t)

        for n, a in zip(harmonics, amplitudes):
            waveform += a * np.sin(2 * np.pi * 440 * n * t)

        ax2.plot(t * 1000, waveform, 'c-', linewidth=1.5)
        ax2.set_xlabel('Time (ms)', color=COLOR_TEXT, fontsize=8)
        ax2.set_ylabel('Amplitude', color=COLOR_TEXT, fontsize=8)
        ax2.set_title('Synthesized Waveform', color=COLOR_GOLD, fontsize=9)
        ax2.set_facecolor(COLOR_DARK)
        ax2.tick_params(colors=COLOR_TEXT, labelsize=7)
        ax2.grid(True, alpha=0.3, color=COLOR_TEXT)

        fig.tight_layout(pad=1.5)

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True)

        self.current_figures.append(fig)

    def create_resonance_modes_section(self):
        """Create resonance modes visualization section."""
        section_frame = tk.LabelFrame(self.scrollable_frame,
                                    text="7. Violin Body Resonance",
                                    font=FONT_SECTION,
                                    fg=COLOR_GOLD,
                                    bg=COLOR_BG,
                                    relief=tk.RIDGE,
                                    borderwidth=2)
        section_frame.pack(fill="x", padx=10, pady=10, ipadx=5, ipady=5)

        # Description
        desc_label = tk.Label(section_frame,
                            text="Chladni patterns show nodal lines where violin body doesn't vibrate.",
                            font=FONT_SMALL,
                            fg=COLOR_TEXT,
                            bg=COLOR_BG,
                            wraplength=WINDOW_WIDTH-60,
                            justify=tk.LEFT)
        desc_label.pack(padx=5, pady=5)

        # Visualization frame
        viz_frame = tk.Frame(section_frame, bg=COLOR_DARK)
        viz_frame.pack(fill="x", padx=5, pady=5, ipadx=2, ipady=2)

        # Create matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(6, 4.5))  # Reduced from (8, 6)
        fig.patch.set_facecolor(COLOR_DARK)

        # Generate Chladni-like patterns
        x = np.linspace(-1, 1, 80)  # Reduced resolution
        y = np.linspace(-1, 1, 80)
        X, Y = np.meshgrid(x, y)

        patterns = [
            ((1, 1), 'Mode (1,1): 440 Hz'),
            ((2, 1), 'Mode (2,1): 650 Hz'),
            ((2, 2), 'Mode (2,2): 880 Hz'),
            ((3, 2), 'Mode (3,2): 1100 Hz')
        ]

        for ax, ((m, n), title) in zip(axes.flat, patterns):
            # Chladni pattern: sin(mπx) * sin(nπy)
            Z = np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)

            im = ax.imshow(Z, cmap='RdBu_r', extent=[-1, 1, -1, 1])
            ax.contour(X, Y, Z, levels=[0], colors='white', linewidths=0.8)
            ax.set_title(title, color=COLOR_GOLD, fontsize=8)
            ax.set_facecolor(COLOR_DARK)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle('Chladni Patterns - Violin Body Resonance',
                    color=COLOR_GOLD, fontsize=10)

        fig.tight_layout(pad=2)

        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x", expand=True)

        self.current_figures.append(fig)

    def create_bowing_cycle_section(self):
        """Create bowing cycle visualization section."""
        section_frame = tk.LabelFrame(self.scrollable_frame,
                                    text="8. Complete Bowing Cycle",
                                    font=FONT_SECTION,
                                    fg=COLOR_GOLD,
                                    bg=COLOR_BG,
                                    relief=tk.RIDGE,
                                    borderwidth=2)
        section_frame.pack(fill="x", padx=10, pady=10, ipadx=5, ipady=5)

        # Description
        desc_label = tk.Label(section_frame,
                            text="Integration of all mathematical models in a single bow stroke.",
                            font=FONT_SMALL,
                            fg=COLOR_TEXT,
                            bg=COLOR_BG,
                            wraplength=WINDOW_WIDTH-60,
                            justify=tk.LEFT)
        desc_label.pack(padx=5, pady=5)

        # Control frame
        control_frame = tk.Frame(section_frame, bg=COLOR_BG)
        control_frame.pack(fill="x", padx=5, pady=5)

        tk.Label(control_frame,
                text="Bow Speed:",
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_BG).pack(side="left", padx=2)

        self.bow_speed_var = tk.StringVar(value="constant")
        speed_menu = tk.OptionMenu(control_frame, self.bow_speed_var,
                                  "constant", "accelerating", "decelerating",
                                  "crescendo", "diminuendo")
        speed_menu.config(font=FONT_SMALL,
                         bg=COLOR_DARK,
                         fg=COLOR_TEXT,
                         activebackground=COLOR_FRAME,
                         activeforeground=COLOR_GOLD,
                         highlightthickness=0,
                         width=12)  # Reduced width
        speed_menu["menu"].config(bg=COLOR_DARK,
                                 fg=COLOR_TEXT,
                                 activebackground=COLOR_FRAME,
                                 activeforeground=COLOR_GOLD)
        speed_menu.pack(side="left", padx=5)

        # Simulate button
        simulate_btn = tk.Button(control_frame,
                               text="🎻 Simulate",
                               font=FONT_SMALL,
                               fg=COLOR_GOLD,
                               bg=COLOR_DARK,
                               command=self.simulate_bowing)
        simulate_btn.pack(side="right", padx=5)

        # Visualization frame
        viz_frame = tk.Frame(section_frame, bg=COLOR_DARK)
        viz_frame.pack(fill="x", padx=5, pady=5, ipadx=2, ipady=2)

        # Create matplotlib figure placeholder
        self.bowing_fig, self.bowing_ax = plt.subplots(figsize=(6, 3))  # Reduced from (8, 4)
        self.bowing_fig.patch.set_facecolor(COLOR_DARK)

        self.bowing_canvas = FigureCanvasTkAgg(self.bowing_fig, viz_frame)
        self.bowing_canvas.get_tk_widget().pack(fill="x", expand=True)

        # Initialize with default plot
        self.update_bowing_plot()

    def create_action_buttons(self):
        """Create action buttons at bottom."""
        actions_frame = tk.Frame(self.scrollable_frame, bg=COLOR_BG)
        actions_frame.pack(fill="x", padx=10, pady=10)

        # Button frame for side-by-side layout
        button_frame = tk.Frame(actions_frame, bg=COLOR_BG)
        button_frame.pack(side="left", padx=5)

        # Generate Report button
        report_btn = tk.Button(button_frame,
                             text="📊 Generate Report",
                             font=FONT_TEXT,
                             fg=COLOR_GOLD,
                             bg=COLOR_DARK,
                             activeforeground=COLOR_GOLD,
                             activebackground=COLOR_FRAME,
                             relief=tk.RAISED,
                             borderwidth=2,
                             padx=15,  # Reduced padding
                             pady=8,
                             command=self.generate_report)
        report_btn.pack(side="left", padx=2)

        # Save Visualizations button
        save_btn = tk.Button(button_frame,
                           text="💾 Save Images",
                           font=FONT_TEXT,
                           fg=COLOR_GOLD,
                           bg=COLOR_DARK,
                           activeforeground=COLOR_GOLD,
                           activebackground=COLOR_FRAME,
                           relief=tk.RAISED,
                           borderwidth=2,
                           padx=15,  # Reduced padding
                           pady=8,
                           command=self.save_visualizations)
        save_btn.pack(side="left", padx=2)

        # Status label
        self.status_var = tk.StringVar(value="✅ Ready")
        status_label = tk.Label(actions_frame,
                              textvariable=self.status_var,
                              font=FONT_SMALL,
                              fg=COLOR_TEXT,
                              bg=COLOR_BG)
        status_label.pack(side="right", padx=5)

    def update_bowing_plot(self, bowing_data=None):
        """Update the bowing cycle plot."""
        self.bowing_ax.clear()

        if bowing_data is None:
            # Show placeholder
            t = np.linspace(0, 2, 100)
            bow_speed = np.ones_like(t)
            sound_pressure = 0.5 * np.sin(2*np.pi*2*t)

            self.bowing_ax.plot(t, bow_speed, 'b-', linewidth=1.5, label='Bow Speed')
            self.bowing_ax.plot(t, sound_pressure, 'r-', linewidth=1.5, label='Sound Output')

            self.bowing_ax.set_xlabel('Time (s)', color=COLOR_TEXT, fontsize=8)
            self.bowing_ax.set_ylabel('Normalized Value', color=COLOR_TEXT, fontsize=8)
            self.bowing_ax.set_title('Bowing Cycle - Click Simulate', color=COLOR_GOLD, fontsize=10)

        else:
            # Plot actual data
            times = [d["time"] for d in bowing_data]
            bow_speeds = [d["bow_speed"] for d in bowing_data]
            sound_pressures = [d["sound_pressure"] for d in bowing_data]

            self.bowing_ax.plot(times, bow_speeds, 'b-', linewidth=1.5, label='Bow Speed')
            self.bowing_ax.plot(times, sound_pressures, 'r-', linewidth=1.5, label='Sound Output')

            self.bowing_ax.fill_between(times, 0, bow_speeds, alpha=0.3, color='blue')
            self.bowing_ax.fill_between(times, 0, sound_pressures, alpha=0.3, color='red')

            self.bowing_ax.set_xlabel('Time (s)', color=COLOR_TEXT, fontsize=8)
            self.bowing_ax.set_ylabel('Normalized Value', color=COLOR_TEXT, fontsize=8)
            self.bowing_ax.set_title(f'Bowing Cycle - {self.bow_speed_var.get().title()}',
                                   color=COLOR_GOLD, fontsize=10)

        self.bowing_ax.legend(fontsize=8, facecolor=COLOR_DARK, edgecolor=COLOR_TEXT)
        self.bowing_ax.set_facecolor(COLOR_DARK)
        self.bowing_ax.tick_params(colors=COLOR_TEXT, labelsize=7)
        self.bowing_ax.grid(True, alpha=0.3, color=COLOR_TEXT)

        self.bowing_fig.tight_layout(pad=1.5)
        self.bowing_canvas.draw()

    def simulate_bowing(self):
        """Simulate a complete bowing cycle."""
        self.status_var.set("⏳ Simulating...")

        try:
            # Generate bow speed profile based on selection
            t = np.linspace(0, 2, 100)
            profile_type = self.bow_speed_var.get()

            if profile_type == "constant":
                speed = np.ones_like(t)
            elif profile_type == "accelerating":
                speed = 0.5 + 0.5 * t
            elif profile_type == "decelerating":
                speed = 1.5 - 0.5 * t
            elif profile_type == "crescendo":
                speed = 0.3 + 0.7 * (1 - np.cos(np.pi * t / 2))
            elif profile_type == "diminuendo":
                speed = 1.0 - 0.7 * (1 - np.cos(np.pi * t / 2))
            else:
                speed = np.ones_like(t)

            # Generate pressure profile (simplified)
            pressure = 10 * np.ones_like(t)

            # Simulate using mathematics model
            bowing_data = self.mathematics.simulate_bowing_cycle(
                {"time": t.tolist(), "speed": speed.tolist()},
                {"time": t.tolist(), "pressure": pressure.tolist()}
            )

            # Update plot
            self.update_bowing_plot(bowing_data)

            self.status_var.set(f"✅ Simulated {profile_type}")

        except Exception as e:
            messagebox.showerror("Simulation Error", f"Error simulating bowing:\n{str(e)}")
            self.status_var.set("❌ Simulation failed")

    def generate_report(self):
        """Generate comprehensive mathematical report."""
        self.status_var.set("⏳ Generating report...")

        try:
            # Generate report using mathematics model
            report = self.mathematics.generate_performance_report({})

            # Create report window
            report_window = tk.Toplevel(self.parent)
            report_window.title("Violin Mathematics Report")
            report_window.geometry("550x600")  # Reduced size
            report_window.configure(bg=COLOR_BG)

            # Set close handler
            report_window.protocol("WM_DELETE_WINDOW", lambda: self.close_report_window(report_window))

            # Add scrollbar
            report_canvas = tk.Canvas(report_window, bg=COLOR_BG, highlightthickness=0)
            scrollbar = ttk.Scrollbar(report_window, orient="vertical", command=report_canvas.yview)

            report_frame = tk.Frame(report_canvas, bg=COLOR_BG, width=530)

            report_frame.bind(
                "<Configure>",
                lambda e: report_canvas.configure(scrollregion=report_canvas.bbox("all"))
            )

            report_canvas.create_window((10, 0), window=report_frame, anchor="nw", width=530)
            report_canvas.configure(yscrollcommand=scrollbar.set)

            report_canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Header
            tk.Label(report_frame,
                   text="🧮 VIOLIN MATHEMATICS REPORT",
                   font=("Georgia", 11, "bold"),  # Smaller font
                   fg=COLOR_GOLD,
                   bg=COLOR_BG).pack(pady=10)

            tk.Label(report_frame,
                   text=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                   font=FONT_SMALL,
                   fg=COLOR_TEXT,
                   bg=COLOR_BG).pack()

            # Performance scores
            scores_frame = tk.Frame(report_frame, bg=COLOR_BG)
            scores_frame.pack(fill="x", padx=10, pady=10)

            performance = report["performance_metrics"]
            score_text = f"""📊 PERFORMANCE METRICS:

Technical Score: {performance['technical_score']:.1f}/100
Musicality Score: {performance['musicality_score']:.1f}/100
Energy Efficiency: {performance['efficiency_score']:.1f}/100

Overall Assessment: {'Excellent' if np.mean(list(performance.values())) > 85
                    else 'Good' if np.mean(list(performance.values())) > 70
                    else 'Developing'}"""

            tk.Label(scores_frame,
                   text=score_text,
                   font=FONT_SMALL,  # Smaller font
                   fg=COLOR_TEXT,
                   bg=COLOR_BG,
                   justify=tk.LEFT).pack(anchor="w")

            # Mathematical models
            for model_name, model_data in report["mathematical_models"].items():
                if "error" not in model_data:
                    model_frame = tk.LabelFrame(report_frame,
                                              text=model_data["name"],
                                              font=("Georgia", 9, "italic"),  # Smaller
                                              fg=COLOR_GOLD,
                                              bg=COLOR_BG,
                                              relief=tk.GROOVE,
                                              borderwidth=1)
                    model_frame.pack(fill="x", padx=10, pady=5)

                    eq_label = tk.Label(model_frame,
                                      text=f"Equation: {model_data['equation']}",
                                      font=("Courier", 8),  # Smaller
                                      fg=COLOR_TEXT,
                                      bg=COLOR_BG)
                    eq_label.pack(anchor="w", padx=5, pady=2)

                    # Display key results
                    result = model_data["result"]
                    if isinstance(result, dict):
                        result_text = "\n".join([f"{k}: {v}" for k, v in list(result.items())[:3]])
                        tk.Label(model_frame,
                               text=result_text,
                               font=FONT_SMALL,
                               fg=COLOR_TEXT,
                               bg=COLOR_BG,
                               justify=tk.LEFT).pack(anchor="w", padx=5, pady=2)

            # Close button
            close_btn = tk.Button(report_frame,
                                text="Close Report",
                                font=FONT_SMALL,
                                fg=COLOR_GOLD,
                                bg=COLOR_DARK,
                                command=lambda: self.close_report_window(report_window))
            close_btn.pack(pady=10)

            self.status_var.set("✅ Report generated")

        except Exception as e:
            messagebox.showerror("Report Error", f"Error generating report:\n{str(e)}")
            self.status_var.set("❌ Report generation failed")

    def close_report_window(self, window):
        """Close report window properly."""
        window.destroy()

    def save_visualizations(self):
        """Save all visualizations as images."""
        try:
            import matplotlib.pyplot as plt

            # Ask for directory
            directory = filedialog.askdirectory(title="Select Save Directory")
            if not directory:
                return

            # Save each figure
            for i, fig in enumerate(self.current_figures):
                filename = os.path.join(directory, f"violin_math_{i+1}.png")
                fig.savefig(filename, dpi=120, facecolor=COLOR_DARK,  # Lower DPI
                          edgecolor='none', bbox_inches='tight')

            # Save bowing figure if it exists
            if hasattr(self, 'bowing_fig'):
                filename = os.path.join(directory, "violin_math_bowing.png")
                self.bowing_fig.savefig(filename, dpi=120, facecolor=COLOR_DARK,
                                      edgecolor='none', bbox_inches='tight')

            messagebox.showinfo("Success", f"Visualizations saved to:\n{directory}")
            self.status_var.set(f"✅ Images saved")

        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving visualizations:\n{str(e)}")
            self.status_var.set("❌ Save failed")

    def on_closing(self):
        """Clean up on closing."""
        self.is_closing = True

        # Close matplotlib figures
        for fig in self.current_figures:
            plt.close(fig)

        if hasattr(self, 'bowing_fig'):
            plt.close(self.bowing_fig)

        # Destroy the window
        self.parent.destroy()

def get_content():
    """Return module description for display in GUI."""
    return [
        "╔══════════════════════════════════════╗",
        "║     VIOLIN MATHEMATICS VISUALIZER    ║",
        "╚══════════════════════════════════════╝",
        "",
        "🎯 PURPOSE: Comprehensive visualization of violin performance mathematics",
        "",
        "🧮 MATHEMATICAL MODELS:",
        "  1. String Vibration: ∂²y/∂t² = c²∂²y/∂x²",
        "  2. Bow Fluid Dynamics: Reynolds number, flow separation",
        "  3. Intonation Calculus: dθ/dt → 0, cents error analysis",
        "  4. Vibrato Waves: f(t) = f₀·2^(Δ·sin(2π·ν·t + φ))",
        "  5. Energy Transfer: η = (1-α)·(1-β)·(1-γ)·(1-δ)·(1-ε)",
        "  6. Harmonic Analysis: Fourier decomposition",
        "  7. Resonance Modes: Chladni patterns",
        "  8. Complete Bowing Cycle: Integration of all models",
        "",
        "📊 VISUALIZATION FEATURES:",
        "  • Live mathematical plots",
        "  • Interactive parameter adjustment",
        "  • Real-time simulations",
        "  • Comparative analysis",
        "  • Performance scoring",
        "",
        "🎻 PHYSICAL PRINCIPLES:",
        "  • Wave propagation on strings",
        "  • Fluid dynamics of bow motion",
        "  • Energy conservation from arm to sound",
        "  • Resonance and standing waves",
        "  • Fourier synthesis of tone color",
        "",
        "🚀 HOW TO USE:",
        "  1. Scroll through mathematical models",
        "  2. Observe visualizations for each principle",
        "  3. Adjust bowing parameters in section 8",
        "  4. Click 'Simulate Bowing' to see integration",
        "  5. Generate reports or save visualizations",
    ]

def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    # Create the violin mathematics GUI
    gui = ViolinMathematicsGUI(parent)
    return gui.scrollable_frame

def test_standalone():
    """Test the module GUI standalone."""
    root = tk.Tk()
    root.title("Violin Mathematics Visualizer - Standalone Test")
    root.configure(bg=COLOR_BG)

    # Create the GUI
    gui = create_gui(root)

    root.mainloop()

if __name__ == "__main__":
    test_standalone()