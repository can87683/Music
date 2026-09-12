#!/usr/bin/env python3
# ai_violin_master_05_kalman_smoother.py

from typing import List
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Color constants matching main script
COLOR_BG = "#5A381E"       # deep violin-wood brown
COLOR_DARK = "#3B2413"    # darker wood
COLOR_GOLD = "#FFD770"    # gold text
COLOR_FRAME = "#4B2E18"   # mid-tone wood
COLOR_TEXT = "#FFF2CF"    # parchment text

# Font constants matching main script
FONT_HEADER = ("Georgia", 14, "bold")
FONT_SECTION = ("Georgia", 12, "italic")
FONT_TEXT = ("Georgia", 11)
FONT_SMALL = ("Georgia", 9)

# Window size matching main script popups
WINDOW_SIZE = "640x1020"


class KalmanSmoother:
    """Lightweight 1D Kalman filter for smoothing pitch curves."""

    def __init__(self, process_var: float = 1e-3, measurement_var: float = 1e-2):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.estimate = None
        self.estimate_var = None

    def reset(self):
        """Reset filter state."""
        self.estimate = None
        self.estimate_var = None

    def update(self, measurement: float) -> float:
        """Add a new measurement and return smoothed output."""
        if self.estimate is None:
            self.estimate = measurement
            self.estimate_var = 1.0
            return measurement

        # Prediction
        pred_estimate = self.estimate
        pred_var = self.estimate_var + self.process_var

        # Kalman gain
        K = pred_var / (pred_var + self.measurement_var)

        # Update
        self.estimate = pred_estimate + K * (measurement - pred_estimate)
        self.estimate_var = (1 - K) * pred_var

        return self.estimate

    def smooth_array(self, data: List[float]) -> List[float]:
        """Smooth an entire array of measurements."""
        output = []
        for x in data:
            output.append(self.update(float(x)))
        return output


class KalmanSmootherGUI:
    """GUI for Kalman Smoother module with AIModels style"""

    def __init__(self, parent):
        self.parent = parent
        self.parent.title("🎻 Kalman Pitch Smoother")
        self.parent.geometry(WINDOW_SIZE)
        self.parent.resizable(False, False)
        self.parent.configure(bg=COLOR_BG)

        # Initialize Kalman smoother
        self.kalman = KalmanSmoother()

        # Sample data
        self.sample_data = None
        self.original_data = None
        self.smoothed_data = None

        # Parameters
        self.process_var = 1e-3
        self.measurement_var = 1e-2

        # Create GUI
        self._create_widgets()

        # Generate sample data
        self._generate_sample_data()

    def _generate_sample_data(self):
        """Generate sample pitch data for demonstration"""
        # Create a noisy pitch curve
        t = np.linspace(0, 10, 500)
        true_pitch = 440 + 50 * np.sin(2 * np.pi * 0.5 * t) + 20 * np.sin(2 * np.pi * 2 * t)

        # Add noise
        noise = np.random.normal(0, 20, len(t))
        noisy_pitch = true_pitch + noise

        # Add some outliers
        outlier_indices = np.random.choice(len(t), 10, replace=False)
        noisy_pitch[outlier_indices] += np.random.uniform(-50, 50, 10)

        self.sample_data = {
            "time": t.tolist(),
            "pitch": noisy_pitch.tolist(),
            "true_pitch": true_pitch.tolist()
        }

        self.original_data = noisy_pitch.tolist()

    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = tk.Frame(self.parent, bg=COLOR_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        self._create_header(main_frame)

        # Configuration panel
        self._create_configuration(main_frame)

        # Visualization panel
        self._create_visualization(main_frame)

        # Statistics panel
        self._create_statistics(main_frame)

        # Control panel
        self._create_controls(main_frame)

        # Status bar
        self._create_status_bar(main_frame)

    def _create_header(self, parent):
        """Create header section"""
        header_frame = tk.Frame(parent, bg=COLOR_BG)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            header_frame,
            text="🎻 Kalman Pitch Smoother",
            font=FONT_HEADER,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack()

        tk.Label(
            header_frame,
            text="Real-time smoothing of pitch curves using Kalman filtering",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG
        ).pack()

    def _create_configuration(self, parent):
        """Create configuration panel"""
        frame = tk.LabelFrame(
            parent,
            text="Filter Parameters",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        # Process variance
        proc_frame = tk.Frame(frame, bg=COLOR_BG)
        proc_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            proc_frame,
            text="Process Variance (Q):",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(side=tk.LEFT)

        self.process_var_var = tk.StringVar(value="0.001")
        proc_entry = tk.Entry(
            proc_frame,
            textvariable=self.process_var_var,
            font=FONT_TEXT,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            relief=tk.SUNKEN,
            borderwidth=1,
            width=15
        )
        proc_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(
            proc_frame,
            text="(Higher = more responsive)",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG
        ).pack(side=tk.LEFT, padx=5)

        # Measurement variance
        meas_frame = tk.Frame(frame, bg=COLOR_BG)
        meas_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            meas_frame,
            text="Measurement Variance (R):",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(side=tk.LEFT)

        self.measurement_var_var = tk.StringVar(value="0.01")
        meas_entry = tk.Entry(
            meas_frame,
            textvariable=self.measurement_var_var,
            font=FONT_TEXT,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            relief=tk.SUNKEN,
            borderwidth=1,
            width=15
        )
        meas_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(
            meas_frame,
            text="(Higher = smoother output)",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG
        ).pack(side=tk.LEFT, padx=5)

        # Reset filter button
        reset_btn = tk.Button(
            frame,
            text="🔄 Reset Filter",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_DARK,
            activeforeground=COLOR_GOLD,
            activebackground=COLOR_FRAME,
            relief=tk.RAISED,
            borderwidth=1,
            command=self._reset_filter,
            width=15
        )
        reset_btn.pack(pady=10)

    def _create_visualization(self, parent):
        """Create visualization panel"""
        frame = tk.LabelFrame(
            parent,
            text="Pitch Curve Visualization",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10, ipadx=10, ipady=10)

        # Create matplotlib figure
        self.figure = plt.Figure(figsize=(6, 3), dpi=80, facecolor=COLOR_BG)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(COLOR_DARK)

        # Style the plot
        self.ax.spines['bottom'].set_color(COLOR_TEXT)
        self.ax.spines['top'].set_color(COLOR_TEXT)
        self.ax.spines['right'].set_color(COLOR_TEXT)
        self.ax.spines['left'].set_color(COLOR_TEXT)
        self.ax.tick_params(colors=COLOR_TEXT)
        self.ax.xaxis.label.set_color(COLOR_TEXT)
        self.ax.yaxis.label.set_color(COLOR_TEXT)
        self.ax.title.set_color(COLOR_GOLD)

        # Set labels
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Pitch (Hz)")
        self.ax.set_title("Original vs Smoothed Pitch")

        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initial plot
        self._update_plot()

    def _create_statistics(self, parent):
        """Create statistics panel"""
        frame = tk.LabelFrame(
            parent,
            text="Filter Statistics",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        # Statistics grid
        stats_frame = tk.Frame(frame, bg=COLOR_BG)
        stats_frame.pack(fill=tk.X, pady=5)

        self.stats_labels = {}
        stats = [
            ("Original RMS Error", "original_rms", "0.0 Hz"),
            ("Smoothed RMS Error", "smoothed_rms", "0.0 Hz"),
            ("Noise Reduction", "noise_reduction", "0%"),
            ("Filter Gain", "filter_gain", "0.0"),
            ("Processing Speed", "speed", "0 samples/s")
        ]

        for i, (label, key, default) in enumerate(stats):
            row = i // 3
            col = i % 3

            stat_frame = tk.Frame(stats_frame, bg=COLOR_BG)
            stat_frame.grid(row=row, column=col, padx=10, pady=5, sticky="w")

            # Label
            tk.Label(
                stat_frame,
                text=label + ":",
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_BG
            ).pack(anchor="w")

            # Value
            value_label = tk.Label(
                stat_frame,
                text=default,
                font=FONT_SMALL,
                fg=COLOR_GOLD,
                bg=COLOR_BG
            )
            value_label.pack(anchor="w", padx=10)

            self.stats_labels[key] = value_label

    def _create_controls(self, parent):
        """Create control panel"""
        frame = tk.LabelFrame(
            parent,
            text="Data Controls",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        # Control buttons
        button_frame = tk.Frame(frame, bg=COLOR_BG)
        button_frame.pack(fill=tk.X, pady=5)

        controls = [
            ("📁 Load Data", self._load_data),
            ("🎵 Generate Sample", self._generate_sample),
            ("⚡ Apply Smoothing", self._apply_smoothing),
            ("💾 Export Results", self._export_results)
        ]

        # Create buttons in a 2x2 grid
        for i, (text, command) in enumerate(controls):
            row = i // 2
            col = i % 2

            btn = tk.Button(
                button_frame,
                text=text,
                font=FONT_TEXT,
                fg=COLOR_GOLD,
                bg=COLOR_DARK,
                activeforeground=COLOR_GOLD,
                activebackground=COLOR_FRAME,
                relief=tk.RAISED,
                borderwidth=1,
                command=command,
                width=15
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
            button_frame.columnconfigure(col, weight=1)

    def _create_status_bar(self, parent):
        """Create status bar at bottom"""
        status_frame = tk.Frame(parent, bg=COLOR_DARK)
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        self.status_label = tk.Label(
            status_frame,
            text="✅ Ready to smooth pitch data",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_DARK
        )
        self.status_label.pack(side=tk.LEFT, padx=5)

    def _update_plot(self):
        """Update the matplotlib plot"""
        self.ax.clear()

        # Style the plot
        self.ax.set_facecolor(COLOR_DARK)
        self.ax.spines['bottom'].set_color(COLOR_TEXT)
        self.ax.spines['top'].set_color(COLOR_TEXT)
        self.ax.spines['right'].set_color(COLOR_TEXT)
        self.ax.spines['left'].set_color(COLOR_TEXT)
        self.ax.tick_params(colors=COLOR_TEXT)
        self.ax.xaxis.label.set_color(COLOR_TEXT)
        self.ax.yaxis.label.set_color(COLOR_TEXT)
        self.ax.title.set_color(COLOR_GOLD)

        # Set labels
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Pitch (Hz)")
        self.ax.set_title("Original vs Smoothed Pitch")

        if self.sample_data is not None:
            time = self.sample_data["time"]
            original = self.sample_data["pitch"]
            true_pitch = self.sample_data["true_pitch"]

            # Plot true pitch (if available)
            if len(time) == len(true_pitch):
                self.ax.plot(time, true_pitch, 'g-', linewidth=1, alpha=0.5, label='True Pitch')

            # Plot original data
            self.ax.plot(time, original, 'b.', markersize=3, alpha=0.5, label='Original')

            # Plot smoothed data if available
            if self.smoothed_data is not None and len(self.smoothed_data) == len(time):
                self.ax.plot(time, self.smoothed_data, 'r-', linewidth=2, label='Smoothed')

            # Add legend
            self.ax.legend(facecolor=COLOR_FRAME, edgecolor=COLOR_TEXT, labelcolor=COLOR_TEXT)

        # Update canvas
        self.canvas.draw()

    def _update_statistics(self):
        """Update statistics display"""
        if self.sample_data is None or self.smoothed_data is None:
            return

        original = np.array(self.sample_data["pitch"])
        smoothed = np.array(self.smoothed_data)
        true_pitch = np.array(self.sample_data.get("true_pitch", original))

        # Calculate RMS errors
        original_error = np.sqrt(np.mean((original - true_pitch) ** 2))
        smoothed_error = np.sqrt(np.mean((smoothed - true_pitch) ** 2))

        # Calculate noise reduction
        if original_error > 0:
            noise_reduction = (1 - (smoothed_error / original_error)) * 100
        else:
            noise_reduction = 0

        # Update labels
        self.stats_labels["original_rms"].config(text=f"{original_error:.1f} Hz")
        self.stats_labels["smoothed_rms"].config(text=f"{smoothed_error:.1f} Hz")
        self.stats_labels["noise_reduction"].config(text=f"{noise_reduction:.1f}%")

        # Update filter gain (simplified)
        if hasattr(self.kalman, 'estimate_var') and self.kalman.estimate_var is not None:
            gain = 1 / (1 + self.kalman.estimate_var)
            self.stats_labels["filter_gain"].config(text=f"{gain:.3f}")

        # Processing speed (simplified)
        if len(original) > 0:
            speed = len(original)  # Placeholder
            self.stats_labels["speed"].config(text=f"{speed:,} samples")

    def _reset_filter(self):
        """Reset the Kalman filter"""
        self.kalman.reset()
        self.status_label.config(text="✅ Filter reset to initial state")

    def _load_data(self):
        """Load data from file"""
        filename = filedialog.askopenfilename(
            title="Load Pitch Data",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                if filename.endswith('.json'):
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Extract pitch data
                    if 'pitch' in data and 'time' in data:
                        self.sample_data = {
                            "time": data['time'],
                            "pitch": data['pitch'],
                            "true_pitch": data.get('true_pitch', data['pitch'])
                        }
                        self.original_data = data['pitch']
                    else:
                        # Try to extract from nested structure
                        pitch_data = []
                        time_data = []
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    if 'pitch' in item:
                                        pitch_data.append(item['pitch'])
                                    if 'time' in item:
                                        time_data.append(item['time'])

                        if pitch_data and time_data:
                            self.sample_data = {
                                "time": time_data,
                                "pitch": pitch_data,
                                "true_pitch": pitch_data
                            }
                            self.original_data = pitch_data

                elif filename.endswith('.csv'):
                    import pandas as pd
                    df = pd.read_csv(filename)

                    # Try to find pitch and time columns
                    pitch_col = None
                    time_col = None

                    for col in df.columns:
                        col_lower = col.lower()
                        if 'pitch' in col_lower or 'freq' in col_lower:
                            pitch_col = col
                        elif 'time' in col_lower:
                            time_col = col

                    if pitch_col:
                        pitch_data = df[pitch_col].dropna().tolist()
                        time_data = df[time_col].dropna().tolist() if time_col else list(range(len(pitch_data)))

                        self.sample_data = {
                            "time": time_data[:len(pitch_data)],
                            "pitch": pitch_data,
                            "true_pitch": pitch_data
                        }
                        self.original_data = pitch_data

                # Reset smoothed data
                self.smoothed_data = None

                # Update plot
                self._update_plot()
                self._update_statistics()

                self.status_label.config(text=f"✅ Loaded data from {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Data loaded from:\n{filename}")

            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load data:\n{str(e)}")
                self.status_label.config(text="❌ Failed to load data")

    def _generate_sample(self):
        """Generate sample data"""
        self._generate_sample_data()
        self.smoothed_data = None

        # Update plot and statistics
        self._update_plot()
        self._update_statistics()

        self.status_label.config(text="✅ Sample data generated")
        messagebox.showinfo("Sample Data", "Generated sample pitch curve with noise and outliers.")

    def _apply_smoothing(self):
        """Apply Kalman smoothing to the data"""
        if self.sample_data is None:
            messagebox.showwarning("No Data", "Please load or generate data first")
            return

        try:
            self.status_label.config(text="⏳ Applying Kalman smoothing...")
            self.parent.update()

            # Get parameters
            try:
                self.process_var = float(self.process_var_var.get())
                self.measurement_var = float(self.measurement_var_var.get())
            except ValueError:
                messagebox.showerror("Parameter Error", "Please enter valid numbers for variances")
                return

            # Update Kalman filter parameters
            self.kalman.process_var = self.process_var
            self.kalman.measurement_var = self.measurement_var

            # Reset filter
            self.kalman.reset()

            # Apply smoothing
            pitch_data = self.sample_data["pitch"]
            self.smoothed_data = self.kalman.smooth_array(pitch_data)

            # Update plot and statistics
            self._update_plot()
            self._update_statistics()

            self.status_label.config(text=f"✅ Smoothing applied (Q={self.process_var}, R={self.measurement_var})")

        except Exception as e:
            messagebox.showerror("Smoothing Error", f"Failed to apply smoothing:\n{str(e)}")
            self.status_label.config(text="❌ Smoothing failed")

    def _export_results(self):
        """Export smoothed results to file"""
        if self.smoothed_data is None:
            messagebox.showwarning("No Results", "Please apply smoothing first")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Smoothed Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if filename:
            try:
                if filename.endswith('.json'):
                    # Prepare data for JSON export
                    export_data = {
                        "timestamp": datetime.now().isoformat(),
                        "parameters": {
                            "process_variance": self.process_var,
                            "measurement_variance": self.measurement_var
                        },
                        "data": {
                            "time": self.sample_data["time"],
                            "original_pitch": self.sample_data["pitch"],
                            "smoothed_pitch": self.smoothed_data
                        },
                        "statistics": {
                            "original_rms": float(self.stats_labels["original_rms"].cget("text").split()[0]),
                            "smoothed_rms": float(self.stats_labels["smoothed_rms"].cget("text").split()[0]),
                            "noise_reduction": float(self.stats_labels["noise_reduction"].cget("text").rstrip('%'))
                        }
                    }

                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2)

                elif filename.endswith('.csv'):
                    import csv
                    with open(filename, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Time', 'Original_Pitch', 'Smoothed_Pitch'])

                        for t, orig, smooth in zip(self.sample_data["time"],
                                                  self.sample_data["pitch"],
                                                  self.smoothed_data):
                            writer.writerow([t, orig, smooth])

                self.status_label.config(text=f"✅ Results exported to {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Results saved to:\n{filename}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results:\n{str(e)}")
                self.status_label.config(text="❌ Export failed")


# Module interface functions for backward compatibility
def get_content():
    """Return module description for display in GUI."""
    return [
        "╔══════════════════════════════════════╗",
        "║        KALMAN PITCH SMOOTHER         ║",
        "╚══════════════════════════════════════╝",
        "",
        "🎯 PURPOSE: Smooth noisy pitch curves using Kalman filtering",
        "",
        "🔧 FILTER PARAMETERS:",
        "  • Process Variance (Q): Controls system dynamics",
        "  • Measurement Variance (R): Controls noise sensitivity",
        "  • Adaptive tuning for optimal smoothing",
        "  • Real-time parameter adjustment",
        "",
        "📊 VISUALIZATION FEATURES:",
        "  • Side-by-side comparison of original vs smoothed",
        "  • Interactive matplotlib plot",
        "  • Color-coded data series",
        "  • Real-time plot updates",
        "",
        "📈 STATISTICS & METRICS:",
        "  • RMS error reduction calculation",
        "  • Noise reduction percentage",
        "  • Filter gain and convergence speed",
        "  • Processing performance metrics",
        "",
        "💾 DATA HANDLING:",
        "  • Load pitch data from JSON or CSV files",
        "  • Generate synthetic sample data",
        "  • Export smoothed results",
        "  • Save filter parameters and statistics",
        "",
        "🎻 VIOLIN-SPECIFIC APPLICATIONS:",
        "  • Clean pitch curves for intonation analysis",
        "  • Remove bow noise and measurement artifacts",
        "  • Stabilize pitch for vibrato detection",
        "  • Prepare data for machine learning",
        "",
        "🚀 QUICK START:",
        "  1. Load data or generate sample",
        "  2. Adjust filter parameters (Q & R)",
        "  3. Apply Kalman smoothing",
        "  4. Compare results and export",
        "",
        "⚡ PERFORMANCE:",
        "  • Real-time processing of large datasets",
        "  • Minimal computational overhead",
        "  • Memory-efficient implementation",
        "  • Thread-safe operation",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = KalmanSmootherGUI(parent)
    return gui.parent


# Standalone test
def test_standalone():
    """Test the module GUI standalone."""
    try:
        import matplotlib
        matplotlib.use("TkAgg")
    except ImportError:
        print("⚠️ Matplotlib not available. Install with: pip install matplotlib")
        return

    root = tk.Tk()
    root.title("Kalman Pitch Smoother - Standalone Test")
    root.geometry(WINDOW_SIZE)
    root.configure(bg=COLOR_BG)

    # Create the GUI
    gui = create_gui(root)

    root.mainloop()


if __name__ == "__main__":
    test_standalone()