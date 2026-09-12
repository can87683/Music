#!/usr/bin/env python3
# ai_violin_master_11_visualization.py

import numpy as np
from typing import List, Tuple, Dict, Any
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
from datetime import datetime

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


class Visualization:
    """ASCII and optional matplotlib visualizations."""

    def __init__(self):
        self.plt = self._try_import_matplotlib()

    def _try_import_matplotlib(self):
        """Safely import matplotlib only if available and usable."""
        try:
            import matplotlib
            matplotlib.use("Agg")  # headless safe
            import matplotlib.pyplot as plt
            return plt
        except Exception:
            return None

    def ascii_bar_chart(self, pairs, width=50):
        """Draw a simple horizontal bar chart."""
        if not pairs:
            return "<no data>"

        maxv = max(v for _, v in pairs)
        if maxv <= 0:
            maxv = 1.0

        bars = []
        for label, val in pairs:
            n = int((val / maxv) * width)
            bars.append(f"{label:<12} | " + "█" * n)

        return "\n".join(bars)

    def ascii_position_map(self, positions_used):
        """Show used violin positions from 1st to 7th."""
        pos_labels = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]
        marks = []
        for i, lab in enumerate(pos_labels):
            used = "█" if (i+1) in positions_used else "░"
            marks.append(f"{lab:<4}: {used}")

        return "\n".join(marks)

    def ascii_string_ribbon(self, string_usage):
        """Show string usage as a simple ribbon."""
        lines = []
        maxv = max(string_usage.values()) if string_usage else 1

        for s in ["G", "D", "A", "E"]:
            v = string_usage.get(s, 0)
            n = int(v / maxv * 30)
            lines.append(f"{s}: " + "█" * n)

        return "\n".join(lines)

    def ascii_bowing_ribbon(self, bow_data):
        """Represent slur/staccato patterns as a symbolic timeline."""
        slurs = bow_data.get("slur_groups", 0)
        avg = bow_data.get("avg_duration", 0.2)

        ribbon = []
        L = 80
        slur_prob = min(slurs / 20, 0.8)
        stacc_prob = min(0.3 / max(avg, 0.02), 0.8)

        import random
        random.seed(1)

        for _ in range(L):
            r = random.random()
            if r < slur_prob:
                ribbon.append("-")
            elif r < slur_prob + stacc_prob:
                ribbon.append("*")
            else:
                ribbon.append(".")
        return "".join(ribbon)

    def ascii_density_map(self, density_values, height=8, width=80):
        """Convert note density into a simple vertical heatmap block."""
        if not density_values:
            return "<no density data>"

        vals = [v for (_, v) in density_values]
        minv, maxv = min(vals), max(vals)
        rng = max(1e-6, maxv - minv)

        # map values → rows (0 to height-1)
        rows = [[] for _ in range(height)]
        for v in vals:
            h = int((v - minv) / rng * (height - 1))
            rows[height - 1 - h].append("█")
        # fill width
        for r in rows:
            while len(r) < width:
                r.append(" ")

        return "\n".join("".join(r) for r in rows)

    def ascii_intonation_plot(self, intonation_curve, width=80, height=8):
        """Plots cents deviation as ASCII graph."""
        if not intonation_curve:
            return "<no intonation data>"

        vals = [c for (_, c) in intonation_curve]
        minv, maxv = min(vals), max(vals)
        rng = max(1e-6, maxv - minv)

        # scale to height
        scaled = [int((v - minv) / rng * (height - 1)) for v in vals]
        rows = [[" "]*len(vals) for _ in range(height)]

        for i, h in enumerate(scaled):
            rows[height - 1 - h][i] = "█"

        # join as lines
        output = []
        for r in rows:
            line = "".join(r)
            if len(line) > width:
                line = line[:width]
            output.append(line)

        return "\n".join(output)

    def ascii_interval_contour(self, intervals, width=80, height=8):
        """Plot the interval direction (+/- semitones) as a contour."""
        if not intervals:
            return "<no interval data>"

        minv, maxv = min(intervals), max(intervals)
        rng = max(1e-6, maxv - minv)

        scaled = [int((v - minv) / rng * (height - 1)) for v in intervals]
        rows = [[" "]*len(intervals) for _ in range(height)]

        for i, h in enumerate(scaled):
            rows[height - 1 - h][i] = "█"

        output = []
        for r in rows:
            line = "".join(r)
            if len(line) > width:
                line = line[:width]
            output.append(line)

        return "\n".join(output)

    def ascii_timeline(self, metadata, violin_data, audio_data):
        """Combine several visualization streams into a timeline."""
        lines = []

        # 1) String ribbon
        lines.append("String Usage Timeline:")
        lines.append(self.ascii_string_ribbon(violin_data["range"]["string_usage"]))

        # 2) Positions
        lines.append("\nPositions Used:")
        lines.append(self.ascii_position_map(violin_data["positions"]["positions_used"]))

        # 3) Bowing
        lines.append("\nBowing Pattern Ribbon:")
        lines.append(self.ascii_bowing_ribbon(violin_data["bowing"]))

        # 4) Density (global)
        density = violin_data.get("density", [])
        if density:
            lines.append("\nNote Density (Heatmap):")
            lines.append(self.ascii_density_map(density, height=6))

        # 5) Intonation (if audio)
        if audio_data and "intonation_curve" in audio_data:
            lines.append("\nIntonation Drift:")
            lines.append(self.ascii_intonation_plot(audio_data["intonation_curve"], height=6))

        return "\n".join(lines)

    def save_png_interval_contour(self, intervals, path):
        """Save interval contour as PNG."""
        if self.plt is None:
            return None

        self.plt.figure(figsize=(10, 2))
        self.plt.plot(intervals)
        self.plt.title("Interval Contour")
        self.plt.xlabel("Note Index")
        self.plt.ylabel("Interval")
        self.plt.savefig(path)
        self.plt.close()
        return path

    def save_png_density(self, density_values, path):
        """Save density plot as PNG."""
        if self.plt is None:
            return None

        times = [t for (t, _) in density_values]
        vals = [v for (_, v) in density_values]

        self.plt.figure(figsize=(10, 2))
        self.plt.plot(times, vals)
        self.plt.title("Note Density Over Time")
        self.plt.xlabel("Time (s)")
        self.plt.ylabel("Density")
        self.plt.savefig(path)
        self.plt.close()
        return path

    def save_png_intonation(self, intonation_curve, path):
        """Save intonation plot as PNG."""
        if self.plt is None:
            return None

        times = [t for (t, _) in intonation_curve]
        cents = [c for (_, c) in intonation_curve]

        self.plt.figure(figsize=(10, 2))
        self.plt.plot(times, cents)
        self.plt.axhline(0, color="gray", linestyle="--")
        self.plt.title("Intonation Drift (Cents)")
        self.plt.xlabel("Time (s)")
        self.plt.ylabel("Cents Deviation")
        self.plt.savefig(path)
        self.plt.close()

        return path


class VisualizationGUI:
    """GUI for visualization module with AIModels style"""

    def __init__(self, parent):
        self.parent = parent
        self.parent.title("🎻 Visualization Module")
        self.parent.geometry(WINDOW_SIZE)
        self.parent.resizable(False, False)
        self.parent.configure(bg=COLOR_BG)

        # Initialize visualization module
        self.viz = Visualization()

        # Sample data for demonstration
        self.sample_data = self._create_sample_data()

        # Create GUI
        self._create_widgets()

    def _create_sample_data(self):
        """Create sample data for demonstration purposes"""
        return {
            "metadata": {
                "title": "Sample Violin Performance",
                "composer": "J.S. Bach",
                "duration": 120,
                "difficulty": "Intermediate"
            },
            "violin_data": {
                "range": {
                    "string_usage": {"G": 15, "D": 40, "A": 35, "E": 10},
                    "min_note": 55,
                    "max_note": 94
                },
                "positions": {
                    "positions_used": [1, 3, 5, 7],
                    "most_common_position": 3
                },
                "bowing": {
                    "slur_groups": 8,
                    "avg_duration": 0.25,
                    "staccato_notes": 12
                },
                "density": [(0, 0.1), (10, 0.8), (20, 0.3), (30, 0.9), (40, 0.2)],
                "intervals": [0, 2, -1, 3, 0, -2, 1, 0, 2, -3]
            },
            "audio_data": {
                "intonation_curve": [(0, -5), (5, 3), (10, -2), (15, 7), (20, -3)],
                "tempo_variation": [(0, 120), (10, 125), (20, 118), (30, 122)],
                "dynamics": [(0, 0.8), (10, 0.9), (20, 0.6), (30, 0.95)]
            }
        }

    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = tk.Frame(self.parent, bg=COLOR_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        self._create_header(main_frame)

        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Style the notebook
        self._style_notebook()

        # Create tabs
        self._create_ascii_tab(notebook)
        self._create_charts_tab(notebook)
        self._create_export_tab(notebook)

        # Status bar
        self._create_status_bar(main_frame)

    def _style_notebook(self):
        """Style the notebook widget to match color scheme"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background=COLOR_DARK, borderwidth=0)
        style.configure("TNotebook.Tab",
                       background=COLOR_DARK,
                       foreground=COLOR_TEXT,
                       font=FONT_SMALL,
                       padding=[10, 5])
        style.map("TNotebook.Tab",
                 background=[("selected", COLOR_FRAME)],
                 foreground=[("selected", COLOR_GOLD)])

    def _create_header(self, parent):
        """Create header section"""
        header_frame = tk.Frame(parent, bg=COLOR_BG)
        header_frame.pack(fill="x", pady=(0, 10))

        tk.Label(
            header_frame,
            text="🎻 Performance Visualization",
            font=FONT_HEADER,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack()

        tk.Label(
            header_frame,
            text="ASCII and graphical representations of violin performance data",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG
        ).pack()

    def _create_ascii_tab(self, notebook):
        """Create ASCII visualizations tab"""
        ascii_frame = tk.Frame(notebook, bg=COLOR_BG)
        notebook.add(ascii_frame, text="ASCII Visualizations")

        # Scrollable canvas for ASCII output
        canvas_frame = tk.Frame(ascii_frame, bg=COLOR_BG)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create scrollbar
        scrollbar = tk.Scrollbar(canvas_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create text widget for ASCII art
        self.ascii_text = tk.Text(
            canvas_frame,
            wrap=tk.NONE,
            font=("Courier", 9),
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            yscrollcommand=scrollbar.set,
            width=70,
            height=25
        )
        self.ascii_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.ascii_text.yview)

        # Buttons for different ASCII visualizations
        button_frame = tk.Frame(ascii_frame, bg=COLOR_BG)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        buttons = [
            ("📊 Timeline", self._show_ascii_timeline),
            ("🎯 Positions", self._show_ascii_positions),
            ("🎻 Strings", self._show_ascii_strings),
            ("🏹 Bowing", self._show_ascii_bowing),
            ("📈 Density", self._show_ascii_density),
            ("🎵 Intonation", self._show_ascii_intonation),
            ("↕️ Intervals", self._show_ascii_intervals)
        ]

        for i, (text, command) in enumerate(buttons):
            btn = tk.Button(
                button_frame,
                text=text,
                font=FONT_SMALL,
                fg=COLOR_GOLD,
                bg=COLOR_DARK,
                activeforeground=COLOR_GOLD,
                activebackground=COLOR_FRAME,
                relief=tk.RAISED,
                borderwidth=1,
                command=command,
                width=12
            )
            btn.grid(row=i//4, column=i%4, padx=2, pady=2, sticky="ew")
            button_frame.columnconfigure(i%4, weight=1)

        # Clear button
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            font=FONT_SMALL,
            fg=COLOR_GOLD,
            bg=COLOR_DARK,
            activeforeground=COLOR_GOLD,
            activebackground=COLOR_FRAME,
            relief=tk.RAISED,
            borderwidth=1,
            command=lambda: self.ascii_text.delete(1.0, tk.END),
            width=12
        )
        clear_btn.grid(row=1 if len(buttons) > 4 else 0,
                      column=3 if len(buttons)%4 == 0 else len(buttons)%4,
                      padx=2, pady=2, sticky="ew")

    def _create_charts_tab(self, notebook):
        """Create graphical charts tab"""
        charts_frame = tk.Frame(notebook, bg=COLOR_BG)
        notebook.add(charts_frame, text="Graphical Charts")

        # Info about matplotlib
        info_frame = tk.Frame(charts_frame, bg=COLOR_BG)
        info_frame.pack(fill=tk.X, padx=10, pady=10)

        if self.viz.plt is None:
            tk.Label(
                info_frame,
                text="⚠️ Matplotlib not available",
                font=FONT_TEXT,
                fg=COLOR_GOLD,
                bg=COLOR_BG
            ).pack()

            tk.Label(
                info_frame,
                text="Install matplotlib for graphical charts:\n\npip install matplotlib",
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                justify=tk.LEFT
            ).pack(pady=5)
        else:
            tk.Label(
                info_frame,
                text="✅ Matplotlib available",
                font=FONT_TEXT,
                fg=COLOR_GOLD,
                bg=COLOR_BG
            ).pack()

            tk.Label(
                info_frame,
                text="Generate and save graphical charts as PNG files",
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_BG
            ).pack(pady=5)

        # Chart generation buttons
        chart_frame = tk.LabelFrame(
            charts_frame,
            text="Generate Charts",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        chart_frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        chart_buttons = [
            ("📈 Interval Contour", self._generate_interval_chart),
            ("📊 Note Density", self._generate_density_chart),
            ("🎵 Intonation Drift", self._generate_intonation_chart)
        ]

        for i, (text, command) in enumerate(chart_buttons):
            btn = tk.Button(
                chart_frame,
                text=text,
                font=FONT_TEXT,
                fg=COLOR_GOLD,
                bg=COLOR_DARK,
                activeforeground=COLOR_GOLD,
                activebackground=COLOR_FRAME,
                relief=tk.RAISED,
                borderwidth=1,
                command=command,
                width=20,
                state=tk.NORMAL if self.viz.plt else tk.DISABLED
            )
            btn.pack(pady=5)

        # Preview area for generated charts
        preview_frame = tk.LabelFrame(
            charts_frame,
            text="Last Generated Chart",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10, ipadx=10, ipady=10)

        self.chart_info_label = tk.Label(
            preview_frame,
            text="No chart generated yet",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            wraplength=500
        )
        self.chart_info_label.pack(pady=20)

    def _create_export_tab(self, notebook):
        """Create export tab"""
        export_frame = tk.Frame(notebook, bg=COLOR_BG)
        notebook.add(export_frame, text="Export")

        # Export options frame
        options_frame = tk.LabelFrame(
            export_frame,
            text="Export Options",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10, ipadx=10, ipady=10)

        # ASCII export
        ascii_frame = tk.Frame(options_frame, bg=COLOR_BG)
        ascii_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            ascii_frame,
            text="Export ASCII Visualization:",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(side=tk.LEFT)

        self.ascii_export_btn = tk.Button(
            ascii_frame,
            text="💾 Save as .txt",
            font=FONT_SMALL,
            fg=COLOR_GOLD,
            bg=COLOR_DARK,
            activeforeground=COLOR_GOLD,
            activebackground=COLOR_FRAME,
            relief=tk.RAISED,
            borderwidth=1,
            command=self._export_ascii,
            width=15
        )
        self.ascii_export_btn.pack(side=tk.RIGHT, padx=5)

        # JSON export
        json_frame = tk.Frame(options_frame, bg=COLOR_BG)
        json_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            json_frame,
            text="Export Data as JSON:",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(side=tk.LEFT)

        self.json_export_btn = tk.Button(
            json_frame,
            text="💾 Save as .json",
            font=FONT_SMALL,
            fg=COLOR_GOLD,
            bg=COLOR_DARK,
            activeforeground=COLOR_GOLD,
            activebackground=COLOR_FRAME,
            relief=tk.RAISED,
            borderwidth=1,
            command=self._export_json,
            width=15
        )
        self.json_export_btn.pack(side=tk.RIGHT, padx=5)

        # Load data button
        load_frame = tk.Frame(options_frame, bg=COLOR_BG)
        load_frame.pack(fill=tk.X, pady=20)

        tk.Label(
            load_frame,
            text="Load Custom Data:",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(side=tk.LEFT)

        self.load_data_btn = tk.Button(
            load_frame,
            text="📂 Load JSON File",
            font=FONT_SMALL,
            fg=COLOR_GOLD,
            bg=COLOR_DARK,
            activeforeground=COLOR_GOLD,
            activebackground=COLOR_FRAME,
            relief=tk.RAISED,
            borderwidth=1,
            command=self._load_json_data,
            width=15
        )
        self.load_data_btn.pack(side=tk.RIGHT, padx=5)

        # Data preview
        preview_frame = tk.LabelFrame(
            options_frame,
            text="Current Data Preview",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10, ipadx=10, ipady=10)

        # Text widget for data preview
        data_preview = tk.Text(
            preview_frame,
            wrap=tk.WORD,
            font=FONT_SMALL,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            height=8
        )
        data_preview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Insert sample data preview
        preview_text = json.dumps(self.sample_data, indent=2)[:500] + "...\n\n(Truncated for preview)"
        data_preview.insert(1.0, preview_text)
        data_preview.config(state=tk.DISABLED)

    def _create_status_bar(self, parent):
        """Create status bar at bottom"""
        status_frame = tk.Frame(parent, bg=COLOR_DARK)
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        self.status_label = tk.Label(
            status_frame,
            text="✅ Ready",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_DARK
        )
        self.status_label.pack(side=tk.LEFT, padx=5)

        # Add timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_label = tk.Label(
            status_frame,
            text=f"🕐 {timestamp}",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_DARK
        )
        time_label.pack(side=tk.RIGHT, padx=5)

    # ASCII visualization methods
    def _show_ascii_timeline(self):
        """Show ASCII timeline visualization"""
        timeline = self.viz.ascii_timeline(
            self.sample_data["metadata"],
            self.sample_data["violin_data"],
            self.sample_data["audio_data"]
        )
        self._display_ascii("Timeline Visualization", timeline)

    def _show_ascii_positions(self):
        """Show ASCII positions visualization"""
        positions = self.viz.ascii_position_map(
            self.sample_data["violin_data"]["positions"]["positions_used"]
        )
        self._display_ascii("Violin Positions Used", positions)

    def _show_ascii_strings(self):
        """Show ASCII string usage visualization"""
        strings = self.viz.ascii_string_ribbon(
            self.sample_data["violin_data"]["range"]["string_usage"]
        )
        self._display_ascii("String Usage", strings)

    def _show_ascii_bowing(self):
        """Show ASCII bowing visualization"""
        bowing = self.viz.ascii_bowing_ribbon(
            self.sample_data["violin_data"]["bowing"]
        )
        self._display_ascii("Bowing Pattern", bowing)

    def _show_ascii_density(self):
        """Show ASCII density visualization"""
        density = self.viz.ascii_density_map(
            self.sample_data["violin_data"]["density"],
            height=6
        )
        self._display_ascii("Note Density Heatmap", density)

    def _show_ascii_intonation(self):
        """Show ASCII intonation visualization"""
        if self.sample_data["audio_data"]["intonation_curve"]:
            intonation = self.viz.ascii_intonation_plot(
                self.sample_data["audio_data"]["intonation_curve"],
                height=6
            )
            self._display_ascii("Intonation Drift", intonation)
        else:
            self._display_ascii("Intonation Drift", "No intonation data available")

    def _show_ascii_intervals(self):
        """Show ASCII interval visualization"""
        intervals = self.viz.ascii_interval_contour(
            self.sample_data["violin_data"]["intervals"],
            height=6
        )
        self._display_ascii("Interval Contour", intervals)

    def _display_ascii(self, title, content):
        """Display ASCII content in the text widget"""
        self.ascii_text.delete(1.0, tk.END)
        self.ascii_text.insert(1.0, f"{title}\n")
        self.ascii_text.insert(2.0, "=" * len(title) + "\n\n")
        self.ascii_text.insert(3.0, content)
        self.status_label.config(text=f"✅ Displayed: {title}")

    # Chart generation methods
    def _generate_interval_chart(self):
        """Generate interval contour chart"""
        if not self.viz.plt:
            messagebox.showerror("Error", "Matplotlib not available")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Interval Chart",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )

        if filename:
            try:
                path = self.viz.save_png_interval_contour(
                    self.sample_data["violin_data"]["intervals"],
                    filename
                )
                if path:
                    self.chart_info_label.config(
                        text=f"✅ Interval chart saved to:\n{path}"
                    )
                    self.status_label.config(text="✅ Interval chart generated")
                else:
                    self.chart_info_label.config(text="❌ Failed to save chart")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save chart: {str(e)}")

    def _generate_density_chart(self):
        """Generate density chart"""
        if not self.viz.plt:
            messagebox.showerror("Error", "Matplotlib not available")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Density Chart",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )

        if filename:
            try:
                path = self.viz.save_png_density(
                    self.sample_data["violin_data"]["density"],
                    filename
                )
                if path:
                    self.chart_info_label.config(
                        text=f"✅ Density chart saved to:\n{path}"
                    )
                    self.status_label.config(text="✅ Density chart generated")
                else:
                    self.chart_info_label.config(text="❌ Failed to save chart")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save chart: {str(e)}")

    def _generate_intonation_chart(self):
        """Generate intonation chart"""
        if not self.viz.plt:
            messagebox.showerror("Error", "Matplotlib not available")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Intonation Chart",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )

        if filename:
            try:
                path = self.viz.save_png_intonation(
                    self.sample_data["audio_data"]["intonation_curve"],
                    filename
                )
                if path:
                    self.chart_info_label.config(
                        text=f"✅ Intonation chart saved to:\n{path}"
                    )
                    self.status_label.config(text="✅ Intonation chart generated")
                else:
                    self.chart_info_label.config(text="❌ Failed to save chart")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save chart: {str(e)}")

    # Export methods
    def _export_ascii(self):
        """Export ASCII visualization to text file"""
        content = self.ascii_text.get(1.0, tk.END)
        if not content.strip():
            messagebox.showwarning("Warning", "No ASCII content to export")
            return

        filename = filedialog.asksaveasfilename(
            title="Save ASCII Visualization",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.status_label.config(text=f"✅ ASCII exported to {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"ASCII visualization saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")

    def _export_json(self):
        """Export data as JSON file"""
        filename = filedialog.asksaveasfilename(
            title="Save Data as JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.sample_data, f, indent=2)
                self.status_label.config(text=f"✅ JSON exported to {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Data saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")

    def _load_json_data(self):
        """Load custom data from JSON file"""
        filename = filedialog.askopenfilename(
            title="Load JSON Data",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    self.sample_data = json.load(f)
                self.status_label.config(text=f"✅ Loaded data from {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Data loaded from:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")


# Module interface functions for backward compatibility
def get_content():
    """Return module description for display in GUI."""
    return [
        "╔══════════════════════════════════════╗",
        "║      VISUALIZATION MODULE            ║",
        "╚══════════════════════════════════════╝",
        "",
        "🎯 PURPOSE: Visualize violin performance data",
        "",
        "📊 ASCII VISUALIZATIONS:",
        "  • Timeline of string usage",
        "  • Violin positions heatmap",
        "  • Bowing pattern ribbon",
        "  • Note density over time",
        "  • Intonation drift graph",
        "  • Interval contour plot",
        "",
        "📈 GRAPHICAL CHARTS (requires matplotlib):",
        "  • Save as PNG images",
        "  • High-resolution plots",
        "  • Publication-ready figures",
        "",
        "💾 EXPORT OPTIONS:",
        "  • ASCII art as text files",
        "  • Performance data as JSON",
        "  • Load custom data sets",
        "",
        "🎻 SAMPLE DATA INCLUDED:",
        "  • Demo violin performance",
        "  • Realistic metrics",
        "  • Multiple visualization types",
        "",
        "🚀 QUICK START:",
        "  1. Explore ASCII visualizations",
        "  2. Generate graphical charts",
        "  3. Export your favorite views",
        "  4. Load your own data (JSON)",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = VisualizationGUI(parent)
    return gui.parent


# Standalone test
def test_standalone():
    """Test the module GUI standalone."""
    root = tk.Tk()
    root.title("Visualization Module - Standalone Test")
    root.geometry(WINDOW_SIZE)
    root.configure(bg=COLOR_BG)

    # Create the GUI
    gui = create_gui(root)

    root.mainloop()


if __name__ == "__main__":
    test_standalone()