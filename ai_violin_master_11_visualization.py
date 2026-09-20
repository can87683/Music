#!/usr/bin/env python3
# ai_violin_master_11_visualization.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683


import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
import os
import json
import random
import configparser

# Color constants
COLOR_BG = "#5A381E"
COLOR_DARK = "#3B2413"
COLOR_GOLD = "#FFD770"
COLOR_FRAME = "#4B2E18"
COLOR_TEXT = "#FFF2CF"
COLOR_MEDIUM = "#4B2E18"

WINDOW_SIZE = "640x1050"


class Visualization:
    """ASCII visualizations."""

    def ascii_bar_chart(self, pairs, width=50):
        if not pairs:
            return "<no data>"
        maxv = max(v for _, v in pairs)
        if maxv <= 0:
            maxv = 1.0
        bars = []
        for label, val in pairs:
            n = int((val / maxv) * width)
            bars.append(f"{label:<12} | " + "#" * n)
        return "\n".join(bars)

    def ascii_position_map(self, positions_used):
        pos_labels = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th"]
        marks = []
        for i, lab in enumerate(pos_labels):
            used = "#" if (i+1) in positions_used else "."
            marks.append(f"{lab:<4}: {used}")
        return "\n".join(marks)

    def ascii_string_ribbon(self, string_usage):
        lines = []
        maxv = max(string_usage.values()) if string_usage else 1
        for s in ["G", "D", "A", "E"]:
            v = string_usage.get(s, 0)
            n = int(v / maxv * 30)
            lines.append(f"{s}: " + "#" * n)
        return "\n".join(lines)

    def ascii_bowing_ribbon(self, bow_data):
        slurs = bow_data.get("slur_groups", 0)
        avg = bow_data.get("avg_duration", 0.2)
        ribbon = []
        L = 80
        slur_prob = min(slurs / 20, 0.8)
        stacc_prob = min(0.3 / max(avg, 0.02), 0.8)
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
        if not density_values:
            return "<no density data>"
        vals = [v for (_, v) in density_values]
        minv, maxv = min(vals), max(vals)
        rng = max(1e-6, maxv - minv)
        rows = [[] for _ in range(height)]
        for v in vals:
            h = int((v - minv) / rng * (height - 1))
            rows[height - 1 - h].append("#")
        for r in rows:
            while len(r) < width:
                r.append(" ")
        return "\n".join("".join(r) for r in rows)

    def ascii_intonation_plot(self, intonation_curve, width=80, height=8):
        if not intonation_curve:
            return "<no intonation data>"
        vals = [c for (_, c) in intonation_curve]
        minv, maxv = min(vals), max(vals)
        rng = max(1e-6, maxv - minv)
        scaled = [int((v - minv) / rng * (height - 1)) for v in vals]
        rows = [[" "]*len(vals) for _ in range(height)]
        for i, h in enumerate(scaled):
            rows[height - 1 - h][i] = "#"
        output = []
        for r in rows:
            line = "".join(r)
            if len(line) > width:
                line = line[:width]
            output.append(line)
        return "\n".join(output)

    def ascii_interval_contour(self, intervals, width=80, height=8):
        if not intervals:
            return "<no interval data>"
        minv, maxv = min(intervals), max(intervals)
        rng = max(1e-6, maxv - minv)
        scaled = [int((v - minv) / rng * (height - 1)) for v in intervals]
        rows = [[" "]*len(intervals) for _ in range(height)]
        for i, h in enumerate(scaled):
            rows[height - 1 - h][i] = "#"
        output = []
        for r in rows:
            line = "".join(r)
            if len(line) > width:
                line = line[:width]
            output.append(line)
        return "\n".join(output)

    def ascii_timeline(self, metadata, violin_data, audio_data):
        lines = []
        lines.append("String Usage Timeline:")
        lines.append(self.ascii_string_ribbon(violin_data["range"]["string_usage"]))
        lines.append("\nPositions Used:")
        lines.append(self.ascii_position_map(violin_data["positions"]["positions_used"]))
        lines.append("\nBowing Pattern Ribbon:")
        lines.append(self.ascii_bowing_ribbon(violin_data["bowing"]))
        density = violin_data.get("density", [])
        if density:
            lines.append("\nNote Density (Heatmap):")
            lines.append(self.ascii_density_map(density, height=6))
        if audio_data and "intonation_curve" in audio_data:
            lines.append("\nIntonation Drift:")
            lines.append(self.ascii_intonation_plot(audio_data["intonation_curve"], height=6))
        return "\n".join(lines)


class VisualizationGUI:
    """GUI for visualization module using CustomTkinter."""

    def __init__(self, parent=None):
        self.ini_file = "ai_violin_master_11_visualization.ini"
        self.config = configparser.ConfigParser()

        if parent is None:
            self.root = ctk.CTk()
            self.is_standalone = True
            self.root.title("Visualization Module")
            self.root.geometry(WINDOW_SIZE)
            self.root.resizable(False, False)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        else:
            self.root = parent
            self.is_standalone = False

        self.viz = Visualization()
        self.sample_data = self._create_sample_data()

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

    def _create_sample_data(self):
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
            }
        }

    def _create_widgets(self):
        # Main container
        main_frame = ctk.CTkFrame(self.root, fg_color=COLOR_BG)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        header_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            header_frame,
            text="PERFORMANCE VISUALIZATION",
            font=("Georgia", 16, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            header_frame,
            text="ASCII and graphical representations of violin performance data",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=(0, 10))

        # Visualization buttons
        btn_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        btn_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            btn_frame,
            text="ASCII VISUALIZATIONS",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        row1 = ctk.CTkFrame(btn_frame, fg_color=COLOR_DARK)
        row1.pack(fill="x", padx=20, pady=5)

        ctk.CTkButton(
            row1,
            text="Timeline",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_FRAME,
            command=self._show_ascii_timeline,
        ).pack(side="left", padx=5, expand=True, fill="x")

        ctk.CTkButton(
            row1,
            text="Positions",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_FRAME,
            command=self._show_ascii_positions,
        ).pack(side="left", padx=5, expand=True, fill="x")

        row2 = ctk.CTkFrame(btn_frame, fg_color=COLOR_DARK)
        row2.pack(fill="x", padx=20, pady=5)

        ctk.CTkButton(
            row2,
            text="Strings",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_FRAME,
            command=self._show_ascii_strings,
        ).pack(side="left", padx=5, expand=True, fill="x")

        ctk.CTkButton(
            row2,
            text="Bowing",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_FRAME,
            command=self._show_ascii_bowing,
        ).pack(side="left", padx=5, expand=True, fill="x")

        row3 = ctk.CTkFrame(btn_frame, fg_color=COLOR_DARK)
        row3.pack(fill="x", padx=20, pady=5)

        ctk.CTkButton(
            row3,
            text="Density",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_FRAME,
            command=self._show_ascii_density,
        ).pack(side="left", padx=5, expand=True, fill="x")

        ctk.CTkButton(
            row3,
            text="Intervals",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_FRAME,
            command=self._show_ascii_intervals,
        ).pack(side="left", padx=5, expand=True, fill="x")

        # Display area
        display_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        display_frame.pack(fill="both", expand=True, pady=10)

        ctk.CTkLabel(
            display_frame,
            text="VISUALIZATION OUTPUT",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        self.display_text = ctk.CTkTextbox(
            display_frame,
            font=("Courier New", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
            wrap="none",
        )
        self.display_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.display_text.insert("1.0", "Select a visualization to display")
        self.display_text.configure(state="disabled")

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

    def _display_ascii(self, title, content):
        self.display_text.configure(state="normal")
        self.display_text.delete("1.0", "end")
        self.display_text.insert("1.0", f"{title}\n")
        self.display_text.insert("2.0", "=" * len(title) + "\n\n")
        self.display_text.insert("3.0", content)
        self.display_text.configure(state="disabled")
        self.status_label.configure(text=f"Displayed: {title}")

    def _show_ascii_timeline(self):
        timeline = self.viz.ascii_timeline(
            self.sample_data["metadata"],
            self.sample_data["violin_data"],
            self.sample_data["audio_data"]
        )
        self._display_ascii("Timeline Visualization", timeline)

    def _show_ascii_positions(self):
        positions = self.viz.ascii_position_map(
            self.sample_data["violin_data"]["positions"]["positions_used"]
        )
        self._display_ascii("Violin Positions Used", positions)

    def _show_ascii_strings(self):
        strings = self.viz.ascii_string_ribbon(
            self.sample_data["violin_data"]["range"]["string_usage"]
        )
        self._display_ascii("String Usage", strings)

    def _show_ascii_bowing(self):
        bowing = self.viz.ascii_bowing_ribbon(
            self.sample_data["violin_data"]["bowing"]
        )
        self._display_ascii("Bowing Pattern", bowing)

    def _show_ascii_density(self):
        density = self.viz.ascii_density_map(
            self.sample_data["violin_data"]["density"],
            height=6
        )
        self._display_ascii("Note Density Heatmap", density)

    def _show_ascii_intervals(self):
        intervals = self.viz.ascii_interval_contour(
            self.sample_data["violin_data"]["intervals"],
            height=6
        )
        self._display_ascii("Interval Contour", intervals)

    def _on_close(self):
        self._save_state()
        self.root.destroy()


def get_content():
    """Return module description for display in GUI."""
    return [
        "VISUALIZATION MODULE",
        "",
        "PURPOSE: Visualize violin performance data",
        "",
        "ASCII VISUALIZATIONS:",
        "  - Timeline of string usage",
        "  - Violin positions heatmap",
        "  - Bowing pattern ribbon",
        "  - Note density over time",
        "  - Intonation drift graph",
        "  - Interval contour plot",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = VisualizationGUI(parent)
    return gui.root


def main():
    """Standalone entry - launches a full window with mainloop."""
    VisualizationGUI()


if __name__ == "__main__":
    main()