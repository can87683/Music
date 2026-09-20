#!/usr/bin/env python3
# ai_violin_master_09_report_generator.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683


import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
import json
import os
from datetime import datetime
import configparser

# Color constants
COLOR_BG = "#5A381E"
COLOR_DARK = "#3B2413"
COLOR_GOLD = "#FFD770"
COLOR_FRAME = "#4B2E18"
COLOR_TEXT = "#FFF2CF"
COLOR_MEDIUM = "#4B2E18"

WINDOW_SIZE = "640x1050"


class ReportGenerator:
    """Comprehensive performance report generation."""

    def _fmt_header(self, title: str) -> str:
        bar = "=" * (len(title) + 6)
        return f"\n{bar}\n== {title} ==\n{bar}\n"

    def _fmt_sub(self, title: str) -> str:
        bar = "-" * (len(title) + 4)
        return f"\n{bar}\n-- {title} --\n{bar}\n"

    def _fmt_kv(self, key: str, val) -> str:
        return f"{key:<30}: {val}"

    def _fmt_table(self, headers, rows):
        widths = [max(len(str(col)) for col in colset) for colset in zip(headers, *rows)]
        line = " | ".join(str(h).ljust(w) for h, w in zip(headers, widths))
        sep = "-+-".join("-" * w for w in widths)
        body = "\n".join(" | ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in rows)
        return line + "\n" + sep + "\n" + body

    def _range_report(self, r):
        lines = []
        lines.append(self._fmt_sub("Range"))
        lines.append(self._fmt_kv("Lowest Note", r["lowest_note"]))
        lines.append(self._fmt_kv("Highest Note", r["highest_note"]))
        lines.append(self._fmt_kv("Semitone Range", r["semitone_range"]))
        lines.append(self._fmt_kv("Uses High Positions", r["requires_high_positions"]))
        lines.append(self._fmt_kv("Extended Range", r["extended_technique_range"]))
        headers = ["String", "Count"]
        rows = [[s, c] for s, c in r["string_usage"].items()]
        lines.append(self._fmt_table(headers, rows))
        return "\n".join(lines)

    def _position_report(self, p):
        lines = []
        lines.append(self._fmt_sub("Left-Hand Positions & Shifts"))
        lines.append(self._fmt_kv("Positions used", p["positions_used"]))
        lines.append(self._fmt_kv("Total Shifts", p["shifts"]))
        lines.append(self._fmt_kv("Large Shifts", p["large_shifts"]))
        lines.append(self._fmt_kv("Shift Density", f"{p['shift_density']:.3f}"))
        lines.append(self._fmt_kv("Shift Complexity", p["shift_complexity"]))
        return "\n".join(lines)

    def _interval_report(self, iv):
        lines = []
        lines.append(self._fmt_sub("Intervals"))
        lines.append(self._fmt_kv("Large Intervals", iv.large_intervals))
        lines.append(self._fmt_kv("Max Interval", iv.max_interval))
        lines.append(self._fmt_kv("Average Interval", f"{iv.avg_interval:.2f}"))
        hist_lines = []
        for k, v in sorted(iv.interval_hist.items()):
            hist_lines.append([k, v])
        if hist_lines:
            lines.append(self._fmt_table(["Interval", "Count"], hist_lines))
        return "\n".join(lines)

    def _bowing_report(self, b):
        lines = []
        lines.append(self._fmt_sub("Bowing & Articulation"))
        lines.append(self._fmt_kv("Slur Groups", b.slur_groups))
        lines.append(self._fmt_kv("Avg Duration", f"{b.avg_duration:.3f}"))
        lines.append(self._fmt_kv("Articulation Consistency", f"{b.articulation_consistency:.3f}"))
        lines.append(self._fmt_kv("Complexity", b.bowing_complexity))
        return "\n".join(lines)

    def _technique_report(self, tstats):
        lines = []
        lines.append(self._fmt_sub("Technique Distribution"))
        lines.append(self._fmt_kv("Fast Passages", tstats["fast_passages"]))
        lines.append(self._fmt_kv("Staccato Notes", tstats["staccato"]))
        lines.append(self._fmt_kv("Double Stops", tstats["double_stops"]))
        lines.append(self._fmt_kv("Large Leaps", tstats["large_leaps"]))
        lines.append(self._fmt_kv("Ricochet", tstats["ricochet_patterns"]))
        lines.append(self._fmt_kv("Tremolo", tstats["tremolo_patterns"]))
        lines.append(self._fmt_kv("Technique Variety", tstats["technique_variety"]))
        return "\n".join(lines)

    def _vibrato_report(self, v):
        lines = []
        lines.append(self._fmt_sub("Vibrato"))
        lines.append(self._fmt_kv("Rate (Hz)", f"{v['rate_hz']:.3f}"))
        lines.append(self._fmt_kv("Depth (cents)", f"{v['depth_cents']:.1f}"))
        lines.append(self._fmt_kv("Stability", f"{v['stability']:.3f}"))
        lines.append(self._fmt_kv("Frames Analyzed", v["segments"]))
        return "\n".join(lines)

    def _timbre_report(self, t):
        lines = []
        lines.append(self._fmt_sub("Tone & Timbre"))
        if "error" in t:
            lines.append("Timbre analysis unavailable.")
            return "\n".join(lines)
        lines.append(self._fmt_kv("Spectral Centroid", f"{t['spectral_centroid']:.1f}"))
        lines.append(self._fmt_kv("Rolloff", f"{t['spectral_rolloff']:.1f}"))
        lines.append(self._fmt_kv("Bandwidth", f"{t['bandwidth']:.1f}"))
        lines.append(self._fmt_kv("ZCR", f"{t['zero_crossing_rate']:.3f}"))
        lines.append(self._fmt_kv("Harmonic Noise Ratio", f"{t['harmonic_noise_ratio']:.3f}"))
        lines.append(self._fmt_kv("Bow Noise", f"{t['bow_noise_level']:.3f}"))
        lines.append(self._fmt_kv("Brightness", f"{t['brightness']:.3f}"))
        lines.append(self._fmt_kv("Tone Quality", t["tone_label"]))
        return "\n".join(lines)

    def _intonation_report(self, icurve):
        lines = []
        lines.append(self._fmt_sub("Intonation (Cents Deviation)"))
        if not icurve:
            lines.append("No intonation data available.")
            return "\n".join(lines)
        cents_vals = [c for (_, c) in icurve]
        avg = float(np.mean(cents_vals))
        spread = float(np.std(cents_vals))
        lines.append(self._fmt_kv("Average Offset", f"{avg:+.2f} cents"))
        lines.append(self._fmt_kv("Stability (StdDev)", f"{spread:.2f}"))
        label = "Stable"
        if abs(avg) > 30:
            label = "Unstable"
        elif abs(avg) > 15:
            label = "Moderate Drift"
        lines.append(self._fmt_kv("Intonation Label", label))
        return "\n".join(lines)

    def _complexity_report(self, c):
        lines = []
        lines.append(self._fmt_sub("Overall Difficulty"))
        lines.append(self._fmt_kv("Rhythmic", f"{c.rhythmic:.1f}"))
        lines.append(self._fmt_kv("Technical", f"{c.technical:.1f}"))
        lines.append(self._fmt_kv("Range", f"{c.range_complexity:.1f}"))
        lines.append(self._fmt_kv("Articulation", f"{c.articulation:.1f}"))
        lines.append(self._fmt_kv("Total Score", f"{c.total:.1f}"))
        lines.append(self._fmt_kv("Difficulty Level", c.level))
        return "\n".join(lines)

    def _practice_report(self, advice, etudes):
        lines = []
        lines.append(self._fmt_sub("Practice Recommendations"))
        for a in advice.items:
            lines.append(f"- {a}")
        if etudes:
            lines.append(self._fmt_sub("Suggested Etudes"))
            for e in etudes:
                lines.append(f"- {e}")
        return "\n".join(lines)

    def _make_practice_schedule(self, complexity: str, advice: list) -> str:
        base = []
        base.append(self._fmt_sub("Practice Schedule"))
        if complexity == "Advanced":
            weeks = 4
        elif complexity == "Upper Intermediate":
            weeks = 3
        elif complexity == "Intermediate":
            weeks = 2
        else:
            weeks = 1
        for w in range(1, weeks+1):
            base.append(f"\nWEEK {w}")
            base.append("-" * 20)
            for tip in advice[: min(len(advice), 5)]:
                base.append(f"- {tip}")
        return "\n".join(base)

    def generate_report(self, metadata, violin_data, audio_data, complexity, advice, etudes):
        r = []
        r.append(self._fmt_header("AI Violin Master Performance Report"))
        r.append(self._range_report(violin_data["range"]))
        r.append(self._position_report(violin_data["positions"]))
        r.append(self._interval_report(violin_data["intervals"]))
        r.append(self._bowing_report(violin_data["bowing"]))
        r.append(self._technique_report(violin_data["technique_stats"]))
        r.append(self._complexity_report(complexity))
        if audio_data:
            vdata = audio_data.get("vibrato", {})
            tdata = audio_data.get("timbre", {})
            icurve = audio_data.get("intonation_curve", [])
            r.append(self._vibrato_report(vdata))
            r.append(self._timbre_report(tdata))
            r.append(self._intonation_report(icurve))
        r.append(self._practice_report(advice, etudes))
        r.append(self._make_practice_schedule(complexity.level, advice.items))
        return "\n".join(r)

    def generate_sample_report(self):
        class MockIntervals:
            def __init__(self):
                self.large_intervals = 12
                self.max_interval = 12
                self.avg_interval = 3.5
                self.interval_hist = {1: 45, 2: 32, 3: 28, 4: 18, 5: 12, 6: 8, 7: 6, 8: 5, 12: 2}

        class MockBowing:
            def __init__(self):
                self.slur_groups = 8
                self.avg_duration = 0.32
                self.articulation_consistency = 0.78
                self.bowing_complexity = "Intermediate"

        class MockComplexity:
            def __init__(self):
                self.rhythmic = 6.5
                self.technical = 7.2
                self.range_complexity = 8.0
                self.articulation = 5.8
                self.total = 6.9
                self.level = "Upper Intermediate"

        class MockAdvice:
            def __init__(self):
                self.items = [
                    "Focus on shifting accuracy in measures 15-24",
                    "Practice double stops with slow bow control",
                    "Work on consistent vibrato in high positions",
                    "Develop better bow distribution in legato passages",
                    "Improve intonation in 3rd and 5th positions"
                ]

        metadata = {
            "title": "Sample Violin Performance",
            "composer": "J.S. Bach",
            "duration": 125,
            "performer": "Student"
        }

        violin_data = {
            "range": {
                "lowest_note": "G3",
                "highest_note": "C7",
                "semitone_range": 36,
                "requires_high_positions": True,
                "extended_technique_range": False,
                "string_usage": {"G": 15, "D": 42, "A": 38, "E": 25}
            },
            "positions": {
                "positions_used": [1, 3, 5, 7],
                "shifts": 28,
                "large_shifts": 6,
                "shift_density": 0.45,
                "shift_complexity": "Medium"
            },
            "intervals": MockIntervals(),
            "bowing": MockBowing(),
            "technique_stats": {
                "fast_passages": 3,
                "staccato": 24,
                "double_stops": 8,
                "large_leaps": 12,
                "ricochet_patterns": 0,
                "tremolo_patterns": 0,
                "technique_variety": "Good"
            }
        }

        audio_data = {
            "vibrato": {
                "rate_hz": 5.8,
                "depth_cents": 45.2,
                "stability": 0.82,
                "segments": 120
            },
            "timbre": {
                "spectral_centroid": 2450.5,
                "spectral_rolloff": 6500.2,
                "bandwidth": 850.3,
                "zero_crossing_rate": 0.12,
                "harmonic_noise_ratio": 0.88,
                "bow_noise_level": 0.15,
                "brightness": 0.72,
                "tone_label": "Rich, focused"
            },
            "intonation_curve": [(i*10, float(np.random.normal(0, 8))) for i in range(13)]
        }

        complexity = MockComplexity()
        advice = MockAdvice()
        etudes = [
            "Kreutzer No. 8 - Shifting Exercises",
            "Schradieck Book 1 - Bow Distribution",
            "Dont Op. 35 - Double Stops",
            "Fiorillo - High Position Studies"
        ]

        return self.generate_report(metadata, violin_data, audio_data, complexity, advice, etudes)


class ReportGeneratorGUI:
    """GUI for Report Generator using CustomTkinter."""

    def __init__(self, parent=None):
        self.ini_file = "ai_violin_master_09_report_generator.ini"
        self.config = configparser.ConfigParser()

        if parent is None:
            self.root = ctk.CTk()
            self.is_standalone = True
            self.root.title("Report Generator")
            self.root.geometry(WINDOW_SIZE)
            self.root.resizable(False, False)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        else:
            self.root = parent
            self.is_standalone = False

        self.report_gen = ReportGenerator()
        self.current_report = ""

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
        # Main container
        main_frame = ctk.CTkFrame(self.root, fg_color=COLOR_BG)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        header_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            header_frame,
            text="PERFORMANCE REPORT GENERATOR",
            font=("Georgia", 16, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            header_frame,
            text="Generate comprehensive violin performance analysis reports",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=(0, 10))

        # Generate button
        self.gen_btn = ctk.CTkButton(
            main_frame,
            text="GENERATE SAMPLE REPORT",
            font=("Georgia", 14, "bold"),
            fg_color=COLOR_DARK,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_MEDIUM,
            height=50,
            command=self._generate_sample_report,
        )
        self.gen_btn.pack(pady=20)

        # Report display
        report_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        report_frame.pack(fill="both", expand=True, pady=10)

        ctk.CTkLabel(
            report_frame,
            text="REPORT PREVIEW",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        self.report_text = ctk.CTkTextbox(
            report_frame,
            font=("Courier New", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
            wrap="none",
        )
        self.report_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.report_text.insert("1.0", "Click GENERATE SAMPLE REPORT to create a demo report.")
        self.report_text.configure(state="disabled")

        # Export buttons
        export_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        export_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            export_frame,
            text="EXPORT OPTIONS",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        btn_row = ctk.CTkFrame(export_frame, fg_color=COLOR_DARK)
        btn_row.pack(fill="x", padx=20, pady=10)

        self.export_txt_btn = ctk.CTkButton(
            btn_row,
            text="EXPORT AS TXT",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_FRAME,
            command=self._export_text,
            state="disabled",
        )
        self.export_txt_btn.pack(side="left", padx=5, expand=True, fill="x")

        self.export_json_btn = ctk.CTkButton(
            btn_row,
            text="EXPORT AS JSON",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_FRAME,
            command=self._export_json,
            state="disabled",
        )
        self.export_json_btn.pack(side="left", padx=5, expand=True, fill="x")

        # Status bar
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Ready to generate reports",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            height=30,
        )
        self.status_label.pack(fill="x", pady=(10, 0))

    def _generate_sample_report(self):
        self.status_label.configure(text="Generating sample report...")
        self.root.update()

        self.current_report = self.report_gen.generate_sample_report()

        self.report_text.configure(state="normal")
        self.report_text.delete("1.0", "end")
        self.report_text.insert("1.0", self.current_report)
        self.report_text.configure(state="disabled")

        self.export_txt_btn.configure(state="normal")
        self.export_json_btn.configure(state="normal")
        self.status_label.configure(text="Sample report generated")

    def _export_text(self):
        filename = filedialog.asksaveasfilename(
            title="Save Report as Text",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.current_report)
            self.status_label.configure(text=f"Report exported to {os.path.basename(filename)}")
            messagebox.showinfo("Success", f"Report saved to:\n{filename}")

    def _export_json(self):
        filename = filedialog.asksaveasfilename(
            title="Save Report as JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            report_data = {
                "title": "Violin Performance Report",
                "generated": datetime.now().isoformat(),
                "content": self.current_report,
            }
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            self.status_label.configure(text=f"JSON exported to {os.path.basename(filename)}")
            messagebox.showinfo("Success", f"JSON report saved to:\n{filename}")

    def _on_close(self):
        self._save_state()
        self.root.destroy()


def get_content():
    """Return module description for display in GUI."""
    return [
        "REPORT GENERATOR MODULE",
        "",
        "PURPOSE: Generate comprehensive violin performance reports",
        "",
        "REPORT SECTIONS:",
        "  - Range and string usage analysis",
        "  - Left-hand positions and shifts",
        "  - Interval analysis and patterns",
        "  - Bowing and articulation assessment",
        "  - Technique distribution and challenges",
        "  - Overall difficulty and complexity score",
        "  - Audio analysis (vibrato, timbre, intonation)",
        "  - Personalized practice recommendations",
        "  - Adaptive practice schedules",
        "",
        "EXPORT OPTIONS:",
        "  - Plain text (.txt) for easy reading",
        "  - JSON (.json) for data processing",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = ReportGeneratorGUI(parent)
    return gui.root


def main():
    """Standalone entry - launches a full window with mainloop."""
    ReportGeneratorGUI()


if __name__ == "__main__":
    main()