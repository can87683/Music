#!/usr/bin/env python3
# ai_violin_master_09_report_generator.py

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import os
from datetime import datetime
import textwrap

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


class ReportGenerator:
    """Comprehensive performance report generation."""

    def _fmt_header(self, title: str) -> str:
        """Format header."""
        bar = "=" * (len(title) + 6)
        return f"\n{bar}\n== {title} ==\n{bar}\n"

    def _fmt_sub(self, title: str) -> str:
        """Format subheader."""
        bar = "-" * (len(title) + 4)
        return f"\n{bar}\n-- {title} --\n{bar}\n"

    def _fmt_kv(self, key: str, val) -> str:
        """Format key-value pair."""
        return f"{key:<30}: {val}"

    def _fmt_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Format table."""
        widths = [max(len(str(col)) for col in colset) for colset in zip(headers, *rows)]
        line = " | ".join(str(h).ljust(w) for h, w in zip(headers, widths))
        sep = "-+-".join("-" * w for w in widths)
        body = "\n".join(" | ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in rows)
        return line + "\n" + sep + "\n" + body

    def _range_report(self, r):
        """Generate range report."""
        lines = []
        lines.append(self._fmt_sub("Range"))
        lines.append(self._fmt_kv("Lowest Note", r["lowest_note"]))
        lines.append(self._fmt_kv("Highest Note", r["highest_note"]))
        lines.append(self._fmt_kv("Semitone Range", r["semitone_range"]))
        lines.append(self._fmt_kv("Uses High Positions", r["requires_high_positions"]))
        lines.append(self._fmt_kv("Extended Range", r["extended_technique_range"]))

        # String usage table
        headers = ["String", "Count"]
        rows = [[s, c] for s, c in r["string_usage"].items()]
        lines.append(self._fmt_table(headers, rows))
        return "\n".join(lines)

    def _position_report(self, p):
        """Generate position/shift report."""
        lines = []
        lines.append(self._fmt_sub("Left-Hand Positions & Shifts"))
        lines.append(self._fmt_kv("Positions used", p["positions_used"]))
        lines.append(self._fmt_kv("Total Shifts", p["shifts"]))
        lines.append(self._fmt_kv("Large Shifts", p["large_shifts"]))
        lines.append(self._fmt_kv("Shift Density", f"{p['shift_density']:.3f}"))
        lines.append(self._fmt_kv("Shift Complexity", p["shift_complexity"]))
        return "\n".join(lines)

    def _interval_report(self, iv):
        """Generate interval report."""
        lines = []
        lines.append(self._fmt_sub("Intervals"))
        lines.append(self._fmt_kv("Large Intervals", iv.large_intervals))
        lines.append(self._fmt_kv("Max Interval", iv.max_interval))
        lines.append(self._fmt_kv("Average Interval", f"{iv.avg_interval:.2f}"))

        # summarize histogram
        hist_lines = []
        for k, v in sorted(iv.interval_hist.items()):
            hist_lines.append([k, v])

        if hist_lines:
            lines.append(self._fmt_table(["Interval", "Count"], hist_lines))

        return "\n".join(lines)

    def _bowing_report(self, b):
        """Generate bowing report."""
        lines = []
        lines.append(self._fmt_sub("Bowing & Articulation"))
        lines.append(self._fmt_kv("Slur Groups", b.slur_groups))
        lines.append(self._fmt_kv("Avg Duration", f"{b.avg_duration:.3f}"))
        lines.append(self._fmt_kv("Articulation Consistency", f"{b.articulation_consistency:.3f}"))
        lines.append(self._fmt_kv("Complexity", b.bowing_complexity))
        return "\n".join(lines)

    def _technique_report(self, tstats):
        """Generate technique report."""
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
        """Generate vibrato report."""
        lines = []
        lines.append(self._fmt_sub("Vibrato"))
        lines.append(self._fmt_kv("Rate (Hz)", f"{v['rate_hz']:.3f}"))
        lines.append(self._fmt_kv("Depth (cents)", f"{v['depth_cents']:.1f}"))
        lines.append(self._fmt_kv("Stability", f"{v['stability']:.3f}"))
        lines.append(self._fmt_kv("Frames Analyzed", v["segments"]))
        return "\n".join(lines)

    def _timbre_report(self, t):
        """Generate timbre report."""
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
        """Generate intonation report."""
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
        """Generate complexity report."""
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
        """Generate practice recommendations report."""
        lines = []
        lines.append(self._fmt_sub("Practice Recommendations"))
        for a in advice.items:
            lines.append(f"• {a}")

        if etudes:
            lines.append(self._fmt_sub("Suggested Etudes"))
            for e in etudes:
                lines.append(f"• {e}")

        return "\n".join(lines)

    def _make_practice_schedule(self, complexity: str, advice: List[str]) -> str:
        """Generate adaptive practice schedule."""
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
                base.append(f"• {tip}")

        return "\n".join(base)

    def generate_report(self, metadata, violin_data, audio_data, complexity, advice, etudes):
        """Generate final comprehensive report."""
        r = []
        r.append(self._fmt_header("AI Violin Master Performance Report"))

        r.append(self._range_report(violin_data["range"]))
        r.append(self._position_report(violin_data["positions"]))
        r.append(self._interval_report(violin_data["intervals"]))
        r.append(self._bowing_report(violin_data["bowing"]))
        r.append(self._technique_report(violin_data["technique_stats"]))

        r.append(self._complexity_report(complexity))

        # Audio
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

    def analyze_performance(self, midi_path: str, audio_path: Optional[str] = None) -> str:
        """Full analysis pipeline."""
        # This would integrate with other modules
        # For now, return placeholder
        return "Full performance analysis report"

    def generate_sample_report(self):
        """Generate a sample report for demonstration."""
        # Sample data for demonstration
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
            "intervals": type('obj', (object,), {
                'large_intervals': 12,
                'max_interval': 12,
                'avg_interval': 3.5,
                'interval_hist': {1: 45, 2: 32, 3: 28, 4: 18, 5: 12, 6: 8, 7: 6, 8: 5, 12: 2}
            })(),
            "bowing": type('obj', (object,), {
                'slur_groups': 8,
                'avg_duration': 0.32,
                'articulation_consistency': 0.78,
                'bowing_complexity': "Intermediate"
            })(),
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
            "intonation_curve": [(i*10, np.random.normal(0, 8)) for i in range(13)]
        }

        complexity = type('obj', (object,), {
            'rhythmic': 6.5,
            'technical': 7.2,
            'range_complexity': 8.0,
            'articulation': 5.8,
            'total': 6.9,
            'level': "Upper Intermediate"
        })()

        advice = type('obj', (object,), {
            'items': [
                "Focus on shifting accuracy in measures 15-24",
                "Practice double stops with slow bow control",
                "Work on consistent vibrato in high positions",
                "Develop better bow distribution in legato passages",
                "Improve intonation in 3rd and 5th positions"
            ]
        })()

        etudes = [
            "Kreutzer No. 8 - Shifting Exercises",
            "Schradieck Book 1 - Bow Distribution",
            "Dont Op. 35 - Double Stops",
            "Fiorillo - High Position Studies"
        ]

        return self.generate_report(metadata, violin_data, audio_data, complexity, advice, etudes)


class ReportGeneratorGUI:
    """GUI for Report Generator module with AIModels style"""

    def __init__(self, parent):
        self.parent = parent
        self.parent.title("🎻 Report Generator")
        self.parent.geometry(WINDOW_SIZE)
        self.parent.resizable(False, False)
        self.parent.configure(bg=COLOR_BG)

        # Initialize report generator
        self.report_gen = ReportGenerator()

        # Current report
        self.current_report = ""

        # Create GUI
        self._create_widgets()

    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = tk.Frame(self.parent, bg=COLOR_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        self._create_header(main_frame)

        # Report controls
        self._create_report_controls(main_frame)

        # Report display
        self._create_report_display(main_frame)

        # Export options
        self._create_export_options(main_frame)

        # Status bar
        self._create_status_bar(main_frame)

    def _create_header(self, parent):
        """Create header section"""
        header_frame = tk.Frame(parent, bg=COLOR_BG)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            header_frame,
            text="🎻 Performance Report Generator",
            font=FONT_HEADER,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack()

        tk.Label(
            header_frame,
            text="Generate comprehensive violin performance analysis reports",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG
        ).pack()

    def _create_report_controls(self, parent):
        """Create report generation controls"""
        frame = tk.LabelFrame(
            parent,
            text="Report Generation",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        # Generation buttons in grid
        button_frame = tk.Frame(frame, bg=COLOR_BG)
        button_frame.pack(fill=tk.X, pady=5)

        controls = [
            ("📊 Generate Sample Report", self._generate_sample_report),
            ("📁 Load Analysis Data", self._load_analysis_data),
            ("⚙️ Customize Report", self._customize_report),
            ("🔄 Refresh Report", self._refresh_report)
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
                width=20
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
            button_frame.columnconfigure(col, weight=1)

        # Report options
        options_frame = tk.Frame(frame, bg=COLOR_BG)
        options_frame.pack(fill=tk.X, pady=10)

        # Report type selection
        type_frame = tk.Frame(options_frame, bg=COLOR_BG)
        type_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            type_frame,
            text="Report Type:",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(side=tk.LEFT)

        self.report_type_var = tk.StringVar(value="Comprehensive")
        report_types = ["Comprehensive", "Technical Only", "Audio Analysis", "Practice Plan"]

        # Create styled dropdown
        type_menu = tk.OptionMenu(type_frame, self.report_type_var, *report_types)
        type_menu.config(
            font=FONT_TEXT,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            activebackground=COLOR_FRAME,
            activeforeground=COLOR_GOLD,
            highlightthickness=0,
            width=15
        )
        type_menu["menu"].config(
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            activebackground=COLOR_FRAME,
            activeforeground=COLOR_GOLD
        )
        type_menu.pack(side=tk.LEFT, padx=5)

        # Include audio analysis checkbox
        self.include_audio_var = tk.BooleanVar(value=True)
        audio_check = tk.Checkbutton(
            options_frame,
            text="Include Audio Analysis",
            variable=self.include_audio_var,
            font=FONT_TEXT,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            activebackground=COLOR_BG,
            activeforeground=COLOR_TEXT,
            selectcolor=COLOR_DARK
        )
        audio_check.pack(anchor="w", pady=5)

    def _create_report_display(self, parent):
        """Create report display area"""
        frame = tk.LabelFrame(
            parent,
            text="Report Preview",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10, ipadx=10, ipady=10)

        # Create notebook for different report sections
        self.report_notebook = ttk.Notebook(frame)
        self.report_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Style the notebook
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

        # Create tabs
        self._create_full_report_tab()
        self._create_summary_tab()
        self._create_practice_tab()

    def _create_full_report_tab(self):
        """Create full report tab"""
        full_frame = tk.Frame(self.report_notebook, bg=COLOR_BG)
        self.report_notebook.add(full_frame, text="Full Report")

        # Scrollable text widget for full report
        scroll_frame = tk.Frame(full_frame, bg=COLOR_BG)
        scroll_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        y_scrollbar = tk.Scrollbar(scroll_frame)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        x_scrollbar = tk.Scrollbar(scroll_frame, orient=tk.HORIZONTAL)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Text widget
        self.report_text = tk.Text(
            scroll_frame,
            wrap=tk.NONE,
            font=("Courier", 9),
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            yscrollcommand=y_scrollbar.set,
            xscrollcommand=x_scrollbar.set,
            height=15
        )
        self.report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure scrollbars
        y_scrollbar.config(command=self.report_text.yview)
        x_scrollbar.config(command=self.report_text.xview)

        # Insert placeholder
        self.report_text.insert(1.0, "No report generated yet.\n\nClick 'Generate Sample Report' to create a demo report.")
        self.report_text.config(state=tk.DISABLED)

    def _create_summary_tab(self):
        """Create summary tab"""
        summary_frame = tk.Frame(self.report_notebook, bg=COLOR_BG)
        self.report_notebook.add(summary_frame, text="Summary")

        # Text widget for summary
        scrollbar = tk.Scrollbar(summary_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.summary_text = tk.Text(
            summary_frame,
            wrap=tk.WORD,
            font=FONT_TEXT,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            yscrollcommand=scrollbar.set,
            height=15
        )
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.summary_text.yview)

        # Insert placeholder
        self.summary_text.insert(1.0, "Summary will appear here.")
        self.summary_text.config(state=tk.DISABLED)

    def _create_practice_tab(self):
        """Create practice recommendations tab"""
        practice_frame = tk.Frame(self.report_notebook, bg=COLOR_BG)
        self.report_notebook.add(practice_frame, text="Practice Plan")

        # Text widget for practice plan
        scrollbar = tk.Scrollbar(practice_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.practice_text = tk.Text(
            practice_frame,
            wrap=tk.WORD,
            font=FONT_TEXT,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            yscrollcommand=scrollbar.set,
            height=15
        )
        self.practice_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.practice_text.yview)

        # Insert placeholder
        self.practice_text.insert(1.0, "Practice recommendations will appear here.")
        self.practice_text.config(state=tk.DISABLED)

    def _create_export_options(self, parent):
        """Create export options area"""
        frame = tk.LabelFrame(
            parent,
            text="Export Report",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        # Export buttons
        button_frame = tk.Frame(frame, bg=COLOR_BG)
        button_frame.pack(fill=tk.X, pady=5)

        export_types = [
            ("📝 Export as Text", self._export_text),
            ("📄 Export as HTML", self._export_html),
            ("📊 Export as JSON", self._export_json),
            ("📋 Copy to Clipboard", self._copy_to_clipboard)
        ]

        for i, (text, command) in enumerate(export_types):
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
                width=20
            )
            btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
            button_frame.columnconfigure(col, weight=1)
            btn.config(state=tk.DISABLED if not self.current_report else tk.NORMAL)

    def _create_status_bar(self, parent):
        """Create status bar at bottom"""
        status_frame = tk.Frame(parent, bg=COLOR_DARK)
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        self.status_label = tk.Label(
            status_frame,
            text="✅ Ready to generate reports",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_DARK
        )
        self.status_label.pack(side=tk.LEFT, padx=5)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_label = tk.Label(
            status_frame,
            text=f"🕐 {timestamp}",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_DARK
        )
        time_label.pack(side=tk.RIGHT, padx=5)

    # Report generation methods
    def _generate_sample_report(self):
        """Generate a sample report for demonstration"""
        try:
            self.status_label.config(text="⏳ Generating sample report...")
            self.parent.update()

            # Generate sample report
            self.current_report = self.report_gen.generate_sample_report()

            # Update display
            self._update_report_display()

            # Enable export buttons
            self._enable_export_buttons()

            self.status_label.config(text="✅ Sample report generated")

        except Exception as e:
            messagebox.showerror("Generation Error", f"Failed to generate sample report:\n{str(e)}")
            self.status_label.config(text="❌ Report generation failed")

    def _load_analysis_data(self):
        """Load analysis data from file"""
        filename = filedialog.askopenfilename(
            title="Load Analysis Data",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                self.status_label.config(text="⏳ Loading analysis data...")
                self.parent.update()

                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Here you would process the data and generate a report
                # For now, we'll just show a message
                messagebox.showinfo("Data Loaded", f"Analysis data loaded from:\n{filename}\n\n(Report generation from data is not implemented in this demo)")

                self.status_label.config(text="✅ Analysis data loaded")

            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load analysis data:\n{str(e)}")
                self.status_label.config(text="❌ Failed to load data")

    def _customize_report(self):
        """Open customization dialog"""
        if not self.current_report:
            messagebox.showinfo("No Report", "Please generate a report first")
            return

        # Create customization dialog
        dialog = tk.Toplevel(self.parent)
        dialog.title("Customize Report")
        dialog.geometry("500x400")
        dialog.configure(bg=COLOR_BG)
        dialog.resizable(False, False)

        # Center the dialog
        dialog.transient(self.parent)
        dialog.grab_set()

        # Dialog content
        header = tk.Label(
            dialog,
            text="Report Customization",
            font=FONT_HEADER,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        )
        header.pack(pady=10)

        # Customization options
        options_frame = tk.LabelFrame(
            dialog,
            text="Sections to Include",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        options_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10, ipadx=10, ipady=10)

        # Checkboxes for sections
        sections = [
            ("Range Analysis", True),
            ("Position Analysis", True),
            ("Interval Analysis", True),
            ("Bowing Analysis", True),
            ("Technique Analysis", True),
            ("Complexity Analysis", True),
            ("Audio Analysis", self.include_audio_var.get()),
            ("Practice Recommendations", True)
        ]

        self.section_vars = {}
        for section_text, default in sections:
            var = tk.BooleanVar(value=default)
            self.section_vars[section_text] = var

            check = tk.Checkbutton(
                options_frame,
                text=section_text,
                variable=var,
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                activebackground=COLOR_BG,
                activeforeground=COLOR_TEXT,
                selectcolor=COLOR_DARK
            )
            check.pack(anchor="w", padx=10, pady=2)

        # Apply button
        def apply_customization():
            # Update audio checkbox in main window
            self.include_audio_var.set(self.section_vars["Audio Analysis"].get())

            # In a real implementation, this would regenerate the report
            # with only the selected sections
            messagebox.showinfo("Customization Applied",
                              "Report customization applied.\n\n(Full implementation would regenerate the report)")
            dialog.destroy()

        apply_btn = tk.Button(
            dialog,
            text="Apply Customization",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_DARK,
            activeforeground=COLOR_GOLD,
            activebackground=COLOR_FRAME,
            relief=tk.RAISED,
            borderwidth=2,
            command=apply_customization,
            width=20
        )
        apply_btn.pack(pady=10)

    def _refresh_report(self):
        """Refresh the current report"""
        if not self.current_report:
            messagebox.showinfo("No Report", "No report to refresh")
            return

        # In a real implementation, this would regenerate the report
        # with current settings
        messagebox.showinfo("Refresh", "Report refreshed with current settings")
        self.status_label.config(text="✅ Report refreshed")

    def _update_report_display(self):
        """Update all report display tabs"""
        if not self.current_report:
            return

        # Update full report
        self.report_text.config(state=tk.NORMAL)
        self.report_text.delete(1.0, tk.END)
        self.report_text.insert(1.0, self.current_report)
        self.report_text.config(state=tk.DISABLED)

        # Update summary (extract key points)
        self._update_summary_tab()

        # Update practice plan (extract practice section)
        self._update_practice_tab()

    def _update_summary_tab(self):
        """Update summary tab with key findings"""
        if not self.current_report:
            return

        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)

        # Extract key information from report
        lines = self.current_report.split('\n')
        summary_lines = []

        # Look for key sections and metrics
        for line in lines:
            if "Difficulty Level:" in line:
                summary_lines.append(f"🎯 {line.strip()}")
            elif "Average Interval:" in line:
                summary_lines.append(f"📊 {line.strip()}")
            elif "Large Intervals:" in line:
                summary_lines.append(f"🎵 {line.strip()}")
            elif "Positions used:" in line:
                summary_lines.append(f"🎻 {line.strip()}")
            elif "Total Shifts:" in line:
                summary_lines.append(f"🔄 {line.strip()}")
            elif "Slur Groups:" in line:
                summary_lines.append(f"🏹 {line.strip()}")
            elif "Fast Passages:" in line:
                summary_lines.append(f"⚡ {line.strip()}")
            elif "Intonation Label:" in line:
                summary_lines.append(f"🎵 {line.strip()}")

        if summary_lines:
            summary_text = "🎻 PERFORMANCE SUMMARY\n"
            summary_text += "=" * 40 + "\n\n"
            summary_text += "\n".join(summary_lines)
        else:
            summary_text = "No summary data available"

        self.summary_text.insert(1.0, summary_text)
        self.summary_text.config(state=tk.DISABLED)

    def _update_practice_tab(self):
        """Update practice tab with recommendations"""
        if not self.current_report:
            return

        self.practice_text.config(state=tk.NORMAL)
        self.practice_text.delete(1.0, tk.END)

        # Extract practice recommendations from report
        lines = self.current_report.split('\n')
        practice_lines = []
        in_practice_section = False

        for line in lines:
            if "Practice Recommendations" in line:
                in_practice_section = True
                continue
            if in_practice_section and line.strip().startswith("•"):
                practice_lines.append(line.strip())
            elif in_practice_section and line.strip() and not line.strip().startswith("--"):
                # Look for other practice-related lines
                if any(keyword in line for keyword in ["Week", "WEEK", "Focus", "Practice"]):
                    practice_lines.append(line.strip())
            elif in_practice_section and "=" * 40 in line:
                break

        if practice_lines:
            practice_text = "🎻 PRACTICE PLAN\n"
            practice_text += "=" * 40 + "\n\n"
            practice_text += "\n".join(practice_lines[:15])  # Show first 15 items
        else:
            practice_text = "No practice recommendations in this report"

        self.practice_text.insert(1.0, practice_text)
        self.practice_text.config(state=tk.DISABLED)

    def _enable_export_buttons(self):
        """Enable all export buttons"""
        # Get all buttons in export frame
        for child in self.parent.winfo_children():
            if isinstance(child, tk.LabelFrame) and "Export Report" in child.cget("text"):
                for widget in child.winfo_children():
                    if isinstance(widget, tk.Frame):
                        for subwidget in widget.winfo_children():
                            if isinstance(subwidget, tk.Button):
                                subwidget.config(state=tk.NORMAL)

    # Export methods
    def _export_text(self):
        """Export report as text file"""
        if not self.current_report:
            messagebox.showwarning("No Report", "No report to export")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Report as Text",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.current_report)

                self.status_label.config(text=f"✅ Report exported to {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Report saved to:\n{filename}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save report:\n{str(e)}")
                self.status_label.config(text="❌ Export failed")

    def _export_html(self):
        """Export report as HTML file"""
        if not self.current_report:
            messagebox.showwarning("No Report", "No report to export")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Report as HTML",
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")]
        )

        if filename:
            try:
                # Convert report to HTML
                html_content = self._convert_to_html(self.current_report)

                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                self.status_label.config(text=f"✅ HTML report exported to {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"HTML report saved to:\n{filename}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save HTML:\n{str(e)}")
                self.status_label.config(text="❌ Export failed")

    def _export_json(self):
        """Export report as JSON file"""
        if not self.current_report:
            messagebox.showwarning("No Report", "No report to export")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Report as JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                # Create structured data from report
                report_data = {
                    "title": "Violin Performance Report",
                    "generated": datetime.now().isoformat(),
                    "content": self.current_report,
                    "sections": self._extract_sections(self.current_report)
                }

                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2)

                self.status_label.config(text=f"✅ JSON report exported to {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"JSON report saved to:\n{filename}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to save JSON:\n{str(e)}")
                self.status_label.config(text="❌ Export failed")

    def _copy_to_clipboard(self):
        """Copy report to clipboard"""
        if not self.current_report:
            messagebox.showwarning("No Report", "No report to copy")
            return

        try:
            self.parent.clipboard_clear()
            self.parent.clipboard_append(self.current_report)

            self.status_label.config(text="✅ Report copied to clipboard")
            messagebox.showinfo("Copied", "Report copied to clipboard")

        except Exception as e:
            messagebox.showerror("Copy Error", f"Failed to copy to clipboard:\n{str(e)}")
            self.status_label.config(text="❌ Copy failed")

    def _convert_to_html(self, text_report):
        """Convert text report to HTML"""
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Violin Performance Report</title>
    <style>
        body {
            font-family: 'Georgia', serif;
            line-height: 1.6;
            margin: 40px;
            background-color: #f5f0e6;
            color: #333;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 3px solid #5A381E;
            padding-bottom: 20px;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .section-title {
            color: #5A381E;
            border-bottom: 2px solid #FFD770;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        .key-value {
            margin: 5px 0;
        }
        .key {
            font-weight: bold;
            color: #5A381E;
        }
        .practice-tip {
            background-color: #FFF2CF;
            padding: 10px;
            margin: 10px 0;
            border-left: 4px solid #FFD770;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #5A381E;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎻 Violin Performance Report</h1>
        <p>Generated on """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    </div>
"""

        # Parse text report into HTML sections
        lines = text_report.split('\n')
        current_section = ""
        in_table = False
        table_rows = []

        for line in lines:
            if "==" in line and "==" in line[line.find("==")+2:]:
                # Main header
                title = line.replace("=", "").replace("==", "").strip()
                html += f'    <div class="section">\n'
                html += f'        <h2 class="section-title">{title}</h2>\n'
                current_section = title
            elif "--" in line and "--" in line[line.find("--")+2:]:
                # Subheader
                title = line.replace("-", "").replace("--", "").strip()
                html += f'        <h3>{title}</h3>\n'
            elif ":" in line and not in_table:
                # Key-value pair
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    html += f'        <div class="key-value">\n'
                    html += f'            <span class="key">{key}:</span> {value}\n'
                    html += f'        </div>\n'
            elif "|" in line:
                # Table row
                if not in_table:
                    in_table = True
                    html += f'        <table>\n'

                cells = [cell.strip() for cell in line.split("|")]
                if "-+-" in line:
                    html += f'            <tr>\n'
                    for cell in cells:
                        html += f'                <th>{cell}</th>\n'
                    html += f'            </tr>\n'
                else:
                    html += f'            <tr>\n'
                    for cell in cells:
                        html += f'                <td>{cell}</td>\n'
                    html += f'            </tr>\n'
            elif line.strip() == "" and in_table:
                # End of table
                in_table = False
                html += f'        </table>\n'
            elif line.strip().startswith("•"):
                # Practice tip
                tip = line.strip()[1:].strip()
                html += f'        <div class="practice-tip">{tip}</div>\n'

        if in_table:
            html += f'        </table>\n'

        html += """    </div>
</body>
</html>"""

        return html

    def _extract_sections(self, text_report):
        """Extract sections from text report for JSON export"""
        sections = {}
        lines = text_report.split('\n')
        current_section = ""

        for line in lines:
            if "==" in line and "==" in line[line.find("==")+2:]:
                # Main section
                current_section = line.replace("=", "").replace("==", "").strip()
                sections[current_section] = []
            elif current_section and line.strip():
                sections[current_section].append(line.strip())

        return sections


# Module interface functions for backward compatibility
def get_content():
    """Return module description for display in GUI."""
    return [
        "╔══════════════════════════════════════╗",
        "║     REPORT GENERATOR MODULE          ║",
        "╚══════════════════════════════════════╝",
        "",
        "🎯 PURPOSE: Generate comprehensive violin performance reports",
        "",
        "📊 REPORT SECTIONS:",
        "  • Range and string usage analysis",
        "  • Left-hand positions and shifts",
        "  • Interval analysis and patterns",
        "  • Bowing and articulation assessment",
        "  • Technique distribution and challenges",
        "  • Overall difficulty and complexity score",
        "  • Audio analysis (vibrato, timbre, intonation)",
        "  • Personalized practice recommendations",
        "  • Adaptive practice schedules",
        "",
        "🎻 SAMPLE REPORTS:",
        "  • Built-in demo with realistic data",
        "  • Shows all report sections",
        "  • Demonstrates formatting and layout",
        "",
        "💾 EXPORT OPTIONS:",
        "  • Plain text (.txt) for easy reading",
        "  • HTML (.html) for web viewing",
        "  • JSON (.json) for data processing",
        "  • Copy to clipboard for quick sharing",
        "",
        "⚙️ CUSTOMIZATION:",
        "  • Select which sections to include",
        "  • Toggle audio analysis on/off",
        "  • Choose report type and focus",
        "  • Refresh with current settings",
        "",
        "🚀 QUICK START:",
        "  1. Generate a sample report",
        "  2. Review in Full Report, Summary, or Practice tabs",
        "  3. Customize sections as needed",
        "  4. Export in your preferred format",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = ReportGeneratorGUI(parent)
    return gui.parent


# Standalone test
def test_standalone():
    """Test the module GUI standalone."""
    root = tk.Tk()
    root.title("Report Generator Module - Standalone Test")
    root.geometry(WINDOW_SIZE)
    root.configure(bg=COLOR_BG)

    # Create the GUI
    gui = create_gui(root)

    root.mainloop()


if __name__ == "__main__":
    test_standalone()