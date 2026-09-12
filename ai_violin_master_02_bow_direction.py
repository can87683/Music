#!/usr/bin/env python3
# ai_violin_master_02_bow_direction.py

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import random

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


class BowDirection:
    """Core rule-based bowing logic for violin."""

    def __init__(self, style="Modern"):
        self.style = style

    def style_rules(self):
        """Returns style-specific weights."""
        if self.style == "Classical":
            return {
                "downbeat_weight": 2.0,
                "accent_weight": 2.5,
                "staccato_upbow_bias": 1.8,
                "pickup_upbow": True,
                "slur_reset": True
            }
        elif self.style == "Baroque":
            return {
                "downbeat_weight": 1.0,
                "accent_weight": 1.2,
                "staccato_upbow_bias": 2.3,
                "pickup_upbow": True,
                "slur_reset": False
            }
        # Modern default
        return {
            "downbeat_weight": 1.5,
            "accent_weight": 2.0,
            "staccato_upbow_bias": 1.5,
            "pickup_upbow": True,
            "slur_reset": True
        }

    def detect_articulations(self, note, next_note):
        """Detect articulations from MIDI data."""
        vel = note.velocity
        slur = False
        staccato = False
        accent = False

        if next_note:
            gap = next_note.start - note.end
            overlap = note.end - next_note.start

            if overlap > -0.01:
                slur = True

            if gap > 0.05 and (note.end - note.start) < 0.15:
                staccato = True

        if vel > 90:
            accent = True

        return slur, staccato, accent

    def initial_pass(self, notes):
        """Construct base bowing sequence with strict alternation."""
        plan = []
        direction = "D"

        for i, n in enumerate(notes):
            plan.append({
                "start": n.start,
                "pitch": n.pitch,
                "direction": direction,
                "slur": False,
                "staccato": False,
                "accent": False,
                "default": True
            })
            direction = "U" if direction == "D" else "D"

        return plan

    def apply_articulations(self, plan):
        """Update slur/staccato/accent flags."""
        N = len(plan)
        for i in range(N):
            if i < N - 1:
                slur, staccato, accent = self.detect_articulations(
                    plan[i]["_note"],
                    plan[i+1]["_note"]
                )
            else:
                slur, staccato, accent = self.detect_articulations(
                    plan[i]["_note"],
                    None
                )

            plan[i]["slur"] = slur
            plan[i]["staccato"] = staccato
            plan[i]["accent"] = accent

        return plan

    def apply_direction_rules(self, plan, metadata):
        """Apply rule-based direction overrides."""
        style = self.style_rules()
        ts = metadata["time_signature"]
        numerator = int(ts.split("/")[0])

        bpm = metadata["tempo"]
        beat_len = 60 / bpm

        for item in plan:
            beat_pos = item["start"] % (beat_len * numerator)
            is_downbeat = abs(beat_pos) < 0.0001

            if is_downbeat:
                if style["downbeat_weight"] > 0:
                    item["direction"] = "D"
                    item["default"] = False

            if item["accent"]:
                item["direction"] = "D"
                item["default"] = False

            if item["staccato"]:
                if np.random.random() < (style["staccato_upbow_bias"] / 2):
                    item["direction"] = "U"
                    item["default"] = False

        if style.get("pickup_upbow", True) and len(plan) >= 2:
            first = plan[0]
            if first["start"] > 0.0001:
                first["direction"] = "U"
                first["default"] = False

        N = len(plan)
        for i in range(N - 1):
            if plan[i]["slur"]:
                plan[i+1]["direction"] = plan[i]["direction"]
                plan[i+1]["default"] = False

        return plan

    def compute_bow_balance(self, plan):
        """Simulate rough bow balancing."""
        pos = 0.0
        history = []

        for item in plan:
            if item["direction"] == "D":
                pos -= 0.12
            else:
                pos += 0.12

            pos *= 0.92
            pos = max(min(pos, 1.0), -1.0)
            history.append(pos)

        difficulty = float(sum(abs(x) for x in history))
        return {
            "balance_score": difficulty,
            "trajectory": history
        }

    def ascii_bow_timeline(self, plan, width=80):
        """Produce ASCII bow timeline."""
        out = []

        for item in plan:
            d = item["direction"]
            tag = d

            if item["slur"]:
                tag += "-"
            if item["staccato"]:
                tag += "*"
            if item["accent"]:
                tag += ">"

            out.append(tag)

        line = " ".join(out)
        if len(line) > width:
            line = line[:width] + "..."
        return line

    def generate_bowing(self, notes, metadata):
        """Main API for bowing generation."""
        if not notes:
            return [], {"balance_score": 0, "trajectory": []}

        plan_list = self.initial_pass(notes)

        for item, n in zip(plan_list, notes):
            item["_note"] = n

        plan_list = self.apply_articulations(plan_list)
        plan_list = self.apply_direction_rules(plan_list, metadata)
        metrics = self.compute_bow_balance(plan_list)

        for item in plan_list:
            del item["_note"]

        return plan_list, metrics


class BowDirectionGUI:
    """GUI for Bow Direction module with AIModels style"""

    def __init__(self, parent):
        self.parent = parent
        self.parent.title("🎻 Bow Direction Analyzer")
        self.parent.geometry(WINDOW_SIZE)
        self.parent.resizable(False, False)
        self.parent.configure(bg=COLOR_BG)

        # Initialize bow direction analyzer
        self.bow_analyzer = BowDirection()

        # Sample data
        self.sample_data = self._create_sample_data()
        self.bow_plan = []
        self.bow_metrics = {}

        # Create GUI
        self._create_widgets()

    def _create_sample_data(self):
        """Create sample data for demonstration"""
        # Create mock note objects
        class MockNote:
            def __init__(self, start, pitch, velocity=80, duration=0.5):
                self.start = start
                self.pitch = pitch
                self.velocity = velocity
                self.end = start + duration
                self.duration = duration

        # Create sample notes
        notes = []
        for i in range(16):
            # Add some articulation variations
            vel = 80
            if i % 4 == 0:  # Accents on downbeats
                vel = 95
            if i == 2 or i == 10:  # Staccato notes
                duration = 0.2
            else:
                duration = 0.5

            notes.append(MockNote(
                start=i * 0.5,
                pitch=60 + (i % 8),
                velocity=vel,
                duration=duration
            ))

        # Add some slurs
        notes[3].end = notes[4].start + 0.01  # Create overlap for slur
        notes[11].end = notes[12].start + 0.01  # Another slur

        return {
            "notes": notes,
            "metadata": {
                "time_signature": "4/4",
                "tempo": 120,
                "title": "Sample Bowing Exercise",
                "composer": "Demo"
            }
        }

    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = tk.Frame(self.parent, bg=COLOR_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        self._create_header(main_frame)

        # Style configuration
        self._create_style_config(main_frame)

        # Bowing visualization
        self._create_visualization(main_frame)

        # Analysis results
        self._create_analysis_results(main_frame)

        # Controls
        self._create_controls(main_frame)

        # Status bar
        self._create_status_bar(main_frame)

    def _create_header(self, parent):
        """Create header section"""
        header_frame = tk.Frame(parent, bg=COLOR_BG)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            header_frame,
            text="🎻 Bow Direction Analyzer",
            font=FONT_HEADER,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack()

        tk.Label(
            header_frame,
            text="Rule-based bowing analysis and direction planning",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG
        ).pack()

    def _create_style_config(self, parent):
        """Create style configuration panel"""
        frame = tk.LabelFrame(
            parent,
            text="Style Configuration",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        # Style selection
        style_frame = tk.Frame(frame, bg=COLOR_BG)
        style_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            style_frame,
            text="Musical Style:",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(side=tk.LEFT)

        self.style_var = tk.StringVar(value="Modern")
        style_menu = tk.OptionMenu(
            style_frame, self.style_var,
            "Modern", "Classical", "Baroque"
        )
        style_menu.config(
            font=FONT_TEXT,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            activebackground=COLOR_FRAME,
            activeforeground=COLOR_GOLD,
            highlightthickness=0,
            width=15
        )
        style_menu["menu"].config(
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            activebackground=COLOR_FRAME,
            activeforeground=COLOR_GOLD
        )
        style_menu.pack(side=tk.LEFT, padx=5)

        # Style descriptions
        desc_frame = tk.Frame(frame, bg=COLOR_BG)
        desc_frame.pack(fill=tk.X, pady=10)

        desc_text = """Style Characteristics:
• Modern: Balanced approach, practical bowing
• Classical: Strong downbeats, traditional rules
• Baroque: Light accents, period-appropriate"""

        desc_label = tk.Label(
            desc_frame,
            text=desc_text,
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            justify=tk.LEFT
        )
        desc_label.pack(anchor="w", padx=5)

    def _create_visualization(self, parent):
        """Create bowing visualization panel"""
        frame = tk.LabelFrame(
            parent,
            text="Bowing Visualization",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10, ipadx=10, ipady=10)

        # Create notebook for different views
        self.viz_notebook = ttk.Notebook(frame)
        self.viz_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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
        self._create_timeline_tab()
        self._create_score_tab()
        self._create_balance_tab()

    def _create_timeline_tab(self):
        """Create bowing timeline tab"""
        timeline_frame = tk.Frame(self.viz_notebook, bg=COLOR_BG)
        self.viz_notebook.add(timeline_frame, text="Timeline")

        # Canvas for bowing visualization
        self.timeline_canvas = tk.Canvas(
            timeline_frame,
            bg=COLOR_DARK,
            highlightthickness=0
        )
        scrollbar = tk.Scrollbar(timeline_frame, orient=tk.HORIZONTAL)

        # Pack canvas and scrollbar
        self.timeline_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Configure scrollbar
        self.timeline_canvas.config(xscrollcommand=scrollbar.set)
        scrollbar.config(command=self.timeline_canvas.xview)

        # Create inner frame for timeline
        self.timeline_inner = tk.Frame(self.timeline_canvas, bg=COLOR_DARK)
        self.timeline_canvas.create_window((0, 0), window=self.timeline_inner, anchor="nw")

    def _create_score_tab(self):
        """Create musical score tab (simplified)"""
        score_frame = tk.Frame(self.viz_notebook, bg=COLOR_BG)
        self.viz_notebook.add(score_frame, text="Score")

        # Placeholder for musical score
        placeholder = tk.Label(
            score_frame,
            text="Musical Score Display\n\n"
                 "Shows bow directions on a staff:\n"
                 "• D = Down-bow (∏)\n"
                 "• U = Up-bow (V)\n"
                 "• - = Slur\n"
                 "• * = Staccato\n"
                 "• > = Accent\n\n"
                 "(Full staff notation would appear here)",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            justify=tk.CENTER
        )
        placeholder.pack(expand=True, fill=tk.BOTH, pady=50)

    def _create_balance_tab(self):
        """Create bow balance visualization tab"""
        balance_frame = tk.Frame(self.viz_notebook, bg=COLOR_BG)
        self.viz_notebook.add(balance_frame, text="Balance")

        # Canvas for balance visualization
        self.balance_canvas = tk.Canvas(
            balance_frame,
            bg=COLOR_DARK,
            highlightthickness=0,
            height=150
        )
        self.balance_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Labels for balance chart
        tk.Label(
            balance_frame,
            text="Bow Balance Trajectory:",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(pady=5)

    def _create_analysis_results(self, parent):
        """Create analysis results panel"""
        frame = tk.LabelFrame(
            parent,
            text="Analysis Results",
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
            ("Total Notes", "total_notes", "0"),
            ("Down-bows", "downbows", "0"),
            ("Up-bows", "upbows", "0"),
            ("Slurs", "slurs", "0"),
            ("Staccato", "staccato", "0"),
            ("Accents", "accents", "0"),
            ("Balance Score", "balance_score", "0.0"),
            ("Style", "style", "Modern")
        ]

        for i, (label, key, default) in enumerate(stats):
            row = i // 4
            col = i % 4

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
            text="Controls",
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
            ("🎵 Generate Bowing", self._generate_bowing),
            ("📁 Load MIDI Data", self._load_midi_data),
            ("🔄 Randomize Style", self._randomize_style),
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
            text="✅ Ready to analyze bowing",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_DARK
        )
        self.status_label.pack(side=tk.LEFT, padx=5)

    def _generate_bowing(self):
        """Generate bowing plan"""
        try:
            self.status_label.config(text="⏳ Generating bowing plan...")
            self.parent.update()

            # Update analyzer style
            self.bow_analyzer.style = self.style_var.get()

            # Generate bowing plan
            self.bow_plan, self.bow_metrics = self.bow_analyzer.generate_bowing(
                self.sample_data["notes"],
                self.sample_data["metadata"]
            )

            # Update visualizations
            self._update_timeline()
            self._update_balance_chart()
            self._update_statistics()

            self.status_label.config(text=f"✅ Bowing generated ({self.style_var.get()} style)")

        except Exception as e:
            messagebox.showerror("Generation Error", f"Failed to generate bowing plan:\n{str(e)}")
            self.status_label.config(text="❌ Generation failed")

    def _update_timeline(self):
        """Update bowing timeline visualization"""
        # Clear existing timeline
        for widget in self.timeline_inner.winfo_children():
            widget.destroy()

        if not self.bow_plan:
            return

        # Create timeline elements
        for i, item in enumerate(self.bow_plan):
            note_frame = tk.Frame(self.timeline_inner, bg=COLOR_DARK, relief=tk.RAISED, borderwidth=1)
            note_frame.pack(side=tk.LEFT, padx=2, pady=5, fill=tk.Y)

            # Note number
            tk.Label(
                note_frame,
                text=f"{i+1}",
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_DARK
            ).pack(pady=2)

            # Bow direction (main indicator)
            direction = item["direction"]
            direction_color = "#FF6B6B" if direction == "D" else "#4ECDC4"  # Red for down, Teal for up
            direction_symbol = "∏" if direction == "D" else "V"

            direction_label = tk.Label(
                note_frame,
                text=direction_symbol,
                font=("Georgia", 16, "bold"),
                fg=direction_color,
                bg=COLOR_DARK
            )
            direction_label.pack(pady=2)

            # Articulation indicators
            articulations = []
            if item["slur"]:
                articulations.append("—")
            if item["staccato"]:
                articulations.append("•")
            if item["accent"]:
                articulations.append(">")

            if articulations:
                tk.Label(
                    note_frame,
                    text=" ".join(articulations),
                    font=FONT_SMALL,
                    fg=COLOR_GOLD,
                    bg=COLOR_DARK
                ).pack(pady=2)

            # Pitch
            tk.Label(
                note_frame,
                text=self._midi_to_note(item["pitch"]),
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_DARK
            ).pack(pady=2)

        # Update canvas scroll region
        self.timeline_inner.update_idletasks()
        self.timeline_canvas.config(scrollregion=self.timeline_canvas.bbox("all"))

    def _update_balance_chart(self):
        """Update bow balance chart"""
        self.balance_canvas.delete("all")

        if not self.bow_metrics or "trajectory" not in self.bow_metrics:
            return

        trajectory = self.bow_metrics["trajectory"]
        if not trajectory:
            return

        width = self.balance_canvas.winfo_width()
        height = self.balance_canvas.winfo_height()

        if width < 10 or height < 10:
            width = 400
            height = 150

        padding = 20
        chart_width = width - 2 * padding
        chart_height = height - 2 * padding

        # Draw center line
        center_y = height // 2
        self.balance_canvas.create_line(
            padding, center_y,
            width - padding, center_y,
            fill=COLOR_TEXT, width=1, dash=(2, 2)
        )

        # Draw balance trajectory
        if len(trajectory) > 1:
            points = []
            for i, value in enumerate(trajectory):
                x = padding + (i / (len(trajectory) - 1)) * chart_width
                y = center_y - (value * chart_height / 2)
                points.extend([x, y])

            self.balance_canvas.create_line(
                points, fill="#4ECDC4", width=2, smooth=True
            )

            # Draw points
            for i, value in enumerate(trajectory):
                x = padding + (i / (len(trajectory) - 1)) * chart_width
                y = center_y - (value * chart_height / 2)

                color = "#FF6B6B" if value < 0 else "#4ECDC4"
                self.balance_canvas.create_oval(
                    x-3, y-3, x+3, y+3,
                    fill=color, outline=color
                )

        # Draw labels
        self.balance_canvas.create_text(
            padding, padding,
            text="Tip",
            fill=COLOR_TEXT,
            font=FONT_SMALL,
            anchor="nw"
        )
        self.balance_canvas.create_text(
            padding, height - padding,
            text="Frog",
            fill=COLOR_TEXT,
            font=FONT_SMALL,
            anchor="sw"
        )

    def _update_statistics(self):
        """Update statistics display"""
        if not self.bow_plan:
            return

        # Calculate statistics
        total = len(self.bow_plan)
        downbows = sum(1 for item in self.bow_plan if item["direction"] == "D")
        upbows = total - downbows
        slurs = sum(1 for item in self.bow_plan if item["slur"])
        staccato = sum(1 for item in self.bow_plan if item["staccato"])
        accents = sum(1 for item in self.bow_plan if item["accent"])
        balance_score = self.bow_metrics.get("balance_score", 0)

        # Update labels
        self.stats_labels["total_notes"].config(text=str(total))
        self.stats_labels["downbows"].config(text=str(downbows))
        self.stats_labels["upbows"].config(text=str(upbows))
        self.stats_labels["slurs"].config(text=str(slurs))
        self.stats_labels["staccato"].config(text=str(staccato))
        self.stats_labels["accents"].config(text=str(accents))
        self.stats_labels["balance_score"].config(text=f"{balance_score:.2f}")
        self.stats_labels["style"].config(text=self.style_var.get())

    def _midi_to_note(self, midi):
        """Convert MIDI number to note name."""
        note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        return f"{note_names[midi % 12]}{midi//12 - 1}"

    def _load_midi_data(self):
        """Load MIDI data from file"""
        # In a real implementation, this would parse MIDI files
        # For now, just show a message
        messagebox.showinfo("MIDI Load",
                          "MIDI file loading would be implemented here.\n\n"
                          "Currently using sample data for demonstration.")
        self.status_label.config(text="✅ MIDI loading placeholder")

    def _randomize_style(self):
        """Randomly select a different style"""
        styles = ["Modern", "Classical", "Baroque"]
        current = self.style_var.get()

        # Pick a different style
        available = [s for s in styles if s != current]
        if available:
            new_style = random.choice(available)
            self.style_var.set(new_style)

            # Regenerate bowing if we already have a plan
            if self.bow_plan:
                self._generate_bowing()

            self.status_label.config(text=f"✅ Switched to {new_style} style")

    def _export_results(self):
        """Export bowing results to file"""
        if not self.bow_plan:
            messagebox.showwarning("No Results", "Please generate bowing plan first")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Bowing Analysis",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                # Prepare export data
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "style": self.style_var.get(),
                    "bow_plan": self.bow_plan,
                    "metrics": self.bow_metrics,
                    "metadata": self.sample_data["metadata"],
                    "statistics": {
                        "total_notes": len(self.bow_plan),
                        "downbows": sum(1 for item in self.bow_plan if item["direction"] == "D"),
                        "upbows": sum(1 for item in self.bow_plan if item["direction"] == "U"),
                        "slurs": sum(1 for item in self.bow_plan if item["slur"]),
                        "staccato": sum(1 for item in self.bow_plan if item["staccato"]),
                        "accents": sum(1 for item in self.bow_plan if item["accent"]),
                        "balance_score": self.bow_metrics.get("balance_score", 0)
                    }
                }

                if filename.endswith('.json'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2)

                elif filename.endswith('.txt'):
                    # Create text report
                    report = self._create_text_report()
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(report)

                self.status_label.config(text=f"✅ Results exported to {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Results saved to:\n{filename}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export results:\n{str(e)}")
                self.status_label.config(text="❌ Export failed")

    def _create_text_report(self):
        """Create text report for export"""
        report = "🎻 BOWING ANALYSIS REPORT\n"
        report += "=" * 50 + "\n\n"

        report += f"Style: {self.style_var.get()}\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        if self.bow_plan:
            report += "BOWING PLAN:\n"
            report += "-" * 40 + "\n"

            for i, item in enumerate(self.bow_plan):
                direction = "Down-bow (∏)" if item["direction"] == "D" else "Up-bow (V)"
                articulations = []
                if item["slur"]:
                    articulations.append("Slur")
                if item["staccato"]:
                    articulations.append("Staccato")
                if item["accent"]:
                    articulations.append("Accent")

                artic_str = ", ".join(articulations) if articulations else "Normal"
                note_name = self._midi_to_note(item["pitch"])

                report += f"Note {i+1}: {note_name} - {direction} - {artic_str}\n"

            report += "\n"

        if self.bow_metrics:
            report += "ANALYSIS METRICS:\n"
            report += "-" * 40 + "\n"
            report += f"Balance Score: {self.bow_metrics.get('balance_score', 0):.2f}\n"
            report += f"(Lower score = better bow balance)\n\n"

        return report


# Module interface functions for backward compatibility
def get_content():
    """Return module description for display in GUI."""
    return [
        "╔══════════════════════════════════════╗",
        "║       BOW DIRECTION ANALYZER         ║",
        "╚══════════════════════════════════════╝",
        "",
        "🎯 PURPOSE: Rule-based bowing analysis and direction planning",
        "",
        "🎵 STYLES SUPPORTED:",
        "  • Modern: Balanced, practical approach",
        "  • Classical: Traditional downbeat emphasis",
        "  • Baroque: Period-appropriate lighter bowing",
        "",
        "📊 ANALYSIS FEATURES:",
        "  • Automatic detection of slurs, staccato, accents",
        "  • Rule-based bow direction assignment",
        "  • Bow balance simulation and scoring",
        "  • Musical style adaptation",
        "",
        "👁️ VISUALIZATION:",
        "  • Interactive bowing timeline",
        "  • Color-coded direction indicators",
        "  • Bow balance trajectory chart",
        "  • Musical score representation",
        "",
        "🎻 BOWING RULES:",
        "  • Downbeats typically receive down-bows",
        "  • Accented notes favor down-bows",
        "  • Staccato notes may use up-bows",
        "  • Slurs maintain bow direction",
        "  • Pickup notes often start with up-bows",
        "",
        "📈 BALANCE ANALYSIS:",
        "  • Simulates bow position during playing",
        "  • Scores overall bow balance efficiency",
        "  • Visualizes bow trajectory",
        "  • Helps identify problematic passages",
        "",
        "🚀 QUICK START:",
        "  1. Select musical style",
        "  2. Click 'Generate Bowing'",
        "  3. Review timeline visualization",
        "  4. Check balance analysis",
        "  5. Export results as needed",
        "",
        "💡 PRACTICAL APPLICATIONS:",
        "  • Practice preparation and planning",
        "  • Teaching bowing technique",
        "  • Analyzing complex passages",
        "  • Developing consistent bowing habits",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = BowDirectionGUI(parent)
    return gui.parent


# Standalone test
def test_standalone():
    """Test the module GUI standalone."""
    root = tk.Tk()
    root.title("Bow Direction Analyzer - Standalone Test")
    root.geometry(WINDOW_SIZE)
    root.configure(bg=COLOR_BG)

    # Create the GUI
    gui = create_gui(root)

    root.mainloop()


if __name__ == "__main__":
    test_standalone()