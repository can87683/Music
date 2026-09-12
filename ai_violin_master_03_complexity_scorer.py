#!/usr/bin/env python3
# ai_violin_master_03_complexity_scorer.py

from typing import Dict, List
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
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


class ComplexityScorer:
    """Quantify musical and technical difficulty."""

    def _rhythmic_complexity(self, metadata: Dict) -> float:
        """Rhythmic complexity based on tempo changes + note density variation."""
        tempo_var = metadata.get("tempo_variation", 0.0)
        nps = metadata.get("notes_per_second", 0.0)

        tempo_score = min(tempo_var * 10, 40)
        density_score = min(nps * 6, 40)

        return tempo_score + density_score

    def _technical_complexity(self, violin_data: Dict) -> float:
        """Calculate technical complexity."""
        iv = violin_data["intervals"]
        pos = violin_data["positions"]
        bow = violin_data["bowing"]
        tstats = violin_data["technique_stats"]

        score = 0
        score += min(iv.large_intervals * 4, 25)
        score += min(pos["shifts"] * 2.5, 25)
        score += min(pos["large_shifts"] * 4, 20)
        score += min(tstats["fast_passages"] * 2.0, 20)
        score += min(tstats["double_stops"] * 6, 20)
        score += min((tstats["ricochet_patterns"] + tstats["tremolo_patterns"]) * 1.5, 15)

        if tstats["technique_variety"] == "High":
            score += 15
        elif tstats["technique_variety"] == "Medium":
            score += 7

        return min(score, 100)

    def _range_difficulty(self, metadata: Dict) -> float:
        """Calculate range difficulty."""
        (low, high) = metadata["pitch_range_midi"]
        width = high - low
        return min((width / 41) * 100, 100)

    def _articulation_complexity(self, violin_data: Dict) -> float:
        """Calculate articulation complexity."""
        bow = violin_data["bowing"]
        slur_score = min(bow.slur_groups * 3.5, 40)
        var_score = min(bow.articulation_consistency * 25, 60)
        return slur_score + var_score

    def _difficulty_level(self, score: float) -> str:
        """Map score to difficulty level."""
        if score >= 85:
            return "Advanced"
        if score >= 70:
            return "Upper Intermediate"
        if score >= 55:
            return "Intermediate"
        if score >= 30:
            return "Early Intermediate"
        return "Beginner"

    def score_complexity(self, metadata: Dict, violin_data: Dict) -> Dict:
        """Compute overall piece difficulty score."""
        rhythmic = self._rhythmic_complexity(metadata)
        technical = self._technical_complexity(violin_data)
        range_c = self._range_difficulty(metadata)
        articulation = self._articulation_complexity(violin_data)

        total = (
            rhythmic * 0.20 +
            technical * 0.45 +
            range_c * 0.20 +
            articulation * 0.15
        )

        level = self._difficulty_level(total)

        return {
            "rhythmic": float(rhythmic),
            "technical": float(technical),
            "range_complexity": float(range_c),
            "articulation": float(articulation),
            "total": float(total),
            "level": level
        }

    def generate_recommendations(self, violin_data: Dict, complexity: Dict) -> list:
        """Generate practice recommendations."""
        rec = []
        pos = violin_data["positions"]
        iv = violin_data["intervals"]
        bow = violin_data["bowing"]
        tstats = violin_data["technique_stats"]
        dstop = violin_data["double_stops"]

        if pos["shifts"] > 12:
            rec.append("Practice slow glissando shifting drills (1–3, 2–4 positions).")
        if pos["large_shifts"] > 5:
            rec.append("Add Sevcik Op.8 shifting exercises for large leaps.")
        if iv.large_intervals > 6:
            rec.append("Practice octave and tenth leaps slowly; isolate wide shifts.")
        if iv.max_interval > 12:
            rec.append("Decompose extreme leaps (>12 semitones) into guide-tone steps.")
        if tstats["fast_passages"] > 12:
            rec.append("Use rhythmic alteration practice (long–short, short–long) on fast runs.")
        if dstop > 0:
            rec.append("Tune double stops slowly; practice with perfect fifth drones.")
        if bow.slur_groups > 6:
            rec.append("Practice smooth bow changes; slow 'son filé' exercises recommended.")
        if bow.articulation_consistency > 0.15:
            rec.append("Stabilize articulation with slow détaché on open strings.")
        if tstats["ricochet_patterns"] > 0:
            rec.append("Isolate ricochet strokes with controlled bounce at the balance point.")
        if tstats["tremolo_patterns"] > 0:
            rec.append("Practice tremolo speed control with metronome subdivisions.")

        if complexity.get("level") == "Advanced":
            rec.append("Piece demands strong technical foundation; consider adding Paganini Caprices for support.")
        elif complexity.get("level") == "Upper Intermediate":
            rec.append("Kreutzer studies 2, 7, 9, and 32 support this piece well.")
        elif complexity.get("level") == "Intermediate":
            rec.append("Focus on Sevcik bowing Op.2 for articulation development.")

        return list(dict.fromkeys(rec))

    def _suggest_etudes(self, violin_data: Dict, complexity: Dict) -> list:
        """Return additional etudes for deeper training."""
        etudes = []

        if violin_data["positions"]["large_shifts"] > 5:
            etudes.append("Kreutzer #11 – shifting control")
        if violin_data["intervals"].large_intervals > 6:
            etudes.append("Dont Op.35 #6 – large leaps & position stability")
        if violin_data["double_stops"] > 4:
            etudes.append("Kreutzer #33 – double-stop intonation")
        if violin_data["bowing"].bowing_complexity == "High":
            etudes.append("Kreutzer #2 – bow control & détaché consistency")
        if complexity.get("level") == "Advanced":
            etudes.append("Paganini Caprice #1 – arpeggio clarity & string crossing")

        return etudes


class ComplexityScoresGUI:
    """GUI for Complexity Scorer module with AIModels style"""

    def __init__(self, parent):
        self.parent = parent
        self.parent.title("🎻 Complexity Scorer")
        self.parent.geometry(WINDOW_SIZE)
        self.parent.resizable(False, False)
        self.parent.configure(bg=COLOR_BG)

        # Initialize complexity scorer
        self.scorer = ComplexityScorer()

        # Sample data for demonstration
        self.sample_data = self._create_sample_data()
        self.complexity_scores = None
        self.recommendations = []
        self.etudes = []

        # Create GUI
        self._create_widgets()

    def _create_sample_data(self):
        """Create sample data for demonstration"""
        # Create mock objects for violin data
        class MockIntervals:
            def __init__(self):
                self.large_intervals = 8
                self.max_interval = 14

        class MockBowing:
            def __init__(self):
                self.slur_groups = 7
                self.articulation_consistency = 0.18
                self.bowing_complexity = "Medium"

        return {
            "metadata": {
                "tempo_variation": 3.5,
                "notes_per_second": 6.2,
                "pitch_range_midi": (55, 92)
            },
            "violin_data": {
                "intervals": MockIntervals(),
                "positions": {
                    "shifts": 15,
                    "large_shifts": 6
                },
                "bowing": MockBowing(),
                "technique_stats": {
                    "fast_passages": 14,
                    "double_stops": 5,
                    "ricochet_patterns": 2,
                    "tremolo_patterns": 1,
                    "technique_variety": "High"
                },
                "double_stops": 5
            }
        }

    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = tk.Frame(self.parent, bg=COLOR_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        self._create_header(main_frame)

        # Score calculation panel
        self._create_score_panel(main_frame)

        # Results display
        self._create_results_display(main_frame)

        # Recommendations panel
        self._create_recommendations_panel(main_frame)

        # Control panel
        self._create_control_panel(main_frame)

        # Status bar
        self._create_status_bar(main_frame)

    def _create_header(self, parent):
        """Create header section"""
        header_frame = tk.Frame(parent, bg=COLOR_BG)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            header_frame,
            text="🎻 Violin Complexity Scorer",
            font=FONT_HEADER,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack()

        tk.Label(
            header_frame,
            text="Quantify musical and technical difficulty of violin pieces",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG
        ).pack()

    def _create_score_panel(self, parent):
        """Create score calculation panel"""
        frame = tk.LabelFrame(
            parent,
            text="Score Calculation",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        # Score weights information
        weights_frame = tk.Frame(frame, bg=COLOR_BG)
        weights_frame.pack(fill=tk.X, pady=5)

        weights_info = """
        Score Components (Weighted):
        • Rhythmic Complexity (20%): Tempo changes, note density
        • Technical Complexity (45%): Intervals, shifts, techniques
        • Range Difficulty (20%): Pitch range width
        • Articulation Complexity (15%): Bowing patterns, consistency
        """

        weights_label = tk.Label(
            weights_frame,
            text=weights_info,
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            justify=tk.LEFT
        )
        weights_label.pack(anchor="w")

        # Calculate button
        calc_btn = tk.Button(
            frame,
            text="📊 Calculate Complexity",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_DARK,
            activeforeground=COLOR_GOLD,
            activebackground=COLOR_FRAME,
            relief=tk.RAISED,
            borderwidth=2,
            command=self._calculate_complexity,
            width=25
        )
        calc_btn.pack(pady=10)

    def _create_results_display(self, parent):
        """Create results display area"""
        frame = tk.LabelFrame(
            parent,
            text="Complexity Scores",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        # Create notebook for different views
        self.results_notebook = ttk.Notebook(frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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
        self._create_scores_tab()
        self._create_radar_tab()
        self._create_details_tab()

    def _create_scores_tab(self):
        """Create scores display tab"""
        scores_frame = tk.Frame(self.results_notebook, bg=COLOR_BG)
        self.results_notebook.add(scores_frame, text="Scores")

        # Score cards in grid
        cards_frame = tk.Frame(scores_frame, bg=COLOR_BG)
        cards_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Define score cards
        self.score_cards = {}
        scores = [
            ("🎵 Rhythmic", "rhythmic", "#FF6B6B"),
            ("⚡ Technical", "technical", "#4ECDC4"),
            ("🎻 Range", "range_complexity", "#FFD166"),
            ("🏹 Articulation", "articulation", "#06D6A0"),
            ("🏆 Total Score", "total", "#118AB2")
        ]

        for i, (label, key, color) in enumerate(scores):
            row = i // 2
            col = i % 2

            card_frame = tk.Frame(cards_frame, bg=COLOR_DARK, relief=tk.RAISED, borderwidth=2)
            card_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            cards_frame.rowconfigure(row, weight=1)
            cards_frame.columnconfigure(col, weight=1)

            # Card label
            tk.Label(
                card_frame,
                text=label,
                font=FONT_TEXT,
                fg=COLOR_GOLD,
                bg=COLOR_DARK
            ).pack(pady=(10, 5))

            # Score value
            score_label = tk.Label(
                card_frame,
                text="0.0",
                font=("Georgia", 24, "bold"),
                fg=color,
                bg=COLOR_DARK
            )
            score_label.pack(pady=5)

            # Score bar
            canvas = tk.Canvas(card_frame, height=20, bg=COLOR_DARK, highlightthickness=0)
            canvas.pack(fill=tk.X, padx=10, pady=5)
            bar = canvas.create_rectangle(0, 0, 0, 20, fill=color, outline="")

            # Max score label
            tk.Label(
                card_frame,
                text="/100",
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_DARK
            ).pack(pady=(0, 10))

            self.score_cards[key] = {
                "label": score_label,
                "bar": bar,
                "canvas": canvas
            }

        # Difficulty level display
        level_frame = tk.Frame(scores_frame, bg=COLOR_BG)
        level_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            level_frame,
            text="Difficulty Level:",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(side=tk.LEFT)

        self.level_label = tk.Label(
            level_frame,
            text="Not Calculated",
            font=("Georgia", 16, "bold"),
            fg=COLOR_GOLD,
            bg=COLOR_BG
        )
        self.level_label.pack(side=tk.LEFT, padx=10)

    def _create_radar_tab(self):
        """Create radar chart tab (simulated)"""
        radar_frame = tk.Frame(self.results_notebook, bg=COLOR_BG)
        self.results_notebook.add(radar_frame, text="Radar Chart")

        # Placeholder for radar chart
        placeholder = tk.Label(
            radar_frame,
            text="Radar Chart Visualization\n\n"
                 "Shows relative strengths across complexity dimensions\n\n"
                 "• Rhythmic Complexity\n"
                 "• Technical Difficulty\n"
                 "• Range Challenge\n"
                 "• Articulation Complexity\n\n"
                 "(Interactive chart would appear here with matplotlib)",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            justify=tk.CENTER
        )
        placeholder.pack(expand=True, fill=tk.BOTH, pady=50)

    def _create_details_tab(self):
        """Create detailed breakdown tab"""
        details_frame = tk.Frame(self.results_notebook, bg=COLOR_BG)
        self.results_notebook.add(details_frame, text="Details")

        # Text widget for detailed breakdown
        scrollbar = tk.Scrollbar(details_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.details_text = tk.Text(
            details_frame,
            wrap=tk.WORD,
            font=FONT_SMALL,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            yscrollcommand=scrollbar.set,
            height=10
        )
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.details_text.yview)

        # Insert placeholder
        self.details_text.insert(1.0, "Detailed breakdown will appear here after calculation.")
        self.details_text.config(state=tk.DISABLED)

    def _create_recommendations_panel(self, parent):
        """Create recommendations panel"""
        frame = tk.LabelFrame(
            parent,
            text="Practice Recommendations",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10, ipadx=10, ipady=10)

        # Create notebook for recommendations
        self.rec_notebook = ttk.Notebook(frame)
        self.rec_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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
        self._create_practice_tab()
        self._create_etudes_tab()
        self._create_schedule_tab()

    def _create_practice_tab(self):
        """Create practice recommendations tab"""
        practice_frame = tk.Frame(self.rec_notebook, bg=COLOR_BG)
        self.rec_notebook.add(practice_frame, text="Practice Tips")

        # Scrollable text widget
        scrollbar = tk.Scrollbar(practice_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.practice_text = tk.Text(
            practice_frame,
            wrap=tk.WORD,
            font=FONT_SMALL,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            yscrollcommand=scrollbar.set,
            height=8
        )
        self.practice_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.practice_text.yview)

        # Insert placeholder
        self.practice_text.insert(1.0, "Practice recommendations will appear here.\n\nClick 'Calculate Complexity' to generate recommendations.")
        self.practice_text.config(state=tk.DISABLED)

    def _create_etudes_tab(self):
        """Create etudes suggestions tab"""
        etudes_frame = tk.Frame(self.rec_notebook, bg=COLOR_BG)
        self.rec_notebook.add(etudes_frame, text="Etudes")

        # Scrollable text widget
        scrollbar = tk.Scrollbar(etudes_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.etudes_text = tk.Text(
            etudes_frame,
            wrap=tk.WORD,
            font=FONT_SMALL,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            yscrollcommand=scrollbar.set,
            height=8
        )
        self.etudes_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.etudes_text.yview)

        # Insert placeholder
        self.etudes_text.insert(1.0, "Suggested etudes will appear here.\n\nThese are additional studies to support technical development.")
        self.etudes_text.config(state=tk.DISABLED)

    def _create_schedule_tab(self):
        """Create practice schedule tab"""
        schedule_frame = tk.Frame(self.rec_notebook, bg=COLOR_BG)
        self.rec_notebook.add(schedule_frame, text="Schedule")

        # Scrollable text widget
        scrollbar = tk.Scrollbar(schedule_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.schedule_text = tk.Text(
            schedule_frame,
            wrap=tk.WORD,
            font=FONT_SMALL,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            yscrollcommand=scrollbar.set,
            height=8
        )
        self.schedule_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.schedule_text.yview)

        # Insert placeholder
        self.schedule_text.insert(1.0, "Practice schedule will appear here.\n\nBased on complexity level, a suggested practice plan will be generated.")
        self.schedule_text.config(state=tk.DISABLED)

    def _create_control_panel(self, parent):
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
            ("📁 Load Analysis", self._load_analysis_data),
            ("🎵 Use Sample Data", self._use_sample_data),
            ("💾 Export Results", self._export_results),
            ("📋 Copy Report", self._copy_report)
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
            text="✅ Ready to analyze complexity",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_DARK
        )
        self.status_label.pack(side=tk.LEFT, padx=5)

    def _calculate_complexity(self):
        """Calculate complexity scores"""
        try:
            self.status_label.config(text="⏳ Calculating complexity scores...")
            self.parent.update()

            # Calculate scores
            self.complexity_scores = self.scorer.score_complexity(
                self.sample_data["metadata"],
                self.sample_data["violin_data"]
            )

            # Generate recommendations
            self.recommendations = self.scorer.generate_recommendations(
                self.sample_data["violin_data"],
                self.complexity_scores
            )

            # Generate etudes
            self.etudes = self.scorer._suggest_etudes(
                self.sample_data["violin_data"],
                self.complexity_scores
            )

            # Update display
            self._update_score_display()
            self._update_recommendations_display()
            self._generate_practice_schedule()

            self.status_label.config(text=f"✅ Complexity calculated: {self.complexity_scores['level']}")

        except Exception as e:
            messagebox.showerror("Calculation Error", f"Failed to calculate complexity:\n{str(e)}")
            self.status_label.config(text="❌ Calculation failed")

    def _update_score_display(self):
        """Update score display with calculated values"""
        if self.complexity_scores is None:
            return

        # Update score cards
        for key, card in self.score_cards.items():
            if key in self.complexity_scores:
                score = self.complexity_scores[key]
                card["label"].config(text=f"{score:.1f}")

                # Update progress bar
                canvas = card["canvas"]
                width = canvas.winfo_width()
                if width < 10:
                    width = 150  # Default width if canvas not yet sized

                bar_width = (score / 100) * width
                canvas.coords(card["bar"], 0, 0, bar_width, 20)

        # Update level label
        level = self.complexity_scores["level"]
        level_colors = {
            "Beginner": "#4CAF50",
            "Early Intermediate": "#8BC34A",
            "Intermediate": "#FFC107",
            "Upper Intermediate": "#FF9800",
            "Advanced": "#F44336"
        }
        color = level_colors.get(level, COLOR_GOLD)

        self.level_label.config(text=level, fg=color)

        # Update details text
        details = f"""Complexity Score Breakdown:

🎵 RHYTHMIC COMPLEXITY: {self.complexity_scores['rhythmic']:.1f}/100
• Tempo variation: {self.sample_data['metadata']['tempo_variation']} BPM variation
• Note density: {self.sample_data['metadata']['notes_per_second']:.1f} notes/second

⚡ TECHNICAL COMPLEXITY: {self.complexity_scores['technical']:.1f}/100
• Large intervals: {self.sample_data['violin_data']['intervals'].large_intervals}
• Position shifts: {self.sample_data['violin_data']['positions']['shifts']}
• Fast passages: {self.sample_data['violin_data']['technique_stats']['fast_passages']}
• Double stops: {self.sample_data['violin_data']['technique_stats']['double_stops']}

🎻 RANGE DIFFICULTY: {self.complexity_scores['range_complexity']:.1f}/100
• Pitch range: {self.sample_data['metadata']['pitch_range_midi'][0]} to {self.sample_data['metadata']['pitch_range_midi'][1]} MIDI
• Range width: {self.sample_data['metadata']['pitch_range_midi'][1] - self.sample_data['metadata']['pitch_range_midi'][0]} semitones

🏹 ARTICULATION COMPLEXITY: {self.complexity_scores['articulation']:.1f}/100
• Slur groups: {self.sample_data['violin_data']['bowing'].slur_groups}
• Articulation consistency: {self.sample_data['violin_data']['bowing'].articulation_consistency:.2f}

🏆 OVERALL SCORE: {self.complexity_scores['total']:.1f}/100
• Difficulty Level: {level}
"""

        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, details)
        self.details_text.config(state=tk.DISABLED)

    def _update_recommendations_display(self):
        """Update recommendations display"""
        if not self.recommendations:
            return

        # Update practice tips
        practice_text = "🎻 PRACTICE RECOMMENDATIONS:\n\n"
        for i, rec in enumerate(self.recommendations, 1):
            practice_text += f"{i}. {rec}\n\n"

        self.practice_text.config(state=tk.NORMAL)
        self.practice_text.delete(1.0, tk.END)
        self.practice_text.insert(1.0, practice_text)
        self.practice_text.config(state=tk.DISABLED)

        # Update etudes
        if self.etudes:
            etudes_text = "📚 SUGGESTED ETUDES:\n\n"
            for etude in self.etudes:
                etudes_text += f"• {etude}\n"
        else:
            etudes_text = "No specific etudes recommended.\n\nFocus on the practice recommendations above."

        self.etudes_text.config(state=tk.NORMAL)
        self.etudes_text.delete(1.0, tk.END)
        self.etudes_text.insert(1.0, etudes_text)
        self.etudes_text.config(state=tk.DISABLED)

    def _generate_practice_schedule(self):
        """Generate practice schedule based on complexity level"""
        if self.complexity_scores is None:
            return

        level = self.complexity_scores["level"]
        total_score = self.complexity_scores["total"]

        # Generate schedule based on level
        schedule = f"🎯 PRACTICE SCHEDULE for {level} LEVEL\n"
        schedule += "=" * 40 + "\n\n"

        if level == "Advanced":
            schedule += "Weekly Structure (4 weeks):\n"
            schedule += "• Week 1-2: Technical foundations (scales, arpeggios)\n"
            schedule += "• Week 3: Piece breakdown by sections\n"
            schedule += "• Week 4: Integration and performance practice\n\n"
            schedule += "Daily (60-90 minutes):\n• 20min Scales & Arpeggios\n• 30min Technical sections\n• 20min Musical expression\n• 10min Sight-reading"

        elif level == "Upper Intermediate":
            schedule += "Weekly Structure (3 weeks):\n"
            schedule += "• Week 1: Technical challenges isolation\n"
            schedule += "• Week 2: Phrase-by-phrase practice\n"
            schedule += "• Week 3: Tempo building and dynamics\n\n"
            schedule += "Daily (45-60 minutes):\n• 15min Scales\n• 25min Section work\n• 15min Whole piece run-through"

        elif level == "Intermediate":
            schedule += "Weekly Structure (2 weeks):\n"
            schedule += "• Week 1: Learn notes and fingerings\n"
            schedule += "• Week 2: Add bowing and expression\n\n"
            schedule += "Daily (30-45 minutes):\n• 10min Warm-up\n• 20min Piece practice\n• 5min Review"

        else:  # Beginner/Early Intermediate
            schedule += "Weekly Structure (1-2 weeks):\n"
            schedule += "• Focus on one section at a time\n"
            schedule += "• Master before moving to next\n\n"
            schedule += "Daily (20-30 minutes):\n• 5min Open strings\n• 15min Section practice\n• 5min Play-through"

        schedule += "\n\n💡 Tips:\n"
        schedule += "• Always use metronome for rhythmic accuracy\n"
        schedule += "• Record yourself regularly for progress tracking\n"
        schedule += "• Focus on quality over quantity of practice\n"

        self.schedule_text.config(state=tk.NORMAL)
        self.schedule_text.delete(1.0, tk.END)
        self.schedule_text.insert(1.0, schedule)
        self.schedule_text.config(state=tk.DISABLED)

    def _load_analysis_data(self):
        """Load analysis data from file"""
        filename = filedialog.askopenfilename(
            title="Load Analysis Data",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Process the loaded data
                # (In a real implementation, you would parse the data structure)
                messagebox.showinfo("Data Loaded",
                                  f"Analysis data loaded from:\n{filename}\n\n"
                                  f"(Sample data is being used for demonstration)")

                self.status_label.config(text=f"✅ Loaded analysis from {os.path.basename(filename)}")

            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load analysis data:\n{str(e)}")
                self.status_label.config(text="❌ Failed to load data")

    def _use_sample_data(self):
        """Use built-in sample data"""
        self.sample_data = self._create_sample_data()
        self.complexity_scores = None
        self.recommendations = []
        self.etudes = []

        # Reset displays
        self._reset_displays()

        self.status_label.config(text="✅ Using sample data. Click 'Calculate Complexity' to analyze.")
        messagebox.showinfo("Sample Data", "Using built-in sample data for demonstration.")

    def _export_results(self):
        """Export results to file"""
        if self.complexity_scores is None:
            messagebox.showwarning("No Results", "Please calculate complexity first")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Complexity Report",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                # Prepare export data
                export_data = {
                    "timestamp": datetime.now().isoformat(),
                    "complexity_scores": self.complexity_scores,
                    "recommendations": self.recommendations,
                    "etudes": self.etudes,
                    "sample_data": self.sample_data
                }

                if filename.endswith('.json'):
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2)

                elif filename.endswith('.txt'):
                    # Create text report
                    report = self._create_text_report()
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(report)

                self.status_label.config(text=f"✅ Report exported to {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Report saved to:\n{filename}")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export report:\n{str(e)}")
                self.status_label.config(text="❌ Export failed")

    def _copy_report(self):
        """Copy report to clipboard"""
        if self.complexity_scores is None:
            messagebox.showwarning("No Results", "Please calculate complexity first")
            return

        try:
            report = self._create_text_report()
            self.parent.clipboard_clear()
            self.parent.clipboard_append(report)

            self.status_label.config(text="✅ Report copied to clipboard")
            messagebox.showinfo("Copied", "Complexity report copied to clipboard")

        except Exception as e:
            messagebox.showerror("Copy Error", f"Failed to copy to clipboard:\n{str(e)}")
            self.status_label.config(text="❌ Copy failed")

    def _create_text_report(self):
        """Create text report for export/copy"""
        report = "🎻 VIOLIN COMPLEXITY ANALYSIS REPORT\n"
        report += "=" * 50 + "\n\n"

        if self.complexity_scores:
            report += f"OVERALL SCORE: {self.complexity_scores['total']:.1f}/100\n"
            report += f"DIFFICULTY LEVEL: {self.complexity_scores['level']}\n\n"

            report += "SCORE BREAKDOWN:\n"
            report += f"• Rhythmic Complexity: {self.complexity_scores['rhythmic']:.1f}/100\n"
            report += f"• Technical Complexity: {self.complexity_scores['technical']:.1f}/100\n"
            report += f"• Range Difficulty: {self.complexity_scores['range_complexity']:.1f}/100\n"
            report += f"• Articulation Complexity: {self.complexity_scores['articulation']:.1f}/100\n\n"

        if self.recommendations:
            report += "PRACTICE RECOMMENDATIONS:\n"
            for i, rec in enumerate(self.recommendations, 1):
                report += f"{i}. {rec}\n"
            report += "\n"

        if self.etudes:
            report += "SUGGESTED ETUDES:\n"
            for etude in self.etudes:
                report += f"• {etude}\n"
            report += "\n"

        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        return report

    def _reset_displays(self):
        """Reset all displays to initial state"""
        # Reset score cards
        for key, card in self.score_cards.items():
            card["label"].config(text="0.0")
            canvas = card["canvas"]
            canvas.coords(card["bar"], 0, 0, 0, 20)

        # Reset level label
        self.level_label.config(text="Not Calculated", fg=COLOR_GOLD)

        # Reset text displays
        texts = [self.details_text, self.practice_text, self.etudes_text, self.schedule_text]
        placeholders = [
            "Detailed breakdown will appear here after calculation.",
            "Practice recommendations will appear here.\n\nClick 'Calculate Complexity' to generate recommendations.",
            "Suggested etudes will appear here.\n\nThese are additional studies to support technical development.",
            "Practice schedule will appear here.\n\nBased on complexity level, a suggested practice plan will be generated."
        ]

        for text_widget, placeholder in zip(texts, placeholders):
            text_widget.config(state=tk.NORMAL)
            text_widget.delete(1.0, tk.END)
            text_widget.insert(1.0, placeholder)
            text_widget.config(state=tk.DISABLED)


# Module interface functions for backward compatibility
def get_content():
    """Return module description for display in GUI."""
    return [
        "╔══════════════════════════════════════╗",
        "║      COMPLEXITY SCORER MODULE        ║",
        "╚══════════════════════════════════════╝",
        "",
        "🎯 PURPOSE: Quantify musical and technical difficulty of violin pieces",
        "",
        "📊 SCORING DIMENSIONS:",
        "  • Rhythmic Complexity: Tempo changes, note density",
        "  • Technical Complexity: Intervals, shifts, techniques",
        "  • Range Difficulty: Pitch range and tessitura",
        "  • Articulation Complexity: Bowing patterns, consistency",
        "",
        "🏆 DIFFICULTY LEVELS:",
        "  • Advanced (85-100): Professional repertoire",
        "  • Upper Intermediate (70-84): Conservatory level",
        "  • Intermediate (55-69): Serious student",
        "  • Early Intermediate (30-54): Developing player",
        "  • Beginner (0-29): First pieces",
        "",
        "🎻 PRACTICE RECOMMENDATIONS:",
        "  • Personalized based on technical challenges",
        "  • Specific exercises for identified difficulties",
        "  • Etude suggestions for supplementary study",
        "  • Practice schedule tailored to difficulty level",
        "",
        "📈 VISUALIZATION FEATURES:",
        "  • Score cards with progress bars",
        "  • Color-coded difficulty levels",
        "  • Detailed breakdown of scoring factors",
        "  • Radar chart for multi-dimensional comparison",
        "",
        "💾 EXPORT OPTIONS:",
        "  • JSON format for data processing",
        "  • Text reports for sharing",
        "  • Copy to clipboard for quick notes",
        "  • Load analysis data from files",
        "",
        "🚀 QUICK START:",
        "  1. Load analysis data or use sample",
        "  2. Click 'Calculate Complexity'",
        "  3. Review scores and difficulty level",
        "  4. Study recommendations and etudes",
        "  5. Export results for future reference",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = ComplexityScoresGUI(parent)
    return gui.parent


# Standalone test
def test_standalone():
    """Test the module GUI standalone."""
    root = tk.Tk()
    root.title("Complexity Scorer Module - Standalone Test")
    root.geometry(WINDOW_SIZE)
    root.configure(bg=COLOR_BG)

    # Create the GUI
    gui = create_gui(root)

    root.mainloop()


if __name__ == "__main__":
    test_standalone()