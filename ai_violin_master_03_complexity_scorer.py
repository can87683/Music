#!/usr/bin/env python3
# ai_violin_master_03_complexity_scorer.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683


import customtkinter as ctk
from tkinter import filedialog, messagebox
import json
import os
from datetime import datetime
import configparser

# Color constants
COLOR_BG = "#5A381E"
COLOR_DARK = "#3B2413"
COLOR_GOLD = "#FFD770"
COLOR_FRAME = "#4B2E18"
COLOR_MEDIUM = "#4B2E18"
COLOR_TEXT = "#FFF2CF"

WINDOW_SIZE = "640x1050"

class ComplexityScorer:
    """Quantify musical and technical difficulty."""

    def _rhythmic_complexity(self, metadata: dict) -> float:
        tempo_var = metadata.get("tempo_variation", 0.0)
        nps = metadata.get("notes_per_second", 0.0)
        tempo_score = min(tempo_var * 10, 40)
        density_score = min(nps * 6, 40)
        return tempo_score + density_score

    def _technical_complexity(self, violin_data: dict) -> float:
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

    def _range_difficulty(self, metadata: dict) -> float:
        (low, high) = metadata["pitch_range_midi"]
        width = high - low
        return min((width / 41) * 100, 100)

    def _articulation_complexity(self, violin_data: dict) -> float:
        bow = violin_data["bowing"]
        slur_score = min(bow.slur_groups * 3.5, 40)
        var_score = min(bow.articulation_consistency * 25, 60)
        return slur_score + var_score

    def _difficulty_level(self, score: float) -> str:
        if score >= 85:
            return "Advanced"
        if score >= 70:
            return "Upper Intermediate"
        if score >= 55:
            return "Intermediate"
        if score >= 30:
            return "Early Intermediate"
        return "Beginner"

    def score_complexity(self, metadata: dict, violin_data: dict) -> dict:
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

    def generate_recommendations(self, violin_data: dict, complexity: dict) -> list:
        rec = []
        pos = violin_data["positions"]
        iv = violin_data["intervals"]
        bow = violin_data["bowing"]
        tstats = violin_data["technique_stats"]
        dstop = violin_data["double_stops"]

        if pos["shifts"] > 12:
            rec.append("Practice slow glissando shifting drills (1-3, 2-4 positions).")
        if pos["large_shifts"] > 5:
            rec.append("Add Sevcik Op.8 shifting exercises for large leaps.")
        if iv.large_intervals > 6:
            rec.append("Practice octave and tenth leaps slowly; isolate wide shifts.")
        if iv.max_interval > 12:
            rec.append("Decompose extreme leaps (>12 semitones) into guide-tone steps.")
        if tstats["fast_passages"] > 12:
            rec.append("Use rhythmic alteration practice (long-short, short-long) on fast runs.")
        if dstop > 0:
            rec.append("Tune double stops slowly; practice with perfect fifth drones.")
        if bow.slur_groups > 6:
            rec.append("Practice smooth bow changes; slow 'son file' exercises recommended.")
        if bow.articulation_consistency > 0.15:
            rec.append("Stabilize articulation with slow detache on open strings.")
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

    def _suggest_etudes(self, violin_data: dict, complexity: dict) -> list:
        etudes = []
        if violin_data["positions"]["large_shifts"] > 5:
            etudes.append("Kreutzer #11 - shifting control")
        if violin_data["intervals"].large_intervals > 6:
            etudes.append("Dont Op.35 #6 - large leaps & position stability")
        if violin_data["double_stops"] > 4:
            etudes.append("Kreutzer #33 - double-stop intonation")
        if violin_data["bowing"].bowing_complexity == "High":
            etudes.append("Kreutzer #2 - bow control & detache consistency")
        if complexity.get("level") == "Advanced":
            etudes.append("Paganini Caprice #1 - arpeggio clarity & string crossing")
        return etudes


class ComplexityScoresGUI:
    """GUI for Complexity Scorer module using CustomTkinter."""

    def __init__(self, parent=None):
        self.ini_file = "ai_violin_master_03_complexity_scorer.ini"
        self.config = configparser.ConfigParser()

        if parent is None:
            self.root = ctk.CTk()
            self.is_standalone = True
            self.root.title("Complexity Scorer")
            self.root.geometry(WINDOW_SIZE)
            self.root.resizable(False, False)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        else:
            self.root = parent
            self.is_standalone = False

        self.scorer = ComplexityScorer()
        self.sample_data = self._create_sample_data()
        self.complexity_scores = None
        self.recommendations = []
        self.etudes = []

        self._load_state()
        ctk.set_appearance_mode("dark")

        if self.is_standalone:
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
        # Main container
        main_frame = ctk.CTkFrame(self.root, fg_color=COLOR_BG)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        header_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            header_frame,
            text="VIOLIN COMPLEXITY SCORER",
            font=("Georgia", 16, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            header_frame,
            text="Quantify musical and technical difficulty of violin pieces",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=(0, 10))

        # Calculate button
        self.calc_btn = ctk.CTkButton(
            main_frame,
            text="CALCULATE COMPLEXITY",
            font=("Georgia", 14, "bold"),
            fg_color=COLOR_DARK,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_MEDIUM,
            height=50,
            command=self._calculate_complexity,
        )
        self.calc_btn.pack(pady=20)

        # Score display
        score_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        score_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            score_frame,
            text="COMPLEXITY SCORES",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        self.score_text = ctk.CTkTextbox(
            score_frame,
            font=("Courier New", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
            height=200,
            wrap="word",
        )
        self.score_text.pack(fill="x", padx=10, pady=10)
        self.score_text.insert("1.0", "Click CALCULATE COMPLEXITY to analyze")
        self.score_text.configure(state="disabled")

        # Recommendations display
        rec_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        rec_frame.pack(fill="both", expand=True, pady=10)

        ctk.CTkLabel(
            rec_frame,
            text="PRACTICE RECOMMENDATIONS",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        self.rec_text = ctk.CTkTextbox(
            rec_frame,
            font=("Courier New", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
            wrap="word",
        )
        self.rec_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.rec_text.insert("1.0", "Recommendations will appear here")
        self.rec_text.configure(state="disabled")

        # Status bar
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Ready to analyze complexity",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            height=30,
        )
        self.status_label.pack(fill="x", pady=(10, 0))

    def _calculate_complexity(self):
        self.status_label.configure(text="Calculating complexity scores...")
        self.root.update()

        self.complexity_scores = self.scorer.score_complexity(
            self.sample_data["metadata"],
            self.sample_data["violin_data"]
        )

        self.recommendations = self.scorer.generate_recommendations(
            self.sample_data["violin_data"],
            self.complexity_scores
        )

        self.etudes = self.scorer._suggest_etudes(
            self.sample_data["violin_data"],
            self.complexity_scores
        )

        # Update score display
        scores = (
            f"RHYTHMIC COMPLEXITY: {self.complexity_scores['rhythmic']:.1f}/100\n"
            f"TECHNICAL COMPLEXITY: {self.complexity_scores['technical']:.1f}/100\n"
            f"RANGE DIFFICULTY: {self.complexity_scores['range_complexity']:.1f}/100\n"
            f"ARTICULATION COMPLEXITY: {self.complexity_scores['articulation']:.1f}/100\n\n"
            f"OVERALL SCORE: {self.complexity_scores['total']:.1f}/100\n"
            f"DIFFICULTY LEVEL: {self.complexity_scores['level']}\n"
        )

        self.score_text.configure(state="normal")
        self.score_text.delete("1.0", "end")
        self.score_text.insert("1.0", scores)
        self.score_text.configure(state="disabled")

        # Update recommendations
        rec_text = "PRACTICE RECOMMENDATIONS:\n\n"
        for i, rec in enumerate(self.recommendations, 1):
            rec_text += f"{i}. {rec}\n\n"

        if self.etudes:
            rec_text += "\nSUGGESTED ETUDES:\n\n"
            for etude in self.etudes:
                rec_text += f"- {etude}\n"

        self.rec_text.configure(state="normal")
        self.rec_text.delete("1.0", "end")
        self.rec_text.insert("1.0", rec_text)
        self.rec_text.configure(state="disabled")

        self.status_label.configure(text=f"Complexity calculated: {self.complexity_scores['level']}")

    def _on_close(self):
        self._save_state()
        self.root.destroy()


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = ComplexityScoresGUI(parent)
    return gui.root


def main():
    """Standalone entry - launches a full window with mainloop."""
    ComplexityScoresGUI()


if __name__ == "__main__":
    main()