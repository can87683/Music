#!/usr/bin/env python3
# ai_violin_master_02_bow_direction.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683


import customtkinter as ctk
from tkinter import filedialog, messagebox
import json
import os
from datetime import datetime
import numpy as np
import random
import configparser

# Color constants
COLOR_BG = "#5A381E"
COLOR_DARK = "#3B2413"
COLOR_GOLD = "#FFD770"
COLOR_FRAME = "#4B2E18"
COLOR_MEDIUM = "#4B2E18"
COLOR_TEXT = "#FFF2CF"

WINDOW_SIZE = "640x1050"

class BowDirection:
    """Core rule-based bowing logic for violin."""

    def __init__(self, style="Modern"):
        self.style = style

    def style_rules(self):
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
        return {
            "downbeat_weight": 1.5,
            "accent_weight": 2.0,
            "staccato_upbow_bias": 1.5,
            "pickup_upbow": True,
            "slur_reset": True
        }

    def detect_articulations(self, note, next_note):
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
        N = len(plan)
        for i in range(N):
            if i < N - 1:
                slur, staccato, accent = self.detect_articulations(
                    plan[i]["_note"], plan[i+1]["_note"]
                )
            else:
                slur, staccato, accent = self.detect_articulations(plan[i]["_note"], None)
            plan[i]["slur"] = slur
            plan[i]["staccato"] = staccato
            plan[i]["accent"] = accent
        return plan

    def apply_direction_rules(self, plan, metadata):
        style = self.style_rules()
        ts = metadata["time_signature"]
        numerator = int(ts.split("/")[0])
        bpm = metadata["tempo"]
        beat_len = 60 / bpm

        for item in plan:
            beat_pos = item["start"] % (beat_len * numerator)
            is_downbeat = abs(beat_pos) < 0.0001
            if is_downbeat and style["downbeat_weight"] > 0:
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
        return {"balance_score": difficulty, "trajectory": history}

    def ascii_bow_timeline(self, plan, width=80):
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
    """GUI for Bow Direction module using CustomTkinter."""

    def __init__(self, parent=None):
        self.ini_file = "ai_violin_master_02_bow_direction.ini"
        self.config = configparser.ConfigParser()

        if parent is None:
            self.root = ctk.CTk()
            self.is_standalone = True
            self.root.title("Bow Direction Analyzer")
            self.root.geometry(WINDOW_SIZE)
            self.root.resizable(False, False)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        else:
            self.root = parent
            self.is_standalone = False

        self.bow_analyzer = BowDirection()
        self.sample_data = self._create_sample_data()
        self.bow_plan = []
        self.bow_metrics = {}

        self._load_state()
        ctk.set_appearance_mode("dark")

        if self.is_standalone:
            self.root.configure(fg_color=COLOR_BG)

        self._create_widgets()

        if self.is_standalone:
            self._restore_position()
            self.root.mainloop()
            self._restore_position()
            self.root.mainloop()

    def _load_state(self):
        if os.path.exists(self.ini_file):
            self.config.read(self.ini_file)
        if not self.config.has_section("window"):
            self.config.add_section("window")
        if not self.config.has_section("state"):
            self.config.add_section("state")

    def _save_state(self):
        self.config.set("window", "x", str(self.root.winfo_x()))
        self.config.set("window", "y", str(self.root.winfo_y()))
        self.config.set("state", "style", self.style_var.get())
        with open(self.ini_file, "w") as f:
            self.config.write(f)

    def _restore_position(self):
        x = self.config.getint("window", "x", fallback=100)
        y = self.config.getint("window", "y", fallback=100)
        self.root.geometry(f"+{x}+{y}")

    def _create_sample_data(self):
        class MockNote:
            def __init__(self, start, pitch, velocity=80, duration=0.5):
                self.start = start
                self.pitch = pitch
                self.velocity = velocity
                self.end = start + duration
                self.duration = duration

        notes = []
        for i in range(16):
            vel = 80
            if i % 4 == 0:
                vel = 95
            if i == 2 or i == 10:
                duration = 0.2
            else:
                duration = 0.5
            notes.append(MockNote(
                start=i * 0.5,
                pitch=60 + (i % 8),
                velocity=vel,
                duration=duration
            ))

        notes[3].end = notes[4].start + 0.01
        notes[11].end = notes[12].start + 0.01

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
        # Main container
        main_frame = ctk.CTkFrame(self.root, fg_color=COLOR_BG)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        header_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            header_frame,
            text="BOW DIRECTION ANALYZER",
            font=("Georgia", 16, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            header_frame,
            text="Rule-based bowing analysis and direction planning",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=(0, 10))

        # Style configuration
        style_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        style_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            style_frame,
            text="Style Configuration",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        style_row = ctk.CTkFrame(style_frame, fg_color=COLOR_DARK)
        style_row.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            style_row,
            text="Musical Style:",
            font=("Georgia", 12),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(side="left")

        self.style_var = ctk.StringVar(value="Modern")
        style_menu = ctk.CTkOptionMenu(
            style_row,
            variable=self.style_var,
            values=["Modern", "Classical", "Baroque"],
            font=("Georgia", 12),
            fg_color=COLOR_DARK,
            text_color=COLOR_TEXT,
            button_color=COLOR_MEDIUM,
            button_hover_color=COLOR_FRAME,
            dropdown_fg_color=COLOR_DARK,
            dropdown_text_color=COLOR_TEXT,
            dropdown_hover_color=COLOR_FRAME,
            width=200,
        )
        style_menu.pack(side="left", padx=10)

        # Generate button
        self.generate_btn = ctk.CTkButton(
            main_frame,
            text="GENERATE BOWING",
            font=("Georgia", 14, "bold"),
            fg_color=COLOR_DARK,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_MEDIUM,
            height=50,
            command=self._generate_bowing,
        )
        self.generate_btn.pack(pady=20)

        # Timeline display
        timeline_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        timeline_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            timeline_frame,
            text="BOWING TIMELINE",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        self.timeline_text = ctk.CTkTextbox(
            timeline_frame,
            font=("Courier New", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
            height=100,
            wrap="none",
        )
        self.timeline_text.pack(fill="x", padx=10, pady=10)
        self.timeline_text.insert("1.0", "Click GENERATE BOWING to create timeline")
        self.timeline_text.configure(state="disabled")

        # Statistics
        stats_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        stats_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            stats_frame,
            text="STATISTICS",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        self.stats_text = ctk.CTkTextbox(
            stats_frame,
            font=("Courier New", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
            height=150,
            wrap="word",
        )
        self.stats_text.pack(fill="x", padx=10, pady=10)
        self.stats_text.insert("1.0", "Statistics will appear here after generation")
        self.stats_text.configure(state="disabled")

        # Status bar
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Ready to analyze bowing",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            height=30,
        )
        self.status_label.pack(fill="x", pady=(10, 0))

    def _generate_bowing(self):
        self.status_label.configure(text="Generating bowing plan...")
        self.root.update()

        self.bow_analyzer.style = self.style_var.get()
        self.bow_plan, self.bow_metrics = self.bow_analyzer.generate_bowing(
            self.sample_data["notes"],
            self.sample_data["metadata"]
        )

        # Update timeline
        timeline = self.bow_analyzer.ascii_bow_timeline(self.bow_plan, width=60)
        self.timeline_text.configure(state="normal")
        self.timeline_text.delete("1.0", "end")
        self.timeline_text.insert("1.0", timeline)
        self.timeline_text.configure(state="disabled")

        # Update statistics
        total = len(self.bow_plan)
        downbows = sum(1 for item in self.bow_plan if item["direction"] == "D")
        upbows = total - downbows
        slurs = sum(1 for item in self.bow_plan if item["slur"])
        staccato = sum(1 for item in self.bow_plan if item["staccato"])
        accents = sum(1 for item in self.bow_plan if item["accent"])
        balance_score = self.bow_metrics.get("balance_score", 0)

        stats = (
            f"Total Notes: {total}\n"
            f"Down-bows: {downbows}\n"
            f"Up-bows: {upbows}\n"
            f"Slurs: {slurs}\n"
            f"Staccato: {staccato}\n"
            f"Accents: {accents}\n"
            f"Balance Score: {balance_score:.2f}\n"
            f"Style: {self.style_var.get()}\n"
        )

        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", stats)
        self.stats_text.configure(state="disabled")

        self.status_label.configure(text=f"Bowing generated ({self.style_var.get()} style)")

    def _on_close(self):
        self._save_state()
        self.root.destroy()


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = BowDirectionGUI(parent)
    return gui.root


def main():
    """Standalone entry - launches a full window with mainloop."""
    BowDirectionGUI()


if __name__ == "__main__":
    main()