#!/usr/bin/env python3
# ai_violin_master_04_fingering_generator.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683


import customtkinter as ctk
from tkinter import filedialog, messagebox
import json
import os
import numpy as np
import configparser

# Color constants
COLOR_BG = "#5A381E"
COLOR_DARK = "#3B2413"
COLOR_GOLD = "#FFD770"
COLOR_FRAME = "#4B2E18"
COLOR_MEDIUM = "#4B2E18"
COLOR_TEXT = "#FFF2CF"

WINDOW_SIZE = "640x1050"

class FingeringGenerator:
    """Automatic fingering generation for violin with position optimization."""

    STRING_BASE = {
        'G': 55,
        'D': 62,
        'A': 69,
        'E': 76
    }

    POSITIONS = {
        1: (0, 1, 2, 3),
        2: (2, 3, 4, 5),
        3: (4, 5, 6, 7),
        4: (5, 6, 7, 8),
        5: (7, 8, 9, 10),
        6: (9, 10, 11, 12),
        7: (11, 12, 13, 14)
    }

    def __init__(self):
        pass

    def valid_fingerings(self, pitch):
        fing_list = []
        for s, base in self.STRING_BASE.items():
            if pitch < base or pitch > base + 14:
                continue
            offset = pitch - base
            for pos, fingers in self.POSITIONS.items():
                if offset in fingers:
                    finger = fingers.index(offset) + 1
                    fing_list.append((s, pos, finger))
        return fing_list

    def cost_transition(self, prev, curr):
        if prev is None:
            return 0.0
        (s1, p1, f1) = prev
        (s2, p2, f2) = curr
        cost = 0.0
        if s1 != s2:
            cost += 1.5
        if p1 != p2:
            cost += 2.0
        if abs(p1 - p2) >= 2:
            cost += 3.0
        if abs(f1 - f2) >= 3:
            cost += 1.0
        if s1 == s2 and p1 == p2:
            cost -= 0.5
        return max(cost, 0.0)

    def generate_plan(self, notes):
        if not notes:
            return []
        candidates = [self.valid_fingerings(n.pitch) for n in notes]
        for c in candidates:
            if not c:
                c.append(('E', 7, 4))
        N = len(notes)
        dp = []
        back = []
        dp.append([0.0] * len(candidates[0]))
        back.append([-1] * len(candidates[0]))
        for i in range(1, N):
            dp.append([float("inf")] * len(candidates[i]))
            back.append([-1] * len(candidates[i]))
            for j, curr in enumerate(candidates[i]):
                for k, prev in enumerate(candidates[i-1]):
                    cost = dp[i-1][k] + self.cost_transition(prev, curr)
                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        back[i][j] = k
        plan = []
        last = np.argmin(dp[-1])
        difficulty = float(dp[-1][last])
        idx = last
        result = []
        for i in reversed(range(N)):
            fing = candidates[i][idx]
            result.append(fing)
            idx = back[i][idx]
        result.reverse()
        for note, fing in zip(notes, result):
            s, p, f = fing
            plan.append({
                "note": note.pitch,
                "note_name": note.pitch % 12,
                "string": s,
                "position": p,
                "finger": f,
                "shift": False
            })
        prev = None
        for item in plan:
            if prev is not None:
                if item["position"] != prev["position"]:
                    item["shift"] = True
            prev = item
        return plan, difficulty

    def generate_easy_plan(self, notes):
        plan = []
        prev_s, prev_p, prev_f = None, None, None
        for note in notes:
            fing_opts = self.valid_fingerings(note.pitch)
            if not fing_opts:
                plan.append({
                    "note": note.pitch,
                    "string": "E",
                    "position": 7,
                    "finger": 4,
                    "shift": False
                })
                continue
            scores = []
            for s, p, f in fing_opts:
                score = 0.0
                score += p * 1.0
                if f == 4:
                    score += 1.0
                if prev_s and s != prev_s:
                    score += 2.0
                scores.append(score)
            best = fing_opts[int(np.argmin(scores))]
            s, p, f = best
            plan.append({
                "note": note.pitch,
                "string": s,
                "position": p,
                "finger": f,
                "shift": prev_p is not None and p != prev_p
            })
            prev_s, prev_p, prev_f = s, p, f
        return plan

    def fingering_difficulty_score(self, plan):
        shifts = 0
        high_pos = 0
        stretch = 0
        prev = None
        for item in plan:
            if item["shift"]:
                shifts += 1
            if item["position"] >= 4:
                high_pos += 1
            if prev:
                if abs(item["finger"] - prev["finger"]) >= 3:
                    stretch += 1
            prev = item
        return {
            "shifts": shifts,
            "high_positions": high_pos,
            "stretch_intervals": stretch,
            "score": shifts * 2 + high_pos * 1.5 + stretch * 1.0
        }

    def ascii_fingering_timeline(self, plan, width=80):
        out = []
        for item in plan:
            tag = f"{item['string']}{item['position']}:{item['finger']}"
            if item["shift"]:
                tag = f"[{tag}]"
            else:
                tag = f" {tag} "
            out.append(tag)
        line = " | ".join(out)
        if len(line) > width:
            line = line[:width] + "..."
        return line


class FingeringGeneratorGUI:
    """GUI for Fingering Generator using CustomTkinter."""

    def __init__(self, parent=None):
        self.ini_file = "ai_violin_master_04_fingering_generator.ini"
        self.config = configparser.ConfigParser()

        if parent is None:
            self.root = ctk.CTk()
            self.is_standalone = True
            self.root.title("Fingering Generator")
            self.root.geometry(WINDOW_SIZE)
            self.root.resizable(False, False)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        else:
            self.root = parent
            self.is_standalone = False

        self.generator = FingeringGenerator()
        self.current_plan = None

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
        if not self.config.has_section("state"):
            self.config.add_section("state")

    def _save_state(self):
        self.config.set("window", "x", str(self.root.winfo_x()))
        self.config.set("window", "y", str(self.root.winfo_y()))
        self.config.set("state", "notes", self.notes_var.get())
        self.config.set("state", "algorithm", self.algo_var.get())
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
            text="FINGERING GENERATOR",
            font=("Georgia", 16, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            header_frame,
            text="Automatic violin fingering generation with position optimization",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=(0, 10))

        # Input section
        input_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        input_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            input_frame,
            text="Enter Notes (MIDI numbers or note names):",
            font=("Georgia", 12),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(anchor="w", padx=20, pady=(10, 5))

        self.notes_var = ctk.StringVar(value="60 62 64 65 67 69 71")
        ctk.CTkEntry(
            input_frame,
            textvariable=self.notes_var,
            font=("Georgia", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
        ).pack(fill="x", padx=20, pady=5)

        # Algorithm selection
        algo_row = ctk.CTkFrame(input_frame, fg_color=COLOR_DARK)
        algo_row.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            algo_row,
            text="Algorithm:",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(side="left")

        self.algo_var = ctk.StringVar(value="optimal")
        ctk.CTkRadioButton(
            algo_row,
            text="Optimal",
            variable=self.algo_var,
            value="optimal",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_GOLD,
        ).pack(side="left", padx=10)

        ctk.CTkRadioButton(
            algo_row,
            text="Easy",
            variable=self.algo_var,
            value="easy",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_GOLD,
        ).pack(side="left", padx=10)

        # Generate button
        self.generate_btn = ctk.CTkButton(
            main_frame,
            text="GENERATE FINGERING",
            font=("Georgia", 14, "bold"),
            fg_color=COLOR_DARK,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_MEDIUM,
            height=50,
            command=self._generate_fingering,
        )
        self.generate_btn.pack(pady=20)

        # Results display
        results_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        results_frame.pack(fill="both", expand=True, pady=10)

        ctk.CTkLabel(
            results_frame,
            text="FINGERING PLAN",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        self.results_text = ctk.CTkTextbox(
            results_frame,
            font=("Courier New", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
            wrap="none",
        )
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.results_text.insert("1.0", "Enter notes and click GENERATE FINGERING")
        self.results_text.configure(state="disabled")

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

    def _parse_notes(self, notes_str):
        class Note:
            def __init__(self, pitch):
                self.pitch = pitch

        notes = []
        for token in notes_str.split():
            if token.strip():
                pitch = int(token.strip())
                if 0 <= pitch <= 127:
                    notes.append(Note(pitch))
        return notes

    def _generate_fingering(self):
        notes_str = self.notes_var.get()
        if not notes_str.strip():
            messagebox.showwarning("No Input", "Please enter some notes to analyze.")
            return

        notes = self._parse_notes(notes_str)
        if not notes:
            messagebox.showerror("Parse Error", "Could not parse notes.")
            return

        self.status_label.configure(text="Generating fingering...")
        self.root.update()

        if self.algo_var.get() == "optimal":
            plan, difficulty = self.generator.generate_plan(notes)
        else:
            plan = self.generator.generate_easy_plan(notes)
            difficulty = self.generator.fingering_difficulty_score(plan)["score"]

        self.current_plan = plan
        diff_score = self.generator.fingering_difficulty_score(plan)

        # Display results
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        output = "Note | MIDI | String | Position | Finger | Shift\n"
        output += "-" * 60 + "\n"

        for item in plan:
            note_name = note_names[item["note_name"]] + str(item["note"] // 12 - 1)
            shift_mark = "X" if item["shift"] else ""
            output += f"{note_name:4s} | {item['note']:4d} | {item['string']:6s} | {item['position']:8d} | {item['finger']:6d} | {shift_mark:5s}\n"

        output += "\n" + "=" * 50 + "\n"
        output += f"Total shifts: {diff_score['shifts']}\n"
        output += f"High positions: {diff_score['high_positions']}\n"
        output += f"Stretch intervals: {diff_score['stretch_intervals']}\n"
        output += f"Difficulty score: {diff_score['score']:.2f}\n"

        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", output)
        self.results_text.configure(state="disabled")

        self.status_label.configure(text="Fingering generated successfully")

    def _on_close(self):
        self._save_state()
        self.root.destroy()


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = FingeringGeneratorGUI(parent)
    return gui.root


def main():
    """Standalone entry - launches a full window with mainloop."""
    FingeringGeneratorGUI()


if __name__ == "__main__":
    main()