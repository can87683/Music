#!/usr/bin/env python3
# ai_violin_master_04_fingering_generator.py

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import json
import os
from pathlib import Path

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

class FingeringGenerator:
    """Automatic fingering generation for violin with position optimization."""

    # Open string MIDI reference
    STRING_BASE = {
        'G': 55,   # G3
        'D': 62,   # D4
        'A': 69,   # A4
        'E': 76    # E5
    }

    # Positions define finger offsets from open string
    POSITIONS = {
        1: (0, 1, 2, 3),      # 1st position
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
        """Returns a list of possible fingerings: (string, position, finger)."""
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
        """Cost model for fingering transitions."""
        if prev is None:
            return 0.0

        (s1, p1, f1) = prev
        (s2, p2, f2) = curr

        cost = 0.0

        # String crossing
        if s1 != s2:
            cost += 1.5

        # Position change
        if p1 != p2:
            cost += 2.0

        # Large shifts
        if abs(p1 - p2) >= 2:
            cost += 3.0

        # Finger stretch
        if abs(f1 - f2) >= 3:
            cost += 1.0

        # Slight bias for staying consistent
        if s1 == s2 and p1 == p2:
            cost -= 0.5

        return max(cost, 0.0)

    def generate_plan(self, notes):
        """Dynamic programming solver for optimal global fingering."""
        if not notes:
            return []

        # Precompute candidate fingerings per note
        candidates = [self.valid_fingerings(n.pitch) for n in notes]

        # If a note is out of range
        for c in candidates:
            if not c:
                c.append(('E', 7, 4))

        N = len(notes)

        # dp[i][j] = cost up to note i choosing j-th fingering option
        dp = []
        back = []

        # Initialize
        dp.append([0.0] * len(candidates[0]))
        back.append([-1] * len(candidates[0]))

        # Build DP
        for i in range(1, N):
            dp.append([float("inf")] * len(candidates[i]))
            back.append([-1] * len(candidates[i]))

            for j, curr in enumerate(candidates[i]):
                for k, prev in enumerate(candidates[i-1]):
                    cost = dp[i-1][k] + self.cost_transition(prev, curr)

                    if cost < dp[i][j]:
                        dp[i][j] = cost
                        back[i][j] = k

        # Backtrack minimal cost solution
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

        # Convert to final structured entries
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

        # Mark shifts
        prev = None
        for item in plan:
            if prev is not None:
                if item["position"] != prev["position"]:
                    item["shift"] = True
            prev = item

        return plan, difficulty

    def generate_easy_plan(self, notes):
        """Simpler heuristic for easier fingering."""
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

            # Rank options
            scores = []
            for s, p, f in fing_opts:
                score = 0.0
                score += p * 1.0  # Prefer low positions
                if f == 4:
                    score += 1.0  # Avoid 4th finger
                if prev_s and s != prev_s:
                    score += 2.0  # Prefer staying on same string
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
        """Calculate fingering complexity metrics."""
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
        """Produce ASCII fingering timeline."""
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

    def export_to_json(self, plan, filename):
        """Export fingering plan to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(plan, f, indent=2)
            return True
        except Exception as e:
            return False, str(e)

class FingeringGeneratorGUI:
    """GUI for Fingering Generator matching the color scheme and size."""

    def __init__(self, parent):
        self.parent = parent
        self.generator = FingeringGenerator()
        self.current_plan = None

        # Create main frame
        self.main_frame = tk.Frame(parent, bg=COLOR_BG)
        self.main_frame.pack(fill="both", expand=True)

        self.create_widgets()

    def create_widgets(self):
        """Create all GUI widgets."""
        # Header
        header_frame = tk.Frame(self.main_frame, bg=COLOR_BG)
        header_frame.pack(fill="x", pady=(0, 10))

        tk.Label(header_frame,
                text="🎻 Fingering Generator",
                font=FONT_HEADER,
                fg=COLOR_GOLD,
                bg=COLOR_BG).pack()

        tk.Label(header_frame,
                text="Automatic violin fingering generation with position optimization",
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_BG).pack()

        # Separator
        sep = tk.Frame(self.main_frame, height=2, bg=COLOR_GOLD)
        sep.pack(fill="x", padx=20, pady=5)

        # Create notebook for tabs
        notebook = ttk.Notebook(self.main_frame)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

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

        # Tab 1: Input Notes
        input_frame = tk.Frame(notebook, bg=COLOR_BG)
        notebook.add(input_frame, text="Input Notes")

        self.create_input_tab(input_frame)

        # Tab 2: Fingering Results
        results_frame = tk.Frame(notebook, bg=COLOR_BG)
        notebook.add(results_frame, text="Fingering Results")

        self.create_results_tab(results_frame)

        # Tab 3: Visualization
        viz_frame = tk.Frame(notebook, bg=COLOR_BG)
        notebook.add(viz_frame, text="Visualization")

        self.create_visualization_tab(viz_frame)

        # Action buttons at bottom
        self.create_action_buttons()

    def create_input_tab(self, parent_frame):
        """Create the input notes tab."""
        # Notes input frame
        input_frame = tk.LabelFrame(parent_frame,
                                  text="Enter Notes (MIDI numbers or note names)",
                                  font=FONT_SECTION,
                                  fg=COLOR_GOLD,
                                  bg=COLOR_BG,
                                  relief=tk.RIDGE,
                                  borderwidth=2)
        input_frame.pack(fill="both", expand=True, padx=20, pady=10, ipadx=10, ipady=10)

        # Example notes
        example_label = tk.Label(input_frame,
                               text="Examples: 60 62 64 65 67 (or) C4 D4 E4 F4 G4",
                               font=FONT_SMALL,
                               fg=COLOR_TEXT,
                               bg=COLOR_BG)
        example_label.pack(anchor="w", padx=10, pady=(0, 10))

        # Notes entry
        notes_frame = tk.Frame(input_frame, bg=COLOR_BG)
        notes_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(notes_frame,
                text="Notes:",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=10,
                anchor="w").pack(side="left")

        self.notes_var = tk.StringVar()
        self.notes_var.set("60 62 64 65 67 69 71")

        notes_entry = tk.Entry(notes_frame,
                             textvariable=self.notes_var,
                             font=FONT_TEXT,
                             bg=COLOR_DARK,
                             fg=COLOR_TEXT,
                             insertbackground=COLOR_TEXT,
                             relief=tk.SUNKEN,
                             borderwidth=1)
        notes_entry.pack(side="left", fill="x", expand=True, padx=5)

        # Preset scales
        preset_frame = tk.LabelFrame(input_frame,
                                   text="Preset Scales",
                                   font=FONT_SMALL,
                                   fg=COLOR_GOLD,
                                   bg=COLOR_BG,
                                   relief=tk.GROOVE,
                                   borderwidth=1)
        preset_frame.pack(fill="x", padx=10, pady=10)

        presets = [
            ("C Major Scale", "60 62 64 65 67 69 71 72"),
            ("G Major Scale", "55 57 59 60 62 64 66 67"),
            ("D Major Scale", "62 64 66 67 69 71 73 74"),
            ("A Minor Scale", "57 59 60 62 64 65 67 69"),
            ("Chromatic C4-C5", "60 61 62 63 64 65 66 67 68 69 70 71 72")
        ]

        for preset_name, preset_notes in presets:
            btn = tk.Button(preset_frame,
                          text=preset_name,
                          font=FONT_SMALL,
                          fg=COLOR_GOLD,
                          bg=COLOR_DARK,
                          command=lambda n=preset_notes: self.load_preset(n))
            btn.pack(side="left", padx=5, pady=5)

        # Algorithm selection
        algo_frame = tk.Frame(input_frame, bg=COLOR_BG)
        algo_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(algo_frame,
                text="Algorithm:",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=10,
                anchor="w").pack(side="left")

        self.algo_var = tk.StringVar(value="optimal")

        tk.Radiobutton(algo_frame,
                      text="Optimal (Dynamic Programming)",
                      variable=self.algo_var,
                      value="optimal",
                      font=FONT_SMALL,
                      fg=COLOR_TEXT,
                      bg=COLOR_BG,
                      selectcolor=COLOR_DARK).pack(side="left", padx=10)

        tk.Radiobutton(algo_frame,
                      text="Easy (Heuristic)",
                      variable=self.algo_var,
                      value="easy",
                      font=FONT_SMALL,
                      fg=COLOR_TEXT,
                      bg=COLOR_BG,
                      selectcolor=COLOR_DARK).pack(side="left", padx=10)

        # Information panel
        info_text = """💡 How to use:
1. Enter notes as MIDI numbers (60 = C4) or note names
2. Choose algorithm:
   • Optimal: Best overall fingering
   • Easy: Simpler, beginner-friendly
3. Click 'Generate Fingering' to calculate"""

        info_label = tk.Label(input_frame,
                            text=info_text,
                            font=FONT_SMALL,
                            fg=COLOR_TEXT,
                            bg=COLOR_BG,
                            justify=tk.LEFT)
        info_label.pack(fill="x", padx=10, pady=10)

    def create_results_tab(self, parent_frame):
        """Create the fingering results tab."""
        # Results display frame
        results_display = tk.LabelFrame(parent_frame,
                                      text="Fingering Plan",
                                      font=FONT_SECTION,
                                      fg=COLOR_GOLD,
                                      bg=COLOR_BG,
                                      relief=tk.RIDGE,
                                      borderwidth=2)
        results_display.pack(fill="both", expand=True, padx=20, pady=10, ipadx=10, ipady=10)

        # Text widget for results
        self.results_text = scrolledtext.ScrolledText(results_display,
                                                    font=("Courier", 10),
                                                    bg=COLOR_DARK,
                                                    fg=COLOR_TEXT,
                                                    insertbackground=COLOR_TEXT,
                                                    relief=tk.SUNKEN,
                                                    borderwidth=1,
                                                    height=15)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Difficulty score frame
        diff_frame = tk.Frame(results_display, bg=COLOR_BG)
        diff_frame.pack(fill="x", padx=10, pady=10)

        self.diff_var = tk.StringVar(value="Difficulty score: --")
        diff_label = tk.Label(diff_frame,
                            textvariable=self.diff_var,
                            font=FONT_TEXT,
                            fg=COLOR_GOLD,
                            bg=COLOR_BG)
        diff_label.pack()

        # ASCII timeline
        timeline_frame = tk.Frame(results_display, bg=COLOR_BG)
        timeline_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(timeline_frame,
                text="ASCII Timeline:",
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_BG).pack(anchor="w")

        self.timeline_var = tk.StringVar(value="")
        timeline_label = tk.Label(timeline_frame,
                                textvariable=self.timeline_var,
                                font=("Courier", 9),
                                fg=COLOR_GOLD,
                                bg=COLOR_DARK,
                                relief=tk.SUNKEN,
                                borderwidth=1,
                                anchor="w",
                                justify=tk.LEFT)
        timeline_label.pack(fill="x", pady=5)

    def create_visualization_tab(self, parent_frame):
        """Create the visualization tab."""
        viz_frame = tk.LabelFrame(parent_frame,
                                text="Fingering Visualization",
                                font=FONT_SECTION,
                                fg=COLOR_GOLD,
                                bg=COLOR_BG,
                                relief=tk.RIDGE,
                                borderwidth=2)
        viz_frame.pack(fill="both", expand=True, padx=20, pady=10, ipadx=10, ipady=10)

        # Canvas for visualization
        self.canvas = tk.Canvas(viz_frame,
                              bg=COLOR_DARK,
                              highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # Visualization controls
        controls_frame = tk.Frame(viz_frame, bg=COLOR_BG)
        controls_frame.pack(fill="x", padx=10, pady=10)

        tk.Label(controls_frame,
                text="String colors:",
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_BG).pack(side="left", padx=5)

        self.string_colors = {
            'G': '#8B4513',  # Brown
            'D': '#D2691E',  # Chocolate
            'A': '#CD853F',  # Peru
            'E': '#F4A460'   # Sandy brown
        }

        for string, color in self.string_colors.items():
            color_label = tk.Label(controls_frame,
                                 text=f" {string} ",
                                 font=FONT_SMALL,
                                 bg=color,
                                 fg='white' if color != '#F4A460' else 'black')
            color_label.pack(side="left", padx=2)

        info_text = """🎨 Visualization Legend:
• Circles represent notes
• Color indicates string (G, D, A, E)
• Numbers inside are finger numbers (1-4)
• Brackets [ ] indicate position shifts
• Y-axis: Position (1st to 7th)
• X-axis: Note sequence"""

        info_label = tk.Label(viz_frame,
                            text=info_text,
                            font=FONT_SMALL,
                            fg=COLOR_TEXT,
                            bg=COLOR_BG,
                            justify=tk.LEFT)
        info_label.pack(fill="x", padx=10, pady=10)

    def create_action_buttons(self):
        """Create action buttons at bottom."""
        actions_frame = tk.Frame(self.main_frame, bg=COLOR_BG)
        actions_frame.pack(fill="x", padx=10, pady=10)

        # Generate button
        generate_btn = tk.Button(actions_frame,
                               text="🎻 Generate Fingering",
                               font=FONT_TEXT,
                               fg=COLOR_GOLD,
                               bg=COLOR_DARK,
                               activeforeground=COLOR_GOLD,
                               activebackground=COLOR_FRAME,
                               relief=tk.RAISED,
                               borderwidth=2,
                               padx=20,
                               pady=5,
                               command=self.generate_fingering)
        generate_btn.pack(side="left", padx=5)

        # Export button
        export_btn = tk.Button(actions_frame,
                             text="💾 Export to JSON",
                             font=FONT_TEXT,
                             fg=COLOR_GOLD,
                             bg=COLOR_DARK,
                             activeforeground=COLOR_GOLD,
                             activebackground=COLOR_FRAME,
                             relief=tk.RAISED,
                             borderwidth=2,
                             padx=20,
                             pady=5,
                             command=self.export_fingering)
        export_btn.pack(side="left", padx=5)

        # Clear button
        clear_btn = tk.Button(actions_frame,
                            text="🗑️ Clear Results",
                            font=FONT_TEXT,
                            fg=COLOR_GOLD,
                            bg=COLOR_DARK,
                            activeforeground=COLOR_GOLD,
                            activebackground=COLOR_FRAME,
                            relief=tk.RAISED,
                            borderwidth=2,
                            padx=20,
                            pady=5,
                            command=self.clear_results)
        clear_btn.pack(side="left", padx=5)

        # Status label
        self.status_var = tk.StringVar(value="✅ Ready")
        status_label = tk.Label(actions_frame,
                              textvariable=self.status_var,
                              font=FONT_SMALL,
                              fg=COLOR_TEXT,
                              bg=COLOR_BG)
        status_label.pack(side="right", padx=5)

    def load_preset(self, notes):
        """Load a preset scale."""
        self.notes_var.set(notes)
        self.status_var.set("✅ Preset loaded")

    def parse_notes(self, notes_str):
        """Parse notes string into list of Note objects."""
        notes = []

        # Simple Note class for compatibility
        class Note:
            def __init__(self, pitch):
                self.pitch = pitch

        # Try to parse as MIDI numbers
        try:
            for token in notes_str.split():
                if token.strip():
                    pitch = int(token.strip())
                    if 0 <= pitch <= 127:
                        notes.append(Note(pitch))
        except ValueError:
            # Try to parse as note names
            note_map = {
                'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
                'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
            }

            for token in notes_str.split():
                token = token.strip().upper()
                if not token:
                    continue

                # Parse note name and octave
                note_name = ''
                octave = ''
                for char in token:
                    if char.isdigit():
                        octave += char
                    else:
                        note_name += char

                if note_name in note_map and octave:
                    pitch = note_map[note_name] + (int(octave) + 1) * 12
                    notes.append(Note(pitch))

        return notes

    def generate_fingering(self):
        """Generate fingering for the entered notes."""
        notes_str = self.notes_var.get()

        if not notes_str.strip():
            messagebox.showwarning("No Input", "Please enter some notes to analyze.")
            return

        # Parse notes
        notes = self.parse_notes(notes_str)

        if not notes:
            messagebox.showerror("Parse Error",
                               "Could not parse notes. Use MIDI numbers (60 62 64) or note names (C4 D4 E4).")
            return

        self.status_var.set("⏳ Generating fingering...")
        self.parent.update()

        try:
            # Generate fingering plan
            if self.algo_var.get() == "optimal":
                plan, difficulty = self.generator.generate_plan(notes)
            else:
                plan = self.generator.generate_easy_plan(notes)
                difficulty = self.generator.fingering_difficulty_score(plan)["score"]

            self.current_plan = plan

            # Calculate difficulty
            diff_score = self.generator.fingering_difficulty_score(plan)

            # Display results
            self.display_results(plan, diff_score, difficulty)

            # Generate ASCII timeline
            timeline = self.generator.ascii_fingering_timeline(plan)
            self.timeline_var.set(timeline)

            # Update visualization
            self.update_visualization(plan)

            self.status_var.set("✅ Fingering generated successfully")

        except Exception as e:
            messagebox.showerror("Generation Error", f"Error generating fingering: {str(e)}")
            self.status_var.set("❌ Error")

    def display_results(self, plan, diff_score, difficulty):
        """Display fingering results in text widget."""
        self.results_text.delete(1.0, tk.END)

        # Header
        self.results_text.insert(tk.END, "FINGERING PLAN\n")
        self.results_text.insert(tk.END, "=" * 50 + "\n\n")

        # Notes and fingerings
        self.results_text.insert(tk.END, "Note | MIDI | String | Position | Finger | Shift\n")
        self.results_text.insert(tk.END, "-" * 60 + "\n")

        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

        for item in plan:
            note_name = note_names[item["note_name"]] + str(item["note"] // 12 - 1)
            shift_mark = "✓" if item["shift"] else ""

            line = f"{note_name:4s} | {item['note']:4d} | {item['string']:6s} | {item['position']:8d} | {item['finger']:6d} | {shift_mark:5s}\n"
            self.results_text.insert(tk.END, line)

        self.results_text.insert(tk.END, "\n" + "=" * 50 + "\n\n")

        # Difficulty score
        self.results_text.insert(tk.END, "DIFFICULTY ANALYSIS\n")
        self.results_text.insert(tk.END, "-" * 30 + "\n")
        self.results_text.insert(tk.END, f"Total shifts: {diff_score['shifts']}\n")
        self.results_text.insert(tk.END, f"High positions (≥4th): {diff_score['high_positions']}\n")
        self.results_text.insert(tk.END, f"Stretch intervals: {diff_score['stretch_intervals']}\n")
        self.results_text.insert(tk.END, f"Difficulty score: {diff_score['score']:.2f}\n")

        if self.algo_var.get() == "optimal":
            self.results_text.insert(tk.END, f"DP optimal cost: {difficulty:.2f}\n")

        # Interpretation
        self.results_text.insert(tk.END, "\nDIFFICULTY INTERPRETATION\n")
        self.results_text.insert(tk.END, "-" * 30 + "\n")

        if diff_score['score'] < 5:
            self.results_text.insert(tk.END, "✅ Easy: Suitable for beginners\n")
        elif diff_score['score'] < 15:
            self.results_text.insert(tk.END, "⚠️ Moderate: Requires intermediate skill\n")
        else:
            self.results_text.insert(tk.END, "🔴 Challenging: Advanced technique needed\n")

        # Configure tags for coloring
        self.results_text.tag_configure("header", foreground=COLOR_GOLD)
        self.results_text.tag_configure("easy", foreground="#90EE90")  # Light green
        self.results_text.tag_configure("moderate", foreground="#FFD700")  # Gold
        self.results_text.tag_configure("challenging", foreground="#FF6B6B")  # Light red

        # Apply tags
        self.results_text.tag_add("header", "1.0", "1.16")

        # Update difficulty label
        self.diff_var.set(f"Difficulty score: {diff_score['score']:.2f} "
                         f"(shifts: {diff_score['shifts']}, "
                         f"high pos: {diff_score['high_positions']}, "
                         f"stretches: {diff_score['stretch_intervals']})")

    def update_visualization(self, plan):
        """Update the fingering visualization on canvas."""
        self.canvas.delete("all")

        if not plan:
            return

        canvas_width = self.canvas.winfo_width() - 20
        canvas_height = self.canvas.winfo_height() - 20

        if canvas_width <= 0 or canvas_height <= 0:
            return

        # Draw strings as horizontal lines
        string_positions = {'G': 0.1, 'D': 0.3, 'A': 0.5, 'E': 0.7}

        for string, pos in string_positions.items():
            y = canvas_height * pos
            color = self.string_colors[string]

            self.canvas.create_line(10, y, canvas_width - 10, y,
                                   fill=color, width=2, dash=(5, 2))

            self.canvas.create_text(canvas_width - 5, y,
                                   text=string, anchor="e",
                                   fill=color, font=FONT_SMALL)

        # Draw notes
        note_spacing = (canvas_width - 40) / max(len(plan) - 1, 1)

        for i, item in enumerate(plan):
            x = 20 + i * note_spacing
            string_y = canvas_height * string_positions[item["string"]]

            # Position affects vertical placement
            pos_factor = 1.0 - (item["position"] / 7.0) * 0.4
            y = string_y * pos_factor

            # Draw note circle
            color = self.string_colors[item["string"]]
            radius = 15

            self.canvas.create_oval(x - radius, y - radius,
                                   x + radius, y + radius,
                                   fill=color, outline=COLOR_GOLD, width=2)

            # Draw finger number
            self.canvas.create_text(x, y,
                                   text=str(item["finger"]),
                                   fill="white" if color != '#F4A460' else 'black',
                                   font=("Arial", 10, "bold"))

            # Draw shift indicator
            if item["shift"]:
                self.canvas.create_rectangle(x - radius - 2, y - radius - 2,
                                           x + radius + 2, y + radius + 2,
                                           outline=COLOR_GOLD, width=2,
                                           dash=(3, 3))

            # Draw note label
            note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            note_name = note_names[item["note_name"]]
            octave = item["note"] // 12 - 1

            self.canvas.create_text(x, y + radius + 10,
                                   text=f"{note_name}{octave}",
                                   fill=COLOR_TEXT, font=FONT_SMALL)

            # Draw position indicator
            self.canvas.create_text(x, y - radius - 10,
                                   text=f"Pos {item['position']}",
                                   fill=COLOR_GOLD, font=FONT_SMALL)

        # Draw connecting lines between notes
        for i in range(len(plan) - 1):
            x1 = 20 + i * note_spacing
            x2 = 20 + (i + 1) * note_spacing

            item1 = plan[i]
            item2 = plan[i + 1]

            y1 = canvas_height * string_positions[item1["string"]]
            y2 = canvas_height * string_positions[item2["string"]]

            pos_factor1 = 1.0 - (item1["position"] / 7.0) * 0.4
            pos_factor2 = 1.0 - (item2["position"] / 7.0) * 0.4

            y1 = y1 * pos_factor1
            y2 = y2 * pos_factor2

            # Line color indicates transition type
            if item1["string"] != item2["string"]:
                line_color = "#FF6B6B"  # Red for string crossing
                width = 3
            elif item1["position"] != item2["position"]:
                line_color = "#FFD700"  # Gold for position shift
                width = 2
            else:
                line_color = "#90EE90"  # Green for same position
                width = 1

            self.canvas.create_line(x1, y1, x2, y2,
                                   fill=line_color, width=width,
                                   arrow=tk.LAST, dash=(5, 2) if width > 1 else None)

    def export_fingering(self):
        """Export current fingering plan to JSON file."""
        if not self.current_plan:
            messagebox.showwarning("No Plan", "Please generate a fingering plan first.")
            return

        from tkinter import filedialog

        filename = filedialog.asksaveasfilename(
            title="Save Fingering Plan",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                success = self.generator.export_to_json(self.current_plan, filename)
                if success:
                    messagebox.showinfo("Success", f"Fingering plan saved to:\n{filename}")
                    self.status_var.set(f"✅ Exported to {os.path.basename(filename)}")
                else:
                    messagebox.showerror("Export Error", "Failed to save file.")
            except Exception as e:
                messagebox.showerror("Export Error", f"Error saving file: {str(e)}")
                self.status_var.set("❌ Export failed")

    def clear_results(self):
        """Clear all results and visualization."""
        self.current_plan = None
        self.results_text.delete(1.0, tk.END)
        self.timeline_var.set("")
        self.diff_var.set("Difficulty score: --")
        self.canvas.delete("all")
        self.status_var.set("✅ Results cleared")

def get_content():
    """Return module description for display in GUI."""
    return [
        "╔══════════════════════════════════════╗",
        "║      FINGERING GENERATOR MODULE      ║",
        "╚══════════════════════════════════════╝",
        "",
        "🎻 PURPOSE: Automatic violin fingering generation with position optimization",
        "",
        "🔧 FEATURES:",
        "  • Dynamic programming algorithm for optimal fingerings",
        "  • Heuristic algorithm for easy/beginner fingerings",
        "  • Position shift detection and optimization",
        "  • String crossing minimization",
        "  • Difficulty scoring system",
        "",
        "📊 VISUALIZATION:",
        "  • Color-coded string representation",
        "  • Position and finger number display",
        "  • Shift indicators",
        "  • Transition visualization",
        "",
        "📈 DIFFICULTY METRICS:",
        "  • Number of position shifts",
        "  • High positions (4th and above)",
        "  • Finger stretch intervals",
        "  • Overall difficulty score",
        "",
        "🚀 QUICK START:",
        "  1. Enter notes as MIDI numbers or note names",
        "  2. Choose algorithm (Optimal or Easy)",
        "  3. Click 'Generate Fingering'",
        "  4. View, visualize, and export results",
    ]

def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    # Create the fingering generator GUI
    gui = FingeringGeneratorGUI(parent)
    return gui.main_frame

def test_standalone():
    """Test the module GUI standalone."""
    root = tk.Tk()
    root.title("Fingering Generator Module - Standalone Test")
    root.geometry(WINDOW_SIZE)
    root.configure(bg=COLOR_BG)

    # Make window non-resizable
    root.resizable(False, False)

    # Create the GUI
    gui = create_gui(root)

    # Add close button for standalone test
    close_btn = tk.Button(root,
                         text="Close Test",
                         command=root.destroy,
                         font=FONT_TEXT,
                         fg=COLOR_GOLD,
                         bg=COLOR_DARK)
    close_btn.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    test_standalone()