#!/usr/bin/env python3
# ai_violin_master_10_structure_analyzer.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683


import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
import json
import os
from datetime import datetime
import pretty_midi
import configparser

# Color constants
COLOR_BG = "#5A381E"
COLOR_DARK = "#3B2413"
COLOR_GOLD = "#FFD770"
COLOR_FRAME = "#4B2E18"
COLOR_TEXT = "#FFF2CF"
COLOR_MEDIUM = "#4B2E18"

WINDOW_SIZE = "640x1050"


class StructureAnalyzer:
    """Advanced musical structure analysis."""

    def analyze_structure(self, notes, pm) -> dict:
        return {
            "ornaments": self._ornamentation(notes),
            "tuplets": self._tuplet_detection(notes),
            "syncopation": self._syncopation_score(notes),
            "swing": self._swing_detection(notes),
            "cross_rhythm": self._cross_rhythm(notes),
            "motifs": self._motif_analysis(notes),
            "sections": self._section_boundaries(notes),
            "repetitions": self._repetition_map(notes),
            "chords": self._harmonic_progression(pm),
            "clusters": self._note_cluster_density(notes)
        }

    def _ornamentation(self, notes) -> list:
        ornaments = []
        pitches = [n.pitch for n in notes]
        durations = [n.duration for n in notes]
        for i in range(len(notes) - 2):
            p0, p1, p2 = pitches[i:i+3]
            if abs(p0 - p1) == 1 and abs(p2 - p1) == 1:
                if durations[i] < 0.15 and durations[i+1] < 0.15:
                    ornaments.append({"type": "trill", "start": notes[i].start, "end": notes[i+2].end})
            if p0 > p1 < p2 and abs(p0 - p1) == 1:
                ornaments.append({"type": "mordent", "start": notes[i].start, "end": notes[i+2].end})
            if i+3 < len(notes):
                p3 = pitches[i+3]
                seq = [p0, p1, p2, p3]
                if seq[0] > seq[1] < seq[2] > seq[3] or seq[0] < seq[1] > seq[2] < seq[3]:
                    ornaments.append({"type": "turn", "start": notes[i].start, "end": notes[i+3].end})
        return ornaments

    def _tuplet_detection(self, notes) -> list:
        tuplets = []
        durations = [n.duration for n in notes]
        if len(durations) < 3:
            return tuplets
        for i in range(len(durations) - 2):
            d0, d1, d2 = durations[i:i+3]
            if abs(d0 - d1) < 0.01 and abs(d1 - d2) < 0.01:
                ref = self._local_average_duration(durations, i, window=8)
                if ref > 0 and (d0 < ref * 0.8):
                    tuplets.append(("triplet", i))
        return tuplets

    def _local_average_duration(self, durations, i, window=6):
        lo = max(0, i - window)
        hi = min(len(durations), i + window)
        region = durations[lo:hi]
        if region:
            return float(np.mean(region))
        return 0.0

    def _syncopation_score(self, notes) -> float:
        if not notes:
            return 0.0
        starts = [n.start for n in notes]
        dstarts = np.diff(starts)
        median = np.median(dstarts) if len(dstarts) > 0 else 0.5
        offbeat = sum(1 for s in starts if (s / median) % 1 > 0.33 and (s / median) % 1 < 0.66)
        return float(min(offbeat / len(starts) * 100, 100))

    def _swing_detection(self, notes) -> str:
        if len(notes) < 3:
            return "straight"
        ratios = []
        for i in range(0, len(notes) - 1, 2):
            d1 = notes[i].duration
            d2 = notes[i+1].duration
            if d2 > 0:
                ratios.append(d1 / d2)
        if not ratios:
            return "straight"
        avg = np.mean(ratios)
        if 1.3 < avg < 2.2:
            return "swing eighths"
        if avg >= 2.2:
            return "hard swing"
        return "straight"

    def _cross_rhythm(self, notes) -> str:
        if len(notes) < 5:
            return "none"
        starts = [n.start for n in notes]
        spacings = np.diff(starts)
        if len(spacings) == 0:
            return "none"
        ratios = []
        for a, b in zip(spacings[:-1], spacings[1:]):
            if b > 0 and a > 0:
                ratios.append(a / b)
        if not ratios:
            return "none"
        hist, bins = np.histogram(ratios, bins=10)
        peak_bin = np.argmax(hist)
        center_ratio = (bins[peak_bin] + bins[peak_bin+1]) / 2
        if 1.4 < center_ratio < 1.6:
            return "3:2 cross-rhythm"
        if 1.8 < center_ratio < 2.2:
            return "2:1 hemiola"
        return "none"

    def _motif_analysis(self, notes) -> list:
        if len(notes) < 6:
            return []
        pitches = [n.pitch for n in notes]
        ivals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
        motifs = []
        window_sizes = [3, 4, 5, 6]
        for w in window_sizes:
            patterns = {}
            for i in range(len(ivals) - w):
                pat = tuple(ivals[i:i+w])
                patterns.setdefault(pat, []).append(i)
            for pat, locs in patterns.items():
                if len(locs) >= 2:
                    motifs.append({"interval_pattern": pat, "occurrences": locs})
        return motifs

    def _section_boundaries(self, notes) -> list:
        if len(notes) < 6:
            return []
        starts = [n.start for n in notes]
        pitches = [n.pitch for n in notes]
        boundaries = []
        current_start = 0
        for i in range(2, len(notes)-2):
            gap = starts[i] - starts[i-1]
            if gap > 0.6:
                boundaries.append({"start_note": current_start, "end_note": i-1})
                current_start = i
            if abs(pitches[i] - pitches[i-1]) > 12:
                boundaries.append({"start_note": current_start, "end_note": i-1})
                current_start = i
        boundaries.append({"start_note": current_start, "end_note": len(notes)-1})
        return boundaries

    def _repetition_map(self, notes) -> list:
        if len(notes) < 10:
            return []
        pitches = [n.pitch for n in notes]
        ivals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
        hashes = {}
        results = []
        window = 8
        for i in range(len(ivals) - window):
            pat = tuple(np.sign(ivals[i:i+window]))
            if pat in hashes:
                for prev in hashes[pat]:
                    results.append((prev, i))
                hashes[pat].append(i)
            else:
                hashes[pat] = [i]
        return results

    def _harmonic_progression(self, pm):
        harmony = []
        for inst in pm.instruments:
            notes = sorted(inst.notes, key=lambda x: x.start)
            for i in range(len(notes)-2):
                a, b, c = notes[i:i+3]
                if a.start == b.start == c.start:
                    chord = sorted([a.pitch, b.pitch, c.pitch])
                    harmony.append({"time": a.start, "chord": chord, "label": self._label_chord(chord)})
        return harmony

    def _label_chord(self, chord):
        pcs = sorted([(p % 12) for p in chord])
        intervals = [(pcs[i] - pcs[0]) % 12 for i in range(len(pcs))]
        if intervals == [0, 4, 7]:
            return "Major"
        if intervals == [0, 3, 7]:
            return "Minor"
        if intervals == [0, 3, 6]:
            return "Diminished"
        if intervals == [0, 4, 8]:
            return "Augmented"
        return "Other"

    def _note_cluster_density(self, notes) -> list:
        times = [n.start for n in notes]
        if not times:
            return []
        result = []
        window = 0.5
        for t in times:
            c = sum(1 for tp in times if abs(tp - t) < window)
            result.append((t, c))
        return result


class StructureAnalyzerGUI:
    """GUI for Structure Analyzer using CustomTkinter."""

    def __init__(self, parent=None):
        self.ini_file = "ai_violin_master_10_structure_analyzer.ini"
        self.config = configparser.ConfigParser()

        if parent is None:
            self.root = ctk.CTk()
            self.is_standalone = True
            self.root.title("Structure Analyzer")
            self.root.geometry(WINDOW_SIZE)
            self.root.resizable(False, False)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        else:
            self.root = parent
            self.is_standalone = False

        self.analyzer = StructureAnalyzer()
        self.sample_data = None
        self.analysis_results = None

        self._load_state()
        ctk.set_appearance_mode("dark")
        self.root.configure(fg_color=COLOR_BG)

        self._create_widgets()
        self._generate_sample_data()

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

    def _generate_sample_data(self):
        pm = pretty_midi.PrettyMIDI()
        violin = pretty_midi.Instrument(program=40)
        notes = []
        for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):
            note = pretty_midi.Note(velocity=100, pitch=pitch, start=i * 0.5, end=i * 0.5 + 0.4)
            violin.notes.append(note)
            notes.append(note)
        for i, chord_notes in enumerate([[60, 64, 67], [62, 65, 69]]):
            for pitch in chord_notes:
                note = pretty_midi.Note(velocity=80, pitch=pitch, start=4 + i * 0.5, end=4 + i * 0.5 + 0.4)
                violin.notes.append(note)
        pm.instruments.append(violin)
        self.sample_data = {"notes": notes, "pm": pm}

    def _create_widgets(self):
        # Main container
        main_frame = ctk.CTkFrame(self.root, fg_color=COLOR_BG)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        header_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            header_frame,
            text="MUSICAL STRUCTURE ANALYZER",
            font=("Georgia", 16, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            header_frame,
            text="Advanced analysis of musical structure, ornaments, and patterns",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=(0, 10))

        # File selection
        file_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        file_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            file_frame,
            text="MIDI FILE INPUT",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        self.file_path_var = ctk.StringVar(value="No file selected")
        ctk.CTkLabel(
            file_frame,
            textvariable=self.file_path_var,
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=5)

        btn_row = ctk.CTkFrame(file_frame, fg_color=COLOR_DARK)
        btn_row.pack(fill="x", padx=20, pady=10)

        ctk.CTkButton(
            btn_row,
            text="LOAD MIDI FILE",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_FRAME,
            command=self._load_midi_file,
        ).pack(side="left", padx=5, expand=True, fill="x")

        ctk.CTkButton(
            btn_row,
            text="USE SAMPLE DATA",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_FRAME,
            command=self._use_sample_data,
        ).pack(side="left", padx=5, expand=True, fill="x")

        # Analysis buttons
        analysis_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        analysis_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            analysis_frame,
            text="ANALYSIS OPTIONS",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        self.run_btn = ctk.CTkButton(
            analysis_frame,
            text="RUN FULL ANALYSIS",
            font=("Georgia", 14, "bold"),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_FRAME,
            height=40,
            command=self._run_full_analysis,
        )
        self.run_btn.pack(pady=10)

        # Results display
        results_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        results_frame.pack(fill="both", expand=True, pady=10)

        ctk.CTkLabel(
            results_frame,
            text="ANALYSIS RESULTS",
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
            wrap="word",
        )
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.results_text.insert("1.0", "Run analysis to see results here.")
        self.results_text.configure(state="disabled")

        # Status bar
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Ready to analyze",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            height=30,
        )
        self.status_label.pack(fill="x", pady=(10, 0))

    def _load_midi_file(self):
        filename = filedialog.askopenfilename(
            title="Select MIDI File",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")]
        )
        if filename:
            pm = pretty_midi.PrettyMIDI(filename)
            notes = []
            for instrument in pm.instruments:
                notes.extend(instrument.notes)
            self.sample_data = {"notes": notes, "pm": pm}
            self.file_path_var.set(os.path.basename(filename))
            self.status_label.configure(text=f"{len(notes)} notes loaded from {os.path.basename(filename)}")

    def _use_sample_data(self):
        if self.sample_data:
            self.file_path_var.set("Sample Data")
            note_count = len(self.sample_data["notes"])
            self.status_label.configure(text=f"{note_count} sample notes loaded")

    def _run_full_analysis(self):
        if not self.sample_data:
            messagebox.showwarning("No Data", "Please load MIDI file or use sample data first")
            return

        self.status_label.configure(text="Analyzing structure...")
        self.root.update()

        self.analysis_results = self.analyzer.analyze_structure(
            self.sample_data["notes"],
            self.sample_data["pm"]
        )

        output = "STRUCTURE ANALYSIS SUMMARY\n"
        output += "=" * 40 + "\n\n"

        if self.analysis_results.get("ornaments"):
            output += f"Ornaments: {len(self.analysis_results['ornaments'])} found\n"
        if self.analysis_results.get("tuplets"):
            output += f"Tuplets: {len(self.analysis_results['tuplets'])} found\n"
        if self.analysis_results.get("syncopation"):
            output += f"Syncopation Score: {self.analysis_results['syncopation']:.1f}/100\n"
        if self.analysis_results.get("swing"):
            output += f"Swing Feel: {self.analysis_results['swing']}\n"
        if self.analysis_results.get("cross_rhythm"):
            output += f"Cross Rhythm: {self.analysis_results['cross_rhythm']}\n"
        if self.analysis_results.get("motifs"):
            output += f"Motifs: {len(self.analysis_results['motifs'])} patterns found\n"
        if self.analysis_results.get("sections"):
            output += f"Sections: {len(self.analysis_results['sections'])} detected\n"
        if self.analysis_results.get("chords"):
            output += f"Chords: {len(self.analysis_results['chords'])} identified\n"
        if self.analysis_results.get("repetitions"):
            output += f"Repetitions: {len(self.analysis_results['repetitions'])} found\n"

        output += "\n\nDETAILED RESULTS:\n"
        output += "=" * 40 + "\n\n"

        for section, data in self.analysis_results.items():
            output += f"\n{section.upper().replace('_', ' ')}:\n"
            output += "-" * 30 + "\n"
            if isinstance(data, list):
                if data:
                    for i, item in enumerate(data[:10]):
                        output += f"  {i+1}. {str(item)}\n"
                    if len(data) > 10:
                        output += f"  ... and {len(data) - 10} more\n"
                else:
                    output += "  None found\n"
            else:
                output += f"  {data}\n"

        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", output)
        self.results_text.configure(state="disabled")

        self.status_label.configure(text="Analysis complete")

    def _on_close(self):
        self._save_state()
        self.root.destroy()


def get_content():
    """Return module description for display in GUI."""
    return [
        "STRUCTURE ANALYZER MODULE",
        "",
        "PURPOSE: Advanced analysis of musical structure",
        "",
        "ANALYSIS CAPABILITIES:",
        "  - Ornament detection (trills, mordents, turns)",
        "  - Tuplet and rhythmic pattern analysis",
        "  - Syncopation and swing detection",
        "  - Cross-rhythm identification",
        "  - Motif and pattern recognition",
        "  - Section boundary detection",
        "  - Harmonic progression analysis",
        "  - Repetition mapping",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = StructureAnalyzerGUI(parent)
    return gui.root


def main():
    """Standalone entry - launches a full window with mainloop."""
    StructureAnalyzerGUI()


if __name__ == "__main__":
    main()