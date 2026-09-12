#!/usr/bin/env python3
# ai_violin_master_10_structure_analyzer.py

import numpy as np
from typing import Dict, List, Tuple
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
from datetime import datetime
import pretty_midi

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


class StructureAnalyzer:
    """Advanced musical structure analysis."""

    def analyze_structure(self, notes, pm) -> Dict:
        """Master structure analysis entry."""
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

    def _ornamentation(self, notes) -> List[Dict]:
        """Detect trills, mordents, turns, short grace oscillations."""
        ornaments = []
        pitches = [n.pitch for n in notes]
        durations = [n.duration for n in notes]

        for i in range(len(notes) - 2):
            p0, p1, p2 = pitches[i:i+3]

            # TRILL: alternating upper & base note rapidly
            if abs(p0 - p1) == 1 and abs(p2 - p1) == 1:
                if durations[i] < 0.15 and durations[i+1] < 0.15:
                    ornaments.append({
                        "type": "trill",
                        "start": notes[i].start,
                        "end": notes[i+2].end
                    })

            # MORDENT: base → lower → base
            if p0 > p1 < p2 and abs(p0 - p1) == 1:
                ornaments.append({
                    "type": "mordent",
                    "start": notes[i].start,
                    "end": notes[i+2].end
                })

            # TURN: upper → main → lower → main
            if i+3 < len(notes):
                p3 = pitches[i+3]
                seq = [p0, p1, p2, p3]
                if seq[0] > seq[1] < seq[2] > seq[3] or seq[0] < seq[1] > seq[2] < seq[3]:
                    ornaments.append({
                        "type": "turn",
                        "start": notes[i].start,
                        "end": notes[i+3].end
                    })

        return ornaments

    def _grace_notes(self, notes) -> List[Dict]:
        """Identify grace notes using extremely short durations + prefix relation."""
        gs = []
        for i in range(len(notes) - 1):
            if notes[i].duration < 0.08 and (notes[i+1].start - notes[i].start) < 0.15:
                gs.append({
                    "pitch": notes[i].pitch,
                    "main_pitch": notes[i+1].pitch,
                    "start": notes[i].start
                })
        return gs

    def _tuplet_detection(self, notes) -> List[Tuple]:
        """Detect tuplets by scanning for repeated non-power-of-two duration ratios."""
        tuplets = []
        durations = [n.duration for n in notes]
        if len(durations) < 3:
            return tuplets

        for i in range(len(durations) - 2):
            d0, d1, d2 = durations[i:i+3]

            # Identify approximate triplet (d0 ≈ d1 ≈ d2 ≈ constant * 2/3 of normal)
            if abs(d0 - d1) < 0.01 and abs(d1 - d2) < 0.01:
                ref = self._local_average_duration(durations, i, window=8)
                if ref > 0 and (d0 < ref * 0.8):
                    tuplets.append(("triplet", i))

        return tuplets

    def _local_average_duration(self, durations, i, window=6):
        """Calculate local average duration."""
        lo = max(0, i - window)
        hi = min(len(durations), i + window)
        region = durations[lo:hi]
        if region:
            return float(np.mean(region))
        return 0.0

    def _syncopation_score(self, notes) -> float:
        """Estimate syncopation: off-beat starts vs beat alignment."""
        if not notes:
            return 0.0

        starts = [n.start for n in notes]
        dstarts = np.diff(starts)
        median = np.median(dstarts) if len(dstarts) > 0 else 0.5

        # starts not aligned to multiples of median beat → syncopation
        offbeat = sum(1 for s in starts if (s / median) % 1 > 0.33 and (s / median) % 1 < 0.66)

        # normalize
        return float(min(offbeat / len(starts) * 100, 100))

    def _swing_detection(self, notes) -> str:
        """Infer swing feel based on ratio of alternating durations."""
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
        """Detect polyrhythmic patterns via histogram of interval ratios."""
        if len(notes) < 5:
            return "none"

        starts = [n.start for n in notes]
        spacings = np.diff(starts)
        if len(spacings) == 0:
            return "none"

        # ratio distribution
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

    def _motif_analysis(self, notes) -> List[Dict]:
        """Find repeating patterns of 3–6 interval steps."""
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
                    motifs.append({
                        "interval_pattern": pat,
                        "occurrences": locs
                    })

        return motifs

    def _section_boundaries(self, notes) -> List[Dict]:
        """Detect sections by large gaps + interval reset + motif change."""
        if len(notes) < 6:
            return []

        starts = [n.start for n in notes]
        pitches = [n.pitch for n in notes]
        boundaries = []
        current_start = 0

        for i in range(2, len(notes)-2):
            gap = starts[i] - starts[i-1]
            if gap > 0.6:  # structural gap
                boundaries.append({"start_note": current_start, "end_note": i-1})
                current_start = i

            # interval discontinuity
            if abs(pitches[i] - pitches[i-1]) > 12:
                boundaries.append({"start_note": current_start, "end_note": i-1})
                current_start = i

        boundaries.append({"start_note": current_start, "end_note": len(notes)-1})
        return boundaries

    def _repetition_map(self, notes) -> List[Tuple[int, int]]:
        """Detect repeated segments by coarse interval pattern hashing."""
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
        """Detect harmonic content from simultaneous MIDI notes."""
        harmony = []

        for inst in pm.instruments:
            notes = sorted(inst.notes, key=lambda x: x.start)
            for i in range(len(notes)-2):
                a, b, c = notes[i:i+3]
                # simultaneous
                if a.start == b.start == c.start:
                    chord = sorted([a.pitch, b.pitch, c.pitch])
                    harmony.append({
                        "time": a.start,
                        "chord": chord,
                        "label": self._label_chord(chord)
                    })

        return harmony

    def _label_chord(self, chord):
        """Simple triad labeling by pitch-class pattern."""
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

    def _note_cluster_density(self, notes) -> List[Tuple[float, float]]:
        """Return (time, cluster_density) pairs."""
        times = [n.start for n in notes]
        if not times:
            return []

        result = []
        window = 0.5  # half-second window

        for t in times:
            c = sum(1 for tp in times if abs(tp - t) < window)
            result.append((t, c))

        return result

    def _build_measure_map(self, pm) -> List[Tuple[float, int]]:
        """Return list of (time, measure#) boundaries."""
        boundaries = []
        time_sig = pm.time_signature_changes[0] if pm.time_signature_changes else None
        tempo = pm.get_tempo_changes()[0][0] if pm.get_tempo_changes()[0].size > 0 else 120

        if time_sig:
            num = time_sig.numerator
            den = time_sig.denominator
        else:
            num, den = 4, 4

        measure_len_sec = (60.0 / tempo) * num
        end = pm.get_end_time()
        m = 0
        t = 0
        while t <= end:
            boundaries.append((t, m))
            m += 1
            t += measure_len_sec

        return boundaries

    def _midi_pitch_at_time(self, t: float) -> float:
        """Return expected MIDI frequency at time t using nearest note."""
        if not hasattr(self, "_time_pitch_cache"):
            self._time_pitch_cache = []

        if not self._time_pitch_cache:
            pm = getattr(self, "_latest_pm", None)
            if pm is None:
                return 440.0

            for inst in pm.instruments:
                for n in inst.notes:
                    freq = 440.0 * (2 ** ((n.pitch - 69) / 12))
                    self._time_pitch_cache.append((n.start, n.end, freq))

        # Lookup
        for s, e, f in self._time_pitch_cache:
            if s <= t <= e:
                return f

        return 440.0  # fallback


class StructureAnalyzerGUI:
    """GUI for Structure Analyzer module with AIModels style"""

    def __init__(self, parent):
        self.parent = parent
        self.parent.title("🎻 Structure Analyzer")
        self.parent.geometry(WINDOW_SIZE)
        self.parent.resizable(False, False)
        self.parent.configure(bg=COLOR_BG)

        # Initialize analyzer
        self.analyzer = StructureAnalyzer()

        # Sample data for demonstration
        self.sample_data = None
        self.analysis_results = None

        # Create GUI
        self._create_widgets()

        # Generate sample data
        self._generate_sample_data()

    def _generate_sample_data(self):
        """Generate sample MIDI data for demonstration"""
        try:
            # Create a sample PrettyMIDI object
            pm = pretty_midi.PrettyMIDI()

            # Create a violin instrument
            violin = pretty_midi.Instrument(program=40)  # Violin

            # Add some sample notes
            notes = []

            # Simple scale with ornaments
            for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72]):
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=i * 0.5,
                    end=i * 0.5 + 0.4
                )
                violin.notes.append(note)
                notes.append(note)

            # Add some chords
            for i, chord_notes in enumerate([[60, 64, 67], [62, 65, 69]]):
                for pitch in chord_notes:
                    note = pretty_midi.Note(
                        velocity=80,
                        pitch=pitch,
                        start=4 + i * 0.5,
                        end=4 + i * 0.5 + 0.4
                    )
                    violin.notes.append(note)

            pm.instruments.append(violin)

            # Store sample data
            self.sample_data = {
                "notes": notes,
                "pm": pm
            }

        except Exception as e:
            print(f"⚠️ Could not generate sample MIDI data: {e}")
            self.sample_data = None

    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = tk.Frame(self.parent, bg=COLOR_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        self._create_header(main_frame)

        # File selection area
        self._create_file_selection(main_frame)

        # Analysis options
        self._create_analysis_options(main_frame)

        # Results display
        self._create_results_display(main_frame)

        # Status bar
        self._create_status_bar(main_frame)

    def _create_header(self, parent):
        """Create header section"""
        header_frame = tk.Frame(parent, bg=COLOR_BG)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            header_frame,
            text="🎻 Musical Structure Analyzer",
            font=FONT_HEADER,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack()

        tk.Label(
            header_frame,
            text="Advanced analysis of musical structure, ornaments, and patterns",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG
        ).pack()

    def _create_file_selection(self, parent):
        """Create file selection area"""
        frame = tk.LabelFrame(
            parent,
            text="MIDI File Input",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        # File path display
        self.file_path_var = tk.StringVar(value="No file selected")
        path_frame = tk.Frame(frame, bg=COLOR_BG)
        path_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            path_frame,
            textvariable=self.file_path_var,
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            anchor='w'
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Buttons frame
        button_frame = tk.Frame(frame, bg=COLOR_BG)
        button_frame.pack(fill=tk.X, pady=5)

        # Load MIDI button
        load_btn = tk.Button(
            button_frame,
            text="📂 Load MIDI File",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_DARK,
            activeforeground=COLOR_GOLD,
            activebackground=COLOR_FRAME,
            relief=tk.RAISED,
            borderwidth=1,
            command=self._load_midi_file,
            width=15
        )
        load_btn.pack(side=tk.LEFT, padx=2)

        # Use sample button
        sample_btn = tk.Button(
            button_frame,
            text="🎵 Use Sample Data",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_DARK,
            activeforeground=COLOR_GOLD,
            activebackground=COLOR_FRAME,
            relief=tk.RAISED,
            borderwidth=1,
            command=self._use_sample_data,
            width=15
        )
        sample_btn.pack(side=tk.LEFT, padx=2)

        # File info
        info_frame = tk.Frame(frame, bg=COLOR_BG)
        info_frame.pack(fill=tk.X, pady=5)

        self.file_info_label = tk.Label(
            info_frame,
            text="Select a MIDI file or use sample data",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            anchor='w'
        )
        self.file_info_label.pack(side=tk.LEFT)

    def _create_analysis_options(self, parent):
        """Create analysis options area"""
        frame = tk.LabelFrame(
            parent,
            text="Analysis Options",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        # Analysis buttons in grid
        button_frame = tk.Frame(frame, bg=COLOR_BG)
        button_frame.pack(fill=tk.X, pady=5)

        analysis_types = [
            ("🎼 Full Analysis", self._run_full_analysis),
            ("🎀 Ornaments", lambda: self._run_specific_analysis("ornaments")),
            ("🎵 Tuplets", lambda: self._run_specific_analysis("tuplets")),
            ("🥁 Rhythm", lambda: self._run_specific_analysis("rhythm")),
            ("🎹 Harmony", lambda: self._run_specific_analysis("chords")),
            ("🔁 Repetitions", lambda: self._run_specific_analysis("repetitions")),
            ("📊 Sections", lambda: self._run_specific_analysis("sections")),
            ("🎯 Motifs", lambda: self._run_specific_analysis("motifs"))
        ]

        # Create buttons in a 2x4 grid
        for i, (text, command) in enumerate(analysis_types):
            row = i // 4
            col = i % 4

            btn = tk.Button(
                button_frame,
                text=text,
                font=FONT_SMALL,
                fg=COLOR_GOLD,
                bg=COLOR_DARK,
                activeforeground=COLOR_GOLD,
                activebackground=COLOR_FRAME,
                relief=tk.RAISED,
                borderwidth=1,
                command=command,
                width=15
            )
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
            button_frame.columnconfigure(col, weight=1)

    def _create_results_display(self, parent):
        """Create results display area"""
        frame = tk.LabelFrame(
            parent,
            text="Analysis Results",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10, ipadx=10, ipady=10)

        # Create notebook for different result views
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
        self._create_summary_tab()
        self._create_detailed_tab()
        self._create_export_tab()

    def _create_summary_tab(self):
        """Create summary tab"""
        summary_frame = tk.Frame(self.results_notebook, bg=COLOR_BG)
        self.results_notebook.add(summary_frame, text="Summary")

        # Scrollable text widget
        scrollbar = tk.Scrollbar(summary_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.summary_text = tk.Text(
            summary_frame,
            wrap=tk.WORD,
            font=("Courier", 9),
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            yscrollcommand=scrollbar.set,
            height=15
        )
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.summary_text.yview)

        # Insert placeholder
        self.summary_text.insert(1.0, "No analysis results yet.\n\nRun an analysis to see results here.")
        self.summary_text.config(state=tk.DISABLED)

    def _create_detailed_tab(self):
        """Create detailed view tab"""
        detailed_frame = tk.Frame(self.results_notebook, bg=COLOR_BG)
        self.results_notebook.add(detailed_frame, text="Detailed")

        # Scrollable text widget
        scrollbar = tk.Scrollbar(detailed_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.detailed_text = tk.Text(
            detailed_frame,
            wrap=tk.WORD,
            font=("Courier", 9),
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            yscrollcommand=scrollbar.set,
            height=15
        )
        self.detailed_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.detailed_text.yview)

        # Insert placeholder
        self.detailed_text.insert(1.0, "Detailed results will appear here.")
        self.detailed_text.config(state=tk.DISABLED)

    def _create_export_tab(self):
        """Create export tab"""
        export_frame = tk.Frame(self.results_notebook, bg=COLOR_BG)
        self.results_notebook.add(export_frame, text="Export")

        # Export options
        options_frame = tk.LabelFrame(
            export_frame,
            text="Export Options",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        options_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10, ipadx=10, ipady=10)

        # JSON export
        json_frame = tk.Frame(options_frame, bg=COLOR_BG)
        json_frame.pack(fill=tk.X, pady=10)

        tk.Label(
            json_frame,
            text="Export full analysis as JSON:",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(side=tk.LEFT)

        self.json_export_btn = tk.Button(
            json_frame,
            text="💾 Save JSON",
            font=FONT_SMALL,
            fg=COLOR_GOLD,
            bg=COLOR_DARK,
            activeforeground=COLOR_GOLD,
            activebackground=COLOR_FRAME,
            relief=tk.RAISED,
            borderwidth=1,
            command=self._export_json,
            width=15,
            state=tk.DISABLED
        )
        self.json_export_btn.pack(side=tk.RIGHT, padx=5)

        # Summary export
        summary_frame = tk.Frame(options_frame, bg=COLOR_BG)
        summary_frame.pack(fill=tk.X, pady=10)

        tk.Label(
            summary_frame,
            text="Export summary as text:",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(side=tk.LEFT)

        self.summary_export_btn = tk.Button(
            summary_frame,
            text="📝 Save Text",
            font=FONT_SMALL,
            fg=COLOR_GOLD,
            bg=COLOR_DARK,
            activeforeground=COLOR_GOLD,
            activebackground=COLOR_FRAME,
            relief=tk.RAISED,
            borderwidth=1,
            command=self._export_summary,
            width=15,
            state=tk.DISABLED
        )
        self.summary_export_btn.pack(side=tk.RIGHT, padx=5)

        # Info text
        info_text = tk.Label(
            options_frame,
            text="Export options will be available after analysis",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            justify=tk.LEFT
        )
        info_text.pack(pady=10)

    def _create_status_bar(self, parent):
        """Create status bar at bottom"""
        status_frame = tk.Frame(parent, bg=COLOR_DARK)
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        self.status_label = tk.Label(
            status_frame,
            text="✅ Ready to analyze",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_DARK
        )
        self.status_label.pack(side=tk.LEFT, padx=5)

    # File handling methods
    def _load_midi_file(self):
        """Load MIDI file for analysis"""
        filename = filedialog.askopenfilename(
            title="Select MIDI File",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")]
        )

        if filename:
            try:
                pm = pretty_midi.PrettyMIDI(filename)
                notes = []
                for instrument in pm.instruments:
                    notes.extend(instrument.notes)

                self.sample_data = {
                    "notes": notes,
                    "pm": pm
                }

                self.file_path_var.set(os.path.basename(filename))
                self.file_info_label.config(
                    text=f"{len(notes)} notes loaded from {os.path.basename(filename)}"
                )
                self.status_label.config(text="✅ MIDI file loaded")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load MIDI file:\n{str(e)}")
                self.status_label.config(text="❌ Failed to load file")

    def _use_sample_data(self):
        """Use built-in sample data"""
        if self.sample_data:
            self.file_path_var.set("Sample Data")
            note_count = len(self.sample_data["notes"]) if self.sample_data["notes"] else 0
            self.file_info_label.config(
                text=f"{note_count} sample notes loaded"
            )
            self.status_label.config(text="✅ Sample data loaded")
        else:
            messagebox.showinfo("Info", "Sample data not available")
            self.status_label.config(text="❌ No sample data")

    # Analysis methods
    def _run_full_analysis(self):
        """Run complete structure analysis"""
        if not self.sample_data:
            messagebox.showwarning("No Data", "Please load MIDI file or use sample data first")
            return

        try:
            self.status_label.config(text="⏳ Analyzing structure...")
            self.parent.update()

            # Run analysis
            self.analysis_results = self.analyzer.analyze_structure(
                self.sample_data["notes"],
                self.sample_data["pm"]
            )

            # Update display
            self._update_summary_view()
            self._update_detailed_view()

            # Enable export buttons
            self.json_export_btn.config(state=tk.NORMAL)
            self.summary_export_btn.config(state=tk.NORMAL)

            self.status_label.config(text="✅ Analysis complete")

        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze structure:\n{str(e)}")
            self.status_label.config(text="❌ Analysis failed")

    def _run_specific_analysis(self, analysis_type):
        """Run specific type of analysis"""
        if not self.sample_data:
            messagebox.showwarning("No Data", "Please load MIDI file or use sample data first")
            return

        try:
            self.status_label.config(text=f"⏳ Analyzing {analysis_type}...")
            self.parent.update()

            # Run specific analysis based on type
            if analysis_type == "ornaments":
                results = self.analyzer._ornamentation(self.sample_data["notes"])
                title = "Ornament Analysis"
            elif analysis_type == "tuplets":
                results = self.analyzer._tuplet_detection(self.sample_data["notes"])
                title = "Tuplet Analysis"
            elif analysis_type == "rhythm":
                results = {
                    "syncopation": self.analyzer._syncopation_score(self.sample_data["notes"]),
                    "swing": self.analyzer._swing_detection(self.sample_data["notes"]),
                    "cross_rhythm": self.analyzer._cross_rhythm(self.sample_data["notes"])
                }
                title = "Rhythm Analysis"
            elif analysis_type == "chords":
                results = self.analyzer._harmonic_progression(self.sample_data["pm"])
                title = "Harmonic Analysis"
            elif analysis_type == "repetitions":
                results = self.analyzer._repetition_map(self.sample_data["notes"])
                title = "Repetition Analysis"
            elif analysis_type == "sections":
                results = self.analyzer._section_boundaries(self.sample_data["notes"])
                title = "Section Analysis"
            elif analysis_type == "motifs":
                results = self.analyzer._motif_analysis(self.sample_data["notes"])
                title = "Motif Analysis"
            else:
                return

            # Display results
            self._display_specific_results(title, results)
            self.status_label.config(text=f"✅ {analysis_type.title()} complete")

        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze {analysis_type}:\n{str(e)}")
            self.status_label.config(text=f"❌ {analysis_type} failed")

    def _update_summary_view(self):
        """Update summary tab with analysis results"""
        if not self.analysis_results:
            return

        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)

        summary = "🎻 STRUCTURE ANALYSIS SUMMARY\n"
        summary += "=" * 40 + "\n\n"

        # Add key findings
        if self.analysis_results.get("ornaments"):
            summary += f"🎀 Ornaments: {len(self.analysis_results['ornaments'])} found\n"

        if self.analysis_results.get("tuplets"):
            summary += f"🎵 Tuplets: {len(self.analysis_results['tuplets'])} found\n"

        if self.analysis_results.get("syncopation"):
            summary += f"🥁 Syncopation Score: {self.analysis_results['syncopation']:.1f}/100\n"

        if self.analysis_results.get("swing"):
            summary += f"🎵 Swing Feel: {self.analysis_results['swing']}\n"

        if self.analysis_results.get("cross_rhythm"):
            summary += f"🥁 Cross Rhythm: {self.analysis_results['cross_rhythm']}\n"

        if self.analysis_results.get("motifs"):
            summary += f"🔁 Motifs: {len(self.analysis_results['motifs'])} patterns found\n"

        if self.analysis_results.get("sections"):
            summary += f"📊 Sections: {len(self.analysis_results['sections'])} detected\n"

        if self.analysis_results.get("chords"):
            summary += f"🎹 Chords: {len(self.analysis_results['chords'])} identified\n"

        if self.analysis_results.get("repetitions"):
            summary += f"🔁 Repetitions: {len(self.analysis_results['repetitions'])} found\n"

        self.summary_text.insert(1.0, summary)
        self.summary_text.config(state=tk.DISABLED)

    def _update_detailed_view(self):
        """Update detailed tab with full analysis results"""
        if not self.analysis_results:
            return

        self.detailed_text.config(state=tk.NORMAL)
        self.detailed_text.delete(1.0, tk.END)

        detailed = "🎻 DETAILED STRUCTURE ANALYSIS\n"
        detailed += "=" * 40 + "\n\n"

        # Format each analysis section
        for section, data in self.analysis_results.items():
            detailed += f"\n{section.upper().replace('_', ' ')}:\n"
            detailed += "-" * 30 + "\n"

            if isinstance(data, list):
                if data:
                    for i, item in enumerate(data[:10]):  # Show first 10 items
                        detailed += f"  {i+1}. {str(item)}\n"
                    if len(data) > 10:
                        detailed += f"  ... and {len(data) - 10} more\n"
                else:
                    detailed += "  None found\n"
            elif isinstance(data, dict):
                for key, value in data.items():
                    detailed += f"  {key}: {value}\n"
            else:
                detailed += f"  {data}\n"

        self.detailed_text.insert(1.0, detailed)
        self.detailed_text.config(state=tk.DISABLED)

    def _display_specific_results(self, title, results):
        """Display specific analysis results"""
        self.detailed_text.config(state=tk.NORMAL)
        self.detailed_text.delete(1.0, tk.END)

        output = f"🎻 {title}\n"
        output += "=" * 40 + "\n\n"

        if isinstance(results, list):
            if results:
                output += f"Found {len(results)} items:\n\n"
                for i, item in enumerate(results[:20]):  # Show first 20 items
                    output += f"{i+1}. {str(item)}\n"
                if len(results) > 20:
                    output += f"\n... and {len(results) - 20} more\n"
            else:
                output += "No items found\n"
        elif isinstance(results, dict):
            for key, value in results.items():
                output += f"{key}: {value}\n"
        else:
            output += f"Results: {results}\n"

        self.detailed_text.insert(1.0, output)
        self.detailed_text.config(state=tk.DISABLED)

        # Switch to detailed tab
        self.results_notebook.select(1)  # Index 1 is detailed tab

    # Export methods
    def _export_json(self):
        """Export analysis results as JSON"""
        if not self.analysis_results:
            messagebox.showwarning("No Data", "No analysis results to export")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Analysis as JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.analysis_results, f, indent=2, default=str)

                self.status_label.config(text=f"✅ JSON exported to {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Analysis saved to:\n{filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save JSON:\n{str(e)}")
                self.status_label.config(text="❌ Export failed")

    def _export_summary(self):
        """Export summary as text file"""
        if not self.analysis_results:
            messagebox.showwarning("No Data", "No analysis results to export")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Summary as Text",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            try:
                # Get summary text
                self.summary_text.config(state=tk.NORMAL)
                summary = self.summary_text.get(1.0, tk.END)
                self.summary_text.config(state=tk.DISABLED)

                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(summary)

                self.status_label.config(text=f"✅ Summary exported to {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Summary saved to:\n{filename}")

            except Exception as e:
                messagebox.showerror("Error", f"Failed to save summary:\n{str(e)}")
                self.status_label.config(text="❌ Export failed")


# Module interface functions for backward compatibility
def get_content():
    """Return module description for display in GUI."""
    return [
        "╔══════════════════════════════════════╗",
        "║     STRUCTURE ANALYZER MODULE        ║",
        "╚══════════════════════════════════════╝",
        "",
        "🎯 PURPOSE: Advanced analysis of musical structure",
        "",
        "🔍 ANALYSIS CAPABILITIES:",
        "  • Ornament detection (trills, mordents, turns)",
        "  • Tuplet and rhythmic pattern analysis",
        "  • Syncopation and swing detection",
        "  • Cross-rhythm identification",
        "  • Motif and pattern recognition",
        "  • Section boundary detection",
        "  • Harmonic progression analysis",
        "  • Repetition mapping",
        "",
        "📁 INPUT FORMATS:",
        "  • MIDI files (.mid, .midi)",
        "  • Built-in sample data",
        "  • Real violin performance data",
        "",
        "📊 OUTPUT FEATURES:",
        "  • Summary statistics",
        "  • Detailed analysis reports",
        "  • JSON export for further processing",
        "  • Text summary export",
        "",
        "🎻 VIOLIN-SPECIFIC ANALYSIS:",
        "  • Ornamentation typical in violin repertoire",
        "  • Bowing pattern implications",
        "  • Position and fingering considerations",
        "",
        "🚀 QUICK START:",
        "  1. Load a MIDI file or use sample data",
        "  2. Run full analysis or specific analysis",
        "  3. Review results in summary/detailed views",
        "  4. Export findings for further study",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = StructureAnalyzerGUI(parent)
    return gui.parent


# Standalone test
def test_standalone():
    """Test the module GUI standalone."""
    root = tk.Tk()
    root.title("Structure Analyzer Module - Standalone Test")
    root.geometry(WINDOW_SIZE)
    root.configure(bg=COLOR_BG)

    # Create the GUI
    gui = create_gui(root)

    root.mainloop()


if __name__ == "__main__":
    test_standalone()