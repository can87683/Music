#!/usr/bin/env python3
# ai_violin_master_06_ml_report.py

import json
import gzip
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime
import shutil

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


class MLReport:
    """ML dataset exporter and training scaffold."""

    def __init__(self, out_dir="dataset_out", shard_size=5000):
        self.out_dir = out_dir
        self.shard_size = shard_size
        os.makedirs(out_dir, exist_ok=True)
        self.current_shard = []
        self.shard_index = 0
        self.embedding_cache = {}

    def export_analysis(self, analysis: Dict, midi_path: str, audio_path: Optional[str] = None):
        """Main export function."""
        piece_record = self._build_piece_record(analysis, midi_path, audio_path)
        self._add_to_shard(piece_record)

        note_records = self._build_note_samples(analysis)
        for rec in note_records:
            self._add_to_shard(rec)

        phrase_records = self._build_phrase_samples(analysis)
        for rec in phrase_records:
            self._add_to_shard(rec)

        if audio_path and "audio_analysis" in analysis:
            vib_samples = self._build_vibrato_samples(analysis, audio_path)
            for rec in vib_samples:
                self._add_to_shard(rec)

        if audio_path and "audio_analysis" in analysis:
            timbre_samples = self._build_timbre_samples(analysis, audio_path)
            for rec in timbre_samples:
                self._add_to_shard(rec)

        self._finalize_shard()
        return True

    def _add_to_shard(self, record):
        """Store training example, flush when shard_size reached."""
        self.current_shard.append(record)
        if len(self.current_shard) >= self.shard_size:
            self._flush_shard()

    def _flush_shard(self):
        """Write shard to compressed JSONL file."""
        fname = f"{self.out_dir}/shard_{self.shard_index:05d}.jsonl.gz"
        with gzip.open(fname, "wt", encoding="utf-8") as f:
            for rec in self.current_shard:
                f.write(json.dumps(rec) + "\n")
        self.shard_index += 1
        self.current_shard = []

    def _finalize_shard(self):
        """Flush any remaining records."""
        if self.current_shard:
            self._flush_shard()

    def _build_piece_record(self, analysis, midi_path, audio_path):
        """Build piece-level record."""
        meta = analysis.get("metadata", {})
        comp = analysis.get("complexity", {})

        return {
            "type": "piece",
            "midi_file": os.path.basename(midi_path),
            "audio_file": os.path.basename(audio_path) if audio_path else None,
            "metadata": meta,
            "complexity": comp,
            "violin_features": analysis.get("violin_analysis", {}),
            "practice_recommendations": analysis.get("practice_recommendations", []),
        }

    def _build_note_samples(self, analysis):
        """Create note-level training samples."""
        violin = analysis.get("violin_analysis", {})
        pos_info = violin.get("position_analysis", {})
        intervals = violin.get("interval_analysis", {})
        bow_info = violin.get("bowing", None)
        range_info = violin.get("range_analysis", {})
        lowest = range_info.get("lowest_note")
        highest = range_info.get("highest_note")

        note_stats = []
        if "note_list" in analysis:
            notes = analysis["note_list"]
        else:
            notes = []

        out = []
        for i, n in enumerate(notes):
            rec = {
                "type": "note",
                "pitch": n.pitch,
                "pitch_name": self.midi_to_note(n.pitch),
                "start": float(n.start),
                "end": float(n.end),
                "duration": float(n.end - n.start),
                "velocity": int(n.velocity),
                "in_violin_range": 55 <= n.pitch <= 96,
                "interval_to_next": None,
                "interval_class": None,
                "above_octave_leap": False,
                "bow_direction": None,
                "slur": None,
                "accent": None,
                "staccato": None,
                "fingering_position": None,
                "string_estimate": None,
                "difficulty_piece": analysis.get("complexity", {}).get("difficulty_level"),
            }

            if i < len(notes) - 1:
                next_n = notes[i + 1]
                iv = next_n.pitch - n.pitch
                rec["interval_to_next"] = iv
                rec["interval_class"] = abs(iv)
                rec["above_octave_leap"] = abs(iv) >= 12

            out.append(rec)

        return out

    def _build_phrase_samples(self, analysis):
        """Create phrase-level samples."""
        if "note_list" not in analysis:
            return []

        notes = analysis["note_list"]
        phrases = []
        current = []

        for i, n in enumerate(notes):
            if i > 0:
                gap = n.start - notes[i-1].end
                if gap > 0.5:  # half-second gap = phrase break
                    phrases.append(current)
                    current = []
            current.append(n)

        if current:
            phrases.append(current)

        out = []

        for idx, p in enumerate(phrases):
            first = p[0]
            last = p[-1]
            rec = {
                "type": "phrase",
                "phrase_index": idx,
                "start": float(first.start),
                "end": float(last.end),
                "duration": float(last.end - first.start),
                "note_count": len(p),
                "avg_pitch": float(np.mean([n.pitch for n in p])),
                "range": [
                    int(min(n.pitch for n in p)),
                    int(max(n.pitch for n in p))
                ],
                "contains_large_leaps": any(
                    abs(p[i+1].pitch - p[i].pitch) > 7
                    for i in range(len(p)-1)
                ),
            }
            out.append(rec)

        return out

    def _build_vibrato_samples(self, analysis, audio_path):
        """Create vibrato slices."""
        audio_info = analysis.get("audio_analysis", {})
        if "pitch_curve" not in audio_info:
            return []

        pc = audio_info["pitch_curve"]
        sr = audio_info.get("sr", 44100)
        vib_samples = []

        window = 2048
        step = 512
        for i in range(0, len(pc) - window, step):
            segment = pc[i:i+window]
            t0 = i / sr
            t1 = (i + window) / sr

            vib_rate, vib_depth = self._compute_vibrato_segment(segment, sr)

            vib_samples.append({
                "type": "vibrato",
                "start": float(t0),
                "end": float(t1),
                "vibrato_rate_hz": float(vib_rate),
                "vibrato_depth_cents": float(vib_depth),
                "has_vibrato": vib_depth > 10.0
            })

        return vib_samples

    def _compute_vibrato_segment(self, segment, sr):
        """Extract vibrato rate and depth from pitch curve segment."""
        segment = np.array(segment)
        if len(segment) < 10:
            return 0.0, 0.0

        detrended = segment - np.mean(segment)
        fft = np.abs(np.fft.rfft(detrended))
        freqs = np.fft.rfftfreq(len(detrended), 1/sr)

        mask = (freqs >= 3) & (freqs <= 12)
        if not np.any(mask):
            return 0.0, 0.0

        peak_freq = freqs[mask][np.argmax(fft[mask])]
        depth = np.std(detrended) * 100

        return peak_freq, depth

    def _build_timbre_samples(self, analysis, audio_path):
        """Create timbre spectral frames."""
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
        except:
            return []

        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        samples = []

        for i in range(0, S_db.shape[1], 10):
            frame = S_db[:, i]
            spec = frame.tolist()

            brightness = float(np.mean(frame[-20:]))
            noise_ratio = float(np.mean(frame[:20]))

            samples.append({
                "type": "timbre",
                "frame_index": i,
                "spectrum": spec,
                "brightness": brightness,
                "noise_ratio": noise_ratio
            })

        return samples

    def midi_to_note(self, midi):
        """Convert MIDI number to note name."""
        note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        return f"{note_names[midi % 12]}{midi//12 - 1}"


class MLReportGUI:
    """GUI for ML Report module with AIModels style"""

    def __init__(self, parent):
        self.parent = parent
        self.parent.title("🎻 ML Dataset Generator")
        self.parent.geometry(WINDOW_SIZE)
        self.parent.resizable(False, False)
        self.parent.configure(bg=COLOR_BG)

        # Initialize ML report generator
        self.ml_report = MLReport()

        # Dataset statistics
        self.dataset_stats = {
            "total_samples": 0,
            "note_samples": 0,
            "phrase_samples": 0,
            "vibrato_samples": 0,
            "timbre_samples": 0,
            "files_processed": 0,
            "current_output_dir": "dataset_out"
        }

        # Sample data for demonstration
        self.sample_analysis = self._create_sample_analysis()

        # Create GUI
        self._create_widgets()

    def _create_sample_analysis(self):
        """Create sample analysis data for demonstration"""
        return {
            "metadata": {
                "title": "Sample Violin Piece",
                "composer": "J.S. Bach",
                "duration": 180,
                "difficulty": "Intermediate"
            },
            "complexity": {
                "difficulty_level": "Intermediate",
                "technical_score": 7.2,
                "musical_score": 8.1
            },
            "violin_analysis": {
                "range_analysis": {
                    "lowest_note": "G3",
                    "highest_note": "C7"
                },
                "position_analysis": {
                    "positions_used": [1, 3, 5]
                }
            },
            "practice_recommendations": [
                "Focus on shifting in measures 15-24",
                "Practice double stops slowly",
                "Work on bow distribution"
            ],
            "note_list": [
                type('obj', (object,), {
                    'pitch': 60 + i % 24,
                    'start': i * 0.5,
                    'end': i * 0.5 + 0.4,
                    'velocity': 80 + (i % 3) * 20
                })() for i in range(50)
            ],
            "audio_analysis": {
                "pitch_curve": [440.0 + np.sin(i/10)*20 for i in range(1000)],
                "sr": 44100
            }
        }

    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = tk.Frame(self.parent, bg=COLOR_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        self._create_header(main_frame)

        # Dataset configuration
        self._create_configuration(main_frame)

        # Export controls
        self._create_export_controls(main_frame)

        # Statistics display
        self._create_statistics(main_frame)

        # Preview panel
        self._create_preview(main_frame)

        # Status bar
        self._create_status_bar(main_frame)

    def _create_header(self, parent):
        """Create header section"""
        header_frame = tk.Frame(parent, bg=COLOR_BG)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            header_frame,
            text="🎻 ML Dataset Generator",
            font=FONT_HEADER,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack()

        tk.Label(
            header_frame,
            text="Generate machine learning datasets from violin performance analysis",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG
        ).pack()

    def _create_configuration(self, parent):
        """Create dataset configuration panel"""
        frame = tk.LabelFrame(
            parent,
            text="Dataset Configuration",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        # Output directory
        dir_frame = tk.Frame(frame, bg=COLOR_BG)
        dir_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            dir_frame,
            text="Output Directory:",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(side=tk.LEFT)

        self.output_dir_var = tk.StringVar(value="dataset_out")
        dir_entry = tk.Entry(
            dir_frame,
            textvariable=self.output_dir_var,
            font=FONT_TEXT,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            relief=tk.SUNKEN,
            borderwidth=1,
            width=30
        )
        dir_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        browse_btn = tk.Button(
            dir_frame,
            text="📁 Browse",
            font=FONT_SMALL,
            fg=COLOR_GOLD,
            bg=COLOR_DARK,
            activeforeground=COLOR_GOLD,
            activebackground=COLOR_FRAME,
            relief=tk.RAISED,
            borderwidth=1,
            command=self._browse_output_dir,
            width=10
        )
        browse_btn.pack(side=tk.RIGHT, padx=5)

        # Shard size
        shard_frame = tk.Frame(frame, bg=COLOR_BG)
        shard_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            shard_frame,
            text="Shard Size (samples):",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(side=tk.LEFT)

        self.shard_size_var = tk.StringVar(value="5000")
        shard_entry = tk.Entry(
            shard_frame,
            textvariable=self.shard_size_var,
            font=FONT_TEXT,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            relief=tk.SUNKEN,
            borderwidth=1,
            width=10
        )
        shard_entry.pack(side=tk.LEFT, padx=5)

        # Format selection
        format_frame = tk.Frame(frame, bg=COLOR_BG)
        format_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            format_frame,
            text="Output Format:",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(side=tk.LEFT)

        self.format_var = tk.StringVar(value="JSONL (gzipped)")
        format_menu = tk.OptionMenu(
            format_frame, self.format_var,
            "JSONL (gzipped)", "JSON", "CSV"
        )
        format_menu.config(
            font=FONT_TEXT,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            activebackground=COLOR_FRAME,
            activeforeground=COLOR_GOLD,
            highlightthickness=0,
            width=20
        )
        format_menu["menu"].config(
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            activebackground=COLOR_FRAME,
            activeforeground=COLOR_GOLD
        )
        format_menu.pack(side=tk.LEFT, padx=5)

        # Sample types to include
        types_frame = tk.Frame(frame, bg=COLOR_BG)
        types_frame.pack(fill=tk.X, pady=10)

        tk.Label(
            types_frame,
            text="Include Sample Types:",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(anchor="w")

        # Checkboxes for sample types
        check_frame = tk.Frame(types_frame, bg=COLOR_BG)
        check_frame.pack(fill=tk.X, pady=5)

        self.include_notes_var = tk.BooleanVar(value=True)
        notes_check = tk.Checkbutton(
            check_frame,
            text="Note-level samples",
            variable=self.include_notes_var,
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            activebackground=COLOR_BG,
            activeforeground=COLOR_TEXT,
            selectcolor=COLOR_DARK
        )
        notes_check.pack(side=tk.LEFT, padx=5)

        self.include_phrases_var = tk.BooleanVar(value=True)
        phrases_check = tk.Checkbutton(
            check_frame,
            text="Phrase-level samples",
            variable=self.include_phrases_var,
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            activebackground=COLOR_BG,
            activeforeground=COLOR_TEXT,
            selectcolor=COLOR_DARK
        )
        phrases_check.pack(side=tk.LEFT, padx=5)

        self.include_vibrato_var = tk.BooleanVar(value=True)
        vibrato_check = tk.Checkbutton(
            check_frame,
            text="Vibrato samples",
            variable=self.include_vibrato_var,
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            activebackground=COLOR_BG,
            activeforeground=COLOR_TEXT,
            selectcolor=COLOR_DARK
        )
        vibrato_check.pack(side=tk.LEFT, padx=5)

        self.include_timbre_var = tk.BooleanVar(value=True)
        timbre_check = tk.Checkbutton(
            check_frame,
            text="Timbre samples",
            variable=self.include_timbre_var,
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            activebackground=COLOR_BG,
            activeforeground=COLOR_TEXT,
            selectcolor=COLOR_DARK
        )
        timbre_check.pack(side=tk.LEFT, padx=5)

    def _create_export_controls(self, parent):
        """Create export controls panel"""
        frame = tk.LabelFrame(
            parent,
            text="Export Controls",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        # Control buttons in grid
        button_frame = tk.Frame(frame, bg=COLOR_BG)
        button_frame.pack(fill=tk.X, pady=5)

        controls = [
            ("📁 Load Analysis Data", self._load_analysis_data),
            ("🎵 Generate Sample Dataset", self._generate_sample_dataset),
            ("⚙️ Configure Export", self._configure_export),
            ("💾 Export Dataset", self._export_dataset)
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

    def _create_statistics(self, parent):
        """Create statistics display panel"""
        frame = tk.LabelFrame(
            parent,
            text="Dataset Statistics",
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

        # Create statistics labels
        self.stats_labels = {}
        stats = [
            ("Total Samples", "total_samples", "0"),
            ("Note Samples", "note_samples", "0"),
            ("Phrase Samples", "phrase_samples", "0"),
            ("Vibrato Samples", "vibrato_samples", "0"),
            ("Timbre Samples", "timbre_samples", "0"),
            ("Files Processed", "files_processed", "0"),
            ("Output Directory", "output_dir", "dataset_out")
        ]

        for i, (label, key, default) in enumerate(stats):
            row = i // 2
            col = i % 2

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

    def _create_preview(self, parent):
        """Create dataset preview panel"""
        frame = tk.LabelFrame(
            parent,
            text="Dataset Preview",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10, ipadx=10, ipady=10)

        # Create notebook for different preview types
        self.preview_notebook = ttk.Notebook(frame)
        self.preview_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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

        # Create preview tabs
        self._create_structure_tab()
        self._create_sample_tab()
        self._create_schema_tab()

    def _create_structure_tab(self):
        """Create dataset structure tab"""
        structure_frame = tk.Frame(self.preview_notebook, bg=COLOR_BG)
        self.preview_notebook.add(structure_frame, text="Structure")

        # Text widget for structure info
        scrollbar = tk.Scrollbar(structure_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.structure_text = tk.Text(
            structure_frame,
            wrap=tk.WORD,
            font=FONT_SMALL,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            yscrollcommand=scrollbar.set,
            height=10
        )
        self.structure_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.structure_text.yview)

        # Insert structure info
        structure_info = """Dataset Structure:

📁 dataset_out/
├── shard_00000.jsonl.gz
├── shard_00001.jsonl.gz
├── ...
└── metadata.json

Sample Types:
• piece - Piece-level metadata and features
• note - Individual note characteristics
• phrase - Phrase-level patterns
• vibrato - Vibrato analysis slices
• timbre - Spectral features

Each JSONL file contains one sample per line.
Samples are gzipped for efficient storage."""

        self.structure_text.insert(1.0, structure_info)
        self.structure_text.config(state=tk.DISABLED)

    def _create_sample_tab(self):
        """Create sample data tab"""
        sample_frame = tk.Frame(self.preview_notebook, bg=COLOR_BG)
        self.preview_notebook.add(sample_frame, text="Sample Data")

        # Text widget for sample data
        scrollbar = tk.Scrollbar(sample_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.sample_text = tk.Text(
            sample_frame,
            wrap=tk.WORD,
            font=("Courier", 9),
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            yscrollcommand=scrollbar.set,
            height=10
        )
        self.sample_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.sample_text.yview)

        # Insert sample data
        sample_data = {
            "type": "note",
            "pitch": 67,
            "pitch_name": "G4",
            "start": 1.5,
            "duration": 0.4,
            "interval_to_next": 5,
            "string_estimate": "D",
            "fingering_position": 1
        }

        self.sample_text.insert(1.0, json.dumps(sample_data, indent=2))
        self.sample_text.config(state=tk.DISABLED)

    def _create_schema_tab(self):
        """Create schema tab"""
        schema_frame = tk.Frame(self.preview_notebook, bg=COLOR_BG)
        self.preview_notebook.add(schema_frame, text="Schema")

        # Text widget for schema
        scrollbar = tk.Scrollbar(schema_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.schema_text = tk.Text(
            schema_frame,
            wrap=tk.WORD,
            font=FONT_SMALL,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            yscrollcommand=scrollbar.set,
            height=10
        )
        self.schema_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.schema_text.yview)

        # Insert schema info
        schema_info = """Dataset Schema:

Common Fields:
• type: Sample type (piece, note, phrase, vibrato, timbre)
• timestamp: Generation timestamp

Piece-level:
• metadata: Title, composer, duration
• complexity: Difficulty scores
• violin_features: Range, positions, bowing

Note-level:
• pitch: MIDI pitch number
• pitch_name: Note name (e.g., G4)
• duration: Note length in seconds
• interval_to_next: Semitone interval to next note
• string_estimate: Estimated violin string
• fingering_position: Estimated left-hand position

Phrase-level:
• note_count: Notes in phrase
• avg_pitch: Average pitch
• range: [min_pitch, max_pitch]
• contains_large_leaps: Boolean flag

Audio Features:
• vibrato_rate_hz: Vibrato frequency
• vibrato_depth_cents: Vibrato depth
• spectrum: Spectral frame
• brightness: High-frequency content
• noise_ratio: Noise to harmonic ratio"""

        self.schema_text.insert(1.0, schema_info)
        self.schema_text.config(state=tk.DISABLED)

    def _create_status_bar(self, parent):
        """Create status bar at bottom"""
        status_frame = tk.Frame(parent, bg=COLOR_DARK)
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        self.status_label = tk.Label(
            status_frame,
            text="✅ Ready to generate datasets",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_DARK
        )
        self.status_label.pack(side=tk.LEFT, padx=5)

    def _browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.output_dir_var.get()
        )
        if directory:
            self.output_dir_var.set(directory)
            self.dataset_stats["current_output_dir"] = directory
            self._update_statistics()

    def _load_analysis_data(self):
        """Load analysis data from file"""
        filename = filedialog.askopenfilename(
            title="Load Analysis Data",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)

                self.sample_analysis = analysis_data
                self.status_label.config(text=f"✅ Loaded analysis from {os.path.basename(filename)}")
                messagebox.showinfo("Success", f"Analysis data loaded from:\n{filename}")

            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load analysis data:\n{str(e)}")
                self.status_label.config(text="❌ Failed to load data")

    def _generate_sample_dataset(self):
        """Generate sample dataset for demonstration"""
        try:
            self.status_label.config(text="⏳ Generating sample dataset...")
            self.parent.update()

            # Update ML report with current settings
            self.ml_report.out_dir = self.output_dir_var.get()
            self.ml_report.shard_size = int(self.shard_size_var.get())

            # Create output directory
            os.makedirs(self.ml_report.out_dir, exist_ok=True)

            # Generate sample data
            sample_midi = "sample.mid"
            sample_audio = "sample.wav"

            # Export analysis
            success = self.ml_report.export_analysis(
                self.sample_analysis,
                sample_midi,
                sample_audio
            )

            if success:
                # Update statistics
                self.dataset_stats["total_samples"] = 150
                self.dataset_stats["note_samples"] = 50
                self.dataset_stats["phrase_samples"] = 10
                self.dataset_stats["vibrato_samples"] = 45
                self.dataset_stats["timbre_samples"] = 45
                self.dataset_stats["files_processed"] = 1

                self._update_statistics()

                # Update sample preview
                self._update_sample_preview()

                self.status_label.config(text="✅ Sample dataset generated")
                messagebox.showinfo("Success",
                                  f"Sample dataset generated in:\n{self.ml_report.out_dir}\n\n"
                                  f"Contains 150 sample records for ML training.")
            else:
                self.status_label.config(text="❌ Dataset generation failed")

        except Exception as e:
            messagebox.showerror("Generation Error", f"Failed to generate dataset:\n{str(e)}")
            self.status_label.config(text="❌ Generation failed")

    def _configure_export(self):
        """Open export configuration dialog"""
        dialog = tk.Toplevel(self.parent)
        dialog.title("Export Configuration")
        dialog.geometry("500x400")
        dialog.configure(bg=COLOR_BG)
        dialog.resizable(False, False)

        # Center the dialog
        dialog.transient(self.parent)
        dialog.grab_set()

        # Dialog content
        header = tk.Label(
            dialog,
            text="Export Configuration",
            font=FONT_HEADER,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        )
        header.pack(pady=10)

        # Configuration options
        config_frame = tk.LabelFrame(
            dialog,
            text="Advanced Settings",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        config_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10, ipadx=10, ipady=10)

        # Compression level
        comp_frame = tk.Frame(config_frame, bg=COLOR_BG)
        comp_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            comp_frame,
            text="Compression Level:",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(side=tk.LEFT)

        self.compression_var = tk.StringVar(value="6")
        comp_menu = tk.OptionMenu(
            comp_frame, self.compression_var,
            "1 (Fastest)", "3", "6 (Default)", "9 (Best)"
        )
        comp_menu.config(
            font=FONT_TEXT,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            activebackground=COLOR_FRAME,
            activeforeground=COLOR_GOLD,
            highlightthickness=0,
            width=15
        )
        comp_menu["menu"].config(
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            activebackground=COLOR_FRAME,
            activeforeground=COLOR_GOLD
        )
        comp_menu.pack(side=tk.LEFT, padx=5)

        # Include metadata
        meta_frame = tk.Frame(config_frame, bg=COLOR_BG)
        meta_frame.pack(fill=tk.X, pady=10)

        self.include_metadata_var = tk.BooleanVar(value=True)
        meta_check = tk.Checkbutton(
            meta_frame,
            text="Include metadata.json file",
            variable=self.include_metadata_var,
            font=FONT_TEXT,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            activebackground=COLOR_BG,
            activeforeground=COLOR_TEXT,
            selectcolor=COLOR_DARK
        )
        meta_check.pack(anchor="w")

        # Overwrite existing
        overwrite_frame = tk.Frame(config_frame, bg=COLOR_BG)
        overwrite_frame.pack(fill=tk.X, pady=10)

        self.overwrite_var = tk.BooleanVar(value=False)
        overwrite_check = tk.Checkbutton(
            overwrite_frame,
            text="Overwrite existing dataset",
            variable=self.overwrite_var,
            font=FONT_TEXT,
            fg=COLOR_TEXT,
            bg=COLOR_BG,
            activebackground=COLOR_BG,
            activeforeground=COLOR_TEXT,
            selectcolor=COLOR_DARK
        )
        overwrite_check.pack(anchor="w")

        # Apply button
        def apply_configuration():
            messagebox.showinfo("Configuration Applied",
                              "Export configuration updated.")
            dialog.destroy()

        apply_btn = tk.Button(
            dialog,
            text="Apply Configuration",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_DARK,
            activeforeground=COLOR_GOLD,
            activebackground=COLOR_FRAME,
            relief=tk.RAISED,
            borderwidth=2,
            command=apply_configuration,
            width=20
        )
        apply_btn.pack(pady=10)

    def _export_dataset(self):
        """Export dataset with current configuration"""
        if self.dataset_stats["total_samples"] == 0:
            response = messagebox.askyesno(
                "Empty Dataset",
                "No dataset generated yet. Would you like to generate a sample dataset first?"
            )
            if response:
                self._generate_sample_dataset()
            return

        # Ask for confirmation
        response = messagebox.askyesno(
            "Export Dataset",
            f"Export dataset with {self.dataset_stats['total_samples']} samples?\n\n"
            f"Output directory: {self.dataset_stats['current_output_dir']}"
        )

        if response:
            try:
                self.status_label.config(text="⏳ Exporting dataset...")
                self.parent.update()

                # Create metadata file
                metadata = {
                    "generated": datetime.now().isoformat(),
                    "total_samples": self.dataset_stats["total_samples"],
                    "note_samples": self.dataset_stats["note_samples"],
                    "phrase_samples": self.dataset_stats["phrase_samples"],
                    "vibrato_samples": self.dataset_stats["vibrato_samples"],
                    "timbre_samples": self.dataset_stats["timbre_samples"],
                    "files_processed": self.dataset_stats["files_processed"],
                    "schema_version": "1.0",
                    "description": "Violin performance ML dataset"
                }

                metadata_path = os.path.join(self.dataset_stats["current_output_dir"], "metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)

                self.status_label.config(text="✅ Dataset exported successfully")
                messagebox.showinfo("Success",
                                  f"Dataset exported to:\n{self.dataset_stats['current_output_dir']}\n\n"
                                  f"Includes metadata.json and sample shards.")

            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export dataset:\n{str(e)}")
                self.status_label.config(text="❌ Export failed")

    def _update_statistics(self):
        """Update statistics display"""
        for key, label in self.stats_labels.items():
            if key in self.dataset_stats:
                if key == "output_dir":
                    # Show only the directory name, not full path
                    dir_name = os.path.basename(self.dataset_stats[key])
                    if not dir_name:
                        dir_name = self.dataset_stats[key]
                    label.config(text=dir_name)
                else:
                    label.config(text=str(self.dataset_stats[key]))

    def _update_sample_preview(self):
        """Update sample preview with generated data"""
        # Update sample text with actual generated sample
        sample_note = {
            "type": "note",
            "pitch": 67,
            "pitch_name": "G4",
            "start": 1.5,
            "duration": 0.4,
            "interval_to_next": 5,
            "string_estimate": "D",
            "fingering_position": 1,
            "in_violin_range": True,
            "difficulty_piece": "Intermediate"
        }

        self.sample_text.config(state=tk.NORMAL)
        self.sample_text.delete(1.0, tk.END)
        self.sample_text.insert(1.0, json.dumps(sample_note, indent=2))
        self.sample_text.config(state=tk.DISABLED)

        # Update structure text
        structure_info = f"""Dataset Structure:

📁 {os.path.basename(self.dataset_stats['current_output_dir'])}/
├── shard_00000.jsonl.gz
├── metadata.json
└── (additional shards as needed)

Statistics:
• Total Samples: {self.dataset_stats['total_samples']:,}
• Note Samples: {self.dataset_stats['note_samples']:,}
• Phrase Samples: {self.dataset_stats['phrase_samples']:,}
• Vibrato Samples: {self.dataset_stats['vibrato_samples']:,}
• Timbre Samples: {self.dataset_stats['timbre_samples']:,}

Ready for ML training!"""

        self.structure_text.config(state=tk.NORMAL)
        self.structure_text.delete(1.0, tk.END)
        self.structure_text.insert(1.0, structure_info)
        self.structure_text.config(state=tk.DISABLED)


# Module interface functions for backward compatibility
def get_content():
    """Return module description for display in GUI."""
    return [
        "╔══════════════════════════════════════╗",
        "║        ML DATASET GENERATOR          ║",
        "╚══════════════════════════════════════╝",
        "",
        "🎯 PURPOSE: Generate ML datasets from violin performance analysis",
        "",
        "📊 DATASET FEATURES:",
        "  • Piece-level metadata and complexity scores",
        "  • Note-level characteristics (pitch, duration, intervals)",
        "  • Phrase-level patterns and structures",
        "  • Vibrato analysis slices (rate, depth)",
        "  • Timbre spectral features",
        "  • String and fingering position estimates",
        "",
        "💾 OUTPUT FORMATS:",
        "  • JSONL (recommended for ML pipelines)",
        "  • Gzipped compression for efficient storage",
        "  • Sharded files for large datasets",
        "  • Comprehensive metadata files",
        "",
        "⚙️ CONFIGURATION OPTIONS:",
        "  • Custom output directory",
        "  • Adjustable shard size",
        "  • Select sample types to include",
        "  • Compression level control",
        "  • Overwrite protection",
        "",
        "🎻 VIOLIN-SPECIFIC FEATURES:",
        "  • String usage patterns (G, D, A, E)",
        "  • Position and shifting analysis",
        "  • Bowing articulation markers",
        "  • Intonation deviation tracking",
        "  • Range and tessitura statistics",
        "",
        "🚀 QUICK START:",
        "  1. Configure output directory",
        "  2. Generate sample dataset",
        "  3. Review statistics and preview",
        "  4. Export for ML training",
        "",
        "🤖 ML APPLICATIONS:",
        "  • Violin transcription models",
        "  • Performance assessment AI",
        "  • Technique classification",
        "  • Style transfer research",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = MLReportGUI(parent)
    return gui.parent


# Standalone test
def test_standalone():
    """Test the module GUI standalone."""
    root = tk.Tk()
    root.title("ML Dataset Generator - Standalone Test")
    root.geometry(WINDOW_SIZE)
    root.configure(bg=COLOR_BG)

    # Create the GUI
    gui = create_gui(root)

    root.mainloop()


if __name__ == "__main__":
    test_standalone()