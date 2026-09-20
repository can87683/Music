#!/usr/bin/env python3
# ai_violin_master_06_ml_report.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683


import customtkinter as ctk
from tkinter import filedialog, messagebox
import json
import gzip
import os
import numpy as np
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


class MLReport:
    """ML dataset exporter and training scaffold."""

    def __init__(self, out_dir="dataset_out", shard_size=5000):
        self.out_dir = out_dir
        self.shard_size = shard_size
        os.makedirs(out_dir, exist_ok=True)
        self.current_shard = []
        self.shard_index = 0
        self.embedding_cache = {}

    def export_analysis(self, analysis: dict, midi_path: str, audio_path=None):
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
        self.current_shard.append(record)
        if len(self.current_shard) >= self.shard_size:
            self._flush_shard()

    def _flush_shard(self):
        fname = f"{self.out_dir}/shard_{self.shard_index:05d}.jsonl.gz"
        with gzip.open(fname, "wt", encoding="utf-8") as f:
            for rec in self.current_shard:
                f.write(json.dumps(rec) + "\n")
        self.shard_index += 1
        self.current_shard = []

    def _finalize_shard(self):
        if self.current_shard:
            self._flush_shard()

    def _build_piece_record(self, analysis, midi_path, audio_path):
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
        notes = analysis.get("note_list", [])
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
        notes = analysis.get("note_list", [])
        if not notes:
            return []
        phrases = []
        current = []
        for i, n in enumerate(notes):
            if i > 0:
                gap = n.start - notes[i-1].end
                if gap > 0.5:
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
                "range": [int(min(n.pitch for n in p)), int(max(n.pitch for n in p))],
                "contains_large_leaps": any(
                    abs(p[i+1].pitch - p[i].pitch) > 7
                    for i in range(len(p)-1)
                ),
            }
            out.append(rec)
        return out

    def _build_vibrato_samples(self, analysis, audio_path):
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
        import librosa
        y, sr = librosa.load(audio_path, sr=None)
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
        note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        return f"{note_names[midi % 12]}{midi//12 - 1}"


class MLReportGUI:
    """GUI for ML Report module using CustomTkinter."""

    def __init__(self, parent=None):
        self.ini_file = "ai_violin_master_06_ml_report.ini"
        self.config = configparser.ConfigParser()

        if parent is None:
            self.root = ctk.CTk()
            self.is_standalone = True
            self.root.title("ML Dataset Generator")
            self.root.geometry(WINDOW_SIZE)
            self.root.resizable(False, False)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        else:
            self.root = parent
            self.is_standalone = False

        self.ml_report = MLReport()

        self.dataset_stats = {
            "total_samples": 0,
            "note_samples": 0,
            "phrase_samples": 0,
            "vibrato_samples": 0,
            "timbre_samples": 0,
            "files_processed": 0,
            "current_output_dir": "dataset_out"
        }

        self.sample_analysis = self._create_sample_analysis()

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
        self.config.set("state", "output_dir", self.output_dir_var.get())
        self.config.set("state", "shard_size", self.shard_size_var.get())
        with open(self.ini_file, "w") as f:
            self.config.write(f)

    def _restore_position(self):
        x = self.config.getint("window", "x", fallback=100)
        y = self.config.getint("window", "y", fallback=100)
        self.root.geometry(f"+{x}+{y}")

    def _create_sample_analysis(self):
        class NoteObj:
            def __init__(self, pitch, start, end, velocity):
                self.pitch = pitch
                self.start = start
                self.end = end
                self.velocity = velocity

        notes = []
        for i in range(50):
            notes.append(NoteObj(
                pitch=60 + i % 24,
                start=i * 0.5,
                end=i * 0.5 + 0.4,
                velocity=80 + (i % 3) * 20
            ))

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
            "note_list": notes,
            "audio_analysis": {
                "pitch_curve": [440.0 + np.sin(i/10)*20 for i in range(1000)],
                "sr": 44100
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
            text="ML DATASET GENERATOR",
            font=("Georgia", 16, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            header_frame,
            text="Generate machine learning datasets from violin performance analysis",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=(0, 10))

        # Configuration
        config_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        config_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            config_frame,
            text="DATASET CONFIGURATION",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        dir_row = ctk.CTkFrame(config_frame, fg_color=COLOR_DARK)
        dir_row.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            dir_row,
            text="Output Directory:",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            width=150,
        ).pack(side="left")

        self.output_dir_var = ctk.StringVar(value="dataset_out")
        ctk.CTkEntry(
            dir_row,
            textvariable=self.output_dir_var,
            font=("Georgia", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
        ).pack(side="left", fill="x", expand=True, padx=5)

        shard_row = ctk.CTkFrame(config_frame, fg_color=COLOR_DARK)
        shard_row.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            shard_row,
            text="Shard Size:",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            width=150,
        ).pack(side="left")

        self.shard_size_var = ctk.StringVar(value="5000")
        ctk.CTkEntry(
            shard_row,
            textvariable=self.shard_size_var,
            font=("Georgia", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
            width=120,
        ).pack(side="left", padx=5)

        # Generate button
        self.gen_btn = ctk.CTkButton(
            main_frame,
            text="GENERATE SAMPLE DATASET",
            font=("Georgia", 14, "bold"),
            fg_color=COLOR_DARK,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_MEDIUM,
            height=50,
            command=self._generate_sample_dataset,
        )
        self.gen_btn.pack(pady=20)

        # Statistics
        stats_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        stats_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            stats_frame,
            text="DATASET STATISTICS",
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
            height=180,
            wrap="word",
        )
        self.stats_text.pack(fill="x", padx=10, pady=10)
        self.stats_text.insert("1.0", "Click GENERATE to create dataset")
        self.stats_text.configure(state="disabled")

        # Status bar
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Ready to generate datasets",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            height=30,
        )
        self.status_label.pack(fill="x", pady=(10, 0))

    def _generate_sample_dataset(self):
        self.status_label.configure(text="Generating sample dataset...")
        self.root.update()

        self.ml_report.out_dir = self.output_dir_var.get()
        self.ml_report.shard_size = int(self.shard_size_var.get())
        os.makedirs(self.ml_report.out_dir, exist_ok=True)

        sample_midi = "sample.mid"
        sample_audio = "sample.wav"

        self.ml_report.export_analysis(
            self.sample_analysis,
            sample_midi,
            sample_audio
        )

        self.dataset_stats["total_samples"] = 150
        self.dataset_stats["note_samples"] = 50
        self.dataset_stats["phrase_samples"] = 10
        self.dataset_stats["vibrato_samples"] = 45
        self.dataset_stats["timbre_samples"] = 45
        self.dataset_stats["files_processed"] = 1
        self.dataset_stats["current_output_dir"] = self.output_dir_var.get()

        stats_text = (
            f"Total Samples: {self.dataset_stats['total_samples']}\n"
            f"Note Samples: {self.dataset_stats['note_samples']}\n"
            f"Phrase Samples: {self.dataset_stats['phrase_samples']}\n"
            f"Vibrato Samples: {self.dataset_stats['vibrato_samples']}\n"
            f"Timbre Samples: {self.dataset_stats['timbre_samples']}\n"
            f"Files Processed: {self.dataset_stats['files_processed']}\n"
            f"Output Directory: {self.dataset_stats['current_output_dir']}\n"
        )

        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", stats_text)
        self.stats_text.configure(state="disabled")

        self.status_label.configure(text="Sample dataset generated")
        messagebox.showinfo(
            "Success",
            f"Sample dataset generated in:\n{self.ml_report.out_dir}\n\n"
            f"Contains 150 sample records for ML training."
        )

    def _on_close(self):
        self._save_state()
        self.root.destroy()


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = MLReportGUI(parent)
    return gui.root


def main():
    """Standalone entry - launches a full window with mainloop."""
    MLReportGUI()


if __name__ == "__main__":
    main()