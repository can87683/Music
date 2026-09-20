#!/usr/bin/env python3
# ai_violin_master_01_audio_analyzer.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683

import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
import os
import threading
import configparser

# Color constants
COLOR_BG = "#5A381E"
COLOR_DARK = "#3B2413"
COLOR_MEDIUM = "#4B2E18"
COLOR_LIGHT = "#6B4E29"
COLOR_GOLD = "#FFD770"
COLOR_TEXT = "#FFF2CF"
COLOR_TEXT_SECONDARY = "#E6D8B5"
COLOR_ACCENT = "#00BFFF"
COLOR_RED = "#FF6B6B"
COLOR_GREEN = "#4CAF50"

WINDOW_SIZE = "640x1050"

class AudioAnalyzer:
    """Pure analysis class - no GUI dependencies."""

    def __init__(self, accel=None, yin_pitch=None, kalman=None):
        self.accel = accel
        self.yin_pitch = yin_pitch
        self.kalman = kalman
        self.deps_ok = True
        self.missing_deps = []
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        required_deps = [
            ("librosa", "librosa"),
            ("pretty_midi", "pretty_midi"),
            ("numpy", "numpy"),
        ]
        for import_name, package_name in required_deps:
            try:
                __import__(import_name)
            except ImportError:
                self.missing_deps.append(package_name)
                self.deps_ok = False

    def analyze_audio(self, audio_path: str, midi_data=None) -> dict:
        if not self.deps_ok:
            return {
                "error": f"Missing dependencies: {', '.join(self.missing_deps)}",
                "suggestion": "Install with: pip install " + " ".join(self.missing_deps),
            }

        import librosa
        y, sr = librosa.load(audio_path, sr=None)

        f0_contour = self._get_pitch_contour(y, sr)
        f0_smooth = self._smooth_pitch(f0_contour)
        alignment = self._align_audio_midi(y, sr, midi_data)
        vibrato_data = self._analyze_vibrato(f0_smooth, sr)
        timbre_data = self._timbre_profile(y, sr)
        intonation_curve = self._intonation_vs_midi(f0_smooth, alignment, midi_data)

        return {
            "pitch_contour_smoothed": f0_smooth.tolist() if hasattr(f0_smooth, "tolist") else list(f0_smooth),
            "vibrato": vibrato_data,
            "timbre": timbre_data,
            "alignment": alignment,
            "intonation_curve": intonation_curve,
            "sample_rate": sr,
            "audio_length_seconds": len(y) / sr if sr > 0 else 0,
        }

    def _get_pitch_contour(self, y: np.ndarray, sr: int) -> np.ndarray:
        if self.yin_pitch:
            return self.yin_pitch.yin_pitch_track(y, sr, hop=512, win=2048)
        import librosa
        return librosa.yin(y, fmin=65, fmax=2093, sr=sr, hop_length=512)

    def _smooth_pitch(self, f0_contour: np.ndarray) -> np.ndarray:
        if self.kalman:
            return self.kalman.smooth_array(f0_contour)
        window = 5
        if len(f0_contour) > window:
            return np.convolve(f0_contour, np.ones(window) / window, mode="valid")
        return f0_contour

    def _align_audio_midi(self, y: np.ndarray, sr: int, midi_data) -> dict:
        import librosa
        if midi_data is None:
            return {
                "map": [],
                "average_timing_offset": 0.0,
                "timing_label": "No MIDI provided",
                "note_count_aligned": 0,
                "status": "No MIDI",
            }
        audio_onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        midi_onsets = []
        for inst in midi_data.instruments:
            for n in inst.notes:
                midi_onsets.append(n.start)
        midi_onsets = np.array(sorted(midi_onsets))
        if len(audio_onsets) == 0 or len(midi_onsets) == 0:
            return {
                "map": [],
                "average_timing_offset": 0.0,
                "timing_label": "No onsets detected",
                "note_count_aligned": 0,
                "status": "No onsets",
            }
        D, wp = librosa.sequence.dtw(
            np.expand_dims(audio_onsets, 0),
            np.expand_dims(midi_onsets, 0),
        )
        mapping = []
        for a_idx, m_idx in wp:
            if m_idx < len(midi_onsets) and a_idx < len(audio_onsets):
                mapping.append((float(audio_onsets[a_idx]), float(midi_onsets[m_idx])))
        diffs = [a - m for a, m in mapping]
        avg_drift = float(np.mean(diffs)) if diffs else 0.0
        if avg_drift < -0.05:
            label = "dragging"
        elif avg_drift > 0.05:
            label = "rushing"
        else:
            label = "on time"
        return {
            "map": mapping,
            "average_timing_offset": avg_drift,
            "timing_label": label,
            "note_count_aligned": len(mapping),
            "status": "Success",
        }

    def _analyze_vibrato(self, f0: np.ndarray, sr: int) -> dict:
        f0 = np.array([x for x in f0 if x > 0])
        if len(f0) < 20:
            return {
                "rate_hz": 0.0,
                "depth_cents": 0.0,
                "stability": 0.0,
                "segments": 0,
                "quality": "Insufficient data",
                "status": "Insufficient data",
            }
        f0_diff = np.diff(f0)
        hp = f0_diff - np.mean(f0_diff)
        spectrum = np.fft.rfft(hp)
        freqs = np.fft.rfftfreq(len(hp), d=1.0 / (sr / 512))
        peak_idx = np.argmax(np.abs(spectrum[1:])) + 1
        vibrato_rate = freqs[peak_idx] if peak_idx < len(freqs) else 0.0
        ref = np.median(f0)
        cents = 1200 * np.log2(f0 / ref) if ref > 0 else np.zeros_like(f0)
        depth = np.std(cents) * 2 if len(cents) > 0 else 0.0
        stability = 1.0 / (1.0 + np.std(hp)) if np.std(hp) > 0 else 0.0
        quality = "Good"
        if depth < 10:
            quality = "Subtle"
        elif depth > 100:
            quality = "Exaggerated"
        elif vibrato_rate < 4:
            quality = "Slow"
        elif vibrato_rate > 8:
            quality = "Fast"
        return {
            "rate_hz": float(vibrato_rate),
            "depth_cents": float(depth),
            "stability": float(stability),
            "segments": len(f0),
            "quality": quality,
            "status": "Success",
        }

    def _intonation_vs_midi(self, f0_smooth, alignment, midi_data):
        if not midi_data or "map" not in alignment or len(alignment["map"]) == 0:
            return []
        expected_pitches = []
        for _, midi_t in alignment["map"]:
            expected_pitches.append(self._midi_pitch_at_time(midi_data, midi_t))
        intonation_curve = []
        map_items = list(zip(alignment["map"], f0_smooth[:len(expected_pitches)], expected_pitches))
        for (audio_t, _), f0, midi_p in map_items:
            if f0 <= 0 or midi_p <= 0:
                continue
            cents_off = 1200 * np.log2(f0 / midi_p)
            intonation_curve.append((float(audio_t), float(cents_off)))
        return intonation_curve

    def _midi_pitch_at_time(self, midi_data, time: float) -> float:
        import librosa
        for inst in midi_data.instruments:
            for note in inst.notes:
                if note.start <= time <= note.end:
                    return librosa.midi_to_hz(note.pitch)
        return 440.0

    def _timbre_profile(self, y: np.ndarray, sr: int) -> dict:
        import librosa
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        y_h, y_p = librosa.effects.hpss(y)
        harmonic_energy = float(np.mean(np.abs(y_h)))
        noise_energy = float(np.mean(np.abs(y_p)))
        hnr = harmonic_energy / (noise_energy + 1e-6)
        S = np.abs(librosa.stft(y, n_fft=4096))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        bow_band = (freqs >= 6000) & (freqs <= 10000)
        bow_noise = float(np.mean(S[bow_band])) if np.any(bow_band) else 0.0
        bright_band = (freqs > 4000)
        brightness = float(np.mean(S[bright_band])) if np.any(bright_band) else 0.0
        resonance_band = (freqs >= 280) & (freqs <= 400)
        resonance = float(np.mean(S[resonance_band])) if np.any(resonance_band) else 0.0
        return {
            "spectral_centroid": float(np.mean(centroid)),
            "spectral_rolloff": float(np.mean(rolloff)),
            "bandwidth": float(np.mean(bandwidth)),
            "zero_crossing_rate": float(np.mean(zcr)),
            "harmonic_energy": harmonic_energy,
            "noise_energy": noise_energy,
            "harmonic_noise_ratio": float(hnr),
            "bow_noise_level": bow_noise,
            "brightness": brightness,
            "resonance": resonance,
            "tone_label": self._tone_label(hnr, brightness, bow_noise, resonance),
            "status": "Success",
        }

    def _tone_label(self, hnr: float, brightness: float, bow_noise: float, resonance: float) -> str:
        if hnr > 8 and brightness < 0.3 and resonance > 0.5:
            return "Warm / Focused"
        if brightness > 0.6 and bow_noise < 0.1:
            return "Bright / Clear"
        if bow_noise > 0.4:
            return "Harsh / Noisy"
        if hnr < 3:
            return "Airy / Weak"
        if resonance > 0.7:
            return "Resonant / Full"
        return "Balanced / Neutral"

    def get_module_info(self) -> list:
        deps_status = ("All dependencies installed" if self.deps_ok
                       else f"Missing: {', '.join(self.missing_deps)}")
        return [
            "AUDIO ANALYZER MODULE",
            "",
            f"DEPENDENCIES: {deps_status}",
            "",
            "PURPOSE: Analyze audio for vibrato, intonation, timbre, and alignment",
            "",
            "ANALYSIS FEATURES:",
            "  - Pitch contour extraction",
            "  - Vibrato rate and depth detection",
            "  - Spectral timbre profiling",
            "  - Audio-MIDI alignment",
            "  - Intonation curve generation",
            "  - Timing deviation detection",
            "  - Tone quality classification",
        ]


class AudioAnalyzerGUI:
    """Standalone GUI for Audio Analyzer module using CustomTkinter."""

    def __init__(self, parent=None):
        self.ini_file = "ai_violin_master_01_audio_analyzer.ini"
        self.config = configparser.ConfigParser()

        if parent is None:
            self.root = ctk.CTk()
            self.is_standalone = True
            self.root.title("Audio Analyzer")
            self.root.geometry(WINDOW_SIZE)
            self.root.resizable(False, False)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        else:
            self.root = parent
            self.is_standalone = False

        self.analyzer = AudioAnalyzer()

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
        self.config.set("state", "audio_path", self.audio_path_var.get())
        self.config.set("state", "midi_path", self.midi_path_var.get())
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
            text="AI VIOLIN MASTER - AUDIO ANALYZER",
            font=("Georgia", 16, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            header_frame,
            text="Analyze vibrato, intonation, timbre & alignment",
            font=("Georgia", 12),
            text_color=COLOR_TEXT_SECONDARY,
            fg_color=COLOR_DARK,
        ).pack(pady=(0, 10))

        # File selection frame
        file_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        file_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            file_frame,
            text="FILE SELECTION",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        # Audio file row
        audio_row = ctk.CTkFrame(file_frame, fg_color=COLOR_DARK)
        audio_row.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            audio_row,
            text="Audio File*:",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            width=120,
        ).pack(side="left")

        self.audio_path_var = ctk.StringVar()
        ctk.CTkEntry(
            audio_row,
            textvariable=self.audio_path_var,
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
        ).pack(side="left", fill="x", expand=True, padx=5)

        ctk.CTkButton(
            audio_row,
            text="Browse...",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            width=100,
            command=self._browse_audio,
        ).pack(side="left")

        # MIDI file row
        midi_row = ctk.CTkFrame(file_frame, fg_color=COLOR_DARK)
        midi_row.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            midi_row,
            text="MIDI File (optional):",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            width=120,
        ).pack(side="left")

        self.midi_path_var = ctk.StringVar()
        ctk.CTkEntry(
            midi_row,
            textvariable=self.midi_path_var,
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
        ).pack(side="left", fill="x", expand=True, padx=5)

        ctk.CTkButton(
            midi_row,
            text="Browse...",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            width=100,
            command=self._browse_midi,
        ).pack(side="left")

        # Analysis settings
        settings_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        settings_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            settings_frame,
            text="ANALYSIS SETTINGS",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        sensitivity_row = ctk.CTkFrame(settings_frame, fg_color=COLOR_DARK)
        sensitivity_row.pack(fill="x", padx=20, pady=10)

        ctk.CTkLabel(
            sensitivity_row,
            text="Vibrato Sensitivity:",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            width=150,
        ).pack(side="left")

        self.vibrato_sensitivity = ctk.IntVar(value=50)
        ctk.CTkSlider(
            sensitivity_row,
            from_=0,
            to=100,
            variable=self.vibrato_sensitivity,
            fg_color=COLOR_MEDIUM,
            progress_color=COLOR_GOLD,
            button_color=COLOR_GOLD,
            button_hover_color=COLOR_TEXT,
        ).pack(side="left", fill="x", expand=True, padx=10)

        ctk.CTkLabel(
            sensitivity_row,
            textvariable=self.vibrato_sensitivity,
            font=("Georgia", 12),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
            width=40,
        ).pack(side="left")

        # Analyze button
        self.analyze_button = ctk.CTkButton(
            main_frame,
            text="START ANALYSIS",
            font=("Georgia", 14, "bold"),
            fg_color=COLOR_DARK,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_MEDIUM,
            height=50,
            command=self._analyze_audio,
        )
        self.analyze_button.pack(pady=20)

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
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
            wrap="word",
        )
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)
        self.results_text.insert("1.0", "Select an audio file and click START ANALYSIS")
        self.results_text.configure(state="disabled")

        # Status bar
        self.status_label = ctk.CTkLabel(
            main_frame,
            text="Ready - Select an audio file to begin",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            height=30,
        )
        self.status_label.pack(fill="x", pady=(10, 0))

    def _browse_audio(self):
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.audio_path_var.set(filename)
            self.status_label.configure(text=f"Selected: {os.path.basename(filename)}")

    def _browse_midi(self):
        filename = filedialog.askopenfilename(
            title="Select MIDI File (optional)",
            filetypes=[("MIDI files", "*.mid *.midi"), ("All files", "*.*")],
        )
        if filename:
            self.midi_path_var.set(filename)
            self.status_label.configure(text=f"Selected MIDI: {os.path.basename(filename)}")

    def _analyze_audio(self):
        audio_path = self.audio_path_var.get()
        midi_path = self.midi_path_var.get() if self.midi_path_var.get() else None

        if not os.path.exists(audio_path):
            messagebox.showerror("File Error", "Selected audio file does not exist!")
            return

        if not self.analyzer.deps_ok:
            missing = ", ".join(self.analyzer.missing_deps)
            messagebox.showerror(
                "Missing Dependencies",
                f"The following required packages are missing:\n\n{missing}\n\n"
                f"Install with: pip install {missing}",
            )
            return

        self.analyze_button.configure(state="disabled", text="ANALYZING...")
        self.status_label.configure(text="Analyzing audio... Please wait.")
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        self.results_text.insert("1.0", "Running analysis... Please wait.")
        self.results_text.configure(state="disabled")

        threading.Thread(
            target=self._run_analysis_thread,
            args=(audio_path, midi_path),
            daemon=True,
        ).start()

    def _run_analysis_thread(self, audio_path, midi_path):
        pm = None
        if midi_path and os.path.exists(midi_path):
            import pretty_midi
            pm = pretty_midi.PrettyMIDI(midi_path)

        analysis_result = self.analyzer.analyze_audio(audio_path, pm)
        self.root.after(0, self._display_results, analysis_result)

    def _display_results(self, analysis_result):
        if "error" in analysis_result:
            error_msg = analysis_result["error"]
            messagebox.showerror("Analysis Error", error_msg)
            self.analyze_button.configure(state="normal", text="START ANALYSIS")
            self.status_label.configure(text="Analysis failed")
            return

        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")

        summary = (
            "ANALYSIS SUMMARY\n"
            "=" * 50 + "\n\n"
            f"AUDIO FILE: {os.path.basename(self.audio_path_var.get())}\n"
            f"DURATION: {analysis_result.get('audio_length_seconds', 0):.1f} seconds\n"
            f"SAMPLE RATE: {analysis_result.get('sample_rate', 0)} Hz\n"
            f"PITCH POINTS: {len(analysis_result.get('pitch_contour_smoothed', []))}\n"
        )

        vibrato = analysis_result.get("vibrato", {})
        summary += "\nVIBRATO ANALYSIS:\n"
        if vibrato.get("status") == "Success":
            summary += (
                f"  Rate: {vibrato.get('rate_hz', 0):.1f} Hz\n"
                f"  Depth: {vibrato.get('depth_cents', 0):.1f} cents\n"
                f"  Stability: {vibrato.get('stability', 0):.3f}\n"
                f"  Quality: {vibrato.get('quality', 'Unknown')}\n"
            )
        else:
            summary += f"  Status: {vibrato.get('status', 'Failed')}\n"

        timbre = analysis_result.get("timbre", {})
        summary += "\nTIMBRE ANALYSIS:\n"
        if timbre.get("status") == "Success":
            summary += (
                f"  Tone Quality: {timbre.get('tone_label', 'Unknown')}\n"
                f"  Harmonic Ratio: {timbre.get('harmonic_noise_ratio', 0):.1f}\n"
                f"  Brightness: {timbre.get('brightness', 0):.2f}\n"
            )
        else:
            summary += f"  Status: {timbre.get('status', 'Failed')}\n"

        alignment = analysis_result.get("alignment", {})
        summary += "\nALIGNMENT ANALYSIS:\n"
        if alignment.get("status") in ["Success", "No MIDI"]:
            summary += (
                f"  Timing Offset: {alignment.get('average_timing_offset', 0) * 1000:.0f} ms\n"
                f"  Timing: {alignment.get('timing_label', 'Unknown')}\n"
                f"  Notes Matched: {alignment.get('note_count_aligned', 0)}\n"
            )
        else:
            summary += f"  Status: {alignment.get('status', 'Failed')}\n"

        self.results_text.insert("1.0", summary)
        self.results_text.configure(state="disabled")

        self.status_label.configure(text="Analysis complete")
        self.analyze_button.configure(state="normal", text="RE-ANALYZE")
        messagebox.showinfo("Analysis Complete", "Audio analysis completed successfully!")

    def _on_close(self):
        self._save_state()
        self.root.destroy()

    def get_content(self):
        return self.analyzer.get_module_info()


def create_gui(parent):
    """Factory used by the CTk entry. Packs the full GUI into parent."""
    return AudioAnalyzerGUI(parent)


def main():
    """Standalone entry - launches a full window with mainloop."""
    AudioAnalyzerGUI()


if __name__ == "__main__":
    main()