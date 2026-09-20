#!/usr/bin/env python3
# ai_violin_master_12_yin_pitch_detector.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683


import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
import threading
import wave
import struct
from scipy.io import wavfile
import os
import json
import configparser

# Color constants
COLOR_BG = "#5A381E"
COLOR_DARK = "#3B2413"
COLOR_GOLD = "#FFD770"
COLOR_FRAME = "#4B2E18"
COLOR_TEXT = "#FFF2CF"
COLOR_MEDIUM = "#4B2E18"

WINDOW_SIZE = "640x1050"


class YinPitchDetector:
    """YIN pitch detection algorithm implementation for violin."""

    def __init__(self, sample_rate=44100, frame_size=2048, hop_size=512):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.min_freq = 196.0
        self.max_freq = 2637.0
        self.min_midi = 55
        self.max_midi = 104
        self.threshold = 0.15
        self.absolute_threshold = 0.1
        self.note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def set_parameters(self, threshold=None, absolute_threshold=None):
        if threshold is not None:
            self.threshold = threshold
        if absolute_threshold is not None:
            self.absolute_threshold = absolute_threshold

    def difference_function(self, x):
        N = len(x)
        diff = np.zeros(N // 2)
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[N-1:]
        for tau in range(len(diff)):
            diff[tau] = autocorr[0] + autocorr[2*tau] - 2*autocorr[tau]
        return diff

    def cumulative_mean_normalized_difference(self, diff):
        cmndf = np.zeros_like(diff)
        cmndf[0] = 1.0
        running_sum = 0.0
        for tau in range(1, len(diff)):
            running_sum += diff[tau]
            cmndf[tau] = diff[tau] * tau / running_sum
        return cmndf

    def absolute_threshold_estimation(self, cmndf):
        for tau in range(1, len(cmndf)):
            if cmndf[tau] < self.absolute_threshold:
                while (tau + 1 < len(cmndf) and cmndf[tau + 1] < cmndf[tau]):
                    tau += 1
                return tau
        return -1

    def parabolic_interpolation(self, cmndf, tau):
        if tau <= 0 or tau >= len(cmndf) - 1:
            return float(tau)
        s0 = cmndf[tau - 1]
        s1 = cmndf[tau]
        s2 = cmndf[tau + 1]
        adjustment = (s2 - s0) / (2 * (2 * s1 - s2 - s0))
        return tau + adjustment

    def detect_pitch_frame(self, audio_frame):
        window = np.hanning(len(audio_frame))
        x = audio_frame * window
        diff = self.difference_function(x)
        cmndf = self.cumulative_mean_normalized_difference(diff)
        tau = self.absolute_threshold_estimation(cmndf)
        if tau == -1:
            return 0.0, 0.0
        refined_tau = self.parabolic_interpolation(cmndf, tau)
        frequency = self.sample_rate / refined_tau if refined_tau > 0 else 0.0
        confidence = 1.0 - cmndf[tau]
        confidence = max(0.0, min(1.0, confidence))
        if frequency < self.min_freq or frequency > self.max_freq:
            return 0.0, 0.0
        return frequency, confidence

    def midi_to_freq(self, midi_note):
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    def freq_to_midi(self, frequency):
        if frequency <= 0:
            return 0
        return 69 + 12 * np.log2(frequency / 440.0)

    def midi_to_note_name(self, midi_note):
        if midi_note < 0 or midi_note > 127:
            return "---"
        note_index = int(midi_note) % 12
        octave = int(midi_note) // 12 - 1
        return f"{self.note_names[note_index]}{octave}"

    def cents_deviation(self, frequency, target_midi):
        target_freq = self.midi_to_freq(target_midi)
        if frequency <= 0 or target_freq <= 0:
            return 0.0
        return 1200 * np.log2(frequency / target_freq)

    def analyze_audio(self, audio_data, progress_callback=None):
        audio_length = len(audio_data)
        num_frames = (audio_length - self.frame_size) // self.hop_size + 1
        if num_frames <= 0:
            return {"error": "Audio too short for analysis"}
        times = np.zeros(num_frames)
        frequencies = np.zeros(num_frames)
        confidences = np.zeros(num_frames)
        midi_notes = np.zeros(num_frames)
        note_names = []
        for i in range(num_frames):
            if progress_callback and i % 10 == 0:
                progress_callback(i / num_frames)
            start = i * self.hop_size
            end = start + self.frame_size
            frame = audio_data[start:end]
            if len(frame) < self.frame_size:
                break
            freq, conf = self.detect_pitch_frame(frame)
            times[i] = start / self.sample_rate
            frequencies[i] = freq
            confidences[i] = conf
            if freq > 0:
                midi = self.freq_to_midi(freq)
                midi_notes[i] = midi
                note_names.append(self.midi_to_note_name(midi))
            else:
                midi_notes[i] = 0
                note_names.append("---")
        valid_freqs = frequencies[frequencies > 0]
        valid_confidences = confidences[confidences > 0]
        valid_midi = midi_notes[midi_notes > 0]
        if len(valid_freqs) == 0:
            return {"error": "No pitch detected in audio"}
        if len(valid_midi) > 0:
            rounded_midi = np.round(valid_midi).astype(int)
            unique, counts = np.unique(rounded_midi, return_counts=True)
            most_common_idx = np.argmax(counts)
            most_common_midi = unique[most_common_idx]
            most_common_note = self.midi_to_note_name(most_common_midi)
            common_note_freqs = valid_freqs[rounded_midi == most_common_midi]
            cents_devs = self.cents_deviation(common_note_freqs, most_common_midi)
            intonation_accuracy = 100.0 * np.mean(np.abs(cents_devs) < 50)
            avg_cents_dev = np.mean(cents_devs) if len(cents_devs) > 0 else 0
        else:
            most_common_note = "---"
            intonation_accuracy = 0
            avg_cents_dev = 0
        return {
            "times": times.tolist(),
            "frequencies": frequencies.tolist(),
            "confidences": confidences.tolist(),
            "midi_notes": midi_notes.tolist(),
            "note_names": note_names,
            "sample_rate": self.sample_rate,
            "analysis_length": audio_length / self.sample_rate,
            "num_frames": num_frames,
            "pitch_detected_frames": len(valid_freqs),
            "detection_rate": len(valid_freqs) / num_frames * 100,
            "average_frequency": float(np.mean(valid_freqs)),
            "average_confidence": float(np.mean(valid_confidences)),
            "frequency_std": float(np.std(valid_freqs)),
            "most_common_note": most_common_note,
            "intonation_accuracy": float(intonation_accuracy),
            "average_cents_deviation": float(avg_cents_dev),
            "frequency_range": {
                "min": float(np.min(valid_freqs)),
                "max": float(np.max(valid_freqs)),
                "range": float(np.max(valid_freqs) - np.min(valid_freqs))
            }
        }

    def export_results(self, results, filename):
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        return True, None

    def load_wav_file(self, filepath):
        sample_rate, audio_data = wavfile.read(filepath)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        audio_data = audio_data.astype(np.float32)
        if np.abs(audio_data).max() > 0:
            audio_data = audio_data / np.abs(audio_data).max()
        return sample_rate, audio_data


class YinPitchDetectorGUI:
    """GUI for YIN Pitch Detector using CustomTkinter."""

    def __init__(self, parent=None):
        self.ini_file = "ai_violin_master_12_yin_pitch_detector.ini"
        self.config = configparser.ConfigParser()

        if parent is None:
            self.root = ctk.CTk()
            self.is_standalone = True
            self.root.title("YIN Pitch Detector")
            self.root.geometry(WINDOW_SIZE)
            self.root.resizable(False, False)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        else:
            self.root = parent
            self.is_standalone = False

        self.detector = YinPitchDetector()
        self.current_results = None
        self.audio_data = None
        self.sample_rate = 44100
        self.is_analyzing = False
        self.current_file = None

        self._load_state()
        ctk.set_appearance_mode("dark")
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

    def _create_widgets(self):
        # Main container
        main_frame = ctk.CTkFrame(self.root, fg_color=COLOR_BG)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        header_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            header_frame,
            text="YIN PITCH DETECTOR",
            font=("Georgia", 16, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            header_frame,
            text="Advanced pitch detection for violin using YIN algorithm",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=(0, 10))

        # File selection
        file_frame = ctk.CTkFrame(main_frame, fg_color=COLOR_DARK, border_width=2, border_color=COLOR_GOLD)
        file_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            file_frame,
            text="AUDIO FILE SELECTION",
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

        ctk.CTkButton(
            file_frame,
            text="BROWSE WAV FILE",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_FRAME,
            command=self._browse_audio_file,
        ).pack(pady=10)

        # Analyze button
        self.analyze_btn = ctk.CTkButton(
            main_frame,
            text="ANALYZE PITCH",
            font=("Georgia", 14, "bold"),
            fg_color=COLOR_DARK,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_MEDIUM,
            height=50,
            command=self._start_analysis,
        )
        self.analyze_btn.pack(pady=20)

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
        self.results_text.insert("1.0", "Load a WAV file and click ANALYZE PITCH")
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

    def _browse_audio_file(self):
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV files", "*.wav *.WAV"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(os.path.basename(filename))
            self._load_audio_file(filename)

    def _load_audio_file(self, filepath):
        detector = YinPitchDetector()
        sample_rate, audio_data = detector.load_wav_file(filepath)
        self.sample_rate = sample_rate
        self.audio_data = audio_data
        self.current_file = filepath
        duration = len(audio_data) / sample_rate
        self.status_label.configure(text=f"Loaded: {duration:.2f}s, {sample_rate}Hz, {len(audio_data)} samples")

    def _start_analysis(self):
        if self.audio_data is None:
            messagebox.showwarning("No Audio", "Please load an audio file first.")
            return
        if self.is_analyzing:
            return
        self.is_analyzing = True
        self.status_label.configure(text="Analyzing pitch...")
        self.analyze_btn.configure(state="disabled", text="ANALYZING...")
        threading.Thread(target=self._perform_analysis, daemon=True).start()

    def _perform_analysis(self):
        detector = YinPitchDetector(
            sample_rate=self.sample_rate,
            frame_size=2048,
            hop_size=512
        )
        detector.set_parameters(threshold=0.15, absolute_threshold=0.1)
        results = detector.analyze_audio(self.audio_data)
        self.root.after(0, lambda: self._display_results(results, detector))

    def _display_results(self, results, detector):
        self.is_analyzing = False
        self.analyze_btn.configure(state="normal", text="ANALYZE PITCH")

        if "error" in results:
            messagebox.showerror("Analysis Error", results["error"])
            self.status_label.configure(text="Analysis failed")
            return

        self.current_results = results
        self.current_detector = detector

        output = "PITCH ANALYSIS RESULTS\n"
        output += "=" * 40 + "\n\n"
        output += "STATISTICS\n"
        output += "-" * 30 + "\n"
        output += f"Audio duration: {results['analysis_length']:.2f}s\n"
        output += f"Analysis frames: {results['num_frames']}\n"
        output += f"Pitch detected frames: {results['pitch_detected_frames']}\n"
        output += f"Detection rate: {results['detection_rate']:.1f}%\n"
        output += f"Average frequency: {results['average_frequency']:.1f} Hz\n"
        output += f"Average confidence: {results['average_confidence']:.3f}\n"
        output += f"Frequency std: {results['frequency_std']:.1f} Hz\n\n"
        output += "VIOLIN-SPECIFIC\n"
        output += "-" * 30 + "\n"
        output += f"Most common note: {results['most_common_note']}\n"
        output += f"Intonation accuracy: {results['intonation_accuracy']:.1f}%\n"
        output += f"Avg cents deviation: {results['average_cents_deviation']:.1f}\n"
        output += f"Frequency range: {results['frequency_range']['min']:.1f} - {results['frequency_range']['max']:.1f} Hz\n"

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
        "YIN PITCH DETECTOR MODULE",
        "",
        "PURPOSE: Advanced pitch detection for violin using YIN algorithm",
        "",
        "FEATURES:",
        "  - YIN fundamental frequency estimation",
        "  - Violin-specific frequency range (G3-E7)",
        "  - Intonation accuracy analysis",
        "  - Cents deviation measurement",
        "  - Confidence scoring for each detection",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = YinPitchDetectorGUI(parent)
    return gui.root


def main():
    """Standalone entry - launches a full window with mainloop."""
    YinPitchDetectorGUI()


if __name__ == "__main__":
    main()