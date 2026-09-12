#!/usr/bin/env python3
# ai_violin_master_05_yin_pitch_detector.py

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import threading
import time
import wave
import struct
from scipy import signal
from scipy.io import wavfile
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

class YinPitchDetector:
    """YIN pitch detection algorithm implementation for violin."""

    def __init__(self, sample_rate=44100, frame_size=2048, hop_size=512):
        """
        Initialize YIN pitch detector.

        Args:
            sample_rate: Audio sample rate in Hz
            frame_size: Size of analysis frame in samples
            hop_size: Hop size between frames in samples
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size

        # Violin pitch range (MIDI 55-104, approx G3-E7)
        self.min_freq = 196.0  # G3
        self.max_freq = 2637.0  # E7
        self.min_midi = 55
        self.max_midi = 104

        # YIN parameters
        self.threshold = 0.15
        self.absolute_threshold = 0.1

        # Note name mapping
        self.note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def set_parameters(self, threshold=None, absolute_threshold=None):
        """Update YIN parameters."""
        if threshold is not None:
            self.threshold = threshold
        if absolute_threshold is not None:
            self.absolute_threshold = absolute_threshold

    def difference_function(self, x):
        """
        Compute YIN difference function.

        Args:
            x: Audio frame

        Returns:
            diff: Difference function
        """
        N = len(x)
        diff = np.zeros(N // 2)

        # Precompute autocorrelation
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[N-1:]  # Keep only positive lags

        # Compute difference function
        for tau in range(len(diff)):
            diff[tau] = autocorr[0] + autocorr[2*tau] - 2*autocorr[tau]

        return diff

    def cumulative_mean_normalized_difference(self, diff):
        """
        Compute cumulative mean normalized difference function.

        Args:
            diff: Difference function

        Returns:
            cmndf: Cumulative mean normalized difference
        """
        cmndf = np.zeros_like(diff)
        cmndf[0] = 1.0

        running_sum = 0.0
        for tau in range(1, len(diff)):
            running_sum += diff[tau]
            cmndf[tau] = diff[tau] * tau / running_sum

        return cmndf

    def absolute_threshold_estimation(self, cmndf):
        """
        Estimate pitch using absolute threshold.

        Args:
            cmndf: Cumulative mean normalized difference

        Returns:
            tau: Estimated period in samples (or -1 if below threshold)
        """
        for tau in range(1, len(cmndf)):
            if cmndf[tau] < self.absolute_threshold:
                # Find local minimum
                while (tau + 1 < len(cmndf) and
                       cmndf[tau + 1] < cmndf[tau]):
                    tau += 1
                return tau
        return -1

    def parabolic_interpolation(self, cmndf, tau):
        """
        Refine period estimate using parabolic interpolation.

        Args:
            cmndf: Cumulative mean normalized difference
            tau: Initial period estimate

        Returns:
            refined_tau: Interpolated period estimate
        """
        if tau <= 0 or tau >= len(cmndf) - 1:
            return float(tau)

        # Get values for interpolation
        s0 = cmndf[tau - 1]
        s1 = cmndf[tau]
        s2 = cmndf[tau + 1]

        # Parabolic interpolation formula
        adjustment = (s2 - s0) / (2 * (2 * s1 - s2 - s0))
        refined_tau = tau + adjustment

        return refined_tau

    def detect_pitch_frame(self, audio_frame):
        """
        Detect pitch for a single frame.

        Args:
            audio_frame: Audio frame to analyze

        Returns:
            frequency: Estimated frequency in Hz (0 if no pitch detected)
            confidence: Detection confidence (0-1)
        """
        # Apply windowing
        window = np.hanning(len(audio_frame))
        x = audio_frame * window

        # Compute difference function
        diff = self.difference_function(x)

        # Compute CMNDF
        cmndf = self.cumulative_mean_normalized_difference(diff)

        # Apply absolute threshold
        tau = self.absolute_threshold_estimation(cmndf)

        if tau == -1:
            # No pitch detected
            return 0.0, 0.0

        # Refine period estimate
        refined_tau = self.parabolic_interpolation(cmndf, tau)

        # Convert to frequency
        frequency = self.sample_rate / refined_tau if refined_tau > 0 else 0.0

        # Compute confidence
        confidence = 1.0 - cmndf[tau]
        confidence = max(0.0, min(1.0, confidence))

        # Filter by violin frequency range
        if frequency < self.min_freq or frequency > self.max_freq:
            return 0.0, 0.0

        return frequency, confidence

    def midi_to_freq(self, midi_note):
        """Convert MIDI note number to frequency."""
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    def freq_to_midi(self, frequency):
        """Convert frequency to MIDI note number."""
        if frequency <= 0:
            return 0
        return 69 + 12 * np.log2(frequency / 440.0)

    def midi_to_note_name(self, midi_note):
        """Convert MIDI note number to note name."""
        if midi_note < 0 or midi_note > 127:
            return "---"

        note_index = int(midi_note) % 12
        octave = int(midi_note) // 12 - 1
        return f"{self.note_names[note_index]}{octave}"

    def cents_deviation(self, frequency, target_midi):
        """Calculate cents deviation from target MIDI note."""
        target_freq = self.midi_to_freq(target_midi)
        if frequency <= 0 or target_freq <= 0:
            return 0.0
        return 1200 * np.log2(frequency / target_freq)

    def analyze_audio(self, audio_data, progress_callback=None):
        """
        Analyze entire audio signal for pitch contour.

        Args:
            audio_data: Mono audio signal
            progress_callback: Callback function for progress updates

        Returns:
            results: Dictionary with analysis results
        """
        audio_length = len(audio_data)
        num_frames = (audio_length - self.frame_size) // self.hop_size + 1

        if num_frames <= 0:
            return {"error": "Audio too short for analysis"}

        # Initialize results arrays
        times = np.zeros(num_frames)
        frequencies = np.zeros(num_frames)
        confidences = np.zeros(num_frames)
        midi_notes = np.zeros(num_frames)
        note_names = []

        # Process each frame
        for i in range(num_frames):
            # Update progress
            if progress_callback and i % 10 == 0:
                progress_callback(i / num_frames)

            # Extract frame
            start = i * self.hop_size
            end = start + self.frame_size
            frame = audio_data[start:end]

            if len(frame) < self.frame_size:
                break

            # Detect pitch
            freq, conf = self.detect_pitch_frame(frame)

            # Store results
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

        # Calculate statistics
        valid_freqs = frequencies[frequencies > 0]
        valid_confidences = confidences[confidences > 0]
        valid_midi = midi_notes[midi_notes > 0]

        if len(valid_freqs) == 0:
            return {"error": "No pitch detected in audio"}

        # Identify most common note
        if len(valid_midi) > 0:
            rounded_midi = np.round(valid_midi).astype(int)
            unique, counts = np.unique(rounded_midi, return_counts=True)
            most_common_idx = np.argmax(counts)
            most_common_midi = unique[most_common_idx]
            most_common_note = self.midi_to_note_name(most_common_midi)

            # Calculate intonation accuracy for most common note
            common_note_freqs = valid_freqs[rounded_midi == most_common_midi]
            target_freq = self.midi_to_freq(most_common_midi)
            cents_devs = self.cents_deviation(common_note_freqs, most_common_midi)
            intonation_accuracy = 100.0 * np.mean(np.abs(cents_devs) < 50)  # Within 50 cents
            avg_cents_dev = np.mean(cents_devs) if len(cents_devs) > 0 else 0
        else:
            most_common_note = "---"
            intonation_accuracy = 0
            avg_cents_dev = 0

        # Compile results
        results = {
            "times": times.tolist(),
            "frequencies": frequencies.tolist(),
            "confidences": confidences.tolist(),
            "midi_notes": midi_notes.tolist(),
            "note_names": note_names,
            "sample_rate": self.sample_rate,
            "frame_size": self.frame_size,
            "hop_size": self.hop_size,
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

        return results

    def export_results(self, results, filename):
        """Export analysis results to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            return True, None
        except Exception as e:
            return False, str(e)

    def load_wav_file(self, filepath):
        """Load WAV file and return mono audio data."""
        try:
            # Try scipy first
            try:
                sample_rate, audio_data = wavfile.read(filepath)
            except:
                # Fall back to wave module
                with wave.open(filepath, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    audio_bytes = wav_file.readframes(n_frames)

                    if wav_file.getsampwidth() == 2:
                        fmt = f"{n_frames}h"
                    else:
                        fmt = f"{n_frames}b"

                    audio_data = np.array(struct.unpack(fmt, audio_bytes))

            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            # Normalize to float
            audio_data = audio_data.astype(np.float32)
            if np.abs(audio_data).max() > 0:
                audio_data = audio_data / np.abs(audio_data).max()

            return sample_rate, audio_data

        except Exception as e:
            raise Exception(f"Error loading WAV file: {str(e)}")

class YinPitchDetectorGUI:
    """GUI for YIN Pitch Detector matching the color scheme and size."""

    def __init__(self, parent):
        self.parent = parent
        self.detector = None
        self.current_results = None
        self.audio_data = None
        self.sample_rate = 44100
        self.is_analyzing = False

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
                text="🎻 YIN Pitch Detector",
                font=FONT_HEADER,
                fg=COLOR_GOLD,
                bg=COLOR_BG).pack()

        tk.Label(header_frame,
                text="Advanced pitch detection for violin using YIN algorithm",
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

        # Tab 1: Audio Input
        input_frame = tk.Frame(notebook, bg=COLOR_BG)
        notebook.add(input_frame, text="Audio Input")

        self.create_input_tab(input_frame)

        # Tab 2: Pitch Analysis
        analysis_frame = tk.Frame(notebook, bg=COLOR_BG)
        notebook.add(analysis_frame, text="Pitch Analysis")

        self.create_analysis_tab(analysis_frame)

        # Tab 3: Visualization
        viz_frame = tk.Frame(notebook, bg=COLOR_BG)
        notebook.add(viz_frame, text="Visualization")

        self.create_visualization_tab(viz_frame)

        # Tab 4: Settings
        settings_frame = tk.Frame(notebook, bg=COLOR_BG)
        notebook.add(settings_frame, text="Settings")

        self.create_settings_tab(settings_frame)

        # Action buttons at bottom
        self.create_action_buttons()

    def create_input_tab(self, parent_frame):
        """Create the audio input tab."""
        # File selection frame
        file_frame = tk.LabelFrame(parent_frame,
                                 text="Audio File Selection",
                                 font=FONT_SECTION,
                                 fg=COLOR_GOLD,
                                 bg=COLOR_BG,
                                 relief=tk.RIDGE,
                                 borderwidth=2)
        file_frame.pack(fill="x", padx=20, pady=10, ipadx=10, ipady=10)

        # File path display
        self.file_path_var = tk.StringVar(value="No file selected")
        path_frame = tk.Frame(file_frame, bg=COLOR_BG)
        path_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(path_frame,
                text="File:",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=10,
                anchor="w").pack(side="left")

        file_label = tk.Label(path_frame,
                            textvariable=self.file_path_var,
                            font=FONT_SMALL,
                            fg=COLOR_TEXT,
                            bg=COLOR_DARK,
                            relief=tk.SUNKEN,
                            borderwidth=1,
                            anchor="w",
                            width=40)
        file_label.pack(side="left", padx=5, fill="x", expand=True)

        # File info
        self.file_info_var = tk.StringVar(value="File info will appear here")
        info_label = tk.Label(file_frame,
                            textvariable=self.file_info_var,
                            font=FONT_SMALL,
                            fg=COLOR_TEXT,
                            bg=COLOR_BG,
                            anchor="w",
                            justify=tk.LEFT)
        info_label.pack(fill="x", padx=10, pady=5)

        # Buttons frame
        buttons_frame = tk.Frame(file_frame, bg=COLOR_BG)
        buttons_frame.pack(fill="x", padx=10, pady=10)

        browse_btn = tk.Button(buttons_frame,
                              text="📁 Browse WAV File",
                              font=FONT_TEXT,
                              fg=COLOR_GOLD,
                              bg=COLOR_DARK,
                              command=self.browse_audio_file)
        browse_btn.pack(side="left", padx=5)

        record_btn = tk.Button(buttons_frame,
                              text="🎤 Record Audio",
                              font=FONT_TEXT,
                              fg=COLOR_GOLD,
                              bg=COLOR_DARK,
                              command=self.record_audio)
        record_btn.pack(side="left", padx=5)

        # Audio visualization placeholder
        viz_frame = tk.LabelFrame(parent_frame,
                                text="Audio Waveform",
                                font=FONT_SECTION,
                                fg=COLOR_GOLD,
                                bg=COLOR_BG,
                                relief=tk.RIDGE,
                                borderwidth=2)
        viz_frame.pack(fill="both", expand=True, padx=20, pady=10, ipadx=10, ipady=10)

        self.waveform_canvas = tk.Canvas(viz_frame,
                                       bg=COLOR_DARK,
                                       highlightthickness=0,
                                       height=150)
        self.waveform_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # Info panel
        info_text = """💡 Audio Requirements:
• WAV format (preferred)
• Mono or stereo
• Sample rate: 44.1kHz recommended
• Duration: 1-30 seconds ideal
• Violin range: G3 (196Hz) to E7 (2637Hz)"""

        info_label = tk.Label(parent_frame,
                            text=info_text,
                            font=FONT_SMALL,
                            fg=COLOR_TEXT,
                            bg=COLOR_BG,
                            justify=tk.LEFT)
        info_label.pack(fill="x", padx=20, pady=10)

    def create_analysis_tab(self, parent_frame):
        """Create the pitch analysis tab."""
        # Results display
        results_frame = tk.LabelFrame(parent_frame,
                                    text="Pitch Analysis Results",
                                    font=FONT_SECTION,
                                    fg=COLOR_GOLD,
                                    bg=COLOR_BG,
                                    relief=tk.RIDGE,
                                    borderwidth=2)
        results_frame.pack(fill="both", expand=True, padx=20, pady=10, ipadx=10, ipady=10)

        # Text widget for results
        self.results_text = tk.Text(results_frame,
                                  font=FONT_SMALL,
                                  bg=COLOR_DARK,
                                  fg=COLOR_TEXT,
                                  insertbackground=COLOR_TEXT,
                                  relief=tk.SUNKEN,
                                  borderwidth=1,
                                  height=15,
                                  wrap=tk.WORD)
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)

        # Add scrollbar
        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)

        # Statistics frame
        stats_frame = tk.Frame(results_frame, bg=COLOR_BG)
        stats_frame.pack(fill="x", padx=10, pady=10)

        self.stats_var = tk.StringVar(value="No analysis results yet")
        stats_label = tk.Label(stats_frame,
                             textvariable=self.stats_var,
                             font=FONT_SMALL,
                             fg=COLOR_GOLD,
                             bg=COLOR_BG,
                             justify=tk.LEFT)
        stats_label.pack(anchor="w")

        # Intonation analysis
        intonation_frame = tk.LabelFrame(parent_frame,
                                       text="Intonation Analysis",
                                       font=FONT_SECTION,
                                       fg=COLOR_GOLD,
                                       bg=COLOR_BG,
                                       relief=tk.GROOVE,
                                       borderwidth=1)
        intonation_frame.pack(fill="x", padx=20, pady=10)

        self.intonation_var = tk.StringVar(value="Intonation accuracy: --")
        intonation_label = tk.Label(intonation_frame,
                                  textvariable=self.intonation_var,
                                  font=FONT_TEXT,
                                  fg=COLOR_TEXT,
                                  bg=COLOR_BG)
        intonation_label.pack(padx=10, pady=5)

        # Cents deviation visualization
        cents_frame = tk.Frame(intonation_frame, bg=COLOR_BG)
        cents_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(cents_frame,
                text="Cents deviation:",
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_BG).pack(side="left")

        self.cents_var = tk.StringVar(value="--")
        cents_label = tk.Label(cents_frame,
                             textvariable=self.cents_var,
                             font=FONT_SMALL,
                             fg=COLOR_GOLD,
                             bg=COLOR_DARK,
                             relief=tk.SUNKEN,
                             borderwidth=1,
                             width=10)
        cents_label.pack(side="left", padx=5)

        # Deviation indicator
        self.deviation_canvas = tk.Canvas(intonation_frame,
                                        bg=COLOR_DARK,
                                        highlightthickness=0,
                                        height=30)
        self.deviation_canvas.pack(fill="x", padx=10, pady=5)

    def create_visualization_tab(self, parent_frame):
        """Create the visualization tab."""
        # Pitch contour plot
        contour_frame = tk.LabelFrame(parent_frame,
                                    text="Pitch Contour",
                                    font=FONT_SECTION,
                                    fg=COLOR_GOLD,
                                    bg=COLOR_BG,
                                    relief=tk.RIDGE,
                                    borderwidth=2)
        contour_frame.pack(fill="both", expand=True, padx=20, pady=10, ipadx=10, ipady=10)

        self.pitch_canvas = tk.Canvas(contour_frame,
                                    bg=COLOR_DARK,
                                    highlightthickness=0)
        self.pitch_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # Confidence plot
        confidence_frame = tk.LabelFrame(parent_frame,
                                       text="Detection Confidence",
                                       font=FONT_SECTION,
                                       fg=COLOR_GOLD,
                                       bg=COLOR_BG,
                                       relief=tk.RIDGE,
                                       borderwidth=2)
        confidence_frame.pack(fill="both", expand=True, padx=20, pady=10, ipadx=10, ipady=10)

        self.confidence_canvas = tk.Canvas(confidence_frame,
                                         bg=COLOR_DARK,
                                         highlightthickness=0,
                                         height=150)
        self.confidence_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # Visualization controls
        controls_frame = tk.Frame(parent_frame, bg=COLOR_BG)
        controls_frame.pack(fill="x", padx=20, pady=10)

        tk.Label(controls_frame,
                text="Display:",
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_BG).pack(side="left", padx=5)

        self.display_var = tk.StringVar(value="pitch")

        tk.Radiobutton(controls_frame,
                      text="Pitch Only",
                      variable=self.display_var,
                      value="pitch",
                      font=FONT_SMALL,
                      fg=COLOR_TEXT,
                      bg=COLOR_BG,
                      selectcolor=COLOR_DARK).pack(side="left", padx=5)

        tk.Radiobutton(controls_frame,
                      text="Pitch + Confidence",
                      variable=self.display_var,
                      value="both",
                      font=FONT_SMALL,
                      fg=COLOR_TEXT,
                      bg=COLOR_BG,
                      selectcolor=COLOR_DARK).pack(side="left", padx=5)

        # Update button
        update_btn = tk.Button(controls_frame,
                              text="🔄 Update Display",
                              font=FONT_SMALL,
                              fg=COLOR_GOLD,
                              bg=COLOR_DARK,
                              command=self.update_visualizations)
        update_btn.pack(side="right", padx=5)

    def create_settings_tab(self, parent_frame):
        """Create the settings tab."""
        # YIN parameters frame
        params_frame = tk.LabelFrame(parent_frame,
                                   text="YIN Algorithm Parameters",
                                   font=FONT_SECTION,
                                   fg=COLOR_GOLD,
                                   bg=COLOR_BG,
                                   relief=tk.RIDGE,
                                   borderwidth=2)
        params_frame.pack(fill="x", padx=20, pady=10, ipadx=10, ipady=10)

        # Threshold setting
        threshold_frame = tk.Frame(params_frame, bg=COLOR_BG)
        threshold_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(threshold_frame,
                text="Threshold:",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=15,
                anchor="w").pack(side="left")

        self.threshold_var = tk.DoubleVar(value=0.15)
        threshold_scale = tk.Scale(threshold_frame,
                                  from_=0.01,
                                  to=0.5,
                                  variable=self.threshold_var,
                                  orient=tk.HORIZONTAL,
                                  resolution=0.01,
                                  length=200,
                                  bg=COLOR_BG,
                                  fg=COLOR_TEXT,
                                  troughcolor=COLOR_DARK,
                                  highlightbackground=COLOR_BG,
                                  sliderrelief=tk.RAISED)
        threshold_scale.pack(side="left", padx=5, fill="x", expand=True)

        threshold_label = tk.Label(threshold_frame,
                                 textvariable=self.threshold_var,
                                 font=FONT_SMALL,
                                 fg=COLOR_GOLD,
                                 bg=COLOR_BG,
                                 width=5)
        threshold_label.pack(side="right", padx=5)

        # Absolute threshold setting
        abs_threshold_frame = tk.Frame(params_frame, bg=COLOR_BG)
        abs_threshold_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(abs_threshold_frame,
                text="Absolute Threshold:",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=15,
                anchor="w").pack(side="left")

        self.abs_threshold_var = tk.DoubleVar(value=0.1)
        abs_threshold_scale = tk.Scale(abs_threshold_frame,
                                      from_=0.01,
                                      to=0.3,
                                      variable=self.abs_threshold_var,
                                      orient=tk.HORIZONTAL,
                                      resolution=0.01,
                                      length=200,
                                      bg=COLOR_BG,
                                      fg=COLOR_TEXT,
                                      troughcolor=COLOR_DARK,
                                      highlightbackground=COLOR_BG,
                                      sliderrelief=tk.RAISED)
        abs_threshold_scale.pack(side="left", padx=5, fill="x", expand=True)

        abs_threshold_label = tk.Label(abs_threshold_frame,
                                      textvariable=self.abs_threshold_var,
                                      font=FONT_SMALL,
                                      fg=COLOR_GOLD,
                                      bg=COLOR_BG,
                                      width=5)
        abs_threshold_label.pack(side="right", padx=5)

        # Frame size setting
        frame_frame = tk.Frame(params_frame, bg=COLOR_BG)
        frame_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_frame,
                text="Frame Size:",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=15,
                anchor="w").pack(side="left")

        self.frame_size_var = tk.StringVar(value="2048")
        frame_options = ["1024", "2048", "4096", "8192"]

        def create_dropdown_menu(parent, variable, options):
            menu = tk.OptionMenu(parent, variable, *options)
            menu.config(font=FONT_TEXT,
                       bg=COLOR_DARK,
                       fg=COLOR_TEXT,
                       activebackground=COLOR_FRAME,
                       activeforeground=COLOR_GOLD,
                       highlightthickness=0,
                       width=15)
            menu["menu"].config(bg=COLOR_DARK,
                               fg=COLOR_TEXT,
                               activebackground=COLOR_FRAME,
                               activeforeground=COLOR_GOLD)
            return menu

        frame_dropdown = create_dropdown_menu(frame_frame, self.frame_size_var, frame_options)
        frame_dropdown.pack(side="left", padx=5)

        # Hop size setting
        hop_frame = tk.Frame(params_frame, bg=COLOR_BG)
        hop_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(hop_frame,
                text="Hop Size:",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=15,
                anchor="w").pack(side="left")

        self.hop_size_var = tk.StringVar(value="512")
        hop_options = ["256", "512", "1024", "2048"]
        hop_dropdown = create_dropdown_menu(hop_frame, self.hop_size_var, hop_options)
        hop_dropdown.pack(side="left", padx=5)

        # Violin range frame
        range_frame = tk.LabelFrame(parent_frame,
                                  text="Violin Pitch Range",
                                  font=FONT_SECTION,
                                  fg=COLOR_GOLD,
                                  bg=COLOR_BG,
                                  relief=tk.RIDGE,
                                  borderwidth=2)
        range_frame.pack(fill="x", padx=20, pady=10, ipadx=10, ipady=10)

        # Minimum frequency
        min_frame = tk.Frame(range_frame, bg=COLOR_BG)
        min_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(min_frame,
                text="Min Frequency (Hz):",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=20,
                anchor="w").pack(side="left")

        self.min_freq_var = tk.DoubleVar(value=196.0)  # G3
        min_spin = tk.Spinbox(min_frame,
                             from_=100.0,
                             to=500.0,
                             textvariable=self.min_freq_var,
                             font=FONT_TEXT,
                             bg=COLOR_DARK,
                             fg=COLOR_TEXT,
                             insertbackground=COLOR_TEXT,
                             relief=tk.SUNKEN,
                             borderwidth=1,
                             width=10,
                             increment=1.0)
        min_spin.pack(side="left", padx=5)

        # Maximum frequency
        max_frame = tk.Frame(range_frame, bg=COLOR_BG)
        max_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(max_frame,
                text="Max Frequency (Hz):",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=20,
                anchor="w").pack(side="left")

        self.max_freq_var = tk.DoubleVar(value=2637.0)  # E7
        max_spin = tk.Spinbox(max_frame,
                             from_=1000.0,
                             to=5000.0,
                             textvariable=self.max_freq_var,
                             font=FONT_TEXT,
                             bg=COLOR_DARK,
                             fg=COLOR_TEXT,
                             insertbackground=COLOR_TEXT,
                             relief=tk.SUNKEN,
                             borderwidth=1,
                             width=10,
                             increment=10.0)
        max_spin.pack(side="left", padx=5)

        # Info panel
        info_text = """🎯 YIN Algorithm Tips:
• Lower threshold = more sensitive detection
• Higher threshold = fewer false positives
• Frame size affects frequency resolution
• Smaller hop size = more temporal resolution
• Violin range: G3 (196Hz) to E7 (2637Hz)"""

        info_label = tk.Label(parent_frame,
                            text=info_text,
                            font=FONT_SMALL,
                            fg=COLOR_TEXT,
                            bg=COLOR_BG,
                            justify=tk.LEFT)
        info_label.pack(fill="x", padx=20, pady=10)

    def create_action_buttons(self):
        """Create action buttons at bottom."""
        actions_frame = tk.Frame(self.main_frame, bg=COLOR_BG)
        actions_frame.pack(fill="x", padx=10, pady=10)

        # Analyze button
        analyze_btn = tk.Button(actions_frame,
                              text="🔍 Analyze Pitch",
                              font=FONT_TEXT,
                              fg=COLOR_GOLD,
                              bg=COLOR_DARK,
                              activeforeground=COLOR_GOLD,
                              activebackground=COLOR_FRAME,
                              relief=tk.RAISED,
                              borderwidth=2,
                              padx=20,
                              pady=5,
                              command=self.start_analysis)
        analyze_btn.pack(side="left", padx=5)

        # Export button
        export_btn = tk.Button(actions_frame,
                             text="💾 Export Results",
                             font=FONT_TEXT,
                             fg=COLOR_GOLD,
                             bg=COLOR_DARK,
                             activeforeground=COLOR_GOLD,
                             activebackground=COLOR_FRAME,
                             relief=tk.RAISED,
                             borderwidth=2,
                             padx=20,
                             pady=5,
                             command=self.export_results)
        export_btn.pack(side="left", padx=5)

        # Clear button
        clear_btn = tk.Button(actions_frame,
                            text="🗑️ Clear Analysis",
                            font=FONT_TEXT,
                            fg=COLOR_GOLD,
                            bg=COLOR_DARK,
                            activeforeground=COLOR_GOLD,
                            activebackground=COLOR_FRAME,
                            relief=tk.RAISED,
                            borderwidth=2,
                            padx=20,
                            pady=5,
                            command=self.clear_analysis)
        clear_btn.pack(side="left", padx=5)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        progress_bar = ttk.Progressbar(actions_frame,
                                      variable=self.progress_var,
                                      maximum=100.0,
                                      length=100)
        progress_bar.pack(side="left", padx=10)

        # Status label
        self.status_var = tk.StringVar(value="✅ Ready")
        status_label = tk.Label(actions_frame,
                              textvariable=self.status_var,
                              font=FONT_SMALL,
                              fg=COLOR_TEXT,
                              bg=COLOR_BG)
        status_label.pack(side="right", padx=5)

    def browse_audio_file(self):
        """Browse for WAV audio file."""
        filetypes = [
            ("WAV files", "*.wav *.WAV"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=filetypes
        )

        if filename:
            self.file_path_var.set(os.path.basename(filename))
            self.load_audio_file(filename)

    def load_audio_file(self, filepath):
        """Load and display audio file information."""
        try:
            # Load audio file
            detector = YinPitchDetector()
            sample_rate, audio_data = detector.load_wav_file(filepath)

            self.sample_rate = sample_rate
            self.audio_data = audio_data
            self.current_file = filepath

            # Update file info
            duration = len(audio_data) / sample_rate
            file_info = f"✓ Loaded: {duration:.2f}s, {sample_rate}Hz, {len(audio_data)} samples"
            self.file_info_var.set(file_info)

            # Update waveform display
            self.draw_waveform(audio_data)

            self.status_var.set(f"✅ Loaded {os.path.basename(filepath)}")

        except Exception as e:
            messagebox.showerror("Load Error", f"Error loading audio file:\n{str(e)}")
            self.status_var.set("❌ Load failed")

    def record_audio(self):
        """Placeholder for audio recording functionality."""
        messagebox.showinfo("Coming Soon",
                          "Audio recording feature will be implemented in a future update.\n"
                          "For now, please use WAV files.")

    def draw_waveform(self, audio_data):
        """Draw audio waveform on canvas."""
        self.waveform_canvas.delete("all")

        if audio_data is None or len(audio_data) == 0:
            return

        canvas_width = self.waveform_canvas.winfo_width()
        canvas_height = self.waveform_canvas.winfo_height()

        if canvas_width <= 0 or canvas_height <= 0:
            return

        # Normalize and downsample for display
        display_samples = min(len(audio_data), 1000)
        indices = np.linspace(0, len(audio_data) - 1, display_samples).astype(int)
        display_data = audio_data[indices]

        # Normalize to canvas height
        max_val = np.abs(display_data).max()
        if max_val > 0:
            display_data = display_data / max_val

        # Draw waveform
        points = []
        for i, val in enumerate(display_data):
            x = (i / display_samples) * canvas_width
            y = canvas_height / 2 - (val * (canvas_height / 2.5))
            points.extend([x, y])

        if len(points) >= 4:
            self.waveform_canvas.create_line(points,
                                           fill=COLOR_GOLD,
                                           width=1,
                                           smooth=True)

        # Draw center line
        self.waveform_canvas.create_line(0, canvas_height/2,
                                       canvas_width, canvas_height/2,
                                       fill=COLOR_FRAME,
                                       width=0.5,
                                       dash=(2, 2))

    def start_analysis(self):
        """Start pitch analysis in a separate thread."""
        if self.audio_data is None:
            messagebox.showwarning("No Audio", "Please load an audio file first.")
            return

        if self.is_analyzing:
            return

        # Update parameters
        self.is_analyzing = True
        self.status_var.set("⏳ Analyzing pitch...")
        self.progress_var.set(0.0)

        # Start analysis in thread
        thread = threading.Thread(target=self.perform_analysis)
        thread.daemon = True
        thread.start()

    def perform_analysis(self):
        """Perform pitch analysis (called in thread)."""
        try:
            # Create detector with current parameters
            detector = YinPitchDetector(
                sample_rate=self.sample_rate,
                frame_size=int(self.frame_size_var.get()),
                hop_size=int(self.hop_size_var.get())
            )

            # Set thresholds
            detector.set_parameters(
                threshold=self.threshold_var.get(),
                absolute_threshold=self.abs_threshold_var.get()
            )

            # Set frequency range
            detector.min_freq = self.min_freq_var.get()
            detector.max_freq = self.max_freq_var.get()

            # Perform analysis
            results = detector.analyze_audio(
                self.audio_data,
                progress_callback=lambda p: self.progress_var.set(p * 100)
            )

            # Update UI in main thread
            self.parent.after(0, lambda: self.display_results(results, detector))

        except Exception as e:
            self.parent.after(0, lambda: messagebox.showerror(
                "Analysis Error", f"Error during analysis:\n{str(e)}"))
            self.parent.after(0, lambda: self.status_var.set("❌ Analysis failed"))
        finally:
            self.is_analyzing = False

    def display_results(self, results, detector):
        """Display analysis results."""
        if "error" in results:
            messagebox.showerror("Analysis Error", results["error"])
            self.status_var.set("❌ Analysis failed")
            return

        self.current_results = results
        self.current_detector = detector

        # Clear and update results text
        self.results_text.delete(1.0, tk.END)

        # Format and display results
        self.results_text.insert(tk.END, "PITCH ANALYSIS RESULTS\n")
        self.results_text.insert(tk.END, "=" * 40 + "\n\n")

        self.results_text.insert(tk.END, "📊 STATISTICS\n")
        self.results_text.insert(tk.END, "-" * 30 + "\n")
        self.results_text.insert(tk.END, f"Audio duration: {results['analysis_length']:.2f}s\n")
        self.results_text.insert(tk.END, f"Analysis frames: {results['num_frames']}\n")
        self.results_text.insert(tk.END, f"Pitch detected frames: {results['pitch_detected_frames']}\n")
        self.results_text.insert(tk.END, f"Detection rate: {results['detection_rate']:.1f}%\n")
        self.results_text.insert(tk.END, f"Average frequency: {results['average_frequency']:.1f} Hz\n")
        self.results_text.insert(tk.END, f"Average confidence: {results['average_confidence']:.3f}\n")
        self.results_text.insert(tk.END, f"Frequency std: {results['frequency_std']:.1f} Hz\n\n")

        self.results_text.insert(tk.END, "🎻 VIOLIN-SPECIFIC\n")
        self.results_text.insert(tk.END, "-" * 30 + "\n")
        self.results_text.insert(tk.END, f"Most common note: {results['most_common_note']}\n")
        self.results_text.insert(tk.END, f"Intonation accuracy: {results['intonation_accuracy']:.1f}%\n")
        self.results_text.insert(tk.END, f"Avg cents deviation: {results['average_cents_deviation']:.1f}\n")
        self.results_text.insert(tk.END, f"Frequency range: {results['frequency_range']['min']:.1f} - "
                                       f"{results['frequency_range']['max']:.1f} Hz "
                                       f"(range: {results['frequency_range']['range']:.1f} Hz)\n\n")

        # Sample pitch contour (first 10 frames)
        self.results_text.insert(tk.END, "🎵 SAMPLE PITCH CONTOUR\n")
        self.results_text.insert(tk.END, "-" * 30 + "\n")

        sample_frames = min(10, results['num_frames'])
        for i in range(sample_frames):
            time_str = f"{results['times'][i]:.2f}s"
            freq = results['frequencies'][i]
            if freq > 0:
                midi = detector.freq_to_midi(freq)
                note = detector.midi_to_note_name(midi)
                conf = results['confidences'][i]
                line = f"{time_str}: {freq:.1f} Hz ({note}), conf: {conf:.3f}\n"
            else:
                line = f"{time_str}: No pitch detected\n"
            self.results_text.insert(tk.END, line)

        # Update statistics display
        stats_text = (f"Detection: {results['detection_rate']:.1f}% | "
                     f"Avg conf: {results['average_confidence']:.3f} | "
                     f"Range: {results['frequency_range']['range']:.1f} Hz")
        self.stats_var.set(stats_text)

        # Update intonation display
        self.intonation_var.set(f"Intonation accuracy: {results['intonation_accuracy']:.1f}%")
        self.cents_var.set(f"{results['average_cents_deviation']:.1f}")

        # Draw deviation indicator
        self.draw_deviation_indicator(results['average_cents_deviation'])

        # Update visualizations
        self.update_visualizations()

        self.status_var.set("✅ Analysis complete")
        self.progress_var.set(100.0)

    def draw_deviation_indicator(self, cents_deviation):
        """Draw cents deviation indicator."""
        self.deviation_canvas.delete("all")

        canvas_width = self.deviation_canvas.winfo_width()
        canvas_height = self.deviation_canvas.winfo_height()

        if canvas_width <= 0 or canvas_height <= 0:
            return

        # Draw scale from -100 to +100 cents
        center_x = canvas_width / 2

        # Draw background
        self.deviation_canvas.create_rectangle(0, 0, canvas_width, canvas_height,
                                             fill=COLOR_DARK, outline="")

        # Draw tick marks
        for cents in [-100, -50, 0, 50, 100]:
            x = center_x + (cents / 100) * (canvas_width / 2)
            self.deviation_canvas.create_line(x, 5, x, canvas_height - 5,
                                            fill=COLOR_FRAME, width=1)
            self.deviation_canvas.create_text(x, canvas_height - 2,
                                            text=f"{cents:+d}",
                                            fill=COLOR_TEXT,
                                            font=("Arial", 8),
                                            anchor="n")

        # Draw center line
        self.deviation_canvas.create_line(center_x, 0, center_x, canvas_height,
                                        fill=COLOR_GOLD, width=2)

        # Draw deviation indicator
        if abs(cents_deviation) <= 100:
            x_pos = center_x + (cents_deviation / 100) * (canvas_width / 2)

            # Color based on deviation
            if abs(cents_deviation) < 10:
                color = "#90EE90"  # Green (excellent)
            elif abs(cents_deviation) < 30:
                color = "#FFD700"  # Gold (good)
            elif abs(cents_deviation) < 50:
                color = "#FFA500"  # Orange (acceptable)
            else:
                color = "#FF6B6B"  # Red (poor)

            # Draw indicator
            radius = 6
            self.deviation_canvas.create_oval(x_pos - radius, canvas_height/2 - radius,
                                            x_pos + radius, canvas_height/2 + radius,
                                            fill=color, outline="white", width=2)

    def update_visualizations(self):
        """Update all visualizations."""
        if self.current_results is None:
            return

        self.draw_pitch_contour()
        self.draw_confidence_plot()

    def draw_pitch_contour(self):
        """Draw pitch contour plot."""
        self.pitch_canvas.delete("all")

        canvas_width = self.pitch_canvas.winfo_width()
        canvas_height = self.pitch_canvas.winfo_height()

        if canvas_width <= 0 or canvas_height <= 0:
            return

        results = self.current_results

        # Convert times to x-coordinates
        times = np.array(results['times'])
        freqs = np.array(results['frequencies'])

        if len(times) == 0 or np.all(freqs == 0):
            return

        # Filter valid frequencies
        valid_mask = freqs > 0
        if not np.any(valid_mask):
            return

        valid_times = times[valid_mask]
        valid_freqs = freqs[valid_mask]

        # Normalize to canvas
        time_range = times[-1] - times[0] if len(times) > 1 else 1.0
        freq_range = results['frequency_range']['max'] - results['frequency_range']['min']
        if freq_range == 0:
            freq_range = 100.0  # Default range

        # Draw grid and axes
        self.draw_pitch_grid(canvas_width, canvas_height,
                           time_range, results['frequency_range'])

        # Draw pitch contour
        points = []
        for t, f in zip(valid_times, valid_freqs):
            x = (t - times[0]) / time_range * canvas_width
            y = canvas_height - ((f - results['frequency_range']['min']) / freq_range * canvas_height)
            points.extend([x, y])

        if len(points) >= 4:
            self.pitch_canvas.create_line(points,
                                        fill=COLOR_GOLD,
                                        width=2,
                                        smooth=True)

        # Draw note names at appropriate positions
        if self.display_var.get() == "both":
            self.draw_note_names(canvas_width, canvas_height,
                               time_range, results['frequency_range'],
                               valid_times, valid_freqs)

    def draw_pitch_grid(self, width, height, time_range, freq_range):
        """Draw grid and axes for pitch plot."""
        # Draw background
        self.pitch_canvas.create_rectangle(0, 0, width, height,
                                         fill=COLOR_DARK, outline="")

        # Draw horizontal lines (frequency grid)
        for freq in np.arange(200, 3000, 100):
            if freq_range['min'] <= freq <= freq_range['max']:
                y = height - ((freq - freq_range['min']) /
                            (freq_range['max'] - freq_range['min']) * height)
                self.pitch_canvas.create_line(0, y, width, y,
                                            fill=COLOR_FRAME, width=0.5, dash=(2, 2))

                # Label every 200 Hz
                if freq % 200 == 0:
                    self.pitch_canvas.create_text(5, y,
                                                text=f"{freq} Hz",
                                                fill=COLOR_TEXT,
                                                font=FONT_SMALL,
                                                anchor="w")

        # Draw vertical lines (time grid)
        for t in np.arange(0, time_range + 0.5, 0.5):
            x = (t / time_range) * width if time_range > 0 else 0
            self.pitch_canvas.create_line(x, 0, x, height,
                                        fill=COLOR_FRAME, width=0.5)

            # Label every second
            if t % 1.0 == 0:
                self.pitch_canvas.create_text(x, height - 5,
                                            text=f"{t:.1f}s",
                                            fill=COLOR_TEXT,
                                            font=FONT_SMALL,
                                            anchor="n")

    def draw_note_names(self, width, height, time_range, freq_range, times, freqs):
        """Draw note names on pitch contour."""
        detector = self.current_detector

        # Sample points for labeling
        sample_indices = np.linspace(0, len(times) - 1, min(20, len(times))).astype(int)

        for idx in sample_indices:
            t = times[idx]
            f = freqs[idx]

            x = (t - self.current_results['times'][0]) / time_range * width
            y = height - ((f - freq_range['min']) /
                         (freq_range['max'] - freq_range['min']) * height)

            midi = detector.freq_to_midi(f)
            note = detector.midi_to_note_name(midi)

            self.pitch_canvas.create_text(x, y - 10,
                                        text=note,
                                        fill=COLOR_TEXT,
                                        font=("Arial", 8),
                                        anchor="s")

    def draw_confidence_plot(self):
        """Draw confidence plot."""
        self.confidence_canvas.delete("all")

        canvas_width = self.confidence_canvas.winfo_width()
        canvas_height = self.confidence_canvas.winfo_height()

        if canvas_width <= 0 or canvas_height <= 0:
            return

        results = self.current_results

        # Draw background
        self.confidence_canvas.create_rectangle(0, 0, canvas_width, canvas_height,
                                              fill=COLOR_DARK, outline="")

        # Draw confidence bars
        times = np.array(results['times'])
        confidences = np.array(results['confidences'])

        if len(times) == 0:
            return

        time_range = times[-1] - times[0] if len(times) > 1 else 1.0

        bar_width = max(1, canvas_width / len(times))

        for i, (t, conf) in enumerate(zip(times, confidences)):
            if conf > 0:
                x = (t - times[0]) / time_range * canvas_width
                bar_height = conf * canvas_height

                # Color based on confidence
                if conf > 0.8:
                    color = "#90EE90"  # Green
                elif conf > 0.6:
                    color = "#FFD700"  # Gold
                elif conf > 0.4:
                    color = "#FFA500"  # Orange
                else:
                    color = "#FF6B6B"  # Red

                self.confidence_canvas.create_rectangle(
                    x, canvas_height - bar_height,
                    x + bar_width, canvas_height,
                    fill=color, outline=""
                )

        # Draw average confidence line
        avg_conf = results['average_confidence']
        if avg_conf > 0:
            y = canvas_height - (avg_conf * canvas_height)
            self.confidence_canvas.create_line(0, y, canvas_width, y,
                                             fill=COLOR_GOLD, width=2, dash=(3, 3))

            self.confidence_canvas.create_text(canvas_width - 5, y,
                                             text=f"Avg: {avg_conf:.3f}",
                                             fill=COLOR_GOLD,
                                             font=FONT_SMALL,
                                             anchor="e")

    def export_results(self):
        """Export analysis results to JSON file."""
        if self.current_results is None:
            messagebox.showwarning("No Results", "Please analyze audio first.")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Analysis Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if filename:
            try:
                success, error = self.current_detector.export_results(
                    self.current_results, filename)

                if success:
                    messagebox.showinfo("Success",
                                      f"Results saved to:\n{filename}")
                    self.status_var.set(f"✅ Exported to {os.path.basename(filename)}")
                else:
                    messagebox.showerror("Export Error",
                                       f"Failed to save:\n{error}")

            except Exception as e:
                messagebox.showerror("Export Error",
                                   f"Error saving file:\n{str(e)}")

    def clear_analysis(self):
        """Clear analysis results."""
        self.current_results = None
        self.results_text.delete(1.0, tk.END)
        self.stats_var.set("No analysis results yet")
        self.intonation_var.set("Intonation accuracy: --")
        self.cents_var.set("--")
        self.deviation_canvas.delete("all")
        self.pitch_canvas.delete("all")
        self.confidence_canvas.delete("all")
        self.progress_var.set(0.0)
        self.status_var.set("✅ Ready")

def get_content():
    """Return module description for display in GUI."""
    return [
        "╔══════════════════════════════════════╗",
        "║       YIN PITCH DETECTOR MODULE      ║",
        "╚══════════════════════════════════════╝",
        "",
        "🎯 PURPOSE: Advanced pitch detection for violin using YIN algorithm",
        "",
        "🔧 FEATURES:",
        "  • YIN (YIN is not) fundamental frequency estimation",
        "  • Violin-specific frequency range (G3-E7)",
        "  • Real-time pitch contour visualization",
        "  • Intonation accuracy analysis",
        "  • Cents deviation measurement",
        "  • Confidence scoring for each detection",
        "",
        "📊 ANALYSIS CAPABILITIES:",
        "  • Pitch tracking over time",
        "  • Most common note identification",
        "  • Intonation accuracy percentage",
        "  • Average cents deviation",
        "  • Detection confidence metrics",
        "  • Frequency range statistics",
        "",
        "🎨 VISUALIZATION:",
        "  • Pitch contour plot with note names",
        "  • Confidence level display",
        "  • Audio waveform preview",
        "  • Cents deviation indicator",
        "  • Real-time analysis feedback",
        "",
        "⚙️ CONFIGURABLE PARAMETERS:",
        "  • YIN threshold (0.01-0.5)",
        "  • Absolute threshold (0.01-0.3)",
        "  • Frame size (1024-8192 samples)",
        "  • Hop size (256-2048 samples)",
        "  • Violin frequency range",
        "",
        "🚀 QUICK START:",
        "  1. Load a WAV file of violin playing",
        "  2. Adjust parameters if needed",
        "  3. Click 'Analyze Pitch'",
        "  4. View pitch contour and statistics",
        "  5. Export results for further analysis",
    ]

def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    # Create the pitch detector GUI
    gui = YinPitchDetectorGUI(parent)
    return gui.main_frame

def test_standalone():
    """Test the module GUI standalone."""
    root = tk.Tk()
    root.title("YIN Pitch Detector Module - Standalone Test")
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