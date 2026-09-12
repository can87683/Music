#!/usr/bin/env python3
# ai_violin_master_01_audio_analyzer.py

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import traceback
import os
import time
import threading
import sys


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
        """Check and report missing dependencies."""
        required_deps = [
            ("librosa", "librosa"),
            ("pretty_midi", "pretty_midi"),
            ("numpy", "numpy")
        ]

        for import_name, package_name in required_deps:
            try:
                __import__(import_name)
            except ImportError:
                self.missing_deps.append(package_name)
                self.deps_ok = False

    def analyze_audio(self, audio_path: str, midi_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        High-level audio analysis: vibrato, intonation, timbre, timing deviation.

        Args:
            audio_path: Path to audio file
            midi_data: PrettyMIDI object or None

        Returns:
            Dictionary with analysis results
        """
        # Check dependencies
        if not self.deps_ok:
            return {
                "error": f"Missing dependencies: {', '.join(self.missing_deps)}",
                "suggestion": "Install with: pip install " + " ".join(self.missing_deps)
            }

        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            return {"error": f"Audio loading failed: {e}", "traceback": traceback.format_exc()}

        # Pitch contour (YIN with GPU if available)
        f0_contour = self._get_pitch_contour(y, sr)
        f0_smooth = self._smooth_pitch(f0_contour)

        # Audio-MIDI alignment
        alignment = self._align_audio_midi(y, sr, midi_data)

        # Vibrato metrics
        vibrato_data = self._analyze_vibrato(f0_smooth, sr)

        # Spectral timbre
        timbre_data = self._timbre_profile(y, sr)

        # Intonation (vs aligned MIDI)
        intonation_curve = self._intonation_vs_midi(f0_smooth, alignment, midi_data)

        return {
            "pitch_contour_smoothed": f0_smooth.tolist() if hasattr(f0_smooth, 'tolist') else list(f0_smooth),
            "vibrato": vibrato_data,
            "timbre": timbre_data,
            "alignment": alignment,
            "intonation_curve": intonation_curve,
            "sample_rate": sr,
            "audio_length_seconds": len(y) / sr if sr > 0 else 0
        }

    def _get_pitch_contour(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Get pitch contour with fallbacks."""
        if self.yin_pitch:
            try:
                return self.yin_pitch.yin_pitch_track(y, sr, hop=512, win=2048)
            except Exception:
                pass

        # Fallback: simple pitch detection
        try:
            import librosa
            return librosa.yin(y, fmin=65, fmax=2093, sr=sr, hop_length=512)
        except Exception:
            # Return dummy data
            n_frames = len(y) // 512
            return np.ones(n_frames) * 440.0

    def _smooth_pitch(self, f0_contour: np.ndarray) -> np.ndarray:
        """Smooth pitch contour."""
        if self.kalman:
            try:
                return self.kalman.smooth_array(f0_contour)
            except Exception:
                pass

        # Simple moving average fallback
        window = 5
        if len(f0_contour) > window:
            return np.convolve(f0_contour, np.ones(window)/window, mode='valid')
        return f0_contour

    def _align_audio_midi(self, y: np.ndarray, sr: int, midi_data: Optional[Any]) -> Dict[str, Any]:
        """Align audio to MIDI using onset detection."""
        try:
            import librosa

            if midi_data is None:
                return {
                    "map": [],
                    "average_timing_offset": 0.0,
                    "timing_label": "No MIDI provided",
                    "note_count_aligned": 0,
                    "status": "No MIDI"
                }

            # Get audio onsets
            audio_onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")

            # Get MIDI onsets
            midi_onsets = []
            try:
                for inst in midi_data.instruments:
                    for n in inst.notes:
                        midi_onsets.append(n.start)
            except AttributeError:
                return {
                    "map": [],
                    "average_timing_offset": 0.0,
                    "timing_label": "Invalid MIDI data",
                    "note_count_aligned": 0,
                    "status": "Error"
                }

            midi_onsets = np.array(sorted(midi_onsets))

            # Simple DTW-based matching
            if len(audio_onsets) == 0 or len(midi_onsets) == 0:
                return {
                    "map": [],
                    "average_timing_offset": 0.0,
                    "timing_label": "No onsets detected",
                    "note_count_aligned": 0,
                    "status": "No onsets"
                }

            D, wp = librosa.sequence.dtw(
                np.expand_dims(audio_onsets, 0),
                np.expand_dims(midi_onsets, 0)
            )

            # Create mapping from warping path
            mapping = []
            for a_idx, m_idx in wp:
                if m_idx < len(midi_onsets) and a_idx < len(audio_onsets):
                    mapping.append((float(audio_onsets[a_idx]), float(midi_onsets[m_idx])))

            # Calculate timing drift
            diffs = [a - m for a, m in mapping]
            avg_drift = float(np.mean(diffs)) if diffs else 0.0

            # Label based on drift
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
                "status": "Success"
            }
        except Exception as e:
            return {
                "error": f"Alignment failed: {e}",
                "traceback": traceback.format_exc(),
                "status": "Error"
            }

    def _analyze_vibrato(self, f0: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze vibrato rate, depth, consistency."""
        try:
            # Filter out zeros and negative values
            f0 = np.array([x for x in f0 if x > 0])

            if len(f0) < 20:
                return {
                    "rate_hz": 0.0,
                    "depth_cents": 0.0,
                    "stability": 0.0,
                    "segments": 0,
                    "quality": "Insufficient data",
                    "status": "Insufficient data"
                }

            # Remove slow drift using high-pass filter
            f0_diff = np.diff(f0)
            hp = f0_diff - np.mean(f0_diff)

            # FFT for vibrato frequency detection
            spectrum = np.fft.rfft(hp)
            freqs = np.fft.rfftfreq(len(hp), d=1.0/(sr/512))

            peak_idx = np.argmax(np.abs(spectrum[1:])) + 1
            vibrato_rate = freqs[peak_idx] if peak_idx < len(freqs) else 0.0

            # Vibrato depth in cents
            ref = np.median(f0)
            cents = 1200 * np.log2(f0 / ref) if ref > 0 else np.zeros_like(f0)
            depth = np.std(cents) * 2 if len(cents) > 0 else 0.0

            # Stability metric
            stability = 1.0 / (1.0 + np.std(hp)) if np.std(hp) > 0 else 0.0

            # Quality assessment
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
                "status": "Success"
            }
        except Exception as e:
            return {
                "rate_hz": 0.0,
                "depth_cents": 0.0,
                "stability": 0.0,
                "segments": 0,
                "error": str(e),
                "status": "Failed"
            }

    def _intonation_vs_midi(self, f0_smooth: np.ndarray, alignment: Dict[str, Any],
                           midi_data: Optional[Any]) -> List[Tuple[float, float]]:
        """Compare smoothed pitch curve to expected MIDI pitches."""
        if not midi_data or "map" not in alignment or len(alignment["map"]) == 0:
            return []

        try:
            # Extract expected MIDI pitches at alignment times
            expected_pitches = []
            for _, midi_t in alignment["map"]:
                expected_pitches.append(self._midi_pitch_at_time(midi_data, midi_t))

            intonation_curve = []
            map_items = list(zip(alignment["map"], f0_smooth[:len(expected_pitches)], expected_pitches))

            for (audio_t, _), f0, midi_p in map_items:
                if f0 <= 0 or midi_p <= 0:
                    continue
                try:
                    cents_off = 1200 * np.log2(f0 / midi_p)
                    intonation_curve.append((float(audio_t), float(cents_off)))
                except (ValueError, ZeroDivisionError):
                    continue

            return intonation_curve
        except Exception:
            return []

    def _midi_pitch_at_time(self, midi_data: Any, time: float) -> float:
        """Find expected MIDI frequency at given time."""
        try:
            import librosa
            # Find the active note at this time
            for inst in midi_data.instruments:
                for note in inst.notes:
                    if note.start <= time <= note.end:
                        return librosa.midi_to_hz(note.pitch)
            return 440.0  # Default to A4 if no note found
        except Exception:
            return 440.0

    def _timbre_profile(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Compute violin tone quality metrics."""
        try:
            import librosa

            # Basic spectral features
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            zcr = librosa.feature.zero_crossing_rate(y)

            # Harmonic-percussive separation
            y_h, y_p = librosa.effects.hpss(y)
            harmonic_energy = float(np.mean(np.abs(y_h)))
            noise_energy = float(np.mean(np.abs(y_p)))
            hnr = harmonic_energy / (noise_energy + 1e-6)

            # Spectral analysis
            S = np.abs(librosa.stft(y, n_fft=4096))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)

            # Bow noise (6-10k Hz)
            bow_band = (freqs >= 6000) & (freqs <= 10000)
            bow_noise = float(np.mean(S[bow_band])) if np.any(bow_band) else 0.0

            # Brightness (>4k Hz)
            bright_band = (freqs > 4000)
            brightness = float(np.mean(S[bright_band])) if np.any(bright_band) else 0.0

            # Resonance (280-400 Hz)
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
                "status": "Success"
            }
        except Exception as e:
            return {
                "error": f"Timbre analysis failed: {e}",
                "traceback": traceback.format_exc(),
                "status": "Failed"
            }

    def _tone_label(self, hnr: float, brightness: float,
                   bow_noise: float, resonance: float) -> str:
        """Classify tone quality."""
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

    def get_module_info(self) -> List[str]:
        """Return module documentation."""
        deps_status = "✅ All dependencies installed" if self.deps_ok else f"❌ Missing: {', '.join(self.missing_deps)}"

        return [
            "╔══════════════════════════════════════╗",
            "║      AUDIO ANALYZER MODULE           ║",
            "╚══════════════════════════════════════╝",
            "",
            f"DEPENDENCIES: {deps_status}",
            "",
            "🎯 PURPOSE: Analyze audio for vibrato, intonation, timbre, and alignment",
            "",
            "ANALYSIS FEATURES:",
            "  • Pitch contour extraction",
            "  • Vibrato rate and depth detection",
            "  • Spectral timbre profiling",
            "  • Audio-MIDI alignment",
            "  • Intonation curve generation",
            "  • Timing deviation detection",
            "  • Tone quality classification",
            "",
            "METHODOLOGY:",
            "  - YIN pitch detection with smoothing",
            "  - FFT-based vibrato analysis (4-8 Hz typical)",
            "  - Spectral centroid/rolloff for timbre",
            "  - DTW alignment for audio-MIDI sync",
            "  - Cents deviation calculation vs tempered scale",
            "",
            "TONE CLASSIFICATIONS:",
            "  • Warm/Focused: High harmonicity, low brightness",
            "  • Bright/Clear: High brightness, low noise",
            "  • Harsh/Noisy: High bow noise levels",
            "  • Airy/Weak: Low harmonic noise ratio",
            "  • Resonant/Full: Strong body resonance",
            "",
            "APPLICATIONS:",
            "  • Live performance monitoring",
            "  • Recording analysis",
            "  • Intonation correction studies",
            "  • Tone quality assessment",
            "  • Practice feedback system"
        ]


class AudioAnalyzerGUI:
    """Standalone GUI for Audio Analyzer module."""

    def __init__(self, *args, **kwargs):
        """
        Initialize the GUI.

        Args:
            Can be called with:
            - No arguments (creates standalone window)
            - One argument (root window)
            - Two arguments (root window, analyzer instance)
        """
        # Handle different calling patterns
        if len(args) == 0:
            # No arguments: create standalone window
            self.root = tk.Tk()
            self.root.title("🎵 Audio Analyzer")
            self.root.geometry("640x800")
            self.root.resizable(False, False)
            self.is_standalone = True
            self.analyzer = AudioAnalyzer()
        elif len(args) == 1:
            # One argument: root window provided
            self.root = args[0]
            self.is_standalone = False
            self.analyzer = AudioAnalyzer()
        elif len(args) == 2:
            # Two arguments: root window and analyzer provided
            self.root = args[0]
            self.analyzer = args[1]
            self.is_standalone = False
        else:
            # Too many arguments - use the first two
            self.root = args[0]
            self.analyzer = args[1] if len(args) > 1 else AudioAnalyzer()
            self.is_standalone = False

        # Also check kwargs
        if 'analyzer' in kwargs:
            self.analyzer = kwargs['analyzer']

        # COLORS - Matching your 2nd, 3rd, 4th scripts
        self.COLOR_BG = "#5A381E"           # Deep wood brown (main background)
        self.COLOR_DARK = "#3B2413"         # Darker brown (panels, darker areas)
        self.COLOR_MEDIUM = "#4B2E18"       # Medium brown (frames, borders)
        self.COLOR_LIGHT = "#6B4E29"        # Light brown (highlights, accents)
        self.COLOR_GOLD = "#FFD770"         # Gold/yellow (text, highlights)
        self.COLOR_TEXT = "#FFF2CF"         # Cream/off-white (primary text)
        self.COLOR_TEXT_SECONDARY = "#E6D8B5"  # Lighter cream (secondary text)
        self.COLOR_ACCENT = "#00BFFF"       # Bright blue (active elements)
        self.COLOR_RED = "#FF6B6B"          # Red (errors, warnings)
        self.COLOR_GREEN = "#4CAF50"        # Green (success)
        self.COLOR_BLUE = "#5D8AA8"         # Muted blue (info)

        # FONTS - Consistent with other scripts
        self.FONT_TITLE = ("Georgia", 16, "bold")
        self.FONT_HEADER = ("Georgia", 12, "bold")
        self.FONT_SUBHEADER = ("Georgia", 11, "italic")
        self.FONT_TEXT = ("Georgia", 10)
        self.FONT_SMALL = ("Georgia", 9)
        self.FONT_MONO = ("Courier New", 9)

        # Configure root window
        self.root.configure(bg=self.COLOR_BG)
        if self.is_standalone:
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._init_variables()
        self._create_widgets()

        if self.is_standalone:
            self.root.mainloop()

    def _init_variables(self):
        """Initialize tkinter variables."""
        self.audio_path_var = tk.StringVar()
        self.midi_path_var = tk.StringVar()
        self.progress_var = tk.IntVar(value=0)
        self.vibrato_sensitivity = tk.IntVar(value=50)

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.COLOR_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        self._create_header(main_frame)

        # Notebook (tabbed interface)
        self._create_notebook(main_frame)

        # Status bar
        self._create_status_bar(main_frame)

    def _create_header(self, parent):
        """Create header section."""
        header_frame = tk.Frame(parent, bg=self.COLOR_BG)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        # Title with decorative frame
        title_frame = tk.Frame(header_frame, bg=self.COLOR_DARK, relief=tk.RAISED, borderwidth=2)
        title_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(
            title_frame,
            text="🎵 AI VIOLIN MASTER - AUDIO ANALYZER",
            font=self.FONT_TITLE,
            fg=self.COLOR_GOLD,
            bg=self.COLOR_DARK
        ).pack(pady=5)

        # Subtitle
        tk.Label(
            header_frame,
            text="Analyze vibrato, intonation, timbre & alignment",
            font=self.FONT_SUBHEADER,
            fg=self.COLOR_TEXT_SECONDARY,
            bg=self.COLOR_BG
        ).pack()

        # Separator line
        separator = tk.Frame(header_frame, height=2, bg=self.COLOR_GOLD)
        separator.pack(fill=tk.X, pady=5)

    def _create_notebook(self, parent):
        """Create tabbed interface."""
        # Create custom notebook style
        style = ttk.Style()
        style.theme_use('clam')

        # Configure notebook style
        style.configure("Custom.TNotebook",
                       background=self.COLOR_BG,
                       borderwidth=0)
        style.configure("Custom.TNotebook.Tab",
                       background=self.COLOR_DARK,
                       foreground=self.COLOR_TEXT,
                       padding=[15, 5],
                       font=self.FONT_TEXT)
        style.map("Custom.TNotebook.Tab",
                 background=[("selected", self.COLOR_MEDIUM)],
                 foreground=[("selected", self.COLOR_GOLD)])

        # Create notebook
        self.notebook = ttk.Notebook(parent, style="Custom.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # Create tabs
        self._create_file_tab()
        self._create_analysis_tab()
        self._create_vibrato_tab()
        self._create_timbre_tab()
        self._create_alignment_tab()

    def _create_file_tab(self):
        """Create file selection tab."""
        tab = tk.Frame(self.notebook, bg=self.COLOR_BG, padx=15, pady=15)
        self.notebook.add(tab, text="📁 FILES")

        # Instructions
        tk.Label(
            tab,
            text="Select audio file to analyze (MIDI optional for alignment):",
            font=self.FONT_TEXT,
            fg=self.COLOR_TEXT,
            bg=self.COLOR_BG,
            wraplength=550
        ).pack(anchor='w', pady=(0, 15))

        # File selection frame with raised border
        file_frame = tk.Frame(
            tab,
            bg=self.COLOR_DARK,
            relief=tk.RAISED,
            borderwidth=2
        )
        file_frame.pack(fill=tk.X, pady=10)

        # Title for file frame
        tk.Label(
            file_frame,
            text="FILE SELECTION",
            font=self.FONT_HEADER,
            fg=self.COLOR_GOLD,
            bg=self.COLOR_DARK
        ).pack(pady=5)

        # Audio file selection
        self._create_file_row(file_frame, "Audio File*:", self.audio_path_var,
                            [("Audio files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aac"),
                             ("WAV files", "*.wav"),
                             ("MP3 files", "*.mp3"),
                             ("All files", "*.*")],
                            self._browse_audio)

        # MIDI file selection (optional)
        self._create_file_row(file_frame, "MIDI File (optional):", self.midi_path_var,
                            [("MIDI files", "*.mid *.midi"), ("All files", "*.*")],
                            self._browse_midi)

        # Settings frame
        settings_frame = tk.Frame(
            tab,
            bg=self.COLOR_DARK,
            relief=tk.RAISED,
            borderwidth=2
        )
        settings_frame.pack(fill=tk.X, pady=10)

        # Title for settings frame
        tk.Label(
            settings_frame,
            text="ANALYSIS SETTINGS",
            font=self.FONT_HEADER,
            fg=self.COLOR_GOLD,
            bg=self.COLOR_DARK
        ).pack(pady=5)

        # Vibrato sensitivity slider
        sensitivity_frame = tk.Frame(settings_frame, bg=self.COLOR_DARK)
        sensitivity_frame.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(
            sensitivity_frame,
            text="Vibrato Sensitivity:",
            font=self.FONT_TEXT,
            fg=self.COLOR_TEXT,
            bg=self.COLOR_DARK,
            width=20,
            anchor='w'
        ).pack(side=tk.LEFT)

        self.vibrato_slider = tk.Scale(
            sensitivity_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.vibrato_sensitivity,
            bg=self.COLOR_DARK,
            fg=self.COLOR_GOLD,
            troughcolor=self.COLOR_MEDIUM,
            highlightthickness=0,
            length=250,
            sliderrelief=tk.RAISED,
            showvalue=True
        )
        self.vibrato_slider.pack(side=tk.LEFT, padx=10)
        self.vibrato_slider.set(50)

        # Analyze button frame
        button_frame = tk.Frame(tab, bg=self.COLOR_BG)
        button_frame.pack(fill=tk.X, pady=20)

        self.analyze_button = tk.Button(
            button_frame,
            text="🎵  START ANALYSIS",
            font=("Georgia", 12, "bold"),
            fg=self.COLOR_GOLD,
            bg=self.COLOR_DARK,
            activeforeground=self.COLOR_GOLD,
            activebackground=self.COLOR_MEDIUM,
            relief=tk.RAISED,
            borderwidth=3,
            padx=30,
            pady=10,
            cursor="hand2",
            command=self._analyze_audio,
            state=tk.DISABLED
        )
        self.analyze_button.pack()

        # Bind entry changes to enable/disable analyze button
        self.audio_path_var.trace_add('write', self._check_files)

    def _create_file_row(self, parent, label_text, var, filetypes, command):
        """Create a file selection row."""
        row = tk.Frame(parent, bg=self.COLOR_DARK, pady=8)
        row.pack(fill=tk.X, padx=20)

        # Label
        tk.Label(
            row,
            text=label_text,
            font=self.FONT_TEXT,
            fg=self.COLOR_TEXT,
            bg=self.COLOR_DARK,
            width=18,
            anchor='w'
        ).pack(side=tk.LEFT)

        # Entry field
        entry = tk.Entry(
            row,
            textvariable=var,
            font=self.FONT_TEXT,
            bg=self.COLOR_MEDIUM,
            fg=self.COLOR_TEXT,
            relief=tk.SUNKEN,
            borderwidth=2,
            insertbackground=self.COLOR_TEXT,
            width=35
        )
        entry.pack(side=tk.LEFT, padx=5)

        # Browse button
        browse_btn = tk.Button(
            row,
            text="Browse...",
            font=self.FONT_TEXT,
            fg=self.COLOR_GOLD,
            bg=self.COLOR_MEDIUM,
            activeforeground=self.COLOR_GOLD,
            activebackground=self.COLOR_LIGHT,
            relief=tk.RAISED,
            borderwidth=2,
            padx=15,
            pady=2,
            cursor="hand2",
            command=lambda: command(filetypes)
        )
        browse_btn.pack(side=tk.LEFT)

    def _create_analysis_tab(self):
        """Create analysis overview tab."""
        tab = tk.Frame(self.notebook, bg=self.COLOR_BG, padx=15, pady=15)
        self.notebook.add(tab, text="📊 OVERVIEW")

        # Frame for results
        results_frame = tk.Frame(
            tab,
            bg=self.COLOR_DARK,
            relief=tk.SUNKEN,
            borderwidth=2
        )
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(
            results_frame,
            text="ANALYSIS OVERVIEW",
            font=self.FONT_HEADER,
            fg=self.COLOR_GOLD,
            bg=self.COLOR_DARK
        ).pack(pady=5)

        # Analysis results text
        self.overview_text = scrolledtext.ScrolledText(
            results_frame,
            height=20,
            bg=self.COLOR_MEDIUM,
            fg=self.COLOR_TEXT,
            font=self.FONT_MONO,
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=0,
            padx=10,
            pady=10
        )
        self.overview_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.overview_text.insert(
            tk.END,
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "                         ANALYSIS OVERVIEW\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "This tab will display a comprehensive summary of all analysis\n"
            "results after processing.\n\n"
            "Expected information includes:\n"
            "• Audio file metadata\n"
            "• Key performance metrics\n"
            "• Overall assessment\n"
            "• Practice recommendations\n\n"
            "Click 'START ANALYSIS' in the FILES tab to begin.\n"
        )
        self.overview_text.config(state=tk.DISABLED)

    def _create_vibrato_tab(self):
        """Create vibrato analysis tab."""
        tab = tk.Frame(self.notebook, bg=self.COLOR_BG, padx=15, pady=15)
        self.notebook.add(tab, text="🎶 VIBRATO")

        # Frame for results
        results_frame = tk.Frame(
            tab,
            bg=self.COLOR_DARK,
            relief=tk.SUNKEN,
            borderwidth=2
        )
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(
            results_frame,
            text="VIBRATO ANALYSIS",
            font=self.FONT_HEADER,
            fg=self.COLOR_GOLD,
            bg=self.COLOR_DARK
        ).pack(pady=5)

        # Vibrato results text
        self.vibrato_results = scrolledtext.ScrolledText(
            results_frame,
            height=20,
            bg=self.COLOR_MEDIUM,
            fg=self.COLOR_TEXT,
            font=self.FONT_MONO,
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=0,
            padx=10,
            pady=10
        )
        self.vibrato_results.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.vibrato_results.insert(
            tk.END,
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "                         VIBRATO ANALYSIS\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Vibrato analysis results will appear here.\n\n"
            "Metrics analyzed:\n"
            "• Rate (Hz) - Typical violin vibrato: 4-7 Hz\n"
            "• Depth (cents) - Typical range: 20-80 cents\n"
            "• Stability score - Higher is better\n"
            "• Quality assessment - Based on rate/depth balance\n\n"
            "Interpretation and recommendations will be provided.\n"
        )
        self.vibrato_results.config(state=tk.DISABLED)

    def _create_timbre_tab(self):
        """Create timbre analysis tab."""
        tab = tk.Frame(self.notebook, bg=self.COLOR_BG, padx=15, pady=15)
        self.notebook.add(tab, text="🎨 TIMBRE")

        # Main container with two columns
        main_container = tk.Frame(tab, bg=self.COLOR_BG)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Left column - numeric metrics
        left_col = tk.Frame(main_container, bg=self.COLOR_BG)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Metrics frame
        metrics_frame = tk.Frame(
            left_col,
            bg=self.COLOR_DARK,
            relief=tk.SUNKEN,
            borderwidth=2
        )
        metrics_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            metrics_frame,
            text="SPECTRAL METRICS",
            font=self.FONT_HEADER,
            fg=self.COLOR_GOLD,
            bg=self.COLOR_DARK
        ).pack(pady=5)

        # Metrics display area
        metrics_display = tk.Frame(metrics_frame, bg=self.COLOR_DARK, padx=20, pady=20)
        metrics_display.pack(fill=tk.BOTH, expand=True)

        self.timbre_labels = {}
        metrics = [
            ("Spectral Centroid:", "centroid", "Hz"),
            ("Rolloff Frequency:", "rolloff", "Hz"),
            ("Brightness Score:", "brightness", ""),
            ("Harmonic Ratio:", "harmonicity", ""),
            ("Noise Level:", "noise", ""),
            ("Resonance:", "resonance", "")
        ]

        for text, key, unit in metrics:
            frame = tk.Frame(metrics_display, bg=self.COLOR_DARK)
            frame.pack(fill=tk.X, pady=8)

            # Metric label
            tk.Label(
                frame,
                text=text,
                font=self.FONT_TEXT,
                fg=self.COLOR_TEXT,
                bg=self.COLOR_DARK,
                width=25,
                anchor='w'
            ).pack(side=tk.LEFT)

            # Value display (initially "--")
            lbl = tk.Label(
                frame,
                text="--",
                font=self.FONT_TEXT,
                fg=self.COLOR_GOLD,
                bg=self.COLOR_MEDIUM,
                relief=tk.SUNKEN,
                borderwidth=1,
                width=15,
                anchor='center',
                padx=5,
                pady=2
            )
            lbl.pack(side=tk.LEFT, padx=5)
            self.timbre_labels[key] = lbl

            # Unit label
            if unit:
                tk.Label(
                    frame,
                    text=unit,
                    font=self.FONT_SMALL,
                    fg=self.COLOR_TEXT_SECONDARY,
                    bg=self.COLOR_DARK,
                    width=5,
                    anchor='w'
                ).pack(side=tk.LEFT)

        # Right column - tone classification
        right_col = tk.Frame(main_container, bg=self.COLOR_BG)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Tone frame
        tone_frame = tk.Frame(
            right_col,
            bg=self.COLOR_DARK,
            relief=tk.SUNKEN,
            borderwidth=2
        )
        tone_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(
            tone_frame,
            text="TONE QUALITY",
            font=self.FONT_HEADER,
            fg=self.COLOR_GOLD,
            bg=self.COLOR_DARK
        ).pack(pady=5)

        # Tone display
        tone_display = tk.Frame(tone_frame, bg=self.COLOR_DARK, padx=20, pady=20)
        tone_display.pack(fill=tk.BOTH, expand=True)

        self.tone_label = tk.Label(
            tone_display,
            text="Not Analyzed",
            font=("Georgia", 14, "bold"),
            fg=self.COLOR_TEXT_SECONDARY,
            bg=self.COLOR_MEDIUM,
            relief=tk.SUNKEN,
            borderwidth=2,
            width=20,
            height=5,
            anchor='center'
        )
        self.tone_label.pack(fill=tk.BOTH, expand=True, pady=(0, 20))

        # Description
        desc_frame = tk.Frame(tone_display, bg=self.COLOR_DARK)
        desc_frame.pack(fill=tk.X)

        tk.Label(
            desc_frame,
            text="Tone Quality Guide:",
            font=self.FONT_SUBHEADER,
            fg=self.COLOR_GOLD,
            bg=self.COLOR_DARK,
            anchor='w'
        ).pack(fill=tk.X, pady=(0, 5))

        desc_text = """• Warm/Focused: Rich, resonant, clear
• Bright/Clear: Brilliant, articulate, clean
• Harsh/Noisy: Scratchy, distorted, tense
• Airy/Weak: Thin, weak, breathy
• Resonant/Full: Powerful, projecting, rich
• Balanced/Neutral: Well-rounded, standard"""

        desc_label = tk.Label(
            desc_frame,
            text=desc_text,
            font=self.FONT_SMALL,
            fg=self.COLOR_TEXT,
            bg=self.COLOR_DARK,
            justify=tk.LEFT,
            anchor='w'
        )
        desc_label.pack(fill=tk.X)

    def _create_alignment_tab(self):
        """Create audio-MIDI alignment tab."""
        tab = tk.Frame(self.notebook, bg=self.COLOR_BG, padx=15, pady=15)
        self.notebook.add(tab, text="🎼 ALIGNMENT")

        # Frame for results
        results_frame = tk.Frame(
            tab,
            bg=self.COLOR_DARK,
            relief=tk.SUNKEN,
            borderwidth=2
        )
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        tk.Label(
            results_frame,
            text="AUDIO-MIDI ALIGNMENT",
            font=self.FONT_HEADER,
            fg=self.COLOR_GOLD,
            bg=self.COLOR_DARK
        ).pack(pady=5)

        # Alignment results text
        self.alignment_results = scrolledtext.ScrolledText(
            results_frame,
            height=20,
            bg=self.COLOR_MEDIUM,
            fg=self.COLOR_TEXT,
            font=self.FONT_MONO,
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=0,
            padx=10,
            pady=10
        )
        self.alignment_results.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.alignment_results.insert(
            tk.END,
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "                     AUDIO-MIDI ALIGNMENT\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Alignment analysis requires both audio and MIDI files.\n\n"
            "When MIDI is provided, this tab will show:\n"
            "• Timing offset (ms) - positive=rushing, negative=dragging\n"
            "• Number of notes successfully matched\n"
            "• Timing consistency assessment\n"
            "• Practice recommendations for timing\n\n"
            "Note: MIDI file is optional. Without it, only basic audio\n"
            "analysis will be performed.\n"
        )
        self.alignment_results.config(state=tk.DISABLED)

    def _create_status_bar(self, parent):
        """Create status bar at bottom."""
        status_frame = tk.Frame(parent, bg=self.COLOR_DARK, height=30, relief=tk.RAISED, borderwidth=2)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))
        status_frame.pack_propagate(False)

        # Status label (left side)
        self.status_label = tk.Label(
            status_frame,
            text="Ready - Select an audio file to begin",
            fg=self.COLOR_TEXT,
            bg=self.COLOR_DARK,
            font=self.FONT_SMALL,
            anchor='w'
        )
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Progress bar (right side)
        progress_frame = tk.Frame(status_frame, bg=self.COLOR_DARK)
        progress_frame.pack(side=tk.RIGHT, padx=10, pady=5)

        tk.Label(
            progress_frame,
            text="Progress:",
            fg=self.COLOR_TEXT_SECONDARY,
            bg=self.COLOR_DARK,
            font=self.FONT_SMALL
        ).pack(side=tk.LEFT, padx=(0, 5))

        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            length=150
        )
        self.progress_bar.pack(side=tk.LEFT)

        # Configure progress bar style
        style = ttk.Style()
        style.configure("TProgressbar",
                       background=self.COLOR_GOLD,
                       troughcolor=self.COLOR_MEDIUM,
                       bordercolor=self.COLOR_DARK,
                       lightcolor=self.COLOR_GOLD,
                       darkcolor=self.COLOR_GOLD)

    def _browse_audio(self, filetypes):
        """Browse for audio file."""
        try:
            filename = filedialog.askopenfilename(
                title="Select Audio File",
                filetypes=filetypes
            )
            if filename:
                self.audio_path_var.set(filename)
                self.status_label.config(text=f"Selected: {os.path.basename(filename)}")
        except Exception as e:
            self._show_error_popup("File Browser Error", f"Could not browse for audio file:\n{e}")

    def _browse_midi(self, filetypes):
        """Browse for MIDI file."""
        try:
            filename = filedialog.askopenfilename(
                title="Select MIDI File (optional)",
                filetypes=filetypes
            )
            if filename:
                self.midi_path_var.set(filename)
                self.status_label.config(text=f"Selected MIDI: {os.path.basename(filename)}")
        except Exception as e:
            self._show_error_popup("File Browser Error", f"Could not browse for MIDI file:\n{e}")

    def _check_files(self, *args):
        """Check if files are selected to enable analyze button."""
        try:
            audio_path = self.audio_path_var.get()
            audio_exists = os.path.exists(audio_path) if audio_path else False

            if audio_exists:
                self.analyze_button.config(state=tk.NORMAL)
                self.status_label.config(text=f"Ready to analyze: {os.path.basename(audio_path)}")
            else:
                self.analyze_button.config(state=tk.DISABLED)
                self.status_label.config(text="Select an audio file to begin")
        except Exception:
            self.analyze_button.config(state=tk.DISABLED)
            self.status_label.config(text="Invalid file path")

    def _analyze_audio(self):
        """Run audio analysis."""
        audio_path = self.audio_path_var.get()
        midi_path = self.midi_path_var.get() if self.midi_path_var.get() else None

        if not os.path.exists(audio_path):
            self._show_error_popup("File Error", "Selected audio file does not exist!")
            return

        # Check analyzer dependencies
        if not self.analyzer.deps_ok:
            missing = ", ".join(self.analyzer.missing_deps)
            self._show_error_popup(
                "Missing Dependencies",
                f"The following required packages are missing:\n\n{missing}\n\n"
                f"Install with: pip install {missing}"
            )
            return

        # Update UI
        self.analyze_button.config(state=tk.DISABLED, text="ANALYZING...")
        self.status_label.config(text="Analyzing audio... Please wait.")
        self.progress_var.set(10)

        # Clear previous results
        self._clear_results()

        # Run analysis in separate thread
        threading.Thread(
            target=self._run_analysis_thread,
            args=(audio_path, midi_path),
            daemon=True
        ).start()

    def _clear_results(self):
        """Clear all previous analysis results."""
        # Clear overview
        self.overview_text.config(state=tk.NORMAL)
        self.overview_text.delete(1.0, tk.END)
        self.overview_text.insert(tk.END, "Running analysis... Please wait.")
        self.overview_text.config(state=tk.DISABLED)

        # Clear vibrato results
        self.vibrato_results.config(state=tk.NORMAL)
        self.vibrato_results.delete(1.0, tk.END)
        self.vibrato_results.insert(tk.END, "Running vibrato analysis...")
        self.vibrato_results.config(state=tk.DISABLED)

        # Clear timbre results
        for key in self.timbre_labels:
            self.timbre_labels[key].config(text="--", fg=self.COLOR_GOLD)
        self.tone_label.config(text="Analyzing...", fg=self.COLOR_TEXT_SECONDARY)

        # Clear alignment results
        self.alignment_results.config(state=tk.NORMAL)
        self.alignment_results.delete(1.0, tk.END)
        self.alignment_results.insert(tk.END, "Running alignment analysis...")
        self.alignment_results.config(state=tk.DISABLED)

    def _run_analysis_thread(self, audio_path, midi_path):
        """Run analysis in background thread."""
        try:
            # Load MIDI file if provided
            pm = None
            if midi_path and os.path.exists(midi_path):
                try:
                    import pretty_midi
                    pm = pretty_midi.PrettyMIDI(midi_path)
                    self.root.after(0, self._log, f"Loaded MIDI: {os.path.basename(midi_path)}")
                except Exception as e:
                    self.root.after(0, self._log, f"Warning: Could not load MIDI: {e}")
                    pm = None

            # Simulate analysis progress
            for i in range(10, 100, 10):
                time.sleep(0.2)
                self.root.after(0, self.progress_var.set, i)

            # Call the actual analyzer
            analysis_result = self.analyzer.analyze_audio(audio_path, pm)

            # Update UI with results
            self.root.after(0, self._display_results, analysis_result)

        except Exception as e:
            self.root.after(0, self._analysis_error, str(e))

    def _display_results(self, analysis_result):
        """Display analysis results."""
        try:
            # Check for errors in main result
            if "error" in analysis_result:
                error_msg = analysis_result["error"]
                if "Missing dependencies" in error_msg:
                    missing = ", ".join(self.analyzer.missing_deps)
                    self._show_error_popup(
                        "Missing Dependencies",
                        f"Required packages are missing:\n\n{missing}\n\n"
                        f"Install with: pip install {missing}"
                    )
                else:
                    self._show_error_popup("Analysis Error", error_msg)
                self._analysis_error("Analysis failed")
                return

            # Update overview tab
            self._update_overview(analysis_result)

            # Update vibrato tab
            self._update_vibrato_results(analysis_result.get('vibrato', {}))

            # Update timbre tab
            self._update_timbre_results(analysis_result.get('timbre', {}))

            # Update alignment tab
            self._update_alignment_results(analysis_result.get('alignment', {}))

            # Update log
            self._log("Analysis complete!")
            self.status_label.config(text=f"Analysis complete - {os.path.basename(self.audio_path_var.get())}")

            # Update UI
            self.analyze_button.config(state=tk.NORMAL, text="🎵  RE-ANALYZE")
            self.progress_var.set(100)

            # Show completion message
            self._show_success_popup("Analysis Complete", "Audio analysis completed successfully!")

        except Exception as e:
            self._analysis_error(f"Error displaying results: {str(e)}")

    def _update_overview(self, analysis_result):
        """Update overview tab with analysis summary."""
        self.overview_text.config(state=tk.NORMAL)
        self.overview_text.delete(1.0, tk.END)

        summary = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                         ANALYSIS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 AUDIO FILE: {os.path.basename(self.audio_path_var.get())}
⏱️  DURATION: {analysis_result.get('audio_length_seconds', 0):.1f} seconds
🎵 SAMPLE RATE: {analysis_result.get('sample_rate', 0)} Hz
📊 PITCH POINTS: {len(analysis_result.get('pitch_contour_smoothed', []))}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                         KEY METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎶 VIBRATO ANALYSIS:"""

        vibrato = analysis_result.get('vibrato', {})
        if vibrato.get('status') == 'Success':
            summary += f"""
  • Rate: {vibrato.get('rate_hz', 0):.1f} Hz
  • Depth: {vibrato.get('depth_cents', 0):.1f} cents
  • Stability: {vibrato.get('stability', 0):.3f}
  • Quality: {vibrato.get('quality', 'Unknown')}
"""
        else:
            summary += f"\n  • Status: {vibrato.get('status', 'Failed')}"

        timbre = analysis_result.get('timbre', {})
        summary += f"\n🎨 TIMBRE ANALYSIS:"
        if timbre.get('status') == 'Success':
            summary += f"""
  • Tone Quality: {timbre.get('tone_label', 'Unknown')}
  • Harmonic Ratio: {timbre.get('harmonic_noise_ratio', 0):.1f}
  • Brightness: {timbre.get('brightness', 0):.2f}
"""
        else:
            summary += f"\n  • Status: {timbre.get('status', 'Failed')}"

        alignment = analysis_result.get('alignment', {})
        summary += f"\n🎼 ALIGNMENT ANALYSIS:"
        if alignment.get('status') in ['Success', 'No MIDI']:
            summary += f"""
  • Timing Offset: {alignment.get('average_timing_offset', 0)*1000:.0f} ms
  • Timing: {alignment.get('timing_label', 'Unknown')}
  • Notes Matched: {alignment.get('note_count_aligned', 0)}
"""
        else:
            summary += f"\n  • Status: {alignment.get('status', 'Failed')}"

        summary += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        self.overview_text.insert(tk.END, summary)
        self.overview_text.config(state=tk.DISABLED)

    def _update_vibrato_results(self, vibrato):
        """Update vibrato results tab."""
        self.vibrato_results.config(state=tk.NORMAL)
        self.vibrato_results.delete(1.0, tk.END)

        if 'error' in vibrato:
            self.vibrato_results.insert(tk.END, f"ERROR: {vibrato['error']}")
        else:
            result_text = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                         VIBRATO ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Rate: {vibrato.get('rate_hz', 0):.1f} Hz
Depth: {vibrato.get('depth_cents', 0):.1f} cents
Stability: {vibrato.get('stability', 0):.3f}
Quality: {vibrato.get('quality', 'Unknown')}
Status: {vibrato.get('status', 'Unknown')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                       INTERPRETATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            rate = vibrato.get('rate_hz', 0)
            depth = vibrato.get('depth_cents', 0)

            if rate > 0:
                if rate < 4:
                    result_text += "• Slow vibrato (expressive, lyrical)\n"
                elif rate > 7:
                    result_text += "• Fast vibrato (intense, dramatic)\n"
                else:
                    result_text += "• Normal vibrato rate (ideal)\n"

            if depth > 0:
                if depth < 20:
                    result_text += "• Subtle vibrato depth\n"
                elif depth > 80:
                    result_text += "• Wide vibrato depth\n"
                else:
                    result_text += "• Moderate vibrato depth (ideal)\n"

            if vibrato.get('stability', 0) > 0.7:
                result_text += "• Good vibrato stability\n"
            elif vibrato.get('stability', 0) < 0.3:
                result_text += "• Unstable vibrato\n"
            else:
                result_text += "• Moderate vibrato stability\n"

            result_text += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            self.vibrato_results.insert(tk.END, result_text)

        self.vibrato_results.config(state=tk.DISABLED)

    def _update_timbre_results(self, timbre):
        """Update timbre results tab."""
        if 'error' not in timbre:
            self.timbre_labels["centroid"].config(text=f"{timbre.get('spectral_centroid', 0):.0f}")
            self.timbre_labels["rolloff"].config(text=f"{timbre.get('spectral_rolloff', 0):.0f}")
            self.timbre_labels["brightness"].config(text=f"{timbre.get('brightness', 0):.3f}")
            self.timbre_labels["harmonicity"].config(text=f"{timbre.get('harmonic_noise_ratio', 0):.1f}")
            self.timbre_labels["noise"].config(text=f"{timbre.get('bow_noise_level', 0):.3f}")
            self.timbre_labels["resonance"].config(text=f"{timbre.get('resonance', 0):.3f}")

            tone = timbre.get('tone_label', 'Unknown')
            self.tone_label.config(text=tone)

            # Color code the tone label based on quality
            if "Warm" in tone or "Bright" in tone or "Resonant" in tone or "Balanced" in tone:
                self.tone_label.config(fg=self.COLOR_GREEN)
            elif "Harsh" in tone or "Noisy" in tone or "Airy" in tone or "Weak" in tone:
                self.tone_label.config(fg=self.COLOR_RED)
            else:
                self.tone_label.config(fg=self.COLOR_GOLD)
        else:
            for key in self.timbre_labels:
                self.timbre_labels[key].config(text="ERROR", fg=self.COLOR_RED)
            self.tone_label.config(text="Analysis Failed", fg=self.COLOR_RED)

    def _update_alignment_results(self, alignment):
        """Update alignment results tab."""
        self.alignment_results.config(state=tk.NORMAL)
        self.alignment_results.delete(1.0, tk.END)

        if 'error' in alignment:
            self.alignment_results.insert(tk.END, f"ERROR: {alignment['error']}")
        elif alignment.get('note_count_aligned', 0) > 0:
            result_text = f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                     AUDIO-MIDI ALIGNMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Timing Offset: {alignment.get('average_timing_offset', 0)*1000:.0f} ms
Timing: {alignment.get('timing_label', 'Unknown')}
Notes Matched: {alignment.get('note_count_aligned', 0)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                       INTERPRETATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            offset = alignment.get('average_timing_offset', 0)
            if offset < -0.05:
                result_text += "• Dragging (playing behind the beat)\n"
                result_text += "  Suggestion: Try to anticipate the beat slightly\n"
            elif offset > 0.05:
                result_text += "• Rushing (playing ahead of the beat)\n"
                result_text += "  Suggestion: Focus on listening to the accompaniment\n"
            else:
                result_text += "• Good timing accuracy\n"
                result_text += "  Keep up the good work!\n"

            result_text += f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            self.alignment_results.insert(tk.END, result_text)
        else:
            self.alignment_results.insert(tk.END, """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                     ALIGNMENT RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

No MIDI file provided or no alignment possible.
Audio-only analysis performed.

To get alignment results, please provide a MIDI file
along with your audio file.""")

        self.alignment_results.config(state=tk.DISABLED)

    def _analysis_error(self, error_msg):
        """Handle analysis error."""
        self.analyze_button.config(state=tk.NORMAL, text="🎵  START ANALYSIS")
        self.status_label.config(text="Analysis failed")
        self.progress_var.set(0)

        self._log(f"ERROR: {error_msg}")
        self._show_error_popup("Analysis Error", error_msg)

    def _log(self, message):
        """Add message to status."""
        self.status_label.config(text=message)

    def _show_error_popup(self, title, message):
        """Show error message in a styled popup."""
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("400x200")
        popup.resizable(False, False)
        popup.configure(bg=self.COLOR_BG)
        popup.transient(self.root)
        popup.grab_set()

        # Center the popup
        popup.geometry(f"+{self.root.winfo_x()+120}+{self.root.winfo_y()+300}")

        # Content
        content = tk.Frame(popup, bg=self.COLOR_BG, padx=20, pady=20)
        content.pack(fill=tk.BOTH, expand=True)

        # Error icon
        tk.Label(
            content,
            text="⚠️",
            font=("Georgia", 24),
            fg=self.COLOR_RED,
            bg=self.COLOR_BG
        ).pack(pady=(0, 10))

        # Error message
        tk.Label(
            content,
            text=message,
            font=self.FONT_TEXT,
            fg=self.COLOR_TEXT,
            bg=self.COLOR_BG,
            wraplength=350,
            justify=tk.LEFT
        ).pack(fill=tk.X, pady=(0, 20))

        # OK button
        ok_btn = tk.Button(
            content,
            text="OK",
            font=self.FONT_TEXT,
            fg=self.COLOR_GOLD,
            bg=self.COLOR_DARK,
            activeforeground=self.COLOR_GOLD,
            activebackground=self.COLOR_MEDIUM,
            relief=tk.RAISED,
            borderwidth=2,
            padx=30,
            pady=5,
            cursor="hand2",
            command=popup.destroy
        )
        ok_btn.pack()

    def _show_success_popup(self, title, message):
        """Show success message in a styled popup."""
        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.geometry("400x200")
        popup.resizable(False, False)
        popup.configure(bg=self.COLOR_BG)
        popup.transient(self.root)
        popup.grab_set()

        # Center the popup
        popup.geometry(f"+{self.root.winfo_x()+120}+{self.root.winfo_y()+300}")

        # Content
        content = tk.Frame(popup, bg=self.COLOR_BG, padx=20, pady=20)
        content.pack(fill=tk.BOTH, expand=True)

        # Success icon
        tk.Label(
            content,
            text="✅",
            font=("Georgia", 24),
            fg=self.COLOR_GREEN,
            bg=self.COLOR_BG
        ).pack(pady=(0, 10))

        # Success message
        tk.Label(
            content,
            text=message,
            font=self.FONT_TEXT,
            fg=self.COLOR_TEXT,
            bg=self.COLOR_BG,
            wraplength=350,
            justify=tk.LEFT
        ).pack(fill=tk.X, pady=(0, 20))

        # OK button
        ok_btn = tk.Button(
            content,
            text="OK",
            font=self.FONT_TEXT,
            fg=self.COLOR_GOLD,
            bg=self.COLOR_DARK,
            activeforeground=self.COLOR_GOLD,
            activebackground=self.COLOR_MEDIUM,
            relief=tk.RAISED,
            borderwidth=2,
            padx=30,
            pady=5,
            cursor="hand2",
            command=popup.destroy
        )
        ok_btn.pack()

    def _on_close(self):
        """Handle window close event."""
        self.root.destroy()

    # =============================================================================
    # COMPATIBILITY METHODS FOR INTEGRATION
    # =============================================================================

    def create_gui(self, parent):
        """Compatibility method - returns the main frame for integration."""
        # Create a frame to hold the integrated GUI
        frame = tk.Frame(parent, bg=self.COLOR_BG)

        # Create a simplified version of the GUI for integration
        title = tk.Label(
            frame,
            text="Audio Analyzer Module",
            font=self.FONT_HEADER,
            fg=self.COLOR_GOLD,
            bg=self.COLOR_BG
        )
        title.pack(pady=10)

        if not self.analyzer.deps_ok:
            status = tk.Label(
                frame,
                text=f"⚠️ Missing dependencies: {', '.join(self.analyzer.missing_deps)}",
                font=self.FONT_TEXT,
                fg=self.COLOR_RED,
                bg=self.COLOR_BG
            )
            status.pack(pady=5)
        else:
            status = tk.Label(
                frame,
                text="✅ Ready to analyze audio",
                font=self.FONT_TEXT,
                fg=self.COLOR_GREEN,
                bg=self.COLOR_BG
            )
            status.pack(pady=5)

        return frame

    def get_content(self):
        """Compatibility method - returns module info."""
        return self.analyzer.get_module_info()


# =============================================================================
# FACTORY FUNCTION FOR INTEGRATION
# =============================================================================

def create_integrated_gui(parent):
    """
    Factory function to create integrated GUI.

    Args:
        parent: Parent tkinter widget

    Returns:
        Frame containing the GUI or None if failed
    """
    try:
        analyzer = AudioAnalyzer()
        gui = AudioAnalyzerGUI(analyzer=analyzer)
        return gui.create_gui(parent)
    except Exception as e:
        print(f"Error creating AudioAnalyzer integrated GUI: {e}")
        return None


# =============================================================================
# MAIN ENTRY POINT - GUI LAUNCHES BY DEFAULT
# =============================================================================

def main():
    """Main entry point - launches GUI by default."""
    # Launch GUI automatically
    app = AudioAnalyzerGUI()


if __name__ == "__main__":
    main()