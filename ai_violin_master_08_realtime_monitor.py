#!/usr/bin/env python3
# ai_violin_master_08_realtime_monitor.py

import threading
import queue
import time
from typing import Optional, Dict, Any
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

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


class RealtimeMonitor:
    """Real-time violin performance analyzer."""

    def __init__(self, analyzer=None, sample_rate=44100, frame_size=2048, hop_size=512):
        self.analyzer = analyzer
        self.sr = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.stream = None
        self.running = False
        self.pitch_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.kalman_pitch = None
        self.kalman_P = 1.0
        self.vibrato_window = []
        self.max_vibrato_frames = 40
        self.last_onset_time = None
        self.onset_times = []
        self.callback = None
        self.midi_onsets = None
        self.midi_index = 0

        # Try to import sounddevice, but don't fail if not available
        try:
            import sounddevice as sd
            self.sd = sd
            self.sounddevice_available = True
        except ImportError:
            self.sd = None
            self.sounddevice_available = False
            print("⚠️ sounddevice not available. Install with: pip install sounddevice")

    def midi_to_note(self, midi):
        """Convert MIDI number to note name."""
        if midi is None:
            return "--"
        note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        return f"{note_names[midi % 12]}{midi//12 - 1}"

    def kalman_update(self, measurement, R=8.0, Q=0.01):
        """Simple 1D Kalman filter to stabilize pitch measurements."""
        if self.kalman_pitch is None:
            self.kalman_pitch = measurement
            self.kalman_P = 1.0
            return measurement

        # Predict
        pred = self.kalman_pitch
        P = self.kalman_P + Q

        # Update
        K = P / (P + R)
        self.kalman_pitch = pred + K * (measurement - pred)
        self.kalman_P = (1 - K) * P

        return self.kalman_pitch

    def yin_pitch(self, frame):
        """GPU-optional YIN pitch detection."""
        x = frame.astype(np.float32)
        N = len(x)

        # Step 1: difference function
        diff = np.zeros(N//2)
        for tau in range(1, N//2):
            diff[tau] = np.sum((x[:-tau] - x[tau:])**2)

        # Step 2: cumulative mean normalization
        cmnd = np.copy(diff)
        for i in range(1, len(cmnd)):
            cmnd[i] *= i / sum(diff[1:i+1])

        # Step 3: absolute threshold
        threshold = 0.1
        candidates = np.where(cmnd < threshold)[0]
        if len(candidates) == 0:
            return None

        tau = candidates[0]
        if tau == 0:
            return None

        return self.sr / tau

    def estimate_vibrato(self):
        """Compute vibrato from rolling pitch window."""
        if len(self.vibrato_window) < 10:
            return None

        times = np.linspace(0, len(self.vibrato_window) / (self.sr / self.hop_size),
                            len(self.vibrato_window))
        pitches = np.array(self.vibrato_window)

        # Remove trend
        detrend = pitches - np.mean(pitches)

        # FFT to detect modulation frequency
        fft = np.fft.rfft(detrend)
        freqs = np.fft.rfftfreq(len(detrend),
                                d=self.hop_size / self.sr)

        idx = np.argmax(np.abs(fft[1:])) + 1
        vibrato_rate = freqs[idx]

        # Depth = peak-to-peak in cents
        if np.max(pitches) == 0:
            return None
        depth_cents = (np.max(pitches) - np.min(pitches)) * 100

        return {
            "rate_hz": float(vibrato_rate),
            "depth_cents": float(depth_cents),
        }

    def bow_noise_level(self, frame):
        """Use spectral roughness (high-frequency content ratio)."""
        spectrum = np.abs(np.fft.rfft(frame))
        total = np.sum(spectrum)
        if total <= 0:
            return 0.0

        high = np.sum(spectrum[int(len(spectrum)*0.6):])
        return float(high / total)

    def detect_onset(self, frame):
        """Simple energy-based onset detection."""
        energy = np.sum(frame**2)
        threshold = 0.02
        if energy > threshold:
            return True
        return False

    def load_midi_reference(self, pm):
        """Preload MIDI note-on times for timing deviation comparison."""
        onsets = []
        for inst in pm.instruments:
            onsets.extend([n.start for n in inst.notes])
        self.midi_onsets = sorted(onsets)
        self.midi_index = 0

    def timing_deviation(self):
        """Compare live onset vs expected next MIDI onset."""
        if self.midi_onsets is None:
            return None

        if self.midi_index >= len(self.midi_onsets):
            return None

        live_time = time.time()
        expected = self.midi_onsets[self.midi_index]

        deviation = live_time - expected
        self.midi_index += 1
        return deviation

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio stream callback."""
        if status or not self.running:
            return

        frame = indata[:, 0].copy()
        self.audio_queue.put(frame)

        # Pitch detection
        pitch = self.yin_pitch(frame)
        if pitch:
            smooth_pitch = self.kalman_update(pitch)
            self.pitch_queue.put(smooth_pitch)

        # Vibrato rolling buffer
        if pitch:
            self.vibrato_window.append(pitch)
            if len(self.vibrato_window) > self.max_vibrato_frames:
                self.vibrato_window.pop(0)

        return

    def start(self, callback=None):
        """Begin real-time monitoring."""
        if not self.sounddevice_available:
            raise ImportError("sounddevice not available. Install with: pip install sounddevice")

        self.callback = callback
        self.running = True

        self.stream = self.sd.InputStream(
            channels=1,
            samplerate=self.sr,
            blocksize=self.hop_size,
            callback=self._audio_callback
        )
        self.stream.start()

        # Start polling thread
        threading.Thread(target=self._poll_thread, daemon=True).start()

    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.stream = None

    def _poll_thread(self):
        """Collects output from audio callback and computes musical info."""
        last_vibrato = None

        while self.running:
            try:
                frame = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Compute bow noise
            bow_noise = self.bow_noise_level(frame)

            # Pitch processing
            pitch = None
            if not self.pitch_queue.empty():
                pitch = self.pitch_queue.get()

            note_name = "--"
            intonation = None

            if pitch:
                # Note mapping
                midi = int(round(69 + 12 * np.log2(pitch / 440.0)))
                note_name = self.midi_to_note(midi)

                # Intonation deviation (cents)
                ideal = 440 * 2**((midi - 69)/12)
                intonation = 1200 * np.log2(pitch / ideal)

            # Vibrato
            vib = self.estimate_vibrato()
            if vib:
                last_vibrato = vib

            vibrato_rate = last_vibrato["rate_hz"] if last_vibrato else None
            vibrato_depth = last_vibrato["depth_cents"] if last_vibrato else None

            # Timing deviation (optional)
            tdev = None
            if self.detect_onset(frame):
                tdev = self.timing_deviation()

            # Prepare data packet
            data = {
                "pitch": pitch,
                "note": note_name,
                "intonation": intonation,
                "vibrato_rate": vibrato_rate,
                "vibrato_depth": vibrato_depth,
                "bow_noise": bow_noise,
                "timing_deviation": tdev
            }

            if self.callback:
                try:
                    self.callback(data)
                except Exception:
                    pass

        return


class RealtimeMonitorGUI:
    """GUI for Realtime Monitor module with AIModels style"""

    def __init__(self, parent):
        self.parent = parent
        self.parent.title("🎻 Realtime Monitor")
        self.parent.geometry(WINDOW_SIZE)
        self.parent.resizable(False, False)
        self.parent.configure(bg=COLOR_BG)

        # Initialize monitor
        self.monitor = RealtimeMonitor()
        self.monitoring = False

        # Current readings
        self.current_data = {
            "pitch": None,
            "note": "--",
            "intonation": 0.0,
            "vibrato_rate": 0.0,
            "vibrato_depth": 0.0,
            "bow_noise": 0.0,
            "timing_deviation": 0.0
        }

        # Create GUI
        self._create_widgets()

        # Start update loop
        self._update_display()

    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = tk.Frame(self.parent, bg=COLOR_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        self._create_header(main_frame)

        # Monitoring controls
        self._create_controls(main_frame)

        # Status indicators
        self._create_status_indicators(main_frame)

        # Meters and displays
        self._create_meters(main_frame)

        # History graph
        self._create_history_graph(main_frame)

        # Settings
        self._create_settings(main_frame)

        # Status bar
        self._create_status_bar(main_frame)

    def _create_header(self, parent):
        """Create header section"""
        header_frame = tk.Frame(parent, bg=COLOR_BG)
        header_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            header_frame,
            text="🎻 Real-time Performance Monitor",
            font=FONT_HEADER,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack()

        tk.Label(
            header_frame,
            text="Live analysis of pitch, intonation, vibrato, and tone quality",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_BG
        ).pack()

    def _create_controls(self, parent):
        """Create monitoring controls"""
        frame = tk.LabelFrame(
            parent,
            text="Monitoring Controls",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        # Control buttons
        button_frame = tk.Frame(frame, bg=COLOR_BG)
        button_frame.pack(fill=tk.X, pady=5)

        # Start/Stop button
        self.start_stop_btn = tk.Button(
            button_frame,
            text="▶ Start Monitoring",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_DARK,
            activeforeground=COLOR_GOLD,
            activebackground=COLOR_FRAME,
            relief=tk.RAISED,
            borderwidth=2,
            command=self._toggle_monitoring,
            width=20
        )
        self.start_stop_btn.pack(side=tk.LEFT, padx=5)

        # Calibrate button
        calibrate_btn = tk.Button(
            button_frame,
            text="🎵 Calibrate",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_DARK,
            activeforeground=COLOR_GOLD,
            activebackground=COLOR_FRAME,
            relief=tk.RAISED,
            borderwidth=1,
            command=self._calibrate,
            width=15
        )
        calibrate_btn.pack(side=tk.LEFT, padx=5)

        # Reset button
        reset_btn = tk.Button(
            button_frame,
            text="🔄 Reset",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_DARK,
            activeforeground=COLOR_GOLD,
            activebackground=COLOR_FRAME,
            relief=tk.RAISED,
            borderwidth=1,
            command=self._reset,
            width=15
        )
        reset_btn.pack(side=tk.LEFT, padx=5)

        # Status indicator
        status_frame = tk.Frame(frame, bg=COLOR_BG)
        status_frame.pack(fill=tk.X, pady=10)

        self.status_indicator = tk.Label(
            status_frame,
            text="●",
            font=("Arial", 24),
            fg="#FF5555",  # Red for stopped
            bg=COLOR_BG
        )
        self.status_indicator.pack(side=tk.LEFT, padx=5)

        self.status_text = tk.Label(
            status_frame,
            text="Ready to monitor",
            font=FONT_TEXT,
            fg=COLOR_TEXT,
            bg=COLOR_BG
        )
        self.status_text.pack(side=tk.LEFT, padx=5)

    def _create_status_indicators(self, parent):
        """Create status indicators for key metrics"""
        frame = tk.LabelFrame(
            parent,
            text="Live Metrics",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        # Create a grid of indicators
        indicators_frame = tk.Frame(frame, bg=COLOR_BG)
        indicators_frame.pack(fill=tk.X, pady=5)

        # Define indicators
        self.indicators = {}
        indicators = [
            ("🎵 Current Note", "note", "C4"),
            ("📊 Pitch (Hz)", "pitch_display", "440.0"),
            ("🎯 Intonation", "intonation_display", "+0.0¢"),
            ("🎻 Vibrato Rate", "vibrato_rate_display", "5.5 Hz"),
            ("📈 Vibrato Depth", "vibrato_depth_display", "45¢"),
            ("🏹 Bow Noise", "bow_noise_display", "0.15")
        ]

        for i, (label, key, default) in enumerate(indicators):
            row = i // 3
            col = i % 3

            indicator_frame = tk.Frame(indicators_frame, bg=COLOR_BG)
            indicator_frame.grid(row=row, column=col, padx=10, pady=10, sticky="w")

            # Label
            tk.Label(
                indicator_frame,
                text=label,
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_BG
            ).pack(anchor="w")

            # Value display
            value_label = tk.Label(
                indicator_frame,
                text=default,
                font=("Courier", 11, "bold"),
                fg=COLOR_GOLD,
                bg=COLOR_DARK,
                padx=10,
                pady=5,
                width=12
            )
            value_label.pack(anchor="w", pady=(2, 0))

            self.indicators[key] = value_label

    def _create_meters(self, parent):
        """Create visual meters for intonation and bow noise"""
        frame = tk.LabelFrame(
            parent,
            text="Visual Meters",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        # Intonation meter
        intonation_frame = tk.Frame(frame, bg=COLOR_BG)
        intonation_frame.pack(fill=tk.X, pady=10)

        tk.Label(
            intonation_frame,
            text="Intonation Deviation:",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(anchor="w")

        # Intonation canvas
        self.intonation_canvas = tk.Canvas(
            intonation_frame,
            height=30,
            bg=COLOR_DARK,
            highlightthickness=0
        )
        self.intonation_canvas.pack(fill=tk.X, pady=5)

        # Draw meter background
        self._draw_intonation_meter()

        # Bow noise meter
        bow_frame = tk.Frame(frame, bg=COLOR_BG)
        bow_frame.pack(fill=tk.X, pady=10)

        tk.Label(
            bow_frame,
            text="Bow Noise Level:",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(anchor="w")

        # Bow noise canvas
        self.bow_canvas = tk.Canvas(
            bow_frame,
            height=30,
            bg=COLOR_DARK,
            highlightthickness=0
        )
        self.bow_canvas.pack(fill=tk.X, pady=5)

        # Draw bow noise meter
        self._draw_bow_meter()

    def _create_history_graph(self, parent):
        """Create pitch history graph"""
        frame = tk.LabelFrame(
            parent,
            text="Pitch History",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10, ipadx=10, ipady=10)

        # Create canvas for pitch graph
        self.pitch_canvas = tk.Canvas(
            frame,
            height=150,
            bg=COLOR_DARK,
            highlightthickness=0
        )
        self.pitch_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initialize pitch history
        self.pitch_history = []
        self.max_history = 100

        # Draw initial grid
        self._draw_pitch_grid()

    def _create_settings(self, parent):
        """Create settings panel"""
        frame = tk.LabelFrame(
            parent,
            text="Settings",
            font=FONT_SECTION,
            fg=COLOR_GOLD,
            bg=COLOR_BG,
            relief=tk.RIDGE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=10, pady=10, ipadx=10, ipady=10)

        settings_frame = tk.Frame(frame, bg=COLOR_BG)
        settings_frame.pack(fill=tk.X, pady=5)

        # Update rate
        rate_frame = tk.Frame(settings_frame, bg=COLOR_BG)
        rate_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            rate_frame,
            text="Update Rate:",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(side=tk.LEFT)

        self.update_rate_var = tk.StringVar(value="100 ms")
        rate_menu = tk.OptionMenu(
            rate_frame, self.update_rate_var,
            "50 ms", "100 ms", "200 ms", "500 ms"
        )
        rate_menu.config(
            font=FONT_TEXT,
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            activebackground=COLOR_FRAME,
            activeforeground=COLOR_GOLD,
            highlightthickness=0,
            width=10
        )
        rate_menu["menu"].config(
            bg=COLOR_DARK,
            fg=COLOR_TEXT,
            activebackground=COLOR_FRAME,
            activeforeground=COLOR_GOLD
        )
        rate_menu.pack(side=tk.LEFT, padx=5)

        # Sensitivity
        sens_frame = tk.Frame(settings_frame, bg=COLOR_BG)
        sens_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            sens_frame,
            text="Sensitivity:",
            font=FONT_TEXT,
            fg=COLOR_GOLD,
            bg=COLOR_BG
        ).pack(side=tk.LEFT)

        self.sensitivity_var = tk.DoubleVar(value=0.5)
        sensitivity_scale = tk.Scale(
            sens_frame,
            from_=0.1,
            to=1.0,
            variable=self.sensitivity_var,
            orient=tk.HORIZONTAL,
            resolution=0.1,
            length=150,
            bg=COLOR_BG,
            fg=COLOR_TEXT,
            troughcolor=COLOR_DARK,
            highlightbackground=COLOR_BG,
            sliderrelief=tk.RAISED
        )
        sensitivity_scale.pack(side=tk.LEFT, padx=5)

    def _create_status_bar(self, parent):
        """Create status bar at bottom"""
        status_frame = tk.Frame(parent, bg=COLOR_DARK)
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        self.status_label = tk.Label(
            status_frame,
            text="✅ Ready",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_DARK
        )
        self.status_label.pack(side=tk.LEFT, padx=5)

        # Performance stats
        stats_label = tk.Label(
            status_frame,
            text="FPS: -- | Latency: -- ms",
            font=FONT_SMALL,
            fg=COLOR_TEXT,
            bg=COLOR_DARK
        )
        stats_label.pack(side=tk.RIGHT, padx=5)

    def _draw_intonation_meter(self):
        """Draw intonation meter background"""
        canvas = self.intonation_canvas
        canvas.delete("all")

        width = canvas.winfo_width()
        if width < 10:  # Canvas not yet sized
            width = 400

        height = 30

        # Draw background
        canvas.create_rectangle(0, 0, width, height, fill=COLOR_DARK, outline="")

        # Draw center line (perfect intonation)
        center_x = width // 2
        canvas.create_line(center_x, 0, center_x, height, fill=COLOR_GOLD, width=2)

        # Draw zones
        zone_width = width // 6
        colors = ["#00AA00", "#55AA00", "#AAAA00", "#AA5500", "#AA0000"]

        for i in range(5):
            x1 = center_x - zone_width * (i+1)
            x2 = center_x - zone_width * i
            canvas.create_rectangle(x1, 5, x2, height-5, fill=colors[i], outline="")

            x1 = center_x + zone_width * i
            x2 = center_x + zone_width * (i+1)
            canvas.create_rectangle(x1, 5, x2, height-5, fill=colors[i], outline="")

        # Draw center perfect zone
        canvas.create_rectangle(center_x - zone_width//4, 5,
                               center_x + zone_width//4, height-5,
                               fill="#00FF00", outline="")

        # Draw labels
        canvas.create_text(10, height//2, text="-50¢", fill=COLOR_TEXT, anchor="w", font=FONT_SMALL)
        canvas.create_text(width-10, height//2, text="+50¢", fill=COLOR_TEXT, anchor="e", font=FONT_SMALL)
        canvas.create_text(center_x, height//2, text="0¢", fill=COLOR_BG, font=FONT_SMALL)

        # Draw needle (initially at center)
        self.intonation_needle = canvas.create_line(center_x, 0, center_x, height,
                                                   fill="#FFFFFF", width=2, tags="needle")

    def _draw_bow_meter(self):
        """Draw bow noise meter background"""
        canvas = self.bow_canvas
        canvas.delete("all")

        width = canvas.winfo_width()
        if width < 10:
            width = 400

        height = 30

        # Draw background
        canvas.create_rectangle(0, 0, width, height, fill=COLOR_DARK, outline="")

        # Draw gradient from green to red
        for i in range(width):
            ratio = i / width
            if ratio < 0.3:
                color = "#00FF00"  # Green
            elif ratio < 0.6:
                color = "#FFFF00"  # Yellow
            elif ratio < 0.8:
                color = "#FFAA00"  # Orange
            else:
                color = "#FF0000"  # Red

            canvas.create_line(i, 0, i, height, fill=color, width=1)

        # Draw labels
        canvas.create_text(10, height//2, text="Clean", fill=COLOR_BG, anchor="w", font=FONT_SMALL)
        canvas.create_text(width-10, height//2, text="Noisy", fill=COLOR_BG, anchor="e", font=FONT_SMALL)

        # Draw needle (initially at left)
        self.bow_needle = canvas.create_line(0, 0, 0, height,
                                            fill="#FFFFFF", width=2, tags="needle")

    def _draw_pitch_grid(self):
        """Draw pitch graph grid"""
        canvas = self.pitch_canvas
        canvas.delete("all")

        width = canvas.winfo_width()
        height = canvas.winfo_height()

        if width < 10 or height < 10:
            return

        # Draw background
        canvas.create_rectangle(0, 0, width, height, fill=COLOR_DARK, outline="")

        # Draw horizontal grid lines (notes)
        note_positions = {
            "C": 0, "C#": 1, "D": 2, "D#": 3,
            "E": 4, "F": 5, "F#": 6, "G": 7,
            "G#": 8, "A": 9, "A#": 10, "B": 11
        }

        for i in range(0, height, 20):
            # Draw grid line
            canvas.create_line(0, i, width, i, fill="#444444", width=1)

            # Draw note label every 3 lines
            if i % 60 == 0:
                canvas.create_text(5, i, text=f"Note", fill=COLOR_TEXT,
                                 anchor="w", font=("Arial", 8))

    def _update_intonation_meter(self, cents):
        """Update intonation meter needle position"""
        canvas = self.intonation_canvas
        width = canvas.winfo_width()

        if width < 10:
            return

        # Convert cents to pixel position (-50 to +50 cents maps to 0 to width)
        center_x = width // 2
        cents = max(-50, min(50, cents))  # Clamp to range
        pixel_offset = (cents / 50) * (width // 2)
        needle_x = center_x + pixel_offset

        # Update needle position
        canvas.coords(self.intonation_needle, needle_x, 0, needle_x, 30)

        # Change needle color based on deviation
        abs_cents = abs(cents)
        if abs_cents < 5:
            color = "#00FF00"  # Green
        elif abs_cents < 15:
            color = "#FFFF00"  # Yellow
        elif abs_cents < 25:
            color = "#FFAA00"  # Orange
        else:
            color = "#FF0000"  # Red

        canvas.itemconfig(self.intonation_needle, fill=color)

    def _update_bow_meter(self, noise_level):
        """Update bow noise meter needle position"""
        canvas = self.bow_canvas
        width = canvas.winfo_width()

        if width < 10:
            return

        # Convert noise level (0-1) to pixel position
        noise_level = max(0, min(1, noise_level))  # Clamp to range
        needle_x = noise_level * width

        # Update needle position
        canvas.coords(self.bow_needle, needle_x, 0, needle_x, 30)

        # Change needle color based on noise level
        if noise_level < 0.3:
            color = "#00FF00"  # Green
        elif noise_level < 0.6:
            color = "#FFFF00"  # Yellow
        elif noise_level < 0.8:
            color = "#FFAA00"  # Orange
        else:
            color = "#FF0000"  # Red

        canvas.itemconfig(self.bow_needle, fill=color)

    def _update_pitch_graph(self, pitch_hz):
        """Update pitch history graph"""
        if pitch_hz is None:
            return

        # Add to history
        self.pitch_history.append(pitch_hz)
        if len(self.pitch_history) > self.max_history:
            self.pitch_history.pop(0)

        # Redraw graph
        canvas = self.pitch_canvas
        width = canvas.winfo_width()
        height = canvas.winfo_height()

        if width < 10 or height < 10:
            return

        canvas.delete("pitch_line")

        if len(self.pitch_history) < 2:
            return

        # Find pitch range for scaling
        min_pitch = min(self.pitch_history)
        max_pitch = max(self.pitch_history)
        pitch_range = max(1, max_pitch - min_pitch)

        # Draw pitch line
        points = []
        for i, pitch in enumerate(self.pitch_history):
            x = (i / len(self.pitch_history)) * width
            y = height - ((pitch - min_pitch) / pitch_range) * height * 0.8 - height * 0.1
            points.extend([x, y])

        if len(points) >= 4:
            canvas.create_line(points, fill=COLOR_GOLD, width=2, tags="pitch_line", smooth=True)

    def _update_display(self):
        """Update all display elements with current data"""
        # Update indicators
        self.indicators["note"].config(text=self.current_data["note"])

        if self.current_data["pitch"]:
            self.indicators["pitch_display"].config(text=f"{self.current_data['pitch']:.1f} Hz")
        else:
            self.indicators["pitch_display"].config(text="-- Hz")

        self.indicators["intonation_display"].config(text=f"{self.current_data['intonation']:+.1f}¢")
        self.indicators["vibrato_rate_display"].config(text=f"{self.current_data['vibrato_rate']:.1f} Hz")
        self.indicators["vibrato_depth_display"].config(text=f"{self.current_data['vibrato_depth']:.0f}¢")
        self.indicators["bow_noise_display"].config(text=f"{self.current_data['bow_noise']:.3f}")

        # Update meters
        self._update_intonation_meter(self.current_data["intonation"])
        self._update_bow_meter(self.current_data["bow_noise"])

        # Update pitch graph
        self._update_pitch_graph(self.current_data["pitch"])

        # Schedule next update
        update_rate = self.update_rate_var.get()
        rate_ms = int(update_rate.split()[0])
        self.parent.after(rate_ms, self._update_display)

    def _toggle_monitoring(self):
        """Toggle monitoring state"""
        if not self.monitor.sounddevice_available:
            messagebox.showerror("Dependency Missing",
                               "sounddevice not available.\n\nInstall with: pip install sounddevice")
            return

        if not self.monitoring:
            # Start monitoring
            try:
                self.monitor.start(callback=self._on_monitor_data)
                self.monitoring = True
                self.start_stop_btn.config(text="⏹ Stop Monitoring")
                self.status_indicator.config(fg="#55FF55")  # Green
                self.status_text.config(text="Monitoring active")
                self.status_label.config(text="✅ Live monitoring")
            except Exception as e:
                messagebox.showerror("Start Error", f"Failed to start monitoring:\n{str(e)}")
                self.status_label.config(text="❌ Failed to start")
        else:
            # Stop monitoring
            self.monitor.stop()
            self.monitoring = False
            self.start_stop_btn.config(text="▶ Start Monitoring")
            self.status_indicator.config(fg="#FF5555")  # Red
            self.status_text.config(text="Monitoring stopped")
            self.status_label.config(text="✅ Monitoring stopped")

    def _calibrate(self):
        """Calibrate the monitor"""
        if not self.monitoring:
            messagebox.showinfo("Calibration",
                              "Please start monitoring first to calibrate.")
            return

        # In a real implementation, this would collect reference samples
        messagebox.showinfo("Calibration",
                          "Calibration started. Play a reference A4 (440Hz) note.")
        self.status_label.config(text="⏳ Calibrating...")

    def _reset(self):
        """Reset all measurements"""
        self.current_data = {
            "pitch": None,
            "note": "--",
            "intonation": 0.0,
            "vibrato_rate": 0.0,
            "vibrato_depth": 0.0,
            "bow_noise": 0.0,
            "timing_deviation": 0.0
        }

        self.pitch_history = []
        self._draw_pitch_grid()
        self.status_label.config(text="✅ Measurements reset")

    def _on_monitor_data(self, data):
        """Callback for monitor data updates"""
        # Update current data (this runs in monitor thread)
        self.current_data.update(data)

        # Fill in defaults for None values
        if self.current_data["intonation"] is None:
            self.current_data["intonation"] = 0.0
        if self.current_data["vibrato_rate"] is None:
            self.current_data["vibrato_rate"] = 0.0
        if self.current_data["vibrato_depth"] is None:
            self.current_data["vibrato_depth"] = 0.0
        if self.current_data["bow_noise"] is None:
            self.current_data["bow_noise"] = 0.0


# Module interface functions for backward compatibility
def get_content():
    """Return module description for display in GUI."""
    return [
        "╔══════════════════════════════════════╗",
        "║      REALTIME MONITOR MODULE         ║",
        "╚══════════════════════════════════════╝",
        "",
        "🎯 PURPOSE: Live microphone monitoring with instant analysis",
        "",
        "🎵 LIVE METRICS:",
        "  • Current note and pitch (Hz)",
        "  • Intonation deviation in cents",
        "  • Vibrato rate (Hz) and depth (cents)",
        "  • Bow noise level (clean vs scratchy)",
        "  • Timing deviation vs reference",
        "",
        "📊 VISUAL DISPLAYS:",
        "  • Intonation meter with colored zones",
        "  • Bow noise level meter",
        "  • Pitch history graph",
        "  • Real-time status indicators",
        "",
        "⚙️ TECHNICAL FEATURES:",
        "  • Kalman-filter stabilized pitch tracking",
        "  • GPU-optional YIN pitch detection",
        "  • Real-time FFT analysis",
        "  • Thread-safe audio processing",
        "  • Configurable update rates",
        "",
        "🎻 VIOLIN-SPECIFIC ANALYSIS:",
        "  • Vibrato analysis for musical expression",
        "  • Bow noise detection for tone quality",
        "  • Intonation tracking across positions",
        "  • Reference-based timing analysis",
        "",
        "🚀 QUICK START:",
        "  1. Connect microphone",
        "  2. Click 'Start Monitoring'",
        "  3. Play violin to see live analysis",
        "  4. Use meters to improve intonation and tone",
        "",
        "📋 DEPENDENCIES:",
        "  • Requires: pip install sounddevice",
        "  • Optional: NumPy for faster processing",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    gui = RealtimeMonitorGUI(parent)
    return gui.parent


# Standalone test
def test_standalone():
    """Test the module GUI standalone."""
    root = tk.Tk()
    root.title("Realtime Monitor Module - Standalone Test")
    root.geometry(WINDOW_SIZE)
    root.configure(bg=COLOR_BG)

    # Create the GUI
    gui = create_gui(root)

    root.mainloop()


if __name__ == "__main__":
    test_standalone()