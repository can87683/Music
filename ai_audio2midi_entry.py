#!/usr/bin/env python3
# ai_audio2midi.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683

"""
pip install beat-this customtkinter librosa muscriptor numpy pretty_midi safetensors setproctitle torch torchaudio
"""

import os
import setproctitle
import configparser
import queue
import sys
import threading
import webbrowser
import customtkinter
import tkinter
import tkinter.filedialog
import tkinter.messagebox
import dataclasses
import pathlib
import typing
import subprocess
import tempfile
import torch
import torchaudio
import librosa
import numpy
import pretty_midi
import safetensors.torch

ctk = customtkinter
tk = tkinter
Path = pathlib.Path
Optional = typing.Optional
Callable = typing.Callable
Dict = typing.Dict
Any = typing.Any
List = typing.List
Tuple = typing.Tuple
dataclass = dataclasses.dataclass
asdict = dataclasses.asdict

setproctitle.setproctitle("ai_audio2midi.py")

os.environ["CUDA_VISIBLE_DEVICES"] = ""

SUPPORTED_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".aif", ".aiff")
FFMPEG_REQUIRED_FOR = (".mp3", ".flac", ".m4a", ".ogg", ".aac", ".aif", ".aiff")


@dataclass
class EngineResult:
    ok: bool
    midi_path: str
    note_count: int = 0
    message: str = ""
    engine_name: str = ""


class MessageBoxPatcher:
    ORIGINAL_SHOWINFO = tkinter.messagebox.showinfo
    ORIGINAL_SHOWWARNING = tkinter.messagebox.showwarning
    ORIGINAL_SHOWERROR = tkinter.messagebox.showerror
    ORIGINAL_ASKYESNO = tkinter.messagebox.askyesno

    @staticmethod
    def patched_showinfo(title: str, message: str, **kwargs):
        print(f"INFO: {title}")
        print(f"  {message}")

    @staticmethod
    def patched_showwarning(title: str, message: str, **kwargs):
        print(f"WARNING: {title}", file=sys.stderr)
        print(f"  {message}", file=sys.stderr)

    @staticmethod
    def patched_showerror(title: str, message: str, **kwargs):
        print(f"ERROR: {title}", file=sys.stderr)
        print(f"  {message}", file=sys.stderr)
        return MessageBoxPatcher.ORIGINAL_SHOWERROR(title, message, **kwargs)

    @staticmethod
    def patched_askyesno(title: str, message: str, **kwargs):
        print(f"QUESTION: {title}")
        print(f"  {message}")
        print(f"  Auto-answering 'yes' for non-exit dialogs...")
        return True

    @classmethod
    def apply(cls):
        tkinter.messagebox.showinfo = cls.patched_showinfo
        tkinter.messagebox.showwarning = cls.patched_showwarning
        tkinter.messagebox.showerror = cls.patched_showerror
        tkinter.messagebox.askyesno = cls.patched_askyesno


class SafetensorsPatcher:
    ORIGINAL_LOAD_FILE = safetensors.torch.load_file

    @staticmethod
    def patched_load_file(filename, device="cpu"):
        tensors = safetensors.torch.load(Path(filename).read_bytes())
        if device != "cpu":
            return {k: v.to(device) for k, v in tensors.items()}
        return tensors

    @classmethod
    def apply(cls):
        safetensors.torch.load_file = cls.patched_load_file


class Audio2MidiEngine:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.device = "cpu"

    def transcribe(self, audio_path: str, midi_path: str, progress_cb: Optional[Callable[[int], None]] = None, log_cb: Optional[Callable[[str], None]] = None, cancel_evt: Optional[threading.Event] = None) -> EngineResult:
        raise NotImplementedError

    def _get_audio_data(self, audio_path: str, target_sr: int = 22050) -> Tuple[numpy.ndarray, int]:
        audio_path_str = str(audio_path)
        ext = Path(audio_path_str).suffix.lower()
        if ext in FFMPEG_REQUIRED_FOR:
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav.close()
            cmd = ["ffmpeg", "-i", audio_path_str, "-ac", "1", "-ar", str(target_sr), temp_wav.name, "-y"]
            subprocess.run(cmd, check=True, capture_output=True)
            audio, sr = librosa.load(temp_wav.name, sr=target_sr, mono=True)
            os.unlink(temp_wav.name)
            return audio, sr
        audio, sr = librosa.load(audio_path_str, sr=target_sr, mono=True)
        return audio, sr

    def _create_midi(self, notes: List[Tuple[float, float, int, float]], midi_path: str, tempo: float = 120.0) -> bool:
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0)
        for start, end, pitch, velocity in notes:
            instrument.notes.append(pretty_midi.Note(velocity=int(velocity * 127), pitch=int(pitch), start=start, end=end))
        pm.instruments.append(instrument)
        pm.write(midi_path)
        return True

    def _count_notes(self, midi_path: str) -> int:
        pm = pretty_midi.PrettyMIDI(midi_path)
        return sum(len(instr.notes) for instr in pm.instruments)


class MuScriptorEngine(Audio2MidiEngine):
    def __init__(self, model_path: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self._model = None
        self._checked_install = False

    def _ensure_installed(self) -> None:
        if self._checked_install:
            return
        import muscriptor
        import beat_this
        self._checked_install = True

    def load_model(self):
        if self._model is None:
            from muscriptor import TranscriptionModel
            if not self.model_path or not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
            model_path = Path(self.model_path)
            if model_path.is_dir():
                weights_files = sorted(model_path.glob("*.safetensors"))
                if not weights_files:
                    raise FileNotFoundError(f"No .safetensors file found in: {model_path}")
                model_path = weights_files[0]
            self._model = TranscriptionModel.load_model(str(model_path), device="cpu")
        return self._model

    def _transcribe_via_event_stream(self, model, audio_path: str, midi_path: str, progress_cb: Optional[Callable[[int], None]], log_cb: Optional[Callable[[str], None]], cancel_evt: Optional[threading.Event]) -> None:
        tempo = self.params.get("midi_tempo", 120.0)
        notes: List[Tuple[float, float, int, float]] = []
        for event in model.transcribe(audio_path):
            if cancel_evt and cancel_evt.is_set():
                return
            if hasattr(event, "start_event"):
                start_event = event.start_event
                notes.append((start_event.start_time, event.end_time, start_event.pitch, 0.8))
            elif progress_cb and hasattr(event, "total") and event.total:
                progress_cb(int(event.completed / event.total * 100))
        self._create_midi(notes, midi_path, tempo=tempo)

    def transcribe(self, audio_path: str, midi_path: str, progress_cb: Optional[Callable[[int], None]] = None, log_cb: Optional[Callable[[str], None]] = None, cancel_evt: Optional[threading.Event] = None) -> EngineResult:
        if log_cb:
            log_cb("Loading MuScriptor model...")
        model = self.load_model()
        if model is None:
            return EngineResult(ok=False, midi_path="", message="MuScriptor model not loaded. Check model path.", engine_name="MuScriptor")
        if cancel_evt and cancel_evt.is_set():
            return EngineResult(ok=False, midi_path="", message="Cancelled by user", engine_name="MuScriptor")
        if log_cb:
            log_cb("Transcribing with MuScriptor...")
            progress_cb(25)
        if hasattr(model, "transcribe_to_midi"):
            midi_bytes = model.transcribe_to_midi(audio_path)
            Path(midi_path).write_bytes(midi_bytes)
        else:
            if log_cb:
                log_cb("transcribe_to_midi unavailable on this MuScriptor build; using event-stream fallback...")
            self._transcribe_via_event_stream(model, audio_path, midi_path, progress_cb, log_cb, cancel_evt)
        if cancel_evt and cancel_evt.is_set():
            return EngineResult(ok=False, midi_path="", message="Cancelled by user", engine_name="MuScriptor")
        if progress_cb:
            progress_cb(100)
        note_count = self._count_notes(midi_path)
        return EngineResult(ok=True, midi_path=midi_path, note_count=note_count, message=f"MuScriptor: {note_count} notes", engine_name="MuScriptor")


class Audio2MidiGUI:
    WINDOW_TITLE = "AI Audio -> MIDI Converter"
    STATE_FILE = Path(__file__).resolve().with_suffix(".ini")
    SCRIPT_DIR = Path(__file__).resolve().parent
    DEFAULT_MUSCRIPTOR_PATH = str(Path.home() / "ai_model_audio")

    DEFAULT_GEOMETRY = "640x1050+120+80"
    DEFAULT_INPUT_PATH = ""
    DEFAULT_SELECTED_ENGINE = "MuScriptor"
    DEFAULT_ONSET_THRESHOLD = 0.5
    DEFAULT_FRAME_THRESHOLD = 0.35
    DEFAULT_MIN_NOTE_LENGTH_MS = 80.0
    DEFAULT_MIN_FREQUENCY = 21.0
    DEFAULT_MAX_FREQUENCY = 108.0
    DEFAULT_MIDI_TEMPO = 120.0
    DEFAULT_LAST_ENGINE_OK = True
    DEFAULT_LAST_NOTES_DETECTED = 0
    POLL_INTERVAL_MS = 80

    STATE_BOOL_FIELDS = ("last_engine_ok",)
    STATE_INT_FIELDS = ("last_notes_detected",)
    STATE_FLOAT_FIELDS = ("onset_threshold", "frame_threshold", "min_note_length_ms", "min_frequency", "max_frequency", "midi_tempo")
    STATE_STR_FIELDS = ("geometry", "input_path", "selected_engine", "muscriptor_path")

    ENGINE_PARAMS = [
        ("Onset Threshold", "onset_threshold", 0.5, 0.0, 1.0, 0.05),
        ("Frame Threshold", "frame_threshold", 0.35, 0.0, 1.0, 0.05),
        ("Min Note Length (ms)", "min_note_length_ms", 80.0, 20.0, 500.0, 10.0),
        ("Min Frequency (MIDI)", "min_frequency", 21, 21, 84, 1),
        ("Max Frequency (MIDI)", "max_frequency", 108, 21, 108, 1),
        ("MIDI Tempo (BPM)", "midi_tempo", 120.0, 60.0, 240.0, 1.0),
    ]

    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self.worker = None
        self.cancel_event = threading.Event()
        self.msg_queue = queue.Queue()
        self._param_vars = {}

        self.geometry = self.DEFAULT_GEOMETRY
        self.input_path = self.DEFAULT_INPUT_PATH
        self.selected_engine = self.DEFAULT_SELECTED_ENGINE
        self.muscriptor_path = self.DEFAULT_MUSCRIPTOR_PATH
        self.onset_threshold = self.DEFAULT_ONSET_THRESHOLD
        self.frame_threshold = self.DEFAULT_FRAME_THRESHOLD
        self.min_note_length_ms = self.DEFAULT_MIN_NOTE_LENGTH_MS
        self.min_frequency = self.DEFAULT_MIN_FREQUENCY
        self.max_frequency = self.DEFAULT_MAX_FREQUENCY
        self.midi_tempo = self.DEFAULT_MIDI_TEMPO
        self.last_engine_ok = self.DEFAULT_LAST_ENGINE_OK
        self.last_notes_detected = self.DEFAULT_LAST_NOTES_DETECTED

        self._load_state()
        self._configure_root()
        self._build_layout()
        self._restore_paths()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close_request)
        self.root.after(self.POLL_INTERVAL_MS, self._poll_queue)

    def _load_state(self) -> None:
        if not self.STATE_FILE.is_file():
            return
        parser = configparser.ConfigParser()
        parser.read(self.STATE_FILE, encoding="utf-8")
        section = parser["state"]
        for name in self.STATE_BOOL_FIELDS:
            if name in section:
                setattr(self, name, section.getboolean(name))
        for name in self.STATE_INT_FIELDS:
            if name in section:
                setattr(self, name, section.getint(name))
        for name in self.STATE_FLOAT_FIELDS:
            if name in section:
                setattr(self, name, section.getfloat(name))
        for name in self.STATE_STR_FIELDS:
            if name in section:
                setattr(self, name, section.get(name))

    def _save_state(self) -> None:
        self.geometry = self.root.geometry()
        for _label, attr, *_ in self.ENGINE_PARAMS:
            var = self._param_vars.get(attr)
            if var is not None:
                setattr(self, attr, float(var.get()))

        state_dict = {}
        for name in self.STATE_BOOL_FIELDS + self.STATE_INT_FIELDS + self.STATE_FLOAT_FIELDS + self.STATE_STR_FIELDS:
            state_dict[name] = str(getattr(self, name))

        parser = configparser.ConfigParser()
        parser["state"] = state_dict
        with self.STATE_FILE.open("w", encoding="utf-8") as f:
            parser.write(f)

    def _configure_root(self) -> None:
        self.root.title(self.WINDOW_TITLE)
        self.root.geometry(self.geometry)
        self.root.minsize(640, 1050)
        self.root.maxsize(640, 1050)
        self.root.resizable(False, False)
        self.root.configure(fg_color="#1c1c1c")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

    def _restore_paths(self) -> None:
        if self.input_path:
            self.input_var.set(self.input_path)
        if self.muscriptor_path:
            self.muscriptor_path_var.set(self.muscriptor_path)

    def _build_layout(self) -> None:
        self.title_label = ctk.CTkLabel(self.root, text="Copyright Su Nie | BSD-3C License | https://github.com/can87683", font=ctk.CTkFont(size=14, weight="bold"), text_color="yellow")
        self.title_label.pack(fill="x", padx=2, pady=2)

        self.engine_frame = ctk.CTkFrame(self.root, fg_color="#2b2b2b", corner_radius=0, border_width=2)
        self.engine_frame.pack(fill="x", padx=2, pady=2)
        ctk.CTkLabel(self.engine_frame, text="Engine Selection", font=ctk.CTkFont(size=14, weight="bold"), text_color="yellow").pack(anchor="w", padx=2, pady=2)

        eng_row = ctk.CTkFrame(self.engine_frame, fg_color="transparent")
        eng_row.pack(fill="x", expand=True, padx=2, pady=2)
        ctk.CTkButton(eng_row, text="📁", width=40, command=self._pick_model_path, border_width=2).pack(side="left", padx=2)
        self.muscriptor_path_var = tk.StringVar(value=self.muscriptor_path)
        ctk.CTkEntry(eng_row, textvariable=self.muscriptor_path_var, border_width=2, state="disabled").pack(side="left", fill="x", expand=True, padx=2)
        ctk.CTkButton(eng_row, text="Download", width=80, command=lambda: webbrowser.open("https://huggingface.co/MuScriptor"), border_width=2).pack(side="left", padx=2)

        self.in_frame = ctk.CTkFrame(self.root, fg_color="#2b2b2b", corner_radius=0, border_width=2)
        self.in_frame.pack(fill="x", padx=2, pady=2)
        ctk.CTkLabel(self.in_frame, text="Audio Input", font=ctk.CTkFont(size=14, weight="bold"), text_color="yellow").pack(anchor="w", padx=2, pady=2)
        in_row = ctk.CTkFrame(self.in_frame, fg_color="transparent")
        in_row.pack(fill="x", expand=True, padx=2, pady=2)
        self.input_var = tk.StringVar(value=self.input_path)
        ctk.CTkButton(in_row, text="Select...", command=self._pick_input, border_width=2).pack(side="left", padx=2)
        ctk.CTkEntry(in_row, textvariable=self.input_var, border_width=2, state="disabled").pack(side="left", fill="x", expand=True, padx=2)

        self.param_frame = ctk.CTkFrame(self.root, fg_color="#2b2b2b", corner_radius=0, border_width=2)
        self.param_frame.pack(fill="x", padx=2, pady=2)
        ctk.CTkLabel(self.param_frame, text="Transcription Parameters", font=ctk.CTkFont(size=14, weight="bold"), text_color="yellow").pack(anchor="w", padx=2, pady=2)
        for label, attr, default, lo, hi, step in self.ENGINE_PARAMS:
            row = ctk.CTkFrame(self.param_frame, fg_color="transparent")
            row.pack(fill="x", expand=True, padx=2, pady=2)
            ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=12), text_color="white", width=180, anchor="w").pack(side="left", padx=2)
            current = getattr(self, attr, default)
            if current is None:
                current = 0.0
            var = tk.DoubleVar(value=float(current))
            self._param_vars[attr] = var
            ctk.CTkSlider(row, from_=lo, to=hi, variable=var, border_width=2).pack(side="left", fill="x", expand=True, padx=2)
            ctk.CTkEntry(row, textvariable=var, width=70, justify="right", border_width=2, font=ctk.CTkFont(size=12)).pack(side="left", padx=2)

        self.btn_row = ctk.CTkFrame(self.root, fg_color="transparent")
        self.btn_row.pack(fill="x", padx=2, pady=2)
        self.start_btn = ctk.CTkButton(self.btn_row, text="Convert", command=self._start_conversion, border_width=2, font=ctk.CTkFont(size=12))
        self.start_btn.pack(side="left", padx=2, pady=2)
        self.cancel_btn = ctk.CTkButton(self.btn_row, text="Cancel", command=self._request_cancel, state="disabled", border_width=2, font=ctk.CTkFont(size=12))
        self.cancel_btn.pack(side="left", padx=2, pady=2)

        self.prog_frame = ctk.CTkFrame(self.root, fg_color="#2b2b2b", corner_radius=0, border_width=2)
        self.prog_frame.pack(fill="x", padx=2, pady=2)
        ctk.CTkLabel(self.prog_frame, text="Conversion Progress", font=ctk.CTkFont(size=14, weight="bold"), text_color="yellow").pack(anchor="w", padx=2, pady=2)
        self.progress_var = tk.IntVar(value=0)
        ctk.CTkProgressBar(self.prog_frame, mode="determinate", variable=self.progress_var, border_width=2).pack(fill="x", expand=True, padx=2, pady=2)

        self.log_frame = ctk.CTkFrame(self.root, fg_color="#2b2b2b", corner_radius=0, border_width=2)
        self.log_frame.pack(fill="both", expand=True, padx=2, pady=2)
        ctk.CTkLabel(self.log_frame, text="Log", font=ctk.CTkFont(size=14, weight="bold"), text_color="yellow").pack(anchor="w", padx=2, pady=2)
        log_row = ctk.CTkFrame(self.log_frame, fg_color="transparent")
        log_row.pack(fill="both", expand=True, padx=2, pady=2)
        self.log_text = ctk.CTkTextbox(log_row, wrap="word", state="normal", fg_color="#003300", text_color="white", border_width=2, font=ctk.CTkFont(size=12))
        self.log_text.pack(side="left", fill="both", expand=True, padx=2, pady=2)

        self.log_text.bind("<Button-3>", self._show_log_context_menu)
        self.log_context_menu = tk.Menu(self.root, tearoff=0)
        self.log_context_menu.add_command(label="Copy", command=self._copy_log_selection)
        self.log_context_menu.add_command(label="Copy All", command=self._copy_log_all)

        self.status_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.status_frame.pack(fill="x", padx=2, pady=2)
        self.status_var = tk.StringVar(value="Ready")
        ctk.CTkLabel(self.status_frame, textvariable=self.status_var, font=ctk.CTkFont(size=12), text_color="white").pack(side="left", padx=2)

    def _pick_model_path(self):
        initial = self.muscriptor_path_var.get()
        path = tkinter.filedialog.askdirectory(title="Select MuScriptor Model Folder", initialdir=initial or str(self.SCRIPT_DIR))
        if path:
            self.muscriptor_path_var.set(path)
            self.muscriptor_path = path
            self._save_state()

    def _pick_input(self) -> None:
        initial = self.input_var.get() or os.path.expanduser("~")
        path = tkinter.filedialog.askopenfilename(
            title="Select audio file",
            initialdir=str(Path(initial).parent) if Path(initial).is_file() else initial,
            filetypes=[
                ("Audio", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac *.aif *.aiff"),
                ("WAV", "*.wav"), ("AIFF", "*.aif *.aiff"), ("MP3", "*.mp3"),
                ("FLAC", "*.flac"), ("M4A / AAC", "*.m4a *.aac"), ("OGG", "*.ogg"), ("All files", "*.*"),
            ],
        )
        if path:
            self.input_var.set(path)

    def _read_params(self) -> dict:
        params = {}
        for _label, attr, default, lo, hi, _step in self.ENGINE_PARAMS:
            val = float(self._param_vars[attr].get())
            if attr in ("min_frequency", "max_frequency") and val <= 0:
                val = 0.0
            params[attr] = val
        return params

    def _start_conversion(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        in_path = self.input_var.get().strip()
        if not in_path:
            print("WARNING: No input file selected", file=sys.stderr)
            return
        out_path = str(Path(in_path).with_suffix(".mid"))
        self.input_path = in_path
        self._save_state()
        self.progress_var.set(0)
        self._set_log("")
        self._log("Starting conversion with MuScriptor...")
        print("INFO: Starting conversion with MuScriptor", file=sys.stderr)
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.cancel_event.clear()
        params = self._read_params()
        model_path = self.muscriptor_path_var.get()
        engine = MuScriptorEngine(model_path=model_path, **params)
        self.worker = threading.Thread(target=self._worker_main, args=(in_path, out_path, engine), daemon=True)
        self.worker.start()

    def _worker_main(self, in_path: str, out_path: str, engine: MuScriptorEngine) -> None:
        result = engine.transcribe(
            in_path, out_path,
            progress_cb=lambda p: self.msg_queue.put(("progress", p)),
            log_cb=lambda m: self.msg_queue.put(("log", m)),
            cancel_evt=self.cancel_event,
        )
        self.msg_queue.put(("done", result))

    def _request_cancel(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self._log("Cancellation requested...")
            print("INFO: Cancellation requested", file=sys.stderr)
            self.cancel_event.set()
            self.cancel_btn.configure(state="disabled")

    def _poll_queue(self) -> None:
        while not self.msg_queue.empty():
            kind, payload = self.msg_queue.get_nowait()
            if kind == "progress":
                self.progress_var.set(int(payload))
            elif kind == "log":
                self._log(str(payload))
            elif kind == "done":
                self._on_done(payload)
        self.root.after(self.POLL_INTERVAL_MS, self._poll_queue)

    def _on_done(self, result: EngineResult) -> None:
        self.start_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        if result.ok:
            self.progress_var.set(100)
            self.last_engine_ok = True
            self.last_notes_detected = result.note_count
            self._log(f"SUCCESS: {result.note_count} notes -> {result.midi_path}")
            print(f"SUCCESS: {result.note_count} notes -> {result.midi_path}", file=sys.stderr)
        else:
            self.last_engine_ok = False
            self._log(f"FAILED: {result.message}")
            print(f"ERROR: {result.message}", file=sys.stderr)
        self._save_state()

    def _log(self, msg: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_log(self, msg: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        if msg:
            self.log_text.insert("end", msg + "\n")
        self.log_text.configure(state="disabled")

    def _on_close_request(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            print("WARNING: Conversion still running, cancel first", file=sys.stderr)
            MessageBoxPatcher.ORIGINAL_SHOWWARNING(self.WINDOW_TITLE, "A conversion is still running.\nPlease cancel it before exiting.")
            return
        if MessageBoxPatcher.ORIGINAL_ASKYESNO(self.WINDOW_TITLE, "Exit AI Audio -> MIDI?"):
            print("INFO: Exiting application", file=sys.stderr)
            self._save_state()
            self.root.destroy()
        else:
            print("INFO: Exit cancelled", file=sys.stderr)

    def _show_log_context_menu(self, event):
        self.log_context_menu.tk_popup(event.x_root, event.y_root)

    def _copy_log_selection(self):
        selection = self.log_text.tag_ranges("sel")
        if selection:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.log_text.get(selection[0], selection[1]))

    def _copy_log_all(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.log_text.get("1.0", "end-1c"))


if __name__ == "__main__":
    MessageBoxPatcher.apply()
    SafetensorsPatcher.apply()
    root = ctk.CTk()
    app = Audio2MidiGUI(root)
    root.mainloop()
    sys.exit(0)