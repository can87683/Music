#!/usr/bin/env python3
# ai_violin_master_07_models.py
# Copyright Su Nie | BSD-3C License | https://github.com/can87683

"""
AI Models Module for AI Violin Master
Complete independent GUI for LLM API and ML Model configuration
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import json
import os
import configparser
from pathlib import Path

# Color constants
COLOR_BG = "#5A381E"
COLOR_DARK = "#3B2413"
COLOR_GOLD = "#FFD770"
COLOR_FRAME = "#4B2E18"
COLOR_MEDIUM = "#4B2E18"
COLOR_TEXT = "#FFF2CF"

WINDOW_SIZE = "640x1050"


class AIModels:
    """AI models configuration and inference for violin performance analysis."""

    def __init__(self):
        self.config_file = "ai_violin_master_models.ini"
        self.default_config = {
            "llm_api_ip": "127.0.0.1",
            "llm_api_port": "12345",
            "api_key": "",
            "model_type": "llama",
            "ml_model_path": "./ai_violin_master_model.pt",
            "use_gpu": False,
            "inference_batch_size": 1,
            "max_context_length": 2048,
            "temperature": 0.7,
            "timeout_seconds": 10
        }
        self.config = self._load_config()
        self.last_error = None

    def _load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, "r", encoding="utf-8") as f:
                loaded_config = json.load(f)
                config = self.default_config.copy()
                config.update(loaded_config)
                return config
        return self.default_config.copy()

    def save_config(self, config_data=None):
        if config_data:
            self.config.update(config_data)
        os.makedirs(os.path.dirname(os.path.abspath(self.config_file)), exist_ok=True)
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, default=str)
        return True

    def get_llm_url(self, ip=None, port=None):
        ip = ip or self.config["llm_api_ip"]
        port = port or self.config["llm_api_port"]
        return f"http://{ip}:{port}"

    def validate_model_path(self, path):
        if not path:
            return False, "Path is empty"
        if path == "./ai_violin_master_model.pt" and not os.path.exists(path):
            return True, "Default path - model not installed yet"
        path_obj = Path(path)
        if not path_obj.exists():
            return False, f"File not found: {path}"
        valid_extensions = [".pt", ".pth", ".ckpt", ".safetensors", ".onnx"]
        if path_obj.suffix.lower() not in valid_extensions:
            return False, f"Invalid extension. Must be one of: {', '.join(valid_extensions)}"
        return True, "Valid model file"

    def get_available_model_types(self):
        return ["llama", "gpt", "claude", "mistral", "custom", "none"]

    def reset_to_defaults(self):
        self.config = self.default_config.copy()
        return self.save_config()

    def test_llm_connection(self, ip, port):
        import requests
        url = self.get_llm_url(ip, port)
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True, f"Success!\nConnected to LLM API at {url}"
            return False, f"Failed!\nServer returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, f"Connection Failed!\nCould not connect to {url}\n\nMake sure your LLM server is running."


class AIModelsGUI:
    """GUI for AI models configuration using CustomTkinter."""

    def __init__(self, parent=None, module=None):
        self.ini_file = "ai_violin_master_07_models.ini"
        self.config = configparser.ConfigParser()

        if parent is None:
            self.root = ctk.CTk()
            self.is_standalone = True
            self.root.title("AI Models Configuration")
            self.root.geometry(WINDOW_SIZE)
            self.root.resizable(False, False)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        else:
            self.root = parent
            self.is_standalone = False

        self.module = module if module is not None else AIModels()

        self._load_state()
        ctk.set_appearance_mode("dark")

        if self.is_standalone:
            self.root.configure(fg_color=COLOR_BG)

        self._create_widgets()

        if self.is_standalone:
            self._restore_position()
            self.root.mainloop()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # widget tree
    # ------------------------------------------------------------------

    def _create_widgets(self):
        self.main_frame = ctk.CTkFrame(self.root, fg_color=COLOR_BG)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Header
        header_frame = ctk.CTkFrame(
            self.main_frame, fg_color=COLOR_DARK,
            border_width=2, border_color=COLOR_GOLD,
        )
        header_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            header_frame,
            text="AI MODELS CONFIGURATION",
            font=("Georgia", 16, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ctk.CTkLabel(
            header_frame,
            text="Configure AI models for violin performance analysis",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
        ).pack(pady=(0, 10))

        # LLM configuration
        llm_frame = ctk.CTkFrame(
            self.main_frame, fg_color=COLOR_DARK,
            border_width=2, border_color=COLOR_GOLD,
        )
        llm_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            llm_frame,
            text="LLM API CONFIGURATION",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        ip_row = ctk.CTkFrame(llm_frame, fg_color=COLOR_DARK)
        ip_row.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            ip_row,
            text="IP Address:",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            width=150,
        ).pack(side="left")

        self.ip_var = ctk.StringVar(value=self.module.config.get("llm_api_ip", "127.0.0.1"))
        ctk.CTkEntry(
            ip_row,
            textvariable=self.ip_var,
            font=("Georgia", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
        ).pack(side="left", fill="x", expand=True, padx=5)

        port_row = ctk.CTkFrame(llm_frame, fg_color=COLOR_DARK)
        port_row.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            port_row,
            text="Port:",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            width=150,
        ).pack(side="left")

        self.port_var = ctk.StringVar(value=self.module.config.get("llm_api_port", "12345"))
        ctk.CTkEntry(
            port_row,
            textvariable=self.port_var,
            font=("Georgia", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
        ).pack(side="left", fill="x", expand=True, padx=5)

        key_row = ctk.CTkFrame(llm_frame, fg_color=COLOR_DARK)
        key_row.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            key_row,
            text="API Key (optional):",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            width=150,
        ).pack(side="left")

        self.api_key_var = ctk.StringVar(value=self.module.config.get("api_key", ""))
        ctk.CTkEntry(
            key_row,
            textvariable=self.api_key_var,
            font=("Georgia", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
            show="*",
        ).pack(side="left", fill="x", expand=True, padx=5)

        self.test_btn = ctk.CTkButton(
            llm_frame,
            text="TEST CONNECTION",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_FRAME,
            command=self._test_connection,
        )
        self.test_btn.pack(pady=10)

        # ML model configuration
        ml_frame = ctk.CTkFrame(
            self.main_frame, fg_color=COLOR_DARK,
            border_width=2, border_color=COLOR_GOLD,
        )
        ml_frame.pack(fill="x", pady=10)

        ctk.CTkLabel(
            ml_frame,
            text="ML MODEL CONFIGURATION",
            font=("Georgia", 12, "bold"),
            text_color=COLOR_GOLD,
            fg_color=COLOR_DARK,
        ).pack(pady=10)

        path_row = ctk.CTkFrame(ml_frame, fg_color=COLOR_DARK)
        path_row.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            path_row,
            text="Model Path:",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            width=150,
        ).pack(side="left")

        self.model_path_var = ctk.StringVar(
            value=self.module.config.get("ml_model_path", "./ai_violin_master_model.pt")
        )
        ctk.CTkEntry(
            path_row,
            textvariable=self.model_path_var,
            font=("Georgia", 12),
            fg_color=COLOR_FRAME,
            text_color=COLOR_TEXT,
            border_width=2,
            border_color=COLOR_GOLD,
        ).pack(side="left", fill="x", expand=True, padx=5)

        browse_row = ctk.CTkFrame(ml_frame, fg_color=COLOR_DARK)
        browse_row.pack(fill="x", padx=20, pady=5)

        ctk.CTkButton(
            browse_row,
            text="BROWSE MODEL",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_FRAME,
            command=self._browse_model,
        ).pack(side="left", padx=5, expand=True, fill="x")

        ctk.CTkButton(
            browse_row,
            text="VALIDATE MODEL",
            font=("Georgia", 12),
            fg_color=COLOR_MEDIUM,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_FRAME,
            command=self._validate_model,
        ).pack(side="left", padx=5, expand=True, fill="x")

        # GPU toggle
        gpu_row = ctk.CTkFrame(ml_frame, fg_color=COLOR_DARK)
        gpu_row.pack(fill="x", padx=20, pady=5)

        self.use_gpu_var = ctk.BooleanVar(value=self.module.config.get("use_gpu", False))
        ctk.CTkCheckBox(
            gpu_row,
            text="Use GPU Acceleration",
            variable=self.use_gpu_var,
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_GOLD,
            border_color=COLOR_GOLD,
            checkmark_color=COLOR_DARK,
        ).pack(anchor="w")

        # Action buttons
        action_row = ctk.CTkFrame(self.main_frame, fg_color=COLOR_DARK)
        action_row.pack(fill="x", pady=20)

        ctk.CTkButton(
            action_row,
            text="SAVE CONFIGURATION",
            font=("Georgia", 14, "bold"),
            fg_color=COLOR_DARK,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_MEDIUM,
            height=50,
            command=self._save_configuration,
        ).pack(side="left", padx=5, expand=True, fill="x")

        ctk.CTkButton(
            action_row,
            text="RESET TO DEFAULTS",
            font=("Georgia", 14, "bold"),
            fg_color=COLOR_DARK,
            text_color=COLOR_GOLD,
            border_width=2,
            border_color=COLOR_GOLD,
            hover_color=COLOR_MEDIUM,
            height=50,
            command=self._reset_configuration,
        ).pack(side="left", padx=5, expand=True, fill="x")

        # Status bar
        self.status_label = ctk.CTkLabel(
            self.main_frame,
            text="Ready",
            font=("Georgia", 12),
            text_color=COLOR_TEXT,
            fg_color=COLOR_DARK,
            height=30,
        )
        self.status_label.pack(fill="x", pady=(10, 0))

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------

    def _test_connection(self):
        success, message = self.module.test_llm_connection(
            self.ip_var.get(), self.port_var.get()
        )
        if success:
            messagebox.showinfo("Connection Test", message)
            self.status_label.configure(text="Connection successful")
        else:
            messagebox.showerror("Connection Test", message)
            self.status_label.configure(text="Connection failed")

    def _browse_model(self):
        filename = filedialog.askopenfilename(
            title="Select ML Model File",
            filetypes=[
                ("Model files", "*.pt *.pth *.ckpt *.safetensors *.onnx"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.model_path_var.set(filename)
            self.status_label.configure(text=f"Selected: {os.path.basename(filename)}")

    def _validate_model(self):
        valid, message = self.module.validate_model_path(self.model_path_var.get())
        if valid:
            messagebox.showinfo("Model Validation", message)
            self.status_label.configure(text="Model valid")
        else:
            messagebox.showerror("Model Validation", message)
            self.status_label.configure(text="Model invalid")

    def _save_configuration(self):
        config_data = {
            "llm_api_ip": self.ip_var.get(),
            "llm_api_port": self.port_var.get(),
            "api_key": self.api_key_var.get(),
            "ml_model_path": self.model_path_var.get(),
            "use_gpu": self.use_gpu_var.get(),
        }
        self.module.config.update(config_data)
        self.module.save_config()
        self._save_state()
        self.status_label.configure(text="Configuration saved")
        messagebox.showinfo("Success", "Configuration saved successfully!")

    def _reset_configuration(self):
        if messagebox.askyesno("Confirm Reset", "Reset all settings to defaults?"):
            self.module.reset_to_defaults()
            self.ip_var.set(self.module.config["llm_api_ip"])
            self.port_var.set(self.module.config["llm_api_port"])
            self.api_key_var.set(self.module.config["api_key"])
            self.model_path_var.set(self.module.config["ml_model_path"])
            self.use_gpu_var.set(self.module.config["use_gpu"])
            self.status_label.configure(text="Reset to defaults")

    def _on_close(self):
        self._save_state()
        self.root.destroy()


# ----------------------------------------------------------------------
# module interface functions
# ----------------------------------------------------------------------

def get_content():
    """Return module description for display in GUI."""
    return [
        "AI MODELS MODULE",
        "",
        "PURPOSE: Configure AI models for violin performance analysis",
        "",
        "LLM API CONFIGURATION (Optional):",
        "  - Connect to local LLM (Ollama, llama.cpp)",
        "  - Or cloud API (OpenAI, Anthropic)",
        "  - Get AI-generated violin feedback",
        "  - Default: 127.0.0.1:12345",
        "",
        "ML MODEL (Future Enhancement):",
        "  - Specialized violin performance models",
        "  - Pitch/rhythm/timbre analysis",
        "  - Technical difficulty scoring",
        "",
        "CURRENT STATUS:",
        "  - Complete configuration interface",
        "  - Connection testing utility",
        "  - Model validation",
        "  - Settings persistence",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module, packed into parent."""
    gui = AIModelsGUI(parent)
    return gui.main_frame


def main():
    """Standalone entry - launches a full window with mainloop."""
    AIModelsGUI()


if __name__ == "__main__":
    main()