#!/usr/bin/env python3
# ai_violin_master_07_models.py
"""
AI Models Module for AI Violin Master
Complete independent GUI for LLM API and ML Model configuration
"""

import json
import os
import sys
from typing import Dict, Any, Optional, Tuple
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


class AIModels:
    """AI models configuration and inference for violin performance analysis."""

    def __init__(self):
        self.config_file = "ai_violin_master_models.ini"
        self.default_config = {
            "llm_api_ip": "127.0.0.1",
            "llm_api_port": "12345",
            "api_key": "",  # Optional API key for authentication
            "model_type": "llama",  # llama, gpt, claude, etc.
            "ml_model_path": "./ai_violin_master_model.pt",
            "use_gpu": False,
            "inference_batch_size": 1,
            "max_context_length": 2048,
            "temperature": 0.7,
            "timeout_seconds": 10
        }
        self.config = self._load_config()
        self.last_error = None

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    config = self.default_config.copy()
                    config.update(loaded_config)
                    return config
        except Exception as e:
            print(f"⚠️  Note: Could not load models config: {e}")

        return self.default_config.copy()

    def save_config(self, config_data: Optional[Dict[str, Any]] = None) -> bool:
        """Save configuration to file."""
        try:
            if config_data:
                self.config.update(config_data)

            os.makedirs(os.path.dirname(os.path.abspath(self.config_file)), exist_ok=True)

            with open(self.config_file, "w", encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, default=str)

            return True

        except Exception as e:
            print(f"❌ Error saving models config: {e}")
            self.last_error = f"Config save error: {str(e)}"
            return False

    def get_llm_url(self, ip: str = None, port: str = None) -> str:
        """Get complete LLM API URL."""
        ip = ip or self.config['llm_api_ip']
        port = port or self.config['llm_api_port']
        return f"http://{ip}:{port}"

    def validate_model_path(self, path: str) -> Tuple[bool, str]:
        """Validate that the ML model path exists and is valid."""
        if not path:
            return False, "Path is empty"

        # If path is the default and file doesn't exist, that's OK for new users
        if path == "./ai_violin_master_model.pt" and not os.path.exists(path):
            return True, "Default path - model not installed yet"

        path_obj = Path(path)

        # Check if file exists
        if not path_obj.exists():
            return False, f"File not found: {path}"

        # Check file extension
        valid_extensions = ['.pt', '.pth', '.ckpt', '.safetensors', '.onnx']
        if path_obj.suffix.lower() not in valid_extensions:
            return False, f"Invalid file extension. Must be one of: {', '.join(valid_extensions)}"

        return True, "Valid model file"

    def get_available_model_types(self) -> list:
        """Get list of available model types."""
        return ["llama", "gpt", "claude", "mistral", "custom", "none"]

    def reset_to_defaults(self) -> bool:
        """Reset configuration to default values."""
        self.config = self.default_config.copy()
        return self.save_config()

    def test_llm_connection(self, ip: str, port: str) -> Tuple[bool, str]:
        """Test connection to LLM API."""
        try:
            import requests

            url = self.get_llm_url(ip, port)
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True, f"✅ Success!\nConnected to LLM API at {url}"
                else:
                    return False, f"❌ Failed!\nServer returned status {response.status_code}"
            except requests.exceptions.ConnectionError:
                return False, f"❌ Connection Failed!\nCould not connect to {url}\n\nMake sure your LLM server is running."
            except Exception as e:
                return False, f"❌ Error!\n{str(e)}"

        except ImportError:
            return False, "⚠️  The 'requests' module is required for LLM connections.\n\nInstall it with:\npip install requests"

class AIModelsGUI:
    """GUI for AI models configuration."""

    def __init__(self, parent, module: AIModels):
        self.parent = parent
        self.module = module

        # Create main frame with uniform colors
        self.main_frame = tk.Frame(parent, bg=COLOR_BG)
        self.main_frame.pack(fill="both", expand=True)

        # UI variables
        self.ip_var = tk.StringVar(value=self.module.config.get('llm_api_ip', '127.0.0.1'))
        self.port_var = tk.StringVar(value=self.module.config.get('llm_api_port', '12345'))
        self.api_key_var = tk.StringVar(value=self.module.config.get('api_key', ''))
        self.model_type_var = tk.StringVar(value=self.module.config.get('model_type', 'llama'))
        self.model_path_var = tk.StringVar(value=self.module.config.get('ml_model_path', './ai_violin_master_model.pt'))
        self.use_gpu_var = tk.BooleanVar(value=self.module.config.get('use_gpu', False))
        self.batch_var = tk.IntVar(value=self.module.config.get('inference_batch_size', 1))
        self.context_var = tk.StringVar(value=str(self.module.config.get('max_context_length', 2048)))
        self.temp_var = tk.DoubleVar(value=self.module.config.get('temperature', 0.7))
        self.timeout_var = tk.IntVar(value=self.module.config.get('timeout_seconds', 10))

        self._setup_gui()

    def _setup_gui(self):
        """Set up the complete GUI."""
        try:
            # Create notebook for tabs
            self.notebook = ttk.Notebook(self.main_frame)
            self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

            # Style the notebook to match color scheme
            self._style_notebook()

            # Create tabs
            self._create_llm_tab()
            self._create_ml_tab()
            self._create_settings_tab()
            self._create_action_buttons()

        except Exception as e:
            self._create_error_gui(str(e))

    def _style_notebook(self):
        """Style the notebook widget."""
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

    def _create_llm_tab(self):
        """Create LLM configuration tab."""
        llm_frame = tk.Frame(self.notebook, bg=COLOR_BG)
        self.notebook.add(llm_frame, text="LLM API")

        # Header
        header_frame = tk.Frame(llm_frame, bg=COLOR_BG)
        header_frame.pack(fill="x", pady=(0, 10))

        tk.Label(header_frame,
                text="🤖 LLM API Configuration",
                font=FONT_HEADER,
                fg=COLOR_GOLD,
                bg=COLOR_BG).pack()

        tk.Label(header_frame,
                text="Configure your language model for AI feedback",
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_BG).pack()

        # Separator
        sep1 = tk.Frame(llm_frame, height=2, bg=COLOR_GOLD)
        sep1.pack(fill="x", padx=20, pady=5)

        # Configuration Frame
        config_frame = tk.LabelFrame(llm_frame,
                                    text="Server Settings",
                                    font=FONT_SECTION,
                                    fg=COLOR_GOLD,
                                    bg=COLOR_BG,
                                    relief=tk.RIDGE,
                                    borderwidth=2)
        config_frame.pack(fill="x", padx=20, pady=10, ipadx=10, ipady=10)

        # IP Address
        self._create_input_field(config_frame, "IP Address:", self.ip_var, width=20)

        # Port
        self._create_input_field(config_frame, "Port:", self.port_var, width=10)

        # API Key
        api_frame = tk.Frame(config_frame, bg=COLOR_BG)
        api_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(api_frame,
                text="API Key (optional):",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=15,
                anchor="w").pack(side="left")

        api_key_entry = tk.Entry(api_frame,
                                textvariable=self.api_key_var,
                                font=FONT_TEXT,
                                bg=COLOR_DARK,
                                fg=COLOR_TEXT,
                                insertbackground=COLOR_TEXT,
                                relief=tk.SUNKEN,
                                borderwidth=1,
                                width=30,
                                show="•")
        api_key_entry.pack(side="left", padx=5)

        # Model Type
        model_frame = tk.Frame(config_frame, bg=COLOR_BG)
        model_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(model_frame,
                text="Model Type:",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=15,
                anchor="w").pack(side="left")

        model_types = self.module.get_available_model_types()
        model_dropdown = self._create_dropdown_menu(model_frame, self.model_type_var, model_types)
        model_dropdown.pack(side="left", padx=5)

        # Connection Test Button
        test_button = tk.Button(llm_frame,
                               text="🔌 Test Connection",
                               font=FONT_TEXT,
                               fg=COLOR_GOLD,
                               bg=COLOR_DARK,
                               activeforeground=COLOR_GOLD,
                               activebackground=COLOR_FRAME,
                               relief=tk.RAISED,
                               borderwidth=2,
                               padx=20,
                               pady=5,
                               command=self._test_connection)
        test_button.pack(pady=10)

        # Info Text
        info_text = """💡 LLM Setup Tips:

• For Ollama: Use 127.0.0.1:11434
• For llama.cpp: Use 127.0.0.1:8080
• API key only needed for cloud services
• Leave API key empty for local models
• Test connection after configuration"""

        info_label = tk.Label(llm_frame,
                             text=info_text,
                             font=FONT_SMALL,
                             fg=COLOR_TEXT,
                             bg=COLOR_BG,
                             justify=tk.LEFT)
        info_label.pack(pady=10, padx=20)

    def _create_ml_tab(self):
        """Create ML model configuration tab."""
        ml_frame = tk.Frame(self.notebook, bg=COLOR_BG)
        self.notebook.add(ml_frame, text="ML Model")

        # Header
        ml_header_frame = tk.Frame(ml_frame, bg=COLOR_BG)
        ml_header_frame.pack(fill="x", pady=(0, 10))

        tk.Label(ml_header_frame,
                text="🤖 Machine Learning Model",
                font=FONT_HEADER,
                fg=COLOR_GOLD,
                bg=COLOR_BG).pack()

        tk.Label(ml_header_frame,
                text="Configure specialized violin analysis models",
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_BG).pack()

        # Separator
        sep2 = tk.Frame(ml_frame, height=2, bg=COLOR_GOLD)
        sep2.pack(fill="x", padx=20, pady=5)

        # Model Configuration Frame
        model_config_frame = tk.LabelFrame(ml_frame,
                                          text="Model Settings",
                                          font=FONT_SECTION,
                                          fg=COLOR_GOLD,
                                          bg=COLOR_BG,
                                          relief=tk.RIDGE,
                                          borderwidth=2)
        model_config_frame.pack(fill="x", padx=20, pady=10, ipadx=10, ipady=10)

        # Model Path
        path_frame = tk.Frame(model_config_frame, bg=COLOR_BG)
        path_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(path_frame,
                text="Model Path:",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=15,
                anchor="w").pack(side="left")

        path_entry = tk.Entry(path_frame,
                             textvariable=self.model_path_var,
                             font=FONT_TEXT,
                             bg=COLOR_DARK,
                             fg=COLOR_TEXT,
                             insertbackground=COLOR_TEXT,
                             relief=tk.SUNKEN,
                             borderwidth=1,
                             width=30)
        path_entry.pack(side="left", padx=5, fill="x", expand=True)

        # Browse Button
        browse_btn = tk.Button(path_frame,
                              text="📁 Browse",
                              font=FONT_SMALL,
                              fg=COLOR_GOLD,
                              bg=COLOR_DARK,
                              command=self._browse_model)
        browse_btn.pack(side="right", padx=5)

        # Validate Button
        validate_btn = tk.Button(model_config_frame,
                                text="🔍 Validate Model",
                                font=FONT_TEXT,
                                fg=COLOR_GOLD,
                                bg=COLOR_DARK,
                                command=self._validate_model)
        validate_btn.pack(pady=10)

        # GPU Toggle
        gpu_frame = tk.Frame(model_config_frame, bg=COLOR_BG)
        gpu_frame.pack(fill="x", padx=10, pady=5)

        gpu_check = tk.Checkbutton(gpu_frame,
                                  text="Use GPU Acceleration",
                                  variable=self.use_gpu_var,
                                  font=FONT_TEXT,
                                  fg=COLOR_TEXT,
                                  bg=COLOR_BG,
                                  activebackground=COLOR_BG,
                                  activeforeground=COLOR_TEXT,
                                  selectcolor=COLOR_DARK)
        gpu_check.pack(anchor="w")

        # ML Model Info
        ml_info_text = """📊 ML Model Information:

• Default path reserved for future models
• Basic audio analysis works without ML model
• Specialized violin models coming soon
• Supports PyTorch (.pt, .pth) formats
• GPU acceleration optional"""

        ml_info_label = tk.Label(ml_frame,
                                text=ml_info_text,
                                font=FONT_SMALL,
                                fg=COLOR_TEXT,
                                bg=COLOR_BG,
                                justify=tk.LEFT)
        ml_info_label.pack(pady=10, padx=20)

    def _create_settings_tab(self):
        """Create inference settings tab."""
        settings_frame = tk.Frame(self.notebook, bg=COLOR_BG)
        self.notebook.add(settings_frame, text="Settings")

        # Header
        settings_header_frame = tk.Frame(settings_frame, bg=COLOR_BG)
        settings_header_frame.pack(fill="x", pady=(0, 10))

        tk.Label(settings_header_frame,
                text="⚙️ Inference Parameters",
                font=FONT_HEADER,
                fg=COLOR_GOLD,
                bg=COLOR_BG).pack()

        tk.Label(settings_header_frame,
                text="Adjust AI model behavior and performance",
                font=FONT_SMALL,
                fg=COLOR_TEXT,
                bg=COLOR_BG).pack()

        # Separator
        sep3 = tk.Frame(settings_frame, height=2, bg=COLOR_GOLD)
        sep3.pack(fill="x", padx=20, pady=5)

        # Settings Frame
        params_frame = tk.LabelFrame(settings_frame,
                                    text="Model Parameters",
                                    font=FONT_SECTION,
                                    fg=COLOR_GOLD,
                                    bg=COLOR_BG,
                                    relief=tk.RIDGE,
                                    borderwidth=2)
        params_frame.pack(fill="x", padx=20, pady=10, ipadx=10, ipady=10)

        # Temperature
        temp_frame = tk.Frame(params_frame, bg=COLOR_BG)
        temp_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(temp_frame,
                text="Temperature:",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=15,
                anchor="w").pack(side="left")

        temp_label = tk.Label(temp_frame,
                             textvariable=self.temp_var,
                             font=FONT_TEXT,
                             fg=COLOR_GOLD,
                             bg=COLOR_BG,
                             width=5)
        temp_label.pack(side="right", padx=5)

        temp_scale = tk.Scale(temp_frame,
                             from_=0.1,
                             to=1.5,
                             variable=self.temp_var,
                             orient=tk.HORIZONTAL,
                             resolution=0.1,
                             length=200,
                             bg=COLOR_BG,
                             fg=COLOR_TEXT,
                             troughcolor=COLOR_DARK,
                             highlightbackground=COLOR_BG,
                             sliderrelief=tk.RAISED)
        temp_scale.pack(side="left", padx=5, fill="x", expand=True)

        # Timeout
        timeout_frame = tk.Frame(params_frame, bg=COLOR_BG)
        timeout_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(timeout_frame,
                text="Timeout (seconds):",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=15,
                anchor="w").pack(side="left")

        timeout_spin = tk.Spinbox(timeout_frame,
                                 from_=5,
                                 to=120,
                                 textvariable=self.timeout_var,
                                 font=FONT_TEXT,
                                 bg=COLOR_DARK,
                                 fg=COLOR_TEXT,
                                 insertbackground=COLOR_TEXT,
                                 relief=tk.SUNKEN,
                                 borderwidth=1,
                                 width=10)
        timeout_spin.pack(side="left", padx=5)

        # Context Length
        context_frame = tk.Frame(params_frame, bg=COLOR_BG)
        context_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(context_frame,
                text="Context Length:",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=15,
                anchor="w").pack(side="left")

        context_options = ["1024", "2048", "4096", "8192"]
        context_dropdown = self._create_dropdown_menu(context_frame, self.context_var, context_options)
        context_dropdown.pack(side="left", padx=5)

        # Batch Size
        batch_frame = tk.Frame(params_frame, bg=COLOR_BG)
        batch_frame.pack(fill="x", padx=10, pady=5)

        tk.Label(batch_frame,
                text="Batch Size:",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=15,
                anchor="w").pack(side="left")

        batch_spin = tk.Spinbox(batch_frame,
                               from_=1,
                               to=16,
                               textvariable=self.batch_var,
                               font=FONT_TEXT,
                               bg=COLOR_DARK,
                               fg=COLOR_TEXT,
                               insertbackground=COLOR_TEXT,
                               relief=tk.SUNKEN,
                               borderwidth=1,
                               width=10)
        batch_spin.pack(side="left", padx=5)

    def _create_action_buttons(self):
        """Create action buttons at bottom of window."""
        actions_frame = tk.Frame(self.main_frame, bg=COLOR_BG)
        actions_frame.pack(fill="x", padx=10, pady=10)

        # Save Button
        save_btn = tk.Button(actions_frame,
                            text="💾 Save Configuration",
                            font=FONT_TEXT,
                            fg=COLOR_GOLD,
                            bg=COLOR_DARK,
                            activeforeground=COLOR_GOLD,
                            activebackground=COLOR_FRAME,
                            relief=tk.RAISED,
                            borderwidth=2,
                            padx=20,
                            pady=5,
                            command=self._save_configuration)
        save_btn.pack(side="left", padx=5)

        # Reset Button
        reset_btn = tk.Button(actions_frame,
                             text="🔄 Reset to Defaults",
                             font=FONT_TEXT,
                             fg=COLOR_GOLD,
                             bg=COLOR_DARK,
                             activeforeground=COLOR_GOLD,
                             activebackground=COLOR_FRAME,
                             relief=tk.RAISED,
                             borderwidth=2,
                             padx=20,
                             pady=5,
                             command=self._reset_configuration)
        reset_btn.pack(side="left", padx=5)

        # Status Label
        self.status_label = tk.Label(actions_frame,
                               text="✅ Ready",
                               font=FONT_SMALL,
                               fg=COLOR_TEXT,
                               bg=COLOR_BG)
        self.status_label.pack(side="right", padx=5)

    def _create_input_field(self, parent, label_text, variable, width=None):
        """Create a labeled input field."""
        frame = tk.Frame(parent, bg=COLOR_BG)
        frame.pack(fill="x", padx=10, pady=5)

        tk.Label(frame,
                text=label_text,
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                width=15,
                anchor="w").pack(side="left")

        entry = tk.Entry(frame,
                        textvariable=variable,
                        font=FONT_TEXT,
                        bg=COLOR_DARK,
                        fg=COLOR_TEXT,
                        insertbackground=COLOR_TEXT,
                        relief=tk.SUNKEN,
                        borderwidth=1,
                        width=width)
        entry.pack(side="left", padx=5)
        return entry

    def _create_dropdown_menu(self, parent, variable, options):
        """Create a dropdown menu with custom styling."""
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

    def _test_connection(self):
        """Test LLM connection."""
        success, message = self.module.test_llm_connection(
            self.ip_var.get(),
            self.port_var.get()
        )

        if success:
            messagebox.showinfo("Connection Test", message)
        else:
            messagebox.showerror("Connection Test", message)

    def _browse_model(self):
        """Browse for ML model file."""
        filetypes = [
            ("PyTorch Models", "*.pt *.pth *.ckpt"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="Select ML Model File",
            filetypes=filetypes
        )
        if filename:
            self.model_path_var.set(filename)

    def _validate_model(self):
        """Validate ML model path."""
        path = self.model_path_var.get()
        valid, message = self.module.validate_model_path(path)

        if valid:
            messagebox.showinfo("Model Validation", f"✅ {message}")
        else:
            messagebox.showerror("Model Validation", f"❌ {message}")

    def _save_configuration(self):
        """Save configuration to file."""
        # Collect all values
        config_data = {
            'llm_api_ip': self.ip_var.get(),
            'llm_api_port': self.port_var.get(),
            'api_key': self.api_key_var.get(),
            'model_type': self.model_type_var.get(),
            'ml_model_path': self.model_path_var.get(),
            'use_gpu': self.use_gpu_var.get(),
            'inference_batch_size': self.batch_var.get(),
            'max_context_length': int(self.context_var.get()),
            'temperature': self.temp_var.get(),
            'timeout_seconds': self.timeout_var.get()
        }

        # Update module config
        self.module.config.update(config_data)

        # Save to file
        if self.module.save_config():
            messagebox.showinfo("Success", "✅ Configuration saved successfully!")
            self.status_label.config(text="✅ Configuration saved")
        else:
            messagebox.showerror("Error", f"❌ Failed to save:\n{self.module.last_error}")
            self.status_label.config(text="❌ Save failed")

    def _reset_configuration(self):
        """Reset configuration to defaults."""
        if messagebox.askyesno("Confirm Reset",
                              "Reset all settings to defaults?\n\nThis will lose any custom configuration."):
            if self.module.reset_to_defaults():
                # Update UI variables
                self.ip_var.set(self.module.config['llm_api_ip'])
                self.port_var.set(self.module.config['llm_api_port'])
                self.api_key_var.set(self.module.config['api_key'])
                self.model_type_var.set(self.module.config['model_type'])
                self.model_path_var.set(self.module.config['ml_model_path'])
                self.use_gpu_var.set(self.module.config['use_gpu'])
                self.batch_var.set(self.module.config['inference_batch_size'])
                self.context_var.set(str(self.module.config['max_context_length']))
                self.temp_var.set(self.module.config['temperature'])
                self.timeout_var.set(self.module.config['timeout_seconds'])

                messagebox.showinfo("Success", "✅ Configuration reset to defaults!")
                self.status_label.config(text="✅ Reset to defaults")
            else:
                messagebox.showerror("Error", "❌ Failed to reset configuration")
                self.status_label.config(text="❌ Reset failed")

    def _create_error_gui(self, error_message):
        """Create fallback error GUI."""
        error_frame = tk.Frame(self.parent, bg=COLOR_BG)

        tk.Label(error_frame,
                text="⚠️  AI Models Configuration",
                font=FONT_HEADER,
                fg=COLOR_GOLD,
                bg=COLOR_BG).pack(pady=20)

        tk.Label(error_frame,
                text=f"Error loading configuration: {error_message[:100]}",
                font=FONT_TEXT,
                fg=COLOR_TEXT,
                bg=COLOR_BG,
                wraplength=500).pack(pady=10, padx=20)

        # Simple save button
        def simple_save():
            messagebox.showinfo("Info", "Using default configuration.")

        save_btn = tk.Button(error_frame,
                            text="Continue with Defaults",
                            font=FONT_TEXT,
                            fg=COLOR_GOLD,
                            bg=COLOR_DARK,
                            command=simple_save)
        save_btn.pack(pady=20)


# =============================================================================
# MODULE INTERFACE FUNCTIONS (Backward Compatibility)
# =============================================================================

def get_content():
    """Return module description for display in GUI."""
    return [
        "╔══════════════════════════════════════╗",
        "║         AI MODELS MODULE             ║",
        "╚══════════════════════════════════════╝",
        "",
        "🎯 PURPOSE: Configure AI models for violin performance analysis",
        "",
        "🔌 LLM API CONFIGURATION (Optional):",
        "  • Connect to local LLM (Ollama, llama.cpp)",
        "  • Or cloud API (OpenAI, Anthropic)",
        "  • Get AI-generated violin feedback",
        "  • Default: 127.0.0.1:12345",
        "",
        "🤖 ML MODEL (Future Enhancement):",
        "  • Specialized violin performance models",
        "  • Pitch/rhythm/timbre analysis",
        "  • Technical difficulty scoring",
        "",
        "⚙️ CURRENT STATUS:",
        "  • Complete configuration interface",
        "  • Connection testing utility",
        "  • Model validation",
        "  • Settings persistence",
        "",
        "🚀 QUICK START:",
        "  1. Configure your LLM server details",
        "  2. Test connection",
        "  3. Save configuration",
        "  4. All other features work independently",
    ]


def create_gui(parent):
    """Create COMPLETE independent GUI for this module."""
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog

    # Initialize module
    module = AIModels()

    # Create and return GUI
    gui = AIModelsGUI(parent, module)
    return gui.main_frame


# =============================================================================
# STANDALONE TEST (Optional)
# =============================================================================

def test_standalone():
    """Test the module GUI standalone."""
    import tkinter as tk

    root = tk.Tk()
    root.title("AI Models Module - Standalone Test")
    root.geometry(WINDOW_SIZE)
    root.configure(bg=COLOR_BG)

    # Create the GUI
    gui = create_gui(root)

    root.mainloop()


if __name__ == "__main__":
    test_standalone()