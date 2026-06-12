#!/usr/bin/env python3
"""
bpe-gui.py — GUI launcher for the BPE-VitalDB experiment pipeline.

Provides a visual pipeline overview, per-step parameter forms, and a live
output console.  All scripts are run via 'uv run python <script>'.
"""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, scrolledtext, ttk
from typing import Optional

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    NUM_GPUS = torch.cuda.device_count()
except ImportError:
    CUDA_AVAILABLE = False
    NUM_GPUS = 0

ROOT = Path(__file__).parent.parent
SCRIPTS = Path(__file__).parent

# Ensure the project root is on sys.path so `bpe` package is importable.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load model names from the registry so the list stays in sync automatically.
# Falls back to a hardcoded list when the package or torch is not yet installed.
try:
    from bpe.models import list_models as _list_models
    MODELS: list[str] = list(_list_models())  # already alphabetically sorted
except Exception:
    MODELS = sorted([
        "resnet1d", "resnet1d_mini", "resnet1d_tiny", "resnet1d_micro",
        "st_resnet", "minception", "xresnet1d", "mtae", "mtae_tr",
        "pulsewoq_resnet1d", "acfa",
    ])

# Models for bpe-browser: all models except pulsewoq_resnet1d (has dedicated browser)
MODELS_BPE: list[str] = [m for m in MODELS if m != "pulsewoq_resnet1d"]


def get_device_choices() -> list[str]:
    """Get available device options based on CUDA availability."""
    devices = ["auto", "cpu", "cuda"]
    if CUDA_AVAILABLE and NUM_GPUS > 0:
        devices.extend(f"cuda:{i}" for i in range(NUM_GPUS))
    return devices


def get_device_choices_with_blank() -> list[str]:
    """Get available device options with blank option for auto-detect."""
    devices = ["", "cpu", "cuda"]
    if CUDA_AVAILABLE and NUM_GPUS > 0:
        devices.extend(f"cuda:{i}" for i in range(NUM_GPUS))
    return devices

# ─── Pipeline definition ──────────────────────────────────────────────────────
# Each param tuple: (flag, widget_type, default, help_text [, choices])
#   widget_type: "entry"       — text; multiple space-separated tokens → multi-arg
#                "int"/"float" — text, single value
#                "bool"        — checkbox; adds --flag when checked
#                "dir"         — text + directory-browse button
#                "file"        — text + file-browse button
#                "positional_dir" — positional arg (no --flag prefix) + browse
#                "combo"       — readonly dropdown
#                "combo_free"  — editable dropdown

PIPELINE = [
    {
        "id": "check_cuda",
        "label": "Check CUDA",
        "category": "Environment",
        "script": "check-cuda.py",
        "desc": "Verify CUDA / GPU availability and list device information.",
        "gui": False,
        "params": [],
    },
    {
        "id": "download",
        "label": "Download VitalDB",
        "category": "Data Acquisition",
        "script": "download-vitaldb.py",
        "desc": "Download .vital files from VitalDB (up to 6,388 surgical cases, ~200 GB total).",
        "gui": False,
        "params": [
            ("output-dir",    "dir",   "data/vitaldb", "Destination directory"),
            ("max-cases",     "int",   "",             "Max cases to download (blank = all)"),
            ("start-case",    "int",   "1",            "First case ID"),
            ("end-case",      "int",   "6388",         "Last case ID"),
            ("workers",       "int",   "4",            "Parallel download threads"),
            ("filter-tracks", "bool",  False,          "Only cases with PPG + ABP tracks"),
            ("no-resume",     "bool",  False,          "Re-download already-existing files"),
        ],
    },
    {
        "id": "vdb_browse",
        "label": "Browse VitalDB",
        "category": "Data Acquisition",
        "script": "vitaldb-browser.py",
        "desc": "Interactive GUI waveform browser for raw .vital files (PPG, ABP, ECG).",
        "gui": True,
        "params": [
            ("data-dir", "dir", "data/vitaldb", "Directory containing .vital files"),
            ("case",     "int", "",             "Open a specific case ID on startup (optional)"),
        ],
    },
    {
        "id": "construct",
        "label": "Build Dataset",
        "category": "Dataset",
        "script": "construct-dataset.py",
        "desc": "Segment .vital files into train/val/test NPZ datasets (8 s windows, 50% overlap).",
        "gui": False,
        "params": [
            ("data-dir",     "dir",   "data/vitaldb", "Source directory of .vital files"),
            ("output-dir",   "dir",   "data/dataset", "Root output directory"),
            ("split",        "entry", "0.7 0.1 0.2",  "Train / val / test ratios (3 space-separated numbers)"),
            ("target-hz",    "int",   "125",           "Target PPG sample rate (Hz)"),
            ("segment-sec",  "int",   "8",             "Window duration in seconds"),
            ("seed",         "int",   "42",            "Random seed for case shuffling"),
        ],
    },
    {
        "id": "ds_browse",
        "label": "Browse Dataset",
        "category": "Dataset",
        "script": "dataset-browser.py",
        "desc": "Interactive GUI browser for preprocessed NPZ segments (PPG + SBP / DBP labels).",
        "gui": True,
        "params": [
            ("dataset-dir", "dir", "data/dataset", "Root dataset directory"),
            ("target-hz",   "int", "125",           "PPG sample rate used when building the dataset"),
        ],
    },
    {
        "id": "psd_browse",
        "label": "PSD Browser",
        "category": "Dataset",
        "script": "psd-browser.py",
        "desc": "Interactive GUI browser for PPG waveform PSD analysis and power_ratio inspection.",
        "gui": True,
        "params": [
            ("dataset-dir", "dir", "data/dataset", "Root dataset directory"),
            ("target-hz",   "int", "125",           "PPG sample rate used when building the dataset"),
            ("nperseg",     "int", "256",           "Welch segment length"),
        ],
    },
    {
        "id": "spectro_browse",
        "label": "Spectrogram Browser",
        "category": "Dataset",
        "script": "spectro-browser.py",
        "desc": "Interactive GUI browser for PPG spectrogram analysis and time-varying frequency inspection.",
        "gui": True,
        "params": [
            ("dataset-dir", "dir", "data/dataset", "Root dataset directory"),
            ("target-hz",   "int", "125",           "PPG sample rate used when building the dataset"),
            ("nperseg",     "int", "128",           "Analysis window length for Welch / spectrogram"),
            ("noverlap",    "int", "64",            "Overlap between adjacent spectrogram windows"),
        ],
    },
    {
        "id": "ds_stats",
        "label": "Dataset Statistics",
        "category": "Dataset",
        "script": "dataset-statistic.py",
        "desc": "Compute dataset statistics and generate SBP / DBP distribution plots.",
        "gui": False,
        "params": [
            ("dataset-dir", "dir", "data/dataset", "Root dataset directory"),
        ],
    },
    {
        "id": "share_data",
        "label": "Share Data",
        "category": "Dataset",
        "script": "share-data.py",
        "desc": "Serve the data/ folder over HTTP so another machine on the LAN can pull it.",
        "gui": False,
        "params": [
            ("port",     "int",   "8888",    "TCP port to listen on"),
            ("bind",     "entry", "0.0.0.0", "Bind address"),
            ("data-dir", "dir",   "data",    "Directory to serve"),
        ],
    },
    {
        "id": "print_model",
        "label": "Print Model",
        "category": "Model",
        "script": "print-model.py",
        "desc": "Print layer structure, output tensor shapes, and parameter count for any model.",
        "gui": False,
        "params": [
            ("model",        "combo_free", "all", "Model name or 'all'", MODELS + ["all"]),
            ("input-length", "int",        "1000", "PPG segment length in samples (8 s @ 125 Hz)"),
            ("batch-size",   "int",        "1",    "Batch size for the dummy forward pass"),
            ("device",       "combo",      "cpu",  "Compute device", ["cpu", "cuda", "auto"]),
        ],
    },
    {
        "id": "train",
        "label": "Train Model",
        "category": "Training",
        "script": "train-model.py",
        "desc": "Train a deep learning model for BP estimation from PPG waveforms.",
        "gui": False,
        "params": [
            ("model",              "combo",      MODELS[0], "Model architecture", MODELS),
            ("dataset-dir",        "dir",        "data/dataset", "Root dataset directory"),
            ("output-dir",         "dir",        "data/models",  "Root directory for saved runs"),
            ("epochs",             "int",        "100",   "Maximum training epochs"),
            ("batch-size",         "int",        "256",   "Mini-batch size"),
            ("lr",                 "float",      "1e-3",  "Initial learning rate"),
            ("weight-decay",       "float",      "1e-4",  "AdamW weight decay"),
            ("patience",           "int",        "15",    "Early-stopping patience (epochs)"),
            ("seed",               "int",        "42",    "Random seed"),
            ("device",             "combo_free", "auto",  "Compute device",
             get_device_choices()),
            ("workers",            "int",        "4",     "DataLoader worker processes"),
            ("preload",            "bool",       False,   "Load all segments into RAM before training"),
            ("no-normalize",       "bool",       False,   "Skip per-segment z-score normalization"),
            ("resume",             "file",       "",      "Resume from checkpoint .pt file (optional)"),
            ("no-aug-noise",       "bool",       False,   "Disable Gaussian noise augmentation"),
            ("no-aug-scale",       "bool",       False,   "Disable amplitude scaling augmentation"),
            ("no-aug-shift",       "bool",       False,   "Disable circular time-shift augmentation"),
            ("no-aug-mask",        "bool",       False,   "Disable random masking augmentation"),
            ("no-patient-balance", "bool",       False,   "Disable per-patient balanced sampling"),
        ],
    },
    {
        "id": "train_status",
        "label": "Training Status",
        "category": "Training",
        "script": "train-status.py",
        "desc": "Plot loss and MAE curves from a completed or in-progress training run.",
        "gui": False,
        "params": [
            ("run_dir", "positional_dir", "data/models/resnet1d",
             "Run directory (data/models/<model>)"),
            ("no-save", "bool", False, "Print summary only; skip writing PNG files"),
        ],
    },
    {
        "id": "eval_model",
        "label": "Eval Model",
        "category": "Evaluation",
        "script": "eval-model.py",
        "desc": "Evaluate a trained model on the held-out test set (MAE, RMSE, BHS grade, AAMI).",
        "gui": False,
        "params": [
            ("run_dir",      "positional_dir", "data/models/resnet1d",
             "Run directory containing best.pt and config.json"),
            ("dataset-dir",  "dir",   "data/dataset", "Root dataset directory"),
            ("device",       "combo", "auto",   "Compute device", ["auto", "cpu", "cuda"]),
            ("batch-size",   "int",   "512",    "Inference batch size"),
            ("no-normalize", "bool",  False,    "Skip z-score normalization"),
        ],
    },
    {
        "id": "eval_pulsewoq",
        "label": "Eval PulseWoQ",
        "category": "Evaluation",
        "script": "eval-model-pulsewoq.py",
        "desc": "Quality-aware evaluation for pulsewoq_resnet1d (Scenarios A and B).",
        "gui": False,
        "params": [
            ("run_dir",      "positional_dir", "data/models/pulsewoq_resnet1d",
             "Run directory containing best.pt and config.json"),
            ("dataset-dir",  "dir",   "data/dataset", "Root dataset directory"),
            ("device",       "combo", "auto",   "Compute device", ["auto", "cpu", "cuda"]),
            ("batch-size",   "int",   "512",    "Inference batch size"),
            ("no-normalize", "bool",  False,    "Skip z-score normalization"),
            ("max-n",        "int",   "16",     "Max repeated measurements per case (Scenario B)"),
            ("n-trials",     "int",   "200",    "Sampling trials per N value (Scenario B)"),
        ],
    },
    {
        "id": "bpe_browse",
        "label": "Browse BPE Results",
        "category": "Evaluation",
        "script": "bpe-browser.py",
        "desc": "Interactive GUI browser for PPG segments with BPE model predictions (all models except pulsewoq_resnet1d).",
        "gui": True,
        "params": [
            ("dataset-dir", "dir",        "data/dataset", "Root dataset directory"),
            ("models-dir",  "dir",        "data/models",  "Root models directory"),
            ("model",       "combo_free", "",             "Model to pre-select on startup (blank = first available)",
             MODELS_BPE),
            ("device",      "combo_free", "",             "Inference device (blank = auto-detect)",
             get_device_choices_with_blank()),
            ("target-hz",   "int",        "125",          "PPG sample rate (Hz)"),
        ],
    },
    {
        "id": "pulse_browse",
        "label": "Browse Pulse Results",
        "category": "Evaluation",
        "script": "pulse-browser.py",
        "desc": "Interactive GUI browser for PPG segments with pulsewoq_resnet1d predictions.",
        "gui": True,
        "params": [
            ("dataset-dir", "dir",        "data/dataset", "Root dataset directory"),
            ("models-dir",  "dir",        "data/models",  "Root models directory"),
            ("device",      "combo_free", "",             "Inference device (blank = auto-detect)",
             get_device_choices_with_blank()),
            ("target-hz",   "int",        "125",          "PPG sample rate (Hz)"),
        ],
    },
    {
        "id": "collect",
        "label": "Collect Results",
        "category": "Analysis",
        "script": "collect-result.py",
        "desc": "Collect eval results, images, and checkpoints from all model run directories.",
        "gui": False,
        "params": [
            ("models-dir",  "dir", "data/models", "Root models directory"),
            ("images-dir",  "dir", "images",       "Output directory for PNG images"),
            ("logs-dir",    "dir", "logs",          "Output directory for JSON / CSV logs"),
            ("pt-dir",      "dir", "models",        "Output directory for best.pt files"),
        ],
    },
    {
        "id": "overview",
        "label": "Overview Graph",
        "category": "Analysis",
        "script": "generate-overview-graph.py",
        "desc": "Generate parameter-count vs metric scatter plots across all trained models.",
        "gui": False,
        "params": [
            ("models-dir",  "dir", "data/models", "Root models directory"),
            ("output-dir",  "dir", "images",       "Output directory for PNG files"),
        ],
    },
]

# ─── Status display helpers ──────────────────────────────────────────────────
_SYM = {"idle": "○", "running": "◉", "done": "●", "error": "✕"}
_CLR = {"idle": "#777788", "running": "#00BFFF", "done": "#4DB870", "error": "#FF5555"}


# ─── Application ─────────────────────────────────────────────────────────────

class BPEApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("BPE-VitalDB  —  Blood Pressure Estimation Pipeline")
        self.geometry("1320x900")
        self.minsize(1000, 660)

        self._selected_id: Optional[str] = None
        self._running_id:  Optional[str] = None
        self._process:     Optional[subprocess.Popen] = None
        self._out_q:       queue.Queue = queue.Queue()
        self._statuses:    dict = {s["id"]: "idle" for s in PIPELINE}
        self._step_map:    dict = {s["id"]: s       for s in PIPELINE}

        # Current config-panel widgets (replaced on each step selection)
        self._run_btn:    Optional[ttk.Button] = None
        self._stop_btn:   Optional[ttk.Button] = None
        self._param_vars: dict = {}           # flag -> (tk.Variable, wtype)

        # Sidebar label refs: id -> (row_frame, status_label, text_label)
        self._sw: dict = {}

        self._build_ui()
        self._select_step(PIPELINE[0]["id"])
        self.after(100, self._poll_q)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame",  background="#f0f0f4")
        style.configure("TLabel",  background="#f0f0f4")
        style.configure("TLabelframe", background="#f0f0f4")
        style.configure("TLabelframe.Label",
                        font=("TkDefaultFont", 9, "bold"), foreground="#445")
        style.configure("Header.TLabel",
                        font=("TkDefaultFont", 13, "bold"), background="#f0f0f4")
        style.configure("Desc.TLabel",
                        foreground="#666677", background="#f0f0f4")
        style.configure("Cat.TLabel",
                        font=("TkDefaultFont", 8, "bold"),
                        foreground="#8888aa", background="#23243a")
        style.configure("Run.TButton",
                        font=("TkDefaultFont", 10, "bold"),
                        foreground="#ffffff", background="#1464c0")
        style.map("Run.TButton",
                  background=[("active", "#1a7de8"), ("disabled", "#8899aa")])

        h = ttk.PanedWindow(self, orient="horizontal")
        h.pack(fill="both", expand=True)

        # ── Sidebar ──────────────────────────────────────────────────────────
        sidebar = tk.Frame(h, bg="#23243a", width=240)
        sidebar.pack_propagate(False)
        h.add(sidebar, weight=0)

        tk.Label(sidebar, text=" Pipeline Steps",
                 font=("TkDefaultFont", 11, "bold"),
                 bg="#23243a", fg="#d0d0e8", anchor="w").pack(
                     fill="x", pady=(12, 6))
        tk.Frame(sidebar, bg="#44475a", height=1).pack(fill="x")

        steps_canvas = tk.Canvas(sidebar, bg="#23243a", highlightthickness=0)
        steps_scroll = ttk.Scrollbar(sidebar, orient="vertical",
                                     command=steps_canvas.yview)
        self._steps_inner = tk.Frame(steps_canvas, bg="#23243a")
        self._steps_inner.bind(
            "<Configure>",
            lambda e: steps_canvas.configure(
                scrollregion=steps_canvas.bbox("all")))
        steps_canvas.create_window((0, 0), window=self._steps_inner, anchor="nw")
        steps_canvas.configure(yscrollcommand=steps_scroll.set)
        steps_canvas.pack(side="left", fill="both", expand=True)
        steps_scroll.pack(side="right", fill="y")

        # Mouse-wheel scrolling on sidebar
        def _on_wheel(event):
            steps_canvas.yview_scroll(
                int(-1 * (event.delta / 120)), "units")
        steps_canvas.bind_all("<MouseWheel>", _on_wheel)

        self._populate_sidebar()

        # ── Right pane ────────────────────────────────────────────────────────
        v = ttk.PanedWindow(h, orient="vertical")
        h.add(v, weight=1)

        self._config_outer = ttk.Frame(v)
        v.add(self._config_outer, weight=2)

        console_wrap = ttk.LabelFrame(v, text=" Output Console ")
        v.add(console_wrap, weight=1)

        self._console = scrolledtext.ScrolledText(
            console_wrap, wrap="word", state="disabled",
            font=("Courier", 9), bg="#1b1c2a", fg="#d0d0d8",
            insertbackground="#d0d0d8", relief="flat")
        self._console.pack(fill="both", expand=True, padx=4, pady=4)
        self._console.tag_config("ts",   foreground="#55557a")
        self._console.tag_config("cmd",  foreground="#8888cc")
        self._console.tag_config("info", foreground="#56b6c2")
        self._console.tag_config("ok",   foreground="#6dbb6d")
        self._console.tag_config("err",  foreground="#e06c75")
        # Mark used to overwrite tqdm-style \r progress lines in place.
        # "left" gravity keeps the mark at the start of the overwriteable
        # region even after text is inserted to its right.
        self._console.mark_set("cr_start", "end")
        self._console.mark_gravity("cr_start", "left")

        ttk.Button(console_wrap, text="Clear",
                   command=self._clear_console).place(
                       relx=1.0, rely=0, anchor="ne", x=-8, y=2)

    def _populate_sidebar(self) -> None:
        current_cat = None
        for step in PIPELINE:
            cat = step.get("category", "")
            if cat != current_cat:
                current_cat = cat
                tk.Label(self._steps_inner, text=f"  {cat.upper()}",
                         font=("TkDefaultFont", 8, "bold"),
                         bg="#23243a", fg="#8888aa",
                         anchor="w").pack(fill="x", pady=(10, 2))

            row = tk.Frame(self._steps_inner, bg="#23243a", cursor="hand2")
            row.pack(fill="x", padx=6, pady=1)

            status_lbl = tk.Label(row, text="○",
                                  font=("TkDefaultFont", 11),
                                  bg="#23243a", fg="#666677", width=2)
            status_lbl.pack(side="left", padx=(4, 0))

            text_lbl = tk.Label(row, text=step["label"],
                                font=("TkDefaultFont", 10),
                                bg="#23243a", fg="#b8b8cc",
                                anchor="w", padx=4, pady=5)
            text_lbl.pack(side="left", fill="x", expand=True)

            sid = step["id"]
            for w in (row, status_lbl, text_lbl):
                w.bind("<Button-1>", lambda e, s=sid: self._select_step(s))
                w.bind("<Enter>",
                       lambda e, r=row, t=text_lbl: self._row_hover(r, t, True))
                w.bind("<Leave>",
                       lambda e, r=row, t=text_lbl, s=sid: self._row_hover(r, t, False, s))

            self._sw[sid] = (row, status_lbl, text_lbl)

    def _row_hover(self, row: tk.Frame, text_lbl: tk.Label,
                   entering: bool, sid: str = "") -> None:
        if entering:
            bg = "#353660"
        else:
            bg = "#2e3052" if sid == self._selected_id else "#23243a"
        row.config(bg=bg)
        text_lbl.config(bg=bg)

    # ── Step selection ────────────────────────────────────────────────────────

    def _select_step(self, step_id: str) -> None:
        if self._selected_id and self._selected_id in self._sw:
            row, _, tl = self._sw[self._selected_id]
            row.config(bg="#23243a")
            tl.config(bg="#23243a", fg="#b8b8cc",
                      font=("TkDefaultFont", 10))

        self._selected_id = step_id
        row, _, tl = self._sw[step_id]
        row.config(bg="#2e3052")
        tl.config(bg="#2e3052", fg="#ffffff",
                  font=("TkDefaultFont", 10, "bold"))

        self._build_config(self._step_map[step_id])

    # ── Config panel ──────────────────────────────────────────────────────────

    def _build_config(self, step: dict) -> None:
        for w in self._config_outer.winfo_children():
            w.destroy()
        self._run_btn  = None
        self._stop_btn = None
        self._param_vars = {}

        # Header
        hdr = ttk.Frame(self._config_outer)
        hdr.pack(fill="x", padx=16, pady=(12, 6))

        badge = "  [GUI app]" if step.get("gui") else ""
        ttk.Label(hdr, text=step["label"] + badge,
                  style="Header.TLabel").pack(anchor="w")
        ttk.Label(hdr, text=step["desc"],
                  style="Desc.TLabel", wraplength=760).pack(anchor="w", pady=(3, 0))

        ttk.Separator(self._config_outer).pack(fill="x", padx=10, pady=(6, 0))

        # Params area
        wrap = ttk.Frame(self._config_outer)
        wrap.pack(fill="both", expand=True, padx=16, pady=8)

        if step["params"]:
            pc = tk.Canvas(wrap, highlightthickness=0, bg="#f0f0f0")
            vs = ttk.Scrollbar(wrap, orient="vertical", command=pc.yview)
            pf = ttk.Frame(pc)
            pf.bind("<Configure>",
                    lambda e: pc.configure(scrollregion=pc.bbox("all")))
            pc.create_window((0, 0), window=pf, anchor="nw")
            pc.configure(yscrollcommand=vs.set)
            pc.pack(side="left", fill="both", expand=True)
            vs.pack(side="right", fill="y")
            pf.columnconfigure(1, weight=0)
            pf.columnconfigure(3, weight=1)

            for i, param in enumerate(step["params"]):
                self._add_param_row(pf, i, *param)
        else:
            ttk.Label(wrap,
                      text="No configurable parameters.  Click Run to execute.",
                      foreground="#888899").pack(pady=28)

        # Action bar
        ttk.Separator(self._config_outer).pack(fill="x", padx=10, pady=(0, 0))
        act = ttk.Frame(self._config_outer)
        act.pack(fill="x", padx=16, pady=10)

        run_label = "▶  Launch" if step.get("gui") else "▶  Run"
        self._run_btn = ttk.Button(act, text=run_label, style="Run.TButton",
                                   command=lambda s=step: self._run(s))
        self._run_btn.pack(side="left", padx=(0, 10))

        if not step.get("gui"):
            self._stop_btn = ttk.Button(act, text="⏹  Stop",
                                        command=self._stop)
            self._stop_btn.pack(side="left")

        self._status_lbl = ttk.Label(act, text="", foreground="#888899")
        self._status_lbl.pack(side="left", padx=14)

        self._sync_buttons()

    def _add_param_row(self, frame: ttk.Frame, row: int,
                       flag: str, wtype: str, default,
                       help_text: str, *extras) -> None:
        choices = extras[0] if extras else []

        # Column 0: flag name
        lbl = flag if wtype == "positional_dir" else f"--{flag}"
        ttk.Label(frame, text=lbl,
                  font=("Courier", 9)).grid(
                      row=row, column=0, sticky="w",
                      padx=(0, 12), pady=3)

        # Column 1: widget  (column 2: optional browse button)
        if wtype == "bool":
            var = tk.BooleanVar(value=bool(default))
            ttk.Checkbutton(frame, variable=var).grid(
                row=row, column=1, sticky="w", pady=3)

        elif wtype in ("dir", "positional_dir", "file"):
            if default and not os.path.isabs(str(default)):
                abs_val = str(ROOT / default)
            else:
                abs_val = str(default) if default else ""
            var = tk.StringVar(value=abs_val)
            ttk.Entry(frame, textvariable=var, width=52).grid(
                row=row, column=1, sticky="ew", pady=3)
            browse = self._browse_file if wtype == "file" else self._browse_dir
            ttk.Button(frame, text="…", width=3,
                       command=lambda v=var: browse(v)).grid(
                           row=row, column=2, padx=(4, 0), pady=3)

        elif wtype in ("combo", "combo_free"):
            state = "readonly" if wtype == "combo" else "normal"
            var = tk.StringVar(value=str(default))
            ttk.Combobox(frame, textvariable=var, values=choices,
                         width=24, state=state).grid(
                             row=row, column=1, sticky="w", pady=3)

        else:  # "entry", "int", "float"
            var = tk.StringVar(value="" if default == "" else str(default))
            ttk.Entry(frame, textvariable=var, width=26).grid(
                row=row, column=1, sticky="w", pady=3)

        # Column 3: help text
        ttk.Label(frame, text=help_text,
                  foreground="#888899",
                  font=("TkDefaultFont", 9)).grid(
                      row=row, column=3, sticky="w",
                      padx=(12, 0), pady=3)

        self._param_vars[flag] = (var, wtype)

    # ── File / directory browsing ─────────────────────────────────────────────

    def _browse_dir(self, var: tk.StringVar) -> None:
        cur = var.get()
        init = cur if os.path.isdir(cur) else str(ROOT)
        d = filedialog.askdirectory(initialdir=init, parent=self)
        if d:
            var.set(d)

    def _browse_file(self, var: tk.StringVar) -> None:
        cur = var.get()
        init_dir = os.path.dirname(cur) if cur else str(ROOT)
        f = filedialog.askopenfilename(
            initialdir=init_dir, parent=self,
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
        if f:
            var.set(f)

    # ── Command building ─────────────────────────────────────────────────────

    def _build_cmd(self, step: dict) -> list:
        script = str(SCRIPTS / step["script"])
        cmd = ["uv", "run", "python", script]
        positionals: list = []
        kwargs: list = []

        for param in step["params"]:
            flag  = param[0]
            wtype = param[1]
            if flag not in self._param_vars:
                continue
            var, _ = self._param_vars[flag]

            if wtype == "bool":
                if var.get():
                    kwargs.append(f"--{flag}")

            elif wtype == "positional_dir":
                val = var.get().strip()
                if val:
                    positionals.append(val)

            else:
                val = var.get().strip()
                if val:
                    if wtype == "entry":
                        # Multi-token support (e.g. --split 0.6 0.2 0.2)
                        kwargs.extend([f"--{flag}"] + val.split())
                    else:
                        kwargs.extend([f"--{flag}", val])

        return cmd + positionals + kwargs

    # ── Running steps ─────────────────────────────────────────────────────────

    def _run(self, step: dict) -> None:
        if self._running_id is not None:
            self._log("⚠  Another process is still running.  Stop it first.", "err")
            return

        cmd = self._build_cmd(step)

        self._log(f"\n{'─' * 68}", "ts")
        self._log(f"▶  {step['label']}", "info")
        self._log("   " + " ".join(cmd), "cmd")
        self._log(f"{'─' * 68}", "ts")

        # GUI apps: launch and do not track stdout
        if step.get("gui"):
            try:
                subprocess.Popen(cmd, cwd=str(ROOT))
                self._log("GUI application launched.", "ok")
                self._set_status(step["id"], "done")
            except Exception as exc:
                self._log(f"Launch failed: {exc}", "err")
                self._set_status(step["id"], "error")
            return

        # Headless scripts: stream output
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT),
                env=env,
            )
        except Exception as exc:
            self._log(f"Failed to start: {exc}", "err")
            self._set_status(step["id"], "error")
            return

        self._running_id = step["id"]
        self._set_status(step["id"], "running")
        self._sync_buttons()

        def _reader() -> None:
            assert self._process is not None
            stdout = self._process.stdout  # binary pipe
            buf = ""
            while True:
                raw = stdout.read(1)  # type: ignore[union-attr]
                if not raw:
                    break
                ch = raw.decode("utf-8", errors="replace")
                if ch == "\n":
                    self._out_q.put(("line", buf))
                    buf = ""
                elif ch == "\r":
                    # Peek at next byte to distinguish \r\n from lone \r.
                    raw2 = stdout.read(1)  # type: ignore[union-attr]
                    if raw2 == b"\n" or not raw2:
                        # Windows line ending or \r at EOF → treat as newline.
                        self._out_q.put(("line", buf))
                        buf = ""
                    else:
                        # Lone \r (tqdm carriage-return update).
                        self._out_q.put(("cr", buf))
                        buf = raw2.decode("utf-8", errors="replace")
                else:
                    buf += ch
            if buf:
                self._out_q.put(("line", buf))
            rc = self._process.wait()
            self._out_q.put(("done", step["id"], rc))

        threading.Thread(target=_reader, daemon=True).start()

    def _stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()

    # ── Output polling ────────────────────────────────────────────────────────

    def _poll_q(self) -> None:
        try:
            while True:
                item = self._out_q.get_nowait()
                if item[0] == "line":
                    self._log(item[1])
                elif item[0] == "cr":
                    self._log_cr(item[1])
                elif item[0] == "done":
                    _, sid, rc = item
                    if rc == 0:
                        self._log("✓  Completed successfully (exit 0).", "ok")
                        self._set_status(sid, "done")
                    else:
                        self._log(f"✕  Exited with code {rc}.", "err")
                        self._set_status(sid, "error")
                    self._running_id = None
                    self._process    = None
                    self._sync_buttons()
        except queue.Empty:
            pass
        self.after(100, self._poll_q)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _sync_buttons(self) -> None:
        is_running = self._running_id is not None
        if self._run_btn:
            self._run_btn.config(state="disabled" if is_running else "normal")
        if self._stop_btn:
            self._stop_btn.config(state="normal" if is_running else "disabled")
        if hasattr(self, "_status_lbl"):
            if is_running:
                sid = self._running_id
                name = self._step_map[sid]["label"] if sid else ""
                self._status_lbl.config(text=f"Running: {name}…",
                                        foreground="#00BFFF")
            else:
                self._status_lbl.config(text="", foreground="#888899")

    def _log(self, text: str, tag: str = "") -> None:
        """Append a complete line. Also finalises any pending \\r progress line."""
        self._console.config(state="normal")
        self._console.delete("cr_start", "end")
        if tag:
            self._console.insert("end", text + "\n", tag)
        else:
            self._console.insert("end", text + "\n")
        self._console.mark_set("cr_start", "end")
        self._console.see("end")
        self._console.config(state="disabled")

    def _log_cr(self, text: str) -> None:
        """Overwrite the current progress line in place (handles tqdm \\r)."""
        self._console.config(state="normal")
        self._console.delete("cr_start", "end")
        self._console.insert("end", text)
        self._console.see("end")
        self._console.config(state="disabled")

    def _clear_console(self) -> None:
        self._console.config(state="normal")
        self._console.delete("1.0", "end")
        self._console.mark_set("cr_start", "end")
        self._console.config(state="disabled")

    def _set_status(self, step_id: str, status: str) -> None:
        self._statuses[step_id] = status
        if step_id in self._sw:
            _, sl, _ = self._sw[step_id]
            sl.config(text=_SYM[status], fg=_CLR[status])


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = BPEApp()
    app.mainloop()
