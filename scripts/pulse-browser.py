"""
Pulse Browser - inspect PPG segments with pulsewo_resnet1d model predictions.

Extends dataset-browser with:
  - live SBP/DBP predictions from the best pulsewo_resnet1d checkpoint
  - quality-weight visualisation: the model learns per-segment quality scores;
    softmax of these scores is shown as a bar chart below the PPG waveform and
    as a coloured intensity overlay behind the signal, revealing which 1-second
    windows contributed most to the prediction.

Layout
------
  Left panel  : split selector + sortable case list + model status
  Right panel : PPG waveform (with quality-weight shading)
              + quality-weight bar chart (time-aligned, below PPG)
  Info bar    : case ID | ground-truth SBP/DBP | predicted SBP/DBP | error

Navigation
----------
  <- / -> : previous / next segment      Up / Down : previous / next case
  Slider : drag to any segment          Jump  : type segment number + Enter

Usage
-----
    uv run python scripts/pulse-browser.py [OPTIONS]

Options
-------
    --dataset-dir   Root dataset directory  (default: data/dataset)
    --models-dir    Root models directory   (default: data/models)
    --device        Inference device        (default: cuda if available, else cpu)
    --target-hz     PPG sample rate         (default: 125)
"""

import argparse
import queue
import sys
import threading
import tkinter as tk
import tkinter.ttk as ttk
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tqdm import tqdm

# Project root on sys.path so that `bpe` package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F


# -- Korean font (no-op if unavailable) ---------------------------------------
def _set_cjk_font() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim"):
        if name in available:
            matplotlib.rc("font", family=name)
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_set_cjk_font()

SPLITS     = ("train", "val", "test")
MODEL_NAME = "pulsewo_resnet1d"

# -- Light colour palette ------------------------------------------------------
BG_DARK   = "#ffffff"   # plot / main background
BG_MID    = "#f0f0f7"   # left panel background
BG_PANEL  = "#e8e8f2"   # info / nav bar background
FG_DIM    = "#888899"   # secondary text
FG_NORM   = "#222233"   # primary text
FG_BRIGHT = "#1133cc"   # accent text (dark blue)
ACCENT    = "#2255cc"   # selection / hover accent

PPG_COLOR    = "#1a8855"   # PPG waveform (dark green)
SBP_COLOR    = "#cc2200"   # ground-truth SBP (dark red)
DBP_COLOR    = "#cc7700"   # ground-truth DBP (dark amber)
GT_SBP_COLOR = "#0044ee"   # GT SBP reference in ax_bp (light blue)
GT_DBP_COLOR = "#0022aa"   # GT DBP reference in ax_bp (blue)
PRED_SBP_CLR = "#dd5533"   # predicted SBP
PRED_DBP_CLR = "#dd9922"   # predicted DBP
WEIGHT_COLOR = "#2255bb"   # quality-weight triangles / shading
GRID_CLR     = "#e8e8ee"   # grid lines

SPLIT_BTN_ACTIVE   = {"bg": "#2255cc", "fg": "white",   "relief": "flat"}
SPLIT_BTN_INACTIVE = {"bg": "#f0f0f7", "fg": "#666677", "relief": "flat"}


# -- Model utilities -----------------------------------------------------------

def find_best_pt(models_dir: Path) -> Path | None:
    """Return path to best.pt in the most recent run of MODEL_NAME, or None."""
    model_dir = models_dir / MODEL_NAME
    if not model_dir.exists():
        return None

    # Check for best.pt directly in model_dir
    pt = model_dir / "best.pt"
    if pt.exists():
        return pt

    # Check for best.pt in run subdirectories
    run_dirs = sorted(d for d in model_dir.iterdir() if d.is_dir())
    for run_dir in reversed(run_dirs):
        pt = run_dir / "best.pt"
        if pt.exists():
            return pt
    return None


def load_model(checkpoint_path: Path, device: str):
    """Load PulseWOResNet1D from a trainer checkpoint. Returns model or None."""
    from bpe.models.pulsewo_resnet1d import PulseWOResNet1D

    try:
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=device)

        # Trainer saves {"model_state_dict": ..., "epoch": ..., ...}
        if isinstance(ckpt, dict):
            state = (
                ckpt.get("model_state_dict")
                or ckpt.get("model")
                or ckpt.get("state_dict")
                or ckpt
            )
        else:
            state = ckpt

        model = PulseWOResNet1D()
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    except Exception as exc:
        print(f"[warn] Failed to load model: {exc}", file=sys.stderr)
        return None


def infer_with_weights(
    model, ppg: np.ndarray, device: str
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Run model forward pass and return
    (sbp_pred, dbp_pred, quality_weights, bp_per_seg, q_raw).

    quality_weights : (S,)    softmax weights over the S overlapping segments.
    bp_per_seg      : (S, 2)  per-segment [SBP, DBP] predictions before weighting.
    q_raw           : (S,)    raw quality scores before softmax (backbone output).
    The model internally computes these but does not expose them through the
    standard forward(); we re-execute the computation graph step-by-step here.
    """
    # Per-segment z-score normalisation (matches training pipeline)
    mu    = float(ppg.mean())
    sigma = float(ppg.std())
    if sigma < 1e-6:
        sigma = 1e-6
    x_norm = (ppg - mu) / sigma

    x = (
        torch.from_numpy(x_norm.astype(np.float32))
        .unsqueeze(0)   # (1, L)
        .unsqueeze(0)   # (1, 1, L)
        .to(device)
    )

    seg_len = model.seg_len   # 125
    stride  = model.stride    # 62

    with torch.no_grad():
        # (1, 1, L) -> (1, 1, S, seg_len)  via overlapping unfold
        x_seg = x.unfold(2, seg_len, stride)
        B, C, S, L_ = x_seg.shape

        x_seg = x_seg.permute(0, 2, 1, 3).contiguous()   # (1, S, 1, seg_len)
        x_seg = x_seg.view(B * S, C, L_)                  # (S, 1, seg_len)

        out  = model.backbone(x_seg)                      # (S, F+1)
        bp   = out[:, : model.out_features]               # (S, F=2)
        q    = out[:, model.out_features]                 # (S,)

        w    = F.softmax(q, dim=0)                        # (S,)  sums to 1
        pred = (w.unsqueeze(-1) * bp).sum(dim=0)          # (F,)

    sbp_pred   = float(pred[0].cpu())
    dbp_pred   = float(pred[1].cpu())
    weights    = w.cpu().numpy()           # (S,)
    bp_per_seg = bp.cpu().numpy()          # (S, 2)  [sbp, dbp] per segment
    q_raw      = q.cpu().numpy()           # (S,)    raw scores before softmax
    return sbp_pred, dbp_pred, weights, bp_per_seg, q_raw


# -- CLI -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Browse PPG segments with pulsewo_resnet1d predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset-dir", type=Path, default=Path("data/dataset"),
        help="Root dataset directory (default: data/dataset)",
    )
    p.add_argument(
        "--models-dir", type=Path, default=Path("data/models"),
        help="Root models directory (default: data/models)",
    )
    p.add_argument(
        "--device", type=str, default="",
        help="Inference device: 'cpu', 'cuda', 'cuda:0', ... "
             "(default: cuda if available, else cpu)",
    )
    p.add_argument(
        "--target-hz", type=int, default=125,
        help="PPG sample rate (default: 125)",
    )
    return p.parse_args()


# -- Browser application -------------------------------------------------------

class PulseBrowser:
    LIST_WIDTH = 320
    CANVAS_W   = 1200
    WIN_H      = 920

    LIST_COLUMNS = [
        ("case", "Case ID",   80, "center"),
        ("segs", "Segments",  80, "center"),
        ("size", "Size",      70, "center"),
    ]

    def __init__(
        self,
        root: tk.Tk,
        dataset_dir: Path,
        models_dir: Path,
        device: str,
        target_hz: int,
    ):
        self.root        = root
        self.dataset_dir = dataset_dir
        self.target_hz   = target_hz
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Model state
        self._model: object | None = None
        self._model_info = "Searching for checkpoint ..."

        # App state
        self._split      = "train"
        self._npz_files: dict[str, list[Path]] = {}
        self._rows:       dict[str, list[dict]] = {}
        self._row_by_path: dict[Path, dict]     = {}
        self._meta_queue: queue.Queue           = queue.Queue()
        self._meta_total  = 0
        self._meta_done   = 0
        self._current_path: Path | None = None
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._seg_idx = 0
        self._slider_updating = False

        self._discover_files()
        self._build_ui()
        self._load_model_async(models_dir)
        self._select_split("train")
        self._start_metadata_worker()

    # -- Model loading (background) --------------------------------------------

    def _load_model_async(self, models_dir: Path) -> None:
        pt_path = find_best_pt(models_dir)
        if pt_path is None:
            self._model_info = f"No checkpoint found in {models_dir / MODEL_NAME}"
            self._refresh_model_label()
            return

        self._model_info = f"Loading {pt_path.parent.name} ..."
        self._refresh_model_label()

        threading.Thread(
            target=self._model_loader_thread,
            args=(pt_path,),
            daemon=True,
            name="model-loader",
        ).start()

    def _model_loader_thread(self, pt_path: Path) -> None:
        model = load_model(pt_path, self.device)
        self._model = model
        if model is not None:
            n_segs = (1000 - model.seg_len) // model.stride + 1
            self._model_info = (
                f"{MODEL_NAME}  |  run {pt_path.parent.name}"
                f"  |  {n_segs} segs  |  {self.device}"
            )
        else:
            self._model_info = f"Failed to load {pt_path.name}"
        self.root.after(0, self._refresh_model_label)
        # Re-draw current segment now that the model is ready
        if self._x is not None:
            self.root.after(0, lambda: self._show_segment(self._seg_idx))

    def _refresh_model_label(self) -> None:
        if hasattr(self, "_model_status_var"):
            self._model_status_var.set(self._model_info)

    # -- File discovery --------------------------------------------------------

    def _discover_files(self) -> None:
        for split in SPLITS:
            d = self.dataset_dir / split
            files = (
                sorted(d.glob("*.npz"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
                if d.exists() else []
            )
            self._npz_files[split] = files
            self._rows[split] = [self._placeholder_row(f) for f in files]
            for row in self._rows[split]:
                self._row_by_path[row["path"]] = row

    @staticmethod
    def _placeholder_row(path: Path) -> dict:
        cid = int(path.stem) if path.stem.isdigit() else 0
        return dict(path=path, case=cid, segs=0, segs_text="...",
                    size="...", size_val=0.0, metadata_loaded=False)

    # -- Metadata worker -------------------------------------------------------

    def _start_metadata_worker(self) -> None:
        self._meta_total = sum(len(f) for f in self._npz_files.values())
        if self._meta_total == 0:
            return
        threading.Thread(
            target=self._metadata_worker, daemon=True, name="metadata-loader"
        ).start()
        self.root.after(50, self._drain_metadata_queue)

    def _metadata_worker(self) -> None:
        for split in SPLITS:
            for path in tqdm(self._npz_files[split],
                             desc=f"Indexing {split}",
                             unit="file",
                             dynamic_ncols=True):
                self._meta_queue.put(("row", split, path, self._file_row(path)))
        self._meta_queue.put(("done", None, None, None))

    def _drain_metadata_queue(self) -> None:
        updated = False
        done    = False
        while True:
            try:
                kind, split, path, row = self._meta_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "done":
                done = True
                continue
            self._meta_done += 1
            stored = self._row_by_path.get(path)
            if stored is None:
                continue
            stored.update(row)
            if split == self._split:
                updated = True
                iid = str(path)
                if self._tree.exists(iid):
                    self._tree.item(iid, values=self._row_values(stored))

        if updated:
            self._update_count()

        if done:
            if self._current_path is None:
                self._status_var.set("Dataset metadata indexing complete.")
            return

        if self._current_path is None:
            self._status_var.set(
                f"Indexing {self._meta_done}/{self._meta_total}..."
            )
        self.root.after(50, self._drain_metadata_queue)

    @staticmethod
    def _file_row(path: Path) -> dict:
        try:
            with np.load(path) as d:
                n_segs = len(d["x"])
        except Exception:
            n_segs = 0
        try:
            size_kb = path.stat().st_size / 1024
        except OSError:
            size_kb = 0.0
        cid = int(path.stem) if path.stem.isdigit() else 0
        return dict(path=path, case=cid, segs=n_segs, segs_text=str(n_segs),
                    size=f"{size_kb:.0f} KB", size_val=size_kb, metadata_loaded=True)

    # -- UI construction -------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.title(f"Pulse Browser - {MODEL_NAME}")
        self.root.configure(bg=BG_DARK)
        self.root.geometry(f"{self.LIST_WIDTH + self.CANVAS_W}x{self.WIN_H}")
        self.root.minsize(860, 560)

        paned = tk.PanedWindow(
            self.root, orient="horizontal",
            bg=BG_DARK, sashwidth=5, sashrelief="flat", handlesize=0,
        )
        paned.pack(fill="both", expand=True)

        left = tk.Frame(paned, bg=BG_MID, width=self.LIST_WIDTH)
        left.pack_propagate(False)
        paned.add(left, minsize=240)

        right = tk.Frame(paned, bg=BG_DARK)
        paned.add(right, minsize=520)

        self._build_list_panel(left)
        self._build_canvas_panel(right)

        bar = tk.Frame(self.root, bg=BG_PANEL, height=22)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self._status_var = tk.StringVar(value="Select a case from the list.")
        tk.Label(
            bar, textvariable=self._status_var,
            bg=BG_PANEL, fg=FG_DIM, font=("Segoe UI", 9), anchor="w",
        ).pack(side="left", padx=8)

    # -- Left panel ------------------------------------------------------------

    def _build_list_panel(self, parent: tk.Frame) -> None:
        btn_row = tk.Frame(parent, bg=BG_MID)
        btn_row.pack(fill="x", padx=8, pady=(8, 4))
        self._split_btns: dict[str, tk.Button] = {}
        for split in SPLITS:
            b = tk.Button(
                btn_row, text=split.capitalize(),
                font=("Segoe UI", 9, "bold"), cursor="hand2",
                bd=0, padx=10, pady=4,
                command=lambda s=split: self._select_split(s),
            )
            b.pack(side="left", padx=2)
            self._split_btns[split] = b

        self._count_var = tk.StringVar()
        tk.Label(
            parent, textvariable=self._count_var,
            bg=BG_MID, fg=FG_DIM, font=("Segoe UI", 8), anchor="w",
        ).pack(fill="x", padx=10, pady=(0, 2))

        frame = tk.Frame(parent, bg=BG_MID)
        frame.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "D.Treeview",
            background=BG_DARK, foreground=FG_NORM,
            fieldbackground=BG_DARK, rowheight=22, font=("Segoe UI", 9),
        )
        style.configure(
            "D.Treeview.Heading",
            background="#d0d8f0", foreground=FG_BRIGHT,
            font=("Segoe UI", 9, "bold"), relief="flat",
        )
        style.map(
            "D.Treeview",
            background=[("selected", ACCENT)],
            foreground=[("selected", "white")],
        )

        col_ids = [c[0] for c in self.LIST_COLUMNS]
        self._tree = ttk.Treeview(
            frame, columns=col_ids, show="headings",
            style="D.Treeview", selectmode="browse",
        )
        for cid, heading, width, anchor in self.LIST_COLUMNS:
            self._tree.heading(cid, text=heading)
            self._tree.column(cid, width=width, anchor=anchor,
                              stretch=(cid == "case"))

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._tree.bind("<<TreeviewSelect>>", self._on_case_select)

        # Model status block at bottom of left panel
        model_frame = tk.Frame(parent, bg="#eeeef6")
        model_frame.pack(fill="x", padx=8, pady=(0, 6))
        tk.Label(
            model_frame, text="Model:",
            bg="#eeeef6", fg=FG_DIM, font=("Segoe UI", 8, "bold"), anchor="w",
        ).pack(anchor="w", padx=6, pady=(4, 0))
        self._model_status_var = tk.StringVar(value=self._model_info)
        tk.Label(
            model_frame, textvariable=self._model_status_var,
            bg="#eeeef6", fg=WEIGHT_COLOR, font=("Segoe UI", 8),
            anchor="w", justify="left",
            wraplength=self.LIST_WIDTH - 24,
        ).pack(anchor="w", padx=6, pady=(0, 4))

    # -- Right panel -----------------------------------------------------------

    def _build_canvas_panel(self, parent: tk.Frame) -> None:
        # -- Info bar ----------------------------------------------------------
        info_row = tk.Frame(parent, bg=BG_PANEL, height=36)
        info_row.pack(fill="x")
        info_row.pack_propagate(False)

        self._case_label = tk.Label(
            info_row, text="", bg=BG_PANEL, fg=FG_BRIGHT,
            font=("Segoe UI", 10, "bold"), anchor="w",
        )
        self._case_label.pack(side="left", padx=10)

        # Ground-truth SBP / DBP
        self._sbp_label = tk.Label(
            info_row, text="", bg=BG_PANEL, fg=SBP_COLOR,
            font=("Segoe UI", 10, "bold"),
        )
        self._sbp_label.pack(side="left", padx=(10, 2))

        self._dbp_label = tk.Label(
            info_row, text="", bg=BG_PANEL, fg=DBP_COLOR,
            font=("Segoe UI", 10, "bold"),
        )
        self._dbp_label.pack(side="left", padx=(0, 8))

        tk.Label(info_row, text="|", bg=BG_PANEL, fg=FG_DIM).pack(side="left")

        # Model predictions
        self._pred_sbp_label = tk.Label(
            info_row, text="", bg=BG_PANEL, fg=PRED_SBP_CLR,
            font=("Segoe UI", 10),
        )
        self._pred_sbp_label.pack(side="left", padx=(8, 2))

        self._pred_dbp_label = tk.Label(
            info_row, text="", bg=BG_PANEL, fg=PRED_DBP_CLR,
            font=("Segoe UI", 10),
        )
        self._pred_dbp_label.pack(side="left", padx=(0, 8))

        tk.Label(info_row, text="|", bg=BG_PANEL, fg=FG_DIM).pack(side="left")

        # Errors
        self._err_sbp_label = tk.Label(
            info_row, text="", bg=BG_PANEL, font=("Segoe UI", 9),
        )
        self._err_sbp_label.pack(side="left", padx=4)

        self._err_dbp_label = tk.Label(
            info_row, text="", bg=BG_PANEL, font=("Segoe UI", 9),
        )
        self._err_dbp_label.pack(side="left", padx=2)

        # -- Placeholder -------------------------------------------------------
        self._placeholder = tk.Label(
            parent, text="<- Select a case from the list",
            bg=BG_DARK, fg="#aaaacc", font=("Segoe UI", 14),
        )
        self._placeholder.pack(expand=True)

        # -- Matplotlib figure (3 vertically stacked axes) ---------------------
        self._fig = plt.Figure(figsize=(8.5, 7.8), facecolor=BG_DARK)
        gs = gridspec.GridSpec(
            3, 1, figure=self._fig,
            height_ratios=[3, 1, 1.5],
            left=0.07, right=0.97,
            top=0.96, bottom=0.06,
            hspace=0.62,
        )
        self._ax_ppg = self._fig.add_subplot(gs[0], facecolor=BG_DARK)
        self._ax_wt  = self._fig.add_subplot(gs[1], facecolor=BG_DARK)
        self._ax_bp  = self._fig.add_subplot(gs[2], facecolor=BG_DARK)

        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas_widget.get_tk_widget().pack_forget()
        self._canvas_packed = False

        # -- Navigation bar ----------------------------------------------------
        nav = tk.Frame(parent, bg=BG_PANEL, height=36)
        nav.pack(fill="x", side="bottom")
        nav.pack_propagate(False)

        btn_cfg = dict(
            font=("Segoe UI", 9, "bold"),
            bg="#d0d8f0", fg=FG_BRIGHT,
            activebackground=ACCENT, activeforeground="white",
            relief="flat", bd=0, padx=16, pady=4, cursor="hand2",
        )
        self._prev_btn = tk.Button(nav, text="< Prev",
                                   command=self._prev_seg, **btn_cfg)
        self._prev_btn.pack(side="left", padx=8, pady=4)

        self._seg_var = tk.StringVar(value="")
        tk.Label(
            nav, textvariable=self._seg_var,
            bg=BG_PANEL, fg=FG_DIM, font=("Segoe UI", 9), width=20,
        ).pack(side="left", padx=4)

        self._next_btn = tk.Button(nav, text="Next >",
                                   command=self._next_seg, **btn_cfg)
        self._next_btn.pack(side="left", padx=4)

        self._seg_slider = tk.Scale(
            nav, from_=1, to=1, orient="horizontal",
            showvalue=False, resolution=1,
            bg=BG_PANEL, fg=FG_DIM,
            troughcolor="#ccccdd", activebackground=ACCENT,
            highlightthickness=0, bd=0,
            sliderlength=16, width=10,
            state="disabled",
            command=self._on_slider,
        )
        self._seg_slider.pack(side="left", fill="x", expand=True, padx=(10, 8))

        tk.Label(nav, text="Jump:", bg=BG_PANEL, fg=FG_DIM,
                 font=("Segoe UI", 9)).pack(side="left", padx=(20, 2))
        self._jump_var = tk.StringVar()
        jump_entry = tk.Entry(
            nav, textvariable=self._jump_var, width=6,
            bg="#ccccdd", fg=FG_NORM, insertbackground=FG_NORM,
            relief="flat", font=("Segoe UI", 9),
        )
        jump_entry.pack(side="left")
        jump_entry.bind("<Return>", self._on_jump)

        self.root.bind("<Left>",  lambda _: self._prev_seg())
        self.root.bind("<Right>", lambda _: self._next_seg())
        self.root.bind("<Up>",    lambda _: self._prev_case())
        self.root.bind("<Down>",  lambda _: self._next_case())

    # -- Split selection -------------------------------------------------------

    def _select_split(self, split: str) -> None:
        self._split        = split
        self._current_path = None
        self._x = self._y  = None
        for s, btn in self._split_btns.items():
            btn.configure(**(SPLIT_BTN_ACTIVE if s == split else SPLIT_BTN_INACTIVE))
        self._refresh_list()
        self._clear_canvas()

    def _refresh_list(self) -> None:
        rows     = self._sorted_rows()
        selected = set(self._tree.selection())
        self._tree.delete(*self._tree.get_children())
        for row in rows:
            iid = str(row["path"])
            self._tree.insert("", "end", iid=iid, values=self._row_values(row))
            if iid in selected:
                self._tree.selection_add(iid)
        self._update_count(rows)

    @staticmethod
    def _row_values(row: dict) -> tuple:
        return (row["case"], row["segs_text"], row["size"])

    def _update_count(self, rows: list[dict] | None = None) -> None:
        rows  = self._rows[self._split] if rows is None else rows
        n     = len(rows)
        known = sum(1 for r in rows if r["metadata_loaded"])
        total = sum(r["segs"] for r in rows if r["metadata_loaded"])
        if known < n:
            self._count_var.set(f"{n} cases | {known}/{n} indexed | {total:,} segs [{self._split}]")
        else:
            self._count_var.set(f"{n} cases | {total:,} segments [{self._split}]")

    def _sorted_rows(self) -> list[dict]:
        rows = self._rows[self._split][:]
        rows.sort(key=lambda r: r["case"])
        return rows

    # -- Case selection --------------------------------------------------------

    def _on_case_select(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        path = Path(sel[0])
        if path != self._current_path:
            self._load_case(path)

    def _load_case(self, path: Path) -> None:
        self._status_var.set(f"Loading {path.name} ...")
        self.root.update_idletasks()
        try:
            data    = np.load(path)
            self._x = data["x"]
            self._y = data["y"]
        except Exception as exc:
            self._status_var.set(f"Error: {exc}")
            return
        self._current_path = path
        self._seg_idx = 0
        self._show_canvas()
        self._show_segment(0)

    def _prev_case(self) -> None:
        rows  = self._sorted_rows()
        paths = [r["path"] for r in rows]
        if self._current_path not in paths:
            return
        idx = paths.index(self._current_path)
        if idx > 0:
            self._load_case(paths[idx - 1])
            self._select_tree_item(paths[idx - 1])

    def _next_case(self) -> None:
        rows  = self._sorted_rows()
        paths = [r["path"] for r in rows]
        if self._current_path not in paths:
            return
        idx = paths.index(self._current_path)
        if idx < len(paths) - 1:
            self._load_case(paths[idx + 1])
            self._select_tree_item(paths[idx + 1])

    def _select_tree_item(self, path: Path) -> None:
        iid = str(path)
        self._tree.selection_set(iid)
        self._tree.see(iid)

    # -- Segment navigation ----------------------------------------------------

    def _prev_seg(self) -> None:
        if self._x is not None and self._seg_idx > 0:
            self._seg_idx -= 1
            self._show_segment(self._seg_idx)

    def _next_seg(self) -> None:
        if self._x is not None and self._seg_idx < len(self._x) - 1:
            self._seg_idx += 1
            self._show_segment(self._seg_idx)

    def _on_slider(self, value: str) -> None:
        if self._slider_updating or self._x is None:
            return
        idx = max(0, min(int(round(float(value))) - 1, len(self._x) - 1))
        if idx != self._seg_idx:
            self._seg_idx = idx
            self._show_segment(idx)

    def _on_jump(self, _event=None) -> None:
        try:
            idx = int(self._jump_var.get()) - 1
            if self._x is not None:
                idx = max(0, min(idx, len(self._x) - 1))
                self._seg_idx = idx
                self._show_segment(idx)
        except ValueError:
            pass
        self._jump_var.set("")

    def _configure_slider(self, n: int, enabled: bool) -> None:
        self._seg_slider.configure(
            from_=1, to=max(n, 1),
            state="normal" if enabled and n > 1 else "disabled",
        )

    def _set_slider(self, idx: int) -> None:
        self._slider_updating = True
        try:
            self._seg_slider.set(idx + 1)
        finally:
            self._slider_updating = False

    # -- Plotting --------------------------------------------------------------

    def _show_canvas(self) -> None:
        if not self._canvas_packed:
            self._placeholder.pack_forget()
            self._canvas_widget.get_tk_widget().pack(fill="both", expand=True)
            self._canvas_packed = True

    def _clear_canvas(self) -> None:
        if self._canvas_packed:
            self._canvas_widget.get_tk_widget().pack_forget()
            self._placeholder.pack(expand=True)
            self._canvas_packed = False
        for lbl in (
            self._case_label, self._sbp_label, self._dbp_label,
            self._pred_sbp_label, self._pred_dbp_label,
            self._err_sbp_label, self._err_dbp_label,
        ):
            lbl.configure(text="")
        self._seg_var.set("")
        self._configure_slider(1, enabled=False)

    def _show_segment(self, idx: int) -> None:
        if self._x is None or self._y is None:
            return

        ppg    = self._x[idx]          # (L,)
        sbp_gt = float(self._y[idx, 0])
        dbp_gt = float(self._y[idx, 1])
        n_segs = len(self._x)
        n_samp = len(ppg)
        seg_sec = n_samp / self.target_hz
        t = np.linspace(0, seg_sec, n_samp, endpoint=False)

        # -- Model inference ---------------------------------------------------
        sbp_pred = dbp_pred = None
        weights:    np.ndarray | None = None
        bp_per_seg: np.ndarray | None = None
        q_raw:      np.ndarray | None = None
        if self._model is not None:
            try:
                sbp_pred, dbp_pred, weights, bp_per_seg, q_raw = infer_with_weights(
                    self._model, ppg, self.device
                )
            except Exception as exc:
                self._status_var.set(f"Inference error: {exc}")

        # -- Info bar ----------------------------------------------------------
        cid = self._current_path.stem if self._current_path else "?"
        self._case_label.configure(text=f"Case {cid}")
        self._sbp_label.configure(text=f"SBP {sbp_gt:.0f}")
        self._dbp_label.configure(text=f"DBP {dbp_gt:.0f} mmHg")

        if sbp_pred is not None:
            self._pred_sbp_label.configure(
                text=f"-> {sbp_pred:.0f}"
            )
            self._pred_dbp_label.configure(
                text=f"/ {dbp_pred:.0f} mmHg  pred"
            )
            err_sbp = sbp_pred - sbp_gt
            err_dbp = dbp_pred - dbp_gt

            def _err_color(e: float) -> str:
                return "#228844" if abs(e) <= 5 else ("#cc7700" if abs(e) <= 10 else "#cc2200")

            self._err_sbp_label.configure(
                text=f"dS {err_sbp:+.0f}", fg=_err_color(err_sbp)
            )
            self._err_dbp_label.configure(
                text=f"dD {err_dbp:+.0f}", fg=_err_color(err_dbp)
            )
        else:
            self._pred_sbp_label.configure(text="")
            self._pred_dbp_label.configure(text="")
            self._err_sbp_label.configure(text="no model", fg=FG_DIM)
            self._err_dbp_label.configure(text="")

        # -- PPG waveform axes -------------------------------------------------
        ax_ppg = self._ax_ppg
        ax_ppg.cla()
        ax_ppg.set_facecolor(BG_DARK)

        # Quality-weight shading - one axvspan per overlapping segment
        top_i: int | None = None
        if weights is not None:
            seg_len = self._model.seg_len   # 125 samples
            stride  = self._model.stride    # 62  samples
            w_max   = float(weights.max())
            top_i   = int(np.argmax(weights))
            for i, w in enumerate(weights):
                x0 = (i * stride) / self.target_hz
                x1 = (i * stride + seg_len) / self.target_hz
                # alpha proportional to relative weight; max span = 0.40 opacity
                alpha = float(w / w_max) * 0.40
                ax_ppg.axvspan(
                    x0, x1,
                    color=WEIGHT_COLOR, alpha=alpha,
                    linewidth=0, zorder=1,
                )

        # PPG signal drawn above the shading
        ax_ppg.plot(t, ppg, color=PPG_COLOR, linewidth=0.85, antialiased=True, zorder=5)

        ppg_min   = float(ppg.min())
        ppg_max   = float(ppg.max())
        ppg_range = max(ppg_max - ppg_min, 1.0)
        margin    = ppg_range * 0.12

        ax_ppg.set_xlim(0, seg_sec)
        ax_ppg.set_ylim(ppg_min - margin, ppg_max + margin)
        ax_ppg.set_xlabel("Time (s)", color=FG_DIM, fontsize=9)
        ax_ppg.set_ylabel("PPG (raw)", color=FG_DIM, fontsize=9)
        ax_ppg.tick_params(colors=FG_DIM, labelsize=8)
        for sp in ax_ppg.spines.values():
            sp.set_edgecolor("#ccccdd")
        ax_ppg.grid(True, color=GRID_CLR, linewidth=0.5,
                    linestyle="--", alpha=0.7, zorder=2)

        # Highlight the highest-weight segment on the x-axis (below the plot)
        if top_i is not None:
            seg_len = self._model.seg_len
            stride  = self._model.stride
            x0_top  = top_i * stride / self.target_hz
            x1_top  = (top_i * stride + seg_len) / self.target_hz
            # get_xaxis_transform(): x in data coords, y in axes coords (0=bottom)
            trans = ax_ppg.get_xaxis_transform()
            ax_ppg.plot(
                [x0_top, x1_top], [-0.04, -0.04],
                transform=trans, clip_on=False,
                color="#3366cc", linewidth=4.0,
                solid_capstyle="butt", zorder=10,
            )
            # Small tick marks at the two ends
            for xv in (x0_top, x1_top):
                ax_ppg.plot(
                    [xv, xv], [-0.06, -0.02],
                    transform=trans, clip_on=False,
                    color="#3366cc", linewidth=1.5, zorder=10,
                )

        ax_ppg.set_title(
            "PPG Waveform  |  Blue shading = quality-weight intensity",
            color=FG_NORM, fontsize=9, pad=4,
        )

        # Annotation boxes (ground truth + prediction)
        gt_str   = f"GT   SBP {sbp_gt:.0f}   DBP {dbp_gt:.0f} mmHg"
        pred_str = (
            f"Pred  SBP {sbp_pred:.0f}   DBP {dbp_pred:.0f} mmHg"
            if sbp_pred is not None else "Pred  -"
        )
        ax_ppg.text(
            0.99, 0.97, gt_str,
            transform=ax_ppg.transAxes,
            ha="right", va="top", color=SBP_COLOR,
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_DARK,
                      edgecolor=SBP_COLOR, alpha=0.85),
            zorder=10,
        )
        ax_ppg.text(
            0.99, 0.81, pred_str,
            transform=ax_ppg.transAxes,
            ha="right", va="top", color=PRED_SBP_CLR,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_DARK,
                      edgecolor=PRED_SBP_CLR, alpha=0.85),
            zorder=10,
        )

        # -- Quality-weight triangle chart -------------------------------------
        ax_wt = self._ax_wt
        ax_wt.cla()
        ax_wt.set_facecolor(BG_DARK)

        if weights is not None:
            seg_len = self._model.seg_len
            stride  = self._model.stride
            S       = len(weights)
            centers = np.array(
                [(i * stride + seg_len / 2) / self.target_hz for i in range(S)]
            )
            half_w   = (seg_len / self.target_hz) / 2   # 0.5 s
            w_pct    = weights * 100.0
            top_i    = int(np.argmax(weights))

            def _fmt_q(v: float) -> str:
                a = abs(v)
                if a == 0:
                    return "0"
                if a >= 1e4 or a < 1e-2:
                    return f"{v:.2e}"
                if a >= 100:
                    return f"{v:.1f}"
                return f"{v:.2f}"

            # Draw one isosceles triangle per segment
            for i, (cx, wp) in enumerate(zip(centers, w_pct)):
                is_top  = (i == top_i)
                fill_c  = "#3366cc" if is_top else WEIGHT_COLOR
                edge_c  = "#1144aa" if is_top else "#5588cc"
                label_c = FG_BRIGHT if is_top else FG_NORM
                tri_x   = [cx - half_w, cx,  cx + half_w, cx - half_w]
                tri_y   = [0,           wp,  0,            0          ]
                ax_wt.fill(tri_x, tri_y, color=fill_c, alpha=0.55, zorder=5)
                ax_wt.plot(
                    tri_x, tri_y,
                    color=edge_c, linewidth=0.9, alpha=0.85, zorder=6,
                )
                # Raw quality score at the apex
                if q_raw is not None:
                    ax_wt.annotate(
                        _fmt_q(float(q_raw[i])),
                        xy=(cx, wp), xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom",
                        color=label_c, fontsize=8.0, zorder=8,
                    )

            # Uniform-weight reference line
            uniform_pct = 100.0 / S
            ax_wt.axhline(
                uniform_pct, color="#aaaaaa",
                linewidth=0.9, linestyle="--", zorder=4,
                label=f"uniform = {uniform_pct:.1f}%",
            )
            ax_wt.set_ylim(bottom=0)
            ax_wt.legend(
                fontsize=7, loc="upper right",
                framealpha=0.0, labelcolor=FG_DIM,
            )
        else:
            ax_wt.text(
                0.5, 0.5, "No model loaded",
                transform=ax_wt.transAxes,
                ha="center", va="center",
                color=FG_DIM, fontsize=9,
            )

        ax_wt.set_xlim(0, seg_sec)
        ax_wt.set_xlabel("Time (s)", color=FG_DIM, fontsize=9)
        ax_wt.set_ylabel("Weight (%)", color=FG_DIM, fontsize=9)
        ax_wt.tick_params(colors=FG_DIM, labelsize=8)
        for sp in ax_wt.spines.values():
            sp.set_edgecolor("#ccccdd")
        ax_wt.grid(True, color=GRID_CLR, linewidth=0.5,
                   linestyle="--", alpha=0.7, zorder=2)
        ax_wt.set_title(
            "Segment Quality Weights (softmax)",
            color=FG_NORM, fontsize=9, pad=4,
        )

        # -- Per-segment SBP / DBP prediction panel ----------------------------
        ax_bp = self._ax_bp
        ax_bp.cla()
        ax_bp.set_facecolor(BG_DARK)

        if bp_per_seg is not None:
            seg_len = self._model.seg_len
            stride  = self._model.stride
            S       = len(weights)
            centers = np.array(
                [(i * stride + seg_len / 2) / self.target_hz for i in range(S)]
            )
            sbp_seg = bp_per_seg[:, 0]
            dbp_seg = bp_per_seg[:, 1]

            # Per-segment lines + markers
            ax_bp.plot(
                centers, sbp_seg,
                color=SBP_COLOR, linewidth=1.0, linestyle="-",
                marker="o", markersize=4, zorder=5,
                label=f"SBP/seg  [{sbp_seg.min():.0f}-{sbp_seg.max():.0f}]",
            )
            ax_bp.plot(
                centers, dbp_seg,
                color=DBP_COLOR, linewidth=1.0, linestyle="-",
                marker="o", markersize=4, zorder=5,
                label=f"DBP/seg  [{dbp_seg.min():.0f}-{dbp_seg.max():.0f}]",
            )

            # Numeric labels at each data point
            for cx, sv, dv in zip(centers, sbp_seg, dbp_seg):
                ax_bp.annotate(
                    f"{sv:.0f}",
                    xy=(cx, sv), xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    color=SBP_COLOR, fontsize=8, zorder=8,
                )
                ax_bp.annotate(
                    f"{dv:.0f}",
                    xy=(cx, dv), xytext=(0, -10),
                    textcoords="offset points",
                    ha="center", va="top",
                    color=DBP_COLOR, fontsize=8, zorder=8,
                )

            # Ground-truth horizontal dashed reference lines
            ax_bp.axhline(
                sbp_gt, color=GT_SBP_COLOR, linewidth=1.0,
                linestyle="--", alpha=0.55, zorder=3,
                label=f"GT SBP {sbp_gt:.0f}",
            )
            ax_bp.axhline(
                dbp_gt, color=GT_DBP_COLOR, linewidth=1.0,
                linestyle="--", alpha=0.55, zorder=3,
                label=f"GT DBP {dbp_gt:.0f}",
            )

            # Weighted-average (final) prediction dotted lines
            if sbp_pred is not None:
                ax_bp.axhline(
                    sbp_pred, color=PRED_SBP_CLR, linewidth=1.2,
                    linestyle=":", alpha=0.85, zorder=4,
                    label=f"Pred SBP {sbp_pred:.0f}",
                )
                ax_bp.axhline(
                    dbp_pred, color=PRED_DBP_CLR, linewidth=1.2,
                    linestyle=":", alpha=0.85, zorder=4,
                    label=f"Pred DBP {dbp_pred:.0f}",
                )

            ax_bp.legend(
                fontsize=7, loc="upper right", ncol=3,
                framealpha=0.0, labelcolor=FG_DIM,
            )
        else:
            ax_bp.text(
                0.5, 0.5, "No model loaded",
                transform=ax_bp.transAxes,
                ha="center", va="center",
                color=FG_DIM, fontsize=9,
            )

        ax_bp.set_xlim(0, seg_sec)
        ax_bp.set_xlabel("Time (s)", color=FG_DIM, fontsize=9)
        ax_bp.set_ylabel("mmHg", color=FG_DIM, fontsize=9)
        ax_bp.tick_params(colors=FG_DIM, labelsize=8)
        for sp in ax_bp.spines.values():
            sp.set_edgecolor("#ccccdd")
        ax_bp.grid(True, color=GRID_CLR, linewidth=0.5,
                   linestyle="--", alpha=0.7, zorder=2)
        ax_bp.set_title(
            "Per-Segment SBP / DBP Predictions"
            "  |  dashed = GT  |  dotted = weighted avg",
            color=FG_NORM, fontsize=9, pad=4,
        )

        # -- Finalize ----------------------------------------------------------
        self._fig.patch.set_facecolor(BG_DARK)
        self._canvas_widget.draw_idle()

        self._seg_var.set(f"Segment  {idx + 1} / {n_segs}")
        self._configure_slider(n_segs, enabled=True)
        self._set_slider(idx)

        err_str = ""
        if sbp_pred is not None:
            err_str = (
                f"  |  dS {sbp_pred - sbp_gt:+.0f}"
                f"  dD {dbp_pred - dbp_gt:+.0f} mmHg"
            )
        self._status_var.set(
            f"Case {cid}  |  seg {idx + 1}/{n_segs}"
            f"  |  {n_samp} samples @ {self.target_hz} Hz"
            f"  |  SBP {sbp_gt:.0f}  DBP {dbp_gt:.0f} mmHg"
            + err_str
            + "  |  [UpDown case  <--> seg]"
        )

        self._prev_btn.configure(state="normal" if idx > 0         else "disabled")
        self._next_btn.configure(state="normal" if idx < n_segs - 1 else "disabled")


# -- Entry point ---------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.dataset_dir.exists():
        print(f"Dataset directory not found: {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    PulseBrowser(
        root,
        dataset_dir=args.dataset_dir,
        models_dir=args.models_dir,
        device=args.device,
        target_hz=args.target_hz,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
