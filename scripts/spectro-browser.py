"""
Spectrogram Browser - inspect PPG segments with time-frequency analysis.

Based on psd-browser.py with a spectrogram view so temporal changes in the
frequency content are visible:
  - waveform plot for the selected PPG segment
  - spectrogram plot with a dominant-frequency trace over time
  - power_ratio = Power(0.67-3.0 Hz) / Power(0.5-10.0 Hz)
  - quick navigation by power_ratio range in a right-side panel

Usage:
    uv run python scripts/spectro-browser.py [OPTIONS]

Options:
    --dataset-dir   Root directory containing train/val/test sub-folders
                    (default: data/dataset)
    --target-hz     PPG sample rate used when the dataset was built
                    (default: 125)
    --nperseg       Analysis window length for Welch / spectrogram
                    (default: 128)
    --noverlap      Overlap between adjacent spectrogram windows
                    (default: nperseg // 2)
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
from scipy.signal import spectrogram, welch
from tqdm import tqdm


def _set_cjk_font() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim"):
        if name in available:
            matplotlib.rc("font", family=name)
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_set_cjk_font()

SPLITS = ("train", "val", "test")

BG_DARK = "#ffffff"
BG_MID = "#f0f0f7"
BG_PANEL = "#e8e8f2"
FG_DIM = "#888899"
FG_NORM = "#222233"
FG_BRIGHT = "#1133cc"
ACCENT = "#2255cc"

PPG_COLOR = "#1a8855"
TRACE_COLOR = "#e78a2f"
SBP_COLOR = "#cc2200"
DBP_COLOR = "#cc7700"
RATIO_COLOR = "#4b3f9f"
GRID_CLR = "#e8e8ee"

SPLIT_BTN_ACTIVE = {"bg": "#2255cc", "fg": "white", "relief": "flat"}
SPLIT_BTN_INACTIVE = {"bg": "#f0f0f7", "fg": "#666677", "relief": "flat"}

PASSBAND = (0.5, 10.0)
HEART_BAND = (0.67, 3.0)
RATIO_BINS = [
    ("0.0 ~ 0.4", 0.0, 0.4),
    ("0.4 ~ 0.5", 0.4, 0.5),
    ("0.5 ~ 0.6", 0.5, 0.6),
    ("0.6 ~ 0.7", 0.6, 0.7),
    ("0.7 ~ 0.8", 0.7, 0.8),
    ("0.8 ~ 0.9", 0.8, 0.9),
    ("0.9 ~ 1.0", 0.9, 1.0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Browse NPZ dataset segments with spectrogram analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/dataset"),
        help="Root dataset directory (default: data/dataset)",
    )
    p.add_argument(
        "--target-hz",
        type=int,
        default=125,
        help="PPG sample rate used when building the dataset (default: 125)",
    )
    p.add_argument(
        "--nperseg",
        type=int,
        default=128,
        help="Analysis window length for Welch / spectrogram (default: 128)",
    )
    p.add_argument(
        "--noverlap",
        type=int,
        default=-1,
        help="Overlap between spectrogram windows (default: nperseg // 2)",
    )
    return p.parse_args()


def compute_psd(signal: np.ndarray, fs: int, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    freqs, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=min(len(signal), nperseg),
        noverlap=None,
        detrend="constant",
        scaling="density",
    )
    return freqs, psd


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return float("nan")
    return float(np.trapezoid(psd[mask], freqs[mask]))


def compute_ratio(signal: np.ndarray, fs: int, nperseg: int) -> tuple[float, float, float]:
    freqs, psd = compute_psd(signal, fs, nperseg)
    heart_power = band_power(freqs, psd, HEART_BAND)
    passband_power = band_power(freqs, psd, PASSBAND)
    ratio = heart_power / passband_power if passband_power > 0 else float("nan")
    return heart_power, passband_power, ratio


def compute_spectrogram(
    signal: np.ndarray,
    fs: int,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    win = min(len(signal), nperseg)
    ov = min(max(0, noverlap), max(win - 1, 0))
    freqs, times, sxx = spectrogram(
        signal,
        fs=fs,
        window="hann",
        nperseg=win,
        noverlap=ov,
        detrend="constant",
        scaling="density",
        mode="psd",
    )
    band_mask = (freqs >= PASSBAND[0]) & (freqs <= PASSBAND[1])
    if np.any(band_mask) and sxx.shape[1] > 0:
        freqs_band = freqs[band_mask]
        sxx_band = sxx[band_mask, :]
        dominant_freq = freqs_band[np.argmax(sxx_band, axis=0)]
    else:
        dominant_freq = np.full(times.shape, np.nan, dtype=np.float64)
    return freqs, times, sxx, dominant_freq


class SpectrogramBrowser:
    LIST_WIDTH = 320
    SEARCH_W = 270
    CANVAS_W = 1040
    WIN_H = 860

    LIST_COLUMNS = [
        ("case", "Case ID", 80, "center"),
        ("segs", "Segments", 80, "center"),
        ("size", "Size", 70, "center"),
    ]

    def __init__(
        self,
        root: tk.Tk,
        dataset_dir: Path,
        target_hz: int,
        nperseg: int,
        noverlap: int,
    ):
        self.root = root
        self.dataset_dir = dataset_dir
        self.target_hz = target_hz
        self.nperseg = nperseg
        self.noverlap = noverlap if noverlap >= 0 else nperseg // 2

        self._split = "train"
        self._npz_files: dict[str, list[Path]] = {}
        self._rows: dict[str, list[dict]] = {}
        self._row_by_path: dict[Path, dict] = {}
        self._metadata_queue: queue.Queue = queue.Queue()
        self._metadata_total = 0
        self._metadata_done = 0
        self._metadata_thread: threading.Thread | None = None
        self._current_path: Path | None = None
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._seg_idx = 0
        self._seg_slider_updating = False
        self._case_ratio_cache: dict[Path, np.ndarray] = {}
        self._ratio_match_indices: list[int] = []
        self._ratio_bin_var = tk.StringVar(value=RATIO_BINS[-1][0])
        self._spec_cbar = None

        self._discover_files()
        self._build_ui()
        self._select_split("train")
        self._start_metadata_worker()

    def _discover_files(self) -> None:
        for split in SPLITS:
            d = self.dataset_dir / split
            if d.exists():
                files = sorted(
                    d.glob("*.npz"),
                    key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
                )
            else:
                files = []
            self._npz_files[split] = files
            self._rows[split] = [self._placeholder_row(f) for f in files]
            for row in self._rows[split]:
                self._row_by_path[row["path"]] = row

    @staticmethod
    def _placeholder_row(path: Path) -> dict:
        cid = int(path.stem) if path.stem.isdigit() else 0
        return {
            "path": path,
            "case": cid,
            "segs": 0,
            "segs_text": "...",
            "size": "...",
            "size_val": 0.0,
            "metadata_loaded": False,
        }

    def _start_metadata_worker(self) -> None:
        self._metadata_total = sum(len(files) for files in self._npz_files.values())
        if self._metadata_total == 0:
            return

        self._metadata_thread = threading.Thread(
            target=self._metadata_worker,
            name="spectro-browser-metadata-loader",
            daemon=True,
        )
        self._metadata_thread.start()
        self.root.after(50, self._drain_metadata_queue)

    def _metadata_worker(self) -> None:
        for split in SPLITS:
            files = self._npz_files[split]
            for path in tqdm(files,
                             desc=f"Indexing {split}",
                             unit="file",
                             dynamic_ncols=True):
                self._metadata_queue.put(("row", split, path, self._file_row(path)))
        self._metadata_queue.put(("done", None, None, None))

    def _drain_metadata_queue(self) -> None:
        updated_current = False
        done = False
        while True:
            try:
                kind, split, path, row = self._metadata_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "done":
                done = True
                continue
            self._metadata_done += 1
            stored = self._row_by_path.get(path)
            if stored is None:
                continue
            stored.update(row)
            updated_current = updated_current or split == self._split
            if split == self._split:
                iid = str(path)
                if self._tree.exists(iid):
                    self._tree.item(iid, values=self._row_values(stored))
        if updated_current:
            self._update_count()
        if done:
            if self._current_path is None:
                self._status_var.set("Dataset metadata indexing complete.")
            return
        if self._current_path is None:
            self._status_var.set(
                f"Indexing dataset metadata {self._metadata_done}/{self._metadata_total}..."
            )
        self.root.after(50, self._drain_metadata_queue)

    @staticmethod
    def _file_row(path: Path) -> dict:
        try:
            with np.load(path) as data:
                n_segs = len(data["x"])
        except Exception:
            n_segs = 0
        try:
            size_kb = path.stat().st_size / 1024
        except OSError:
            size_kb = 0.0
        cid = int(path.stem) if path.stem.isdigit() else 0
        return {
            "path": path,
            "case": cid,
            "segs": n_segs,
            "segs_text": str(n_segs),
            "size": f"{size_kb:.0f} KB",
            "size_val": size_kb,
            "metadata_loaded": True,
        }

    def _build_ui(self) -> None:
        self.root.title("Spectrogram Browser")
        self.root.configure(bg=BG_DARK)
        self.root.geometry(f"{self.LIST_WIDTH + self.CANVAS_W + self.SEARCH_W}x{self.WIN_H}")
        self.root.minsize(1180, 620)

        paned = tk.PanedWindow(
            self.root,
            orient="horizontal",
            bg=BG_DARK,
            sashwidth=5,
            sashrelief="flat",
            handlesize=0,
        )
        paned.pack(fill="both", expand=True)

        left = tk.Frame(paned, bg=BG_MID, width=self.LIST_WIDTH)
        left.pack_propagate(False)
        paned.add(left, minsize=240)

        center = tk.Frame(paned, bg=BG_DARK, width=self.CANVAS_W)
        center.pack_propagate(False)
        paned.add(center, minsize=680)

        right = tk.Frame(paned, bg="#f6f6fb", width=self.SEARCH_W)
        right.pack_propagate(False)
        paned.add(right, minsize=220)

        self._build_list_panel(left)
        self._build_canvas_panel(center)
        self._build_ratio_panel(right)

        bar = tk.Frame(self.root, bg=BG_PANEL, height=22)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        self._status_var = tk.StringVar(value="Select a case from the list.")
        tk.Label(
            bar,
            textvariable=self._status_var,
            bg=BG_PANEL,
            fg=FG_DIM,
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(side="left", padx=8)

    def _build_list_panel(self, parent: tk.Frame) -> None:
        btn_row = tk.Frame(parent, bg=BG_MID)
        btn_row.pack(fill="x", padx=8, pady=(8, 4))
        self._split_btns: dict[str, tk.Button] = {}
        for split in SPLITS:
            b = tk.Button(
                btn_row,
                text=split.capitalize(),
                font=("Segoe UI", 9, "bold"),
                cursor="hand2",
                bd=0,
                padx=10,
                pady=4,
                command=lambda s=split: self._select_split(s),
            )
            b.pack(side="left", padx=2)
            self._split_btns[split] = b

        self._count_var = tk.StringVar()
        tk.Label(
            parent,
            textvariable=self._count_var,
            bg=BG_MID,
            fg=FG_DIM,
            font=("Segoe UI", 8),
            anchor="w",
        ).pack(fill="x", padx=10, pady=(0, 2))

        frame = tk.Frame(parent, bg=BG_MID)
        frame.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "D.Treeview",
            background=BG_DARK,
            foreground=FG_NORM,
            fieldbackground=BG_DARK,
            rowheight=22,
            font=("Segoe UI", 9),
        )
        style.configure(
            "D.Treeview.Heading",
            background="#d0d8f0",
            foreground=FG_BRIGHT,
            font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        style.map(
            "D.Treeview",
            background=[("selected", ACCENT)],
            foreground=[("selected", "white")],
        )

        col_ids = [c[0] for c in self.LIST_COLUMNS]
        self._tree = ttk.Treeview(
            frame,
            columns=col_ids,
            show="headings",
            style="D.Treeview",
            selectmode="browse",
        )
        for cid, heading, width, anchor in self.LIST_COLUMNS:
            self._tree.heading(cid, text=heading)
            self._tree.column(cid, width=width, anchor=anchor, stretch=(cid == "case"))

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._tree.bind("<<TreeviewSelect>>", self._on_case_select)

    def _build_canvas_panel(self, parent: tk.Frame) -> None:
        info_row = tk.Frame(parent, bg=BG_PANEL, height=36)
        info_row.pack(fill="x")
        info_row.pack_propagate(False)

        self._case_label = tk.Label(
            info_row,
            text="",
            bg=BG_PANEL,
            fg=FG_BRIGHT,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        )
        self._case_label.pack(side="left", padx=12)

        self._bp_label = tk.Label(
            info_row,
            text="",
            bg=BG_PANEL,
            fg=SBP_COLOR,
            font=("Segoe UI", 10, "bold"),
        )
        self._bp_label.pack(side="left", padx=(12, 4))

        self._ratio_label = tk.Label(
            info_row,
            text="",
            bg=BG_PANEL,
            fg=RATIO_COLOR,
            font=("Segoe UI", 10, "bold"),
        )
        self._ratio_label.pack(side="left", padx=(8, 4))

        self._summary_label = tk.Label(
            info_row,
            text="",
            bg=BG_PANEL,
            fg=FG_DIM,
            font=("Segoe UI", 9),
        )
        self._summary_label.pack(side="right", padx=12)

        self._placeholder = tk.Label(
            parent,
            text="<- Select a case from the list",
            bg=BG_DARK,
            fg="#aaaacc",
            font=("Segoe UI", 14),
        )
        self._placeholder.pack(expand=True)

        self._fig = plt.Figure(figsize=(9.4, 6.6), facecolor=BG_DARK)
        self._gs = gridspec.GridSpec(2, 1, figure=self._fig, height_ratios=[1.0, 1.2])
        self._ax_wave = self._fig.add_subplot(self._gs[0], facecolor=BG_DARK)
        self._ax_spec = self._fig.add_subplot(self._gs[1], facecolor=BG_DARK)
        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas_widget.get_tk_widget().pack_forget()
        self._canvas_frame_packed = False

        nav = tk.Frame(parent, bg=BG_PANEL, height=36)
        nav.pack(fill="x", side="bottom")
        nav.pack_propagate(False)

        btn_cfg = dict(
            font=("Segoe UI", 9, "bold"),
            bg="#d0d8f0",
            fg=FG_BRIGHT,
            activebackground=ACCENT,
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=16,
            pady=4,
            cursor="hand2",
        )
        self._prev_btn = tk.Button(nav, text="< Prev", command=self._prev_seg, **btn_cfg)
        self._prev_btn.pack(side="left", padx=8, pady=4)

        self._seg_var = tk.StringVar(value="")
        tk.Label(
            nav,
            textvariable=self._seg_var,
            bg=BG_PANEL,
            fg=FG_DIM,
            font=("Segoe UI", 9),
            width=20,
        ).pack(side="left", padx=4)

        self._next_btn = tk.Button(nav, text="Next >", command=self._next_seg, **btn_cfg)
        self._next_btn.pack(side="left", padx=4)

        self._seg_slider = tk.Scale(
            nav,
            from_=1,
            to=1,
            orient="horizontal",
            showvalue=False,
            resolution=1,
            bg=BG_PANEL,
            fg=FG_DIM,
            troughcolor="#ccccdd",
            activebackground=ACCENT,
            highlightthickness=0,
            bd=0,
            sliderlength=16,
            width=10,
            state="disabled",
            command=self._on_segment_slider,
        )
        self._seg_slider.pack(side="left", fill="x", expand=True, padx=(10, 8))

        tk.Label(
            nav,
            text="Jump:",
            bg=BG_PANEL,
            fg=FG_DIM,
            font=("Segoe UI", 9),
        ).pack(side="left", padx=(20, 2))
        self._jump_var = tk.StringVar()
        jump_entry = tk.Entry(
            nav,
            textvariable=self._jump_var,
            width=6,
            bg="#ccccdd",
            fg=FG_NORM,
            insertbackground=FG_NORM,
            relief="flat",
            font=("Segoe UI", 9),
        )
        jump_entry.pack(side="left")
        jump_entry.bind("<Return>", self._on_jump)

        self.root.bind("<Left>", lambda _: self._prev_seg())
        self.root.bind("<Right>", lambda _: self._next_seg())
        self.root.bind("<Up>", lambda _: self._prev_case())
        self.root.bind("<Down>", lambda _: self._next_case())

    def _build_ratio_panel(self, parent: tk.Frame) -> None:
        top = tk.Frame(parent, bg="#f6f6fb")
        top.pack(fill="x", padx=10, pady=(10, 6))

        tk.Label(
            top,
            text="Power Ratio Search",
            bg="#f6f6fb",
            fg="#222233",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            top,
            text="Current case only",
            bg="#f6f6fb",
            fg="#888899",
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(2, 6))

        btn_wrap = tk.Frame(top, bg="#f6f6fb")
        btn_wrap.pack(fill="x")
        for i, (label, _, _) in enumerate(RATIO_BINS):
            rb = tk.Radiobutton(
                btn_wrap,
                text=label,
                value=label,
                variable=self._ratio_bin_var,
                command=self._update_ratio_results,
                bg="#f6f6fb",
                fg=FG_NORM,
                selectcolor="#dbe6ff",
                activebackground="#f6f6fb",
                activeforeground=FG_BRIGHT,
                font=("Segoe UI", 9),
                anchor="w",
                relief="flat",
                highlightthickness=0,
            )
            rb.grid(row=i, column=0, sticky="w", pady=1)

        results = tk.Frame(parent, bg="#f6f6fb")
        results.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        self._ratio_results = tk.Listbox(
            results,
            bg="#ffffff",
            fg="#222233",
            selectbackground="#2255cc",
            selectforeground="white",
            relief="flat",
            activestyle="none",
            font=("Consolas", 9),
        )
        self._ratio_results.pack(side="left", fill="both", expand=True)
        self._ratio_results.bind("<<ListboxSelect>>", self._on_ratio_result_select)
        self._ratio_results.bind("<Double-1>", self._on_ratio_result_select)
        self._ratio_results.bind("<Return>", self._on_ratio_result_select)

        vsb = ttk.Scrollbar(results, orient="vertical", command=self._ratio_results.yview)
        self._ratio_results.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")

        self._ratio_status = tk.StringVar(value="Load a case to browse by power_ratio.")
        tk.Label(
            parent,
            textvariable=self._ratio_status,
            bg="#f6f6fb",
            fg="#888899",
            font=("Segoe UI", 8),
            anchor="center",
            justify="center",
            wraplength=self.SEARCH_W - 24,
        ).pack(fill="x", padx=10, pady=(0, 10))

    def _select_split(self, split: str) -> None:
        self._split = split
        self._current_path = None
        self._x = None
        self._y = None
        for s, btn in self._split_btns.items():
            cfg = SPLIT_BTN_ACTIVE if s == split else SPLIT_BTN_INACTIVE
            btn.configure(**cfg)
        self._refresh_list()
        self._clear_canvas()

    def _refresh_list(self) -> None:
        rows = self._sorted_rows()
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
        rows = self._rows[self._split] if rows is None else rows
        n = len(rows)
        known = sum(1 for r in rows if r["metadata_loaded"])
        total_segs = sum(r["segs"] for r in rows if r["metadata_loaded"])
        if known < n:
            self._count_var.set(
                f"{n} cases - indexed {known}/{n} - {total_segs:,} segments known [{self._split}]"
            )
        else:
            self._count_var.set(f"{n} cases - {total_segs:,} segments [{self._split}]")

    def _sorted_rows(self) -> list[dict]:
        rows = self._rows[self._split][:]
        rows.sort(key=lambda r: r["case"])
        return rows

    def _on_case_select(self, _event=None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        path = Path(sel[0])
        if path == self._current_path:
            return
        self._load_case(path)

    def _load_case(self, path: Path) -> None:
        self._status_var.set(f"Loading {path.name} ...")
        self.root.update_idletasks()
        try:
            with np.load(path) as data:
                self._x = data["x"]
                self._y = data["y"]
        except Exception as exc:
            self._status_var.set(f"Error loading {path.name}: {exc}")
            return
        self._current_path = path
        self._seg_idx = 0
        self._compute_case_ratio_cache(path)
        self._show_canvas()
        self._update_ratio_results()
        self._show_segment(0)

    def _compute_case_ratio_cache(self, path: Path) -> None:
        if path in self._case_ratio_cache or self._x is None:
            return
        ratios = []
        for signal in self._x:
            _, _, ratio = compute_ratio(signal, self.target_hz, self.nperseg)
            ratios.append(ratio)
        self._case_ratio_cache[path] = np.asarray(ratios, dtype=np.float64)

    def _current_ratio_bin(self) -> tuple[str, float, float | None]:
        label = self._ratio_bin_var.get()
        for item in RATIO_BINS:
            if item[0] == label:
                return item
        return RATIO_BINS[-1]

    def _update_ratio_results(self) -> None:
        self._ratio_results.delete(0, tk.END)
        self._ratio_match_indices = []
        if self._current_path is None or self._x is None:
            self._ratio_status.set("Load a case to browse by power_ratio.")
            return
        ratios = self._case_ratio_cache.get(self._current_path)
        if ratios is None:
            self._ratio_status.set("power_ratio cache is not available for this case.")
            return
        label, lo, hi = self._current_ratio_bin()
        mask = np.isfinite(ratios) & (ratios >= lo)
        if hi is not None:
            mask &= ratios < hi
        matches = np.flatnonzero(mask)
        self._ratio_match_indices = matches.astype(int).tolist()
        for seg_idx in self._ratio_match_indices:
            ratio = float(ratios[seg_idx])
            self._ratio_results.insert(tk.END, f"Seg {seg_idx + 1:4d}   ratio {ratio:0.4f}")
        if self._ratio_match_indices:
            self._ratio_status.set(f"{len(self._ratio_match_indices)} segments in range {label}")
        else:
            self._ratio_status.set(f"No segments in range {label}")
        self._sync_ratio_result_selection()

    def _on_ratio_result_select(self, _event=None) -> None:
        sel = self._ratio_results.curselection()
        if not sel:
            return
        row_idx = sel[0]
        if 0 <= row_idx < len(self._ratio_match_indices):
            seg_idx = self._ratio_match_indices[row_idx]
            if self._x is not None:
                self._seg_idx = seg_idx
                self._show_segment(seg_idx)

    def _sync_ratio_result_selection(self) -> None:
        if self._ratio_match_indices and self._seg_idx in self._ratio_match_indices:
            row_idx = self._ratio_match_indices.index(self._seg_idx)
            self._ratio_results.selection_clear(0, tk.END)
            self._ratio_results.selection_set(row_idx)
            self._ratio_results.activate(row_idx)
            self._ratio_results.see(row_idx)
        else:
            self._ratio_results.selection_clear(0, tk.END)

    def _prev_case(self) -> None:
        rows = self._sorted_rows()
        paths = [r["path"] for r in rows]
        if self._current_path not in paths:
            return
        idx = paths.index(self._current_path)
        if idx > 0:
            self._load_case(paths[idx - 1])
            self._select_tree_item(paths[idx - 1])

    def _next_case(self) -> None:
        rows = self._sorted_rows()
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

    def _prev_seg(self) -> None:
        if self._x is not None and self._seg_idx > 0:
            self._seg_idx -= 1
            self._show_segment(self._seg_idx)

    def _next_seg(self) -> None:
        if self._x is not None and self._seg_idx < len(self._x) - 1:
            self._seg_idx += 1
            self._show_segment(self._seg_idx)

    def _on_segment_slider(self, value: str) -> None:
        if self._seg_slider_updating or self._x is None:
            return
        idx = int(round(float(value))) - 1
        idx = max(0, min(idx, len(self._x) - 1))
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

    def _configure_segment_slider(self, n_segs: int, enabled: bool) -> None:
        self._seg_slider.configure(
            from_=1,
            to=max(n_segs, 1),
            state="normal" if enabled and n_segs > 1 else "disabled",
        )

    def _set_segment_slider(self, idx: int) -> None:
        self._seg_slider_updating = True
        try:
            self._seg_slider.set(idx + 1)
        finally:
            self._seg_slider_updating = False

    def _show_canvas(self) -> None:
        if not self._canvas_frame_packed:
            self._placeholder.pack_forget()
            self._canvas_widget.get_tk_widget().pack(fill="both", expand=True)
            self._canvas_frame_packed = True

    def _clear_canvas(self) -> None:
        if self._canvas_frame_packed:
            self._canvas_widget.get_tk_widget().pack_forget()
            self._placeholder.pack(expand=True)
            self._canvas_frame_packed = False
        self._case_label.configure(text="")
        self._bp_label.configure(text="")
        self._ratio_label.configure(text="")
        self._summary_label.configure(text="")
        self._seg_var.set("")
        self._configure_segment_slider(1, enabled=False)
        self._ratio_results.delete(0, tk.END)
        self._ratio_status.set("Load a case to browse by power_ratio.")
        self._ratio_match_indices = []

    def _show_segment(self, idx: int) -> None:
        if self._x is None:
            return
        ppg = self._x[idx]
        label = None if self._y is None or idx >= len(self._y) else self._y[idx]
        sbp = float(label[0]) if label is not None else float("nan")
        dbp = float(label[1]) if label is not None else float("nan")
        n_segs = len(self._x)
        n_samp = len(ppg)
        seg_sec = n_samp / self.target_hz
        t = np.arange(n_samp) / self.target_hz

        heart_power, passband_power, ratio = compute_ratio(ppg, self.target_hz, self.nperseg)
        freqs, times, sxx, dominant_freq = compute_spectrogram(
            ppg,
            self.target_hz,
            self.nperseg,
            self.noverlap,
        )
        ratio_stats = self._case_ratio_cache.get(self._current_path)
        cid = self._current_path.stem if self._current_path else "?"

        self._case_label.configure(text=f"Case {cid}")
        if np.isfinite(sbp) and np.isfinite(dbp):
            self._bp_label.configure(text=f"SBP {sbp:.0f} mmHg   DBP {dbp:.0f} mmHg")
        else:
            self._bp_label.configure(text="SBP/DBP unavailable")
        self._ratio_label.configure(text=f"power_ratio {ratio:.4f}")
        if ratio_stats is not None and len(ratio_stats) > 0:
            self._summary_label.configure(
                text=(
                    f"case mean {np.nanmean(ratio_stats):.4f}   "
                    f"median {np.nanmedian(ratio_stats):.4f}"
                )
            )
        else:
            self._summary_label.configure(text="")

        self._seg_var.set(f"Segment  {idx + 1} / {n_segs}")
        self._configure_segment_slider(n_segs, enabled=True)
        self._set_segment_slider(idx)
        self._sync_ratio_result_selection()

        ax_wave = self._ax_wave
        ax_spec = self._ax_spec
        ax_wave.cla()
        ax_spec.cla()
        ax_wave.set_facecolor(BG_DARK)
        ax_spec.set_facecolor(BG_DARK)

        ax_wave.plot(t, ppg, color=PPG_COLOR, linewidth=0.9, antialiased=True)
        ax_wave.set_xlim(0, seg_sec)
        ax_wave.set_title(f"PPG Segment - case {cid}, segment {idx + 1}", color=FG_NORM, fontsize=11)
        ax_wave.set_xlabel("Time (s)", color=FG_DIM, fontsize=9)
        ax_wave.set_ylabel("PPG amplitude", color=FG_DIM, fontsize=9)

        band_mask = (freqs >= PASSBAND[0]) & (freqs <= PASSBAND[1])
        freqs_plot = freqs[band_mask]
        sxx_plot = sxx[band_mask, :] if np.any(band_mask) else sxx
        if sxx_plot.size == 0:
            sxx_plot = np.zeros((1, 1), dtype=np.float64)
            freqs_plot = np.array([0.0], dtype=np.float64)
            times_plot = np.array([0.0], dtype=np.float64)
        else:
            times_plot = times

        power_db = 10.0 * np.log10(np.maximum(sxx_plot, 1e-12))
        mesh = ax_spec.pcolormesh(times_plot, freqs_plot, power_db, shading="auto", cmap="viridis")
        if len(times) > 0 and len(dominant_freq) == len(times):
            ax_spec.plot(times, dominant_freq, color=TRACE_COLOR, linewidth=1.8, label="Dominant freq")
        ax_spec.set_ylim(PASSBAND[0], PASSBAND[1])
        ax_spec.set_xlim(0, seg_sec)
        ax_spec.set_title("Spectrogram (0.5-10.0 Hz)", color=FG_NORM, fontsize=11)
        ax_spec.set_xlabel("Time (s)", color=FG_DIM, fontsize=9)
        ax_spec.set_ylabel("Frequency (Hz)", color=FG_DIM, fontsize=9)
        if len(times) > 0 and len(dominant_freq) == len(times):
            ax_spec.legend(loc="upper right", fontsize=8, frameon=True)
        if self._spec_cbar is None:
            self._spec_cbar = self._fig.colorbar(mesh, ax=ax_spec, pad=0.015)
        else:
            self._spec_cbar.update_normal(mesh)
        self._spec_cbar.set_label("PSD (dB/Hz)", color=FG_DIM, fontsize=8)
        self._spec_cbar.ax.yaxis.set_tick_params(color=FG_DIM, labelsize=8)
        plt.setp(self._spec_cbar.ax.get_yticklabels(), color=FG_DIM)

        dom_mean = float(np.nanmean(dominant_freq)) if dominant_freq.size else float("nan")
        dom_std = float(np.nanstd(dominant_freq)) if dominant_freq.size else float("nan")
        summary_text = (
            f"Power(0.67-3.0 Hz) = {heart_power:.5f}\n"
            f"Power(0.5-10.0 Hz) = {passband_power:.5f}\n"
            f"power_ratio = {ratio:.5f}\n"
            f"dom.freq mean = {dom_mean:.3f} Hz\n"
            f"dom.freq std = {dom_std:.3f} Hz"
        )
        ax_spec.text(
            0.98,
            0.97,
            summary_text,
            transform=ax_spec.transAxes,
            ha="right",
            va="top",
            color=RATIO_COLOR,
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": BG_DARK,
                "edgecolor": RATIO_COLOR,
                "alpha": 0.9,
            },
        )

        for ax in (ax_wave, ax_spec):
            ax.tick_params(colors=FG_DIM, labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#ccccdd")
            ax.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--", alpha=0.7)

        self._fig.patch.set_facecolor(BG_DARK)
        self._fig.tight_layout(pad=1.2)
        self._canvas_widget.draw_idle()

        self._status_var.set(
            f"Case {cid}  |  segment {idx + 1}/{n_segs}"
            f"  |  {n_samp} samples @ {self.target_hz} Hz"
            f"  |  ratio {ratio:.4f}"
            f"  |  dom.freq mean {dom_mean:.3f} Hz"
            f"  |  [UpDown case  <--> segment]"
        )

        self._prev_btn.configure(state="normal" if idx > 0 else "disabled")
        self._next_btn.configure(state="normal" if idx < n_segs - 1 else "disabled")


def main() -> None:
    args = parse_args()
    if not args.dataset_dir.exists():
        print(f"Dataset directory not found: {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    SpectrogramBrowser(
        root,
        args.dataset_dir,
        args.target_hz,
        args.nperseg,
        args.noverlap,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
