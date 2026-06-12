"""
Dataset Browser - inspect NPZ segment files produced by construct-dataset.py

Left panel:   split selector (Train / Val / Test) + sortable case list
Right panel:  PPG waveform for the selected segment, SBP / DBP labels,
              and Prev / Next buttons to navigate through a case's segments

Usage:
    uv run python scripts/dataset-browser.py [OPTIONS]

Options:
    --dataset-dir   Root directory containing train/val/test sub-folders
                    (default: data/dataset)
    --target-hz     PPG sample rate used when the dataset was built
                    (default: 125)
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

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tqdm import tqdm


# -- Korean font (no-op if unavailable) ---------------------------------------
def _set_cjk_font():
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim"):
        if name in available:
            matplotlib.rc("font", family=name)
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_set_cjk_font()

SPLITS = ("train", "val", "test")

# Light colour palette
BG_DARK   = "#ffffff"
BG_MID    = "#f0f0f7"
BG_PANEL  = "#e8e8f2"
FG_DIM    = "#888899"
FG_NORM   = "#222233"
FG_BRIGHT = "#1133cc"
ACCENT    = "#2255cc"

PPG_COLOR = "#1a8855"
SBP_COLOR = "#cc2200"
DBP_COLOR = "#cc7700"
GRID_CLR  = "#e8e8ee"

SPLIT_BTN_ACTIVE   = {"bg": "#2255cc", "fg": "white",    "relief": "flat"}
SPLIT_BTN_INACTIVE = {"bg": "#f0f0f7", "fg": "#666677",  "relief": "flat"}


# -- argparse ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Browse NPZ dataset segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset-dir", type=Path, default=Path("data/dataset"),
        help="Root dataset directory (default: data/dataset)",
    )
    p.add_argument(
        "--target-hz", type=int, default=125,
        help="PPG sample rate used when building the dataset (default: 125)",
    )
    return p.parse_args()


# -- Browser application -------------------------------------------------------
class DatasetBrowser:
    LIST_WIDTH  = 320
    CANVAS_W    = 860
    WIN_H       = 700

    LIST_COLUMNS = [
        ("case",  "Case ID",   80,  "center"),
        ("segs",  "Segments",  80,  "center"),
        ("size",  "Size",      70,  "center"),
    ]

    def __init__(self, root: tk.Tk, dataset_dir: Path, target_hz: int):
        self.root        = root
        self.dataset_dir = dataset_dir
        self.target_hz   = target_hz

        # App state
        self._split      = "train"
        self._npz_files: dict[str, list[Path]] = {}
        self._rows:       dict[str, list[dict]] = {}
        self._row_by_path: dict[Path, dict] = {}
        self._metadata_queue: queue.Queue = queue.Queue()
        self._metadata_total = 0
        self._metadata_done = 0
        self._metadata_thread: threading.Thread | None = None
        self._current_path: Path | None = None
        self._x: np.ndarray | None = None   # (N, samples)
        self._y: np.ndarray | None = None   # (N, 2) [SBP, DBP]
        self._seg_idx = 0
        self._seg_slider_updating = False

        self._discover_files()
        self._build_ui()
        self._select_split("train")
        self._start_metadata_worker()

    # -- File discovery --------------------------------------------------------

    def _discover_files(self):
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
        return dict(
            path=path, case=cid, segs=0, segs_text="...",
            size="...", size_val=0.0, metadata_loaded=False,
        )

    def _start_metadata_worker(self):
        self._metadata_total = sum(len(files) for files in self._npz_files.values())
        if self._metadata_total == 0:
            return

        self._metadata_thread = threading.Thread(
            target=self._metadata_worker,
            name="dataset-metadata-loader",
            daemon=True,
        )
        self._metadata_thread.start()
        self.root.after(50, self._drain_metadata_queue)

    def _metadata_worker(self):
        for split in SPLITS:
            files = self._npz_files[split]
            for path in tqdm(files, desc=f"Indexing {split}", unit="file"):
                self._metadata_queue.put(("row", split, path, self._file_row(path)))
        self._metadata_queue.put(("done", None, None, None))

    def _drain_metadata_queue(self):
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
                f"Indexing dataset metadata "
                f"{self._metadata_done}/{self._metadata_total}..."
            )
        self.root.after(50, self._drain_metadata_queue)

    @staticmethod
    def _file_row(path: Path) -> dict:
        try:
            with np.load(path) as data:
                n_segs = len(data["x"])
        except Exception:
            n_segs  = 0
        try:
            size_kb = path.stat().st_size / 1024
        except OSError:
            size_kb = 0.0
        cid     = int(path.stem) if path.stem.isdigit() else 0
        return dict(
            path=path, case=cid, segs=n_segs, segs_text=str(n_segs),
            size=f"{size_kb:.0f} KB", size_val=size_kb,
            metadata_loaded=True,
        )

    # -- UI construction -------------------------------------------------------

    def _build_ui(self):
        self.root.title("Dataset Browser")
        self.root.configure(bg=BG_DARK)
        self.root.geometry(f"{self.LIST_WIDTH + self.CANVAS_W}x{self.WIN_H}")
        self.root.minsize(800, 500)

        paned = tk.PanedWindow(
            self.root, orient="horizontal",
            bg=BG_DARK, sashwidth=5, sashrelief="flat", handlesize=0,
        )
        paned.pack(fill="both", expand=True)

        left = tk.Frame(paned, bg=BG_MID, width=self.LIST_WIDTH)
        left.pack_propagate(False)
        paned.add(left, minsize=240)

        right = tk.Frame(paned, bg=BG_DARK)
        paned.add(right, minsize=500)

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

    def _build_list_panel(self, parent: tk.Frame):
        # Split selector buttons
        btn_row = tk.Frame(parent, bg=BG_MID)
        btn_row.pack(fill="x", padx=8, pady=(8, 4))
        self._split_btns: dict[str, tk.Button] = {}
        for split in SPLITS:
            b = tk.Button(
                btn_row, text=split.capitalize(),
                font=("Segoe UI", 9, "bold"),
                cursor="hand2", bd=0, padx=10, pady=4,
                command=lambda s=split: self._select_split(s),
            )
            b.pack(side="left", padx=2)
            self._split_btns[split] = b

        # Case count label
        self._count_var = tk.StringVar()
        tk.Label(
            parent, textvariable=self._count_var,
            bg=BG_MID, fg=FG_DIM, font=("Segoe UI", 8), anchor="w",
        ).pack(fill="x", padx=10, pady=(0, 2))

        # Treeview
        frame = tk.Frame(parent, bg=BG_MID)
        frame.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "D.Treeview",
            background=BG_DARK, foreground=FG_NORM,
            fieldbackground=BG_DARK, rowheight=22,
            font=("Segoe UI", 9),
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

    # -- Right panel -----------------------------------------------------------

    def _build_canvas_panel(self, parent: tk.Frame):
        # Info bar at the top
        info_row = tk.Frame(parent, bg=BG_PANEL, height=32)
        info_row.pack(fill="x")
        info_row.pack_propagate(False)

        self._case_label = tk.Label(
            info_row, text="",
            bg=BG_PANEL, fg=FG_BRIGHT,
            font=("Segoe UI", 10, "bold"), anchor="w",
        )
        self._case_label.pack(side="left", padx=12)

        self._sbp_label = tk.Label(
            info_row, text="",
            bg=BG_PANEL, fg=SBP_COLOR,
            font=("Segoe UI", 11, "bold"),
        )
        self._sbp_label.pack(side="left", padx=(16, 4))

        self._dbp_label = tk.Label(
            info_row, text="",
            bg=BG_PANEL, fg=DBP_COLOR,
            font=("Segoe UI", 11, "bold"),
        )
        self._dbp_label.pack(side="left", padx=(0, 8))

        # Placeholder shown before any case is selected
        self._placeholder = tk.Label(
            parent,
            text="<- Select a case from the list",
            bg=BG_DARK, fg="#aaaacc",
            font=("Segoe UI", 14),
        )
        self._placeholder.pack(expand=True)

        # Matplotlib figure (hidden until a case is loaded)
        self._fig = plt.Figure(figsize=(8, 4.5), facecolor=BG_DARK)
        self._ax  = self._fig.add_subplot(111, facecolor=BG_DARK)
        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=parent)
        # nav bar hidden - we provide our own Prev / Next

        self._canvas_widget.get_tk_widget().pack_forget()
        self._canvas_frame_packed = False

        # Navigation bar at the bottom
        nav = tk.Frame(parent, bg=BG_PANEL, height=36)
        nav.pack(fill="x", side="bottom")
        nav.pack_propagate(False)

        btn_cfg = dict(
            font=("Segoe UI", 9, "bold"),
            bg="#d0d8f0", fg=FG_BRIGHT,
            activebackground=ACCENT, activeforeground="white",
            relief="flat", bd=0, padx=16, pady=4,
            cursor="hand2",
        )
        self._prev_btn = tk.Button(nav, text="< Prev", command=self._prev_seg, **btn_cfg)
        self._prev_btn.pack(side="left", padx=8, pady=4)

        self._seg_var = tk.StringVar(value="")
        tk.Label(
            nav, textvariable=self._seg_var,
            bg=BG_PANEL, fg=FG_DIM, font=("Segoe UI", 9), width=20,
        ).pack(side="left", padx=4)

        self._next_btn = tk.Button(nav, text="Next >", command=self._next_seg, **btn_cfg)
        self._next_btn.pack(side="left", padx=4)

        self._seg_slider = tk.Scale(
            nav,
            from_=1, to=1, orient="horizontal",
            showvalue=False, resolution=1,
            bg=BG_PANEL, fg=FG_DIM,
            troughcolor="#ccccdd",
            activebackground=ACCENT,
            highlightthickness=0, bd=0,
            sliderlength=16, width=10,
            state="disabled",
            command=self._on_segment_slider,
        )
        self._seg_slider.pack(side="left", fill="x", expand=True, padx=(10, 8))

        # Jump-to-segment entry
        tk.Label(
            nav, text="Jump:", bg=BG_PANEL, fg=FG_DIM,
            font=("Segoe UI", 9),
        ).pack(side="left", padx=(20, 2))
        self._jump_var = tk.StringVar()
        jump_entry = tk.Entry(
            nav, textvariable=self._jump_var,
            width=6, bg="#ccccdd", fg=FG_NORM,
            insertbackground=FG_NORM, relief="flat", font=("Segoe UI", 9),
        )
        jump_entry.pack(side="left")
        jump_entry.bind("<Return>", self._on_jump)

        # Keyboard bindings on root
        self.root.bind("<Left>",  lambda _: self._prev_seg())
        self.root.bind("<Right>", lambda _: self._next_seg())
        self.root.bind("<Up>",    lambda _: self._prev_case())
        self.root.bind("<Down>",  lambda _: self._next_case())

    # -- Split selection -------------------------------------------------------

    def _select_split(self, split: str):
        self._split     = split
        self._current_path = None
        self._x         = None
        self._y         = None

        for s, btn in self._split_btns.items():
            cfg = SPLIT_BTN_ACTIVE if s == split else SPLIT_BTN_INACTIVE
            btn.configure(**cfg)

        self._refresh_list()
        self._clear_canvas()

    def _refresh_list(self):
        rows = self._sorted_rows()
        selected = set(self._tree.selection())
        self._tree.delete(*self._tree.get_children())
        for row in rows:
            iid = str(row["path"])
            self._tree.insert(
                "", "end",
                iid=iid,
                values=self._row_values(row),
            )
            if iid in selected:
                self._tree.selection_add(iid)
        self._update_count(rows)

    @staticmethod
    def _row_values(row: dict) -> tuple:
        return (row["case"], row["segs_text"], row["size"])

    def _update_count(self, rows: list[dict] | None = None):
        rows = self._rows[self._split] if rows is None else rows
        n = len(rows)
        known = sum(1 for r in rows if r["metadata_loaded"])
        total_segs = sum(r["segs"] for r in rows if r["metadata_loaded"])

        if known < n:
            self._count_var.set(
                f"{n} cases - indexed {known}/{n} - "
                f"{total_segs:,} segments known [{self._split}]"
            )
        else:
            self._count_var.set(
                f"{n} cases - {total_segs:,} segments [{self._split}]"
            )

    def _sorted_rows(self) -> list[dict]:
        rows = self._rows[self._split][:]
        rows.sort(key=lambda r: r["case"])
        return rows

    # -- Case selection --------------------------------------------------------

    def _on_case_select(self, _event=None):
        sel = self._tree.selection()
        if not sel:
            return
        path = Path(sel[0])
        if path == self._current_path:
            return
        self._load_case(path)

    def _load_case(self, path: Path):
        self._status_var.set(f"Loading {path.name} ...")
        self.root.update_idletasks()
        try:
            data = np.load(path)
            self._x = data["x"]   # (N, samples)
            self._y = data["y"]   # (N, 2)
        except Exception as e:
            self._status_var.set(f"Error loading {path.name}: {e}")
            return
        self._current_path = path
        self._seg_idx = 0
        self._show_canvas()
        self._show_segment(0)

    def _prev_case(self):
        rows = self._sorted_rows()
        paths = [r["path"] for r in rows]
        if self._current_path not in paths:
            return
        idx = paths.index(self._current_path)
        if idx > 0:
            self._load_case(paths[idx - 1])
            self._select_tree_item(paths[idx - 1])

    def _next_case(self):
        rows = self._sorted_rows()
        paths = [r["path"] for r in rows]
        if self._current_path not in paths:
            return
        idx = paths.index(self._current_path)
        if idx < len(paths) - 1:
            self._load_case(paths[idx + 1])
            self._select_tree_item(paths[idx + 1])

    def _select_tree_item(self, path: Path):
        iid = str(path)
        self._tree.selection_set(iid)
        self._tree.see(iid)

    # -- Segment navigation ----------------------------------------------------

    def _prev_seg(self):
        if self._x is not None and self._seg_idx > 0:
            self._seg_idx -= 1
            self._show_segment(self._seg_idx)

    def _next_seg(self):
        if self._x is not None and self._seg_idx < len(self._x) - 1:
            self._seg_idx += 1
            self._show_segment(self._seg_idx)

    def _on_segment_slider(self, value: str):
        if self._seg_slider_updating or self._x is None:
            return
        idx = int(round(float(value))) - 1
        idx = max(0, min(idx, len(self._x) - 1))
        if idx != self._seg_idx:
            self._seg_idx = idx
            self._show_segment(idx)

    def _on_jump(self, _event=None):
        try:
            idx = int(self._jump_var.get()) - 1  # 1-based input
            if self._x is not None:
                idx = max(0, min(idx, len(self._x) - 1))
                self._seg_idx = idx
                self._show_segment(idx)
        except ValueError:
            pass
        self._jump_var.set("")

    def _configure_segment_slider(self, n_segs: int, enabled: bool):
        self._seg_slider.configure(
            from_=1, to=max(n_segs, 1),
            state="normal" if enabled and n_segs > 1 else "disabled",
        )

    def _set_segment_slider(self, idx: int):
        self._seg_slider_updating = True
        try:
            self._seg_slider.set(idx + 1)
        finally:
            self._seg_slider_updating = False

    # -- Plotting --------------------------------------------------------------

    def _show_canvas(self):
        if not self._canvas_frame_packed:
            self._placeholder.pack_forget()
            self._canvas_widget.get_tk_widget().pack(fill="both", expand=True)
            self._canvas_frame_packed = True

    def _clear_canvas(self):
        if self._canvas_frame_packed:
            self._canvas_widget.get_tk_widget().pack_forget()
            self._placeholder.pack(expand=True)
            self._canvas_frame_packed = False
        self._case_label.configure(text="")
        self._sbp_label.configure(text="")
        self._dbp_label.configure(text="")
        self._seg_var.set("")
        self._configure_segment_slider(1, enabled=False)

    def _show_segment(self, idx: int):
        if self._x is None or self._y is None:
            return

        ppg    = self._x[idx]
        sbp    = float(self._y[idx, 0])
        dbp    = float(self._y[idx, 1])
        n_segs = len(self._x)
        n_samp = len(ppg)
        seg_sec = n_samp / self.target_hz
        t      = np.linspace(0, seg_sec, n_samp)

        # Update info labels
        cid = self._current_path.stem if self._current_path else "?"
        self._case_label.configure(text=f"Case {cid}")
        self._sbp_label.configure(text=f"SBP  {sbp:.0f} mmHg")
        self._dbp_label.configure(text=f"DBP  {dbp:.0f} mmHg")
        self._seg_var.set(f"Segment  {idx + 1} / {n_segs}")
        self._configure_segment_slider(n_segs, enabled=True)
        self._set_segment_slider(idx)

        # Draw waveform
        ax = self._ax
        ax.cla()
        ax.set_facecolor(BG_DARK)

        ax.plot(t, ppg, color=PPG_COLOR, linewidth=0.8, antialiased=True)

        # Horizontal reference lines for SBP and DBP (normalised to signal range)
        # - shown as coloured dashed annotations instead of separate axes
        ppg_min, ppg_max = float(ppg.min()), float(ppg.max())
        ppg_range = ppg_max - ppg_min if ppg_max != ppg_min else 1.0
        margin = ppg_range * 0.12

        ax.set_xlim(0, seg_sec)
        ax.set_ylim(ppg_min - margin, ppg_max + margin)
        ax.set_xlabel("Time (s)", color=FG_DIM, fontsize=9)
        ax.set_ylabel("PPG amplitude", color=FG_DIM, fontsize=9)
        ax.tick_params(colors=FG_DIM, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#ccccdd")
        ax.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--", alpha=0.7)

        # SBP / DBP text annotation inside the plot
        ax.text(
            0.98, 0.97,
            f"SBP  {sbp:.0f} mmHg",
            transform=ax.transAxes,
            ha="right", va="top",
            color=SBP_COLOR, fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_DARK,
                      edgecolor=SBP_COLOR, alpha=0.8),
        )
        ax.text(
            0.98, 0.84,
            f"DBP  {dbp:.0f} mmHg",
            transform=ax.transAxes,
            ha="right", va="top",
            color=DBP_COLOR, fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_DARK,
                      edgecolor=DBP_COLOR, alpha=0.8),
        )

        self._fig.patch.set_facecolor(BG_DARK)
        self._fig.tight_layout(pad=1.2)
        self._canvas_widget.draw_idle()

        self._status_var.set(
            f"Case {cid}  |  segment {idx + 1}/{n_segs}"
            f"  |  {n_samp} samples @ {self.target_hz} Hz"
            f"  |  SBP {sbp:.0f}  DBP {dbp:.0f} mmHg"
            f"  |  [UpDown case  <--> segment]"
        )

        # Enable / disable nav buttons
        self._prev_btn.configure(state="normal" if idx > 0 else "disabled")
        self._next_btn.configure(state="normal" if idx < n_segs - 1 else "disabled")


# -- Entry point ---------------------------------------------------------------

def main():
    args = parse_args()

    if not args.dataset_dir.exists():
        print(f"Dataset directory not found: {args.dataset_dir}", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    DatasetBrowser(root, args.dataset_dir, args.target_hz)
    root.mainloop()


if __name__ == "__main__":
    main()
