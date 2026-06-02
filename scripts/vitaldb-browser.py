"""
VitalDB waveform browser — unified single-window UI
Left panel: case list (always visible, sortable, searchable)
Right panel: matplotlib waveform canvas (updates on case click)

Usage:
    uv run python scripts/vitaldb-browser.py [--data-dir data/vitaldb] [--case CASEID]
"""

from vitaldb.utils import VitalFile
from matplotlib.widgets import Slider
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import vitaldb
import numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import sys
import tkinter as tk
import tkinter.ttk as ttk
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")  # must be set before importing pyplot


# ── Korean font (renders cleanly on Windows; no-op on others) ─────────────────
def _set_cjk_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rc("font", family=name)
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_set_cjk_font()


# ── Constants ─────────────────────────────────────────────────────────────────
API_URL = "https://api.vitaldb.net"
SRATE = 500    # waveform sample rate (Hz)
WINDOW_SEC = 30     # visible time window (s)
STEP_SEC = 10     # navigation step (s)

# (track_name, label, color, is_waveform)
TRACK_DEFS = [
    ("SNUADC/PLETH",      "PPG",           "#00cc88", True),
    ("SNUADC/ART",        "ABP",           "#e05050", True),
    ("SNUADC/ECG_II",     "ECG II",        "#5588ff", True),
    ("Solar8000/ART_SBP", "SBP (mmHg)",    "#ff6644", False),
    ("Solar8000/ART_DBP", "DBP (mmHg)",    "#ffaa00", False),
    ("Solar8000/ART_MBP", "MBP (mmHg)",    "#ff88cc", False),
    ("Solar8000/HR",      "HR (/min)",     "#44ddff", False),
]
WAVE_TRACKS = [(n, l, c) for n, l, c, w in TRACK_DEFS if w]
NUMERIC_TRACKS = [(n, l, c) for n, l, c, w in TRACK_DEFS if not w]

# Row highlight tags: (foreground, background)
TAG_COLORS = {
    "none":    ("#ccccdd", "#0d0d1f"),
    "ppg":     ("#44ffaa", "#0d0d1f"),   # bright green text
    "abp":     ("#ccccdd", "#400d0d"),   # dark crimson bg
    "ppg_abp": ("#44ffaa", "#400d0d"),   # both
}


# ── Data helpers ──────────────────────────────────────────────────────────────

def list_vital_files(data_dir: Path) -> list[Path]:
    return sorted(
        data_dir.glob("*.vital"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )


def fetch_clinical_map(files: list[Path]) -> dict:
    try:
        caseids = [int(f.stem) for f in files if f.stem.isdigit()]
        ci = vitaldb.load_clinical_data(
            caseids=caseids,
            params=["caseid", "age", "sex", "opname", "caseend"],
        )
        return {row["caseid"]: row for _, row in ci.iterrows()}
    except Exception:
        return {}


def fetch_track_flags(files: list[Path]) -> dict[int, tuple[bool, bool]]:
    """Return {caseid: (has_ppg, has_abp)} from the public trks index."""
    import warnings
    import pandas as pd

    caseids = {int(f.stem) for f in files if f.stem.isdigit()}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            df = pd.read_csv(f"{API_URL}/trks")
        df = df[df["caseid"].isin(caseids)]
        ppg = set(df.loc[df["tname"] == "SNUADC/PLETH", "caseid"])
        abp = set(df.loc[df["tname"] == "SNUADC/ART",   "caseid"])
        return {cid: (cid in ppg, cid in abp) for cid in caseids}
    except Exception:
        return {}


def load_vital(path: Path) -> tuple[VitalFile, dict[str, np.ndarray]]:
    """Load a .vital file; return (VitalFile, {track_name: ndarray})."""
    vf = VitalFile(str(path))
    available = set(vf.get_track_names())

    wave_names = [n for n, *_ in WAVE_TRACKS if n in available]
    numeric_names = [n for n, *_ in NUMERIC_TRACKS if n in available]
    data: dict[str, np.ndarray] = {}

    if wave_names:
        arr = vf.to_numpy(wave_names, interval=1 / SRATE)
        for i, name in enumerate(wave_names):
            data[name] = arr[:, i]

    if numeric_names:
        arr = vf.to_numpy(numeric_names, interval=1)
        for i, name in enumerate(numeric_names):
            data[name] = arr[:, i]

    return vf, data


def duration_sec(data: dict) -> float:
    for name, *_ in WAVE_TRACKS:
        if name in data:
            return len(data[name]) / SRATE
    for name in data:
        return float(len(data[name]))
    return 0.0


# ── Main application window ───────────────────────────────────────────────────

class VitalDBBrowser:
    """Single unified window: case list on the left, waveform canvas on the right."""

    LIST_WIDTH = 460   # px — left panel minimum width
    CANVAS_W = 950   # px — right panel minimum width
    WIN_H = 700

    LIST_COLUMNS = [
        ("case",     "Case",      58,  "center"),
        ("duration", "Dur.",      72,  "center"),
        ("age",      "Age",       42,  "center"),
        ("sex",      "Sex",       36,  "center"),
        ("size",     "Size",      66,  "center"),
        ("opname",   "Operation", 160, "w"),
    ]

    def __init__(self, root: tk.Tk, data_dir: Path,
                 files: list[Path], ci_map: dict,
                 track_flags: dict[int, tuple[bool, bool]]):
        self.root = root
        self.data_dir = data_dir
        self.files = files
        self.ci_map = ci_map
        self.track_flags = track_flags

        # Waveform state
        self._vf:   VitalFile | None = None
        self._data: dict[str, np.ndarray] | None = None
        self._t0 = 0.0
        self._dur = 0.0
        self._wave_axes: list[plt.Axes] = []
        self._num_ax:    plt.Axes | None = None
        self._slider:    Slider | None = None
        self._time_text = None

        # Sorted row cache
        self._sort_col = "case"
        self._sort_rev = False
        self._all_rows = self._build_rows()

        self._build_ui()
        self._refresh_list()

    # ── Row data ──────────────────────────────────────────────────────────────

    def _build_rows(self) -> list[dict]:
        rows = []
        for f in self.files:
            cid = int(f.stem) if f.stem.isdigit() else 0
            info = self.ci_map.get(cid)
            size_mb = f.stat().st_size / 1024 / 1024

            if info is not None:
                dur = int(info.get("caseend", 0) or 0)
                age = str(info.get("age", ""))
                sex = str(info.get("sex", ""))
                opname = str(info.get("opname", ""))
                dur_s = f"{dur//60}m{dur%60:02d}s"
            else:
                dur = 0
                age = ""
                sex = ""
                opname = ""
                dur_s = ""

            has_ppg, has_abp = self.track_flags.get(cid, (False, False))
            tag = ("ppg_abp" if has_ppg and has_abp
                   else "ppg" if has_ppg
                   else "abp" if has_abp
                   else "none")

            rows.append(dict(path=f, case=cid, duration=dur_s, dur_sec=dur,
                             age=age, sex=sex, size=f"{size_mb:.1f}MB",
                             size_val=size_mb, opname=opname, tag=tag))
        return rows

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.title("VitalDB Browser")
        self.root.configure(bg="#1a1a2e")
        self.root.geometry(f"{self.LIST_WIDTH + self.CANVAS_W}x{self.WIN_H}")
        self.root.minsize(900, 500)

        # ── Top-level paned window ────────────────────────────────────────────
        paned = tk.PanedWindow(self.root, orient="horizontal",
                               bg="#1a1a2e", sashwidth=5, sashrelief="flat",
                               handlesize=0)
        paned.pack(fill="both", expand=True)

        # ── Left panel ───────────────────────────────────────────────────────
        left = tk.Frame(paned, bg="#1a1a2e", width=self.LIST_WIDTH)
        left.pack_propagate(False)
        paned.add(left, minsize=280)

        self._build_list_panel(left)

        # ── Right panel ──────────────────────────────────────────────────────
        right = tk.Frame(paned, bg="#0d0d1f")
        paned.add(right, minsize=500)

        self._build_canvas_panel(right)

        # ── Bottom status bar ─────────────────────────────────────────────────
        bar = tk.Frame(self.root, bg="#111128", height=22)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._status_var = tk.StringVar(value="Select a case from the list.")
        tk.Label(bar, textvariable=self._status_var,
                 bg="#111128", fg="#666688",
                 font=("Segoe UI", 9), anchor="w").pack(side="left", padx=8)

    # ── Left panel: case list ─────────────────────────────────────────────────

    def _build_list_panel(self, parent: tk.Frame):
        # Search bar
        top = tk.Frame(parent, bg="#1a1a2e")
        top.pack(fill="x", padx=8, pady=(8, 4))

        tk.Label(top, text="Search:", bg="#1a1a2e", fg="#aaaacc",
                 font=("Segoe UI", 9)).pack(side="left")
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._refresh_list())
        tk.Entry(top, textvariable=self._search_var,
                 bg="#2a2a4a", fg="white", insertbackground="white",
                 relief="flat", font=("Segoe UI", 9), width=18
                 ).pack(side="left", padx=(4, 0))

        # Legend
        legend = tk.Frame(parent, bg="#1a1a2e")
        legend.pack(fill="x", padx=8, pady=(0, 4))
        tk.Label(legend, text="●", bg="#1a1a2e", fg="#44ffaa",
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Label(legend, text="PPG  ", bg="#1a1a2e", fg="#888899",
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Label(legend, text="■", bg="#280d0d", fg="#ff6666",
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Label(legend, text="ABP", bg="#1a1a2e", fg="#888899",
                 font=("Segoe UI", 9)).pack(side="left")

        # Treeview
        frame = tk.Frame(parent, bg="#1a1a2e")
        frame.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("L.Treeview",
                        background="#0d0d1f", foreground="#ccccdd",
                        fieldbackground="#0d0d1f", rowheight=22,
                        font=("Segoe UI", 9))
        style.configure("L.Treeview.Heading",
                        background="#2a2a5a", foreground="#aaaaff",
                        font=("Segoe UI", 9, "bold"), relief="flat")
        style.map("L.Treeview",
                  background=[("selected", "#3344aa")],
                  foreground=[("selected", "white")])

        col_ids = [c[0] for c in self.LIST_COLUMNS]
        self._tree = ttk.Treeview(frame, columns=col_ids, show="headings",
                                  style="L.Treeview", selectmode="browse")
        for cid, heading, width, anchor in self.LIST_COLUMNS:
            self._tree.heading(cid, text=heading,
                               command=lambda c=cid: self._sort_by(c))
            self._tree.column(cid, width=width, anchor=anchor,
                              stretch=(cid == "opname"))

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        for tag, (fg, bg) in TAG_COLORS.items():
            self._tree.tag_configure(tag, foreground=fg, background=bg)

        self._tree.bind("<<TreeviewSelect>>", self._on_case_select)
        self._tree.bind("<Double-1>",         self._on_case_select)

        # List status
        self._list_status = tk.StringVar()
        tk.Label(parent, textvariable=self._list_status,
                 bg="#1a1a2e", fg="#666688",
                 font=("Segoe UI", 8), anchor="w").pack(fill="x", padx=8, pady=(0, 4))

    # ── Right panel: matplotlib canvas ───────────────────────────────────────

    def _build_canvas_panel(self, parent: tk.Frame):
        # Placeholder shown before any case is selected
        self._placeholder = tk.Label(
            parent,
            text="← Select a case from the list",
            bg="#0d0d1f", fg="#333355",
            font=("Segoe UI", 14),
        )
        self._placeholder.pack(expand=True)

        # Figure (hidden until a case is loaded)
        self._fig = plt.Figure(facecolor="#1a1a2e")
        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas_widget = self._canvas.get_tk_widget()
        # Not packed yet — shown after first case load

        # Navigation bar below the canvas
        self._nav_frame = tk.Frame(parent, bg="#1a1a2e")
        # Also deferred

    def _show_canvas(self):
        """Swap placeholder for the matplotlib canvas on first load."""
        if self._placeholder.winfo_ismapped():
            self._placeholder.pack_forget()
            self._canvas_widget.pack(fill="both", expand=True)
            self._nav_frame.pack(fill="x")
            self._build_nav_bar(self._nav_frame)

    def _build_nav_bar(self, parent: tk.Frame):
        bc = "#2a2a5a"
        hc = "#4455aa"
        kw = dict(bg=bc, fg="white", activebackground=hc, activeforeground="white",
                  relief="flat", font=("Segoe UI", 9), padx=10, pady=3, cursor="hand2")

        bar = tk.Frame(parent, bg="#1a1a2e")
        bar.pack(fill="x", padx=6, pady=(2, 4))

        tk.Button(bar, text="<< 60s", command=lambda: self._shift(-60),
                  **kw).pack(side="left", padx=2)
        tk.Button(bar, text=f"< {STEP_SEC}s", command=lambda: self._shift(
            -STEP_SEC), **kw).pack(side="left", padx=2)

        # Slider (canvas-less, pure tkinter)
        self._tk_slider = tk.Scale(
            bar, orient="horizontal", from_=0, to=1,
            resolution=1, showvalue=False,
            bg="#1a1a2e", fg="#aaaacc", troughcolor="#2a2a4a",
            highlightthickness=0, bd=0, sliderlength=14,
            command=self._on_tk_slider,
        )
        self._tk_slider.pack(side="left", fill="x", expand=True, padx=4)

        tk.Button(bar, text=f"{STEP_SEC}s >", command=lambda: self._shift(
            +STEP_SEC), **kw).pack(side="left", padx=2)
        tk.Button(bar, text="60s >>", command=lambda: self._shift(+60),
                  **kw).pack(side="left", padx=2)

        kw2 = dict(kw, bg="#1a3a4a", activebackground="#2a5a6a")
        tk.Button(bar, text="Track Info", command=self._show_track_info,
                  **kw2).pack(side="right", padx=(8, 2))

        self._nav_time = tk.StringVar()
        tk.Label(bar, textvariable=self._nav_time,
                 bg="#1a1a2e", fg="#aaaacc",
                 font=("Consolas", 9)).pack(side="right", padx=8)

        # Keyboard bindings on root
        self.root.bind("<Left>", lambda e: self._shift(-STEP_SEC))
        self.root.bind("<Right>", lambda e: self._shift(+STEP_SEC))
        self.root.bind("<Control-Left>", lambda e: self._shift(-60))
        self.root.bind("<Control-Right>", lambda e: self._shift(+60))

    # ── List management ───────────────────────────────────────────────────────

    def _refresh_list(self):
        q = self._search_var.get().lower()
        rows = [
            r for r in self._all_rows
            if not q
            or q in str(r["case"])
            or q in r["opname"].lower()
            or q in r["age"]
            or q in r["sex"].lower()
        ]

        key_fn = {
            "case": lambda r: r["case"],
            "duration": lambda r: r["dur_sec"],
            "age": lambda r: int(r["age"]) if r["age"].isdigit() else 0,
            "sex": lambda r: r["sex"],
            "size": lambda r: r["size_val"],
            "opname": lambda r: r["opname"],
        }.get(self._sort_col, lambda r: r["case"])
        rows.sort(key=key_fn, reverse=self._sort_rev)

        # Remember current selection
        sel = self._tree.selection()
        sel_iid = sel[0] if sel else None

        self._tree.delete(*self._tree.get_children())
        for r in rows:
            self._tree.insert("", "end", iid=str(r["path"]), tags=(r["tag"],),
                              values=(r["case"], r["duration"], r["age"],
                                      r["sex"], r["size"], r["opname"]))

        # Restore selection or select first
        if sel_iid and self._tree.exists(sel_iid):
            self._tree.selection_set(sel_iid)
            self._tree.see(sel_iid)
        elif self._tree.get_children():
            first = self._tree.get_children()[0]
            self._tree.selection_set(first)

        n_ppg = sum(1 for r in rows if r["tag"] in ("ppg", "ppg_abp"))
        n_abp = sum(1 for r in rows if r["tag"] in ("abp", "ppg_abp"))
        arr = (" ▲" if not self._sort_rev else " ▼")
        for cid, heading, *_ in self.LIST_COLUMNS:
            self._tree.heading(cid, text=heading +
                               (arr if cid == self._sort_col else ""))
        self._list_status.set(
            f"{len(rows)}/{len(self._all_rows)}  PPG:{n_ppg}  ABP:{n_abp}"
        )

    def _sort_by(self, col: str):
        self._sort_rev = (col == self._sort_col) and not self._sort_rev
        self._sort_col = col
        self._refresh_list()

    # ── Case loading ──────────────────────────────────────────────────────────

    def _on_case_select(self, event=None):
        sel = self._tree.selection()
        if not sel:
            return
        path = Path(sel[0])
        if self._vf is not None and path == getattr(self, "_current_path", None):
            return  # same case — no reload
        self._load_case(path)

    def _load_case(self, path: Path):
        self._set_status(f"Loading {path.name} …")
        self.root.update_idletasks()

        try:
            vf, data = load_vital(path)
        except Exception as e:
            self._set_status(f"[ERROR] {path.name}: {e}")
            return

        avail = [n for n, *_ in WAVE_TRACKS + NUMERIC_TRACKS if n in data]
        if not avail:
            self._set_status(f"[WARNING] No displayable tracks in {path.name}")
            return

        self._vf = vf
        self._data = data
        self._dur = duration_sec(data)
        self._t0 = 0.0
        self._current_path = path

        self._show_canvas()
        self._rebuild_figure()

        dur_m, dur_s = int(self._dur) // 60, int(self._dur) % 60
        self._set_status(
            f"Case {path.stem}  |  tracks: {', '.join(avail)}"
            f"  |  duration: {dur_m}m {dur_s}s"
            f"  |  Keys: ←→ {STEP_SEC}s   Ctrl+←→ 60s"
        )

    # ── Figure construction ───────────────────────────────────────────────────

    def _rebuild_figure(self):
        """Reconstruct the matplotlib layout for the newly loaded case."""
        self._fig.clear()
        self._wave_axes = []
        self._num_ax = None
        self._time_text = None

        avail_waves = [(n, l, c) for n, l, c in WAVE_TRACKS if n in self._data]
        avail_numeric = [(n, l, c)
                         for n, l, c in NUMERIC_TRACKS if n in self._data]
        self._avail_waves = avail_waves
        self._avail_numeric = avail_numeric

        n_wave = len(avail_waves)
        n_num = len(avail_numeric)
        n_rows = n_wave + (1 if n_num else 0)

        if n_rows == 0:
            return

        gs = gridspec.GridSpec(
            n_rows, 1, figure=self._fig,
            hspace=0.06, top=0.94, bottom=0.06,
            height_ratios=[2] * n_wave + ([1.4] if n_num else []),
        )

        for i in range(n_wave):
            ax = self._fig.add_subplot(gs[i, 0])
            self._wave_axes.append(ax)
        if n_num:
            self._num_ax = self._fig.add_subplot(gs[n_wave, 0])

        # Share x-axis
        for ax in self._wave_axes[1:]:
            ax.sharex(self._wave_axes[0])

        # Case title
        caseid = int(
            self._current_path.stem) if self._current_path.stem.isdigit() else 0
        try:
            ci = vitaldb.load_clinical_data(
                caseids=[caseid],
                params=["caseid", "age", "sex", "height",
                        "weight", "opname", "ane_type"],
            )
            row = ci.iloc[0]
            title = (f"Case {caseid}  |  {row.get('age','?')}y/{row.get('sex','?')}  |  "
                     f"{row.get('height','?')}cm {row.get('weight','?')}kg  |  "
                     f"{str(row.get('opname',''))[:38]}  |  Ane: {row.get('ane_type','?')}")
        except Exception:
            title = f"Case {caseid}"
        self._fig.suptitle(title, color="white", fontsize=10, y=0.99)

        # Update slider range
        if hasattr(self, "_tk_slider"):
            self._tk_slider.config(to=max(int(self._dur) - WINDOW_SEC, 1))
            self._tk_slider.set(0)

        self._draw()

    # ── Waveform drawing ──────────────────────────────────────────────────────

    def _wave_slice(self, name: str):
        arr = self._data[name]
        i0 = int(self._t0 * SRATE)
        i1 = min(i0 + WINDOW_SEC * SRATE, len(arr))
        return np.arange(i0, i1) / SRATE, arr[i0:i1]

    def _num_slice(self, name: str):
        arr = self._data[name]
        i0 = int(self._t0)
        i1 = min(i0 + WINDOW_SEC, len(arr))
        return np.arange(i0, i1, dtype=float), arr[i0:i1]

    def _draw(self):
        if self._data is None:
            return

        t_end = self._t0 + WINDOW_SEC

        for ax, (name, label, color) in zip(self._wave_axes, self._avail_waves):
            ax.cla()
            ax.set_facecolor("#0d0d1f")
            t, y = self._wave_slice(name)
            valid = ~np.isnan(y)
            if valid.any():
                ax.plot(t[valid], y[valid], color=color,
                        lw=0.6, rasterized=True)
                p1, p99 = np.nanpercentile(y, [1, 99])
                margin = max((p99 - p1) * 0.15, 0.5)
                ax.set_ylim(p1 - margin, p99 + margin)
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="#444466", fontsize=9)
            ax.set_xlim(self._t0, t_end)
            ax.set_ylabel(label, color=color, fontsize=8, labelpad=2)
            ax.tick_params(colors="#666688", labelsize=7)
            ax.spines[:].set_color("#222244")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Hide x-tick labels on all waveform panels except the last
        for ax in self._wave_axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        if self._wave_axes:
            self._wave_axes[-1].set_xlabel("Time (s)",
                                           color="#aaaacc", fontsize=8)

        if self._num_ax:
            self._num_ax.cla()
            self._num_ax.set_facecolor("#0d0d1f")
            for name, label, color in self._avail_numeric:
                t, y = self._num_slice(name)
                valid = ~np.isnan(y)
                if valid.any():
                    self._num_ax.plot(t[valid], y[valid], color=color,
                                      lw=1.1, marker=".", ms=2, label=label)
            self._num_ax.set_xlim(self._t0, t_end)
            self._num_ax.set_ylabel("Numeric", color="white", fontsize=8)
            self._num_ax.set_xlabel("Time (s)", color="#aaaacc", fontsize=8)
            self._num_ax.legend(loc="upper right", fontsize=7, framealpha=0.3,
                                labelcolor="white", facecolor="#0d0d1f")
            self._num_ax.tick_params(colors="#666688", labelsize=7)
            self._num_ax.spines[:].set_color("#222244")
            self._num_ax.spines["top"].set_visible(False)
            self._num_ax.spines["right"].set_visible(False)

        # Time stamp in nav bar
        def _fmt(s):
            s = int(s)
            return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

        if hasattr(self, "_nav_time"):
            total_m, total_s = int(self._dur) // 60, int(self._dur) % 60
            self._nav_time.set(
                f"{_fmt(self._t0)} ~ {_fmt(t_end)}"
                f"  / {total_m}m{total_s:02d}s"
            )

        self._canvas.draw_idle()

    # ── Navigation ────────────────────────────────────────────────────────────

    def _shift(self, delta: float):
        if self._data is None:
            return
        self._t0 = float(np.clip(self._t0 + delta, 0,
                                 max(self._dur - WINDOW_SEC, 0)))
        if hasattr(self, "_tk_slider"):
            self._tk_slider.set(int(self._t0))
        self._draw()

    def _on_tk_slider(self, val):
        if self._data is None:
            return
        self._t0 = float(val)
        self._draw()

    # ── Track info window ─────────────────────────────────────────────────────

    def _show_track_info(self):
        if self._vf is None:
            return

        win = tk.Toplevel(self.root)
        win.title(f"Track Info — Case {self._current_path.stem}")
        win.configure(bg="#1a1a2e")
        win.geometry("620x460")

        cols = [("track", "Track Name", 255, "w"), ("type", "Type", 88, "center"),
                ("unit", "Unit", 68, "center"), ("recs", "Records", 68, "center")]
        frame = tk.Frame(win, bg="#1a1a2e")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        style = ttk.Style()
        style.configure("TI.Treeview",
                        background="#0d0d1f", foreground="#ccccdd",
                        fieldbackground="#0d0d1f", rowheight=21,
                        font=("Consolas", 9))
        style.configure("TI.Treeview.Heading",
                        background="#2a2a5a", foreground="#aaaaff",
                        font=("Segoe UI", 9, "bold"), relief="flat")
        style.map("TI.Treeview",
                  background=[("selected", "#3344aa")],
                  foreground=[("selected", "white")])

        col_ids = [c[0] for c in cols]
        tree = ttk.Treeview(frame, columns=col_ids, show="headings",
                            style="TI.Treeview", selectmode="browse")
        for cid, heading, width, anchor in cols:
            tree.heading(cid, text=heading)
            tree.column(cid, width=width, anchor=anchor,
                        stretch=(cid == "track"))

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        for i, (name, trk) in enumerate(self._vf.trks.items()):
            kind = f"{trk.srate:.0f} Hz" if trk.srate > 0 else "numeric"
            unit = getattr(trk, "unit", "") or ""
            tag = "even" if i % 2 == 0 else "odd"
            tree.insert("", "end", values=(
                name, kind, unit, len(trk.recs)), tags=(tag,))

        tree.tag_configure("even", background="#0d0d1f")
        tree.tag_configure("odd",  background="#111128")

        bot = tk.Frame(win, bg="#1a1a2e")
        bot.pack(fill="x", padx=10, pady=(0, 10))
        n_w = sum(1 for t in self._vf.trks.values() if t.srate > 0)
        n_n = sum(1 for t in self._vf.trks.values() if t.srate == 0)
        tk.Label(bot, text=f"{len(self._vf.trks)} tracks — {n_w} waveform, {n_n} numeric",
                 bg="#1a1a2e", fg="#666688", font=("Segoe UI", 9)).pack(side="left")
        tk.Button(bot, text="Close", command=win.destroy,
                  bg="#2a2a5a", fg="white", activebackground="#4455aa",
                  activeforeground="white", relief="flat",
                  font=("Segoe UI", 9), padx=12, pady=3, cursor="hand2"
                  ).pack(side="right")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, text: str):
        self._status_var.set(text)
        self.root.update_idletasks()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="VitalDB waveform browser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/vitaldb"),
                   help="Directory containing .vital files (default: data/vitaldb)")
    p.add_argument("--case", type=int, default=None,
                   help="Open a specific case ID on startup")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir: Path = args.data_dir

    files = list_vital_files(data_dir)
    if not files:
        # Minimal error dialog before Tk main loop
        root = tk.Tk()
        root.withdraw()
        import tkinter.messagebox as mb
        mb.showerror("VitalDB Browser",
                     f"No .vital files found in:\n{data_dir.resolve()}\n\n"
                     "Run  bin\\download-vitaldb.bat  first.")
        root.destroy()
        sys.exit(1)

    ci_map = fetch_clinical_map(files)
    track_flags = fetch_track_flags(files)

    root = tk.Tk()
    app = VitalDBBrowser(root, data_dir, files, ci_map, track_flags)

    # Auto-open case if specified
    if args.case is not None:
        path = data_dir / f"{args.case}.vital"
        if path.exists():
            app._load_case(path)
            # Scroll list to that case
            iid = str(path)
            if app._tree.exists(iid):
                app._tree.selection_set(iid)
                app._tree.see(iid)
        else:
            print(f"[ERROR] File not found: {path}", file=sys.stderr)
            sys.exit(1)

    root.mainloop()


if __name__ == "__main__":
    main()
