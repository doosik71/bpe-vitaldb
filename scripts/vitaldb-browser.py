"""
VitalDB waveform browser - unified single-window UI
Left panel: case list (always visible, sortable, searchable)
Center panel: matplotlib waveform canvas (updates on case click)
Right panel: SBP/DBP search within the current case

Usage:
    uv run python scripts/vitaldb-browser.py [--data-dir data/vitaldb] [--case CASEID]
"""

from vitaldb.utils import VitalFile
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
from queue import Empty, LifoQueue, Queue
import threading

from bpe.utils.qc_v2 import (
    QCResult,
    compute_window_qc,
    SOURCE_HZ as QC_SOURCE_HZ,
    TARGET_HZ as QC_TARGET_HZ,
    SEGMENT_SEC as QC_SEGMENT_SEC,
)

import matplotlib
matplotlib.use("TkAgg")  # must be set before importing pyplot


# -- Korean font (renders cleanly on Windows; no-op on others) -----------------
def _set_cjk_font():
    candidates = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Gulim"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rc("font", family=name)
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


_set_cjk_font()


# -- Constants -----------------------------------------------------------------
API_URL = "https://api.vitaldb.net"
SRATE = 500         # waveform sample rate (Hz)
WINDOW_SEC = 32     # visible time window (s) — 4 × 8 s QC segments
STEP_SEC = 8        # navigation step (s) — 1 QC segment
LARGE_STEP_SEC = 64 # Ctrl+arrow step (s) — 8 QC segments

# (track_name, label, color, is_waveform)
TRACK_DEFS = [
    ("SNUADC/PLETH",       "PPG",      "#1a8855", True),
    ("SNUADC/ART",         "ABP",      "#cc2200", True),
    ("SNUADC/ECG_II",      "ECG II",   "#2255bb", True),
    ("Solar8000/ART_SBP",  "SBP",      "#cc2200", False),
    ("Solar8000/ART_DBP",  "DBP",      "#2255bb", False),
    ("Solar8000/ART_MBP",  "MBP",      "#228844", False),
    ("Solar8000/NIBP_SBP", "NIBP SBP", "#cc2200", False),
    ("Solar8000/NIBP_DBP", "NIBP DBP", "#2255bb", False),
    ("Solar8000/NIBP_MBP", "NIBP MBP", "#228844", False),
]
WAVE_TRACKS = [(n, l, c) for n, l, c, w in TRACK_DEFS if w]
NUMERIC_TRACKS = [(n, l, c) for n, l, c, w in TRACK_DEFS if not w]

# NIBP tracks use dashed lines to distinguish from invasive ART tracks
_NIBP_TRACKS = frozenset({
    "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Solar8000/NIBP_MBP",
})
# Tracks whose integer values are annotated on the numeric panel
_LABEL_TRACKS = frozenset({
    "Solar8000/ART_SBP",  "Solar8000/ART_DBP",
    "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP",
})
_LABEL_VA = {
    "Solar8000/ART_SBP":  "bottom",
    "Solar8000/ART_DBP":  "top",
    "Solar8000/NIBP_SBP": "bottom",
    "Solar8000/NIBP_DBP": "top",
}

# Row highlight tags: (foreground, background)
TAG_COLORS = {
    "unknown": ("#777777", "#cccccc"),
    "none":    ("#0000dd", "#cccccc"),
    "ppg":     ("#dd0000", "#cccccc"),
    "abp":     ("#008800", "#cccccc"),
    "ppg_abp": ("#000000", "#ffffff"),
}

# -- Data helpers --------------------------------------------------------------


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


def scan_track_flags(path: Path) -> tuple[bool, bool]:
    """Read only track headers needed for list coloring."""
    try:
        names = set(VitalFile(str(path), header_only=True).get_track_names())
        return "SNUADC/PLETH" in names, "SNUADC/ART" in names
    except Exception:
        return False, False


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


# -- Main application window ---------------------------------------------------

class VitalDBBrowser:
    """Single unified window with case list, waveform canvas, and BP search."""

    LIST_WIDTH = 460   # px - left panel width
    CANVAS_W = 1000   # px - center panel minimum width
    SEARCH_W = 280   # px - right panel width
    WIN_H = 800

    LIST_COLUMNS = [
        ("case",     "Case",      58,  "center"),
        ("duration", "Dur.",      72,  "center"),
        ("age",      "Age",       42,  "center"),
        ("sex",      "Sex",       36,  "center"),
        ("size",     "Size",      66,  "center"),
        ("opname",   "Operation", 160, "w"),
    ]

    def __init__(self, root: tk.Tk, data_dir: Path,
                 files: list[Path], ci_map: dict):
        self.root = root
        self.data_dir = data_dir
        self.files = files
        self.ci_map = ci_map
        self.track_flags: dict[int, tuple[bool, bool]] = {}
        self._row_by_path: dict[Path, dict] = {}
        self._filtered_rows: list[dict] = []
        max_cid = max((int(f.stem)
                      for f in files if f.stem.isdigit()), default=0)
        self._track_scanned = np.zeros(max_cid + 1, dtype=bool)
        self._track_has_ppg = np.zeros(max_cid + 1, dtype=bool)
        self._track_has_abp = np.zeros(max_cid + 1, dtype=bool)
        self._scan_queue: LifoQueue[Path | None] = LifoQueue()
        self._scan_results: Queue[tuple[int, Path, bool, bool]] = Queue()
        self._scan_stop = threading.Event()

        # Waveform state
        self._vf: VitalFile | None = None
        self._data: dict[str, np.ndarray] | None = None
        self._t0 = 0.0
        self._dur = 0.0
        self._wave_axes: list[plt.Axes] = []
        self._num_ax: plt.Axes | None = None
        self._time_text = None
        self._bp_match_times: list[int] = []
        self._current_bp_mode = "exact"
        self._canvas_ready = False

        # Dataset-v2 QC overlay state
        self._has_ppg_abp = False
        self._ppg_dec: np.ndarray | None = None
        self._abp_dec: np.ndarray | None = None
        self._qc_cache: dict[float, QCResult] = {}
        self._qc_ax: plt.Axes | None = None

        # Sorted row cache
        self._sort_col = "case"
        self._sort_rev = False
        self._all_rows = self._build_rows()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._start_track_scan_worker()
        self.root.after(120, self._drain_track_scan_results)
        self._refresh_list()

    # -- Row data --------------------------------------------------------------

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

            row = dict(path=f, case=cid, duration=dur_s, dur_sec=dur,
                       age=age, sex=sex, size=f"{size_mb:.1f}MB",
                       size_val=size_mb, opname=opname, tag="unknown")
            rows.append(row)
            self._row_by_path[f] = row
        return rows

    # -- UI construction -------------------------------------------------------

    def _build_ui(self):
        self.root.title("VitalDB Browser")
        self.root.configure(bg="#f0f0f7")
        self.root.geometry(
            f"{self.LIST_WIDTH + self.CANVAS_W + self.SEARCH_W}x{self.WIN_H}"
        )
        self.root.minsize(1180, 500)

        content = tk.Frame(self.root, bg="#f0f0f7")
        content.pack(fill="both", expand=True)
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=0, minsize=self.LIST_WIDTH)
        content.grid_columnconfigure(1, weight=1, minsize=500)
        content.grid_columnconfigure(2, weight=0, minsize=self.SEARCH_W)

        # -- Left panel -------------------------------------------------------
        left = tk.Frame(content, bg="#f0f0f7", width=self.LIST_WIDTH)
        left.grid(row=0, column=0, sticky="nsew")
        left.grid_propagate(False)

        self._build_list_panel(left)

        # -- Center panel -----------------------------------------------------
        center = tk.Frame(content, bg="#ffffff", width=self.CANVAS_W)
        center.grid(row=0, column=1, sticky="nsew")
        center.grid_propagate(False)

        self._build_canvas_panel(center)

        # -- Right panel ------------------------------------------------------
        right = tk.Frame(content, bg="#f6f6fb", width=self.SEARCH_W)
        right.grid(row=0, column=2, sticky="nsew")
        right.grid_propagate(False)

        self._build_search_panel(right)

        # -- Bottom status bar -------------------------------------------------
        bar = tk.Frame(self.root, bg="#e8e8f2", height=22)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._status_var = tk.StringVar(value="Select a case from the list.")
        tk.Label(bar, textvariable=self._status_var,
                 bg="#e8e8f2", fg="#888899",
                 font=("Segoe UI", 9), anchor="w").pack(side="left", padx=8)

    # -- Left panel: case list -------------------------------------------------

    def _build_list_panel(self, parent: tk.Frame):
        # Search bar
        top = tk.Frame(parent, bg="#f0f0f7")
        top.pack(fill="x", padx=8, pady=(8, 4))

        tk.Label(top, text="Search:", bg="#f0f0f7", fg="#888899",
                 font=("Segoe UI", 9)).pack(side="left")
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._refresh_list())
        tk.Entry(top, textvariable=self._search_var,
                 bg="#ccccdd", fg="#222233", insertbackground="#222233",
                 relief="flat", font=("Segoe UI", 9), width=18
                 ).pack(side="left", padx=(4, 0))

        # Legend
        legend = tk.Frame(parent, bg="#f0f0f7")
        legend.pack(fill="x", padx=8, pady=(0, 4))
        tk.Label(legend, text="o", bg="#f0f0f7", fg="#228844",
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Label(legend, text="No PPG  ", bg="#f0f0f7", fg="#888899",
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Label(legend, text="#", bg="#fde8e8", fg="#cc2200",
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Label(legend, text="No ABP", bg="#f0f0f7", fg="#888899",
                 font=("Segoe UI", 9)).pack(side="left")

        # Treeview
        frame = tk.Frame(parent, bg="#f0f0f7")
        frame.pack(fill="both", expand=True, padx=8, pady=(0, 4))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("L.Treeview",
                        background="#ffffff", foreground="#222233",
                        fieldbackground="#ffffff", rowheight=22,
                        font=("Segoe UI", 9))
        style.configure("L.Treeview.Heading",
                        background="#d0d8f0", foreground="#1133cc",
                        font=("Segoe UI", 9, "bold"), relief="flat")
        style.map("L.Treeview",
                  background=[("selected", "#2255cc")],
                  foreground=[("selected", "white")])

        col_ids = [c[0] for c in self.LIST_COLUMNS]
        self._tree = ttk.Treeview(frame, columns=col_ids, show="headings",
                                  style="L.Treeview", selectmode="browse")
        for cid, heading, width, anchor in self.LIST_COLUMNS:
            self._tree.heading(cid, text=heading,
                               command=lambda c=cid: self._sort_by(c))
            self._tree.column(cid, width=width, anchor=anchor,
                              stretch=(cid == "opname"))

        vsb = ttk.Scrollbar(frame, orient="vertical",
                            command=self._on_tree_scroll)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        for tag, (fg, bg) in TAG_COLORS.items():
            self._tree.tag_configure(tag, foreground=fg, background=bg)

        self._tree.bind("<<TreeviewSelect>>", self._on_case_select)
        self._tree.bind("<Double-1>", self._on_case_select)
        self._tree.bind("<Up>", self._on_case_key_nav)
        self._tree.bind("<Down>", self._on_case_key_nav)
        self._tree.bind("<Prior>", self._on_tree_view_change)
        self._tree.bind("<Next>", self._on_tree_view_change)
        self._tree.bind("<Home>", self._on_tree_view_change)
        self._tree.bind("<End>", self._on_tree_view_change)
        self._tree.bind("<MouseWheel>", self._on_tree_view_change)
        self._tree.bind("<Button-4>", self._on_tree_view_change)
        self._tree.bind("<Button-5>", self._on_tree_view_change)
        self._tree.bind("<Configure>", self._on_tree_view_change)

        # List status
        self._list_status = tk.StringVar()
        tk.Label(parent, textvariable=self._list_status,
                 bg="#f0f0f7", fg="#888899",
                 font=("Segoe UI", 8), anchor="w").pack(fill="x", padx=8, pady=(0, 4))

    # -- Center panel: matplotlib canvas --------------------------------------

    def _build_canvas_panel(self, parent: tk.Frame):
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=0)
        parent.grid_columnconfigure(0, weight=1)

        self._canvas_host = tk.Frame(parent, bg="#ffffff")
        self._canvas_host.grid(row=0, column=0, sticky="nsew")
        self._canvas_host.grid_rowconfigure(0, weight=1)
        self._canvas_host.grid_columnconfigure(0, weight=1)

        # Placeholder shown before any case is selected
        self._placeholder = tk.Label(
            self._canvas_host,
            text="<- Select a case from the list",
            bg="#ffffff", fg="#aaaacc",
            font=("Segoe UI", 14),
        )
        self._placeholder.grid(row=0, column=0, sticky="nsew")

        # Figure (hidden until a case is loaded)
        self._fig = plt.Figure(facecolor="#f0f0f7")
        self._canvas = FigureCanvasTkAgg(self._fig, master=self._canvas_host)
        self._canvas_widget = self._canvas.get_tk_widget()
        # Not gridded yet - shown after first case load

        # Navigation bar below the canvas
        self._nav_frame = tk.Frame(parent, bg="#f0f0f7", height=52)
        self._nav_frame.grid(row=1, column=0, sticky="ew")
        self._nav_frame.grid_propagate(False)

    # -- Right panel: SBP/DBP search ------------------------------------------

    def _build_search_panel(self, parent: tk.Frame):
        top = tk.Frame(parent, bg="#f6f6fb")
        top.pack(fill="x", padx=10, pady=(10, 6))

        tk.Label(top, text="SBP/DBP Search", bg="#f6f6fb", fg="#222233",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(top, text="Current case only", bg="#f6f6fb", fg="#888899",
                 font=("Segoe UI", 8)).pack(anchor="w", pady=(2, 0))

        results = tk.Frame(parent, bg="#f6f6fb")
        results.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        self._bp_results = tk.Listbox(
            results,
            bg="#ffffff", fg="#222233",
            selectbackground="#2255cc", selectforeground="white",
            relief="flat", activestyle="none",
            font=("Consolas", 9),
        )
        self._bp_results.pack(side="left", fill="both", expand=True)
        self._bp_results.bind("<<ListboxSelect>>", self._on_bp_result_select)
        self._bp_results.bind("<Double-1>", self._on_bp_result_select)
        self._bp_results.bind("<Return>", self._on_bp_result_select)

        vsb = ttk.Scrollbar(results, orient="vertical",
                            command=self._bp_results.yview)
        self._bp_results.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")

        self._bp_status = tk.StringVar(value="Enter SBP and DBP below.")
        tk.Label(parent, textvariable=self._bp_status,
                 bg="#f6f6fb", fg="#888899",
                 font=("Segoe UI", 8), anchor="center", justify="center",
                 wraplength=self.SEARCH_W - 24
                 ).pack(fill="x", padx=10, pady=(0, 8))

        bottom = tk.Frame(parent, bg="#eef0f8")
        bottom.pack(fill="x", side="bottom", padx=10, pady=(0, 10))

        tk.Label(bottom, text="SBP", bg="#eef0f8", fg="#555577",
                 font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w", padx=(10, 6), pady=(10, 4))
        tk.Label(bottom, text="DBP", bg="#eef0f8", fg="#555577",
                 font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w", padx=(10, 6), pady=(4, 10))

        self._sbp_var = tk.StringVar()
        self._dbp_var = tk.StringVar()
        self._sbp_var.trace_add("write", lambda *_: self._run_bp_search())
        self._dbp_var.trace_add("write", lambda *_: self._run_bp_search())

        entry_kw = dict(bg="#ffffff", fg="#222233", insertbackground="#222233",
                        relief="flat", font=("Segoe UI", 9), width=10)
        tk.Entry(bottom, textvariable=self._sbp_var, **entry_kw).grid(
            row=0, column=1, sticky="ew", padx=(0, 10), pady=(10, 4)
        )
        tk.Entry(bottom, textvariable=self._dbp_var, **entry_kw).grid(
            row=1, column=1, sticky="ew", padx=(0, 10), pady=(4, 10)
        )
        bottom.grid_columnconfigure(1, weight=1)

    def _show_canvas(self):
        """Swap placeholder for the matplotlib canvas on first load."""
        if self._canvas_ready:
            return
        if self._placeholder.winfo_ismapped():
            self._placeholder.grid_remove()
        self._canvas_widget.grid(row=0, column=0, sticky="nsew")
        self._build_nav_bar(self._nav_frame)
        self._canvas_ready = True

    def _build_nav_bar(self, parent: tk.Frame):
        bc = "#d0d8f0"
        hc = "#3366cc"
        kw = dict(bg=bc, fg="#222233", activebackground=hc, activeforeground="white",
                  relief="flat", font=("Segoe UI", 9), padx=10, pady=3, cursor="hand2")

        bar = tk.Frame(parent, bg="#f0f0f7")
        bar.pack(fill="x", padx=6, pady=(2, 4))

        tk.Button(bar, text=f"<< {LARGE_STEP_SEC}s", command=lambda: self._shift(-LARGE_STEP_SEC),
                  **kw).pack(side="left", padx=2)
        tk.Button(bar, text=f"< {STEP_SEC}s", command=lambda: self._shift(
            -STEP_SEC), **kw).pack(side="left", padx=2)

        # Slider (canvas-less, pure tkinter)
        self._tk_slider = tk.Scale(
            bar, orient="horizontal", from_=0, to=1,
            resolution=1, showvalue=False,
            bg="#f0f0f7", fg="#888899", troughcolor="#ccccdd",
            highlightthickness=0, bd=0, sliderlength=14,
            command=self._on_tk_slider,
        )
        self._tk_slider.pack(side="left", fill="x", expand=True, padx=4)

        tk.Button(bar, text=f"{STEP_SEC}s >", command=lambda: self._shift(
            +STEP_SEC), **kw).pack(side="left", padx=2)
        tk.Button(bar, text=f"{LARGE_STEP_SEC}s >>", command=lambda: self._shift(+LARGE_STEP_SEC),
                  **kw).pack(side="left", padx=2)

        kw2 = dict(kw, bg="#d0dde8", activebackground="#3366cc")
        tk.Button(bar, text="Track Info", command=self._show_track_info,
                  **kw2).pack(side="right", padx=(8, 2))

        self._nav_time = tk.StringVar()
        tk.Label(bar, textvariable=self._nav_time,
                 bg="#f0f0f7", fg="#888899",
                 font=("Consolas", 9)).pack(side="right", padx=8)

        # Keyboard bindings on root
        self.root.bind("<Left>", lambda e: self._shift(-STEP_SEC))
        self.root.bind("<Right>", lambda e: self._shift(+STEP_SEC))
        self.root.bind("<Control-Left>", lambda e: self._shift(-LARGE_STEP_SEC))
        self.root.bind("<Control-Right>", lambda e: self._shift(+LARGE_STEP_SEC))

    # -- List management -------------------------------------------------------

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
        self._filtered_rows = rows

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

        arr = (" ^" if not self._sort_rev else " v")
        for cid, heading, *_ in self.LIST_COLUMNS:
            self._tree.heading(cid, text=heading +
                               (arr if cid == self._sort_col else ""))
        self._update_list_status()
        self.root.after_idle(self._queue_visible_track_scan)

    def _sort_by(self, col: str):
        self._sort_rev = (col == self._sort_col) and not self._sort_rev
        self._sort_col = col
        self._refresh_list()

    def _on_case_key_nav(self, event=None):
        self.root.after_idle(self._on_case_select)
        self.root.after_idle(self._queue_visible_track_scan)

    def _on_tree_scroll(self, *args):
        self._tree.yview(*args)
        self.root.after_idle(self._queue_visible_track_scan)

    def _on_tree_view_change(self, event=None):
        self.root.after_idle(self._queue_visible_track_scan)

    def _tag_from_flags(self, has_ppg: bool, has_abp: bool) -> str:
        if has_ppg and has_abp:
            return "ppg_abp"
        if has_ppg:
            return "ppg"
        if has_abp:
            return "abp"
        return "none"

    def _visible_tree_iids(self) -> list[str]:
        children = self._tree.get_children()
        if not children:
            return []

        first = self._tree.identify_row(0) or children[0]
        last = self._tree.identify_row(
            max(self._tree.winfo_height() - 1, 0)) or children[-1]

        try:
            i0 = children.index(first)
            i1 = children.index(last)
        except ValueError:
            return []
        if i0 > i1:
            i0, i1 = i1, i0
        return list(children[i0:i1 + 1])

    def _queue_visible_track_scan(self):
        for iid in reversed(self._visible_tree_iids()):
            row = self._row_by_path.get(Path(iid))
            if row is None or row["tag"] != "unknown":
                continue
            cid = row["case"]
            if cid < len(self._track_scanned) and self._track_scanned[cid]:
                continue
            self._scan_queue.put(row["path"])

    def _start_track_scan_worker(self):
        def worker():
            while not self._scan_stop.is_set():
                try:
                    path = self._scan_queue.get(timeout=0.2)
                except Empty:
                    continue
                if path is None:
                    break
                cid = int(path.stem) if path.stem.isdigit() else 0
                if cid < len(self._track_scanned) and self._track_scanned[cid]:
                    continue
                has_ppg, has_abp = scan_track_flags(path)
                self._scan_results.put((cid, path, has_ppg, has_abp))

        self._scan_thread = threading.Thread(
            target=worker, name="track-scan", daemon=True
        )
        self._scan_thread.start()

    def _drain_track_scan_results(self):
        updated = False
        while True:
            try:
                cid, path, has_ppg, has_abp = self._scan_results.get_nowait()
            except Empty:
                break

            row = self._row_by_path.get(path)
            if row is None:
                continue

            if cid < len(self._track_scanned):
                self._track_scanned[cid] = True
                self._track_has_ppg[cid] = has_ppg
                self._track_has_abp[cid] = has_abp
            tag = self._tag_from_flags(has_ppg, has_abp)
            row["tag"] = tag
            self.track_flags[row["case"]] = (has_ppg, has_abp)
            iid = str(path)
            if self._tree.exists(iid):
                self._tree.item(iid, tags=(tag,))
            updated = True

        if updated:
            self._update_list_status()
            self.root.after_idle(self._queue_visible_track_scan)

        if not self._scan_stop.is_set():
            try:
                self.root.after(120, self._drain_track_scan_results)
            except tk.TclError:
                pass

    def _update_list_status(self):
        rows = self._filtered_rows
        n_ppg = sum(1 for r in rows if r["tag"] in ("ppg", "ppg_abp"))
        n_abp = sum(1 for r in rows if r["tag"] in ("abp", "ppg_abp"))
        n_scanned = sum(1 for r in rows if r["tag"] != "unknown")
        self._list_status.set(
            f"{len(rows)}/{len(self._all_rows)}  scanned:{n_scanned}  PPG:{n_ppg}  ABP:{n_abp}"
        )

    # -- Case loading ----------------------------------------------------------

    def _on_case_select(self, event=None):
        sel = self._tree.selection()
        if not sel:
            return
        path = Path(sel[0])
        if self._vf is not None and path == getattr(self, "_current_path", None):
            return  # same case - no reload
        self._load_case(path)

    def _load_case(self, path: Path):
        self._set_status(f"Loading {path.name} ...")
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

        # Pre-decimate once so per-view QC queries are fast
        ppg_raw = data.get("SNUADC/PLETH")
        abp_raw = data.get("SNUADC/ART")
        factor = QC_SOURCE_HZ // QC_TARGET_HZ
        if ppg_raw is not None and abp_raw is not None:
            self._ppg_dec = ppg_raw[::factor].astype(np.float32)
            self._abp_dec = abp_raw[::factor].astype(np.float32)
        else:
            self._ppg_dec = None
            self._abp_dec = None
        self._qc_cache = {}

        self._show_canvas()
        self._rebuild_figure()
        self._run_bp_search()

        dur_m, dur_s = int(self._dur) // 60, int(self._dur) % 60
        self._set_status(
            f"Case {path.stem}  |  tracks: {', '.join(avail)}"
            f"  |  duration: {dur_m}m {dur_s}s"
            f"  |  Keys: <--> {STEP_SEC}s   Ctrl+<--> {LARGE_STEP_SEC}s"
        )

    # -- Figure construction ---------------------------------------------------

    def _rebuild_figure(self):
        """Reconstruct the matplotlib layout for the newly loaded case."""
        self._fig.clear()
        self._wave_axes = []
        self._num_ax = None
        self._qc_ax = None
        self._time_text = None

        avail_waves = [(n, l, c) for n, l, c in WAVE_TRACKS if n in self._data]
        avail_numeric = [(n, l, c)
                         for n, l, c in NUMERIC_TRACKS if n in self._data]
        self._avail_waves = avail_waves
        self._avail_numeric = avail_numeric

        self._has_ppg_abp = (
            "SNUADC/PLETH" in self._data and "SNUADC/ART" in self._data
        )

        n_wave = len(avail_waves)
        n_num = len(avail_numeric)
        add_qc = self._has_ppg_abp
        n_rows = n_wave + (1 if n_num else 0) + (1 if add_qc else 0)

        if n_rows == 0:
            return

        height_ratios = (
            [2] * n_wave
            + ([1.4] if n_num else [])
            + ([0.55] if add_qc else [])
        )
        gs = gridspec.GridSpec(
            n_rows, 1, figure=self._fig,
            hspace=0.06, top=0.94, bottom=0.06,
            height_ratios=height_ratios,
        )

        for i in range(n_wave):
            ax = self._fig.add_subplot(gs[i, 0])
            self._wave_axes.append(ax)

        next_row = n_wave
        if n_num:
            self._num_ax = self._fig.add_subplot(gs[next_row, 0])
            next_row += 1
        if add_qc:
            self._qc_ax = self._fig.add_subplot(gs[next_row, 0])
            if self._wave_axes:
                self._qc_ax.sharex(self._wave_axes[0])

        # Share x-axis among wave axes
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
        self._fig.suptitle(title, color="#222233", fontsize=10, y=0.99)

        # Update slider range
        if hasattr(self, "_tk_slider"):
            self._tk_slider.config(to=max(int(self._dur) - WINDOW_SEC, 1))
            self._tk_slider.set(0)

        self._draw()

    # -- Waveform drawing ------------------------------------------------------

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
            ax.set_facecolor("#ffffff")
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
                        ha="center", va="center", color="#888899", fontsize=9)
            ax.set_xlim(self._t0, t_end)
            ax.set_ylabel(label, color=color, fontsize=8, labelpad=2)
            ax.tick_params(colors="#888899", labelsize=7)
            ax.spines[:].set_color("#ccccdd")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Hide x-tick labels on all waveform panels except the last
        # (or all of them when the QC strip is present, which carries the x-axis)
        hide_wave_xlabel = self._qc_ax is not None
        for ax in self._wave_axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        if self._wave_axes:
            if hide_wave_xlabel:
                plt.setp(self._wave_axes[-1].get_xticklabels(), visible=False)
                self._wave_axes[-1].set_xlabel("")
            else:
                self._wave_axes[-1].set_xlabel("Time (s)",
                                               color="#888899", fontsize=8)

        # Dataset-v2 QC segment overlays on wave axes
        self._draw_qc_overlay()

        if self._num_ax:
            ax = self._num_ax
            ax.cla()
            ax.set_facecolor("#ffffff")

            for name, label, color in self._avail_numeric:
                t, y = self._num_slice(name)
                valid = ~np.isnan(y)
                if valid.any():
                    ls = "--" if name in _NIBP_TRACKS else "-"
                    ax.plot(t[valid], y[valid], color=color,
                            lw=1.1, linestyle=ls, marker=".", ms=3, label=label)
                    if name in _LABEL_TRACKS:
                        va = _LABEL_VA[name]
                        for xi, yi in zip(t[valid], y[valid]):
                            ax.text(xi, yi, str(int(round(yi))),
                                    color=color, fontsize=6,
                                    ha="center", va=va, clip_on=True)

            # Expand y-limits to give the value labels room to breathe
            y_lo, y_hi = ax.get_ylim()
            pad = (y_hi - y_lo) * 0.12
            ax.set_ylim(y_lo - pad, y_hi + pad)

            ax.set_xlim(self._t0, t_end)
            ax.set_ylabel("SBP / DBP / MBP (mmHg)",
                          color="#222233", fontsize=8)
            ax.set_xlabel("Time (s)", color="#888899", fontsize=8)
            ax.legend(loc="upper right", fontsize=7, framealpha=0.3,
                      labelcolor="#222233", facecolor="#ffffff")
            ax.tick_params(colors="#888899", labelsize=7)
            ax.spines[:].set_color("#ccccdd")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if self._qc_ax is not None:
            self._draw_qc_strip()

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

    # -- BP search ------------------------------------------------------------

    def _parse_bp_value(self, text: str) -> int | None:
        text = text.strip()
        if not text:
            return None
        return int(round(float(text)))

    def _bp_values_are_integral(self, values: np.ndarray) -> bool:
        valid = values[~np.isnan(values)]
        return bool(valid.size == 0 or np.allclose(valid, np.round(valid)))

    def _format_time_row(self, sec: int, sbp: int, dbp: int) -> str:
        return f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d} ({sec}s)   {sbp:>3}/{dbp:<3}"

    def _run_bp_search(self):
        if not hasattr(self, "_bp_results"):
            return

        self._bp_results.delete(0, tk.END)
        self._bp_match_times = []
        self._current_bp_mode = "exact"

        try:
            sbp = self._parse_bp_value(self._sbp_var.get())
            dbp = self._parse_bp_value(self._dbp_var.get())
        except ValueError:
            self._bp_status.set("SBP and DBP must be numeric.")
            return

        if sbp is None or dbp is None:
            self._bp_status.set("Enter SBP and DBP below.")
            return

        if self._data is None:
            self._bp_status.set("Load a case to search within it.")
            return

        if "Solar8000/ART_SBP" not in self._data or "Solar8000/ART_DBP" not in self._data:
            self._bp_status.set(
                "This case does not contain numeric SBP/DBP tracks.")
            return

        sbp_arr = self._data["Solar8000/ART_SBP"]
        dbp_arr = self._data["Solar8000/ART_DBP"]
        n = min(len(sbp_arr), len(dbp_arr))
        sbp_arr = sbp_arr[:n]
        dbp_arr = dbp_arr[:n]

        sbp_integral = self._bp_values_are_integral(sbp_arr)
        dbp_integral = self._bp_values_are_integral(dbp_arr)
        self._current_bp_mode = "exact" if sbp_integral and dbp_integral else "rounded"

        if self._current_bp_mode == "exact":
            match_mask = (sbp_arr == sbp) & (dbp_arr == dbp)
        else:
            match_mask = (np.round(sbp_arr) == sbp) & (
                np.round(dbp_arr) == dbp)
        match_mask &= ~np.isnan(sbp_arr) & ~np.isnan(dbp_arr)

        matches = np.flatnonzero(match_mask)
        self._bp_match_times = matches.astype(int).tolist()
        for t in self._bp_match_times:
            self._bp_results.insert(tk.END, self._format_time_row(t, sbp, dbp))

        self._bp_status.set(
            f"{len(self._bp_match_times)} matches for SBP {sbp} / DBP {dbp} ({self._current_bp_mode})"
        )

    def _on_bp_result_select(self, event=None):
        sel = self._bp_results.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self._bp_match_times):
            self._set_time(self._bp_match_times[idx], center=True)

    # -- Navigation ------------------------------------------------------------

    def _set_time(self, t0: float, *, center: bool = False):
        if self._data is None:
            return

        if center:
            t0 = float(t0) - (WINDOW_SEC / 2)
        self._t0 = float(np.clip(t0, 0, max(self._dur - WINDOW_SEC, 0)))
        if hasattr(self, "_tk_slider"):
            self._tk_slider.set(int(self._t0))
        self._draw()

    def _shift(self, delta: float):
        if self._data is None:
            return
        self._set_time(self._t0 + delta)

    def _on_tk_slider(self, val):
        if self._data is None:
            return
        self._t0 = float(val)
        self._draw()

    # -- Track info window -----------------------------------------------------

    def _show_track_info(self):
        if self._vf is None:
            return

        win = tk.Toplevel(self.root)
        win.title(f"Track Info - Case {self._current_path.stem}")
        win.configure(bg="#f0f0f7")
        win.geometry("620x460")

        cols = [("track", "Track Name", 255, "w"), ("type", "Type", 88, "center"),
                ("unit", "Unit", 68, "center"), ("recs", "Records", 68, "center")]
        frame = tk.Frame(win, bg="#f0f0f7")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        style = ttk.Style()
        style.configure("TI.Treeview",
                        background="#ffffff", foreground="#222233",
                        fieldbackground="#ffffff", rowheight=21,
                        font=("Consolas", 9))
        style.configure("TI.Treeview.Heading",
                        background="#d0d8f0", foreground="#1133cc",
                        font=("Segoe UI", 9, "bold"), relief="flat")
        style.map("TI.Treeview",
                  background=[("selected", "#2255cc")],
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

        tree.tag_configure("even", background="#ffffff")
        tree.tag_configure("odd",  background="#e8e8f2")

        bot = tk.Frame(win, bg="#f0f0f7")
        bot.pack(fill="x", padx=10, pady=(0, 10))
        n_w = sum(1 for t in self._vf.trks.values() if t.srate > 0)
        n_n = sum(1 for t in self._vf.trks.values() if t.srate == 0)
        tk.Label(bot, text=f"{len(self._vf.trks)} tracks - {n_w} waveform, {n_n} numeric",
                 bg="#f0f0f7", fg="#888899", font=("Segoe UI", 9)).pack(side="left")
        tk.Button(bot, text="Close", command=win.destroy,
                  bg="#d0d8f0", fg="#222233", activebackground="#3366cc",
                  activeforeground="white", relief="flat",
                  font=("Segoe UI", 9), padx=12, pady=3, cursor="hand2"
                  ).pack(side="right")

    # -- Dataset-v2 QC --------------------------------------------------------

    def _qc_window_results(self) -> list[QCResult]:
        """Return QC results for segments overlapping the current view, using cache."""
        if not self._has_ppg_abp or self._ppg_dec is None or self._abp_dec is None:
            return []
        return compute_window_qc(
            self._ppg_dec,
            self._abp_dec,
            t0=self._t0,
            t1=self._t0 + WINDOW_SEC,
            cache=self._qc_cache,
        )

    def _draw_qc_overlay(self):
        """Overlay per-segment QC spans on every wave axis."""
        if not self._wave_axes:
            return
        t_end = self._t0 + WINDOW_SEC
        for qr in self._qc_window_results():
            ts = qr.t_start
            te = ts + QC_SEGMENT_SEC
            color = "#22bb66" if qr.passed else "#ee3311"
            alpha = 0.07 if qr.passed else 0.15
            for ax in self._wave_axes:
                ax.axvspan(ts, te, alpha=alpha, color=color,
                           linewidth=0, zorder=0)
            if not qr.passed:
                lx = max(ts, self._t0)
                self._wave_axes[0].text(
                    lx, 0.97, f"R{qr.failed_rule}",
                    transform=self._wave_axes[0].get_xaxis_transform(),
                    ha="left", va="top",
                    color="#cc1100", fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="#cc1100", alpha=0.75),
                    clip_on=True, zorder=5,
                )

    def _draw_qc_strip(self):
        """Draw the compact v2-QC strip at the bottom of the figure."""
        ax = self._qc_ax
        if ax is None:
            return
        ax.cla()
        ax.set_facecolor("#f0f0f6")

        t_end = self._t0 + WINDOW_SEC
        ax.set_xlim(self._t0, t_end)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel("v2 QC", color="#888899", fontsize=7,
                      rotation=0, labelpad=30, va="center")
        ax.set_xlabel("Time (s)", color="#888899", fontsize=8)
        ax.tick_params(colors="#888899", labelsize=7)
        ax.spines[:].set_color("#ccccdd")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        n_pass = n_fail = 0
        for qr in self._qc_window_results():
            ts = qr.t_start
            te = ts + QC_SEGMENT_SEC
            color = "#22bb66" if qr.passed else "#ee3311"
            ax.axvspan(ts, te, alpha=0.80, color=color,
                       linewidth=0.5, edgecolor="white")
            if qr.passed:
                lx = (max(ts, self._t0) + min(te, t_end)) / 2
                ha = "center"
            else:
                lx = max(ts, self._t0)
                ha = "left"
            lbl = "✓" if qr.passed else f"R{qr.failed_rule}"
            ax.text(lx, 0.5, lbl,
                    transform=ax.get_xaxis_transform(),
                    ha=ha, va="center",
                    color="white", fontsize=7, fontweight="bold",
                    clip_on=True)
            if qr.passed:
                n_pass += 1
            else:
                n_fail += 1

        # Right-side summary
        total = n_pass + n_fail
        if total:
            summary = f"  {n_pass}/{total} pass"
            ax.text(0.995, 0.5, summary,
                    transform=ax.transAxes,
                    ha="right", va="center",
                    color="#444455", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="#f0f0f6", edgecolor="none", alpha=0.85))

    # -- Helpers ---------------------------------------------------------------

    def _set_status(self, text: str):
        self._status_var.set(text)
        self.root.update_idletasks()

    def _on_close(self):
        self._scan_stop.set()
        self._scan_queue.put(None)
        self.root.destroy()


# -- CLI -----------------------------------------------------------------------

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

    root = tk.Tk()
    app = VitalDBBrowser(root, data_dir, files, ci_map)

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
