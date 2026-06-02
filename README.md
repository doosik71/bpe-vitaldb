# bpe-vitaldb (Blood Pressure Estimation from PPG using VitalDB)

A PyTorch deep learning project that estimates arterial blood pressure (ABP)
from photoplethysmography (PPG) waveforms, trained on the publicly available
[VitalDB](https://vitaldb.net) intraoperative biosignal dataset.

## Project Goal

Develop a deep learning model that takes a raw PPG waveform segment as input
and predicts continuous blood pressure values (SBP / DBP) as output,
without requiring invasive arterial-line measurement at inference time.

```text
PPG waveform (125 Hz)  ──►  [ Deep Learning Model ]  ──►  SBP / DBP (mmHg)
```

## Dataset — VitalDB

[VitalDB](https://vitaldb.net) is the world's largest open intraoperative
biosignal dataset, built by the VitalLab research group at Seoul National
University Hospital (Prof. Chul-Woo Jung, Prof. Hyung-Chul Lee et al.).

| Item              | Detail                                                       |
| ----------------- | ------------------------------------------------------------ |
| Total cases       | 6,388 surgical patients                                      |
| Waveform format   | `.vital` (packed binary, all tracks aligned)                 |
| PPG track         | `SNUADC/PLETH` — 500 Hz                                      |
| ABP waveform      | `SNUADC/ART` — 500 Hz (invasive radial arterial line)        |
| Numeric BP        | `Solar8000/ART_SBP`, `ART_DBP`, `ART_MBP` — ~1 Hz            |
| Clinical metadata | Age, sex, height, weight, operation name, anesthesia type, … |

Cases that contain **both** PPG and invasive ABP waveforms (~3,000 cases)
form the core training dataset for this project.

## Project Roadmap

```text
Phase 1  Data acquisition     ✅  download & browse VitalDB .vital files
Phase 2  Preprocessing        ✅  segment, filter, align PPG ↔ ABP signals
Phase 3  Model development        design & train PyTorch architecture
Phase 4  Evaluation               benchmark against published BPE methods
Phase 5  Analysis                 clinical validation, error analysis
```

## Repository Layout

```text
bpe-vitaldb/
├── bin/                            # Launcher scripts (Windows .bat + POSIX sh)
│   ├── download-vitaldb.bat        # run scripts/download-vitaldb.py
│   ├── vitaldb-browser.bat         # run scripts/vitaldb-browser.py
│   ├── construct-dataset.bat       # run scripts/construct-dataset.py
│   └── dataset-browser.bat         # run scripts/dataset-browser.py
├── scripts/
│   ├── download-vitaldb.py         # parallel .vital file downloader
│   ├── vitaldb-browser.py          # GUI waveform browser (tkinter + matplotlib)
│   ├── construct-dataset.py        # build train/val/test NPZ datasets
│   └── dataset-browser.py          # GUI dataset segment browser
├── data/
│   ├── vitaldb/                    # downloaded .vital files (git-ignored)
│   └── dataset/                    # NPZ segment files (git-ignored)
│       ├── train/
│       ├── val/
│       └── test/
├── AGENTS.md                       # contribution rules for AI agents
├── pyproject.toml                  # uv project configuration
└── README.md
```

> **Planned additions**
> `scripts/train.py` · `scripts/evaluate.py`
> `src/model.py` · `src/metrics.py`

## Getting Started

### 1. Install dependencies

```bash
# requires uv  (https://docs.astral.sh/uv/)
uv sync
```

### 2. Download VitalDB data

Download all 6,388 cases (full dataset, ~200 GB):

```bash
bin\download-vitaldb.bat
```

Download a small subset for exploration:

```bash
bin\download-vitaldb.bat --max-cases 50
```

Only cases that have both PPG and ABP tracks:

```bash
bin\download-vitaldb.bat --filter-tracks
```

| Option                        | Default        | Description                                     |
| ----------------------------- | -------------- | ----------------------------------------------- |
| `--output-dir`                | `data/vitaldb` | Download destination                            |
| `--max-cases`                 | all            | Limit number of cases                           |
| `--start-case` / `--end-case` | 1 / 6388       | Case ID range                                   |
| `--workers`                   | 4              | Parallel download threads                       |
| `--no-resume`                 | off            | Re-download existing files                      |
| `--filter-tracks`             | off            | Only PPG + ABP cases (uses deprecated trks API) |

### 3. Browse raw waveforms

```bash
bin\vitaldb-browser.bat
# or open a specific case directly:
bin\vitaldb-browser.bat --case 1
```

The browser shows a **unified single window**:

- **Left panel** — sortable case list; rows highlighted by available signals
  - Green text → PPG present
  - Crimson background → invasive ABP present
- **Right panel** — live waveform canvas (PPG, ABP, ECG II, numeric BP/HR)
- **Navigation** — slider, buttons, or keyboard (`←` / `→` 10 s, `Ctrl+←/→` 60 s)
- **Track Info** button — lists all tracks in the loaded case

### 4. Build the dataset

Segment the downloaded `.vital` files into train / val / test NPZ files:

```bash
bin\construct-dataset.bat
```

Each `.vital` case is resampled to 125 Hz, sliced into 8-second windows with
50 % overlap (4-second stride), and saved as `data/dataset/{split}/{caseid}.npz`.
Windows that contain NaN values or physiologically implausible BP readings are
discarded.  Cases are split at the **case level** (not segment level) to prevent
data leakage.

| Option            | Default          | Description                               |
| ----------------- | ---------------- | ----------------------------------------- |
| `--data-dir`      | `data/vitaldb`   | Source directory of `.vital` files        |
| `--output-dir`    | `data/dataset`   | Root output directory                     |
| `--split`         | `0.6 0.2 0.2`    | Train / val / test case ratios            |
| `--target-hz`     | `125`            | Output PPG sample rate (Hz)               |
| `--segment-sec`   | `8`              | Window duration in seconds                |
| `--seed`          | `42`             | Random seed for case shuffling            |

Each output `.npz` contains:

```text
x  float32  (N, segment_samples)   PPG segments
y  float32  (N, 2)                 [SBP_mean, DBP_mean] in mmHg
```

### 5. Browse dataset segments

Inspect the preprocessed NPZ segments in a GUI:

```bash
bin\dataset-browser.bat
```

The browser shows a **unified single window**:

- **Left panel** — split selector (`Train` / `Val` / `Test`) + sortable case
  list with case ID, segment count, and file size; metadata loads in the
  background so the UI is immediately responsive
- **Right panel** — PPG waveform plot for the selected segment; SBP and DBP
  values shown both in the top info bar and as annotated boxes on the graph
- **Navigation** — `◀ Prev` / `Next ▶` buttons, a slider for fast scrubbing,
  a jump-to-segment entry field, and keyboard shortcuts

| Keyboard     | Action                   |
| ------------ | ------------------------ |
| `←` / `→`    | Previous / next segment  |
| `↑` / `↓`    | Previous / next case     |

```bash
# if the dataset was built with a non-default sample rate:
bin\dataset-browser.bat --target-hz 250
```

## Signals of Interest

| Signal               | Track                       | Rate   | Role                      |
| -------------------- | --------------------------- | ------ | ------------------------- |
| PPG                  | `SNUADC/PLETH`              | 500 Hz | Model **input**           |
| Arterial BP waveform | `SNUADC/ART`                | 500 Hz | Ground truth (continuous) |
| SBP / DBP            | `Solar8000/ART_SBP/DBP`     | ~1 Hz  | Ground truth (numeric)    |
| ECG II               | `SNUADC/ECG_II`             | 500 Hz | Auxiliary / quality check |

## Environment

| Tool             | Version                                             |
| ---------------- | --------------------------------------------------- |
| Python           | ≥ 3.13                                              |
| Package manager  | [uv](https://docs.astral.sh/uv/)                    |
| Key dependencies | `vitaldb`, `torch` (planned), `numpy`, `matplotlib` |

> **Do not use `pip install`.**  
> All dependency management must go through `uv`. See `AGENTS.md`.

## References

- Kachuee, M. et al. (2017). *Cuffless Blood Pressure Estimation Algorithms for Continuous Health-Care Monitoring.* IEEE TBME.
- Slapničar, G. et al. (2019). *Blood Pressure Estimation from Photoplethysmogram Using a Spectro-Temporal Deep Neural Network.* Sensors.
- Lee, H.-C. et al. (2022). *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients.* Scientific Data.
