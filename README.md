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
Phase 3  Model development    ✅  design & train PyTorch architecture
Phase 4  Evaluation           ✅  benchmark against published BPE methods
Phase 5  Analysis                 clinical validation, error analysis
```

## Repository Layout

```text
bpe-vitaldb/
├── bin/                            # Launcher scripts (Windows .bat + POSIX sh)
│   ├── download-vitaldb.bat        # run scripts/download-vitaldb.py
│   ├── vitaldb-browser.bat         # run scripts/vitaldb-browser.py
│   ├── construct-dataset.bat       # run scripts/construct-dataset.py
│   ├── dataset-browser.bat         # run scripts/dataset-browser.py
│   ├── share-data.bat              # run scripts/share-data.py (HTTP server)
│   ├── download-shared-data.bat    # download data/ from a remote share-data host
│   ├── print-model.bat             # run scripts/print-model.py
│   ├── train-model.bat             # run scripts/train-model.py
│   ├── train-status.bat            # run scripts/train-status.py
│   └── eval-model.bat              # run scripts/eval-model.py
├── scripts/
│   ├── download-vitaldb.py         # parallel .vital file downloader
│   ├── vitaldb-browser.py          # GUI waveform browser (tkinter + matplotlib)
│   ├── construct-dataset.py        # build train/val/test NPZ datasets
│   ├── dataset-browser.py          # GUI dataset segment browser
│   ├── share-data.py               # multi-threaded HTTP file server
│   ├── print-model.py              # layer structure and output shape inspector
│   ├── train-model.py              # model training pipeline
│   ├── train-status.py             # plot training metrics from a run directory
│   └── eval-model.py               # evaluate best.pt on the test split
├── data/
│   ├── vitaldb/                    # downloaded .vital files (git-ignored)
│   ├── dataset/                    # NPZ segment files (git-ignored)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── models/                     # training checkpoints & metrics (git-ignored)
├── AGENTS.md                       # contribution rules for AI agents
├── pyproject.toml                  # uv project configuration
└── README.md
```

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

| Option          | Default        | Description                        |
| --------------- | -------------- | ---------------------------------- |
| `--data-dir`    | `data/vitaldb` | Source directory of `.vital` files |
| `--output-dir`  | `data/dataset` | Root output directory              |
| `--split`       | `0.6 0.2 0.2`  | Train / val / test case ratios     |
| `--target-hz`   | `125`          | Output PPG sample rate (Hz)        |
| `--segment-sec` | `8`            | Window duration in seconds         |
| `--seed`        | `42`           | Random seed for case shuffling     |

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

| Keyboard  | Action                  |
| --------- | ----------------------- |
| `←` / `→` | Previous / next segment |
| `↑` / `↓` | Previous / next case    |

```bash
# if the dataset was built with a non-default sample rate:
bin\dataset-browser.bat --target-hz 250
```

### 6. Share data between machines

`share-data` serves the entire `data/` folder over HTTP so that another
machine on the same network can pull it down without needing SSH, cloud
storage, or physical drives.

**On the machine that has the data (server):**

```bash
bin\share-data.bat          # Windows — listens on port 8888
bin/share-data              # Linux / macOS
# optional: specify a different port
bin\share-data.bat 9000
bin/share-data 9000
```

The LAN address is printed on startup, e.g. `http://192.168.1.10:8888/`.

| Option       | Default   | Description           |
| ------------ | --------- | --------------------- |
| `--port`     | `8888`    | TCP port to listen on |
| `--bind`     | `0.0.0.0` | Bind address          |
| `--data-dir` | `data`    | Directory to serve    |

**On the machine that needs the data (client):**

```bash
bin\download-shared-data.bat 192.168.1.10          # Windows
bin/download-shared-data     192.168.1.10          # Linux / macOS
# with a non-default port:
bin\download-shared-data.bat 192.168.1.10 9000
```

Files are written to `data/` in the project root, mirroring the remote
directory tree.  Re-running the command resumes an interrupted transfer;
already-complete files are skipped automatically.

> **Windows prerequisite:** `wget.exe` must be on `PATH`.  
> Install with `winget install GnuWin32.Wget` or download from
> [eternallybored.org/misc/wget](https://eternallybored.org/misc/wget/).

### 7. Inspect model architecture

Print every layer, its output shape, and its parameter count for any registered
model.  A single forward pass with a dummy input is run internally, so the
output shapes reflect the actual tensor dimensions at each layer.

```bash
bin\print-model.bat                          # Windows — print all models
bin\print-model.bat --model resnet1d         # one model only
bin/print-model                              # Linux / macOS — print all models
bin/print-model     --model st_resnet
```

Example output (truncated):

```text
============================================================================================
  Model: resnet1d
============================================================================================
Layer (name)                    Type                Output shape        Params
--------------------------------------------------------------------------------------------
stem                            Sequential          (1, 32, 250)
stem.0                          ConvBnAct1d         (1, 32, 500)
stem.0.0                        Conv1d              (1, 32, 500)           480
stem.0.1                        BatchNorm1d         (1, 32, 500)            64
...
head.fc                         Linear              (1, 2)                 514
--------------------------------------------------------------------------------------------
  Total params    : 2,184,866  (2.18 M)
  Trainable params: 2,184,866  (2.18 M)
  Input shape     : (1, 1000)
```

| Option           | Default | Description                                  |
| ---------------- | ------- | -------------------------------------------- |
| `--model`        | `all`   | Model name or `all`                          |
| `--input-length` | `1000`  | PPG segment length in samples (8 s @ 125 Hz) |
| `--batch-size`   | `1`     | Batch size for the dummy forward pass        |
| `--device`       | `cpu`   | `cpu` \| `cuda` \| `auto`                    |

### 8. Train a model

Run the training pipeline against the NPZ dataset built in step 4:

```bash
bin\train-model.bat --model resnet1d              # Windows
bin/train-model     --model resnet1d              # Linux / macOS
```

Available model architectures:

| Model name          | Description                                                             | Layers |  Params |
| ------------------- | ----------------------------------------------------------------------- | -----: | ------: |
| `resnet1d`          | 1D ResNet — lightweight, fast baseline                                  |    100 |  2.18 M |
| `resnet1d_mini`     | ResNet1D 50 % depth (4 stages × 1 block)                                |     60 | 964.4 K |
| `resnet1d_tiny`     | ResNet1D 25 % depth (2 stages × 1 block)                                |     34 |  60.6 K |
| `resnet1d_micro`    | ResNet1D ~10 % depth (1 stage × 1 block)                                |     21 |  15.1 K |
| `st_resnet`         | Spectro-Temporal ResNet (PPG + VPG + APG branches)                      |    140 | 478.9 K |
| `minception`        | Multi-scale Inception 1D CNN                                            |    134 | 440.7 K |
| `xresnet1d`         | Deep XResNet-101-style 1D CNN                                           |    484 |  9.47 M |
| `mtae`              | Multi-Task AutoEncoder (reconstruction + BP head)                       |     37 | 119.5 K |
| `mtae_tr`           | MTAE with Transformer encoder/decoder (MAE-style)                       |     93 | 109.4 K |
| `pulsewoq_resnet1d` | Overlapping-segment ResNet with explicit quality supervision and output |     38 |  30.1 K |
| `acfa`              | ACFA: DyCASNet + xLSTM + Transformer + FKAN (Li et al., 2026)           |    108 | 542.6 K |
| `cnn_bilstm_at`     | CNN–BiLSTM with additive attention (Mohammadi et al., 2025)             |     17 | 691.3 K |

> Layers = total named modules (forward hooks); Params = trainable parameters.  
> Input: PPG segment (1, 1000) — 8 s @ 125 Hz.

Common usage examples:

```bash
# default hyperparameters
bin\train-model.bat --model resnet1d

# longer run with larger batches
bin\train-model.bat --model st_resnet --epochs 150 --batch-size 512

# custom learning rate and early-stopping patience
bin\train-model.bat --model minception --lr 5e-4 --patience 20

# resume from a previous checkpoint
bin\train-model.bat --model resnet1d --resume data\models\resnet1d\last.pt
```

Checkpoints and a metrics CSV are saved under
`data/models/<model>/`.

| Option           | Default        | Description                                |
| ---------------- | -------------- | ------------------------------------------ |
| `--model`        | *(required)*   | Model name from the registry               |
| `--dataset-dir`  | `data/dataset` | Root dataset directory                     |
| `--output-dir`   | `data/models`  | Root directory for saved runs              |
| `--epochs`       | `100`          | Maximum training epochs                    |
| `--batch-size`   | `256`          | Mini-batch size                            |
| `--lr`           | `1e-3`         | Initial learning rate                      |
| `--weight-decay` | `1e-4`         | AdamW weight decay                         |
| `--patience`     | `15`           | Early-stopping patience (val loss)         |
| `--seed`         | `42`           | Random seed                                |
| `--device`       | `auto`         | `auto` \| `cpu` \| `cuda` \| `cuda:N`      |
| `--workers`      | `4`            | DataLoader worker processes                |
| `--preload`      | off            | Load all segments into RAM before training |
| `--no-normalize` | off            | Skip per-segment z-score normalization     |
| `--resume`       | —              | Path to a checkpoint `.pt` to resume from  |

Run `bin\train-model.bat --help` for the full option listing.

### 9. Check training status

Plot loss and MAE curves for any run while training is in progress or after it completes:

```bash
bin\train-status.bat data\models\resnet1d   # Windows
bin/train-status     data/models/resnet1d   # Linux / macOS
```

Two PNG files are written next to `metrics.csv` inside the run directory:

| File             | Contents                                                                 |
| ---------------- | ------------------------------------------------------------------------ |
| `loss_graph.png` | `train_loss` vs `val_loss` per epoch                                     |
| `mae_graph.png`  | `train_sbp_mae`, `train_dbp_mae`, `val_sbp_mae`, `val_dbp_mae` per epoch |

A summary table is also printed to the terminal.

| Option      | Default | Description                                |
| ----------- | ------- | ------------------------------------------ |
| `--no-save` | off     | Print summary only; skip writing PNG files |

### 10. Evaluate a model

Run the trained model on the held-out test split and compute clinical metrics:

```bash
bin\eval-model.bat data\models\resnet1d   # Windows
bin/eval-model     data/models/resnet1d   # Linux / macOS
```

Three result files are saved in the run directory:

| File                | Contents                                                      |
| ------------------- | ------------------------------------------------------------- |
| `eval_results.json` | MAE, RMSE, ME, SD; BHS cumulative error grade; AAMI pass/fail |
| `eval_plot.png`     | Predicted vs actual scatter plots for SBP and DBP             |
| `error_hist.png`    | Error distribution histograms for SBP and DBP                 |

| Option           | Default        | Description                            |
| ---------------- | -------------- | -------------------------------------- |
| `--dataset-dir`  | `data/dataset` | Root dataset directory                 |
| `--device`       | `auto`         | `auto` \| `cpu` \| `cuda` \| `cuda:N`  |
| `--batch-size`   | `512`          | Inference batch size                   |
| `--no-normalize` | off            | Skip per-segment z-score normalization |

## Signals of Interest

| Signal               | Track                   | Rate   | Role                      |
| -------------------- | ----------------------- | ------ | ------------------------- |
| PPG                  | `SNUADC/PLETH`          | 500 Hz | Model **input**           |
| Arterial BP waveform | `SNUADC/ART`            | 500 Hz | Ground truth (continuous) |
| SBP / DBP            | `Solar8000/ART_SBP/DBP` | ~1 Hz  | Ground truth (numeric)    |
| ECG II               | `SNUADC/ECG_II`         | 500 Hz | Auxiliary / quality check |

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
