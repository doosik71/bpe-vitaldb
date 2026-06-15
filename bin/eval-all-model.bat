@echo off
rem Evaluate every registered BPE model on the held-out test set.
rem
rem Enumerates all models from bpe.models.list_models(), skips any model that
rem lacks a best.pt checkpoint, then dispatches eval-model.py once per model.
rem Runs one job per CUDA device in parallel; falls back to sequential CPU runs.
rem
rem Usage:
rem   bin\eval-all-model.bat [OPTIONS]
rem   bin\eval-all-model.bat --models-dir data\models-v1
rem   bin\eval-all-model.bat --batch-size 256 --no-normalize
rem
rem Options:
rem   --models-dir <path>    Root directory of trained models  (default: data\models)
rem   --poll-sec <secs>      Scheduler polling interval        (default: 2.0)
rem   --dry-run              Print commands without launching them
rem   --dataset-dir <path>   Forwarded to eval-model.py        (default: data\dataset)
rem   --batch-size <N>       Forwarded to eval-model.py        (default: 512)
rem   --no-normalize         Forwarded to eval-model.py

cd /d "%~dp0.."
uv run python scripts\eval-all-model.py %*
