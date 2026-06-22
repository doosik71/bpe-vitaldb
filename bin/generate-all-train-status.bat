@echo off
rem Generate training status graphs for all registered BPE models.
rem
rem For each model in bpe.models.list_models(), runs generate-train-status.py
rem if metrics.csv exists. Models without metrics.csv are skipped.
rem
rem Usage:
rem   bin\generate-all-train-status.bat
rem   bin\generate-all-train-status.bat --models-dir data\models-v1
rem   bin\generate-all-train-status.bat --dry-run
rem
rem Options:
rem   --models-dir <path>    Root directory of trained models  (default: data\models)
rem   --dry-run              Print commands without executing them
rem   --no-save              Forwarded to generate-train-status.py; print only, no PNG

cd /d "%~dp0.."
uv run python scripts\generate-all-train-status.py %*
