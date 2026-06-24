@echo off
rem Print layer structure for all registered BPE models.
rem
rem For each model in bpe.models.list_models(), runs print-model.py and saves
rem the output to <models-dir>\<model>\struct.txt.
rem
rem Usage:
rem   bin\print-all-model.bat
rem   bin\print-all-model.bat --models-dir data\models-v1
rem   bin\print-all-model.bat --dry-run
rem
rem Options:
rem   --models-dir <path>    Root directory for struct.txt outputs  (default: data\models)
rem   --dry-run              Print commands without executing them
rem   --input-length N       Forwarded to print-model.py            (default: 1000)
rem   --batch-size N         Forwarded to print-model.py            (default: 1)
rem   --device cpu|cuda      Forwarded to print-model.py            (default: cpu)

cd /d "%~dp0.."
uv run python scripts\print-all-model.py %*
