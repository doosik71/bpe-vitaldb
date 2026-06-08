@echo off
rem Print layer structure and output shapes for BPE model architectures.
rem
rem Usage:
rem   bin\print-model.bat                     print all models
rem   bin\print-model.bat --model resnet1d    print one model
rem   bin\print-model.bat --model all         print all models
rem
rem Model names:
rem   resnet1d  st_resnet  minception  xresnet1d
rem
rem Options forwarded to scripts/print-model.py:
rem   --model <name|all>   Model to display   (default: all)
rem   --input-length N     PPG segment length  (default: 1000)
rem   --batch-size N       Dummy batch size    (default: 1)
rem   --device cpu|cuda    Compute device      (default: cpu)

cd /d "%~dp0.."
uv run python scripts\print-model.py %*
