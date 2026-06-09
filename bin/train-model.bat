@echo off
rem Train a BPE model on the VitalDB NPZ dataset.
rem
rem Usage:
rem   bin\train-model.bat --model <name> [OPTIONS]
rem
rem Model names:
rem   resnet1d            1D ResNet (light, fast baseline)
rem   st_resnet           Spectro-Temporal ResNet  (PPG + VPG + APG branches)
rem   minception          Multi-scale Inception 1D CNN
rem   xresnet1d           Deep XResNet-101-style 1D CNN
rem
rem Examples:
rem   bin\train-model.bat --model resnet1d
rem   bin\train-model.bat --model st_resnet --epochs 150 --batch-size 512
rem   bin\train-model.bat --model minception --lr 5e-4 --patience 20
rem   bin\train-model.bat --model xresnet1d --batch-size 128 --workers 2
rem   bin\train-model.bat --model resnet1d --resume data\models\resnet1d\last.pt
rem
rem All options are forwarded to scripts/train-model.py.
rem Run "bin\train-model.bat --help" for a full option listing.

cd /d "%~dp0.."
uv run python scripts\train-model.py %*
