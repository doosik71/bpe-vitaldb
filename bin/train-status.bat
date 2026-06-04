@echo off
rem Show training progress for a BPE model run.
rem
rem Reads metrics.csv from the given run directory and writes:
rem   loss_graph.png  -- train_loss vs val_loss per epoch
rem   mae_graph.png   -- SBP/DBP MAE (train + val) per epoch
rem
rem Usage:
rem   bin\train-status.bat <run_dir>
rem   bin\train-status.bat data\models\resnet1d\20260101_120000
rem   bin\train-status.bat data\models\resnet1d\20260101_120000 --no-save

cd /d "%~dp0.."
uv run python scripts\train-status.py %*
