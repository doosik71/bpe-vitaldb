@echo off
cd /d "%~dp0.."
uv run python scripts\train-all-model.py %*
