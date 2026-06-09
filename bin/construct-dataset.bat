@echo off
cls
cd /d "%~dp0.."
uv run python scripts\construct-dataset.py %*
