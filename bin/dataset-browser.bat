@echo off
cd /d "%~dp0.."
uv run python scripts\dataset-browser.py %*
