@echo off
cd /d "%~dp0.."
uv run python scripts\bpe-gui.py %*
