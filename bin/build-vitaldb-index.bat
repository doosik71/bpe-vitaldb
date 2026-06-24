@echo off
cd /d "%~dp0.."
uv run python scripts\build-vitaldb-index.py %*
