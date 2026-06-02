@echo off
cd /d "%~dp0.."
uv run python scripts\download-vitaldb.py %*
