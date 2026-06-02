@echo off
setlocal
cd /d "%~dp0.."
call uv run python scripts\check-cuda.py
