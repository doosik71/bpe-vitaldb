@echo off
cd /d "%~dp0.."
uv run python scripts\psd-browser.py %*
