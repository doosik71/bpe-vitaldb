@echo off
cd /d "%~dp0.."
uv run python scripts\pulse-browser.py %*
