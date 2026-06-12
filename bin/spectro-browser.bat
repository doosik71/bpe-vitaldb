@echo off
cd /d "%~dp0.."
uv run python scripts\spectro-browser.py %*
