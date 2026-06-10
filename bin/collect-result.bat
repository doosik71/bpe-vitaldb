@echo off

cd /d "%~dp0.."
uv run python scripts\collect-result.py %*
