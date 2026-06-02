@echo off
rem Serve the data/ folder over HTTP (multi-threaded, all interfaces).
rem
rem Usage:
rem   bin\share-data.bat [PORT]
rem
rem   PORT defaults to 8888.
rem   The LAN address is printed on startup so remote machines know where to connect.

cd /d "%~dp0.."
uv run python scripts\share-data.py --port %1 %2 %3 %4
