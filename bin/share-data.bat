@echo off
rem Serve the data/ folder over HTTP using Python's built-in file server.
rem
rem Usage:
rem   bin\share-data.bat [PORT]
rem
rem   PORT defaults to 8000.
rem   Browse to http://localhost:PORT/ or share the LAN address shown on screen.

cd /d "%~dp0.."

set PORT=%1
if "%PORT%"=="" set PORT=8000

uv run python -m http.server %PORT% --directory data
