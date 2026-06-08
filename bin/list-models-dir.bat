@echo off
rem List directories under data\models\ that contain a trained model (best.pt).
rem
rem Usage:
rem   bin\list-models-dir.bat

cd /d "%~dp0.."

for /r "data\models" %%F in (best.pt) do (
    set "p=%%~dpF"
    setlocal enabledelayedexpansion
    set "p=!p:~0,-1!"
    echo !p!
    endlocal
)
