@echo off
rem Download the entire shared data/ folder from a remote host running
rem bin/share-data.bat (Python http.server).
rem
rem Usage:
rem   bin\download-shared-data.bat <host> [port]
rem
rem   host  IP address or hostname of the remote machine
rem   port  HTTP port (default: 8000)
rem
rem Example:
rem   bin\download-shared-data.bat 192.168.1.10
rem   bin\download-shared-data.bat 192.168.1.10 9000
rem
rem Requires: wget.exe on PATH  (e.g. via winget install GnuWin32.Wget
rem           or https://eternallybored.org/misc/wget/)

cd /d "%~dp0.."

set HOST=%1
set PORT=%2

if "%HOST%"=="" (
    echo Usage: bin\download-shared-data.bat ^<host^> [port]
    exit /b 1
)
if "%PORT%"=="" set PORT=8000

set BASE_URL=http://%HOST%:%PORT%/

echo Downloading data/ from %BASE_URL% ...

wget ^
    --recursive ^
    --no-host-directories ^
    --cut-dirs=0 ^
    --directory-prefix=data ^
    --no-parent ^
    --reject "index.html*" ^
    --progress=dot:mega ^
    "%BASE_URL%"

echo Done. Files saved to data\
