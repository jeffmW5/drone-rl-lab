@echo off
REM Flight Watch -- check the drone, then record the camera while you fly.
REM This tool never commands the drone.
setlocal
cd /d "%~dp0"

set "PYEXE=py -3"
py -3 --version >/dev/null 2>&1
if errorlevel 1 (
    set "PYEXE=python"
    python --version >/dev/null 2>&1
    if errorlevel 1 (
        echo Python 3.10+ was not found on PATH.
        echo Install from https://www.python.org/downloads/windows/
        echo and tick "Add python.exe to PATH".
        pause
        exit /b 9009
    )
)

%PYEXE% flight_watch_gui.py
