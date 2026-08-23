@echo off
REM AI Deck WiFi Test Bed launcher.
REM Double-click for the GUI. Pass arguments to run a console test instead:
REM     run_testbed.bat throughput --throughput-duration 300
setlocal
cd /d "%~dp0"

set "PYEXE=py -3"
py -3 --version >nul 2>&1
if errorlevel 1 (
    set "PYEXE=python"
    python --version >nul 2>&1
    if errorlevel 1 (
        echo Python 3.10+ was not found on PATH.
        echo Install from https://www.python.org/downloads/windows/
        echo and tick "Add python.exe to PATH".
        pause
        exit /b 9009
    )
)

if "%~1"=="" (
    %PYEXE% testbed_gui.py
) else (
    %PYEXE% testbed_cli.py %*
    pause
)