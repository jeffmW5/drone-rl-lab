@echo off
REM Play back an AI Deck recording made by Flight Watch.
REM Optionally pass a run folder:  run_playback.bat path\to\flight_YYYYmmdd_HHMMSS
setlocal
cd /d "%~dp0"
set "PYEXE=py -3"
py -3 --version >nul 2>&1
if errorlevel 1 set "PYEXE=python"
%PYEXE% playback_gui.py %*
if errorlevel 1 pause
