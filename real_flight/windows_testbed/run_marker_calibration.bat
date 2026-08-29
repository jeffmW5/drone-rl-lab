@echo off
REM Calibrate apparent marker size against real distance.
REM Pass --self-test to check the fit maths without hardware.
setlocal
cd /d "%~dp0"
set "PYEXE=py -3"
py -3 --version >nul 2>&1
if errorlevel 1 set "PYEXE=python"
%PYEXE% marker_calibration_gui.py %*
if errorlevel 1 pause
