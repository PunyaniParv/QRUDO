@echo off
rem Double-click me to start QRUDO on Windows.
rem
rem main.py builds its own environment on a new device; this only has
rem to find a real Python 3.  The py launcher is tried first because on
rem a fresh Windows, `python` is a Microsoft Store decoy that opens the
rem Store instead of running anything.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 main.py %*
) else (
  python main.py %*
)
rem Keep the window open on failure so the reason can be read.
if errorlevel 1 pause
