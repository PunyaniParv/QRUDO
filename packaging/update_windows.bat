@echo off
rem Keep the Windows QRUDO current.
rem
rem Pulls the latest code, rebuilds only when something new arrived,
rem and keeps a Start Menu shortcut named QRUDO pointing at the
rem result -- so the shortcut a person clicks never goes stale.
rem
rem Double-click this whenever, or let Windows run it for you daily:
rem
rem   schtasks /create /tn "QRUDO Update" /tr "%~f0" /sc daily /st 13:00

cd /d "%~dp0.."

for /f %%i in ('git rev-parse HEAD') do set BEFORE=%%i
git pull --ff-only
if errorlevel 1 (
  echo Could not pull -- check the network or local changes.
  pause
  exit /b 1
)
for /f %%i in ('git rev-parse HEAD') do set AFTER=%%i

if "%BEFORE%"=="%AFTER%" if exist dist\QRUDO\QRUDO.exe (
  echo Nothing new; QRUDO is already current.
  goto shortcut
)

call packaging\build_windows.bat
if errorlevel 1 (
  pause
  exit /b 1
)

:shortcut
powershell -NoProfile -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut([Environment]::GetFolderPath('Programs') + '\QRUDO.lnk'); $s.TargetPath = '%cd%\dist\QRUDO\QRUDO.exe'; $s.WorkingDirectory = '%cd%\dist\QRUDO'; $s.Save()"

echo QRUDO is current.
