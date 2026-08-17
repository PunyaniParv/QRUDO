@echo off
rem Build QRUDO for Windows, then optionally wrap it in an installer.
rem
rem   packaging\build_windows.bat
rem
rem Produces dist\QRUDO\QRUDO.exe -- a folder anyone can run from
rem directly.  If Inno Setup is installed (https://jrsoftware.org),
rem it is also compiled into dist\QRUDO-Setup.exe, the thing to hand
rem to a user: Start Menu entry, per-user install, clean uninstall.
rem
rem Windows needs no camera-permission ceremony; the first camera use
rem asks by itself.  Code signing (signtool, an EV certificate) can be
rem added here the same way notarization is on the Mac -- until then,
rem SmartScreen will warn once per download, which is survivable for
rem early users and the reason to buy the certificate eventually.

cd /d "%~dp0.."

rem A fresh Windows machine has no environment yet; build it the same
rem way a first run of QRUDO would, so this script is one command on a
rem machine that has only Python and this repository.
set PYTHON=.venv\Scripts\python.exe
if not exist %PYTHON% (
  where py >nul 2>nul
  if %errorlevel%==0 (py -3 -m venv .venv) else (python -m venv .venv)
  %PYTHON% -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1
)

%PYTHON% -c "import PyInstaller" 2>nul || %PYTHON% -m pip install pyinstaller
%PYTHON% -m PyInstaller --noconfirm --clean packaging\qrudo.spec
if errorlevel 1 exit /b 1

where iscc >nul 2>nul
if %errorlevel%==0 (
  iscc packaging\qrudo.iss
) else (
  echo Inno Setup not found; dist\QRUDO\QRUDO.exe is ready as-is.
)
