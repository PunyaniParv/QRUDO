; The Windows installer: what turns a folder of files into a product.
; Compiled by packaging\build_windows.bat when Inno Setup is present.

[Setup]
AppName=QRUDO
; Keep in step with version.py, which is what the update check compares.
AppVersion=0.1.0
AppPublisher=QRUDO
DefaultDirName={autopf}\QRUDO
DefaultGroupName=QRUDO
UninstallDisplayName=QRUDO
OutputDir=..\dist
OutputBaseFilename=QRUDO-Setup
; Per-user by default: no admin prompt between a curious person and
; the first run.
PrivilegesRequired=lowest
Compression=lzma2
SolidCompression=yes
; The eclipse Q, on the installer itself as well as the app.
SetupIconFile=..\assets\qrudo.ico

[Files]
Source: "..\dist\QRUDO\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\QRUDO"; Filename: "{app}\QRUDO.exe"
Name: "{autodesktop}\QRUDO"; Filename: "{app}\QRUDO.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop icon"; Flags: unchecked

[Run]
Filename: "{app}\QRUDO.exe"; Description: "Start QRUDO now"; \
  Flags: postinstall nowait skipifsilent
