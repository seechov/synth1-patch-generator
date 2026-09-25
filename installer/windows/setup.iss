#ifndef MyAppVersion
  #define MyAppVersion "1"
#endif

#define MyAppName      "Seechov Forge"
#define MyAppPublisher "Aleksei Sychev"
#define MyAppExeName   "seechov-forge.exe"
#define MyAppRoot      "..\.."

[Setup]
AppId={{3B8F7C2A-D5E4-4B6F-9A1C-8D2E7F3B5A90}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir={#MyAppRoot}\dist
OutputBaseFilename=seechov-forge-setup-{#MyAppVersion}-windows-x64
Compression=lzma2/ultra64
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#MyAppRoot}\gui\target\release\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
; Bundled model — shipped as the default model so a first-time user only
; needs to pick an output folder and click Generate.
Source: "{#MyAppRoot}\installer\trained-model\generator.onnx"; DestDir: "{app}\model"; Flags: ignoreversion
Source: "{#MyAppRoot}\installer\trained-model\normalization.json"; DestDir: "{app}\model"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; \
  Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; \
  Flags: nowait postinstall skipifsilent
