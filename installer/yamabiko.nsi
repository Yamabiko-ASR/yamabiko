; Yamabiko NSIS installer - v0.2.0-beta
; Requires NSIS 3.x Unicode build (default in modern distributions).
; Verify with: makensis /HDRINFO | findstr NSIS_UNICODE
;
; Look ("Full landscape", design spec Surface 7): MUI2's stock window,
; buttons and progress bar with Segoe UI 9 on every control, the Mist
; colors, and full-window nsDialogs pages (Welcome or Update, Finish,
; Remove, Removed) over landscape bitmaps. The bitmaps (installer\art) and
; the license text (LICENSE.rtf) are committed; regenerate them with
; installer\art\build_art.py and installer\license_rtf.py.
; Keep this file UTF-8 with a BOM: makensis reads it as UTF-8 only then,
; and the copy has non-ASCII characters ("Copying files…").

Unicode true
; System DPI aware: Windows draws the window at the display scale (instead
; of stretching a 96 DPI window), and the bitmaps are picked for it.
ManifestDPIAware true
; Per-file (non-solid) lzma. Originally chosen because /SOLID lzma hit
; "error mmapping datablock" on the pre-torch-drop bundle (individual DLLs
; >600 MB exceeded NSIS's ~32 MB mmap window). Bundle is now ~1.4 GB after
; the torch drop; /SOLID would likely work and shrink the installer maybe
; 10-20%, but we ship non-solid: faster to compile during iteration, no
; functional difference for users, and removes the mmap-limit footgun if
; someone later adds back a large dependency.
SetCompressor lzma

!define APP_NAME "Yamabiko"
!define APP_VERSION "0.2.0-beta"
; The caption's form of APP_VERSION ("0.2.0 beta"), derived so it can't
; drift from it.
!searchreplace APP_VERSION_DISPLAY "${APP_VERSION}" "-" " "
; Binary VERSIONINFO quad for the VIProductVersion/VIFileVersion block
; below. MUST equal APP_VERSION's numeric X.Y.Z prefix + ".0" (the block's
; comment explains why 4 numeric parts are required; the "-beta" suffix
; cannot appear here). Bump it together with APP_VERSION —
; scripts/preflight.py's version-consistency section fails the release
; gate if the two drift (installer/version_info.txt too).
!define APP_VERSION_QUAD "0.2.0.0"
!define APP_PUBLISHER "Christian Allen"
!define APP_EXE "yamabiko.exe"
!define APP_REGKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"
; Colors of the art (installer\art\*.svg): the skies the page text sits on,
; the ink and muted text, and the link.
!define SKY_MIST F4F3EF
!define SKY_DUSK F6EEE6
!define INK 1F2421
!define MUTED 646A65
!define ACCENT C8402C

!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "nsDialogs.nsh"
!include "WinMessages.nsh"
!include "WordFunc.nsh"

; Name feeds MUI's default texts; the version shows only in the caption.
Name "${APP_NAME}"
Caption "Install ${APP_NAME} ${APP_VERSION_DISPLAY}"
UninstallCaption "Uninstall ${APP_NAME}"
BrandingText " "
OutFile "yamabiko-${APP_VERSION}-setup.exe"

; PE VERSIONINFO on the installer .exe -- without it the file shows "no
; information" in Properties -> Details, which adds to SmartScreen friction on
; the unsigned build. VIProductVersion/VIFileVersion require 4 numeric parts;
; the "-beta" pre-release suffix lives only in the string keys.
VIProductVersion "${APP_VERSION_QUAD}"
VIFileVersion "${APP_VERSION_QUAD}"
VIAddVersionKey "ProductName" "${APP_NAME}"
VIAddVersionKey "ProductVersion" "${APP_VERSION}"
VIAddVersionKey "FileVersion" "${APP_VERSION}"
VIAddVersionKey "FileDescription" "${APP_NAME} installer"
VIAddVersionKey "CompanyName" "${APP_PUBLISHER}"
VIAddVersionKey "LegalCopyright" "(c) Christian Allen. PolyForm Noncommercial 1.0.0."

InstallDir "$LOCALAPPDATA\${APP_NAME}"
; Reuse the prior install location on re-install so an upgrade never
; creates a second copy at a different path. The runtime
; data dir (paths.app_data_dir) is hardcoded to $LOCALAPPDATA\Yamabiko,
; so the install dir MUST stay there too -- the MUI directory page is
; deliberately omitted below.
InstallDirRegKey HKCU "${APP_REGKEY}" "InstallLocation"
RequestExecutionLevel user   ; no UAC; LOCALAPPDATA is per-user
; Go on to the Finish page by itself when the files are in.
AutoCloseWindow true

!define MUI_ICON "..\assets\icon.ico"
!define MUI_UNICON "..\assets\icon.ico"
!define MUI_BGCOLOR ${SKY_MIST}
!define MUI_TEXTCOLOR ${INK}
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_RIGHT
; MUI shows this 100 % header first; GuiInit swaps in the one picked for
; the display's scale.
!define MUI_HEADERIMAGE_BITMAP "art\header-100.bmp"
!define MUI_CUSTOMFUNCTION_GUIINIT GuiInit
!define MUI_CUSTOMFUNCTION_UNGUIINIT un.GuiInit

Var Dpi        ; system DPI, 96 = 100 %
Var Pct        ; bitmap scale picked for it: 100, 150 or 200
Var FontBody   ; Segoe UI 9
Var FontBold   ; Segoe UI 9 bold
Var FontTitle  ; Segoe UI 15 bold (20 px at 100 %)
Var FontLink   ; Segoe UI 9 underlined
Var IsUpdate   ; 1 when the Apps entry shows an installed version
Var OldVersion ; that version, dashes as spaces ("0.1.0 beta")
Var Dlg        ; the full-window page
Var Art        ; its landscape: mist, dusk or removed
Var SealArt    ; its seal: seal-mist or seal-dusk
Var BmpH       ; landscape HBITMAP, freed when the page closes
Var SealH      ; seal HBITMAP, freed when the page closes
Var Chk        ; Finish: Start Yamabiko
Var Kept       ; uninstall: 1 when the folder stays (user data)
Var SizeText   ; uninstall: its measured size ("6.8 GB")

; --- Pages ------------------------------------------------------------------
; Welcome/Update, Finish, Remove and Removed are nsDialogs pages on MUI2's
; full-window area (control 1044), as MUI2's own Welcome and Finish pages
; are: MUI_PAGE_FUNCTION_FULLWINDOW defines muiPageLoadFullWindow /
; muiPageUnloadFullWindow, which hide the header and branding around them.
!insertmacro MUI_PAGE_INIT
!insertmacro MUI_PAGE_FUNCTION_FULLWINDOW
Page custom WelcomeCreate WelcomeLeave

!define MUI_PAGE_HEADER_TEXT "License"
!define MUI_PAGE_HEADER_SUBTEXT "Free for personal, noncommercial use."
!define MUI_LICENSEPAGE_TEXT_TOP "Yamabiko is licensed under PolyForm Noncommercial 1.0.0."
!define MUI_LICENSEPAGE_TEXT_BOTTOM "Click I Agree to accept the license and install Yamabiko."
!define MUI_PAGE_CUSTOMFUNCTION_PRE LicensePre
!define MUI_PAGE_CUSTOMFUNCTION_SHOW LicenseShow
!insertmacro MUI_PAGE_LICENSE "LICENSE.rtf"
; MUI_PAGE_DIRECTORY intentionally omitted: the install dir must equal
; paths.app_data_dir() ($LOCALAPPDATA\Yamabiko) or user data + the
; uninstaller diverge. Power users / CI can still override with /D= --
; an explicit, documented choice.
!define MUI_PAGE_HEADER_TEXT "Installing Yamabiko"
!define MUI_PAGE_HEADER_SUBTEXT "This takes a minute or two."
; The completed header shows for a few ms before the Finish page: keep the
; installing copy so nothing flashes.
!define MUI_INSTFILESPAGE_FINISHHEADER_TEXT "Installing Yamabiko"
!define MUI_INSTFILESPAGE_FINISHHEADER_SUBTEXT "This takes a minute or two."
!define MUI_INSTFILESPAGE_ABORTHEADER_TEXT "Yamabiko isn't fully installed"
!define MUI_INSTFILESPAGE_ABORTHEADER_SUBTEXT "Run the installer again to finish."
!define MUI_PAGE_CUSTOMFUNCTION_SHOW InstFilesShow
!insertmacro MUI_PAGE_INSTFILES
Page custom FinishCreate FinishLeave

!insertmacro MUI_UNPAGE_INIT
!insertmacro MUI_PAGE_FUNCTION_FULLWINDOW
UninstPage custom un.RemoveCreate un.RemoveLeave
!define MUI_PAGE_HEADER_TEXT "Removing Yamabiko"
!define MUI_PAGE_HEADER_SUBTEXT "This takes a moment."
!define MUI_INSTFILESPAGE_FINISHHEADER_TEXT "Removing Yamabiko"
!define MUI_INSTFILESPAGE_FINISHHEADER_SUBTEXT "This takes a moment."
; MUI2 unsets the finish header defines after each progress page but not
; the abort header ones.
!undef MUI_INSTFILESPAGE_ABORTHEADER_TEXT
!undef MUI_INSTFILESPAGE_ABORTHEADER_SUBTEXT
!define MUI_INSTFILESPAGE_ABORTHEADER_TEXT "Removal stopped"
!define MUI_INSTFILESPAGE_ABORTHEADER_SUBTEXT "Run the uninstaller again to finish."
!define MUI_PAGE_CUSTOMFUNCTION_SHOW un.InstFilesShow
!insertmacro MUI_UNPAGE_INSTFILES
UninstPage custom un.RemovedCreate

!insertmacro MUI_LANGUAGE "English"

; --- Refuse to install / uninstall while Yamabiko is running ----------
; The install dir IS the runtime dir; a running yamabiko.exe / its loaded
; _internal DLLs / the bundled llama-server.exe hold exclusive locks, so
; File /r + RMDir /r would half-complete and corrupt the install. Uses
; only the stock nsExec plugin (bundled with
; NSIS) -- no third-party nsProcess/FindProcDLL dependency. Two probes:
; yamabiko.exe by image name via tasklist (the name is unique to this
; app), llama-server.exe path-scoped under $INSTDIR via powershell (the
; name is generic llama.cpp). Both probes fail OPEN (a probe
; that cannot launch -> proceed) so a probe hiccup never blocks a
; legitimate install; tasklist + powershell are universally present on
; Windows. ${UN} expands to "" for the installer and "un." for the
; uninstaller, generating two function copies with function-local labels.
; Called from .onInit / un.onInit for silent runs (the box takes /SD
; IDCANCEL -> Abort -> exit code 2) and from the Welcome / Remove pages'
; leave functions for interactive runs, so the box belongs to the window
; and Cancel (Abort) keeps that page up.
!macro _EnsureNotRunning UN
Function ${UN}EnsureNotRunning
  recheck:
  nsExec::ExecToStack 'cmd /c tasklist /NH /FI "IMAGENAME eq ${APP_EXE}" | findstr /I "${APP_EXE}"'
  Pop $0
  Pop $1
  ${If} $0 == 0
    MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION \
      "Yamabiko is still running. Close Yamabiko setup, or right-click its tray icon and choose Quit Yamabiko, then click Retry." \
      /SD IDCANCEL IDRETRY recheck
    Abort
  ${EndIf}
  ; llama-server.exe is the GENERIC llama.cpp server name -- unrelated
  ; apps (LM Studio, ollama frontends, dev checkouts) run binaries with
  ; the same image name, so this probe must be scoped to OUR copy by
  ; executable path under $INSTDIR (watchdog.py's orphan
  ; sweep path-scopes its match set for the same reason). $INSTDIR is
  ; resolved before .onInit/un.onInit run (InstallDir + InstallDirRegKey
  ; + /D=), so every caller sees it. Exit 0 = a Yamabiko-owned
  ; llama-server is alive (block); any launch/parse failure exits
  ; nonzero -> fail OPEN, matching the tasklist probe above. The path
  ; sits in a PS single-quoted literal: a pathological $INSTDIR (quote
  ; char) makes the script fail to parse -> nonzero -> proceed.
  ;
  ; BITNESS NOTE: NSIS is a 32-bit process, so this launches the SysWOW64 (32-bit)
  ; powershell, and from 32-bit PowerShell `Get-Process ... | Where
  ; { $_.Path ... }` CANNOT read a 64-bit process's module path -- the
  ; predicate evaluates falsy and the probe fails OPEN for every real
  ; (64-bit) llama-server. Query WMI via Get-CimInstance instead:
  ; Win32_Process.ExecutablePath is served by the (64-bit) WMI
  ; provider and is bitness-agnostic. ExecutablePath is $null for
  ; protected processes; -like on $null is falsy -> still fail-open.
  ; The CIM filter uses a PS single-quoted string with doubled inner
  ; quotes so no nested double-quote reaches the -Command parser.
  nsExec::ExecToStack `powershell -NoProfile -NonInteractive -Command "if (Get-CimInstance Win32_Process -Filter 'Name=''llama-server.exe''' -ErrorAction SilentlyContinue | Where-Object { $$_.ExecutablePath -like '$INSTDIR\*' }) { exit 0 } else { exit 1 }"`
  Pop $0
  Pop $1
  ${If} $0 == 0
    MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION \
      "Yamabiko's translator is still running. Quit Yamabiko, or end llama-server.exe in Task Manager, then click Retry." \
      /SD IDCANCEL IDRETRY recheck
    Abort
  ${EndIf}
FunctionEnd
!macroend
!insertmacro _EnsureNotRunning ""
!insertmacro _EnsureNotRunning "un."

; --- Look: DPI, fonts, bitmaps (installer and uninstaller copies) ------------
; Positions in the mockup's px at 100 %, scaled by the system DPI.
!macro _Px VAR VALUE
  IntOp ${VAR} ${VALUE} * $Dpi
  IntOp ${VAR} ${VAR} / 96
!macroend

; A page label over the art's flat sky. Opaque, in that sky's color: it
; looks the same as a transparent label, and repainting it can't thicken
; the ClearType text. Leaves the label's handle in $R9.
!macro _Label X Y W H FONT COLOR SKY TEXT
  !insertmacro _Px $R5 ${X}
  !insertmacro _Px $R6 ${Y}
  !insertmacro _Px $R7 ${W}
  !insertmacro _Px $R8 ${H}
  ${NSD_CreateLabel} $R5 $R6 $R7 $R8 "${TEXT}"
  Pop $R9
  SetCtlColors $R9 ${COLOR} ${SKY}
  SendMessage $R9 ${WM_SETFONT} ${FONT} 0
!macroend

; Extract only the picked scale of installer\art\<NAME>-<pct>.bmp.
!macro _PickArt NAME
  ${If} $Pct == 150
    File "/oname=$PLUGINSDIR\${NAME}.bmp" "art\${NAME}-150.bmp"
  ${ElseIf} $Pct == 200
    File "/oname=$PLUGINSDIR\${NAME}.bmp" "art\${NAME}-200.bmp"
  ${Else}
    File "/oname=$PLUGINSDIR\${NAME}.bmp" "art\${NAME}-100.bmp"
  ${EndIf}
!macroend

; A page's landscape and seal, named once: extracted at the picked scale
; and handed to CreateArt.
!macro _PageArt ART SEAL
  !insertmacro _PickArt ${ART}
  !insertmacro _PickArt ${SEAL}
  StrCpy $Art ${ART}
  StrCpy $SealArt ${SEAL}
!macroend

; Make a control PX pixels taller (Segoe UI's descenders reach the last row
; of MUI's single-line statics at 200 %).
!macro _GrowControl HWND PX
  System::Call '*(i,i,i,i) p .r0'
  System::Call 'user32::GetWindowRect(p ${HWND}, p r0)'
  System::Call '*$0(i .r1, i .r2, i .r3, i .r4)'
  System::Free $0
  IntOp $3 $3 - $1
  IntOp $4 $4 - $2
  IntOp $4 $4 + ${PX}
  System::Call 'user32::SetWindowPos(p ${HWND}, p 0, i 0, i 0, i r3, i r4, i 0x16)'
!macroend

; Move a control PX pixels down inside its parent window.
!macro _MoveControlDown HWND PARENT PX
  System::Call '*(i,i,i,i) p .r0'
  System::Call 'user32::GetWindowRect(p ${HWND}, p r0)'
  System::Call 'user32::MapWindowPoints(p 0, p ${PARENT}, p r0, i 2)'
  System::Call '*$0(i .r1, i .r2)'
  System::Free $0
  IntOp $2 $2 + ${PX}
  System::Call 'user32::SetWindowPos(p ${HWND}, p 0, i r1, i r2, i 0, i 0, i 0x15)'
!macroend

!macro _UiFunctions UN
Function ${UN}GuiInit
  ; ManifestDPIAware makes the dialog follow the system DPI; GetDeviceCaps
  ; reads it on every Windows version.
  System::Call 'user32::GetDC(p 0) p .r0'
  System::Call 'gdi32::GetDeviceCaps(p r0, i 88) i .r1'   ; LOGPIXELSX
  System::Call 'user32::ReleaseDC(p 0, p r0)'
  StrCpy $Dpi $1
  ; 125 % scales the 150 % art down, 175 % the 200 % art.
  ${If} $Dpi <= 96
    StrCpy $Pct 100
  ${ElseIf} $Dpi <= 144
    StrCpy $Pct 150
  ${Else}
    StrCpy $Pct 200
  ${EndIf}
  ; Segoe UI 9 on every control. The dialog template keeps MUI2's MS Shell
  ; Dlg font, so the window keeps its stock size.
  CreateFont $FontBody "Segoe UI" 9
  CreateFont $FontBold "Segoe UI" 9 700
  CreateFont $FontTitle "Segoe UI" 15 700
  CreateFont $FontLink "Segoe UI" 9 400 /UNDERLINE
  SendMessage $mui.Button.Next ${WM_SETFONT} $FontBody 1
  SendMessage $mui.Button.Back ${WM_SETFONT} $FontBody 1
  SendMessage $mui.Button.Cancel ${WM_SETFONT} $FontBody 1
  SendMessage $mui.Header.Text ${WM_SETFONT} $FontBold 1
  SendMessage $mui.Header.SubText ${WM_SETFONT} $FontBody 1
  !insertmacro _GrowControl $mui.Header.Text 2
  Call ${UN}HeaderImage
  ; BrandingText " " still leaves a short static over the etched line on
  ; the License and progress pages: move both branding statics out of the
  ; window (the full-window pages only hide and show them).
  System::Call 'user32::SetWindowPos(p $mui.Branding.Text, p 0, i -2000, i 0, i 0, i 0, i 0x15)'
  System::Call 'user32::SetWindowPos(p $mui.Branding.Background, p 0, i -2000, i 0, i 0, i 0, i 0x15)'
FunctionEnd

Function ${UN}HeaderImage
  ; The header bitmap at the picked scale, fitted to the header control's
  ; height (keeping its aspect) and kept at the window's right edge.
  InitPluginsDir
  !insertmacro _PickArt header
  System::Call '*(i,i,i,i) p .r0'
  System::Call 'user32::GetWindowRect(p $mui.Header.Image, p r0)'
  System::Call 'user32::MapWindowPoints(p 0, p $HWNDPARENT, p r0, i 2)'
  System::Call '*$0(i .r1, i .r2, i .r3, i .r4)'
  System::Free $0
  StrCpy $R0 $mui.Header.Image
  StrCpy $R1 "$PLUGINSDIR\header.bmp"
  StrCpy $R2 0
  IntOp $R3 $4 - $2
  Call ${UN}LoadFit
  ; The static took the image's size; move it so its right edge stays put.
  System::Call '*(i,i,i,i) p .r0'
  System::Call 'user32::GetClientRect(p $mui.Header.Image, p r0)'
  System::Call '*$0(i, i, i .r5, i)'
  System::Free $0
  IntOp $1 $3 - $5
  System::Call 'user32::SetWindowPos(p $mui.Header.Image, p 0, i r1, i r2, i 0, i 0, i 0x15)'
FunctionEnd

Function ${UN}LoadFit
  ; Show a .bmp in an SS_BITMAP static, scaled with HALFTONE (smooth; the
  ; LoadImage scaling behind NSD_SetStretchedImage and /RESIZETOFIT is
  ; nearest-neighbor, which breaks the thin arcs).
  ; In: $R0 static, $R1 .bmp path, $R2 width (0 = from $R3 keeping the
  ; aspect), $R3 height. Out: $R4 the HBITMAP the static now shows (delete
  ; it after the page closes). Every other register is kept.
  System::Store "s"
  System::Call 'user32::LoadImageW(p 0, w R1, i 0, i 0, i 0, i 0x2010) p .r0'   ; IMAGE_BITMAP, LR_LOADFROMFILE|LR_CREATEDIBSECTION
  System::Call '*(i,i,i,i,i,i) p .r1'
  System::Call 'gdi32::GetObjectW(p r0, i 24, p r1)'
  System::Call '*$1(i, i .r2, i .r3)'
  System::Free $1
  ${If} $R2 == 0
    IntOp $R2 $2 * $R3
    IntOp $4 $3 / 2
    IntOp $R2 $R2 + $4
    IntOp $R2 $R2 / $3
  ${EndIf}
  System::Call 'user32::GetDC(p 0) p .r4'
  System::Call 'gdi32::CreateCompatibleDC(p r4) p .r5'
  System::Call 'gdi32::CreateCompatibleDC(p r4) p .r6'
  System::Call 'gdi32::CreateCompatibleBitmap(p r4, i R2, i R3) p .r7'
  System::Call 'user32::ReleaseDC(p 0, p r4)'
  System::Call 'gdi32::SelectObject(p r5, p r0) p .r8'
  System::Call 'gdi32::SelectObject(p r6, p r7) p .r9'
  System::Call 'gdi32::SetStretchBltMode(p r6, i 4)'   ; HALFTONE
  System::Call 'gdi32::SetBrushOrgEx(p r6, i 0, i 0, p 0)'
  System::Call 'gdi32::StretchBlt(p r6, i 0, i 0, i R2, i R3, p r5, i 0, i 0, i r2, i r3, i 0x00CC0020)'   ; SRCCOPY
  System::Call 'gdi32::SelectObject(p r5, p r8)'
  System::Call 'gdi32::SelectObject(p r6, p r9)'
  System::Call 'gdi32::DeleteDC(p r5)'
  System::Call 'gdi32::DeleteDC(p r6)'
  System::Call 'gdi32::DeleteObject(p r0)'
  SendMessage $R0 ${STM_SETIMAGE} ${IMAGE_BITMAP} $7 $8
  ${If} $8 <> 0
    System::Call 'gdi32::DeleteObject(p r8)'
  ${EndIf}
  Push $7
  System::Store "l"
  Pop $R4
FunctionEnd

Function ${UN}CreateArt
  ; Call after the page's labels: a control created later sits under the
  ; earlier ones. Uses $Dlg, $Art and $SealArt (set and extracted by _PageArt).
  ; The page's real size isn't 497 x 314 times the scale (dialog units
  ; don't scale linearly), so the landscape is scaled to the page width
  ; keeping its aspect and anchored to the bottom; the page background, the
  ; same sky color, fills any gap above it.
  System::Call '*(i,i,i,i) p .r0'
  System::Call 'user32::GetClientRect(p $Dlg, p r0)'
  System::Call '*$0(i, i, i .r1, i .r2)'
  System::Free $0
  ; The seal: its own square control at (28, 26), 32 px at 100 %.
  !insertmacro _Px $3 28
  !insertmacro _Px $4 26
  !insertmacro _Px $5 32
  ${NSD_CreateBitmap} $3 $4 $5 $5 ""
  Pop $R0
  StrCpy $R1 "$PLUGINSDIR\$SealArt.bmp"
  StrCpy $R2 $5
  StrCpy $R3 $5
  Call ${UN}LoadFit
  StrCpy $SealH $R4
  ; The landscape: 314/497 of the page width tall, at the page bottom.
  IntOp $6 314 * $1
  IntOp $6 $6 + 248
  IntOp $6 $6 / 497
  IntOp $7 $2 - $6
  ; nsDialogs reads a negative position as "from the far edge": create the
  ; control at y 0, then move it.
  ${NSD_CreateBitmap} 0 0 $1 $6 ""
  Pop $R0
  System::Call 'user32::SetWindowPos(p R0, p 0, i 0, i r7, i 0, i 0, i 0x15)'
  StrCpy $R1 "$PLUGINSDIR\$Art.bmp"
  StrCpy $R2 $1
  StrCpy $R3 $6
  Call ${UN}LoadFit
  StrCpy $BmpH $R4
FunctionEnd

Function ${UN}FreeArt
  System::Call 'gdi32::DeleteObject(p $BmpH)'
  System::Call 'gdi32::DeleteObject(p $SealH)'
  StrCpy $BmpH 0
  StrCpy $SealH 0
FunctionEnd

Function ${UN}InstFilesShow
  SendMessage $mui.InstFilesPage.Text ${WM_SETFONT} $FontBody 1
  SendMessage $mui.InstFilesPage.ShowLogButton ${WM_SETFONT} $FontBody 1
  SendMessage $mui.InstFilesPage.Log ${WM_SETFONT} $FontBody 1
  ; Room for the status line's descenders at 200 %: 2 px taller, and the
  ; progress bar 2 px lower.
  !insertmacro _GrowControl $mui.InstFilesPage.Text 2
  !insertmacro _MoveControlDown $mui.InstFilesPage.ProgressBar $mui.InstFilesPage 2
FunctionEnd
!macroend
!insertmacro _UiFunctions ""
!insertmacro _UiFunctions "un."

; --- Installer ----------------------------------------------------------------
Function .onInit
  ; An installed version in the Apps entry (any version, this one too) makes
  ; this an update: Welcome shows the Update copy and the license is skipped.
  ReadRegStr $OldVersion HKCU "${APP_REGKEY}" "DisplayVersion"
  StrCpy $IsUpdate 0
  ${If} $OldVersion != ""
    StrCpy $IsUpdate 1
    ${WordReplace} "$OldVersion" "-" " " "+" $OldVersion
  ${EndIf}
  ; Silent installs check here (exit code 2 while Yamabiko runs);
  ; interactive installs check when Next is clicked on Welcome.
  ${If} ${Silent}
    Call EnsureNotRunning
  ${EndIf}
FunctionEnd

Function WelcomeCreate
  nsDialogs::Create 1044
  Pop $Dlg
  ${If} $Dlg == error
    Abort
  ${EndIf}
  SetCtlColors $Dlg "" ${SKY_MIST}
  InitPluginsDir
  !insertmacro _PageArt mist seal-mist
  ${If} $IsUpdate == 1
    !insertmacro _Label 70 28 400 28 $FontTitle ${INK} ${SKY_MIST} "Update Yamabiko"
    !insertmacro _Label 28 72 330 34 $FontBody ${INK} ${SKY_MIST} "Yamabiko $OldVersion is installed. This updates it to ${APP_VERSION_DISPLAY}."
    !insertmacro _Label 28 98 330 34 $FontBody ${MUTED} ${SKY_MIST} "Your settings and downloaded models stay."
  ${Else}
    !insertmacro _Label 70 28 400 28 $FontTitle ${INK} ${SKY_MIST} "Welcome to Yamabiko"
    !insertmacro _Label 28 72 330 34 $FontBody ${INK} ${SKY_MIST} "Yamabiko listens to Japanese audio on your PC and shows subtitles with furigana and an English translation."
    !insertmacro _Label 28 114 330 34 $FontBody ${MUTED} ${SKY_MIST} "The first time you start it, a short setup downloads the models it needs (up to 7 GB)."
  ${EndIf}
  Call CreateArt
  Call muiPageLoadFullWindow
  ; NSIS sets the button text on every page entry, so this needs no reset.
  ${If} $IsUpdate == 1
    SendMessage $mui.Button.Next ${WM_SETTEXT} 0 "STR:&Update"
  ${EndIf}
  nsDialogs::Show
  Call muiPageUnloadFullWindow
  Call FreeArt
FunctionEnd

Function WelcomeLeave
  ; Cancel in the Still running box aborts the leave: Welcome stays up.
  Call EnsureNotRunning
FunctionEnd

Function LicensePre
  ${If} $IsUpdate == 1
    Abort   ; updates skip the license page
  ${EndIf}
FunctionEnd

Function LicenseShow
  SendMessage $mui.LicensePage.TopText ${WM_SETFONT} $FontBody 1
  SendMessage $mui.LicensePage.Text ${WM_SETFONT} $FontBody 1
  ; Keep the license's URLs plain text: NSIS turns on URL detection, which
  ; draws them as links. Turn it off and clear the link effect it applied
  ; (CHARFORMAT2W with CFM_LINK and no effects, over all the text; the
  ; control takes it only while it isn't read-only).
  SendMessage $mui.LicensePage.LicenseText ${EM_AUTOURLDETECT} 0 0
  System::Alloc 116
  Pop $1
  System::Call '*$1(i 116, i 0x20, i 0)'
  SendMessage $mui.LicensePage.LicenseText ${EM_SETREADONLY} 0 0
  SendMessage $mui.LicensePage.LicenseText 0x444 4 $1   ; EM_SETCHARFORMAT, SCF_ALL
  SendMessage $mui.LicensePage.LicenseText ${EM_SETREADONLY} 1 0
  System::Free $1
  ; 6 px inner padding left and right, as the mockup.
  !insertmacro _Px $3 6
  IntOp $4 $3 << 16
  IntOp $4 $4 | $3
  SendMessage $mui.LicensePage.LicenseText ${EM_SETMARGINS} 3 $4   ; EC_LEFTMARGIN|EC_RIGHTMARGIN
FunctionEnd

Section "Yamabiko" SecMain
    SectionIn RO
    ; The status line reads "Copying files…" throughout; file names go to
    ; the details list only (listonly must come before SetOutPath, or the
    ; status shows "Output folder: …" first).
    SetDetailsPrint textonly
    DetailPrint "Copying files…"
    SetDetailsPrint listonly
    ; Remove the previous program tree before extracting so an upgrade
    ; cannot leave orphaned DLLs that shadow the new ones in the flattened
    ; _internal root (the DLL-shadowing class, via the upgrade
    ; path). User data ($INSTDIR\hf-cache, logs, *.yaml, state.json) lives
    ; outside _internal and is untouched; the running check (Welcome's
    ; leave function, or .onInit when silent) already ensured nothing is
    ; running, so these files are unlocked.
    retry_clean_internal:
    RMDir /r "$INSTDIR\_internal"
    ; RMDir /r half-completes SILENTLY when another process holds a file
    ; open inside _internal (antivirus on-access scan, Windows Search
    ; indexer, backup tools -- none of which EnsureNotRunning can see).
    ; Extracting over the leftovers would ship a mixed-version _internal
    ; where a stale DLL shadows a new one -- the DLL-shadowing class via
    ; the upgrade path. Postcondition-check the tree rather
    ; than IfErrors: RMDir on a missing dir (fresh install) is not a
    ; failure; what matters is that nothing of the old tree survives.
    ; (FindFirstFile treats *.* as *, and an empty-but-undeletable dir
    ; still matches via its "." entry -- conservative by design.)
    ${If} ${FileExists} "$INSTDIR\_internal\*.*"
        MessageBox MB_RETRYCANCEL|MB_ICONEXCLAMATION \
            "Couldn't replace the old version's files in $INSTDIR\_internal. Another program may have them open, such as an antivirus scan or a backup tool. Close it, then click Retry." \
            /SD IDCANCEL IDRETRY retry_clean_internal
        ; Messages reach the status line again, so the stop clears
        ; "Copying files…" under "Yamabiko isn't fully installed".
        SetDetailsPrint both
        Abort
    ${EndIf}
    Delete "$INSTDIR\${APP_EXE}"
    SetOutPath "$INSTDIR"
    File /r "..\dist\yamabiko\*.*"

    ; Start Menu shortcut
    CreateDirectory "$SMPROGRAMS\${APP_NAME}"
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" \
        "$INSTDIR\${APP_EXE}" "" "$INSTDIR\_internal\assets\icon.ico"
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk" \
        "$INSTDIR\Uninstall.exe"

    ; Uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"

    ; Add/Remove Programs registration
    WriteRegStr HKCU "${APP_REGKEY}" "DisplayName" "${APP_NAME}"
    WriteRegStr HKCU "${APP_REGKEY}" "DisplayVersion" "${APP_VERSION}"
    WriteRegStr HKCU "${APP_REGKEY}" "Publisher" "${APP_PUBLISHER}"
    WriteRegStr HKCU "${APP_REGKEY}" "DisplayIcon" "$INSTDIR\_internal\assets\icon.ico"
    WriteRegStr HKCU "${APP_REGKEY}" "UninstallString" "$INSTDIR\Uninstall.exe"
    WriteRegStr HKCU "${APP_REGKEY}" "InstallLocation" "$INSTDIR"
    WriteRegDWORD HKCU "${APP_REGKEY}" "NoModify" 1
    WriteRegDWORD HKCU "${APP_REGKEY}" "NoRepair" 1
    SetDetailsPrint both
SectionEnd

Function FinishCreate
  ; The last page's button reads Close unless told otherwise.
  SendMessage $mui.Button.Next ${WM_SETTEXT} 0 "STR:&Finish"
  nsDialogs::Create 1044
  Pop $Dlg
  ${If} $Dlg == error
    Abort
  ${EndIf}
  SetCtlColors $Dlg "" ${SKY_DUSK}
  InitPluginsDir
  !insertmacro _PageArt dusk seal-dusk
  !insertmacro _Label 70 28 400 28 $FontTitle ${INK} ${SKY_DUSK} "Yamabiko is installed."
  !insertmacro _Label 28 72 330 34 $FontBody ${INK} ${SKY_DUSK} "When it starts, a short setup picks your audio device and downloads the models."
  ; A themed checkbox can't be transparent: it sits on flat sky and takes
  ; that color as its background (Windows draws its text black).
  !insertmacro _Px $R5 28
  !insertmacro _Px $R6 118
  !insertmacro _Px $R7 110
  !insertmacro _Px $R8 17
  ${NSD_CreateCheckbox} $R5 $R6 $R7 $R8 "&Start Yamabiko"
  Pop $Chk
  SetCtlColors $Chk ${INK} ${SKY_DUSK}
  SendMessage $Chk ${WM_SETFONT} $FontBody 0
  ${NSD_Check} $Chk
  Call CreateArt
  Call muiPageLoadFullWindow
  nsDialogs::Show
  Call muiPageUnloadFullWindow
  Call FreeArt
FunctionEnd

Function FinishLeave
  ; Start Yamabiko as its Start menu shortcut does: the install folder as
  ; the working directory, no arguments. Leave functions never run in
  ; silent mode, so scripted installs never start it.
  ${NSD_GetState} $Chk $0
  ${If} $0 == ${BST_CHECKED}
    SetOutPath "$INSTDIR"
    Exec '"$INSTDIR\${APP_EXE}"'
  ${EndIf}
FunctionEnd

; --- Uninstaller --------------------------------------------------------------
Function un.onInit
  ; Silent uninstalls check here (exit code 2 while Yamabiko runs);
  ; interactive ones when Uninstall is clicked on the Remove page.
  ${If} ${Silent}
    Call un.EnsureNotRunning
  ${EndIf}
FunctionEnd

Function un.RemoveCreate
  nsDialogs::Create 1044
  Pop $Dlg
  ${If} $Dlg == error
    Abort
  ${EndIf}
  SetCtlColors $Dlg "" ${SKY_MIST}
  InitPluginsDir
  !insertmacro _PageArt mist seal-mist
  !insertmacro _Label 70 28 400 28 $FontTitle ${INK} ${SKY_MIST} "Remove Yamabiko?"
  !insertmacro _Label 28 72 330 34 $FontBody ${INK} ${SKY_MIST} "This removes the app and its Start menu shortcuts."
  !insertmacro _Label 28 98 330 34 $FontBody ${MUTED} ${SKY_MIST} "Your downloaded models and settings stay on your PC."
  Call un.CreateArt
  Call un.muiPageLoadFullWindow
  nsDialogs::Show
  Call un.muiPageUnloadFullWindow
  Call un.FreeArt
FunctionEnd

Function un.RemoveLeave
  ; Cancel in the Still running box aborts the leave: Remove stays up.
  Call un.EnsureNotRunning
FunctionEnd

Section "Uninstall"
    SetDetailsPrint textonly
    DetailPrint "Removing files…"
    SetDetailsPrint listonly
    ; PyInstaller 6.x onedir puts yamabiko.exe at the root and everything
    ; else under _internal/. So uninstall nukes yamabiko.exe + _internal/
    ; plus the standalone Uninstall.exe we wrote — then tries to remove
    ; $INSTDIR without /r, which fails silently if user data remains
    ; (models/, logs/, config.yaml, user_overrides.yaml, state.json).

    Delete "$INSTDIR\${APP_EXE}"
    Delete "$INSTDIR\Uninstall.exe"
    RMDir /r "$INSTDIR\_internal"

    ; Legacy layout (PyInstaller <6.0 onedir) — files at root. Harmless
    ; no-ops on a 6.x install; covers in-place upgrades from an older
    ; Yamabiko that might have used the flat layout.
    Delete "$INSTDIR\*.dll"
    Delete "$INSTDIR\*.pyd"
    Delete "$INSTDIR\LICENSE"
    Delete "$INSTDIR\NOTICE"
    Delete "$INSTDIR\README.md"
    Delete "$INSTDIR\CHANGELOG.md"

    ; Try to remove $INSTDIR — succeeds only if no user data is present.
    ; The multi-GB HF model cache + config/logs deliberately survive so a
    ; reinstall keeps the user's models + settings; the Removed page shows
    ; what stayed, its size and where.
    RMDir "$INSTDIR"

    Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
    Delete "$SMPROGRAMS\${APP_NAME}\Uninstall.lnk"
    RMDir "$SMPROGRAMS\${APP_NAME}"

    DeleteRegKey HKCU "${APP_REGKEY}"

    ; Measure what stayed for the Removed page while the progress page is
    ; still up (about 1 ms per file). Silent uninstalls show no page.
    StrCpy $Kept 0
    ${IfNot} ${Silent}
    ${AndIf} ${FileExists} "$INSTDIR\*.*"
        StrCpy $Kept 1
        StrCpy $R0 "$INSTDIR"
        Call un.DirSize64
        Call un.FormatSize
    ${EndIf}
    SetDetailsPrint both
    ; Go on to the Removed page by itself.
    SetAutoClose true
SectionEnd

Function un.DirSize64
  ; In: $R0 folder. Out: $R1 its total size in bytes (64-bit decimal).
  ; FileFunc's GetSize and Locate size files with FileSeek, a 32-bit
  ; position, and drop every file of 2 GiB or more (model files are 5 GB).
  ; This reads each entry's 64-bit size with GetFileAttributesExW and sums
  ; with System::Int64Op. Reparse points (junctions, symlinks) are skipped,
  ; so nothing outside the folder or twice counted adds to the size, and an
  ; entry that can't be read counts 0. Every register but $R1 is kept.
  System::Store "s"
  StrCpy $1 0   ; bytes
  StrCpy $3 1   ; folders waiting on the stack
  Push $R0
  System::Call '*(i,i,i,i,i,i,i,i,i) p .r9'   ; WIN32_FILE_ATTRIBUTE_DATA, 36 bytes
  ${DoWhile} $3 > 0
    Pop $4
    IntOp $3 $3 - 1
    FindFirst $5 $6 "$4\*.*"
    ${DoWhile} $6 != ""
      ${If} $6 != "."
      ${AndIf} $6 != ".."
        System::Call 'kernel32::GetFileAttributesExW(w "$4\$6", i 0, p r9) i .r0'
        ${If} $0 <> 0
          ; attributes at offset 0, size high/low at 28/32
          System::Call '*$9(i .r0, i, i, i, i, i, i, i .r7, i .r8)'
          IntOp $2 $0 & 0x400   ; FILE_ATTRIBUTE_REPARSE_POINT
          ${If} $2 = 0
            IntOp $2 $0 & 0x10   ; FILE_ATTRIBUTE_DIRECTORY
            ${If} $2 <> 0
              Push "$4\$6"
              IntOp $3 $3 + 1
            ${Else}
              System::Int64Op $7 << 32
              Pop $7
              System::Int64Op $8 & 0xFFFFFFFF
              Pop $8
              System::Int64Op $1 + $7
              Pop $1
              System::Int64Op $1 + $8
              Pop $1
            ${EndIf}
          ${EndIf}
        ${EndIf}
      ${EndIf}
      FindNext $5 $6
    ${Loop}
    FindClose $5
  ${Loop}
  System::Free $9
  Push $1
  System::Store "l"
  Pop $R1
FunctionEnd

Function un.FormatSize
  ; In: $R1 bytes (64-bit). Out: $SizeText, 1024-based like Explorer:
  ; "6.8 GB" from 1 GB up (tenths, rounded half up), else whole MB rounded
  ; down, at least "1 MB". Int64Op compares only with < = > (no >=).
  System::Store "s"
  System::Int64Op $R1 > 1073741823
  Pop $0
  ${If} $0 == 1
    System::Int64Op $R1 * 10
    Pop $1
    System::Int64Op $1 + 536870912
    Pop $1
    System::Int64Op $1 / 1073741824
    Pop $1
    IntOp $2 $1 / 10
    IntOp $3 $1 % 10
    StrCpy $SizeText "$2.$3 GB"
  ${Else}
    System::Int64Op $R1 / 1048576
    Pop $1
    ${If} $1 < 1
      StrCpy $1 1
    ${EndIf}
    StrCpy $SizeText "$1 MB"
  ${EndIf}
  System::Store "l"
FunctionEnd

Function un.RemovedCreate
  nsDialogs::Create 1044
  Pop $Dlg
  ${If} $Dlg == error
    Abort
  ${EndIf}
  SetCtlColors $Dlg "" ${SKY_MIST}
  InitPluginsDir
  !insertmacro _PageArt removed seal-mist
  !insertmacro _Label 70 28 400 28 $FontTitle ${INK} ${SKY_MIST} "Yamabiko has been removed."
  ; With nothing kept the folder is gone: the title alone.
  ${If} $Kept == 1
    !insertmacro _Label 28 72 330 17 $FontBody ${INK} ${SKY_MIST} "Your models and settings ($SizeText) are still in:"
    !insertmacro _Label 28 90 440 17 $FontBody ${INK} ${SKY_MIST} "$INSTDIR"
    ; A long path keeps its ends ("C:\Users\…\Yamabiko"); an "&" in it stays.
    ${NSD_AddStyle} $R9 ${SS_PATHELLIPSIS}
    ${NSD_AddStyle} $R9 ${SS_NOPREFIX}
    !insertmacro _Label 28 116 330 34 $FontBody ${MUTED} ${SKY_MIST} "Delete the folder to free up the space, or keep it for a quicker reinstall."
    !insertmacro _Px $R5 28
    !insertmacro _Px $R6 158
    !insertmacro _Px $R7 96
    !insertmacro _Px $R8 17
    ${NSD_CreateLink} $R5 $R6 $R7 $R8 "Open the folder"
    Pop $R9
    SetCtlColors $R9 ${ACCENT} transparent
    SendMessage $R9 ${WM_SETFONT} $FontLink 0
    ${NSD_OnClick} $R9 un.OpenFolder
  ${EndIf}
  Call un.CreateArt
  Call un.muiPageLoadFullWindow
  nsDialogs::Show
  Call un.muiPageUnloadFullWindow
  Call un.FreeArt
FunctionEnd

Function un.OpenFolder
  Pop $0   ; the link (nsDialogs passes the clicked control)
  ExecShell "open" "$INSTDIR"
FunctionEnd
