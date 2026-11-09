"""Meta-tests for the NSIS installer script (installer/yamabiko.nsi).

The .nsi is install-surface infrastructure with contracts the rest of the
project relies on but that the unit suite otherwise can't see (the real
validation is `makensis` + the installer-smoke workflow + the manual
wizard E2E). These structural guards exist because a prior audit found
the install/upgrade/uninstall lifecycle naïve: a bare `File /r` overwrite
with no running-process detection, no upgrade handling, and a custom-
install-dir vs hardcoded-data-dir divergence.

Each assertion is paired with a one-line seeded defect: revert that one
line in the .nsi and the test goes red. They guard against a future edit
silently dropping a lifecycle safety.
"""
from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NSI = _REPO_ROOT / "installer" / "yamabiko.nsi"


def _nsi_text() -> str:
    return _NSI.read_text(encoding="utf-8")


def _macro_body(text: str) -> str:
    """The EnsureNotRunning function template (the !macro body), so probe
    assertions see the actual guard code -- not a comment elsewhere that
    would survive deleting the probe."""
    m = re.search(r"!macro _EnsureNotRunning.*?!macroend", text, re.S)
    return m.group(0) if m else ""


def _function_body(text: str, name: str) -> str:
    m = re.search(rf"Function\s+{re.escape(name)}\b(.*?)FunctionEnd", text, re.S)
    return m.group(1) if m else ""


def _code(text: str) -> str:
    """The script without its comments (``/* ... */`` blocks and full
    ";" or "#" lines), so a commented-out statement doesn't count as
    present."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith((";", "#"))
    )


def test_nsi_exists() -> None:
    assert _NSI.is_file(), f"missing installer script at {_NSI}"


def test_nsi_versioninfo_binary_quad_comes_from_single_define() -> None:
    """VIProductVersion / VIFileVersion must reference the one
    ``${APP_VERSION_QUAD}`` define rather than repeating hardcoded quad
    literals: with literals, a version bump edits APP_VERSION (used by
    the VIAddVersionKey string keys) while the binary VERSIONINFO
    silently stays at the old quad, and the shipped setup.exe carries a
    string/binary version mismatch that AV / reputation tooling reads
    off the PE.

    Seeded defect: revert ``VIProductVersion "${APP_VERSION_QUAD}"`` to
    the literal ``VIProductVersion "0.1.0.0"`` -> this fails.
    """
    text = _nsi_text()
    assert re.search(r'(?m)^\s*!define\s+APP_VERSION_QUAD\s+"[^"]+"', text), (
        "no APP_VERSION_QUAD define; the binary VERSIONINFO quad has no "
        "single source of truth next to APP_VERSION"
    )
    for key in ("VIProductVersion", "VIFileVersion"):
        m = re.search(rf'(?m)^\s*{key}\s+"([^"]+)"', text)
        assert m, f"no {key} statement in the .nsi"
        assert m.group(1) == "${APP_VERSION_QUAD}", (
            f'{key} is {m.group(1)!r}, not "${{APP_VERSION_QUAD}}"; a '
            f"hardcoded quad literal drifts silently on the next version bump"
        )


def test_nsi_app_version_quad_equals_numeric_prefix_of_app_version() -> None:
    """APP_VERSION_QUAD must equal APP_VERSION's X.Y.Z numeric prefix
    plus a ".0" build part: the PE VERSIONINFO quad requires 4 numeric
    parts and cannot carry the "-beta" pre-release suffix, so the two
    defines are separate values that must be bumped together.
    scripts/preflight.py enforces the same relationship at the release
    gate; this is the per-PR unit-suite tripwire.

    Seeded defect: bump ``!define APP_VERSION`` to "0.2.0-beta" without
    touching APP_VERSION_QUAD -> this fails.
    """
    text = _nsi_text()
    app_version = re.search(r'!define\s+APP_VERSION\s+"([^"]+)"', text)
    quad = re.search(r'!define\s+APP_VERSION_QUAD\s+"([^"]+)"', text)
    assert app_version, "no APP_VERSION define in the .nsi"
    assert quad, "no APP_VERSION_QUAD define in the .nsi"
    prefix = re.match(r"(\d+)\.(\d+)\.(\d+)", app_version.group(1))
    assert prefix, (
        f"APP_VERSION {app_version.group(1)!r} has no X.Y.Z numeric prefix"
    )
    expected = ".".join(prefix.groups()) + ".0"
    assert quad.group(1) == expected, (
        f"APP_VERSION_QUAD {quad.group(1)!r} != {expected!r} (numeric prefix "
        f"of APP_VERSION {app_version.group(1)!r} + '.0'); the installer PE "
        f"would ship a stale binary version"
    )


def test_nsi_blocks_install_uninstall_while_app_running() -> None:
    """Install + uninstall must refuse to run while yamabiko.exe or the
    bundled llama-server.exe is live -- their loaded DLLs are locked and
    File /r / RMDir /r would half-complete and corrupt the install.

    Seeded defect: delete the EnsureNotRunning tasklist probe (or its
    Abort) -> this test fails.
    """
    text = _nsi_text()
    assert "EnsureNotRunning" in text, (
        "no EnsureNotRunning guard; install/uninstall over a running app "
        "will corrupt the bundle"
    )
    body = _macro_body(text)
    assert body, "no _EnsureNotRunning macro found"
    # Assert on the GUARD BODY, not the whole file -- a comment mentioning
    # llama-server.exe must not satisfy this after the probe is deleted.
    assert "tasklist" in body, "guard does not probe tasklist"
    assert body.lower().count("llama-server.exe") >= 1, (
        "guard body does not probe llama-server.exe"
    )
    assert ("${APP_EXE}" in body) or ("yamabiko.exe" in body.lower()), (
        "guard body does not probe the app exe"
    )
    assert "Abort" in body, "running-process guard never aborts the operation"


def test_nsi_llama_probe_is_scoped_to_install_dir() -> None:
    """The llama-server.exe pre-install/uninstall probe must filter by
    executable path under $INSTDIR, not bare image name: llama-server.exe
    is the generic llama.cpp server binary shipped by unrelated apps
    (LM Studio, ollama frontends, dev checkouts), and an unscoped
    tasklist image-name match blocks install/uninstall for a process
    that has nothing to do with Yamabiko. watchdog.py's orphan sweep
    already path-scopes its match set for exactly this reason; the
    installer guard must mirror it.

    The yamabiko.exe probe may stay image-name based (the name is
    unique to this app).

    Seeded defect: revert the llama probe to
    ``tasklist /FI "IMAGENAME eq llama-server.exe"`` -> this fails.
    """
    body = _macro_body(_nsi_text())
    assert body, "no _EnsureNotRunning macro found"
    probe_lines = [
        ln for ln in body.splitlines()
        if "nsExec" in ln and "llama-server" in ln
    ]
    assert probe_lines, "guard body has no llama-server probe at all"
    for ln in probe_lines:
        assert "$INSTDIR" in ln, (
            "llama-server probe is not scoped to $INSTDIR; any "
            "unrelated llama-server.exe on the machine blocks "
            "install/uninstall"
        )
        assert (".Path" in ln) or ("ExecutablePath" in ln), (
            "llama-server probe does not filter on the process "
            "executable path; image-name matching false-positives on "
            "other llama.cpp-based apps"
        )
        assert "tasklist" not in ln, (
            "llama-server probe still uses tasklist, which cannot "
            "filter by executable path"
        )


def _silent_branch(body: str) -> str:
    """The ``${If} ${Silent}`` branch of a function body, up to its
    ``${Else}``/``${ElseIf}`` or ``${EndIf}``."""
    m = re.search(r"\$\{If\}\s+\$\{Silent\}(.*?)\$\{(?:Else|EndIf)", body, re.S)
    return m.group(1) if m else ""


def test_nsi_guard_runs_for_both_install_and_uninstall() -> None:
    """The running check must fire on every install and uninstall path.
    Silent runs check in .onInit / un.onInit, inside ``${If} ${Silent}``
    (the box answers Cancel by itself -> exit code 2, which ci-smoke
    Scenario B asserts). Interactive runs check in the leave functions of
    the Welcome and Remove pages, so the box belongs to the window and
    Cancel keeps the page up. A call in .onInit outside the Silent branch
    would raise the box before the window opens. A user hitting Uninstall
    while translating is just as destructive as a re-install over a
    running app.

    Seeded defect: delete (or comment out, with ";" or "/* */") the call in
    un.RemoveLeave (interactive uninstall while running unguarded) or in
    .onInit's ${Silent} branch (RELEASE.md drill row 9's seed), or move
    .onInit's call out of that branch (also into an ``${Else}``) -> this
    test fails.
    """
    text = _code(_nsi_text())
    for init, call in (
        (".onInit", "Call EnsureNotRunning"),
        ("un.onInit", "Call un.EnsureNotRunning"),
    ):
        body = _function_body(text, init)
        assert call in _silent_branch(body), (
            f"{init} does not call the guard inside ${{If}} ${{Silent}}; "
            f"silent runs while Yamabiko is running are unguarded"
        )
        assert body.count(call) == 1, (
            f"{init} also calls the guard outside the Silent branch; "
            f"interactive runs would get the box before the window opens"
        )
    for page, leave, call in (
        (r"Page\s+custom\s+WelcomeCreate\s+WelcomeLeave", "WelcomeLeave",
         "Call EnsureNotRunning"),
        (r"UninstPage\s+custom\s+un\.RemoveCreate\s+un\.RemoveLeave",
         "un.RemoveLeave", "Call un.EnsureNotRunning"),
    ):
        # Wired AND reached (TESTING.md section 3.9): the leave function is
        # registered on its page and its body calls the guard.
        assert re.search(rf"(?m)^\s*{page}\s*$", text), (
            f"{leave} is not the leave function of its page; the "
            f"interactive running check never runs"
        )
        assert call in _function_body(text, leave), (
            f"{leave} does not call the guard; an interactive run while "
            f"Yamabiko is running is unguarded"
        )


def test_nsi_omits_directory_page_to_keep_install_dir_eq_data_dir() -> None:
    """paths.app_data_dir() is hardcoded to $LOCALAPPDATA\\Yamabiko, so the
    install dir must stay there too -- otherwise user data + the
    uninstaller diverge. The MUI directory page (which lets a user pick
    an arbitrary install dir) must be omitted.

    Seeded defect: re-add `!insertmacro MUI_PAGE_DIRECTORY` -> this fails.
    """
    text = _nsi_text()
    # Assert the DIRECTIVE is absent. A comment mentioning the macro name
    # (we keep one explaining why it was removed) must not trip this.
    assert not re.search(r"(?m)^\s*!insertmacro\s+MUI_PAGE_DIRECTORY\b", text), (
        "MUI_PAGE_DIRECTORY is inserted: a user-chosen install dir diverges "
        "from the hardcoded $LOCALAPPDATA data dir; uninstaller cleans the "
        "wrong place"
    )


def test_nsi_reuses_prior_install_location_on_reinstall() -> None:
    """A re-install must default to the previously-registered location so
    an upgrade never creates a second copy at a different path.

    Seeded defect: remove the InstallDirRegKey line -> reinstall ignores
    the prior InstallLocation -> this test fails.
    """
    text = _nsi_text()
    assert "InstallDirRegKey" in text, (
        "no InstallDirRegKey: a reinstall ignores the user's prior install "
        "location and can create a duplicate install"
    )


def test_nsi_clears_internal_before_extract() -> None:
    """An upgrade must remove the old _internal tree before extracting the
    new one, or orphaned DLLs accumulate and a stale cuBLAS/cuDNN copy can
    shadow the new one in the flattened _internal root.

    Seeded defect: remove the `RMDir /r "$INSTDIR\\_internal"` before
    `File /r` -> stale orphans survive an upgrade -> this test fails.
    """
    text = _nsi_text()
    # The cleanup must precede the extraction.
    clean = text.find(r'RMDir /r "$INSTDIR\_internal"')
    extract = text.find(r'File /r "..\dist\yamabiko')
    assert clean != -1, "no pre-extraction RMDir /r of _internal"
    assert extract != -1, "could not locate the File /r extraction line"
    assert clean < extract, (
        "the _internal cleanup must run BEFORE File /r, or orphaned DLLs "
        "from the prior version survive the upgrade"
    )


def test_nsi_aborts_when_internal_cleanup_leaves_files() -> None:
    """SecMain must verify the pre-extraction ``RMDir /r _internal``
    actually emptied the tree, and Abort if files survived. NSIS RMDir
    half-completes silently when another process holds a file open
    inside _internal (antivirus on-access scan, Windows Search indexer,
    backup tools -- none of which appear as yamabiko.exe or
    llama-server.exe, so EnsureNotRunning cannot see them). Extracting
    over the leftovers produces a mixed-version _internal where a stale
    DLL shadows the new one.

    The guard must be a POSTCONDITION check (``${If} ${FileExists}
    "$INSTDIR\\_internal\\*.*"`` after the RMDir), not IfErrors: RMDir
    on a missing dir (fresh install) must not abort, and what matters
    is that no old file survives, not which API call failed.

    Seeded defect: delete the ``${If} ${FileExists}`` guard (or its
    Abort) between the RMDir and File /r -> this fails.
    """
    text = _nsi_text()
    sec_match = re.search(r'Section "Yamabiko".*?SectionEnd', text, re.S)
    assert sec_match, "could not locate the SecMain section body"
    sec = sec_match.group(0)

    clean = sec.find(r'RMDir /r "$INSTDIR\_internal"')
    extract = sec.find(r'File /r')
    assert clean != -1 and extract != -1, "SecMain lost RMDir or File /r"

    guard_re = re.compile(
        r'\$\{If\}\s+\$\{FileExists\}\s+"\$INSTDIR\\_internal\\\*\.\*"'
        r'(?P<block>.*?)\$\{EndIf\}',
        re.S,
    )
    guard = guard_re.search(sec)
    assert guard, (
        "SecMain has no FileExists postcondition after the _internal "
        "RMDir: a partial deletion (locked DLL) silently continues into "
        "File /r and ships a mixed-version _internal"
    )
    assert clean < guard.start() < extract, (
        "the _internal FileExists guard must sit between the RMDir "
        "cleanup and the File /r extraction"
    )
    assert "Abort" in guard.group("block"), (
        "the _internal leftover guard never Aborts; the install would "
        "warn (at best) and still extract over stale files"
    )


def test_nsi_stopped_install_does_not_keep_the_copying_status() -> None:
    """SecMain sends its messages to the details list only while it copies
    (``SetDetailsPrint listonly``), so the status line keeps "Copying
    files…". An ``Abort`` in that mode leaves the status line as it is:
    the progress page would read "Yamabiko isn't fully installed" over
    "Copying files…". Every Abort in the section must come after a
    ``SetDetailsPrint both``, which blanks the status line.

    Seeded defect: delete the ``SetDetailsPrint both`` before the Abort in
    the Old files locked branch -> this fails.
    """
    sec = re.search(r'Section "Yamabiko"(.*?)SectionEnd', _code(_nsi_text()), re.S)
    assert sec, "could not locate the SecMain section body"
    body = sec.group(1)
    aborts = [m.start() for m in re.finditer(r"(?m)^\s*Abort\b", body)]
    assert aborts, "SecMain no longer stops when the old files are locked"
    for at in aborts:
        modes = re.findall(r"(?m)^\s*SetDetailsPrint\s+(\w+)", body[:at])
        assert modes[-1:] == ["both"], (
            f"an Abort in SecMain runs with SetDetailsPrint {modes[-1:]}: the "
            'stopped install keeps "Copying files…" as its status line'
        )


def test_nsi_declares_dpi_awareness() -> None:
    """The installer must be DPI aware: otherwise Windows draws it at
    96 DPI and stretches the whole window (blurry text and art at 125 to
    200 %), and the system DPI it reads is always 96, so the 100 %
    bitmaps are picked on every display.

    Seeded defect: delete ``ManifestDPIAware true`` -> this fails.
    """
    assert re.search(r"(?m)^\s*ManifestDPIAware\s+true\b", _nsi_text()), (
        "no ManifestDPIAware true: Windows stretches the installer window "
        "and the 150/200 % bitmaps are never picked"
    )


def test_nsi_license_page_shows_the_rtf_copy() -> None:
    """The license page must load installer/LICENSE.rtf (formatted text),
    not the Markdown LICENSE, which shows as raw "#" and "**" markup.

    Seeded defect: revert to ``!insertmacro MUI_PAGE_LICENSE "..\\LICENSE"``
    -> this fails.
    """
    m = re.search(
        r'(?m)^\s*!insertmacro\s+MUI_PAGE_LICENSE\s+"([^"]+)"', _nsi_text()
    )
    assert m, "no MUI_PAGE_LICENSE page"
    assert m.group(1) == "LICENSE.rtf", (
        f"the license page loads {m.group(1)!r}, not LICENSE.rtf; the "
        f"license shows as raw Markdown"
    )


def test_nsi_measures_the_kept_folder_with_64bit_sizes() -> None:
    """The Removed page's size must count files of 2 GiB and more (the
    translation model is about 5 GB). FileFunc's GetSize and Locate size
    each file with FileSeek, a 32-bit position, and silently drop those
    files, so a 7.5 GB folder read "169 KB" in the spike. The walk must
    read 64-bit sizes and skip reparse points (a junction would count
    another folder, or the same files twice).

    Seeded defect: measure with ``${un.GetSize}`` instead of
    un.DirSize64, drop the reparse-point test, or drop the high DWORD's
    shift or sum (a file of 4 GiB or more then counts only its low 32
    bits: the 5 GB model reads about 1 GB) -> this fails.
    """
    text = _nsi_text()
    code = _code(text)
    assert not re.search(r"\$\{(un\.)?(GetSize|Locate)\}", code), (
        "the script sizes files with FileFunc GetSize/Locate, which drop "
        "every file of 2 GiB or more"
    )
    sec = re.search(r'Section\s+"Uninstall"(.*?)SectionEnd', code, re.S)
    assert sec and "Call un.DirSize64" in sec.group(1), (
        "the uninstall section does not measure the kept folder with "
        "un.DirSize64"
    )
    walk = _function_body(code, "un.DirSize64")
    for needle, why in (
        ("GetFileAttributesExW", "64-bit size per file"),
        ("System::Int64Op", "64-bit sum"),
        ("0x400", "FILE_ATTRIBUTE_REPARSE_POINT skip"),
        ("System::Int64Op $7 << 32", "high DWORD shift"),
        ("System::Int64Op $1 + $7", "high DWORD sum"),
    ):
        assert needle in walk, f"un.DirSize64 lost its {why} ({needle})"


def test_nsi_starts_yamabiko_only_from_the_finish_page() -> None:
    """Yamabiko may start only from the Finish page's leave function, and
    only when Start Yamabiko is ticked: leave functions never run in silent
    mode, so scripted and CI installs never start it, and it starts after
    the running check and the copy are done.

    Seeded defect: move the Exec into Section "Yamabiko" (every silent
    install starts the app) or drop the checkbox test -> this fails.
    """
    text = _nsi_text()
    code = _code(text)
    launches = re.findall(
        r"(?m)^\s*(?:Exec|ExecWait|ExecShell)\b.*\$\{APP_EXE\}.*$", code
    )
    assert len(launches) == 1, f"expected one launch of the app, found {launches}"
    leave = _function_body(code, "FinishLeave")
    assert launches[0] in leave, "the app is started outside FinishLeave"
    assert "${NSD_GetState} $Chk" in leave and "${BST_CHECKED}" in leave, (
        "FinishLeave starts Yamabiko without checking Start Yamabiko"
    )
    assert re.search(r"(?m)^\s*Page\s+custom\s+FinishCreate\s+FinishLeave\s*$", text), (
        "FinishLeave is not the Finish page's leave function"
    )


# Spec "Surface 7" copy (final), the copy the spec leaves open (plan
# decisions), and the frame strings, exactly as the script writes them.
_COPY = (
    'Name "${APP_NAME}"',
    'Caption "Install ${APP_NAME} ${APP_VERSION_DISPLAY}"',
    'UninstallCaption "Uninstall ${APP_NAME}"',
    'BrandingText " "',
    '!searchreplace APP_VERSION_DISPLAY "${APP_VERSION}" "-" " "',
    '"Welcome to Yamabiko"',
    '"Yamabiko listens to Japanese audio on your PC and shows subtitles with furigana and an English translation."',
    '"The first time you start it, a short setup downloads the models it needs (up to 7 GB)."',
    '"Update Yamabiko"',
    '"Yamabiko $OldVersion is installed. This updates it to ${APP_VERSION_DISPLAY}."',
    '"Your settings and downloaded models stay."',
    '"STR:&Update"',
    '"License"',
    '"Free for personal, noncommercial use."',
    '"Yamabiko is licensed under PolyForm Noncommercial 1.0.0."',
    '"Click I Agree to accept the license and install Yamabiko."',
    '!define MUI_PAGE_HEADER_TEXT "Installing Yamabiko"',
    '!define MUI_PAGE_HEADER_SUBTEXT "This takes a minute or two."',
    '!define MUI_INSTFILESPAGE_FINISHHEADER_TEXT "Installing Yamabiko"',
    '!define MUI_INSTFILESPAGE_FINISHHEADER_SUBTEXT "This takes a minute or two."',
    '"Copying files…"',
    '!define MUI_INSTFILESPAGE_ABORTHEADER_TEXT "Yamabiko isn\'t fully installed"',
    '!define MUI_INSTFILESPAGE_ABORTHEADER_SUBTEXT "Run the installer again to finish."',
    '"Yamabiko is installed."',
    '"When it starts, a short setup picks your audio device and downloads the models."',
    '"&Start Yamabiko"',
    '"STR:&Finish"',
    '"Yamabiko is still running. Close Yamabiko setup, or right-click its tray icon and choose Quit Yamabiko, then click Retry."',
    '"Yamabiko\'s translator is still running. Quit Yamabiko, or end llama-server.exe in Task Manager, then click Retry."',
    '"Couldn\'t replace the old version\'s files in $INSTDIR\\_internal. Another program may have them open, such as an antivirus scan or a backup tool. Close it, then click Retry."',
    '"Remove Yamabiko?"',
    '"This removes the app and its Start menu shortcuts."',
    '"Your downloaded models and settings stay on your PC."',
    '!define MUI_PAGE_HEADER_TEXT "Removing Yamabiko"',
    '!define MUI_PAGE_HEADER_SUBTEXT "This takes a moment."',
    '!define MUI_INSTFILESPAGE_FINISHHEADER_TEXT "Removing Yamabiko"',
    '!define MUI_INSTFILESPAGE_FINISHHEADER_SUBTEXT "This takes a moment."',
    '"Removing files…"',
    '"Removal stopped"',
    '"Run the uninstaller again to finish."',
    '"Yamabiko has been removed."',
    '"Your models and settings ($SizeText) are still in:"',
    '"Delete the folder to free up the space, or keep it for a quicker reinstall."',
    '"Open the folder"',
)


def test_nsi_shows_the_approved_copy() -> None:
    """Every string the installer and uninstaller show is the approved copy
    (design spec Surface 7, quoted copy final; plain, American, no em
    dashes), and the file keeps its UTF-8 BOM: without it makensis reads
    the script in the ANSI code page and "Copying files…" shows garbled.

    Seeded defect: reword a page line or a progress page header (the
    finish header next to it repeats it), comment out
    ``DetailPrint "Copying files…"``, bring back the old "Setup could not
    remove ..." box, put an em dash in a shown string, or save the file
    without its BOM -> this fails.
    """
    raw = _NSI.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "yamabiko.nsi lost its UTF-8 BOM"
    code = _code(_nsi_text())
    missing = [s for s in _COPY if s not in code]
    assert not missing, "approved copy missing from yamabiko.nsi:\n" + "\n".join(missing)
    shown = re.findall(r'"([^"]*)"', code)
    dashed = [s for s in shown if "\u2014" in s]
    assert not dashed, f"em dash in shown text: {dashed}"


def test_nsi_bitmaps_and_license_it_embeds_exist() -> None:
    """Every bitmap and the license file the script embeds must exist
    under installer/: makensis stops on a missing one, but only after the
    ~25 minute bundle build, and CI never regenerates them.

    Seeded defect: rename a page's art (``!insertmacro _PageArt dusk
    seal-dusk`` -> ``dusk2``) or delete installer/LICENSE.rtf -> this fails.
    """
    text = _nsi_text()
    macro = re.search(r"!macro _PickArt NAME(.*?)!macroend", text, re.S)
    assert macro, "no _PickArt macro"
    scales = set(re.findall(r'"art\\\$\{NAME\}-(\d+)\.bmp"', macro.group(1)))
    assert scales == {"100", "150", "200"}, f"_PickArt scales {sorted(scales)}"
    # Pages name their landscape and seal once, in _PageArt; the header
    # art is picked by name.
    names = set(re.findall(r"(?m)^\s*!insertmacro\s+_PickArt\s+([^\s$]\S*)", text))
    for page_art in re.findall(r"(?m)^\s*!insertmacro\s+_PageArt\s+(\S+)\s+(\S+)", text):
        names.update(page_art)
    assert {"mist", "removed", "dusk", "seal-mist", "seal-dusk", "header"} <= names, names
    paths = {f"art/{n}-{s}.bmp" for n in names for s in scales}
    paths.update(
        p.replace("\\", "/")
        for p in re.findall(r'(?m)^\s*!define\s+MUI_HEADERIMAGE_BITMAP\s+"([^"]+)"', text)
    )
    license_page = re.search(r'(?m)^\s*!insertmacro\s+MUI_PAGE_LICENSE\s+"([^"]+)"', text)
    assert license_page, "no MUI_PAGE_LICENSE page"
    paths.add(license_page.group(1))
    missing = sorted(p for p in paths if not (_NSI.parent / p).is_file())
    assert not missing, f"yamabiko.nsi embeds files that don't exist: {missing}"
