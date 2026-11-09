"""Real Windows desktop: show_in_folder_outside_job opens Explorer with the
file selected (@pytest.mark.real).

This needs a desktop that shows Explorer's selection, so it runs in the
real-gui release phase (scripts/release-gates.json) and is kept out of
real-deps and the CI subset (.github/workflows/ci.yml). On the GitHub-hosted
windows-2025 runner the window opens on the right folder but its selection
stays empty for the whole 20 s wait, while at a dev box desktop the file is
selected by the first read. Run it at an unlocked desktop:

    .venv\\Scripts\\python.exe -m pytest tests/lifecycle/test_real_show_in_folder.py -m real -v

It opens one Explorer window and closes it.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.real,
    pytest.mark.skipif(not sys.platform.startswith("win"), reason="Explorer is Windows-only"),
]


_SHELL_WINDOWS_PS = (
    "$shell = New-Object -ComObject Shell.Application; "
    "foreach ($w in @($shell.Windows())) { try { "
    "$folder = $w.Document.Folder.Self.Path; "
    "$picked = @($w.Document.SelectedItems() | ForEach-Object { $_.Path }) -join '|'; "
    "Write-Output ($folder + [char]9 + $picked) } catch {} }"
)


def _explorer_windows() -> list[tuple[str, list[str]]]:
    """(folder path, selected item paths) of every open Explorer window."""
    out = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _SHELL_WINDOWS_PS],
        capture_output=True, text=True, timeout=30,
        creationflags=subprocess.CREATE_NO_WINDOW,
    ).stdout
    rows = []
    for line in out.splitlines():
        folder, _, picked = line.partition("\t")
        rows.append((folder, [p for p in picked.split("|") if p]))
    return rows


def test_real_show_in_folder_opens_explorer_with_the_file_selected(tmp_path):
    """The real explorer.exe call site: a folder and a file name with spaces
    (a OneDrive desktop has them) open in an Explorer window with the file
    selected, read back from the Shell.Application window list.

    Fails red if: the command is ["explorer.exe", f"/select,{path}"] (the
    whole token gets quoted; Explorer opens Documents), or "/select," is
    dropped (the folder opens without the selection)."""
    import time
    import uuid

    import yamabiko.lifecycle.jobs as jobs_mod

    folder = tmp_path / f"show in folder {uuid.uuid4().hex[:8]}"
    folder.mkdir()
    target = folder / "yamabiko diagnostics.zip"
    target.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    jobs_mod.show_in_folder_outside_job(target)
    try:
        deadline = time.monotonic() + 20.0
        seen: list = []
        while time.monotonic() < deadline:
            seen = _explorer_windows()
            if any(
                shown.lower().endswith(folder.name.lower())
                and any(p.lower().endswith(target.name.lower()) for p in picked)
                for shown, picked in seen
            ):
                break
            time.sleep(0.5)
        else:
            pytest.fail(f"no Explorer window showed {target.name} selected in {folder.name}: {seen}")
    finally:
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
             "$shell = New-Object -ComObject Shell.Application; "
             "foreach ($w in @($shell.Windows())) { try { "
             f"if ($w.Document.Folder.Self.Path -like '*{folder.name}') {{ $w.Quit() }} "
             "} catch {} }"],
            capture_output=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW,
        )
