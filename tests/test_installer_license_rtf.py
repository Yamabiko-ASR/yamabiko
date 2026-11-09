"""The installer's license page text (installer/LICENSE.rtf).

installer/license_rtf.py converts LICENSE's Markdown to the RTF the license
page shows; the RTF is committed so the installer build needs no generator.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GENERATOR = _REPO_ROOT / "installer" / "license_rtf.py"
_LICENSE = _REPO_ROOT / "LICENSE"
_RTF = _REPO_ROOT / "installer" / "LICENSE.rtf"


def _convert(markdown: str) -> str:
    spec = importlib.util.spec_from_file_location("license_rtf", _GENERATOR)
    assert spec is not None and spec.loader is not None, f"cannot load {_GENERATOR}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert(markdown)


def _lines(text: str) -> list[str]:
    # Newline-safe: a checkout may turn LF into CRLF.
    return text.replace("\r\n", "\n").split("\n")


def test_committed_license_rtf_matches_license() -> None:
    """The RTF the installer ships is the current LICENSE, converted.

    Seeded defect: edit a sentence in LICENSE without running
    installer/license_rtf.py -> the installer would show the old terms ->
    this fails.
    """
    expected = _convert(_LICENSE.read_text(encoding="utf-8"))
    assert _RTF.is_file(), f"missing {_RTF}; run installer/license_rtf.py"
    committed = _RTF.read_text(encoding="ascii")
    assert _lines(committed) == _lines(expected), (
        "installer/LICENSE.rtf is stale; run "
        ".venv\\Scripts\\python.exe installer\\license_rtf.py"
    )


def test_license_rtf_has_no_raw_markdown() -> None:
    """The license page shows formatted text, not Markdown markup: no
    heading hashes, emphasis stars, link brackets or angle-bracket URLs.

    Seeded defect: drop the **bold** handling from license_rtf.py and
    regenerate -> "**licensor**" shows on the license page -> this fails.
    Or match ``**...**`` before ``***...***`` -> the No Liability paragraph
    turns bold with a stray star at each end -> this fails too.
    """
    rtf = _RTF.read_text(encoding="ascii")
    body = rtf.split("\n", 1)[1]  # past the font table header
    leaks = [mark for mark in ("*", "](", "<http", "`") if mark in body]
    assert not leaks, f"installer/LICENSE.rtf shows raw Markdown: {leaks}"
    for line in _lines(body):
        text = line.split(" ", 1)[1] if line.startswith("\\pard") else line
        assert not text.startswith(("#", ">")), f"raw Markdown line: {line[:60]!r}"


def test_license_rtf_writes_characters_above_u_ffff_as_surrogate_pairs() -> None:
    """RTF's \\uN escape holds one UTF-16 code unit, so a character above
    U+FFFF (an emoji a future LICENSE edit adds) needs its surrogate pair,
    each unit a signed 16-bit \\uN? escape. The drift test can't catch a
    wrong escape: it compares with the same converter.

    Seeded defect: escape the code point itself (U+1F600 -> \\u62976?, the
    private-use U+F600, a wrong glyph on the license page) -> this fails.
    """
    rtf = _convert("Thanks \U0001F600 and \u2014 done.")
    assert r"Thanks \u-10179?\u-8704? and \u8212? done." in rtf, rtf
