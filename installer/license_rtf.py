"""Write installer/LICENSE.rtf, the license page's copy of LICENSE (stdlib only).

The installer's license page is a RichEdit control: it shows RTF as
formatted text, and plain text as raw Markdown. This converts LICENSE's
Markdown to RTF in the approved mockup's styling (Segoe UI, sizes in px at
100 %; RTF uses half-points for type and twips for spacing, 15 twips = 1 px):

  # heading   14 px bold, 20 px line, 4 px below   \\fs21 \\b \\sl-300 \\sa60
  ## heading  12 px bold, 16 px line, 2 px below   \\b \\sl-240 \\sa30
  paragraph   12 px, 16 px line, 8 px below        \\fs18 \\sl-240 \\sa120
  > quote     a paragraph indented 16 px            \\li240

A negative \\sl is an exact line height (RichEdit's own pitch for Segoe UI
9 pt is 15 px, the mockup's 16). **bold** and ***bold italic*** become RTF
runs, links and <urls> plain text, `code` loses its backticks, and runs of
spaces collapse. Non-ASCII characters become \\uN? escapes, one per UTF-16
code unit (a surrogate pair above U+FFFF).

The committed LICENSE.rtf must match this script's output
(tests/test_installer_license_rtf.py). After editing LICENSE run:
    .venv\\Scripts\\python.exe installer\\license_rtf.py
"""
from __future__ import annotations

import re
from pathlib import Path

INSTALLER = Path(__file__).resolve().parent
LICENSE = INSTALLER.parent / "LICENSE"
OUTPUT = INSTALLER / "LICENSE.rtf"

_HEAD = (
    r"{\rtf1\ansi\ansicpg1252\deff0\uc1"
    r"{\fonttbl{\f0\fswiss\fcharset0 Segoe UI;}}"
    r"\viewkind4\f0\fs18"
)
_H1 = r"\pard\sb0\sa60\sl-300\slmult0\b\fs21 "
_H2 = r"\pard\sb0\sa30\sl-240\slmult0\b "
_QUOTE = r"\pard\li240\sb0\sa120\sl-240\slmult0 "
_BODY = r"\pard\sb0\sa120\sl-240\slmult0 "
_EMPHASIS = re.compile(r"\*\*\*(.+?)\*\*\*|\*\*(.+?)\*\*")


def _escape(text: str) -> str:
    out = []
    for ch in text:
        if ch in "\\{}":
            out.append("\\" + ch)
        elif ord(ch) < 128:
            out.append(ch)
        else:
            # \uN holds one UTF-16 code unit as a signed 16-bit number, so a
            # character above U+FFFF becomes its surrogate pair.
            units = ch.encode("utf-16-le")
            for i in range(0, len(units), 2):
                unit = int.from_bytes(units[i:i + 2], "little")
                out.append(f"\\u{unit if unit < 32768 else unit - 65536}?")
    return "".join(out)


def _plain(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<(https?://[^>]+)>", r"\1", text)
    text = text.replace("`", "")
    return re.sub(r"\s{2,}", " ", text).strip()


def _runs(text: str) -> str:
    """Escape ``text`` and turn **bold** and ***bold italic*** into RTF groups."""
    out = []
    pos = 0
    for match in _EMPHASIS.finditer(text):
        out.append(_escape(text[pos:match.start()]))
        if match.group(1) is not None:
            out.append(r"{\b\i " + _escape(match.group(1)) + "}")
        else:
            out.append(r"{\b " + _escape(match.group(2)) + "}")
        pos = match.end()
    out.append(_escape(text[pos:]))
    return "".join(out)


def convert(markdown: str) -> str:
    """RTF for ``markdown`` (LICENSE's subset: headings, paragraphs, a quote)."""
    paragraphs = []
    for block in re.split(r"\n\s*\n", markdown.replace("\r\n", "\n").strip()):
        line = _plain(" ".join(block.split("\n")))
        if line.startswith("# "):
            paragraphs.append(_H1 + _runs(line[2:]) + r"\b0\fs18\par")
        elif line.startswith("## "):
            paragraphs.append(_H2 + _runs(line[3:]) + r"\b0\par")
        elif line.startswith("> "):
            paragraphs.append(_QUOTE + _runs(line[2:]) + r"\par")
        else:
            paragraphs.append(_BODY + _runs(line) + r"\par")
    return _HEAD + "\n" + "\n".join(paragraphs) + "\n}\n"


def main() -> None:
    rtf = convert(LICENSE.read_text(encoding="utf-8"))
    OUTPUT.write_bytes(rtf.encode("ascii"))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
