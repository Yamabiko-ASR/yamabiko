"""The installer's committed bitmaps (installer/art/*.bmp).

installer/art/build_art.py renders them from the SVG sources at dev time;
the installer build and CI only read the committed files (makensis embeds
them, and the installer picks one scale by the system DPI at runtime and
loads it with LoadImage). Stdlib only: Pillow is not in the release venv.
"""
from __future__ import annotations

import struct
from pathlib import Path

_ART = Path(__file__).resolve().parents[1] / "installer" / "art"

# name -> pixel size at 100 %, 150 %, 200 % (half-up rounding, as build_art.py)
_EXPECTED = {
    "mist": ((497, 314), (746, 471), (994, 628)),
    "removed": ((497, 314), (746, 471), (994, 628)),
    "dusk": ((497, 314), (746, 471), (994, 628)),
    "seal-mist": ((32, 32), (48, 48), (64, 64)),
    "seal-dusk": ((32, 32), (48, 48), (64, 64)),
    "header": ((150, 57), (225, 86), (300, 114)),
}

# name -> the sky color of its SVG (the background rect), which every render's
# top corners carry when the art fills the bitmap edge to edge
_SKY = {
    "mist": "f4f3ef",
    "removed": "f4f3ef",
    "dusk": "f6eee6",
    "seal-mist": "f4f3ef",
    "seal-dusk": "f6eee6",
    "header": "f4f3ef",
}


def test_installer_bitmaps_are_24bit_bmps_at_each_shipped_scale() -> None:
    """Every page, seal and header bitmap exists at 100, 150 and 200 % as an
    uncompressed 24-bit BMP of exactly the size the installer's layout
    assumes (the page art is scaled to the page width and anchored to the
    bottom, the seal and header are fitted from these sizes), with its art
    reaching the bitmap's edges: both top corners are the sky color.

    Seeded defect: regenerate one scale at the wrong size (or save it as a
    32-bit or PNG file under the .bmp name), or delete one file -> this fails.
    Render a scale whose size was rounded (150 %: 745.5 -> 746 px) with the
    art centered instead of stretched, leaving a light sub-pixel edge -> this
    fails.
    """
    problems = []
    for name, sizes in _EXPECTED.items():
        for pct, (width, height) in zip((100, 150, 200), sizes):
            path = _ART / f"{name}-{pct}.bmp"
            if not path.is_file():
                problems.append(f"{path.name}: missing")
                continue
            data = path.read_bytes()
            if data[:2] != b"BM" or len(data) < 54:
                problems.append(f"{path.name}: not a BMP")
                continue
            file_size, = struct.unpack_from("<I", data, 2)
            header_size, w, h, planes, bits, compression = struct.unpack_from(
                "<IiiHHI", data, 14
            )
            stride = (w * 3 + 3) & ~3
            got = (header_size, w, abs(h), planes, bits, compression)
            if got != (40, width, height, 1, 24, 0):
                problems.append(
                    f"{path.name}: header {got}, expected (40, {width}, {height}, 1, 24, 0)"
                )
            elif file_size != len(data) or len(data) != 54 + stride * height:
                problems.append(f"{path.name}: {len(data)} bytes, not a whole 24-bit image")
            else:
                # Rows are stored bottom-up when h > 0, pixels as BGR.
                top = 54 + ((height - 1) * stride if h > 0 else 0)
                corners = (data[top:top + 3], data[top + (width - 1) * 3:top + width * 3])
                sky = bytes.fromhex(_SKY[name])[::-1]
                if corners != (sky, sky):
                    found = ", ".join(c[::-1].hex() for c in corners)
                    problems.append(
                        f"{path.name}: top corners {found}, expected the sky {_SKY[name]}"
                        " (the art doesn't reach the edge)"
                    )
    assert not problems, "installer bitmaps are wrong:\n" + "\n".join(problems)
