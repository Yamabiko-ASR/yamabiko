"""Rasterise the app icon SVGs to a multi-resolution .ico + preview PNG.

Hand-tuned sources, rendered with headless Edge (present on the Windows 11
dev/build boxes) and packed with Pillow:

  icon.svg     128-unit master: tile, bold 山, ridge line (64 px and up, preview)
  icon-48.svg  48-unit frame, 山 and ridge on the pixel grid (48 px)
  icon-32.svg  32-unit frame with a heavier ridge (32 px)
  icon-24.svg  24-unit frame, 山 and ridge on the pixel grid (24 px)
  icon-16.svg  16-unit frame, 山 only (16 px)

Outputs:
  assets/icon.ico   - multi-resolution Windows icon (16/24/32/48/64/128/256)
  assets/icon.png   - 512x512 preview for docs/README + wizard welcome page
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ASSETS = Path(__file__).parent
SVG_PATH = ASSETS / "icon.svg"
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]
HAND_TUNED_SIZES = (16, 24, 32, 48)


def _source_for(size: int) -> Path:
    """The SVG drawn for ``size``: its own frame up to 48 px, the master above."""
    if size in HAND_TUNED_SIZES:
        return ASSETS / f"icon-{size}.svg"
    return SVG_PATH


def _markup(path: Path) -> str:
    return re.sub(r"^<\?xml[^>]*\?>\s*", "", path.read_text(encoding="utf-8"))


EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def _find_edge() -> str:
    for candidate in EDGE_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    found = shutil.which("msedge")
    if found:
        return found
    raise RuntimeError("msedge.exe not found; headless Edge is required to rasterise the SVG")


def _render(edge: str, svg_markup: str, size: int, workdir: Path) -> Image.Image:
    html = (
        "<!doctype html><meta charset='utf-8'>"
        f"<style>html,body{{margin:0}} svg{{display:block;width:{size}px;height:{size}px}}</style>"
        f"{svg_markup}"
    )
    page = workdir / f"icon-{size}.html"
    out = workdir / f"icon-{size}.png"
    page.write_text(html, encoding="utf-8")
    result = subprocess.run(
        [
            edge,
            "--headless=new",
            "--disable-gpu",
            "--default-background-color=00000000",
            f"--window-size={size},{size}",
            f"--screenshot={out}",
            page.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    if not out.is_file():
        detail = result.stderr.decode(errors="replace").strip() or "no stderr output"
        raise RuntimeError(
            f"headless Edge wrote no {size}x{size} screenshot: {detail}"
        )
    img = Image.open(out).convert("RGBA")
    if img.size != (size, size):
        raise RuntimeError(f"expected {size}x{size} render, got {img.size}")
    return img


def main() -> None:
    edge = _find_edge()
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        variants = [
            _render(edge, _markup(_source_for(s)), s, workdir) for s in ICO_SIZES
        ]
        preview = _render(edge, _markup(SVG_PATH), 512, workdir)

    ico_path = ASSETS / "icon.ico"
    variants[-1].save(ico_path, format="ICO",
                      sizes=[(s, s) for s in ICO_SIZES],
                      append_images=variants[:-1])
    preview.save(ASSETS / "icon.png", format="PNG")
    print(f"wrote {ico_path} ({len(ICO_SIZES)} resolutions) + icon.png preview")


if __name__ == "__main__":
    main()
