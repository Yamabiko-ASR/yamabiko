"""Render the installer's bitmaps from their SVG sources (dev-time, run with .venv).

Sources (the approved mockup's art, opt-inst-full.tpl, without the seal on the
pages so the seal stays square at every scale):

  mist.svg       Welcome, Update and Remove pages: Mist landscape with echo arcs
  removed.svg    Removed page: the same landscape without the echo arcs
  dusk.svg       Finish page: the landscape at dusk
  seal-mist.svg  the 32 px seal on the Mist sky (its own control on the page)
  seal-dusk.svg  the 32 px seal on the dusk sky
  header.svg     150 x 57 ridgeline sliver and seal for the license and
                 progress pages' header

Each source is rendered with headless Edge (present on the Windows 11 dev and
build boxes) at 100, 150 and 200 % and saved by Pillow as a 24-bit BMP named
``<source>-<pct>.bmp`` next to it. The BMPs are committed: the installer build
and CI need no renderer. The seal's 山彦 uses the bundled BIZ UDPMincho Bold,
as the app's seal does, so every machine renders the same glyphs.

Usage: .venv\\Scripts\\python.exe installer\\art\\build_art.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ART = Path(__file__).resolve().parent
REPO = ART.parents[1]
SEAL_FONT = REPO / "yamabiko" / "display" / "static" / "fonts" / "BIZUDPMincho-Bold.ttf"
# name -> size at 100 % (the SVG's own width/height)
SOURCES = {
    "mist": (497, 314),
    "removed": (497, 314),
    "dusk": (497, 314),
    "seal-mist": (32, 32),
    "seal-dusk": (32, 32),
    "header": (150, 57),
}
SCALES = (100, 150, 200)

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]


def scaled(size: tuple[int, int], pct: int) -> tuple[int, int]:
    """Pixel size at ``pct`` percent, rounded half up."""
    return tuple((v * pct + 50) // 100 for v in size)


def _find_edge() -> str:
    for candidate in EDGE_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    found = shutil.which("msedge")
    if found:
        return found
    raise RuntimeError("msedge.exe not found; headless Edge is required to render the SVGs")


def _render(edge: str, svg_markup: str, width: int, height: int, workdir: Path) -> Image.Image:
    # A size rounded to whole pixels (150 %: 745.5 -> 746) is off the viewBox's
    # aspect by under a pixel. Stretch the art over it: the default centering
    # leaves a light sub-pixel edge of the page's white background.
    svg_markup = svg_markup.replace("<svg ", '<svg preserveAspectRatio="none" ', 1)
    html = (
        "<!doctype html><meta charset='utf-8'><style>"
        "@font-face{font-family:'BIZ UDPMincho';font-weight:700;"
        f"src:url('{SEAL_FONT.as_uri()}')}}"
        "html,body{margin:0;overflow:hidden}"
        f"svg{{display:block;width:{width}px;height:{height}px}}"
        f"</style>{svg_markup}"
    )
    page = workdir / "page.html"
    out = workdir / "page.png"
    out.unlink(missing_ok=True)
    page.write_text(html, encoding="utf-8")
    env = dict(os.environ)
    env.pop("__COMPAT_LAYER", None)  # headless Edge writes nothing under a compat layer
    result = subprocess.run(
        [
            edge,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            "--force-color-profile=srgb",
            f"--window-size={width},{height}",
            f"--screenshot={out}",
            page.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=60,
        env=env,
    )
    if not out.is_file():
        detail = result.stderr.decode(errors="replace").strip() or "no stderr output"
        raise RuntimeError(f"headless Edge wrote no {width}x{height} screenshot: {detail}")
    with Image.open(out) as shot:
        img = shot.convert("RGB")
    if img.size != (width, height):
        raise RuntimeError(f"expected a {width}x{height} render, got {img.size}")
    return img


def main() -> None:
    edge = _find_edge()
    if not SEAL_FONT.is_file():
        raise RuntimeError(f"missing {SEAL_FONT}")
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for name, size in SOURCES.items():
            markup = (ART / f"{name}.svg").read_text(encoding="utf-8")
            for pct in SCALES:
                width, height = scaled(size, pct)
                bmp = ART / f"{name}-{pct}.bmp"
                _render(edge, markup, width, height, workdir).save(bmp, format="BMP")
                print(f"wrote {bmp.name} ({width}x{height})")


if __name__ == "__main__":
    main()
