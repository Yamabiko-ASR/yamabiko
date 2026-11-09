"""Render the README banners and the GitHub social preview card.

The artwork reuses the app's own brand pieces, so it cannot drift from them:
the landscape and echo arcs (overlay_app/landscape.py), the 山彦 seal
(overlay_app/seal.py), the light and dark palettes (overlay_app/theme.py) and
the bundled fonts. Qt paints those pieces offscreen; headless Edge, as in
build_icon.py, sets the type, including the furigana sample, in the same
bundled fonts.

Run from the repo root with the dev venv:

  .venv\\Scripts\\python.exe assets\\build_banner.py

Outputs:
  assets/banner-light.png    2560x640 (1280x320 at 2x), README header, light theme
  assets/banner-dark.png     the same in the dark palette, README header, dark theme
  assets/social-preview.png  1280x640, upload under Settings > General >
                             Social preview (GitHub has no API for it)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ASSETS = Path(__file__).resolve().parent
REPO = ASSETS.parent
sys.path.insert(0, str(REPO))

from PIL import Image  # noqa: E402
from PySide6.QtCore import QRectF, Qt  # noqa: E402
from PySide6.QtGui import QGuiApplication, QImage, QPainter  # noqa: E402

from build_icon import _find_edge  # noqa: E402
from yamabiko.overlay_app.fonts import register_bundled_fonts  # noqa: E402
from yamabiko.overlay_app.landscape import paint_landscape  # noqa: E402
from yamabiko.overlay_app.seal import paint_seal  # noqa: E402
from yamabiko.overlay_app.theme import MIST, NIGHT  # noqa: E402

FONTS = REPO / "yamabiko" / "display" / "static" / "fonts"

# A real line from the demo capture (assets/yamabiko-demo.gif): the app's own
# transcription, furigana and translation, not invented copy.
SAMPLE_JP = (
    "<ruby>水面<rt>すいめん</rt></ruby>から<ruby>顔<rt>かお</rt></ruby>"
    "をのぞかせているのは<ruby>毒<rt>どく</rt></ruby>ヘビのコブラです"
)
SAMPLE_EN = "A cobra is seen with its head above the water surface."

TAGLINE = "Live Japanese subtitles with furigana and English translation"
SUBLINE = "Runs on your own NVIDIA graphics card. No accounts, no usage fees."

# Height of the landscape inside the social card's screen: tall enough that the
# echo arcs rise clear of the subtitle stack.
SCENE_H = 226


# --------------------------------------------------------------------------
# Qt-painted brand pieces
# --------------------------------------------------------------------------

def _qt_png(path: Path, width: int, height: int, scale: int, paint) -> None:
    image = QImage(width * scale, height * scale, QImage.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(scale)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    paint(painter, QRectF(0, 0, width, height))
    painter.end()
    if not image.save(str(path)):
        raise RuntimeError(f"Qt could not write {path}")


def _landscape_png(path: Path, palette: dict, width: int, height: int, scale: int) -> None:
    _qt_png(path, width, height, scale,
            lambda p, r: paint_landscape(p, r, palette, arcs=True))


def _seal_png(path: Path, size: int, scale: int) -> None:
    _qt_png(path, size, size, scale, paint_seal)


# --------------------------------------------------------------------------
# HTML layouts
# --------------------------------------------------------------------------

def _font_faces() -> str:
    faces = (
        ("Gothic", 400, "BIZUDPGothic-Regular.ttf"),
        ("Gothic", 700, "BIZUDPGothic-Bold.ttf"),
        ("Mincho", 400, "BIZUDPMincho-Regular.ttf"),
        ("Mincho", 700, "BIZUDPMincho-Bold.ttf"),
    )
    rules = []
    for family, weight, name in faces:
        path = FONTS / name
        if not path.is_file():
            raise RuntimeError(f"bundled font missing: {path}")
        rules.append(
            f"@font-face{{font-family:{family};font-weight:{weight};"
            f"src:url('{path.as_uri()}')}}"
        )
    return "".join(rules)


def _page(css: str, body: str) -> str:
    return (
        "<!doctype html><meta charset='utf-8'>"
        f"<style>{_font_faces()}"
        "html,body{margin:0;background:transparent}"
        f"{css}</style>{body}"
    )


def _brand(seal: Path, palette: dict, seal_px: int, word_px: int, kanji_px: int) -> str:
    return (
        "<div class='brand'>"
        f"<img src='{seal.as_uri()}' width='{seal_px}' height='{seal_px}'>"
        f"<span class='word' style='font-size:{word_px}px;color:{palette['ink']}'>Yamabiko</span>"
        f"<span class='kanji' style='font-size:{kanji_px}px;color:{palette['muted']}'>山彦</span>"
        "</div>"
    )


BRAND_CSS = """
.brand{display:flex;align-items:center}
.brand img{flex:none}
.word{font-family:Mincho,serif;font-weight:700;line-height:1;margin-left:.3em}
.kanji{font-family:Mincho,serif;font-weight:400;line-height:1;margin-left:.3em}
"""


def _banner_html(palette: dict, seal: Path, land: Path) -> str:
    p = palette
    css = BRAND_CSS + f"""
.card{{position:relative;width:1280px;height:320px;box-sizing:border-box;
  overflow:hidden;border-radius:18px;border:1px solid {p['line']};
  background:{p['ground']};font-family:Gothic,sans-serif}}
.land{{position:absolute;left:0;bottom:0;width:1280px;height:104px}}
.body{{position:absolute;left:0;right:0;top:58px;display:flex;
  flex-direction:column;align-items:center}}
.tag{{margin-top:30px;font-size:27px;color:{p['ink']}}}
"""
    body = (
        "<div class='card'>"
        f"<img class='land' src='{land.as_uri()}'>"
        "<div class='body'>"
        f"{_brand(seal, p, 62, 78, 62)}"
        f"<div class='tag'>{TAGLINE}</div>"
        "</div></div>"
    )
    return _page(css, body)


def _social_html(palette: dict, seal: Path, scene: Path) -> str:
    p = palette
    css = BRAND_CSS + f"""
.card{{position:relative;width:1280px;height:640px;box-sizing:border-box;
  padding:62px 80px 0;background:{p['raised']};font-family:Gothic,sans-serif}}
.tag{{margin-top:26px;font-size:30px;color:{p['ink']}}}
.sub{{margin-top:10px;font-size:21px;color:{p['muted']}}}
.screen{{position:absolute;left:80px;right:80px;top:296px;bottom:62px;
  overflow:hidden;border-radius:14px;border:1px solid {p['line']};
  background:linear-gradient(180deg,#222a26 0,{p['ground']} calc(100% - {SCENE_H}px))}}
.screen img{{position:absolute;left:0;bottom:0;width:1120px;height:{SCENE_H}px}}
.subs{{position:absolute;left:34px;bottom:20px;display:flex;
  flex-direction:column;align-items:flex-start}}
.jp{{background:{MIST['ground']};color:{MIST['ink']};font-size:25px;
  padding:6px 14px 5px;line-height:1.5}}
.jp rt{{font-size:11px;color:{MIST['muted']}}}
.en{{background:rgba(10,13,11,.9);color:#ffffff;font-weight:700;font-size:29px;
  padding:9px 16px 10px}}
"""
    body = (
        "<div class='card'>"
        f"{_brand(seal, p, 66, 82, 66)}"
        f"<div class='tag'>{TAGLINE}</div>"
        f"<div class='sub'>{SUBLINE}</div>"
        "<div class='screen'>"
        f"<img src='{scene.as_uri()}'>"
        f"<div class='subs'><div class='jp'>{SAMPLE_JP}</div>"
        f"<div class='en'>{SAMPLE_EN}</div></div>"
        "</div></div>"
    )
    return _page(css, body)


def _screenshot(edge: str, html: str, size: tuple[int, int], scale: int,
                name: str, workdir: Path) -> Image.Image:
    page = workdir / f"{name}.html"
    out = workdir / f"{name}.png"
    page.write_text(html, encoding="utf-8")
    w, h = size
    result = subprocess.run(
        [
            edge,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--allow-file-access-from-files",
            "--default-background-color=00000000",
            f"--force-device-scale-factor={scale}",
            f"--window-size={w},{h}",
            f"--screenshot={out}",
            page.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    if not out.is_file():
        detail = result.stderr.decode(errors="replace").strip() or "no stderr output"
        raise RuntimeError(f"headless Edge wrote no {name} screenshot: {detail}")
    img = Image.open(out).convert("RGBA")
    expected = (w * scale, h * scale)
    if img.size != expected:
        raise RuntimeError(f"expected {expected} render for {name}, got {img.size}")
    return img


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
    if not register_bundled_fonts(FONTS):
        raise RuntimeError(f"no bundled fonts registered from {FONTS}")
    edge = _find_edge()
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        seal = work / "seal.png"
        _seal_png(seal, 66, 2)
        outputs = []
        for name, palette in (("banner-light", MIST), ("banner-dark", NIGHT)):
            land = work / f"{name}-land.png"
            _landscape_png(land, palette, 1280, 104, 2)
            img = _screenshot(edge, _banner_html(palette, seal, land),
                              (1280, 320), 2, name, work)
            outputs.append((ASSETS / f"{name}.png", img))
        scene = work / "social-scene.png"
        _landscape_png(scene, NIGHT, 1120, SCENE_H, 1)
        social = _screenshot(edge, _social_html(NIGHT, seal, scene),
                             (1280, 640), 1, "social-preview", work)
        # GitHub's social preview is shown without transparency.
        outputs.append((ASSETS / "social-preview.png", social.convert("RGB")))
    for path, img in outputs:
        img.save(path, format="PNG", optimize=True)
        print(f"wrote {path.relative_to(REPO)} {img.size[0]}x{img.size[1]}")
    del app


if __name__ == "__main__":
    main()
