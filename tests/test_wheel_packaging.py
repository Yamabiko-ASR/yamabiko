"""Wheel data-completeness guard.

A built wheel must ship every runtime-read data file, not just the .py
modules. The runtime resolves these via importlib.resources /
paths.bundle_data_dir():

  * yamabiko/default_config.yaml   -- paths.shipped_defaults_path()
  * yamabiko/display/static/*      -- browser_server static mount /
                                      FileResponse(index.html)
  * yamabiko/glossaries/*          -- genre JSONs + user.yaml seeded by
                                      main._seed_user_glossaries();
                                      _seed.yaml read directly
                                      (main.py); _denylist.txt read by
                                      the romanization pipeline
  * yamabiko/vad/data/*            -- vad.sound_tagger.default_model_path()
                                      (bundled sound-tagger ONNX + licence)

Nothing in-repo currently ships a wheel (the installer is PyInstaller
with its own spec ``datas=``, CI uses ``pip install -e .``), so this is
a dormant-packaging-debt guard: it fails red the moment
``[tool.setuptools.package-data]`` stops covering any runtime data set.

Marked ``real`` so it stays out of the default ``-m "not real"`` unit
loop: it shells out to ``pip wheel`` and takes a few seconds. It needs
NO network -- ``--no-build-isolation`` reuses the venv's
already-installed setuptools -- so it is safe in the real tier with no
hardware or live endpoints.

Seeded defect: delete the ``[tool.setuptools.package-data]`` table from
pyproject.toml -> the built wheel drops every data file -> red on the
``missing`` assertion.
"""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Every runtime data file that must survive into the wheel, keyed by the
# critical members that anchor the guard against a vacuous pass.
_CRITICAL_MEMBERS = (
    "yamabiko/default_config.yaml",
    "yamabiko/display/static/index.html",
    "yamabiko/display/static/overlay.css",
    "yamabiko/display/static/overlay.js",
    "yamabiko/display/static/fonts/KleeOne-SemiBold.ttf",
    "yamabiko/glossaries/anime.json",
    "yamabiko/glossaries/drama.json",
    "yamabiko/glossaries/news.json",
    "yamabiko/glossaries/variety.json",
    "yamabiko/glossaries/_seed.yaml",
    "yamabiko/glossaries/_denylist.txt",
    "yamabiko/glossaries/user.yaml",
    "yamabiko/vad/data/efficientsed-fmn10-tf256-kd.onnx",
)


def _expected_data_members() -> set[str]:
    """Wheel arcnames every runtime data file must appear under.

    Derived from the SOURCE tree (recursively) so a newly-added glossary,
    static asset or font is automatically required -- the exact bug class
    this guards (a data file added but left out of the package-data glob).
    """
    pkg = _REPO_ROOT / "yamabiko"
    expected = {"yamabiko/default_config.yaml"}
    for sub in ("display/static", "glossaries", "vad/data", "overlay_app/assets"):
        root = pkg / sub
        for f in root.rglob("*"):
            if f.is_file():
                expected.add(f"yamabiko/{sub}/{f.relative_to(root).as_posix()}")
    return expected


@pytest.mark.real
def test_built_wheel_contains_all_runtime_data(tmp_path):
    try:
        import setuptools  # noqa: F401
    except ImportError:  # pragma: no cover - modern build envs always have it
        pytest.skip("setuptools unavailable for --no-build-isolation build")

    expected = _expected_data_members()
    # Sanity floor: the source-tree derivation must actually contain the
    # critical files, so the superset check below can never pass because
    # `expected` collapsed to (almost) nothing.
    for critical in _CRITICAL_MEMBERS:
        assert critical in expected, (
            f"source tree is missing {critical}; the guard would be vacuous"
        )

    proc = subprocess.run(
        [
            sys.executable, "-m", "pip", "wheel", str(_REPO_ROOT),
            "--no-deps", "--no-build-isolation", "-w", str(tmp_path),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        f"pip wheel failed (rc={proc.returncode}):\n{proc.stderr}"
    )

    wheels = list(tmp_path.glob("yamabiko-*.whl"))
    assert len(wheels) == 1, f"expected exactly one yamabiko wheel, got {wheels}"
    names = set(zipfile.ZipFile(wheels[0]).namelist())

    missing = expected - names
    assert not missing, (
        "built wheel omits runtime data files -- the package-data globs "
        f"in pyproject.toml do not cover them: {sorted(missing)}"
    )
