"""Unit-tier battery for CJK / non-ASCII app-data paths (issue #26).

Why this exists: Yamabiko ships to JP-locale Windows users, whose
Windows username — and therefore ``%LOCALAPPDATA%`` and the whole
``YAMABIKO_APPDATA`` tree — routinely contains CJK characters
(``C:\\Users\\山彦\\AppData\\Local\\Yamabiko``). Before #26 NOT ONE
test exercised the app-data stack with a non-ASCII root. The classic
failure mode is a stray ``str.encode()`` / ``.encode("ascii")`` / a
``cp932`` (Shift-JIS) text mode or an ``open()`` missing
``encoding="utf-8"`` somewhere in the path-building, config-writing,
logging, or wizard-state code: each one is invisible on an ASCII dev
box and raises ``UnicodeEncodeError`` / ``UnicodeDecodeError`` /
mojibake on a JP user's machine.

These are unit-tier: the subjects (``paths``, ``config_writer``,
``config``, ``log_config``, ``wizard_state``) are pure-Python +
filesystem, so a CJK root either works end-to-end or it doesn't. No
mocks on the subject — the only indirection is the
``YAMABIKO_APPDATA`` env override, which is the same seam the real app
honours.

If any test here fails, that is a REAL #26 production bug (a Unicode
seam in our own code), not a test artifact — fix it in product code.

The long-path budget test at the bottom is the static (no-I/O) half of
#26: it asserts the deepest path Yamabiko itself can produce fits the
default Windows MAX_PATH of 260, so a default install on a
not-absurdly-long username never needs LongPathsEnabled.
"""
from __future__ import annotations

import pytest

from yamabiko import config, config_writer, paths, wizard_state


# A username/app-data segment a real JP user would have. Mixes kanji,
# katakana, and hiragana so a partial-codec bug (e.g. one that survives
# katakana but not kanji) still trips.
_CJK_SEGMENT = "山彦ユーザー試験"


@pytest.fixture
def cjk_appdata(tmp_path, monkeypatch):
    """Point YAMABIKO_APPDATA at a CJK-named directory under tmp_path.

    ``paths.app_data_dir()`` re-reads the env on every call (no
    caching), so setting it here isolates every consuming module to the
    CJK root for the duration of the test.
    """
    root = tmp_path / _CJK_SEGMENT
    root.mkdir()
    monkeypatch.setenv("YAMABIKO_APPDATA", str(root))
    return root


def test_ensure_dirs_creates_tree_under_cjk_root(cjk_appdata):
    """paths.ensure_dirs() creates logs/ under a CJK root, and not the v0.1
    models/ folder.

    Seeded defect (one line in paths.py): wrap the root in
    ``Path(str(app_data_dir()).encode('ascii').decode())`` — or any
    ``.encode('ascii')`` on the path — and ensure_dirs raises
    UnicodeEncodeError before logs/ is created -> red; put root / "models"
    back in ensure_dirs -> red.
    """
    paths.ensure_dirs()

    assert (cjk_appdata / "logs").is_dir(), (
        f"logs/ not created under CJK root {cjk_appdata}"
    )
    assert not (cjk_appdata / "models").exists(), (
        f"models/ created under CJK root {cjk_appdata}"
    )


def test_override_write_read_roundtrip_under_cjk_root(cjk_appdata):
    """A config override written under a CJK root reads back through the
    real merged-config loader.

    Exercises config_writer.write_override (filelock + atomic write,
    both of which open files under the CJK app-data dir) AND
    config.load_full_config (which reads the same file). quant_tier's
    shipped default is "default", so writing "small" is a genuine
    override that must survive the round trip.

    Seeded defect: drop ``encoding="utf-8"`` from config_writer's
    atomic write (so it opens in the locale default cp932 on a JP box)
    — round-trip still works on an ASCII path but a CJK *value* would
    mojibake; more directly, an ``.encode('ascii')`` on the overrides
    path in either module raises before the value is persisted, so the
    read-back assertion goes red.
    """
    paths.ensure_dirs()

    config_writer.write_override(["translation", "quant_tier"], "small")

    assert paths.user_overrides_path().is_file(), (
        f"user_overrides.yaml not written under CJK root {cjk_appdata}"
    )
    cfg = config.load_full_config()
    assert cfg["translation"]["quant_tier"] == "small", (
        f"override did not round-trip under CJK root; got "
        f"{cfg['translation'].get('quant_tier')!r}"
    )


def test_log_setup_writes_under_cjk_root(cjk_appdata, hermetic_log_handlers):
    """The production setup_main_log idiom writes a log file containing a
    CJK log line under a CJK app-data root, readable back as UTF-8.

    Uses the exact call shape main.py uses
    (``setup_main_log(logging.getLogger(), level="INFO",
    filename="main-pipeline.log")``) so the RotatingFileHandler is
    constructed against ``app_data_dir()/logs`` for real. The
    hermetic_log_handlers guard closes the handler at teardown so
    Windows can delete the CJK tmp dir.

    Seeded defect: change the RotatingFileHandler construction in
    log_config.py to ``encoding=None`` (locale default = cp932 on a JP
    box) — the CJK log line then encodes as Shift-JIS, and the UTF-8
    read-back below either raises UnicodeDecodeError or fails to find
    the marker, going red. Equally, an ``.encode('ascii')`` on the log
    path raises at handler construction.
    """
    import logging

    from yamabiko import log_config

    paths.ensure_dirs()
    marker = "marker-7741"
    cjk_line = f"日本語のログ行 {marker}"

    # The file is opened, written, AND read entirely INSIDE the
    # hermetic_log_handlers block on purpose: the block's teardown
    # closes the RotatingFileHandler it added, releasing the OS handle
    # so the tmp_path cleanup can delete the CJK log dir on Windows. Do
    # NOT move the read_text() below the `with` — a future refactor that
    # hoists it out keeps the assertion green but reintroduces the
    # open-handle that makes Windows tmp teardown flaky.
    with hermetic_log_handlers("", "yamabiko"):
        log_config.setup_main_log(
            logging.getLogger(), level="INFO", filename="main-pipeline.log",
        )
        logging.getLogger("yamabiko").info(cjk_line)
        for h in logging.getLogger().handlers:
            h.flush()

        log_file = cjk_appdata / "logs" / "main-pipeline.log"
        assert log_file.is_file(), (
            f"main-pipeline.log not written under CJK root; logs dir "
            f"contents: "
            f"{[p.name for p in (cjk_appdata / 'logs').iterdir()] if (cjk_appdata / 'logs').is_dir() else 'absent'}"
        )
        text = log_file.read_text(encoding="utf-8")

    assert marker in text, (
        f"unique marker {marker!r} not found in the CJK log file — the "
        f"CJK line did not survive the log write/read round trip"
    )
    assert "日本語のログ行" in text, (
        "the CJK characters themselves were lost/mangled in the log file "
        "(encoding regression — handler not writing UTF-8)"
    )


def test_wizard_state_roundtrip_under_cjk_root(cjk_appdata, monkeypatch):
    """First-run wizard completion persists and reads back under a CJK root.

    Drives the real wizard_state contract: needs_wizard() is True on a
    fresh CJK install, save(WizardResult(first_run_complete=True))
    writes state.json under the CJK app-data dir, and needs_wizard()
    then reads it back as False. save() opens state.json.tmp + fsyncs +
    renames, all under the CJK root.

    needs_wizard()'s model-cache trigger (models.required_models_cached,
    the upgrade-migration gate) is stubbed True: it probes the machine's
    real HF cache, which is not the subject here (the CJK *path* stack
    is) and would make this test depend on whether the dev/CI box has
    the models downloaded. Its own contract is covered in
    tests/test_wizard_state.py against a temp HF cache.

    Seeded defect: drop ``encoding="utf-8"`` from wizard_state.save's
    open() (locale default cp932) — on this CJK path the write itself
    still works (the bytes written are ASCII JSON) but flipping load()'s
    read to ``.read_text()`` without encoding, or an ``.encode('ascii')``
    on the state path, raises and needs_wizard() stays True, going red.
    More fundamentally: if app_data_dir() ascii-mangles the CJK root,
    save writes to one path and load reads another → needs_wizard()
    never flips.
    """
    monkeypatch.setattr(
        "yamabiko.models.required_models_cached", lambda **kw: True,
    )

    assert wizard_state.needs_wizard() is True, (
        "fresh CJK install should need the wizard, but needs_wizard() "
        "returned False before any state was saved"
    )

    wizard_state.save(wizard_state.WizardResult(first_run_complete=True))

    assert paths.state_path().is_file(), (
        f"state.json not written under CJK root {cjk_appdata}"
    )
    assert wizard_state.needs_wizard() is False, (
        "after saving first_run_complete=True under a CJK root, "
        "needs_wizard() still returned True — the saved state did not "
        "round-trip (likely save/load resolved different CJK paths)"
    )


# ---------------------------------------------------------------------------
# Long-path budget — the static (no-I/O) half of #26.
#
# Windows' default MAX_PATH is 260 chars unless LongPathsEnabled is set
# (which the no-UAC installer cannot set). This asserts the deepest path
# Yamabiko itself produces — the shisa GGUF in the HF cache, under a
# generously-long username — stays under that ceiling, so a default
# install never silently fails to load a model for path-length reasons.
# ---------------------------------------------------------------------------

def test_worst_case_shipped_path_fits_default_max_path():
    """The deepest path Yamabiko ships fits in the default 260-char MAX_PATH.

    Builds the longest concrete path the app resolves: the LONGEST
    translator-tier GGUF path inside the HF hub cache, under a
    40-character Windows username (a deliberately generous upper bound —
    a 40-char username is already rare). HF cache layout is
    ``<HF_HOME>/hub/models--<repo with '/'->'-'->'-'>/snapshots/<40-char
    commit sha>/<filename>``.

    Seeded defect: give a tier pin in models.py a much longer
    filename/repo, or add another nesting level to the cache-path
    assembly here to model a deeper real layout — the assembled length
    crosses 260 and this goes red, flagging that a default install would
    need LongPathsEnabled.
    """
    from yamabiko import models

    # Deepest *shipped* root: %LOCALAPPDATA% under a long username +
    # the app's own subtree, matching hf_env.py
    # (app_data_dir()/"hf-cache") and the real HF hub cache layout.
    username = "W" * 40  # generous upper bound on a real Windows username
    local_appdata = f"C:\\Users\\{username}\\AppData\\Local"
    hf_home = f"{local_appdata}\\Yamabiko\\hf-cache"

    # HF replaces '/' with '--' in the repo id for the cache dir name.
    commit_sha = "0" * 40  # a git commit sha is always 40 hex chars
    deepest = max(
        (
            f"{hf_home}\\hub\\models--{pin.repo.replace('/', '--')}\\"
            f"snapshots\\{commit_sha}\\{pin.filename}"
            for pin in models.TRANSLATOR_PINS.values()
        ),
        key=len,
    )

    assert len(deepest) < 260, (
        f"deepest shipped path is {len(deepest)} chars (>= 260 MAX_PATH); "
        f"a default install under a {len(username)}-char username would "
        f"fail to load the model without LongPathsEnabled.\n  path: {deepest}"
    )
