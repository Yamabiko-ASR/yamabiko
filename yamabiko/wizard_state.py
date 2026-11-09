"""Persistence for first-run wizard results.

Stored as JSON at `paths.state_path()`. The wizard writes this once on
completion; `main.py` reads `needs_wizard()` at startup to branch between
wizard and pipeline.

Contract lives in `WizardResult` — adding a field? Bump `wizard_version`
and handle the old shape gracefully in `load` so users who ran an older
wizard don't get forced through it again.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from yamabiko import __version__, models, paths

_log = logging.getLogger(__name__)


@dataclass
class WizardResult:
    first_run_complete: bool = False
    wizard_version: str = __version__
    wizard_completed_at: Optional[str] = None
    audio_device_name: Optional[str] = None
    genre: str = "variety"
    llama_smoke_ok: bool = False


_LEGACY_IGNORED_FIELDS = frozenset({"livekit_license_accepted"})
_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    "first_run_complete": (bool,),
    "wizard_version": (str,),
    "wizard_completed_at": (str, type(None)),
    "audio_device_name": (str, type(None)),
    "genre": (str,),
    "llama_smoke_ok": (bool,),
}
_COMPLETE_FIELDS = frozenset(_FIELD_TYPES)


def _decode_result(data: object) -> WizardResult:
    if not isinstance(data, dict):
        raise ValueError(
            f"state.json root must be an object, got {type(data).__name__}"
        )
    unknown = set(data) - set(_FIELD_TYPES) - _LEGACY_IGNORED_FIELDS
    if unknown:
        raise ValueError(f"state.json has unsupported fields: {sorted(unknown)}")
    clean = {key: value for key, value in data.items() if key in _FIELD_TYPES}
    for key, value in clean.items():
        if type(value) not in _FIELD_TYPES[key]:
            raise ValueError(
                f"state.json field {key} has unsupported type "
                f"{type(value).__name__}"
            )
    if clean.get("first_run_complete") is True:
        missing = _COMPLETE_FIELDS - set(clean)
        if missing:
            raise ValueError(
                f"completed state.json is missing fields: {sorted(missing)}"
            )
    return WizardResult(**clean)


def load() -> Optional[WizardResult]:
    """Read state.json. Return None if missing or unreadable.

    Corrupt JSON, OS errors, and schema mismatches are all
    collapsed into ``None`` — intentionally, so a bad file causes the
    wizard to re-run rather than crashing the app. The trade-off is
    that we can't distinguish "fresh install" from "corrupt state" at
    this layer. Log output preserves the distinction for debugging.
    """
    path = paths.state_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _decode_result(data)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        _log.warning("state.json unreadable (%s); treating as missing", exc)
        return None


def save(result: WizardResult) -> WizardResult:
    """Persist result to state.json and return the effective object.

    If ``first_run_complete`` is True and ``wizard_completed_at`` is None,
    the returned object carries a freshly-stamped ISO-8601 UTC timestamp.
    The input ``result`` is NOT mutated — use the return value if you
    need the stamped timestamp.
    """
    paths.ensure_dirs()
    if result.first_run_complete and result.wizard_completed_at is None:
        result = dataclasses.replace(
            result,
            wizard_completed_at=datetime.now(timezone.utc).isoformat(),
        )
    path = paths.state_path()
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Flush + fsync before the rename, same
    # idiom as config_writer._atomic_write -- without it a power loss
    # in the post-rename window can publish a zero-length state.json,
    # which needs_wizard() reads as "run the first-run wizard again".
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(result), indent=2, ensure_ascii=False))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        # A write/flush/fsync failure (e.g. ENOSPC) must not leave the
        # partial state.json.tmp behind as litter; same cleanup idiom as
        # config_writer._atomic_write. Re-raise so the caller still sees
        # the original failure.
        try:
            os.unlink(tmp)
        except OSError:
            # Swallow KEPT (the original write error re-raised below is
            # the failure that matters), but warn: silent .tmp litter in
            # %LOCALAPPDATA%\Yamabiko was undiagnosable from field logs.
            _log.warning(
                "could not remove partial %s after a failed save; "
                "leaving it behind", tmp.name, exc_info=True,
            )
        raise
    return result


def needs_wizard() -> bool:
    """True when the first-run wizard must run before the pipeline.

    Three triggers:
      - state.json missing/unreadable (fresh install, corrupt state);
      - ``first_run_complete`` False (wizard never finished, or cancelled);
      - the models the wizard is responsible for downloading are absent
        from the HF cache (``models.required_models_cached()``). This is
        the UPGRADE-migration path: the NSIS installer deliberately
        preserves state.json across upgrades, and the wizard is the only
        downloader -- so a release that changes a model pin (e.g. the
        whisper deepdml -> kotoba -> efwkjn -> Kodama-1 repo swaps) would
        otherwise boot the pipeline with the new model uncached: a surprise
        multi-GB synchronous download inside startup, or an offline hard
        failure.

    The model probe is LOCAL-ONLY (cache stats, no network), runs only
    when the state file says the wizard already completed, and fails OPEN
    (probe exception ⇒ "cached") so a probe bug can never force the
    wizard on every launch of a healthy install.
    """
    result = load()
    if result is None or not result.first_run_complete:
        return True
    return not models.required_models_cached()


def speech_model_update_pending(result: Optional[WizardResult]) -> bool:
    """True when setup opens on an install that finished setup before and
    the speech model the runtime will load is not cached: an update moved
    the speech model pin. ``result`` is the state setup opened with;
    ``wizard_completed_at`` marks a finished setup, and setup's interim
    saves keep it (wizard.py ``_current_result``), so a cancelled update
    still counts. Drives the Welcome step's update line. Never raises: a
    probe error reads as False, which only hides that line.
    """
    if result is None or result.wizard_completed_at is None:
        return False
    try:
        return models.default_whisper_missing()
    except Exception:
        _log.warning(
            "speech model cache probe failed; Welcome shows no update line",
            exc_info=True,
        )
        return False


def reset() -> None:
    """Delete state.json so the wizard runs again on next launch."""
    path = paths.state_path()
    if path.exists():
        path.unlink()
