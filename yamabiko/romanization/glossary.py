"""Per-genre glossary of proper-noun translations.

Stores `{jp_surface: GlossaryEntry}` where each entry carries the canonical
English form and (optionally) the hiragana reading used for furigana.

Lookup priority: user.yaml (hand-curated) > shipped seed > active genre's
on-disk JSON. The user file is read-only from the manager's perspective.

The per-genre JSON is a legacy layer. The learning loop that once wrote it
was removed, so nothing adds entries at runtime any more; it survives only
so readings a prior version auto-learned still feed furigana on upgraded
installs (its entries are never served as translation hints -- see
RomanizationPipeline). It is re-flushed on a genre switch through an atomic
temp-file + fsync + os.replace so a crash cannot truncate it.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

_log = logging.getLogger(__name__)

VALID_GENRES = ("variety", "drama", "news", "anime")


@dataclass
class GlossaryEntry:
    en: str
    reading: str = ""
    count: int = 1
    first_seen: str = ""

    def __post_init__(self) -> None:
        if not self.first_seen:
            self.first_seen = datetime.now(timezone.utc).isoformat(timespec="seconds")


class GlossaryManager:
    """Genre-aware glossary store with JSON + YAML backing files.

    Thread-safe via an internal lock. Writes to the genre JSON go through
    save() (an atomic temp-file + fsync + os.replace); a genre switch
    flushes the outgoing genre first.
    """

    def __init__(
        self,
        glossaries_dir: Path,
        genre: str,
        seed_path: Path | None = None,
    ):
        if genre not in VALID_GENRES:
            raise ValueError(f"invalid genre: {genre!r}; must be one of {VALID_GENRES}")
        self._dir = Path(glossaries_dir)
        # Ensure the (now user-writable) dir exists so _save_unlocked's
        # temp-file write can't FileNotFound on the first save, even if
        # seeding from the bundle failed or an override dir doesn't exist
        # yet.
        self._dir.mkdir(parents=True, exist_ok=True)
        self._genre = genre
        self._lock = threading.Lock()
        self._user: dict[str, GlossaryEntry] = {}
        self._entries: dict[str, GlossaryEntry] = {}
        self._seed: dict[str, GlossaryEntry] = {}
        self._load_user()
        self._load_genre()
        if seed_path is not None:
            from .glossary_yaml import load_yaml_glossary
            self._seed = load_yaml_glossary(seed_path)

    # ---- public API ----

    def lookup(self, jp: str) -> GlossaryEntry | None:
        # The legacy genre layer comes last: its machine-learned readings
        # (日本 -> にっぽん) must not override the curated seed (にほん).
        with self._lock:
            if jp in self._user:
                return self._user[jp]
            if jp in self._seed:
                return self._seed[jp]
            return self._entries.get(jp)

    def lookup_user(self, jp: str) -> GlossaryEntry | None:
        """user.yaml layer only: the hand-curated, authoritative layer.
        The hint path checks it BEFORE the series gazetteer (spec contract:
        a user-curated spelling always wins)."""
        with self._lock:
            return self._user.get(jp)

    def lookup_seed(self, jp: str) -> GlossaryEntry | None:
        """Shipped seed layer only. Deliberately EXCLUDES the writable genre
        JSON: legacy v0.1.0-beta installs' genre files hold machine-romanized
        entries auto-added by the removed learning loop (incl. macron-corrupted
        forms the deleted heal pass used to repair); hinting those as exact
        English names would corrupt translations. build_ruby still uses
        lookup() (all layers) for readings."""
        with self._lock:
            return self._seed.get(jp)

    def set_genre(self, genre: str) -> None:
        if genre not in VALID_GENRES:
            raise ValueError(f"invalid genre: {genre!r}")
        with self._lock:
            self._save_unlocked()
            self._genre = genre
            self._entries = {}
            self._load_genre_unlocked()

    def save(self) -> None:
        with self._lock:
            self._save_unlocked()

    def current_genre(self) -> str:
        with self._lock:
            return self._genre

    # ---- internals ----

    def _load_user(self) -> None:
        from .glossary_yaml import load_yaml_glossary
        self._user = load_yaml_glossary(self._dir / "user.yaml")

    def _load_genre(self) -> None:
        with self._lock:
            self._load_genre_unlocked()

    def _load_genre_unlocked(self) -> None:
        path = self._dir / f"{self._genre}.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            entries = {
                jp: GlossaryEntry(**payload)
                for jp, payload in data.get("entries", {}).items()
            }
        except (json.JSONDecodeError, TypeError, ValueError, OSError) as exc:
            # A corrupt/truncated genre JSON must NOT crash startup (it is read
            # unconditionally from __init__). Quarantine it and start the genre
            # empty -- mirrors load_yaml_glossary's guard for the YAML side.
            backup = path.with_suffix(".json.corrupt.bak")
            _log.warning(
                "corrupt genre glossary %s (%s); starting empty, backing up to %s",
                path, exc, backup.name,
            )
            try:
                path.replace(backup)
            except OSError:
                pass
            return
        self._entries.update(entries)

    def _save_unlocked(self) -> None:
        path = self._dir / f"{self._genre}.json"
        payload = {
            "version": 1,
            "entries": {jp: asdict(e) for jp, e in self._entries.items()},
        }
        data = json.dumps(payload, ensure_ascii=False, indent=2)
        # Atomic write: temp file in the same dir + os.replace, so an
        # interrupted write (crash / power loss) can't truncate or corrupt the
        # genre glossary -- save() runs on a genre switch and at shutdown
        # (same pattern config_writer.py already uses).
        fd, tmp = tempfile.mkstemp(
            dir=str(self._dir), prefix=f".{self._genre}.", suffix=".json.tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(data)
                # os.replace journals the NTFS rename atomically
                # but does NOT flush data blocks. Flush the userspace
                # buffer, then fsync (Windows: FlushFileBuffers), so
                # power loss after the rename can't publish a zero-
                # length/partial genre JSON -- which the corrupt-file
                # guard would then quarantine, silently dropping every
                # entry in the genre.
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
