"""Per-series proper-noun gazetteer built from AniList characters. Maps JA surface
forms to AniList's romanized character names (AniList's `full` name, which is not
necessarily an official English spelling); used as SOFT prompt hints (never source
substitution). A miss returns None so the pipeline hints nothing."""
from __future__ import annotations
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from yamabiko import paths
from yamabiko.fsutil import atomic_write_text

from . import fuzzy as _fuzzy
from .anilist import Character

_log = logging.getLogger(__name__)

_KATA = re.compile(r"^[゠-ヿ]+$")   # a JA-looking (katakana) key
# A key made only of kanji (and the iteration mark 々): a native name
# AniList stores unbroken, family name first (竈門炭治郎).
_KANJI_NAME = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff々]+$")

def _key(s: str) -> str:
    return s.strip()


def feature_enabled(cfg: dict) -> bool:
    """Effective on-switch for the whole series-gazetteer feature.

    Requires BOTH gazetteer.enabled and romanization.enabled: a series hint
    only ever reaches the model through RomanizationPipeline, so
    gazetteer-on with romanization-off is UI that silently does nothing.
    Every entry point (hotkey bind, palette / tray construction, boot
    restore, the set_series wire command) gates on this one helper so the
    decision cannot drift per entry point. Warns when the two flags diverge
    so the silent-no-op config is visible in the log.
    """
    # Fallbacks mirror the SHIPPED defaults (default_config.yaml ships
    # both flags true): a key dropped by a validator bug or a hand edit
    # must fail loud (feature visibly on) rather than silently disable
    # the feature. test_gazetteer pins this against the yaml.
    #
    # Keep each section literal adjacent to the leaf it reads: the
    # preflight consumer gate anchors a collision-prone leaf ("enabled"
    # names two different keys) to its parent within a short window, so
    # prose wedged between the two reads it as an orphan.
    gaz = cfg.get("gazetteer") or {}
    rom = cfg.get("romanization") or {}
    enabled = bool(gaz.get("enabled", True))
    if enabled and not bool(rom.get("enabled", True)):
        _log.warning(
            "gazetteer.enabled is true but romanization.enabled is false; "
            "the series feature stays off (hints flow through the "
            "romanization pipeline)"
        )
        return False
    return enabled


def gazetteer_path(media_id: int) -> Path:
    """Per-series disk-cache path: <app_data_dir>/gazetteers/<media_id>.json.

    Always routed through paths.app_data_dir() -- never a hardcoded or
    cwd-relative path (the frozen-bundle trap; CLAUDE.md hard rule #8).
    """
    return paths.app_data_dir() / "gazetteers" / f"{media_id}.json"


def cover_path(media_id: int) -> Path:
    """The picked series' saved poster: cover-<media_id>.png beside its
    gazetteer cache. The series palette writes it for the active series, so
    a series restored at boot shows its cover in the tray flyout and the
    series bar, offline too. The diagnostics zip never reads this folder
    (it collects the logs folder only)."""
    return gazetteer_path(media_id).parent / f"cover-{media_id}.png"


# Cache envelope schema written by SeriesGazetteer.save. Bump when an
# extractor fix should invalidate every existing cache: any stamp other
# than this exact version -- lower, missing (the v1 flat-map era), or
# HIGHER (a newer binary's cache met after a downgrade, whose map
# semantics this code cannot know) -- reads as "rebuild on the next
# pick", with the old file kept as the offline fallback only when its
# shape is one this code understands.
_SCHEMA_VERSION = 2
# Pick-time freshness horizon: a cache built longer ago than this is
# rebuilt on the next pick (an airing series' cast must not stay frozen
# at first-pick state forever). Network still happens ONLY on a user
# pick -- there is deliberately no background/startup refresh, so the
# verified offline-launch property holds.
_MAX_CACHE_AGE_S = 24 * 60 * 60


def cache_needs_rebuild(path: Path, *, now=time.time) -> bool:
    """Pick-time freshness gate for the on-disk gazetteer cache at `path`.

    True when the next pick should attempt a (re)build: the cache is
    absent or unreadable, predates the versioned envelope (a v1 flat map
    never benefits from extractor fixes otherwise), claims any schema
    other than the current one, or its built_at is _MAX_CACHE_AGE_S or
    more in the past -- or ahead of the clock entirely. Read-only and
    never raises: the file itself stays put whatever shape it is in,
    because the caller (tray._GazetteerBuildThread) keeps it as the
    offline fallback -- a failed rebuild degrades to loading the old
    cache via `load`, never to a lost series. `now` is the injectable
    clock (tests pin the 24 h boundary without sleeping).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    if not isinstance(data, dict):
        return True
    schema = data.get("schema")
    if (not isinstance(schema, int) or isinstance(schema, bool)
            or schema != _SCHEMA_VERSION):
        return True
    built_at = data.get("built_at")
    if not isinstance(built_at, (int, float)) or isinstance(built_at, bool):
        return True
    if built_at > now():
        # A clock that was ahead at save time and corrected afterwards
        # (NTP) would otherwise read as fresh until the wall clock
        # catches up, freezing an airing series' cast meanwhile.
        return True
    return (now() - built_at) >= _MAX_CACHE_AGE_S


def active_series_path() -> Path:
    """Pointer file naming the user's currently-selected series:
    <app_data_dir>/active_series.json  ({"media_id": int, "label": str,
    "native"?: str}). Read at pipeline boot and rewritten on every palette
    selection. Routed through paths.app_data_dir() (frozen-bundle trap;
    hard rule #8)."""
    return paths.app_data_dir() / "active_series.json"


def save_active_series(media_id: int, label: str, native: str | None = None) -> None:
    """Atomically persist the active-series pointer (tmp file + fsync + os.replace
    so a crash mid-write never leaves a half-written pointer that breaks boot
    restore, and a failed write leaves no tmp litter). ``native`` is the
    series' Japanese title for the overlay banner; omitted when unknown."""
    data: dict = {"media_id": media_id, "label": label}
    if native:
        data["native"] = native
    atomic_write_text(active_series_path(), json.dumps(data, ensure_ascii=False))


def clear_active_series() -> None:
    """Forget the active series (the tray's Clear): delete the pointer so a
    starting pipeline restores none, the way a pick persists one. A missing
    pointer is already clear; any other OSError reaches the caller, which
    logs it like a failed save."""
    active_series_path().unlink(missing_ok=True)


def load_active_series() -> "tuple[int, str, str | None] | None":
    """Read the pointer -> (media_id, label, native), or None when it is
    absent, unreadable, malformed, missing a valid int media_id, or missing a
    str label. A missing or non-string native is None. Never raises: a bad
    pointer means "no series selected"."""
    path = active_series_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    media_id = data.get("media_id")
    label = data.get("label")
    if not isinstance(media_id, int) or isinstance(media_id, bool):
        return None
    if not isinstance(label, str):
        return None
    native = data.get("native")
    return media_id, label, native if isinstance(native, str) and native else None


# Crude katakana->romaji used ONLY to score zip orientations in
# _align_tokens -- never emitted as output. Coverage gaps just lower a
# similarity score; they cannot corrupt the map.
_KATA_ROMAJI = {
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo",
    "ラ": "ra", "リ": "ri", "ル": "ru", "レ": "re", "ロ": "ro",
    "ワ": "wa", "ヲ": "o", "ン": "n",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
    "ヴ": "vu",
}
_KATA_SMALL = {
    "ャ": "ya", "ュ": "yu", "ョ": "yo",
    "ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o",
}


def _kata_to_romaji(s: str) -> str:
    """Katakana -> rough romaji for alignment scoring; "" if any char is
    outside the table (kanji, Latin), which callers treat as unalignable."""
    out: list[str] = []
    sokuon = False
    for ch in s:
        if ch == "ッ":
            sokuon = True
            continue
        if ch == "ー":
            if out and out[-1][-1] in "aiueo":
                out.append(out[-1][-1])
            continue
        if ch in _KATA_SMALL and out:
            base = out[-1]
            glide = _KATA_SMALL[ch]
            if base[-1] in "aiueo":
                stem = base[:-1]
                if glide[0] == "y" and stem.endswith(("sh", "ch", "j")):
                    glide = glide[1:]
                out[-1] = stem + glide
            else:
                out.append(glide)
            continue
        r = _KATA_ROMAJI.get(ch)
        if r is None:
            return ""
        if sokuon:
            r = r[0] + r
            sokuon = False
        out.append(r)
    return "".join(out)


def _align_tokens(parts: list[str], tokens: list[str]) -> "list[str] | None":
    """Order-reconcile English tokens against katakana native parts.

    AniList's `full` is Western order while `native` may be either order,
    so a positional zip can pair every part with the WRONG token. Score
    both orientations by transliterating each part and fuzzy-matching its
    would-be token; return tokens in winning order, or None when neither
    orientation is confident (the caller then maps every part to the full
    English name instead of risking a swapped single token).
    """
    def score(order: list[str]) -> float:
        total = 0.0
        for jp, tok in zip(parts, order):
            r = _kata_to_romaji(jp)
            if not r:
                return 0.0
            total += SequenceMatcher(None, r, tok.lower()).ratio()
        return total / len(parts)

    straight, flipped = score(tokens), score(tokens[::-1])
    if max(straight, flipped) < 0.6:
        return None
    if straight == flipped and len(tokens) > 1:
        return None
    return tokens if straight >= flipped else tokens[::-1]


def _quarantine(path: Path) -> None:
    """Move a corrupt gazetteer cache aside so the next pick refetches.

    A corrupt file also reads as unparseable to `cache_needs_rebuild`
    (rebuild path), but it must not linger as the rebuild-failure
    fallback either: a fallback that loads to None is "no hints", so the
    file is moved aside the moment corruption is observed."""
    backup = path.with_suffix(".json.corrupt.bak")
    _log.warning("corrupt gazetteer %s; quarantining to %s", path, backup.name)
    try:
        path.replace(backup)
    except OSError:
        _log.warning("could not quarantine %s", path, exc_info=True)


def _is_person_name(tok, pos4: str) -> bool:
    """True when UniDic tags `tok` as a person-name proper noun whose
    sub-type (姓 family / 名 given) is `pos4`."""
    f = tok.feature
    return (
        getattr(f, "pos2", None) == "固有名詞"
        and getattr(f, "pos3", None) == "人名"
        and getattr(f, "pos4", None) == pos4
    )


def _family_given(key: str, tagger, family_en: str) -> "tuple[str, str] | None":
    """Split an all-kanji native name into (family, given) with the tagger,
    or None when the tagging does not clearly say where the split is.

    The first token must be a family name (人名/姓) and the last a given
    name (人名/名), and the token surfaces must spell the key exactly.
    UniDic sometimes splits a given name further (竈門 / 炭 / 治郎); then the
    family token's reading must spell `family_en` (カマド -> kamado), which
    pins the boundary after the first token. Without that check a split
    such as 祈 / 本 / 里香 (family 祈本) would hint 本里香 as the given name.
    """
    toks = list(tagger(key))
    if len(toks) < 2 or "".join(t.surface for t in toks) != key:
        return None
    first, last = toks[0], toks[-1]
    if not (_is_person_name(first, "姓") and _is_person_name(last, "名")):
        return None
    if len(toks) > 2:
        kana = getattr(first.feature, "kana", None) or ""
        if _kata_to_romaji(kana) != family_en.lower():
            return None
    return first.surface, key[len(first.surface):]


@dataclass
class SeriesGazetteer:
    map: dict[str, str] = field(default_factory=dict)
    # Folded fuzzy index, built lazily on first fuzzy_lookup and cached so a
    # hot loop does not re-fold every key per sentence. Not part of the map's
    # identity: init=False (SeriesGazetteer(data) still takes one positional),
    # compare/repr=False so equality and dumps stay about `map`.
    _fuzzy_index: "_fuzzy.FuzzyIndex | None" = field(
        default=None, init=False, compare=False, repr=False
    )

    @classmethod
    def from_characters(cls, chars: list[Character]) -> "SeriesGazetteer":
        m: dict[str, str] = {}
        for c in chars:
            if c.role == "BACKGROUND":
                # BACKGROUND "names" include plain generic nouns (店員 /
                # 職員, live-observed on AniList): an entry would hint
                # 店員 -> "Tenin" on ordinary dialogue, so BACKGROUND
                # characters never become entries.
                continue
            en = (c.full or "").strip()
            if not en:
                continue
            if c.native:
                nat = _key(c.native)
                m.setdefault(nat, en)
                parts = [p for p in re.split(r"[・=＝]", nat) if p]
                en_tokens = en.split()
                aligned = (
                    _align_tokens(parts, en_tokens)
                    if len(parts) == len(en_tokens) else None
                )
                if aligned is not None:
                    for jp, et in zip(parts, aligned):
                        m.setdefault(_key(jp), et)
                else:
                    for jp in parts:
                        m.setdefault(_key(jp), en)   # safe fallback: full EN
            for alt in c.alternative:
                if alt and _KATA.match(alt.strip()):
                    m.setdefault(_key(alt), en)
        return cls(m)

    def lookup(self, surface: str) -> str | None:
        return self.map.get(_key(surface))

    def fuzzy_lookup(self, surface: str, *, max_distance: float) -> str | None:
        """Best-effort recovery of an ASR-garbled katakana `surface` to its
        exact English name, or None. STRICT by construction (see
        fuzzy.match): a wrong-name hint is worse than none, so this refuses
        short surfaces, non-katakana noise, over-distance garbles, and any
        near-tie between two DIFFERENT characters.

        Only ever a fallback BELOW the exact `lookup` in the pipeline's hint
        precedence -- it never overrides an exact hit, and the source is
        never mutated (soft prompt hint only, same contract as v1).

        The folded key index is built on first use and cached on the instance.
        """
        if self._fuzzy_index is None:
            self._fuzzy_index = _fuzzy.build_index(self.map)
        return _fuzzy.match(surface, self._fuzzy_index, max_distance=max_distance)

    def with_name_parts(self, tagger) -> "SeriesGazetteer":
        """Return a copy that also maps the family and given parts of each
        all-kanji native name. AniList stores 竈門炭治郎 unbroken, so
        from_characters cannot split it, while dialogue says 竈門 or 炭治郎.

        A part is added only when the tagger reads the key as a family name
        followed by a given name (see _family_given) and the romanized name
        is exactly two words; AniList's `full` is given name first, so the
        family part takes the last word and the given part the first. Parts
        shorter than two characters are skipped (a lone kanji is too common
        a word to hint), and existing keys are never overwritten.
        """
        m = dict(self.map)
        for key, en in self.map.items():
            if not _KANJI_NAME.match(key):
                continue
            en_tokens = en.split()
            if len(en_tokens) != 2:
                continue
            parts = _family_given(key, tagger, en_tokens[-1])
            if parts is None:
                continue
            family, given = parts
            if len(family) >= 2:
                m.setdefault(family, en_tokens[-1])
            if len(given) >= 2:
                m.setdefault(given, en_tokens[0])
        return SeriesGazetteer(m)

    def save(self, path: Path, *, now=time.time) -> None:
        """Write the gazetteer to `path` as UTF-8 JSON (JA keys stay
        readable), atomically (tmp + fsync + os.replace) so a crash
        mid-write never publishes a truncated cache that quarantine would
        then discard. Persists the versioned envelope
        ``{"schema": _SCHEMA_VERSION, "built_at": <epoch s>, "map": ...}``
        so `cache_needs_rebuild` can tell a fresh cache from a stale or
        pre-versioning one; `now` is the injectable clock for tests."""
        atomic_write_text(
            path,
            json.dumps(
                {"schema": _SCHEMA_VERSION, "built_at": int(now()),
                 "map": self.map},
                ensure_ascii=False, indent=2,
            ),
        )

    @staticmethod
    def load(path: Path) -> "SeriesGazetteer | None":
        """Load a gazetteer previously written by `save`.

        Accepts BOTH shapes this code understands: the current-schema
        envelope `save` writes AND the legacy v1 flat map -- a stale
        cache is the offline fallback when a pick-time rebuild fails, so
        both must keep loading (freshness decisions live in
        `cache_needs_rebuild`, never here). An envelope stamped with any
        OTHER schema is refused like a corrupt file: its map semantics
        belong to a binary this code is not, so serving it could hint
        wrong. Returns None if the file is absent, unreadable, or not
        valid JSON -- callers treat that identically to "no gazetteer for
        this series" (a soft-hint miss), never a crash. A corrupt or
        wrong-schema file (non-dict JSON, an envelope without a dict map,
        or a non-current schema stamp) is quarantined to
        `.json.corrupt.bak` so the next pick refetches instead of
        short-circuiting on the still-present file.
        """
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _quarantine(path)
            return None
        if not isinstance(data, dict):
            _quarantine(path)
            return None
        if "schema" in data:
            # Versioned envelope. A legacy flat map cannot carry a
            # "schema" key (its values are all strings, its keys JA
            # surfaces), so this discriminates the two shapes safely.
            if data.get("schema") != _SCHEMA_VERSION:
                _quarantine(path)
                return None
            m = data.get("map")
            if not isinstance(m, dict):
                _quarantine(path)
                return None
            return SeriesGazetteer(m)
        return SeriesGazetteer(data)  # legacy v1 flat map


# Relations carry: related-season casts share speakers with the picked
# media (a PREQUEL protagonist keeps talking in the SEQUEL), so up to
# _CARRY_MAX_RELATED related ANIME, ranked by _CARRY_RELATION_ORDER, are
# merged in at _CARRY_MAX_PAGES pages each. The caps keep a whole pick
# inside the ~13-request budget under AniList's 30 req/min limit.
_CARRY_RELATION_ORDER = ("PREQUEL", "SEQUEL", "PARENT", "SIDE_STORY")
_CARRY_MAX_RELATED = 2
_CARRY_MAX_PAGES = 2

# Staged pick progress (the series bar's loading line, phase-5 L10): shares
# of the whole build. The intel request, then the picked cast's pages, then
# the relations carry. The caller fills to 1.0 when the build is saved.
_PROGRESS_PROFILE = 0.15
_PROGRESS_CAST = 0.6
_PROGRESS_CARRY = 0.25


def _carry_media_ids(relations) -> list[int]:
    """Related media ids worth carrying: eligible relation types ranked
    PREQUEL > SEQUEL > PARENT > SIDE_STORY (stable within a type), capped
    at _CARRY_MAX_RELATED. `relations` is SeriesProfile.relations, which
    is ANIME-only by ingest contract."""
    ranked = sorted(
        (r for r in relations
         if r.relation_type in _CARRY_RELATION_ORDER and r.media_id is not None),
        key=lambda r: _CARRY_RELATION_ORDER.index(r.relation_type),
    )
    return [r.media_id for r in ranked[:_CARRY_MAX_RELATED]]


def _makes_entries(c: Character) -> bool:
    """True when from_characters will index this character: not a
    BACKGROUND extra, and carrying a non-empty English full name. Only
    such characters may claim a native-name dedup slot in the relations
    carry -- a character from_characters filters out produces no entry
    itself, so letting it occupy the slot would silently block a
    legitimately-named same-native character from a related media
    (unsignaled data loss: neither ends up in the gazetteer)."""
    return c.role != "BACKGROUND" and bool((c.full or "").strip())


def build_series_gazetteer(
    client, media_id: int, cache_dir: "Path | None" = None, *,
    on_progress: "Callable[[float], None] | None" = None,
) -> SeriesGazetteer:
    """Full per-pick build: the picked media's characters plus the
    relations carry, with the series profile built (and persisted) from
    the same single intel request.

    The merge is deduped by native name and the PICKED media wins every
    collision: a related character with an already-seen native is dropped
    whole, so not even its alternative aliases can attach a related
    season's spelling. Dedup slots are only ever claimed by characters
    that will actually produce entries (_makes_entries): a BACKGROUND or
    nameless character -- picked or related -- must not block a real
    same-native entry from a later source. A related fetch failing must
    not throw away the already-fetched picked cast -- the pick degrades
    to a single-series gazetteer (logged) instead of failing outright.
    The profile is optional the same way: if its intel request fails, the
    pick continues with no related-season carry and no profile.

    ``on_progress`` hears the build's progress (0..1, staged, monotonic):
    0.15 once the intel request is done or failed, 0.15 + 0.6 x the share
    of the picked cast fetched after each page (0.75 on its last page), then
    0.75 + 0.25 x (i + 1) / n after each of the n related fetches, failed
    ones included.
    """
    # Local import: series_profile resolves its default cache dir via
    # gazetteer_path from THIS module, so a top-level import is circular.
    from . import series_profile

    def report(fraction: float) -> None:
        if on_progress is not None:
            on_progress(fraction)

    try:
        relations = series_profile.build_profile(client, media_id, cache_dir).relations
    except Exception:
        _log.warning(
            "series profile fetch failed; continuing without related seasons",
            exc_info=True)
        relations = []
    report(_PROGRESS_PROFILE)
    chars = list(client.fetch_characters(
        media_id,
        on_page=lambda share: report(_PROGRESS_PROFILE + _PROGRESS_CAST * share),
    ))
    seen = {_key(c.native) for c in chars if c.native and _makes_entries(c)}
    carry_ids = _carry_media_ids(relations)
    cast_done = _PROGRESS_PROFILE + _PROGRESS_CAST
    for i, rel_id in enumerate(carry_ids):
        try:
            related = client.fetch_characters(rel_id, max_pages=_CARRY_MAX_PAGES)
        except Exception:
            # No media id in this line: it names a series the user watches,
            # and the build runs in the overlay process, whose log ships in
            # the diagnostics zip (the tray worker's own rule).
            _log.warning(
                "relations carry: a related media's character fetch failed; "
                "continuing without it", exc_info=True)
            related = []
        report(cast_done + _PROGRESS_CARRY * (i + 1) / len(carry_ids))
        for c in related:
            if not _makes_entries(c):
                # from_characters would drop it anyway; skipping it HERE
                # keeps it from claiming a dedup slot against a later
                # relation's same-native character.
                continue
            nat = _key(c.native) if c.native else None
            if nat is not None:
                if nat in seen:
                    continue
                seen.add(nat)
            chars.append(c)
    return SeriesGazetteer.from_characters(chars)
