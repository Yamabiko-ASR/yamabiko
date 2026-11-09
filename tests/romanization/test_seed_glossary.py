"""Schema and content guards for the shipped seed glossary.

These tests are intentionally strict: the seed is authored by hand
(or adapted from upstream data) and regressions manifest as subtle
romanization mistakes (`Tokyo` vs `Tōkyō`) that are easy to miss in
review but quietly break every downstream furigana/romaji feature.
"""
import re
from pathlib import Path

import yaml

from yamabiko.romanization.glossary_yaml import load_yaml_glossary


_SEED_PATH = Path(__file__).parent.parent.parent / "yamabiko" / "glossaries" / "_seed.yaml"
_HIRAGANA_RE = re.compile(r"^[ぁ-ゖー]+$")
# Start-of-word unromanized digraphs that indicate a missing macron.
# Checked per-word, so compound names like "New York" split on whitespace.
_DIGRAPHS = ("Ou", "Oo", "Uu", "Aa", "Ee")


def _entries():
    return load_yaml_glossary(_SEED_PATH)


def test_seed_file_exists():
    assert _SEED_PATH.exists(), f"missing seed file at {_SEED_PATH}"


def test_all_entries_have_en():
    for surface, entry in _entries().items():
        assert entry.en, f"{surface} missing en"


def test_all_entries_have_reading():
    for surface, entry in _entries().items():
        assert entry.reading, f"{surface} missing reading"


def test_all_readings_are_hiragana():
    for surface, entry in _entries().items():
        assert _HIRAGANA_RE.match(entry.reading), (
            f"{surface} reading={entry.reading!r} contains non-hiragana"
        )


def test_no_unromanized_digraphs_at_word_start():
    for surface, entry in _entries().items():
        for word in entry.en.split():
            for dg in _DIGRAPHS:
                assert not word.startswith(dg), (
                    f"{surface} en={entry.en!r} word {word!r} "
                    f"starts with {dg!r} — missing macron (use Ō/ū/ā/ē)"
                )


def test_no_duplicate_surfaces():
    # load_yaml_glossary collapses duplicates silently; parse raw to catch
    # duplicate keys at file level.
    raw = yaml.safe_load(_SEED_PATH.read_text(encoding="utf-8"))
    keys = list(raw.get("entries", {}).keys())
    assert len(keys) == len(set(keys)), "duplicate keys in _seed.yaml"


def test_has_at_least_50_entries():
    assert len(_entries()) >= 50


def test_all_47_prefectures_present():
    # The 47 prefecture surfaces by plain form (not suffixed with 都/府/県/道).
    prefectures = {
        "北海道", "青森", "岩手", "宮城", "秋田", "山形", "福島",
        "茨城", "栃木", "群馬", "埼玉", "千葉", "東京", "神奈川",
        "新潟", "富山", "石川", "福井", "山梨", "長野", "岐阜",
        "静岡", "愛知", "三重", "滋賀", "京都", "大阪", "兵庫",
        "奈良", "和歌山", "鳥取", "島根", "岡山", "広島", "山口",
        "徳島", "香川", "愛媛", "高知", "福岡", "佐賀", "長崎",
        "熊本", "大分", "宮崎", "鹿児島", "沖縄",
    }
    entries = _entries()
    missing = prefectures - set(entries.keys())
    assert not missing, f"missing prefectures: {sorted(missing)}"
