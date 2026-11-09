"""The pilot export's CTranslate2 folder against the shipped one, outside
``model.bin``: a converter that re-serializes a JSON file (same content,
other bytes) must not fail the ``ct2_files_equal`` check, and a JSON file
whose content differs must."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.asr_finetune.common import adopt_reference_json, compare_model_dirs, ct2_files_equal

_VOCABULARY = ["<|endoftext|>", "<|startoftranscript|>", "あ", "ねえ"]


def _folders(tmp_path: Path, converted_vocabulary: str) -> tuple[Path, Path]:
    reference, converted = tmp_path / "shipped", tmp_path / "ct2"
    for folder in (reference, converted):
        folder.mkdir()
        (folder / "tokenizer.json").write_bytes(b'{"model": {"type": "BPE"}}')
    (reference / "vocabulary.json").write_text(json.dumps(_VOCABULARY, ensure_ascii=False), encoding="utf-8")
    (converted / "vocabulary.json").write_text(converted_vocabulary, encoding="utf-8")
    return reference, converted


def test_ft_export_gives_reserialized_json_the_shipped_bytes_and_passes(tmp_path: Path) -> None:
    reference, converted = _folders(tmp_path, json.dumps(_VOCABULARY, indent=4))  # \u escapes, other bytes

    replaced = adopt_reference_json(reference, converted)
    files = compare_model_dirs(reference, converted)

    assert (converted / "vocabulary.json").read_bytes() == (reference / "vocabulary.json").read_bytes()
    assert sorted(replaced) == ["vocabulary.json"]
    assert files["files"]["vocabulary.json"]["identical"]
    assert ct2_files_equal(files)


def test_ft_export_keeps_content_different_json_and_fails(tmp_path: Path) -> None:
    different = json.dumps([*_VOCABULARY[:-1], "ねぇ"], indent=4)
    reference, converted = _folders(tmp_path, different)

    replaced = adopt_reference_json(reference, converted)
    files = compare_model_dirs(reference, converted)

    assert replaced == {}
    assert (converted / "vocabulary.json").read_text(encoding="utf-8") == different
    assert not ct2_files_equal(files)
