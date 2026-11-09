"""Real-path check: tokenizers + HF cache stack is functional.

faster-whisper (the shipped default whisper CT2 model) loads a Rust
`tokenizers.Tokenizer` from the model's ``tokenizer.json`` via the Rust
lib's ``Tokenizer.from_file``. The Rust lib's ``from_pretrained`` has no
offline switch at all — it hits the Hub on every call — so the
production path downloads via the Python SDK (which honours
``local_files_only``) and loads from the landed file. This test mirrors
that exact call shape against the real cached whisper ``tokenizer.json``;
when the real stack breaks (pinned `tokenizers` version incompatible with
the landed file, or Rust binary missing from the wheel) mocked tests
happily pass while the production path 500s.

Skip discipline: the ONLY skippable condition is network/Hub
unavailability, and it is guarded by a narrow huggingface_hub pre-fetch
— the stack-breakage failure modes this file exists to catch
(cache-layout incompatibility, unparseable tokenizer.json, Rust API
drift) run OUTSIDE any try and fail red. An earlier version of this
body wrapped the subject itself in a blanket
``except Exception: pytest.skip``, which skipped on exactly the
defects the module docstring names as its purpose.

Uses the shipped whisper ``tokenizer.json`` (models.WHISPER_REPO at the
pinned models.WHISPER_REVISION) — the exact file faster-whisper loads at
runtime. Cached after first run (WHISPER_FILES includes tokenizer.json,
so the wizard/preflight warm it); with a warm cache this test runs fully
offline and CANNOT skip.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.real  # a test_real_* file that hits the
# network must never run in the unit tier, and must be INCLUDED in
# `pytest -m real` (preflight section 8).


def test_tokenizer_from_file_via_hf_hub_download() -> None:
    """End-to-end: fetch the whisper ``tokenizer.json`` from HF via
    ``hf_hub_download`` and instantiate via the production API
    (``Tokenizer.from_file``, the same call shape faster-whisper uses),
    then encode.

    Seeded-defect check: plant a corrupted ``tokenizer.json`` in an
    isolated HF cache (HF_HOME override) -> ``from_file`` raises
    outside any try -> this fails red (the old body skipped).
    """
    pytest.importorskip("huggingface_hub")
    tokenizers = pytest.importorskip("tokenizers")

    import httpx
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
    )

    from yamabiko import models

    # Narrow network pre-flight: warm the cache (no-op when already
    # cached). ONLY connectivity / Hub-availability classes may skip;
    # anything else here (corrupted cache, parse error) is a stack
    # break and propagates red.
    try:
        tok_path = hf_hub_download(
            repo_id=models.WHISPER_REPO, filename="tokenizer.json",
            revision=models.WHISPER_REVISION,
        )
    except (HfHubHTTPError, LocalEntryNotFoundError,
            OfflineModeIsEnabled, httpx.HTTPError) as exc:
        pytest.skip(f"HF Hub unreachable (env/network): {exc!r}")

    # The subject — OUTSIDE any try. Cache-layout incompatibility,
    # unparseable tokenizer.json, or a Rust-side API rename fails RED.
    tok = tokenizers.Tokenizer.from_file(tok_path)

    encoding = tok.encode("こんにちは、今日は良い天気ですね。")
    assert encoding.ids, "tokenizer returned no token ids for simple input"
    assert len(encoding.tokens) == len(encoding.ids)
