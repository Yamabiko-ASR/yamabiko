"""Decoder targets of the pilot's training windows (plan D3).

A window's label is the shipped decode format with timestamps, without a
prompt: ``<|ja|><|transcribe|><|t_start|>text<|t_end|>...<|endoftext|>``.
The model puts ``<|startoftranscript|>`` in front (the targets shifted right);
the trainer masks the first ``MASKED_PREFIX`` positions (-100), as the decode
forces them (round 4: only ``<|transcribe|>``, so the first predicted
position, where the decode reads ``no_speech_prob``, learns ``<|ja|>``). A
no-speech window's label is ``NOSPEECH_LABELS``: ``<|nospeech|>`` at that
first position, then ``<|endoftext|>``. Token ids are those of the base
export's ``tokenizer.json`` (``load_tokenizer`` checks them).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

EOT, SOT, JA, NOSPEECH = 8631, 8632, 8640, 8737
TS0, TS_STEP, TS_MAX = 8739, 0.02, 30.0
MASKED_PREFIX = 2  # <|ja|> and <|transcribe|> positions get -100
MAX_LABEL_TOKENS = 440
NOSPEECH_LABELS = [NOSPEECH, EOT]
FIRST_POSITIONS = ("masked", "target")  # label_mask: the first predicted position left out, or trained
_FIXED = {"<|endoftext|>": EOT, "<|startoftranscript|>": SOT, "<|ja|>": JA, "<|nospeech|>": NOSPEECH,
          "<|0.00|>": TS0, f"<|{TS_MAX:.2f}|>": TS0 + round(TS_MAX / TS_STEP)}


def timestamp_token(seconds: float) -> int:
    if not 0.0 <= seconds <= TS_MAX:
        raise ValueError(f"timestamp {seconds} outside [0, {TS_MAX}]")
    return TS0 + round(seconds / TS_STEP)


def build_labels(tokenizer, segments: Sequence[tuple[float, float, str]]) -> list[int]:
    """[<|ja|>, <|transcribe|>, ts, text..., ts, ..., <|endoftext|>] for the
    decoder targets; the model shifts them right behind <|startoftranscript|>."""
    transcribe = tokenizer.token_to_id("<|transcribe|>")
    ids = [JA, transcribe]
    last = 0.0
    for start, end, text in segments:
        if start < last or end <= start:
            raise ValueError("segments must be ordered and non-empty")
        ids.append(timestamp_token(start))
        ids.extend(tokenizer.encode(text, add_special_tokens=False).ids)
        ids.append(timestamp_token(end))
        last = end
    ids.append(EOT)
    if len(ids) > MAX_LABEL_TOKENS:
        raise ValueError(f"{len(ids)} label tokens exceed {MAX_LABEL_TOKENS}")
    return ids


def label_length(text_tokens: Sequence[int]) -> int:
    """Tokens of a label whose segments' texts have ``text_tokens`` tokens each."""
    return MASKED_PREFIX + sum(count + 2 for count in text_tokens) + 1


def label_mask(labels: Sequence[int], transcribe: int, first_position: str) -> tuple[int, ...] | None:
    """The label positions a window's loss leaves out (-100), or None for
    labels a run with this ``first_position`` does not train on. A speech
    label (``<|ja|>``, ``<|transcribe|>``, ..., ``<|endoftext|>``) leaves out
    its first ``MASKED_PREFIX`` positions (``masked``, the rounds before 4) or
    only ``<|transcribe|>``'s (``target``: the first predicted position, where
    the decode reads ``no_speech_prob``, learns ``<|ja|>``). A no-speech label
    (``NOSPEECH_LABELS``) leaves out nothing and needs ``target``."""
    if tuple(labels) == tuple(NOSPEECH_LABELS):
        return () if first_position == "target" else None
    if len(labels) <= MASKED_PREFIX or labels[0] != JA or labels[1] != transcribe or labels[-1] != EOT:
        return None
    return tuple(range(MASKED_PREFIX)) if first_position == "masked" else (1,)


def loss_weights(length: int, masked: Sequence[int], first_weight: float) -> tuple[float, ...]:
    """Each label position's loss weight (round 5, the trainer's
    ``first_position_weight``): 0 where ``masked``, ``first_weight`` at the
    first predicted position (``<|ja|>`` or ``<|nospeech|>``) when it is
    trained, 1 elsewhere. The trainer divides the weighted sum of the
    token losses by the sum of the weights, so a weight of 1 is the token mean."""
    weights = [0.0 if position in masked else 1.0 for position in range(length)]
    if weights and weights[0]:
        weights[0] = float(first_weight)
    return tuple(weights)


def load_tokenizer(path: Path | None = None) -> Any:
    """``tokenizers.Tokenizer`` of the base export (``tokenizer.json`` of
    ``common.BASE_EXPORT_REPO`` at ``BASE_EXPORT_REVISION``, from the app's
    Hugging Face cache or downloaded there), or of ``path``. Raises
    ``ValueError`` when a special-token id differs from the constants here."""
    from tokenizers import Tokenizer

    if path is None:
        from yamabiko import hf_env  # noqa: F401  (pins HF_HOME before the hub import)
        from huggingface_hub import hf_hub_download

        from scripts.asr_finetune.common import BASE_EXPORT_REPO, BASE_EXPORT_REVISION

        path = Path(hf_hub_download(BASE_EXPORT_REPO, "tokenizer.json", revision=BASE_EXPORT_REVISION))
    tokenizer = Tokenizer.from_file(str(path))
    wrong = [name for name, token in _FIXED.items() if tokenizer.token_to_id(name) != token]
    if wrong or tokenizer.token_to_id("<|transcribe|>") is None:
        raise ValueError(f"{path}: token ids differ from scripts.asr_finetune.labels for {', '.join(wrong) or 'transcribe'}")
    return tokenizer
