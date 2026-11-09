"""Decode keyword overrides: the ``fw_kwargs`` window strategy.

Each window is decoded by production's own ``FasterWhisperASR.transcribe``
(boundary pad, style prompt, beam, the per-segment confidence backstop, the
assembled-text compression gate and the outcome), run on a shallow copy of
the runner's backend whose ``WhisperModel`` stands in a recording wrapper.
The wrapper merges the candidate's faster-whisper ``transcribe`` keywords
over the ones production passes and records every call and segment. With no
keyword the decode is production's, call for call::

    "strategy": {"mode": "fw_kwargs", "without_timestamps": true}

Keywords (all optional, unknown keys rejected):

- ``without_timestamps`` (bool), ``suppress_blank`` (bool);
- ``max_initial_timestamp`` (seconds, 0 to 30);
- ``suppress_tokens``: token ids, passed as ``[-1, *ids]`` so faster-whisper
  still suppresses its default non-speech set; each id must be a text
  token of the model's vocabulary (below its end-of-text token);
- ``temperature``: a number or a non-empty list of numbers in [0, 1], the
  fallback schedule (production uses faster-whisper's default
  ``[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]``);
- ``no_speech_threshold`` (0 to 1), ``log_prob_threshold`` (at most 0),
  ``compression_ratio_threshold`` (positive). Production reads these from
  constants of ``yamabiko.asr.asr_client`` for its ``transcribe`` call and
  again in its own gates, so an override replaces the constant for the
  duration of each call, as a change of the constant would.

Each window record adds ``fw_segments``: every segment faster-whisper
yielded (start and end in seconds of the padded decode audio, text, token
ids, temperature, average log-probability, no-speech probability and
compression ratio), dropped or kept. The run's ``strategy`` section holds the
keywords production passed, the complete effective keywords (faster-whisper
defaults filled in), the suppressed token ids faster-whisper hands
CTranslate2, the gate constants in effect, and the window, segment,
temperature and outcome counts.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import math
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from scripts.asr_bench.sets import SAMPLE_RATE_HZ

if TYPE_CHECKING:
    from scripts.asr_bench.run import StrategyContext, Window

MODE = "fw_kwargs"
MAX_INITIAL_TIMESTAMP_S = 30.0
# Overridable thresholds -> the asr_client constant production reads for its
# transcribe call and its own gates.
GATE_CONSTANTS = {
    "no_speech_threshold": "_NO_SPEECH_THRESHOLD",
    "log_prob_threshold": "_LOGPROB_THRESHOLD",
    "compression_ratio_threshold": "_COMPRESSION_RATIO_THRESHOLD",
}
KEYWORDS = (
    "without_timestamps", "max_initial_timestamp", "suppress_tokens", "suppress_blank",
    "no_speech_threshold", "log_prob_threshold", "compression_ratio_threshold", "temperature",
)
# The keys of FwKwargsStrategy.record that add up over items; the runner
# sums them over its worker processes (run._ADDITIVE_STRATEGY_KEYS).
ADDITIVE_RECORD_KEYS = frozenset({"windows", "segments", "segment_temperatures", "outcomes"})


@dataclass(frozen=True)
class FwKwargsSpec:
    """The ``strategy`` of a ``fw_kwargs`` candidate: validated keywords."""

    keywords: Mapping[str, Any] = field(default_factory=dict)
    mode: str = MODE

    def to_json(self) -> dict[str, Any]:
        return {"mode": self.mode, **copy.deepcopy(dict(self.keywords))}


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _bounded(value: object, label: str, low: float, high: float) -> float:
    number = _finite(value, label)
    if not low <= number <= high:
        raise ValueError(f"{label} must lie in [{low}, {high}]")
    return number


def parse_strategy(raw: Mapping[str, Any], where: str) -> FwKwargsSpec:
    """Parse a candidate's ``{"mode": "fw_kwargs", ...}`` strategy."""
    unknown = set(raw) - {"mode", *KEYWORDS}
    if unknown:
        raise ValueError(f"{where} has unknown keys {sorted(unknown)} (known: {list(KEYWORDS)})")
    keywords: dict[str, Any] = {}
    for key in KEYWORDS:
        if key not in raw:
            continue
        value = raw[key]
        label = f"{where}.{key}"
        if key in ("without_timestamps", "suppress_blank"):
            if not isinstance(value, bool):
                raise ValueError(f"{label} must be a bool")
        elif key == "max_initial_timestamp":
            value = _bounded(value, label, 0.0, MAX_INITIAL_TIMESTAMP_S)
        elif key == "suppress_tokens":
            if (not isinstance(value, list) or not value
                    or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in value)):
                raise ValueError(f"{label} must be a non-empty list of token ids (ints >= 0)")
            if len(set(value)) != len(value):
                raise ValueError(f"{label} lists a token twice")
            value = list(value)
        elif key == "temperature":
            schedule = value if isinstance(value, list) else [value]
            if not schedule:
                raise ValueError(f"{label} must be a number or a non-empty list")
            value = [_bounded(t, label, 0.0, 1.0) for t in schedule]
        elif key == "no_speech_threshold":
            value = _bounded(value, label, 0.0, 1.0)
        elif key == "log_prob_threshold":
            value = _finite(value, label)
            if value > 0.0:
                raise ValueError(f"{label} must be at most 0")
        elif key == "compression_ratio_threshold":
            value = _finite(value, label)
            if value <= 0.0:
                raise ValueError(f"{label} must be positive")
        keywords[key] = value
    return FwKwargsSpec(keywords=keywords)


class _RecordingModel:
    """Stands in for the backend's ``WhisperModel``: merges ``overrides``
    over the keywords production passes and records each call."""

    def __init__(self, model: Any, overrides: Mapping[str, Any]) -> None:
        self._model = model
        self._overrides = dict(overrides)
        self.production_kwargs: dict[str, Any] = {}
        self.kwargs: dict[str, Any] = {}
        self.segments: list[Any] = []

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Iterable[Any], Any]:
        self.production_kwargs = dict(kwargs)
        self.kwargs = {**kwargs, **self._overrides}
        self.segments = []
        segments, info = self._model.transcribe(audio, **self.kwargs)
        return self._record(segments), info

    def _record(self, segments: Iterable[Any]) -> Iterator[Any]:
        for segment in segments:
            self.segments.append(segment)
            yield segment


@contextlib.contextmanager
def _gate_constants(values: Mapping[str, float]) -> Iterator[None]:
    """Replace asr_client threshold constants for the duration of one call."""
    from yamabiko.asr import asr_client

    saved = {name: getattr(asr_client, name) for name in values}
    try:
        for name, value in values.items():
            setattr(asr_client, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(asr_client, name, value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


class FwKwargsStrategy:
    """The runner's ``fw_kwargs`` window strategy."""

    def __init__(self, context: StrategyContext) -> None:
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import get_suppressed_tokens

        from yamabiko.asr import asr_client

        spec = context.candidate.strategy
        if not isinstance(spec, FwKwargsSpec):
            raise TypeError("fw_kwargs needs a FwKwargsSpec strategy")
        backend = context.backend
        if backend is None:
            raise ValueError("fw_kwargs drives the faster_whisper backend only")
        self._spec = spec
        model = backend._model
        tokenizer = Tokenizer(
            model.hf_tokenizer, model.model.is_multilingual, task="transcribe", language="ja",
        )
        overrides = {key: value for key, value in spec.keywords.items() if key not in GATE_CONSTANTS}
        if "suppress_tokens" in overrides:
            bad = [token for token in overrides["suppress_tokens"] if token >= tokenizer.eot]
            if bad:
                raise ValueError(
                    f"fw_kwargs.suppress_tokens {bad} are not text tokens (end of text is {tokenizer.eot})"
                )
            overrides["suppress_tokens"] = [-1, *overrides["suppress_tokens"]]
        self._gates = {
            GATE_CONSTANTS[key]: value for key, value in spec.keywords.items() if key in GATE_CONSTANTS
        }
        self._recorder = _RecordingModel(model, overrides)
        # Production's own transcribe, on a copy whose model is the recorder.
        self._asr = copy.copy(backend)
        self._asr._model = self._recorder
        self._transcribe_signature = inspect.signature(model.transcribe)
        self._tokenizer = tokenizer
        self._get_suppressed_tokens = get_suppressed_tokens
        self._gate_values = {
            name: self._gates.get(name, getattr(asr_client, name)) for name in GATE_CONSTANTS.values()
        }
        self._gate_values["_MIN_COMPRESSION_GATE_LEN"] = asr_client._MIN_COMPRESSION_GATE_LEN
        self._boundary_pad_ms = backend._boundary_pad_ms
        self._production_kwargs: dict[str, Any] | None = None
        self._kwargs: dict[str, Any] | None = None
        self._audio = np.zeros(0, np.float32)
        self._windows = 0
        self._segments = 0
        self._temperatures: Counter = Counter()
        self._outcomes: Counter = Counter()

    def start_item(self, item: Any, asr_audio: np.ndarray, probabilities: np.ndarray) -> None:
        self._audio = asr_audio

    def decode_window(self, window: Window) -> dict[str, Any]:
        with _gate_constants(self._gates):
            result = self._asr.transcribe(
                self._audio[window.start_sample:window.end_sample], SAMPLE_RATE_HZ,
            )
        recorder = self._recorder
        if self._kwargs is None:
            self._production_kwargs = recorder.production_kwargs
            self._kwargs = recorder.kwargs
        elif recorder.kwargs != self._kwargs or recorder.production_kwargs != self._production_kwargs:
            raise RuntimeError("production passed different transcribe keywords between windows")
        segments = [
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": list(segment.tokens),
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
                "compression_ratio": segment.compression_ratio,
            }
            for segment in recorder.segments
        ]
        self._windows += 1
        self._segments += len(segments)
        self._temperatures.update(f"{segment['temperature']:g}" for segment in segments)
        self._outcomes[result.outcome.value] += 1
        return {"text": result.text, "asr_outcome": result.outcome.value, "fw_segments": segments}

    def _effective(self) -> dict[str, Any] | None:
        """Every keyword of the call, faster-whisper defaults filled in."""
        if self._kwargs is None:
            return None
        bound = self._transcribe_signature.bind(None, **self._kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        arguments.pop("audio")
        return _jsonable(arguments)

    def record(self) -> dict[str, Any]:
        effective = self._effective()
        suppressed = None
        if effective is not None and effective["suppress_tokens"]:
            suppressed = list(self._get_suppressed_tokens(
                self._tokenizer, list(effective["suppress_tokens"]),
            ))
        return {
            "mode": MODE,
            "spec": self._spec.to_json(),
            "decode": (
                "yamabiko.asr.asr_client.FasterWhisperASR.transcribe on a shallow copy of the"
                " runner's backend whose WhisperModel merges the keywords over production's"
            ),
            "boundary_pad_ms": self._boundary_pad_ms,
            "production_kwargs": _jsonable(self._production_kwargs),
            "effective_kwargs": effective,
            "suppress_tokens_resolved": suppressed,
            "gate_constants": self._gate_values,
            "windows": self._windows,
            "segments": self._segments,
            "segment_temperatures": dict(self._temperatures),
            "outcomes": dict(self._outcomes),
        }


def make_strategy(context: StrategyContext) -> FwKwargsStrategy:
    return FwKwargsStrategy(context)
