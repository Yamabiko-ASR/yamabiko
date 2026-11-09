"""Export one EfficientSED frame tagger (a frame-wise MobileNet with its
sequence model) to ONNX, mel frontend included, so ``tagger_onnx.py`` can
run it with ONNX Runtime and NumPy alone.

Usage (inside the tagger venv, which needs ``onnx`` besides the worker's
packages; ``tag_sounds_worker.py`` must sit next to this file)::

    <tagger venv python> scripts/asr_bench/export_tagger_onnx.py --repo <EfficientSED>
        --checkpoint <.pt> --model fmn10 --seq-model-type tf --seq-model-dim 256
        --out <.onnx> [--opset 17]

The model is loaded exactly as the worker loads it (``PredictionsWrapper``
and its key remapping). The graph maps ``audio`` (batch x 160000 float32,
one 10 s window at 16 kHz) to ``probs`` (batch x 250 x 447 float32, the
sigmoid of the strong head; frame ``j`` covers window samples
``[640 j, 640 (j + 1))``, classes in ``as_strong_train_classes`` order).
The frontend is the repository's ``AugmentMelSTFT`` in eval mode written
with exportable operators: pre-emphasis, a 512-point STFT (ONNX ``STFT``,
opset 17) with the 400-sample symmetric Hann window centred in 512 and
reflect padding, power, the 128-band Kaldi mel matrix (0 to 7000 Hz, a
constant), ``log(max(x, 1e-7))`` and ``(x + 4.5) / 5``. Before writing, the
script checks the rewritten frontend and the whole wrapper against the
repository's own ``mel_forward`` and forward pass in PyTorch.

The file's ``yamabiko_tagger`` metadata entry is a JSON object with the
class list, the input and output layout, the source checkpoint (name, size,
SHA-256), the EfficientSED commit and the export versions. A report with
the output's SHA-256 goes to stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

SCHEMA = "yamabiko-sound-tagger-onnx-1"
METADATA_KEY = "yamabiko_tagger"
SAMPLE_RATE = 16000
OPSET = 17


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                              check=True).stdout.strip()

    try:
        return {"head": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "dirty": None}


def _tagger_type(torch: Any) -> type:
    """The graph wrapper class, built here so the module imports nothing
    but the standard library at load time."""
    import torch.nn.functional as functional
    from torchaudio.compliance import kaldi

    class ExportMel(torch.nn.Module):
        """``AugmentMelSTFT`` in eval mode with exportable operators."""

        def __init__(self, mel: Any) -> None:
            super().__init__()
            if (len(mel.n_fft) != 1 or not mel.preamp or mel.padding != "center"
                    or not mel.fast_norm or mel.training):
                raise ValueError("unsupported AugmentMelSTFT settings")
            n_fft, win_length = mel.n_fft[0], mel.win_length[0]
            window = getattr(mel, f"window_{win_length}")
            left = (n_fft - win_length) // 2
            self.register_buffer("window", functional.pad(window, (left, n_fft - win_length - left)))
            self.register_buffer("preemphasis", mel.preemphasis_coefficient.clone())
            basis, _ = kaldi.get_mel_banks(mel.n_mels, n_fft, mel.sr, mel.fmin, mel.fmax,
                                           vtln_low=100.0, vtln_high=-500.0,
                                           vtln_warp_factor=1.0)
            self.register_buffer("basis", functional.pad(basis, (0, 1), mode="constant", value=0))
            self.n_fft, self.hop = n_fft, mel.hopsize

        def forward(self, audio: Any) -> Any:
            x = functional.conv1d(audio.unsqueeze(1), self.preemphasis).squeeze(1)
            spec = torch.stft(x, self.n_fft, hop_length=self.hop, win_length=self.n_fft,
                              window=self.window, center=True, pad_mode="reflect",
                              normalized=False, onesided=True, return_complex=False)
            power = spec.pow(2).sum(-1)
            mel = torch.log(torch.clamp(torch.matmul(self.basis, power), min=1e-7))
            return ((mel + 4.5) / 5.0).unsqueeze(1)

    class Tagger(torch.nn.Module):
        """Audio windows to strong-head probabilities, frames before classes."""

        def __init__(self, model: Any) -> None:
            super().__init__()
            self.mel = ExportMel(model.model.mel)
            self.model = model

        def forward(self, audio: Any) -> Any:
            logits, _ = self.model(self.mel(audio))
            return torch.sigmoid(logits).transpose(1, 2)

    return Tagger


def export(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import onnx
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tag_sounds_worker import WINDOW_FRAMES, WINDOW_SAMPLES, load_model

    torch.set_num_threads(args.threads)
    job = {"repo": str(args.repo), "checkpoint": str(args.checkpoint), "model": args.model,
           "seq_model_type": args.seq_model_type, "seq_model_dim": args.seq_model_dim}
    model, classes = load_model(job, torch)
    tagger = _tagger_type(torch)(model).eval()

    # Check the rewrite against the repository's own path: silence, noise
    # and a loud tone-plus-noise window.
    rng = np.random.default_rng(0)
    time_axis = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE
    probe = np.stack([
        np.zeros(WINDOW_SAMPLES),
        0.05 * rng.standard_normal(WINDOW_SAMPLES),
        0.5 * np.sin(2 * np.pi * 440 * time_axis) + 0.01 * rng.standard_normal(WINDOW_SAMPLES),
    ]).astype(np.float32)
    with torch.inference_mode():
        audio = torch.from_numpy(probe)
        mel_reference = model.mel_forward(audio)
        mel_export = tagger.mel(audio)
        logits, _ = model(mel_reference)
        reference = torch.sigmoid(logits).transpose(1, 2)
        exported = tagger(audio)
    mel_diff = float((mel_reference - mel_export).abs().max())
    prob_diff = float((reference - exported).abs().max())
    if exported.shape != (len(probe), WINDOW_FRAMES, len(classes)):
        raise RuntimeError(f"unexpected output shape {tuple(exported.shape)}")
    if mel_diff > 1e-4 or prob_diff > 1e-5:
        raise RuntimeError(f"rewritten frontend differs: mel {mel_diff}, probs {prob_diff}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.stem + ".part.onnx")
    torch.onnx.export(
        tagger, (torch.from_numpy(probe[:1]),), str(temporary), dynamo=False,
        opset_version=args.opset, input_names=["audio"], output_names=["probs"],
        dynamic_axes={"audio": {0: "batch"}, "probs": {0: "batch"}},
        do_constant_folding=True,
    )
    graph = onnx.load(str(temporary))
    # The STFT hides the frame count from shape inference; state it.
    frames_dim = graph.graph.output[0].type.tensor_type.shape.dim[1]
    frames_dim.Clear()
    frames_dim.dim_value = WINDOW_FRAMES
    checkpoint = Path(args.checkpoint)
    metadata = {
        "schema": SCHEMA,
        "input": {"name": "audio", "shape": ["batch", WINDOW_SAMPLES], "dtype": "float32",
                  "sample_rate": SAMPLE_RATE},
        "output": {"name": "probs", "shape": ["batch", WINDOW_FRAMES, len(classes)],
                   "dtype": "float32",
                   "meaning": "sigmoid of the strong head; frame j covers window samples "
                              f"[{WINDOW_SAMPLES // WINDOW_FRAMES} j, "
                              f"{WINDOW_SAMPLES // WINDOW_FRAMES} (j + 1))"},
        "frontend": "in graph: pre-emphasis 0.97, STFT 512 points, hop 160, 400-sample "
                    "symmetric Hann centred, reflect padding, power, Kaldi mel 128 bands "
                    "0 to 7000 Hz, log(max(x, 1e-7)), (x + 4.5) / 5",
        "classes": classes,
        "source": {
            "checkpoint": checkpoint.name, "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": _sha256(checkpoint), "efficientsed": _git_head(Path(args.repo)),
            "model": args.model, "seq_model_type": args.seq_model_type,
            "seq_model_dim": args.seq_model_dim,
        },
        "export": {"exporter": "torch.onnx.export (TorchScript)", "opset": args.opset,
                   "torch": torch.__version__, "torchaudio": version("torchaudio"),
                   "onnx": onnx.__version__},
    }
    entry = graph.metadata_props.add()
    entry.key = METADATA_KEY
    entry.value = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(temporary))
    temporary.replace(out)
    return {
        "onnx": str(out), "bytes": out.stat().st_size, "sha256": _sha256(out),
        "torch_check": {"mel_max_abs_diff": mel_diff, "probs_max_abs_diff": prob_diff},
        "source": metadata["source"], "export": metadata["export"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", required=True, type=Path, help="EfficientSED checkout")
    parser.add_argument("--checkpoint", required=True, type=Path, help="tagger .pt file")
    parser.add_argument("--model", required=True, help="fmn04 to fmn30")
    parser.add_argument("--seq-model-type", default=None, help="tf, gru, ... or omit")
    parser.add_argument("--seq-model-dim", type=int, default=None)
    parser.add_argument("--out", required=True, type=Path, help="output .onnx file")
    parser.add_argument("--opset", type=int, default=OPSET)
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    args = parser.parse_args(argv)
    if not args.model.startswith("fmn"):
        parser.error("only the fmn models export here")
    # The EfficientSED code prints the model and progress; keep stdout for
    # the report.
    with contextlib.redirect_stdout(sys.stderr):
        report = export(args)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
