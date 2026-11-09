"""Write a trained decoder into the 760M's float16 weights and convert them
with the base export's settings, the encoder byte for byte unchanged (plan D1;
pod task T1).

    python -m scripts.asr_finetune.export --source <transformers dir> [--decoder <decoder .pt> [<fallback .pt> ...]]
        [--reference <base export dir>] --out <dir> [--hf <dir>] [--copy-decoder] [--exit-zero]

``--source`` is the transformers snapshot of ``SOURCE_REPO`` at
``SOURCE_REVISION`` (fetched there first when it holds no weights); it loads
in float16. ``--decoder`` names a checkpoint of ``scripts.asr_finetune.train``
(``decoder-epoch<e>.pt``, ``decoder-final.pt`` or ``decoder-last.pt``, each
the decoder ``state_dict`` in float32), optionally followed by fallbacks:
the first that exists is exported (on a pod: ``decoder-final.pt``, then
``decoder-last.pt`` after a deadline stop, then ``decoder-epoch1.pt`` after
a fault that ended training). Its tensors are rounded once to
float16 and replace the decoder's, names and shapes exactly matching.
Without ``--decoder`` the model stays unchanged. ``save_pretrained``
writes ``--hf`` (default ``<out>\\hf``; on a pod a folder outside the
fetched ``{out}``), and every other file of the source (config, generation
config, tokenizer and preprocessor files) is copied over it, so the
converter reads the source's metadata; ``common.convert_to_ct2`` converts
it into ``<out>\\ct2`` with the base export's settings. ``--copy-decoder`` copies
the chosen checkpoint into ``--out`` before any of that.

``--reference`` is the base export (``common.BASE_EXPORT_REPO``); its
``model.bin`` must be the pinned one (``BASE_EXPORT_MODEL_BIN_SHA256``).
Without it, the base export at ``BASE_EXPORT_REVISION`` is fetched by repo id
into the app's model cache (as on a pod) and checked the same way.

``export.json`` in ``--out``: the source weights' and checkpoint's SHA-256,
which ``--decoder`` path was used (``decoder.file``, ``decoder.candidate``
counted from 0) and whether it was copied;
``lora.compare_safetensors(<source>\\model.safetensors,
<hf>\\model.safetensors)``; ``common.compare_model_dirs(<reference>,
<out>\\ct2)`` with ``common.compare_ct2`` of the two ``model.bin``; the
encoder and decoder variable counts (identical, changed); the SHA-256 of
the new ``model.bin`` and whether it equals the reference's; the checks:

- ``weights_same_tensors``: no tensor only in one safetensors file;
- ``weights_outside_decoder_identical``: every differing tensor is named
  ``model.decoder.*``;
- ``ct2_same_layout``: same header, variables in the same order, same
  aliases;
- ``ct2_outside_decoder_identical``: every variable outside ``decoder/``
  (all of ``encoder/``) byte-identical;
- ``ct2_files_equal``: every file of ``<out>\\ct2`` but ``model.bin``
  byte-identical to the reference's (``common.ct2_files_equal``;
  ``config.json`` may also be the same JSON). Before the comparison, each
  converted ``.json`` file whose content equals the reference file's but
  whose bytes do not (some converter builds re-serialize ``config.json``
  and ``vocabulary.json``) gets the reference file's bytes
  (``common.adopt_reference_json``); its ``ct2.files`` entry records
  ``replaced_with_reference`` and the converter's SHA-256 and size. A JSON
  file whose content differs stays as converted and fails the check;
- without ``--decoder`` also ``unchanged_model_bin_identical``: the new
  ``model.bin`` is byte-identical to the reference's.

Exit 0 when every check holds; 1 when one fails (an encoder tensor or
variable differs, among others) or the export raises (a Hub download, the
converter, a reference that is not the pinned one); 2 on a missing or
invalid input, no ``--decoder`` path existing among them. Every failure
that stops the export early still writes ``export.json`` with its
``error``, ``ok`` false and the chosen ``decoder`` (file, candidate,
copied). ``--copy-decoder`` copies the chosen checkpoint first, before
anything that can fail, so a paid checkpoint always reaches ``--out`` for
a local export. ``--exit-zero`` (a pod task whose failure must not stop
the tasks after it) exits 0 in every case: ``export.json`` ``ok`` is then
the only verdict. ``export.json`` is written atomically.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch

from scripts.asr_finetune.common import (
    SOURCE_WEIGHTS,
    adopt_reference_json,
    base_export_dir,
    compare_model_dirs,
    convert_to_ct2,
    ct2_files_equal,
    fetch_source,
    not_base_export,
    package_versions,
    read_ct2_model,
    sha256_file,
    weight_summary,
)
from scripts.asr_finetune.lora import compare_safetensors, load_base

SCHEMA = "ft-pilot-export-1"
_HF_DECODER = "model.decoder."
_CT2_DECODER = "decoder/"
_CT2_ENCODER = "encoder/"


class InputError(ValueError):
    """A missing or invalid input (exit 2)."""


def load_decoder(path: Path, expected: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The checkpoint's tensors rounded once to float16, checked against
    the decoder's names and shapes."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # torch.load raises several unrelated types on a file that is no checkpoint
        raise InputError(f"{path}: not a torch checkpoint ({type(error).__name__})") from None
    if not isinstance(state, dict) or set(state) != set(expected):
        raise InputError(f"{path}: not a decoder state_dict of this model")
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 or tensor.shape != expected[name].shape:
            raise InputError(f"{path}: {name} is not a float32 tensor of shape {tuple(expected[name].shape)}")
    return {name: tensor.to(torch.float16) for name, tensor in state.items()}


def write_hf(source: Path, decoder: Path | None, out: Path) -> dict[str, Any] | None:
    """``<out>`` as a float16 transformers checkpoint of the source with the
    checkpoint's decoder; the checkpoint's record, or None without one."""
    model = load_base(source, torch.float16)
    record = None
    if decoder is not None:
        state = load_decoder(decoder, model.model.decoder.state_dict())
        model.model.decoder.load_state_dict(state, strict=True)
        record = {"file": decoder.name, "sha256": sha256_file(decoder), "tensors": len(state)}
    if out.exists():
        shutil.rmtree(out)
    model.save_pretrained(out, safe_serialization=True)
    for path in source.iterdir():
        if path.is_file() and path.name != SOURCE_WEIGHTS and not path.name.startswith("."):
            shutil.copy2(path, out / path.name)
    return record


def checks(weights: dict[str, Any], files: dict[str, Any], unchanged: bool) -> dict[str, bool]:
    model_bin = files["model_bin"]
    result = {
        "weights_same_tensors": not weights["only_in_reference"] and not weights["only_in_other"],
        "weights_outside_decoder_identical": all(name.startswith(_HF_DECODER) for name in weights["differing_tensors"]),
        "ct2_same_layout": (model_bin["header_identical"] and model_bin["order_identical"]
                            and model_bin["aliases_identical"] and not model_bin["only_in_reference"]
                            and not model_bin["only_in_other"]),
        "ct2_outside_decoder_identical": all(name.startswith(_CT2_DECODER)
                                             for name in model_bin["differing_variables"]),
        "ct2_files_equal": ct2_files_equal(files),
    }
    if unchanged:
        result["unchanged_model_bin_identical"] = files["files"]["model.bin"]["identical"]
    return result


def variable_counts(reference_bin: Path, differing: dict[str, Any], prefix: str) -> dict[str, int]:
    names = [name for name in read_ct2_model(reference_bin).variables if name.startswith(prefix)]
    changed = sum(1 for name in names if name in differing)
    return {"total": len(names), "identical": len(names) - changed, "changed": changed}


def run(args: argparse.Namespace, chosen_record: dict[str, Any]) -> int:
    """The export; ``chosen_record`` receives the chosen checkpoint (file,
    candidate, copied) as soon as it is known, for the failure record."""
    out_record = args.out / "export.json"
    args.out.mkdir(parents=True, exist_ok=True)
    out_record.unlink(missing_ok=True)
    try:
        candidates = args.decoder or []
        chosen = next((path for path in candidates if path.is_file()), None)
        if candidates and chosen is None:
            raise InputError(f"no checkpoint among {', '.join(str(path) for path in candidates)}")
        if chosen is not None:
            chosen_record.update(file=chosen.name, candidate=candidates.index(chosen), copied=False)
            print(f"decoder: {chosen.name} (candidate {chosen_record['candidate'] + 1} of {len(candidates)})",
                  flush=True)
            if args.copy_decoder:  # first: a failure below must not keep the paid checkpoint on the pod
                shutil.copy2(chosen, args.out / chosen.name)
                chosen_record["copied"] = True
        if args.reference is None:
            reference = base_export_dir(download=True)
        elif not (args.reference / "model.bin").is_file():
            raise InputError(f"{args.reference}: no model.bin")
        elif problem := not_base_export(args.reference):
            raise InputError(problem)
        else:
            reference = args.reference
        source = args.source if (args.source / SOURCE_WEIGHTS).is_file() else fetch_source(args.source)
        hf, ct2 = args.hf or args.out / "hf", args.out / "ct2"
        decoder = write_hf(source, chosen, hf)
    except InputError as error:
        write_json(out_record, {"schema": SCHEMA, "error": str(error), "decoder": chosen_record or None, "ok": False})
        print(f"error: {error}", flush=True)
        return 2

    command = convert_to_ct2(hf, ct2)
    replaced = adopt_reference_json(reference, ct2)
    weights = compare_safetensors(source / SOURCE_WEIGHTS, hf / SOURCE_WEIGHTS)
    files = compare_model_dirs(reference, ct2)
    for name, converted in replaced.items():
        files["files"][name].update(replaced_with_reference=True, **converted)
    differing = files["model_bin"]["differing_variables"]
    model_bin_sha256 = files["files"]["model.bin"]["sha256"]
    result = checks(weights, files, unchanged=chosen is None)
    if decoder is not None:
        decoder.update(chosen_record)
    record = {
        "schema": SCHEMA,
        "source": {"weights_sha256": sha256_file(source / SOURCE_WEIGHTS)},
        "decoder": decoder,
        "reference_model_bin_sha256": files["files"]["model.bin"]["reference_sha256"],
        "converter_command": command,
        "weights": weights,
        "ct2": files,
        "encoder_variables": variable_counts(reference / "model.bin", differing, _CT2_ENCODER),
        "decoder_variables": variable_counts(reference / "model.bin", differing, _CT2_DECODER),
        "model_bin_sha256": model_bin_sha256,
        "identical_to_reference": files["files"]["model.bin"]["identical"],
        "checks": result,
        "ok": all(result.values()),
        "packages": package_versions(),
    }
    write_json(out_record, record)

    encoder, decoder_vars = record["encoder_variables"], record["decoder_variables"]
    if replaced:
        print(f"converter JSON re-serialized, content equal: {', '.join(replaced)} given the reference bytes")
    print(f"weights vs source: {weight_summary(weights)}")
    print(f"model.bin vs reference: {weight_summary(files['model_bin'])}")
    print(f"encoder variables byte-identical: {encoder['identical']} of {encoder['total']}; "
          f"decoder variables changed: {decoder_vars['changed']} of {decoder_vars['total']}")
    print(f"model.bin SHA-256 {model_bin_sha256}; identical to the reference file: {record['identical_to_reference']}")
    failed = [name for name, passed in result.items() if not passed]
    print(f"checks: {'all passed' if not failed else 'FAILED ' + ', '.join(failed)}; record: {out_record}")
    return 0 if not failed else 1


def write_json(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, required=True, help="transformers snapshot of the 760M")
    parser.add_argument("--decoder", type=Path, nargs="+",
                        help="decoder checkpoint of scripts.asr_finetune.train, then fallbacks (the first that exists)")
    parser.add_argument("--reference", type=Path, help="base export dir (default: fetched by repo id)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hf", type=Path, help="the float16 transformers checkpoint (default: <out>/hf)")
    parser.add_argument("--copy-decoder", action="store_true",
                        help="copy the chosen checkpoint into --out, before anything that can fail")
    parser.add_argument("--exit-zero", action="store_true", help="exit 0 in every case; export.json holds the verdict")
    args = parser.parse_args(argv)
    chosen: dict[str, Any] = {}
    try:
        code = run(args, chosen)
    except (Exception, SystemExit) as error:
        with contextlib.suppress(OSError):
            write_json(args.out / "export.json", {"schema": SCHEMA, "error": f"{type(error).__name__}: {error}",
                                                  "decoder": chosen or None, "ok": False})
        print(f"error: {type(error).__name__}: {error}", flush=True)
        if not args.exit_zero:
            raise
        code = 1
    if code and args.exit_zero:
        print(f"exit {code} recorded in export.json; exiting 0 (--exit-zero)", flush=True)
        return 0
    return code


if __name__ == "__main__":
    raise SystemExit(main())
