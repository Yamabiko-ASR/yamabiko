"""VOICEVOX ENGINE for the synthetic ASR bench set: install, run, synthesize.

The Windows CPU build of VOICEVOX ENGINE is downloaded from its official
GitHub release and unpacked under ``%LOCALAPPDATA%\\Yamabiko\\tools``, never
into the repository. ``running_engine`` starts it on 127.0.0.1:50021 for one
build and stops it afterwards; an engine of the same version that is already
listening there is used as is and left running.

Synthesized speech is private bench data. Each VOICEVOX character's terms
require the credit ``VOICEVOX:<name>``; builders record it in the set's build
provenance.

Run ``python -m scripts.asr_bench.voicevox`` to install the engine and print
the voice plan resolved against its ``/speakers``.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .sets import SAMPLE_RATE_HZ

ENGINE_VERSION = "0.26.0"
RELEASE_URL = (
    "https://github.com/VOICEVOX/voicevox_engine/releases/download/"
    f"{ENGINE_VERSION}"
)
ARCHIVE_STEM = f"voicevox_engine-windows-cpu-{ENGINE_VERSION}"
SEVEN_ZIP_REDUCED_URL = "https://www.7-zip.org/a/7zr.exe"
HOST = "127.0.0.1"
PORT = 50021
STARTUP_TIMEOUT_S = 300.0

# Characters with clearly different voices. Each style is tagged with the
# register it adds to the bench; whisper and high-energy styles are the hard
# cases for VAD. Styles missing from the installed engine are skipped.
VOICE_PLAN: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("ずんだもん", (("ノーマル", "normal"), ("ささやき", "whisper"))),
    ("四国めたん", (("ノーマル", "normal"), ("ヒソヒソ", "whisper"), ("ツンツン", "other"))),
    ("春日部つむぎ", (("ノーマル", "normal"),)),
    ("冥鳴ひまり", (("ノーマル", "normal"),)),
    ("玄野武宏", (("ノーマル", "normal"), ("ツンギレ", "high_energy"))),
    ("青山龍星", (("ノーマル", "normal"), ("熱血", "high_energy"), ("囁き", "whisper"))),
    ("白上虎太郎", (("ふつう", "normal"), ("わーい", "high_energy"), ("おこ", "high_energy"))),
    ("ちび式じい", (("ノーマル", "normal"),)),
)
MIN_CHARACTERS = 6
MIN_WHISPER_STYLES = 2
MIN_HIGH_ENERGY_STYLES = 2


@dataclass(frozen=True)
class Voice:
    character: str
    style: str
    style_id: int
    register: str

    @property
    def label(self) -> str:
        return f"{self.character}/{self.style}"

    @property
    def credit(self) -> str:
        return f"VOICEVOX:{self.character}"


def tools_root() -> Path:
    return Path(os.environ["LOCALAPPDATA"]) / "Yamabiko" / "tools"


def install_dir() -> Path:
    return tools_root() / f"voicevox_engine-{ENGINE_VERSION}"


def _download(url: str, target: Path) -> Path:
    """Stream ``url`` to ``target`` through a ``.part`` file; a finished
    ``target`` is reused."""
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        expected = response.headers.get("Content-Length")
        written = 0
        with partial.open("wb") as handle:
            while chunk := response.read(1 << 20):
                handle.write(chunk)
                written += len(chunk)
    if expected is not None and written != int(expected):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"{url}: got {written} bytes, expected {expected}")
    os.replace(partial, target)
    return target


def _seven_zip() -> tuple[Path, bool]:
    """Return a 7-Zip executable and whether it is the full build (which
    opens split ``.7z.001`` volumes itself). Falls back to downloading the
    reduced ``7zr.exe`` from 7-zip.org."""
    found = shutil.which("7z")
    if found:
        return Path(found), True
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    installed = program_files / "7-Zip" / "7z.exe"
    if installed.is_file():
        return installed, True
    return _download(SEVEN_ZIP_REDUCED_URL, tools_root() / "7zr.exe"), False


def install_engine() -> Path:
    """Install the engine once and return its ``run.exe``."""
    target = install_dir()
    run_exe = target / "run.exe"
    if run_exe.is_file():
        return run_exe
    downloads = tools_root() / "downloads"
    listing = _download(
        f"{RELEASE_URL}/{ARCHIVE_STEM}.7z.txt", downloads / f"{ARCHIVE_STEM}.7z.txt"
    )
    part_names = listing.read_text(encoding="utf-8").split()
    if not part_names:
        raise RuntimeError(f"{listing.name} lists no archive parts")
    parts = [_download(f"{RELEASE_URL}/{name}", downloads / name) for name in part_names]
    seven_zip, full_build = _seven_zip()
    staging = tools_root() / f".staging-voicevox_engine-{ENGINE_VERSION}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    archive = parts[0]
    if not full_build:
        # 7zr reads plain .7z archives only: join the volumes first.
        archive = staging / f"{ARCHIVE_STEM}.7z"
        with archive.open("wb") as joined:
            for part in parts:
                with part.open("rb") as handle:
                    shutil.copyfileobj(handle, joined, 1 << 24)
    extracted = staging / "extracted"
    subprocess.run(
        [str(seven_zip), "x", "-y", f"-o{extracted}", str(archive)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    candidates = sorted(extracted.rglob("run.exe"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one run.exe in the archive, found {len(candidates)}")
    os.replace(candidates[0].parent, target)
    shutil.rmtree(staging)
    for part in parts:
        part.unlink()
    return run_exe


class EngineClient:
    """Minimal client for the VOICEVOX ENGINE HTTP API."""

    def __init__(self, host: str = HOST, port: int = PORT) -> None:
        self.base = f"http://{host}:{port}"

    def _request(
        self, method: str, path: str, query: Mapping[str, Any] | None = None,
        body: bytes | None = None, timeout: float = 120.0,
    ) -> bytes:
        url = self.base + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(url, data=body, method=method)
        if body is not None:
            request.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def version(self) -> str | None:
        """The engine version, or ``None`` when nothing answers."""
        try:
            return json.loads(self._request("GET", "/version", timeout=2.0))
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            return None

    def speakers(self) -> list[dict[str, Any]]:
        return json.loads(self._request("GET", "/speakers"))

    def synthesize(self, text: str, style_id: int) -> np.ndarray:
        """Render ``text`` as 16 kHz mono float32 samples. The engine
        resamples its output to 16 kHz (``outputSamplingRate``)."""
        query = json.loads(
            self._request("POST", "/audio_query", {"text": text, "speaker": style_id})
        )
        query["outputSamplingRate"] = SAMPLE_RATE_HZ
        query["outputStereo"] = False
        payload = self._request(
            "POST", "/synthesis", {"speaker": style_id},
            body=json.dumps(query, ensure_ascii=False).encode("utf-8"),
        )
        with wave.open(io.BytesIO(payload)) as wav:
            if (
                wav.getframerate() != SAMPLE_RATE_HZ
                or wav.getnchannels() != 1
                or wav.getsampwidth() != 2
            ):
                raise RuntimeError(
                    f"unexpected VOICEVOX output {wav.getframerate()} Hz, "
                    f"{wav.getnchannels()} ch, {wav.getsampwidth() * 8} bit"
                )
            frames = wav.readframes(wav.getnframes())
        return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        # The engine may start helper processes; end the whole tree.
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
    else:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


@contextlib.contextmanager
def running_engine() -> Iterator[EngineClient]:
    """Yield a client for a running engine of ``ENGINE_VERSION``.

    Starts the installed engine when nothing listens on the port and stops
    it on exit; refuses an engine of another version.
    """
    client = EngineClient()
    found = client.version()
    if found is not None:
        if found != ENGINE_VERSION:
            raise RuntimeError(
                f"VOICEVOX ENGINE {found} already listens on {HOST}:{PORT}; "
                f"the bench needs {ENGINE_VERSION}"
            )
        yield client
        return
    run_exe = install_engine()
    log_path = tools_root() / f"voicevox_engine-{ENGINE_VERSION}.log"
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [str(run_exe), "--host", HOST, "--port", str(PORT)],
            cwd=run_exe.parent, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, creationflags=flags,
        )
        try:
            deadline = time.monotonic() + STARTUP_TIMEOUT_S
            while (found := client.version()) is None:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"VOICEVOX ENGINE exited with {process.returncode}; see {log_path}"
                    )
                if time.monotonic() > deadline:
                    raise RuntimeError(f"VOICEVOX ENGINE did not start; see {log_path}")
                time.sleep(1.0)
            if found != ENGINE_VERSION:
                raise RuntimeError(f"started VOICEVOX ENGINE reports version {found}")
            yield client
        finally:
            _stop(process)


def resolve_voices(
    speakers: Sequence[Mapping[str, Any]],
    plan: Sequence[tuple[str, Sequence[tuple[str, str]]]] = VOICE_PLAN,
) -> list[Voice]:
    """Map the voice plan onto the engine's ``/speakers`` talk styles."""
    styles: dict[tuple[str, str], int] = {}
    for speaker in speakers:
        for style in speaker.get("styles", ()):
            # Speech styles are "talk" (0.26 reports "streaming_talk");
            # singing and humming styles are not dialogue.
            if style.get("type", "talk") in ("talk", "streaming_talk"):
                styles[(speaker["name"], style["name"])] = int(style["id"])
    voices = [
        Voice(character, style, styles[(character, style)], register)
        for character, wanted in plan
        for style, register in wanted
        if (character, style) in styles
    ]
    characters = {voice.character for voice in voices}
    whispers = sum(voice.register == "whisper" for voice in voices)
    energetic = sum(voice.register == "high_energy" for voice in voices)
    if (
        len(characters) < MIN_CHARACTERS
        or whispers < MIN_WHISPER_STYLES
        or energetic < MIN_HIGH_ENERGY_STYLES
    ):
        raise RuntimeError(
            f"voice plan resolved to {len(characters)} characters, {whispers} "
            f"whisper and {energetic} high-energy styles; the engine lacks "
            "planned voices"
        )
    return voices


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    print(f"engine: {install_engine()}")
    with running_engine() as client:
        for voice in resolve_voices(client.speakers()):
            print(f"{voice.style_id:4d} {voice.label} ({voice.register}) {voice.credit}")


if __name__ == "__main__":
    main()
