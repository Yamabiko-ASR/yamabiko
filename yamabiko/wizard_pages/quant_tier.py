"""Wizard step "Quality": which translation model to download.

Sits right before Download: it is a "what do we download" decision, and a
user who goes back from Download lands here. GPU detection runs on a
worker QThread (see _GpuDetectThread); the page seeds the conservative
Standard immediately, the detected card marks the fitting tier "Recommended
for your card" (a card under the Smaller tier's floor fits neither: no mark,
and the requirement line shows as for no card), and the suggestion moves the
selection only while the user has neither saved a tier nor picked one. The
chosen tier is persisted on "Start download" (validatePage), so the Download
step reads it.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWizard, QWizardPage

from yamabiko import gpu as _gpu
from yamabiko.overlay_app.controls import ui_font
from yamabiko.overlay_app.segmented import SegmentOption, SegmentedControl
from yamabiko.overlay_app.settings_dialogs import SaveFailedDialog
from yamabiko.overlay_app.theme import MIST
from yamabiko.ui import thread_keeper
from yamabiko.wizard_pages.rail import BRAND

_log = logging.getLogger(__name__)

CHECKING = "Checking\u2026"
NONE_FOUND = "None found"
NO_GPU_HELP = ("Yamabiko needs an NVIDIA graphics card with "
               f"{_gpu.SMALL_TIER_MIN_GB} GB of video memory or more.")
RECOMMENDED = "Recommended for your card"


def _saved_quant_tier() -> Optional[str]:
    """The ``translation.quant_tier`` saved in ``user_overrides.yaml``, or
    None when the user never saved one.

    Reads the overrides file ONLY, never the merged config: the merged
    value is ``default`` both for a deliberate Standard pick and for "never
    chosen", and only the second may be replaced by the GPU suggestion.
    ``config_writer.write_quant_tier`` always writes the tier key (it is
    exempt from sparse pruning) so a saved ``default`` stays visible here.
    The upgrade path forces completed installs back through this wizard,
    and the async GPU suggestion must not move a saved pick on
    click-through (wrong multi-GB download, runtime VRAM pressure on the
    card the user overrode for). Fails open to None on any read failure so
    a broken overrides file degrades to today's detection behavior.
    """
    try:
        from yamabiko import paths
        from yamabiko.config import _read_yaml
        overrides = _read_yaml(paths.user_overrides_path())
        tier = (overrides.get("translation") or {}).get("quant_tier")
    except Exception:
        _log.warning(
            "user_overrides.yaml read for translation.quant_tier failed; "
            "deferring to GPU detection", exc_info=True,
        )
        return None
    if isinstance(tier, str) and tier.strip() in ("default", "small"):
        return tier.strip()
    return None


def _whole_gb(vram_mib: int) -> int:
    """Video memory in the whole GB the GPU line shows (an 8 GB card may
    report 8188 MiB). The Decision 150 floor compares this same number."""
    return round(vram_mib / 1024)


def _gpu_text(name: Optional[str], vram_mib: Optional[int]) -> str:
    """The GPU line's value: "NVIDIA GeForce RTX 4080, 16 GB" (the error
    report's format), or "None found"."""
    if vram_mib is None:
        return NONE_FOUND
    return f"{name or 'NVIDIA graphics card'}, {_whole_gb(vram_mib)} GB"


class _GpuDetectThread(QThread):
    """Runs the two nvidia-smi shell-outs off the GUI thread.

    QuantTierPage used to call ``detect_vram_mib()`` /
    ``detect_gpu_name()`` synchronously in ``__init__`` -- which runs
    inside ``YamabikoWizard.__init__``, BEFORE ``exec()`` shows anything.
    Each subprocess carries a 5 s timeout (gpu.py), so a wedged driver
    meant up to ~10 s of no-window wizard startup. QThread-subclass idiom
    mirrors ``overlay_app.model_fetch._DownloadThread`` /
    ``tray._DiagnosticsThread``; the result crosses back to the page's
    slot as a queued cross-thread signal.

    ``object`` payloads on purpose: ``vram_mib`` may be None (no NVIDIA
    driver / timeout) and a typed int signal would coerce it.
    """

    detected = Signal(object, object)  # (vram_mib: int|None, name: str|None)

    def run(self) -> None:  # worker thread
        vram_mib = _gpu.detect_vram_mib()
        name = _gpu.detect_gpu_name()
        self.detected.emit(vram_mib, name)


class QuantTierPage(QWizardPage):
    """The Quality step: Standard or Smaller (``translation.quant_tier``)."""

    def __init__(self, *, palette: dict = MIST) -> None:
        super().__init__()
        self.setTitle("Pick a model size")
        self.setButtonText(QWizard.WizardButton.NextButton, "Start download")
        self._palette = palette

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(0)

        # "Graphics card" + the detected card. Seeded with "Checking…"; the
        # async detection result replaces it (queued signal). The page never
        # blocks on nvidia-smi -- see _GpuDetectThread.
        gpu_row = QHBoxLayout()
        gpu_row.setSpacing(10)
        key = QLabel("Graphics card")
        key.setObjectName("muted")
        key.setFont(ui_font(14))
        self.gpu_label = QLabel(CHECKING)
        self.gpu_label.setFont(ui_font(14))
        gpu_row.addWidget(key)
        gpu_row.addWidget(self.gpu_label)
        gpu_row.addStretch(1)
        layout.addLayout(gpu_row)
        self.gpu_help = QLabel(NO_GPU_HELP)
        self.gpu_help.setObjectName("help")
        self.gpu_help.setFont(ui_font(13))
        self.gpu_help.hide()
        layout.addWidget(self.gpu_help)

        # A tier saved in user_overrides.yaml wins outright: seed the
        # selection from it and keep the async suggestion off it (see
        # _on_gpu_detected). Otherwise seed the conservative Standard
        # IMMEDIATELY (matches gpu.suggest_quant_tier(None)); the detection
        # result only refines it, and only while the user hasn't picked.
        self._saved_tier = _saved_quant_tier()
        layout.addSpacing(20)
        self.tier_control = SegmentedControl(
            [
                SegmentOption("default", "Standard",
                              "Best translations. For cards with 10 GB or more."),
                SegmentOption("small", "Smaller",
                              f"Needs less video memory. For {_gpu.SMALL_TIER_MIN_GB} GB cards."),
            ],
            palette, variant="cards", wizard=True,
            value="small" if self._saved_tier == "small" else "default",
        )
        self.tier_control.setFont(ui_font(14))
        # The two rows keep the handles the frozen E2E drives (L10).
        self.default_radio = self.tier_control.segment("default")
        self.small_radio = self.tier_control.segment("small")
        self.default_radio.setObjectName("quant_default_radio")
        self.small_radio.setObjectName("quant_small_radio")
        layout.addWidget(self.tier_control)

        layout.addSpacing(10)
        note = QLabel("You can change this later in Settings.")
        note.setObjectName("help")
        note.setFont(ui_font(13))
        layout.addWidget(note)
        layout.addStretch(1)

        # A user pick: a row's click (also a click on the chosen row) or an
        # arrow-key move (valueChanged). setValue from the detection emits
        # neither, so a late result never counts as a pick.
        self._user_picked = False
        self.tier_control.valueChanged.connect(self._on_user_pick)
        self.default_radio.clicked.connect(self._on_user_pick)
        self.small_radio.clicked.connect(self._on_user_pick)

        self._detect_thread: Optional[_GpuDetectThread] = None
        self._start_detection()

    def _start_detection(self) -> None:
        """Kick off the nvidia-smi probes on a worker QThread."""
        thread = _GpuDetectThread(parent=self)
        thread.detected.connect(self._on_gpu_detected)
        self._detect_thread = thread
        thread.start()

    def _on_user_pick(self, *_args) -> None:
        self._user_picked = True

    def _on_gpu_detected(self, vram_mib, gpu_name) -> None:
        """Main-thread slot: the GPU line, the recommendation, and the
        suggestion when nobody picked yet."""
        self.gpu_label.setText(_gpu_text(gpu_name, vram_mib))
        # A card under the Smaller tier's floor fits neither tier: no
        # recommendation, and the requirement line shows as for no card
        # (Decision 150), in the whole GB the GPU line shows.
        fits = vram_mib is not None and _whole_gb(vram_mib) >= _gpu.SMALL_TIER_MIN_GB
        self.gpu_help.setVisible(not fits)
        if vram_mib is None:
            return  # no card: no recommendation; Standard stays seeded
        suggested = _gpu.suggest_quant_tier(vram_mib)
        if fits:
            self.tier_control.set_note(suggested, RECOMMENDED)
        if self._user_picked or self._saved_tier is not None:
            # A pick already exists (made this session, or saved from an
            # earlier run); detection only informs.
            return
        self.tier_control.setValue(suggested)

    def shutdown_thread(self, wait_ms: int = 2000) -> None:
        """Wizard-teardown hook: join the detection thread, or detach it
        into ``thread_keeper`` when a wedged nvidia-smi still holds it
        (each shell-out can block up to 5 s). Destroying a page whose
        child QThread still runs is a Qt 6 qFatal abort -- and the wizard
        can be cancelled while the probe runs. The detached thread exits
        on its own once gpu.py's subprocess timeouts fire; its signal is
        disconnected first so a late result can't drive a dying page.
        """
        thread = self._detect_thread
        self._detect_thread = None
        if thread is None:
            return
        thread.quit()
        if thread.wait(wait_ms):
            return
        try:
            thread.detected.disconnect(self._on_gpu_detected)
        except (RuntimeError, TypeError):
            pass
        thread_keeper.release(thread)

    def selected_tier(self) -> str:
        """``'default'`` or ``'small'``."""
        return self.tier_control.value()

    def isComplete(self) -> bool:  # noqa: N802 -- Qt convention
        # A tier is always selected (seeded in __init__).
        return True

    def validatePage(self) -> bool:  # noqa: N802 -- Qt convention
        """Persist the chosen tier before the Download step starts.

        Writing here (rather than in the wizard's accept()) means the
        Download step -- which reads the merged config in initializePage --
        sees the pick on this very "Start download". Going back and
        changing it rewrites idempotently.

        The coupled pair (translation.quant_tier *and*
        asr.faster_whisper.compute_type) is persisted through
        :func:`~yamabiko.config_writer.write_quant_tier`, so it lands
        together or not at all -- a mismatched pair would run one tier's
        model against the other tier's ASR compute type.

        A persist failure BLOCKS the advance (the Download step would fetch
        the wrong-tier model): "Couldn't save your settings" asks; Retry
        writes again, Cancel keeps the user here.
        """
        tier = self.selected_tier()
        try:
            from yamabiko.config_writer import write_quant_tier
            write_quant_tier(tier)
        except Exception as exc:
            _log.exception("failed to persist quant tier pair (%s)", tier)
            from yamabiko import paths as _paths
            try:
                target = str(_paths.user_overrides_path())
            except Exception:
                target = "user_overrides.yaml"
            if SaveFailedDialog.ask(self, self._palette, target, exc, title=BRAND):
                return self.validatePage()
            return False
        return True
