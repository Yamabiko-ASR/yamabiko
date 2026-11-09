"""Set HuggingFace environment variables at import time.

Must be imported BEFORE any ``huggingface_hub`` code runs — `main.py`,
`watchdog.py`, and `wizard_pages/downloads.py` all import this first.

What gets pinned:

- ``HF_HOME`` → ``%LOCALAPPDATA%\\Yamabiko\\hf-cache\\`` so the wizard and
  runtime share one cache. Without this, the wizard warms HF's default
  user cache (``~/.cache/huggingface/hub``) while runtime may be looking
  elsewhere, which can cause a re-download on the first translation.

- ``HF_XET_HIGH_PERFORMANCE`` → ``1``. Tunes hf_xet for gigabit+ machines:
  larger chunk buffers, more parallel workers, higher concurrency. HF
  still throttles anonymous users to ~10 MB/s per IP but this gets us
  to the ceiling instead of the ~3 MB/s single-stream fallback.

Setting env vars in a module imported at app startup is the idiomatic
way to configure HF; setting them per-download is a footgun because
some HF internals cache the values on first import.
"""
from __future__ import annotations

import os

from yamabiko import paths

# setdefault — if a contributor / packager has already pinned HF_HOME
# (e.g. to share a cache with another tool) we respect their choice.
os.environ.setdefault("HF_HOME", str(paths.app_data_dir() / "hf-cache"))
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
