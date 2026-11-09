"""Unit tests for the HealthProbe wedge detector.

HealthProbe polls the pipeline's /livez (liveness, NOT readiness). On
fail_budget_s of continuous liveness failure it invokes on_wedged
(wired to the pipeline supervisor's request_restart) and re-arms for
the fresh pipeline. It never touches llama_stop_event and never returns
on a trip (a re-wedge must be caught again).

Seeded-defect drills are inline per test.
"""
from __future__ import annotations
import threading, time
from unittest.mock import MagicMock, patch
import pytest
from yamabiko.watchdog import HealthProbe


def _probe(on_wedged, *, stop_event, poll_interval_s=0.01, fail_budget_s=0.05):
    # One child for the whole test (generation never changes) and no
    # startup dialog: these tests exercise the per-child timers only.
    return HealthProbe(url="http://127.0.0.1:8765/livez",
                       stop_event=stop_event,
                       on_wedged=on_wedged,
                       spawn_generation=lambda: 1,
                       startup_dialog_up=lambda: False,
                       poll_interval_s=poll_interval_s,
                       fail_budget_s=fail_budget_s)


def test_probe_stays_quiet_while_live():
    """Fails red if: probe restarts a healthy pipeline."""
    stop = threading.Event(); wedged = []
    with patch("yamabiko.watchdog.loopback_http") as h:
        h.get.return_value = MagicMock(status_code=200)
        p = _probe(lambda: wedged.append(1), stop_event=stop, fail_budget_s=0.1)
        t = threading.Thread(target=p.run, daemon=True); t.start()
        time.sleep(0.3); stop.set(); t.join(1.0)
        assert wedged == []


def test_probe_fires_on_wedged_after_budget():
    """Fails red if: a sustained liveness failure never restarts the
    pipeline (the whole point of the wedge detector).

    First probe 200 arms the budget; then 503s simulate a wedge.
    Seeded-defect drill: delete the on_wedged() call in the budget
    branch -> red.
    """
    stop = threading.Event(); wedged = threading.Event()
    calls = [0]
    def fake_get(*a, **k):
        calls[0] += 1
        return MagicMock(status_code=200 if calls[0] == 1 else 503)
    with patch("yamabiko.watchdog.loopback_http") as h:
        h.get.side_effect = fake_get
        p = _probe(lambda: wedged.set(), stop_event=stop, fail_budget_s=0.05)
        t = threading.Thread(target=p.run, daemon=True); t.start()
        assert wedged.wait(0.5), "on_wedged never fired"
        stop.set(); t.join(1.0)


def test_probe_rearms_after_restart_and_can_fire_again():
    """Fails red if: probe returns after the first trip (a second wedge
    of the freshly-restarted pipeline would go undetected).

    Pattern: 200 (arm), 503xN (trip #1), 200 (re-arm), 503xN (trip #2).
    Asserts on_wedged fires at least twice.

    Seeded-defect drill: change the re-arm `seen_live = False` to
    `seen_live = True` (or restore a `return` after on_wedged) -> only
    one trip fires -> `count[0] >= 2` goes red.
    """
    stop = threading.Event(); count = [0]
    def on_wedged(): count[0] += 1
    seq = iter([200] + [503]*5 + [200] + [503]*20)
    def fake_get(*a, **k):
        try: code = next(seq)
        except StopIteration: code = 503
        return MagicMock(status_code=code)
    with patch("yamabiko.watchdog.loopback_http") as h:
        h.get.side_effect = fake_get
        p = _probe(on_wedged, stop_event=stop, fail_budget_s=0.03)
        t = threading.Thread(target=p.run, daemon=True); t.start()
        time.sleep(0.5); stop.set(); t.join(1.0)
        assert count[0] >= 2, f"expected >=2 wedge restarts, got {count[0]}"


def test_probe_survives_raising_httpx_during_cold_boot_without_firing():
    """The first poll of every real launch raises -- connection refused
    while uvicorn is still binding. If the broad `except Exception` around
    httpx.get is narrowed or removed in a refactor, the probe thread dies
    on that first poll, permanently disabling the wedge detector, with the
    suite green.

    Contract side 1: a raise is 'not live' + cold-boot grace (no
    on_wedged), and the thread SURVIVES (call count keeps growing across
    two observation points -- survival, not snapshot, §3.6).

    Seeded-defect drill: raise a BaseException subclass from fake_get,
    which escapes the current except clause exactly as ConnectError would
    escape a narrowed one -> the thread dies on poll 1,
    `second > first` goes red.
    """
    import httpx as real_httpx

    stop = threading.Event(); wedged = []
    calls = [0]

    def fake_get(*a, **k):
        calls[0] += 1
        raise real_httpx.ConnectError("connection refused (uvicorn still binding)")

    with patch("yamabiko.watchdog.loopback_http") as h:
        h.get.side_effect = fake_get
        p = _probe(lambda: wedged.append(1), stop_event=stop, fail_budget_s=0.05)
        t = threading.Thread(target=p.run, daemon=True); t.start()
        time.sleep(0.15); first = calls[0]
        time.sleep(0.15); second = calls[0]
        stop.set(); t.join(1.0)

    assert first >= 1, "probe never polled"
    assert second > first, (
        f"probe thread stopped polling after a raising httpx.get "
        f"({first} -> {second} calls); the cold-boot ConnectError killed "
        f"it — wedge detection is permanently dead on every real launch"
    )
    assert wedged == [], (
        "raises before the first 200 must stay in cold-boot grace, "
        "not trip a force-restart of a still-starting pipeline"
    )


def test_probe_fires_on_wedged_when_armed_probe_starts_raising():
    """After the first 200 arms the budget, a wedge usually manifests as
    a raised ConnectError/timeout (process hung, socket gone) -- not a
    polite 503. Sustained raises must consume the fail budget and fire
    on_wedged just like 503s do.

    Seeded-defect drill: same BaseException-escaping-except seed as the
    cold-boot test -> thread dies on poll 2, on_wedged never fires ->
    `wedged.wait` red.
    """
    import httpx as real_httpx

    stop = threading.Event(); wedged = threading.Event()
    calls = [0]

    def fake_get(*a, **k):
        calls[0] += 1
        if calls[0] == 1:
            return MagicMock(status_code=200)
        raise real_httpx.ConnectError("socket gone: pipeline hung")

    with patch("yamabiko.watchdog.loopback_http") as h:
        h.get.side_effect = fake_get
        p = _probe(lambda: wedged.set(), stop_event=stop, fail_budget_s=0.05)
        t = threading.Thread(target=p.run, daemon=True); t.start()
        fired = wedged.wait(0.5)
        stop.set(); t.join(1.0)

    assert fired, (
        "sustained raising polls after the budget armed never fired "
        "on_wedged — a hung pipeline whose socket is gone (the common "
        "wedge signature) would never be force-restarted"
    )


def test_probe_does_not_fire_before_first_live():
    """Fails red if: cold-boot grace removed -- a still-starting pipeline
    (never served /livez) would be force-restarted in a loop.

    Seeded-defect drill: remove the `if not seen_live: pass` cold-boot
    grace (start counting toward the budget before the first 200) ->
    the all-503 stream trips on_wedged -> `wedged == []` goes red.
    """
    stop = threading.Event(); wedged = []
    with patch("yamabiko.watchdog.loopback_http") as h:
        h.get.return_value = MagicMock(status_code=503)
        p = _probe(lambda: wedged.append(1), stop_event=stop, fail_budget_s=0.05)
        t = threading.Thread(target=p.run, daemon=True); t.start()
        time.sleep(0.3); stop.set(); t.join(1.0)
        assert wedged == []


def test_probe_restart_startup_wedge_fires_again_within_startup_budget():
    """A pipeline that wedges DURING STARTUP after a probe restart used
    to be unsupervised forever: the post-restart re-arm reset to
    cold-boot grace, which is unbounded, so a recurring boot-time hang
    (same CUDA wedge in build_pipeline, before /livez binds) left the
    process alive, the probe waiting, and the supervisor blind -- app
    hung silently holding the single-instance mutex, Task Manager the
    only recovery. The INITIAL cold boot must stay unbounded (the
    first-run wizard legitimately takes as long as its downloads), but a
    probe-initiated restart follows a previously-live pipeline: model
    loads, not wizards -- a generous startup budget applies.

    Seeded defect: drop the startup_deadline arming after on_wedged ->
    only ONE wedge event ever fires -> the second wait times out -> red.
    """
    stop = threading.Event()
    wedges = []
    second = threading.Event()

    def on_wedged():
        wedges.append(1)
        if len(wedges) >= 2:
            second.set()

    calls = [0]

    def fake_get(*a, **k):
        calls[0] += 1
        # One healthy poll arms the budget; everything after is a wedge,
        # INCLUDING the restarted process (startup wedge).
        return MagicMock(status_code=200 if calls[0] == 1 else 503)

    with patch("yamabiko.watchdog.loopback_http") as h:
        h.get.side_effect = fake_get
        p = HealthProbe(
            url="http://127.0.0.1:8765/livez", stop_event=stop,
            on_wedged=on_wedged, spawn_generation=lambda: 1,
            startup_dialog_up=lambda: False, poll_interval_s=0.01,
            fail_budget_s=0.05, startup_budget_s=0.05,
        )
        t = threading.Thread(target=p.run, daemon=True); t.start()
        assert second.wait(1.5), (
            "restarted pipeline never served a 200 and the probe never "
            "re-fired: the startup wedge is unsupervised"
        )
        stop.set(); t.join(1.0)
