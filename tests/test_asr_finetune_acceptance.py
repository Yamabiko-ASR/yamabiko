"""Acceptance gate A2 of the fine-tuning pilot (``scripts.asr_finetune.acceptance``):
long word lines hold when the paired delta's upper bound is at least 0 for the
band 13-20, the band 21+ and the pooled 13+ lines."""

from scripts.asr_finetune.acceptance import SetResult, gate_a2


def _band(caught: int, lines: int) -> dict:
    return {"lines": lines, "aligned_caught_reading": caught, "aligned_recall_reading": caught / lines}


def _delta(delta: float, lower: float, upper: float) -> dict:
    return {"delta": delta, "lower": lower, "upper": upper}


def test_gate_a2_fails_the_21_plus_band_whose_upper_bound_is_below_zero() -> None:
    bins = {"base": {"13-20": _band(3000, 3083), "21+": _band(1926, 1959)},
            "cand": {"13-20": _band(2990, 3083), "21+": _band(1900, 1959)}}
    silver = SetResult(
        set_id="silver-v5",
        entry={
            "metrics": {label: {"kind=words": {"length_bins": by_band}} for label, by_band in bins.items()},
            "deltas": {"cand": {"kind=words": {"length_bins": {
                # An upper bound of exactly 0 is no significant regression.
                "13-20": {"aligned_recall_reading": _delta(-0.0032, -0.0071, 0.0)},
                "21+": {"aligned_recall_reading": _delta(-0.0133, -0.0199, -0.0066)},
            }}}},
        },
        baseline="base",
        candidate="cand",
        pooled=_delta(-0.0071, -0.0110, 0.0004),
        phantom={},
    )

    checks = gate_a2(silver)

    assert [(check["check"], check["pass"]) for check in checks] == [
        ("word lines 13-20 recall", True),
        ("word lines 21+ recall", False),
        ("word lines 13+ pooled recall", True),
    ]
    assert checks[2]["candidate"] == {"rate": 4890 / 5042, "hits": 4890, "total": 5042}
