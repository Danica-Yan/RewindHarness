"""W7's attribution-accuracy validation, against synthetic ground truth (see
`flywheel/eval_attribution.py`'s module docstring for why synthetic — there
is no real failure corpus yet to hand-label 30-50 trajectories from) and
using auto-paired patches (`auto_patch_for_step`) rather than hand-written
ones, so this also exercises the W7 "auto-pairing" item end to end."""
from __future__ import annotations

import asyncio

from rewind.flywheel.eval_attribution import evaluate_attribution_accuracy, make_scenario


def test_scenario_construction_is_internally_consistent_at_every_root_position():
    for rp in range(1, 6):
        s = make_scenario(rp)
        assert s.candidates == list(range(1, rp + 1))
        assert len(s.bad_script) == rp + 1     # rp-1 reads + 1 mutating call + 1 closing text
        assert len(s.good_script) == rp + 1


def test_bisection_finds_the_true_root_step_exactly_at_every_position():
    rows, metrics = asyncio.run(evaluate_attribution_accuracy([1, 2, 3, 4, 5]))
    for row in rows:
        assert row.predicted == row.root_position, row
        assert row.monotonic
    assert metrics["exact_match_accuracy"] == 1.0
    assert metrics["within_1_step_accuracy"] == 1.0
    assert metrics["n"] == 5


def test_bisection_uses_o_log_n_oracle_calls_not_a_linear_scan():
    rows, _metrics = asyncio.run(evaluate_attribution_accuracy([5]))
    row = rows[0]
    # A linear scan over 5 candidates would cost up to 5 oracle calls; even
    # with the neighborhood re-check on top of a clean bisection, this
    # should stay well under that.
    assert row.oracle_calls <= 4, row
