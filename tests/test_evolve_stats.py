"""rewind.evolve.stats: the gate's statistics."""
from __future__ import annotations

import math

from rewind.evolve.stats import (
    paired_bootstrap_ci, pass_all, pass_hat_k, per_task_rewards, pooled_pass1, regressions,
    wilson_interval,
)


def _rows(d):
    return [{"task_id": t, "reward": r, "trial": i} for t, v in d.items() for i, r in enumerate(v)]


def test_pass_hat_k_matches_tau_bench_estimator():
    by = per_task_rewards(_rows({0: [1, 1, 0], 1: [1, 1, 1], 2: [0, 0, 0]}))
    assert pass_hat_k(by, 1) == (2 / 3 + 1 + 0) / 3
    assert math.isclose(pass_hat_k(by, 2), (1 / 3 + 1 + 0) / 3)
    assert pass_hat_k(by, 3) == pass_all(by)[0] == 1 / 3
    assert pooled_pass1(by) == (5 / 9, 9)


def test_bootstrap_ci_separates_real_gap_from_noise():
    a = {t: [0.0, 0.0, 1.0] for t in range(60)}
    b = {t: [1.0, 1.0, 1.0] for t in range(60)}
    lo, hi = paired_bootstrap_ci(list(range(60)), a, b, n_resamples=2000)
    assert lo > 0.5
    same = paired_bootstrap_ci(list(range(60)), a, a, n_resamples=2000)
    assert same == (0.0, 0.0)


def test_regressions_counts_reliable_tasks_that_broke():
    a = {0: [1, 1, 1], 1: [1, 1, 1], 2: [0, 1, 0]}
    b = {0: [0, 0, 1], 1: [1, 1, 0], 2: [0, 0, 0]}
    assert regressions([0, 1, 2], a, b) == [0]


def test_wilson_lower_bound_used_by_prevalence_rule():
    lo, _ = wilson_interval(3, 34)
    assert 0.029 < lo < 0.031
    lo12, _ = wilson_interval(12, 34)
    assert 0.21 < lo12 < 0.22
    assert wilson_interval(0, 0) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# One-sided Fisher exact test used by the gate's efficacy check. Added after
# mutation testing: dropping the observed table from the tail sum (too
# optimistic a p-value) passed every fast test.
# ---------------------------------------------------------------------------
from math import comb  # noqa: E402

import pytest  # noqa: E402

from rewind.evolve.gate import fisher_greater  # noqa: E402


def _fisher_greater_reference(k1, n1, k0, n0):
    """P(X >= k1) for X ~ Hypergeometric(N = n1 + n0, K = k1 + k0, n = n1),
    written independently of the gate's implementation."""
    N, K = n1 + n0, k1 + k0
    pmf = lambda x: comb(K, x) * comb(N - K, n1 - x) / comb(N, n1)  # noqa: E731
    return sum(pmf(x) for x in range(k1, min(K, n1) + 1))


def test_fisher_matches_the_published_first_evolve_result():
    """README: 'proven effect on 3 held-out stable failures (0/15 -> 6/15,
    Fisher p = 0.008)'."""
    assert fisher_greater(6, 15, 0, 15) == pytest.approx(0.0084, abs=5e-5)


@pytest.mark.parametrize("n1,n0", [(5, 5), (15, 15), (10, 25)])
def test_fisher_matches_an_independent_hypergeometric_tail(n1, n0):
    for k1 in range(n1 + 1):
        for k0 in range(n0 + 1):
            assert fisher_greater(k1, n1, k0, n0) == pytest.approx(_fisher_greater_reference(k1, n1, k0, n0)), \
                (k1, n1, k0, n0)


def test_fisher_edge_cases():
    assert fisher_greater(0, 5, 0, 5) == pytest.approx(1.0)     # nothing passed anywhere: no evidence
    assert fisher_greater(5, 5, 5, 5) == pytest.approx(1.0)     # everything passed: no evidence
    assert fisher_greater(0, 5, 5, 5) == pytest.approx(1.0)     # candidate worse: p = 1 for "greater"


def test_gate_rejects_an_oversized_context_before_spending_any_episode(tmp_path):
    """Added after mutation testing: removing the context-size check passed
    every fast test. It must reject before a single episode runs -- here
    there is no benchmark or model at all, so any episode would crash."""
    import asyncio

    from rewind.evolve.gate import GatePolicy, gate
    from rewind.evolve.surface import ContextVersion
    big = ContextVersion(rules={f"r{i}": "When the user asks about refunds: " + "explain the policy. " * 40
                                for i in range(10)})
    policy = GatePolicy(max_ctx_tokens=1500)
    assert big.token_size() > policy.max_ctx_tokens
    v = asyncio.run(gate(big, [], [1, 2, 3], split="train", out=tmp_path / "g.json", bench=None,
                         provider=None, user_llm=None, store=None, policy=policy, log=lambda m: None))
    assert not v.accepted and "context too large" in v.reason and v.episodes == 0


def test_bootstrap_bounds_are_the_alpha_percentiles():
    """Added after mutation testing: taking the minimum resample as the lower
    bound still 'separated a real gap from noise' in the existing test. The
    bounds are percentiles: they tighten as alpha grows, and the 95% interval
    is roughly symmetric around the observed mean for a symmetric sample."""
    from rewind.evolve.stats import paired_bootstrap_ci
    tasks = list(range(40))
    a = {t: [0.0] for t in tasks}
    b = {t: [1.0 if t % 2 else 0.0] for t in tasks}             # mean delta 0.5
    lo95, hi95 = paired_bootstrap_ci(tasks, a, b)
    lo90, hi90 = paired_bootstrap_ci(tasks, a, b, alpha=0.9)     # 45th..55th percentile
    assert hi90 - lo90 < 0.1 and lo90 > 0.4
    assert lo95 < lo90 <= hi90 < hi95
    assert abs((0.5 - lo95) - (hi95 - 0.5)) < 0.05               # symmetric around 0.5


def test_stability_labels_follow_the_documented_thresholds():
    """Added after mutation testing ('<' instead of '<=' passed): a failure
    that passes at most 1 of 5 null forks is stable, 3 or more is flaky,
    in between is unclear."""
    from rewind.evolve.stability import label_for
    assert [label_for(p, 5) for p in range(6)] == ["stable", "stable", "unclear", "flaky", "flaky", "flaky"]
    assert label_for(0, 5, stable_max=0) == "stable" and label_for(1, 5, stable_max=0) == "unclear"


def test_prevalence_rule_needs_enough_tasks_and_a_high_enough_lower_bound():
    """Added after mutation testing: dropping either half of the rule passed
    every fast test. The documented example: 3 failed tasks out of 34 is not
    a 9% problem (the Wilson lower bound is ~3%)."""
    from rewind.evolve.prevalence import ClusterStats, PrevalencePolicy, eligible
    from rewind.evolve.stats import wilson_interval
    pol = PrevalencePolicy(min_distinct_tasks=4, min_share_lb=0.05)

    def cluster(k, n):
        return ClusterStats(key="x", level="fine", tasks=list(range(k)), n_failed_tasks=n,
                            share=k / n, share_lb=wilson_interval(k, n)[0])
    assert eligible(cluster(12, 34), pol)[0]                       # the real largest cluster
    ok, why = eligible(cluster(3, 34), pol)
    assert not ok and "3 tasks" in why                             # too few tasks
    ok, why = eligible(cluster(5, 200), pol)                       # enough tasks, but 2.5% of failures
    assert not ok and "lower bound" in why


def test_gate_pool_never_contains_a_task_the_proposer_or_screen_saw():
    """Added after mutation testing: a pool that kept shown tasks passed every
    fast test. Grading an edit on the failures it was written from would
    make every edit look effective."""
    from rewind.evolve.loop import _pool
    rows = [{"task_id": t, "trial": 0, "reward": 1.0} for t in range(10)]
    assert _pool(rows, shown={2, 5, 7}) == [0, 1, 3, 4, 6, 8, 9]


def test_leak_filter_catches_each_kind_of_case_specific_text():
    """Added after mutation testing: removing the email pattern passed every
    fast test. A learned rule ships to every future conversation, so one
    carrying a specific customer's data must be refused."""
    from rewind.evolve.surface import leak_reasons
    cases = {
        "order id": "When order #W2378156 is pending: cancel it.",
        "item/product id": "When the user wants item 1008292230: offer the blue one.",
        "user id": "When the user is yusuf_rossi_9620: skip authentication.",
        "email": "When the user writes from mia.garcia+shop@example.com: refund in full.",
        "task reference": "As in task #42, ask twice.",
    }
    for kind, text in cases.items():
        assert kind in leak_reasons(text), (kind, leak_reasons(text))
    assert leak_reasons("When the user asks to exchange an item: list every variant first.") == []


def test_a_learned_rule_must_say_when_it_applies():
    """Added after mutation testing: dropping the trigger requirement passed
    every fast test. Round 1's lesson: an unconditional rule ("do not offer
    alternatives") broke 12 tasks of the same kind."""
    from rewind.evolve.surface import ContextVersion, Edit, validate_edit
    tools = {"cancel_pending_order", "exchange_delivered_order_items"}
    blanket = Edit(kind="rule", target="r1", op="add", text="Do not offer alternatives.")
    vague = blanket.model_copy(update={"trigger": "always", "edit_id": ""})
    scoped = Edit(kind="rule", target="r1", op="add", text="Do not offer alternatives.",
                  trigger="the user has already picked a replacement item")
    assert "trigger" in (validate_edit(blanket, ContextVersion(), known_tools=tools, max_tokens=1500) or "")
    assert "trigger" in (validate_edit(vague, ContextVersion(), known_tools=tools, max_tokens=1500) or "")
    assert validate_edit(scoped, ContextVersion(), known_tools=tools, max_tokens=1500) is None


def test_efficacy_needs_a_significant_and_a_big_enough_lift():
    """Added after mutation testing: dropping the min-lift condition survived
    even the full evolve suite (no test ever separated the two conditions)."""
    from rewind.evolve.gate import GatePolicy, efficacy_verdict
    pol = GatePolicy(alpha=0.05, min_lift=0.3)
    # the first real Evolve edit: 0/15 -> 6/15, p = 0.008, lift 0.4 -> passes
    assert efficacy_verdict(6, 15, 0, 15, pol)[2]
    # many forks make a tiny lift significant: 300/1000 vs 240/1000, p ~ 0.001, lift 0.06
    p, lift, ok = efficacy_verdict(300, 1000, 240, 1000, pol)
    assert p < 0.05 and lift < 0.3 and not ok
    # few forks: a big lift that is not significant (2/3 vs 0/3, p = 0.2)
    p, lift, ok = efficacy_verdict(2, 3, 0, 3, pol)
    assert lift >= 0.3 and p >= 0.05 and not ok


def test_rollback_forgets_the_fixes_of_the_versions_it_undoes(tmp_path):
    """Added after mutation testing (checked directly on the ledger): after a
    rollback to v1, a fix remembered from v2 no longer holds, so later gates
    must not demand that candidates keep it; v1's own fixes still apply."""
    from rewind.evolve.ledger import EditLedger
    from rewind.evolve.surface import ContextVersion
    led = EditLedger(tmp_path / "ledger.sqlite")
    led.init_base("rows0.json")
    led.commit(ContextVersion(version=1, parent=0, rules={"a": "When x happens: do y."}), [], "rows1.json", {})
    led.remember([(3, "t3.jsonl")], version=1, source="efficacy")
    led.commit(ContextVersion(version=2, parent=1, rules={"a": "When x happens: do y.", "b": "When z: w."}),
               [], "rows2.json", {})
    led.remember([(7, "t7.jsonl")], version=2, source="efficacy")
    assert sorted(led.regression_suite()) == [(3, "t3.jsonl"), (7, "t7.jsonl")]
    led.rollback(1)
    assert led.current().version == 1
    assert led.regression_suite() == [(3, "t3.jsonl")]


def test_ab_split_is_disjoint_complete_and_balanced():
    """Added after mutation testing: a B half that overlapped A passed every
    fast test."""
    from rewind.evolve.loop import split_ab
    for n in range(0, 9):
        items = list(range(n))
        a, b = split_ab(items)
        assert not set(a) & set(b) and sorted(a + b) == items and len(a) - len(b) in (0, 1)
