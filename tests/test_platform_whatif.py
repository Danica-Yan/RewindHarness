"""Platform P5: judging recorded runs under a new access policy, offline.

The corpus (tests/whatif_corpus.py) is real rewind execution under the
example multi-tenant policy. No model calls, except the scripted
continuation in the fork-confirmation test.
"""
from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

import pytest
from conftest import call, text
from test_platform import ev, go, platform_registry, principal

import whatif_corpus as wc
from rewind import ScriptedProvider, SnapshotStore, TraceWriter
from rewind.env import ToyRetailEnv
from rewind.platform import InMemoryApprovalStore, Principal
from rewind.platform.whatif import classify, confirm_first_divergence, what_if

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus():
    return wc.build(48, seed=3)


def rules(*, drop=(), set_effect=None, add=()):
    out = [copy.deepcopy(r) for r in wc.BASE_RULES if r["id"] not in drop]
    for r in out:
        if set_effect and r["id"] in set_effect:
            r["effect"] = set_effect[r["id"]]
    return out + list(add)


def recorded_principal_or(p):
    return lambda s: Principal.model_validate(s["cfg"]["principal"]) if s["cfg"].get("principal") else p


# ---------------------------------------------------------------------------
# the basis: the same policy reproduces every recorded decision
# ---------------------------------------------------------------------------
def test_same_policy_changes_nothing(corpus):
    rep = what_if(corpus.traces, wc.new_access(wc.BASE_RULES), run_outcomes=corpus.outcomes)
    own_authz = sum(1 for t in corpus.traces for e in t if e.type == "authz" and not e.inherited)
    assert rep.calls == own_authz > 0
    assert rep.diffs == [] and rep.old_versions == [rep.new_version]
    seen = {o for (o, _n) in rep.transitions}
    assert {"allow", "approval", "deny", "hidden", "unknown"} <= seen        # every outcome kind exercised
    assert set(rep.skipped) == {r for r, s in corpus.scenario.items() if s == "legacy"}


@pytest.mark.parametrize("old,new,want", [
    ("allow", "allow", "same"), ("allow", "approval", "tightened"), ("approval", "deny", "tightened"),
    ("deny", "allow", "loosened"), ("hidden", "approval", "loosened"), ("deny", "hidden", "reclassified"),
    ("unenforced", "allow", "same"), ("unenforced", "deny", "tightened"),
])
def test_classify(old, new, want):
    assert classify(old, new, None, None) == want


def test_same_outcome_by_another_rule_is_reported_but_is_not_a_change():
    assert classify("allow", "allow", "rule:a", "rule:b") == "rule_changed"


# ---------------------------------------------------------------------------
# what a change does
# ---------------------------------------------------------------------------
def test_dropping_tenant_isolation_surfaces_the_cross_tenant_writes(corpus):
    rep = what_if(corpus.traces, wc.new_access(rules(drop={"files-outside-own-workspace"})))
    loosened = rep.loosened()
    assert loosened and all(d.name == "write_file" and "/../" in d.args["path"] for d in loosened)
    assert {(d.old, d.new, d.new_reason) for d in loosened} == {("deny", "allow", "rule:files-editor")}
    assert rep.summary()["blocked_calls_now_running"] == len(loosened)
    assert rep.tightened() == []


def test_tightening_counts_the_writes_that_really_happened(corpus):
    new = rules(add=[{"id": "no-cancel-t2", "effect": "deny", "tenants": ["t2"],
                      "capabilities": ["local.cancel_order"]}])
    rep = what_if(corpus.traces, wc.new_access(new), run_outcomes=corpus.outcomes)
    stopped = rep.stopped_writes()
    assert stopped and {d.tenant for d in stopped} == {"t2"}
    assert all((d.name, d.ran, d.ok, d.risk) == ("cancel_order", True, True, "write") for d in stopped)
    interrupted = rep.runs_interrupted()
    assert sorted(interrupted["passed"]) == sorted({d.run_id for d in stopped})


def test_taint_is_judged_on_the_history_the_call_really_had(corpus):
    """A fetch and an email in the SAME turn: when the email was authorised
    the page's content was not in the conversation yet. Only the email that
    came in a later turn is judged as tainted."""
    new = rules(set_effect={"external-after-untrusted-input": "deny"})
    rep = what_if(corpus.traces, wc.new_access(new))
    by_scenario = {corpus.scenario[d.run_id] for d in rep.diffs}
    assert by_scenario == {"inject_then_email", "resumed_after_approval"}
    assert all((d.old, d.new, d.tainted) == ("approval", "deny", True) for d in rep.diffs)


def test_catalog_changes_are_judged_too(corpus):
    access = wc.new_access(wc.BASE_RULES)
    access.catalog.resolve("fetch_url").untrusted_output = False           # "the docs site is trusted now"
    rep = what_if(corpus.traces, access)
    assert {(d.name, d.old, d.new) for d in rep.diffs} == {("send_email", "approval", "allow")}


def test_only_the_first_change_in_a_run_is_clean():
    _, _, trace, *_ = go([call("list_orders", user_id="u1"),
                          call("cancel_order", order_id="O1", reason="ordered_by_mistake"), text("done")],
                         principal())
    rep = what_if([trace.events], wc.new_access(rules(add=[
        {"id": "no-orders", "effect": "deny", "capabilities": ["local.list_orders", "local.cancel_order"]}])))
    first, second = rep.diffs
    assert (first.name, first.downstream, second.name, second.downstream) == \
        ("list_orders", False, "cancel_order", True)
    assert rep.first_divergences() == [first] and rep.summary()["downstream_changes"] == 1


def test_runs_recorded_before_access_control(corpus):
    legacy = [t for t in corpus.traces if corpus.scenario[t[0].run_id] == "legacy"]
    assert what_if(legacy, wc.new_access(wc.BASE_RULES)).skipped
    rep = what_if(legacy, wc.new_access(wc.BASE_RULES), principal_for=recorded_principal_or(principal(roles=())))
    assert {(d.name, d.old, d.new) for d in rep.diffs} == {("list_orders", "unenforced", "hidden"),
                                                           ("cancel_order", "unenforced", "hidden")}
    assert len(rep.stopped_writes()) == len(legacy)


def test_a_fork_is_judged_only_on_its_own_calls(corpus):
    fork = next(t for t in corpus.traces if corpus.scenario[t[0].run_id] == "fork")
    own = sum(1 for e in fork if e.type == "authz" and not e.inherited)
    inherited = sum(1 for e in fork if e.type == "authz" and e.inherited)
    assert own and inherited
    assert what_if([fork], wc.new_access(wc.BASE_RULES)).calls == own


def test_what_if_has_no_side_effects(corpus):
    approvals = InMemoryApprovalStore()
    access = wc.new_access(rules(add=[{"id": "all-cancels-need-a-human", "effect": "require_approval",
                                       "capabilities": ["local.cancel_order"]}]), approvals)
    rep = what_if(corpus.traces, access)
    assert any(d.new == "approval" for d in rep.diffs)
    assert approvals.pending() == []                     # judged, never filed as approval requests


def test_report_renders_and_serialises(corpus):
    rep = what_if(corpus.traces, wc.new_access(rules(drop={"files-outside-own-workspace"})))
    text_ = rep.render_text()
    assert "Blocked calls that would now run" in text_ and "deny -> allow" in text_
    data = json.loads(rep.to_json())
    assert data["summary"]["changes"]["loosened"] == len(data["diffs"]) == len(rep.loosened())


# ---------------------------------------------------------------------------
# confirming a divergence for real
# ---------------------------------------------------------------------------
def test_confirm_first_divergence_replays_the_recorded_action_under_the_new_policy():
    snaps = SnapshotStore()
    from rewind import AgentConfig, AgentRunner
    reg = platform_registry()
    cfg = AgentConfig(system_prompt="sys", principal=principal().model_dump(mode="json"))
    rec = TraceWriter("rec-1")
    asyncio.run(AgentRunner(ScriptedProvider([call("list_orders", user_id="u1"),
                                              call("cancel_order", order_id="O1", reason="ordered_by_mistake"),
                                              text("done")]), reg, cfg, snaps, sleep=wc._no_sleep)
                .run(ToyRetailEnv(), "cancel my order", rec))
    new = rules(add=[{"id": "freeze-cancels", "effect": "deny", "capabilities": ["local.cancel_order"]}])
    (diff,) = what_if([rec.events], wc.new_access(new)).first_divergences()
    assert (diff.step, diff.name, diff.new) == (2, "cancel_order", "hidden")   # unconditional deny hides it

    provider = ScriptedProvider([text("I can't cancel orders right now; a human will follow up.")])
    out = TraceWriter("confirm-1")
    res = asyncio.run(confirm_first_divergence(rec.events, diff, snapshots=snaps,
                                               registry=platform_registry(rules=new), provider=provider,
                                               env_factory=ToyRetailEnv.restore, trace=out))
    assert res.status == "completed" and provider.calls == 1          # step 2 was not asked again
    own = [e for e in out.events if not e.inherited]
    # what really happened under the new policy is what the offline report predicted
    assert [a["verdict"] for a in (e.data for e in own if e.type == "authz")] == [diff.new]
    assert [m.data["message"]["content"] for m in own if m.type == "message"
            and m.data["message"]["role"] == "tool"][0].startswith("ERROR[HALLUCINATED_TOOL]")
    end = snaps.get(res.env_hash)
    assert end["orders"]["O1"]["status"] == "pending"                 # the write did not happen


# ---------------------------------------------------------------------------
# the CLI, as a CI check on a policy change
# ---------------------------------------------------------------------------
def test_cli_fails_a_loosening_policy(corpus, tmp_path, capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import policy_diff
    tdir = tmp_path / "traces"
    tdir.mkdir()
    for t in corpus.traces:
        (tdir / f"{t[0].run_id}.jsonl").write_text("".join(e.model_dump_json() + "\n" for e in t))
    (tmp_path / "loose.json").write_text(json.dumps(rules(drop={"files-outside-own-workspace"})))
    (tmp_path / "same.json").write_text(json.dumps(wc.BASE_RULES))
    (tmp_path / "cat.json").write_text(json.dumps(wc.CATALOG_OVERRIDES))
    common = [str(tdir), "--registry", "whatif_corpus:plain_registry", "--catalog", str(tmp_path / "cat.json")]

    assert policy_diff.main(common + ["--policy", str(tmp_path / "same.json"), "--fail-on", "any"]) == 0
    assert policy_diff.main(common + ["--policy", str(tmp_path / "loose.json"), "--fail-on", "tightened"]) == 0
    out = tmp_path / "report.json"
    assert policy_diff.main(common + ["--policy", str(tmp_path / "loose.json"), "--fail-on", "loosened",
                                      "--json", str(out)]) == 1
    assert "FAIL" in capsys.readouterr().err
    assert json.loads(out.read_text())["summary"]["changes"]["loosened"] > 0
    rep = policy_diff.main(common + ["--policy", str(tmp_path / "same.json"), "--as", "t9:u1:legacy-bot:"])
    assert rep == 0 and "skipped" not in capsys.readouterr().out


MIXED_CHANGE = rules(
    drop={"files-outside-own-workspace"},                                   # loosen
    set_effect={"external-after-untrusted-input": "deny"},                 # tighten approval -> deny
    add=[{"id": "t3-cancels-need-a-human", "effect": "require_approval", "tenants": ["t3"],
          "capabilities": ["local.cancel_order"]},                         # tighten allow -> approval
         {"id": "docs-for-everyone", "effect": "allow", "capabilities": ["local.get_order"]}])  # loosen hidden -> allow


def test_every_first_divergence_is_confirmed_by_a_live_fork(corpus):
    """The offline report is a prediction. Check it: fork every run at its
    first divergence under the new policy and compare the decision the live
    loop makes there with the predicted one."""
    from rewind.platform.whatif import recorded_outcome
    rep = what_if(corpus.traces, wc.new_access(MIXED_CHANGE))
    firsts = rep.first_divergences()
    kinds = {(d.old, d.new) for d in firsts}
    assert {"tightened", "loosened"} <= {d.change for d in firsts} and len(kinds) >= 3
    by_run = {t[0].run_id: t for t in corpus.traces}
    for d in firsts:
        out = TraceWriter(f"confirm-{d.run_id}")
        asyncio.run(confirm_first_divergence(by_run[d.run_id], d, snapshots=corpus.snapshots,
                                             registry=platform_registry(rules=MIXED_CHANGE),
                                             provider=ScriptedProvider([text("ok")] * 3),
                                             env_factory=ToyRetailEnv.restore, trace=out))
        live = next(e.data for e in out.events if e.type == "authz" and not e.inherited
                    and e.step == d.step and e.data["idx"] == d.idx)
        assert recorded_outcome(live) == d.new, (d.run_id, d.name, d.old, d.new, live)
