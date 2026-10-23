"""The agent loop (harness).

Determinism rules: no direct use of wall time or global randomness. Time comes
from the injected clock, jitter from an RNG seeded by run_id, and every provider
attempt (success or failure) is recorded so a replay can reproduce it.
"""
from __future__ import annotations

import asyncio
import copy
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .events import Event
from .providers import LLMResponse, MalformedOutput, ProviderError, ToolCall
from .snapshot import SnapshotStore
from .tools import ToolOutcome, ToolRegistry
from .trace import TraceWriter
from .util import SystemClock, request_hash, sha

REPAIR_PROMPT = ("Your previous output could not be parsed. Respond again with either a "
                 "valid tool call or a final answer.")
ARG_ERRORS = ("TOOL_ARG_INVALID", "HALLUCINATED_TOOL")


@dataclass
class AgentConfig:
    system_prompt: str = "You are a helpful agent. Use the provided tools."
    # budgets
    max_steps: int = 20
    max_tokens: int = 200_000
    wall_seconds: float = 300.0
    max_output_tokens: int = 1024
    temperature: float | None = 0.0
    # reliability
    max_llm_retries: int = 3
    backoff_base: float = 0.5
    backoff_cap: float = 8.0
    max_invalid_args: int = 2        # consecutive arg/hallucination errors before abort
    loop_warn_at: int = 3            # identical (tool,args) calls: warn (block) at Nth
    loop_kill_at: int = 5            # ... abort at Nth
    max_transient_tool_retries: int = 2
    # compatibility with other harnesses (e.g. tau-bench's own ToolCallingAgent,
    # which executes only the first tool call of a turn and reports tool
    # failures as "Error: <msg>"); defaults keep rewind's native behaviour
    single_tool_call: bool = False
    tool_error_format: str = "ERROR[{cls}]: {msg}"


@dataclass
class RunResult:
    run_id: str
    status: str
    reason: str | None
    steps: int
    tokens: int
    final_text: str | None
    env_hash: str


class RunAbort(Exception):
    def __init__(self, status: str, reason: str | None = None) -> None:
        super().__init__(f"{status}: {reason}")
        self.status = status
        self.reason = reason


class Checkpointer(Protocol):
    """Durability seam. `save` must raise RunAbort("fenced_out", ...) itself
    when the write is rejected (a newer worker took over) rather than
    returning a status, so the loop's existing RunAbort handling covers it.
    Declared async so a real (Postgres-backed) checkpointer can actually
    await network I/O here; the sqlite-backed one just returns immediately."""

    async def save(self, kind: str, step: int, trace: TraceWriter, env: Any,
                   messages: list[dict], state: LoopState,
                   action: LLMResponse | None = None) -> None: ...


@dataclass
class LoopState:
    """Loop counters that must survive a fork/resume; rebuildable from events."""
    tokens: int = 0
    call_counts: dict[str, int] = field(default_factory=dict)
    consecutive_invalid: int = 0

    def bump_call(self, name: str, args_hash: str) -> int:
        key = f"{name}|{args_hash}"
        self.call_counts[key] = self.call_counts.get(key, 0) + 1
        return self.call_counts[key]

    def note_result(self, error_class: str | None) -> None:
        self.consecutive_invalid = self.consecutive_invalid + 1 if error_class in ARG_ERRORS else 0

    @classmethod
    def from_events(cls, events: list[Event]) -> LoopState:
        st = cls()
        for e in events:
            if e.type == "llm_call" and "response" in e.data:
                u = e.data["response"].get("usage") or {}
                st.tokens += u.get("input_tokens", 0) + u.get("output_tokens", 0)
            elif e.type == "tool_call":
                st.bump_call(e.data["name"], e.data["args_hash"])
                st.note_result(e.data["error_class"])
        return st


class AgentRunner:
    def __init__(self, provider, registry: ToolRegistry, cfg: AgentConfig,
                 snapshots: SnapshotStore, clock=None, fallback=None, sleep=None,
                 checkpointer: Checkpointer | None = None,
                 debug_hook: Callable[[str, int], None] | None = None) -> None:
        self.provider = provider
        self.registry = registry
        self.cfg = cfg
        self.snapshots = snapshots
        self.clock = clock or SystemClock()
        self.fallback = fallback
        self.sleep = sleep or asyncio.sleep
        self.checkpointer = checkpointer
        # Test/debug seam only: lets tests simulate a process crash at an exact
        # point (e.g. "after this tool call, before the step checkpoint") by
        # raising from the hook. Never used in production wiring.
        self.debug_hook = debug_hook or (lambda tag, step: None)

    # ---- public entry points -------------------------------------------------
    async def run(self, env, task: str, trace: TraceWriter) -> RunResult:
        messages = [{"role": "system", "content": self.cfg.system_prompt},
                    {"role": "user", "content": task}]
        trace.emit("run_start", 0, {
            "task": task, "cfg": asdict(self.cfg),
            "initial_env_hash": self.snapshots.put(env.snapshot()),
            "has_fallback": self.fallback is not None,
        })
        for m in messages:
            trace.emit("message", 0, {"message": m})
        return await self._loop(env, messages, LoopState(), 1, trace, None)

    async def resume(self, env, messages: list[dict], state: LoopState, start_step: int,
                     trace: TraceWriter, forced: LLMResponse | None = None) -> RunResult:
        """Continue from a reconstructed state (used by fork). `forced` replaces the
        first LLM call (counterfactual action)."""
        return await self._loop(env, messages, state, start_step, trace, forced)

    # ---- internals -------------------------------------------------------------
    async def _loop(self, env, messages, state: LoopState, step: int, trace: TraceWriter,
                    forced: LLMResponse | None) -> RunResult:
        cfg = self.cfg
        rng = random.Random(int(sha(trace.run_id)[7:19], 16))
        t0 = self.clock.perf()
        tools = self.registry.schemas()
        status, reason, final_text = "completed", None, None
        last_step = step - 1
        try:
            while True:
                if step > cfg.max_steps:
                    raise RunAbort("budget_exceeded", "max_steps")
                if self.clock.perf() - t0 > cfg.wall_seconds:
                    raise RunAbort("budget_exceeded", "wall_time")
                if state.tokens > cfg.max_tokens:
                    raise RunAbort("budget_exceeded", "max_tokens")

                trace.emit("step_start", step, {"env_hash": self.snapshots.put(env.snapshot())})
                last_step = step
                resp = await self._llm_step(messages, tools, step, trace, state, rng, forced)
                forced = None
                if cfg.single_tool_call and len(resp.tool_calls) > 1:
                    trace.emit("guard", step, {"guard": "truncate_tool_calls",
                                               "dropped": len(resp.tool_calls) - 1})
                    resp = resp.model_copy(update={"tool_calls": resp.tool_calls[:1]})

                assistant = {"role": "assistant", "content": resp.text or "",
                             "tool_calls": [{"id": c.id, "name": c.name, "args": c.args}
                                            for c in resp.tool_calls]}
                messages.append(assistant)
                trace.emit("message", step, {"message": assistant})

                # Durably record the chosen action *before* touching any tool. If the
                # process dies before the matching "step" checkpoint, resume redoes
                # this same recorded action rather than asking the model again.
                # `pre_tool_messages` (not the growing `messages`) is what every
                # checkpoint for this step stores: on resume the step is always
                # replayed from idx 0 of `resp.tool_calls`, so restoring a message
                # list that already contains tool-result messages would duplicate
                # them. Mutating tools stay exactly-once anyway, because the "tool"
                # checkpoints below advance the *env* (and with it env.applied, the
                # idempotency ledger) after every call — a replayed idx whose
                # action_id is already in env.applied returns the cached outcome
                # instead of re-running the tool.
                pre_tool_messages = list(messages)
                if self.checkpointer is not None:
                    await self.checkpointer.save("intent", step, trace, env, pre_tool_messages, state,
                                                 action=resp)
                self.debug_hook("after_intent_checkpoint", step)

                if not resp.tool_calls:
                    # Dialogue envs (e.g. a simulated user) answer a text turn
                    # instead of ending the run. The reply is recorded twice on
                    # purpose: as a `user_turn` event (what the user said, and
                    # whether they ended the conversation -- this is what
                    # replay serves back instead of calling the user model) and
                    # as an ordinary `message` (so fork's prefix rebuild and
                    # LoopState.from_events need no special case).
                    on_text = getattr(env, "on_text_turn", None)
                    if on_text is None:
                        final_text = resp.text
                        break
                    turn = await on_text(resp.text or "")
                    trace.emit("user_turn", step, {"reply": turn.reply, "done": turn.done})
                    if turn.done:
                        final_text = resp.text
                        break
                    user_msg = {"role": "user", "content": turn.reply}
                    messages.append(user_msg)
                    trace.emit("message", step, {"message": user_msg})
                    if self.checkpointer is not None:
                        await self.checkpointer.save("step", step, trace, env, messages, state)
                    self.debug_hook("after_step_checkpoint", step)
                    step += 1
                    continue

                outcomes = await self._run_tool_batch(env, resp, step, pre_tool_messages, state, trace, rng)
                for idx, call in enumerate(resp.tool_calls):
                    outcome = outcomes[idx]
                    tool_msg = {"role": "tool", "tool_call_id": call.id,
                                "content": (outcome.content if outcome.ok
                                            else cfg.tool_error_format.format(
                                                cls=outcome.error_class, msg=outcome.content)),
                                "is_error": not outcome.ok}
                    messages.append(tool_msg)
                    trace.emit("message", step, {"message": tool_msg})

                if state.consecutive_invalid > cfg.max_invalid_args:
                    raise RunAbort("invalid_args", f"{state.consecutive_invalid} consecutive invalid calls")

                # Some envs end the episode from a tool (tau-bench's
                # transfer_to_human_agents). The step is complete either way.
                if getattr(env, "terminated", False):
                    if self.checkpointer is not None:
                        await self.checkpointer.save("step", step, trace, env, messages, state)
                    break

                # All of this step's tools durably applied: confirm the step so a
                # resume starts step+1 fresh instead of replaying it.
                if self.checkpointer is not None:
                    await self.checkpointer.save("step", step, trace, env, messages, state)
                self.debug_hook("after_step_checkpoint", step)
                step += 1
        except RunAbort as a:
            status, reason = a.status, a.reason

        end_hash = self.snapshots.put(env.snapshot())
        trace.emit("run_end", last_step, {"status": status, "reason": reason, "steps": last_step,
                                          "tokens": state.tokens, "env_hash": end_hash,
                                          "final_text": final_text})
        return RunResult(trace.run_id, status, reason, last_step, state.tokens, final_text, end_hash)

    @staticmethod
    def _normalize(resp: LLMResponse, step: int) -> LLMResponse:
        resp = resp.model_copy(deep=True)
        for i, c in enumerate(resp.tool_calls):
            if not c.id:
                c.id = f"tc_{step}_{i}"
        return resp

    def _backoff(self, attempt: int, rng: random.Random) -> float:
        cfg = self.cfg
        return min(cfg.backoff_cap, cfg.backoff_base * 2 ** attempt) * (0.5 + rng.random() / 2)

    async def _llm_step(self, messages, tools, step, trace, state, rng, forced) -> LLMResponse:
        cfg = self.cfg
        params = {"temperature": cfg.temperature, "max_tokens": cfg.max_output_tokens}

        if forced is not None:  # counterfactual action injected by fork
            resp = self._normalize(forced, step)
            trace.emit("llm_call", step, {
                "request_hash": request_hash(messages, tools, params),
                "response": resp.model_dump(), "patched": True,
                "provider": "patched", "latency_ms": 0})
            state.tokens += sum(resp.usage.get(k, 0) for k in ("input_tokens", "output_tokens"))
            return resp

        provider, pname = self.provider, "primary"
        attempt, repaired = 0, False
        while True:
            rh = request_hash(messages, tools, params)
            t = self.clock.perf()
            try:
                resp = await provider.complete(copy.deepcopy(messages), tools, params)
            except MalformedOutput as e:
                trace.emit("llm_call", step, {"request_hash": rh, "provider": pname, "error": {
                    "kind": "malformed", "class": "OUTPUT_MALFORMED", "retryable": False,
                    "message": str(e)}})
                if repaired:
                    raise RunAbort("malformed", str(e))
                repaired = True
                msg = {"role": "user", "content": REPAIR_PROMPT}
                messages.append(msg)
                trace.emit("message", step, {"message": msg})
                trace.emit("guard", step, {"guard": "repair_prompt"})
                continue
            except ProviderError as e:
                trace.emit("llm_call", step, {"request_hash": rh, "provider": pname, "error": {
                    "kind": "provider", "class": "LLM_API_ERROR", "retryable": e.retryable,
                    "status": e.status, "message": str(e)}})
                if e.retryable and attempt < cfg.max_llm_retries:
                    delay = self._backoff(attempt, rng)
                    attempt += 1
                    await self.sleep(delay)
                    continue
                if pname == "primary" and self.fallback is not None:
                    provider, pname, attempt = self.fallback, "fallback", 0
                    trace.emit("guard", step, {"guard": "fallback_provider"})
                    continue
                raise RunAbort("provider_error", str(e))
            resp = self._normalize(resp, step)
            state.tokens += sum(resp.usage.get(k, 0) for k in ("input_tokens", "output_tokens"))
            trace.emit("llm_call", step, {
                "request_hash": rh, "response": resp.model_dump(), "patched": False,
                "provider": pname, "latency_ms": round((self.clock.perf() - t) * 1000, 3)})
            return resp

    async def _run_tool_batch(self, env, resp: LLMResponse, step: int, pre_tool_messages: list[dict],
                              state: LoopState, trace: TraceWriter, rng) -> dict[int, ToolOutcome]:
        """Execute every tool call from one LLM turn and checkpoint as needed.
        All of `resp.tool_calls` were produced in the SAME turn, before the model
        saw any of their results, so none can depend on another's outcome --
        running them out of relative order is always safe.

        Non-mutating calls have no side effect on `env.applied` (the idempotency
        ledger), so re-running any of them after a crash is free. When there are
        2+ of them in this step, they run CONCURRENTLY: each one's slow part
        (`registry.execute`, inside `_tool_step`, via `asyncio.to_thread`) is
        dispatched to a thread pool at once with `asyncio.gather`, and ONE
        checkpoint covers the whole batch afterwards instead of one per call.

        Mutating calls always run SEQUENTIALLY, in their original relative
        order -- a later mutation can depend on an earlier one's effect on
        `env` (see `scripts/taubench_round1_wiki.patch` for a real example of
        this ordering mattering), and each keeps its own checkpoint exactly as
        before this batching existed.

        A step with 0 or 1 non-mutating calls (every existing demo/test
        scenario -- `tests/test_checkpoint.py`'s exhaustive crash-point test
        included) takes the SAME single-call-at-a-time path, with the SAME
        "after_tool_call" hook and one checkpoint per call, as before this
        method existed -- the concurrent path is new code that only ever
        triggers for a genuinely new situation (2+ non-mutating calls in one
        turn), so it can't change the checkpoint-hook sequence any existing
        test already pins down."""
        outcomes: dict[int, ToolOutcome] = {}
        indexed = list(enumerate(resp.tool_calls))
        readonly = [(i, c) for i, c in indexed if not self.registry.is_mutating(c.name)]
        mutating = [(i, c) for i, c in indexed if self.registry.is_mutating(c.name)]

        if len(readonly) >= 2:
            results = await asyncio.gather(
                *(self._tool_step(env, c, step, i, state, trace, rng, pre_tool_messages) for i, c in readonly))
            for (i, _c), outcome in zip(readonly, results):
                outcomes[i] = outcome
            self.debug_hook("after_readonly_batch", step)
            if self.checkpointer is not None:
                await self.checkpointer.save("tool", step, trace, env, pre_tool_messages, state,
                                             action=resp)
        else:
            for i, c in readonly:
                outcomes[i] = await self._tool_step(env, c, step, i, state, trace, rng, pre_tool_messages)
                if self.checkpointer is not None:
                    await self.checkpointer.save("tool", step, trace, env, pre_tool_messages, state,
                                                 action=resp)
                self.debug_hook("after_tool_call", step)

        for i, c in mutating:
            outcomes[i] = await self._tool_step(env, c, step, i, state, trace, rng, pre_tool_messages)
            if self.checkpointer is not None:
                await self.checkpointer.save("tool", step, trace, env, pre_tool_messages, state,
                                             action=resp)
            self.debug_hook("after_tool_call", step)

        return outcomes

    def _policy_block(self, env, call: ToolCall, step: int, history: list[dict] | None,
                      trace: TraceWriter) -> ToolOutcome | None:
        """Run the registry's policy checks on a call. Returns the blocking
        outcome (the call is then NOT executed), or None to proceed. Every
        failing check is recorded as a guard event, including observe-only
        checks and blocks waived because the check's per-episode budget ran
        out, so a trace shows exactly what the policy did."""
        policy = getattr(self.registry, "policy", None)
        if not policy or history is None:
            return None
        blocked = None
        for d in policy.evaluate(getattr(env, "data", {}) or {}, call.name, call.args, history,
                                 self.registry.is_mutating):
            trace.emit("guard", step, {"guard": "policy_check", "check": d.check_id,
                                       "action": d.action, "reason": d.reason, "call_id": call.id})
            if d.action == "block" and blocked is None:
                blocked = ToolOutcome(False, d.message, "POLICY_BLOCKED")
        return blocked

    async def _tool_step(self, env, call: ToolCall, step: int, idx: int, state: LoopState,
                         trace: TraceWriter, rng, history: list[dict] | None = None) -> ToolOutcome:
        cfg = self.cfg
        args_hash = sha(call.args)
        n = state.bump_call(call.name, args_hash)
        action_id = f"s{step}:c{idx}"
        t = self.clock.perf()
        base = {"call_id": call.id, "idx": idx, "name": call.name, "args": call.args,
                "args_hash": args_hash, "action_id": action_id}

        if n >= cfg.loop_kill_at:
            msg = f"identical call to {call.name} made {n} times"
            trace.emit("guard", step, {"guard": "loop_kill", "tool": call.name, "count": n})
            trace.emit("tool_call", step, {**base, "ok": False, "error_class": "LOOP_KILLED",
                                           "content": msg, "latency_ms": 0})
            raise RunAbort("loop", msg)

        if n >= cfg.loop_warn_at:
            trace.emit("guard", step, {"guard": "loop_warn", "tool": call.name, "count": n})
            outcome = ToolOutcome(
                False, f"you have called {call.name} with identical arguments {n} times. "
                       "Change your approach or give a final answer.", "LOOP_DETECTED")
        elif (blocked := self._policy_block(env, call, step, history, trace)) is not None:
            outcome = blocked
        else:
            attempt = 0
            while True:
                # Dispatched to a thread pool (not called directly) so that when
                # `_run_tool_batch` gathers several of these coroutines at once
                # for non-mutating calls, their actual work genuinely overlaps
                # instead of running one-at-a-time on the event loop thread.
                outcome = await asyncio.to_thread(self.registry.execute, env, call.name,
                                                  call.args, action_id)
                if outcome.transient and attempt < cfg.max_transient_tool_retries:
                    trace.emit("error", step, {"class": outcome.error_class, "attempt": attempt,
                                               "message": outcome.content, "call_id": call.id})
                    delay = self._backoff(attempt, rng)
                    attempt += 1
                    await self.sleep(delay)
                    continue
                break

        state.note_result(outcome.error_class)
        trace.emit("tool_call", step, {
            **base, "ok": outcome.ok, "error_class": outcome.error_class,
            "content": outcome.content,
            "latency_ms": round((self.clock.perf() - t) * 1000, 3)})
        trace.emit("env_snapshot", step, {"env_hash": self.snapshots.put(env.snapshot())})
        return outcome
