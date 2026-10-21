"""Real multi-turn tau-bench retail episodes inside rewind's own loop.

`taubench_adapter.py` wires tau-bench's real tools and data into rewind but
stops at the golden-script check, because a real episode also needs
tau-bench's LLM *user simulator*. Until now real runs therefore went through
tau-bench's own `run.py`, which leaves no rewind trace and no env snapshots,
so a real failure could not be replayed, forked or attributed. This module
closes that gap:

* `TauUserSim` is tau-bench's `LLMUserSimulationEnv` with its state made
  explicit: the same system prompt (taken from tau-bench itself, not
  copied), the same role convention, but the conversation lives in a plain
  list that is part of the env snapshot, and the model call is injected.
* `TauBenchDialogueEnv` adds that simulator to the real retail env. An agent
  text turn goes to the simulator (`on_text_turn`, called by the loop) and
  its reply comes back as the next user message; `###STOP###` or
  `transfer_to_human_agents` ends the episode -- the same two ways
  tau-bench's own `Env.step` ends one.
* `grade` is tau-bench's full reward: `r_actions` (data hash vs. a replay of
  the ground-truth actions) AND, for tasks that have them, `r_outputs`
  (required strings in what the agent said). The 37 test tasks the golden
  adapter had to skip are graded here.

Snapshots: the 1.3 MB retail DB is stored once per distinct content (by
hash, in a shared `SnapshotStore`) and referenced from the per-step env
snapshot, which otherwise holds only the dialogue. The DB is re-hashed only
after a mutating tool ran, so read-only steps cost almost nothing.

Fork semantics: restoring a step-k snapshot restores the simulator's full
conversation up to that point; the continuation talks to a *live*
simulator. Replay instead serves recorded replies (`recorded_user_llm`).
"""
from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from ..events import Event
from ..snapshot import SnapshotStore
from ..tools import Tool, ToolError, ToolRegistry
from .taubench_adapter import (
    MUTATING_TOOLS, RESPOND_ACTION_NAME, TERMINATE_TOOLS, TauBenchRetailEnv,
    _ensure_importable, _schema_to_args_model,
)

STOP_TOKEN = "###STOP###"
OPENING_LINE = "Hi! How can I help you today?"  # tau-bench's fixed first agent line

# messages (user-sim's own view) -> reply text
UserLLM = Callable[[list[dict]], Awaitable[str]]


# ---------------------------------------------------------------------------
# user simulator
# ---------------------------------------------------------------------------
def _user_system_prompt(instruction: str | None) -> str:
    _ensure_importable()
    from tau_bench.envs.user import LLMUserSimulationEnv
    # build_system_prompt never touches `self`; calling it unbound keeps the
    # prompt byte-identical to tau-bench's without constructing the class
    # (whose __init__ makes an LLM call).
    return LLMUserSimulationEnv.build_system_prompt(None, instruction)


class TauUserSim:
    """Role convention matches tau-bench: the *agent's* words are `user`
    messages here and the simulator's own replies are `assistant` messages."""

    def __init__(self, llm: UserLLM, messages: list[dict] | None = None) -> None:
        self.llm = llm
        self.messages: list[dict] = copy.deepcopy(messages) if messages else []

    async def reset(self, instruction: str) -> str:
        self.messages = [{"role": "system", "content": _user_system_prompt(instruction)},
                         {"role": "user", "content": OPENING_LINE}]
        return await self._next()

    async def step(self, agent_text: str) -> str:
        self.messages.append({"role": "user", "content": agent_text})
        return await self._next()

    async def _next(self) -> str:
        reply = await self.llm(copy.deepcopy(self.messages))
        reply = reply or ""
        self.messages.append({"role": "assistant", "content": reply})
        return reply


def litellm_user_llm(model: str = "deepseek-chat", provider: str = "deepseek",
                     retries: int = 3, backoff: float = 2.0) -> UserLLM:
    """The same call tau-bench's simulator makes (litellm, provider default
    temperature), off the event loop, with a few retries on transient
    errors -- tau-bench itself has none, and one flaky call should not cost
    a whole episode."""
    async def call(messages: list[dict]) -> str:
        from litellm import completion
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                res = await asyncio.to_thread(completion, model=model,
                                              custom_llm_provider=provider, messages=messages)
                return res.choices[0].message.content or ""
            except Exception as e:  # litellm wraps provider errors in many types
                last = e
                if attempt < retries:
                    await asyncio.sleep(backoff * 2 ** attempt)
        raise RuntimeError(f"user simulator failed after {retries + 1} attempts: {last}")
    return call


def recorded_user_llm(events: list[Event]) -> UserLLM:
    """Serves the replies recorded in a trace, in order -- for deterministic
    replay of a dialogue run. Raises if asked for more than were recorded."""
    replies = [e.data["reply"] for e in events if e.type == "user_turn"]
    it = iter(replies)

    async def call(_messages: list[dict]) -> str:
        try:
            return next(it)
        except StopIteration:
            raise RuntimeError("replay asked the user simulator for more turns than were recorded")
    return call


# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------
@dataclass
class TextTurn:
    reply: str
    done: bool


class TauBenchDialogueEnv(TauBenchRetailEnv):
    def __init__(self, store: SnapshotStore, user: TauUserSim, split: str, task_index: int,
                 data: dict[str, Any] | None = None) -> None:
        super().__init__(data)
        self.store = store
        self.user = user
        self.split = split
        self.task_index = task_index
        self.responses: list[str] = []   # every text turn the agent sent (for r_outputs)
        self.terminated = False          # set by a terminate tool
        self.user_stopped = False
        self._data_ref: str | None = None  # cached store key of self.data; None = dirty

    # -- episode -------------------------------------------------------------
    @property
    def task(self) -> Any:
        return load_task(self.split, self.task_index)

    async def start(self) -> str:
        """First user utterance (what `run()` should receive as the task)."""
        return await self.user.reset(self.task.instruction)

    async def on_text_turn(self, text: str) -> TextTurn:
        self.responses.append(text)
        reply = await self.user.step(text)
        done = STOP_TOKEN in reply
        self.user_stopped = done
        return TextTurn(reply=reply, done=done)

    def mark_dirty(self) -> None:
        self._data_ref = None

    # -- snapshot protocol ---------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        if self._data_ref is None:
            self._data_ref = self.store.put(self.data)
        return {"data_ref": self._data_ref, "_applied": copy.deepcopy(self._applied),
                "user": copy.deepcopy(self.user.messages), "responses": list(self.responses),
                "terminated": self.terminated, "user_stopped": self.user_stopped,
                "split": self.split, "task_index": self.task_index}

    @classmethod
    def restore(cls, state: dict[str, Any], *, store: SnapshotStore,
                user_llm: UserLLM) -> "TauBenchDialogueEnv":
        env = cls(store, TauUserSim(user_llm, state["user"]), state["split"],
                  state["task_index"], data=store.get(state["data_ref"]))
        env._data_ref = state["data_ref"]
        env._applied = copy.deepcopy(state["_applied"])
        env.responses = list(state["responses"])
        env.terminated = state["terminated"]
        env.user_stopped = state["user_stopped"]
        return env


# ---------------------------------------------------------------------------
# tools (faithful to tau-bench's ToolCallingAgent view)
# ---------------------------------------------------------------------------
def _faithful_tool_fn(tool_cls, name: str) -> Callable[[TauBenchDialogueEnv, BaseModel], Any]:
    mutating = name in MUTATING_TOOLS
    terminates = name in TERMINATE_TOOLS

    def fn(env: TauBenchDialogueEnv, args: BaseModel) -> Any:
        kwargs = args.model_dump(exclude_none=True)
        if mutating:
            env.mark_dirty()
        try:
            result = tool_cls.invoke(data=env.data, **kwargs)
        except Exception as e:  # tau-bench's Env.step catch-all
            raise ToolError(str(e)) from e
        if terminates:
            env.terminated = True
        if isinstance(result, str) and result.startswith("Error:"):
            raise ToolError(result[len("Error:"):].strip())
        return result
    return fn


def build_dialogue_registry() -> ToolRegistry:
    """The 16 retail tools as tau-bench's own agent sees them: tau-bench's
    published JSON schemas (not regenerated), and results passed through as
    the exact strings the tools return."""
    _ensure_importable()
    from tau_bench.envs.retail.tools import ALL_TOOLS

    reg = ToolRegistry()
    for tool_cls in ALL_TOOLS:
        info = tool_cls.get_info()["function"]
        name = info["name"]
        reg.register(Tool(name=name, description=info["description"],
                          args_model=_schema_to_args_model(f"{name}_args", info["parameters"]),
                          fn=_faithful_tool_fn(tool_cls, name), mutating=name in MUTATING_TOOLS,
                          input_schema=info["parameters"], raw_text_result=True))
    return reg


def wiki() -> str:
    _ensure_importable()
    from tau_bench.envs.retail.wiki import WIKI
    return WIKI


def faithful_config(**overrides: Any):
    """AgentConfig matching tau-bench's ToolCallingAgent as closely as rewind
    allows: wiki as the whole system prompt, 30 model calls per episode,
    one tool call per turn, "Error: ..." tool failures, and rewind's own
    loop-kill / invalid-args aborts disabled (tau-bench has neither)."""
    from ..loop import AgentConfig
    cfg = dict(system_prompt=wiki(), max_steps=30, max_tokens=10_000_000, wall_seconds=1800.0,
               max_output_tokens=4096, temperature=0.0, max_invalid_args=10_000,
               loop_warn_at=10_000, loop_kill_at=10_000, single_tool_call=True,
               tool_error_format="Error: {msg}")
    cfg.update(overrides)
    return AgentConfig(**cfg)


# ---------------------------------------------------------------------------
# tasks + grading
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _tasks(split: str) -> tuple:
    _ensure_importable()
    if split == "train":
        from tau_bench.envs.retail.tasks_train import TASKS_TRAIN as tasks
    elif split == "dev":
        from tau_bench.envs.retail.tasks_dev import TASKS_DEV as tasks
    elif split == "test":
        from tau_bench.envs.retail.tasks_test import TASKS_TEST as tasks
    else:
        raise ValueError(f"unknown split {split!r}")
    return tuple(tasks)


def load_task(split: str, index: int) -> Any:
    return _tasks(split)[index]


def n_tasks(split: str) -> int:
    return len(_tasks(split))


@lru_cache(maxsize=None)
def _gt_hash(split: str, index: int) -> str:
    """Hash of the DB after replaying the task's ground-truth actions on fresh
    data -- tau-bench's `calculate_reward` recomputes this every episode."""
    from tau_bench.envs.base import consistent_hash, to_hashable
    from tau_bench.envs.retail.data import load_data
    from tau_bench.envs.retail.tools import ALL_TOOLS
    tool_map = {t.get_info()["function"]["name"]: t for t in ALL_TOOLS}
    fresh = load_data()
    for action in load_task(split, index).actions:
        if action.name == RESPOND_ACTION_NAME or action.name in TERMINATE_TOOLS:
            continue
        tool_map[action.name].invoke(data=fresh, **action.kwargs)
    return consistent_hash(to_hashable(fresh))


class Reward(BaseModel):
    reward: float
    r_actions: bool
    r_outputs: float | None = None
    outputs: dict[str, bool] | None = None


def grade(env: TauBenchDialogueEnv) -> Reward:
    """tau-bench's full reward, same rules as `Env.calculate_reward`: data
    hash must match the ground-truth replay, and every required output
    string must appear (case-insensitive, commas removed) in some agent text
    turn."""
    task = env.task
    r_actions = env.db_hash() == _gt_hash(env.split, env.task_index)
    reward = 1.0 if r_actions else 0.0
    if not task.outputs:
        return Reward(reward=reward, r_actions=r_actions)
    said = [r.lower().replace(",", "") for r in env.responses]
    outputs = {o: any(o.lower() in s for s in said) for o in task.outputs}
    r_outputs = 1.0 if all(outputs.values()) else 0.0
    if r_outputs == 0.0:
        reward = 0.0
    return Reward(reward=reward, r_actions=r_actions, r_outputs=r_outputs, outputs=outputs)


def result_record(env: TauBenchDialogueEnv, rr: Any, trial: int, trace_path: str | None,
                  events: list[Event]) -> dict:
    """One row in the same shape tau-bench's run.py writes (`task_id`,
    `reward`, `info`, `traj`, `trial`), so the existing failure-classification
    and multi-trial comparison scripts read rewind results unchanged. `traj`
    is the agent-side conversation in OpenAI chat format, as tau-bench
    stores it."""
    from ..providers import to_openai_messages
    r = grade(env)
    task = env.task
    # fork traces carry the parent's prefix as `inherited` events; the
    # conversation is the whole thing either way
    traj = to_openai_messages([e.data["message"] for e in events if e.type == "message"])
    return {
        "task_id": env.task_index, "reward": r.reward, "trial": trial,
        "info": {"reward_info": {"reward": r.reward,
                                 "info": ({"r_outputs": r.r_outputs, "outputs": r.outputs}
                                          if r.outputs is not None else {"r_actions": r.r_actions}),
                                 "actions": [a.model_dump() for a in task.actions
                                             if a.name != RESPOND_ACTION_NAME]},
                 "task": task.model_dump(),
                 "rewind": {"run_id": rr.run_id, "status": rr.status, "reason": rr.reason,
                            "steps": rr.steps, "tokens": rr.tokens, "trace": trace_path,
                            "terminated_by_tool": env.terminated, "user_stopped": env.user_stopped,
                            "r_actions": r.r_actions}},
        "traj": traj,
    }


def dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# episode helpers: run / fork / replay one real task
# ---------------------------------------------------------------------------
@dataclass
class Episode:
    result: Any                    # loop.RunResult
    env: TauBenchDialogueEnv
    events: list[Event]
    reward: Reward


async def run_episode(split: str, index: int, *, provider: Any, user_llm: UserLLM,
                      store: SnapshotStore, cfg: Any = None, registry: ToolRegistry | None = None,
                      run_id: str | None = None, trace_path: str | None = None) -> Episode:
    from ..loop import AgentRunner
    from ..trace import TraceWriter
    cfg = cfg or faithful_config()
    registry = registry or build_dialogue_registry()
    env = TauBenchDialogueEnv(store, TauUserSim(user_llm), split, index)
    first = await env.start()
    runner = AgentRunner(provider, registry, cfg, store)
    with TraceWriter(run_id or f"tau-{split}-{index}", path=trace_path) as trace:
        rr = await runner.run(env, first, trace)
        events = list(trace.events)
    return Episode(rr, env, events, grade(env))


async def fork_episode(base_events: list[Event], at_step: int, *, provider: Any, user_llm: UserLLM,
                       store: SnapshotStore, cfg: Any = None, registry: ToolRegistry | None = None,
                       patch: Any = None, run_id: str | None = None,
                       trace_path: str | None = None) -> Episode:
    """Restore env + simulator at `at_step` of a recorded episode and continue
    live. With `patch=None` this is the *null fork* used to calibrate how much
    a continuation varies on its own; with a `ForkPatch` it is a
    counterfactual. `cfg` defaults to the parent's recorded config, so a
    context edit is expressed by passing a different cfg (system prompt)."""
    from ..fork import fork_run
    from ..loop import AgentConfig, AgentRunner
    from ..trace import TraceWriter
    if cfg is None:
        cfg = AgentConfig(**base_events[0].data["cfg"])
    registry = registry or build_dialogue_registry()
    holder: dict[str, TauBenchDialogueEnv] = {}

    def env_factory(state: dict) -> TauBenchDialogueEnv:
        holder["env"] = TauBenchDialogueEnv.restore(state, store=store, user_llm=user_llm)
        return holder["env"]

    runner = AgentRunner(provider, registry, cfg, store)
    rid = run_id or f"{base_events[0].run_id}~fork{at_step}"
    with TraceWriter(rid, path=trace_path) as trace:
        rr = await fork_run(base_events, store, at_step=at_step, runner=runner,
                            env_factory=env_factory, trace=trace, patch=patch)
        events = list(trace.events)
    env = holder["env"]
    return Episode(rr, env, events, grade(env))


async def replay_episode(events: list[Event], store: SnapshotStore,
                         registry: ToolRegistry | None = None):
    """Deterministic replay of a recorded dialogue episode: recorded model
    responses AND recorded user replies, no API calls."""
    from ..replay import replay_run
    registry = registry or build_dialogue_registry()
    user_llm = recorded_user_llm(events)
    return await replay_run(events, store, registry,
                            lambda s: TauBenchDialogueEnv.restore(s, store=store, user_llm=user_llm))
