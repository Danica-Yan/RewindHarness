"""The benchmark port: what evaluation and self-improvement need from a task suite.

`experiment.py` (run / fork / replay one episode) and everything in
`evolve/` are written against this Protocol only. A benchmark is passed in
explicitly, the same way a provider or a SnapshotStore is, so the
self-improvement loop has no idea which benchmark it is improving on.
`env/taubench_benchmark.py` is the tau-bench retail implementation.

What a benchmark supplies, grouped by who uses it:

  tasks      `task`, `n_tasks`, `write_tools` -- the goal text and gold
             actions of task i in a split, and which tools change state
  agent      `policy_text`, `agent_config`, `build_registry` -- the base
             system prompt, the loop config that matches the benchmark's own
             agent, and its tool surface (evolve layers edits on top)
  episodes   `start_episode`, `restore_env`, `recorded_user` -- a fresh env
             plus its opening message; an env rebuilt from a snapshot (fork,
             resume); and a counterparty that serves recorded replies
             (replay without calling the live user model)
  results    `grade`, `result_record`, `failure_signature` -- the reward,
             one result row in the benchmark's own format, and a coarse
             signature of why a failed row failed (what evolve clusters)

`user` is the benchmark's counterparty handle (tau-bench: the user
simulator's LLM). It is opaque here: experiment.py and evolve only pass it
through.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from .environment import Environment
from .events import Event
from .snapshot import SnapshotStore
from .tools import ToolRegistry

if TYPE_CHECKING:
    from .loop import AgentConfig, RunResult


class GoldAction(Protocol):
    name: str
    kwargs: dict[str, Any]


class TaskSpec(Protocol):
    instruction: str
    actions: Sequence[GoldAction]


class Grade(Protocol):
    reward: float


class FailureSignature(BaseModel):
    """Why a failed episode failed, coarsely: what evolve clusters on."""
    category: str
    tool: str | None = None
    detail: str | None = None

    @property
    def key(self) -> str:
        return f"{self.category}|{self.tool or '-'}|{self.detail or '-'}"

    @property
    def coarse_key(self) -> str:
        return f"{self.category}|{self.tool or '-'}"


@runtime_checkable
class Benchmark(Protocol):
    name: str

    # -- tasks -------------------------------------------------------------------
    @property
    def write_tools(self) -> frozenset[str]: ...

    def task(self, split: str, index: int) -> TaskSpec: ...

    def n_tasks(self, split: str) -> int: ...

    # -- agent -------------------------------------------------------------------
    def policy_text(self) -> str: ...

    def agent_config(self, **overrides: Any) -> AgentConfig: ...

    def build_registry(self) -> ToolRegistry: ...

    # -- episodes ----------------------------------------------------------------
    async def start_episode(self, split: str, index: int, *, store: SnapshotStore,
                            user: Any) -> tuple[Environment, str]:
        """A fresh env for task `index` and the opening message the agent
        receives as its task."""

    def restore_env(self, state: dict[str, Any], *, store: SnapshotStore, user: Any) -> Environment: ...

    def recorded_user(self, events: list[Event]) -> Any:
        """A `user` that replays the counterparty turns recorded in `events`."""

    # -- results -----------------------------------------------------------------
    def grade(self, env: Environment) -> Grade: ...

    def result_record(self, env: Environment, result: RunResult, trial: int,
                      trace_path: str | None, events: list[Event]) -> dict: ...

    def failure_signature(self, row: dict, split: str) -> FailureSignature | None:
        """None for a row that did not fail (or cannot be classified)."""
