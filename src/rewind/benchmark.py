"""The benchmark port: what evaluation and self-improvement need from a task suite.

`experiment.py` (run / fork / replay one episode) and everything in
`evolve/` are written against this Protocol only. A benchmark is passed in
explicitly, the same way a provider or a SnapshotStore is, so the
self-improvement loop has no idea which benchmark it is improving on.
`env/taubench_benchmark.py` is the tau-bench retail implementation.

What a benchmark supplies, grouped by who uses it:

  tasks      `task`, `gold_calls`, `n_tasks`, `write_tools` -- the goal text
             of task i in a split, its gold solution as tool calls, and
             which tools change state
  agent      `policy_text`, `agent_config`, `build_registry` -- the base
             system prompt, the loop config that matches the benchmark's own
             agent, and its tool surface (evolve layers edits on top)
  episodes   `start_episode`, `restore_env`, `recorded_user` -- a fresh env
             plus its opening message; an env rebuilt from a snapshot (fork,
             resume); and a counterparty that serves recorded replies
             (replay without calling the live user model)
  results    `grade`, `result_record`, `failure_signature` -- the reward,
             one result row (a `ResultRow`, see below), and a coarse
             signature of why a failed row failed (what evolve clusters)

`user` is the benchmark's counterparty handle (tau-bench: the user
simulator's LLM). It is opaque here: experiment.py and evolve only pass it
through.

Result rows are the other half of the contract. Evolve never sees an env or
a trace object for an episode it did not just run; it reads results files,
one `ResultRow` per (task, trial). A row may carry any extra fields the
benchmark likes (tau-bench rows are in its own run.py format); evolve reads
only these:

  task_id, trial, reward     which episode, and how it scored (pass = 1.0)
  traj                       the agent-side conversation in OpenAI chat
                             format: {"role", "content"} messages, and
                             assistant `tool_calls` of the form
                             {"function": {"name", "arguments": <JSON str>}}
  info.rewind.trace          path of the rewind trace (null forks and gate
                             forks start from it; rows without one are
                             skipped wherever a fork is needed)
  info.rewind.ctx_version    which context version produced the row; evolve
                             writes it (`set_ctx_version`), a benchmark need not

`check_row` validates that shape and `run_batch` rejects a benchmark row that
fails it; `row_trace` / `row_ctx_version` are the only way evolve reads the
nested fields, so the contract lives in one place.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable

from pydantic import BaseModel

from .environment import Environment
from .events import Event
from .snapshot import SnapshotStore
from .tools import ToolRegistry

if TYPE_CHECKING:
    from .loop import AgentConfig, RunResult


class GoldAction(Protocol):
    """One tool call of a task's gold solution."""
    name: str
    kwargs: dict[str, Any]


class TaskSpec(Protocol):
    instruction: str


# ---------------------------------------------------------------------------
# result rows
# ---------------------------------------------------------------------------
class RewindInfo(TypedDict, total=False):
    trace: str | None
    ctx_version: int


class RowInfo(TypedDict, total=False):
    rewind: RewindInfo


class ResultRow(TypedDict):
    task_id: int
    trial: int
    reward: float
    traj: list[dict[str, Any]]
    info: RowInfo


def check_row(row: Any) -> list[str]:
    """Every way `row` breaks the ResultRow contract (empty if it holds).
    Cheap: checks keys, types and the shape of `traj`, not its content."""
    if not isinstance(row, dict):
        return [f"row is {type(row).__name__}, not a dict"]
    out = []
    for key, types in (("task_id", (int,)), ("trial", (int,)), ("reward", (int, float))):
        if key not in row:
            out.append(f"missing {key!r}")
        elif isinstance(row[key], bool) or not isinstance(row[key], types):
            out.append(f"{key!r} is {type(row[key]).__name__}")
    traj = row.get("traj")
    if not isinstance(traj, list):
        out.append("missing 'traj' (list of OpenAI chat messages)")
    else:
        for i, m in enumerate(traj):
            if not isinstance(m, dict) or "role" not in m:
                out.append(f"traj[{i}] is not a chat message with a 'role'")
                break
            for c in m.get("tool_calls") or []:
                fn = c.get("function") if isinstance(c, dict) else None
                if not isinstance(fn, dict) or "name" not in fn:
                    out.append(f"traj[{i}] has a tool call without function.name")
                    break
    info = row.get("info")
    if not isinstance(info, dict):
        out.append("missing 'info' (dict)")
    else:
        rw = info.get("rewind")
        if rw is not None and not isinstance(rw, dict):
            out.append("info.rewind is not a dict")
        elif rw:
            if rw.get("trace") is not None and not isinstance(rw["trace"], str):
                out.append("info.rewind.trace is not a path string")
            if "ctx_version" in rw and not isinstance(rw["ctx_version"], int):
                out.append("info.rewind.ctx_version is not an int")
    return out


class RowContractError(ValueError):
    pass


def validate_row(row: Any, where: str = "") -> ResultRow:
    problems = check_row(row)
    if problems:
        raise RowContractError(f"result row breaks the ResultRow contract{' (' + where + ')' if where else ''}: "
                               + "; ".join(problems))
    return row


def row_trace(row: ResultRow) -> str | None:
    return ((row.get("info") or {}).get("rewind") or {}).get("trace")


def row_ctx_version(row: ResultRow) -> int:
    return ((row.get("info") or {}).get("rewind") or {}).get("ctx_version", 0)


def set_ctx_version(row: ResultRow, version: int) -> None:
    row.setdefault("info", {}).setdefault("rewind", {})["ctx_version"] = version


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

    def gold_calls(self, split: str, index: int) -> Sequence[GoldAction]:
        """The gold solution as tool calls only -- no benchmark-specific
        pseudo-actions (tau-bench's `respond`)."""

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
                      trace_path: str | None, events: list[Event]) -> ResultRow: ...

    def failure_signature(self, row: ResultRow, split: str) -> FailureSignature | None:
        """None for a row that did not fail (or cannot be classified)."""
