"""Adapter wiring rewind's harness to the *real* tau-bench retail domain.

Optional dependency, not on PyPI::

    pip install "tau_bench @ git+https://github.com/sierra-research/tau-bench.git"

(or clone it yourself and point `REWIND_TAUBENCH_SRC` at the checkout -- see
`_ensure_importable`). The import is lazy, same pattern as the optional
`anthropic` import in providers.py: the rest of rewind works without this
installed.

What this module gives you, and what it deliberately does NOT give you
-------------------------------------------------------------------------
tau-bench's retail domain ships as real, unmodified data: 500 users, 1000
orders, 50 products (`tau_bench.envs.retail.data`), 16 real tools with real
validation/mutation logic (`tau_bench.envs.retail.tools`), and a real
hash-based reward function (`tau_bench.envs.base.calculate_reward`, reused
here via `tau_bench.envs.base.to_hashable`/`consistent_hash` so grading is
byte-for-byte the same convention tau-bench itself uses).

What tau-bench tasks ALSO need, that this adapter does not provide: a live LLM
for (a) the agent and (b) tau-bench's own user-simulator, which plays the
human side of a multi-turn conversation (`tau_bench.envs.user`, driven by
`litellm`). Without either, there is no way to produce a real baseline
number -- that requires an actual model attempt, which this offline adapter
does not make (same constraint documented for every other "real LLM"
step in this project: W5's AnthropicProvider, W8's flywheel).

So this adapter is scoped to what IS honestly achievable without an LLM:

1. `TauBenchRetailEnv` + `build_registry()` -- real env/tool integration.
   Every tool call here runs tau-bench's own, unmodified `Tool.invoke()`
   against tau-bench's own data. This is not reimplemented or mocked.
2. `golden_script(task)` + `replay_grade(env, task)` -- a *golden-trajectory
   sanity check*: replay the task's own recorded ground-truth tool calls
   (skipping only `respond`, which never touches `data`, and
   `transfer_to_human_agents`, tau-bench's own "terminate_tools" exclusion)
   through a `ScriptedProvider`, then grade with tau-bench's real reward
   convention. This is NOT a model baseline -- the trajectory IS the
   answer key, so a correct adapter should score ~100% by construction.
   Its value is proving the integration is wired correctly (real tools,
   real data, real reward semantics) before ever spending an API call on
   it, exactly the same role `tests/test_checkpoint.py` plays for the
   checkpoint protocol.
3. Scoped to the `train`/`dev` splits for the headline number, because
   those tasks grade purely on the data-mutation hash (`r_actions`); the
   37/115 `test`-split tasks that also require matching spoken `outputs`
   need the dialogue layer this adapter does not model (see
   `UNSUPPORTED_TASK_REASONS` below) and are reported separately, honestly,
   not folded into one blended number.

A real baseline (pass^1 of an actual model, with a real or human user
simulator) is the next step once an API key is available -- see
`scripts/taubench_baseline.py` and the README/project-plan "next steps".
"""
from __future__ import annotations

import copy
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model

from ..providers import LLMResponse, ScriptedProvider, ToolCall
from ..tools import ToolError, ToolRegistry

RESPOND_ACTION_NAME = "respond"
TERMINATE_TOOLS = {"transfer_to_human_agents"}  # tau-bench's own exclusion list

# tau-bench doesn't expose "does this tool mutate `data`" as metadata, so this
# is an explicit, auditable allowlist (checked against the 7 tools that write
# to `data` in tau_bench/envs/retail/tools/*.py as of the version pinned
# above -- everything else in ALL_TOOLS is read-only or a dialogue action).
MUTATING_TOOLS = {
    "cancel_pending_order", "exchange_delivered_order_items",
    "modify_pending_order_address", "modify_pending_order_items",
    "modify_pending_order_payment", "modify_user_address",
    "return_delivered_order_items",
}

_JSON_TYPE_MAP: dict[str, Any] = {"string": str, "integer": int, "number": float, "boolean": bool}


def _ensure_importable() -> None:
    try:
        import tau_bench  # noqa: F401
        return
    except ImportError:
        pass
    src = os.environ.get("REWIND_TAUBENCH_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)
    try:
        import tau_bench  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "tau_bench is not installed. It is not on PyPI, so pip can't find it by "
            "name. Install the real benchmark with:\n"
            '  pip install "tau_bench @ git+https://github.com/sierra-research/tau-bench.git"\n'
            "or clone https://github.com/sierra-research/tau-bench and set "
            "REWIND_TAUBENCH_SRC=/path/to/that/clone."
        ) from e


def _schema_to_args_model(name: str, schema: dict) -> type[BaseModel]:
    """tau-bench tools describe arguments as an OpenAI-function-calling JSON
    schema (`Tool.get_info()["function"]["parameters"]`), not a pydantic
    model. Build one dynamically so `ToolRegistry` (which requires a
    pydantic `args_model`) can validate against the *same* schema tau-bench
    itself publishes -- rather than hand-duplicating 16 argument lists."""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for pname, pschema in props.items():
        t = pschema.get("type")
        if t == "array":
            item_t = _JSON_TYPE_MAP.get((pschema.get("items") or {}).get("type", ""), str)
            py_t = list[item_t]
        else:
            py_t = _JSON_TYPE_MAP.get(t, Any)
        fields[pname] = (py_t, ... if pname in required else None)
    return create_model(name, __config__=ConfigDict(extra="forbid"), **fields)


def _wrap_tool_fn(tool_cls) -> Any:
    """tau-bench tools return a plain string: JSON-encoded on success, or a
    string starting with "Error: " on failure (not an exception) -- base.py
    also catches any *actual* exception a tool raises and turns it into the
    same "Error: ..." shape. Translate both into rewind's ToolError
    convention so the rest of the harness (retry/classification/etc.) sees
    the same signal it sees from the toy env."""
    def fn(env: "TauBenchRetailEnv", args: BaseModel) -> Any:
        kwargs = args.model_dump()
        try:
            result = tool_cls.invoke(data=env.data, **kwargs)
        except Exception as e:  # matches tau_bench.envs.base.Env.step's own catch-all
            raise ToolError(str(e)) from e
        if isinstance(result, str):
            if result.startswith("Error:"):
                raise ToolError(result[len("Error:"):].strip())
            try:
                return json.loads(result)
            except (json.JSONDecodeError, ValueError):
                return result
        return result
    return fn


def build_registry() -> ToolRegistry:
    """All 16 real retail-domain tools, wired to real `invoke()` logic."""
    _ensure_importable()
    from tau_bench.envs.retail.tools import ALL_TOOLS

    reg = ToolRegistry()
    for tool_cls in ALL_TOOLS:
        info = tool_cls.get_info()["function"]
        name, description = info["name"], info["description"]
        args_model = _schema_to_args_model(f"{name}_args", info["parameters"])
        reg.tool(name, description, args_model, mutating=name in MUTATING_TOOLS)(_wrap_tool_fn(tool_cls))
    return reg


class TauBenchRetailEnv:
    """Wraps tau-bench's own in-memory retail DB (`data`: orders/users/
    products) behind rewind's env protocol (`snapshot`/`restore`/`applied`/
    `db_hash`). `self.data` is passed to tau-bench's own tools exactly as
    tau-bench's own `Env.step()` would pass it -- nothing here reimplements
    tool behavior, it only adds the bookkeeping rewind's harness needs."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        _ensure_importable()
        from tau_bench.envs.retail.data import load_data
        self.data: dict[str, Any] = copy.deepcopy(data) if data is not None else load_data()
        self._applied: dict[str, Any] = {}

    @property
    def applied(self) -> dict[str, Any]:
        return self._applied

    def snapshot(self) -> dict[str, Any]:
        return {"data": copy.deepcopy(self.data), "_applied": copy.deepcopy(self._applied)}

    @classmethod
    def restore(cls, state: dict[str, Any]) -> "TauBenchRetailEnv":
        env = cls(state["data"])
        env._applied = copy.deepcopy(state["_applied"])
        return env

    def db_hash(self) -> str:
        """Same hash tau-bench's own `calculate_reward` uses, so a reward
        computed here is directly comparable to one tau-bench would report."""
        _ensure_importable()
        from tau_bench.envs.base import consistent_hash, to_hashable
        return consistent_hash(to_hashable(self.data))


@dataclass
class TaskResult:
    task_index: int
    split: str
    supported: bool
    reason: str | None
    passed: bool | None
    n_actions: int


def load_tasks(split: str) -> list[Any]:
    """`split` is "train", "dev", or "test" -- tau-bench's own names."""
    _ensure_importable()
    if split == "train":
        from tau_bench.envs.retail.tasks_train import TASKS_TRAIN
        return list(TASKS_TRAIN)
    if split == "dev":
        from tau_bench.envs.retail.tasks_dev import TASKS_DEV
        return list(TASKS_DEV)
    if split == "test":
        from tau_bench.envs.retail.tasks_test import TASKS_TEST
        return list(TASKS_TEST)
    raise ValueError(f"unknown split {split!r}")


def unsupported_reason(task: Any) -> str | None:
    """Why this adapter can't honestly grade a task without an LLM, or None
    if it can. A task needs the dialogue layer (`respond`'s *content*, not
    just that it happened) whenever it has a non-empty `outputs` list --
    those strings are checked against what the agent actually said over the
    conversation, which this adapter does not simulate (see module
    docstring, point 3)."""
    if task.outputs:
        return "task.outputs is non-empty: grading needs matching spoken text, " \
               "which requires the real multi-turn dialogue this adapter doesn't simulate"
    return None


def golden_script(task: Any) -> list[LLMResponse]:
    """Turn tau-bench's own recorded ground-truth `actions` into a
    ScriptedProvider script: one LLM turn per action, in the recorded
    order, skipping `respond` (never mutates `data`) and
    `transfer_to_human_agents` (tau-bench's own terminate_tools exclusion
    from the ground-truth replay in `calculate_reward`) -- same exclusion
    tau-bench's own reward function applies, not an extra simplification on
    top of it. A final no-tool-call turn ends the rewind run."""
    script: list[LLMResponse] = []
    for i, action in enumerate(task.actions):
        if action.name == RESPOND_ACTION_NAME or action.name in TERMINATE_TOOLS:
            continue
        script.append(LLMResponse(tool_calls=[ToolCall(id=f"gt_{i}", name=action.name, args=action.kwargs)]))
    script.append(LLMResponse(text="Done."))
    return script


def golden_provider(task: Any) -> ScriptedProvider:
    return ScriptedProvider(golden_script(task))


def replay_grade(env: TauBenchRetailEnv, task: Any) -> bool:
    """tau-bench's own `r_actions` reward component: replay the task's
    ground-truth actions on a FRESH copy of the data, hash it, and compare
    to the hash of whatever state the agent (here: the golden script)
    actually produced. Mirrors `tau_bench.envs.base.Env.calculate_reward`
    exactly, except it skips `respond` actions during the replay -- in the
    real implementation those call into the LLM user simulator too (just to
    produce a "done" signal that can't move the hash anyway, since `respond`
    never touches `data`), which would require an LLM API key
    for no effect on the number being computed."""
    _ensure_importable()
    from tau_bench.envs.base import consistent_hash, to_hashable
    from tau_bench.envs.retail.data import load_data
    from tau_bench.envs.retail.tools import ALL_TOOLS
    tool_map = {t.get_info()["function"]["name"]: t for t in ALL_TOOLS}

    fresh = load_data()
    for action in task.actions:
        if action.name == RESPOND_ACTION_NAME or action.name in TERMINATE_TOOLS:
            continue
        tool_map[action.name].invoke(data=fresh, **action.kwargs)
    return env.db_hash() == consistent_hash(to_hashable(fresh))
