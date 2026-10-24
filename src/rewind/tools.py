"""Tool registry: pydantic-validated arguments, structured errors, idempotent
mutations.

Contract for tool functions: a tool that raises must NOT have mutated the env
(validate first, mutate last), because failed calls are not recorded as applied.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Protocol

from pydantic import BaseModel, ValidationError

from .util import canonical_json


class ToolError(Exception):
    """Permanent tool failure (bad state, not found, policy)."""


class TransientToolError(Exception):
    """Retryable failure (timeout, 5xx from a backing service)."""


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Any, Any], Any]
    mutating: bool = False
    # Optional overrides for wrapping a third-party tool set verbatim: publish
    # its own JSON schema instead of the pydantic-generated one, and pass a
    # string result through as-is instead of JSON-encoding it.
    input_schema: dict | None = None
    raw_text_result: bool = False


@dataclass
class ToolOutcome:
    ok: bool
    content: str
    error_class: str | None = None
    transient: bool = False


def _fmt_validation(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "<root>"
        parts.append(f"{loc}: {err['msg']}")
    return "; ".join(parts)


class GateDecision(Protocol):
    check_id: str
    action: str          # "block" stops the call; anything else is recorded only
    reason: str
    message: str         # what the agent sees as the tool error when blocked


class ToolGate(Protocol):
    """Pre-execution check on a tool call. Declared here, at the tool layer,
    so the registry and the loop depend on this contract rather than on
    rewind.policy; `PolicySet` satisfies it structurally."""

    def evaluate(self, data: dict, tool: str, args: dict, history: list[dict],
                 is_write: Callable[[str], bool]) -> list[GateDecision]: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # Optional gate the agent loop runs before executing a call (in
        # practice a rewind.policy.PolicySet). Kept on the registry so every
        # path that builds an agent -- run, fork, replay, the Evolve gate --
        # carries it along with the tool surface it guards.
        self.policy: ToolGate | None = None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def tool(self, name: str, description: str, args_model: type[BaseModel],
             mutating: bool = False):
        def deco(fn):
            self.register(Tool(name, description, args_model, fn, mutating))
            return fn
        return deco

    def schemas(self) -> list[dict]:
        """Sorted by name so the request hash is stable."""
        return [
            {"name": t.name, "description": t.description,
             "input_schema": t.input_schema if t.input_schema is not None
             else t.args_model.model_json_schema()}
            for t in sorted(self._tools.values(), key=lambda t: t.name)
        ]

    def is_mutating(self, name: str) -> bool:
        t = self._tools.get(name)
        return bool(t and t.mutating)

    def execute(self, env: Any, name: str, raw_args: Any, action_id: str) -> ToolOutcome:
        tool = self._tools.get(name)
        if tool is None:
            avail = ", ".join(sorted(self._tools))
            return ToolOutcome(False, f"unknown tool '{name}'. Available tools: {avail}",
                               "HALLUCINATED_TOOL")
        try:
            args = tool.args_model.model_validate(raw_args)
        except ValidationError as e:
            return ToolOutcome(False, "invalid arguments: " + _fmt_validation(e),
                               "TOOL_ARG_INVALID")
        # Idempotency: a re-executed action (e.g. a worker resumed after a crash)
        # returns the recorded outcome instead of mutating twice.
        if tool.mutating and action_id in env.applied:
            return ToolOutcome(**env.applied[action_id])
        try:
            result = tool.fn(env, args)
        except TransientToolError as e:
            return ToolOutcome(False, str(e), "TOOL_TRANSIENT_ERROR", transient=True)
        except ToolError as e:
            return ToolOutcome(False, str(e), "TOOL_RUNTIME_ERROR")
        content = result if (tool.raw_text_result and isinstance(result, str)) else canonical_json(result)
        outcome = ToolOutcome(True, content)
        if tool.mutating:
            env.applied[action_id] = asdict(outcome)
        return outcome
