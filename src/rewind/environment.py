"""The environment port: what the agent runtime needs from an environment.

Before this module the contract lived in a docstring and in `getattr(env,
"...", default)` probes inside loop.py. It is now split into one required
protocol and three optional capabilities (interface segregation): an env
implements only the capabilities it actually has, and the loop asks with
`isinstance` against a named protocol instead of probing attribute names.

Required -- every env the loop runs:

    Environment      snapshot() for env hashing / fork points, and the
                     `applied` idempotency ledger that ToolRegistry uses to
                     keep mutating tools exactly-once across crashes.

Optional capabilities:

    DialogueEnv      answers an agent *text* turn (a simulated user) instead
                     of ending the run on it.
    TerminatingEnv   can end the episode from inside a tool (`terminated`),
                     e.g. tau-bench's transfer_to_human_agents.
    DataBackedEnv    exposes a read-only `data` dict that policy checks
                     evaluate against.

Construction is not part of the instance protocol: resume and fork rebuild
an env from a snapshot through an `EnvFactory` (usually a bound
`SomeEnv.restore`). Grading (`db_hash`, rewards) is a benchmark concern and
deliberately not part of what the kernel requires.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Environment(Protocol):
    @property
    def applied(self) -> dict[str, Any]:
        """action_id -> recorded ToolOutcome for every mutating call already
        applied. Must be part of `snapshot()` so it survives restore."""

    def snapshot(self) -> dict[str, Any]:
        """A JSON-able state that an `EnvFactory` can rebuild the env from."""


EnvFactory = Callable[[dict[str, Any]], Environment]


@dataclass
class TextTurn:
    """A dialogue env's answer to one agent text turn."""
    reply: str
    done: bool          # True: the user ended the conversation; the run ends


@runtime_checkable
class DialogueEnv(Protocol):
    async def on_text_turn(self, text: str) -> TextTurn: ...


@runtime_checkable
class TerminatingEnv(Protocol):
    terminated: bool


@runtime_checkable
class DataBackedEnv(Protocol):
    data: dict[str, Any]
