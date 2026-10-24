"""A tiny offline retail environment used for tests and the demo.

`env` is a package: this file is the toy env unchanged from W1-W7; the real
tau-bench adapter (W1 followup, see taubench_adapter.py) lives alongside it.
What the loop needs from an env is the `Environment` protocol in
rewind/environment.py (`snapshot()` + the `applied` idempotency ledger),
plus whichever optional capabilities an env has; `restore()` is the
`EnvFactory` resume/fork use, and `db_hash()` is for grading (bookkeeping
fields are excluded from grading on purpose).
"""
from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..tools import ToolError, ToolRegistry
from ..util import sha


def initial_state() -> dict[str, Any]:
    return {
        "orders": {
            "O1": {"user_id": "u1", "status": "pending", "items": ["headphones"], "total": 59.0},
            "O2": {"user_id": "u1", "status": "shipped", "items": ["keyboard"], "total": 89.0},
            "O3": {"user_id": "u2", "status": "pending", "items": ["mouse"], "total": 25.0},
        },
        "_applied": {},
    }


class ToyRetailEnv:
    def __init__(self, state: dict[str, Any] | None = None) -> None:
        self.state = copy.deepcopy(state) if state is not None else initial_state()

    @property
    def applied(self) -> dict[str, Any]:
        return self.state["_applied"]

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)

    @classmethod
    def restore(cls, state: dict[str, Any]) -> "ToyRetailEnv":
        return cls(state)

    def db_hash(self) -> str:
        return sha(self.state["orders"])


def expected_db_hash() -> str:
    """Ground truth for the demo task: O1 cancelled, everything else untouched."""
    env = ToyRetailEnv()
    env.state["orders"]["O1"]["status"] = "cancelled"
    return env.db_hash()


def grade(env: ToyRetailEnv) -> bool:
    return env.db_hash() == expected_db_hash()


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ListOrdersArgs(_Strict):
    user_id: str


class GetOrderArgs(_Strict):
    order_id: str


class CancelOrderArgs(_Strict):
    order_id: str
    reason: Literal["ordered_by_mistake", "no_longer_needed"]


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool("list_orders", "List all orders for a user.", ListOrdersArgs)
    def list_orders(env: ToyRetailEnv, a: ListOrdersArgs):
        return [{"order_id": k, **v} for k, v in sorted(env.state["orders"].items())
                if v["user_id"] == a.user_id]

    @reg.tool("get_order", "Get one order by id.", GetOrderArgs)
    def get_order(env: ToyRetailEnv, a: GetOrderArgs):
        o = env.state["orders"].get(a.order_id)
        if o is None:
            raise ToolError(f"order {a.order_id} not found")
        return {"order_id": a.order_id, **o}

    @reg.tool("cancel_order", "Cancel a pending order. Shipped orders cannot be cancelled.",
              CancelOrderArgs, mutating=True)
    def cancel_order(env: ToyRetailEnv, a: CancelOrderArgs):
        o = env.state["orders"].get(a.order_id)
        if o is None:
            raise ToolError(f"order {a.order_id} not found")
        if o["status"] != "pending":
            raise ToolError(f"order {a.order_id} is {o['status']} and cannot be cancelled")
        o["status"] = "cancelled"
        return {"order_id": a.order_id, "status": "cancelled", "reason": a.reason}

    return reg
