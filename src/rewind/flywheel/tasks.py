"""A small, varied task suite over the toy retail env, plus a fixed 60/20/20
split persisted to disk (plan W8's first item). `env.py`'s original single
hard-coded task ("cancel O1 for u1") is fine for demoing the harness itself,
but the data flywheel needs *many* tasks with *known-correct* outcomes to
train/validate/heldout against — this module generates those deterministically
(seeded, so regenerating reproduces the same suite byte-for-byte) rather than
requiring a hand-written task per scenario.

Each task's "correct policy" (baked into `expected_cancelled`, not something
a model decides) is: cancel the named order iff it belongs to the requesting
user AND is currently `pending`. Three task categories exercise that policy
from different angles:

  - "normal": target order is pending and owned by the requester -> should
    be cancelled.
  - "shipped_trap": target order is owned by the requester but already
    shipped -> should NOT be cancelled (the tool itself would refuse this
    one — see `TOOL_RUNTIME_ERROR` in flywheel/failures.py).
  - "ownership_trap": target order is pending but owned by someone else ->
    should NOT be cancelled (the *tool* has no ownership check, so only the
    model refusing to call it makes this correct — see `REASONING_ERROR`).

Distractor orders (never the task's target) are included in every task's
order book purely so grading can catch a provider that touches the wrong
order, not just whether the target order ended up in the right state.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

CATEGORIES: tuple[str, ...] = ("normal", "shipped_trap", "ownership_trap")
REQUESTER = "u1"
OTHER_USER = "u2"


@dataclass
class TaskSpec:
    task_id: str
    category: str
    requester: str
    task_text: str
    target_order: str
    orders: dict[str, dict]              # order_id -> {user_id, status, items, total}
    expected_cancelled: list[str]        # order ids that should end up cancelled (0 or 1 here)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpec":
        return cls(**d)

    def initial_state(self) -> dict:
        return {"orders": {oid: dict(o) for oid, o in self.orders.items()}, "_applied": {}}


def _make_order(order_id: str, user_id: str, status: str, item: str, total: float) -> dict:
    return {"user_id": user_id, "status": status, "items": [item], "total": total}


_ITEMS = ["headphones", "keyboard", "mouse", "monitor", "webcam", "charger", "speaker", "cable"]


def make_task(idx: int, category: str, rng: random.Random) -> TaskSpec:
    target_id = f"O{idx}-t"
    if category == "normal":
        target_status, target_owner, expected = "pending", REQUESTER, [target_id]
    elif category == "shipped_trap":
        target_status, target_owner, expected = "shipped", REQUESTER, []
    elif category == "ownership_trap":
        target_status, target_owner, expected = "pending", OTHER_USER, []
    else:
        raise ValueError(category)

    orders = {target_id: _make_order(target_id, target_owner, target_status,
                                     rng.choice(_ITEMS), round(rng.uniform(10, 200), 2))}
    n_distractors = rng.randint(1, 3)
    for d in range(n_distractors):
        did = f"O{idx}-d{d}"
        orders[did] = _make_order(did, rng.choice([REQUESTER, OTHER_USER]),
                                  rng.choice(["pending", "shipped", "cancelled"]),
                                  rng.choice(_ITEMS), round(rng.uniform(10, 200), 2))

    return TaskSpec(
        task_id=f"task-{idx:03d}-{category}", category=category, requester=REQUESTER,
        task_text=f"I'm {REQUESTER}, please cancel order {target_id}.",
        target_order=target_id, orders=orders, expected_cancelled=expected)


def generate_suite(n: int = 30, seed: int = 20260928) -> list[TaskSpec]:
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        category = CATEGORIES[i % len(CATEGORIES)]
        tasks.append(make_task(i, category, rng))
    rng.shuffle(tasks)
    return tasks


Split = Literal["train", "dev", "heldout"]


def split_suite(tasks: list[TaskSpec], train: float = 0.6, dev: float = 0.2
               ) -> dict[Split, list[TaskSpec]]:
    n = len(tasks)
    n_train = round(n * train)
    n_dev = round(n * dev)
    return {"train": tasks[:n_train], "dev": tasks[n_train:n_train + n_dev],
           "heldout": tasks[n_train + n_dev:]}


def save_split(split: dict[Split, list[TaskSpec]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: [t.to_dict() for t in tasks] for name, tasks in split.items()}
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def load_split(path: str | Path) -> dict[Split, list[TaskSpec]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: [TaskSpec.from_dict(d) for d in tasks] for name, tasks in payload.items()}


def grade_task(env, spec: TaskSpec) -> bool:
    """Every order must end exactly where the policy says it should: the
    expected-cancelled order(s) cancelled, every other order (including
    every distractor) untouched from its original status."""
    for oid, original in spec.orders.items():
        final_status = env.state["orders"][oid]["status"]
        if oid in spec.expected_cancelled:
            if final_status != "cancelled":
                return False
        elif final_status != original["status"]:
            return False
    return True
