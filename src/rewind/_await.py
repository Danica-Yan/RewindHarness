"""One helper so the same calling code works against a synchronous backend
(the sqlite `Coordinator`, used for offline tests/demos) and an asynchronous
one (the Postgres `PgCoordinator` in `dist/`, used for real deployments)
without duplicating `run_worker_step` / `StepCheckpointer` per backend."""
from __future__ import annotations

import inspect
from typing import Any, Awaitable, TypeVar

T = TypeVar("T")


async def maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value
