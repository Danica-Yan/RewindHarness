"""Both coordination backends implement the same `CoordinatorStore` port.

The worker and the checkpointer are typed against the port only; this pins
down that neither backend drifts from it (a renamed method, a sync method
sneaking back in) without needing a live Postgres for the structural check.
"""
from __future__ import annotations

import inspect

import pytest

from rewind.coordination import CoordinatorStore, SqliteCoordinator

PORT_METHODS = [n for n, _ in inspect.getmembers(CoordinatorStore, inspect.isfunction)
                if not n.startswith("_")]


def _backends():
    yield SqliteCoordinator
    try:
        from rewind.dist.pg_coordinator import PgCoordinator
    except ImportError:  # [dist] extra not installed
        return
    yield PgCoordinator


def test_port_lists_the_full_protocol():
    assert set(PORT_METHODS) == {"submit", "claim", "heartbeat", "get", "write_checkpoint",
                                 "latest_checkpoint", "complete"}


@pytest.mark.parametrize("backend", list(_backends()), ids=lambda b: b.__name__)
def test_backend_implements_every_port_method_as_async(backend):
    for name in PORT_METHODS:
        impl = getattr(backend, name, None)
        assert impl is not None, f"{backend.__name__} is missing {name}"
        assert inspect.iscoroutinefunction(impl), f"{backend.__name__}.{name} must be async"
        port_params = list(inspect.signature(getattr(CoordinatorStore, name)).parameters)
        impl_params = list(inspect.signature(impl).parameters)
        assert impl_params == port_params, f"{backend.__name__}.{name}: {impl_params} != {port_params}"


def test_sqlite_backend_is_a_runtime_instance_of_the_port():
    assert isinstance(SqliteCoordinator(), CoordinatorStore)
