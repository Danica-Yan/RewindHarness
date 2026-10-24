"""Each shipped env implements the environment port, and exactly the
optional capabilities it claims -- the loop's behaviour switches on these
isinstance checks, so a capability gained or lost by accident changes how
a run ends (e.g. a text turn ending the run vs. going to a simulated user)."""
from __future__ import annotations

import pytest

from rewind import SnapshotStore
from rewind.env import ToyRetailEnv
from rewind.environment import DataBackedEnv, DialogueEnv, Environment, TerminatingEnv
from rewind.policy import PolicySet
from rewind.tools import ToolGate, ToolRegistry


def _caps(env) -> set[str]:
    return {name for name, proto in [("dialogue", DialogueEnv), ("terminating", TerminatingEnv),
                                     ("data", DataBackedEnv)] if isinstance(env, proto)}


def test_toy_env_is_a_plain_environment():
    env = ToyRetailEnv()
    assert isinstance(env, Environment)
    assert _caps(env) == set()


def test_taubench_envs_declare_their_capabilities():
    pytest.importorskip("tau_bench", reason="needs the [taubench] extra")
    from rewind.env.taubench_adapter import TauBenchRetailEnv
    from rewind.env.taubench_dialogue import TauBenchDialogueEnv, TauUserSim

    single = TauBenchRetailEnv()
    assert isinstance(single, Environment)
    assert _caps(single) == {"data"}

    async def never_called(messages):  # pragma: no cover
        raise AssertionError
    dialogue = TauBenchDialogueEnv(SnapshotStore(), TauUserSim(never_called), "test", 0)
    assert isinstance(dialogue, Environment)
    assert _caps(dialogue) == {"dialogue", "terminating", "data"}


def test_policy_set_fits_the_registry_gate_slot():
    reg = ToolRegistry()
    reg.policy = PolicySet([])
    assert callable(getattr(reg.policy, "evaluate"))
    assert ToolGate.evaluate.__code__.co_varnames[:6] == PolicySet.evaluate.__code__.co_varnames[:6]
