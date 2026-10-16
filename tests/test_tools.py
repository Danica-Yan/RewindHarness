from rewind import ToolRegistry
from rewind.env import ToyRetailEnv, build_registry
from pydantic import BaseModel


def test_schema_sorted_and_strict():
    s = build_registry().schemas()
    assert [t["name"] for t in s] == ["cancel_order", "get_order", "list_orders"]
    assert s[0]["input_schema"]["additionalProperties"] is False


def test_validation_and_hallucination():
    reg, env = build_registry(), ToyRetailEnv()
    bad = reg.execute(env, "cancel_order", {"orderid": "O1"}, "a")
    assert not bad.ok and bad.error_class == "TOOL_ARG_INVALID" and "order_id" in bad.content
    ghost = reg.execute(env, "fly", {}, "a")
    assert ghost.error_class == "HALLUCINATED_TOOL" and "cancel_order" in ghost.content


def test_runtime_error_does_not_mutate():
    reg, env = build_registry(), ToyRetailEnv()
    before = env.db_hash()
    out = reg.execute(env, "cancel_order", {"order_id": "O2", "reason": "no_longer_needed"}, "a")
    assert out.error_class == "TOOL_RUNTIME_ERROR" and env.db_hash() == before


def test_mutation_is_idempotent_per_action_id():
    class A(BaseModel):
        pass

    reg, env = ToolRegistry(), ToyRetailEnv()
    env.state["counter"] = 0

    @reg.tool("bump", "increment", A, mutating=True)
    def bump(e, a):
        e.state["counter"] += 1
        return e.state["counter"]

    first = reg.execute(env, "bump", {}, "s1:c0")
    again = reg.execute(env, "bump", {}, "s1:c0")   # e.g. a resumed worker re-executes the step
    other = reg.execute(env, "bump", {}, "s2:c0")
    assert first.content == again.content == "1"
    assert other.content == "2" and env.state["counter"] == 2
