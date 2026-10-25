"""SnapshotStore isolation. Added after mutation testing: handing out (or
keeping) the live object instead of a copy passed every fast test, but it means a
restored env that is then mutated -- which every fork does -- silently
rewrites the stored snapshot, and the next fork from it starts from the
wrong state."""
from rewind import SnapshotStore
from rewind.env import ToyRetailEnv


def test_mutating_what_get_returns_does_not_change_the_snapshot():
    """Checked on the store itself: the envs' own restore() happens to copy
    again, which hid this -- but the worker passes a checkpoint bundle's
    `messages` from get() straight to the loop, which appends to it."""
    store = SnapshotStore()
    h = store.put({"messages": [{"role": "user", "content": "hi"}]})
    got = store.get(h)
    got["messages"].append({"role": "assistant", "content": "resumed once"})
    assert store.get(h) == {"messages": [{"role": "user", "content": "hi"}]}


def test_mutating_the_live_state_after_put_does_not_change_the_snapshot():
    store = SnapshotStore()
    state = ToyRetailEnv().state
    h = store.put(state)
    state["orders"]["O1"]["status"] = "cancelled"                  # the env keeps running
    assert store.get(h)["orders"]["O1"]["status"] == "pending"


def test_ordered_snapshots_keep_key_order_through_disk(tmp_path):
    """The fork-fidelity bug: tau-bench shows records in stored key order."""
    store = SnapshotStore(tmp_path)
    h = store.put({"b": 1, "a": {"z": 1, "y": 2}}, ordered=True)
    assert list(SnapshotStore(tmp_path).get(h)) == ["b", "a"]
    assert list(SnapshotStore(tmp_path).get(h)["a"]) == ["z", "y"]
