"""Per-turn tool routing: routed tools/prompt per LLM call, cumulative routes, escalation, route
violations, fallback, deterministic replay without the router, and LoopState rebuild."""
from dataclasses import dataclass

from conftest import call, run, text

from rewind import AgentConfig, AgentRunner, ScriptedProvider, SnapshotStore, TraceWriter, replay_run
from rewind.env import ToyRetailEnv, build_registry
from rewind.loop import LoopState
from rewind.routing import ESCALATE_TOOL, RouteUpdate, escalate_tool

READ = ["get_order", "list_orders"]


@dataclass
class Turn:
    reply: str
    done: bool


class DialogueToyEnv(ToyRetailEnv):
    """Toy retail env with a scripted user: each agent text turn gets the next reply."""

    def __init__(self, replies=None, state=None):
        super().__init__(state)
        self.replies = list(replies or [])

    async def on_text_turn(self, text):
        if not self.replies:
            return Turn("", True)
        return Turn(self.replies.pop(0), False)

    @classmethod
    def restore(cls, state):
        return cls([], state)


class Recording:
    """Provider wrapper that records which tools and system prompt each LLM call received."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    async def complete(self, messages, tools, params):
        self.calls.append((sorted(t["name"] for t in tools), messages[0]["content"]))
        return await self.inner.complete(messages, tools, params)


class KeywordRouter:
    """Routes 'cancel' when the user (or the agent's proposal) mentions it; never confident about 'help me'."""

    def __init__(self, unsure_on=()):
        self.unsure_on = unsure_on

    def update(self, units, full, user_text, agent_text, full_prompt):
        if any(w in user_text for w in self.unsure_on):
            return RouteUpdate(units=units, full=True, tools=None, prompt=None)
        units = sorted(set(units) | ({"cancel"} if "cancel" in (user_text + agent_text) else set()))
        tools = READ + (["cancel_order"] if "cancel" in units else [])
        return RouteUpdate(units=units, full=full, tools=tools, prompt=f"{full_prompt} [route: {units}]",
                           detail={"user": user_text[:20]})


def _runner(script, router, replies):
    snaps = SnapshotStore()
    registry = build_registry()
    registry.register(escalate_tool())
    prov = Recording(ScriptedProvider(script))
    runner = AgentRunner(prov, registry, AgentConfig(system_prompt="sys"), snaps, router=router)
    env = DialogueToyEnv(replies)
    return runner, env, TraceWriter("route-1"), snaps, registry, prov


def test_routes_grow_with_user_turns_and_prompt_is_routed():
    script = [text("How can I help?"), call("cancel_order", order_id="O1", reason="ordered_by_mistake"),
              text("Done.")]
    runner, env, trace, snaps, registry, prov = _runner(script, KeywordRouter(), ["please cancel O1"])
    rr = run(runner.run(env, "hi there", trace))
    assert rr.status == "completed"
    tools1, prompt1 = prov.calls[0]
    assert tools1 == sorted(READ + [ESCALATE_TOOL]) and prompt1 == "sys [route: []]"
    tools2, prompt2 = prov.calls[1]
    assert "cancel_order" in tools2 and prompt2 == "sys [route: ['cancel']]"
    # history keeps the canonical system prompt; the routed one only goes to the LLM
    assert [e for e in trace.events if e.type == "message"][0].data["message"]["content"] == "sys"
    routes = [e.data for e in trace.events if e.type == "route"]
    assert [r["units"] for r in routes] == [[], ["cancel"]] and all(r["reason"] == "user_turn" for r in routes)
    assert env.state["orders"]["O1"]["status"] == "cancelled"


def test_escalation_tool_switches_to_full_toolset():
    script = [call(ESCALATE_TOOL), call("cancel_order", order_id="O1", reason="ordered_by_mistake"),
              text("Done.")]
    runner, env, trace, snaps, registry, prov = _runner(script, KeywordRouter(), [])
    run(runner.run(env, "something unusual", trace))
    assert "cancel_order" not in prov.calls[0][0]
    tools2, prompt2 = prov.calls[1]
    assert "cancel_order" in tools2 and ESCALATE_TOOL not in tools2 and prompt2 == "sys"
    assert any(e.type == "route" and e.data["reason"] == "escalate" for e in trace.events)


def test_route_violation_executes_and_falls_back_to_full():
    script = [call("cancel_order", order_id="O1", reason="ordered_by_mistake"), text("Done.")]
    runner, env, trace, snaps, registry, prov = _runner(script, KeywordRouter(), [])
    run(runner.run(env, "something unusual", trace))
    assert env.state["orders"]["O1"]["status"] == "cancelled"   # the call itself still runs
    assert any(e.type == "route" and e.data["reason"] == "route_violation:cancel_order" for e in trace.events)
    assert "cancel_order" in prov.calls[1][0] and prov.calls[1][1] == "sys"


def test_unconfident_router_falls_back_to_full_prompt():
    script = [text("Sure, what do you need?"), text("Done.")]
    runner, env, trace, snaps, registry, prov = _runner(script, KeywordRouter(unsure_on=("help me",)),
                                                        ["help me with something"])
    run(runner.run(env, "hello", trace))
    assert ESCALATE_TOOL in prov.calls[0][0]                         # routed first turn
    tools2, prompt2 = prov.calls[1]
    assert ESCALATE_TOOL not in tools2 and "cancel_order" in tools2 and prompt2 == "sys"


def test_routed_run_replays_without_the_router():
    script = [text("How can I help?"), call("cancel_order", order_id="O1", reason="ordered_by_mistake"),
              text("Done.")]
    runner, env, trace, snaps, registry, prov = _runner(script, KeywordRouter(), ["please cancel O1"])
    run(runner.run(env, "hi there", trace))

    def env_factory(state):
        e = DialogueToyEnv.restore(state)
        e.replies = ["please cancel O1"]
        return e
    rep = run(replay_run(trace.events, snaps, registry, env_factory))
    assert rep.ok, rep


def test_loop_state_rebuilds_route_from_events():
    script = [text("How can I help?"), call(ESCALATE_TOOL), text("Done.")]
    runner, env, trace, snaps, registry, prov = _runner(script, KeywordRouter(), ["please cancel O1"])
    run(runner.run(env, "hi there", trace))
    st = LoopState.from_events(trace.events)
    assert st.route_full and st.route_tools is None and st.route_units == ["cancel"] and st.route_user_seen == 2


def test_no_router_sends_every_tool_except_the_escalation_tool():
    snaps = SnapshotStore()
    registry = build_registry()
    registry.register(escalate_tool())
    prov = Recording(ScriptedProvider([text("done")]))
    run(AgentRunner(prov, registry, AgentConfig(system_prompt="sys"), snaps).run(ToyRetailEnv(), "t",
                                                                                 TraceWriter("r")))
    assert prov.calls[0] == (sorted(["cancel_order", "get_order", "list_orders"]), "sys")
