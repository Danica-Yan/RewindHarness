"""Policy checks: rules enforced in code before a tool call runs.

A rule in the system prompt is a soft constraint -- the agent may ignore it
(the first rule Evolve learned fixed its targets only 6 times in 15). Some
policies are mechanical enough to check instead: "get an explicit yes before
any write", "only act for the user you authenticated". A `Check` names the
tools it guards and ONE condition from a small fixed library; when the
condition fails, the call is not executed and the agent gets the check's
message back as the tool error, in the same conversation, so it can fix the
mistake (ask for confirmation, authenticate, ...).

Checks are data, not code: the self-improvement loop may compose them but
cannot write new condition logic. Every condition is a pure function of

  * the call (tool name, arguments),
  * a read-only view of the environment's database (`env.data`), and
  * the conversation so far (rewind's message list),

so the same check can be evaluated live (`PolicySet.evaluate`, from the agent
loop) and offline over recorded traces (`rewind.evolve.backtest`) with
identical results.

What checks cannot do: make the agent act. "Agent refused instead of calling
the tool" leaves nothing to intercept -- that stays a prompt rule's job.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

CheckMode = Literal["block", "observe"]
BLOCK_TAG = "[policy:{id}]"

AFFIRMATIVE = re.compile(
    r"\b(yes|yeah|yep|yup|correct|confirm(ed|s)?|sure|go ahead|proceed|please do|do it|"
    r"that'?s right|sounds good|ok(ay)?|absolutely|definitely)\b", re.IGNORECASE)
NEGATIVE_START = re.compile(r"^\s*(no|nope|wait|hold on|actually,? no|don'?t|do not|not yet)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# the conversation, as the conditions see it
# ---------------------------------------------------------------------------
@dataclass
class Turn:
    kind: Literal["user", "agent", "call"]
    text: str = ""
    name: str = ""
    args: dict | None = None
    ok: bool | None = None          # calls only; None = no result yet (the call being checked)
    result: str = ""


def turns(history: list[dict]) -> list[Turn]:
    """Flatten rewind messages into user text / agent text / tool call turns,
    each call joined with its result. The system message is dropped."""
    results = {m.get("tool_call_id"): m for m in history if m.get("role") == "tool"}
    out: list[Turn] = []
    for m in history:
        role = m.get("role")
        if role == "user":
            out.append(Turn("user", text=m.get("content") or ""))
        elif role == "assistant":
            # Text that came with a tool call never reached the user (the
            # loop only forwards text-only turns), so it is not "said".
            if (m.get("content") or "").strip() and not m.get("tool_calls"):
                out.append(Turn("agent", text=m["content"]))
            for c in m.get("tool_calls") or []:
                r = results.get(c.get("id"))
                out.append(Turn("call", name=c.get("name", ""), args=c.get("args") or {},
                                ok=None if r is None else not r.get("is_error", False),
                                result="" if r is None else str(r.get("content", ""))))
    return out


@dataclass
class CheckContext:
    tool: str
    args: dict
    data: dict                      # env.data: read-only by convention
    history: list[Turn]             # prior turns; the call being checked is NOT included
    is_write: Callable[[str], bool]


# ---------------------------------------------------------------------------
# condition library: (ctx, params) -> None if satisfied, else why not
# ---------------------------------------------------------------------------
Condition = Callable[[CheckContext, dict], "str | None"]
CONDITIONS: dict[str, Condition] = {}
PARAMS: dict[str, set[str]] = {}


def condition(name: str, params: set[str]):
    def deco(fn: Condition) -> Condition:
        CONDITIONS[name] = fn
        PARAMS[name] = params
        return fn
    return deco


@condition("args_same_length", {"a", "b"})
def _same_length(ctx: CheckContext, p: dict) -> str | None:
    a, b = ctx.args.get(p["a"]), ctx.args.get(p["b"])
    if isinstance(a, list) and isinstance(b, list) and len(a) != len(b):
        return f"{p['a']} has {len(a)} entries but {p['b']} has {len(b)}"
    return None


@condition("arg_in", {"arg", "values"})
def _arg_in(ctx: CheckContext, p: dict) -> str | None:
    v = ctx.args.get(p["arg"])
    if v is not None and v not in p["values"]:
        return f"{p['arg']} must be one of {p['values']}, got {v!r}"
    return None


@condition("state_field_in", {"collection", "key_arg", "field", "values"})
def _state_field_in(ctx: CheckContext, p: dict) -> str | None:
    key = ctx.args.get(p["key_arg"])
    rec = (ctx.data.get(p["collection"]) or {}).get(key) if key is not None else None
    if rec is None:
        return None                 # unknown object: the tool reports it itself
    v = rec.get(p["field"])
    if v not in p["values"]:
        return f"{p['collection']} {key} has {p['field']} {v!r}, needs one of {p['values']}"
    return None


@condition("called_before", {"tool", "same_args"})
def _called_before(ctx: CheckContext, p: dict) -> str | None:
    """A successful call to `tool` earlier in the conversation whose arguments
    named in `same_args` equal this call's."""
    for t in ctx.history:
        if t.kind == "call" and t.ok and t.name == p["tool"] and all(
                (t.args or {}).get(a) == ctx.args.get(a) for a in p["same_args"]):
            return None
    shown = ", ".join(f"{a}={ctx.args.get(a)!r}" for a in p["same_args"])
    return f"call {p['tool']}({shown}) first"


@condition("user_confirmed", {"max_writes"})
def _user_confirmed(ctx: CheckContext, p: dict) -> str | None:
    """The latest user message is an explicit yes, it answers an agent text
    turn (not the opening request), and at most `max_writes - 1` successful
    writes have used that yes already."""
    last_user = max((i for i, t in enumerate(ctx.history) if t.kind == "user"), default=None)
    if last_user is None:
        return "no user message to confirm this action"
    u = ctx.history[last_user]
    said = [t for t in ctx.history[:last_user] if t.kind in ("agent", "user")]
    asked = bool(said) and said[-1].kind == "agent"      # the yes answers the agent's last words
    if not asked or not AFFIRMATIVE.search(u.text) or NEGATIVE_START.search(u.text):
        return "the user has not explicitly confirmed (yes) since you last described the action"
    used = sum(1 for t in ctx.history[last_user + 1:] if t.kind == "call" and t.ok and ctx.is_write(t.name))
    if used >= p.get("max_writes", 1):
        return "that confirmation was already used for an earlier action; confirm this one separately"
    return None


@condition("authenticated_owner", {"auth_tools", "owners"})
def _authenticated_owner(ctx: CheckContext, p: dict) -> str | None:
    """Exactly one user was authenticated (a successful call to one of
    `auth_tools`, whose result is the user id), and the object this call acts
    on belongs to that user. `owners` maps an argument name to
    [collection, owner_field] -- or to null when the argument IS the user id."""
    ids = []
    for t in ctx.history:
        if t.kind == "call" and t.ok and t.name in p["auth_tools"]:
            uid = t.result.strip().strip('"')
            if uid and uid not in ids:
                ids.append(uid)
    if not ids:
        return "authenticate the user (find their user id) before acting on their account"
    if len(ids) > 1:
        return f"more than one user was authenticated ({', '.join(ids)}); only one user per conversation"
    for arg, where in p["owners"].items():
        v = ctx.args.get(arg)
        if v is None:
            continue
        if where is None:
            owner = v
        else:
            rec = (ctx.data.get(where[0]) or {}).get(v)
            if rec is None:
                continue
            owner = rec.get(where[1])
        if owner != ids[0]:
            return f"{arg} {v!r} does not belong to the authenticated user"
    return None


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
class Check(BaseModel):
    id: str
    tools: list[str]                         # which calls it guards
    condition: str
    params: dict[str, Any] = Field(default_factory=dict)
    message: str                             # what the agent is told when blocked
    mode: CheckMode = "block"
    max_blocks: int = 2                      # per episode; afterwards the call goes through

    @model_validator(mode="after")
    def _known(self) -> Check:
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}; known: {sorted(CONDITIONS)}")
        missing = PARAMS[self.condition] - set(self.params)
        if missing and not (self.condition == "user_confirmed" and missing == {"max_writes"}):
            raise ValueError(f"{self.condition} needs params {sorted(missing)}")
        extra = set(self.params) - PARAMS[self.condition]
        if extra:
            raise ValueError(f"{self.condition} does not take {sorted(extra)}")
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,60}", self.id):
            raise ValueError("check id must be snake_case")
        return self

    @property
    def tag(self) -> str:
        return BLOCK_TAG.format(id=self.id)


@dataclass
class Decision:
    check_id: str
    action: Literal["block", "observe", "allow_over_budget"]
    reason: str
    message: str


class PolicySet:
    def __init__(self, checks: list[Check]) -> None:
        self.checks = list(checks)

    def __bool__(self) -> bool:
        return bool(self.checks)

    def evaluate(self, data: dict, tool: str, args: dict, history: list[dict],
                 is_write: Callable[[str], bool]) -> list[Decision]:
        """Every check on `tool` whose condition fails. `history` is the
        message list up to and including the assistant message that made this
        call (its result not yet present)."""
        mine = [c for c in self.checks if tool in c.tools]
        if not mine:
            return []
        hist = [t for t in turns(history) if not (t.kind == "call" and t.ok is None)]
        ctx = CheckContext(tool=tool, args=args or {}, data=data or {}, history=hist, is_write=is_write)
        out = []
        for c in mine:
            why = CONDITIONS[c.condition](ctx, c.params)
            if why is None:
                continue
            action: Any = c.mode
            if c.mode == "block":
                used = sum(1 for m in history if m.get("role") == "tool" and c.tag in str(m.get("content", "")))
                if used >= c.max_blocks:
                    action = "allow_over_budget"
            out.append(Decision(c.id, action, why, f"{c.tag} {c.message} ({why})"))
        return out

    def fingerprint_data(self) -> list[dict]:
        return [c.model_dump() for c in self.checks]


def check_from_json(text: str) -> Check:
    return Check.model_validate(json.loads(text))
