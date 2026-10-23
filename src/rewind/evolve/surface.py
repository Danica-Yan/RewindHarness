"""What the self-improvement loop is allowed to change, as plain data.

A `ContextVersion` is an immutable, versioned set of additions on top of the
benchmark's own agent context:

* `rules`      -- learned policy lines, rendered as one extra section appended
                  to the original wiki (the wiki text itself is never edited,
                  so every learned line is visible and attributable)
* `tool_notes` -- one note appended to a tool's description
* `skills`     -- step-by-step procedures the agent loads on demand
                  (`rewind.skills`); only name + description go in the
                  prompt; an edit's `text` is {"description","steps","tools"}
* `checks`     -- policy checks run in code before a tool call
                  (`rewind.policy`); an edit's `text` is the check as JSON,
                  its `target` the check id

Edits are data (`Edit`), applied functionally (`apply` returns a new
version), so a version is reproducible from its lineage and any version can
be rolled back to. Skills and guard templates are later surfaces (design P3).
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, Field

from ..policy import Check, PolicySet
from ..skills import LOAD_SKILL, MAX_SKILLS, Skill, load_skill_tool, skill_index
from ..tools import Tool, ToolRegistry

SurfaceKind = Literal["rule", "tool_note", "check", "skill"]
EditOp = Literal["add", "replace", "delete"]

LEARNED_HEADER = "## Additional policies (learned)"


class Edit(BaseModel):
    edit_id: str = ""
    kind: SurfaceKind
    target: str                       # rule id, or tool name
    op: EditOp
    text: str | None = None           # new text for add / replace
    trigger: str | None = None        # the situation the policy applies to ("the customer ...")
    rationale: str = ""
    source_cluster: str = ""
    evidence_tasks: list[int] = Field(default_factory=list)

    def model_post_init(self, _ctx) -> None:
        if not self.edit_id:
            h = hashlib.sha256(json.dumps(
                [self.kind, self.target, self.op, self.text, self.trigger],
                ensure_ascii=False).encode()).hexdigest()
            self.edit_id = "e_" + h[:12]


class EditError(ValueError):
    pass


def scoped_text(edit: Edit) -> str:
    """What is actually written into the context: the policy prefixed by its
    trigger, so the agent reads a conditional rule, never a blanket one."""
    body = (edit.text or "").strip()
    trig = (edit.trigger or "").strip().rstrip(".:")
    if not trig:
        return body
    trig = trig[0].lower() + trig[1:] if trig[:1].isupper() and not trig[:2].isupper() else trig
    return f"When {trig.removeprefix('when ').removeprefix('When ')}: {body}"


class ContextVersion(BaseModel):
    version: int = 0
    parent: int | None = None
    rules: dict[str, str] = Field(default_factory=dict)       # rule id -> text (insertion order kept)
    tool_notes: dict[str, str] = Field(default_factory=dict)  # tool name -> note
    checks: dict[str, dict] = Field(default_factory=dict)     # check id -> Check fields
    skills: dict[str, dict] = Field(default_factory=dict)     # skill name -> Skill fields

    # -- edits -----------------------------------------------------------------
    def apply(self, edit: Edit, *, known_tools: set[str] | None = None) -> ContextVersion:
        if edit.kind in ("check", "skill"):
            table = dict(self.checks if edit.kind == "check" else self.skills)
            self._apply_structured(edit, table, known_tools)
            return self._next(**{"checks" if edit.kind == "check" else "skills": table})
        rules, notes = dict(self.rules), dict(self.tool_notes)
        table = rules if edit.kind == "rule" else notes
        if edit.kind == "tool_note" and known_tools is not None and edit.target not in known_tools:
            raise EditError(f"unknown tool {edit.target!r}")
        if edit.op == "add":
            if edit.target in table:
                raise EditError(f"{edit.kind} {edit.target!r} already exists (use replace)")
            if not (edit.text or "").strip():
                raise EditError("add needs text")
            table[edit.target] = scoped_text(edit)
        elif edit.op == "replace":
            if edit.target not in table:
                raise EditError(f"{edit.kind} {edit.target!r} does not exist")
            if not (edit.text or "").strip():
                raise EditError("replace needs text")
            table[edit.target] = scoped_text(edit)
        else:
            if edit.target not in table:
                raise EditError(f"{edit.kind} {edit.target!r} does not exist")
            del table[edit.target]
        return self._next(rules=rules, tool_notes=notes)

    def _next(self, **update) -> ContextVersion:
        """The successor version: everything carried over except `update`."""
        return self.model_copy(deep=True, update={"version": self.version + 1, "parent": self.version,
                                                  **update})

    def _apply_structured(self, edit: Edit, table: dict[str, dict],
                          known_tools: set[str] | None) -> None:
        if edit.op == "delete":
            if edit.target not in table:
                raise EditError(f"{edit.kind} {edit.target!r} does not exist")
            del table[edit.target]
            return
        if (edit.op == "add") == (edit.target in table):
            raise EditError(f"{edit.kind} {edit.target!r} " + ("already exists (use replace)"
                                                              if edit.op == "add" else "does not exist"))
        obj = parse_check(edit) if edit.kind == "check" else parse_skill(edit)
        known = (known_tools or set()) - {LOAD_SKILL}
        unknown = sorted(set(obj.tools) - known) if known_tools is not None else []
        if unknown:
            raise EditError(f"{edit.kind} names unknown tool(s) {unknown}")
        if edit.kind == "skill" and edit.op == "add" and len(table) >= MAX_SKILLS:
            raise EditError(f"skill library is full ({MAX_SKILLS}): merge or delete one instead")
        table[edit.target] = obj.model_dump()

    def policy(self) -> PolicySet:
        return PolicySet([Check.model_validate(c) for c in self.checks.values()])

    def skill_objects(self) -> dict[str, Skill]:
        return {k: Skill.model_validate(v) for k, v in self.skills.items()}

    def apply_all(self, edits: list[Edit], **kw) -> ContextVersion:
        v = self
        for e in edits:
            v = v.apply(e, **kw)
        # several edits still make ONE new version
        return v.model_copy(update={"version": self.version + 1, "parent": self.version})

    # -- rendering ---------------------------------------------------------------
    def render_system_prompt(self, base: str) -> str:
        if not self.rules and not self.skills:
            return base
        out = base.rstrip()
        if self.rules:
            lines = "\n".join(f"- {t}" for t in self.rules.values())
            out += f"\n\n{LEARNED_HEADER}\n\n{lines}"
        if self.skills:
            out += "\n\n" + skill_index(self.skill_objects())
        return out + "\n"

    def patch_registry(self, reg: ToolRegistry) -> ToolRegistry:
        if not self.tool_notes and not self.checks and not self.skills:
            return reg
        out = ToolRegistry()
        out.policy = self.policy() if self.checks else getattr(reg, "policy", None)
        for t in reg._tools.values():
            note = self.tool_notes.get(t.name)
            out.register(t if note is None else Tool(
                name=t.name, description=f"{t.description}\n\nNote: {note}", args_model=t.args_model,
                fn=t.fn, mutating=t.mutating, input_schema=copy.deepcopy(t.input_schema),
                raw_text_result=t.raw_text_result))
        if self.skills:
            out.register(load_skill_tool(self.skill_objects()))
        return out

    def token_size(self) -> int:
        """Rough size of everything this version adds (~4 chars per token)."""
        n = sum(len(t) for t in self.rules.values()) + sum(len(t) for t in self.tool_notes.values())
        n += sum(len(s["name"]) + len(s["description"]) + 4 for s in self.skills.values())  # steps load on demand
        return n // 4

    def fingerprint(self) -> str:
        parts = [self.rules, self.tool_notes] + ([self.checks] if self.checks or self.skills else []) \
            + ([self.skills] if self.skills else [])
        return hashlib.sha256(json.dumps(parts, ensure_ascii=False)
                              .encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# leak / overfit checks on edit text
# ---------------------------------------------------------------------------
_LEAK_PATTERNS = [
    (re.compile(r"#W\d{5,}"), "order id"),
    (re.compile(r"\b\d{10}\b"), "item/product id"),
    (re.compile(r"\b[a-z]+_[a-z]+_\d{4}\b"), "user id"),
    (re.compile(r"[\w.+-]+@[\w-]+\.\w+"), "email"),
    (re.compile(r"\b(task|example)\s*#?\d+", re.IGNORECASE), "task reference"),
]


def leak_reasons(text: str) -> list[str]:
    """Why an edit's text looks like it encodes specific cases instead of a
    general policy (ids, emails, task numbers). Empty list = looks general."""
    return [why for pat, why in _LEAK_PATTERNS if pat.search(text or "")]


def validate_edit(edit: Edit, ctx: ContextVersion, *, known_tools: set[str],
                  max_tokens: int) -> str | None:
    """None if the edit may be tried, else the reason it is refused before
    spending any evaluation on it."""
    both = f"{edit.text or ''} {edit.trigger or ''}"
    if edit.kind == "check":
        return _validate_check(edit, ctx, known_tools)
    if edit.kind == "skill":
        return _validate_skill(edit, ctx, known_tools, max_tokens)
    if leak_reasons(both):
        return "looks case-specific: " + ", ".join(leak_reasons(both))
    if edit.op != "delete":
        # Scope (round-1 lesson: an unconditional "do not offer alternatives"
        # clause broke 12 tasks of the same kind). A global rule must say when
        # it applies; a rule about one tool belongs on that tool.
        if edit.kind == "rule" and len((edit.trigger or "").split()) < 3:
            return ("a rule needs a trigger: the specific situation it applies to "
                    "(at least a few words), not a blanket instruction")
        if edit.kind == "rule":
            named = sorted(t for t in known_tools if t in both)
            if len(named) == 1:
                return f"this rule is about one tool -- make it a tool_note on {named[0]}"
    if edit.text and len(edit.text) > 600:
        return "too long (>600 chars): a learned rule should be one policy, not a procedure dump"
    try:
        nxt = ctx.apply(edit, known_tools=known_tools)
    except EditError as e:
        return str(e)
    if edit.op == "add" and nxt.token_size() > max_tokens:
        return f"context budget exceeded ({nxt.token_size()} > {max_tokens} tokens): replace or delete instead"
    return None


def parse_check(edit: Edit) -> Check:
    """The check an edit carries: `text` is its JSON; `target` is its id."""
    try:
        raw = json.loads(edit.text or "")
    except json.JSONDecodeError as e:
        raise EditError(f"check text is not JSON: {e}") from e
    if not isinstance(raw, dict):
        raise EditError("check text must be a JSON object")
    raw["id"] = edit.target
    try:
        return Check.model_validate(raw)
    except ValueError as e:
        raise EditError(f"invalid check: {e}") from e


def _validate_check(edit: Edit, ctx: ContextVersion, known_tools: set[str]) -> str | None:
    if edit.op != "delete":
        try:
            c = parse_check(edit)
        except EditError as e:
            return str(e)
        if leak_reasons(f"{c.message} {json.dumps(c.params)}"):
            return "looks case-specific: " + ", ".join(leak_reasons(f"{c.message} {json.dumps(c.params)}"))
    try:
        ctx.apply(edit, known_tools=known_tools)
    except EditError as e:
        return str(e)
    return None


def parse_skill(edit: Edit) -> Skill:
    """The skill an edit carries: `text` is {"description", "steps", "tools"};
    `target` is its name."""
    try:
        raw = json.loads(edit.text or "")
    except json.JSONDecodeError as e:
        raise EditError(f"skill text is not JSON: {e}") from e
    if not isinstance(raw, dict):
        raise EditError("skill text must be a JSON object")
    raw["name"] = edit.target
    try:
        return Skill.model_validate(raw)
    except ValueError as e:
        raise EditError(f"invalid skill: {e}") from e


def _validate_skill(edit: Edit, ctx: ContextVersion, known_tools: set[str], max_tokens: int) -> str | None:
    if edit.op != "delete":
        try:
            sk = parse_skill(edit)
        except EditError as e:
            return str(e)
        text = " ".join([sk.description, *sk.steps])
        if leak_reasons(text):
            return "looks case-specific: " + ", ".join(leak_reasons(text))
    try:
        nxt = ctx.apply(edit, known_tools=known_tools)
    except EditError as e:
        return str(e)
    if edit.op == "add" and nxt.token_size() > max_tokens:
        return f"context budget exceeded ({nxt.token_size()} > {max_tokens} tokens)"
    return None
