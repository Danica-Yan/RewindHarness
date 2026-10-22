"""What the self-improvement loop is allowed to change, as plain data.

A `ContextVersion` is an immutable, versioned set of additions on top of the
benchmark's own agent context:

* `rules`      -- learned policy lines, rendered as one extra section appended
                  to the original wiki (the wiki text itself is never edited,
                  so every learned line is visible and attributable)
* `tool_notes` -- one note appended to a tool's description

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

from ..tools import Tool, ToolRegistry

SurfaceKind = Literal["rule", "tool_note"]
EditOp = Literal["add", "replace", "delete"]

LEARNED_HEADER = "## Additional policies (learned)"


class Edit(BaseModel):
    edit_id: str = ""
    kind: SurfaceKind
    target: str                       # rule id, or tool name
    op: EditOp
    text: str | None = None           # new text for add / replace
    rationale: str = ""
    source_cluster: str = ""
    evidence_tasks: list[int] = Field(default_factory=list)

    def model_post_init(self, _ctx) -> None:
        if not self.edit_id:
            h = hashlib.sha256(json.dumps(
                [self.kind, self.target, self.op, self.text], ensure_ascii=False).encode()).hexdigest()
            self.edit_id = "e_" + h[:12]


class EditError(ValueError):
    pass


class ContextVersion(BaseModel):
    version: int = 0
    parent: int | None = None
    rules: dict[str, str] = Field(default_factory=dict)       # rule id -> text (insertion order kept)
    tool_notes: dict[str, str] = Field(default_factory=dict)  # tool name -> note

    # -- edits -----------------------------------------------------------------
    def apply(self, edit: Edit, *, known_tools: set[str] | None = None) -> "ContextVersion":
        rules, notes = dict(self.rules), dict(self.tool_notes)
        table = rules if edit.kind == "rule" else notes
        if edit.kind == "tool_note" and known_tools is not None and edit.target not in known_tools:
            raise EditError(f"unknown tool {edit.target!r}")
        if edit.op == "add":
            if edit.target in table:
                raise EditError(f"{edit.kind} {edit.target!r} already exists (use replace)")
            if not (edit.text or "").strip():
                raise EditError("add needs text")
            table[edit.target] = edit.text.strip()
        elif edit.op == "replace":
            if edit.target not in table:
                raise EditError(f"{edit.kind} {edit.target!r} does not exist")
            if not (edit.text or "").strip():
                raise EditError("replace needs text")
            table[edit.target] = edit.text.strip()
        else:
            if edit.target not in table:
                raise EditError(f"{edit.kind} {edit.target!r} does not exist")
            del table[edit.target]
        return ContextVersion(version=self.version + 1, parent=self.version,
                              rules=rules, tool_notes=notes)

    def apply_all(self, edits: list[Edit], **kw) -> "ContextVersion":
        v = self
        for e in edits:
            v = v.apply(e, **kw)
        # several edits still make ONE new version
        return v.model_copy(update={"version": self.version + 1, "parent": self.version})

    # -- rendering ---------------------------------------------------------------
    def render_system_prompt(self, base: str) -> str:
        if not self.rules:
            return base
        lines = "\n".join(f"- {t}" for t in self.rules.values())
        return f"{base.rstrip()}\n\n{LEARNED_HEADER}\n\n{lines}\n"

    def patch_registry(self, reg: ToolRegistry) -> ToolRegistry:
        if not self.tool_notes:
            return reg
        out = ToolRegistry()
        for t in reg._tools.values():  # noqa: SLF001 -- same package family, read-only use
            note = self.tool_notes.get(t.name)
            out.register(t if note is None else Tool(
                name=t.name, description=f"{t.description}\n\nNote: {note}", args_model=t.args_model,
                fn=t.fn, mutating=t.mutating, input_schema=copy.deepcopy(t.input_schema),
                raw_text_result=t.raw_text_result))
        return out

    def token_size(self) -> int:
        """Rough size of everything this version adds (~4 chars per token)."""
        n = sum(len(t) for t in self.rules.values()) + sum(len(t) for t in self.tool_notes.values())
        return n // 4

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps([self.rules, self.tool_notes], ensure_ascii=False)
                              .encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# leak / overfit checks on edit text
# ---------------------------------------------------------------------------
_LEAK_PATTERNS = [
    (re.compile(r"#W\d{5,}"), "order id"),
    (re.compile(r"\b\d{10}\b"), "item/product id"),
    (re.compile(r"\b[a-z]+_[a-z]+_\d{4}\b"), "user id"),
    (re.compile(r"[\w.+-]+@[\w-]+\.\w+"), "email"),
    (re.compile(r"\b(task|example)\s*#?\d+", re.I), "task reference"),
]


def leak_reasons(text: str) -> list[str]:
    """Why an edit's text looks like it encodes specific cases instead of a
    general policy (ids, emails, task numbers). Empty list = looks general."""
    return [why for pat, why in _LEAK_PATTERNS if pat.search(text or "")]


def validate_edit(edit: Edit, ctx: ContextVersion, *, known_tools: set[str],
                  max_tokens: int) -> str | None:
    """None if the edit may be tried, else the reason it is refused before
    spending any evaluation on it."""
    if edit.text and leak_reasons(edit.text):
        return "looks case-specific: " + ", ".join(leak_reasons(edit.text))
    if edit.text and len(edit.text) > 600:
        return "too long (>600 chars): a learned rule should be one policy, not a procedure dump"
    try:
        nxt = ctx.apply(edit, known_tools=known_tools)
    except EditError as e:
        return str(e)
    if edit.op == "add" and nxt.token_size() > max_tokens:
        return f"context budget exceeded ({nxt.token_size()} > {max_tokens} tokens): replace or delete instead"
    return None
