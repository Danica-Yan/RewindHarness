"""Skills: step-by-step procedures the agent loads on demand.

A rule is one policy line, a check stops a wrong call -- neither can carry a
multi-step procedure ("to exchange items: authenticate, look the order up,
confirm it was delivered, get every item's product details, find the
variants, ask the user for ALL items at once, confirm, call the tool once").
A skill can. Only each skill's name and one-line description go into the
system prompt; the agent calls `load_skill(name)` to read the steps, so the
prompt stays small as the library grows -- and the trace records WHETHER and
WHEN a skill was loaded, which lets the evaluation compare episodes that
loaded it with episodes that did not.

Skills are data (name, when to use it, steps, tools it involves); the
self-improvement loop adds, rewrites, merges and deletes them through the
same versioned, gated edits as rules.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

from .tools import Tool, ToolError

LOAD_SKILL = "load_skill"
SKILLS_HEADER = "## Skills (procedures you can load)"
MAX_SKILLS = 12


class Skill(BaseModel):
    name: str
    description: str                   # when to use it -- this line is all the agent sees up front
    steps: list[str]
    tools: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,40}", v):
            raise ValueError("skill name must be snake_case, 3-41 chars")
        return v

    @field_validator("description")
    @classmethod
    def _desc(cls, v: str) -> str:
        v = " ".join(v.split())
        if not 10 <= len(v) <= 220:
            raise ValueError("description must be one line of 10-220 chars saying when to use the skill")
        return v

    @field_validator("steps")
    @classmethod
    def _steps(cls, v: list[str]) -> list[str]:
        v = [" ".join(s.split()) for s in v if s and s.strip()]
        if not 2 <= len(v) <= 12:
            raise ValueError("a skill has 2-12 steps")
        if any(len(s) > 320 for s in v):
            raise ValueError("each step at most 320 chars")
        return v

    def render(self) -> str:
        lines = [f"Skill: {self.name} -- {self.description}", ""]
        lines += [f"{i}. {s}" for i, s in enumerate(self.steps, 1)]
        if self.tools:
            lines += ["", "Tools involved: " + ", ".join(self.tools)]
        return "\n".join(lines)


def skill_index(skills: dict[str, Skill]) -> str:
    """The system-prompt section: names and descriptions only."""
    lines = [f"- {s.name}: {s.description}" for s in skills.values()]
    return (f"{SKILLS_HEADER}\n\nBefore handling a request one of these covers, call "
            f"{LOAD_SKILL}(name) and follow its steps.\n\n" + "\n".join(lines))


class _LoadSkillArgs(BaseModel):
    name: str


def load_skill_tool(skills: dict[str, Skill]) -> Tool:
    names = list(skills)

    def fn(_env, args: _LoadSkillArgs) -> str:
        s = skills.get(args.name.strip())
        if s is None:
            raise ToolError(f"unknown skill {args.name!r}; available: {', '.join(names)}")
        return s.render()

    return Tool(
        name=LOAD_SKILL,
        description=("Load the step-by-step procedure of one of the skills listed in your instructions. "
                     "Call it before handling a request the skill covers, then follow the steps."),
        args_model=_LoadSkillArgs, fn=fn, mutating=False, raw_text_result=True,
        input_schema={"type": "object", "properties": {"name": {
            "type": "string", "enum": names, "description": "The skill's name."}},
            "required": ["name"]})
