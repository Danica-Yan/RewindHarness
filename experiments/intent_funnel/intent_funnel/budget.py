"""Prompt token cost of a route (DeepSeek tokenizer). Fallback / unclear = full prompt."""
import json
import re
from functools import lru_cache
from pathlib import Path

from .taxonomy import INTENT_WIKI_SECTIONS, tools_for

ALWAYS = {"", "# Retail agent policy", "## Domain basic"}


def split_wiki(wiki: str):
    parts, cur_h, buf = [], "", []
    for line in wiki.splitlines(keepends=True):
        if re.match(r"^#{2,3} ", line):
            parts.append((cur_h, "".join(buf)))
            cur_h, buf = line.strip(), [line]
        else:
            buf.append(line)
    parts.append((cur_h, "".join(buf)))
    return parts


class PromptBudget:
    def __init__(self, data_dir="data/taubench"):
        from deepseek_tokenizer import ds_token
        self._tok = ds_token
        d = Path(data_dir)
        self.wiki = (d / "wiki.md").read_text()
        self.sections = split_wiki(self.wiki)
        tools = json.loads((d / "tools.json").read_text())
        self.tool_tok = {t["function"]["name"]: self.ntok(json.dumps(t, separators=(",", ":"))) for t in tools}
        self.full = self.ntok(self.wiki) + sum(self.tool_tok.values())

    def ntok(self, s):
        return len(self._tok.encode(s))

    @lru_cache(maxsize=4096)
    def _routed(self, key):
        intents = list(key)
        keep = set(ALWAYS)
        for i in intents:
            keep.update(INTENT_WIKI_SECTIONS.get(i, []))
        w = "".join(txt for h, txt in self.sections if h in keep)
        return self.ntok(w) + sum(self.tool_tok[n] for n in tools_for(intents) if n in self.tool_tok)

    def cost(self, intents):
        """intents=None -> fallback (full prompt)."""
        if intents is None:
            return self.full
        return self._routed(tuple(sorted(intents)))
