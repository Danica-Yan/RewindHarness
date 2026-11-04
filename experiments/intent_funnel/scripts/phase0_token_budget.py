"""Phase 0 go/no-go: how much of the retail prompt can per-intent routing remove?

Counts DeepSeek tokens (deepseek-tokenizer package) for:
  full prompt = wiki (system) + all 16 tool schemas
  routed prompt = preamble/Domain-basic + intent wiki sections + shared tools + intent write tools
Tool schemas are serialized as compact JSON; the provider's internal serialization differs a bit,
so treat absolute numbers as estimates and ratios as the signal.

Usage: python phase0_token_budget.py data/taubench out/phase0.json
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepseek_tokenizer import ds_token  # noqa: E402

from intent_funnel.taxonomy import (INTENT_WIKI_SECTIONS, INTENTS, intents_from_actions,  # noqa: E402
                                    tools_for)


def ntok(s: str) -> int:
    return len(ds_token.encode(s))


def split_wiki(wiki: str):
    """Return [(heading, text)], heading '' for the preamble."""
    parts, cur_h, buf = [], "", []
    for line in wiki.splitlines(keepends=True):
        if re.match(r"^#{2,3} ", line):
            parts.append((cur_h, "".join(buf)))
            cur_h, buf = line.strip(), [line]
        else:
            buf.append(line)
    parts.append((cur_h, "".join(buf)))
    return parts


def main(data_dir, out_path):
    d = Path(data_dir)
    wiki = (d / "wiki.md").read_text()
    tools = json.loads((d / "tools.json").read_text())
    by_name = {t["function"]["name"]: t for t in tools}
    sections = split_wiki(wiki)
    always = {"", "# Retail agent policy", "## Domain basic"}

    tool_tok = {n: ntok(json.dumps(t, separators=(",", ":"))) for n, t in by_name.items()}
    sec_tok = {h or "<preamble>": ntok(txt) for h, txt in sections}
    full_wiki, full_tools = ntok(wiki), sum(tool_tok.values())
    full = full_wiki + full_tools

    def routed(intents):
        keep = set(always)
        for i in intents:
            keep.update(INTENT_WIKI_SECTIONS[i])
        w = "".join(txt for h, txt in sections if h in keep)
        tnames = tools_for(intents)
        return ntok(w) + sum(tool_tok[n] for n in tnames if n in tool_tok)

    per_intent = {i: {"tokens": routed([i]), "saving": 1 - routed([i]) / full} for i in INTENTS}

    # weight by real task mix (multi-intent tasks get the union of routes)
    res_split = {}
    for split in ["train", "test"]:
        tasks = json.loads((d / f"tasks_{split}.json").read_text())
        mix = Counter()
        savings = []
        for t in tasks:
            labs = intents_from_actions(t["actions"])
            mix["+".join(sorted(labs))] += 1
            savings.append(1 - routed(labs) / full)
        n_multi = sum(c for k, c in mix.items() if "+" in k)
        res_split[split] = {
            "n_tasks": len(tasks),
            "multi_intent_tasks": n_multi,
            "mean_saving_if_routed_perfectly": sum(savings) / len(savings),
            "label_mix_top": mix.most_common(15),
        }

    out = {
        "full_prompt_tokens": full,
        "wiki_tokens": full_wiki,
        "tool_schema_tokens": full_tools,
        "wiki_sections": sec_tok,
        "tool_tokens": dict(sorted(tool_tok.items(), key=lambda x: -x[1])),
        "per_intent_routed": per_intent,
        "splits": res_split,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("label_mix_top",)}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
