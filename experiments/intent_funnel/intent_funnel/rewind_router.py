"""TauRouter: the intent router as a rewind `ToolRouter` (src/rewind/routing.py), for tau-bench retail.

Per new user turn:
  1. user-text model -> intent probabilities -> decide(t_include, gate) with route groups
     (no confident intent -> fallback to the full toolset for the rest of the run)
  2. + intents the agent's previous message proposes (agent_rules, proposal sentences only)
  3. cumulative union with the intents routed so far
  4. tools   = shared read-only tools + the write tools of those intents
     prompt  = the system prompt with only the wiki sections those intents need. Only the original
               wiki text inside the prompt is pruned; anything else (e.g. Evolve rules) is kept verbatim.

Builders (thresholds come from the frozen held-out files):
  build_router("tfidf")   TF-IDF trained on trajectory turns + agent rules       CPU only
  build_router("hybrid", qwen_adapters=[...])  Qwen3-0.6B LoRA avg ⊕ TF-IDF + agent rules
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .agent_rules import proposed_intents
from .budget import ALWAYS, split_wiki
from .router import decide
from .taxonomy import INTENT_WIKI_SECTIONS, INTENTS, expand_groups, tools_for
from .textclean import clean

ROOT = Path(__file__).resolve().parents[1]


def _rewind_types():
    try:
        from rewind.routing import RouteUpdate
    except ImportError:  # running from experiments/ without rewind installed
        sys.path.insert(0, str(ROOT.parents[1] / "src"))
        from rewind.routing import RouteUpdate
    return RouteUpdate


class TauRouter:
    def __init__(self, model, t_include: float, gate: float, wiki: str, agent_rules: bool = True,
                 name: str = "tau", always_tools=()):
        self.model, self.t_include, self.gate = model, t_include, gate
        self.wiki, self.agent_rules, self.name = wiki, agent_rules, name
        self.always_tools = sorted(always_tools)  # non-retail tools (e.g. Evolve's skill loader): never hidden
        self.sections = split_wiki(wiki)
        self.RouteUpdate = _rewind_types()

    # ---- prompt -----------------------------------------------------------------
    def pruned_wiki(self, intents) -> str:
        keep = set(ALWAYS)
        for i in intents:
            keep.update(INTENT_WIKI_SECTIONS.get(i, []))
        return "".join(txt for h, txt in self.sections if h in keep)

    def routed_prompt(self, intents, full_prompt: str) -> str | None:
        w = self.wiki.rstrip()  # Evolve's render_system_prompt rstrips the wiki before appending rules
        if w not in full_prompt:  # unknown prompt layout: never prune what we can't locate
            return None
        return full_prompt.replace(w, self.pruned_wiki(intents).rstrip(), 1)

    # ---- rewind ToolRouter ----------------------------------------------------------
    def update(self, units, full, user_text, agent_text, full_prompt):
        RU = self.RouteUpdate
        if full:
            return RU(units=list(units), full=True, tools=None, prompt=None)
        p = self.model.predict_proba([clean(user_text)[0]])
        d = decide(p, self.model.classes, t_include=self.t_include, gate=self.gate)[0]
        top = sorted(zip(self.model.classes, p[0].tolist()), key=lambda x: -x[1])[:3]
        detail = {"router": self.name, "top": [[c, round(v, 3)] for c, v in top]}
        if d is None:
            detail["fallback"] = "low_confidence"
            return RU(units=list(units), full=True, tools=None, prompt=None, detail=detail)
        add = set(d)
        if self.agent_rules:
            prop = proposed_intents(agent_text)
            if prop:
                detail["agent_proposed"] = prop
            add |= set(prop)
        new = expand_groups(set(units) | add)
        prompt = self.routed_prompt(new, full_prompt)
        if prompt is None:
            detail["prompt"] = "not_pruned"
        return RU(units=new, full=False, tools=sorted(set(tools_for(new)) | set(self.always_tools)),
                  prompt=prompt, detail=detail)


# ---- builders ------------------------------------------------------------------------
class _Hybrid:
    """p = max(p_qwen_avg, p_tfidf) per intent, in INTENTS order."""

    def __init__(self, qwen_models, tfidf):
        self.qwen, self.tfidf, self.classes = qwen_models, tfidf, list(INTENTS)

    def predict_proba(self, texts):
        def ordered(m):
            P = m.predict_proba(texts)
            return P[:, [list(m.classes).index(c) for c in self.classes]]
        q = np.mean([ordered(m) for m in self.qwen], 0)
        return np.maximum(q, ordered(self.tfidf))


def _tfidf_traj():
    from .data import load, make_train
    from .router import TfidfLRRouter
    rows = load()
    tr = make_train(rows, lambda r: r["source"] == "tau_traj", seed=0)
    return TfidfLRRouter().fit([r["text"] for r in tr], [r["labels"] for r in tr])


def _qwen(adapter_dir, base="Qwen/Qwen3-0.6B", max_len=256):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    sys.path.insert(0, str(ROOT / "scripts"))
    from finetune import HFRouter, _disable_peft_torchao
    _disable_peft_torchao()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForSequenceClassification.from_pretrained(base, num_labels=len(INTENTS),
                                                           problem_type="multi_label_classification")
    m.config.pad_token_id = tok.pad_token_id
    m = PeftModel.from_pretrained(m, adapter_dir).to(device)
    return HFRouter(m, tok, device, max_len)


def build_router(kind: str = "tfidf", qwen_adapters=(), wiki: str | None = None, always_tools=()) -> TauRouter:
    frozen = json.loads((ROOT / "out/frozen_thresholds_agent_rules_proposal.json").read_text())
    wiki = wiki if wiki is not None else (ROOT / "data/taubench/wiki.md").read_text()
    tfidf = _tfidf_traj()
    if kind == "tfidf":
        f = frozen["tfidf_traj+agent_rules"]
        return TauRouter(tfidf, f["t_include"], f["gate"], wiki, name="tfidf_traj+agent_rules",
                         always_tools=always_tools)
    if kind == "hybrid":
        assert len(qwen_adapters) == 3, "hybrid uses the 3-seed Qwen average: pass 3 adapter dirs"
        f = frozen["hybrid_avg3+agent_rules"]
        return TauRouter(_Hybrid([_qwen(a) for a in qwen_adapters], tfidf), f["t_include"], f["gate"], wiki,
                         name="hybrid_avg3+agent_rules", always_tools=always_tools)
    raise ValueError(kind)
