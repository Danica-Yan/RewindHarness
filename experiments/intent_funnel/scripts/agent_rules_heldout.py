"""Agent channel v2: the pre-existing MENTION keyword patterns (written for weak labelling, before any
held-out data was touched; unchanged) applied to the agent's message before each user turn.
Thresholds frozen on the DeepSeek dev replay, then applied unchanged to held-out. Run from repo root."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import intent_funnel.router as rm
from intent_funnel.agent_channel import combine
from intent_funnel.budget import PromptBudget
from intent_funnel.data import load_replays
from intent_funnel.evaluate import GATES, INCLUDES, replay_metrics
from intent_funnel.taxonomy import WRITE_INTENTS
from extract_trajectories import MENTION
from validate_heldout import all_probs, cpu_models, split
rm.GROUP_EXPANSION = True
B = PromptBudget('data/taubench')
ROUTERS = ['tfidf_traj', 'qwen_avg3', 'hybrid_avg3']


import re
PROPOSAL = re.compile(r"\b(would you like|shall i|should i|do you want|please confirm|confirm (that|if|whether)|"
                      r"i can|we can|you can|let me|i'?ll|can i)\b", re.I)
SENT = re.compile(r"(?<=[.!?\n])\s+")
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"   # all | proposal


def agent_rules(replays):
    rows = []
    for rp in replays:
        for a in rp['agent_before']:
            text = a
            if MODE == "proposal" and a:
                text = " ".join(x for x in SENT.split(a) if PROPOSAL.search(x))
            rows.append([1.0 if (text and MENTION[c].search(text)) else 0.0 for c in WRITE_INTENTS])
    return np.array(rows)


def build(R, tag, models):
    probs = all_probs(R, tag, models); Pa = agent_rules(R); out = {}
    for n in ROUTERS:
        P, cls = probs[n]; out[n] = (P, cls); out[n + '+agent_rules'] = (combine(P, cls, Pa), cls)
    return out


models = cpu_models()
dev = load_replays('data/processed/agentctx/tau_traj_test.replay.jsonl')
fr = {}
print('== dev (DeepSeek): thresholds frozen here')
for n, (P, cls) in build(dev, 'alldata', models).items():
    Ps = split(P, dev)
    grid = {(ti, g): replay_metrics(dev, Ps, cls, B, ti, g) for ti in INCLUDES for g in GATES}
    ok = [k for k, v in grid.items() if v['write_coverage'] >= 0.99]
    best = min(ok, key=lambda k: grid[k]['step_token']) if ok else max(grid, key=lambda k: grid[k]['write_coverage'])
    fr[n] = {'t_include': best[0], 'gate': best[1], 'dev': grid[best]}
    print(f"{n:<24} cov {grid[best]['write_coverage']:.1%} tokens {grid[best]['step_token']:.3f}  (t_include={best[0]}, gate={best[1]})")
Path(f'out/frozen_thresholds_agent_rules_{MODE}.json').write_text(json.dumps(fr, indent=1))
rep = {}
for tag in ['gpt-4o_retail', 'sonnet-35-new_retail']:
    R = load_replays(f'data/heldout/{tag}.replay.jsonl'); rep[tag] = {}
    print(f'\n== {tag}')
    for n, (P, cls) in build(R, tag, models).items():
        m = replay_metrics(R, split(P, R), cls, B, fr[n]['t_include'], fr[n]['gate']); rep[tag][n] = m
        print(f"{n:<24} cov {m['write_coverage']:.1%}  task {m['task_coverage']:.1%}  tokens {m['step_token']:.3f}  fell back {m['traj_fell_back']:.1%}")
Path(f'out/heldout_report_agent_rules_{MODE}.json').write_text(json.dumps(rep, indent=1))
