"""Agent channel: freeze config + thresholds on the DeepSeek dev replay, then apply unchanged to held-out.
Config frozen from the dev sweep: class_weight=None, horizon=2, t_agent=0.8 (middle of the 0.7–0.9 plateau).
Run from repo root:  python scripts/agent_heldout.py freeze  |  python scripts/agent_heldout.py eval"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import intent_funnel.router as rm
import intent_funnel.agent_channel as acm
from intent_funnel.budget import PromptBudget
from intent_funnel.data import load_replays
from intent_funnel.evaluate import GATES, INCLUDES, replay_metrics
from validate_heldout import all_probs, cpu_models, split
rm.GROUP_EXPANSION = True
CFG = {"class_weight": None, "horizon": 2, "t_agent": 0.8}
FROZEN = Path('out/frozen_thresholds_agent.json')
B = PromptBudget('data/taubench')
acm.HORIZON = CFG['horizon']
ag = acm.AgentChannel(); ag.pipe.steps[-1][1].estimator.set_params(class_weight=CFG['class_weight'])
ag.fit(load_replays('data/processed/agentctx/tau_traj_train.replay.jsonl'))
ROUTERS = ['tfidf_traj', 'qwen_avg3', 'hybrid_avg3']


def with_agent(probs, replays):
    Pa = ag.predict_proba([a for rp in replays for a in rp['agent_before']])
    Pb = np.where(Pa >= CFG['t_agent'], 1.0, 0.0)
    out = {}
    for n in ROUTERS:
        P, cls = probs[n]
        out[n] = (P, cls)
        out[n + '+agent'] = (acm.combine(P, cls, Pb), cls)
    return out


def freeze():
    dev = load_replays('data/processed/agentctx/tau_traj_test.replay.jsonl')
    probs = with_agent(all_probs(dev, 'alldata', cpu_models()), dev)
    fr = {}
    for n, (P, cls) in probs.items():
        Ps = split(P, dev)
        grid = {(ti, g): replay_metrics(dev, Ps, cls, B, ti, g) for ti in INCLUDES for g in GATES}
        ok = [k for k, v in grid.items() if v['write_coverage'] >= 0.99]
        best = min(ok, key=lambda k: grid[k]['step_token']) if ok else max(grid, key=lambda k: grid[k]['write_coverage'])
        fr[n] = {'t_include': best[0], 'gate': best[1], 'dev': grid[best]}
        print(f"{n:<18} t_include={best[0]:<5} gate={best[1]:<4} dev cov {grid[best]['write_coverage']:.1%} tokens {grid[best]['step_token']:.3f}")
    FROZEN.write_text(json.dumps({'agent_cfg': CFG, 'routers': fr}, indent=1))


def evaluate():
    fr = json.loads(FROZEN.read_text())['routers']
    models = cpu_models(); rep = {}
    for tag in ['gpt-4o_retail', 'sonnet-35-new_retail']:
        R = load_replays(f'data/heldout/{tag}.replay.jsonl')
        probs = with_agent(all_probs(R, tag, models), R)
        print(f"\n== {tag}: {len(R)} trajectories, {sum(len(r['writes']) for r in R)} writes")
        rep[tag] = {}
        for n, f in fr.items():
            P, cls = probs[n]
            m = replay_metrics(R, split(P, R), cls, B, f['t_include'], f['gate'])
            rep[tag][n] = m
            print(f"{n:<18} cov {m['write_coverage']:.1%}  task {m['task_coverage']:.1%}  tokens {m['step_token']:.3f}  fell back {m['traj_fell_back']:.1%}")
    Path('out/heldout_report_agent.json').write_text(json.dumps(rep, indent=1))


if __name__ == '__main__':
    {'freeze': freeze, 'eval': evaluate}[sys.argv[1]]()
