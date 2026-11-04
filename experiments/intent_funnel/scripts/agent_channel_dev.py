"""Develop the agent-proposal channel on the DeepSeek dev replay only (trained on train-split agent turns).
Compares user-only routers vs + agent channel, thresholds cross-fit, route groups on. Run from repo root."""
import sys
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import intent_funnel.router as rm
from intent_funnel.agent_channel import AgentChannel, combine
from intent_funnel.budget import PromptBudget
from intent_funnel.data import load_replays
from intent_funnel.evaluate import replay_crossfit
from validate_heldout import all_probs, cpu_models

rm.GROUP_EXPANSION = True
B = PromptBudget('data/taubench')
TR = load_replays('data/processed/agentctx/tau_traj_train.replay.jsonl')
DEV = load_replays('data/processed/agentctx/tau_traj_test.replay.jsonl')
ag = AgentChannel().fit(TR)
P_ag = ag.predict_proba([a for rp in DEV for a in rp['agent_before']])


class Saved:
    def __init__(s, P, c): s.P, s.classes = P, list(c)
    def predict_proba(s, t): return s.P


probs = all_probs(DEV, 'alldata', cpu_models())
print(f"{'router':<16} {'':<10} write_cov  task_cov  tokens  fell_back")
for name in ['tfidf_traj', 'qwen_avg3', 'hybrid_avg3']:
    P, cls = probs[name]
    for label, PP in [('user only', P), ('+ agent', combine(P, cls, P_ag))]:
        r, picks, _ = replay_crossfit(Saved(PP, cls), DEV, B, target=0.99)
        print(f"{name:<16} {label:<10} {r['write_coverage']:.1%}     {r['task_coverage']:.1%}    {r['step_token']:.3f}   {r['traj_fell_back']:.1%}   picks {picks}")
