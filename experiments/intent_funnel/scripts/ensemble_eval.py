"""Offline: Qwen seed ensemble and Qwen ⊕ TF-IDF (max-probability) hybrid, from saved probabilities. Run from repo root."""
import sys, numpy as np
sys.path.insert(0,'.')
import intent_funnel.router as rm
from intent_funnel.budget import PromptBudget
from intent_funnel.data import load, load_replays, make_train
from intent_funnel.evaluate import replay_crossfit, replay_probs
from intent_funnel.router import TfidfLRRouter
rm.GROUP_EXPANSION=True
B=PromptBudget('data/taubench'); R=load_replays(); rows=load()
class Saved:
    def __init__(s,P,c): s.P,s.classes=P,list(c)
    def predict_proba(s,t): return s.P
Z=[np.load(f'out/qwen/seed{s}_probs_alldata.npz') for s in range(3)]
cls=list(Z[0]['classes'])
Q=np.mean([z['replay'] for z in Z],0)
tr=make_train(rows,lambda r:r['source']=='tau_traj',seed=0)
m=TfidfLRRouter().fit([r['text'] for r in tr],[r['labels'] for r in tr])
T=np.vstack(replay_probs(m,R)); T=T[:,[m.classes.index(c) for c in cls]]
def show(name,P):
    p,picks,curve=replay_crossfit(Saved(P,cls),R,B)
    f99=[v['step_token'] for v in curve.values() if v['write_coverage']>=0.99]
    print(f"{name:<34} cov {p['write_coverage']:.1%}  tokens {p['step_token']:.3f}  fell back {p['traj_fell_back']:.1%}  | min tokens for >=99% in-sample: {min(f99) if f99 else None}")
show('qwen 3-seed average',Q)
for s in range(3):
    show(f'qwen seed{s} ⊕ tfidf_traj (max)',np.maximum(Z[s]['replay'],T))
show('qwen avg ⊕ tfidf_traj (max)',np.maximum(Q,T))
