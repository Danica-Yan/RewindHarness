"""Offline comparison across seeds (TF-IDF vs saved Qwen probabilities), all with route groups. Run from repo root."""
import sys, json, numpy as np, collections
sys.path.insert(0,'.')
import intent_funnel.router as rm
from intent_funnel.budget import PromptBudget
from intent_funnel.data import load, load_replays, make_train
from intent_funnel.evaluate import replay_crossfit, replay_probs, INCLUDES, GATES, replay_metrics
from intent_funnel.router import TfidfLRRouter, decide
from intent_funnel.taxonomy import ACTION_TO_INTENT, tools_for
from intent_funnel.textclean import clean
rm.GROUP_EXPANSION=True
B=PromptBudget('data/taubench'); R=load_replays(); rows=load()
def front_at(curve, target):
    ok=[v['step_token'] for v in curve.values() if v['write_coverage']>=target]
    return min(ok) if ok else None
res={}
for name,keep in [('tfidf_all',lambda r:True),('tfidf_traj',lambda r:r['source']=='tau_traj')]:
    out=[]
    for s in [0,1,2]:
        tr=make_train(rows,keep,seed=s)
        m=TfidfLRRouter().fit([r['text'] for r in tr],[r['labels'] for r in tr])
        pooled,picks,curve=replay_crossfit(m,R,B)
        out.append((pooled,curve))
    res[name]=out
class Saved:
    def __init__(s,P,c): s.P,s.classes=P,list(c)
    def predict_proba(s,t): return s.P
qs=[]
for s in [0,1,2]:
    z=np.load(f'out/qwen/seed{s}_probs_alldata.npz')
    pooled,picks,curve=replay_crossfit(Saved(z['replay'],z['classes']),R,B)
    qs.append((pooled,curve))
res['qwen_all']=qs
print(f"{'router':<12} {'write_cov (cross-fit)':<24} {'tokens':<16} {'fell back':<14} min tokens for >=98% / >=99% (in-sample)")
summary={}
for k,v in res.items():
    wc=[p['write_coverage'] for p,_ in v]; tk=[p['step_token'] for p,_ in v]; fb=[p['traj_fell_back'] for p,_ in v]
    f98=[front_at(c,0.98) for _,c in v]; f99=[front_at(c,0.99) for _,c in v]
    summary[k]=dict(write_cov=(np.mean(wc),np.std(wc)),tokens=(np.mean(tk),np.std(tk)),fell_back=(np.mean(fb),np.std(fb)),per_seed_cov=wc,f98=f98,f99=f99)
    print(f"{k:<12} {np.mean(wc):.1%} ± {np.std(wc):.1%}{'':<9} {np.mean(tk):.3f} ± {np.std(tk):.3f}   {np.mean(fb):.1%}          {[round(x,3) if x else None for x in f98]} / {[round(x,3) if x else None for x in f99]}   per-seed cov {[round(x,4) for x in wc]}")
json.dump({k:{kk:(list(vv) if isinstance(vv,(tuple,list)) else vv) for kk,vv in d.items()} for k,d in summary.items()},open('out/seed_summary.json','w'),indent=1,default=float)
# seed-2 Qwen misses at its pick
z=np.load('out/qwen/seed2_probs_alldata.npz'); flat=z['replay']; cls=list(z['classes']); P=[];k=0
for rp in R: n=len(rp['user_turns']); P.append(flat[k:k+n]); k+=n
miss=collections.Counter(); ex=[]
for rp,p in zip(R,P):
    route=set();per=[];full=False
    for d in decide(p,cls,t_include=0.05,gate=0.0):
        if d is None: full=True
        elif not full: route|=set(d)
        per.append(None if full else set(route))
    for w in rp['writes']:
        t=w['after_user_turn']; rt=per[t]
        if rt is None or w['tool'] in tools_for(sorted(rt)): continue
        it=ACTION_TO_INTENT[w['tool']]; miss[it]+=1
        ex.append((it,[clean(x)[0][:80] for x in rp['user_turns'][:t+1]][:3]))
print('qwen seed2 misses by intent:',miss)
for e in ex[:6]: print('  ',e)
