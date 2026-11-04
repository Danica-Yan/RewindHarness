"""Categorise held-out misses at frozen thresholds and check overlap between routers. Run from repo root."""
import json, sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import intent_funnel.router as rm
from intent_funnel.data import load_replays
from intent_funnel.router import decide
from intent_funnel.taxonomy import ACTION_TO_INTENT, tools_for
from intent_funnel.textclean import clean
from extract_trajectories import MENTION
from validate_heldout import all_probs, cpu_models, split
rm.GROUP_EXPANSION = True
frozen = json.loads(Path('out/frozen_thresholds.json').read_text())['routers']
models = cpu_models()
for tag in ['gpt-4o_retail', 'sonnet-35-new_retail']:
    R = load_replays(f'data/heldout/{tag}.replay.jsonl')
    probs = all_probs(R, tag, models)
    missed = {}
    for name in ['tfidf_traj', 'qwen_avg3', 'hybrid_avg3']:
        P, cls = probs[name]; f = frozen[name]; S = set()
        for i, (rp, p) in enumerate(zip(R, split(P, R))):
            route, full, per = set(), False, []
            for d in decide(p, cls, t_include=f['t_include'], gate=f['gate']):
                if d is None: full = True
                elif not full: route |= set(d)
                per.append(None if full else set(route))
            for wi, w in enumerate(rp['writes']):
                rt = per[w['after_user_turn']]
                if not (rt is None or w['tool'] in tools_for(sorted(rt))): S.add((i, wi))
        missed[name] = S
    allm = set.union(*missed.values()); both = set.intersection(*missed.values())
    print(f'\n== {tag}: misses ' + ', '.join(f'{k} {len(v)}' for k, v in missed.items()) + f'; missed by all three: {len(both)}')
    cats = collections.Counter(); ex = collections.defaultdict(list)
    for (i, wi) in sorted(missed['hybrid_avg3']):
        rp = R[i]; w = rp['writes'][wi]; it = ACTION_TO_INTENT[w['tool']]
        turns = [clean(t)[0] for t in rp['user_turns'][:w['after_user_turn'] + 1]]
        c = 'user never states it (keyword-free)' if not any(MENTION[it].search(t) for t in turns) else 'stated but missed'
        cats[(c, it)] += 1
        ex[c].append((it, [t[:85] for t in turns if (MENTION[it].search(t) if c.startswith('stated') else True)][:2]))
    for k, v in sorted(cats.items(), key=lambda x: -x[1]): print('   ', v, k)
    for c, e in ex.items():
        print('  --', c)
        for x in e[:4]: print('     ', x)
