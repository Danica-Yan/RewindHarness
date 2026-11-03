# intent_funnel — per-turn tool router for RewindHarness (standalone stage)

Lives in `experiments/intent_funnel/`. It is developed and evaluated offline here; it is not yet wired into the
rewind agent loop (see *Integration sketch*).

**Headline (held-out, thresholds frozen in advance):** on τ-bench retail conversations from models never used for
development (gpt-4o and Claude 3.5 Sonnet agents; 915 trajectories), the router keeps **99.7–100% of required
write-tool calls** while cutting the static prompt by **35–38%**. Router = fine-tuned Qwen3-0.6B (LoRA) ⊕ TF-IDF on user
turns + agent-proposal rules + route groups. End-to-end task success and billed cost are not measured yet.

Goal: put a cheap router in front of the DeepSeek agent on τ-bench retail. The router predicts which
intents a user turn contains, and the agent then gets **only the tools and policy sections those intents need**.
If the router isn't confident, it **falls back** to the full prompt. On τ-bench retail this works as
routing + context pruning, not as skipping LLM calls: almost every retail task needs tool calls.

Everything here runs offline. The only scripts that spend API budget are `deepseek_zeroshot.py`
(capped and cached) and, later, the end-to-end τ-bench rerun.

## Results so far

### Phase 0 — is pruning worth it? (`scripts/phase0_token_budget.py`, DeepSeek tokenizer)
| | tokens |
|---|---|
| full static prompt (wiki + 16 tool schemas) | **3,730** (wiki 1,205 · tools 2,525) |
| routed, single intent | 1,323 – 1,839 (−51% … −65%) |
| perfect routing, averaged over the real test task mix | **−51%** |

Tool schemas are two-thirds of the static prompt. 34/115 test tasks (30%) are **multi-intent**, so the
router has to be multi-label: its route is the union of tool sets.

Caveats before claiming dollar savings:
1. These numbers cover the *static* prefix. Each agent step also resends the growing conversation
   (tool outputs such as order/product JSON). Measure the per-step share on real trajectories.
2. DeepSeek bills cached prefix tokens at a lower rate. The full prompt is a perfectly stable prefix;
   a routed prompt is stable *per route*, so cache hits are spread across ~10 prefixes instead of 1.
   The end-to-end run has to report billed cost (cache hit/miss), not just token counts.

### Data findings
- **τ-bench retail train instructions are template-generated** (mean 8 words, e.g. *"Cancel order #W… because
  ordered by mistake."*). **Test instructions are human-written** (mean 69 words, 30% multi-intent). That leaves
  no natural-language in-domain training data and no natural valid split. Thresholds are therefore chosen by
  2-fold user-group cross-fitting on test (the model never trains on test; only two scalars are fitted, and every
  reported τ number is out-of-fold).
- **BANKING77-OOS `ood-oos/test` is byte-identical to CLINC150 `oos_test`.** It is counted once.
- No external dataset contains `exchange`, the largest τ test class. In-domain data has to come from
  τ trajectories or paraphrases.

### Real trajectories (your DeepSeek runs)
- train: `runs/p1/train_base.json` + `runs/p1/extend/v1_tasks_200-499.json` → 448 successful trajectories, 2,776 user turns
- test: `results_multitrial_unpatched/…_1016031023.json` (3 trials) → 249 successful trajectories, 1,597 turns, used as **replay**
- `tau-bench/historical_trajectories/*retail*` are official **test-split** runs: never train on them.
- About 55% of turns carry no new intent (identity details, confirmations, vague openers). They are labelled
  **`continue`**, meaning "keep the current route". Without this class every "my zip is 12345" would trigger a
  fallback, and in cumulative routing one fallback costs the rest of the conversation.

### Baselines + fine-tuned router — primary metric: replay
Replay = feed each real test trajectory turn by turn through the router with cumulative routing and check,
at every write call the agent actually made, whether that tool was in the route. This uses the agent's real
tool calls, **not** the weak labels. Thresholds are cross-fit over test tasks.

| router | route groups | write coverage | task coverage | static tokens / step | p50 latency |
|---|---|---|---|---|---|
| keyword rules | yes | 93.8% | 91.2% | 0.56 | 0.06 ms CPU |
| TF-IDF+LR, external + τ templates (no trajectories) | yes | 89.0% | 84.7% | 0.68 | 3 ms CPU |
| TF-IDF+LR, all data | **no** | 95.7% | 94.4% | 0.56 | 3 ms CPU |
| TF-IDF+LR, all data | yes | 98.3% | 97.6% | 0.60 | 3 ms CPU |
| **TF-IDF+LR, trajectory turns only** | yes | **99.4%** | **99.2%** | **0.62** | 3 ms CPU |
| Qwen3-0.6B + LoRA, all data, seed 0 | **no** | 94.3% | 92.8% | 0.52 | ~95 ms T4 |

**Three seeds, all with route groups** (`out/seed_summary.json`; TF-IDF seeds = training-sample seeds):

| router | write coverage (cross-fit) | static tokens / step | trajectories fell back | per-seed coverage |
|---|---|---|---|---|
| TF-IDF+LR, all data | 98.0% ± 0.4% | 0.603 | 11.0% | 98.3 / 98.3 / 97.5 |
| TF-IDF+LR, trajectory turns only | 99.4% ± 0.0% | 0.617 | 3.6% | 99.4 / 99.4 / 99.4 |
| Qwen3-0.6B LoRA, all data | 98.5% ± 0.9% | **0.568** | 3.2% | 99.4 / 98.9 / 97.3 |
| Qwen, 3-seed probability average | 99.2% | **0.563** | 0.8% | (one ensemble) |
| **Qwen (any single seed) ⊕ TF-IDF traj-only, p = max(p_qwen, p_tfidf)** | **99.4% for all 3 seeds** | **0.578–0.592** | 3.6–6.4% | 99.4 / 99.4 / 99.4 |

Reading:
- Qwen is consistently the cheapest (0.568 vs 0.60–0.62, ≈130–185 static tokens/step), but its coverage varies by seed
  (97.3–99.4%). The seed-0 result alone overstated it. Seed 2's misses include clear requests such as "cancel or return
  all my orders" → `cancel` missed, i.e. variance in rare-ish classes, not systematic ambiguity.
- TF-IDF trained on trajectories only is the most *stable* (99.4% for every seed) but costs more tokens.
- **Hybrid** (take the larger probability per intent): the stability of TF-IDF and most of Qwen's savings — 99.4% on
  every seed at 0.58–0.59 tokens, for +3 ms. Recommended configuration.
- Caveat: the hybrid rule was chosen after seeing these test results (only two combinations were tried, and thresholds
  stay cross-fit). Confirm it on data not used for any decision before quoting it as final.

### Held-out validation (thresholds frozen first)
`scripts/validate_heldout.py freeze` picked every router's thresholds on the DeepSeek test replay (target ≥99% write
coverage) and wrote `out/frozen_thresholds.json` (sha1 `30e0738974ac…`) **before** any held-out number was computed.
Held-out = tau-bench's published retail runs (`historical_trajectories/`): the same 115 test tasks, but **different user
simulator, agent and conversations** (gpt-4o: 278 successful trajectories / 381 writes; Claude 3.5 Sonnet: 637 / 946). This
tests robustness to phrasing, not to new tasks.

| router | gpt-4o: coverage / tokens | Sonnet: coverage / tokens | dev (DeepSeek) |
|---|---|---|---|
| rules | 90.8% / 0.54 | 92.2% / 0.54 | 93.6% / 0.56 |
| TF-IDF all data | 97.9% / 0.62 | 98.5% / 0.64 | 99.4% / 0.63 |
| TF-IDF trajectories only | 98.2% / 0.62 | 97.9% / 0.60 | 99.4% / 0.62 |
| Qwen seed 0 / 1 / 2 | 97.1 / 97.6 / 99.0% · 0.57 / 0.56 / 0.70 | 96.8 / 98.3 / 98.4% · 0.58 / 0.59 / 0.70 | 99.2 / 99.2 / 98.1% |
| Qwen 3-seed average | 97.9% / **0.56** | 97.6% / **0.57** | 99.2% / 0.56 |
| Hybrid: Qwen avg ⊕ TF-IDF traj | **98.2%** / 0.57 | 97.7% / 0.58 | 99.4% / 0.58 |

**Held-out reading (the honest headline):**
- Every router loses ~1–2 points of write coverage under a new user simulator. None holds the 99% dev level; the best
  sit at **~98%**.
- Qwen is **not** more robust than TF-IDF on held-out (97.6–97.9% vs 97.9–98.2%). What it keeps is the token saving
  (0.56–0.57 vs 0.60–0.62). The hybrid matches TF-IDF's coverage at Qwen-like cost. That is the defensible claim.
- The misses are largely **the same examples for all routers** (`scripts/heldout_misses.py`). Most are writes the user
  never asks for in words: "my card only has $1131 left" → the agent proposes cheaper items (`modify_items`); "I want to
  upgrade some stuff"; "can I apply my gift card balance to this order?" (`modify_payment`). A router that only reads
  user turns cannot see these, whatever the model. They need **the agent's proposal as router input** and an
  **escalation tool** as a safety net.
- Caveat: these held-out examples have now been inspected. Any fix designed from them must be confirmed on fresh
  conversations; the end-to-end run in rewind provides those by construction.

### Agent context: closing the user-text blind spot
`agent_before` (the agent's last message before each user turn) is now stored in replays. The router adds a
write intent when the agent proposes it. Configs and thresholds are frozen on the DeepSeek dev replay, then applied
unchanged to held-out.

| variant (hybrid = Qwen avg ⊕ TF-IDF traj) | dev cov / tokens | gpt-4o cov / tokens | Sonnet cov / tokens |
|---|---|---|---|
| user text only | 99.4% / 0.58 | 98.2% / 0.57 | 97.7% / 0.58 |
| + learned agent channel (TF-IDF on DeepSeek agent turns) | 99.7% / 0.58 | 98.2% / 0.57 | 97.7% / 0.58 |
| + agent keyword rules, whole message | 100% / 0.75 | 100% / 0.72 | 99.9% / 0.76 |
| **+ agent keyword rules, proposal sentences only** | **100% / 0.63** | **100% / 0.62** | **99.7% / 0.65** |

- The learned agent channel did not transfer. It was trained on DeepSeek's agent phrasing, and gpt-4o/Sonnet's clear
  proposals ("you can modify the order to remove this item") scored only 0.25–0.77.
- Keyword rules on agent text transfer because agents state actions explicitly. The patterns are the *existing*
  `MENTION` patterns from weak labelling, unchanged. Restricting them to proposal/confirmation sentences ("would you
  like", "shall I", "please confirm", "I can", "let me", …) drops the cost of policy chatter such as "this order can't be
  cancelled".
- Trade-off: user-only routing saves ~43% of the static prompt at ~98% write coverage. With agent proposals it saves
  **35–38% at 99.7–100%**. The second is the safer default.
- Caveat: the proposal-cue list is generic, but it was written after inspecting held-out misses. Confirm it end-to-end
  on fresh conversations.

**Route groups:** `modify_items`/`exchange` and order/account address are routed together (`taxonomy.ROUTE_GROUPS`).
The text often can't separate them: the difference is the order's status or which address is meant, and that is
only known after `get_order_details`. Error analysis of TF-IDF misses at high coverage (12 misses): 6 were order↔account
address confusion, 2 were modify↔exchange confusion, and 2 were users saying "exchange" for what was actually a pending-order
modification. Grouping costs ~0.04 tokens and lifts write coverage +2.6 pts. That design change is worth more than the
model choice.

**Fine-tuned Qwen3-0.6B:** without groups it does not beat TF-IDF (94.3% vs 95.7%). With groups (re-scored offline from
its saved probabilities, `scripts/eval_saved_probs.py`) it reaches 99.4% write coverage at 0.57 tokens. TF-IDF needs
0.62–0.67 for the same coverage. It also generalises better to human-written ABCD (macro-F1 0.79 vs 0.73). Known data gap:
**the τ train split has zero `modify_user_address` tasks**, so trajectories never teach it; groups cover it via the
order-address sibling. Latency (~95 ms on T4) is small next to a DeepSeek agent step (seconds), but the router does add
one model call per user turn.

### Hand audit of weak labels (done)
All 200 sampled train turns audited by hand (`out/audit_train_final.csv`; 4 rows adjusted for consistency, marked `claude:` in `note`).
- weak labels at audit time: **95.0%** correct (190/200)
- after the rule fixes the audit prompted: **97.5%** (195/200). The audit found three things:
  1. turn-0 lookback credited `modify_items` for "update my **address**" (tightened the pattern: 18 train rows fixed)
  2. two writes after the same user turn ("proceed with both") collapsed the second write's segment (12 rows fixed)
  3. user-simulator leaks: DeepSeek's tool-call markup (`<｜｜DSML｜｜ …>`) and role confusion (`\n\nuser` followed by a
     hallucinated agent reply) inside user turns. Now cleaned in `intent_funnel/textclean.py` and kept out of training:
     train 17 turns, test 1.
- remaining disagreements: "swap" without status cues (truth `modify_items+exchange`) and vague "update/change my order"
  openers (truth `continue`).

## Layout
```
intent_funnel/            # library (no API calls)
  taxonomy.py             # 10 intents, intent -> write tools + wiki sections, gold-action labelling
  label_maps.py           # ABCD / Bitext label -> intent (ambiguous mappings excluded on purpose)
  budget.py               # DeepSeek-token cost of a route
  router.py               # rules, TF-IDF+LR, SentenceUnion wrapper, shared decision rule
  metrics.py, evaluate.py # one protocol for every router (cross-fit thresholds, risk/coverage/token, ECE)
  data.py                 # loading + training-set construction (OOS cap, in-domain upsampling)
scripts/
  parse_taubench.py       # static (ast) parse of tau-bench tools/tasks — no repo code executed
  phase0_token_budget.py
  build_datasets.py       # ABCD, Bitext, CLINC150, BANKING77-OOS -> data/processed/external.jsonl
  build_tau.py            # tau tasks -> pseudo-utterances + style flag (terse/natural)
  extract_trajectories.py # YOUR tau result jsons -> replay fixture + per-turn weak labels (--from-replay to relabel)
  find_results.py         # locate tau-bench result jsons and identify their split
  make_audit.py / apply_audit.py  # hand-audit sheet for weak labels
  run_baselines.py        # primary metric: replay write coverage (intent_funnel/evaluate.py)
  finetune.py             # Qwen3-0.6B LoRA (or encoder) multi-label router; --smoke runs on CPU
  deepseek_zeroshot.py    # big-model router baseline; cached, capped, --dry-run
```

## Data kept out of git
Large files (>500 KB) stay local and are listed in `.gitignore`. All of them can be regenerated:

| file | how to regenerate |
|---|---|
| `data/processed/external.jsonl`, `tau_instr.jsonl` | `build_datasets.py` / `build_tau.py` (see *Reproduce*) |
| `data/processed/tau_traj_train.jsonl` | `extract_trajectories.py --split train --from-replay data/processed/tau_traj_train.jsonl data/processed/tau_traj_train.replay.jsonl`, then `apply_audit.py out/audit_train_final.csv data/processed/tau_traj_train.jsonl data/processed/tau_traj_train.replay.jsonl` (re-applies the 200 hand-audited labels) |
| `data/processed/agentctx/*.replay.jsonl` | `extract_trajectories.py` on the original DeepSeek result jsons (adds `agent_before`) |
| `data/heldout/*` | `extract_trajectories.py --split test data/heldout/<name>_retail.jsonl ext/tau-bench/historical_trajectories/<name>-retail.json` for `gpt-4o` and `sonnet-35-new` |

The replay files committed in `data/processed/tau_traj_{train,test}.replay.jsonl` are the source of truth for the
DeepSeek trajectories; the original result jsons are not in the repo.

## Reproduce
```bash
# data (GitHub only)
git clone --depth 1 https://github.com/sierra-research/tau-bench ext/tau-bench
git clone --depth 1 https://github.com/asappresearch/abcd ext/abcd
git clone --depth 1 https://github.com/bitext/customer-support-llm-chatbot-training-dataset ext/customer-support-llm-chatbot-training-dataset
git clone --depth 1 https://github.com/clinc/oos-eval ext/oos-eval
git clone --depth 1 https://github.com/jianguoz/Few-Shot-Intent-Detection ext/Few-Shot-Intent-Detection
pip install deepseek-tokenizer scikit-learn pandas
python scripts/parse_taubench.py ext/tau-bench data/taubench
python scripts/phase0_token_budget.py data/taubench out/phase0.json
python scripts/build_datasets.py ext data/processed
python scripts/build_tau.py data/taubench data/processed/tau_instr.jsonl
python scripts/run_baselines.py
```

## Next steps (in order)
1. ~~Turn trajectories into in-domain data~~ (done: 448 train / 249 test trajectories).
2. ~~Hand-audit weak labels~~ (done: 97.5% after fixes).
3. **Fine-tune on Kaggle (T4):** upload this folder *including* `data/processed/`, run `pip uninstall -y torchao`
   (Kaggle's preinstalled 0.10 breaks recent peft), then
   `python scripts/finetune.py --model Qwen/Qwen3-0.6B --seeds 0` (then `--seeds 0 1 2`). Encoder control:
   `--model answerdotai/ModernBERT-base --no-lora --lr 3e-5 --epochs 3`. Target: ≥99% replay write coverage at ≈0.6 tokens.
4. **DeepSeek router baseline:** `python scripts/deepseek_zeroshot.py --dry-run`, then `--sets tau_traj --max-calls ...`.
   Report its routing quality *and* its own overhead (one extra call per user turn).
5. **End-to-end** only once replay looks good: treatment arm on τ test with the existing multi-trial + bootstrap CI;
   the unpatched 3-trial run is the control (rerun one control trial first to check DeepSeek hasn't drifted).

## Integration into rewind (done; end-to-end run pending)
- `src/rewind/routing.py`: a generic `ToolRouter` protocol, the `request_full_toolset` escalation tool, and
  `RecordedRouter` for replay. rewind itself knows nothing about intents.
- `AgentRunner(router=...)`: on every new user turn the router updates the route. Each LLM call then gets only
  the routed tool schemas (+ escalation tool) and the routed system prompt; the history keeps the canonical
  prompt. Routes only grow within a run. The run falls back to the full toolset when the router is unsure, when
  the agent calls `request_full_toolset`, or when it calls a registered tool outside its route (the call still runs).
- Route state lives in `LoopState` (checkpoint/resume/fork carry it). Every decision is a `route` trace event.
  The routed prompt goes to the SnapshotStore, so `replay_run` re-serves recorded routes and needs no router
  model. Covered by `tests/test_routing.py`; with no router the loop is unchanged (full suite passes).
- `intent_funnel/rewind_router.py`: `TauRouter`, with thresholds from the frozen proposal-rules file.
  `build_router("tfidf")` is CPU-only; `build_router("hybrid", qwen_adapters=[3 dirs])` adds the Qwen avg.
  Only the wiki text inside the prompt is pruned, so Evolve rules are kept verbatim. Extra non-retail
  tools (e.g. the skill loader) are never hidden.
- `scripts/router_parity.py`: drives the live `TauRouter` through replays exactly as the loop does. It reproduces
  the offline numbers (dev 100% / 0.659, gpt-4o 100% / 0.666, Sonnet 99.8% / 0.672 vs offline 0.658 / 0.665 / 0.670).
  The escalation tool's schema adds ~72 tokens (0.019) per routed step on top of that.

### Running it
```bash
export DEEPSEEK_API_KEY=... REWIND_TAUBENCH_SRC=../tau-bench
# pilot (3 tasks): check traces for route events, escalations, crashes
python scripts/run_taubench_rewind.py --split test --task-ids 0-2 --router tfidf --out runs/router/pilot.json
# treatment arm
python scripts/run_taubench_rewind.py --split test --num-trials 1 --router tfidf --out runs/router/test_tfidf.json
# report vs the existing rewind control run (same harness)
python experiments/intent_funnel/scripts/router_e2e_report.py --control runs/p0/test_rewind.json \
    --treatment runs/router/test_tfidf.json --paired [--price-miss .. --price-hit .. --price-out ..]
python scripts/taubench_multitrial_compare.py runs/p0/test_rewind.json runs/router/test_tfidf.json
```

### Still open
- Flywheel: add a `misroute` failure label (escalation fired, or a failed task whose needed tool was missing
  from the route) so misroutes feed the router's next training round.
