#!/usr/bin/env bash
# Multi-trial rerun -- the real follow-up to "noise floor found a bigger
# effect than the round-1 patch" (see docs/DEVLOG.zh-CN.md, noise-floor section). A single trial's
# pass^1 on this task set swings by double digits just from run-to-run
# variance, so a real before/after comparison needs several trials per
# condition and a comparison of DISTRIBUTIONS, not two single numbers.
# tau-bench's own run.py already supports this natively via --num-trials:
# one results file with N trials per task (trial 0..N-1), instead of N
# separate files -- see scripts/taubench_multitrial_compare.py for how to
# read it.
#
# Usage:
#   export DEEPSEEK_API_KEY=sk-...
#   bash scripts/run_taubench_multitrial.sh unpatched 3   # baseline, 3 trials
#   bash scripts/run_taubench_multitrial.sh patched 3     # round-1 patch, 3 trials
#
# First argument: "patched" or "unpatched" -- which state scripts/
# taubench_round1_wiki.patch should be in for this run (applied/reverted
# before running, same idempotent logic as run_taubench_round1.sh /
# run_taubench_noise_floor.sh). Second argument: number of trials (default 3).
#
# Cost note: --num-trials 3 means 3x the API calls of a single run. Running
# both conditions at num-trials=3 is 6x the cost of the original baseline
# run you already did -- still cheap with DeepSeek's pricing, but worth
# knowing before kicking this off twice.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-}"
TRIALS="${2:-3}"
if [ "$MODE" != "patched" ] && [ "$MODE" != "unpatched" ]; then
  echo "Usage: $0 <patched|unpatched> [num_trials]" >&2
  exit 1
fi

TAUBENCH_DIR="${TAUBENCH_DIR:-../tau-bench}"
if [ ! -d "$TAUBENCH_DIR" ]; then
  echo "tau-bench not found at $TAUBENCH_DIR -- run scripts/run_taubench_deepseek.sh first." >&2
  exit 1
fi
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "DEEPSEEK_API_KEY is not set -- export it first." >&2
  exit 1
fi

WIKI="$TAUBENCH_DIR/tau_bench/envs/retail/wiki.md"
CURRENTLY_PATCHED=false
grep -q "modifying items must be done LAST" "$WIKI" && CURRENTLY_PATCHED=true

if [ "$MODE" = "patched" ] && [ "$CURRENTLY_PATCHED" = false ]; then
  echo "Applying round-1 patch..."
  patch -p1 -d "$TAUBENCH_DIR" < scripts/taubench_round1_wiki.patch
elif [ "$MODE" = "unpatched" ] && [ "$CURRENTLY_PATCHED" = true ]; then
  echo "Reverting round-1 patch (restoring original wiki.md)..."
  patch -R -p1 -d "$TAUBENCH_DIR" < scripts/taubench_round1_wiki.patch
else
  echo "wiki.md is already in the '$MODE' state -- nothing to change."
fi

pip show tau_bench >/dev/null 2>&1 || pip install -e "$TAUBENCH_DIR"

LOG_DIR="results_multitrial_${MODE}"
cd "$TAUBENCH_DIR"
python run.py \
  --env retail \
  --task-split test \
  --model deepseek-chat \
  --model-provider deepseek \
  --user-model deepseek-chat \
  --user-model-provider deepseek \
  --agent-strategy tool-calling \
  --max-concurrency 2 \
  --num-trials "$TRIALS" \
  --log-dir "$LOG_DIR" \
  "${@:3}"

echo
echo "Done. $TRIALS-trial results for '$MODE' written under $TAUBENCH_DIR/$LOG_DIR/."
echo "Run this for the OTHER mode too, then compare both multi-trial files with:"
echo "  python scripts/taubench_multitrial_compare.py \\"
echo "    $TAUBENCH_DIR/results_multitrial_unpatched/<file>.json \\"
echo "    $TAUBENCH_DIR/results_multitrial_patched/<file>.json"
