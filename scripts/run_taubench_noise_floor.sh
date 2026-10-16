#!/usr/bin/env bash
# Noise-floor control: rerun the exact same baseline protocol a SECOND time,
# with wiki.md reverted to its pristine (unpatched) state -- so the only
# thing different from the original baseline run is "it's a different API
# call", not "the prompt changed". Compare the two baseline runs against
# each other to see how much pass^1/task-level churn exists with literally
# nothing deliberately changed; that's the number you need before crediting
# scripts/run_taubench_round1.sh's +2 to the wiki.md patch rather than to
# plain run-to-run variance (temperature=0 does not guarantee identical
# outputs across calls, and a multi-turn conversation with an LLM user
# simulator can diverge from one differing token). See docs/DEVLOG.zh-CN.md
# (round 1 section) for why this matters.
#
# Usage:
#   export DEEPSEEK_API_KEY=sk-...
#   bash scripts/run_taubench_noise_floor.sh
set -euo pipefail
cd "$(dirname "$0")/.."

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
if grep -q "modifying items must be done LAST" "$WIKI"; then
  echo "wiki.md currently has the round-1 patch applied -- reverting it so this run is a true"
  echo "re-run of the ORIGINAL (unpatched) baseline, not a second round-1 run."
  patch -R -p1 -d "$TAUBENCH_DIR" < scripts/taubench_round1_wiki.patch
else
  echo "wiki.md is already unpatched -- good, nothing to revert."
fi

pip show tau_bench >/dev/null 2>&1 || pip install -e "$TAUBENCH_DIR"

NOISE_LOG_DIR="results_noise_floor"
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
  --log-dir "$NOISE_LOG_DIR" \
  "$@"

echo
echo "Done. Results written under $TAUBENCH_DIR/$NOISE_LOG_DIR/ -- compare against your"
echo "ORIGINAL baseline results file (not round 1) with:"
echo "  python scripts/taubench_compare.py <original-baseline.json> $TAUBENCH_DIR/$NOISE_LOG_DIR/<new-file>.json \\"
echo "    --label-a baseline --label-b \"baseline rerun (noise floor)\""
echo
echo "NOTE: wiki.md is left unpatched after this script. If you want to run round 1 again"
echo "afterwards, scripts/run_taubench_round1.sh re-applies the patch (it's idempotent)."
