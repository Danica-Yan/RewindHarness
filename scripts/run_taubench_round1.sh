#!/usr/bin/env bash
# Round 1 of the real-failure-driven fix loop: apply one concrete patch for a
# specific, reproducible bug found in the real DeepSeek failures (see
# flywheel/taubench_failures.py and docs/DEVLOG.zh-CN.md), then re-run the
# test split with exactly the same protocol as the baseline, writing to a new
# file (never overwriting the baseline results) so the two can be compared
# with scripts/taubench_round1_compare.py.
#
# The bug: tau-bench's retail policy (tau_bench/envs/retail/wiki.md) already
# says an order is locked after modify_pending_order_items is called, but only
# as a warning inside the "Modify items" subsection, not as an ordering rule at
# the decision point. Failed tasks 72/98/105 all modify an order's items first
# and then try to change its address, which must fail. The patch lifts that
# rule into the shared "Modify pending order" instructions as an explicit
# ordering constraint.
#
# Usage:
#   export DEEPSEEK_API_KEY=sk-...
#   bash scripts/run_taubench_round1.sh
#
# TAUBENCH_DIR (default ../tau-bench, same as run_taubench_deepseek.sh) must be
# the tau-bench clone the baseline was run from, so that apart from this one
# wiki.md change everything (code, task set, split order) is identical.
set -euo pipefail
cd "$(dirname "$0")/.."

TAUBENCH_DIR="${TAUBENCH_DIR:-../tau-bench}"
if [ ! -d "$TAUBENCH_DIR" ]; then
  echo "tau-bench not found at $TAUBENCH_DIR -- run scripts/run_taubench_deepseek.sh first (it clones it)." >&2
  exit 1
fi
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "DEEPSEEK_API_KEY is not set -- export it first." >&2
  exit 1
fi

WIKI="$TAUBENCH_DIR/tau_bench/envs/retail/wiki.md"
if grep -q "modifying items must be done LAST" "$WIKI"; then
  echo "wiki.md already patched -- skipping (idempotent, re-running run.py below)."
else
  echo "Applying scripts/taubench_round1_wiki.patch to $WIKI ..."
  patch -p1 -d "$TAUBENCH_DIR" < scripts/taubench_round1_wiki.patch
fi

pip show tau_bench >/dev/null 2>&1 || pip install -e "$TAUBENCH_DIR"

ROUND1_LOG_DIR="results_round1"
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
  --log-dir "$ROUND1_LOG_DIR" \
  "$@"

echo
echo "Done. Results written under $TAUBENCH_DIR/$ROUND1_LOG_DIR/ -- compare against your"
echo "baseline results file with:"
echo "  python scripts/taubench_round1_compare.py <baseline.json> $TAUBENCH_DIR/$ROUND1_LOG_DIR/<new-file>.json"
