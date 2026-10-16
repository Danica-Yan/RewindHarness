#!/usr/bin/env bash
# Round 2 of the real-failure-driven fix loop -- see docs/DEVLOG.zh-CN.md "真实失败归因
# round 1" for round 1 (the "modifying items must be done LAST" wiki patch,
# which turned out to not be statistically significant once tested with
# --num-trials 3 + paired bootstrap).
#
# Round 2 targets a DIFFERENT failure cluster, found by batch-classifying
# ALL 34 real baseline failures (not just the 3 round-1 targeted ones) with
# scripts/taubench_failure_report.py: the single largest cluster is
# WRONG_ACTION tasks where the agent calls the *right* tool with the
# *wrong* arguments. Within that cluster, the largest identifiable
# sub-pattern (tasks 93/95/99/111 in the original baseline file) is the
# agent confusing identifiers (order id / payment method id / address)
# across two similar orders or requests in the same multi-step
# conversation -- not a sequencing bug like round 1, and not something
# `scripts/taubench_round1_wiki.patch` touches (round 1 edits the "Modify
# items" section; round 2 edits the general policy section at the top).
#
# scripts/taubench_round2_wiki.patch applies ON TOP of round 1 (so this
# script always ensures round 1 is applied first, then applies round 2) --
# this tests the ACCUMULATED fix, not round 2 in isolation. If you want
# round 2's effect in isolation instead, revert round 1 first
# (`patch -R -p1 -d "$TAUBENCH_DIR" < scripts/taubench_round1_wiki.patch`)
# before running this script, and interpret the result as "round 2 alone
# vs totally unpatched", not "round 2 alone vs round 1".
#
# IMPORTANT: you do NOT need to re-run the unpatched baseline for this
# comparison -- your existing 3-trial unpatched multi-trial result file
# (pooled pass^1 72.17%, the one you already have) is still the correct
# "before" side, since round 2 doesn't change anything round 1's unpatched
# baseline already covered. Only run THIS script, then compare its output
# against that existing unpatched file.
#
# Usage:
#   export DEEPSEEK_API_KEY=sk-...
#   bash scripts/run_taubench_round2.sh 3   # 3 trials, same protocol as before
#
# Then:
#   python scripts/taubench_multitrial_compare.py \
#     /path/to/your/existing/results_multitrial_unpatched/<file>.json \
#     "$TAUBENCH_DIR/results_round2_patched/<new-file>.json" \
#     --label-a unpatched --label-b round2_patched
set -euo pipefail
cd "$(dirname "$0")/.."

TRIALS="${1:-3}"

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

if ! grep -q "modifying items must be done LAST" "$WIKI"; then
  echo "Applying round-1 patch first (round 2 builds on top of it)..."
  patch -p1 -d "$TAUBENCH_DIR" < scripts/taubench_round1_wiki.patch
else
  echo "Round-1 patch already applied."
fi

if ! grep -q "Never reuse an order id, item id, or payment method id" "$WIKI"; then
  echo "Applying round-2 patch..."
  patch -p1 -d "$TAUBENCH_DIR" < scripts/taubench_round2_wiki.patch
else
  echo "Round-2 patch already applied -- nothing to do."
fi

pip show tau_bench >/dev/null 2>&1 || pip install -e "$TAUBENCH_DIR"

LOG_DIR="results_round2_patched"
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
  --log-dir "$LOG_DIR"

echo
echo "Done. $TRIALS-trial round-2 (round1+round2 patched) results written under"
echo "  $TAUBENCH_DIR/$LOG_DIR/"
echo "Compare against your EXISTING unpatched multi-trial file (no need to re-run it):"
echo "  python scripts/taubench_multitrial_compare.py \\"
echo "    /path/to/your/results_multitrial_unpatched/<file>.json \\"
echo "    $TAUBENCH_DIR/$LOG_DIR/<new-file>.json \\"
echo "    --label-a unpatched --label-b round2_patched"
