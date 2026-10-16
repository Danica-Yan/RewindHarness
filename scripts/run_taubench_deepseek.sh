#!/usr/bin/env bash
# A real tau-bench baseline with DeepSeek -- using tau-bench's OWN run.py,
# not rewind's AgentRunner. tau-bench's run.py already drives the full
# multi-turn protocol (agent + LLM user simulator) via litellm, which has
# DeepSeek built in; reimplementing that inside rewind would mean solving
# the same mid-run-user-turn problem env/taubench_adapter.py's module
# docstring documents as out of scope, for a result tau-bench already
# produces correctly. See docs/DEVLOG.zh-CN.md (DeepSeek section) for the full
# reasoning (and for the DeepSeekProvider rewind DOES ship, for running
# rewind's own harness against the toy env with a real model).
#
# Usage:
#   export DEEPSEEK_API_KEY=sk-...
#   bash scripts/run_taubench_deepseek.sh                       # dev split, 20 tasks
#   bash scripts/run_taubench_deepseek.sh --task-split test     # extra args pass straight to run.py
#
# TAUBENCH_DIR (default ../tau-bench relative to this repo): where tau-bench
# is or should be cloned. Cloned automatically on first run if missing.
set -euo pipefail
cd "$(dirname "$0")/.."

TAUBENCH_DIR="${TAUBENCH_DIR:-../tau-bench}"
if [ ! -d "$TAUBENCH_DIR" ]; then
  echo "tau-bench not found at $TAUBENCH_DIR -- cloning it there now." >&2
  git clone https://github.com/sierra-research/tau-bench.git "$TAUBENCH_DIR"
fi
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "DEEPSEEK_API_KEY is not set -- export it first (get one at https://platform.deepseek.com)." >&2
  exit 1
fi

pip show tau_bench >/dev/null 2>&1 || pip install -e "$TAUBENCH_DIR"

cd "$TAUBENCH_DIR"
python run.py \
  --env retail \
  --task-split dev \
  --model deepseek-chat \
  --model-provider deepseek \
  --user-model deepseek-chat \
  --user-model-provider deepseek \
  --agent-strategy tool-calling \
  --max-concurrency 2 \
  "$@"
