#!/usr/bin/env bash
# W8 CI regression gate: the small, fast, no-external-services regression set a
# PR should block on. Excludes the three dist/ test files and test_chaos.py
# wholesale (not just their infra-needing tests) so this gate never needs
# Postgres, Redis, or toxiproxy-server to be reachable — a PR's CI runner
# might not have any of them, and this gate should never flake because of
# that. The full set (including those) is `ci_nightly.sh`.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pytest -q \
  --ignore=tests/test_dist_pg.py \
  --ignore=tests/test_dist_gateway.py \
  --ignore=tests/test_dist_eventstream.py \
  --ignore=tests/test_chaos.py
