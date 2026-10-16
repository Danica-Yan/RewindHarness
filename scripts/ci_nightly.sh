#!/usr/bin/env bash
# W8 nightly full run: the full suite, including everything that needs real
# Postgres/Redis (dist/) and real kill -9 + real Toxiproxy (chaos). Meant to
# run somewhere those services are actually available (see README "Running the
# infrastructure tests" for setup); each of those test files skips itself with a
# clear reason if a service it needs isn't reachable, so this script never
# hard-fails just because of environment gaps — it fails only on an actual
# test failure.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pytest -q
