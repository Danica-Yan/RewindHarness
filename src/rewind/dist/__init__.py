"""Real-infrastructure implementations (plan section 4.4/4.5/5, week 5).

Everything here needs a real Postgres and/or Redis reachable at runtime (see
each module's docstring for the DSN/URL it expects) — these are NOT exercised
by the default offline test suite. `tests/test_dist_*.py` skip themselves
with a clear reason when the service isn't reachable, so `pytest -q` stays
green offline; run them explicitly once Postgres/Redis are up to exercise
this package.
"""
