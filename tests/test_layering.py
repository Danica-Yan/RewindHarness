"""Import rules between layers, checked statically so a violation fails CI
the moment it is written, not when someone notices the coupling later.

    evolve/      depends on the Benchmark port, never on a benchmark plugin
    env/         plugins: never on orchestration (fork, replay, experiment)
                 or on anything above it (flywheel, evolve)
    experiment   generic over Benchmark: never on a plugin
    durability   checkpoint / coordination never on the agent runner

Relative and absolute imports are both resolved; imports inside functions
count too (a lazy import is still a dependency).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "rewind"


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imports(path: Path) -> set[str]:
    mod = _module_name(path)
    pkg = mod if path.name == "__init__.py" else mod.rsplit(".", 1)[0]
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")[: len(pkg.split(".")) - (node.level - 1)]
                target = ".".join(base + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            out.add(target)
            out.update(f"{target}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return {m for m in out if m.startswith("rewind")}


def _modules(prefix: str) -> list[Path]:
    p = SRC / prefix.replace(".", "/")
    return sorted(p.rglob("*.py")) if p.is_dir() else [p.with_suffix(".py")]


def _hits(imports: set[str], forbidden: tuple[str, ...]) -> list[str]:
    return sorted(i for i in imports
                  if any(i == f or i.startswith(f + ".") for f in forbidden))


RULES = [
    # (modules under, must not import, why)
    ("evolve", ("rewind.env", "rewind.flywheel.taubench_failures"),
     "evolve must reach a benchmark only through the Benchmark port"),
    ("experiment", ("rewind.env", "rewind.evolve", "rewind.flywheel"),
     "experiment is generic over Benchmark"),
    ("env", ("rewind.fork", "rewind.replay", "rewind.experiment", "rewind.worker",
             "rewind.flywheel", "rewind.evolve", "rewind.loop.AgentRunner"),
     "a plugin builds, restores and grades envs; it does not orchestrate runs"),
    ("benchmark", ("rewind.env", "rewind.evolve", "rewind.experiment", "rewind.fork", "rewind.replay"),
     "the port sits below everything that uses or implements it"),
    ("checkpoint", ("rewind.loop", "rewind.worker"), "durability does not depend on the runner"),
    ("coordination", ("rewind.loop", "rewind.worker", "rewind.checkpoint"),
     "coordination backends know nothing about runs"),
]


@pytest.mark.parametrize("under,forbidden,why", RULES, ids=[r[0] for r in RULES])
def test_layer_rule(under, forbidden, why):
    violations = {}
    for path in _modules(under):
        bad = _hits(_imports(path), forbidden)
        if bad:
            violations[str(path.relative_to(SRC))] = bad
    assert not violations, f"{why}:\n" + "\n".join(f"  {k}: {v}" for k, v in violations.items())


def test_evolve_has_no_benchmark_vocabulary():
    """An import rule cannot see a benchmark's names hard-coded as strings
    (the old `a.name != "respond"`). No string constant in evolve/ may equal
    one of tau-bench's tool or action names; anything benchmark-specific
    belongs behind the port (e.g. `Benchmark.gold_calls`)."""
    from rewind.env.taubench_adapter import MUTATING_TOOLS, RESPOND_ACTION_NAME, TERMINATE_TOOLS
    vocab = {RESPOND_ACTION_NAME, *MUTATING_TOOLS, *TERMINATE_TOOLS}
    try:
        from rewind.env.taubench_dialogue import build_dialogue_registry
        vocab |= set(build_dialogue_registry()._tools)  # noqa: SLF001
    except Exception:  # tau_bench not installed: the constants above still apply
        pass
    found = {}
    for path in _modules("evolve"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in vocab:
                found.setdefault(str(path.relative_to(SRC)), set()).add(node.value)
    assert not found, f"benchmark names hard-coded in evolve: {found}"


def test_tau_benchmark_implements_the_port():
    from rewind.benchmark import Benchmark
    from rewind.env.taubench_benchmark import TauBenchRetail
    assert isinstance(TauBenchRetail(), Benchmark)
