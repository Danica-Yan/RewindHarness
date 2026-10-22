"""Batch execution of real tau-bench episodes under a given ContextVersion.

Shared by scripts/run_taubench_rewind.py (baselines) and the evolution loop
(gate runs, re-runs after a commit). Results are written in tau-bench's
run.py format after every episode, and finished (task, trial) pairs are
skipped on restart, so an interrupted batch resumes where it stopped.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable

from ..env.taubench_dialogue import (
    build_dialogue_registry, faithful_config, result_record, run_episode, wiki,
)
from ..snapshot import SnapshotStore
from .surface import ContextVersion


def version_setup(ctx: ContextVersion):
    """(AgentConfig, ToolRegistry) for episodes under `ctx`."""
    cfg = faithful_config(system_prompt=ctx.render_system_prompt(wiki()))
    return cfg, ctx.patch_registry(build_dialogue_registry())


def load_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def write_rows(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(sorted(rows, key=lambda r: (r["trial"], r["task_id"])),
                              ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


async def run_batch(split: str, task_ids: list[int], trials: int, out: str | Path, *,
                    ctx: ContextVersion, provider: Any, user_llm: Any, store: SnapshotStore,
                    trace_dir: str | Path | None = None, concurrency: int = 4,
                    log: Callable[[str], None] = print) -> list[dict]:
    out = Path(out)
    trace_dir = Path(trace_dir or out.with_suffix("").as_posix() + "_traces")
    rows = load_rows(out)
    done = {(r["task_id"], r["trial"]) for r in rows}
    jobs = [(t, i) for t in range(trials) for i in task_ids if (i, t) not in done]
    log(f"{out.name}: {len(jobs)} episodes to run ({len(done)} already done)")
    cfg, registry = version_setup(ctx)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    t0 = time.time()

    async def one(trial: int, idx: int) -> None:
        async with sem:
            name = f"{split}-{idx}-t{trial}" + (f"-v{ctx.version}" if ctx.version else "")
            path = trace_dir / f"{name}.jsonl"
            try:
                ep = await run_episode(split, idx, provider=provider, user_llm=user_llm, store=store,
                                       cfg=cfg, registry=registry, run_id=name, trace_path=str(path))
            except Exception as e:  # one broken episode must not kill the batch
                log(f"[error] {name}: {type(e).__name__}: {e}")
                return
            row = result_record(ep.env, ep.result, trial, str(path), ep.events)
            row["info"]["rewind"]["ctx_version"] = ctx.version
            async with lock:
                rows.append(row)
                write_rows(out, rows)
            log(f"[{len(rows):4d}] {name}: reward={row['reward']:.0f} status={ep.result.status} "
                f"steps={ep.result.steps} ({time.time() - t0:.0f}s)")

    await asyncio.gather(*(one(t, i) for t, i in jobs))
    return rows
