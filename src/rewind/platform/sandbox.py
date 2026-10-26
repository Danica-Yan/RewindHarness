"""Running untrusted tool code in containers.

Trust tiers, chosen per tool:

  trusted    tools written with the platform (the toy retail tools, the
             MCP gateway's client side): run in-process, as before; access
             control and quotas still decide whether they run
  untrusted  code registered by tenants, or code an agent wrote: runs in a
             DockerSandbox container
  hostile    same, with `SandboxPolicy(runtime="runsc")` -- gVisor puts a
             user-space kernel between the code and the host; same interface,
             one field

What a container gets (SandboxPolicy.container_flags):

  * no network at all (--network none)
  * read-only root filesystem; a small tmpfs at /tmp (noexec, nosuid)
  * an unprivileged user, every capability dropped, no-new-privileges
  * memory (no swap on top), CPU and process-count limits; a wall-clock
    timeout after which the container is killed
  * exactly ONE host directory: the calling tenant's workspace, at /work.
    Another tenant's files are not hidden by a rule -- they are not in the
    container at all
  * secrets only for the call that needs them, handed to `docker exec` by
    name through the CLI's environment: not in argv, not in the container's
    config (`docker inspect` does not show them)

Pool: containers can be started ahead of time per tenant ("warm"), but each
one runs exactly one call and is then destroyed; a replacement starts in the
background. That buys the start-up latency back without letting two calls --
even of the same tenant -- share /tmp or a leftover process.

Which tenant? A tool function only sees (env, args); the agent loop publishes
the caller in `platform.context` and `sandboxed_tool` reads it there. No
caller identity, no execution (fail closed).

Results are external to the env snapshot (they depend on workspace files and
the container), so sandboxed tools are marked `external`: replay serves the
recorded outcome and never starts a container.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from ..tools import Tool, ToolError
from .context import current_call

_TENANT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}")
LABEL = "rewind.sandbox"


class SandboxPolicy(BaseModel):
    image: str = "rewind-sandbox:py"
    python: str = "/usr/bin/python3"
    runtime: str | None = None            # e.g. "runsc" (gVisor) for hostile code
    memory_mb: int = Field(default=256, gt=0)
    cpus: float = Field(default=1.0, gt=0)
    pids: int = Field(default=64, gt=0)
    tmpfs_mb: int = Field(default=64, gt=0)
    timeout_s: float = Field(default=10.0, gt=0)
    max_output_bytes: int = Field(default=64 * 1024, gt=0)
    user: str = "65534:65534"             # nobody

    def container_flags(self) -> list[str]:
        f = ["--network", "none", "--ipc", "none", "--read-only",
             "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
             "--user", self.user,
             "--memory", f"{self.memory_mb}m", "--memory-swap", f"{self.memory_mb}m",
             "--cpus", f"{self.cpus:g}", "--pids-limit", str(self.pids),
             # mode=1777: otherwise the tmpfs is root-owned and the sandbox user cannot use it
             "--tmpfs", f"/tmp:rw,size={self.tmpfs_mb}m,noexec,nosuid,nodev,mode=1777",
             "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1"]
        if self.runtime:
            f += ["--runtime", self.runtime]
        return f


class Workspaces:
    """One directory per tenant under `root`, owned by the sandbox user."""

    def __init__(self, root: str | Path, uid: int = 65534, gid: int = 65534) -> None:
        self.root = Path(root)
        self.uid, self.gid = uid, gid

    def path(self, tenant: str) -> Path:
        if not _TENANT.fullmatch(tenant):
            raise ValueError(f"invalid tenant id {tenant!r}")
        if not self.root.exists():
            self.root.mkdir(parents=True)
        self.root.chmod(0o700)              # other host users cannot reach any tenant's directory
        p = self.root / tenant
        if not p.exists():
            p.mkdir()
            if os.geteuid() == 0:
                os.chown(p, self.uid, self.gid)
                p.chmod(0o700)
            else:
                # Not root (e.g. a CI runner): cannot hand the directory to the
                # sandbox user, so open the leaf. The container mounts the leaf
                # directly; host users are still stopped by the 0700 root.
                p.chmod(0o777)
        return p


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    killed: bool              # SIGKILL other than our timeout: in practice the memory limit
    duration_ms: float
    warm: bool                # came from the pre-started pool


class DockerSandbox:
    def __init__(self, policy: SandboxPolicy, workspaces: Workspaces, *, pool_size: int = 0,
                 docker: str = "docker") -> None:
        self.policy = policy
        self.workspaces = workspaces
        self.pool_size = pool_size
        self.docker = docker
        self.session = uuid.uuid4().hex[:8]
        self._pools: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        self._bg = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sandbox-bg")
        self._closed = False

    # ---- containers --------------------------------------------------------------
    def _cli(self, *args: str, timeout: float = 30, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run([self.docker, *args], capture_output=True, text=True, timeout=timeout, check=check)

    def _create(self, tenant: str) -> str:
        name = f"rewind-sbx-{tenant.lower()[:20]}-{uuid.uuid4().hex[:10]}"
        self._cli("run", "-d", "--name", name, "--label", f"{LABEL}={self.session}",
                  "--label", f"rewind.tenant={tenant}", *self.policy.container_flags(),
                  "-v", f"{self.workspaces.path(tenant)}:/work:rw", "-w", "/work",
                  self.policy.image, self.policy.python, "-c", "import time; time.sleep(10**9)")
        return name

    def _destroy(self, name: str) -> None:
        self._cli("rm", "-f", name, check=False)

    def _refill(self, tenant: str) -> None:
        if self._closed:
            return
        try:
            name = self._create(tenant)
        except Exception:                                   # noqa: BLE001 -- the next acquire starts one cold
            return
        with self._lock:
            if self._closed:
                self._bg.submit(self._destroy, name)
                return
            self._pools.setdefault(tenant, []).append(name)

    def warm(self, tenant: str) -> None:
        """Fill `tenant`'s pool now (blocking), e.g. when a tenant's agent starts."""
        with self._lock:
            need = self.pool_size - len(self._pools.get(tenant, []))
        for _ in range(max(0, need)):
            self._refill(tenant)

    def _acquire(self, tenant: str) -> tuple[str, bool]:
        with self._lock:
            pool = self._pools.get(tenant, [])
            name = pool.pop() if pool else None
        if self.pool_size:
            self._bg.submit(self._refill, tenant)
        return (name, True) if name else (self._create(tenant), False)

    # ---- execution -----------------------------------------------------------------
    def run(self, tenant: str, code: str, stdin: str = "", env: dict[str, str] | None = None) -> SandboxResult:
        """Run `code` (Python) in a fresh container of `tenant`, `stdin` on
        its standard input, `env` set for this call only. The container is
        destroyed afterwards, whatever happened."""
        if self._closed:
            raise RuntimeError("sandbox is closed")
        env = env or {}
        bad = [k for k in env if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", k)]
        if bad:
            raise ValueError(f"invalid env names {bad}")
        t0 = time.perf_counter()                # includes getting a container: that is the point of the pool
        name, warm = self._acquire(tenant)
        cmd = [self.docker, "exec", "-i"]
        for k in env:
            cmd += ["-e", k]                    # value comes from the CLI's own environment, not argv
        cmd += [name, self.policy.python, "-c", code]
        cli_env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **{k: v for k, v in os.environ.items()
                                                                        if k.startswith("DOCKER_")}, **env}
        timed_out = False
        try:
            p = subprocess.run(cmd, input=stdin.encode(), capture_output=True, timeout=self.policy.timeout_s,
                               env=cli_env)
            code_, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired as e:
            timed_out = True
            self._cli("kill", name, check=False)
            code_, out, err = 137, e.stdout or b"", e.stderr or b""
        finally:
            self._bg.submit(self._destroy, name)
        cap = self.policy.max_output_bytes
        return SandboxResult(
            exit_code=code_, stdout=_cap(out, cap), stderr=_cap(err, cap), timed_out=timed_out,
            killed=code_ == 137 and not timed_out, duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            warm=warm)

    def close(self) -> None:
        """Destroy every container this sandbox started. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pools.clear()
        self._bg.shutdown(wait=True)
        ids = self._cli("ps", "-aq", "--filter", f"label={LABEL}={self.session}", check=False).stdout.split()
        if ids:
            self._cli("rm", "-f", *ids, check=False)


def _cap(b: bytes, n: int) -> str:
    s = b[:n].decode("utf-8", errors="replace")
    return s + (f"\n[truncated: {len(b)} bytes total]" if len(b) > n else "")


def docker_available(image: str) -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# as a rewind tool
# ---------------------------------------------------------------------------
def sandboxed_tool(name: str, description: str, args_model: type[BaseModel], code: str,
                   sandbox: DockerSandbox, *, mutating: bool = False,
                   secrets: dict[str, str] | None = None, resolve_secret=None) -> Tool:
    """A tool whose implementation is `code`, run in the caller's tenant
    sandbox. The code reads its arguments as JSON from stdin and prints its
    result. `secrets` maps an env var name to a secret reference, resolved per
    call with `resolve_secret(ref)`."""
    secrets = secrets or {}

    def fn(_env, args: BaseModel) -> str:
        ctx = current_call()
        if ctx is None or ctx.principal is None:
            raise ToolError(f"{name} runs in a tenant sandbox and needs a caller identity")
        env = {k: resolve_secret(ref) for k, ref in secrets.items()} if secrets else {}
        r = sandbox.run(ctx.principal.tenant_id, code, stdin=args.model_dump_json(), env=env)
        if r.timed_out:
            raise ToolError(f"{name} exceeded its {sandbox.policy.timeout_s:g}s time limit")
        if r.killed:
            raise ToolError(f"{name} was killed (memory limit of {sandbox.policy.memory_mb} MB)")
        if r.exit_code != 0:
            tail = r.stderr.strip().splitlines()[-1:] or [f"exit code {r.exit_code}"]
            raise ToolError(f"{name} failed: {tail[0][:300]}")
        return r.stdout

    return Tool(name=name, description=description, args_model=args_model, fn=fn, mutating=mutating,
                raw_text_result=True, external=True)
