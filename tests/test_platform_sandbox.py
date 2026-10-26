"""Platform P4: running untrusted tool code in containers.

The policy tests need nothing. The rest start real containers and skip when
Docker or the sandbox image is unavailable (build it with
`docker build -t rewind-sandbox:py -f docker/sandbox.Dockerfile .`, or set
REWIND_SANDBOX_IMAGE). Each container test is an attack the sandbox must
stop, run for real.
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest
from conftest import call, run, text
from pydantic import BaseModel
from test_platform import RULES, ev, go, platform_registry, principal, tool_msgs

from rewind import replay_run
from rewind.env import ToyRetailEnv
from rewind.platform import CapabilityDescriptor
from rewind.platform.sandbox import (DockerSandbox, SandboxPolicy, Workspaces, docker_available,
                                     sandboxed_tool)

IMAGE = os.environ.get("REWIND_SANDBOX_IMAGE", "rewind-sandbox:py")
PYTHON = os.environ.get("REWIND_SANDBOX_PYTHON", "/usr/bin/python3")
needs_docker = pytest.mark.skipif(not docker_available(IMAGE), reason=f"needs Docker and image {IMAGE}")


# ---------------------------------------------------------------------------
# policy (no Docker)
# ---------------------------------------------------------------------------
def test_container_flags_deny_by_default():
    f = SandboxPolicy(memory_mb=128, pids=32).container_flags()
    pairs = {(f[i], f[i + 1]) for i in range(len(f) - 1)}
    for want in [("--network", "none"), ("--cap-drop", "ALL"), ("--security-opt", "no-new-privileges"),
                 ("--user", "65534:65534"), ("--memory", "128m"), ("--memory-swap", "128m"),
                 ("--pids-limit", "32")]:
        assert want in pairs, want
    assert "--read-only" in f and any(x.startswith("/tmp:rw,") and "noexec" in x for x in f)
    assert "--runtime" not in f
    assert ("--runtime", "runsc") in {(g[i], g[i + 1]) for g in [SandboxPolicy(runtime="runsc").container_flags()]
                                      for i in range(len(g) - 1)}


def test_workspace_ids_cannot_escape(tmp_path):
    ws = Workspaces(tmp_path)
    assert ws.path("t1") == tmp_path / "t1"
    for bad in ["../t2", "t1/../../etc", "", "a" * 80]:
        with pytest.raises(ValueError):
            ws.path(bad)


@pytest.mark.parametrize("euid,leaf_mode", [(0, 0o700), (1000, 0o777)])
def test_workspace_permissions(tmp_path, monkeypatch, euid, leaf_mode):
    monkeypatch.setattr(os, "geteuid", lambda: euid)
    monkeypatch.setattr(os, "chown", lambda *a: None)
    ws = Workspaces(tmp_path / "ws")
    leaf = ws.path("t1")
    assert (tmp_path / "ws").stat().st_mode & 0o777 == 0o700     # host users stop here either way
    assert leaf.stat().st_mode & 0o777 == leaf_mode


def test_secret_values_never_reach_argv(tmp_path, monkeypatch):
    seen = []
    real = subprocess.run

    def spy(cmd, *a, **kw):
        seen.append((list(cmd), dict(kw.get("env") or {})))
        if cmd[1] == "exec":
            return subprocess.CompletedProcess(cmd, 0, b"ok", b"")
        return subprocess.CompletedProcess(cmd, 0, "cid", "")
    monkeypatch.setattr(subprocess, "run", spy)
    sb = DockerSandbox(SandboxPolicy(), Workspaces(tmp_path))
    sb.run("t1", "print(1)", env={"API_TOKEN": "s3cr3t"})
    monkeypatch.setattr(subprocess, "run", real)
    exec_cmd, exec_env = next((c, e) for c, e in seen if c[1] == "exec")
    assert "s3cr3t" not in " ".join(exec_cmd) and ["-e", "API_TOKEN"] == exec_cmd[3:5]
    assert exec_env["API_TOKEN"] == "s3cr3t"
    assert all("s3cr3t" not in " ".join(c) for c, _ in seen)


# ---------------------------------------------------------------------------
# real containers
# ---------------------------------------------------------------------------
@pytest.fixture
def sandbox(tmp_path):
    sb = DockerSandbox(SandboxPolicy(image=IMAGE, python=PYTHON, memory_mb=128, pids=32, timeout_s=3),
                       Workspaces(tmp_path / "ws"))
    yield sb
    sb.close()


def py(sb, code, tenant="t1", **kw):
    return sb.run(tenant, code, **kw)


@needs_docker
def test_runs_unprivileged_with_no_capabilities(sandbox):
    r = py(sandbox, "import os; print(os.getuid(), open('/proc/self/status').read().split('CapEff:')[1].split()[0])")
    assert r.stdout.split() == ["65534", "0000000000000000"]


@needs_docker
def test_no_network(sandbox):
    # Not "a connection failed" -- this host's own egress filter could cause that.
    # The container must have no interface but loopback at all.
    r = py(sandbox, "import os, socket\nprint(sorted(os.listdir('/sys/class/net')))\ntry:\n"
                    " socket.create_connection(('1.1.1.1', 53), 2); print('CONNECTED')\n"
                    "except OSError as e: print('errno', e.errno)")
    assert r.stdout.split("\n")[:2] == ["['lo']", "errno 101"]          # ENETUNREACH: no route anywhere


@needs_docker
def test_root_filesystem_is_read_only_and_tmp_is_noexec(sandbox):
    r = py(sandbox, "\ntry:\n open('/usr/evil','w'); print('WROTE')\nexcept OSError as e: print(e.strerror)")
    assert r.stdout.strip() == "Read-only file system"
    r = py(sandbox, "import os, subprocess, shutil\nshutil.copy('/usr/bin/python3', '/tmp/p')\nos.chmod('/tmp/p', 0o755)\n"
                    "try:\n subprocess.run(['/tmp/p', '-c', 'print(1)']); print('EXECUTED')\n"
                    "except PermissionError: print('noexec')")
    assert r.stdout.strip() == "noexec"


@needs_docker
def test_tenants_see_only_their_own_workspace(sandbox):
    assert py(sandbox, "open('/work/plan.txt','w').write('t1 secret'); print('ok')").stdout.strip() == "ok"
    assert py(sandbox, "import os; print(os.listdir('/work'))", tenant="t2").stdout.strip() == "[]"
    assert py(sandbox, "print(open('/work/plan.txt').read())").stdout.strip() == "t1 secret"


@needs_docker
@pytest.mark.parametrize("attack,expect", [
    ("x = bytearray(512 * 1024 * 1024); print('ALLOCATED')", "killed"),
    ("while True: pass", "timed_out"),
])
def test_resource_attacks_are_stopped(sandbox, attack, expect):
    r = py(sandbox, attack)
    assert getattr(r, expect) and "ALLOCATED" not in r.stdout


@needs_docker
def test_fork_bomb_hits_the_process_limit(sandbox):
    r = py(sandbox, "import os, time\nn = 0\ntry:\n  while True:\n    if os.fork() == 0:\n      time.sleep(30)\n"
                    "    n += 1\nexcept OSError as e: print('stopped', n)")
    assert r.stdout.startswith("stopped") and int(r.stdout.split()[1]) < 32


@needs_docker
def test_secrets_exist_only_for_their_call(sandbox):
    probe = "import os; print(os.environ.get('API_TOKEN', '<unset>'))"
    assert py(sandbox, probe, env={"API_TOKEN": "s3cr3t"}).stdout.strip() == "s3cr3t"
    assert py(sandbox, probe).stdout.strip() == "<unset>"


@needs_docker
def test_warm_containers_are_single_use(tmp_path):
    sb = DockerSandbox(SandboxPolicy(image=IMAGE, python=PYTHON), Workspaces(tmp_path / "ws"), pool_size=2)
    try:
        sb.warm("t1")
        a = sb.run("t1", "open('/tmp/left-behind','w').write('x'); print('ok')")
        b = sb.run("t1", "import os; print(os.path.exists('/tmp/left-behind'))")
        assert a.stdout.strip() == "ok"        # the first call really did leave a file behind ...
        assert a.warm and b.warm and b.stdout.strip() == "False"   # ... and the next call cannot see it
    finally:
        sb.close()
    left = subprocess.run(["docker", "ps", "-aq", "--filter", f"label=rewind.sandbox={sb.session}"],
                          capture_output=True, text=True).stdout.split()
    assert left == []


@needs_docker
def test_output_is_capped(tmp_path):
    sb = DockerSandbox(SandboxPolicy(image=IMAGE, python=PYTHON, max_output_bytes=100), Workspaces(tmp_path))
    try:
        r = sb.run("t1", "print('x' * 10000)")
        assert len(r.stdout) < 200 and "[truncated: 10001 bytes total]" in r.stdout
    finally:
        sb.close()


# ---------------------------------------------------------------------------
# as a tool in the agent loop
# ---------------------------------------------------------------------------
class CodeArgs(BaseModel):
    path: str
    text: str


WRITE_CODE = ("import json, sys, os\na = json.load(sys.stdin)\n"
              "open(os.path.join('/work', os.path.basename(a['path'])), 'w').write(a['text'])\n"
              "print('saved', sorted(os.listdir('/work')))")


def sandbox_registry(sb):
    reg = platform_registry(rules=RULES + [
        {"id": "analysts-run-code", "effect": "allow", "roles": ["analyst"], "capabilities": ["local.save_note"]}])
    reg.register(sandboxed_tool("save_note", "Save a note in the team workspace.", CodeArgs, WRITE_CODE, sb,
                                mutating=True))
    reg.access.catalog.add(CapabilityDescriptor(id="local.save_note", name="save_note", risk="write"))
    return reg


@needs_docker
def test_sandboxed_tool_runs_in_the_callers_tenant(sandbox):
    reg = sandbox_registry(sandbox)
    _, _, t1, *_ = go([call("save_note", path="a.txt", text="hi"), text("ok")],
                      principal(tenant="t1", roles=("analyst",)), registry=reg, run_id="r1")
    _, _, t2, *_ = go([call("save_note", path="b.txt", text="yo"), text("ok")],
                      principal(tenant="t2", roles=("analyst",)), registry=reg, run_id="r2")
    assert tool_msgs(t1)[0]["content"].strip() == "saved ['a.txt']"
    assert tool_msgs(t2)[0]["content"].strip() == "saved ['b.txt']"     # t1's file is not there
    _, _, t3, *_ = go([call("save_note", path="c.txt", text="no"), text("ok")],
                      principal(tenant="t1", roles=("support",)), registry=reg, run_id="r3")
    assert ev(t3, "authz")[0]["verdict"] == "hidden"


@needs_docker
def test_replay_never_starts_a_container(sandbox):
    reg = sandbox_registry(sandbox)
    _, _, trace, _p, snaps, _ = go([call("save_note", path="a.txt", text="hi"), text("ok")],
                                   principal(roles=("analyst",)), registry=reg)
    sandbox.close()                                          # no sandbox any more ...
    assert run(replay_run(trace.events, snaps, reg, ToyRetailEnv.restore)).ok   # ... replay still exact


def test_sandboxed_tool_without_a_caller_identity_fails_closed(tmp_path):
    from rewind import ToolRegistry
    reg = ToolRegistry()
    sb = DockerSandbox(SandboxPolicy(image=IMAGE), Workspaces(tmp_path))
    reg.register(sandboxed_tool("save_note", "Save a note.", CodeArgs, WRITE_CODE, sb))
    out = reg.execute(ToyRetailEnv(), "save_note", {"path": "a", "text": "b"}, "s1:c0")
    assert (out.ok, out.error_class) == (False, "TOOL_RUNTIME_ERROR") and "caller identity" in out.content
