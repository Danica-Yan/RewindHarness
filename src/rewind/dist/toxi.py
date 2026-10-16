"""Real network-fault injection via a real Toxiproxy server (plan W6's
"Toxiproxy" line) — not a Python-side simulation of latency/loss, an actual
TCP proxy that sits between the client and Postgres/Redis and misbehaves on
purpose. `ToxiproxyController` starts/stops the real `toxiproxy-server`
binary as a subprocess and drives its HTTP admin API (default
`127.0.0.1:8474`) to create proxies and add/remove toxics, so a test that
adds a `latency` or `reset_peer` toxic is exercising genuine socket-level
misbehavior end to end.

No Docker needed: `toxiproxy-server`/`toxiproxy-cli` are single static Go
binaries fetched straight from the project's GitHub releases (no Docker
daemon required — see README). `tests/test_chaos.py` skips itself with a clear reason if the
binary isn't on PATH.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

import requests


class ToxiproxyUnavailable(RuntimeError):
    """Raised when the toxiproxy-server binary can't be found or won't come up."""


@dataclass
class Toxic:
    name: str
    type: str
    stream: str = "downstream"          # "downstream" (server->client) or "upstream"
    attributes: dict[str, Any] = field(default_factory=dict)


# A few named constructors for the toxics W6 asks for, so call sites read as
# intent ("cut the connection") rather than raw Toxiproxy JSON shapes.
def latency(name: str, ms: int, jitter_ms: int = 0) -> Toxic:
    return Toxic(name, "latency", attributes={"latency": ms, "jitter": jitter_ms})


def timeout(name: str, ms: int) -> Toxic:
    """Holds the connection open and silent for `ms`, then resets it — the
    "server accepted the request and then never answered" failure mode."""
    return Toxic(name, "timeout", attributes={"timeout": ms})


def connection_reset(name: str) -> Toxic:
    """Slams the TCP connection shut immediately (RST) — models a killed
    upstream process or a dropped network path, not a slow one."""
    return Toxic(name, "reset_peer", attributes={"timeout": 0})


def bandwidth_limit(name: str, rate_kb_s: int) -> Toxic:
    return Toxic(name, "bandwidth", attributes={"rate": rate_kb_s})


class ToxiproxyController:
    """Owns one `toxiproxy-server` process (or reuses one already running on
    the target admin port) and its admin API."""

    def __init__(self, admin_host: str = "127.0.0.1", admin_port: int = 8474) -> None:
        self.admin_host = admin_host
        self.admin_port = admin_port
        self.base_url = f"http://{admin_host}:{admin_port}"
        self._proc: subprocess.Popen | None = None
        self._owns_process = False

    @staticmethod
    def binary_available() -> bool:
        return shutil.which("toxiproxy-server") is not None

    def _reachable(self) -> bool:
        try:
            r = requests.get(self.base_url + "/version", timeout=0.3)
            return r.ok
        except requests.RequestException:
            return False

    def start(self, timeout_s: float = 5.0) -> None:
        """Idempotent: if a toxiproxy-server is already answering on this
        admin port (e.g. started by an earlier test in the same session),
        reuse it rather than spawning a second one."""
        if self._reachable():
            return
        if not self.binary_available():
            raise ToxiproxyUnavailable("toxiproxy-server binary not found on PATH")
        self._proc = subprocess.Popen(
            ["toxiproxy-server", "-host", self.admin_host, "-port", str(self.admin_port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._owns_process = True
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._reachable():
                return
            time.sleep(0.1)
        raise ToxiproxyUnavailable(f"toxiproxy-server did not come up on {self.base_url} in time")

    def stop(self) -> None:
        if self._proc is not None and self._owns_process:
            self._proc.kill()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self._proc = None

    # ---- proxies ---------------------------------------------------------------
    def create_proxy(self, name: str, listen: str, upstream: str) -> None:
        self.delete_proxy(name)          # start from a clean slate (idempotent for re-runs)
        r = requests.post(f"{self.base_url}/proxies",
                          json={"name": name, "listen": listen, "upstream": upstream})
        r.raise_for_status()

    def delete_proxy(self, name: str) -> None:
        requests.delete(f"{self.base_url}/proxies/{name}")     # 404 if absent is fine, ignored

    def set_enabled(self, proxy: str, enabled: bool) -> None:
        """Toggling `enabled=False` drops the listener entirely — every new
        connection attempt fails immediately, modeling "the network path to
        this host is gone" rather than "the path is degraded"."""
        r = requests.post(f"{self.base_url}/proxies/{proxy}", json={"enabled": enabled})
        r.raise_for_status()

    # ---- toxics ------------------------------------------------------------------
    def add_toxic(self, proxy: str, toxic: Toxic) -> None:
        body = {"name": toxic.name, "type": toxic.type, "stream": toxic.stream,
                "attributes": toxic.attributes}
        r = requests.post(f"{self.base_url}/proxies/{proxy}/toxics", json=body)
        r.raise_for_status()

    def remove_toxic(self, proxy: str, toxic_name: str) -> None:
        requests.delete(f"{self.base_url}/proxies/{proxy}/toxics/{toxic_name}")

    def remove_all_toxics(self, proxy: str) -> None:
        r = requests.get(f"{self.base_url}/proxies/{proxy}/toxics")
        if r.ok:
            for t in r.json():
                self.remove_toxic(proxy, t["name"])
