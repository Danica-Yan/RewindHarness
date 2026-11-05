"""MCP gateway: call latency per transport, and recovery time after kill -9.

    python scripts/mcp_bench.py --calls 300 --kills 10
Needs the [mcp] extra. Runs tests/mcp_servers/demo_server.py locally.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = str(ROOT / "tests" / "mcp_servers" / "demo_server.py")
sys.path.insert(0, str(ROOT / "tests" / "mcp_servers"))

from rewind.platform.mcp_gateway import McpGateway, McpServerConfig  # noqa: E402
from rewind.tools import TransientToolError  # noqa: E402


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(len(xs) * q))] * 1000, 2)


def latency(gw: McpGateway, sid: str, n: int) -> dict:
    gw.call(sid, "write_note", {"path": "/t1/a", "text": "x"})
    lat = []
    for _ in range(n):
        t = time.perf_counter()
        gw.call(sid, "read_note", {"path": "/t1/a"})
        lat.append(time.perf_counter() - t)
    return {"transport": sid, "p50_ms": round(statistics.median(lat) * 1000, 2), "p99_ms": pct(lat, 0.99)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=300)
    ap.add_argument("--kills", type=int, default=10)
    a = ap.parse_args()
    import demo_server

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    http = subprocess.Popen([sys.executable, SERVER, "--http", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp = Path(tempfile.mkdtemp())
    gw = McpGateway()
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), 0.2).close()
                break
            except OSError:
                time.sleep(0.05)
        common = {"trust_on_first_use": True}
        gw.add_server(McpServerConfig(id="inproc", transport="inproc", **common), inproc=demo_server.server)
        gw.add_server(McpServerConfig(id="stdio", transport="stdio", command=[sys.executable, SERVER],
                                      env={"DEMO_PIDFILE": str(tmp / "pid")}, **common))
        gw.add_server(McpServerConfig(id="http", transport="http", url=f"http://127.0.0.1:{port}/mcp", **common))
        gw.sync()
        for sid in ("inproc", "stdio", "http"):
            print(latency(gw, sid, a.calls))

        rec = []
        for _ in range(a.kills):
            os.kill(int((tmp / "pid").read_text()), signal.SIGKILL)
            t0 = time.perf_counter()
            while True:
                try:
                    gw.call("stdio", "write_note", {"path": "/t1/a", "text": "x"})
                    break
                except TransientToolError:
                    time.sleep(0.01)
            rec.append(time.perf_counter() - t0)
        print({"kill -9 recovery (stdio)": len(rec), "p50_ms": round(statistics.median(rec) * 1000),
               "max_ms": round(max(rec) * 1000)})
    finally:
        gw.close()
        http.kill()


if __name__ == "__main__":
    main()
