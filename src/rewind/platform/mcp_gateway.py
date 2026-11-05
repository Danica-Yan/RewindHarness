"""MCP gateway: third-party MCP servers as governed rewind capabilities.

    gw = McpGateway(PinStore("pins.json"))
    gw.add_server(McpServerConfig(id="notes", transport="stdio", command=[...]))
    gw.sync()                       # list tools, compare with approved pins
    gw.approve("notes", "read_note", approver="admin@t1")
    gw.register(registry, catalog)  # approved tools become rewind tools + capabilities

What the gateway is responsible for, and what it hands to the rest of the
platform:

  * Naming. A server's tool `read_note` is the capability
    `mcp.notes.read_note`, shown to the model as `notes__read_note` -- so
    access rules and quotas govern MCP tools exactly like local ones, and two
    servers' tools never collide.
  * Pinning (mcp_pins.py). Only tools whose exact definition was approved are
    registered. A definition that changes later is quarantined: still listed
    by the server, refused by the gateway, until someone approves the new one.
  * Risk. Server annotations may raise risk above the admin's floor, never
    lower it. Anything above `read` is registered as mutating, so the loop
    runs it sequentially and the idempotency ledger keeps it exactly-once
    across crash/resume. Output from MCP servers is untrusted by default,
    which is what arms the context-taint rules of the policy engine.
  * Secrets. A server config holds secret REFERENCES; the values are
    resolved when connecting and go only to that server (its process env or
    its HTTP headers) -- never into config dumps, traces or the model's view.
    A stdio server inherits only the SDK's minimal default environment.
  * Connections. One supervisor task per server owns the client session:
    connect, reconnect with capped exponential backoff after the transport
    breaks, a per-server concurrency limit and call timeout. Transport
    failures and timeouts surface as TransientToolError (the loop's retry
    path), errors the tool itself reports as ToolError.
  * Output limits. Results are capped at `max_output_chars`.

Replay never calls a server: gateway tools are marked `external`, and
`replay_run` serves their recorded outcomes. A fork does call servers --
their state is not part of env snapshots, which is the honest limit of
forking a run that touched the outside world.

The gateway runs its own event loop in a background thread, because rewind
executes tools synchronously in worker threads (ToolRegistry.execute is
called via asyncio.to_thread). Needs the [mcp] extra.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import os
import re
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..tools import Tool, ToolError, ToolRegistry, TransientToolError
from .capability import CapabilityCatalog, CapabilityDescriptor, Risk
from .context import current_call
from .mcp_pins import PinStore, ToolStatus, derive_risk, fingerprint, reconcile

SEP = "__"
IDEMPOTENCY_META = "io.rewind/idempotency-key"
# Python 3.10: asyncio.wait_for and Future.result raise their own TimeoutError
# classes, which only became aliases of the builtin in 3.11.
_TIMEOUTS = (TimeoutError, asyncio.TimeoutError, concurrent.futures.TimeoutError)


class ToolOverride(BaseModel):
    risk: Risk | None = None
    untrusted_output: bool | None = None


class McpServerConfig(BaseModel):
    id: str
    transport: Literal["stdio", "http", "inproc"]
    command: list[str] = Field(default_factory=list)           # stdio
    url: str | None = None                                      # http
    env: dict[str, str] = Field(default_factory=dict)           # non-secret extras (stdio)
    secret_env: dict[str, str] = Field(default_factory=dict)    # VAR -> secret ref (stdio)
    secret_headers: dict[str, str] = Field(default_factory=dict)  # header -> secret ref (http)
    risk_floor: Risk = "write"
    untrusted_output: bool = True
    tool_overrides: dict[str, ToolOverride] = Field(default_factory=dict)
    trust_on_first_use: bool = False        # dev convenience: pin everything on the first sync
    call_timeout_s: float = Field(default=30.0, gt=0)
    connect_timeout_s: float = Field(default=10.0, gt=0)
    max_concurrency: int = Field(default=4, gt=0)
    max_output_chars: int = Field(default=20_000, gt=0)
    recheck_interval_s: float | None = Field(default=300.0, gt=0)   # re-list while connected
    stderr_path: str | None = None          # stdio server's stderr (default: discarded)

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9]{1,30}", v):
            raise ValueError("server id: lowercase letters and digits, 2-31 chars (it is part of tool names)")
        return v

    @model_validator(mode="after")
    def _transport(self) -> McpServerConfig:
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio server needs a command")
        if self.transport == "http" and not self.url:
            raise ValueError("http server needs a url")
        overlap = set(self.env) & set(self.secret_env)
        if overlap:
            raise ValueError(f"{sorted(overlap)} set both as plain env and as secret")
        return self


class SecretResolver:
    """Resolves a secret reference to its value. The default reads the
    environment variable REWIND_SECRET_<REF>; a deployment would plug in its
    vault here. Values are never cached on the gateway."""

    def __init__(self, prefix: str = "REWIND_SECRET_", lookup: Callable[[str], str | None] | None = None) -> None:
        self.prefix = prefix
        self.lookup = lookup or os.environ.get

    def get(self, ref: str) -> str:
        v = self.lookup(self.prefix + ref)
        if v is None:
            raise KeyError(f"secret {ref!r} is not available")
        return v


@dataclass
class ListedTool:
    name: str
    description: str
    input_schema: dict
    annotations: dict
    fingerprint: str


@dataclass
class _Health:
    connected: bool = False
    connects: int = 0
    failures: int = 0
    last_error: str | None = None


class _Conn:
    """Owns one server's client session inside the gateway's event loop."""

    def __init__(self, cfg: McpServerConfig, opener: Callable[[], Any],
                 on_listing: Callable[[list[Any]], None]) -> None:
        self.cfg = cfg
        self._open = opener
        self._on_listing = on_listing
        self.client: Any = None
        self.health = _Health()
        self._ready = asyncio.Event()
        self._broken = asyncio.Event()
        self._closing = False
        self._sem = asyncio.Semaphore(cfg.max_concurrency)
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._supervise())

    async def _supervise(self) -> None:
        attempt = 0
        while not self._closing:
            self._broken = asyncio.Event()
            try:
                async with self._open() as client:
                    # Definitions are re-verified on EVERY connect before the
                    # connection is usable: a server restarted with different
                    # tools must not get a window in which the old approval
                    # still covers them.
                    self._on_listing(await self._list(client))
                    self.client = client
                    self.health.connected = True
                    self.health.connects += 1
                    attempt = 0
                    self._ready.set()
                    while not self._broken.is_set():
                        try:
                            await asyncio.wait_for(self._broken.wait(), self.cfg.recheck_interval_s)
                        except _TIMEOUTS:
                            self._on_listing(await self._list(client))
            except Exception as e:                       # noqa: BLE001 -- any failure means reconnect
                self.health.failures += 1
                self.health.last_error = f"{type(e).__name__}: {e}"[:300]
            finally:
                self.client = None
                self.health.connected = False
                self._ready.clear()
            if self._closing:
                break
            await asyncio.sleep(min(2.0, 0.05 * 2 ** attempt))
            attempt += 1

    async def _client(self) -> Any:
        try:
            await asyncio.wait_for(self._ready.wait(), self.cfg.connect_timeout_s)
        except _TIMEOUTS:
            raise TransientToolError(f"MCP server {self.cfg.id} is unavailable "
                                     f"({self.health.last_error or 'not connected'})") from None
        return self.client

    def _lost(self, e: BaseException) -> TransientToolError:
        self._broken.set()
        return TransientToolError(f"connection to MCP server {self.cfg.id} lost ({type(e).__name__})")

    async def _list(self, client: Any) -> list[Any]:
        tools, cursor = [], None
        while True:
            page = await asyncio.wait_for(client.list_tools(cursor=cursor), self.cfg.call_timeout_s)
            tools.extend(page.tools)
            cursor = page.next_cursor
            if not cursor:
                return tools

    async def list_tools(self) -> list[Any]:
        from mcp.shared.exceptions import MCPError
        client = await self._client()
        try:
            return await self._list(client)
        except _TIMEOUTS:
            raise TransientToolError(f"listing tools of {self.cfg.id} timed out") from None
        except MCPError as e:
            if _is_connection_error(e):
                raise self._lost(e) from e
            raise
        except Exception as e:                           # noqa: BLE001
            raise self._lost(e) from e

    async def call(self, tool: str, args: dict, meta: dict | None = None) -> str:
        from mcp.shared.exceptions import MCPError
        client = await self._client()
        async with self._sem:
            try:
                res = await asyncio.wait_for(client.call_tool(tool, args, meta=meta), self.cfg.call_timeout_s)
            except _TIMEOUTS:
                raise TransientToolError(f"{tool} on {self.cfg.id} timed out after "
                                         f"{self.cfg.call_timeout_s:g}s") from None
            except MCPError as e:
                if _is_connection_error(e):
                    raise self._lost(e) from e
                if _is_timeout(e):
                    raise TransientToolError(f"{tool} on {self.cfg.id} timed out") from e
                raise ToolError(f"{tool}: {e}") from e
            except Exception as e:                       # noqa: BLE001 -- transport failure
                raise self._lost(e) from e
        parts = [c.text if getattr(c, "type", None) == "text" else f"[{getattr(c, 'type', '?')} content omitted]"
                 for c in res.content]
        text = "\n".join(parts)
        if len(text) > self.cfg.max_output_chars:
            text = text[:self.cfg.max_output_chars] + f"\n[truncated: {len(text)} chars total]"
        if res.is_error:
            raise ToolError(text or f"{tool} failed")
        return text

    async def close(self) -> None:
        self._closing = True
        self._broken.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, 5)
            except Exception:            # noqa: BLE001
                self._task.cancel()


def _is_connection_error(e: Any) -> bool:
    from mcp_types import CONNECTION_CLOSED
    return getattr(e, "code", None) == CONNECTION_CLOSED


def _is_timeout(e: Any) -> bool:
    from mcp_types import REQUEST_TIMEOUT
    return getattr(e, "code", None) == REQUEST_TIMEOUT


class _ArgsBase(BaseModel):
    """Arguments go to the server as given; the server validates them against
    its own schema. Only the schema's required keys are checked here, so a
    call missing one fails as TOOL_ARG_INVALID before any traffic."""
    model_config = ConfigDict(extra="allow")


def _args_model(tool: str, required: list[str]) -> type[BaseModel]:
    req = tuple(required)

    class Args(_ArgsBase):
        @model_validator(mode="before")
        @classmethod
        def _required(cls, v: Any) -> Any:
            if not isinstance(v, dict):
                raise ValueError("arguments must be an object")
            missing = [k for k in req if k not in v]
            if missing:
                raise ValueError(f"missing required arguments: {', '.join(missing)}")
            return v

    Args.__name__ = f"Args_{tool}"
    return Args


class McpGateway:
    def __init__(self, pins: PinStore | None = None, secrets: SecretResolver | None = None) -> None:
        self.pins = pins or PinStore()
        self.secrets = secrets or SecretResolver()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="mcp-gateway", daemon=True)
        self._thread.start()
        self._conns: dict[str, _Conn] = {}
        self._listed: dict[str, dict[str, ListedTool]] = {}
        self._status: dict[str, dict[str, ToolStatus]] = {}
        self._lock = threading.Lock()

    # ---- plumbing ------------------------------------------------------------
    def _run(self, coro, timeout: float):
        if not self._thread.is_alive():
            coro.close()
            # Fail fast: a coroutine handed to a stopped loop would never run,
            # and the caller would sit out the whole timeout.
            raise TransientToolError("the MCP gateway is closed")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def _opener(self, cfg: McpServerConfig, inproc: Any) -> Callable[[], Any]:
        from mcp import Client
        # The SDK's own read timeout is set past ours, so the gateway's timeout
        # (a TransientToolError) is the one that fires; REQUEST_TIMEOUT from the
        # SDK is still mapped to transient below in case it wins anyway.
        kw = {"cache": None, "read_timeout_seconds": cfg.call_timeout_s + 2}
        if cfg.transport == "inproc":
            if inproc is None:
                raise ValueError("inproc transport needs the server object")
            return lambda: Client(inproc, **kw)
        if cfg.transport == "stdio":
            from mcp.client.stdio import StdioServerParameters, stdio_client

            def open_stdio():
                env = dict(cfg.env)
                env.update({k: self.secrets.get(ref) for k, ref in cfg.secret_env.items()})
                params = StdioServerParameters(command=cfg.command[0], args=cfg.command[1:], env=env)

                @asynccontextmanager
                async def transport():
                    with open(cfg.stderr_path or os.devnull, "a") as errlog:
                        async with stdio_client(params, errlog=errlog) as streams:
                            yield streams
                return Client(transport(), **kw)
            return open_stdio

        def open_http():
            import httpx2
            from mcp.client.streamable_http import streamable_http_client
            headers = {h: self.secrets.get(ref) for h, ref in cfg.secret_headers.items()}

            @asynccontextmanager
            async def transport():
                async with httpx2.AsyncClient(headers=headers, timeout=cfg.call_timeout_s + 5) as hc:
                    async with streamable_http_client(cfg.url, http_client=hc) as streams:
                        yield streams
            return Client(transport(), **kw)
        return open_http

    # ---- servers -------------------------------------------------------------
    def add_server(self, cfg: McpServerConfig, inproc: Any = None) -> None:
        if cfg.id in self._conns:
            raise ValueError(f"server {cfg.id} already added")
        conn = _Conn(cfg, self._opener(cfg, inproc), lambda raw, sid=cfg.id: self._apply_listing(sid, raw))

        self._conns[cfg.id] = conn          # before start: the first connect looks the config up

        async def start():
            conn.start()
        self._run(start(), 5)

    def config(self, server: str) -> McpServerConfig:
        return self._conns[server].cfg

    def health(self) -> dict[str, dict]:
        return {s: vars(c.health).copy() for s, c in self._conns.items()}

    def close(self) -> None:
        """Disconnect every server and stop the gateway's loop. Idempotent."""
        if not self._thread.is_alive():
            return
        for c in self._conns.values():
            try:
                self._run(c.close(), 10)
            except Exception:                            # noqa: BLE001
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)

    # ---- definitions and pins ---------------------------------------------------
    def sync(self, server: str | None = None) -> dict[str, dict[str, ToolStatus]]:
        """List each server's tools, fingerprint them and compare with the
        approved pins. Re-run it periodically (or on reconnect) to catch a
        definition that changes under you; a changed tool is refused at call
        time from the moment this returns."""
        out = {}
        for sid in ([server] if server else list(self._conns)):
            conn = self._conns[sid]
            raw = self._run(conn.list_tools(), conn.cfg.connect_timeout_s + conn.cfg.call_timeout_s + 5)
            out[sid] = self._apply_listing(sid, raw)
        return out

    def _apply_listing(self, sid: str, raw: list[Any]) -> dict[str, ToolStatus]:
        cfg = self._conns[sid].cfg if sid in self._conns else None
        listed = {}
        for t in raw:
            ann = t.annotations.model_dump() if t.annotations is not None else {}
            listed[t.name] = ListedTool(t.name, t.description or "", t.input_schema or {}, ann,
                                        fingerprint(t.name, t.description, t.input_schema, ann))
        pins = self.pins.pins(sid)
        if cfg is not None and cfg.trust_on_first_use and not pins:
            for lt in listed.values():
                self.pins.approve(sid, lt.name, lt.fingerprint, "trust-on-first-use")
            pins = self.pins.pins(sid)
        status = reconcile(pins, {n: lt.fingerprint for n, lt in listed.items()})
        with self._lock:
            self._listed[sid], self._status[sid] = listed, status
        return dict(status)

    def approve(self, server: str, tool: str, approver: str) -> str:
        """Pin the definition the server listed at the last sync."""
        lt = self._listed.get(server, {}).get(tool)
        if lt is None:
            raise KeyError(f"{server} has not listed a tool {tool!r}; run sync() first")
        self.pins.approve(server, tool, lt.fingerprint, approver)
        with self._lock:
            self._status[server][tool] = "approved"
        return lt.fingerprint

    def status(self, server: str) -> dict[str, ToolStatus]:
        with self._lock:
            return dict(self._status.get(server, {}))

    # ---- exposure ------------------------------------------------------------
    def descriptor(self, server: str, tool: str) -> CapabilityDescriptor:
        cfg, lt = self._conns[server].cfg, self._listed[server][tool]
        ov = cfg.tool_overrides.get(tool, ToolOverride())
        return CapabilityDescriptor(
            id=f"mcp.{server}.{tool}", name=f"{server}{SEP}{tool}", source="mcp",
            risk=derive_risk(lt.annotations, cfg.risk_floor, ov.risk),
            untrusted_output=cfg.untrusted_output if ov.untrusted_output is None else ov.untrusted_output,
            schema_hash=lt.fingerprint, description=lt.description)

    def register(self, registry: ToolRegistry, catalog: CapabilityCatalog | None = None) -> list[str]:
        """Register every approved tool. Returns their capability ids."""
        ids = []
        for sid in self._conns:
            for name, st in self.status(sid).items():
                if st != "approved":
                    continue
                lt, cap = self._listed[sid][name], self.descriptor(sid, name)
                registry.register(Tool(
                    name=cap.name, description=f"[{sid}] {lt.description}",
                    args_model=_args_model(cap.name, list(lt.input_schema.get("required", []))),
                    fn=self._tool_fn(sid, name), mutating=cap.risk != "read",
                    input_schema=lt.input_schema or {"type": "object", "properties": {}},
                    raw_text_result=True, external=True))
                if catalog is not None:
                    catalog.add(cap)
                ids.append(cap.id)
        return ids

    def _tool_fn(self, server: str, tool: str):
        def fn(_env, args: BaseModel) -> str:
            return self.call(server, tool, args.model_dump())
        return fn

    def call(self, server: str, tool: str, args: dict) -> str:
        st = self.status(server).get(tool)
        if st != "approved":
            raise ToolError(f"{server}{SEP}{tool} is unavailable: its definition is not approved"
                            + (" (it changed since approval)" if st == "changed" else ""))
        conn = self._conns[server]
        # The same key on every attempt of one action -- including after a
        # crash/resume -- so a server that honours it can deduplicate the
        # case rewind's own ledger cannot cover (crash after the server acted,
        # before the checkpoint).
        ctx = current_call()
        meta = {IDEMPOTENCY_META: ctx.idempotency_key} if ctx is not None else None
        try:
            return self._run(conn.call(tool, args, meta),
                             conn.cfg.connect_timeout_s + conn.cfg.call_timeout_s + 5)
        except _TIMEOUTS:
            raise TransientToolError(f"{server}{SEP}{tool} did not answer in time") from None
