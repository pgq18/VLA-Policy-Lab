"""Long synchronous node dispatch must use the RPC deadline, not WS pings."""
from contextlib import contextmanager
import asyncio
import functools
import inspect
import threading
import time

import pytest
import websockets
from websockets.sync import client as websocket_client

from pressb.runtime.client import NodeClient
from pressb.runtime.inference_transport import _pack, _unpack
from pressb.runtime.rpc import RPCError


ORIGINAL_CONNECT = websocket_client.connect
HAS_KEEPALIVE = "ping_timeout" in inspect.signature(ORIGINAL_CONNECT).parameters
requires_keepalive = pytest.mark.skipif(
    not HAS_KEEPALIVE, reason="The synchronous keepalive regression requires websockets 15+")


@contextmanager
def blocking_node(delay=.15):
    """A real socket peer with the production node's synchronous-dispatch shape.

    Its event loop cannot answer Ping frames while the simulated inference
    handler is sleeping. No model, CUDA context or external port is needed.
    """
    from websockets.asyncio.server import serve

    ready = threading.Event()
    shared = {"calls": [], "errors": []}

    async def handler(connection):
        await connection.send(_pack({"schema_version": 2, "node_type": "inference"}))
        try:
            async for frame in connection:
                request = _unpack(frame)["__pressb_rpc__"]
                shared["calls"].append(request)
                # Deliberately block this server event loop, as the owner-thread
                # model/physics dispatch does in serve_websocket.
                time.sleep(delay)
                await connection.send(_pack({"protocol_version": 1, "ok": True}))
        except websockets.exceptions.ConnectionClosed:
            # Expected when the control client or the actual RPC deadline closes.
            pass

    async def run():
        shared["loop"] = asyncio.get_running_loop()
        shared["stop"] = asyncio.Event()
        async with serve(handler, "127.0.0.1", 0, compression=None, ping_interval=None) as server:
            shared["endpoint"] = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            ready.set()
            await shared["stop"].wait()

    def worker():
        try:
            asyncio.run(run())
        except BaseException as error:
            shared["errors"].append(error)
            ready.set()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        assert ready.wait(3), "WebSocket fixture failed to start"
        assert not shared["errors"]
        yield shared
    finally:
        if "stop" in shared and thread.is_alive():
            shared["loop"].call_soon_threadsafe(shared["stop"].set)
        thread.join(3)
        assert not thread.is_alive(), "WebSocket fixture did not stop"
        assert not shared["errors"]


def accelerate_library_keepalive(monkeypatch):
    """Expose the real v15 signature while shortening only library defaults."""
    options = []

    @functools.wraps(ORIGINAL_CONNECT)
    def connect(*args, **kwargs):
        options.append(dict(kwargs))
        kwargs.setdefault("ping_interval", .02)
        kwargs.setdefault("ping_timeout", .02)
        return ORIGINAL_CONNECT(*args, **kwargs)

    monkeypatch.setattr(websocket_client, "connect", connect)
    return options


@requires_keepalive
def test_library_heartbeat_reproduces_failure_during_blocking_dispatch():
    """A fast heartbeat closes a still-running request before its RPC deadline."""
    with blocking_node() as server:
        with ORIGINAL_CONNECT(server["endpoint"], compression=None,
                              ping_interval=.02, ping_timeout=.02,
                              open_timeout=1, close_timeout=.2) as client:
            assert _unpack(client.recv(timeout=1))["schema_version"] == 2
            client.send(_pack({"__pressb_rpc__": {
                "version": 1, "method": "slow", "payload": {}}}))
            with pytest.raises(websockets.exceptions.ConnectionClosedError) as caught:
                client.recv(timeout=1)
            assert caught.value.sent.code == 1011
            assert caught.value.sent.reason == "keepalive ping timeout"
        assert len(server["calls"]) == 1


@requires_keepalive
def test_node_request_outlasts_heartbeat_but_completes_within_rpc_deadline(monkeypatch):
    options = accelerate_library_keepalive(monkeypatch)
    with blocking_node() as server:
        client = NodeClient(server["endpoint"], timeout=1)
        try:
            assert client.call("/slow", {"value": 7}) == {"protocol_version": 1, "ok": True}
            assert len(server["calls"]) == 1
            assert server["calls"][0]["payload"] == {"value": 7}
            assert len(options) == 1
            assert options[0]["ping_timeout"] is None
        finally:
            client.close()


@requires_keepalive
def test_real_rpc_timeout_still_fails_closed_without_replaying(monkeypatch):
    options = accelerate_library_keepalive(monkeypatch)
    with blocking_node(delay=.2) as server:
        client = NodeClient(server["endpoint"], timeout=.06)
        try:
            with pytest.raises(TimeoutError):
                client.call("/slow", {"value": 11})
            with pytest.raises(RPCError, match="closed or failed.*not retried"):
                client.call("/slow", {"value": 12})
            assert len(server["calls"]) == 1
            assert server["calls"][0]["payload"] == {"value": 11}
            assert len(options) == 1
            assert options[0]["ping_timeout"] is None
        finally:
            client.close()


def test_legacy_connect_without_keepalive_arguments_remains_supported(monkeypatch):
    """The strict signature models websockets 12's actual connect API."""
    calls, sent = [], []

    class Wire:
        def __init__(self):
            self.frames = iter((_pack({"schema_version": 2}),
                                _pack({"protocol_version": 1, "ok": True})))

        def recv(self, *, timeout):
            return next(self.frames)

        def send(self, frame):
            sent.append(frame)

        def close(self):
            pass

    def legacy_connect(uri, *, additional_headers, compression, max_size,
                       open_timeout, close_timeout):
        calls.append(uri)
        return Wire()

    monkeypatch.setattr(websocket_client, "connect", legacy_connect)
    client = NodeClient("ws://127.0.0.1:1", timeout=1)
    try:
        assert client.call("/slow", {}) == {"protocol_version": 1, "ok": True}
        assert calls == ["ws://127.0.0.1:1"]
        assert len(sent) == 1
    finally:
        client.close()
