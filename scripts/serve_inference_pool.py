#!/usr/bin/env python3
"""Serve dedicated inference replicas through one schema-v2 WS/HTTP endpoint.

Config sections: workers (NodeClient endpoint/options, optional token_env),
server (same host/port/transport/token_env/message limit as other nodes), and
service ({"max_batch_size": ...}, optional). Worker processes are externally
managed; this host never starts or kills accelerator jobs.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pressb.runtime.client import NodeClient
from pressb.runtime.inference_pool import InferencePool
from pressb.runtime.rpc import MAX_BODY_BYTES, RPCServer
from pressb.runtime.server import _message_limit, serve_websocket


def serve(config):
    if set(config) - {"workers", "server", "service"}:
        raise ValueError("Unknown inference pool config section")
    specs = config.get("workers")
    if not isinstance(specs, list) or len(specs) < 2:
        raise ValueError("workers must contain at least two NodeClient configurations")
    endpoints = [spec.get("endpoint") for spec in specs if isinstance(spec, dict)]
    if len(endpoints) != len(specs) or len(set(endpoints)) != len(endpoints):
        raise ValueError("workers must select distinct endpoints")
    server = config.get("server", {})
    if set(server) - {"host", "port", "transport", "token_env", "max_message_bytes"}:
        raise ValueError("Unknown pool server option")
    host, port = server.get("host", "127.0.0.1"), server.get("port", 8100)
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Invalid server port")
    transport = server.get("transport", "ws")
    if transport not in ("ws", "http"):
        raise ValueError("Pool transport must be ws or http")
    limit = _message_limit(server.get("max_message_bytes", MAX_BODY_BYTES))
    token = os.environ.get(server["token_env"]) if server.get("token_env") else None
    if server.get("token_env") and not token:
        raise ValueError("Server token_env is unset or empty")
    clients, pool = [], None
    def stop_service(_signum, _frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, stop_service)
    try:
        for spec in specs:
            spec = dict(spec)
            key = spec.pop("token_env", None)
            if "token" in spec:
                raise ValueError("Use token_env instead of inline worker credentials")
            if key:
                spec["token"] = os.environ.get(key)
                if not spec["token"]:
                    raise ValueError("Worker token_env is unset or empty")
            clients.append(NodeClient(**spec))
        pool = InferencePool(clients, **config.get("service", {}))
        print(json.dumps({"event": "ready", "endpoint": f"{transport}://{host}:{port}",
                          "metadata": pool.health()}, allow_nan=False), flush=True)
        if transport == "ws":
            asyncio.run(serve_websocket(pool, host, port, token, max_message_bytes=limit))
        else:
            with RPCServer((host, port), pool.handlers, token=token) as http:
                http.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        if pool is not None:
            pool.close()
        else:
            for client in clients:
                client.close()
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    serve(json.loads(Path(parser.parse_args().config).read_text()))
