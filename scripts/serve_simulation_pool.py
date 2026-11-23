#!/usr/bin/env python3
"""Expose dedicated schema-v2 simulation workers as one HTTP simulation service."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pressb.runtime.client import NodeClient
from pressb.runtime.rpc import RPCServer
from pressb.runtime.simulation_pool import SimulationPool


def _token(options):
    key = options.get("token_env")
    if key and not os.environ.get(key):
        raise ValueError(f"Token environment variable {key} is unset or empty")
    return os.environ[key] if key else None


def serve(config):
    if set(config) - {"workers", "server", "service"}:
        raise ValueError("Unknown simulation pool config section")
    workers = config.get("workers")
    if not isinstance(workers, list) or len(workers) < 2 or not all(isinstance(w, dict) for w in workers):
        raise ValueError("Specify at least two worker NodeClient configurations")
    endpoints = [w.get("endpoint", "").rstrip("/") for w in workers]
    if len(set(endpoints)) != len(endpoints):
        raise ValueError("Simulation worker endpoints must be distinct")
    server = config.get("server", {})
    if set(server) - {"host", "port", "transport", "token_env"} or server.get("transport", "http") != "http":
        raise ValueError("Simulation pool server accepts HTTP host, port and optional token_env")
    address = (server.get("host", "127.0.0.1"), server.get("port", 22993))
    token = _token(server)
    clients, pool = [], None

    def stop(_signal, _frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        for worker in workers:
            if "token" in worker:
                raise ValueError("Use token_env instead of inline worker credentials")
            client_options = {key: value for key, value in worker.items() if key != "token_env"}
            clients.append(NodeClient(**client_options, token=_token(worker)))
        pool = SimulationPool(clients, **config.get("service", {}))
        with RPCServer(address, pool.handlers, token=token) as http:
            print(json.dumps(dict(event="ready", endpoint=f"http://{address[0]}:{http.server_port}",
                                  metadata=pool.health()), allow_nan=False), flush=True)
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
    parser.add_argument("--config", required=True, type=Path)
    serve(json.loads(parser.parse_args().config.read_text()))
