"""Resolve learner checkpoint references against a completed training run."""
from __future__ import annotations

import json
from os.path import normpath
from pathlib import Path


def resolve_checkpoint(reference, method):
    """Leave direct checkpoints intact and read local status.json on the learner."""
    if reference is None or str(reference).startswith(("http://", "https://")):
        return reference
    status_file = Path(reference)
    if status_file.name != "status.json":
        return reference
    status_file = status_file.expanduser()
    try:
        status = json.loads(status_file.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid checkpoint status JSON in {status_file}: {error}") from error
    if not isinstance(status, dict) or status.get("state") != "completed":
        raise ValueError(f"Checkpoint source {status_file} must describe a completed training run")
    if status.get("mode", "train") != "train" or status.get("method") != method:
        raise ValueError(f"Checkpoint source {status_file} must be from {method} training")
    location = status.get("last_checkpoint")
    if not isinstance(location, str) or not location.strip():
        raise ValueError(f"Checkpoint source {status_file} has no last_checkpoint")
    if location.startswith(("http://", "https://")):
        if method != "dppo":
            raise ValueError(f"Only DPPO supports an HTTP(S) last_checkpoint in {status_file}")
        return location
    checkpoint = Path(location)
    if not checkpoint.is_absolute():
        # DPPO records paths under a relative output_dir from its training cwd,
        # e.g. outputs/run/checkpoints/step_*.pt. Bind these to the source run.
        directory = status_file.resolve().parent
        run_parts = tuple(part for part in Path(normpath(checkpoint.parent.parent)).parts if part != "..")
        same_run = not run_parts or directory.parts[-len(run_parts):] == run_parts
        if checkpoint.parent.name == "checkpoints" and same_run:
            checkpoint = directory / "checkpoints" / checkpoint.name
        else:
            checkpoint = directory / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint reported by {status_file} does not exist: {location}")
    return str(checkpoint.resolve())
