#!/usr/bin/env python3
"""Run DSRL, action residuals, joint action/noise residuals or DPPO against inference and environment nodes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--inference")
    parser.add_argument("--simulation")
    parser.add_argument("--resume", help="Full RL checkpoint path (or HTTP(S) URL for DPPO); simulation starts fresh episodes")
    parser.add_argument("--mode", choices=("train", "eval"))
    parser.add_argument("--max-transitions", type=int)
    parser.add_argument("--eval-episodes", type=int)
    console = parser.add_mutually_exclusive_group()
    console.add_argument("--operator-console", dest="operator_console", action="store_true",
                         help="Enter real-robot readiness and outcomes in this evaluation terminal")
    console.add_argument("--no-operator-console", dest="operator_console", action="store_false")
    parser.set_defaults(operator_console=None)
    parser.add_argument("--token-env", default="PRESSB_RL_TOKEN")
    args = parser.parse_args()
    data = json.loads(args.config.read_text())
    console_settings = data.pop("operator_console", None)
    console_enabled = bool(console_settings) if args.operator_console is None else args.operator_console
    if console_enabled and not isinstance(console_settings, dict):
        parser.error("--operator-console requires operator_console SSH settings in the run config")
    for name in ("device", "mode", "seed"):
        if getattr(args, name) is not None:
            data[name] = getattr(args, name)
    if args.output:
        data["output_dir"] = str(args.output)
    for name in ("inference", "simulation"):
        if getattr(args, name):
            data.setdefault("endpoints", {})[name] = getattr(args, name)
    if os.environ.get(args.token_env):
        data.setdefault("endpoints", {})["token"] = os.environ[args.token_env]
    if args.resume:
        data.setdefault("checkpoint", {})["resume"] = str(args.resume)
    for name in ("max_transitions", "eval_episodes"):
        if getattr(args, name) is not None:
            data.setdefault("rollout", {})[name] = getattr(args, name)
    if data.get("mode") == "eval":
        # A saved training config can be used with --mode eval --resume ... .
        data.pop("residual_warmup", None)
        data.pop("residual_initialize_from", None)
    if data.get("method") == "dppo":
        from pressb.runtime.dppo_trainer import DPPOTrainer as GenericTrainer, DPPOTrainerConfig as TrainerConfig
    else:
        from pressb.runtime.trainer import GenericTrainer, TrainerConfig
    runner = GenericTrainer(TrainerConfig(**data))
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: runner.stop())
    if console_enabled:
        from pressb.runtime.operator_console import run_with_operator_console
        result = run_with_operator_console(runner, console_settings)
    else:
        result = runner.run()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
