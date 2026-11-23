import argparse
import json
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from pressb.runtime.client import NodeClient
from pressb.runtime.inference_pool import InferencePool
from pressb.runtime.trainer import GenericTrainer, TrainerConfig
from paired_fr3 import PairedFR3Client


class DirectPool(InferencePool):
    def call(self, path, payload=None):
        return self.handlers[path](payload or {})


class Trainer(GenericTrainer):
    def __init__(self, *args, segment, overflow_dir, **kwargs):
        self.segment = segment
        self.overflow_dir = overflow_dir
        super().__init__(*args, **kwargs)

    def _status_write(self, state, **extra):
        super()._status_write(state, **extra)
        if hasattr(self.simulation, "verified_layouts"):
            p = self.segment / "layout_status.json"
            tmp = p.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "state": state,
                        "verified_by_shard": self.simulation.verified_layouts,
                        "transitions": self.transitions,
                    }
                )
                + "\n"
            )
            tmp.replace(p)

    def _negotiate(self):
        super()._negotiate()
        if self.overflow_dir is None:
            return
        save = self.learner.save
        overflow = self.overflow_dir

        def save_to_disk(path, *, extra):
            path = Path(path)
            step = extra["transitions"]
            # Retain every full snapshot. The data disk also carries standalone
            # policy weights for every snapshot whose replay lives on the system disk.
            if (
                step // self.config.checkpoint.interval % 2 == 0
                or step >= self.config.rollout.max_transitions
            ):
                return save(path, extra=extra)
            overflow.mkdir(parents=True, exist_ok=True)
            if (
                overflow == path.parent.resolve()
                or overflow == (path.parent / "weights_only").resolve()
            ):
                raise ValueError(
                    "Overflow directory must be separate from both checkpoint directories"
                )
            full = overflow / path.name
            save(full, extra=extra)
            weights = path.parent / "weights_only" / path.name
            state = {k: v for k, v in extra.items() if k != "replay"}
            state["full_resume_checkpoint"] = str(full)
            save(weights, extra=state)
            link = path.with_suffix(".link-stage")
            link.unlink(missing_ok=True)
            link.symlink_to(full)
            os.replace(link, path)

        self.learner.save = save_to_disk


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the FR3 two-shard residual learner against four running nodes."
    )
    parser.add_argument(
        "segment",
        type=Path,
        help="Run segment containing configs/trainer.json and simulation0.json",
    )
    parser.add_argument(
        "--overflow-dir", type=Path, help="Store alternate full checkpoints on this filesystem"
    )
    args = parser.parse_args()
    segment = args.segment.resolve()
    config = TrainerConfig(**json.loads((segment / "configs/trainer.json").read_text()))
    clients = [
        NodeClient(
            f"ws://127.0.0.1:{port}",
            timeout=300,
            max_message_bytes=536870912,
            image_transport="auto",
            image_workers=8,
        )
        for port in (21953, 21954)
    ]
    pool = DirectPool(clients, max_batch_size=128)
    sim = PairedFR3Client(
        [
            NodeClient(f"http://127.0.0.1:{port}", timeout=300, max_message_bytes=536870912)
            for port in (21951, 21952)
        ],
        json.loads((segment / "configs/simulation0.json").read_text())["backend"]["config"],
    )
    try:
        (segment / "simulation_health.json").write_text(
            json.dumps(sim.call("/health"), indent=2) + "\n"
        )
        (segment / "pool_health.json").write_text(json.dumps(pool.health(), indent=2) + "\n")
        runner = Trainer(
            config,
            inference_client=pool,
            simulation_client=sim,
            segment=segment,
            overflow_dir=args.overflow_dir.resolve() if args.overflow_dir else None,
        )
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: runner.stop())
        print(json.dumps(runner.run(), indent=2), flush=True)
    finally:
        sim.close()
        pool.close()
