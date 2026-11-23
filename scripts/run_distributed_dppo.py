#!/usr/bin/env python3
"""Run one DPPO learner across synchronous local accelerator replicas.

Launch with torchrun (or ascend/torchrun_local.py on the configured Ascend host).
Only rank zero owns external RPCs, logging/checkpoints and policy publication.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import signal
import threading

from benchmark_dppo_training import PhaseRecorder, load_runtime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--backend', choices=('auto', 'gloo', 'nccl', 'hccl'), default='auto')
    parser.add_argument('--collective-timeout', type=int, default=900)
    parser.add_argument('--control-timeout', type=int, default=3600,
                        help='Worker control wait must cover leader rollout, failure saving and RPC cleanup')
    parser.add_argument('--gradient-bucket-mib', type=int, default=32)
    parser.add_argument('--phase-log', type=Path)
    parser.add_argument('--phase-file', type=Path)
    parser.add_argument('--skip-checkpoints', action='store_true')
    parser.add_argument('--stop-file', type=Path,
                        help='Rank zero requests a graceful iteration-boundary stop when this file appears')
    parser.add_argument('--leader-pid-file', type=Path,
                        help='Write rank-zero process identity outside the trainer output directory')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--token-env', default='PRESSB_RL_TOKEN')
    args = parser.parse_args(argv)
    if min(args.collective_timeout, args.control_timeout, args.gradient_bucket_mib) < 1:
        parser.error('Collective timeout and gradient bucket size must be positive')
    runtime, provenance = load_runtime(args.source_root)
    import torch
    import torch.distributed as dist
    from pressb.runtime.accelerator import initialize_device
    from pressb.runtime.distributed_dppo import DistributedDPPOTrainer, LearnerCollectives
    import pressb.runtime.distributed_dppo as distributed_module

    data = json.loads(args.config.read_text())
    if args.output is not None:
        data['output_dir'] = str(args.output.expanduser().resolve())
    if os.environ.get(args.token_env):
        data.setdefault('endpoints', {})['token'] = os.environ[args.token_env]
    config = runtime.DPPOTrainerConfig(**data)
    if config.mode != 'train':
        raise ValueError('Use a single evaluator for evaluation; this launcher is for distributed training')
    device_type = str(config.device).split(':', 1)[0]
    expected_backend = {'cpu': 'gloo', 'cuda': 'nccl', 'npu': 'hccl'}.get(device_type)
    backend = expected_backend if args.backend == 'auto' else args.backend
    if backend != expected_backend:
        raise ValueError(f'Device {device_type} requires {expected_backend}, received {backend}')
    output = Path(config.output_dir).expanduser().resolve()
    directory = output.with_name(output.name + '_benchmark')
    phase_log = (args.phase_log or directory / 'phases.jsonl').expanduser().resolve()
    phase_file = (args.phase_file or directory / 'phase.json').expanduser().resolve()
    if phase_log == phase_file or any(p.is_relative_to(output) for p in (phase_log, phase_file)):
        raise ValueError('Phase logs must be distinct and outside the learner output')
    provenance.update(distributed_module=str(Path(distributed_module.__file__).resolve()),
        distributed_sha256=hashlib.sha256(Path(distributed_module.__file__).read_bytes()).hexdigest(),
        config=str(args.config.resolve()), output_dir=str(output), backend=backend,
        global_minibatch_size=config.dppo.minibatch_size, skip_checkpoints=args.skip_checkpoints)
    if args.check_only:
        print(json.dumps(dict(passed=True, processes_started=False, **provenance)))
        return 0
    if args.stop_file is not None and args.stop_file.exists():
        raise FileExistsError('Choose a new stop-file path; an existing request must not be ignored')
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world < 2:
        raise ValueError('Use the normal DPPO runner for a single learner')
    config.device = 'cpu' if device_type == 'cpu' else f'{device_type}:{local_rank}'
    device = initialize_device(config.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    timeout = timedelta(seconds=args.collective_timeout)
    dist.init_process_group(backend=backend, timeout=timeout)
    control_group = dist.new_group(backend='gloo', timeout=timedelta(seconds=args.control_timeout))
    context = LearnerCollectives(device, control_group=control_group,
                                gradient_bucket_bytes=args.gradient_bucket_mib * 1024**2)
    runner, recorder, failure, result = None, None, None, None
    monitor_stop, monitor_thread = threading.Event(), None
    handlers = {}
    try:
        runner = DistributedDPPOTrainer(config, collectives=context)
        if rank:
            runner.serve_worker()
            return 0
        if args.leader_pid_file is not None:
            path = args.leader_pid_file.expanduser().resolve()
            if path.is_relative_to(output) or path.exists():
                raise ValueError('Leader receipt must be a fresh path outside trainer output')
            runtime._atomic_json(path, {'pid': os.getpid(), 'role': 'learner_rank0',
                'start_ticks': Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'world_size': world, 'device': str(device)})
        if args.stop_file is not None:
            def monitor():
                while not monitor_stop.wait(.5):
                    if args.stop_file.exists():
                        runner.stop()
                        return
            monitor_thread = threading.Thread(target=monitor, name='dppo-stop-file', daemon=True)
            monitor_thread.start()
        provenance.update(world_size=world, rank_zero_only_rpc=True)
        print(json.dumps(dict(event='distributed_benchmark_import_verified', **provenance)), flush=True)
        recorder = PhaseRecorder(phase_log, phase_file, skip_checkpoints=args.skip_checkpoints)
        recorder.record('benchmark_start', active_phase='startup', provenance=provenance)
        recorder.attach(runner)
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda *_: runner.stop())
        result = runner.run()
        print(json.dumps(result, indent=2), flush=True)
        return 0
    except BaseException as error:
        failure = error
        raise
    finally:
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=1)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if runner is not None and rank == 0:
            runner.shutdown_workers()
        if recorder is not None:
            state = 'failed' if failure else result.get('state', 'completed') if result else 'completed'
            recorder.record('benchmark_end', active_phase=state, outcome=state,
                error_type=type(failure).__name__ if failure else None,
                error=str(failure) if failure else None, result=result)
        if control_group is not None:
            dist.destroy_process_group(control_group)
        dist.destroy_process_group()


if __name__ == '__main__':
    raise SystemExit(main())
