#!/usr/bin/env python3
"""Replay a LeRobot FR3 demonstration through the shared simulation backend.

This sends joint targets to reset/step; objects remain governed by physics.
It neither connects to real hardware nor attaches objects to the gripper.
Comparison layout: real base | simulated base, real wrist | simulated wrist.
Each cell keeps the original 640x480 pixels and playback uses dataset FPS.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def load_episode(dataset, episode):
    """Resolve v3 shards and media offsets without changing the source dataset."""
    import pyarrow.parquet as pq
    dataset = Path(dataset).expanduser().resolve()
    info = json.loads((dataset / 'meta/info.json').read_text())
    if info.get('codebase_version') != 'v3.0':
        raise ValueError('FR3 replay requires a LeRobot v3 dataset')
    fps = float(info['fps'])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Dataset fps must be positive and finite')
    if type(episode) is not int or episode < 0:
        raise ValueError('Episode index must be a nonnegative integer')
    matches = []
    for path in sorted((dataset / 'meta/episodes').rglob('*.parquet')):
        matches.extend(pq.read_table(path, filters=[('episode_index', '=', episode)]).to_pylist())
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one episode metadata row, got {len(matches)}')
    meta = matches[0]
    rows = []
    for path in sorted((dataset / 'data').rglob('*.parquet')):
        rows.extend(pq.read_table(path, columns=['episode_index', 'frame_index', 'timestamp',
            'observation.state', 'action'], filters=[('episode_index', '=', episode)]).to_pylist())
    rows.sort(key=lambda row: row['frame_index'])
    count = int(meta['length'])
    if count < 1 or len(rows) != count or [row['frame_index'] for row in rows] != list(range(count)):
        raise ValueError('Episode frames must be complete, unique and contiguous from zero')
    state = np.asarray([row['observation.state'] for row in rows], dtype=np.float64)
    action = np.asarray([row['action'] for row in rows], dtype=np.float64)
    if any(value.shape != (count, 8) or not np.isfinite(value).all() for value in (state, action)):
        raise ValueError('Expected finite seven-joint-plus-closure state and action arrays')
    if any(np.any((value[:, 7] < 0) | (value[:, 7] > 1)) for value in (state, action)):
        raise ValueError('The eighth coordinate is normalized closure in [0,1], not width in metres')
    timestamps = np.asarray([row['timestamp'] for row in rows])
    if not np.allclose(timestamps, np.arange(count) / fps, rtol=0, atol=1e-4):
        raise ValueError('Dataset frame timestamps disagree with frame_index/fps')
    cameras = {}
    for camera in ('base', 'wrist'):
        prefix = f'videos/observation.images.{camera}'
        filename = info['video_path'].format(video_key=f'observation.images.{camera}',
            chunk_index=meta[prefix + '/chunk_index'], file_index=meta[prefix + '/file_index'])
        start = float(meta[prefix + '/from_timestamp'])
        stop = float(meta[prefix + '/to_timestamp'])
        if not math.isfinite(start) or start < 0 or not math.isclose(stop-start, count/fps, abs_tol=1e-4):
            raise ValueError(f'{camera} video interval does not cover the episode')
        cameras[camera] = dict(path=str(dataset/filename), from_timestamp=start, to_timestamp=stop)
    return dict(dataset=str(dataset), episode=episode, fps=fps, frames=count,
        task=meta.get('tasks', []), state=state, action=action, cameras=cameras)


def replay_commands(episode, source, max_frames=None):
    if source not in ('observation.state', 'action'):
        raise ValueError('Replay source must be observation.state or action')
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError('max_frames must be a positive integer')
    count = min(episode['frames'], max_frames or episode['frames'])
    # Frame zero is reset. Measured state i belongs at t=i/fps; submitted
    # action i-1 drives the interval ending at t=i/fps.
    commands = episode['state'][1:count] if source == 'observation.state' else episode['action'][:count-1]
    return count, commands.copy()


def read_layout(path):
    if path is None:
        return {}, {}
    value = json.loads(Path(path).expanduser().read_text())
    if not isinstance(value, dict):
        raise ValueError('Layout must be a JSON mapping')
    if set(value) <= {'cube', 'bowl'}:
        objects, overrides = value, {}
    else:
        if set(value) - {'objects', 'scene_overrides'}:
            raise ValueError('Layout supports objects and scene_overrides only')
        objects, overrides = value.get('objects', {}), value.get('scene_overrides', {})
    if (not isinstance(objects, dict) or set(objects)-{'cube', 'bowl'} or not isinstance(overrides, dict)):
        raise ValueError('Layout objects must contain cube/bowl mappings')
    for name, item in objects.items():
        if not isinstance(item, dict) or set(item)-{'center_xy', 'yaw_deg'}:
            raise ValueError(f'{name} supports center_xy and yaw_deg only')
        if 'center_xy' in item:
            xy = np.asarray(item['center_xy'], dtype=float)
            if xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError(f'{name} center_xy must contain two finite coordinates')
        if 'yaw_deg' in item and not math.isfinite(float(item['yaw_deg'])):
            raise ValueError(f'{name} yaw must be finite')
    return deepcopy(objects), deepcopy(overrides)


class DatasetVideo:
    """Decode the original episode frames sequentially, without time rescaling."""
    def __init__(self, spec, fps):
        import cv2
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(spec['path'])
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open source video {spec['path']}")
        actual_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not math.isclose(actual_fps, fps, abs_tol=1e-3):
            self.close()
            raise ValueError(f'Source video FPS {actual_fps} differs from dataset FPS {fps}')
        start_frame = round(spec['from_timestamp'] * actual_fps)
        if not math.isclose(start_frame / actual_fps, spec['from_timestamp'], abs_tol=1e-4):
            self.close()
            raise ValueError('Episode video offset does not lie on the declared frame grid')
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    def next(self):
        ok, image = self.capture.read()
        if not ok:
            raise RuntimeError('Source video ended before the selected episode frames')
        if image.shape != (480, 640, 3):
            raise ValueError('This calibrated comparison requires native 640x480 RGB images')
        return self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)

    def close(self):
        self.capture.release()


def compare_grid(real, sim):
    for group in (real, sim):
        if set(group) != {'base', 'wrist'} or any(np.asarray(v).shape != (480, 640, 3) for v in group.values()):
            raise ValueError('Comparison requires base and wrist at native 640x480')
    return np.ascontiguousarray(np.concatenate([
        np.concatenate([real['base'], sim['base']], axis=1),
        np.concatenate([real['wrist'], sim['wrist']], axis=1)], axis=0), dtype=np.uint8)


def assess(history, initial, dataset, requested_frames):
    states = np.asarray([row['state'] for row in history])
    references = dataset['state'][:len(history)]
    error = states - references
    positions = np.asarray([row['objects']['cube']['position_world'] for row in history])
    outcomes = [row.get('physical_outcome') or {} for row in history]
    contacts = [row.get('contacts') or {} for row in history]
    measured_initial_z = float(initial['objects']['cube']['position_world'][2])
    bilateral = [bool(len(c.get('finger_forces_n', [])) == 2 and
        min(c['finger_forces_n']) >= .02) for c in contacts]
    lift = positions[:, 2] - measured_initial_z
    return dict(full_requested_replay=len(history) == requested_frames,
        full_dataset_episode=len(history) == dataset['frames'],
        frame_count=len(history), physical_duration_s=history[-1]['elapsed_s'],
        max_cube_center_lift_m=float(np.max(lift)),
        ever_lifted_2cm=bool(np.any(lift > .02)),
        ever_bilateral_finger_contact=bool(any(bilateral)),
        ever_lifted_with_bilateral_contact=bool(any(b and z > .02 for b, z in zip(bilateral, lift))),
        ever_contained=bool(any(o.get('contained') for o in outcomes)),
        ever_supported_by_bowl=bool(any(o.get('supported') for o in outcomes)),
        ever_task_success=bool(any(o.get('success') or o.get('ever_success') for o in outcomes)),
        final_task_success=bool(outcomes[-1].get('success', False)), final_physical_outcome=outcomes[-1],
        state_tracking={'reference': 'dataset observation.state at matching frame timestamps',
            'joint_mae_rad': np.abs(error[:, :7]).mean(axis=0).tolist(),
            'joint_max_abs_rad': np.abs(error[:, :7]).max(axis=0).tolist(),
            'closure_mae': float(np.abs(error[:, 7]).mean()),
            'closure_max_abs': float(np.abs(error[:, 7]).max())},
        final_cube_position_world=positions[-1].tolist())


def verify_video(path, frames, fps, size):
    result = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=width,height,r_frame_rate,nb_read_frames','-of','json',str(path)],
        capture_output=True, text=True, check=True, timeout=60)
    stream = json.loads(result.stdout)['streams'][0]
    numerator, denominator = map(float, stream['r_frame_rate'].split('/'))
    rate = numerator/denominator
    if (int(stream['nb_read_frames']) != frames or not math.isclose(rate,fps,abs_tol=1e-6)
            or [stream['width'],stream['height']] != list(size)):
        raise RuntimeError(f'Encoded video timing or dimensions differ: {path}')
    return dict(frame_count=frames, fps=rate, dimensions=list(size), duration_s=frames/rate)


def persist_then_close(output, report, backend):
    """Keep the completed evidence even when Isaac closes the interpreter."""
    from pressb.runtime.recording import write
    # SimulationApp.close(fast_shutdown=True) may exit the interpreter
    # directly. closed=True is recorded only if the call actually returns.
    report['finished_unix'] = time.time()
    report['close_requested'] = backend is not None
    write(output/'report.json', report)
    print(json.dumps({'event':'replay_results_persisted','status':report['status'],
        'report':str(output/'report.json'),'execution':report.get('execution')}), flush=True)
    if backend is not None:
        try:
            backend.close()
            report['closed'] = True
        except Exception as exc:
            report.update(status='error',close_error=str(exc))
    report['close_returned_unix'] = time.time()
    write(output/'report.json', report)


def run(args):
    from PIL import Image
    from pressb.runtime.contract import decode_image
    from pressb.runtime.recording import Video, write
    from pressb.tasks.fr3_pap.backend import create_backend
    from pressb.tasks.fr3_pap.parallel import Fr3PapTask

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = load_episode(args.dataset, args.episode)
    count, commands = replay_commands(dataset, args.source, 1 if args.preview_only else args.max_frames)
    objects, scene_overrides = read_layout(args.layout)
    config = dict(scene_config=str(args.scene_config), scene_overrides=scene_overrides,
        num_envs=1, gpu=args.gpu, headless=True, horizon=1, control_hz=dataset['fps'],
        physics_dt=1/(4*dataset['fps']), max_episode_seconds=dataset['frames']/dataset['fps']+1.,
        render_resolution=[640, 480], image_output_resolution=[640, 480],
        smoothing={'mode': 'none'}, terminate_on_success=False,
        randomization={'enabled': False, 'bowl_enabled': False},
        log_dir=str(output/'isaac_logs'))
    # Validate all coordinates and action bounds before constructing Isaac.
    task = Fr3PapTask(config)
    specs = task.sample_reset(0, args.seed, {'initial_state': dataset['state'][0].tolist(), 'objects': objects})
    if np.any(commands < task.lower) or np.any(commands > task.upper):
        raise ValueError('Recorded targets exceed the FR3 joint/closure bounds')
    source = {key: value for key, value in dataset.items() if key not in ('state', 'action')}
    plan = dict(source=source, replay_source=args.source, requested_frames=count,
        reset=specs, target_count=len(commands), targets=commands.tolist(),
        frame_semantics='reset at real frame0; state[i] or action[i-1] drives the interval ending at frame i',
        configuration=config)
    write(output/'plan.json', plan)
    report = dict(kind='fr3_dataset_physical_replay', status='running', simulation_only=True,
        hardware_access=False, learned_policy_evaluation=False, object_attachments=False,
        object_teleport_after_reset=False, started_unix=time.time(), closed=False,
        source=source, replay_source=args.source, requested_frames=count,
        comparison_grid=[['real base', 'simulation base'], ['real wrist', 'simulation wrist']],
        fps=dataset['fps'], cell_size=[640,480], playback_speed=1.,
        cube_contact_model=deepcopy(task.config['scene']['cube'].get('compliant_contact', {})),
        limitations=[
            'The real cube is compressible foam with a50mm uncompressed edge; recorded held widths have median about43.7mm. The configured compliant-contact spring/damper approximates compression at contact, not volumetric deformation or FEM. Its stiffness and damping have not been fitted to measured forces.',
            'Joint targets are replayed through simulation drives; measured joint and contact trajectories need not match the real controller.',
            'Camera frames and joints have capture latency; equal video frame indices are nominal dataset time, not calibrated hardware synchronization.',
            'This is one recorded demonstration replay, not a learned-policy success-rate evaluation.'])
    backend, readers, videos, history = None, {}, {}, []
    try:
        readers = {camera: DatasetVideo(spec, dataset['fps']) for camera, spec in dataset['cameras'].items()}
        backend = create_backend(config)
        from pressb.runtime.parallel import SharedWorldBackend
        if not isinstance(backend, SharedWorldBackend):
            raise RuntimeError('FR3 replay must use the common SharedWorldBackend')
        report['metadata'] = backend.describe()
        rows = backend.reset([0], args.seed, {'initial_state': dataset['state'][0].tolist(), 'objects': objects})
        initial = backend.snapshot([0])['0']

        def record(row, index, command=None):
            snapshot = backend.snapshot([0])['0']
            info = deepcopy(row.get('info', {}))
            sample = dict(frame_index=index, elapsed_s=float(backend.ticks[0])*backend.execution['physics_dt'],
                state=row['observation']['state'], command=command,
                objects=deepcopy(snapshot['objects']), physical_outcome=deepcopy(snapshot.get('physical_outcome', {})),
                contacts=info.get('contacts', {}), reward=row.get('reward'),
                terminated=row.get('terminated', False), truncated=row.get('truncated', False), info=info)
            if index and not math.isclose(sample['elapsed_s'], index/dataset['fps'], abs_tol=1e-7):
                # A terminal may occur inside one control interval. Retain its
                # exact physical timestamp rather than pretending a full step.
                if not (sample['terminated'] or sample['truncated']):
                    raise RuntimeError('Simulation and dataset sample clocks differ')
            sim = {name: np.asarray(decode_image(image)) for name, image in row['observation']['images'].items()}
            real = {name: reader.next() for name, reader in readers.items()}
            grid = compare_grid(real, sim)
            if index == 0:
                Image.fromarray(grid).save(output/'initial_comparison.png')
                for name in ('base', 'wrist'):
                    Image.fromarray(real[name]).save(output/f'initial_real_{name}.png')
                    Image.fromarray(sim[name]).save(output/f'initial_sim_{name}.png')
            if not args.preview_only:
                if not videos:
                    videos.update(comparison=Video(output/'real_sim_comparison.mp4', (1280,960), dataset['fps']),
                        simulation=Video(output/'simulation_base_wrist.mp4', (640,960), dataset['fps']))
                videos['comparison'].add(grid)
                videos['simulation'].add(np.ascontiguousarray(np.concatenate([sim['base'],sim['wrist']],axis=0)))
            history.append(sample)
            return grid

        last_grid = record(rows[0], 0)
        with (output/'physics.jsonl').open('w') as stream:
            stream.write(json.dumps(history[0], allow_nan=False)+'\n')
            for index, command in enumerate(commands, 1):
                if not backend.ready[0]:
                    break
                row = backend.step([dict(env_id=0, actions=[command.tolist()])])[0]
                last_grid = record(row, index, command.tolist())
                stream.write(json.dumps(history[-1], allow_nan=False)+'\n')
                if index % 50 == 0:
                    print(json.dumps(dict(event='replay_progress',frame=index,requested_frames=count,
                        elapsed_s=history[-1]['elapsed_s'],outcome=history[-1]['physical_outcome'])), flush=True)
        Image.fromarray(last_grid).save(output/'final_comparison.png')
        audit = assess(history, initial, dataset, count)
        report.update(status='preview' if args.preview_only else 'completed', execution=audit,
            initial_snapshot=initial, final_snapshot=backend.snapshot([0])['0'],
            timings=deepcopy(backend.timings), videos={key: str(pipe.path) for key,pipe in videos.items()},
            video_frames={key:pipe.count for key,pipe in videos.items()})
        np.savez(output/'trajectory.npz', simulated_state=np.asarray([h['state'] for h in history]),
            reference_state=dataset['state'][:len(history)],
            cube_world=np.asarray([h['objects']['cube']['position_world'] for h in history]),
            time_seconds=np.asarray([h['elapsed_s'] for h in history]))
        if backend.ready[0]:
            backend.end_episode([0])
    except Exception as exc:
        import traceback
        report.update(status='error', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    finally:
        for reader in readers.values():
            reader.close()
        for video in videos.values():
            try:
                video.close()
            except Exception as exc:
                report.update(status='error', video_error=str(exc))
        if videos and report['status'] != 'error':
            try:
                report['encoded_video_validation'] = {name:verify_video(video.path,video.count,
                    dataset['fps'], (1280,960) if name=='comparison' else (640,960))
                    for name,video in videos.items()}
            except Exception as exc:
                report.update(status='error',video_error=str(exc))
        persist_then_close(output, report, backend)
    print(json.dumps({'status':report['status'],'report':str(output/'report.json'),
        'execution':report.get('execution')}), flush=True)
    return 0 if report['status'] in ('completed','preview') else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=Path('/home/pengguanqi/Datasets/pap95_20261116'))
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--scene-config', type=Path, default=Path('configs/fr3_pap.json'))
    p.add_argument('--layout', type=Path, help='JSON cube/bowl reset layout plus optional scene_overrides')
    p.add_argument('--source', choices=['observation.state','action'], default='observation.state')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--seed', type=int, default=20261116)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--preview-only', action='store_true', help='Only reset and render the initial comparison; no physical action step')
    return p


def main(argv=None):
    p = parser(); args = p.parse_args(argv)
    if args.gpu < 0 or args.episode < 0 or args.seed < 0 or args.max_frames is not None and args.max_frames < 1:
        p.error('gpu, episode and seed must be nonnegative; max-frames must be positive')
    if args.output.expanduser().exists():
        p.error('--output must be a new directory')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
