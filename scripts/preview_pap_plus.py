#!/usr/bin/env python3
"""Render one PAP variant and check the shared GPU reset/step path.

Run each variant in a separate process: Isaac may exit the interpreter during
fast shutdown. This script never loads a policy or commands a physical robot.
Images use native camera resolution; only rollout duration and preview transport
are overridden. Task cameras, robot, geometry, materials and renderer stay as
configured.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import importlib
import io
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from smoke_pap_task_backend import check_rows, hold_items, require


def write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--task', choices=('pap','pap_plus','pap_plus_plus'), required=True)
    result.add_argument('--config', type=Path, required=True, help='Resolved simulation node JSON')
    result.add_argument('--num-envs', type=int, default=2)
    result.add_argument('--gpu-id', type=int, default=0)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--output-dir', type=Path, required=True)
    return result


def prepare(args):
    require(args.num_envs >= 2, 'Use at least two environments for reset-isolation checks')
    require(args.gpu_id >= 0 and args.seed >= 0, 'GPU id and seed must be nonnegative')
    document=json.loads(args.config.expanduser().resolve().read_text())
    config=deepcopy(document['backend']['config'])
    # Use the exact same recorded initial state in both variants. The plugin
    # resolves its own scene defaults; a node file may also specify the path.
    config.update(num_envs=args.num_envs,gpu_ids=[args.gpu_id],gpu=args.gpu_id,
                  python=sys.executable,physics_backend='gpu',gpu_dynamics=True,
                  log_dir=str(args.output_dir/'workers'),image_preprocessing=None)
    config.pop('workers',None)
    spec='pressb.tasks.'+args.task+'.parallel'
    module=importlib.import_module(spec)
    factory=module.create_backend
    # Parent configuration is fixed at 30Hz for PAP/PAPPlus, but obtain the rate
    # from its effective scene rather than relying on that implementation fact.
    task_class=getattr(module, {'pap':'PAPTask', 'pap_plus':'PAPPlusTask',
                               'pap_plus_plus':'PAPPlusPlusTask'}[args.task])
    task=task_class(config)
    hz=float(task.execution['control_hz'])
    config['max_episode_seconds']=2./hz
    return config,factory


def save_images(rows, directory):
    from PIL import Image
    evidence={}
    for row in rows:
        paths={}
        for camera,payload in row['observation']['images'].items():
            require(payload['encoding']=='png_base64','Expected PNG base64 backend preview')
            raw=base64.b64decode(payload['data'],validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                require(image.mode=='RGB','Expected RGB camera image')
                image.load()
                name=f'env{row["env_id"]}_{camera}.png'
                image.save(directory/name)
                paths[camera]={'path':str(directory/name),'size':list(image.size)}
        evidence[str(row['env_id'])]=paths
    return evidence


def scene_evidence(backend,task_name):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    engine=backend.engine
    scene=engine.template.data['scene']
    cfg=backend.task.config['scene']
    outer=np.asarray(cfg['box']['size'],dtype=float)
    thickness=float(cfg['box']['wall_thickness'])
    cavity=scene['box_inside']
    expected_inner=np.r_[outer[:2]-2*thickness,outer[2]-thickness]
    measured_inner=np.asarray(cavity['max_local'])-cavity['min_local']
    np.testing.assert_allclose(measured_inner,expected_inner,atol=1e-9,rtol=0)
    require(np.all(expected_inner>0),'Open box has no usable cavity')
    expected_faces={
        'Bottom':([0,0,thickness/2],[outer[0],outer[1],thickness]),
        'Front':([-(outer[0]-thickness)/2,0,(outer[2]+thickness)/2],[thickness,outer[1],outer[2]-thickness]),
        'Back':([(outer[0]-thickness)/2,0,(outer[2]+thickness)/2],[thickness,outer[1],outer[2]-thickness]),
        'Right':([0,-(outer[1]-thickness)/2,(outer[2]+thickness)/2],[outer[0]-2*thickness,thickness,outer[2]-thickness]),
        'Left':([0,(outer[1]-thickness)/2,(outer[2]+thickness)/2],[outer[0]-2*thickness,thickness,outer[2]-thickness]),
    }
    clones=[]
    for env_id in range(backend.num_envs):
        box_path=engine.relocate(env_id,scene['box_path'])
        box_prim=engine.stage.GetPrimAtPath(box_path)
        require(bool(box_prim),'Box prim missing from cloned environment')
        colliders=[]
        for prim in Usd.PrimRange(box_prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):colliders.append(prim)
        require({prim.GetName() for prim in colliders}==set(expected_faces),'Box must have exactly five physical collision faces')
        for prim in colliders:
            position,size=expected_faces[prim.GetName()]
            require(prim.IsA(UsdGeom.Cube),'Expected cubic static collision faces')
            require(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False,'Box collider disabled')
            np.testing.assert_allclose(prim.GetAttribute('xformOp:translate').Get(),position,atol=1e-8,rtol=0)
            scale=np.asarray(prim.GetAttribute('xformOp:scale').Get())*UsdGeom.Cube(prim).GetSizeAttr().Get()
            np.testing.assert_allclose(scale,size,atol=1e-8,rtol=0)
        root=engine.relocate(env_id,scene['root_path'])
        markers=[str(prim.GetPath()) for prim in Usd.PrimRange(engine.stage.GetPrimAtPath(root)) if '/Tape/Marker' in str(prim.GetPath())]
        expected_markers=len(cfg.get('tape_markers',[]))
        require(len(markers)==expected_markers,'Rendered marker count differs from scene configuration')
        if task_name in ('pap_plus','pap_plus_plus'):
            require(not markers,'PAP variants must have no tabletop marker geometry')
        clones.append({'env_id':env_id,'box_path':box_path,'collision_face_count':len(colliders),
                       'marker_count':len(markers),'marker_paths':markers})
    return {'outer_size_m':outer.tolist(),'inner_size_m':measured_inner.tolist(),
            'wall_thickness_m':thickness,'box_center_xy':cfg['box']['center_xy'],
            'box_yaw_deg':cfg['box']['yaw_deg'],'clones':clones,
            'cameras':{key:deepcopy(cfg[key]) for key in ('up_camera','wrist_camera')},
            'object_is_free_rigid_body':True,'full_grasp_policy_not_evaluated':True}


def run(args):
    args.output_dir=args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report={'status':'preparing','task':args.task,'num_envs':args.num_envs,'gpu_id':args.gpu_id,
            'seed':args.seed,'closed':False,'close_attempted':False,'timing':{}}
    backend=None
    start=time.monotonic()
    def timed(name,call):
        begin=time.monotonic()
        try:return call()
        finally:report['timing'][name+'_seconds']=time.monotonic()-begin
    def record():
        report['timing']['wall_seconds']=time.monotonic()-start
        write(args.output_dir/'report.json',report)
    try:
        config,factory=prepare(args)
        write(args.output_dir/'config.json',config)
        report['preview_overrides']={'num_envs':args.num_envs,'gpu_ids':[args.gpu_id],
            'max_episode_seconds':config['max_episode_seconds'],'physics_backend':'gpu',
            'image_preprocessing':None,'purpose':'native-resolution images plus bounded reset/hold/timeout validation'}
        backend=timed('startup',lambda:factory(config))
        metadata=backend.describe()
        write(args.output_dir/'metadata.json',metadata)
        require(metadata['task_id']==args.task and metadata['num_envs']==args.num_envs,'Wrong task metadata')
        require(metadata['execution']['gpu_dynamics'] is True and metadata['execution']['physics_backend_verified'] is True,'GPU PhysX did not activate')
        report['scene']=scene_evidence(backend,args.task)
        cfg=backend.task.config['scene']
        obj=cfg.get('object') or cfg['cube']
        fixed={'center_xy':obj['center_xy'],'yaw_deg':obj.get('yaw_deg',0.),'color':'red'}
        ids=list(range(args.num_envs))
        before_ticks=backend.timings['world_ticks']
        initial=timed('reset',lambda:backend.reset(ids,args.seed,fixed))
        check_rows(initial,ids)
        require(backend.timings['world_ticks']==before_ticks,'Initial reset advanced physics')
        report['fixed_initial_object']=fixed
        report['images']=save_images(initial,args.output_dir)
        report['initial_states']={str(row['env_id']):row['observation']['state'] for row in initial}
        peers=ids[1:]
        before=backend.snapshot(peers)
        subset=timed('indexed_reset',lambda:backend.reset([0],args.seed+1,{}))
        require(backend.timings['world_ticks']==before_ticks,'Indexed reset advanced physics')
        require(before==backend.snapshot(peers),'Indexed reset changed peer physical state')
        check_rows(subset,[0])
        initial=[subset[0],*initial[1:]]
        report['indexed_reset']={'reset_env_ids':[0],'peer_env_ids':peers,'zero_ticks':True,
                                 'peer_state_identical':True,'sampled_reset':subset[0]['info']['reset']}
        ordinary=timed('step',lambda:backend.step(hold_items(initial,metadata)))
        check_rows(ordinary,ids)
        require(all(not (row['terminated'] or row['truncated']) for row in ordinary),'Episode ended before exact timeout')
        before_fused=backend.timings['world_ticks']
        fused=timed('fused',lambda:backend.step_and_reset(hold_items(ordinary,metadata),
            reset_seed=args.seed+2,reset_options={},max_active_envs=args.num_envs))
        terminal,resets=fused['items'],fused['reset_items']
        check_rows(terminal,ids,images=False);check_rows(resets,ids)
        expected_ticks=2*backend.execution['physics_ticks_per_action']
        require(all(row['truncated'] and not row['terminated'] and row['info']['termination_reason']=='timeout'
                    and row['info']['episode_physics_steps']==expected_ticks
                    and row.get('terminal_observation_omitted') is True for row in terminal),'Timeout/terminal observation semantics differ')
        require(all(row['info']['episode_physics_steps']==0 for row in resets),'Fused reset did not clear episode clock')
        advanced=backend.timings['world_ticks']-before_fused
        require(advanced==max(row['info']['physics_steps'] for row in terminal),'Fused reset advanced extra physics')
        report['fused']={'timeout_env_ids':ids,'reset_env_ids':ids,'episode_physics_steps':expected_ticks,
                         'world_ticks':advanced,'terminal_images_omitted':True,'reset_images_present':True}
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        import traceback
        traceback.print_exc()
    finally:
        if backend is not None:
            report['timing']['backend']=dict(backend.timings)
            report['operations_status']=report['status']
            report['close_attempted']=True
            if report['status']=='passed':report['status']='closing'
            record()
            try:
                timed('close',backend.close)
                report['closed']=True
                if report['status']=='closing':report['status']='passed'
            except Exception as error:report.update(status='failed',close_error=str(error))
        record()
        print(json.dumps(report,allow_nan=False),flush=True)
    return 0 if report['status']=='passed' else 1


def main():
    args=parser().parse_args()
    if args.output_dir.expanduser().exists():raise SystemExit('--output-dir must be new')
    raise SystemExit(run(args))


if __name__=='__main__':main()
