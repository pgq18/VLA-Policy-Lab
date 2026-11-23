"""CPU checks for dataset time alignment and honest physical replay reporting."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

spec = importlib.util.spec_from_file_location('replay_fr3_dataset',
    Path(__file__).parents[1]/'scripts/replay_fr3_dataset.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def make_dataset(tmp_path):
    (tmp_path/'meta/episodes/chunk-000').mkdir(parents=True)
    (tmp_path/'data/chunk-000').mkdir(parents=True)
    info = dict(codebase_version='v3.0',fps=25,
        video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4')
    (tmp_path/'meta/info.json').write_text(json.dumps(info))
    meta = dict(episode_index=2,length=3,tasks=['Pick up blue cube and place it in the bowl.'])
    for camera in ('base','wrist'):
        prefix = f'videos/observation.images.{camera}'
        meta.update({prefix+'/chunk_index':0,prefix+'/file_index':1,
            prefix+'/from_timestamp':10.,prefix+'/to_timestamp':10.12})
    pq.write_table(pa.Table.from_pylist([meta]),tmp_path/'meta/episodes/chunk-000/file-000.parquet')
    rows = [dict(episode_index=2,frame_index=i,timestamp=i/25,
        **{'observation.state':[i*.1]*7+[0.], 'action':[.5+i*.1]*7+[1.]}) for i in range(3)]
    # Deliberately span two shards: replay must use all rows and preserve order.
    pq.write_table(pa.Table.from_pylist(rows[:2]),tmp_path/'data/chunk-000/file-000.parquet')
    pq.write_table(pa.Table.from_pylist(rows[2:]),tmp_path/'data/chunk-000/file-001.parquet')
    return rows


def test_v3_shards_and_offsets(tmp_path):
    make_dataset(tmp_path)
    result = replay.load_episode(tmp_path,2)
    assert result['state'].shape == (3,8)
    assert result['cameras']['base']['from_timestamp'] == 10.
    assert result['cameras']['base']['path'].endswith('observation.images.base/chunk-000/file-001.mp4')


def test_measured_and_submitted_targets_use_different_time_indices(tmp_path):
    make_dataset(tmp_path)
    ep = replay.load_episode(tmp_path,2)
    _, state = replay.replay_commands(ep,'observation.state')
    _, action = replay.replay_commands(ep,'action')
    np.testing.assert_array_equal(state,ep['state'][1:])
    np.testing.assert_array_equal(action,ep['action'][:-1])
    assert replay.replay_commands(ep,'action',1)[1].shape == (0,8)


def test_duplicate_or_bad_timestamp_rejected(tmp_path):
    rows = make_dataset(tmp_path)
    rows[-1]['timestamp'] = .12
    pq.write_table(pa.Table.from_pylist(rows[2:]),tmp_path/'data/chunk-000/file-001.parquet')
    with pytest.raises(ValueError,match='timestamps'):
        replay.load_episode(tmp_path,2)


def test_layout_does_not_accept_per_frame_object_poses(tmp_path):
    path = tmp_path/'layout.json'
    path.write_text(json.dumps({'objects':{'cube':{'position_world':[.4,.1,.025]}}}))
    with pytest.raises(ValueError,match='center_xy'):
        replay.read_layout(path)


def test_no_lift_or_bowl_contact_is_not_reported_success():
    def sample(i,z,outcome):
        return dict(state=[0.]*8,elapsed_s=i/25,
            objects={'cube':{'position_world':[.4,.1,z]}},
            physical_outcome=outcome,contacts={'finger_forces_n':[0.,0.]})
    history=[sample(0,.025,{}),sample(1,.026,{'contained':True,'supported':False,'success':False})]
    result=replay.assess(history,history[0],{'state':np.zeros((3,8)),'frames':3},3)
    assert result['ever_contained']
    assert not result['ever_task_success']
    assert not result['ever_supported_by_bowl']
    assert not result['ever_lifted_2cm']
    assert not result['full_dataset_episode']
    assert not result['full_requested_replay']


def test_real_sim_grid_has_correct_camera_and_source_order():
    im=lambda value:np.full((480,640,3),value,dtype=np.uint8)
    result=replay.compare_grid({'base':im(10),'wrist':im(30)}, {'base':im(20),'wrist':im(40)})
    assert result.shape==(960,1280,3)
    assert [int(result[y,x,0]) for y,x in [(0,0),(0,640),(480,0),(480,640)]] == [10,20,30,40]


def test_fast_shutdown_cannot_discard_replay_evidence(tmp_path):
    class FastShutdown:
        def close(self):
            raise SystemExit(0)
    report=dict(status='completed',closed=False,
        execution={'ever_task_success':False},encoded_video_validation={'frame_count':437})
    with pytest.raises(SystemExit):
        replay.persist_then_close(tmp_path,report,FastShutdown())
    saved=json.loads((tmp_path/'report.json').read_text())
    assert saved['execution']=={'ever_task_success':False}
    assert saved['encoded_video_validation']=={'frame_count':437}
    assert saved['close_requested'] and not saved['closed']
    assert 'close_returned_unix' not in saved
