"""PAP++ keeps the PAP interaction contract with a smaller, less shifted box."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from pressb.experiments.config import load_experiment
from pressb.registry import resolve
from pressb.tasks.pap.parallel import PAPTask
from pressb.tasks.pap.task import PickPlaceOutcome, object_inside_box
from pressb.tasks.pap_plus.parallel import PAPPlusTask
from pressb.tasks.pap_plus_plus.configuration import load_config, sample_reset
from pressb.tasks.pap_plus_plus.parallel import PAPPlusPlusTask, create_backend


ROOT = Path(__file__).resolve().parents[1]


def _scene(suffix=""):
    return json.loads((ROOT / "configs" / f"piper_pick_place{suffix}.json").read_text())


def _cavity(scene):
    box = scene["box"]
    return {
        "center_world": [*np.add(scene["robot_base_xy"], box["center_xy"]), scene["table_height"]],
        "yaw_deg": box["yaw_deg"],
        "inner_size_xy": (np.asarray(box["size"][:2]) - 2 * box["wall_thickness"]).tolist(),
        "floor_z_world": scene["table_height"] + box["wall_thickness"],
        "rim_z_world": scene["table_height"] + box["size"][2],
    }


def _rest_pose(scene, local_xy=(0., 0.)):
    box = _cavity(scene)
    angle = np.deg2rad(box["yaw_deg"])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = np.asarray(box["center_world"][:2]) + rotation @ local_xy
    position = np.r_[xy, box["floor_z_world"] + .015]
    return position, [np.cos(angle / 2), 0., 0., np.sin(angle / 2)]


def test_scene_changes_only_box_size_and_offset():
    base, plus, smaller = _scene(), _scene("_plus"), _scene("_plus_plus")
    np.testing.assert_allclose(smaller["box"]["size"], np.asarray(plus["box"]["size"]) * [.648, .648, 1.], atol=1e-12)
    np.testing.assert_allclose(smaller["box"]["size"], np.asarray(base["box"]["size"]) * [.5184, .5184, 1.], atol=1e-12)
    np.testing.assert_allclose(smaller["box"]["center_xy"], np.asarray(base["box"]["center_xy"]) + [.03, 0.], atol=1e-12)
    for key in ("yaw_deg", "wall_thickness"):
        assert smaller["box"][key] == plus["box"][key] == base["box"][key]
    # This catches accidental camera, initial-pose, material or object changes.
    assert {k: v for k, v in smaller.items() if k not in ("task", "box", "difficulty_variant")} == {
        k: v for k, v in plus.items() if k not in ("task", "box", "difficulty_variant")}
    assert smaller["task"] == "piper_pick_place_plus_plus"
    assert smaller["tape_markers"] == []
    assert smaller["sampling_region_edges"] == base["tape_markers"]


def test_seeded_sampling_is_unchanged_and_input_is_not_mutated():
    options = {"randomization": {"position": "markers", "colors": ["red", "blue"],
                                  "yaw_range_deg": [-180., 180.]}}
    before = deepcopy(options)
    tasks = [PAPTask(options), PAPPlusTask(options), PAPPlusPlusTask(options)]
    for seed, env_id in [(0, 0), (42, 3), (4242, 7), (999, 1)]:
        for overrides in ({}, {"yaw_deg": 37., "color": "yellow"}):
            expected = tasks[0].sample_reset(env_id, seed, overrides)
            assert tasks[1].sample_reset(env_id, seed, overrides) == expected
            assert tasks[2].sample_reset(env_id, seed, overrides) == expected
    assert options == before
    config = load_config(options)
    assert sample_reset(config, 4242, 7, {}) == tasks[2].sample_reset(7, 4242, {})
    assert config["scene"]["tape_markers"] == []


def test_independent_identity_preserves_observation_action_and_reward_contracts():
    tasks = [PAPTask({}), PAPPlusTask({}), PAPPlusPlusTask({})]
    assert [task.metadata()["task_id"] for task in tasks] == ["pap", "pap_plus", "pap_plus_plus"]
    assert len({task.identity["task_config_sha256"] for task in tasks}) == 3
    assert len({task.identity["source_revision"] for task in tasks}) == 3
    for task in tasks[1:]:
        for key in ("observation", "action", "reward_schema"):
            assert task.metadata()[key] == tasks[0].metadata()[key]
        np.testing.assert_array_equal(task.initial, tasks[0].initial)
        assert task.execution["reset_contract"] == tasks[0].execution["reset_contract"]
    assert "tasks/pap_plus_plus/parallel.py" in tasks[2].identity["source_identity"]
    assert tasks[2].identity["task_config_sha256"] != PAPPlusPlusTask({
        "randomization": {"position": "fixed"}}).identity["task_config_sha256"]


def test_smaller_cavity_rejects_an_object_that_fits_plus_at_same_relative_location():
    plus, smaller = _scene("_plus"), _scene("_plus_plus")
    # Both centres are inside their own boxes, but the whole 3 cm cube only fits
    # PAP+ when displaced 22 mm along the box's short axis.
    for scene, expected in ((plus, True), (smaller, False)):
        position, quat = _rest_pose(scene, (0., .022))
        assert object_inside_box(position, quat, .03, _cavity(scene)) == expected


def test_success_uses_new_physical_box_and_still_requires_lift_release_and_rest():
    scene = _scene("_plus_plus")
    center, quat = _rest_pose(scene)
    checker = PickPlaceOutcome(.03, scene["table_height"], _cavity(scene))
    # Merely starting inside the box is not a completed pick/place episode.
    assert not checker.update(1., center, quat, np.zeros(6), .08)["success"]
    assert not checker.update(2., center, quat, np.zeros(6), .08)["success"]
    lifted = center.copy()
    lifted[2] += .07
    checker.update(3., lifted, quat, np.zeros(6), .025, finger_contact=True)
    held = checker.update(4., center, quat, np.zeros(6), .025, finger_contact=True)
    assert held["inside_box"] and held["lifted"] and not held["success"]
    assert not checker.update(5., center, quat, np.zeros(6), .08)["success"]
    assert checker.update(5.6, center, quat, np.zeros(6), .08)["success"]
    # The old +6 cm centre is beyond the new cavity once cube extent is counted.
    old_center, old_quat = _rest_pose(_scene("_plus"))
    assert not object_inside_box(old_center, old_quat, .03, _cavity(scene))


def test_custom_scene_and_geometry_overrides_are_respected(tmp_path):
    scene = _scene("_plus_plus")
    scene["box"]["size"] = [.09, .08, .063]
    scene["wrist_camera"]["fx"] += 10.
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(scene))
    config = {"scene_config": str(path), "scene_overrides": {
        "box": {**scene["box"], "center_xy": [.34, .21]}}}
    task = PAPPlusPlusTask(config)
    loaded = load_config(config)
    assert task.config["scene"] == loaded["scene"]
    assert task.config["scene"]["box"]["size"] == [.09, .08, .063]
    assert task.config["scene"]["box"]["center_xy"] == [.34, .21]
    assert task.config["scene"]["wrist_camera"]["fx"] == scene["wrist_camera"]["fx"]


def test_registry_factory_uses_common_parallel_runtime(monkeypatch):
    import pressb.runtime.parallel as shared
    monkeypatch.setattr(shared, "SharedWorldBackend", lambda task: ("shared", task))
    assert resolve("task", "pap_plus_plus") == "pressb.tasks.pap_plus_plus.parallel:create_backend"
    kind, task = create_backend({})
    assert kind == "shared" and isinstance(task, PAPPlusPlusTask)


@pytest.mark.parametrize("method", ["dsrl", "action_residual", "dppo"])
def test_new_task_uses_common_gpu_rl_default(tmp_path, method):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps({"schema_version": 1, "name": "pappp_defaults",
        "task": {"plugin": "pap_plus_plus", "config": {}},
        "model": {"plugin": "walloss", "config": {}},
        "run": {"method": method, "mode": "train"}}))
    compiled = load_experiment(path)
    assert compiled["nodes"]["simulation"]["backend"]["config"]["physics_backend"] == "gpu"


@pytest.mark.parametrize("mode", ["eval", "dsrl"])
def test_example_experiments_select_new_task_and_scene(mode):
    compiled = load_experiment(ROOT / "configs" / "experiments" / f"pap_plus_plus_walloss_{mode}.json")
    simulation = compiled["nodes"]["simulation"]
    config = simulation["backend"]["config"]
    assert compiled["factories"]["task"] == "pressb.tasks.pap_plus_plus.parallel:create_backend"
    assert Path(config["scene_config"]).name == "piper_pick_place_plus_plus.json"
    assert config["physics_backend"] == "gpu"
    task = PAPPlusPlusTask(config)
    assert task.metadata()["task_id"] == "pap_plus_plus"
    assert task.config["scene"]["box"] == _scene("_plus_plus")["box"]
    assert simulation["server"]["port"] == 8221
    assert compiled["nodes"]["trainer"]["mode"] == ("eval" if mode == "eval" else "train")
