"""Exercise FR3 returns with an in-memory arm and the real operator mailbox."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event
import time

import numpy as np
import pytest

from pressb.robots.franka.environment import FrankaReturnBackend
from pressb.runtime.real_robot_operator import ManualTask, send_command
from pressb.runtime.simulation import SimulationService
from test_real_robot_backend import FakeCameras, FakeClock


class Limits:
    velocity = np.ones(7)

    def check(self, q, name):
        assert np.all(np.abs(q) < 2)
        return np.asarray(q[:3])


class Robot:
    limits = Limits()

    def __init__(self, enabled):
        self.enabled = enabled
        self.state = np.array([.03] * 7 + [.72])
        self.last_command = None
        self.sent = []
        self.hand_commands = []
        self.connected = self.stopped = False

    def describe(self):
        state = {"names": [f"joint{i}" for i in range(1, 8)] + ["gripper_closure"],
                 "units": ["rad"] * 7 + ["unitless"]}
        return dict(robot_type="synthetic_fr3", actuation_enabled=self.enabled, state=state,
                    action={**state, "representation": "joint_position", "frame": "robot_base",
                            "control_hz": 25., "lower": [-2.] * 7 + [0.], "upper": [2.] * 7 + [1.]})

    def connect(self):
        self.connected = True

    def read_state(self):
        assert self.connected
        return self.state.copy()

    def _apply(self, target):
        assert self.enabled and self.connected and not self.stopped
        self.last_command = np.array(target, copy=True)
        self.state = self.last_command.copy()
        self.sent.append(self.state.copy())
        return self.last_command.copy()

    def command(self, target):
        applied = self._apply(target)
        self.hand_commands.append((len(self.sent), applied[-1]))
        return applied

    def command_arm(self, target):
        held = self.state if self.last_command is None else self.last_command
        return self._apply(np.r_[target[:-1], held[-1]])

    def hold(self):
        return self.command_arm(self.state)

    def gripper_command_idle(self):
        return True

    def stop(self):
        self.stopped = True

    def close(self):
        self.connected = False


@contextmanager
def environment(tmp_path, *, enabled=True, return_enabled=True, watchdog=False, gripper_closure=.1):
    clock = FakeClock()
    task = ManualTask(dict(directory=str(tmp_path), reset_timeout_s=2., poll_interval_s=.005))
    robot = Robot(enabled)
    config = dict(instruction="Synthetic test", control_hz=25, horizon=2, max_episode_steps=1)
    if return_enabled:
        config["return_to_initial"] = dict(enabled=True, joints_rad=[0.] * 7, gripper_closure=gripper_closure,
                                           min_duration_s=.05, gripper_stable_s=.01)
    backend = FrankaReturnBackend(config, robot=robot, cameras=FakeCameras(), task=task,
                                  clock=clock, sleep=clock.sleep, _start_watchdog=watchdog)
    service = SimulationService(backend)
    try:
        yield service, robot, task
    finally:
        service.close()


def ready(service, task, request_id="reset"):
    payload = dict(run_id="test", request_id=request_id, env_ids=[0], seed=123)
    with ThreadPoolExecutor(1) as pool:
        result = pool.submit(service.reset, payload)
        deadline = time.monotonic() + 2
        while not result.done() and time.monotonic() < deadline:
            status = task.status()
            if status["phase"] == "waiting_reset":
                send_command(task.directory, "ready", status["episode_token"])
                break
            time.sleep(.001)
        return result.result(timeout=2)["items"][0]


def terminal(service, row):
    return service.step(dict(run_id="test", request_id=row["episode_id"], items=[
        dict(env_id=0, episode_id=row["episode_id"], step_id=row["step_id"],
             actions=[[.04] * 7 + [.8]])]))["items"][0]


def test_return_before_ready_then_queued_for_next_reset_and_final_close(tmp_path):
    with environment(tmp_path) as (service, robot, task):
        row = ready(service, task)
        report = row["info"]["return_to_initial"]
        assert report["status"] == "completed" and report["plan"]["path_samples"] >= 101
        assert report["gripper_reset"]["status"] == "completed"
        assert all(q[-1] == .72 for q in robot.sent[:report["control_steps"] - 1])
        assert robot.hand_commands == [(report["control_steps"], .1)]
        assert robot.sent[report["control_steps"] - 1] == pytest.approx([0.] * 7 + [.1])
        assert row["observation"]["state"] == pytest.approx([0.] * 7 + [.1])

        for episode in range(2):
            result = terminal(service, row)
            assert result["truncated"] and result["info"]["hardware_control_steps"] == 1
            assert result["discount"] == pytest.approx(.995)
            assert result["info"]["return_to_initial"]["status"] == "pending"
            assert task.status()["phase"] == "returning"
            assert result["observation"]["state"] == pytest.approx([.04] * 7 + [.8])
            assert robot.state == pytest.approx(result["observation"]["state"])
            if episode == 0:
                row = ready(service, task, request_id="next")
                assert row["info"]["return_to_initial"]["status"] == "completed"
            else:
                service.close_run({"run_id": "test"})
            assert robot.state == pytest.approx([0.] * 7 + [.1])


def test_arm_return_without_gripper_reset_never_submits_a_hand_goal(tmp_path):
    with environment(tmp_path, gripper_closure=None) as (service, robot, task):
        # A nearly closed measured Hand could start a force grasp if replayed as a target.
        robot.state[-1] = .98
        row = ready(service, task)
        assert row["info"]["return_to_initial"]["control_steps"] > 0
        assert robot.state == pytest.approx([0.] * 7 + [.98])
        assert robot.hand_commands == []


def test_abort_pending_return_keeps_terminal_result_and_stops_before_recovery(tmp_path):
    with environment(tmp_path) as (service, robot, task):
        result = terminal(service, ready(service, task))
        sent = len(robot.sent)
        send_command(task.directory, "abort", task.status()["episode_token"])
        with pytest.raises(RuntimeError, match="abort"):
            service.close_run({"run_id": "test"})
        assert result["truncated"]
        assert len(robot.sent) == sent and robot.stopped
        assert task.status()["phase"] == "aborted"
        assert task.status()["return_to_initial"]["status"] == "aborted"


def test_abort_during_arm_return_keeps_gripper_and_latches_fault(tmp_path, monkeypatch):
    with environment(tmp_path) as (service, robot, task):
        command = robot.command_arm

        def abort_after_three_ticks(target):
            applied = command(target)
            if len(robot.sent) == 3:
                status = task.status()
                assert status["phase"] == "returning"
                send_command(task.directory, "abort", status["episode_token"])
            return applied

        monkeypatch.setattr(robot, "command_arm", abort_after_three_ticks)
        with pytest.raises(RuntimeError, match="abort"):
            service.reset(dict(run_id="test", request_id="reset", env_ids=[0], seed=0))
        assert robot.stopped and len(robot.sent) == 3
        assert all(command[-1] == .72 for command in robot.sent)
        assert not robot.hand_commands
        assert task.status()["return_to_initial"]["status"] == "aborted"
        with pytest.raises(RuntimeError, match="restart"):
            service.reset(dict(run_id="test", request_id="retry", env_ids=[0], seed=0))
        assert len(robot.sent) == 3


def test_disabled_motion_and_shutdown(tmp_path):
    with environment(tmp_path / "passive", enabled=False) as (service, robot, task):
        with pytest.raises((RuntimeError, ValueError), match="[Aa]ctuation"):
            service.reset(dict(run_id="test", request_id="reset", env_ids=[0], seed=0))
        assert not robot.connected and not robot.sent
        assert task.status()["phase"] == "idle"
    with environment(tmp_path / "manual", return_enabled=False) as (service, robot, task):
        row = ready(service, task)
        assert "return_to_initial" not in row["info"]
        assert robot.state == pytest.approx([.03] * 7 + [.72])
    with environment(tmp_path / "shutdown") as (service, robot, task):
        terminal(service, ready(service, task))
        sent = len(robot.sent)
    assert len(robot.sent) == sent


def test_operator_can_abort_while_return_command_is_blocked(tmp_path, monkeypatch):
    with environment(tmp_path, watchdog=True) as (service, robot, task):
        entered, stopped = Event(), Event()

        def blocked_command(target):
            entered.set()
            if not stopped.wait(2):
                pytest.fail("Operator abort did not reach the blocked return")
            raise RuntimeError("Synthetic controller stopped")

        monkeypatch.setattr(robot, "command_arm", blocked_command)
        monkeypatch.setattr(robot, "stop", stopped.set)
        with ThreadPoolExecutor(1) as pool:
            pending = pool.submit(service.reset, dict(run_id="test", request_id="reset", env_ids=[0], seed=0))
            assert entered.wait(2)
            status = task.status()
            assert status["phase"] == "returning"
            send_command(task.directory, "abort", status["episode_token"])
            with pytest.raises(RuntimeError, match="controller stopped"):
                pending.result(timeout=2)
        assert stopped.is_set() and not robot.sent
        assert task.status()["return_to_initial"]["status"] == "aborted"
