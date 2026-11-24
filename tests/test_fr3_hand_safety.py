import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.robots.franka.gripper import GripperWorker
from pressb.robots.franka.safety import SafetyLimits, checked_state


class CommandError(RuntimeError):
    pass


class Hand:
    def __init__(self):
        self.width = 0.08
        self.grasped = False
        self.calls = []
        self.entered_grasp = threading.Event()
        self.release_grasp = threading.Event()
        self.stop_result = True
        self.read_delay_ns = 0

    def gripper_sample(self):
        timestamp = time.monotonic_ns() - self.read_delay_ns
        return {
            "width": self.width,
            "is_grasped": self.grasped,
            "max_width": 0.08,
            "device_time_ms": 0,
            "read_started_monotonic_ns": timestamp,
            "read_returned_monotonic_ns": timestamp,
        }

    def gripper_move(self, width, speed):
        self.calls.append(("move", width))
        self.width = max(width, 0.04)  # model a 40 mm object
        self.grasped = False
        return self.width == width

    def gripper_grasp(self, width, speed, force):
        self.calls.append(("grasp", width, force))
        self.entered_grasp.set()
        if not self.release_grasp.wait(2):
            raise RuntimeError("Test grasp did not receive stop")
        return False

    def gripper_stop(self):
        self.calls.append(("stop",))
        if self.entered_grasp.is_set():
            self.release_grasp.set()
        return self.stop_result

    def worker(self):
        return GripperWorker(self, 0.08, 0.1, self.width, self.grasped, command_error=CommandError)


def wait_until(condition, timeout=2):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            pytest.fail("Timed out waiting for Franka Hand worker")
        time.sleep(0.002)


def test_threshold_bypasses_deadband_then_open_preempts_grasp():
    hand = Hand()
    worker = hand.worker()
    try:
        worker.submit(0.94)
        wait_until(lambda: not worker.snapshot()["busy"])
        worker.submit(0.951)
        assert hand.entered_grasp.wait(1)
        assert worker.snapshot()["busy"]
        assert [c[0] for c in hand.calls] == ["move", "stop", "grasp"]
        assert hand.calls[-1] == ("grasp", 0.0, 20.0)

        worker.submit(0.0)
        wait_until(lambda: not worker.snapshot()["busy"])
        assert hand.calls[-2:] == [("stop",), ("move", 0.08)]
        assert worker.width() == pytest.approx(0.08)
    finally:
        hand.release_grasp.set()
        worker.close()


def test_held_object_is_preserved_on_connect_and_during_closure():
    hand = Hand()
    hand.width, hand.grasped = 0.04, True
    worker = hand.worker()
    try:
        worker.submit(1.0)
        wait_until(lambda: not worker.snapshot()["busy"])
        assert worker.width() == 0.04
        assert hand.calls == []
    finally:
        worker.close()


def test_failed_mode_switch_stop_blocks_the_grasp():
    hand = Hand()
    hand.stop_result = False
    worker = hand.worker()
    try:
        worker.submit(1)
        wait_until(lambda: worker._error is not None)
        with pytest.raises(RuntimeError, match="did not acknowledge position-mode stop"):
            worker.submit(0)
        assert hand.calls == [("stop",)]
    finally:
        hand.stop_result = True
        worker.close()


def test_native_read_return_time_determines_feedback_age():
    hand = Hand()
    hand.read_delay_ns = 1_000_000_000
    worker = hand.worker()
    try:
        wait_until(lambda: worker._sample_sequence > 0)
        with pytest.raises(TimeoutError, match="stale"):
            worker.snapshot()
    finally:
        worker.close()
    assert hand.calls == []


def test_live_urdf_and_native_workspace_limits_are_both_enforced():
    joints = "".join(
        f'<joint name="fr3_joint{i}" type="revolute"><limit lower="-2" upper="2"/></joint>'
        for i in range(1, 8)
    )
    urdf = f"<robot>{joints}</robot>"
    native = SimpleNamespace(
        lower=[-1] * 7,
        upper=[3] * 7,
        velocity=[2] * 7,
        xyz_lower=[-1] * 3,
        xyz_upper=[1] * 3,
    )
    model = SimpleNamespace(position=lambda q: q[:3])
    limits = SafetyLimits.from_urdf(urdf, model, native)
    np.testing.assert_array_equal(limits.check([0] * 7), [0] * 3)
    np.testing.assert_array_equal(limits.velocity, [1.5] * 7)
    for last in (-1.1, 2.1):
        with pytest.raises(ValueError, match="J7"):
            limits.check([0] * 6 + [last])
    with pytest.raises(ValueError, match="flange position"):
        limits.check([0.96] + [0] * 6)
    with pytest.raises(ValueError, match="seven Franka"):
        SafetyLimits.from_urdf(urdf.replace("fr3_joint2", "finger_joint"), model, native)


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"errors": "reflex"}, "reports errors"),
        ({"flange_timestamp_ns": 0}, "stale"),
        ({"flange_timestamp_ns": float("nan")}, "stale"),
        ({"dq": [0.051] * 7}, "already moving"),
    ],
)
def test_startup_rejects_fault_stale_or_moving_state(override, error):
    state = {
        "q": [0] * 7,
        "dq": [0] * 7,
        "flange_timestamp_ns": time.monotonic_ns(),
        "errors": "",
    }
    state.update(override)
    with pytest.raises(RuntimeError, match=error):
        checked_state(SimpleNamespace(state=lambda: state), stationary=True)
