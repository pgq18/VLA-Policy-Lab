"""Exercise the production FR3 driver against in-process FCI and Hand doubles."""
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from pressb.robots.franka import real
from pressb.robots.franka.gripper import GripperWorker


URDF = Path(__file__).resolve().parents[1] / "assets/robots/fr3/source/controller_fr3.urdf"
POSE = np.array([.21, -.12, .25, -1.77, .068, 1.62, -.47])


class FakeHardware:
    def __init__(self):
        self.q, self.dq = POSE.copy(), np.zeros(7)
        self.width, self.maximum = .03, .075
        self.age = 0.
        self.errors = ""
        self.running = False
        self.started, self.targets = [], []
        self.stop_calls = 0

    def urdf(self):
        return URDF.read_text()

    def set_limits(self, lower, upper):
        self.lower, self.upper = np.array(lower), np.array(upper)

    def state(self):
        return {"q": self.q, "dq": self.dq, "errors": self.errors, "running": self.running,
                "flange_timestamp_ns": time.monotonic_ns() - int(self.age * 1e9)}

    def gripper_state(self):
        return self.width, self.maximum, False

    def start(self, q, duration):
        assert not self.running
        self.started.append((q, duration))
        self.running = True

    def set_target(self, q):
        assert self.running
        self.targets.append(q)

    def stop_control(self):
        self.stop_calls += 1
        self.running = False


class FakeHand:
    def __init__(self, hardware, maximum, speed, width, grasped, *, command_error):
        self.hardware = hardware
        self.maximum, self.speed, self.width = maximum, speed, width
        self.commands, self.closed, self.busy = [], False, False

    def snapshot(self):
        assert not self.closed
        return {"width": self.width, "max_width": self.hardware.maximum, "is_grasped": False, "busy": self.busy}

    def submit(self, closure):
        assert not self.closed
        self.commands.append(closure)

    def close(self):
        self.closed = True


@pytest.fixture
def station(monkeypatch):
    hardware = FakeHardware()
    made = []
    hands = []

    def connect(ip, realtime):
        made.append((ip, realtime))
        return hardware

    def hand(*args, **kwargs):
        worker = FakeHand(*args, **kwargs)
        hands.append(worker)
        return worker

    # The station checks the URDF/native intersection. Broader fake native
    # bounds ensure a mismatch in the offline advertised URDF fails this test.
    limits = SimpleNamespace(lower=[-10.] * 7, upper=[10.] * 7, velocity=[2.5] * 7,
                             xyz_lower=[-.85, -.85, -.05], xyz_upper=[.85, .85, .85])
    model = SimpleNamespace(position=lambda q: [q[0], 0., .4])
    native = SimpleNamespace(Hardware=connect, Limits=lambda: limits, Kinematics=lambda urdf: model,
                             CommandError=ValueError, libfranka_version="test", gripper_stop_concurrency_fixed=True)
    monkeypatch.setattr(real, "_load_native", lambda: native)
    monkeypatch.setattr(real, "GripperWorker", hand)
    return SimpleNamespace(hardware=hardware, hands=hands, made=made, native=native, limits=limits)


def test_offline_metadata_and_readonly_connect_never_submit_motion(station):
    driver = real.FrankaHardware()
    description = driver.describe()
    assert not description["actuation_enabled"]
    assert description["state"]["names"] == [f"joint{i}" for i in range(1, 8)] + ["gripper_closure"]
    assert description["action"]["units"] == ["rad"] * 7 + ["unitless"]
    assert description["action"]["control_hz"] == 25
    assert driver.status()["device"] == {} and not station.made

    driver.connect()
    driver.connect()
    np.testing.assert_allclose(driver.read_state(), [*POSE, .6])
    assert driver.describe() == description
    assert station.made == [("172.16.0.2", True)]
    assert station.hands[0].maximum == .075 and station.hands[0].speed == .1
    for fn in (lambda: driver.command([*POSE, .9]), driver.hold):
        with pytest.raises(RuntimeError, match="disabled"):
            fn()
    assert driver.last_command is None
    assert not station.hardware.started and not station.hands[0].commands
    driver.close()
    driver.close()
    assert station.hands[0].closed and station.hardware.stop_calls == 1
    with pytest.raises(RuntimeError, match="stopped or closed"):
        driver.connect()


def test_readonly_real_worker_releases_reader_without_hand_command(station, monkeypatch):
    # The double implements only SDK feedback, with no move/grasp/stop methods.
    monkeypatch.setattr(real, "GripperWorker", GripperWorker)
    station.hardware.maximum = .08003654
    station.hardware.width = .07989947
    driver = real.FrankaHardware()
    try:
        driver.connect()
        state = driver.read_state()
        assert state[7] == pytest.approx(1. - station.hardware.width / .08)
        assert driver.status()["device"]["reported_gripper_max_width_m"] == .08003654
        assert driver.gripper_command_idle()
    finally:
        driver.close()
    assert not station.hardware.started


def test_initial_hold_and_arm_return_leave_existing_hand_untouched(station, monkeypatch):
    # No SDK move, grasp or stop exists on the double; a previously held
    # object must retain the Hand's own control mode across the whole return.
    monkeypatch.setattr(real, "GripperWorker", GripperWorker)
    monkeypatch.setattr(station.hardware, "gripper_state", lambda: (.001, .08, True))
    driver = real.FrankaHardware({"actuation_enabled": True})
    try:
        driver.connect()
        initial = driver.hold()
        assert initial[-1] == pytest.approx(.9875)
        applied = driver.command_arm([*POSE, 0.])
        assert applied[-1] == initial[-1]
        assert driver.gripper_command_idle()
        assert driver.status()["last_feedback"]["gripper"]["last_command"] is None
    finally:
        driver.close()
    assert len(station.hardware.targets) == 2


@pytest.mark.parametrize("operation", ["command", "hold", "command_arm"])
def test_invalid_aperture_is_reported_readonly_and_rejected_before_actuation(operation, station):
    station.hardware.width, station.hardware.maximum = .16054490, .16054443
    driver = real.FrankaHardware({"actuation_enabled": True})
    driver.connect()
    assert driver.read_state()[-1] == 0.
    assert driver.status()["last_feedback"]["gripper"]["width"] == .16054490
    with pytest.raises(RuntimeError, match="physical travel"):
        if operation == "hold":
            driver.hold()
        else:
            getattr(driver, operation)([*POSE, 0.])
    assert not station.hardware.started and not station.hands[0].commands


@pytest.mark.parametrize("config", [
    {"actuation_enabled": 1}, {"enforce_realtime": "true"}, {"robot_ip": ""},
    {"max_joint_step": .2}, {"control_hz": float("nan")}, {"alignment_tolerance": .4},
    {"gripper_speed": 0}, {"home": True},
])
def test_invalid_config_fails_before_connect(config, station):
    with pytest.raises((ValueError, TypeError)):
        real.FrankaHardware(config)
    assert not station.made


def test_private_sdk_limits_and_feedback_checked_before_motion(station):
    driver = real.FrankaHardware({"actuation_enabled": True})
    station.native.gripper_stop_concurrency_fixed = False
    with pytest.raises(RuntimeError, match="stop fix"):
        driver.connect()
    assert not station.made


@pytest.mark.parametrize("condition", ["moving", "live_limits", "uncalibrated", "nan_width"])
def test_failed_preflight_releases_hardware(condition, station):
    driver = real.FrankaHardware({"actuation_enabled": True})
    if condition == "moving":
        station.hardware.dq[3] = .2
    elif condition == "live_limits":
        station.limits.lower[0] = -1.
    elif condition == "uncalibrated":
        station.hardware.maximum = 0
    else:
        station.hardware.width = float("nan")
    with pytest.raises(RuntimeError):
        driver.connect()
    assert station.hardware.stop_calls == 1 and not station.hardware.started
    assert not station.hands and driver.status()["stopped"]


def test_projection_measured_hold_and_applied_history(station):
    driver = real.FrankaHardware({"actuation_enabled": True})
    driver.connect()
    first = np.r_[POSE, .97]
    first[6] += .13
    expected = first.copy()
    expected[6] = POSE[6] + .05
    sent = driver.command(first)
    np.testing.assert_allclose(sent, expected)
    np.testing.assert_allclose(station.hardware.targets[-1], sent[:7])
    assert np.isclose(first[6], POSE[6] + .13)
    assert station.hardware.started == [(POSE.tolist(), 0.)]
    assert station.hands[0].commands == [.97]

    # A fresh measured position is the origin for each control tick, even if
    # the last accepted target has not been reached. Hand closure is unfiltered.
    station.hardware.q[6] += .02
    second = np.r_[POSE, 1.]
    second[6] += .3
    applied = driver.command(second)
    assert applied[6] == pytest.approx(POSE[6] + .07) and applied[7] == 1.
    assert len(station.hardware.started) == 1
    np.testing.assert_allclose(driver.hold(), [*station.hardware.q, 1.])
    assert station.hands[0].commands[-1] == 1. and driver.read_state()[7] == pytest.approx(.6)
    driver.last_command[:] = 0.
    assert driver.status()["last_applied_action"] == [*station.hardware.q, 1.]
    assert driver.gripper_command_idle()
    station.hands[0].busy = True
    assert not driver.gripper_command_idle()
    driver.stop()
    assert not station.hardware.running and station.hands[0].closed


def test_project_wire_rounded_joint_limit_within_true_bound(station):
    driver = real.FrankaHardware({"actuation_enabled": True})
    driver.connect()
    driver.hold()
    arm_upper = driver.describe()["action"]["upper"]
    command = np.array([*POSE, 0.], dtype=np.float32)
    command[6] = arm_upper[6]
    assert float(command[6]) > arm_upper[6]
    applied = driver.command(command)
    assert applied[6] == pytest.approx(POSE[6] + .05)
    driver.close()


@pytest.mark.parametrize("target, message", [
    ([*POSE[:6], POSE[6] + .3, .4], "startup alignment"),
    ([.9, *POSE[1:], .4], "workspace"),
    ([4., *POSE[1:], .4], "requested FR3 target"),
    ([*POSE, 1.1], "gripper_closure"),
    ([*POSE[:6], float("nan"), .4], "finite"),
])
def test_bad_first_action_does_not_start_arm_or_hand(target, message, station):
    driver = real.FrankaHardware({"actuation_enabled": True})
    driver.connect()
    with pytest.raises(ValueError, match=message):
        driver.command(target)
    assert station.hardware.started == [] and station.hands[0].commands == []


@pytest.mark.parametrize("failure", ["stale", "hardware", "stopped", "hand"])
def test_feedback_fault_stops_arm_and_hand_without_restarting(failure, station):
    driver = real.FrankaHardware({"actuation_enabled": True})
    driver.connect()
    driver.hold()
    if failure == "stale":
        station.hardware.age = 1.
    elif failure == "hardware":
        station.hardware.errors = "reflex"
    elif failure == "stopped":
        station.hardware.running = False
    else:
        station.hands[0].width = float("nan")
    with pytest.raises(RuntimeError):
        driver.read_state()
    assert driver.status()["fault"] and not driver.status()["connected"]
    assert not station.hardware.running and station.hands[0].closed
    with pytest.raises(RuntimeError, match="stopped or closed"):
        driver.hold()
    assert len(station.hardware.started) == 1


def test_command_failure_after_arm_submit_latches_and_cleans_up(station, monkeypatch):
    driver = real.FrankaHardware({"actuation_enabled": True})
    driver.connect()

    def fail(closure):
        raise RuntimeError("worker fault")

    monkeypatch.setattr(station.hands[0], "submit", fail)
    with pytest.raises(RuntimeError, match="worker fault"):
        driver.command(np.r_[POSE, .8])
    assert station.hardware.targets and not station.hardware.running
    assert driver.status()["submitted_commands"] == 0 and driver.last_command is None
    assert station.hands[0].closed


def test_watchdog_during_start_prevents_joint_and_hand_submission(station, monkeypatch):
    driver = real.FrankaHardware({"actuation_enabled": True})
    driver.connect()
    starting, resume = threading.Event(), threading.Event()
    original = station.hardware.start
    errors = []

    def slow_start(q, duration):
        original(q, duration)
        starting.set()
        assert resume.wait(2.)

    def command():
        try:
            driver.command([*POSE, 1.])
        except RuntimeError as error:
            errors.append(error)

    monkeypatch.setattr(station.hardware, "start", slow_start)
    active = threading.Thread(target=command)
    active.start()
    assert starting.wait(2.)
    watchdog = threading.Thread(target=driver.stop)
    watchdog.start()
    try:
        assert driver._stop_requested.wait(2.)
    finally:
        resume.set()
        active.join(timeout=2.)
        watchdog.join(timeout=2.)
    assert not active.is_alive() and not watchdog.is_alive()
    assert errors and not station.hardware.targets and not station.hands[0].commands
    assert not station.hardware.running and station.hardware.stop_calls == 1
