"""Evaluator terminal feedback through real mailbox/episode code, fake hardware."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from pressb.runtime.inference import InferenceService
from pressb.runtime.operator_console import (RUN_OPTION, OperatorCommandError, OperatorConsole,
                                             SSHOperatorClient, run_with_operator_console)
from pressb.runtime.real_robot import RealRobotBackend
from pressb.runtime.real_robot_operator import ManualTask, read_status, send_command
from pressb.runtime.simulation import SimulationService
from pressb.runtime.trainer import GenericTrainer
from test_real_robot_backend import FakeRobot, FakeCameras, FakeInferenceBackend


class MailboxClient:
    timeout = 1.
    poll_interval = .01

    def __init__(self, directory):
        self.directory = directory
        self.commands = []

    def status(self):
        return read_status(self.directory)

    def send(self, command, token):
        self.commands.append((command, token))
        return send_command(self.directory, command, token)


class ServiceClient:
    def __init__(self, service):
        self.service = service

    def call(self, path, payload=None):
        return self.service.handlers[path](payload or {})

    def close(self):
        pass


@contextmanager
def evaluation(tmp_path, episodes=2):
    task = ManualTask({"directory": str(tmp_path / "operator"), "reset_timeout_s": 3,
                       "poll_interval_s": .005})
    robot = FakeRobot()
    backend = RealRobotBackend({"instruction": "Synthetic test task", "task_id": "console-test",
        "horizon": 2, "control_hz": 30., "max_episode_steps": 1000, "watchdog_seconds": 10.},
        robot=robot, cameras=FakeCameras(), task=task, _start_watchdog=False)
    simulation = SimulationService(backend)
    inference = InferenceService(FakeInferenceBackend(backend), cache_size=4)
    runner = GenericTrainer({"method": "base", "mode": "eval", "device": "cpu",
        "output_dir": str(tmp_path / "run"), "rollout": {"eval_episodes": episodes, "execution_horizon": 2,
        "num_envs": 1, "action_bounds": "clip"}},
        inference_client=ServiceClient(inference), simulation_client=ServiceClient(simulation))
    reader, writer = os.pipe()
    source = os.fdopen(reader, "r")
    client = MailboxClient(task.directory)
    try:
        yield runner, client, source, writer, robot, task
    finally:
        source.close()
        try:
            os.close(writer)
        except OSError:
            pass  # The EOF test deliberately closes the writer first.
        backend.close()
        inference.close()


def wait_for(predicate, timeout=5.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.005)
    raise AssertionError("Timed out waiting for the test episode")


def test_two_episodes_same_terminal_success_failure_and_ready(tmp_path):
    with evaluation(tmp_path) as (runner, client, source, writer, robot, task), ThreadPoolExecutor(1) as executor:
        output = io.StringIO()
        future = executor.submit(run_with_operator_console, runner, {}, client=client,
                                 input_stream=source, output_stream=output, require_tty=False)
        previous = None
        for index, label in enumerate((b"s\n", b"f\n"), 1):
            status = wait_for(lambda: (s if (s := task.status())["phase"] == "waiting_reset"
                                      and s["episode_token"] != previous else None))
            previous = status["episode_token"]
            assert status["options"][RUN_OPTION] == runner.run_id
            assert not robot.sent or runner.episodes == 1
            wait_for(lambda: output.getvalue().count("[第 ") >= index)
            os.write(writer, b"r\n")
            wait_for(lambda: task.status()["phase"] == "running")
            wait_for(lambda: output.getvalue().count("[执行中]") >= index)
            os.write(writer, label)
        result = future.result(timeout=5)
        assert result["state"] == "completed"
        assert result["episodes"] == 2 and result["successes"] == 1
        assert result["success_rate"] == .5
        assert [c[0] for c in client.commands] == ["ready", "success", "ready", "failure"]
        events = [json.loads(line) for line in (runner.output / "operator_events.jsonl").read_text().splitlines()]
        assert [e["command"] for e in events] == ["ready", "success", "ready", "failure"]
        assert all(e["accepted"] for e in events)
        assert "episode_token" not in output.getvalue()


@pytest.mark.parametrize("quit_input", [b"q\n", b""])
def test_quit_or_eof_unblocks_pending_manual_reset_without_counting_failure(tmp_path, quit_input):
    with evaluation(tmp_path) as (runner, client, source, writer, robot, task), ThreadPoolExecutor(1) as executor:
        future = executor.submit(run_with_operator_console, runner, {}, client=client,
                                 input_stream=source, output_stream=io.StringIO(), require_tty=False)
        wait_for(lambda: task.status()["phase"] == "waiting_reset")
        if quit_input:
            os.write(writer, quit_input)
        else:
            os.close(writer)
        result = future.result(timeout=5)
        assert result["state"] == "stopped" and result["episodes"] == 0
        assert [c[0] for c in client.commands] == ["abort"]
        assert not robot.sent and robot.stop_event.is_set()


def test_stale_foreign_and_duplicate_inputs_do_not_change_another_episode(tmp_path):
    reader, writer = os.pipe()
    source = os.fdopen(reader, "r")
    runner = SimpleNamespace(run_id="our-run", output=tmp_path, stop_event=threading.Event(), stop=lambda: None)
    client = SimpleNamespace(timeout=1., poll_interval=.01, sent=[], status=lambda: None)
    client.send = lambda command, token: (client.sent.append((command, token)) or
                                         {"accepted": True, "episode_token": token})
    console = OperatorConsole(runner, client, input_stream=source, output_stream=io.StringIO(), require_tty=False)
    foreign = {"phase": "running", "episode_token": "foreign", "options": {RUN_OPTION: "other"}}
    own = {"phase": "waiting_reset", "episode_token": "ours", "options": {RUN_OPTION: "our-run"}}
    try:
        console.submit_line("s", foreign)
        console.submit_line("s", own)
        assert not client.sent
        console.submit_line("r", own)
        console.submit_line("r", own)
        assert client.sent == [("ready", "ours")]
    finally:
        source.close()
        os.close(writer)


def test_ssh_argv_is_quoted_and_label_failure_not_retried(monkeypatch):
    commands = []
    settings = {"ssh_host": "franka", "command": ["/home/pnp/Workspace/a b/operator.sh"]}
    client = SSHOperatorClient(settings)

    def run(argv, **kwargs):
        commands.append(argv)
        assert kwargs["stdin"] == subprocess.DEVNULL
        return SimpleNamespace(returncode=2, stderr="stale token", stdout="")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(OperatorCommandError):
        client.send("success", "token; echo should-not-run")
    assert len(commands) == 1
    assert shlex.split(commands[0][-1]) == [settings["command"][0], "success", "--episode", "token; echo should-not-run"]


def test_active_other_evaluator_rejected_before_reset(tmp_path):
    with evaluation(tmp_path) as (runner, client, source, _, robot, task):
        client.status = lambda: {"phase": "running"}
        with pytest.raises(ValueError, match="active episode"):
            run_with_operator_console(runner, {}, client=client, input_stream=source,
                                      output_stream=io.StringIO(), require_tty=False)
        assert not robot.connected and not robot.sent


def test_cli_pops_console_config_and_no_console_override(tmp_path, monkeypatch):
    import pressb.runtime.trainer as trainer_module
    import pressb.runtime.operator_console as console_module

    calls = []

    class Runner:
        def __init__(self, config):
            self.config = config
        def stop(self):
            pass
        def run(self):
            calls.append("plain")
            return {}

    monkeypatch.setattr(trainer_module, "GenericTrainer", Runner)
    monkeypatch.setattr("signal.signal", lambda *_: None)
    monkeypatch.setattr(console_module, "run_with_operator_console", lambda runner, config: calls.append(config) or {})
    settings = {"ssh_host": "franka", "command": ["/operator.sh"]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"method": "base", "mode": "eval", "operator_console": settings}))
    spec = importlib.util.spec_from_file_location("console_cli", Path(__file__).resolve().parents[1] / "scripts/run_rl_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for flags in ([], ["--no-operator-console"]):
        monkeypatch.setattr(sys, "argv", ["run_rl_training.py", "--config", str(path), *flags])
        module.main()
    assert calls == [settings, "plain"]
