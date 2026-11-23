"""Terminal feedback in the evaluator, using the existing real-node mailbox CLI.

Policy observations/actions keep using the normal node protocol. Operator
messages use SSH independently, so readiness can arrive while /reset blocks.
The input console never confirms readiness or generates policy actions itself.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
import select
import shlex
import subprocess
import sys
import threading
import time


RUN_OPTION = "operator_console_run_id"


class OperatorCommandError(RuntimeError):
    """The existing operator CLI rejected a stale or wrong-phase command."""


class SSHOperatorClient:
    def __init__(self, settings):
        allowed = {"ssh_host", "command", "timeout_s", "poll_interval_s"}
        if not isinstance(settings, dict) or set(settings) - allowed:
            raise ValueError("operator_console requires ssh_host and command")
        self.host = settings.get("ssh_host")
        self.command = settings.get("command")
        self.timeout = settings.get("timeout_s", 5.)
        self.poll_interval = settings.get("poll_interval_s", .5)
        if (not isinstance(self.host, str) or not self.host or self.host.startswith("-")
                or any(c.isspace() for c in self.host)):
            raise ValueError("operator_console.ssh_host must be one SSH host/alias")
        if (not isinstance(self.command, list) or not self.command
                or not all(isinstance(value, str) and value for value in self.command)):
            raise ValueError("operator_console.command must be a nonempty argv list")
        for name, value in (("timeout_s", self.timeout), ("poll_interval_s", self.poll_interval)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"operator_console.{name} must be finite and positive")

    def _call(self, arguments):
        command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", self.host,
                   shlex.join([*self.command, *arguments])]
        try:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                    text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("Operator SSH request timed out; command was not retried") from error
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()[-1500:]
            error = OperatorCommandError if result.returncode == 2 else RuntimeError
            raise error(f"Operator command failed ({result.returncode}): {detail}")
        try:
            value = json.loads(result.stdout)
        except ValueError as error:
            raise RuntimeError("Operator CLI returned invalid JSON") from error
        if not isinstance(value, dict):
            raise RuntimeError("Operator CLI must return a JSON object")
        return value

    def status(self):
        result = self._call(["status"])
        if result.get("schema_version") != 1 or not isinstance(result.get("phase"), str):
            raise RuntimeError("Unsupported real-robot operator status")
        return result

    def send(self, command, token):
        return self._call([command, "--episode", token])


class OperatorConsole:
    """Poll one real node and forward line commands bound to this evaluation."""

    def __init__(self, runner, client, *, input_stream=None, output_stream=None, require_tty=True):
        self.runner, self.client = runner, client
        self.input = sys.stdin if input_stream is None else input_stream
        self.output = sys.stderr if output_stream is None else output_stream
        if require_tty and not self.input.isatty():
            raise ValueError("Operator console needs an interactive terminal; disable it for unattended runs")
        self.fd = self.input.fileno()
        self.done = threading.Event()
        self.thread = None
        self.error = None
        self.quit_requested = False
        self.current = None
        self.last_prompt = None
        self.pending = None
        self.aborted_token = None
        self.episode_tokens = set()
        self.buffer = b""
        self.eof = False

    def _print(self, message):
        print(message, file=self.output, flush=True)

    def _owned(self, status):
        return (isinstance(status, dict)
                and status.get("options", {}).get(RUN_OPTION) == self.runner.run_id)

    def _log(self, command, status, accepted, detail=None):
        # The trainer owns creation of its run directory. Never make a fresh
        # output nonempty before GenericTrainer.run validates it.
        if not (self.runner.output / "config.json").exists():
            return
        row = {"time_unix_s": time.time(), "run_id": self.runner.run_id,
               "episode_token": status.get("episode_token"), "command": command,
               "accepted": accepted, "detail": detail}
        with (self.runner.output / "operator_events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    def refresh(self):
        status = self.client.status()
        self.current = deepcopy(status)
        if not self._owned(status):
            return status
        identity = (status.get("owner_id"), status.get("episode_token"), status["phase"])
        if identity != self.last_prompt:
            self.last_prompt = identity
            self.pending = None
            token, phase = status.get("episode_token"), status["phase"]
            if phase == "waiting_reset":
                self.episode_tokens.add(token)
                self._print(f"\n[第 {len(self.episode_tokens)} 局] 摆好物体并离开运动范围后，输入 r 回车开始；q 回车退出。")
            elif phase == "running":
                self._print("[执行中] s 回车：成功结束本局；f 回车：失败结束本局；q 回车：中止整轮评测。")
            elif phase == "returning":
                self._print("[回位中] 正在恢复机械臂和夹爪初始状态，请暂勿摆放物体；q 回车可中止。")
            elif phase == "aborted" and status.get("return_to_initial", {}).get("status") in ("failed", "aborted"):
                self._print("[回位未完成] 回位已停止；已完成的任务结果会保留。")
            elif phase == "ended":
                outcome = status.get("outcome", {})
                home = status.get("return_to_initial", {})
                if home.get("before_reset") and home.get("status") == "completed":
                    self._print("[初始位姿就绪] 等待本局准备确认。")
                    return status
                label = "成功" if outcome.get("success") else (
                    "失败" if outcome.get("terminated") else "已结束")
                returned = home.get("status") == "completed"
                self._print(f"[本局结束] {label}。" + ("机械臂已回到初始位姿。" if returned else ""))
        return status

    def _send(self, command, status):
        token = status.get("episode_token")
        try:
            result = self.client.send(command, token)
        except OperatorCommandError as error:
            self._log(command, status, False, str(error))
            self._print(f"[未接受] {error}")
            return False
        if result.get("accepted") is not True or result.get("episode_token") != token:
            raise RuntimeError("Operator acknowledgment does not match the selected episode")
        self._log(command, status, True)
        self.pending = (token, command)
        self._print({"ready": "[已确认] 开始本局。", "success": "[已标注成功] 正在结束本局。",
                     "failure": "[已标注失败] 正在结束本局。", "abort": "[已中止] 等待评测端退出。"}[command])
        return True

    def submit_line(self, line, status=None):
        text = line.strip().lower()
        status = self.current if status is None else status
        if text in ("q", "quit", "abort"):
            self.quit_requested = True
            self.runner.stop()
            self._cancel_active()
            return
        command = {"r": "ready", "ready": "ready", "s": "success", "success": "success",
                   "f": "failure", "failure": "failure"}.get(text)
        if command is None:
            if text:
                self._print("请输入 r / s / f / q，然后回车。")
            return
        if not self._owned(status):
            self._print("[等待] 尚未收到本轮评测的重置请求。")
            return
        expected = "waiting_reset" if command == "ready" else "running"
        if status["phase"] != expected:
            self._print(f"[未发送] 当前阶段为 {status['phase']}，不能输入 {command}。")
            return
        if self.pending is not None:
            self._print("[等待] 上一条输入正在处理，请等待下一条阶段提示。")
            return
        self._send(command, status)

    def _cancel_active(self):
        status = self.current
        if (self._owned(status) and status.get("phase") in ("waiting_reset", "running", "returning")
                and status.get("episode_token") != self.aborted_token):
            if self._send("abort", status):
                self.aborted_token = status["episode_token"]

    def _loop(self):
        try:
            while not self.done.is_set():
                status = self.refresh()
                if self.runner.stop_event.is_set():
                    self.quit_requested = True
                    self._cancel_active()
                if self.eof:
                    self.done.wait(self.client.poll_interval)
                    continue
                readable, _, _ = select.select([self.fd], [], [], self.client.poll_interval)
                if readable:
                    chunk = os.read(self.fd, 4096)
                    if not chunk:
                        self.eof = True
                        self.submit_line("q", status)
                        continue
                    self.buffer += chunk
                    while b"\n" in self.buffer:
                        raw, self.buffer = self.buffer.split(b"\n", 1)
                        # All lines read together retain the displayed token;
                        # pasted input cannot silently target a later episode.
                        self.submit_line(raw.decode("utf-8", errors="replace"), status)
        except Exception as error:
            self.error = error
            self.runner.stop()
            self._print(f"[人工反馈通道失败] {error}")

    def start(self):
        self._print("人工反馈已接入本评测终端：r 开始，s 成功，f 失败，q 退出；每次输入后回车。")
        self.thread = threading.Thread(target=self._loop, name="eval-operator-console", daemon=True)
        self.thread.start()

    def close(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=self.client.timeout + self.client.poll_interval + 1.)


def run_with_operator_console(runner, settings, *, client=None, input_stream=None,
                              output_stream=None, require_tty=True):
    """Run evaluation with input in the same terminal; preserve the RL core."""
    if runner.config.mode != "eval":
        raise ValueError("operator_console currently supports evaluation mode")
    if runner.config.human_feedback.get("mode", "environment") != "environment":
        raise ValueError("operator_console uses the real node's labels; human_feedback.mode must be environment")
    client = SSHOperatorClient(settings) if client is None else client
    console = OperatorConsole(runner, client, input_stream=input_stream,
                              output_stream=output_stream, require_tty=require_tty)
    metadata = runner.simulation.call("/health")
    if (metadata.get("environment_type") != "real_robot" or metadata.get("num_envs") != 1
            or metadata.get("reward_schema", {}).get("kind") != "manual_operator"
            or not metadata.get("reward_schema", {}).get("labels_enabled")):
        raise ValueError("Operator console requires one physical environment with manual labels enabled")
    if not metadata.get("execution", {}).get("actuation_enabled"):
        raise ValueError("Actuation is disabled on the real node; enable execution before evaluating")
    status = client.status()
    if status.get("phase") in ("waiting_reset", "running", "terminal", "returning"):
        raise ValueError("The real node already has an active episode; do not attach a second evaluator")
    if status.get("phase") not in ("idle", "ended"):
        raise ValueError("The real operator task is closed or aborted; restart the real node")
    # Bind remote mailbox tokens to this evaluator's own reset requests.
    runner.config.rollout.reset_options = {**runner.config.rollout.reset_options, RUN_OPTION: runner.run_id}
    console.start()
    try:
        try:
            result = runner.run()
        except Exception as error:
            if console.quit_requested and ("operator_abort" in str(error) or "Operator requested abort" in str(error)):
                runner._status_write("stopped", stop_reason="operator_abort")
                result = runner.health()
            else:
                raise
    finally:
        console.close()
    if console.error:
        raise RuntimeError("Operator console failed") from console.error
    final_operator = client.status()
    home_report = final_operator.get("return_to_initial", {})
    if home_report.get("status") == "failed":
        error = home_report.get("error", "Initial return failed")
        runner._status_write("failed", error_type="ReturnToInitialError", error=error, return_to_initial=home_report)
        raise RuntimeError(f"Return to initial pose failed after recording episode results: {error}")
    if home_report:
        state = "stopped" if home_report.get("status") == "aborted" else result["state"]
        runner._status_write(state, return_to_initial=home_report)
        result = runner.health()
    if result.get("episodes"):
        console._print(f"评测结束：成功 {result['successes']}/{result['episodes']} 局，"
                       f"成功率 {100 * result['successes'] / result['episodes']:.1f}%。")
    return result
