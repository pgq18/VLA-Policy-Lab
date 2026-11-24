"""Asynchronous Franka Hand commands and feedback for PressB.

Adapted from the Franka station follower.
"""

import logging
import threading
import time
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
PREPARATION_STOP_TIMEOUT_S = 1.0


class GripperWorker:
    def __init__(
        self,
        hardware: Any,
        max_width: float,
        speed: float,
        initial_width: float,
        initial_grasped: bool,
        *,
        command_error: type[Exception],
    ):
        if not np.isfinite([max_width, speed, initial_width]).all() or max_width <= 0 or speed <= 0:
            raise ValueError(
                "Franka Hand requires finite feedback, a positive maximum width and positive speed"
            )
        self.hardware = hardware
        self.max_width = max_width
        self.speed = speed
        self._command_error = command_error
        self._condition = threading.Condition()
        self._pending: float | None = None
        self._pending_submitted_ns: int | None = None
        self._last: float | None = None
        self._error: Exception | None = None
        self._executing = False
        self._closed = False
        self._thread: threading.Thread | None = None
        self._state_stop = threading.Event()
        self._width = initial_width
        self._is_grasped = initial_grasped
        self._grasp_requested = initial_grasped
        # A completed position move may leave the device's position controller
        # active. Clear it once before switching to force grasp; a successful
        # preemption stop already clears it, so must not be repeated.
        # No command history is available on connect, so initially clear once
        # unless feedback already reports a held object (preserved below).
        self._position_mode_active: bool | None = None
        self._state_timestamp = time.monotonic()
        self._sample_metadata: dict[str, Any] = {
            "max_width": max_width,
            "device_message_id": None,
            "device_time_ms": None,
            "read_started_monotonic_ns": None,
            "read_returned_monotonic_ns": None,
        }
        self._sample_sequence = 0
        self._command_sequence = 0
        self._command_metadata: dict[str, Any] | None = None
        self._completed_commands: deque[dict[str, Any]] = deque(maxlen=8)
        self._stop_lock = threading.Lock()
        self._state_thread = threading.Thread(target=self._poll_state, name="franka-hand-state", daemon=True)
        self._state_thread.start()

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Franka Hand I/O failed: {self._error}") from self._error

    def width(self) -> float:
        with self._condition:
            self._feedback_age()
            return self._width

    def _feedback_age(self) -> float:
        self.check()
        age = time.monotonic() - self._state_timestamp
        if not -0.1 <= age <= 0.5:
            raise TimeoutError("Franka Hand state is stale")
        return age

    def snapshot(self) -> dict[str, Any]:
        """Return the current received sample and asynchronous command status."""
        with self._condition:
            age = self._feedback_age()
            return {
                **self._sample_metadata,
                "width": self._width,
                "is_grasped": self._is_grasped,
                "host_receive_age_s": age,
                "sample_sequence": self._sample_sequence,
                "pending_closure": self._pending,
                "pending_submitted_monotonic_ns": self._pending_submitted_ns,
                "last_command": None if self._command_metadata is None else dict(self._command_metadata),
                "completed_commands": [dict(command) for command in self._completed_commands],
                "busy": self._executing or self._pending is not None,
            }

    def _poll_state(self) -> None:
        try:
            while not self._state_stop.is_set():
                read_sample = getattr(self.hardware, "gripper_sample", None)
                if callable(read_sample):
                    sample = read_sample()
                    width, is_grasped = sample["width"], sample["is_grasped"]
                    metadata = {
                        key: sample[key]
                        for key in (
                            "max_width",
                            "device_time_ms",
                            "read_started_monotonic_ns",
                            "read_returned_monotonic_ns",
                        )
                    }
                    metadata["device_message_id"] = sample.get("device_message_id", sample["device_time_ms"])
                else:
                    # Compatibility for existing fake/older backends. These
                    # host call bounds cannot supply a device sampling time.
                    started = time.monotonic_ns()
                    width, maximum, is_grasped = self.hardware.gripper_state()
                    returned = time.monotonic_ns()
                    metadata = {
                        "max_width": maximum,
                        "device_message_id": None,
                        "device_time_ms": None,
                        "read_started_monotonic_ns": started,
                        "read_returned_monotonic_ns": returned,
                    }
                if not np.isfinite(width):
                    raise RuntimeError("Non-finite Franka Hand width")
                with self._condition:
                    self._width = float(width)
                    self._is_grasped = bool(is_grasped)
                    self._sample_metadata = metadata
                    self._sample_sequence += 1
                    # Native read-return precedes reacquiring the GIL. Do not
                    # relabel a delayed sample as fresh when Python publishes it.
                    self._state_timestamp = metadata["read_returned_monotonic_ns"] / 1e9
                    self._condition.notify_all()
                self._state_stop.wait(0.005)
        except Exception as exc:  # noqa: BLE001 - propagate SDK reader failures to the owning session
            if not self._state_stop.is_set():
                with self._condition:
                    if self._error is None:
                        self._error = exc
                    self._condition.notify_all()

    def submit(self, closure: float) -> None:
        if not np.isfinite(closure) or not 0 <= closure <= 1:
            raise ValueError("Franka Hand closure must be finite and in [0, 1]")
        with self._condition:
            self.check()
            if self._closed:
                raise RuntimeError("Franka Hand worker is closed")
            # Grasp and re-arm transitions must bypass the position deadband.
            mode_changed = self._last is not None and (
                self._last < 0.95 <= closure or closure < 0.90 <= self._last
            )
            if self._last is None or mode_changed or abs(closure - self._last) >= 0.02:
                self._pending = self._last = closure
                self._pending_submitted_ns = time.monotonic_ns()
                if self._thread is None:
                    self._thread = threading.Thread(target=self._run, name="franka-hand", daemon=True)
                    self._thread.start()
                self._condition.notify_all()

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._pending is not None)
                    if self._closed:
                        return
                    self._executing = True
                    closure, self._pending = self._pending, None
                    submitted_ns, self._pending_submitted_ns = (
                        self._pending_submitted_ns,
                        None,
                    )
                assert closure is not None
                self._execute(self.max_width * (1 - closure), submitted_ns=submitted_ns)
                with self._condition:
                    self._executing = False
                    self._condition.notify_all()
        except Exception as exc:  # noqa: BLE001 - propagate SDK worker failures to the owning session
            with self._condition:
                if not self._closed:
                    if self._error is None:
                        self._error = exc
                    self._condition.notify_all()

    def _execute(self, target_width: float, *, submitted_ns: int | None = None) -> None:
        with self._condition:
            width = self.width()
            is_grasped = self._is_grasped
        # Preserve proportional width tracking. grasp() closes until contact;
        # its width argument describes the expected object, not a position stop.
        # Match frankateleop: grasp only at >=95% closure. Re-arm below 90% so
        # encoder noise near the threshold cannot repeatedly restart a grasp.
        if target_width > self.max_width * 0.10:
            self._grasp_requested = False
        elif self._grasp_requested:
            return
        # Keep holding while the requested gap is inside the detected object.
        if is_grasped and target_width <= width + 0.001:
            return
        if abs(target_width - width) <= 0.001:
            return
        grasp = target_width <= self.max_width * 0.05 + 1e-12
        command = "grasp" if grasp else "move"
        rejection = None
        success = False
        fatal: Exception | None = None
        with self._condition:
            self._command_sequence += 1
            command_metadata: dict[str, Any] = {
                "sequence": self._command_sequence,
                "type": command,
                "target_width_m": 0.0 if grasp else target_width,
                "speed_m_s": self.speed,
                "initial_width_m": width,
                "submitted_monotonic_ns": submitted_ns,
                "started_monotonic_ns": None,
                "returned_monotonic_ns": None,
                "status": "preparing",
                "success": None,
                "error": None,
                "cancel_reason": None,
                "stop_started_monotonic_ns": None,
                "stop_returned_monotonic_ns": None,
                "preparation_type": None,
                "preparation_reason": None,
                "preparation_started_monotonic_ns": None,
                "preparation_returned_monotonic_ns": None,
                "preparation_success": None,
            }
            self._command_metadata = command_metadata
        done = threading.Event()
        # libfranka calls block until completion. The monitor only issues stop;
        # it never starts a second move/grasp. Join it before the next command,
        # so a late stop cannot cancel a newer command. Requires the private
        # SDK response-lock fix verified by the native backend at connection.
        monitor: threading.Thread | None = None
        try:
            prepared = not grasp or self._prepare_grasp(command_metadata)
            with self._condition:
                closed = self._closed
                if closed:
                    command_metadata["cancel_reason"] = "shutdown_before_dispatch"
                self.check()
                if grasp and self._pending is not None and self._pending < 0.95:
                    prepared = False
                    command_metadata["cancel_reason"] = "leader_changed_before_grasp"
                if prepared and not closed:
                    command_metadata["started_monotonic_ns"] = time.monotonic_ns()
                    command_metadata["status"] = "running"
            if not prepared or closed:
                success = False
            else:
                # Only motion execution counts toward the travel deadline.
                # No monitor races a preparation stop or a never-sent grasp.
                monitor = threading.Thread(
                    target=self._monitor_command,
                    args=(command_metadata, done),
                    name="franka-hand-stop",
                    daemon=True,
                )
                monitor.start()
                if grasp:
                    self._grasp_requested = True
                    success = self.hardware.gripper_grasp(0.0, self.speed, 20.0)
                else:
                    with self._condition:
                        self._position_mode_active = True
                    success = self.hardware.gripper_move(target_width, self.speed)
        except self._command_error as exc:
            # Match FrankaHandClient::applyGripperCommand in frankateleop:
            # rejecting one command must not permanently kill the worker.
            # Transport/protocol failures are separate exception types.
            rejection = exc
            success = False
        except Exception as exc:  # noqa: BLE001 - save context and chain the original SDK failure
            fatal = RuntimeError(
                f"Franka Hand {command} failed: target={target_width:.4f} m, "
                f"measured={width:.4f} m, is_grasped={is_grasped}: "
                f"{type(exc).__name__}: {exc}"
            )
            fatal.__cause__ = exc
        finally:
            with self._condition:
                command_metadata["returned_monotonic_ns"] = time.monotonic_ns()
                command_metadata["success"] = bool(success)
                command_metadata["error"] = str(fatal or rejection) if fatal or rejection else None
                command_metadata["status"] = (
                    "error"
                    if fatal
                    else "cancelled"
                    if command_metadata["cancel_reason"]
                    else "succeeded"
                    if success
                    else "rejected"
                    if rejection
                    else "unsuccessful"
                )
                done.set()
                self._condition.notify_all()
            if monitor is not None:
                monitor.join(timeout=3)
                if monitor.is_alive():
                    error = TimeoutError("Franka Hand stop call did not return; no further commands will run")
                    with self._condition:
                        if self._error is None:
                            self._error = error
                        self._condition.notify_all()
                    # Keep the command worker alive until its monitor exits. close()
                    # can then time out safely without racing an outstanding SDK stop.
                    monitor.join()
                    raise error
        # Keep the post-command read barrier: stale width can otherwise make a
        # queued reversal look like a no-op. Use the actual completion boundary,
        # not the later monitor join, which can needlessly cost another read.
        completed_ns = max(
            command_metadata["returned_monotonic_ns"],
            command_metadata["stop_returned_monotonic_ns"] or 0,
        )
        with self._condition:
            fresh = self._condition.wait_for(
                lambda: (
                    fatal is not None
                    or self._closed
                    or self._error is not None
                    or self._feedback_read_after(completed_ns)
                ),
                timeout=0.5,
            )
            self._completed_commands.append(dict(command_metadata))
            self._condition.notify_all()
            if fatal is not None:
                raise fatal
            if self._closed:
                return
            self.check()
            if not fresh:
                raise TimeoutError("Franka Hand state did not update after command")
            measured = self._width
            is_grasped = self._is_grasped
        if not success and command_metadata["cancel_reason"] is None:
            logger.warning(
                "Franka Hand %s unsuccessful: target=%.4f m, measured=%.4f m, is_grasped=%s, error=%s; "
                "waiting for the next target",
                command,
                target_width,
                measured,
                is_grasped,
                rejection if rejection is not None else "command returned False",
            )

    def _feedback_read_after(self, boundary_ns: int) -> bool:
        """Require a completed SDK read begun after the command/stop boundary.

        readOnce drains queued UDP packets before waiting. A read already in
        progress at completion can return immediately afterward with an older
        sample, so its return time alone cannot provide this ordering barrier.
        This orders delivery only; it cannot prove device width freshness.
        """
        started = self._sample_metadata["read_started_monotonic_ns"]
        returned = self._sample_metadata["read_returned_monotonic_ns"]
        return started is not None and returned is not None and started > boundary_ns and returned >= started

    def _prepare_grasp(self, command: dict[str, Any]) -> bool:
        """Exit position mode once, retaining the latest leader/shutdown intent."""
        with self._condition:
            if self._closed:
                command["cancel_reason"] = "shutdown_before_dispatch"
                return False
            self.check()
            if self._pending is not None and self._pending < 0.95:
                command["cancel_reason"] = "leader_changed_before_grasp"
                return False
            # A newly received grasp indication can arrive after _execute's
            # initial snapshot. Never stop an already detected held object.
            if self._is_grasped:
                command["cancel_reason"] = "grasp_already_detected"
                return False
            if self._position_mode_active is False:
                return True
            command["preparation_type"] = "stop_position_mode"
            command["preparation_reason"] = (
                "unknown_initial_mode" if self._position_mode_active is None else "position_move"
            )
            command["preparation_started_monotonic_ns"] = time.monotonic_ns()
            command["preparation_timeout_s"] = PREPARATION_STOP_TIMEOUT_S
            command["preparation_timed_out"] = False
        stop_done = threading.Event()
        watchdog = threading.Thread(
            target=self._watch_preparation_stop,
            args=(command, stop_done),
            name="franka-hand-prepare-watchdog",
            daemon=True,
        )
        watchdog.start()
        try:
            with self._stop_lock:
                stopped = self.hardware.gripper_stop()
        except Exception as exc:
            # Even SDK CommandError is fatal for a failed mode-reset stop:
            # do not treat it like an ordinary rejected motion and continue.
            raise RuntimeError("Franka Hand position-mode stop failed") from exc
        finally:
            returned_ns = time.monotonic_ns()
            stop_done.set()
            watchdog.join(timeout=1)
            if watchdog.is_alive():
                raise RuntimeError("Franka Hand preparation watchdog did not stop")
            with self._condition:
                command["preparation_returned_monotonic_ns"] = returned_ns
                # The watchdog itself can be descheduled. A late SDK return
                # must never slip through merely because it set done first.
                elapsed_s = (
                    command["preparation_returned_monotonic_ns"] - command["preparation_started_monotonic_ns"]
                ) / 1e9
                if elapsed_s >= PREPARATION_STOP_TIMEOUT_S:
                    self._preparation_stop_timed_out(command)
        with self._condition:
            command["preparation_success"] = bool(stopped)
            if not stopped:
                raise RuntimeError("Franka Hand did not acknowledge position-mode stop")
            self._position_mode_active = False
            self.check()  # A stop returning after its deadline cannot dispatch a grasp.
            returned_ns = command["preparation_returned_monotonic_ns"]
            fresh = self._condition.wait_for(
                lambda: self._closed or self._error is not None or self._feedback_read_after(returned_ns),
                timeout=0.5,
            )
            if self._closed:
                command["cancel_reason"] = "shutdown_before_dispatch"
                return False
            self.check()
            if not fresh:
                raise TimeoutError("Franka Hand feedback did not update after position-mode stop")
            if self._pending is not None and self._pending < 0.95:
                command["cancel_reason"] = "leader_changed_before_grasp"
                return False
            if self._is_grasped:
                command["cancel_reason"] = "grasp_already_detected"
                return False
            command["initial_width_m"] = self._width
            command["preparation_feedback_started_monotonic_ns"] = self._sample_metadata[
                "read_started_monotonic_ns"
            ]
            command["preparation_feedback_monotonic_ns"] = self._sample_metadata["read_returned_monotonic_ns"]
        return True

    def _preparation_stop_timed_out(self, command: dict[str, Any]) -> None:
        """Publish a fatal preparation failure while holding the condition."""
        if command["preparation_timed_out"]:
            return
        command["preparation_timed_out"] = True
        command["preparation_timeout_monotonic_ns"] = time.monotonic_ns()
        error = TimeoutError(
            f"Franka Hand position-mode stop exceeded {PREPARATION_STOP_TIMEOUT_S:.1f} s; "
            "no further commands will run"
        )
        command["preparation_error"] = str(error)
        if self._error is None:
            self._error = error
        self._condition.notify_all()

    def _watch_preparation_stop(self, command: dict[str, Any], done: threading.Event) -> None:
        """Flag a stalled SDK call without retrying stop or starting motion."""
        deadline = command["preparation_started_monotonic_ns"] / 1e9 + PREPARATION_STOP_TIMEOUT_S
        if done.wait(max(0.0, deadline - time.monotonic())):
            return
        with self._condition:
            if not done.is_set():
                self._preparation_stop_timed_out(command)

    def _monitor_command(self, command: dict[str, Any], done: threading.Event) -> None:
        """Interrupt superseded commands and bound contact-blocked position moves."""
        started = command["started_monotonic_ns"] / 1e9
        # Leave margin for acceleration, device processing and force settling.
        distance = abs(command["initial_width_m"] - command["target_width_m"])
        budget = (
            max(3.0, distance / self.speed + 2.0)
            if command["type"] == "grasp"
            else max(0.5, distance / self.speed + 0.5)
        )
        try:
            with self._condition:
                while not done.is_set():
                    age = time.monotonic() - started
                    reason = None
                    if self._closed or self._error is not None:
                        reason = "shutdown"
                    elif self._pending is not None:
                        pending_width = self.max_width * (1 - self._pending)
                        if command["type"] == "grasp":
                            if self._pending <= 0.1 or (
                                self._pending < 0.9 and pending_width > self._width + 0.002
                            ):
                                reason = "leader_opened"
                        elif (
                            self._pending >= 0.95
                            and command["target_width_m"] > self.max_width * 0.05
                            and command["target_width_m"] > command["initial_width_m"]
                        ):
                            # Full closure reverses an opening move. During a
                            # closing move it continues the same intent: let the
                            # move finish, then enter grasp via the mode-reset
                            # path. Active stop at contact can stall for seconds.
                            # A blocked closing move still has the travel budget.
                            reason = "leader_requested_grasp"
                        else:
                            direction = command["target_width_m"] - command["initial_width_m"]
                            change = pending_width - command["target_width_m"]
                            if direction * change < 0 and abs(change) >= 0.004:
                                reason = "leader_reversed"
                    if reason is None and age > budget:
                        reason = "command_timeout"
                    # Allow the native call to enter before attempting stop.
                    # The final shutdown stop also runs after joining workers.
                    if reason is not None and age >= 0.05:
                        command["preemption_reason"] = reason
                        break
                    self._condition.wait(timeout=0.02)
                else:
                    return
            # A stop can race ahead of a delayed SDK request. Its successful
            # reply alone is insufficient: keep stopping until the original
            # call returns. No next command starts until this monitor joins.
            while not done.is_set():
                with self._stop_lock:
                    # The original command can finish while this thread waits
                    # for the stop lock. Do not send a now-unnecessary stop.
                    with self._condition:
                        if done.is_set():
                            command["stop_skipped_command_completed"] = True
                            break
                        command["cancel_reason"] = reason
                        command["stop_attempts"] = command.get("stop_attempts", 0) + 1
                        if command["stop_started_monotonic_ns"] is None:
                            command["stop_started_monotonic_ns"] = time.monotonic_ns()
                        command["stop_last_started_monotonic_ns"] = time.monotonic_ns()
                    stopped = False
                    try:
                        stopped = self.hardware.gripper_stop()
                    except Exception as exc:
                        with self._condition:
                            command["stop_error"] = str(exc)
                            command["stop_error_type"] = type(exc).__name__
                        # A failed stop is still fatal, even if the motion call
                        # returned meanwhile: abort is not stop acknowledgement.
                        raise
                    finally:
                        with self._condition:
                            command["stop_returned_monotonic_ns"] = time.monotonic_ns()
                            command["stop_success"] = bool(stopped)
                            command["command_done_at_stop_return"] = done.is_set()
                            self._condition.notify_all()
                    with self._condition:
                        if not stopped:
                            raise RuntimeError("Franka Hand did not acknowledge stop")
                        self._position_mode_active = False
                done.wait(0.05)
            if command["cancel_reason"] == "command_timeout":
                logger.warning(
                    "Franka Hand %s exceeded %.2f s travel budget and was stopped; waiting for latest target",
                    command["type"],
                    budget,
                )
        except Exception as exc:  # noqa: BLE001 - propagate monitor faults to the owning session
            with self._condition:
                if self._error is None:
                    self._error = exc
                self._condition.notify_all()
            logger.error(
                "Franka Hand stop failed: command=%s #%s, reason=%s, attempts=%s, error=%s: %s",
                command["type"],
                command["sequence"],
                command.get("preemption_reason"),
                command.get("stop_attempts", 0),
                type(exc).__name__,
                exc,
            )

    def close(self) -> None:
        self._state_stop.set()
        with self._condition:
            self._closed = True
            self._pending = None
            self._pending_submitted_ns = None
            self._condition.notify_all()
        try:
            if self._thread is not None:
                # The command monitor handles an in-flight call. Do not send
                # an independent late stop while the worker starts another.
                self._thread.join(timeout=3)
                if self._thread.is_alive():
                    raise TimeoutError("Franka Hand worker did not stop")
                with self._stop_lock:
                    if not self.hardware.gripper_stop():
                        raise RuntimeError("Franka Hand did not acknowledge shutdown stop")
        finally:
            self._state_thread.join(timeout=3)
            if self._state_thread.is_alive():
                raise TimeoutError("Franka Hand state reader did not stop")
