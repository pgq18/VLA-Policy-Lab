import numpy as np
import pytest

from pressb.control.joint_return import GripperResetTimeout, JointReturnTimeout, execute_joint_return, plan_joint_return, reset_gripper
from pressb.runtime.real_robot_operator import ManualTask, send_command


def test_quintic_return_preserves_gripper_and_limits_planned_speed():
    now, sent = [0.], []
    state = np.array([.5, -.2, .4, -1.6, .1, 1.8, -.3, .97])
    target = state[:7] + .2

    def command(q):
        state[:] = q
        sent.append(q.copy())
        return q

    report = execute_joint_return(state.copy(), target, duration_s=5., control_hz=25,
        read_state=lambda: state.copy(), command=command,
        clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0]+seconds))
    assert report['status'] == 'completed'
    assert np.allclose(state[:7], target)
    assert np.all(np.asarray(sent)[:, -1] == .97)
    assert np.max(np.abs(np.diff(sent, axis=0)[:, :7])) * 25 <= .2
    assert 5 <= report['wall_seconds'] < 5.1


def test_return_aborts_before_next_command_and_times_out_if_arm_does_not_follow():
    for abort in (True, False):
        now, sent = [0.], []
        initial = np.array([0.] * 7 + [.5])

        def progress(_):
            if abort and len(sent) == 3:
                raise RuntimeError('operator_abort')

        with pytest.raises(RuntimeError if abort else TimeoutError):
            execute_joint_return(initial, [.2] * 7, duration_s=1., control_hz=25,
                read_state=lambda: initial, command=lambda q: (sent.append(q.copy()) or q),
                progress=progress, settle_timeout_s=.1,
                clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0]+seconds))
        if abort:
            assert len(sent) == 3


def test_operator_return_has_owned_token_abort_and_no_early_ready(tmp_path):
    task = ManualTask({'directory': str(tmp_path), 'reset_timeout_s': 1})
    try:
        task.begin_return([0.] * 7, reset_seed=1, reset_options={'operator_console_run_id': 'run'})
        status = task.status()
        assert status['phase'] == 'returning'
        assert status['options']['operator_console_run_id'] == 'run'
        with pytest.raises(ValueError, match='phase'):
            send_command(tmp_path, 'ready', status['episode_token'])
        send_command(tmp_path, 'abort', status['episode_token'])
        with pytest.raises(RuntimeError, match='abort'):
            task.check_abort()
        task.fail_return('operator_abort')
        assert task.status()['return_to_initial']['status'] == 'aborted'
    finally:
        task.close()


@pytest.mark.parametrize('bias,tolerance,accepted', [(.0205, .02, False), (.0205, .05, True), (.06, .05, False)])
def test_impedance_offset_matches_reference_completion_rule(tmp_path, bias, tolerance, accepted):
    now = [0.]
    state = np.zeros(8)
    state[-1] = .97
    sent = []

    def command(q):
        sent.append(q.copy())
        state[:] = q
        state[5] += bias  # Static tracking offset in J6, matching the reported failure.
        return q.copy()

    kwargs = dict(duration_s=1., control_hz=25, read_state=lambda: state.copy(),
                  command=command, tolerance_rad=tolerance, settle_timeout_s=.2,
                  clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0]+seconds))
    if accepted:
        result = execute_joint_return(state.copy(), [.2]*7, **kwargs)
        assert result['max_joint_error_rad'] == pytest.approx(bias)
        assert result['worst_joint'] == 6 and result['tolerance_rad'] == tolerance
        assert result['wall_seconds'] >= 1.  # Never accept before the complete trajectory.
    else:
        with pytest.raises(JointReturnTimeout) as error:
            execute_joint_return(state.copy(), [.2]*7, **kwargs)
        report = error.value.report
        assert report['worst_joint'] == 6
        assert report['joint_errors_rad'][5] == pytest.approx(bias)
        assert 1 <= len(report['last_samples']) <= 25
        assert np.allclose(report['last_applied_action'][:7], [.2]*7)
        task = ManualTask({'directory': str(tmp_path/'operator')})
        try:
            task.begin_return([.2]*7, reset_seed=1, reset_options={})
            task.fail_return(error.value)
            assert task.status()['return_to_initial']['worst_joint'] == 6
            assert task.status()['return_to_initial']['last_samples'] == report['last_samples']
        finally:
            task.close()
    assert np.allclose(sent[-1][:7], [.2]*7)  # No goal correction or controller gain change.
    assert sent[-1][-1] == .97


def test_gripper_reset_waits_for_async_feedback_and_preserves_arm():
    now, commands = [0.], []
    initial = np.array([.1, -.2, .3, -1.7, .05, 1.6, -.4, .97])
    target = .005459577310830355

    def read():
        state = initial.copy()
        state[-1] = target if now[0] >= .4 else initial[-1]
        return state

    report = reset_gripper(initial, target, read_state=read,
        command=lambda q: (commands.append(q.copy()) or q), control_hz=25,
        command_ready=lambda: now[0] >= .4, clock=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0]+seconds))
    assert len(commands) == 1 and np.array_equal(commands[0][:-1], initial[:-1])
    assert commands[0][-1] == target
    assert report['status'] == 'completed' and report['wall_seconds'] >= .6
    assert report['measured'] == target


def test_gripper_reset_does_not_accept_old_open_feedback_while_worker_is_busy():
    now, sent = [0.], []
    state = np.array([0.] * 7 + [.005])
    report = reset_gripper(state, .005, read_state=lambda: state,
        command=lambda q: (sent.append(q.copy()) or q), control_hz=25,
        command_ready=lambda: now[0] >= .4, clock=lambda: now[0],
        sleep=lambda seconds: now.__setitem__(0, now[0]+seconds))
    assert report['already_at_initial'] and report['wall_seconds'] >= .6
    assert len(sent) == 1  # Replace an old close intent even if feedback currently appears open.


def test_gripper_reset_abort_and_failure_do_not_issue_repeated_commands():
    now, sent = [0.], []
    state = np.array([0.] * 7 + [.97])

    def abort(_):
        raise RuntimeError('operator_abort')

    kwargs = dict(read_state=lambda: state, command=lambda q: (sent.append(q.copy()) or q),
                  control_hz=25, timeout_s=.4, stable_s=.1, clock=lambda: now[0],
                  sleep=lambda seconds: now.__setitem__(0, now[0]+seconds))
    with pytest.raises(RuntimeError, match='operator_abort'):
        reset_gripper(state, .005, progress=abort, **kwargs)
    assert sent == []

    def abort_during_wait(_):
        if now[0] >= .08:
            raise RuntimeError('operator_abort')

    with pytest.raises(RuntimeError, match='operator_abort'):
        reset_gripper(state, .005, progress=abort_during_wait, **kwargs)
    assert len(sent) == 1
    now[0] = 0.
    sent.clear()
    with pytest.raises(GripperResetTimeout) as error:
        reset_gripper(state, .005, **kwargs)
    assert len(sent) == 1
    assert error.value.report['gripper_reset']['measured'] == .97
    assert error.value.report['gripper_reset']['target'] == .005


@pytest.mark.parametrize('delta', [.02, .5, 1.2])
def test_faster_return_retimes_same_path_with_velocity_and_acceleration_bounds(delta):
    class Limits:
        velocity = np.array([.25, 2., 2., 2., 2., 2., 2.])
        def check(self, q, label):
            assert np.all(np.abs(q) <= 2.)
            return np.array([.4+.2*q[0], .1+.1*q[1], .4+.05*q[2]])

    start, target = np.zeros(7), np.array([delta, -delta/2, delta/3, 0, 0, 0, 0])
    old = plan_joint_return(start, target, limits=Limits())
    new = plan_joint_return(start, target, limits=Limits(), speed_rad_s=.3,
                            min_duration_s=3.5, max_flange_speed_m_s=.075)
    assert new['duration_s'] < old['duration_s']
    assert new['start_joints'] == old['start_joints'] and new['target_joints'] == old['target_joints']
    assert new['path_samples'] == old['path_samples']
    fractions = np.linspace(0, 1, 10001)
    blend = 10*fractions**3 - 15*fractions**4 + 6*fractions**5
    q = start + blend[:, None]*(target-start)
    dt = new['duration_s']/10000
    dq = np.diff(q, axis=0)/dt
    ddq = np.diff(dq, axis=0)/dt
    assert np.all(np.max(np.abs(dq), axis=0) <= np.minimum(Limits.velocity, .3)+1e-6)
    assert np.max(np.abs(ddq)) <= .4+1e-6
    xyz = np.array([Limits().check(row, 'test') for row in q])
    assert np.max(np.linalg.norm(np.diff(xyz, axis=0)/dt, axis=1)) <= .075+1e-6
