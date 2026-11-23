"""Explicit RL-only state representations, independent of model wire inputs.

Inference still receives the producer's original observation. These encodings
only change the state segment concatenated with frozen features for SAC.
"""
from copy import deepcopy

import numpy as np

from pressb.control.pose_protocol import pose8_to_pose9
from .contract import fingerprint, negotiate


POSE8_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper_width"]
POSE8_UNITS = ["m"] * 3 + ["unitless"] * 4 + ["m"]
POSE9_NAMES = ["x", "y", "z"] + [f"rot6d_{i}" for i in range(6)]
POSE9_UNITS = ["m"] * 3 + ["unitless"] * 6


def negotiate_rl_contract(inference_metadata, simulation_metadata, method, execution_horizon=None, *,
                          state_encoding="raw", terminal_observation_mode="full", fused_step_reset=False):
    """Bind the SAC method and observation encoding to the existing node wire contract."""
    joint = method == "joint_residual"
    # Negotiate the external-noise wire geometry; the method stored in joint
    # checkpoints also identifies the additive decode and paired policy action.
    contract = negotiate(inference_metadata, simulation_metadata, "dsrl" if joint else method,
                         execution_horizon, terminal_observation_mode=terminal_observation_mode,
                         fused_step_reset=fused_step_reset)
    if joint:
        if "additive_noise_residual" not in inference_metadata.get("capabilities", []):
            raise ValueError("Inference missing capability: additive_noise_residual")
        contract["method"] = method
        contract["fingerprint"] = fingerprint({k: v for k, v in contract.items()
                                               if k not in ("fingerprint", "num_envs")})
    return configure_rl_observation(contract, state_encoding, inference_metadata, simulation_metadata)


def configure_rl_observation(contract, encoding, inference_metadata, simulation_metadata):
    """Return learner geometry without changing the producer/model contract.

    The raw default deliberately leaves every contract field/fingerprint
    untouched, so existing checkpoints retain their original schema identity.
    pose9 describes its transformed state explicitly and adds one RL input.
    """
    if encoding not in ("raw", "pose9"):
        raise ValueError("observation_state_encoding must be raw or pose9")
    result = deepcopy(contract)
    if encoding == "raw":
        return result
    for role, metadata in (("inference", inference_metadata), ("simulation", simulation_metadata)):
        state = metadata.get("observation", {}).get("state", {})
        if state.get("names") != POSE8_NAMES or state.get("units") != POSE8_UNITS:
            raise ValueError(f"observation_state_encoding=pose9 requires {role} XYZ+wxyz+gripper pose8 schema in metres")
        if metadata.get("quaternion_order", "wxyz") != "wxyz":
            raise ValueError(f"observation_state_encoding=pose9 requires {role} quaternion_order=wxyz")
    if result.get("state_dim") != 8:
        raise ValueError("observation_state_encoding=pose9 requires a wire state_dim of 8")
    result["learner_state_dim"] = 9
    result["learner_observation"] = {
        "version": 1,
        "state_encoding": "pose9",
        "state": {"names": POSE9_NAMES.copy(), "units": POSE9_UNITS.copy()},
        "rotation6d": "first_two_matrix_rows_row_major",
        "gripper": "omitted_from_rl_state_only",
    }
    result["obs_dim"] = sum(result[key] for key in
                            ("feature_dim", "learner_state_dim", "control_state_dim", "task_features_dim"))
    result["fingerprint"] = fingerprint({key: value for key, value in result.items()
                                         if key not in ("fingerprint", "num_envs")})
    return result


def encode_rl_state(state, encoding):
    """Encode an already shape/finite-validated float32 producer state."""
    if encoding == "raw":
        return state
    if encoding == "pose9":
        # Same absolute XYZ and first two *rows* of R as the old pose9 policy.
        # The converter validates unit wxyz quaternions, including terminal
        # observations; silently normalizing invalid observations is forbidden.
        return pose8_to_pose9(state).astype(np.float32)
    raise ValueError("observation_state_encoding must be raw or pose9")
