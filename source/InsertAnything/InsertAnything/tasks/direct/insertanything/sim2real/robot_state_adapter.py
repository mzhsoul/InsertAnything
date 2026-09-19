from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
from transforms import euler_xyz_to_quat, quat_to_euler_xyz


def _as_np_1d(x: Any, expected_len: int | None = None, name: str = "array") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if expected_len is not None and arr.shape[0] != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got shape {arr.shape}")
    return arr


@dataclass
class RobotState:
    """Implementation helper."""

    pose_BA: np.ndarray
    ee_pose6: np.ndarray
    q: np.ndarray
    dq: np.ndarray
    vel_A: np.ndarray
    gripper_pos: float | None
    force_K: np.ndarray | None = None
    torque_K: np.ndarray | None = None
    raw: dict[str, Any] | None = None

    @property
    def p_BA(self) -> np.ndarray:
        return self.pose_BA[:3]

    @property
    def q_BA(self) -> np.ndarray:
        return self.pose_BA[3:7]

    @property
    def rpy_BA(self) -> np.ndarray:
        return quat_to_euler_xyz(self.q_BA)

    @property
    def v_A(self) -> np.ndarray:
        return self.vel_A[:3]

    @property
    def w_A(self) -> np.ndarray:
        return self.vel_A[3:6]


def _pose7_from_pose6_xyzrpy(pose6: np.ndarray) -> np.ndarray:
    """Implementation helper."""
    pose6 = _as_np_1d(pose6, 6, "ee_pose6")
    pos = pose6[:3]
    rpy = pose6[3:6]
    quat = euler_xyz_to_quat(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    return np.concatenate([pos, quat], axis=0)


def robot_state_from_json(state_json: dict[str, Any]) -> RobotState:
    """Implementation helper."""
    if not isinstance(state_json, dict):
        raise TypeError(f"state_json must be a dict, got {type(state_json).__name__}")

    if "pose" in state_json:
        pose_BA = _as_np_1d(state_json["pose"], 7, 'state_json["pose"]')
    elif "ee" in state_json:
        ee_pose6_tmp = _as_np_1d(state_json["ee"], 6, 'state_json["ee"]')
        pose_BA = _pose7_from_pose6_xyzrpy(ee_pose6_tmp)
    else:
        raise KeyError('state_json must contain at least "pose" or "ee"')

    if "ee" in state_json:
        ee_pose6 = _as_np_1d(state_json["ee"], 6, 'state_json["ee"]')
    else:
        ee_pose6 = np.concatenate([pose_BA[:3], quat_to_euler_xyz(pose_BA[3:7])], axis=0)

    q = _as_np_1d(state_json.get("q", np.zeros(7)), 7, 'state_json["q"]')
    dq = _as_np_1d(state_json.get("dq", np.zeros(7)), 7, 'state_json["dq"]')
    vel_A = _as_np_1d(state_json.get("vel", np.zeros(6)), 6, 'state_json["vel"]')

    gripper_pos_raw = state_json.get("gripper_pos")
    gripper_pos = None if gripper_pos_raw is None else float(gripper_pos_raw)

    force_K = None
    torque_K = None
    if "force" in state_json:
        force_K = _as_np_1d(state_json["force"], 3, 'state_json["force"]')
    if "torque" in state_json:
        torque_K = _as_np_1d(state_json["torque"], 3, 'state_json["torque"]')

    return RobotState(
        pose_BA=pose_BA,
        ee_pose6=ee_pose6,
        q=q,
        dq=dq,
        vel_A=vel_A,
        gripper_pos=gripper_pos,
        force_K=force_K,
        torque_K=torque_K,
        raw=state_json,
    )


def fetch_robot_state(server_url: str, timeout: float = 2.0) -> RobotState:
    """Implementation helper."""
    url = server_url.rstrip("/") + "/getstate"
    resp = requests.post(url, json={}, timeout=timeout)
    resp.raise_for_status()
    state_json = resp.json()
    return robot_state_from_json(state_json)


__all__ = [
    "RobotState",
    "robot_state_from_json",
    "fetch_robot_state",
]
