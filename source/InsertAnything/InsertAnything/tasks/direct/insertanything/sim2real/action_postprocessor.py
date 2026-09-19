from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from transforms import euler_xyz_to_quat, make_pose7, pose_position, pose_rpy_xyz


def _as_np_1d(x, expected_len: int | None = None, name: str = "array") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if expected_len is not None and arr.shape[0] != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got shape {arr.shape}")
    return arr


def _wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _clip_with_flag(value: float, low: float, high: float) -> tuple[float, bool]:
    clipped = float(np.clip(value, low, high))
    return clipped, not np.isclose(clipped, value)


@dataclass
class ActionPostprocessorConfig:
    """Implementation helper."""

    ema_factor: float = 0.2

    pos_action_threshold: tuple[float, float, float] = (
        0.02,
        0.02,
        0.02,
    )
    rot_action_threshold: tuple[float, float, float] = (
        0.097,
        0.097,
        0.097,
    )

    pos_action_bounds: tuple[float, float, float] = (
        0.05,
        0.05,
        0.05,
    )

    raw_action_clip: float | None = 1.5

    yaw_enable: bool = False
    yaw_abs_limit_deg: float = 15.0
    yaw_step_limit_deg: float = 2.0


@dataclass
class ActionPostprocessorState:
    action_ema: np.ndarray


@dataclass
class ActionPostprocessorOutput:
    """Implementation helper."""

    raw_action: np.ndarray
    clipped_action: np.ndarray
    action_ema: np.ndarray

    delta_pos: np.ndarray
    delta_rot_scaled: np.ndarray
    delta_yaw: float

    p_target_unclipped: np.ndarray
    p_target_clipped: np.ndarray

    p_target: np.ndarray
    p_rel_H: np.ndarray
    p_rel_PRE: np.ndarray

    yaw_ref: float
    yaw_current: float
    yaw_candidate: float
    yaw_target: float

    T_BT_target: np.ndarray

    was_raw_clipped: bool
    was_position_clipped: bool
    was_yaw_step_clipped: bool
    was_yaw_abs_clipped: bool

    was_action_clipped: bool
    was_pos_bound_clipped: bool


class ActionPostprocessor:
    def __init__(self, cfg: ActionPostprocessorConfig):
        self.cfg = cfg

        self._pos_threshold = np.asarray(cfg.pos_action_threshold, dtype=np.float64)
        self._rot_threshold = np.asarray(cfg.rot_action_threshold, dtype=np.float64)
        self._pos_bounds = np.asarray(cfg.pos_action_bounds, dtype=np.float64)

    def get_initial_state(
        self,
    ) -> ActionPostprocessorState:
        return ActionPostprocessorState(action_ema=np.zeros(6, dtype=np.float64))

    def reset_state(self) -> ActionPostprocessorState:
        return self.get_initial_state()

    def process(
        self,
        current_T_BT: np.ndarray,
        fixed_pos_frame_pose: np.ndarray,
        orientation_ref_pose: np.ndarray | None,
        action_raw: np.ndarray,
        state: ActionPostprocessorState,
    ) -> tuple[ActionPostprocessorOutput, ActionPostprocessorState]:
        """Implementation helper."""
        action_raw = _as_np_1d(action_raw, 6, "action_raw")
        current_T_BT = _as_np_1d(current_T_BT, 7, "current_T_BT")
        fixed_pos_frame_pose = _as_np_1d(fixed_pos_frame_pose, 7, "fixed_pos_frame_pose")

        if orientation_ref_pose is None:
            orientation_ref_pose = fixed_pos_frame_pose
        orientation_ref_pose = _as_np_1d(orientation_ref_pose, 7, "orientation_ref_pose")

        if self.cfg.raw_action_clip is not None:
            clip_val = float(self.cfg.raw_action_clip)
            clipped_action = np.clip(action_raw, -clip_val, +clip_val).astype(np.float64)
            was_raw_clipped = not np.allclose(clipped_action, action_raw)
        else:
            clipped_action = action_raw.copy()
            was_raw_clipped = False

        prev_ema = _as_np_1d(state.action_ema, 6, "state.action_ema")
        alpha = float(self.cfg.ema_factor)
        action_ema = alpha * clipped_action + (1.0 - alpha) * prev_ema
        new_state = ActionPostprocessorState(action_ema=action_ema.copy())

        delta_pos = action_ema[0:3] * self._pos_threshold

        p_current = pose_position(current_T_BT)
        p_fixed = pose_position(fixed_pos_frame_pose)
        p_ref_orientation = pose_position(orientation_ref_pose)

        p_target_unclipped = p_current + delta_pos

        p_low = p_fixed - self._pos_bounds
        p_high = p_fixed + self._pos_bounds
        p_target_clipped = np.clip(p_target_unclipped, p_low, p_high).astype(np.float64)
        was_position_clipped = not np.allclose(p_target_clipped, p_target_unclipped)

        p_target = p_target_clipped.copy()
        p_rel_H = p_target - p_fixed
        p_rel_PRE = p_target - p_ref_orientation

        delta_rot_scaled = action_ema[3:6] * self._rot_threshold

        rpy_current = pose_rpy_xyz(current_T_BT)
        rpy_ref = pose_rpy_xyz(orientation_ref_pose)

        roll_ref = float(rpy_ref[0])
        pitch_ref = float(rpy_ref[1])
        yaw_ref = float(rpy_ref[2])
        yaw_current = float(rpy_current[2])

        if self.cfg.yaw_enable:
            max_step = np.deg2rad(self.cfg.yaw_step_limit_deg)
            max_abs = np.deg2rad(self.cfg.yaw_abs_limit_deg)

            raw_delta_yaw = float(delta_rot_scaled[2])
            delta_yaw, was_yaw_step_clipped = _clip_with_flag(raw_delta_yaw, -max_step, +max_step)

            yaw_candidate = _wrap_to_pi(yaw_current + delta_yaw)

            yaw_low = yaw_ref - max_abs
            yaw_high = yaw_ref + max_abs
            yaw_target, was_yaw_abs_clipped = _clip_with_flag(yaw_candidate, yaw_low, yaw_high)
            yaw_target = _wrap_to_pi(yaw_target)
        else:
            delta_yaw = 0.0
            yaw_candidate = yaw_ref
            yaw_target = yaw_ref
            was_yaw_step_clipped = False
            was_yaw_abs_clipped = False

        q_target = euler_xyz_to_quat(roll_ref, pitch_ref, yaw_target)
        T_BT_target = make_pose7(p_target, q_target)

        output = ActionPostprocessorOutput(
            raw_action=action_raw.copy(),
            clipped_action=clipped_action.copy(),
            action_ema=action_ema.copy(),
            delta_pos=delta_pos.copy(),
            delta_rot_scaled=delta_rot_scaled.copy(),
            delta_yaw=float(delta_yaw),
            p_target_unclipped=p_target_unclipped.copy(),
            p_target_clipped=p_target_clipped.copy(),
            p_target=p_target.copy(),
            p_rel_H=p_rel_H.copy(),
            p_rel_PRE=p_rel_PRE.copy(),
            yaw_ref=float(yaw_ref),
            yaw_current=float(yaw_current),
            yaw_candidate=float(yaw_candidate),
            yaw_target=float(yaw_target),
            T_BT_target=T_BT_target.copy(),
            was_raw_clipped=bool(was_raw_clipped),
            was_position_clipped=bool(was_position_clipped),
            was_yaw_step_clipped=bool(was_yaw_step_clipped),
            was_yaw_abs_clipped=bool(was_yaw_abs_clipped),
            was_action_clipped=bool(was_raw_clipped),
            was_pos_bound_clipped=bool(was_position_clipped),
        )
        return output, new_state


__all__ = [
    "ActionPostprocessorConfig",
    "ActionPostprocessorState",
    "ActionPostprocessorOutput",
    "ActionPostprocessor",
]
