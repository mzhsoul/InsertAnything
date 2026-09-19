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


def _clip_vec_with_flags(v: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    clipped = np.clip(v, low, high)
    flags = ~np.isclose(clipped, v)
    return clipped, flags


@dataclass
class SafetyGuardConfig:
    """Implementation helper."""

    hole_box_lower: tuple[float, float, float] = (-0.055, -0.055, 0.0)
    hole_box_upper: tuple[float, float, float] = (+0.055, +0.055, 0.06)

    enable_hole_box_clip: bool = True

    enable_step_clip: bool = False

    step_max_xyz: tuple[float, float, float] = (0.02, 0.02, 0.02)

    yaw_abs_limit_deg: float = 15.0
    yaw_step_limit_deg: float = 2.0

    enforce_ref_roll_pitch: bool = True


@dataclass
class SafetyGuardOutput:
    T_BT_in: np.ndarray
    T_BT_safe: np.ndarray

    p_target_in: np.ndarray
    p_target_safe: np.ndarray

    p_rel_H_in: np.ndarray
    p_rel_H_safe: np.ndarray

    p_rel_PRE_safe: np.ndarray

    delta_step_in: np.ndarray
    delta_step_safe: np.ndarray

    yaw_ref: float
    yaw_current: float
    yaw_in: float
    yaw_safe: float

    was_hole_box_clipped: bool
    was_step_clipped: bool
    was_yaw_abs_clipped: bool
    was_yaw_step_clipped: bool

    raw_ok_without_guard: bool
    final_safe: bool


class SafetyGuard:
    def __init__(self, cfg: SafetyGuardConfig):
        self.cfg = cfg
        self._hole_box_lower = np.asarray(cfg.hole_box_lower, dtype=np.float64)
        self._hole_box_upper = np.asarray(cfg.hole_box_upper, dtype=np.float64)
        self._step_max_xyz = np.asarray(cfg.step_max_xyz, dtype=np.float64)

    def apply(
        self,
        current_T_BT: np.ndarray,
        candidate_T_BT: np.ndarray,
        T_BH: np.ndarray,
        T_BPRE: np.ndarray,
        orientation_ref_pose: np.ndarray | None = None,
    ) -> SafetyGuardOutput:
        """Implementation helper."""
        cfg = self.cfg

        p_current = pose_position(current_T_BT)
        p_in = pose_position(candidate_T_BT)
        p_hole = pose_position(T_BH)
        p_pre = pose_position(T_BPRE)

        p_rel_H_in = p_in - p_hole
        delta_step_in = p_in - p_current

        p_safe = p_in.copy()

        if cfg.enable_hole_box_clip:
            p_rel_H_candidate = p_safe - p_hole
            p_rel_H_clipped, hole_flags = _clip_vec_with_flags(
                p_rel_H_candidate,
                self._hole_box_lower,
                self._hole_box_upper,
            )
            was_hole_box_clipped = bool(np.any(hole_flags))
            p_safe = p_hole + p_rel_H_clipped
        else:
            was_hole_box_clipped = False

        if cfg.enable_step_clip:
            delta_step_candidate = p_safe - p_current
            delta_step_clipped, step_flags = _clip_vec_with_flags(
                delta_step_candidate,
                -self._step_max_xyz,
                +self._step_max_xyz,
            )
            was_step_clipped = bool(np.any(step_flags))
            p_safe = p_current + delta_step_clipped
        else:
            was_step_clipped = False

        delta_step_safe = p_safe - p_current

        ref_pose = orientation_ref_pose if orientation_ref_pose is not None else T_BH

        rpy_ref = pose_rpy_xyz(ref_pose)
        rpy_current = pose_rpy_xyz(current_T_BT)
        rpy_in = pose_rpy_xyz(candidate_T_BT)

        roll_ref, pitch_ref, yaw_ref = (float(x) for x in rpy_ref)
        yaw_current = float(rpy_current[2])
        yaw_in = float(rpy_in[2])

        yaw_rel_ref = _wrap_to_pi(yaw_in - yaw_ref)
        yaw_rel_ref_clipped = float(
            np.clip(
                yaw_rel_ref,
                -np.deg2rad(cfg.yaw_abs_limit_deg),
                +np.deg2rad(cfg.yaw_abs_limit_deg),
            )
        )
        was_yaw_abs_clipped = not np.isclose(yaw_rel_ref_clipped, yaw_rel_ref)
        yaw_after_abs = _wrap_to_pi(yaw_ref + yaw_rel_ref_clipped)

        yaw_rel_current = _wrap_to_pi(yaw_after_abs - yaw_current)
        yaw_rel_current_clipped = float(
            np.clip(
                yaw_rel_current,
                -np.deg2rad(cfg.yaw_step_limit_deg),
                +np.deg2rad(cfg.yaw_step_limit_deg),
            )
        )
        was_yaw_step_clipped = not np.isclose(yaw_rel_current_clipped, yaw_rel_current)
        yaw_safe = _wrap_to_pi(yaw_current + yaw_rel_current_clipped)

        if cfg.enforce_ref_roll_pitch:
            roll_safe = roll_ref
            pitch_safe = pitch_ref
        else:
            roll_safe = float(rpy_in[0])
            pitch_safe = float(rpy_in[1])

        q_safe = euler_xyz_to_quat(roll_safe, pitch_safe, yaw_safe)
        T_BT_safe = make_pose7(p_safe, q_safe)

        p_rel_H_safe = p_safe - p_hole
        p_rel_PRE_safe = p_safe - p_pre

        raw_ok_without_guard = not (
            was_hole_box_clipped or was_step_clipped or was_yaw_abs_clipped or was_yaw_step_clipped
        )

        final_safe = True

        return SafetyGuardOutput(
            T_BT_in=candidate_T_BT.copy(),
            T_BT_safe=T_BT_safe.copy(),
            p_target_in=p_in.copy(),
            p_target_safe=p_safe.copy(),
            p_rel_H_in=p_rel_H_in.copy(),
            p_rel_H_safe=p_rel_H_safe.copy(),
            p_rel_PRE_safe=p_rel_PRE_safe.copy(),
            delta_step_in=delta_step_in.copy(),
            delta_step_safe=delta_step_safe.copy(),
            yaw_ref=float(yaw_ref),
            yaw_current=float(yaw_current),
            yaw_in=float(yaw_in),
            yaw_safe=float(yaw_safe),
            was_hole_box_clipped=bool(was_hole_box_clipped),
            was_step_clipped=bool(was_step_clipped),
            was_yaw_abs_clipped=bool(was_yaw_abs_clipped),
            was_yaw_step_clipped=bool(was_yaw_step_clipped),
            raw_ok_without_guard=bool(raw_ok_without_guard),
            final_safe=bool(final_safe),
        )


__all__ = [
    "SafetyGuardConfig",
    "SafetyGuardOutput",
    "SafetyGuard",
]
