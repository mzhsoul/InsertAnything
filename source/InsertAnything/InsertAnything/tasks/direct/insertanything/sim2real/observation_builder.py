from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from robot_state_adapter import RobotState
from transforms import (
    compose_pose,
    compute_fingertip_pos_rel_fixed,
    compute_fingertip_pose,
    compute_fingertip_quat_rel_fixed,
    euler_xyz_to_quat,
    make_pose7,
    pose_position,
    pose_quat,
    quat_inv,
    quat_mul,
    quat_to_euler_xyz,
    rotate_vector,
    shift_twist_from_A_to_T,
)


def _as_np_1d(x, expected_len: int | None = None, name: str = "array") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if expected_len is not None and arr.shape[0] != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got shape {arr.shape}")
    return arr


def _pose_from_cfg(node: dict) -> np.ndarray:
    pos = _as_np_1d(node["position"], 3, "position")
    quat = _as_np_1d(node["quat_xyzw"], 4, "quat_xyzw")
    return make_pose7(pos, quat)


def _get_policy_obs_cfg(calib: CalibrationData) -> dict:
    raw = calib.raw if calib.raw is not None else {}
    return raw.get("policy_observation", {})


def _quat_xyzw_to_ordered(q_xyzw: np.ndarray, order: str, canonicalize_sign: bool) -> np.ndarray:
    """
    Convert quaternion from xyzw to the requested order.
    Optionally canonicalize the sign so that the scalar part is non-negative.
    """
    q_xyzw = _as_np_1d(q_xyzw, 4, "q_xyzw")

    if order == "xyzw":
        q = q_xyzw.copy()
        scalar_idx = 3
    elif order == "wxyz":
        q = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
        scalar_idx = 0
    else:
        raise ValueError(f"Unsupported quat order: {order}")

    if canonicalize_sign and q[scalar_idx] < 0.0:
        q = -q

    return q


def _align_relative_quat_sign_by_dim3(
    q_xyzw: np.ndarray,
    *,
    order: str,
    expected_sign: int,
) -> np.ndarray:
    """
    Align q and -q so the 3rd quaternion component in the final ordered
    observation carries the expected sign.
    """
    expected_sign = int(expected_sign)
    if expected_sign == 0:
        return q_xyzw

    q_ordered = _quat_xyzw_to_ordered(q_xyzw, order=order, canonicalize_sign=False)
    if q_ordered.shape[0] == 4 and float(q_ordered[2]) * float(expected_sign) < 0.0:
        return -q_xyzw
    return q_xyzw


def _apply_policy_yaw_offset(T_BT: np.ndarray, yaw_offset_deg: float) -> np.ndarray:
    """
    Apply a fixed yaw offset for policy observation only.
    This does NOT change the physical pose, only what is fed into the policy.
    """
    if abs(yaw_offset_deg) < 1e-12:
        return T_BT

    q_offset = euler_xyz_to_quat(0.0, 0.0, np.deg2rad(yaw_offset_deg))
    T_offset = make_pose7([0.0, 0.0, 0.0], q_offset)
    return compose_pose(T_BT, T_offset)


def _override_policy_roll_pitch(
    T_BT_for_policy: np.ndarray,
    mode: str,
    *,
    roll_mean_deg: float = 180.0,
    pitch_mean_deg: float = 0.0,
    roll_std_deg: float = 0.0,
    pitch_std_deg: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Implementation helper."""
    if mode == "none":
        return T_BT_for_policy

    if rng is None:
        rng = np.random.default_rng()

    p = pose_position(T_BT_for_policy)
    q_xyzw = pose_quat(T_BT_for_policy)
    roll_real, pitch_real, yaw_real = quat_to_euler_xyz(q_xyzw)

    if mode == "force_train_mean":
        roll_fake = np.deg2rad(roll_mean_deg)
        pitch_fake = np.deg2rad(pitch_mean_deg)
    elif mode == "sample_gaussian":
        roll_fake = np.deg2rad(rng.normal(loc=roll_mean_deg, scale=roll_std_deg))
        pitch_fake = np.deg2rad(rng.normal(loc=pitch_mean_deg, scale=pitch_std_deg))
    else:
        raise ValueError(f"Unsupported fake roll/pitch mode: {mode}")

    q_fake_xyzw = euler_xyz_to_quat(
        float(roll_fake),
        float(pitch_fake),
        float(yaw_real),
    )
    return make_pose7(p, q_fake_xyzw)


@dataclass
class CalibrationData:
    T_AT: np.ndarray
    T_BH: np.ndarray
    T_BPRE: np.ndarray
    raw: dict | None = None


@dataclass
class PolicyObservationConfig:
    """Implementation helper."""

    obs_order: list[str]
    quat_order: str = "wxyz"
    canonicalize_quat_sign: bool = False
    yaw_offset_deg: float = 0.0
    zero_vel_for_debug: bool = False
    pos_rel_bias_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    requires_orientation_logic: bool = False
    symmetry_angles_deg: list[float] = field(default_factory=list)
    contact_force_mode: str = "error"
    append_prev_actions: bool = True

    fake_policy_roll_pitch_mode: str = "none"
    fake_policy_roll_mean_deg: float = 180.0
    fake_policy_pitch_mean_deg: float = 0.0
    fake_policy_roll_std_deg: float = 0.0
    fake_policy_pitch_std_deg: float = 0.0
    grasp_yaw_comp_deg: float = 0.0
    quat_rel_dim3_expected_sign: int = 0


@dataclass
class ObservationComponents:
    """Implementation helper."""

    fingertip_pos_rel_fixed: np.ndarray
    fingertip_quat: np.ndarray
    fingertip_quat_rel_fixed: np.ndarray | None
    ee_linvel: np.ndarray
    ee_angvel: np.ndarray
    contact_force: np.ndarray | None
    prev_actions: np.ndarray
    obs: np.ndarray

    obs_order: list[str]
    obs_order_with_prev_actions: list[str]
    obs_dim: int
    obs_component_slices: dict[str, tuple[int, int]]
    obs_components_by_name: dict[str, np.ndarray]

    quat_order: str
    canonicalize_quat_sign: bool
    requires_orientation_logic: bool
    symmetry_angles_deg: list[float]

    T_BA: np.ndarray
    T_BT: np.ndarray
    T_BT_for_policy: np.ndarray
    p_rel_pre: np.ndarray
    pos_rel_bias_m: np.ndarray

    fingertip_quat_rel_fixed_raw_xyzw: np.ndarray | None = None
    fingertip_quat_rel_fixed_after_symmetry_xyzw: np.ndarray | None = None


def _get_policy_observation_config(calib: CalibrationData) -> PolicyObservationConfig:
    """Implementation helper."""
    raw_cfg = _get_policy_obs_cfg(calib)

    default_obs_order = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
    ]
    obs_order_raw = raw_cfg.get("obs_order", default_obs_order)
    if not isinstance(obs_order_raw, (list, tuple)):
        raise TypeError("policy_observation.obs_order must be a list or tuple.")

    obs_order = [str(x) for x in obs_order_raw]
    allowed_names = {
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "fingertip_quat_rel_fixed",
        "ee_linvel",
        "ee_angvel",
        "contact_force",
    }

    if any(name == "prev_actions" for name in obs_order):
        raise ValueError(
            'Do not put "prev_actions" inside policy_observation.obs_order; '
            "it is appended automatically by build_observation()."
        )

    unknown = [name for name in obs_order if name not in allowed_names]
    if unknown:
        raise ValueError(f"Unsupported observation names in obs_order: {unknown}")

    if len(set(obs_order)) != len(obs_order):
        raise ValueError(f"Duplicate observation names in obs_order: {obs_order}")

    quat_order = str(raw_cfg.get("quat_order", "wxyz"))
    if quat_order not in ("xyzw", "wxyz"):
        raise ValueError(f"Unsupported quat_order: {quat_order}")

    symmetry_angles_raw = raw_cfg.get("symmetry_angles_deg", [])
    if symmetry_angles_raw is None:
        symmetry_angles_deg = []
    elif isinstance(symmetry_angles_raw, (list, tuple)):
        symmetry_angles_deg = [float(x) for x in symmetry_angles_raw]
    else:
        raise TypeError("policy_observation.symmetry_angles_deg must be a list/tuple of numbers.")

    contact_force_mode = str(raw_cfg.get("contact_force_mode", "error"))
    if contact_force_mode not in ("error", "zeros"):
        raise ValueError(f"Unsupported contact_force_mode: {contact_force_mode}")

    return PolicyObservationConfig(
        obs_order=obs_order,
        quat_order=quat_order,
        canonicalize_quat_sign=bool(raw_cfg.get("canonicalize_quat_sign", False)),
        yaw_offset_deg=float(raw_cfg.get("yaw_offset_deg", 0.0)),
        zero_vel_for_debug=bool(raw_cfg.get("zero_vel_for_debug", False)),
        pos_rel_bias_m=_as_np_1d(
            raw_cfg.get("pos_rel_bias_m", [0.0, 0.0, 0.0]),
            3,
            "policy_observation.pos_rel_bias_m",
        ),
        requires_orientation_logic=bool(raw_cfg.get("requires_orientation_logic", False)),
        symmetry_angles_deg=symmetry_angles_deg,
        contact_force_mode=contact_force_mode,
        append_prev_actions=bool(raw_cfg.get("append_prev_actions", True)),
        fake_policy_roll_pitch_mode=str(raw_cfg.get("fake_policy_roll_pitch_mode", "none")),
        fake_policy_roll_mean_deg=float(raw_cfg.get("fake_policy_roll_mean_deg", 180.0)),
        fake_policy_pitch_mean_deg=float(raw_cfg.get("fake_policy_pitch_mean_deg", 0.0)),
        fake_policy_roll_std_deg=float(raw_cfg.get("fake_policy_roll_std_deg", 0.0)),
        fake_policy_pitch_std_deg=float(raw_cfg.get("fake_policy_pitch_std_deg", 0.0)),
        grasp_yaw_comp_deg=float(raw_cfg.get("grasp_yaw_comp_deg", 0.0)),
        quat_rel_dim3_expected_sign=int(raw_cfg.get("quat_rel_dim3_expected_sign", 0)),
    )


def _get_obs_component_dim(name: str) -> int:
    """Implementation helper."""
    dims = {
        "fingertip_pos_rel_fixed": 3,
        "fingertip_quat": 4,
        "fingertip_quat_rel_fixed": 4,
        "ee_linvel": 3,
        "ee_angvel": 3,
        "contact_force": 3,
        "prev_actions": 6,
    }
    if name not in dims:
        raise KeyError(f"Unknown observation component: {name}")
    return dims[name]


def _apply_world_yaw_offset_to_pose(T_BT: np.ndarray, yaw_offset_deg: float) -> np.ndarray:
    """
    Apply a world-Z yaw left-multiply to pose orientation only.

    Position is kept unchanged. This is used to compensate task-level grasp
    yaw offsets in policy observation so the policy sees the same orientation
    semantics as simulation.
    """
    if abs(yaw_offset_deg) < 1e-12:
        return T_BT

    p = pose_position(T_BT)
    q_xyzw = pose_quat(T_BT)
    q_offset = euler_xyz_to_quat(0.0, 0.0, np.deg2rad(yaw_offset_deg))
    q_new = quat_mul(q_offset, q_xyzw)
    return make_pose7(p, q_new)


def _wrap_to_pi(angle_rad: float) -> float:
    """Implementation helper."""
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def _apply_orientation_symmetry_to_relative_quat(
    current_quat_xyzw: np.ndarray,
    target_quat_xyzw: np.ndarray,
    symmetry_angles_deg: list[float],
) -> np.ndarray:
    """Implementation helper."""
    current_quat_xyzw = _as_np_1d(current_quat_xyzw, 4, "current_quat_xyzw")
    target_quat_xyzw = _as_np_1d(target_quat_xyzw, 4, "target_quat_xyzw")

    if len(symmetry_angles_deg) == 0:
        return compute_fingertip_quat_rel_fixed(
            make_pose7([0.0, 0.0, 0.0], current_quat_xyzw),
            make_pose7([0.0, 0.0, 0.0], target_quat_xyzw),
        )

    _, _, current_yaw = quat_to_euler_xyz(current_quat_xyzw)

    target_roll, target_pitch, target_yaw = quat_to_euler_xyz(target_quat_xyzw)

    best_abs_error = None
    best_target_yaw = None

    for angle_deg in symmetry_angles_deg:
        sym_target_yaw = float(target_yaw) + np.deg2rad(float(angle_deg))

        diff = (float(current_yaw) - sym_target_yaw + np.pi) % (2.0 * np.pi) - np.pi
        abs_error = abs(diff)

        if best_abs_error is None or abs_error < best_abs_error:
            best_abs_error = abs_error
            best_target_yaw = sym_target_yaw

    closest_target_quat_xyzw = euler_xyz_to_quat(
        float(target_roll),
        float(target_pitch),
        float(best_target_yaw),
    )

    current_quat_inv_xyzw = quat_inv(current_quat_xyzw)
    relative_quat_to_closest_xyzw = quat_mul(
        current_quat_inv_xyzw,
        closest_target_quat_xyzw,
    )

    return relative_quat_to_closest_xyzw


def _resolve_optional_obs_component(
    component_name: str,
    provider,
    mode: str,
    expected_dim: int,
    robot_state: RobotState,
    calib: CalibrationData,
) -> np.ndarray:
    """Implementation helper."""
    if provider is not None:
        value = provider(robot_state, calib)
        return _as_np_1d(value, expected_dim, component_name)

    if mode == "zeros":
        return np.zeros(expected_dim, dtype=np.float64)

    raise RuntimeError(
        f'Observation component "{component_name}" is requested by obs_order, '
        f'but no provider is supplied and mode="{mode}".'
    )


def load_calibration_data(yaml_path: str | Path) -> CalibrationData:
    yaml_path = Path(yaml_path)
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    transforms_cfg = cfg["transforms"]

    T_AT = _pose_from_cfg(transforms_cfg["T_AT"])

    if "T_BH_final" in transforms_cfg:
        T_BH = _pose_from_cfg(transforms_cfg["T_BH_final"])
    elif "T_BPRE_origin" in transforms_cfg:
        T_BH = _pose_from_cfg(transforms_cfg["T_BPRE_origin"])
    else:
        raise KeyError("Calibration YAML must provide transforms.T_BH_final or transforms.T_BPRE_origin.")

    if "T_BPRE_final" in transforms_cfg:
        T_BPRE = _pose_from_cfg(transforms_cfg["T_BPRE_final"])
    elif "T_BPRE_origin" in transforms_cfg:
        T_BPRE = _pose_from_cfg(transforms_cfg["T_BPRE_origin"])
    else:
        T_BPRE = T_BH.copy()

    return CalibrationData(
        T_AT=T_AT,
        T_BH=T_BH,
        T_BPRE=T_BPRE,
        raw=cfg,
    )


def _get_relative_quat_reference_xyzw(calib) -> np.ndarray:
    """Implementation helper."""
    raw_cfg = _get_policy_obs_cfg(calib)
    ref = raw_cfg.get("relative_quat_reference_quat_xyzw", [0.0, 0.0, 0.0, 1.0])
    return _as_np_1d(ref, 4, "policy_observation.relative_quat_reference_quat_xyzw")


def build_observation(
    robot_state: RobotState,
    calib: CalibrationData,
    prev_actions: np.ndarray | None = None,
    contact_force_provider: Callable[..., np.ndarray] | None = None,
    grasp_yaw_comp_deg: float | None = None,
) -> ObservationComponents:
    """Implementation helper."""
    if prev_actions is None:
        prev_actions = np.zeros(6, dtype=np.float64)
    prev_actions = _as_np_1d(prev_actions, 6, "prev_actions")

    policy_cfg = _get_policy_observation_config(calib)

    T_BA = robot_state.pose_BA
    T_BT = compute_fingertip_pose(T_BA, calib.T_AT)

    fingertip_pos_rel_fixed = compute_fingertip_pos_rel_fixed(T_BT, calib.T_BH)
    fingertip_pos_rel_fixed = fingertip_pos_rel_fixed + policy_cfg.pos_rel_bias_m

    T_BT_for_policy = _apply_policy_yaw_offset(T_BT, policy_cfg.yaw_offset_deg)

    T_BT_for_policy = _override_policy_roll_pitch(
        T_BT_for_policy,
        mode=policy_cfg.fake_policy_roll_pitch_mode,
        roll_mean_deg=policy_cfg.fake_policy_roll_mean_deg,
        pitch_mean_deg=policy_cfg.fake_policy_pitch_mean_deg,
        roll_std_deg=policy_cfg.fake_policy_roll_std_deg,
        pitch_std_deg=policy_cfg.fake_policy_pitch_std_deg,
    )

    fingertip_quat_xyzw_for_policy = pose_quat(T_BT_for_policy)
    fingertip_quat = _quat_xyzw_to_ordered(
        fingertip_quat_xyzw_for_policy,
        order=policy_cfg.quat_order,
        canonicalize_sign=policy_cfg.canonicalize_quat_sign,
    )

    fingertip_quat_rel_fixed_raw_xyzw = None
    fingertip_quat_rel_fixed_after_symmetry_xyzw = None
    fingertip_quat_rel_fixed = None

    if "fingertip_quat_rel_fixed" in policy_cfg.obs_order:
        grasp_yaw_comp_deg_effective = (
            float(policy_cfg.grasp_yaw_comp_deg) if grasp_yaw_comp_deg is None else float(grasp_yaw_comp_deg)
        )
        T_BT_for_rel = _apply_world_yaw_offset_to_pose(
            T_BT_for_policy,
            grasp_yaw_comp_deg_effective,
        )
        current_quat_xyzw = pose_quat(T_BT_for_rel)
        fixed_quat_ref_xyzw = _get_relative_quat_reference_xyzw(calib)

        fingertip_quat_rel_fixed_raw_xyzw = quat_mul(
            quat_inv(current_quat_xyzw),
            fixed_quat_ref_xyzw,
        )
        fingertip_quat_rel_fixed_after_symmetry_xyzw = fingertip_quat_rel_fixed_raw_xyzw.copy()

        if policy_cfg.requires_orientation_logic:
            fingertip_quat_rel_fixed_after_symmetry_xyzw = _apply_orientation_symmetry_to_relative_quat(
                current_quat_xyzw=current_quat_xyzw,
                target_quat_xyzw=fixed_quat_ref_xyzw,
                symmetry_angles_deg=policy_cfg.symmetry_angles_deg,
            )

        fingertip_quat_rel_fixed_after_symmetry_xyzw = _align_relative_quat_sign_by_dim3(
            fingertip_quat_rel_fixed_after_symmetry_xyzw,
            order=policy_cfg.quat_order,
            expected_sign=policy_cfg.quat_rel_dim3_expected_sign,
        )

        fingertip_quat_rel_fixed = _quat_xyzw_to_ordered(
            fingertip_quat_rel_fixed_after_symmetry_xyzw,
            order=policy_cfg.quat_order,
            canonicalize_sign=policy_cfg.canonicalize_quat_sign,
        )

    v_A = robot_state.v_A
    w_A = robot_state.w_A

    r_AT_A = pose_position(calib.T_AT)
    r_AT_base = rotate_vector(robot_state.pose_BA, r_AT_A)
    ee_linvel, ee_angvel = shift_twist_from_A_to_T(v_A, w_A, r_AT_base)

    if policy_cfg.zero_vel_for_debug:
        ee_linvel = np.zeros(3, dtype=np.float64)
        ee_angvel = np.zeros(3, dtype=np.float64)

    p_rel_pre = pose_position(T_BT) - pose_position(calib.T_BPRE)

    contact_force = None
    if "contact_force" in policy_cfg.obs_order:
        contact_force = _resolve_optional_obs_component(
            component_name="contact_force",
            provider=contact_force_provider,
            mode=policy_cfg.contact_force_mode,
            expected_dim=_get_obs_component_dim("contact_force"),
            robot_state=robot_state,
            calib=calib,
        )

    component_map: dict[str, np.ndarray | None] = {
        "fingertip_pos_rel_fixed": fingertip_pos_rel_fixed,
        "fingertip_quat": fingertip_quat,
        "fingertip_quat_rel_fixed": fingertip_quat_rel_fixed,
        "ee_linvel": ee_linvel,
        "ee_angvel": ee_angvel,
        "contact_force": contact_force,
        "prev_actions": prev_actions,
    }

    obs_order_with_prev_actions = list(policy_cfg.obs_order)
    if policy_cfg.append_prev_actions:
        obs_order_with_prev_actions.append("prev_actions")

    obs_parts: list[np.ndarray] = []
    obs_component_slices: dict[str, tuple[int, int]] = {}
    obs_components_by_name: dict[str, np.ndarray] = {}
    cursor = 0

    for name in obs_order_with_prev_actions:
        value = component_map.get(name)
        if value is None:
            raise RuntimeError(
                f'Observation component "{name}" is required by current schema but was not constructed successfully.'
            )

        expected_dim = _get_obs_component_dim(name)
        value = _as_np_1d(value, expected_dim, name)

        obs_parts.append(value)
        obs_components_by_name[name] = value.copy()

        next_cursor = cursor + expected_dim
        obs_component_slices[name] = (cursor, next_cursor)
        cursor = next_cursor

    obs = np.concatenate(obs_parts, axis=0).astype(np.float64)
    obs_dim = int(obs.shape[0])

    return ObservationComponents(
        fingertip_pos_rel_fixed=fingertip_pos_rel_fixed,
        fingertip_quat=fingertip_quat,
        fingertip_quat_rel_fixed=fingertip_quat_rel_fixed,
        ee_linvel=ee_linvel,
        ee_angvel=ee_angvel,
        contact_force=contact_force,
        prev_actions=prev_actions,
        obs=obs,
        obs_order=list(policy_cfg.obs_order),
        obs_order_with_prev_actions=obs_order_with_prev_actions,
        obs_dim=obs_dim,
        obs_component_slices=obs_component_slices,
        obs_components_by_name=obs_components_by_name,
        quat_order=policy_cfg.quat_order,
        canonicalize_quat_sign=policy_cfg.canonicalize_quat_sign,
        requires_orientation_logic=policy_cfg.requires_orientation_logic,
        symmetry_angles_deg=list(policy_cfg.symmetry_angles_deg),
        T_BA=T_BA,
        T_BT=T_BT,
        T_BT_for_policy=T_BT_for_policy,
        p_rel_pre=p_rel_pre,
        pos_rel_bias_m=policy_cfg.pos_rel_bias_m,
        fingertip_quat_rel_fixed_raw_xyzw=fingertip_quat_rel_fixed_raw_xyzw,
        fingertip_quat_rel_fixed_after_symmetry_xyzw=(fingertip_quat_rel_fixed_after_symmetry_xyzw),
    )


__all__ = [
    "CalibrationData",
    "ObservationComponents",
    "load_calibration_data",
    "build_observation",
    "PolicyObservationConfig",
]
