from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _as_np_1d(x, expected_len: int | None = None, name: str = "array") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    if expected_len is not None and arr.shape[0] != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got shape {arr.shape}")
    return arr


def _as_np_mat(x, shape: tuple[int, int], name: str = "matrix") -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def _skew(v: np.ndarray) -> np.ndarray:
    v = _as_np_1d(v, 3, "v")
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float64,
    )


def _check_quat_input_no_autofix(q, name: str = "q") -> np.ndarray:
    """Implementation helper."""
    q = _as_np_1d(q, 4, name)

    if not np.all(np.isfinite(q)):
        raise ValueError(f"{name} contains non-finite values: {q}")

    n2 = float(np.dot(q, q))
    if n2 < _EPS:
        raise ValueError(f"{name} norm is too small.")

    return q


def quat_normalize(q) -> np.ndarray:
    """Implementation helper."""
    q = _check_quat_input_no_autofix(q, "q")
    n = np.linalg.norm(q)
    return q / n


def quat_conjugate(q) -> np.ndarray:
    q = _as_np_1d(q, 4, "q")
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_inv(q) -> np.ndarray:
    q = _as_np_1d(q, 4, "q")
    n2 = np.dot(q, q)
    if n2 < _EPS:
        raise ValueError("Quaternion norm is too small.")
    return quat_conjugate(q) / n2


def quat_mul(q1, q2) -> np.ndarray:
    """Implementation helper."""
    q1 = _check_quat_input_no_autofix(q1, "q1")
    q2 = _check_quat_input_no_autofix(q2, "q2")

    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return np.array([x, y, z, w], dtype=np.float64)


def quat_to_rotmat(q) -> np.ndarray:
    """Implementation helper."""
    q = _check_quat_input_no_autofix(q, "q")
    x, y, z, w = q

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


def rotmat_to_quat(R) -> np.ndarray:
    """Implementation helper."""
    R = _as_np_mat(R, (3, 3), "R")

    trace = np.trace(R)
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return np.array([x, y, z, w], dtype=np.float64)


def euler_xyz_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Implementation helper."""
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([x, y, z, w], dtype=np.float64)


def quat_to_euler_xyz(q) -> np.ndarray:
    """Implementation helper."""
    q = _check_quat_input_no_autofix(q, "q")
    x, y, z, w = q

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * (np.pi / 2.0)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float64)


def make_pose7(position, quat) -> np.ndarray:
    """Implementation helper."""
    position = _as_np_1d(position, 3, "position")
    quat = _check_quat_input_no_autofix(quat, "quat")
    return np.concatenate([position, quat], axis=0)


def split_pose7(pose7) -> tuple[np.ndarray, np.ndarray]:
    """Implementation helper."""
    pose7 = _as_np_1d(pose7, 7, "pose7")
    position = pose7[:3].copy()
    quat = _check_quat_input_no_autofix(pose7[3:7], "pose7[3:7]")
    return position, quat


def pose7_to_mat(
    pose7,
) -> np.ndarray:
    p, q = split_pose7(pose7)
    R = quat_to_rotmat(q)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def mat_to_pose7(T) -> np.ndarray:

    T = _as_np_mat(T, (4, 4), "T")
    p = T[:3, 3]
    R = T[:3, :3]
    q = rotmat_to_quat(R)
    return make_pose7(p, q)


def compose_pose(T_XY, T_YZ) -> np.ndarray:
    """Implementation helper."""
    p_XY, q_XY = split_pose7(T_XY)
    p_YZ, q_YZ = split_pose7(T_YZ)

    R_XY = quat_to_rotmat(q_XY)
    p_XZ = p_XY + R_XY @ p_YZ
    q_XZ = quat_mul(q_XY, q_YZ)
    return make_pose7(p_XZ, q_XZ)


def invert_pose(T_XY) -> np.ndarray:
    """
    Return T_YX.
    """
    p_XY, q_XY = split_pose7(T_XY)
    R_XY = quat_to_rotmat(q_XY)

    R_YX = R_XY.T
    p_YX = -R_YX @ p_XY
    q_YX = quat_inv(q_XY)
    return make_pose7(p_YX, q_YX)


def relative_pose(T_XA, T_XB) -> np.ndarray:
    """Implementation helper."""
    return compose_pose(invert_pose(T_XA), T_XB)


def transform_point(T_XY, p_Y) -> np.ndarray:

    p_XY, q_XY = split_pose7(T_XY)
    p_Y = _as_np_1d(p_Y, 3, "p_Y")
    R_XY = quat_to_rotmat(q_XY)
    return p_XY + R_XY @ p_Y


def rotate_vector(T_XY, v_Y) -> np.ndarray:

    _, q_XY = split_pose7(T_XY)
    v_Y = _as_np_1d(v_Y, 3, "v_Y")
    R_XY = quat_to_rotmat(q_XY)
    return R_XY @ v_Y


def compute_fingertip_pose(T_BA, T_AT) -> np.ndarray:
    """Implementation helper."""
    return compose_pose(T_BA, T_AT)


def compute_fingertip_pos_rel_fixed(T_BT, T_BH) -> np.ndarray:
    """Implementation helper."""
    p_BT, _ = split_pose7(T_BT)
    p_BH, _ = split_pose7(T_BH)
    return p_BT - p_BH


def compute_fingertip_quat_rel_fixed(T_BT, T_BH) -> np.ndarray:
    """Implementation helper."""
    _, q_BT = split_pose7(T_BT)
    _, q_BH = split_pose7(T_BH)
    return quat_mul(quat_inv(q_BT), q_BH)


def shift_twist_from_A_to_T(v_A, w_A, r_AT) -> tuple[np.ndarray, np.ndarray]:
    """Implementation helper."""
    v_A = _as_np_1d(v_A, 3, "v_A")
    w_A = _as_np_1d(w_A, 3, "w_A")
    r_AT = _as_np_1d(r_AT, 3, "r_AT")

    v_T = v_A + np.cross(w_A, r_AT)
    w_T = w_A.copy()
    return v_T, w_T


def pose_position(pose7) -> np.ndarray:
    p, _ = split_pose7(pose7)
    return p


def pose_quat(pose7) -> np.ndarray:
    _, q = split_pose7(pose7)
    return q


def pose_rpy_xyz(pose7) -> np.ndarray:
    _, q = split_pose7(pose7)
    return quat_to_euler_xyz(q)


__all__ = [
    "quat_normalize",
    "quat_conjugate",
    "quat_inv",
    "quat_mul",
    "quat_to_rotmat",
    "rotmat_to_quat",
    "euler_xyz_to_quat",
    "quat_to_euler_xyz",
    "make_pose7",
    "split_pose7",
    "pose7_to_mat",
    "mat_to_pose7",
    "compose_pose",
    "invert_pose",
    "relative_pose",
    "transform_point",
    "rotate_vector",
    "compute_fingertip_pose",
    "compute_fingertip_pos_rel_fixed",
    "compute_fingertip_quat_rel_fixed",
    "shift_twist_from_A_to_T",
    "pose_position",
    "pose_quat",
    "pose_rpy_xyz",
]
