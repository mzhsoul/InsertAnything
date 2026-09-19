"""Run InsertAnything closed-loop sim-to-real evaluation on a Franka server.

The runner keeps the deployment path intentionally small:
robot state -> observation -> policy -> action postprocessor -> safety guard
-> pose command. Manual episode labels are entered from stdin.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import yaml
from action_postprocessor import ActionPostprocessor, ActionPostprocessorConfig
from observation_builder import (
    CalibrationData,
    build_observation,
    load_calibration_data,
)
from robot_state_adapter import RobotState, fetch_robot_state
from safety_guard import SafetyGuard, SafetyGuardConfig
from torch_policy import TorchPolicyConfig, TorchRecurrentActorPolicy
from transforms import (
    compose_pose,
    euler_xyz_to_quat,
    invert_pose,
    make_pose7,
    pose_position,
    pose_quat,
    pose_rpy_xyz,
)

DEFAULT_GRASP_LOAD_POSE6 = [
    0.50736036401316785,
    -0.0008526372540086282,
    0.2036872599762553,
    -3.1398522357613947,
    -0.0026717628628585954,
    0.006630397673314414,
]


def _as_np_1d(value: Any, expected_len: int | None = None, name: str = "array") -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if expected_len is not None and arr.shape[0] != expected_len:
        raise ValueError(f"{name} must have length {expected_len}, got shape {arr.shape}")
    return arr


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _fmt_vec(value: Any, digits: int = 5) -> str:
    arr = _as_np_1d(value)
    return "[" + ", ".join(f"{x:.{digits}f}" for x in arr.tolist()) + "]"


def _tensor_to_np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_ready(_tensor_to_np(value))
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def post_json(server_url: str, endpoint: str, payload: dict | None = None, timeout: float = 3.0):
    url = server_url.rstrip("/") + endpoint
    response = requests.post(url, json={} if payload is None else payload, timeout=timeout)
    response.raise_for_status()
    return response


def send_pose6(server_url: str, pose6: np.ndarray, timeout: float = 3.0) -> None:
    pose6 = _as_np_1d(pose6, 6, "pose6")
    post_json(server_url, "/pose", {"arr": pose6.tolist()}, timeout=timeout)


def open_gripper(server_url: str, endpoint: str, timeout: float = 3.0) -> None:
    post_json(server_url, endpoint, {}, timeout=timeout)


def close_gripper(server_url: str, endpoint: str, timeout: float = 3.0) -> None:
    post_json(server_url, endpoint, {}, timeout=timeout)


def pose7_to_pose6_xyzrpy(pose7: np.ndarray) -> np.ndarray:
    pose7 = _as_np_1d(pose7, 7, "pose7")
    return np.concatenate([pose_position(pose7), pose_rpy_xyz(pose7)], axis=0)


def fingertip_target_to_api_target(t_bt_target: np.ndarray, t_at: np.ndarray) -> np.ndarray:
    return compose_pose(t_bt_target, invert_pose(t_at))


def _with_policy_observation_pos_rel_bias(
    calib: CalibrationData,
    pos_rel_bias_m: np.ndarray,
) -> CalibrationData:
    raw = dict(calib.raw) if calib.raw is not None else {}
    policy_obs = dict(raw.get("policy_observation", {}))
    policy_obs["pos_rel_bias_m"] = _as_np_1d(pos_rel_bias_m, 3, "pos_rel_bias_m").tolist()
    raw["policy_observation"] = policy_obs
    return CalibrationData(
        T_AT=calib.T_AT.copy(),
        T_BH=calib.T_BH.copy(),
        T_BPRE=calib.T_BPRE.copy(),
        raw=raw,
    )


def _sample_xy_uniform_annulus(
    rng: np.random.Generator,
    r_inner_m: float,
    r_outer_m: float,
) -> np.ndarray:
    r0 = float(r_inner_m)
    r1 = float(r_outer_m)
    if r0 < 0.0 or r1 < 0.0:
        raise ValueError(f"Observation hole-center error radii must be non-negative, got {r0}, {r1}.")
    if r1 < r0:
        raise ValueError(f"Outer radius must be >= inner radius, got {r0}, {r1}.")
    if r1 == 0.0:
        return np.zeros(2, dtype=np.float64)

    theta = rng.uniform(0.0, 2.0 * np.pi)
    radius = np.sqrt(rng.uniform(r0 * r0, r1 * r1))
    return np.array([radius * np.cos(theta), radius * np.sin(theta)], dtype=np.float64)


def sample_obs_hole_center_error(args: argparse.Namespace, rng: np.random.Generator) -> np.ndarray:
    radii = _as_np_1d(args.obs_hole_center_error_annulus_m, 2, "obs_hole_center_error_annulus_m")
    xy = _sample_xy_uniform_annulus(rng, float(radii[0]), float(radii[1]))
    return np.array([float(xy[0]), float(xy[1]), 0.0], dtype=np.float64)


def sample_episode_start_pose(
    *,
    rng: np.random.Generator,
    base_pose7: np.ndarray,
    random_start: bool,
    rand_x_range: float,
    rand_y_range: float,
    rand_z_up_range: float,
    randomize_start_yaw: bool,
    yaw_enable: bool,
    yaw_abs_limit_deg: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    base_pose7 = _as_np_1d(base_pose7, 7, "base_pose7")
    pos = pose_position(base_pose7).copy()
    quat = pose_quat(base_pose7).copy()

    offset = np.zeros(3, dtype=np.float64)
    if random_start:
        offset[0] = rng.uniform(-float(rand_x_range), float(rand_x_range))
        offset[1] = rng.uniform(-float(rand_y_range), float(rand_y_range))
        offset[2] = rng.uniform(0.0, float(rand_z_up_range))

    yaw_offset = 0.0
    if random_start and randomize_start_yaw and yaw_enable:
        yaw_offset = float(rng.uniform(-np.deg2rad(yaw_abs_limit_deg), np.deg2rad(yaw_abs_limit_deg)))
        roll, pitch, yaw = pose_rpy_xyz(base_pose7)
        quat = euler_xyz_to_quat(float(roll), float(pitch), float(yaw + yaw_offset))

    return make_pose7(pos + offset, quat), offset, yaw_offset


class InputManager:
    """Non-blocking stdin reader for manual experiment commands."""

    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        while True:
            try:
                line = input()
            except EOFError:
                return
            self._queue.put(line.strip().lower())

    def drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def get_nowait(self) -> str | None:
        try:
            value = self._queue.get_nowait()
        except queue.Empty:
            return None
        return value or None

    def wait_for_choice(self, *, valid: set[str], prompt: str) -> str:
        valid = {x.lower() for x in valid}
        print(prompt)
        while True:
            value = self.get_nowait()
            if value is None:
                time.sleep(0.05)
                continue
            if value in valid:
                return value
            print(f"[INPUT] Ignore unknown command: {value}. Valid commands: {sorted(valid)}")


class ExperimentLogger:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.root_dir / "steps.jsonl"
        self.episodes_path = self.root_dir / "episodes.csv"
        self._episode_fieldnames: list[str] | None = None

    def log_step(self, record: dict[str, Any]) -> None:
        with self.steps_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_ready(record), ensure_ascii=False) + "\n")

    def log_episode(self, record: dict[str, Any]) -> None:
        record = _json_ready(record)
        if self._episode_fieldnames is None:
            self._episode_fieldnames = list(record.keys())
            write_header = True
        else:
            write_header = False

        with self.episodes_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._episode_fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(record)


def _policy_config_from_yaml(cfg: dict[str, Any], args: argparse.Namespace) -> TorchPolicyConfig:
    policy_obs = cfg.get("policy_observation", {})
    policy_structure = cfg.get("policy_structure", {})
    obs_order = policy_obs.get("obs_order")
    if obs_order is None:
        raise KeyError("policy_observation.obs_order is required.")

    mlp_units = tuple(int(x) for x in policy_structure.get("mlp_units", [512, 128, 64]))
    return TorchPolicyConfig.from_basic_spec(
        obs_order=[str(x) for x in obs_order],
        append_prev_actions=bool(policy_obs.get("append_prev_actions", True)),
        action_dim=int(policy_structure.get("action_dim", 6)),
        hidden_size=int(policy_structure.get("hidden_size", 1024)),
        num_layers=int(policy_structure.get("num_layers", 2)),
        mlp_units=mlp_units,
        obs_norm_eps=float(args.obs_norm_eps),
        obs_clip=policy_structure.get("obs_clip"),
        device=str(args.device),
        dtype=str(policy_structure.get("dtype", "float32")),
    )


def _postprocessor_config_from_yaml(cfg: dict[str, Any], args: argparse.Namespace) -> ActionPostprocessorConfig:
    node = dict(cfg.get("action_postprocessor", {}))
    return ActionPostprocessorConfig(
        ema_factor=float(args.ema_factor),
        pos_action_threshold=tuple(float(x) for x in node.get("pos_action_threshold", [0.02, 0.02, 0.02])),
        rot_action_threshold=tuple(float(x) for x in node.get("rot_action_threshold", [0.097, 0.097, 0.097])),
        pos_action_bounds=tuple(float(x) for x in node.get("pos_action_bounds", [0.05, 0.05, 0.05])),
        raw_action_clip=float(args.raw_action_clip),
        yaw_enable=bool(args.yaw_enable),
        yaw_abs_limit_deg=float(args.yaw_abs_limit_deg),
        yaw_step_limit_deg=float(args.yaw_step_limit_deg),
    )


def _safety_guard_config_from_args(args: argparse.Namespace) -> SafetyGuardConfig:
    return SafetyGuardConfig(
        enable_hole_box_clip=True,
        hole_box_lower=(
            -float(args.hole_box_x),
            -float(args.hole_box_y),
            float(args.hole_box_z_low),
        ),
        hole_box_upper=(
            +float(args.hole_box_x),
            +float(args.hole_box_y),
            float(args.hole_box_z_high),
        ),
        enable_step_clip=True,
        step_max_xyz=(
            float(args.step_max_x),
            float(args.step_max_y),
            float(args.step_max_z),
        ),
        yaw_abs_limit_deg=float(args.yaw_abs_limit_deg),
        yaw_step_limit_deg=float(args.yaw_step_limit_deg),
        enforce_ref_roll_pitch=True,
    )


def _make_contact_force_provider(source: str):
    if source == "none":
        return None
    if source != "robot_state.force_K_debug":
        raise ValueError(f"Unsupported contact-force source: {source}")

    def provider(robot_state: RobotState, _calib: CalibrationData) -> np.ndarray:
        raw = robot_state.raw or {}
        if "force_K_debug" in raw:
            return _as_np_1d(raw["force_K_debug"], 3, "force_K_debug")
        raise RuntimeError(
            'contact_force_source="robot_state.force_K_debug" was requested, '
            "but the robot state does not contain force_K_debug."
        )

    return provider


def _make_step_record(
    *,
    episode_idx: int,
    step_idx: int,
    robot_state: RobotState,
    obs_comp,
    policy_out: dict[str, torch.Tensor],
    action_raw: np.ndarray,
    post_out,
    guard_out,
    target_pose6_cmd: np.ndarray,
    policy_period_s: float,
    step_elapsed_s: float,
) -> dict[str, Any]:
    return {
        "episode_idx": int(episode_idx),
        "step_idx": int(step_idx),
        "wall_time": datetime.now().isoformat(timespec="milliseconds"),
        "policy_period_s": float(policy_period_s),
        "step_elapsed_s": float(step_elapsed_s),
        "robot_pose_BA": robot_state.pose_BA,
        "robot_ee_pose6": robot_state.ee_pose6,
        "robot_q": robot_state.q,
        "robot_dq": robot_state.dq,
        "robot_vel_A": robot_state.vel_A,
        "robot_force_K": None if robot_state.force_K is None else robot_state.force_K,
        "obs": obs_comp.obs,
        "obs_order": obs_comp.obs_order_with_prev_actions,
        "obs_components": obs_comp.obs_components_by_name,
        "pos_rel_bias_m": obs_comp.pos_rel_bias_m,
        "action_raw": action_raw,
        "mu": _tensor_to_np(policy_out["mu"].squeeze(0)),
        "sigma": _tensor_to_np(policy_out["sigma"].squeeze(0)),
        "value": _tensor_to_np(policy_out["value"].squeeze(0)),
        "obs_norm": _tensor_to_np(policy_out["obs_norm"].squeeze(0)),
        "post_action_ema": post_out.action_ema,
        "post_target_T_BT": post_out.T_BT_target,
        "guard_target_T_BT": guard_out.T_BT_safe,
        "target_pose6_cmd": target_pose6_cmd,
        "was_raw_clipped": bool(post_out.was_raw_clipped),
        "was_pos_bound_clipped": bool(post_out.was_pos_bound_clipped),
        "was_hole_box_clipped": bool(guard_out.was_hole_box_clipped),
        "was_step_clipped": bool(guard_out.was_step_clipped),
        "was_yaw_abs_clipped": bool(guard_out.was_yaw_abs_clipped),
        "was_yaw_step_clipped": bool(guard_out.was_yaw_step_clipped),
    }


def _make_episode_record(
    *,
    episode_idx: int,
    result: str,
    steps: int,
    duration_s: float,
    clip_stats: dict[str, int],
    obs_hole_center_error_m: np.ndarray,
) -> dict[str, Any]:
    return {
        "episode_idx": int(episode_idx),
        "result": str(result),
        "steps": int(steps),
        "duration_s": float(duration_s),
        "obs_hole_center_error_dx_m": float(obs_hole_center_error_m[0]),
        "obs_hole_center_error_dy_m": float(obs_hole_center_error_m[1]),
        "raw_clip_count": int(clip_stats.get("raw", 0)),
        "pos_bound_clip_count": int(clip_stats.get("pos_bound", 0)),
        "hole_box_clip_count": int(clip_stats.get("hole_box", 0)),
        "step_clip_count": int(clip_stats.get("step", 0)),
        "yaw_abs_clip_count": int(clip_stats.get("yaw_abs", 0)),
        "yaw_step_clip_count": int(clip_stats.get("yaw_step", 0)),
    }


def _update_clip_stats(stats: dict[str, int], post_out, guard_out) -> dict[str, int]:
    stats["raw"] = stats.get("raw", 0) + int(post_out.was_raw_clipped)
    stats["pos_bound"] = stats.get("pos_bound", 0) + int(post_out.was_pos_bound_clipped)
    stats["hole_box"] = stats.get("hole_box", 0) + int(guard_out.was_hole_box_clipped)
    stats["step"] = stats.get("step", 0) + int(guard_out.was_step_clipped)
    stats["yaw_abs"] = stats.get("yaw_abs", 0) + int(guard_out.was_yaw_abs_clipped)
    stats["yaw_step"] = stats.get("yaw_step", 0) + int(guard_out.was_yaw_step_clipped)
    return stats


def run_regrasp_flow(args: argparse.Namespace, input_mgr: InputManager) -> None:
    print("[REGRASP] Move to grasp load pose.")
    send_pose6(args.server_url, np.asarray(args.grasp_load_pose6, dtype=np.float64))
    time.sleep(args.move_wait)

    print("[REGRASP] Open gripper.")
    open_gripper(args.server_url, args.open_gripper_endpoint)
    input_mgr.wait_for_choice(
        valid={"g", "grasp", "ready"},
        prompt="[REGRASP] Place the peg, then input g/ready.",
    )

    print("[REGRASP] Close gripper.")
    close_gripper(args.server_url, args.close_gripper_endpoint)
    time.sleep(args.move_wait)
    input_mgr.wait_for_choice(valid={"y", "yes", "ok"}, prompt="[REGRASP] Confirm the grasp with y/ok.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InsertAnything sim-to-real closed-loop runner.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Calibration and deployment YAML path.",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="RL-Games checkpoint path.")
    parser.add_argument("--server-url", type=str, default="http://172.16.0.1:5000")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--policy-period-s", type=float, default=8.0 / 120.0)
    parser.add_argument("--move-wait", type=float, default=0.2)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--rand-x-range", type=float, default=0.05)
    parser.add_argument("--rand-y-range", type=float, default=0.05)
    parser.add_argument("--rand-z-up-range", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize-start-yaw", action="store_true")
    parser.add_argument("--grasp-load-pose6", type=float, nargs=6, default=DEFAULT_GRASP_LOAD_POSE6)
    parser.add_argument("--open-gripper-endpoint", type=str, default="/open_gripper")
    parser.add_argument("--close-gripper-endpoint", type=str, default="/close_gripper")
    parser.add_argument("--obs-norm-eps", type=float, default=1e-5)
    parser.add_argument("--ema-factor", type=float, default=0.2)
    parser.add_argument("--raw-action-clip", type=float, default=1.5)
    parser.add_argument("--yaw-enable", action="store_true")
    parser.add_argument("--yaw-abs-limit-deg", type=float, default=15.0)
    parser.add_argument("--yaw-step-limit-deg", type=float, default=2.0)
    parser.add_argument("--step-max-x", type=float, default=0.005)
    parser.add_argument("--step-max-y", type=float, default=0.005)
    parser.add_argument("--step-max-z", type=float, default=0.004)
    parser.add_argument("--hole-box-x", type=float, default=0.045)
    parser.add_argument("--hole-box-y", type=float, default=0.045)
    parser.add_argument("--hole-box-z-low", type=float, default=0.0)
    parser.add_argument("--hole-box-z-high", type=float, default=0.05)
    parser.add_argument("--log-dir", type=str, default="sim2real_logs")
    parser.add_argument("--assume-grasped", action="store_true")
    parser.add_argument("--return-to-load-pose-at-end", action="store_true")
    parser.add_argument("--open-gripper-at-end", action="store_true")
    parser.add_argument(
        "--contact-force-source",
        type=str,
        choices=["none", "robot_state.force_K_debug"],
        default="none",
    )
    parser.add_argument(
        "--obs-hole-center-error-annulus-m",
        type=float,
        nargs=2,
        metavar=("R_IN", "R_OUT"),
        default=[0.0, 0.0],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    rng = np.random.default_rng(int(args.seed))
    calib = load_calibration_data(config_path)
    policy_cfg = _policy_config_from_yaml(raw_cfg, args)
    post_cfg = _postprocessor_config_from_yaml(raw_cfg, args)
    guard_cfg = _safety_guard_config_from_args(args)

    print(f"[INFO] Config: {config_path}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Policy config: {policy_cfg.to_serializable_dict()}")
    print(f"[INFO] Postprocessor config: {asdict(post_cfg)}")
    print(f"[INFO] Safety guard config: {asdict(guard_cfg)}")

    policy = TorchRecurrentActorPolicy.from_checkpoint(checkpoint_path, policy_cfg)
    policy.eval()
    post = ActionPostprocessor(post_cfg)
    guard = SafetyGuard(guard_cfg)
    contact_force_provider = _make_contact_force_provider(args.contact_force_source)

    log_root = Path(args.log_dir) / f"run_{_now_tag()}"
    logger = ExperimentLogger(log_root)
    with (log_root / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            _json_ready({
                "args": vars(args),
                "policy_config": policy_cfg.to_serializable_dict(),
                "postprocessor_config": asdict(post_cfg),
                "safety_guard_config": asdict(guard_cfg),
                "calibration": raw_cfg,
            }),
            f,
            sort_keys=False,
            allow_unicode=False,
        )

    input_mgr = InputManager()
    input_mgr.drain()

    if not args.assume_grasped:
        run_regrasp_flow(args, input_mgr)

    success_count = 0
    fail_count = 0
    abort_count = 0

    try:
        for episode_idx in range(1, int(args.num_episodes) + 1):
            print("\n" + "=" * 100)
            print(f"EPISODE {episode_idx}/{args.num_episodes}")
            print("=" * 100)

            obs_hole_center_error_m = sample_obs_hole_center_error(args, rng)
            print(
                f"[EP {episode_idx}] Observation hole-center error xy = ({_fmt_vec(obs_hole_center_error_m[:2], 6)}) m"
            )

            start_pose, start_offset, start_yaw_offset = sample_episode_start_pose(
                rng=rng,
                base_pose7=calib.T_BPRE,
                random_start=bool(args.random_start),
                rand_x_range=float(args.rand_x_range),
                rand_y_range=float(args.rand_y_range),
                rand_z_up_range=float(args.rand_z_up_range),
                randomize_start_yaw=bool(args.randomize_start_yaw),
                yaw_enable=bool(args.yaw_enable),
                yaw_abs_limit_deg=float(args.yaw_abs_limit_deg),
            )
            print(
                f"[EP {episode_idx}] Start offset xyz={_fmt_vec(start_offset, 5)}, "
                f"yaw_offset={np.rad2deg(start_yaw_offset):+.2f} deg"
            )
            send_pose6(
                args.server_url,
                pose7_to_pose6_xyzrpy(fingertip_target_to_api_target(start_pose, calib.T_AT)),
            )
            time.sleep(args.move_wait)

            input_mgr.wait_for_choice(
                valid={"start", "go", "g"},
                prompt=f"[EP {episode_idx}] Robot is at the start pose. Input start/go/g to run closed loop.",
            )
            input_mgr.drain()

            policy_state = policy.get_initial_state(batch_size=1, device=args.device)
            post_state = post.get_initial_state()
            prev_actions = np.zeros(6, dtype=np.float64)
            episode_start_time = time.time()
            episode_result: str | None = None
            episode_steps = 0
            clip_stats: dict[str, int] = {}

            for step_idx in range(1, int(args.max_steps) + 1):
                step_t0 = time.monotonic()
                robot_state = fetch_robot_state(args.server_url)

                obs_calib = _with_policy_observation_pos_rel_bias(
                    calib,
                    -obs_hole_center_error_m,
                )
                obs_comp = build_observation(
                    robot_state=robot_state,
                    calib=obs_calib,
                    prev_actions=prev_actions,
                    contact_force_provider=contact_force_provider,
                )

                obs_t = torch.as_tensor(obs_comp.obs, dtype=policy_cfg.dtype, device=args.device).unsqueeze(0)
                with torch.inference_mode():
                    policy_out = policy.forward(
                        obs=obs_t,
                        state=policy_state,
                        reset_mask=None,
                        return_aux=True,
                    )

                policy_state = policy_out["new_state"]
                action_raw = _tensor_to_np(policy_out["action"].squeeze(0))
                post_out, post_state = post.process(
                    current_T_BT=obs_comp.T_BT,
                    fixed_pos_frame_pose=calib.T_BH,
                    orientation_ref_pose=calib.T_BPRE,
                    action_raw=action_raw,
                    state=post_state,
                )
                guard_out = guard.apply(
                    current_T_BT=obs_comp.T_BT,
                    candidate_T_BT=post_out.T_BT_target,
                    T_BH=calib.T_BH,
                    T_BPRE=calib.T_BPRE,
                    orientation_ref_pose=calib.T_BPRE,
                )

                target_pose6 = pose7_to_pose6_xyzrpy(fingertip_target_to_api_target(guard_out.T_BT_safe, calib.T_AT))
                send_pose6(args.server_url, target_pose6)

                episode_steps = step_idx
                clip_stats = _update_clip_stats(clip_stats, post_out, guard_out)
                logger.log_step(
                    _make_step_record(
                        episode_idx=episode_idx,
                        step_idx=step_idx,
                        robot_state=robot_state,
                        obs_comp=obs_comp,
                        policy_out=policy_out,
                        action_raw=action_raw,
                        post_out=post_out,
                        guard_out=guard_out,
                        target_pose6_cmd=target_pose6,
                        policy_period_s=float(args.policy_period_s),
                        step_elapsed_s=time.monotonic() - step_t0,
                    )
                )

                p_rel = obs_comp.fingertip_pos_rel_fixed
                mu = _tensor_to_np(policy_out["mu"].squeeze(0))
                print(
                    f"[EP {episode_idx:02d} STEP {step_idx:03d}] "
                    f"p_rel=({p_rel[0]:+.4f},{p_rel[1]:+.4f},{p_rel[2]:+.4f}) "
                    f"mu=({mu[0]:+.3f},{mu[1]:+.3f},{mu[2]:+.3f}) "
                    f"target=({guard_out.p_target_safe[0]:+.4f},"
                    f"{guard_out.p_target_safe[1]:+.4f},{guard_out.p_target_safe[2]:+.4f}) "
                    f"clip(step={int(guard_out.was_step_clipped)}, box={int(guard_out.was_hole_box_clipped)})"
                )

                prev_actions = post_out.action_ema.copy()

                deadline = step_t0 + float(args.policy_period_s)
                user_cmd = None
                while time.monotonic() < deadline:
                    cmd = input_mgr.get_nowait()
                    if cmd in {"s", "success"}:
                        user_cmd = "success"
                        break
                    if cmd in {"f", "fail"}:
                        user_cmd = "fail"
                        break
                    if cmd in {"a", "abort"}:
                        user_cmd = "abort"
                        break
                    if cmd in {"q", "quit"}:
                        user_cmd = "quit"
                        break
                    if cmd is not None:
                        print(f"[EP {episode_idx}] Ignore unknown command during episode: {cmd}")
                    time.sleep(0.01)

                if user_cmd == "success":
                    episode_result = "success"
                    break
                if user_cmd == "fail":
                    episode_result = "fail"
                    break
                if user_cmd in {"abort", "quit"}:
                    episode_result = "abort"
                    break

            if episode_result is None:
                choice = input_mgr.wait_for_choice(
                    valid={"s", "success", "f", "fail", "a", "abort", "q", "quit"},
                    prompt=f"[EP {episode_idx}] Reached max_steps={args.max_steps}. Input s/f/a/q.",
                )
                if choice in {"s", "success"}:
                    episode_result = "success"
                elif choice in {"f", "fail"}:
                    episode_result = "fail"
                else:
                    episode_result = "abort"

            print(f"[EP {episode_idx}] Retreat to T_BPRE.")
            send_pose6(
                args.server_url,
                pose7_to_pose6_xyzrpy(fingertip_target_to_api_target(calib.T_BPRE, calib.T_AT)),
            )
            time.sleep(args.move_wait)

            if episode_result == "success":
                success_count += 1
            elif episode_result == "fail":
                fail_count += 1
            else:
                abort_count += 1

            logger.log_episode(
                _make_episode_record(
                    episode_idx=episode_idx,
                    result=episode_result,
                    steps=episode_steps,
                    duration_s=time.time() - episode_start_time,
                    clip_stats=clip_stats,
                    obs_hole_center_error_m=obs_hole_center_error_m,
                )
            )
            print(f"[EP {episode_idx}] Result={episode_result}, steps={episode_steps}")
            print(f"[STATS] success={success_count}, fail={fail_count}, abort={abort_count}")

            next_choice = input_mgr.wait_for_choice(
                valid={"c", "continue", "r", "regrasp", "q", "quit"},
                prompt="[NEXT] Input c/continue, r/regrasp, or q/quit.",
            )
            if next_choice in {"r", "regrasp"}:
                run_regrasp_flow(args, input_mgr)
            elif next_choice in {"q", "quit"}:
                print("[INFO] User requested experiment shutdown.")
                break
    finally:
        if args.return_to_load_pose_at_end:
            print("[END] Return to grasp load pose.")
            send_pose6(args.server_url, np.asarray(args.grasp_load_pose6, dtype=np.float64))
            time.sleep(args.move_wait)
        if args.open_gripper_at_end:
            print("[END] Open gripper.")
            open_gripper(args.server_url, args.open_gripper_endpoint)
            time.sleep(1.0)

    print("=" * 100)
    print("Experiment finished.")
    print(f"Logs saved to: {log_root}")
    print(f"Final stats: success={success_count}, fail={fail_count}, abort={abort_count}")
    print("=" * 100)


if __name__ == "__main__":
    main()
