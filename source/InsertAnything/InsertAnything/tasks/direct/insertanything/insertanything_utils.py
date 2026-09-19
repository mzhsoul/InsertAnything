import isaacsim.core.utils.torch as torch_utils
import numpy as np
import torch


def get_keypoint_offsets(num_keypoints, device):
    """Return evenly spaced local keypoint offsets along the local z-axis."""
    keypoint_offsets = torch.zeros((num_keypoints, 3), device=device)
    keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=device) - 0.5
    return keypoint_offsets


def get_deriv_gains(prop_gains, rot_deriv_scale=1.0):
    """Compute critically damped derivative gains from proportional gains."""
    deriv_gains = 2 * torch.sqrt(prop_gains)
    deriv_gains[:, 3:6] /= rot_deriv_scale
    return deriv_gains


def wrap_yaw(angle):
    """Wrap yaw angles above 235 degrees back by one full turn."""
    return torch.where(angle > np.deg2rad(235), angle - 2 * np.pi, angle)


def set_friction(asset, value, env_ids, device="cpu"):
    """Set static and dynamic friction for all shapes in selected environments."""
    del device
    materials = asset.root_physx_view.get_material_properties()

    if isinstance(env_ids, int):
        env_ids_cpu = torch.arange(env_ids, device="cpu")
    else:
        env_ids_cpu = env_ids.to("cpu") if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device="cpu")

    if isinstance(value, torch.Tensor):
        val_cpu = value.to("cpu")
        if val_cpu.dim() == 1:
            val_cpu = val_cpu.unsqueeze(-1)
        materials[env_ids_cpu, :, 0] = val_cpu
        materials[env_ids_cpu, :, 1] = val_cpu
    else:
        materials[env_ids_cpu, :, 0] = value
        materials[env_ids_cpu, :, 1] = value

    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)


def set_body_inertias(robot, num_envs):
    """Add a small diagonal inertia offset for simulation stability."""
    inertias = robot.root_physx_view.get_inertias()
    offset = torch.zeros_like(inertias)
    offset[:, :, [0, 4, 8]] += 0.01
    robot.root_physx_view.set_inertias(inertias + offset, torch.arange(num_envs))


def get_held_base_pos_local(num_envs, device):
    """Return the task reference point in the held asset local frame."""
    held_base_pos_local = torch.zeros((num_envs, 3), device=device)
    return held_base_pos_local


def get_held_base_pose(held_pos, held_quat, num_envs, device):
    """Transform the held asset reference point to the world frame."""
    held_base_pos_local = get_held_base_pos_local(num_envs, device)
    held_base_quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

    held_base_quat, held_base_pos = torch_utils.tf_combine(
        held_quat,
        held_pos,
        held_base_quat_local,
        held_base_pos_local,
    )
    return held_base_pos, held_base_quat


def get_target_held_base_pose(fixed_pos, fixed_quat, cfg_task, num_envs, device):
    """Compute the target pose for the held asset reference point."""
    fixed_success_pos_local = torch.zeros((num_envs, 3), device=device)
    base_height = getattr(cfg_task.fixed_asset_cfg, "base_height", 0.0)

    if getattr(cfg_task, "use_decoupled_reward", False):
        fixed_success_pos_local[:, 2] = base_height
    else:
        fixed_success_pos_local[:, 2] = base_height + 0.02

    fixed_success_quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)
    target_held_base_quat, target_held_base_pos = torch_utils.tf_combine(
        fixed_quat,
        fixed_pos,
        fixed_success_quat_local,
        fixed_success_pos_local,
    )
    return target_held_base_pos, target_held_base_quat


def squashing_fn(x, a, b):
    """Bound an unbounded distance-like value into a smooth reward term."""
    return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))


def collapse_obs_dict(obs_dict, obs_order):
    """Concatenate observation tensors according to the configured order."""
    return torch.cat([obs_dict[obs_name] for obs_name in obs_order], dim=-1)


def compute_orientation_reward(min_yaw_error: torch.Tensor, coef: list):
    """Compute an orientation reward from the closest yaw error."""
    a, b = coef
    return squashing_fn(min_yaw_error, a, b)


def get_closest_symmetry_transform(held_quat, target_quat, symmetry_angles: list):
    """Return the nearest symmetry-equivalent target orientation."""
    _, _, held_yaw = torch_utils.get_euler_xyz(held_quat)
    roll, pitch, target_yaw = torch_utils.get_euler_xyz(target_quat)

    all_errors = []
    all_target_yaws = []
    for sym_angle in symmetry_angles:
        sym_target_yaw = target_yaw + sym_angle
        diff = (held_yaw - sym_target_yaw + np.pi) % (2 * np.pi) - np.pi
        all_errors.append(torch.abs(diff))
        all_target_yaws.append(sym_target_yaw)

    all_errors_tensor = torch.stack(all_errors)
    all_target_yaws_tensor = torch.stack(all_target_yaws)

    min_error_indices = torch.argmin(all_errors_tensor, dim=0)
    min_yaw_error_wrapped = torch.gather(all_errors_tensor, 0, min_error_indices.unsqueeze(0)).squeeze(0)
    closest_target_yaw = torch.gather(all_target_yaws_tensor, 0, min_error_indices.unsqueeze(0)).squeeze(0)

    closest_target_quat = torch_utils.quat_from_euler_xyz(roll, pitch, closest_target_yaw)
    held_quat_inv = torch_utils.quat_conjugate(held_quat)
    relative_quat_to_closest = torch_utils.quat_mul(held_quat_inv, closest_target_quat)

    return closest_target_quat, relative_quat_to_closest, min_yaw_error_wrapped


def canonicalize_quat(q):
    """Choose the quaternion representative with non-negative scalar component."""
    return torch.where(q[:, 0:1] < 0, -q, q)


def compose_single_hole_world_pose(anchor_pos, anchor_quat, hole_local_pos, hole_local_quat):
    """Compose a local hole pose with an anchor pose."""
    world_quat, world_pos = torch_utils.tf_combine(anchor_quat, anchor_pos, hole_local_quat, hole_local_pos)
    return world_pos, world_quat
