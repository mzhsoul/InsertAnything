from __future__ import annotations

import os

import carb
import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils
import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat

from . import insertanything_control, insertanything_utils
from .insertanything_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, InsertAnythingEnvCfg
from .tactile_datalogger import CSVDataLogger


class InsertAnythingEnv(DirectRLEnv):
    cfg: InsertAnythingEnvCfg

    def _enable_hole_view_camera(self) -> bool:
        return self.cfg.evaluation_mode and self.num_envs == 1 and getattr(self.cfg_task, "show_visual_markers", False)

    def _get_hole_view_camera_pose(self, target_pos=None):
        if target_pos is None:
            target = np.array(self.cfg_task.fixed_asset.init_state.pos, dtype=np.float64)
        else:
            target = np.asarray(target_pos, dtype=np.float64)

        target = target.copy()
        target[2] += 0.04
        eye = target + np.array([0.32, -0.46, 0.30], dtype=np.float64)
        return eye.tolist(), target.tolist()

    def _set_hole_view_camera(self, target_pos=None):
        if not self._enable_hole_view_camera() or getattr(self, "hole_view_camera_path", None) is None:
            return

        eye, target = self._get_hole_view_camera_pose(target_pos)
        try:
            self.sim.set_camera_view(eye=eye, target=target, camera_prim_path=self.hole_view_camera_path)
        except Exception as e:
            print(f"[WARN] Failed to set hole view camera: {e}")

    def _setup_scene(self):
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(),
            translation=(0.0, 0.0, -1.05),
        )

        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table",
            cfg,
            translation=(0.55, 0.0, 0.0),
            orientation=(0.70711, 0.0, 0.0, 0.70711),
        )

        self._robot = Articulation(self.cfg.robot)
        self._held_asset = Articulation(self.cfg_task.held_asset)
        self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        self.scene.articulations["fixed_asset"] = self._fixed_asset

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        self.scene.articulations["held_asset"] = self._held_asset

        self.hole_view_camera_path = None
        if self._enable_hole_view_camera():
            self.hole_view_camera_path = "/World/HoleViewCamera"
            try:
                eye, target = self._get_hole_view_camera_pose()
                camera_cfg = sim_utils.PinholeCameraCfg(
                    focal_length=28.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.01, 10.0),
                    lock_camera=False,
                )
                camera_cfg.func(self.hole_view_camera_path, camera_cfg, translation=tuple(eye))
                self._set_hole_view_camera()
                print(f"[INFO] Created hole view camera: {self.hole_view_camera_path}")
            except Exception as e:
                print(f"[WARN] Failed to create hole view camera: {e}")
                self.hole_view_camera_path = None

        if self.cfg.evaluation_mode and self.num_envs == 1 and getattr(self.cfg_task, "show_visual_markers", False):
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/TargetMarkers",
                markers={
                    "true_hole": sim_utils.SphereCfg(
                        radius=0.0005,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                    "perceived_hole": sim_utils.SphereCfg(
                        radius=0.0005,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    ),
                    "fingertip": sim_utils.SphereCfg(
                        radius=0.01,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                    ),
                },
            )
            self.target_markers = VisualizationMarkers(marker_cfg)
        else:
            self.target_markers = None

        contact_cfg = getattr(self.cfg_task, "contact_force", {})
        if contact_cfg.get("enabled", False):
            self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
            self.scene.sensors["contact_sensor"] = self.contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _set_default_dynamics_parameters(self):
        self.default_gains = torch.tensor(
            self.cfg.ctrl.default_task_prop_gains,
            dtype=torch.float32,
            device=self.device,
        ).repeat((self.num_envs, 1))

        self.pos_threshold = torch.tensor(
            self.cfg.ctrl.pos_action_threshold, dtype=torch.float32, device=self.device
        ).repeat((self.num_envs, 1))

        self.rot_threshold = torch.tensor(
            self.cfg.ctrl.rot_action_threshold, dtype=torch.float32, device=self.device
        ).repeat((self.num_envs, 1))

        insertanything_utils.set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction, self.scene.num_envs)
        insertanything_utils.set_friction(self._robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs)
        insertanything_utils.set_friction(
            self._fixed_asset,
            self.cfg_task.fixed_asset_cfg.friction,
            self.scene.num_envs,
        )

    def _init_tensors(self):
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ema_factor = self.cfg.ctrl.ema_factor
        self.dead_zone_thresholds = torch.zeros((self.num_envs, 6), device=self.device)
        self.dr_frictions = torch.zeros((self.num_envs, 1), device=self.device)

        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        self.init_fingertip_quat_obs_noise = torch.zeros((self.num_envs, 4), device=self.device)
        self.init_fingertip_quat_obs_noise[:, 0] = 1.0

        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        self.last_update_timestamp = 0.0
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        self.contact_force_local = torch.zeros((self.num_envs, 3), device=self.device)
        self.smoothed_contact_force = torch.zeros((self.num_envs, 3), device=self.device)
        self.contact_force_tare = torch.zeros((self.num_envs, 3), device=self.device)
        self.contact_force_input_mask = torch.ones((self.num_envs, 1), device=self.device)
        self.contact_force_input_scale = torch.ones((self.num_envs, 1), device=self.device)

        self.target_hole_local_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hole_local_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.target_hole_local_quat[:, 0] = 1.0

        self.correction_field_mode = False
        self.correction_field_xy_offsets = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        self.correction_field_obs_bias = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.correction_field_hand_z_offset = 0.0

    def configure_correction_field_eval(
        self,
        xy_offsets,
        obs_bias,
        hand_z_offset: float = 0.0,
    ):
        xy_offsets = torch.as_tensor(xy_offsets, dtype=torch.float32, device=self.device)
        if xy_offsets.shape != (self.num_envs, 2):
            raise ValueError(f"xy_offsets must have shape ({self.num_envs}, 2), got {tuple(xy_offsets.shape)}")

        obs_bias = torch.as_tensor(obs_bias, dtype=torch.float32, device=self.device).reshape(1, 3)
        self.correction_field_mode = True
        self.correction_field_xy_offsets[:] = xy_offsets
        self.correction_field_obs_bias[:] = obs_bias.repeat(self.num_envs, 1)
        self.correction_field_hand_z_offset = float(hand_z_offset)

    def get_correction_field_snapshot(self):
        held_base_pos, _ = insertanything_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.num_envs, self.device
        )
        return {
            "true_hole_pos": self.fixed_pos_obs_frame.detach().clone(),
            "obs_hole_pos": (self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise).detach().clone(),
            "peg_base_pos": held_base_pos.detach().clone(),
            "fingertip_pos": self.fingertip_midpoint_pos.detach().clone(),
            "contact_force": self.contact_force_local.detach().clone(),
            "success": self._get_curr_successes(self.cfg_task.success_threshold).detach().clone(),
            "engaged": self._get_curr_successes(self.cfg_task.engage_threshold).detach().clone(),
            "engaged_half": self._get_curr_successes(self.cfg_task.engage_half_threshold).detach().clone(),
        }

    def __init__(self, cfg: InsertAnythingEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg_task = cfg.task
        self.obs_order = list(self.cfg_task.obs_order)
        cfg.state_order = list(cfg.state_order)
        self._obs_dim_cfg = dict(OBS_DIM_CFG)
        self._state_dim_cfg = dict(STATE_DIM_CFG)

        contact_cfg = getattr(self.cfg_task, "contact_force", {})

        force_enabled = contact_cfg.get("enabled", False)
        use_as_obs = contact_cfg.get("use_as_obs", False)
        use_as_state = contact_cfg.get("use_as_state", use_as_obs)

        if force_enabled:
            if use_as_obs and "contact_force" not in self.obs_order:
                self.obs_order.append("contact_force")
            if use_as_state and "contact_force" not in cfg.state_order:
                insert_idx = cfg.state_order.index("joint_pos")
                cfg.state_order.insert(insert_idx, "contact_force")

        if getattr(self.cfg_task, "dr_randomize_dynamics", False):
            if "dr_friction" not in cfg.state_order:
                cfg.state_order.append("dr_friction")
            if "dr_dead_zone" not in cfg.state_order:
                cfg.state_order.append("dr_dead_zone")
            if "pos_threshold" not in cfg.state_order:
                cfg.state_order.append("pos_threshold")
            if "rot_threshold" not in cfg.state_order:
                cfg.state_order.append("rot_threshold")
            if "ema_factor" not in cfg.state_order:
                cfg.state_order.append("ema_factor")
            if "task_prop_gains" not in cfg.state_order:
                cfg.state_order.append("task_prop_gains")

        cfg.observation_space = sum([self._obs_dim_cfg[obs] for obs in self.obs_order])
        cfg.state_space = sum([self._state_dim_cfg[state] for state in cfg.state_order])
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space

        super().__init__(cfg, render_mode, **kwargs)
        insertanything_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()

        if self.cfg.evaluation_mode:
            self.eval_ep_count = 0

        self.contact_force_logger = None
        if (
            self.cfg.evaluation_mode
            and self.cfg_task.contact_force.get("enabled", False)
            and getattr(self.cfg_task, "contact_force", {}).get("log_contact_force", False)
            and self.cfg.scene.num_envs == 1
        ):
            log_directory = os.path.join(os.path.dirname(__file__), "contact_force_logs")
            self.contact_force_logger = CSVDataLogger(task_name=self.cfg_task.name, log_dir=log_directory)

    def _compute_intermediate_values(self):
        current_sim_time = self._robot._data._sim_timestamp
        actual_dt = current_sim_time - self.last_update_timestamp
        board_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        board_quat = self._fixed_asset.data.root_quat_w

        self.fixed_quat, self.fixed_pos = torch_utils.tf_combine(
            board_quat,
            board_pos,
            self.target_hole_local_quat,
            self.target_hole_local_pos,
        )

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        if actual_dt > 1e-5:
            self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / actual_dt
            self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

            rot_diff_quat = torch_utils.quat_mul(
                self.fingertip_midpoint_quat,
                torch_utils.quat_conjugate(self.prev_fingertip_quat),
            )
            rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
            rot_diff_aa = axis_angle_from_quat(rot_diff_quat)

            self.ee_angvel_fd = rot_diff_aa / actual_dt
            self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

            joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
            self.joint_vel_fd = joint_diff / actual_dt
            self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

            self.last_update_timestamp = current_sim_time

        contact_cfg = getattr(self.cfg_task, "contact_force", {})
        if contact_cfg.get("enabled", False):
            contact_sensor = self.scene.sensors["contact_sensor"]
            net_forces_w = contact_sensor.data.net_forces_w

            total_force_w = torch.sum(net_forces_w, dim=1)

            q_inv = torch_utils.quat_conjugate(self.held_quat)
            total_force_local = torch_utils.quat_apply(q_inv, total_force_w)

            force_source = contact_cfg.get("force_source", "peg_hole")
            if force_source == "gripper_peg":
                is_first_step = self.episode_length_buf == 0
                if is_first_step.any():
                    self.contact_force_tare[is_first_step] = total_force_local[is_first_step]

                external_force_local = total_force_local - self.contact_force_tare
            else:
                external_force_local = total_force_local

            alpha = contact_cfg.get("ema_alpha", 0.25)
            self.smoothed_contact_force = alpha * external_force_local + (1 - alpha) * self.smoothed_contact_force

            self.contact_force_local = self.smoothed_contact_force.clone()
        else:
            self.contact_force_local.zero_()
            self.smoothed_contact_force.zero_()

    def _reset_contact_force_input_randomization(self, env_ids):
        self.contact_force_input_mask[env_ids] = 1.0
        self.contact_force_input_scale[env_ids] = 1.0

        contact_cfg = getattr(self.cfg_task, "contact_force", {})
        if (
            self.cfg.evaluation_mode
            or not contact_cfg.get("enabled", False)
            or not (
                contact_cfg.get("use_as_obs", False)
                or contact_cfg.get("use_as_state", contact_cfg.get("use_as_obs", False))
            )
        ):
            return

        dropout_prob = float(contact_cfg.get("dropout_prob", 0.0))
        dropout_prob = max(0.0, min(1.0, dropout_prob))
        if dropout_prob > 0.0:
            keep = (torch.rand((len(env_ids), 1), device=self.device) >= dropout_prob).float()
            self.contact_force_input_mask[env_ids] = keep

        scale_range = contact_cfg.get("scale_range", [1.0, 1.0])
        if scale_range is None or len(scale_range) != 2:
            scale_min, scale_max = 1.0, 1.0
        else:
            scale_min, scale_max = float(scale_range[0]), float(scale_range[1])
        if scale_max < scale_min:
            scale_min, scale_max = scale_max, scale_min
        if scale_min != 1.0 or scale_max != 1.0:
            rand = torch.rand((len(env_ids), 1), device=self.device)
            self.contact_force_input_scale[env_ids] = scale_min + (scale_max - scale_min) * rand

    def _process_contact_force_input(self, contact_force):
        contact_cfg = getattr(self.cfg_task, "contact_force", {})
        input_scale = float(contact_cfg.get("input_scale", 1.0))
        return contact_force * self.contact_force_input_mask * self.contact_force_input_scale * input_scale

    def _get_insertanything_obs_state_dict(self):
        clean_fingertip_pos = self.fingertip_midpoint_pos
        clean_fingertip_quat = self.fingertip_midpoint_quat
        clean_ee_linvel = self.ee_linvel_fd
        clean_ee_angvel = self.ee_angvel_fd

        clean_contact_force = getattr(
            self,
            "contact_force_local",
            torch.zeros((self.num_envs, 3), device=self.device),
        )

        base_perceived_hole_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        base_perceived_fingertip_quat = torch_utils.quat_mul(self.init_fingertip_quat_obs_noise, clean_fingertip_quat)

        if getattr(self, "target_markers", None) is not None:
            true_pos_w = self.fixed_pos_obs_frame + self.scene.env_origins
            perceived_pos_w = base_perceived_hole_pos + self.scene.env_origins

            fingertip_pos_w = clean_fingertip_pos + self.scene.env_origins

            marker_locations = torch.cat([true_pos_w, perceived_pos_w, fingertip_pos_w], dim=0)

            marker_orientations = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(3, 1)

            marker_indices = torch.tensor([0, 1, 2], dtype=torch.int32, device=self.device)

            self.target_markers.visualize(marker_locations, marker_orientations, marker_indices=marker_indices)

        if self.cfg.obs_rand.use_all_noise:
            noise = torch.randn_like(self.fixed_pos_obs_frame)
            noise_std = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, device=self.device)
            noisy_fixed_pos = base_perceived_hole_pos + (noise @ torch.diag(noise_std))

            noise = torch.randn_like(clean_fingertip_pos)
            noise_std = torch.tensor(self.cfg.obs_rand.fingertip_pos, device=self.device)
            noisy_fingertip_pos = clean_fingertip_pos + (noise @ torch.diag(noise_std))

            noise = torch.randn((self.num_envs, 3), device=self.device)
            noise_std = torch.tensor(self.cfg.obs_rand.fingertip_quat, device=self.device)
            aa_noise = noise @ torch.diag(noise_std)
            angle = torch.norm(aa_noise, p=2, dim=-1)
            axis = aa_noise / (angle.unsqueeze(-1) + 1e-6)
            dynamic_quat_noise = torch_utils.quat_from_angle_axis(angle, axis)

            noisy_fingertip_quat = torch_utils.quat_mul(dynamic_quat_noise, base_perceived_fingertip_quat)

            noise = torch.randn_like(clean_ee_linvel)
            noise_std = torch.tensor(self.cfg.obs_rand.ee_linvel, device=self.device)
            noisy_ee_linvel = clean_ee_linvel + (noise @ torch.diag(noise_std))

            noise = torch.randn_like(clean_ee_angvel)
            noise_std = torch.tensor(self.cfg.obs_rand.ee_angvel, device=self.device)
            noisy_ee_angvel = clean_ee_angvel + (noise @ torch.diag(noise_std))

            if getattr(self.cfg_task, "contact_force", {}).get("enabled", False):
                noise = torch.randn_like(clean_contact_force)
                noise_std = torch.tensor(self.cfg.obs_rand.contact_force, device=self.device)
                noisy_contact_force = clean_contact_force + (noise @ torch.diag(noise_std))
            else:
                noisy_contact_force = clean_contact_force

        else:
            noisy_fixed_pos = base_perceived_hole_pos
            noisy_fingertip_pos = clean_fingertip_pos
            noisy_fingertip_quat = base_perceived_fingertip_quat
            noisy_ee_linvel = clean_ee_linvel
            noisy_ee_angvel = clean_ee_angvel

            noisy_contact_force = clean_contact_force

        prev_actions = self.actions.clone()
        fixed_quat_obs_target = self.fixed_quat
        grasp_yaw_comp_rad = self._get_grasp_yaw_comp_rad()

        clean_fingertip_quat_for_rel = self._apply_world_yaw_comp_to_quat(clean_fingertip_quat, grasp_yaw_comp_rad)
        noisy_fingertip_quat_for_rel = self._apply_world_yaw_comp_to_quat(noisy_fingertip_quat, grasp_yaw_comp_rad)

        clean_current_quat_inv = torch_utils.quat_conjugate(clean_fingertip_quat_for_rel)
        clean_fingertip_quat_rel_fixed = torch_utils.quat_mul(clean_current_quat_inv, fixed_quat_obs_target)
        if self.cfg_task.requires_orientation_logic:
            symmetry_rad = [angle * np.pi / 180.0 for angle in self.cfg_task.symmetry_angles_deg]
            _, clean_relative_quat_to_closest, _ = insertanything_utils.get_closest_symmetry_transform(
                clean_fingertip_quat_for_rel, fixed_quat_obs_target, symmetry_rad
            )
            clean_fingertip_quat_rel_fixed = clean_relative_quat_to_closest

        noisy_current_quat_inv = torch_utils.quat_conjugate(noisy_fingertip_quat_for_rel)
        noisy_fingertip_quat_rel_fixed = torch_utils.quat_mul(noisy_current_quat_inv, fixed_quat_obs_target)
        if self.cfg_task.requires_orientation_logic:
            symmetry_rad = [angle * np.pi / 180.0 for angle in self.cfg_task.symmetry_angles_deg]
            _, noisy_relative_quat_to_closest, _ = insertanything_utils.get_closest_symmetry_transform(
                noisy_fingertip_quat_for_rel, fixed_quat_obs_target, symmetry_rad
            )
            noisy_fingertip_quat_rel_fixed = noisy_relative_quat_to_closest

        state_contact_force = self._process_contact_force_input(clean_contact_force)
        obs_contact_force = self._process_contact_force_input(noisy_contact_force)

        state_dict = {
            "fingertip_pos": clean_fingertip_pos,
            "fingertip_pos_rel_fixed": clean_fingertip_pos - self.fixed_pos_obs_frame,
            "fingertip_quat": clean_fingertip_quat,
            "fingertip_quat_rel_fixed": clean_fingertip_quat_rel_fixed,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "contact_force": state_contact_force,
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "task_prop_gains": self.task_prop_gains,
            "ema_factor": torch.tensor([[self.ema_factor]], device=self.device).repeat(self.num_envs, 1),
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "dr_friction": self.dr_frictions,
            "dr_dead_zone": self.dead_zone_thresholds,
            "prev_actions": prev_actions,
        }

        obs_dict = {
            "fingertip_pos": noisy_fingertip_pos,
            "fingertip_pos_rel_fixed": noisy_fingertip_pos - noisy_fixed_pos,
            "fingertip_quat": noisy_fingertip_quat,
            "fingertip_quat_rel_fixed": noisy_fingertip_quat_rel_fixed,
            "ee_linvel": noisy_ee_linvel,
            "ee_angvel": noisy_ee_angvel,
            "contact_force": obs_contact_force,
            "prev_actions": prev_actions,
        }

        return obs_dict, state_dict

    def _get_observations(self):
        obs_dict, state_dict = self._get_insertanything_obs_state_dict()

        obs_tensors = insertanything_utils.collapse_obs_dict(obs_dict, self.obs_order + ["prev_actions"])
        state_tensors = insertanything_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        return {"policy": obs_tensors, "critic": state_tensors}

    def _reset_buffers(self, env_ids):
        self.ep_succeeded[env_ids] = 0
        self.ep_success_times[env_ids] = 0

    def _pre_physics_step(self, action):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.actions = self.ema_factor * action.clone().to(self.device) + (1 - self.ema_factor) * self.actions

    def generate_ctrl_signals(
        self,
        ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat,
        ctrl_target_gripper_dof_pos,
    ):
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        self.joint_torque, self.applied_wrench = insertanything_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
            dead_zone_thresholds=self.dead_zone_thresholds,
        )

        self.ctrl_target_joint_pos[:, 7:9] = ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def close_gripper_in_place(self):
        actions = torch.zeros((self.num_envs, 6), device=self.device)

        pos_actions = actions[:, 0:3] * self.pos_threshold
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        rot_actions = actions[:, 3:6]

        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0],
            pitch=target_euler_xyz[:, 1],
            yaw=target_euler_xyz[:, 2],
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def _apply_action(self):
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values()

        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        rot_actions = self.actions[:, 3:6] * self.rot_threshold

        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        delta_pos = ctrl_target_fingertip_midpoint_pos - fixed_pos_action_frame
        bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds, device=self.device)
        pos_error_clipped = torch.clip(delta_pos, -bounds, bounds)
        ctrl_target_fingertip_midpoint_pos = fixed_pos_action_frame + pos_error_clipped

        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0],
            pitch=target_euler_xyz[:, 1],
            yaw=target_euler_xyz[:, 2],
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def _get_dones(self):
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _get_grasp_yaw_comp_rad(self) -> float:
        return np.deg2rad(float(getattr(self.cfg_task, "grasp_yaw_comp_deg", 0.0)))

    def _apply_world_yaw_comp_to_quat(self, quat: torch.Tensor, yaw_comp_rad: float) -> torch.Tensor:
        if abs(yaw_comp_rad) <= 1e-8:
            return quat
        yaw_comp_quat = torch_utils.quat_from_euler_xyz(
            torch.zeros((quat.shape[0],), device=self.device),
            torch.zeros((quat.shape[0],), device=self.device),
            torch.full((quat.shape[0],), yaw_comp_rad, device=self.device),
        )
        return torch_utils.quat_mul(yaw_comp_quat, quat)

    def _apply_local_yaw_comp_to_quat(self, quat: torch.Tensor, yaw_comp_rad: float) -> torch.Tensor:
        if abs(yaw_comp_rad) <= 1e-8:
            return quat
        yaw_comp_quat = torch_utils.quat_from_euler_xyz(
            torch.zeros((quat.shape[0],), device=self.device),
            torch.zeros((quat.shape[0],), device=self.device),
            torch.full((quat.shape[0],), yaw_comp_rad, device=self.device),
        )
        return torch_utils.quat_mul(quat, yaw_comp_quat)

    def _get_curr_successes(self, success_threshold):
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        held_base_pos, held_base_quat = insertanything_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = insertanything_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task,
            self.num_envs,
            self.device,
        )
        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        is_centered = torch.where(
            xy_dist < 0.0025,
            torch.ones_like(curr_successes),
            torch.zeros_like(curr_successes),
        )
        fixed_cfg = self.cfg_task.fixed_asset_cfg

        height_threshold = fixed_cfg.height * success_threshold
        is_close_or_below = torch.where(
            z_disp < height_threshold,
            torch.ones_like(curr_successes),
            torch.zeros_like(curr_successes),
        )
        curr_successes = torch.logical_and(is_centered, is_close_or_below)

        if self.cfg_task.requires_orientation_logic:
            symmetry_rad = [angle * np.pi / 180.0 for angle in self.cfg_task.symmetry_angles_deg]

            _, _, min_yaw_error = insertanything_utils.get_closest_symmetry_transform(
                held_base_quat, target_held_base_quat, symmetry_rad
            )

            is_oriented = min_yaw_error < self.cfg_task.yaw_success_threshold

            curr_successes = torch.logical_and(curr_successes, is_oriented)

        return curr_successes

    def _get_curr_failures(self):
        return torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    def _log_factory_metrics(self, rew_dict, curr_successes):
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

    def _get_factory_rew_dict(self):
        rew_dict, rew_scales = {}, {}
        held_base_pos, held_base_quat = insertanything_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = insertanything_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task,
            self.num_envs,
            self.device,
        )

        if self.cfg_task.use_decoupled_reward:

            xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
            z_dist = torch.abs(target_held_base_pos[:, 2] - held_base_pos[:, 2])

            xy_coef_a, xy_coef_b = self.cfg_task.xy_dist_coef
            rew_xy_align = insertanything_utils.squashing_fn(xy_dist, xy_coef_a, xy_coef_b)

            z_coef_a, z_coef_b = self.cfg_task.z_dist_coef
            rew_z_insert = insertanything_utils.squashing_fn(z_dist, z_coef_a, z_coef_b)

            gate_sharpness = self.cfg_task.z_reward_gate_sharpness
            z_reward_mask = torch.exp(-gate_sharpness * xy_dist)
            rew_z_insert_gated = rew_z_insert * z_reward_mask

            if self.cfg_task.requires_orientation_logic:
                symmetry_rad = [angle * np.pi / 180.0 for angle in self.cfg_task.symmetry_angles_deg]
                _, _, min_yaw_error = insertanything_utils.get_closest_symmetry_transform(
                    held_base_quat, target_held_base_quat, symmetry_rad
                )
                orientation_rew = insertanything_utils.compute_orientation_reward(
                    min_yaw_error, self.cfg_task.orientation_coef
                )

                orientation_threshold = np.deg2rad(self.cfg_task.orientation_reward_threshold)
                orientation_mask = (min_yaw_error < orientation_threshold).float()
            else:
                orientation_mask = torch.ones((self.num_envs,), device=self.device)

            action_penalty_ee = torch.norm(self.actions, p=2, dim=-1)
            action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
            curr_engaged = self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold)
            curr_engaged_half = self._get_curr_successes(success_threshold=self.cfg_task.engage_half_threshold)
            curr_successes = self._get_curr_successes(success_threshold=self.cfg_task.success_threshold)

            rew_dict = {
                "xy_align": rew_xy_align,
                "z_insert": rew_z_insert_gated * orientation_mask,
                "action_penalty_ee": action_penalty_ee,
                "action_grad_penalty": action_grad_penalty,
                "curr_engaged": curr_engaged.float(),
                "curr_engaged_half": curr_engaged_half.float(),
                "curr_success": curr_successes.float(),
            }
            rew_scales = {
                "xy_align": self.cfg_task.xy_dist_reward_scale,
                "z_insert": self.cfg_task.z_dist_reward_scale,
                "action_penalty_ee": -self.cfg_task.action_penalty_ee_scale,
                "action_grad_penalty": -self.cfg_task.action_grad_penalty_scale,
                "curr_engaged": self.cfg_task.engage_threshold_scale,
                "curr_engaged_half": self.cfg_task.engage_half_threshold_scale,
                "curr_success": self.cfg_task.success_threshold_scale,
            }

            if self.cfg_task.requires_orientation_logic:
                rew_dict["orientation"] = orientation_rew
                rew_scales["orientation"] = self.cfg_task.orientation_reward_scale

        else:

            keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
            keypoints_fixed = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
            offsets = insertanything_utils.get_keypoint_offsets(self.cfg_task.num_keypoints, self.device)
            keypoint_offsets = offsets * self.cfg_task.keypoint_scale
            for idx, keypoint_offset in enumerate(keypoint_offsets):
                keypoints_held[:, idx] = torch_utils.tf_combine(
                    held_base_quat,
                    held_base_pos,
                    torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                    keypoint_offset.repeat(self.num_envs, 1),
                )[1]
                keypoints_fixed[:, idx] = torch_utils.tf_combine(
                    target_held_base_quat,
                    target_held_base_pos,
                    torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
                    keypoint_offset.repeat(self.num_envs, 1),
                )[1]
            keypoint_dist = torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(-1)

            a0, b0 = self.cfg_task.keypoint_coef_baseline
            a1, b1 = self.cfg_task.keypoint_coef_coarse
            a2, b2 = self.cfg_task.keypoint_coef_fine

            action_penalty_ee = torch.norm(self.actions, p=2, dim=-1)
            action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
            curr_engaged = self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold)
            curr_engaged_half = self._get_curr_successes(success_threshold=self.cfg_task.engage_half_threshold)
            curr_successes = self._get_curr_successes(success_threshold=self.cfg_task.success_threshold)

            rew_dict = {
                "kp_baseline": insertanything_utils.squashing_fn(keypoint_dist, a0, b0),
                "kp_coarse": insertanything_utils.squashing_fn(keypoint_dist, a1, b1),
                "kp_fine": insertanything_utils.squashing_fn(keypoint_dist, a2, b2),
                "action_penalty_ee": action_penalty_ee,
                "action_grad_penalty": action_grad_penalty,
                "curr_engaged": curr_engaged.float(),
                "curr_engaged_half": curr_engaged_half.float(),
                "curr_success": curr_successes.float(),
            }
            rew_scales = {
                "kp_baseline": self.cfg_task.kp_baseline_scale,
                "kp_coarse": self.cfg_task.kp_coarse_scale,
                "kp_fine": self.cfg_task.kp_fine_scale,
                "action_penalty_ee": -self.cfg_task.action_penalty_ee_scale,
                "action_grad_penalty": -self.cfg_task.action_grad_penalty_scale,
                "curr_engaged": self.cfg_task.engage_threshold_scale,
                "curr_engaged_half": self.cfg_task.engage_half_threshold_scale,
                "curr_success": self.cfg_task.success_threshold_scale,
            }

            if self.cfg_task.requires_orientation_logic:
                symmetry_rad = [angle * np.pi / 180.0 for angle in self.cfg_task.symmetry_angles_deg]
                _, _, min_yaw_error = insertanything_utils.get_closest_symmetry_transform(
                    held_base_quat, target_held_base_quat, symmetry_rad
                )
                orientation_rew = insertanything_utils.compute_orientation_reward(
                    min_yaw_error, self.cfg_task.orientation_coef
                )

                rew_dict["orientation"] = orientation_rew
                rew_scales["orientation"] = self.cfg_task.orientation_reward_scale

                orientation_threshold = np.deg2rad(self.cfg_task.orientation_reward_threshold)
                orientation_mask = (min_yaw_error < orientation_threshold).float()
                rew_dict["kp_fine"] *= orientation_mask

        contact_cfg = getattr(self.cfg_task, "contact_force", {})
        if contact_cfg.get("enabled", False) and contact_cfg.get("use_as_reward", False):
            z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]
            margin = self.cfg_task.insert_depth_margin

            force_sq = torch.sum(torch.square(self.contact_force_local), dim=-1)
            force_xy_sq = torch.sum(torch.square(self.contact_force_local[:, 0:2]), dim=-1)

            is_attempt = (z_disp > margin).float()
            is_insert = (z_disp <= margin).float()

            penalty = (
                force_sq * is_attempt * self.cfg_task.contact_force_penalty_attempt_scale
                + force_xy_sq * is_insert * self.cfg_task.contact_force_penalty_insertion_scale
            )

            rew_dict["contact_force_penalty"] = penalty
            rew_scales["contact_force_penalty"] = -1.0

        if self.contact_force_logger is not None:
            self.contact_force_logger.log_step(
                step=self.episode_length_buf[0].item(),
                is_engaged=rew_dict["curr_engaged"][0].item(),
                is_engaged_half=rew_dict["curr_engaged_half"][0].item(),
                is_success=rew_dict["curr_success"][0].item(),
                contact_force=self.contact_force_local[0],
            )

        return rew_dict, rew_scales, curr_successes

    def _get_rewards(self):
        if self.cfg.evaluation_mode:
            self.extras.clear()
        rew_dict, rew_scales, curr_successes_for_training_log = self._get_factory_rew_dict()

        rew_buf = torch.zeros((self.num_envs,), device=self.device)
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self.prev_actions = self.actions.clone()

        if self.cfg.evaluation_mode:
            eval_success_mask = rew_dict["curr_success"].bool()

            self._log_factory_metrics(rew_dict, eval_success_mask)

            if torch.any(self.reset_buf):
                self.eval_ep_count += 1

                num_successes = torch.sum(self.ep_succeeded)
                success_rate = (num_successes / self.num_envs) * 100

                print_str = f"\n--- Episode {self.eval_ep_count} Finished ---\n"
                print_str += (
                    f"    Success Rate (insert_success): {success_rate:.2f}% ({num_successes.item()}/{self.num_envs})\n"
                )

                if num_successes > 0:
                    successful_times_steps = self.ep_success_times[self.ep_succeeded.bool()]
                    avg_success_time_sec = torch.mean(successful_times_steps.float()) * self.step_dt
                    print_str += f"    Average Success Time: {avg_success_time_sec:.2f} seconds\n"

                self.extras["eval_printout"] = print_str
        else:
            self._log_factory_metrics(rew_dict, curr_successes_for_training_log)
        return rew_buf

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

        if self._enable_hole_view_camera() and hasattr(self, "fixed_pos"):
            self._set_hole_view_camera(target_pos=self.fixed_pos[0].detach().cpu().numpy())

        if self.contact_force_logger is not None:
            self.contact_force_logger.start_new_episode()

        self.contact_force_local[env_ids] = 0.0
        self.smoothed_contact_force[env_ids] = 0.0
        self._reset_contact_force_input_randomization(env_ids)

    def _set_assets_to_default_pose(self, env_ids):
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def set_pos_inverse_kinematics(
        self,
        ctrl_target_fingertip_midpoint_pos,
        ctrl_target_fingertip_midpoint_quat,
        env_ids,
    ):
        ik_time = 0.0
        while ik_time < 0.25:
            pos_error, axis_angle_error = insertanything_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            delta_dof_pos = insertanything_control.get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def get_handheld_asset_relative_pose(self):

        held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
        held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
        held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length

        held_asset_relative_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        grasp_yaw_comp_rad = self._get_grasp_yaw_comp_rad()
        held_asset_relative_quat = self._apply_local_yaw_comp_to_quat(held_asset_relative_quat, -grasp_yaw_comp_rad)

        return held_asset_relative_pos, held_asset_relative_quat

    def _set_franka_to_default_pose(self, joints, env_ids):
        gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values()

    def randomize_initial_state(self, env_ids):
        correction_mode = bool(getattr(self, "correction_field_mode", False))
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]

        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)

        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_range,
            dtype=torch.float32,
            device=self.device,
        )
        if correction_mode:
            fixed_asset_init_pos_rand.zero_()
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)

        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]

        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_half_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)

        if correction_mode:
            yaw_noise = torch.zeros((len(env_ids),), dtype=torch.float32, device=self.device)
        else:
            yaw_noise = (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * fixed_orn_yaw_half_range

        fixed_orn_euler = torch.zeros((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler[:, 2] = fixed_orn_init_yaw + yaw_noise

        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat

        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        self.target_hole_local_pos[env_ids] = 0.0
        self.target_hole_local_quat[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self.step_sim_no_action()

        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            fixed_tip_pos_local,
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        bad_envs = env_ids.clone()
        ik_attempt = 0
        max_ik_attempts = 30

        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        while ik_attempt < max_ik_attempts:
            n_bad = bad_envs.shape[0]

            above_fixed_pos = fixed_tip_pos.clone()
            above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2]
            if correction_mode:
                above_fixed_pos[env_ids, 0:2] += self.correction_field_xy_offsets[env_ids]
                above_fixed_pos[env_ids, 2] += self.correction_field_hand_z_offset

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)
            hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_range, device=self.device)
            if correction_mode:
                hand_init_pos_rand.zero_()
            above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
            above_fixed_pos[bad_envs] += above_fixed_pos_rand

            hand_down_euler = (
                torch.tensor(self.cfg_task.hand_init_orn, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .repeat(n_bad, 1)
            )

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_range, device=self.device)
            if correction_mode:
                hand_init_orn_rand.zero_()
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise
            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0],
                pitch=hand_down_euler[:, 1],
                yaw=hand_down_euler[:, 2],
            )

            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_fingertip_midpoint_pos=above_fixed_pos,
                ctrl_target_fingertip_midpoint_quat=hand_down_quat,
                env_ids=bad_envs,
            )
            pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
            angle_error = torch.norm(aa_error, dim=1) > 1e-3
            any_error = torch.logical_or(pos_error, angle_error)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            if bad_envs.shape[0] == 0:
                break

            self._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0],
                env_ids=bad_envs,
            )

            ik_attempt += 1

        if bad_envs.shape[0] > 0:
            print(
                f"[ERROR] {bad_envs.shape[0]} envs failed IK reachability after {max_ik_attempts} attempts. Force"
                " starting episode."
            )
            print(f"Failed env IDs: {bad_envs.cpu().numpy()}")

        self.step_sim_no_action()

        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
            q1=self.fingertip_midpoint_quat,
            t1=self.fingertip_midpoint_pos,
            q2=flip_z_quat,
            t2=torch.zeros((self.num_envs, 3), device=self.device),
        )

        held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat, held_asset_relative_pos
        )

        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=fingertip_flipped_quat,
            t1=fingertip_flipped_pos,
            q2=asset_in_hand_quat,
            t2=asset_in_hand_pos,
        )

        rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
        held_asset_pos_noise = 2 * (rand_sample - 0.5)
        held_asset_pos_noise_level = torch.tensor(self.cfg_task.held_asset_pos_range, device=self.device)
        if correction_mode:
            held_asset_pos_noise_level.zero_()
        held_asset_pos_noise = held_asset_pos_noise @ torch.diag(held_asset_pos_noise_level)
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            t2=held_asset_pos_noise,
        )

        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
        held_state[:, 3:7] = translated_held_asset_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()

        reset_task_prop_gains = torch.tensor(
            self.cfg.ctrl.reset_task_prop_gains, dtype=torch.float32, device=self.device
        ).repeat((self.num_envs, 1))
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = insertanything_utils.get_deriv_gains(
            reset_task_prop_gains, self.cfg.ctrl.reset_rot_deriv_scale
        )

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0
            self.close_gripper_in_place()
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        if getattr(self.cfg_task, "dr_randomize_dynamics", False):
            n_envs = len(env_ids)
            frictions = torch.empty(n_envs, device=self.device).uniform_(*self.cfg_task.dr_friction_range)
            insertanything_utils.set_friction(self._held_asset, frictions, env_ids, device=self.device)
            insertanything_utils.set_friction(self._fixed_asset, frictions, env_ids, device=self.device)
            self.dr_frictions[env_ids, 0] = frictions

            trans_kp = torch.empty((n_envs, 3), device=self.device).uniform_(*self.cfg_task.dr_gains_trans_range)
            rot_kp = torch.empty((n_envs, 3), device=self.device).uniform_(*self.cfg_task.dr_gains_rot_range)
            self.default_gains[env_ids] = torch.cat([trans_kp, rot_kp], dim=-1)

            trans_dz = torch.empty((n_envs, 3), device=self.device).uniform_(*self.cfg_task.dr_dead_zone_trans_range)
            rot_dz = torch.empty((n_envs, 3), device=self.device).uniform_(*self.cfg_task.dr_dead_zone_rot_range)
            self.dead_zone_thresholds[env_ids] = torch.cat([trans_dz, rot_dz], dim=-1)

            if n_envs <= 5:
                for i in range(n_envs):
                    print(
                        f"[DR] Env {env_ids[i].item()}: Friction={frictions[i].item():.3f}, "
                        f"Gains={self.default_gains[env_ids[i]].cpu().numpy()}, "
                        f"DeadZone={self.dead_zone_thresholds[env_ids[i]].cpu().numpy()}"
                    )

        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = insertanything_utils.get_deriv_gains(self.default_gains)

        static_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        obs_noise_static_std = torch.tensor(self.cfg.obs_rand.fixed_asset_pos_static_error, device=self.device)
        if correction_mode:
            self.init_fixed_pos_obs_noise[env_ids] = self.correction_field_obs_bias[env_ids]
        else:
            self.init_fixed_pos_obs_noise[env_ids] = static_noise @ torch.diag(obs_noise_static_std)

        static_noise_euler = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        obs_noise_static_std_euler = torch.tensor(
            self.cfg.obs_rand.fingertip_quat_static_error,
            dtype=torch.float32,
            device=self.device,
        )
        if correction_mode:
            static_noise_euler.zero_()
        else:
            static_noise_euler = static_noise_euler @ torch.diag(obs_noise_static_std_euler)

        quat_noise = torch_utils.quat_from_euler_xyz(
            roll=static_noise_euler[:, 0],
            pitch=static_noise_euler[:, 1],
            yaw=static_noise_euler[:, 2],
        )
        self.init_fingertip_quat_obs_noise[env_ids] = quat_noise

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))

    def close(self):
        if self.contact_force_logger is not None:
            self.contact_force_logger.close()
        super().close()
