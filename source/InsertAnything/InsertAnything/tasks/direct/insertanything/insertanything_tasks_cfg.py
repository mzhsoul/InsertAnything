import os
from dataclasses import field

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

INSERTANYTHING_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


def contact_force_cfg(
    *,
    enabled: bool = False,
    force_source: str = "peg_hole",
    use_as_obs: bool = False,
    use_as_state: bool = False,
    use_as_reward: bool = False,
    log_contact_force: bool = False,
    ema_alpha: float = 0.25,
) -> dict:
    return {
        "enabled": enabled,
        "force_source": force_source,
        "use_as_obs": use_as_obs,
        "use_as_state": use_as_state,
        "use_as_reward": use_as_reward,
        "log_contact_force": log_contact_force,
        "ema_alpha": ema_alpha,
    }


@configclass
class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@configclass
class RobotCfg:
    robot_usd: str = f"{ASSET_DIR}/franka_mimic.usd"
    franka_fingerpad_length: float = 0.017608
    friction: float = 0.75


@configclass
class InsertAnythingTask:
    robot_cfg: RobotCfg = RobotCfg()
    name: str = ""
    duration_s: float = 5.0

    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()
    asset_size: float = 0.0

    hand_init_pos: list = [0.0, 0.0, 0.047]
    hand_init_pos_range: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_range: list = [0.0, 0.0, 0.0]

    fixed_asset_init_pos_range: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 0.0
    grasp_yaw_comp_deg: float = 0.0

    held_asset_pos_range: list = [0.003, 0.0, 0.003]
    held_asset_rot_init: float = 0.0

    dr_randomize_dynamics: bool = True
    dr_friction_range: list = [0.3, 0.6]
    dr_gains_trans_range: list = [100.0, 300.0]
    dr_gains_rot_range: list = [30.0, 50.0]
    dr_dead_zone_trans_range: list = [0.0, 0.2]
    dr_dead_zone_rot_range: list = [0.0, 0.04]

    action_penalty_ee_scale: float = 0.008
    action_grad_penalty_scale: float = 0.05

    num_keypoints: int = 4
    keypoint_scale: float = 0.15
    keypoint_coef_baseline: list = [5, 4]
    kp_baseline_scale: float = 1.0
    keypoint_coef_coarse: list = [50, 2]
    kp_coarse_scale: float = 1.0
    keypoint_coef_fine: list = [100, 0]
    kp_fine_scale: float = 1.0

    success_threshold: float = 0.04
    success_threshold_scale: float = 1.0
    engage_threshold: float = 0.9
    engage_threshold_scale: float = 1.0
    engage_half_threshold: float = 0.55
    engage_half_threshold_scale: float = 1.0

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.5
    z_reward_gate_sharpness: float = 100.0

    orientation_reward_scale: float = 3.0
    yaw_success_threshold: float = 0.05
    symmetry_angles_deg: list = []
    orientation_coef: list = [5, 4]
    orientation_reward_threshold: float = 1.0

    requires_orientation_logic: bool = False
    use_decoupled_reward: bool = False
    show_visual_markers: bool = True

    contact_force: dict = contact_force_cfg()
    contact_force_penalty_attempt_scale: float = 0.01
    contact_force_penalty_insertion_scale: float = 0.03
    insert_depth_margin: float = 0.002

    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "fingertip_quat_rel_fixed",
        "ee_linvel",
        "ee_angvel",
    ]

    fixed_asset: ArticulationCfg = field(default=None, init=False)
    held_asset: ArticulationCfg = field(default=None, init=False)

    def __post_init__(self):
        self.fixed_asset = self._make_fixed_asset_cfg(
            prim_path="/World/envs/env_.*/FixedAsset",
            usd_path=self.fixed_asset_cfg.usd_path,
            pos=(0.6, 0.0, 0.05),
        )
        self.held_asset = self._make_held_asset_cfg()

    def _make_fixed_asset_cfg(self, prim_path: str, usd_path: str, pos: tuple[float, float, float]) -> ArticulationCfg:
        return ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.fixed_asset_cfg.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=pos,
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={},
                joint_vel={},
            ),
            actuators={},
        )

    def _make_held_asset_cfg(self) -> ArticulationCfg:
        return ArticulationCfg(
            prim_path="/World/envs/env_.*/HeldAsset",
            spawn=sim_utils.UsdFileCfg(
                usd_path=self.held_asset_cfg.usd_path,
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=192,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.held_asset_cfg.mass),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.4, 0.1),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={},
                joint_vel={},
            ),
            actuators={},
        )


@configclass
class CircleHole(FixedAssetCfg):
    usd_path: str = f"{INSERTANYTHING_ASSETS_DIR}/circle/circle_hole.usd"
    diameter: float = 0.010
    height: float = 0.025
    base_height: float = 0.0


@configclass
class CirclePeg_I(HeldAssetCfg):
    usd_path: str = f"{INSERTANYTHING_ASSETS_DIR}/circle/circle_peg_I.usd"
    diameter: float = 0.008
    height: float = 0.050
    mass: float = 0.009


@configclass
class InsertAnythingCircleHole_I(InsertAnythingTask):
    name: str = "insertanything_circle_I"
    fixed_asset_cfg: FixedAssetCfg = CircleHole()
    held_asset_cfg: HeldAssetCfg = CirclePeg_I()
    asset_size: float = held_asset_cfg.diameter
    duration_s: float = 10.0

    hand_init_orn_range: list = [0.0, 0.0, 0.2]
    fixed_asset_init_orn_range_deg: float = 30.0
    z_reward_gate_sharpness: float = 100.0
    action_penalty_ee_scale: float = 0.3
    action_grad_penalty_scale: float = 0.6

    use_decoupled_reward: bool = True
    requires_orientation_logic: bool = True
    symmetry_angles_deg: list = [0.0, 90.0, 180.0, 270.0]
    yaw_success_threshold: float = 0.5
    orientation_reward_scale: float = 3.0
    orientation_coef: list = [5, 4]
    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "fingertip_quat_rel_fixed",
        "ee_linvel",
        "ee_angvel",
    ]


@configclass
class SquareHole(FixedAssetCfg):
    usd_path: str = f"{INSERTANYTHING_ASSETS_DIR}/square/square_hole.usd"
    diameter: float = 0.010
    height: float = 0.025
    base_height: float = 0.0


@configclass
class SquarePeg_II(HeldAssetCfg):
    usd_path: str = f"{INSERTANYTHING_ASSETS_DIR}/square/square_peg_II.usd"
    diameter: float = 0.0095
    height: float = 0.050
    mass: float = 0.024


@configclass
class InsertAnythingSquareHole_II(InsertAnythingTask):
    name: str = "insertanything_square_II"
    fixed_asset_cfg: FixedAssetCfg = SquareHole()
    held_asset_cfg: HeldAssetCfg = SquarePeg_II()
    asset_size: float = held_asset_cfg.diameter
    duration_s: float = 15.0

    hand_init_orn_range: list = [0.0, 0.0, 0.785]
    fixed_asset_init_orn_range_deg: float = 90.0
    action_penalty_ee_scale: float = 0.03
    action_grad_penalty_scale: float = 0.1

    orientation_reward_scale: float = 3.0
    yaw_success_threshold: float = 0.05
    symmetry_angles_deg: list = [0.0, 90.0, 180.0, 270.0]
    orientation_coef: list = [5, 4]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.0
    z_reward_gate_sharpness: float = 100.0

    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True


@configclass
class LHole(FixedAssetCfg):
    usd_path: str = f"{INSERTANYTHING_ASSETS_DIR}/L_hole/L_hole.usd"
    diameter: float = 0.015
    height: float = 0.025
    base_height: float = 0.0


@configclass
class LPeg_III(HeldAssetCfg):
    usd_path: str = f"{INSERTANYTHING_ASSETS_DIR}/L_hole/L_peg_III.usd"
    diameter: float = 0.0149
    height: float = 0.050
    mass: float = 0.022


@configclass
class InsertAnythingLHole_III(InsertAnythingTask):
    name: str = "insertanything_L_III"
    fixed_asset_cfg: FixedAssetCfg = LHole()
    held_asset_cfg: HeldAssetCfg = LPeg_III()
    asset_size: float = held_asset_cfg.diameter
    duration_s: float = 20.0

    hand_init_orn_range: list = [0.0, 0.0, 0.2]
    fixed_asset_init_orn_deg: float = 180.0
    fixed_asset_init_orn_range_deg: float = 30.0
    action_penalty_ee_scale: float = 0.05
    action_grad_penalty_scale: float = 0.1

    orientation_reward_scale: float = 2.5
    yaw_success_threshold: float = 0.1
    orientation_reward_threshold: float = 5.0
    symmetry_angles_deg: list = [0.0]
    orientation_coef: list = [50, 2]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.5
    z_dist_coef: list = [50, 1]
    z_dist_reward_scale: float = 2.5
    z_reward_gate_sharpness: float = 100.0

    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True
    contact_force: dict = contact_force_cfg(
        enabled=True,
        force_source="gripper_peg",
        use_as_obs=True,
        use_as_state=True,
        use_as_reward=True,
        log_contact_force=True,
    )


@configclass
class TriangleHole(FixedAssetCfg):
    usd_path: str = f"{INSERTANYTHING_ASSETS_DIR}/triangle/triangle_hole.usd"
    diameter: float = 0.012
    height: float = 0.025
    base_height: float = 0.0


@configclass
class TrianglePeg_IV(HeldAssetCfg):
    usd_path: str = f"{INSERTANYTHING_ASSETS_DIR}/triangle/triangle_peg_IV.usd"
    diameter: float = 0.01196
    height: float = 0.050
    mass: float = 0.019


@configclass
class InsertAnythingTriangleHole_IV(InsertAnythingTask):
    name: str = "insertanything_triangle_IV"
    fixed_asset_cfg: FixedAssetCfg = TriangleHole()
    held_asset_cfg: HeldAssetCfg = TrianglePeg_IV()
    asset_size: float = held_asset_cfg.diameter
    duration_s: float = 20.0

    hand_init_orn_range: list = [0.0, 0.0, 0.2]
    fixed_asset_init_orn_deg: float = 180.0
    fixed_asset_init_orn_range_deg: float = 30.0
    action_penalty_ee_scale: float = 0.03
    action_grad_penalty_scale: float = 0.1

    orientation_reward_scale: float = 3.0
    yaw_success_threshold: float = 0.005
    symmetry_angles_deg: list = [0.0, 120.0, 240.0]
    orientation_coef: list = [5, 4]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.5
    z_reward_gate_sharpness: float = 100.0

    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True


@configclass
class HexagonHole(FixedAssetCfg):
    usd_path: str = f"{INSERTANYTHING_ASSETS_DIR}/hexagon/hexagon_hole.usd"
    diameter: float = 0.012
    height: float = 0.025
    base_height: float = 0.0


@configclass
class HexagonPeg_III(HeldAssetCfg):
    usd_path: str = f"{INSERTANYTHING_ASSETS_DIR}/hexagon/hexagon_peg_III.usd"
    diameter: float = 0.011884
    height: float = 0.050
    mass: float = 0.019


@configclass
class HexagonPeg_IV(HeldAssetCfg):
    usd_path: str = f"{INSERTANYTHING_ASSETS_DIR}/hexagon/hexagon_peg_IV.usd"
    diameter: float = 0.011976
    height: float = 0.050
    mass: float = 0.019


@configclass
class InsertAnythingHexagonHole_III(InsertAnythingTask):
    name: str = "insertanything_hexagon_III"
    fixed_asset_cfg: FixedAssetCfg = HexagonHole()
    held_asset_cfg: HeldAssetCfg = HexagonPeg_III()
    asset_size: float = held_asset_cfg.diameter
    duration_s: float = 20.0

    hand_init_orn_range: list = [0.0, 0.0, 0.2]
    fixed_asset_init_orn_deg: float = 180.0
    fixed_asset_init_orn_range_deg: float = 22.5
    action_penalty_ee_scale: float = 0.05
    action_grad_penalty_scale: float = 0.1

    orientation_reward_scale: float = 2.5
    yaw_success_threshold: float = 0.1
    orientation_reward_threshold: float = 1.0
    symmetry_angles_deg: list = [0.0]
    orientation_coef: list = [50, 2]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.5
    z_dist_coef: list = [50, 1]
    z_dist_reward_scale: float = 2.5
    z_reward_gate_sharpness: float = 100.0

    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True
    contact_force: dict = contact_force_cfg(
        enabled=True,
        force_source="gripper_peg",
        use_as_obs=True,
        use_as_state=True,
        use_as_reward=True,
        log_contact_force=True,
        ema_alpha=0.25,
    )


@configclass
class InsertAnythingHexagonHole_IV(InsertAnythingTask):
    name: str = "insertanything_hexagon_IV"
    fixed_asset_cfg: FixedAssetCfg = HexagonHole()
    held_asset_cfg: HeldAssetCfg = HexagonPeg_IV()
    asset_size: float = held_asset_cfg.diameter
    duration_s: float = 20.0

    hand_init_orn_range: list = [0.0, 0.0, 0.2]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 30.0
    action_penalty_ee_scale: float = 0.03
    action_grad_penalty_scale: float = 0.1

    orientation_reward_scale: float = 3.0
    yaw_success_threshold: float = 0.005
    symmetry_angles_deg: list = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    orientation_coef: list = [5, 4]

    xy_dist_coef: list = [50, 2]
    xy_dist_reward_scale: float = 2.0
    z_dist_coef: list = [20, 4]
    z_dist_reward_scale: float = 2.5
    z_reward_gate_sharpness: float = 100.0

    requires_orientation_logic: bool = True
    use_decoupled_reward: bool = True
