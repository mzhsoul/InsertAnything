from dataclasses import field

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from .insertanything_tasks_cfg import (
    InsertAnythingCircleHole_I,
    InsertAnythingHexagonHole_III,
    InsertAnythingHexagonHole_IV,
    InsertAnythingLHole_III,
    InsertAnythingSquareHole_II,
    InsertAnythingTask,
    InsertAnythingTriangleHole_IV,
)

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "fingertip_quat_rel_fixed": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "contact_force": 3,
}


STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "fingertip_quat_rel_fixed": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "contact_force": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
    "dr_friction": 1,
    "dr_dead_zone": 6,
}


@configclass
class ObsRandCfg:
    fixed_asset_pos_static_error: list = [0.001, 0.001, 0.001]
    fingertip_quat_static_error: list = [0.08, 0.08, 0.01]

    use_all_noise: bool = True
    fixed_asset_pos: list = [0.0005, 0.0005, 0.0005]
    fingertip_pos: list = [0.0005, 0.0005, 0.0005]
    fingertip_quat: list = [0.04, 0.04, 0.005]
    ee_linvel: list = [0.01, 0.01, 0.01]
    ee_angvel: list = [0.01, 0.01, 0.01]
    contact_force: list = [0.1, 0.1, 0.1]


@configclass
class CtrlCfg:
    ema_factor: float = 0.2

    pos_action_bounds: list = [0.05, 0.05, 0.05]
    rot_action_bounds: list = [1.0, 1.0, 1.0]
    pos_action_threshold: list = [0.02, 0.02, 0.02]
    rot_action_threshold: list = [0.097, 0.097, 0.097]

    reset_joints: list = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
    reset_task_prop_gains: list = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale: float = 10.0
    default_task_prop_gains: list = [100, 100, 100, 30, 30, 30]

    default_dof_pos_tensor: list = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null: float = 10.0
    kd_null: float = 6.3246


@configclass
class InsertAnythingEnvCfg(DirectRLEnvCfg):
    decimation: int = 8
    action_space: int = 6
    observation_space: int = 21
    state_space: int = 72

    state_order: list = [
        "fingertip_pos",
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "fingertip_quat_rel_fixed",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
    ]

    task_name: str = "insertanything_circle_I"
    task: InsertAnythingTask = InsertAnythingTask()
    obs_rand: ObsRandCfg = ObsRandCfg()
    ctrl: CtrlCfg = CtrlCfg()

    evaluation_mode: bool = False
    episode_length_s: float = 10.0

    sim: SimulationCfg = SimulationCfg(
        device="cuda:0",
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        render_interval=decimation,
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,
        ),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    scene: InteractiveSceneCfg = field(
        default_factory=lambda: InteractiveSceneCfg(
            num_envs=128,
            env_spacing=2.0,
        )
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
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
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.0005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.00871,
                "panda_joint2": -0.10368,
                "panda_joint3": -0.00794,
                "panda_joint4": -1.49139,
                "panda_joint5": -0.00083,
                "panda_joint6": 1.38774,
                "panda_joint7": 0.0,
                "panda_finger_joint2": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_arm1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=87,
                velocity_limit_sim=124.6,
            ),
            "panda_arm2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                armature=0.0,
                effort_limit_sim=12,
                velocity_limit_sim=149.5,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint[1-2]"],
                effort_limit_sim=40.0,
                velocity_limit_sim=0.04,
                stiffness=7500.0,
                damping=173.0,
                friction=0.1,
                armature=0.0,
            ),
        },
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/HeldAsset/.*",
        filter_prim_paths_expr=["/World/envs/env_.*/FixedAsset/.*"],
        update_period=0.0,
    )

    def __post_init__(self):
        self.robot.spawn.usd_path = self.task.robot_cfg.robot_usd

        contact_cfg = getattr(self.task, "contact_force", {})
        contact_enabled = bool(contact_cfg.get("enabled", False))
        if contact_enabled:
            force_source = contact_cfg.get("force_source", "peg_hole")
            if force_source == "gripper_peg":
                self.contact_sensor.filter_prim_paths_expr = [
                    "/World/envs/env_.*/Robot/panda_leftfinger",
                    "/World/envs/env_.*/Robot/panda_rightfinger",
                ]
            else:
                self.contact_sensor.filter_prim_paths_expr = ["/World/envs/env_.*/FixedAsset/.*"]

        self.scene.clone_in_fabric = not contact_enabled


@configclass
class InsertAnythingCircleHole_I_Cfg(InsertAnythingEnvCfg):
    task: InsertAnythingTask = InsertAnythingCircleHole_I()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingSquareHole_II_Cfg(InsertAnythingEnvCfg):
    task: InsertAnythingTask = InsertAnythingSquareHole_II()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingLHole_III_Cfg(InsertAnythingEnvCfg):
    task: InsertAnythingTask = InsertAnythingLHole_III()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingTriangleHole_IV_Cfg(InsertAnythingEnvCfg):
    task: InsertAnythingTask = InsertAnythingTriangleHole_IV()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingHexagonHole_III_Cfg(InsertAnythingEnvCfg):
    task: InsertAnythingTask = InsertAnythingHexagonHole_III()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingHexagonHole_IV_Cfg(InsertAnythingEnvCfg):
    task: InsertAnythingTask = InsertAnythingHexagonHole_IV()
    task_name: str = task.name
    episode_length_s: float = task.duration_s
