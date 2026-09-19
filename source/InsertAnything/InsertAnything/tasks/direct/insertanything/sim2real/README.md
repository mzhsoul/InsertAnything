# InsertAnything Sim-to-Real Deployment

This directory contains the real-robot deployment utilities for InsertAnything single-hole peg-in-hole experiments. The main entry point is:

```text
run_multi_episode_closed_loop.py
```

The script runs a deterministic closed-loop policy on a Franka robot:

```text
robot state -> InsertAnything observation -> Torch policy -> action postprocessor -> safety guard -> Cartesian pose command
```

It is designed for controlled laboratory experiments where a human operator marks each episode as success, failure, or abort from the terminal.

## 1. Robot-Control Backend

The current deployment script assumes a Franka robot server compatible with the HTTP control style used by Berkeley SERL and HIL-SERL:

- SERL: https://github.com/rail-berkeley/serl
- HIL-SERL: https://github.com/rail-berkeley/hil-serl

Those projects provide real-robot infrastructure for Franka manipulation experiments, including robot-server components and impedance-based Franka control. This InsertAnything runner is therefore expected to work directly only when a compatible robot server is running and the robot is controlled with a stable Cartesian-space impedance controller.

The script does not import SERL or HIL-SERL Python modules. It communicates with the robot through a small HTTP contract. If you use another Franka Python interface, such as `panda-py`, the runner can still be used after adapting the robot I/O layer, provided your backend supports:

- reliable Cartesian pose commands for the Franka end-effector,
- stable and robust Cartesian impedance control,
- real-time or near-real-time state feedback,
- gripper open and close commands,
- optional 3D contact-force feedback if using force-conditioned policies.

For a different backend, the main code locations to modify are:

| File | What to adapt |
| --- | --- |
| `robot_state_adapter.py` | Replace `fetch_robot_state()` and/or `robot_state_from_json()` to read your robot state source. |
| `run_multi_episode_closed_loop.py` | Replace `post_json()`, `send_pose6()`, `open_gripper()`, and `close_gripper()` if your backend is not HTTP-based. |
| `transforms.py` | Usually unchanged, unless your backend uses a different pose convention. |
| `observation_builder.py` | Usually unchanged, unless your robot state semantics or observation schema changes. |

## 2. Robot-Server Interface Contract

The default runner expects the following endpoints.

### 2.1 `POST /getstate`

Returns robot state as JSON. Required fields:

| Field | Shape | Meaning |
| --- | ---: | --- |
| `pose` | 7 | API frame `A` pose in Franka base frame `B`, `[x, y, z, qx, qy, qz, qw]`. |
| `ee` | 6 | Alternative to `pose`, `[x, y, z, roll, pitch, yaw]`. Used only if `pose` is missing. |

At least one of `pose` or `ee` must be present.

Optional fields:

| Field | Shape | Meaning |
| --- | ---: | --- |
| `q` | 7 | Joint positions. Defaults to zeros if missing. |
| `dq` | 7 | Joint velocities. Defaults to zeros if missing. |
| `vel` | 6 | End-effector twist-like vector `[vx, vy, vz, wx, wy, wz]`. Defaults to zeros if missing. |
| `gripper_pos` | 1 | Gripper opening or position. |
| `force_K_debug` | 3 | Contact-force observation source used by `--contact-force-source robot_state.force_K_debug`. |

The contact-force path is strict: when `--contact-force-source robot_state.force_K_debug` is selected, the JSON must contain `force_K_debug`.

### 2.2 `POST /pose`

Receives Cartesian pose commands:

```json
{"arr": [x, y, z, roll, pitch, yaw]}
```

The command controls the robot API frame `A`. Internally the policy, postprocessor, and safety guard work with fingertip frame `T`, so the runner converts:

```text
T_BA_target = T_BT_target * inv(T_AT)
```

before sending `/pose`.

### 2.3 Gripper Endpoints

The default endpoints are:

```text
POST /open_gripper
POST /close_gripper
```

They can be changed with:

```text
--open-gripper-endpoint
--close-gripper-endpoint
```

## 3. Directory Layout

| File or directory | Role |
| --- | --- |
| `run_multi_episode_closed_loop.py` | Main multi-episode closed-loop deployment script. |
| `observation_builder.py` | Builds InsertAnything actor observations from robot state and calibration. |
| `torch_policy.py` | Loads RL-Games recurrent actor checkpoints and runs deterministic inference. |
| `action_postprocessor.py` | Converts raw policy actions into candidate fingertip targets. |
| `safety_guard.py` | Clips candidate targets with step, hole-box, and yaw safety limits. |
| `robot_state_adapter.py` | Converts robot-server JSON into a normalized `RobotState`. |
| `transforms.py` | Pose, quaternion, and frame-transform utilities. |
| `configs/` | Calibration and deployment YAML templates. |
| `checkpoints/` | Optional local deployment checkpoint location. Large binaries are ignored by git. |

## 4. Calibration YAML

The runner reads a YAML file through `--config`. The YAML defines transforms, policy observation schema, policy architecture, and default action-processing values.

Current templates:

| File | Use case |
| --- | --- |
| `configs/calibration_insertanything_basic.yaml` | Pose-only single-hole policy, no contact-force observation. |
| `configs/calibration_insertanything_with_contact_force_template.yaml` | Contact-force policy using `force_K_debug`. |

### 4.1 Required Transforms

```yaml
transforms:
  T_AT:
    position: [0.0, 0.0, 0.0]
    quat_xyzw: [0.0, 0.0, 0.0, 1.0]

  T_BH_final:
    position: [0.50, 0.00, 0.04]
    quat_xyzw: [1.0, 0.0, 0.0, 0.0]

  T_BPRE_final:
    position: [0.50, 0.00, 0.09]
    quat_xyzw: [1.0, 0.0, 0.0, 0.0]
```

Meanings:

| Transform | Meaning |
| --- | --- |
| `T_AT` | API frame `A` to fingertip frame `T`. Use identity only if the API frame is already close enough to the policy fingertip frame. |
| `T_BH_final` | Single-hole reference pose in Franka base frame `B`. |
| `T_BPRE_final` | Pre-hover pose above the hole in Franka base frame `B`. |

All quaternions in YAML use `xyzw` order.

### 4.2 Policy Observation Schema

The basic template uses:

```yaml
policy_observation:
  obs_order:
    - fingertip_pos_rel_fixed
    - fingertip_quat
    - fingertip_quat_rel_fixed
    - ee_linvel
    - ee_angvel
  append_prev_actions: true
```

This schema produces a 23-dimensional actor observation after the previous six-dimensional action is appended.

The contact-force template uses:

```yaml
policy_observation:
  obs_order:
    - fingertip_pos_rel_fixed
    - fingertip_quat
    - fingertip_quat_rel_fixed
    - ee_linvel
    - ee_angvel
    - contact_force
  append_prev_actions: true
```

This schema produces a 26-dimensional actor observation after the previous six-dimensional action is appended. The YAML observation schema must match the checkpoint exactly. A checkpoint trained with contact force will not load correctly with a force-disabled observation schema.

Only load checkpoints from trusted sources. PyTorch checkpoints may contain serialized Python objects in addition to model weights.

## 5. Basic Usage

Run from this directory or from the repository root. Example from the repository root:

```powershell
python source/InsertAnything/InsertAnything/tasks/direct/insertanything/sim2real/run_multi_episode_closed_loop.py `
  --config source/InsertAnything/InsertAnything/tasks/direct/insertanything/sim2real/configs/calibration_insertanything_basic.yaml `
  --checkpoint source/InsertAnything/InsertAnything/tasks/direct/insertanything/sim2real/checkpoints/your_policy.pth `
  --server-url http://172.16.0.1:5000 `
  --device cpu `
  --num-episodes 10 `
  --max-steps 300 `
  --policy-period-s 0.0667 `
  --move-wait 0.2 `
  --random-start `
  --rand-x-range 0.05 `
  --rand-y-range 0.05 `
  --rand-z-up-range 0.03 `
  --assume-grasped
```

Contact-force deployment:

```powershell
python source/InsertAnything/InsertAnything/tasks/direct/insertanything/sim2real/run_multi_episode_closed_loop.py `
  --config source/InsertAnything/InsertAnything/tasks/direct/insertanything/sim2real/configs/calibration_insertanything_with_contact_force_template.yaml `
  --checkpoint source/InsertAnything/InsertAnything/tasks/direct/insertanything/sim2real/checkpoints/your_force_policy.pth `
  --server-url http://172.16.0.1:5000 `
  --device cpu `
  --num-episodes 10 `
  --max-steps 300 `
  --policy-period-s 0.0667 `
  --move-wait 0.2 `
  --random-start `
  --rand-x-range 0.05 `
  --rand-y-range 0.05 `
  --rand-z-up-range 0.03 `
  --contact-force-source robot_state.force_K_debug `
  --assume-grasped
```

## 6. Command-Line Arguments

### 6.1 Required Arguments

| Argument | Meaning |
| --- | --- |
| `--config` | Calibration and deployment YAML path. |
| `--checkpoint` | RL-Games checkpoint path. |

### 6.2 Robot Server and Runtime

| Argument | Default | Meaning |
| --- | --- | --- |
| `--server-url` | `http://172.16.0.1:5000` | Robot server base URL. |
| `--device` | `cpu` | Torch inference device. |
| `--num-episodes` | `10` | Number of episodes to run. |
| `--max-steps` | `300` | Maximum policy steps per episode. |
| `--policy-period-s` | `8.0 / 120.0` | Control period in seconds. |
| `--move-wait` | `0.2` | Wait time after non-policy pose moves. |
| `--seed` | `0` | Random seed for episode start and observation bias sampling. |

### 6.3 Randomized Start Pose

| Argument | Default | Meaning |
| --- | --- | --- |
| `--random-start` | disabled | Enables randomized episode start around `T_BPRE_final`. |
| `--rand-x-range` | `0.05` | Uniform start offset range in x, meters. |
| `--rand-y-range` | `0.05` | Uniform start offset range in y, meters. |
| `--rand-z-up-range` | `0.03` | Uniform upward start offset range in z, meters. |
| `--randomize-start-yaw` | disabled | Randomizes start yaw when `--yaw-enable` is also set. |

### 6.4 Gripper and End-of-Run Behavior

| Argument | Default | Meaning |
| --- | --- | --- |
| `--grasp-load-pose6` | built-in lab pose | Pose used by the regrasp flow, `[x y z roll pitch yaw]`. |
| `--open-gripper-endpoint` | `/open_gripper` | Robot-server endpoint for opening the gripper. |
| `--close-gripper-endpoint` | `/close_gripper` | Robot-server endpoint for closing the gripper. |
| `--assume-grasped` | disabled | Skips the initial regrasp flow. |
| `--return-to-load-pose-at-end` | disabled | Moves to `grasp_load_pose6` after the experiment. |
| `--open-gripper-at-end` | disabled | Opens the gripper after the experiment. |

### 6.5 Policy and Action Postprocessing

| Argument | Default | Meaning |
| --- | --- | --- |
| `--obs-norm-eps` | `1e-5` | Observation normalization epsilon used by the Torch policy wrapper. |
| `--ema-factor` | `0.2` | EMA factor for smoothed policy actions. |
| `--raw-action-clip` | `1.5` | Symmetric clipping bound for raw policy actions before EMA. |
| `--yaw-enable` | disabled | Allows the policy yaw action to affect target yaw. |
| `--yaw-abs-limit-deg` | `15.0` | Absolute yaw limit around the reference yaw. |
| `--yaw-step-limit-deg` | `2.0` | Per-step yaw change limit. |

### 6.6 Safety Guard

| Argument | Default | Meaning |
| --- | --- | --- |
| `--step-max-x` | `0.005` | Maximum target displacement in x per policy step. |
| `--step-max-y` | `0.005` | Maximum target displacement in y per policy step. |
| `--step-max-z` | `0.004` | Maximum target displacement in z per policy step. |
| `--hole-box-x` | `0.045` | Half-width of the safety box around the hole in x. |
| `--hole-box-y` | `0.045` | Half-width of the safety box around the hole in y. |
| `--hole-box-z-low` | `0.0` | Lower z bound of the safety box relative to the hole. |
| `--hole-box-z-high` | `0.05` | Upper z bound of the safety box relative to the hole. |

### 6.7 Logging and Optional Observations

| Argument | Default | Meaning |
| --- | --- | --- |
| `--log-dir` | `sim2real_logs` | Root directory for deployment logs. |
| `--contact-force-source` | `none` | `none` or `robot_state.force_K_debug`. |
| `--obs-hole-center-error-annulus-m R_IN R_OUT` | `0.0 0.0` | Samples a per-episode static XY hole-center observation error from an annulus. |

The observation hole-center error only changes what the policy sees. It does not move the physical hole reference used by the postprocessor or safety guard.

## 7. Manual Interaction Flow

The terminal interaction is intentionally operator-driven.

If `--assume-grasped` is not set, startup begins with regrasp:

1. Move to `grasp_load_pose6`.
2. Open the gripper.
3. Wait for `g`, `grasp`, or `ready`.
4. Close the gripper.
5. Wait for `y`, `yes`, or `ok`.

At each episode start:

1. Sample or reuse the start pose around `T_BPRE_final`.
2. Move to the episode start pose.
3. Wait for `start`, `go`, or `g`.
4. Reset policy recurrent state, action EMA state, and previous actions.

During closed-loop execution, the operator can type:

| Command | Meaning |
| --- | --- |
| `s` or `success` | Mark the current episode as successful. |
| `f` or `fail` | Mark the current episode as failed. |
| `a` or `abort` | Abort the current episode. |
| `q` or `quit` | Abort the current episode and then choose whether to end the run. |

After each episode, the robot retreats to `T_BPRE_final`. The operator then chooses:

| Command | Meaning |
| --- | --- |
| `c` or `continue` | Continue to the next episode. |
| `r` or `regrasp` | Run the regrasp flow before the next episode. |
| `q` or `quit` | End the experiment. |

## 8. Logs

Each run creates:

```text
<log-dir>/run_YYYYMMDD_HHMMSS/
```

Files:

| File | Contents |
| --- | --- |
| `steps.jsonl` | One JSON record per policy step. Includes robot state, observation components, policy outputs, postprocessor output, guard output, target command, and clip flags. |
| `episodes.csv` | One row per episode. Includes result, step count, duration, observation hole-center error, and clip counts. |
| `config_snapshot.yaml` | CLI arguments, policy config, postprocessor config, safety config, and the raw calibration YAML. |

These logs are intended for debugging deployment differences between simulation and the real robot.

## 9. Safety Notes

Real-robot deployment must be supervised. Before running:

1. Verify the robot server controls the expected robot.
2. Verify `/pose` accepts `[x, y, z, roll, pitch, yaw]` in meters and radians.
3. Verify the robot is in a stable Cartesian impedance control mode.
4. Start with conservative `--step-max-*` and `--hole-box-*` values.
5. Run a low-risk dry test above the hole before allowing contact.
6. Keep an emergency stop available.

The safety guard clips commanded targets, but it is not a replacement for a reliable low-level controller, workspace limits, collision avoidance, or human supervision.

## 10. Adapting to Another Franka Interface

To use `panda-py` or another Franka interface, implement the same logical contract:

1. A state reader that returns `pose_BA`, `q`, `dq`, `vel_A`, and optional `force_K_debug`.
2. A Cartesian pose sender that accepts `pose6 = [x, y, z, roll, pitch, yaw]`.
3. Gripper open and close functions.
4. A stable Cartesian impedance control loop underneath the pose sender.

The smallest adaptation path is:

1. Modify `robot_state_adapter.fetch_robot_state()` so it reads from your backend instead of HTTP.
2. Modify `run_multi_episode_closed_loop.send_pose6()` so it sends a Cartesian target through your backend.
3. Modify `open_gripper()` and `close_gripper()` if your gripper interface is not HTTP-based.
4. Keep `observation_builder.py`, `action_postprocessor.py`, and `safety_guard.py` unchanged unless your coordinate conventions differ.

After adapting the backend, validate in this order:

1. Print one converted `RobotState` and verify pose units and quaternion order.
2. Send one small safe Cartesian move through `send_pose6()`.
3. Run with `--num-episodes 1 --max-steps 1`.
4. Inspect `steps.jsonl` before running longer closed-loop trials.
