# InsertAnything Single-Hole Tasks

This directory contains the Isaac Lab implementation for force-conditioned, contact-rich peg-in-hole insertion. The released simulation tasks use one fixed hole and one held peg per environment. Each task is configured for a single geometry and tolerance; no task-specific sampler or task identity code is required.

## Registered tasks

| Gym ID | Geometry | Tolerance |
| --- | --- | --- |
| `InsertAnything-CircleHole-I-Direct-v0` | Circle | I |
| `InsertAnything-SquareHole-II-Direct-v0` | Square | II |
| `InsertAnything-LHole-III-Direct-v0` | L-shape | III |
| `InsertAnything-TriangleHole-IV-Direct-v0` | Triangle | IV |
| `InsertAnything-HexagonHole-III-Direct-v0` | Hexagon | III |
| `InsertAnything-HexagonHole-IV-Direct-v0` | Hexagon | IV |

The task classes are defined in `insertanything_tasks_cfg.py`, environment classes in `insertanything_env.py`, and environment registrations in `__init__.py`.

## Observations

The actor observation is built from the task's `obs_order`, optional fingertip force feedback, and the six-dimensional previous action:

- `fingertip_pos_rel_fixed`: fingertip position relative to the target hole
- `fingertip_quat`: fingertip orientation
- `fingertip_quat_rel_fixed`: relative orientation when enabled by the task
- `ee_linvel`, `ee_angvel`: fingertip linear and angular velocity
- `contact_force`: compact three-axis force feedback when enabled
- `prev_actions`: previous six-dimensional action

The critic additionally receives privileged simulation quantities specified by `state_order`. The observation dimension is computed at runtime, so the printed environment summary should be checked before loading a checkpoint.

## Training

Run training from the repository root:

```bash
python scripts/rl_games/train.py --task InsertAnything-CircleHole-I-Direct-v0 --num_envs 128 --headless
```

Replace the task ID to train another single-hole configuration. The corresponding RL-Games configuration files are in `agents/`:

| File | Task |
| --- | --- |
| `rl_games_ppo_circle_cfg.yaml` | Circle |
| `rl_games_ppo_square_cfg.yaml` | Square |
| `rl_games_ppo_L_cfg.yaml` | L-shape |
| `rl_games_ppo_triangle_cfg.yaml` | Triangle |
| `rl_games_ppo_hexagon_cfg.yaml` | Hexagon |

## Evaluation

A headless evaluation command is:

```bash
python scripts/rl_games/play.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 128 \
  --headless \
  --checkpoint source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-LHole-III-Direct-v0.pth
```

The terminal reports the success rate and average success time for each completed episode. For a visual single-environment run, omit `--headless` and use `--num_envs 1`.

## Configuration

The most commonly adjusted task fields are:

| Field | Purpose |
| --- | --- |
| `fixed_asset_cfg` | USD asset and geometry parameters for the fixed hole |
| `held_asset_cfg` | USD asset and geometry parameters for the held peg |
| `fixed_asset_init_pos_range` | Initial target-position randomization |
| `fixed_asset_init_orn_range_deg` | Initial target-yaw randomization |
| `hand_init_pos_range`, `hand_init_orn_range` | Initial end-effector randomization |
| `contact_force` | Force observation, reward, and logging options |
| `dr_randomize_dynamics` | Dynamics randomization switch |
| `obs_order` | Actor observation order before automatic force and action fields |

The four tolerance levels used by the released single-hole assets are defined in the task configurations and assets. They are not encoded as an additional observation.

## Contact-force logging

For a task with `contact_force["log_contact_force"] = True`, CSV logging is enabled in evaluation mode with one environment:

```bash
python scripts/rl_games/play.py \
  --task InsertAnything-LHole-III-Direct-v0 \
  --num_envs 1 \
  --headless \
  --checkpoint source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-LHole-III-Direct-v0.pth
```

Logs are written to `contact_force_logs/`, which is ignored by git.

## Real-robot deployment

The real-robot closed-loop runner is:

```text
source/InsertAnything/InsertAnything/tasks/direct/insertanything/sim2real/run_multi_episode_closed_loop.py
```

Its deployment configuration and observation builder use the same single-hole observation schema as the simulation policy. The runner requires a calibrated target pose, a compatible Franka control server, and the sensor/control interfaces described in `sim2real/README.md`.

## Source layout

| Path | Role |
| --- | --- |
| `assets/` | Released single-hole USD assets |
| `agents/` | RL-Games training configurations |
| `checkpoints/` | Optional local checkpoint files and download notes |
| `sim2real/` | Real-robot observation, policy, safety, and closed-loop execution code |
| `insertanything_env.py` | Isaac Lab environment |
| `insertanything_env_cfg.py` | Environment and observation configuration |
| `insertanything_tasks_cfg.py` | Geometry, reward, and randomization configurations |
