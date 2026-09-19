---
license: bsd-3-clause
library_name: rl-games
tags:
  - isaac-lab
  - reinforcement-learning
  - robotics
  - peg-in-hole
  - contact-rich-manipulation
  - tactile-sensing
  - sim-to-real
  - insertanything
---

# InsertAnything Checkpoints

Pretrained RL-Games policies for the single-hole precision-insertion environments released with **InsertAnything: Generalizable Contact-Rich Precision Insertion from Simulation to Reality**.

The policies were trained entirely in Isaac Lab. They use deployable robot observations together with a compact three-axis fingertip force signal for contact-aware motion correction. The files are intended for evaluation, visualization, and reproducibility checks with the accompanying InsertAnything codebase.

## Released files

| Checkpoint | Isaac Lab task | Observation dim | Configuration | SHA256 |
| --- | --- | ---: | --- | --- |
| `InsertAnything-LHole-III-Direct-v0.pth` | `InsertAnything-LHole-III-Direct-v0` | 26 | L-shape, Tol. III, fingertip force observation, state-independent policy standard deviation | `FC158B0D72AFDAB0D860CEB379F8E3EC2B0B693D49C9B24629608A5BEA4B58AC` |
| `InsertAnything-HexagonHole-III-Direct-v0.pth` | `InsertAnything-HexagonHole-III-Direct-v0` | 26 | Hexagon, Tol. III, fingertip force observation, state-independent policy standard deviation | `D11D9A4849238DBDEC4602EFB86DF6D260118FB7CA59E0F159A05546D2B14C08` |

The checkpoint and task names must match because geometry, observation construction, symmetry handling, and control settings are defined by the registered Isaac Lab task.

## Download

Install the Hugging Face CLI and download the files from the root of the InsertAnything repository:

```powershell
python -m pip install -U huggingface_hub

hf download tOMRmA/InsertAnything-checkpoints InsertAnything-LHole-III-Direct-v0.pth `
  --repo-type model `
  --local-dir source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints

hf download tOMRmA/InsertAnything-checkpoints InsertAnything-HexagonHole-III-Direct-v0.pth `
  --repo-type model `
  --local-dir source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints
```

## Evaluation

Install the InsertAnything extension and its Isaac Lab dependencies before running these commands.

### L-shape Tol. III

```powershell
python scripts/rl_games/play.py `
  --task InsertAnything-LHole-III-Direct-v0 `
  --num_envs 128 `
  --headless `
  --seed 0 `
  --max_episodes 3 `
  --checkpoint source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-LHole-III-Direct-v0.pth
```

With Isaac Sim 5.0, seed 0, and 128 environments, the release validation produced episode success rates of 97.66%, 96.88%, and 94.53% (370/384 trials; aggregate: 96.35%). The corresponding average success times were 5.95 s, 6.65 s, and 6.08 s.

### Hexagon Tol. III

```powershell
python scripts/rl_games/play.py `
  --task InsertAnything-HexagonHole-III-Direct-v0 `
  --num_envs 128 `
  --headless `
  --seed 0 `
  --max_episodes 3 `
  --checkpoint source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/InsertAnything-HexagonHole-III-Direct-v0.pth
```

Under the same protocol, the release validation produced episode success rates of 94.53%, 98.44%, and 96.88% (371/384 trials; aggregate: 96.61%). The corresponding average success times were 5.64 s, 5.57 s, and 4.99 s.

Results can vary with the simulator version, hardware, and random seed.

## Integrity check

```powershell
Get-FileHash `
  source/InsertAnything/InsertAnything/tasks/direct/insertanything/checkpoints/*.pth `
  -Algorithm SHA256
```

Compare the returned hashes with the table above before evaluation.

## Intended use and limitations

- These checkpoints are provided for the corresponding InsertAnything simulation tasks and research use.
- They are not standalone models: task configuration, assets, observation construction, and controller code are required from the InsertAnything repository.
- Real-robot deployment additionally requires calibration, compatible robot and tactile-sensor interfaces, safety limits, and operator supervision.
- Performance on modified assets, simulator settings, observation schemas, or physical systems is not guaranteed by the released simulation validation.

## License

The checkpoints are released under the BSD 3-Clause license.
