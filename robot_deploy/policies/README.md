# policies/ (exported, deployed to the robot)

Empty for now. Synced to `~/g1_deploy/policies/` by `scripts/robot_sync.sh`.

## Manifest (fill in when exporting)

| file | policy | trained from | obs to actions | format | exported on |
|---|---|---|---|---|---|
| _(none yet)_ | walking "v4" | lab `logs/rsl_rl/g1_walk_flat_inspire_arms/2026-10-29_03-22-08/model_2999.pt` | 84 to 15 | | |
| _(none yet)_ | standing "v1" | laptop `logs/rsl_rl/g1_walk_flat_inspire/2026-10-28_20-48-49/model_1499.pt` (lab copy: `logs/ckpt_v1/model_1499_rslrl5.pt`) | 47 to 12 | | |

## Notes for exporting
- Only the **actor** is needed (deterministic mean, no exploration noise); both are `MLP in-256-128-128-out`, ELU, no
  observation normaliser. `scripts/evaluate_switch.py: load_actor()` already rebuilds exactly this network from either
  checkpoint format (rsl-rl 3.x laptop, 5.x lab); an exporter can reuse it.
- Preferred formats: a dependency-free `.npz` of the weight matrices (run with numpy), or TorchScript (`torch.jit.save`).
  The robot has PyTorch 2.4.1 and Python 3.8.
- Name files `<role>_<version>_<yyyymmdd>.<ext>`, record the checkpoint it came from in the table above, and verify the exported
  network reproduces the simulation actor's outputs on recorded observations before copying it to the robot.
- The checkpoints are about 1.5-1.8 MB each, small enough to commit; decide whether to track them with the repo.
