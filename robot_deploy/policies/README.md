# policies/ (exported, deployed to the robot)

Exported 2026-10-29 (ONNX + npz + test vectors; `manifest.json` has hashes and verification). Synced to `~/g1_deploy/policies/` by `scripts/robot_sync.sh`.

## Manifest (fill in when exporting)

| file | policy | trained from | obs to actions | format | exported on |
|---|---|---|---|---|---|
| `walk_v4_20261029.{onnx,npz}` (+ `_testvec.npz`) | walking "v4" | laptop copy `logs/rsl_rl/g1_walk_flat_inspire_stand6/warm_start_v4/model_2999.pt` (iteration 2999, 84-256-128-128-15; the file the v6 run warm-starts from; hash vs the lab file `.../arms/2026-10-29_03-22-08/model_2999.pt` not compared) | 84 to 15 | ONNX opset 13, fp32 + numpy npz | 2026-10-29 |
| `stand_v1_20261029.{onnx,npz}` (+ `_testvec.npz`) | standing "v1" | laptop `logs/rsl_rl/g1_walk_flat_inspire/2026-10-28_20-48-49/model_1499.pt` | 47 to 12 | ONNX opset 13, fp32 + numpy npz | 2026-10-29 |

## Notes for exporting
- Only the **actor** is needed (deterministic mean, no exploration noise); both are `MLP in-256-128-128-out`, ELU, no
  observation normaliser. `scripts/evaluate_switch.py: load_actor()` already rebuilds exactly this network from either
  checkpoint format (rsl-rl 3.x laptop, 5.x lab); an exporter can reuse it.
- Preferred formats: a dependency-free `.npz` of the weight matrices (run with numpy), or TorchScript (`torch.jit.save`).
  The robot has PyTorch 2.4.1 and Python 3.8.
- Name files `<role>_<version>_<yyyymmdd>.<ext>`, record the checkpoint it came from in the table above, and verify the exported
  network reproduces the simulation actor's outputs on recorded observations before copying it to the robot.
- The checkpoints are about 1.5-1.8 MB each, small enough to commit; decide whether to track them with the repo.

## Running them on the robot (measured 2026-10-29, see `docs/open_issues.md`)
- Export with `scripts/export_policy.py` (laptop, `isaaclab` conda env). Verify against the torch actor before copying.
- On the Jetson, single-threaded numpy needs ~0.3 ms per call at 50 Hz: set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (default threads stalled up to 168 ms on the busy Jetson).
- TensorRT works (engine built on the robot with `trtexec --noTF32`, ~1 ms per call paced at 50 Hz) but gives no latency benefit for these MLPs.
- The retrained walking policy (no waist roll/pitch) will have a different output size: re-run the exporter and re-verify.
