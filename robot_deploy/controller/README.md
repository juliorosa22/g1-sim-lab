# controller/ (runs ON the robot)

Not written yet. Synced to `~/g1_deploy/controller/` by `scripts/robot_sync.sh`.

## Constraints from the robot (checked 2026-10-29)
- Jetson, Linux 5.10.104-tegra, **Python 3.8.10**, PyTorch 2.4.1 (CUDA availability not checked), `unitree_sdk2py` installed.
- Write Python 3.8-compatible code. The policies are small MLPs: running them with numpy on CPU is enough and avoids
  torch/CUDA concerns inside a 20 ms loop (decide after measuring).

## Planned shape
```
controller/
  run.py              entry: python run.py <net_interface> --config ../configs/g1_walk_stand.yaml [--dry-run]
  state.py            LowState to joint q/dq, IMU gyro + gravity, in policy joint order
  observation.py      builds the 84- and 47-value observations exactly as the configs/yaml says
  policy.py           loads the two exported MLPs
  selector.py         hysteresis, debounce, command ramp, action blend (mirror of scripts/evaluate_switch.py)
  safety.py           watchdog, NaN/limit checks, command clamp and deadband, state machine
  commands.py         planner velocity input (ROS2 or a test script)
```
Reference implementation: `external_refs/unitree_rl_gym/deploy/deploy_real/deploy_real.py` (state machine
zero torque, default pose, control; uses `rt/lowstate` and `rt/lowcmd`, message type `hg`).

## Requirements for the first version
1. `--dry-run` is the default: compute and log everything, publish nothing to `rt/lowcmd`.
2. Sending commands requires an explicit flag AND the user's confirmation in the session (see `../CLAUDE.md`).
3. Observation code must be tested against the simulation: replay recorded simulation states through it and
   compare to the observations the training environment produced (the "contract test" of the training plan).
4. Same selector logic as `scripts/evaluate_switch.py`; the command deadband (0.05) and the gait-clock rules
   from the config must match what the policies were trained with.
