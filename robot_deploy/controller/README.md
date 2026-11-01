# controller/ (runs ON the robot)

Status 2026-11-01: v11 (stand + turn in place) controller written, **tested on the laptop only, never run on the robot**. Synced to `~/g1_deploy/controller/` by `scripts/robot_sync.sh` (dry-run first; `--apply` needs the user's yes).

| file | role |
|---|---|
| `observation.py`, `policy.py`, `commands.py` | simulation session: 78-value observation, numpy network, command deadband + ramp (contract test passes) |
| `state.py` | `DdsLink` (rt/lowstate in, rt/lowcmd out, foreign-publisher detection), `ReplayState` (laptop), `NullPublisher` |
| `safety.py` | limits and checks: stale state, NaN, tilt 25 deg, speed 5 rad/s, per-motor torque caps, tracking error 0.30 rad, motor temperature, loop period |
| `run.py` | phase machine RAMP, HOLD, BLEND, RUN, STOP; `--plan`, dry run (default), `--publish` |

```
python3 run.py --plan                                   # preflight + the plan text for approval; publishes nothing
python3 run.py                                          # DRY RUN: logs the commands it WOULD send to ~/g1_recordings/controller/<stamp>/log.csv
CONTROLLER_CONFIRM=yes-<stage> python3 run.py --publish --stage <stage>   # publishes rt/lowcmd: only after the user's yes for THIS run
```
Tests (laptop, no robot): `tests/contract_test.py` (vs Isaac Lab), `tests/real_state_check.py` (real standing recording), `tests/test_runner.py` (phases, every safety trip -> stop sequence,
dry run sends nothing, publish refused without confirmation; also mutation-checked: removing the kd damping, the tilt check or the invalid-state stop makes it fail).
Known gaps: damping from PC2 is unverified in debug mode; after publishing ends the motors go to mode 0 (limp); torque caps and the tracking limit are proposals to review per run;
the gait clock starts when BLEND starts; stage B yaw commands need `--allow-turn` and are clamped at 0.5 rad/s.

(Original plan below, kept for the parts not written yet.)

## Constraints from the robot (checked 2026-10-29)
- Jetson, Linux 5.10.104-tegra, **Python 3.8.10**, PyTorch 2.4.1 (CPU-only build: `cuda.is_available()` False), `unitree_sdk2py` installed.
- Write Python 3.8-compatible code. The policies are small MLPs: running them with numpy on CPU is enough and avoids
  torch/CUDA concerns inside a 20 ms loop. **Measured 2026-10-29:** single-threaded numpy ~0.3 ms per call at 50 Hz (set OPENBLAS_NUM_THREADS=1), TensorRT FP32 ~1 ms,
  torch CPU 1.5 ms mean with a 48 ms worst case. See `docs/open_issues.md`.

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
