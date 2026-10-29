# Open issues before any hardware run

Everything here is a known mismatch or an assumption that has not been validated on the robot.
Resolve or consciously accept each item before the step it blocks. Update this file whenever you learn something.

## Blocking: the controller does not exist yet
- `controller/` is empty. Needed: LowState reader, observation builder (layout in `configs/g1_walk_stand.yaml`),
  the two policies, the selector, LowCmd writer, a state machine like Unitree's (zero torque to default pose to control),
  a dry-run mode (log actions, send nothing), and a watchdog (stop and go limp on stale state or NaN).
  Reference: `external_refs/unitree_rl_gym/deploy/deploy_real/deploy_real.py`.
- Policies are not exported yet (`policies/README.md`).

## Gain mismatches between our simulation and Unitree's own deploy config
The legs match exactly. Waist and arms do not:

| joints | trained in simulation (kp / kd) | Unitree deploy_real (kp / kd) |
|---|---|---|
| waist yaw, roll, pitch | 200 / 5 | **300 / 3** |
| wrist roll, pitch, yaw | 40 / 2 | **20 / 1** |
| shoulders, elbow | 100/2 pitch, roll; 50/2 yaw, elbow | same |

The walking policy actively drives the waist (v4), so the waist gain difference is a real sim-to-real error.
Decide which values the robot will run and, if they differ from training, re-train or at least evaluate in simulation with them.

## Policies: known behaviour (simulation only)
- Walking policy (v4): steps in place at zero command (about 4 foot lifts/s, ~3 cm/s drift) and never has a slow-walk gait;
  below ~0.3 m/s it marches. Speed tracking is accurate otherwise. It was never trained or evaluated on non-flat floors.
- Standing policy (v1): stands without stepping, but 38% of envs fall in three shoves up to 0.7 m/s, only 62% survive
  with the arms held far forward, and it has no waist output (waist held at default) and no arm observation.
- The selector has only been smoke-tested (`scripts/evaluate_switch.py`); thresholds are provisional.

## Unvalidated assumptions
- **Observation noise** in training (gyro +/-0.2, gravity +/-0.05, joint pos +/-0.01, joint vel +/-1.5, encoder bias +/-0.015 rad
  on joint positions) was chosen from reference repos, not measured on this robot.
- **Latency**: training randomises 0-4 physics steps (0-20 ms) of actuation delay. Real round-trip latency over DDS is unmeasured.
- **IMU**: Unitree's config uses the pelvis IMU (`imu_type: pelvis`). The sign/frame conventions of gyro and quaternion
  must be checked against the simulation's body-frame angular velocity and projected gravity before closing the loop.
- **Joint order / zero offsets**: motor indices 0-28 follow Unitree's deploy config; verify at runtime that joint
  signs and zero positions match the default pose (print `q` in the default pose and compare).
- **Torque limits**: simulation uses ankle 50, knee 139, hip roll 139, hip pitch/yaw 88, waist yaw 88, waist roll/pitch 35 N*m.
  Confirm against the real motor limits.
- **Arms**: the walking policy only observes them. Whatever commands the arms must keep them inside the poses it
  saw in training (`g1_rl/body_env_cfg.py: ARM_POSE_RANGES`, one-sided shoulder roll) and tucked when passing a 50 cm door
  (the robot is ~0.37 m wide with arms at the sides; standing pose width not measured).
- **Command limits**: the proposed planner envelope is in the config; the true safe envelope on hardware is unknown.
  Unitree's own deploy limit is vx 0.8, vy 0.5, wz 1.57.
- **Hardware has never run a g1_rl policy.** Expect the first runs to be dry-run and hoisted only.
