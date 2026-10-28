# Plan: train a G1 walking policy in Isaac Lab

Goal: replace Unitree's pretrained `motion.pt` (trained on a 12-DOF legs-only
model in Isaac Gym) with a policy trained on **our** robot model in Isaac Lab,
with the end goal of **deployment on the real G1 EDU**.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Env base class | `ManagerBasedRLEnv` | Isaac Lab's stock G1 velocity task and `unitree_sim_isaaclab` are both manager-based; reward/event/curriculum terms are reusable and swappable. |
| Training approach | From scratch with RSL-RL PPO; `motion.pt` is the baseline, not a starting point | `motion.pt` is a recurrent TorchScript actor with no critic or optimizer state, and RSL-RL's default policy is an MLP, so PPO cannot resume from it. Distillation is a fallback only if scratch training stalls. |
| Robot model | Full Inspire USD (`g1_29dof_with_inspire_rev_1_0.usd`), 53 DOF | Matches what we simulate and deploy. The 12-DOF vs full-body mass mismatch is a suspected cause of the current falls. |
| Actions | 12 leg joints only; arms, waist and hands held at the standing pose | Same split as `g1_walking/leg_controller.py`. |
| Real-robot deployment | Yes, a goal | Actuator, latency and sensor-noise randomization are in from the start, not added later. |

## Baseline (measured, current setup)

`run_walking.py` with Unitree's `motion.pt`, 50 Hz control (physics dt 0.005 s):

| Command vx | Result |
|---|---|
| 0.0 | upright 10 s |
| 0.3 | upright 15 s, 0.16 m/s average |
| 0.5 | falls at 7.0 s, 0.30 m/s average |

Observed: base sags to ~0.65-0.75 m from 0.79 m; actions reach 3-5. These are
the numbers the new policy must beat.

## Task design (`g1_rl/`, to be created in this repo)

Register a gym id such as `G1-Walk-Flat-Inspire-v0`.

- **Robot cfg:** reuse `G129_CFG_WITH_INSPIRE_WHOLEBODY` from
  `unitree_sim_isaaclab/robots/unitree.py` (real actuator groups and gains).
- **Timing:** `sim.dt = 0.005`, `decimation = 4` -> 50 Hz control.
- **Actions:** joint-position offsets on the 12 leg joints around the default
  pose, scaled (0.25 in the current port). Other joints held at the standing
  pose from `g1_sim/config.py::StandingPoseConfig`.
- **Actor observations (proprioceptive only; must be obtainable on hardware):**
  base angular velocity (body frame), projected gravity, velocity command,
  joint positions relative to default, joint velocities, last action, and
  optionally a gait-phase clock. Exclude base linear velocity (the stock task
  includes it; the real robot cannot measure it).
- **Critic observations (privileged, training only):** base linear velocity,
  contact forces, randomized physical parameters (asymmetric actor-critic).
- **Rewards:** start from the stock G1 terms (velocity tracking, feet air time,
  foot slide, joint-limit and deviation penalties, termination penalty); add
  action-rate and torque penalties, since jerky policies transfer poorly.

## Sim-to-real robustness (included from the start)

- Observation noise on gyro, gravity and joint readings
- Action delay / latency (randomized, 0-2 control steps)
- Actuator randomization: PD gain scaling, motor strength, friction, armature
- Randomized base mass and COM offset (covers hand/arm payload differences)
- Ground friction and restitution randomization
- Random push disturbances
- Randomized initial pose and joint offsets

Exact ranges are to be set conservatively first and widened when training is
stable; record them with each run.

## Phases

0. **Baseline** - done (table above).
1. **Validate pipeline:** run stock `Isaac-Velocity-Flat-G1-v0` with RSL-RL
   headless for a few hundred iterations. Confirms install, VRAM budget
   (RTX 3060, 6 GB: expect ~1-2k envs rather than 4k) and training speed.
2. **Custom task package** per the design above.
3. **Train in stages:** flat ground -> rough terrain with terrain curriculum
   -> wider command ranges. Keep every stage's checkpoint and TensorBoard log.
4. **Contract test before deployment:** record observations from the training
   env, feed the same raw state through `ObservationBuilder`, assert they match
   numerically (catches joint order, scale and sign bugs). Export with the
   RSL-RL exporter (JIT/ONNX).
5. **Evaluate against fixed criteria** (new policy and `motion.pt`, 20+ seeds):
   - survival rate over 30 s at vx in {0.2, 0.5, 0.8}, plus lateral and yaw
   - velocity-tracking error
   - recovery from random pushes
   - torque and action smoothness (and action-rate)
6. **Sim-to-sim:** run the exported policy in standalone `run_walking.py`
   (different code path from the training env) and in `unitree_mujoco`
   (different physics engine).
7. **Integration:** connect to `/cmd_vel` per `docs/architecture.md`.
8. **Hardware bring-up:** staged and cautious - gantry/harness first,
   zero-command standing, then low-speed commands; confirm IMU frame and
   joint-order conventions against the `unitree_sdk2` `LowState` before any
   free-standing test.

## Risks

- Training and the GUI sim compete for the 6 GB GPU; train headless and do not
  run `run_walking.py` concurrently.
- Joint ordering differs between Isaac Lab's articulation and
  `LEG_JOINT_NAMES`; the contract test (phase 4) is the guard.
- Reward tuning will take several runs.
- Passing phases 5-6 does not guarantee hardware success; hardware bring-up
  (phase 8) needs a safety setup (harness, e-stop).
- The standing-pose gains for non-leg joints must match the real robot's
  low-level gains, or the held arms/waist will behave differently on hardware.

## Related

- `docs/architecture.md` - planner + locomotion policy split
- `g1_walking/` - current Unitree-policy port (the baseline and the deployment
  code the new policy will plug into)
