# Task for the hardware session: standing-only deployment baseline

Source: the simulation session, 2026-10-31, at the user's request. Status: **NOT STARTED**. The user decides when to start and
gives the explicit yes for every run that publishes `rt/lowcmd` (`CLAUDE.md`, `bringup_checklist.md`).

## Goal
Run ONE standing policy on the real robot, standing only (no walking, no selector), to establish a deployment baseline and
verify the whole path: export -> sync -> state read -> observation build -> policy -> command publish -> safe stop.
The result is a measured comparison of real against simulation, not a walking demo.

## Which policy
Both candidates have the same contract (47 observations -> 12 leg actions), so the same controller code runs either.

| | v10 (recommended first) | v1 (the user's choice for the baseline) |
|---|---|---|
| Checkpoint | `logs/lab_checkpoints/v10/model_1499.pt` (rsl-rl 5 format; lab: `logs/rsl_rl/g1_v10_stand_pose/2026-10-30_08-59-54/`) | `logs/lab_checkpoints/v1/model_1499_rslrl5.pt` (already exported in `policies/stand_v1_20261029.*`) |
| Trained on | the real plant: Unitree gains, waist roll/pitch locked, base-height + pose reward | the original sim plant (leg kd 2, ankle kp 40) |
| Sim behaviour at real gains | base 0.783 m, knee 0.28 rad, knee torque 2.4 N*m, 0 foot lifts, 0.5 cm drift (arms at rest) | base 0.79 m, **knee -0.08 rad (the joint stop is -0.087) with ~42 N*m commanded into it**, 0 lifts, 1.3 cm drift |
| Known weakness | fails with large arm poses (not part of this task); ~89% survive shoves up to 0.7 m/s | ~48% survive shoves at the real gains |

Reason for the recommendation: v1 stands by locking its knee at the stop with a large torque. That is a risk on hardware (a joint
pressed at its stop) and it is not representative of a good standing posture. v10 uses a normal stance and low torque. Export v10
with `scripts/export_policy.py` (same as v1) and run it first; run v1 afterwards in the same pipeline if the user wants the comparison.

## Fixed parameters (from `configs/g1_walk_stand.yaml` and the sim configs)
- 50 Hz control loop, publish at 50 Hz only. Command = default pose + 0.25 * action, PD with the real gains (legs kp 100/100/100/150/60/60,
  kd 4/4/4/5/3/3).
- Command input is always [0, 0, 0] for this task. Gait clock input: [sin, cos] of 2*pi*(t mod 0.8)/0.8 (for the standing policy too).
- The policy drives the 12 leg joints only. The controller must also HOLD every other live motor at its default with the gains the sim
  used for held joints: waist yaw 200/6; arms 60/3; wrists 40/3 (waist roll/pitch are locked/absent: skip motors 13, 14). Hands are separate.
- Observation (47): gyro*0.25 (3), projected gravity (3), command*[2,2,0.25] (3), leg joint pos - default (12), leg joint vel*0.05 (12),
  last action (12), gait clock (2). No normaliser. Deterministic output = the actor mean.
- The robot's onboard gyro has a small bias (about 0.004 rad/s measured): do not "correct" it in the observation.

## Stages (each needs its own dry-run plan and the user's yes; stop at the first surprise)
0. **Contract test, laptop only.** Replay recorded simulation observations (from `logs/eval` runs or a new dump) through the controller's
   observation code and the exported network; the actions must match the torch actor to <= 1e-4. Also feed the real `rt/lowstate`
   recording from `measurements/20261029/standing_lowstate_raw.csv` and check the observation ranges are plausible
   (gravity near [0,0,-1], joint velocities small, no NaN).
1. **Robot, read-only dry run** (hoisted, zero torque, nothing published): run the controller with `--dry-run`, log observations and
   the actions the policy WOULD send. Check the actions are bounded and near zero for a near-default pose, and the loop time is well
   inside 20 ms (numpy single-threaded: OPENBLAS_NUM_THREADS=1).
2. **Hoisted, feet off the ground, commands enabled.** Ramp from the current pose to the default pose over >= 3 s, hold, then
   blend the policy's actions in over >= 5 s (0 -> 1). Watch joint torques and oscillation. Kill the controller mid-run: the robot must
   go to a safe damping state (`estop_damping.py` path). Abort on: any joint speed > 5 rad/s, torque above a limit set in the dry-run
   plan, tilt > 25 deg, NaN, or loop overrun.
3. **Feet touching down, hoist taking weight, then hoist slack.** Standing policy only. Slowly transfer weight (the user decides the
   steps); an assistant holds the robot. Abort on the same limits.
4. **Free standing for a short, fixed time** (start with 10 s, then 30 s), hoist or spotter still in reach. No pushes in this task.

## Measure and report (compare each number with simulation)
| Quantity | v10 in sim | How to measure |
|---|---|---|
| Base height | 0.783 m | not measured directly: use leg joint angles / forward kinematics |
| Knee angle, standing | 0.28 rad | `rt/lowstate` |
| Knee torque, standing | 2.4 N*m | `tau_est` |
| Ankle pitch angle / torque | -0.21 rad (default -0.20) | `rt/lowstate` |
| Foot lifts | 0 | contact / joint motion |
| Tilt (IMU) | < 3 deg RMS | IMU |
| Drift / sway | ~0.5 cm in 20 s | state estimate or markings on the floor |
| Loop time and command latency | 3-8 ms at 50 Hz | controller log |

Also record: the first stage where reality differed from sim, any joint that saturated, how the robot recovered from a gentle
manual disturbance at stage 3 (if the user allows it), and whether the policy output chatters (action rate).

## Deliverables
- `controller/` code that runs the stages above with `--dry-run` as the default and an explicit flag to publish.
- `measurements/<date>/standing_baseline/` with logs, a short `REPORT.md` section and the table above filled in.
- A list of surprises and open questions for the simulation session (e.g. differences in joint offsets, IMU sign, gain effects).

## Not part of this task
Walking, the velocity-command selector, pushes, arm motion while standing (a later manipulation policy), hands, mass and plant
identification (see `hardware_requests_from_sim.md`).
