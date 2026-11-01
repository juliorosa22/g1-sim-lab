# Task for the hardware session: first trial = v11 standing, then turning in place

Source: the simulation session, 2026-11-01, updated at the user's request (the first version used v10 and standing only).
Status: **NOT STARTED**. The user decides when to start and gives the explicit yes for every run that publishes `rt/lowcmd`
(`CLAUDE.md`, `bringup_checklist.md`).

## Goal
Run ONE policy on the real robot, **v11 (stand + turn in place)**, to establish the deployment baseline and verify the whole path:
export -> sync -> state read -> observation build -> policy -> command publish -> safe stop.
Stage A is standing at zero command. Stage B is yaw-only commands (turning in place). **No translation, no walking, no v12, no selector** in this task.
The result is a measured comparison of real against simulation, not a demo.

## Why v11 (simulation evidence, 128 envs unless noted, noise off)
| | v11 | v10 (the earlier candidate) |
|---|---|---|
| Standing posture | base 0.784 m, knee 0.288 rad, knee torque 3.5 N*m | base 0.783 m, knee 0.281, 2.4 N*m |
| Turn in place, achieved yaw rate for a 0.3 / 0.5 / 1.0 / -0.5 rad/s command | 0.34 / 0.53 / 1.00 / -0.51 | 0.01 / 0.01 / 0.44 / -0.01 (does not turn) |
| Planted again after a 3 s turn (0 foot lifts, <5 cm, <30 mm/s) | 100 / 100 / 98 / 98 %, settles in 1.2-2.1 s | 95 / 91 / 84 / 92 % |
| While turning | wanders 6-10 cm, ~240-260 foot lifts/min (it steps around) | n/a |
| Standing with the arms moving (12 scenarios, 64 envs, 20 s) | 0.3-2.2 cm drift, 0-0.2 lifts/min, 100% survive | 0.5 cm at rest but 770 cm with arms held back, 41 cm with random reaches |
| Shoves 0.3 / 0.5 / 0.7 m/s, arms at rest | 100% survive, calm again in ~1.2 s | 81-89% survive |
| Low-speed commands | yaw 0.2 -> 0.26, forward 0.2 -> 0.23, sideways 0.2 -> 0.19 | ignores them |

**Not yet verified in simulation:** v11 with observation noise on (the same scenarios), and nothing here has run on hardware.
v10 stays the simpler fallback (47 observations -> 12 actions, no arm or waist inputs) if the 78-value interface below causes trouble.

## The interface (different from v10 and v1: read this)
- Network: MLP 78-256-128-128-13, ELU, no normaliser, deterministic output = the actor mean. Checkpoint (rsl-rl 5 format):
  `logs/lab_checkpoints/v11/model_5997.pt`, task `G1-Walk-Flat-Inspire-V11-v0`, trained from the v8 interface. Export with `scripts/export_policy.py`
  (`--role stand_turn --version v11`) and verify against the torch actor on recorded observations.
- 50 Hz control loop; publish at 50 Hz only. Command = default + scale * action, PD with the real gains.
- **Observation, 78 values, in this order:**
  1. gyro (pelvis IMU, body frame) x 0.25 (3). Do not subtract a bias: training added a +-0.005 rad/s per-environment bias on purpose.
  2. projected gravity, body frame (3)
  3. velocity command [vx, vy, wz] x [2, 2, 0.25] (3), AFTER the deadband and ramp below
  4. joint positions minus default, **27 joints** (see order below) (27)
  5. joint velocities x 0.05, same 27 joints (27)
  6. last action: the previous raw policy output, 13 values (13)
  7. gait clock [sin, cos] of 2*pi*(t mod 0.8)/0.8 with t = seconds since the policy started (0 at start), always running (2)
- **Action, 13 values:** 12 legs then waist yaw. `scale` 0.25 for legs, **0.2 for waist yaw**.
- Joint order for items 4, 5 and the action (indices into the motor arrays from `configs/g1_walk_stand.yaml`, motor in brackets):
  left hip pitch [0], left hip roll [1], left hip yaw [2], left knee [3], left ankle pitch [4], left ankle roll [5],
  right hip pitch [6], right hip roll [7], right hip yaw [8], right knee [9], right ankle pitch [10], right ankle roll [11],
  **waist yaw [12]**, then (observation only) left shoulder pitch [15], roll [16], yaw [17], elbow [18], wrist roll [19], pitch [20], yaw [21],
  right shoulder pitch [22], roll [23], yaw [24], elbow [25], wrist roll [26], pitch [27], yaw [28].
  Waist roll and pitch (motors 13, 14) are locked on this unit and are **not** in the observation or the action.
- Defaults (rad): hip pitch -0.1, knee 0.3, ankle pitch -0.2, all other leg joints and waist yaw 0; arms: shoulder pitch 0.35, shoulder roll +0.18 (left) / -0.18 (right),
  elbow 0.87, shoulder yaw and all wrists 0.
- Gains (Unitree's onboard values, what v11 was trained with): legs kp 100/100/100/150/60/60 and kd 4/4/4/5/3/3 (hip pitch/roll/yaw, knee, ankle pitch/roll);
  waist yaw 200/6. **The policy does not control the arms:** hold them at the default pose with kp 60 / kd 3 (wrists 40 / 3). In simulation the arms were swept
  through random poses by a separate generator while v11 balanced; for this first trial hold them still (the observation still reads their real positions).
- Hands (Inspire) are a separate system; do not command them in this task.

## The velocity command (what the planner side does before the policy sees it)
1. Deadband: if norm(vx, vy) < 0.05 set both to 0; if |wz| < 0.05 set wz to 0.
2. Ramp toward the requested value at 1.0 m/s^2 (vector) and 2.0 rad/s^2 (yaw), at 50 Hz. The ramped value is what goes into observation item 3.
3. Trained ranges: vx -0.3..0.6, vy +-0.3, wz +-1.0. **This task only uses [0, 0, wz].**

## Stages (each needs its own dry-run plan and the user's yes; stop at the first surprise)
0. **Contract test, laptop only.** The simulation session provides recorded 78-value observations and the policy outputs (from the Isaac Lab Play task); replay them through the controller's
   observation code and the exported network. Actions must match to <= 1e-4. Also feed the real `rt/lowstate` recording from `measurements/20261029/standing_lowstate_raw.csv`
   and check observation ranges (gravity near [0,0,-1], small velocities, no NaN) and the joint order.
1. **Robot, read-only dry run** (hoisted, zero torque, nothing published): controller with `--dry-run`, log observations and the actions v11 WOULD send. Actions bounded, near zero for a
   near-default pose, loop time well inside 20 ms (numpy single-threaded: OPENBLAS_NUM_THREADS=1).
2. **Hoisted, feet off the ground, commands enabled, zero velocity command.** Ramp from the current pose to the default over >= 3 s, hold, then blend the policy in over >= 5 s (0 -> 1).
   Hold the arms at their default with the gains above. Watch joint torques and oscillation. Kill the controller mid-run: the robot must go to a safe damping state (`estop_damping.py`).
3. **Feet touching down, hoist taking weight, then slack.** Zero command. Slow weight transfer, assistant holding the robot.
4. **Free standing for a fixed time** (10 s, then 30 s), hoist or spotter within reach. No pushes.
   *Stage A is complete here. The simulation session reviews the report before stage B.*
5. **Turning in place (stage B).** Only after stage A passed and the user says so. First hoisted with the feet just off the floor (to watch the stepping pattern), then on the floor with the
   hoist slack and a spotter, in an open area of at least 1.5 m radius (the robot steps around: 6-10 cm of wander per 3 s turn in simulation, with feet lifting).
   Yaw-only commands [0, 0, wz], ramped at 2 rad/s^2, held 3 s, then back to zero (ramped), then 6 s of zero command. Order: wz = 0.2, 0.3, then 0.5 rad/s, one direction first,
   then the other. Do not exceed 0.5 rad/s in this task. One run per yes.
- Abort on: any joint speed > 5 rad/s, torque above a limit set in the dry-run plan, tilt > 25 deg, NaN, or loop overrun, or the user's stop.

## Measure and report (compare each number with simulation)
| Quantity | v11 in simulation | How to measure |
|---|---|---|
| Knee angle, standing | 0.288 rad | `rt/lowstate` |
| Knee torque, standing | 3.5 N*m | `tau_est` |
| Ankle pitch angle, standing | -0.20 rad (default) | `rt/lowstate` |
| Base height, standing | 0.784 m | from leg joint angles / forward kinematics |
| Foot lifts at zero command | 0 | contact / joint motion |
| Tilt (IMU), standing | < 2 deg RMS | IMU |
| Drift / sway, standing | 0.3 cm in 17 s | floor markings or state estimate |
| Yaw rate reached, command 0.3 / 0.5 rad/s | 0.34 / 0.53 | IMU gyro z |
| Time to be still after the turn | 1.2-2.1 s | gyro + joint velocity |
| Foot lifts, 6 s after the turn | 0 | contact / joint motion |
| Position wander during a 3 s turn | 6-10 cm | floor markings |
| Loop time, command latency | 3-8 ms at 50 Hz | controller log |

Also record: the first stage where reality differed from simulation, any joint that saturated, action chatter (RMS of a_t - a_{t-1}), the waist yaw motor's
temperature and torque, and anything the arms did while held.

## Deliverables
- `controller/` code that runs the stages above with `--dry-run` as the default and an explicit flag to publish.
- A `configs/` entry for the 78 -> 13 contract (the existing `g1_walk_stand.yaml` describes the v4/v1 interface).
- `measurements/<date>/stand_turn_baseline/` with logs, a short `REPORT.md` section and the table above filled in.
- A list of surprises and open questions for the simulation session.

## Not part of this task
Translation, walking (v12) and the selector between v11 and v12, pushes, arm motion (a later step: v11 stood with the arms moving in simulation), hands,
mass and plant identification (`hardware_requests_from_sim.md`).
