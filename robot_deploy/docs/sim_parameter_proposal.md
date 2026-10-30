# Simulation parameter proposal from the first hardware session (2026-10-29)

For the simulation/training session. Source data: `measurements/20261029/` (raw recordings stay on the laptop and on the robot,
see the end). Per-joint table: `measurements/20261029/joints.csv`. Everything below was **measured on one G1 unit** while Unitree's own
onboard controller was running; **no g1_rl policy has run on this robot**, and nothing was commanded by us (we only listened).
Confidence is stated per item: HIGH = directly observed and repeated, MEDIUM = observed once or inferred, LOW = weak.

## Limits of the data (read first)
- ~74 s of clean walking (25 short segments, forward stick mean 0.27 of full), plus standing, plus manual pushes at unrecorded times.
  The robot was on a support at least part of the time. Torque extremes (ankle pitch 63-67 N*m, knee up to 101) include the pushes;
  use the 99th percentiles for walking loads.
- `tau_est` may be derived from the command or motor current, not an independent torque measurement.
- Rest-noise figures come from a standing robot; noise during motion was not separated from the motion itself.
- Not measured at all: friction, reflected inertia (armature), backlash, command-to-motion delay, true torque and speed limits,
  total mass, push strength. These need the hoisted single-joint tests (approval per run).

## Changes to consider

| # | Parameter | Current sim | Real / onboard | Evidence | Conf. | Proposal |
|---|---|---|---|---|---|---|
| 1 | Waist roll, waist pitch (motors 13, 14) | driven, observed (kp 200, limit 35 N*m) | motor mode 0, q = dq = tau = 0 in every run (zero torque, standing, walking, pushes) | 4 recordings | HIGH for this unit | Remove both from actions and observations (hold at 0). v3/v4 need retraining. Waist yaw stays (moves +/-0.13 rad, tau p99 9 N*m). |
| 2 | Leg kp (hip p/r/y, knee, ankle p/r) | 100/100/100/150/40/40 | 100/100/100/150/60/60, constant while nominal | ~94k + 340k command messages | HIGH (what the onboard controller sends) | Ankle kp 40 -> 60 if matching Unitree. Whether to match is a decision, see below. |
| 3 | Leg kd | 2/2/2/4/2/2 | 4/4/4/5/3/3 | same | HIGH | Roughly double ours (hips 4 vs 2, ankle 3 vs 2). Train/evaluate at these. |
| 4 | Waist gains | 200 / 5 | 200 / 6 (Unitree's `deploy_real` file says 300 / 3) | same | HIGH for the onboard value | Use 200 / 6 for waist yaw. |
| 5 | Arm gains (held, not policy-driven) | shoulder, elbow 100/100/50/50 kp, kd 2; wrists 40 / 2 | shoulders and elbow 60 / 3, wrists 40 / 3 | same | HIGH | Use 60 / 3 and 40 / 3 for the held arms. They drop as low as 35 / 1.2 (first recording) or 14 / 0.9 (walk10) during the user's gesture combos (not a policy concern). |
| 6 | Actuator model | ideal PD, armature 0.01 | `tau_est` = 0.96 x kp*(q_cmd - q) + kd*(0 - dq), corr 0.98, lag 0-4 ms (4 ms grid) | standing + walking, 12 leg joints | MEDIUM | Keep ideal PD; allow a 3-7% torque shortfall (per-joint slope 0.91-0.99). Command is pure PD: dq target 0, torque feed-forward 0, same form as our action. |
| 7 | Ankle pitch torque limit | 50 N*m | p99 37-39, max 63-67 (pushes included) | walk10 | LOW (limit itself not read from firmware or URDF) | Do not raise on this evidence alone. Check the manufacturer limit; consider randomising 50-65 or testing the policy's sensitivity. Knee (139) and hips are not close to limits at p99. |
| 8 | Observation noise (at rest) | gyro +/-0.2, gravity +/-0.05, joint pos +/-0.01, joint vel +/-1.5 | gyro std 0.001/0.002/0.002 rad/s, gravity std 0.0004, joint q std <= 1.5e-4 rad, joint dq std 0.007 median / 0.018 max | standing run | HIGH at rest, LOW in motion | Training noise is 1-2 orders larger than rest noise. It is a robustness margin, not a measurement; do not tighten before noise in motion is measured. Joint-vel +/-1.5 rad/s looks the most excessive. |
| 9 | IMU conventions | body-frame gyro, projected gravity | pelvis IMU, quaternion order w,x,y,z (inferred), gravity z = -1 upright, constant gyro bias ~[-0.0004, 0.0043, 0.0010] rad/s | all runs | MEDIUM | Matches our convention. Optionally randomise a gyro bias of ~+/-0.005 rad/s. |
| 10 | State timing / latency | 0-20 ms actuation delay, 50 Hz | `rt/lowstate` ~700-800 Hz, gaps p99 2-6 ms, max 32 ms; `rt/lowcmd` ~740 Hz; command-to-state round trip NOT measured | timing runs, walk10 | MEDIUM (rates), none (latency) | Keep the delay range; do not narrow it. Optionally add 0-6 ms observation age. **UPDATE (hoisted step tests, right wrist yaw):** published at 200 Hz the joint responded ~115-120 ms late, but at **50 Hz (our control rate) it responds 3-6 ms after the command** (run 4, recorder timestamps). The 120 ms was a high-publish-rate effect (probable backlog), not a path latency, so the 0-20 ms training range is fine for 50 Hz control. Legs and other publish rates not yet tested; see `open_issues.md` (run 4). |
| 11 | Default pose | hip p -0.1, knee 0.3, ankle p -0.2 | onboard standing: hip p -0.18/-0.15, knee 0.30/0.28, ankle p -0.14, hip r/y within 0.05, shoulder roll +/-0.26, elbow 0.84/0.80 | standing run | MEDIUM | Same sign on every joint, within ~0.06 rad on legs. These are not encoder offsets (the robot was in its own pose). Keep ours unless retraining anyway. |
| 12 | Gait clock | 0.8 s period | hip-pitch cycle 0.9-1.6 Hz from two segments | walk10 | LOW | No change; need steadier walking data. |
| 13 | Joint static friction (unloaded, hanging) | not modelled / unknown | breakaway ~0.2 (wrist yaw), ~0.4-0.6 (elbow), ~0.6-0.65 (knee, plus a ~0.85-1.15 N*m gravity holding load at 0.26 rad from the hanging shank and foot), **~0.8-1.0 N*m (ankle pitch, clean on the right; the left shows an extra unexplained growing resistance up to 2.3 N*m)**; direction-dependent offset 0.1-0.3 N*m | hoisted step/ramp tests (right ankle ramp: 17 slips, up 0.69 / down 1.32 N*m; left knee step) | LOW-MEDIUM | Consider a static/Coulomb friction term: a few tenths of N*m on wrist/elbow, ~1 N*m on the ankle pitch. Loaded values on the ground are unmeasured and probably higher. |

## Decisions for the simulation session
1. **Match Unitree's leg gains (2, 3) or keep ours?** Unitree's are the only gains with a hardware track record (for their controller),
   but a policy trained at 40/2 ankles may behave differently at 60/3. Either train at Unitree's gains or evaluate the current policy
   at both and report the gap.
2. **Retrain without waist roll/pitch (1).** v3/v4 as they stand cannot be deployed on this unit's waist.
3. **Torque limit and noise (7, 8)** need a sensitivity check, not a one-way edit.

## What would settle the rest
- Hoisted single-joint tests (ankle pitch first, then knee, hip pitch, waist yaw, one arm joint): delay, effective kp/kd, friction,
  armature, backlash. Requires putting the robot in Unitree's debug mode; in the default zero-torque state, damping commands from PC2
  had no visible effect (`open_issues.md`).
- A longer, steadier walk (5+ min, no F1 combos) for gait and noise in motion.
- Push times from the user to turn the manual pushes into response data for the standing policy.
- Mass, torque and speed limits from the SDK/URDF/manufacturer documentation.

## Data
`measurements/20261029/`: `joints.csv`, `imu.json`, `timing.json`, `walk/walk10` (537 s, full rate), `walk/test30*`, step A and
standing listens, damping tests, `start_chain_watch.log`. Raw `.bin` and `*_raw.*` files are git-ignored. Copies of the walk
recordings are also on the robot in `/home/unitree/g1_recordings/` (checksums verified identical to the laptop). Format of the
`.bin` files: `meta.json` holds the numpy dtype; `np.fromfile(path, dtype=...)`. Recorder: `scripts/record_walk.{sh,py}`.
