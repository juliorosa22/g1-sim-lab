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

Where these numbers come from: our values were copied from `unitree_sim_isaaclab`'s simulation config; Unitree's are the gains its
`deploy_real` script uses to hold the waist and arms at fixed targets while its 12-joint leg policy runs. Neither set was validated for a
policy that DRIVES the waist (our v3/v4 do). So this is a decision, not a proven error: either run 200/5 as trained (and check it is
safe on the real robot) or evaluate/retrain in simulation at 300/3. The joint attribution of Unitree's 7-value arm vectors assumes
the order shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw, which matches the SDK's `G1JointIndex` for the left arm (15-21);
the right arm (22-28) was not read.

## Is the waist usable on this robot?
Unitree's SDK example marks `WaistRoll` and `WaistPitch` (motors 13, 14) as **"INVALID for g1 23dof/29dof with waist locked"**.
Our walking policy v4 (and v3) commands all three waist joints. Find out whether THIS unit's waist roll/pitch are unlocked and
actuated. If they are locked, those two outputs must be held at default and v3/v4 would need retraining without them.
(`unitree_sdk2_python/example/g1/high_level/g1_arm7_sdk_dds_example.py`, class `G1JointIndex`.)

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

## Findings from hardware step A (2026-10-29, read-only listen, robot on support, zero torque)
Raw data and summaries: `measurements/20261029/`. One ~22 s run, so treat as preliminary.
- **Waist roll / pitch (motors 13, 14) look absent or locked on this unit.** They report motor mode 0, temperature 0/0,
  voltage 0, and q, dq, tau_est exactly 0, while all 27 other motors report mode 1 and 35-52 degC. This matches Unitree's
  "INVALID with waist locked" note. Not proof they cannot be unlocked, but v3/v4 command and observe them: expect to hold those
  two outputs at default and retrain without them (or confirm with Unitree how to unlock).
- **Something already publishes rt/lowcmd**: ~16k messages in 20 s, every motor kp=0, kd=0, tau=0, q=0, mode=1,
  mode_machine=6 (the robot's own zero-torque state, source not yet identified). A second publisher would compete with it.
- **IMU (pelvis)** at rest: gyro bias ~[-0.0004, 0.0043, 0.0010] rad/s, std ~[0.0010, 0.0023, 0.0015]; quaternion order
  looks w,x,y,z; projected gravity z ~ -1 upright, as in simulation. Joint q noise at rest ~1e-5 rad.
- **lowstate rate looks like ~740-760 Hz, not 500**; jitter not yet measured (timestamp precision bug in the logger).
- Joint signs/zeros NOT yet checked: the legs hang relaxed (knee 0.90, ankle pitch -0.73), not in the default pose.
- **Timing rerun (2026-10-29, ns timestamps, 15 s each, robot on support, zero torque; `measurements/20261029/timing.json`)**:
  `rt/lowstate` arrives at ~780-810 Hz (not 500), gap mean ~1.25 ms, p99 ~1.8-2.0 ms, p99.9 2.5-4.3 ms, max ~6-7 ms, no gap above 20 ms.
  `rt/lowcmd` (PC1's zero-torque stream) ~740-770 Hz. Jitter is small against a 20 ms control period; the `tick` field
  mostly steps by 1 but has ~4% repeats and some 4-5 step jumps (its unit and the true source rate are unresolved).
  Round-trip latency still unmeasured (needs commands).
- **`unitree-start-joystick.service` was stopped** (sudo, user-requested) to free CPU; it used ~73% of a core. Torch walk-net
  forward, 1 thread: mean 6.5 ms before vs 1.5 ms after (p50 6.3 -> 1.1 ms; p99 13.4 -> 6.9 ms; max 76 -> 48 ms). Still far
  above what the arithmetic needs and the tail still exceeds 20 ms, so Python/torch on this CPU is not yet fit for the 50 Hz loop
  as is. Single runs, different iteration counts: indicative only. The service starts again on reboot.
- **Live damping test, step 1 (2026-10-29 20:31, motor 21 only, kd=2, kp=tau=0, 5 s, robot on support at zero torque;
  `measurements/20261029/damp_step1_raw.csv`)**: command accepted, no errors, motor modes unchanged, no motion. Our messages
  (~340 Hz, not 500: Python sleep overhead) interleave on `rt/lowcmd` with PC1's zero stream: 23% ours (kd21=2), 77% PC1's
  (kd21=0). **INCONCLUSIVE on whether the motor obeys it**: the wrist was not moved (dq at noise level, tau_est change nil), and at
  kd=2 and rest speed the expected torque (~0.03 N*m) is below tau_est noise. Needs a hand-guided repeat.
- **Live damping test, step 1b (2026-10-29 20:36, same command, wrist hand-guided; `measurements/20261029/damp_step1b_raw.csv`)**:
  the wrist did move (q21 0.157..0.269 rad, |dq| up to 0.5 rad/s) while 954 of our kd=2 messages shared the topic with PC1's zero
  stream, yet `tau_est` of motor 21 stayed flat (~-0.016, slope vs dq 0.000; a kd=2 command obeyed even a quarter of the time
  would give about -0.5 N*m per rad/s). **No measurable effect: in this state (robot at zero torque, PC1 streaming zeros), a damping
  command sent from PC2 does not act on the joint**, or `tau_est` does not reflect it. Do not rely on it as a stop. To retest in the
  real condition (our controller running in Unitree's debug mode, possibly without PC1's zero stream) before trusting it.
- **Standing pose, read-only listen (2026-10-29 ~21:10, robot in the onboard balance mode FSM 801 via F1+Up, on the support;
  `measurements/20261029/standing_*`)**:
  - Signs/zeros consistent with our default pose: same sign on every joint; legs within ~0.06 rad (knee 0.30/0.28 vs 0.30, ankle
    pitch -0.14 vs -0.20, hip pitch -0.18/-0.15 vs -0.10), shoulder roll L +0.26 / R -0.28 (default +/-0.18), elbows 0.84/0.80 vs 0.87.
    This is the onboard controller's pose, not ours, so these differences are NOT encoder offsets; a real offset check needs the robot
    held exactly at default.
  - Waist roll/pitch (13, 14) still mode 0, q=dq=tau=0 while standing: locked/absent, as in zero-torque.
  - IMU upright: projected gravity (0.007, 0.003, -1.000); gyro bias/std unchanged from the zero-torque run.
  - tau_est (standing, partly supported): L knee -9.7, L hip roll +6.9, R knee -2.0, R hip roll -2.4 N*m: strongly asymmetric, and
    low for a 35 kg robot carrying its full weight, so the support probably takes part of it.
  - **Onboard controller gains (last rt/lowcmd message only, single snapshot, mode 1, tau 0)**: legs kp 100/100/100/150/60/60, kd 4/4/4/5/3/3;
    waist kp 200 kd 6; shoulders+elbow kp 60 kd 3; wrists kp 40 kd 3. Ours (sim): ankle kp 40 (theirs 60); leg kd 2/2/2/4/2/2 (theirs
    4/4/4/5/3/3); waist kd 5 (6); shoulders/elbow 100/100/50/50 (60 all); wrist kd 2 (3). Not a measurement of the plant; need the
    gains over time (they may vary with state).
- **Motion recording, onboard balance/walk controller driven by the RC (2026-10-29 21:08, 128 s, read-only, robot moved around on the
  support; `measurements/20261029/motion_*`)**:
  - **Leg gains are constant** over all ~94k command messages: kp 100/100/100/150/60/60, kd 4/4/4/5/3/3 (ankles differ from our 40/2).
  - **Waist and arm gains are scheduled, not constant**: nominal waist 200/6, shoulders+elbow 60/3, wrists 40/3, dropped to 35/1.2
    from 19.0 to 27.4 s, then ramped back up in small steps. Timing matches the user's F1+B (salute) at 18.2 s and F1+Right (return
    to balance) at 23.4 s, so this is the user's `movimentos_militares` flow, not the walking controller itself.
  - **The onboard command is a pure PD**: q target only, dq target = 0, tau feed-forward = 0 (same form as our sim action).
  - **tau_est tracks the PD torque from the onboard commands**: legs mean corr 0.98, slope 0.97 (0.91-0.99), lag 0-4 ms (4 ms grid).
    Caveat: tau_est may be derived from motor current/command rather than an independent torque measurement.
  - Peaks while walking: ankle pitch ~41 N*m (limit 50), knee 46, hip roll 51, hip pitch 34; joint speeds up to 7.3 rad/s (knee);
    max tilt 8.3 deg; waist yaw within +/-0.11 rad. Waist roll/pitch stayed exactly 0.
- **Walking recording `walk10` (2026-10-29, robot driven by the RC under Unitree's controller, read-only full-rate recorder over Wi-Fi;
  `measurements/20261029/walk/walk10/`, 537 s, 362k state + 342k cmd messages, ~670 Hz, p99 gap 5.9 ms, 9 gaps >20 ms, max 32 ms)**:
  - **Only ~74 s of clean walking** (sticks active AND leg gains nominal; 25 segments, longest 4 s, mean forward stick 0.27 of 1.0).
    Sticks were active 147 s in total; 12 "return to balance", 4 attention, 3 salute, 3 rest combos were pressed (66-417 s).
    Far too little for behaviour cloning; thin even as a reference-motion set. Record a longer, steadier walk if we want it.
  - **Leg gains are NOT always nominal**: reduced for 62 s in six intervals between 202 s and 301 s (hip kp down to 40, ankle down to
    28.5; knee ~99): not explained by the logs, to be matched with what was done to the robot then. Outside those, legs are
    kp 100/100/100/150/60/60, kd 4/4/4/5/3/3 as before. Waist/arm gains were scheduled again.
  - **tau_est vs PD torque while walking**: corr 0.98 (min 0.97), slope 0.96 (0.93-0.99): ideal-PD actuator with ~4% shortfall holds.
    dq target = 0 and tau feed-forward = 0 throughout (pure PD).
  - Torque (tau_est) peaks, whole run, 99th percentile / max N*m: ankle pitch 37-39 / 63-67 (**above our sim limit 50**), knee 26-32 / 87-101,
    hip roll 40-43 / 64-76, hip pitch 22-24 / 52-56, waist yaw 9 / 24. Max joint speeds: knee 16 rad/s, ankle roll 28 rad/s (single spikes).
  - Gait: hip-pitch cycle ~0.9-1.6 Hz from two usable segments (low confidence); our policies use a 0.8 s clock (1.25 Hz).
  - Tilt mean 2.8 deg, p99 9.1, max 14.9; gyro max 2.8 rad/s; hottest motor 61 degC (right shoulder pitch). Waist roll/pitch stayed 0.
- **Correction to the `walk10` numbers (user, 2026-10-29): the robot was pushed by hand at times during that recording** (times not
  noted). So the extremes listed above (ankle pitch 63-67 N*m, knee up to 101, hip roll 76, tilt max 14.9 deg, gyro max 2.8 rad/s)
  are probably from pushes, NOT from ordinary walking; use the 99th percentiles for walking loads. Likely push responses seen while
  the sticks were idle: ~494 s (tilt 13.7 deg, gyro 2.0 rad/s, torque 101), ~499 s (gyro 2.5, torque 83), ~452 s (torque 87),
  ~260 s (tilt 14.9, torque 76, leg gains reduced), ~211 s, ~275 s, ~439-447 s. Unconfirmed: a simple IMU detector cannot separate pushes
  from foot impacts while walking. No force was measured.
- **Hoisted joint tests, runs 1 and 1b (2026-10-29 ~22:18-22:21, joint 21 left wrist yaw, `hold`, kp 20 / kd 1, others held at 0.5 x nominal;
  `measurements/20261029/jointtests/jt_21_hold_*`)**: tooling worked end to end (200 Hz loop, mean 5.0 ms, worst 7.5 ms, no aborts, publishing stopped
  cleanly, motors returned to mode 0 afterwards). **Our commands from PC2 are accepted in this state: all 27 live motors switched from mode 0 to mode 1
  (enabled) during the run** (logged in run 1b; waist roll/pitch stayed 0). Whether they actually TRACK a target is not yet shown: with the hold at zero
  position error the commanded torque is ~0 by design (tau_est 0.0-0.4 N*m, no limb moved more than 1.1 mrad), and in run 1b no hand nudge reached
  the wrist (largest excursion anywhere 0.0011 rad), so neither run discriminates. Next: a commanded `step` on the wrist.
- **Hoisted test run 1c (2026-10-29 22:23, `jt_21_hold_222338`): the held left shoulder pitch (joint 15, hold kp 30 / kd 1.5) was pushed by hand and pushed back.**
  Displacement up to 0.15 rad against the hold: tau_est vs displacement slope -27.7 N*m/rad (corr -0.97); fit tau = a*err + b*dq gives a = -28.5
  (commanded 30, i.e. 95%) and b = -1.03 (commanded 1.5; dq here is differentiated from q, noisy, so b is weaker evidence). Peak 4.9 N*m.
  **So commands from PC2 act as a PD in this state (stiffness within ~5-8% of the commanded kp).** Caveat: tau_est may be computed from the command rather than
  measured, so this alone shows the command is applied, not the physical torque; the user's hand feel should corroborate (asked). The run aborted
  by the held-joint drift limit (0.15 rad) at 16.1 s as designed. The wrist (joint 21) itself did not move (peak 3 mrad).
- **Hoisted test run 1d (2026-10-29 22:25, `jt_21_hold_222546`, aborted at 31.3 s by the held-joint drift limit on joint 19)**: hand-moved held joints again pushed back
  in proportion to displacement: left wrist roll (19, held kp 20): slope -19.9 N*m/rad, fit stiffness 20.4, damping 1.34 (kd 1.5); right shoulder pitch (22, kp 30):
  -30.2, fit 32.1 / 1.54; right shoulder roll (23, kp 30): -29.7, fit 25.6 / 1.45. With run 1c (joint 15: 27.7), four joints agree: **the torque estimate follows
  kp x position error to within ~5-15% (stiffness) and damping within ~10-30%, so PD commands from PC2 act in this debug state.** Same caveat: tau_est may be computed
  from the command. The wrist yaw (21) itself was never moved in the hold runs; the user was turning the wrist roll (19) instead.
- **Hoisted test run 2: `step` on joint 21 (left wrist yaw), kp 20 / kd 1, +-0.05 rad, 2026-10-29 22:34 (`jt_21_step_223437`, 200 Hz log, finished, no abort).** Single run, single joint.
  - **Dead time of ~100-150 ms** between our commanded step and the first response: tau_est and dq both start moving 100-151 ms after each of the six target changes
    (mean ~120 ms); best lag for the position response 125 ms (corr 0.97). Position is 0% complete at +100 ms, ~35% at +150 ms, ~72% at +250 ms. tau_est and dq begin
    together, so the delay is in the command path (PC2 -> PC1 -> motor), not in the joint dynamics. For comparison the onboard controller's own commands
    reach tau_est within 0-4 ms (walk10). **If this holds on other joints and at other publish rates, it is 5-6x the 0-20 ms actuation delay we train with.** Unconfirmed: needs the
    same step on more joints, a different publish rate, and a cross-check of state-age.
  - **The wrist stalls short of the target** (77-78% of the step, then flat): static residual error 5 mrad after up-steps (kp*err = 0.10 N*m) and 17 mrad after down-steps
    (0.34 N*m), matching tau_est (0.13 and -0.30). Estimated breakaway friction + gravity offset of this joint at rest: ~0.1 to ~0.3 N*m, direction dependent (method: residual
    error x kp; assumes tau_est tracks the real torque). Consistent with the user's observation that this joint resists hand motion.
  - tau_est does NOT equal the command law here (corr 0.32 / slope 0.16 over the run; 0.90 at a 105 ms lag), so it is a real response signal, not an echo of kp*err: this supports using
    it as a torque estimate (earlier caveat weakened, not removed).
  - Peak tau_est 1.28 N*m (cap 3), peak speed 1.5 rad/s, max excursion 0.045 rad (limit 0.10).
- **Hoisted test run 3: `step` on joint 28 (right wrist yaw), same plan as run 2 (2026-10-29 22:44, `jt_28_step_224413`, finished, no abort).** Replicates run 2 on the other arm:
  response delay (dq > 0.05 rad/s) 96-125 ms, mean **115 ms** (left wrist yaw: 106-141, mean 121); progress 69% at +0.25 s and 73% at +1.45 s (left 72% / 78%); peak torque estimate
  1.32 N*m (left 1.28); max excursion 0.045 rad. So the ~120 ms dead time appears on both arms (same joint type). Not yet tested: other joint types, other publish rates.
  Static residuals mirror left to right: the left wrist stalls short after DOWN steps (17 mrad, 0.34 N*m), the right after UP steps (16 mrad, 0.31 N*m). Reading the two directions as
  friction +/- gravity offset: breakaway friction ~0.2 N*m on both (0.22 / 0.21), gravity-like offset ~0.11-0.12 N*m with opposite sign on the two sides (estimates; assumes tau_est tracks the real torque).
- **CORRECTION (run 4, 2026-10-29 22:48, `jt_28_step_224819` + recorder `walk/rate50`): the ~120 ms "dead time" of runs 2 and 3 was caused by publishing at 200 Hz, not a fixed
  latency of the command path.** Same step, same joint (28), same gains, published at **50 Hz**: with the full-rate recorder's timestamps (our command message vs `rt/lowstate`, ~1 ms
  resolution) the joint starts moving **3-6 ms** after the command (q > 0.5 mrad: 4-6 ms; dq > 0.05 rad/s: 3-4 ms; tau_est > 0.15 N*m: 3 ms; 5% of the step: 7-12 ms, six steps). At 200 Hz the
  same criterion gave 96-125 ms. The test's own log at 50 Hz only shows ~20-27 ms because of its 20 ms row spacing. Likely mechanism (UNCONFIRMED, PC1 is not inspectable from here):
  PC1/DDS consumes commands slower than 200 Hz, so a backlog builds (a bounded queue of ~25 samples x 5 ms would give ~125 ms). Consequence: at 50 Hz, the rate Unitree's `deploy_real` and
  our policies use, the command-to-motion latency is only a few ms. The earlier warning that the 0-20 ms training delay was "5-6x too short" is WITHDRAWN for 50 Hz control; it only
  applies to publishing at 200 Hz or faster. Do not publish faster than needed. The stall at ~77% of each step (static friction) is unchanged: 71% done at +250 ms at 50 Hz.
  Open: where the backlog starts (try 100 Hz); the same check on legs.
- **Hoisted test run 5: `step` on joint 25 (right elbow), kp 30 / kd 1.5, +-0.05 rad, 50 Hz publish, 2026-10-29 22:50-22:51 (`jt_25_step_225058` + recorder `walk/elbow50`; finished, no abort).**
  Confirms the 50 Hz result on a heavier joint: response starts **3-6 ms** after our command message (q > 0.5 mrad: 3-6 ms, dq > 0.05: 3-5 ms, tau_est > 0.15 N*m: 3-4 ms, 5% of the step 6-8 ms; six steps,
  recorder timestamps). The 200 Hz dead time of ~120 ms is therefore a publish-rate effect, not a joint property.
  The elbow stalls short of each step because of static friction: residual error 14-16 mrad after up-steps (kp*err 0.43-0.49 N*m, tau_est 0.51-0.63) and 27 mrad after down-steps (0.80 N*m, tau_est -0.62).
  Reading the two directions as friction +/- gravity offset: breakaway friction ~0.4-0.6 N*m, gravity-like offset ~0.05-0.2 N*m (estimates; assumes tau_est tracks the real torque). Wrist yaw was ~0.2 / ~0.11.
  Peak tau_est from the 1 kHz state 1.9 N*m (cap 4); the test's own 50 Hz log shows only 0.69, i.e. it MISSES fast torque transients, and the runtime torque trip samples state at the publish rate: at 50 Hz
  the watchdog can miss spikes shorter than 20 ms (margins in the caps cover this for these small amplitudes; keep it in mind for the legs). "Percent of step completed" in these tests is relative to the
  previous TARGET, so steps that follow a stall look smaller than they are; use the residual error instead.
- **Hoisted test run 6: `step` on joint 4 (left ankle pitch), kp 30 / kd 1.5, +-0.03 rad, 50 Hz, feet off the ground, 2026-10-29 22:56 (`jt_4_step_225621` + recorder `walk/ankle50`; finished, no abort).**
  - **Command path at 50 Hz is fast on a leg too**: tau_est responds 3-5 ms after our command, joint speed 5-9 ms, first position motion (> 0.5 mrad) 7-12 ms (slightly later than the arms because the joint is stuck).
  - **The ankle barely moves: static friction dominates.** The joint moved only 1-9 mrad of the +-30 mrad steps (max excursion 0.009 rad) and stopped with residual errors of 21 mrad after up-steps (kp*err 0.63 N*m, tau_est 0.43)
    and 36 mrad after down-steps (1.08 N*m, tau_est -1.0). Reading the two directions as friction +/- offset: breakaway friction **~0.7-0.9 N*m**, gravity-like offset ~0.2-0.3 N*m (estimates, hanging foot, unloaded; assumes
    tau_est tracks the real torque). So with kp 30 the ankle has a dead band of roughly +-0.03 rad: errors below that produce no motion. A step of this size cannot show the dynamics; use a bigger step, a sine or a ramp.
  - Peak tau_est 1.23 N*m (1 kHz), peak speed 0.18 rad/s; the same-leg knee moved 0.015 rad, hip pitch 0.002 (held, limit 0.15). The ankle is driven in PR (pitch/roll) mode (mode_pr = 0).
  - Friction so far (unloaded, hanging): wrist yaw ~0.2, elbow ~0.4-0.6, ankle pitch ~0.7-0.9 N*m.
- **Hoisted test run 7: `ramp` on joint 4 (left ankle pitch), kp 30 / kd 1.5, target 0 -> +0.08 -> -0.08 -> 0 at 0.02 rad/s, 50 Hz, feet off the ground, 2026-10-29 23:00 (`jt_4_ramp_230003` + recorder `walk/ramp50`; finished, no abort).**
  - **The joint covered only 22% of the commanded span**: the target moved +-0.080 rad, the ankle stayed within -0.014..+0.022 rad, in 13 small slips (0.5-8.5 mrad, peak speed 0.23 rad/s).
  - **The torque needed to make it slip kept growing**: kp*err at slip onset 0.58 -> 1.75 N*m going up, 0.75 -> 2.28 N*m going down (peak tau_est 2.25 N*m). Constant Coulomb friction would not do this.
    **This SUPERSEDES the ~0.7-0.9 N*m ankle friction estimate from run 6 (WITHDRAWN as a clean friction value).** What is established: with the ankle held by neighbours as in this test, the joint resists at least 2.3 N*m of commanded
    torque without moving through most of the +-0.08 rad span. Cause NOT identified. Candidates: the toe/foot touching the floor or a support, the foot dragging the held ankle roll through the parallel linkage
    (ankle roll moved 0.021 rad in step with the pitch, corr 0.85, its torque stayed within -0.5..+0.3 N*m), a mechanical or soft stop, or friction that depends on history/position.
  - **Command law holds**: over the run tau_est = 0.94 x kp*(target - q), corr 1.00 (stuck samples), so the commanded torque is really being applied; the joint just does not move.
  - Knee and hip pitch of the same leg (held) did not move (0.0001 rad).
- **Hoisted test run 8: `ramp` on joint 10 (right ankle pitch), same plan as run 7 (2026-10-29 23:07, `jt_10_ramp_230714` + recorder `walk/ramp50r`; finished, no abort). The right ankle behaves very differently from the left:**
  it covered **69%** of the commanded +-0.080 rad span (offset -0.046..+0.065 rad) vs 22% on the left, in 17 slips (12 up, 5 down; sizes 2-26 mrad), peak speed 0.31 rad/s, peak tau_est 1.56 N*m.
  Slip onset kp*err does NOT grow: up-moves 0.31-1.03 N*m (mean 0.69), down-moves 0.87-1.55 N*m (mean 1.32) => breakaway friction ~1.0 N*m (mean of the directions) with a direction offset ~0.3 N*m (estimates;
  assumes tau_est tracks the real torque; tau_est/(kp*err) = 0.92). The left ankle's FIRST slips (0.6-0.8 up, 0.75-1.1 down) are consistent with the same ~0.8-1.0 N*m, but it then developed a growing extra resistance
  (up to 2.3 N*m) that the right side does not show: **something specific to the left ankle/foot** (contact, strap or cable tension, mechanics) is still unidentified. Hand test by the user: both ankles need "a little more push" to move when disabled.
  Working number for the simulation: ankle pitch breakaway friction ~0.8-1.0 N*m unloaded (right side clean, left side contaminated).
- **Hoisted test run 9: `step` on joint 3 (left knee), kp 40 / kd 2.0, +-0.05 rad, 50 Hz, feet off the ground, 2026-10-29 23:11 (`jt_3_step_231107` + recorder `walk/knee50`; finished, no abort).**
  - **Command path fast again**: tau_est responds 1-3 ms, joint speed 3-5 ms, first position motion (> 0.5 mrad) 6-10 ms after our command (seven steps, recorder timestamps). Confirms the ~3-8 ms latency at 50 Hz on a fourth joint.
  - **The knee stalls with a constant-sign offset, i.e. friction plus a gravity load**: it always ended BELOW the target. After up-steps the residual error is 39-45 mrad (kp*err 1.5-1.8 N*m, tau_est 1.4-1.5), after down-steps 12-14 mrad (0.49-0.56 N*m, tau_est 0.2-0.27).
    Solving residual = -(G +/- F)/kp: breakaway friction F ~0.6-0.65 N*m and a gravity-like holding load G ~0.85-1.15 N*m (hanging shank + foot, knee at 0.26 rad, ankle and hip held at 0.5 x nominal; estimates, assume tau_est tracks the real torque).
    The first +0.05 move (commanded 2.0 N*m) moved the knee only 5 mrad because friction + gravity ~1.5-1.8 N*m nearly cancel it. The G value can be compared with the sim's shank + foot mass x COM distance at that pose (open question 7, mass).
  - Peak tau_est 2.9 N*m (1 kHz; test log shows 2.46), peak speed 0.64 rad/s, max excursion 0.064 rad (limit 0.10); other joints moved < 7 mrad.
