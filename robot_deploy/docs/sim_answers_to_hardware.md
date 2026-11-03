# Answers from the simulation session to `sim_requests_from_hardware.md`
Written 2026-11-03. Policy: v11 (`stand_turn_v11_20261101`, checkpoint `model_5997`). Everything below is **simulation** (Isaac Lab, flat ground, feet in contact, arms held at rest, no hoist).
Request 1 and Request 3 are answered. **Request 2 (what to do after a trip while standing on the ground) is not done yet.**
Reproduce: `scripts/evaluate_delay.py`, `scripts/summarize_delay.py` (tables in `docs/sim_delay_sweep_v11.md`), `scripts/evaluate_joint_demands.py`.

## Request 1: delay budget

### Exactly what v11 was trained with
- The 12 leg joints and waist yaw use a **delayed ideal-PD actuator** (Isaac Lab `DelayedPDActuator`, explicit PD at the real gains). The arms are held by an implicit PD with no delay.
- The **joint position target** (and the velocity/effort setpoints, which are zero) reaches the PD law **lagged by a whole number of physics steps**. A physics step is 5 ms. The lag is drawn **uniformly from {0, 1, 2, 3, 4} steps = 0, 5, ..., 20 ms**, **once per environment at every episode reset, and fixed for the episode** (not per step, not per tick). Legs and waist yaw draw it independently.
- On top of that the action is held for one control tick: the policy runs at 50 Hz and its target is applied for 4 physics steps (**zero-order hold, 0-20 ms, mean 10 ms**).
- **No observation delay** in training. The observation is the state at the tick, plus observation noise (gyro +-0.2, gravity +-0.05, joint position +-0.01 rad, joint velocity +-1.5 rad/s uniform, a fixed per-episode gyro bias +-0.005 rad/s and encoder bias). No late ticks in training.
- Physics 200 Hz, control 50 Hz.

### How each sweep was done (`evaluate_delay.py`)
Always on top of the trained model, 3 seeds x 64 envs per setting (3 groups in one batch; "seed std" below is between those groups), each scenario below.
- **A. Extra action delay E (0, 10, 20, 30, 40, 60 ms):** E/5 physics steps are ADDED to every env's trained lag, so the target lag is uniform in [E, E+20] ms.
- **B. Extra observation age a (10 ... 60 ms):** the sensor part of the observation (gyro, gravity, joint positions, joint velocities) is the linear interpolation between the observations of the last control ticks, a ms old. The command, the previous action and the gait clock stay current. The trained 0-20 ms action lag is untouched. (Sub-tick ages are interpolations.)
- **C. Late ticks:** at each tick, with probability 1% per env, the policy "stalls" for k = 1, 2, 3, 5 consecutive ticks (20-100 ms): the previously applied action is sent again (PD keeps the last target and gains). The policy keeps reading observations.
- Scenarios: **stand** (zero command, 30 s); **turn 0.3** and **turn 0.5 rad/s** (stand 3 s, yaw command ramped at 2 rad/s^2, held 6 s, then zero for 6 s); **push** (stand, a 0.3 m/s shove in a random direction at 4, 9 and 14 s, 20 s).
- Normal band per metric = baseline (trained model) mean +- max(3 x between-seed std, 15 % of the mean, a floor: tilt 0.3 deg, drift 0.5 cm, torque 1 N*m, chatter 0.01, yaw rate 0.03 rad/s, settle time 0.3 s). The fall rate leaves its band above baseline + 0.02.

### Results (fall rate; full tables with tilt, drift, yaw rate, settle time, torque peaks and chatter in `docs/sim_delay_sweep_v11.md`)
| setting | stand | turn 0.3 | turn 0.5 | push 0.3 |
|---|---|---|---|---|
| **baseline (no extra delay)** | 0 | 0 | 0 | 0 |
| extra action delay 10 ms | 0 | 0 | 0 | 0 |
| 20 ms | 0 | 0 | 0 | 0 |
| 30 ms | 0.01 | 0 | 0 | 0.02 |
| 40 ms | 0.02 | 0.02 | 0.02 | 0.17 |
| 60 ms | 0.09 | 0.27 | 0.29 | 0.61 |
| observation age 10 ms | 0 | 0 | 0 | 0 |
| 20 ms | 0.01 | 0 | 0 | 0.01 |
| 30 ms | 0.01 | 0.01 | 0.01 | 0.06 |
| 40 ms | 0.02 | 0.02 | 0.02 | 0.21 |
| 60 ms | 0.13 | 0.29 | 0.32 | 0.60 |
| stall 1, 2, 3 or 5 ticks (1 % per tick) | 0 | 0 | 0 | 0 |

Behaviour inside the band: tilt RMS 0.9 deg standing, 3.0-3.3 deg turning; drift 0.3 cm standing, 17 / 12 cm of wander while turning at 0.3 / 0.5; yaw rate reached 0.34 / 0.54; settle time 1.5 / 2.3 s after the turn.

**First setting where a metric leaves its band** (any scenario): extra action delay **10 ms** (only the turn-0.5 drift: 12.3 -> 14.9 cm) and observation age **10 ms** (the same, 15.1 cm). At 20 ms several metrics leave the band: peak tilt while turning (5.7 -> 6.8 deg), turn drift (16.9 -> 21 cm), standing drift (0.3 -> 0.9 cm), push chatter and hip-pitch torque. These are 10-30 % changes. At 30-40 ms the yaw rate reached falls and the settle time grows (turn 0.3: yaw 0.34 -> 0.28 at 40 ms).

**Late ticks:** no falls in any scenario for stalls up to 5 ticks (100 ms) at 1 % per tick. Only the peak tilt while turning rises out of its band at 5 ticks (5.7 -> 7.3 deg, 6.3 -> 7.9 deg). This tested isolated random stalls (about 0.5 per second at 1 %), not frequent or long ones.

### What this means
| Quantity | Behaves like baseline up to | No extra falls up to | Clearly worse from |
|---|---|---|---|
| Extra action delay (target lag uniform in [E, E+20] ms) | **10 ms** | **20 ms** (30 ms: 2 % push falls) | 40 ms (push 17 % falls), 60 ms (turning 27-29 %) |
| Observation age | **10 ms** | **20 ms** (1 % falls) | 30 ms (push 6 %), 40 ms (21 %), 60 ms |
| Late ticks, held last action, 1 % per tick | **3 ticks** (60 ms) | **5 ticks** (100 ms) tested | not found up to 5 ticks |

Reading your measurements against this (your numbers: controller cycle <= 2 ms, state age at publish <= 5 ms, motor path 3-8 ms, controller->bridge phase delay constant in 0-20 ms, not yet measured):
- The measured sensor-to-motor delay of about 5-30 ms is **inside** what v11 was trained on plus the 10 ms margin. A bridge phase delay of 0 would put you at about 5-15 ms total, equal to the trained range. A phase delay near 20 ms puts you at about 25-35 ms, i.e. an extra of 10-20 ms: still no extra falls but already outside the "baseline behaviour" band (more drift, slower settling).
- So the bridge phase delay is the only thing worth fighting for: it is the one you can remove, and up to 20 ms of it is as large as the whole budget.

### Recommendations (grounded in the above)
1. **Bridge: publish when a new command arrives** (event-driven), not on a free-running 50 Hz tick. The phase-locked wait is 0-20 ms of pure delay and the margin before behaviour changes is about 10 ms. If you keep the fixed tick, shift its phase so the publish happens just after the controller writes (<= 2 ms), and log that offset.
2. **Keep holding the last target on a late tick** (what the bridge does now), but stop treating one late tick as a fault: isolated stalls of 1-5 ticks (20-100 ms) were harmless in stand, turn and push at 1 % per tick. A sensible rule: hold the last command for up to 3 ticks (60 ms); trip on a gap longer than 100 ms, or on 3 or more late ticks inside one second (frequent stalls are untested).
3. **Stale-state trip:** your 50 ms is too late. Observation age of 30 ms already makes push recovery worse (6 % falls) and 40 ms gives 21 %. Suggest a warning at 20 ms and a trip at 30-40 ms. Your measured state age (mean 0.5 ms, max 5 ms) leaves a large margin.
4. **Dead-man 60 ms** is conservative relative to the 100 ms of isolated stalls tolerated; keep it for safety, or raise to 100 ms if it trips on harmless events.
5. The loop-period trip (40 ms) is stricter than the policy needs, as in 2.
Caveats: all of this is flat ground and feet in contact; the hoisted robot (feet free) responds differently. The simulated motors are ideal PD with a position-target lag; real friction, backlash and the torque shortfall (3-7 % measured earlier) are not in these numbers.

## Request 3: what v11 asks of the joints
Turn in place: stand 3 s, then yaw command +-0.2, +-0.3, +-0.5 rad/s (ramp 2 rad/s^2, 3 s hold); 128 envs; statistics over the turning segment (ramp + hold) of all envs and ticks. "|target - q|" = the target computed at tick k (default + scale x action) against the joint position measured at the same tick k, i.e. what a tracking trip compares. Standing = the same run with zero command, the first 1.5 s skipped. Clean / with the training observation noise are given where they differ.

| Joint | standing p99 / max |tgt-q| (rad) | turning p99 (6 cases) / max |tgt-q| (rad) | standing p99 / max torque (N*m) | turning p99 / max torque (N*m) | turning max speed (rad/s) |
|---|---|---|---|---|---|
| hip pitch | 0.13 / 0.26 | 0.16-0.22 / 0.28-0.40 | 12 / 31 | 15-22 / 28-38 | 2.7-3.4 |
| hip roll | 0.13 / 0.15 (noise 0.34) | 0.24-0.29 / 0.38-0.42 | 13 / 18 (noise 30) | 22-28 / 37-40 | 2.2-2.7 |
| hip yaw | 0.05 / 0.07 | 0.15-0.18 / 0.27-0.28 | 5 / 7 | 14-18 / 25-26 | 4.2-4.8 |
| **knee** | 0.06 / 0.38 | **0.44-0.53 / 0.63-0.76** | 9 / 46 (noise 15 / 28) | **57-70 / 101-111** | 9-11 |
| **ankle pitch** | 0.26 / 0.47 (noise 0.38 / 0.87) | **0.50-0.76 / 1.06-1.18** | 14 / 28 (noise 21 / 49) | **29-41 / 50 (the cap)** | 18-23 |
| ankle roll | 0.09 / 0.13 | 0.20-0.29 / 0.53-0.63 | 5 / 9 | 13-20 / **50 (the cap)** | 19-20 |
| waist yaw | 0.00 / 0.02 | 0.03-0.05 / 0.07 | 0.2 / 2.3 (noise 1.9 / 4.5) | 3-5 / 9-12 | 1.3-1.4 |

Findings:
- The demands barely depend on the yaw rate (0.2-0.5): they come from stepping itself. v11 lifts a foot about 4 times a second while turning.
- **Standing-era limits are exceeded by normal stepping:** the tracking error reaches 0.5-0.75 rad at the 99th percentile on knee and ankle pitch (limit 0.30), and a few joints spike to 1.0-1.2 rad (ankle pitch). Knee torque reaches 60-70 N*m at the 99th percentile and 111 N*m at the maximum (stage-3 cap 40). Ankle pitch and roll reach their 50 N*m limit (stage-3 cap 25). Hip roll reaches 40 N*m (cap 40). Waist yaw is far below its cap (max 12 vs 25).
- Joint speeds reach 18-23 rad/s at the ankles and about 10 rad/s at the knee, 5 at the hip yaw, so a global 5 rad/s speed trip cannot be kept for stepping.
- Noise changes things only slightly (standing ankle pitch error and torque are up, p99 0.26 -> 0.38 rad and 14 -> 21 N*m).
- **First-step kick:** your finding that the first step produces a big target-minus-q error on the real robot is consistent with a policy that steps: nothing in the observation tells it about foot contact. These simulation numbers do not include a hoist or feet in the air, where the error is larger because the legs are unloaded.

Suggested Stage B trip limits **from this data** (about 1.3x the largest value seen, rounded; simulation, ground contact):
| Joint | |target - q| (rad) | |torque| (N*m) | joint speed (rad/s) |
|---|---|---|---|
| hip pitch | 0.55 | 50 | 6 |
| hip roll | 0.55 | 55 | 4 |
| hip yaw | 0.40 | 35 | 8 |
| knee | 1.00 | 140 (the URDF limit is 139) | 14 |
| ankle pitch | 1.50 | 50 (hardware limit) | 30 |
| ankle roll | 0.85 | 50 (hardware limit) | 26 |
| waist yaw | 0.15 | 25 | 3 |
Use a looser limit for a few ticks around each foot set-down (the speed and torque spikes are there), or trip on a sustained violation (for example 3 consecutive ticks) rather than on one sample. With weight on the legs a knee torque cap well below 100 N*m will trip on ordinary steps. Because stepping asks for the actuator limits at the ankles, a pure-ankle fault is hard to separate from a normal step by torque alone; the tilt (IMU) and base-height checks are better indicators.

## Not done
- **Request 2** (what the robot should do after a trip while standing on the ground: fade, reduced gains, longer fade): not simulated yet. The sweep scripts are in place, so this is a small addition; say if you want it.
- Delay sweeps for the stand-only case with arms moving, and for v12 / the v11-v12 switch, were not run.
