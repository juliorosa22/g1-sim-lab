# v11 stand + turn baseline: report (Stage A: stages 0-4 done; the turning stage B not done)

Task: `docs/first_standing_baseline_task.md` (simulation session, 2026-11-01). One G1 unit, controller in `controller/`, policy `stand_turn_v11_20261101`.
All robot runs: Ethernet, Unitree debug mode entered by the user on the RC, `unitree-start-joystick.service` stopped, e-stop in the user's hand, an explicit yes for each run. Every published run went through the C++ bridge (`controller_cpp/bridge/`), which owns all DDS traffic.

## Stage 0 (laptop): done
- `tests/contract_test.py`: observation 1.5e-6, policy 1.1e-6, targets exact over 2400 simulation steps; joint order and defaults match.
- `tests/real_state_check.py` on our real standing recording: no NaN, tilt 0.45 deg, |joint_pos - default| <= 0.10 rad, raw actions <= 0.39, targets <= 0.097 rad from the measured pose.
- `tests/test_runner.py`: phases, every safety trip -> stop sequence (right reason), dry run sends nothing, publish refused without confirmation, start-pose guard, stage2/stage3 torque caps, the 2 s fade. Mutation-checked.

## Stage 1 (robot, read-only dry run, hoisted, zero torque, 2026-11-01 ~12:15): done
`run.py` dry run, 23.5 s, 1176 ticks: 0 commands sent, no trips, all phases visited. Log: `stage1_dry1/log.csv`. Loop time mean 2.1 ms (max 9.7), tilt at rest 1.85 deg, gyro at rest <= 0.012 rad/s.
**Finding:** the hanging pose was far from the default (left wrist roll -0.88 rad, left hip yaw +0.79, ...), and the raw actions computed there are large (up to 5.2) because the state is outside the training distribution. They do not predict later stages (see "dry runs at a non-default pose" below). The controller refuses to publish when a joint is farther than `--max-start-diff` (0.6 rad) from its default; the robot was hung closer to the default by hand before Stage 2.

## Tooling built on the way (all laptop-tested before use; details in `docs/open_issues.md`)
- **Damping from the Jetson verified** (debug mode): `tau_est` follows -kd x speed (arm test slope -2.91 vs -3.00; left knee, hand-guided through the bridge: slope -4.45 vs -5, correlation -1.00).
- **C++ bridge** `g1_bridge`: the only process that publishes `rt/lowcmd`. Python hands it commands through shared memory. Command firewall, state checks, dead-man (no new command for 60 ms -> fade and damping; measured 44.5 ms after a SIGKILL on the robot), one run per process, two-key publish gate. Reason: the Python DDS publish stalled up to 32 ms, the C++ write is 0.15 ms mean, 0.45 max.
- **`--on-arrival`**: the bridge publishes a new command as soon as it arrives. Controller-write to publish delay: 0.34 ms mean (max 0.78) vs 5.3 ms (max 6.0, phase dependent) with the fixed 50 Hz tick.
- **Torque caps by stage** (`--caps stage2|stage3`): stage2 (hoisted in the air) hips 30, knee 35, ankle 20, waist yaw 20, arms 15 N*m; stage3 (weight on the legs) hips 40, knee 40, ankle 25, waist yaw 25, arms 15; defaults 60-80 / 80 / 40 / 50 / 25. Tracking limit 0.30 rad kept (it trips before the caps everywhere except waist yaw).
- **`--stop-fade-s 2` with bridge `--fade-s 2`**: a trip fades kp over 2 s (not 0.5 s) and then damps for 3 s, so that a trip with weight on the legs lets the robot settle onto the hoist.

## Stage 2 (robot hoisted, feet in the air, commands enabled, zero velocity command): done, 3 runs
Runs `stage2-hoisted`, `stage2-lat`, `stage2-arr` (`../stage2/`). RAMP 3 s, HOLD 2 s, BLEND 5 s, RUN 10 s, stop. No trips in any run; bridge ended STOPPED "clean end", 0 rejected commands, 0 dead-man events.

| Quantity | Result | Simulation / target |
|---|---|---|
| Tilt | <= 0.83 deg | |
| Max joint speed | <= 0.55 rad/s | abort at 5 |
| Max `tau_est` (any joint) | 3.2 N*m (legs <= 2.6) | stage2 caps 15-35 |
| Max tracking error | 0.046 rad | limit 0.30 |
| Leg joint range during the 10 s RUN | <= 0.008 rad (arms 0.05) | |
| Action RMS RAMP / HOLD / BLEND / RUN | 0.45 / 0.24 / 0.15 / 0.09 (RUN max 0.21, chatter 0.004) | |
| Python cycle with the policy | mean 1.2 ms, p99 1.6, max 2.0 | well inside 20 ms |
| Tick period | mean 20.000 ms, max 20.3 | trip at 40 ms |

## Stage 3 (feet on the floor, hoist taking weight, then slack): done, 4 runs, 3a-3c
The user lowered the hoist by hand during the RUN (an assistant stood by or steadied the torso). `--caps stage3 --stop-fade-s 2` from 3b on. Data in `../stage3/`.
- **`stage3a-lower` (run1): airborne.** The soles were NOT lowered (user's correction); the run is another quiet airborne run (max 3.7 N*m, tilt 1.14 deg).
- **`stage3a-lower2` (run2): first ground contact, hoist carrying the weight.** Ankle pitch torque rose from about 0 to +3.4/+1.9 N*m over 4 s and stayed flat; knees stayed at about -0.2 N*m (< 10% of full standing); joint angles moved < 0.02 rad; action RMS 0.13, tilt 0.8 deg; max `tau_est` 4.4 N*m.
- **`stage3b` (run3): legs carried most of the weight** (an assistant handling error; the aim was about half). Knee torque -3.2 N*m flat for 32 s, no oscillation; `tau_est` max 7.1 N*m; tilt 1.13 deg in RUN, 3.56 deg in the 2 s fade.
- **`stage3c` (run4): smooth transition from fully on the hoist to fully on the legs, hoist slack by a few cm, assistant standing by** (user's account). 2251 ticks, no trips. Knee torque grew gradually from ~0 to -3.3/-3.8 N*m, ankle pitch torque to +7.2/+5.6 N*m (15-21 s), then flat to the end of RUN. Tilt max 1.29 deg in RUN; 6.58 deg in the 2 s fade/damping, ended at 2.03 deg, no fall.

### Standing measurements against the task's table (3c, the last 12 s of RUN, legs alone; 3b, 32 s with most of the weight, in brackets)
| Quantity | Robot | v11 in simulation |
|---|---|---|
| Knee angle, standing | 0.285 / 0.288 rad L/R (3b: 0.281 / 0.283) | 0.288 rad |
| Knee torque, standing | -2.47 / -3.20 N*m L/R (3b: -3.18 / -3.19) | 3.5 N*m (magnitude; sign by log convention) |
| Ankle pitch angle, standing | -0.198 / -0.191 rad (3b: -0.184 / -0.178) | -0.20 rad (default) |
| Ankle pitch torque, standing | +7.19 / +5.61 N*m (3b: +3.3 / +2.9, assistant carrying part) | not provided: please send the simulation's full-standing ankle and hip pitch torques |
| Hip pitch torque, standing | -0.30 / -0.50 N*m | not provided |
| Tilt (IMU), standing | mean 1.11 deg, std 0.10, max 1.29 (3b: mean 0.47, std 0.06) | < 2 deg RMS |
| Gyro z RMS / max leg joint speed / leg angle std | 0.0023 rad/s / 0.058 rad/s / 0.0012 rad | |
| Waist yaw torque | -0.07 N*m mean, 0.28 max | |
| Action RMS / max / chatter | 0.19 / 0.56 / 0.0036 | |
| Base height | not measured (no kinematics or markers) | 0.784 m |
| Drift / sway | not measured (no state estimator, no floor markers) | 0.3 cm in 17 s |
| Foot lifts at zero command | not measured (no contact sensor in `rt/lowstate`; none seen by the user) | 0 |
| Loop time, command latency | cycle 1.25 ms mean (max 2.4); bridge publish 0.34 ms after the controller write; motor response 3-8 ms after (earlier session) | 3-8 ms |
| Max motor temperature after the runs | 53 C (right shoulder roll), legs cooler | |

The knee angle and ankle angle agree with the simulation within 0.01 rad, the knee torque within about 10% (left knee 29% lower in 3c). The stationary forward lean of 1.1 deg and the ankle pitch torque (7.2 / 5.6 N*m, about twice that of 3b) need the simulation's number to be judged.

## Stage 4 (free standing on the legs for 30 s, hoist slack by a few cm, assistant standing by, zero command, no pushes): done (2026-11-01 ~17:00)
Run `stage4` (`../stage4/`): `--caps stage3 --stop-fade-s 2 --run-s 50`, bridge `--on-arrival --fade-s 2`. 3252 ticks, **no trips**, bridge STOPPED "clean end", 0 rejected commands, 0 dead-man events. The user lowered the hoist during RUN and slackened it as in 3c (details of the slack to be added by the user).
The window RUN +20..+50 s (30 s, the robot on its legs alone):

| Quantity | Result | Simulation / target |
|---|---|---|
| Tilt | mean 1.27 deg, std 0.11, max 1.44; first vs last 10 s: 1.281 -> 1.248 (no trend) | < 2 deg RMS |
| Knee angle L / R | 0.284 / 0.288 rad | 0.288 rad |
| Hip pitch / knee / ankle pitch (left) first vs last 10 s | -0.0847 -> -0.0846 / 0.2839 -> 0.2841 / -0.2017 -> -0.2019 rad: no drift | |
| Ankle pitch torque L / R first vs last 10 s | 7.94 / 5.79 -> 7.90 / 5.79 N*m | flat |
| Knee torque L / R | -2.0..-2.2 / -3.2 N*m | 3.5 |
| Gyro z RMS / max leg joint speed | 0.0023 rad/s / 0.08 rad/s | |
| Action RMS / max | 0.20 / 0.60, steady | |
| Max tracking error (policy target - measured q) | 0.152 rad, constant from 15 s on | limit 0.30 |
| Max `tau_est` | 9.1 N*m (left ankle pitch, 36% of the stage3 cap 25), 8.1 in the fade | |
| Motor temperature after the run | 53 C max (right shoulder roll; legs cooler), no change from before | trip at 70 |
| Bridge publish delay | 0.35 ms mean, 1.5 ms max (3251 ticks, the counter advanced by 1 on every tick) | |

Reading: the robot stands still on its own legs for 30 s with a constant 1.3 deg forward lean, constant joint angles (changes of 2e-4 rad) and constant torques; the policy output is steady (no oscillation, no drift). The constant 0.15 rad between the left-ankle target and the measured angle is the ankle held against the load (ankle kp 60 x 0.13 rad = 7.9 N*m, the measured torque): half of the tracking-error trip of 0.30 rad, so that trip would be the first to fire if this grew. The 2 s fade onto the hoist and the assistant ended with a forward tilt of 9.99 deg at the peak and 6.55 deg at the end (3c: 6.58 / 2.03), leg speed 0.98 rad/s, no fall; this is the robot going limp onto its supports, not a measure of balance. What is NOT shown: balance against a push, standing without the hoist and the assistant within reach, base height, drift on the floor.

## Findings and cautions
1. **Dry runs at a non-default pose over-predict the actions.** With the robot held still away from the default, v11's own last-action input makes the open loop oscillate (action RMS 1.4, max 4.8, chatter 0.29 at the Stage 3a pose, deterministic: reproduced offline). In the closed loop the ramp brings the pose to the default and the actions are small (RUN RMS 0.09-0.19); `scripts/closed_loop_toy.py` reproduced the real Stage 2 numbers. Do not judge action size from a dry run at a non-default pose.
2. **On the ground, every trip and the dead-man end in a fade and damping, which drops a standing robot.** Hence the hoist and spotter stay part of the safety system; the 2 s fade only gives time. What a trip should do on the ground is asked of the simulation session (Request 2 in `docs/sim_requests_from_hardware.md`).
3. **Delay budget.** Sensor to publish about 2.5 ms (state age 0.6 + compute 1.3 + publish 0.3-0.7), plus the motor path 3-8 ms: roughly 6-11 ms, inside the 0-20 ms the simulation was trained with (to be confirmed by the delay sweep requested in Request 1).
4. **The robot's clock is about 11 h behind the laptop's** (no NTP); a rebuild with `make` can silently keep the old binary. Always check the binary.
5. The ankle-pitch torque cap of 40 N*m (default) sits at the 99th percentile of the walking torques (37-39): revisit before any walking policy.
6. Not covered: left ankle extra resistance (up to 2.3 N*m, unexplained), the user's hands/mass items, the leg sine identification (`docs/hardware_requests_from_sim.md`).

## Not yet done
- **Stage 5 (turning in place, stage B):** only after the simulation session has reviewed this report and the user says so; yaw-only commands 0.2, 0.3, 0.5 rad/s. The controller supports it behind `--allow-turn`/`--wz-script` and has been tested on the laptop only.
- Motor temperature in the controller log; the simulation session's answers to the delay sweep, the ground-trip question and the full-standing ankle/hip torques.
