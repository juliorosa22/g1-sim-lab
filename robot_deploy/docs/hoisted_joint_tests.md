# Hoisted single-joint tests (plant identification)

Goal: measure what one real joint does when commanded, so the simulation's actuator model has measured numbers instead of copied ones
(`docs/sim_parameter_proposal.md`, "What would settle the rest"). **These runs publish `rt/lowcmd`.** Safety rules in `CLAUDE.md` apply
to the letter: one specific run at a time, only after the user's explicit yes in the session, after showing the plan below.

Tools: `scripts/jointtest.sh` (laptop) and `scripts/jointtest.py` (runs on the robot). Not yet run on hardware; the profile generators,
plan refusal and safety trips were tested locally with stand-in data only.

## Preconditions (all of them, every session)
1. **User attests**: robot hoisted with **feet off the ground**, weight fully on the hoist, nothing can swing into a person, **e-stop or
   power in the user's hand**, battery charged. (The RC is the e-stop only when on; the software stop is not a catch.)
2. **Debug mode, entered by the user on the RC** per Unitree's `deploy_real` README (zero torque, then debug mode via L2+R2). We never
   switch modes from software (`CLAUDE.md` rule 3). In the default zero-torque state PC1 keeps publishing zero commands and our
   commands had no visible effect (`open_issues.md`, damping tests).
3. **`unitree-start-joystick.service` stopped** (rule 7; sudo, only on the user's ok) and no `movimentos_militares.py` running.
4. **`jointtest.sh check` reports no problems.** It refuses, among others, if any other publisher is active on `rt/lowcmd` (that means
   debug mode is not active, do not run), if the robot is moving (|dq| > 0.5 rad/s), a motor is not in mode 1, or any motor is above 60 C.
5. Wi-Fi link is up (`ROBOT_HOST_OVERRIDE=10.173.210.67`). The test itself runs detached and finishes on its own if the link drops.

## What a run does
One joint follows a profile; all others are **held at their starting position** with 0.5 x the onboard nominal gains, ramped up over 2 s so
nothing jumps. Waist roll/pitch (13, 14) are absent on this unit and always get kp = kd = 0. The test joint has deliberately low gains and a
tiny amplitude, so the worst commanded torque is a few N*m (`jointtest.py` refuses any plan where kp x largest command error exceeds the cap).
Every run ends the same way: kp fades to 0 over 0.5 s while the target follows the measured position (no jump), 2 s of damping
(kd = held-joint kd, test joint its own kd), then publishing stops. Log: `log.csv` (200 Hz: target, q, dq, tau_est, gains, all q and tau), `meta.json`.

| Profile | Duration | What it asks | Identifies |
|---|---|---|---|
| `hold` | 6 s | nothing: hold the start position | **Acceptance test**: do our commands act at all in debug mode (tau_est becomes non-zero, limb is held)? Sag under gravity (tau_est / kp). |
| `step` | 11 s | +amp, -amp, 0 x 2 (a 2 x amp step each) | Command-to-motion delay, rise time, overshoot (effective damping ratio), steady-state error (stiffness incl. gravity) |
| `sine` | 14 s | 0.2 -> 1.5 Hz chirp, 1 s fade in/out | Effective stiffness and damping vs frequency, phase lag |
| `ramp` | 12 s at amp 0.05 | +-amp at 0.02 rad/s | Static / Coulomb friction (hysteresis between up and down), backlash (dead zone on reversal) |

## Starting parameters (defaults in `jointtest.py`; the user may lower, not raise, without a new discussion)

| Joints | kp | kd | amp (rad) | torque cap (N*m) | max excursion |
|---|---|---|---|---|---|
| wrists 19-21, 26-28 | 20 | 1.0 | 0.05 | 3 | 0.10 |
| shoulders, elbows 15-18, 22-25 | 30 | 1.5 | 0.05 | 4 | 0.10 |
| waist yaw 12 | 40 | 2.0 | 0.03 | 6 | 0.06 |
| hips 0-2, 6-8 | 40 | 2.0 | 0.03 | 6 | 0.06 |
| knees 3, 9 | 40 | 2.0 | 0.03 | 8 | 0.06 |
| ankles 4, 5, 10, 11 | 30 | 1.5 | 0.03 | 5 | 0.06 |

## Automatic abort (any one: ramp to damping, then stop publishing)
no or stale `rt/lowstate` (> 50 ms) | NaN/inf | tested joint beyond 1.5 x max excursion | tested joint speed > 2 rad/s | |tau_est| of the tested
joint > 1.5 x torque cap | a held joint drifts > 0.15 rad | any joint faster than 6 rad/s | any motor above 70 C | file `STOP` (`jointtest.sh abort`)
| SIGINT/SIGTERM. `robot_estop.sh` also kills a running `jointtest.py` (hard stop: publishing just ends).

## Order (each step needs its own yes; stop the ladder at the first surprise)
| # | Run | Why here |
|---|---|---|
| 0 | `check` (read-only) | preconditions |
| 1 | `hold` on joint 21 (left wrist yaw) | does anything we send act? Least dangerous joint. If tau_est stays ~0 and the limb is not held, stop: debug mode is not active |
| 2 | hand-guided damping on joint 21 (`--profile hold --kp 0 --kd 1`, user moves the wrist) | settles the inconclusive damping tests: expect tau_est ~ -kd x dq |
| 3 | `step`, `sine`, `ramp` on wrist yaw 21, then wrist pitch 20, wrist roll 19 | cleanest joints, low inertia |
| 4 | `step` on elbow 18, shoulder pitch 15 | larger arm joints |
| 5 | `step`, `sine` on waist yaw 12 | single waist DoF that matters |
| 6 | `step`, `sine`, `ramp` on ankle pitch 4 (most critical), then knee 3, hip pitch 0 | the legs; limb weight loads the joint, so expect sag |

## Publish rate and timing runs
`--rate HZ` (50..250, default 200) sets the publish rate. The test's own log has one row per publish, so at 50 Hz its resolution is 20 ms. For delay
comparisons across rates, run the full-rate read-only recorder alongside (`record_walk.sh start NAME SECONDS` before, `stop` and `pull` after): it timestamps
our commands and the state at ~700 Hz on the same clock. Unitree's own `deploy_real` publishes at 50 Hz, so rates below 50 Hz are not allowed here.

## Per-run protocol
1. I run `jointtest.sh dry JOINT PROFILE` and show you its `PLAN` / `PREFLIGHT` output (joint, profile, amplitude, gains, torque cap, abort conditions, expected result).
2. You say **yes** to that exact run. Nothing else counts, and a yes covers one run.
3. I run `JOINTTEST_CONFIRM=yes-JOINT-PROFILE jointtest.sh run JOINT PROFILE` (the variable only forces a deliberate step), report the END line, and pull the log.
4. Between runs: check temperatures (`check` prints warnings above 50 C; start is refused above 60 C). Rest the joint if it is hot.

## How the data will be analysed (methods and honest limits)
- **Delay**: time from the step command to the first motion above 3 x the rest noise of q (1.5e-4 rad). Resolution is 5 ms (200 Hz log), so a
  delay under ~10 ms is reported as "< 10 ms", not a number. Cross-check with the 700 Hz state stream recorded alongside if we run `record_walk.sh`.
- **Stiffness / damping**: second-order fit to the step and chirp responses with the commanded kp, kd known. Reported with the fit spread over repeats.
- **Friction / backlash**: from the ramp hysteresis (q error at velocity reversal x kp), not from tau_est alone.
- **Reflected inertia (armature)**: only from the early acceleration of steps; weakly identifiable when the limb mass dominates. Reported as an upper
  bound with the method named, never as a measurement.
- **tau_est** may be derived from the command or current, not an independent torque sensor (the PD-vs-tau_est agreement of 0.96 is consistent with
  either). Anything that rests on tau_est alone is labelled "estimated".
- Hanging limbs are not in their operating posture or load, so friction and gravity effects are only indicative for the loaded legs on the ground.

## Known unknowns / risks
- Whether PC2 commands act in debug mode on this unit is unproven; run 1 answers it before anything else is attempted.
- Whether the firmware limits torque or position below our caps, and its true limits, are unknown (not read from firmware or URDF).
- The held joints may sag or drift a little as their gains ramp in (the drift abort is 0.15 rad).
- Nothing here has run on hardware; the first run is also a test of this tooling.
