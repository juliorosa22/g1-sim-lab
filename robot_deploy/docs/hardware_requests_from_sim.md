# Open requests from the simulation session (for the next hardware sessions)

Source: the simulation session, relayed by the user on 2026-10-29 after reading `docs/sim_parameter_proposal.md` and
`measurements/20261029/REPORT.md`. Status of every item: **NOT STARTED**. Every run that publishes `rt/lowcmd` keeps the usual protocol
(`CLAUDE.md`, `docs/hoisted_joint_tests.md`): a dry-run plan shown first, the user's explicit yes for that exact run, robot hoisted with feet
off the ground, e-stop in hand, `unitree-start-joystick.service` stopped, robot in debug mode entered by the user on the RC.

## Request 1: mass and hands

**Why.** The simulation model has the Inspire hands. Total mass is unmeasured, and the report says whether hands are fitted is unknown.
Mass and its distribution feed the gravity loads and the standing/walking dynamics.

**Proposed steps (cheapest first; steps a-b need no robot commands):**
- a. **Are Inspire hands fitted?** Ask the user to look. Then, read-only on the Jetson: check for hand services/processes and, if the cyclonedds
  python package allows, list DDS topics for Inspire hand state/command topics (`rt/lowstate` carries only the 29 body motors; the hands
  are not in it). Record the answer, the hand model, and whether the hands are commanded separately.
- b. **Total mass.** Read the hoist's scale/load cell if it has one, or weigh the robot on a platform scale. Record with and without the battery and
  hands, and the configuration (arms, head/camera modules fitted: the Jetson shows RealSense nodes). Also look up Unitree's published mass for this variant and compare.
- c. **Mass distribution hints from static torques** (commanded holds, small, hoisted): the left knee already showed ~0.85-1.15 N*m of gravity-like load
  at 0.26 rad from the hanging shank + foot (REPORT.md section 3). Repeating `hold` at 2-3 poses on the knee, one shoulder pitch and one elbow gives
  segment mass x lever estimates the simulation can compare against its model. These are estimates (they assume `tau_est` tracks the real torque, and friction adds +-0.2-0.6 N*m of ambiguity).

**Deliverable:** `measurements/<date>/mass.md` (+ rows in `joints.csv` or a new `mass.csv`): total mass, hands fitted yes/no and model, method, error bars.

## Request 2: leg plant identification at the operating gains

**Why.** Estimate effective inertia and damping of the leg joints at the gains the policy will actually run, using the `sine` profile (never run) and
steps on **ankle pitch, knee and hip pitch**.

**What the first session already established (constraints for the design):**
- Commands act as a pure PD; at 50 Hz publish the response starts 3-8 ms after our message. **Never publish above 50-100 Hz** (200 Hz gave a ~120 ms backlog; 100 Hz is untested).
- Static friction hides the dynamics at small amplitudes (ankle: ~1.0 N*m on the right side; knee ~0.6; the **left ankle shows an extra unexplained resistance** up to 2.3 N*m and should be inspected first). Amplitude x kp must clearly exceed the friction band: with the old low gains (kp 30) a +-0.03 rad ankle step stayed inside the dead band.
- Our test tool's defaults use deliberately LOW gains and amplitudes; operating gains are 3-5x higher, so torque caps, the amplitude and the abort limits must be re-derived per joint and shown in the dry-run.

**Needed changes before running (tool work, laptop only):**
1. `jointtest.py` `sine` profile is fixed at 0.2-1.5 Hz. A leg joint's natural frequency at operating gains is higher: e.g. J ~0.05-0.1 kg*m^2 with kp 150 gives w_n = sqrt(kp/J) ~ 40-55 rad/s (6-9 Hz). Add `--f0/--f1` options (chirp up to ~8 Hz, amplitude chosen so peak speed stays <= 1 rad/s), keeping the fade in/out.
2. At 50 Hz publish the highest useful excitation is ~8 Hz (>= 6 samples/cycle) and the test's own log has 20 ms rows: rely on the full-rate recorder (`record_walk.sh`, ~1 kHz) alongside every run, as in runs 4-9.
3. Make the torque trip use the 1 kHz state, not the publish-rate state (REPORT.md: the 50 Hz watchdog can miss spikes shorter than 20 ms), or keep generous caps and low amplitudes.

**Gains to identify at.** Decide with the simulation session which set: ours (legs 100/100/100/150/40/40, kd 2/2/2/4/2/2) or Unitree's onboard (100/100/100/150/60/60, kd 4/4/4/5/3/3). Run both for the ankle pitch if time allows: it is the joint where they differ.

**Run list (each its own dry run + yes; stop at the first surprise):**
1. Right ankle pitch (clean side): `step` then `sine` at the chosen gains, recorder alongside. 2. Left ankle pitch (only after the extra resistance is inspected). 3. Knee. 4. Hip pitch.
Hold the other joints as before (0.5 x onboard nominal gains).

**Analysis (stated limits):** second-order fit to step and chirp responses with kp, kd known. Report effective inertia and damping with the spread over repeats and over amplitudes, and say that:
the hanging leg's inertia is the limb's own (shank + foot for knee/ankle, whole leg for the hip) in the hanging posture; friction (Coulomb + the direction offset) must be fitted too or it biases both numbers; armature is only weakly identifiable when limb mass dominates (report an upper bound, not a measurement); loaded behaviour on the ground is not covered.

**Deliverable:** per-joint rows in `joints.csv` (effective inertia, damping, friction, delay, with confidence), raw logs under `jointtests/` and `walk/`, and a `REPORT.md` section for the new date.

## Order and dependencies
Request 1a-1b first (no commands, cheap). Request 2 needs the tool changes above and the left-ankle inspection. Both are independent of the ONNX/TensorRT work (`docs/open_issues.md`, scripts `export_policy.py`, `trt_bench.*`).
