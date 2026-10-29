# Prompt for the hardware session: identify the real G1 parameters (safely)

Paste everything below the line into a new session started inside `robot_deploy/`.

---

You are the hardware session for our Unitree G1 project. Read `CLAUDE.md` (safety rules), `docs/open_issues.md`
and `configs/g1_walk_stand.yaml` first. The robot is reachable with `scripts/robot_check.sh`.

## Goal

The simulation team (another session) will **retrain** the walking policy (legs + waist, observes all 29 joints) and the
standing policy (legs only) so the simulated joints behave like the real ones. To do that we need **measured** real
parameters instead of numbers copied from other simulators. Your job is to plan and, only with my explicit confirmation
for each step, carry out SAFE tests that identify them, and to hand the results back in a form the simulation can use.

A framing point: PD gains (`kp`, `kd`) are not a property of the robot. They are values our controller sends in the command.
What we need to learn is (a) what the real joints do under a given kp/kd, and (b) the plant behind it. Please keep these
two apart in your reports.

## Safety (non-negotiable)

- Start read-only. Nothing that publishes `rt/lowcmd` runs without my explicit "yes" for that specific run.
- Any test that moves a joint: robot **hoisted with feet off the ground**, e-stop/remote in my hand, one joint at a time with
  the others in a safe state, small amplitudes, hard caps on position excursion and torque, a watchdog that goes limp on
  stale state / NaN / limit breach / my abort, and temperature checks between runs. Never drive a joint toward its limits.
- Do not test joints whose status is unknown (see question 1). Do not change robot modes, firmware or system settings.
- Before each moving test, show me: joint, command profile, amplitude, gains, torque cap, abort conditions, expected result.
  Wait for my yes.

## Questions to answer first (read-only wherever possible)

1. **Which joints exist and are actuated?** Unitree's SDK marks `WaistRoll`/`WaistPitch` (motors 13, 14) "INVALID for g1 23dof/29dof
   with waist locked". Are this unit's waist roll/pitch unlocked and live? (Check `mode`, `motorstate`, `tau_est`, whether `q` is valid.)
   Which hands are fitted (Inspire) and are they commanded separately?
2. **Zero offsets and signs.** In the pose I put the robot in (we will agree on one), what is `q` for each of the 29 motors, and does
   the sign convention match our default pose in `configs/g1_walk_stand.yaml` (legs -0.1/0/0/0.3/-0.2/0, arms 0.35/±0.18/0/0.87/0/0/0)?
3. **IMU.** At rest: gyro bias and noise (std), gravity-vector noise, orientation convention (frame, quaternion order), and the
   pelvis vs torso IMU. Our policies assume body-frame angular velocity and a gravity vector in the body frame.
4. **Timing.** State message rate and jitter on `rt/lowstate`; if later allowed, command-to-state round-trip latency (ms).
   Our training randomises 0-20 ms of actuation delay at a 50 Hz control rate.
5. **Limits.** Real per-joint position limits, velocity limits and peak/continuous torque limits (from the SDK/URDF/docs first, then
   confirm what the firmware enforces). Our simulation uses ankle 50, knee 139, hip roll 139, hip pitch/yaw 88, waist yaw 88,
   waist roll/pitch 35 N*m.
6. **What gains does the robot itself use?** Does the onboard controller publish its own `rt/lowcmd` (listen only: kp, kd per motor)?
   What gains do Unitree's SDK low-level examples and `deploy_real` use? List each source; none of these is a measurement.
7. **Mass.** Total mass and what is attached (hands, battery). Our model's masses come from the Inspire USD.

## Tests to propose (in this order; I approve each)

A. **Read-only listen** (10-30 s, nothing published): answers 1-4 and part of 6.
B. **Passive checks in zero-torque mode**, hand-guided only if I choose to: confirm signs and ranges of motion.
C. **Hoisted, one joint at a time, low gain, tiny amplitude:** position steps and slow sine sweeps. From the response estimate per joint:
   effective stiffness and damping actually achieved for the commanded kp/kd, reflected inertia (our sim uses armature 0.01),
   static and viscous friction (slow velocity ramps), command-to-motion delay, torque saturation behaviour, backlash.
   Start with joints whose failure is least dangerous (wrists, arms) before legs and waist.
D. **At the operating gains we plan to run** (legs 100/100/100/150/40/40 with kd 2/2/2/4/2/2; waist and wrist values to be
   decided with me; see the gain table in `docs/open_issues.md`): confirm the joint tracks a small step without oscillation.
   Raise gains gradually from a lower value; stop at the first sign of oscillation or heat.

## Deliverables

Create `measurements/<yyyymmdd>/` in this folder with:
- `joints.csv`: per joint: motor index, name, sign, zero offset, position/velocity/torque limits, measured effective kp/kd for each
  commanded pair, reflected inertia estimate, static and viscous friction estimate, delay (ms), notes and confidence.
- `imu.json`: noise and bias statistics, frame conventions.
- `timing.json`: state rate, jitter, latency.
- raw logs (CSV) and a short `REPORT.md`: what was run, with what limits, what was found, what is uncertain, and which simulation
  parameters should change. Update `docs/open_issues.md` and `configs/g1_walk_stand.yaml` comments where reality differs.

State your uncertainty honestly: say "estimated" and give the method and error bars; do not present a fit as a measurement.
Start by listing what you would do for step A and wait for my go.
