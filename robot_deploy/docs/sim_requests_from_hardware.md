# Requests from the hardware session to the simulation session
(Opposite direction of `hardware_requests_from_sim.md`. Written 2026-11-01. Nothing here needs the robot.)

## Request 1: how much pipeline delay and how many late ticks can v11 tolerate?
**Why.** v11 runs on the robot at 50 Hz through a C++ bridge (`controller_cpp/bridge/`) and a Python controller. We need a delay budget grounded in the policy, not a guess: the trip limits we set (stale state 50 ms, loop period 40 ms, dead-man 60 ms) are conservative engineering choices, not derived from the dynamics.

**What we measured on the robot (Stage 2, hoisted, 2026-11-01, `measurements/20261101/stage2/`):**
- Controller cycle with the policy running: mean 1.2 ms, p99 1.6, max 2.0 ms (policy forward pass alone about 0.1-0.5 ms). Tick period max 20.24 ms. Bridge DDS write mean 0.15, max 0.45 ms.
- State age when the bridge publishes: mean 0.5 ms, max 5 ms.
- Motor path (our message to first joint response): 3-8 ms (earlier session).
- NOT yet measured: the delay between the controller writing a command and the bridge publishing it. The two 50 Hz loops are phase-locked, so it is a constant somewhere in 0-20 ms. A bridge rebuild that logs it is being prepared.
- So the real sensor-to-motor delay is roughly 5-30 ms, i.e. at the upper end of the 0-20 ms actuation delay the simulation trained with (please state exactly what v11 was trained with: observation delay, action delay, range, per-episode or per-step).

**A. Added action delay.** Evaluate v11 with EXTRA action delay of 0, 10, 20, 30, 40 and 60 ms on top of the trained model (interpolate or use the physics step; say how you did it).
**B. Added observation (state) age.** Same sweep for observations only, if the setup allows it separately from A.
**C. Late ticks.** At random times hold the previous action for k = 1, 2, 3, 5 consecutive ticks (20-100 ms, PD keeps the last target and gains), with probability e.g. 1% per tick. Does v11 recover or fall?
**Scenarios** (each with several seeds): stand at zero command (30 s); turn in place at 0.3 and 0.5 rad/s (the task's stage B); a small push (the same one used for the stander evaluation) during standing.
**Report, per delay value:** fall rate; tilt RMS and max; position drift; time to be still after a turn; yaw rate reached; peak joint torque for knee, ankle pitch, hip pitch, waist yaw; action chatter (RMS of a_t - a_(t-1)); the first delay at which each metric leaves its normal band.
**Deliverable.** One table and the largest extra delay (and longest stall) that keeps v11 inside its baseline behaviour. We use it to set the bridge design (publish when a new command arrives, or keep the fixed tick) and the trip limits.

## Request 2 (optional): what should the robot do after a trip while standing on the ground?
Every trip and the dead-man currently fade kp to 0 over 0.5 s and then damp (kd only). Hoisted that is benign; **on the ground it drops a standing robot**. In simulation, compare after a trip at t0 while standing: (a) the current fade + damping, (b) keep the last target at reduced gains (50%, 25%) for 1-2 s then damp, (c) fade over 2 s. Report time to fall and the impact (base velocity at contact, peak joint torque). Informs the ground-stage design (hoist and spotter remain mandatory regardless).

## Request 3: what does v11 ask of the joints while turning in place? (for the torque caps and the tracking limit of Stage B)
The trips on the robot are tuned for standing: |target - measured q| <= 0.30 rad and, on the ground, stage3 caps (hips 40, knee 40, ankle 25, waist yaw 25 N*m). Turning in place makes v11 step (240-260 foot lifts/min). In simulation, for yaw commands 0.2, 0.3 and 0.5 rad/s (both signs), 3 s hold, please report per leg joint and waist yaw: the 99th percentile and the maximum of |target - measured q| (rad) and of |torque| (N*m), and the largest joint speed, so we can set the Stage B limits from data instead of guessing. Also the full-standing hip pitch and ankle pitch torques (Request 1). Why now: the first turning run is being prepared and a limit that trips on normal stepping would waste a run and, with weight on the legs, drop the robot onto the hoist.
