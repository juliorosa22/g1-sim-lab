# Request 2: what should the robot do after a trip while standing on the ground?  (simulation answer)

Script: `scripts/evaluate_trip_stop.py` (256 envs per strategy and scenario, observation noise on; raw results `logs/eval/trip/trip_v11_d05.json` and `trip_v11_d10.json` on the laptop, 2026-11-18).
Policy v11 (`model_5997`). The robot stands (scenario "stand") or turns in place at 0.3 rad/s (scenario "turn 0.3", the false alarm of Stage 5a) and a trip happens at a random time in 5.0-5.8 s
(every gait-clock phase). From the trip on, the policy output is FROZEN (the target stays where it was) and the legs and waist yaw get the strategy's stiffness schedule; damping kd = 0.5 x the
trained kd (about what `scripts/estop_damping.py` sends) or 1.0 x. "Fell" = the simulation's own termination (pelvis or torso touches the ground with more than 1 N, or the tilt exceeds 57 deg)
within 6 s. Impact = base speed just before that step (total, and the downward part).

## Result in one line
**No stop strategy keeps a free-standing robot up: every one of them falls 100% of the time** (stand and turn, kd x0.5 and x1.0). Holding stiffness longer only delays the fall (0.85 s to 1.5 s)
and makes it harder (impact 1.1 m/s to 1.8-2.3 m/s). Only NOT stopping (the policy keeps running) stays up: 0% falls.

## Table, damping kd x0.5 (stand | turn 0.3): time from trip to fall (s), impact speed (m/s), peak leg torque p95 after the trip (N*m)
| strategy | stand: fall s / impact / torque | turn: fall s / impact / torque |
|---|---|---|
| none (policy keeps running) | no fall / - / 35 | no fall / - / 92 |
| fade 0.5 s (today) | 0.85 / 1.07 / 24 | 0.82 / 1.36 / 56 |
| fade 1.0 s | 1.25 / 1.14 / 31 | 1.00 / 1.60 / 63 |
| fade 2.0 s | 1.46 / 1.78 / 24 | 1.01 / 2.33 / 59 |
| hold 50% for 1 s, then damp | 1.45 / 1.58 / 28 | 1.16 / 1.64 / 48 |
| hold 50% for 2 s, then damp | 1.48 / 1.96 / 26 | 1.15 / 2.08 / 49 |
| hold 25% for 1 s, then damp | 1.30 / 1.08 / 40 | 0.98 / 1.14 / 49 |
| hold 25% for 2 s, then damp | 1.32 / 1.47 / 37 | 1.01 / 1.25 / 49 |
| full stiffness 1 s, then fade 1 s | 1.52 / 1.86 / 27 | 0.98 / 2.26 / 61 |
| **damp now (kp = 0 at once)** | **0.57 / 0.65 / 24** | **0.60 / 0.54 / 27** |
With kd x1.0 the picture is the same: fall times 0.66-1.54 s (stand) and 0.64-1.16 s (turn), impact 0.9-1.9 and 0.8-2.4 m/s, damp now still the lowest (0.90 and 0.81 m/s), fade 2.0 the highest (1.76 and 2.41).
Stronger damping makes the peak torque higher (the legs resist more), not the outcome different. 100% fell in every row; no run "sagged" and stayed on the ground in a heap without a termination.

## Reading
- A frozen target is an open-loop pose: a standing robot is only stable with feedback, and v11 stands because it reacts every 20 ms. Any stop that freezes the output (even at full stiffness for 1 s) falls once the
  disturbance (the turn, the step) is not corrected; what the stop gains is time, not balance. Fade 2.0 and "hold" buy 0.6-1.0 s and cost impact: the robot falls from a higher, more committed lean.
- The softest landing in simulation is **damp now**: the legs collapse at once, 0.55-0.65 m/s at impact, about half of today's 1.1-1.4 m/s, and the lowest torques. Today's fade 0.5 s is close to it (0.85 s, 1.07-1.36 m/s).
- During a turn the legs carry much more load (p95 torque 56-92 N*m against 24-35 standing): a trip in mid-turn is harsher, and a torque cap of 40 N*m (stage 3) trips there regardless of the strategy.
- "none" shows what a **false alarm** costs when it is ignored: nothing, the policy keeps balancing (0% falls in both scenarios). The 5a trips (tracking 0.30 rad, joint speed 5 rad/s) were false alarms of this kind, and stopping on them
  is what would drop the robot on the ground.

## Recommendation for the controller (the hardware session decides; `run.py` and the bridge are theirs)
1. **Do not stop for marginal limits while the robot is on the ground.** Tracking error, joint speed and torque limits that normal stepping crosses should log and continue (or stop only above a hard bound, or when they persist
   for several ticks), as the demand table in `sim_answers_to_hardware.md` allows (tracking about 0.8 rad, speed about 10 rad/s, torque caps above the turn demand). Keep the immediate stop for: tilt above the limit, stale or invalid
   state, NaN, the dead-man, the user's stop. This is the only change that keeps a free-standing robot up.
2. **Do not lengthen the fade.** Keep today's 0.5 s or go to damp-now for the hard stops; fade 2.0 and holding stiffness raise the impact speed by 1.5-2x in simulation. (Stage 3/4 used `--stop-fade-s 2.0` hoisted with an assistant: fine there, wrong for the ground.)
3. **For the ground, the protection is mechanical**: a gantry or harness carrying the weight (and a soft floor or mat). No electronic stop sequence saves a free-standing robot in simulation.
4. **A hand-over to Unitree's own controller is the only strategy that might keep it standing** (it is a feedback controller), but it cannot be tested here: it is the console test (`docs/task_hardware_console_test.md`, steps D and E), hoisted first. Until it is measured, treat it as a possible
   improvement, not a safety guarantee.

## Caveats
Arms keep their implicit hold in simulation (on the robot they fade too); flat rigid floor, no mat; no hoist or gantry; policy frozen at the trip (a controller that held the LAST ACTION but kept reading the sensors would be the same); the
trip is random in a 0.8 s window, one policy (v11), observation noise on; "fell" includes the pelvis touching the ground, which on the robot may be a controlled sit; the impact speed is the base speed one control step before the termination.
