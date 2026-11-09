# Task for a new session: simulate the lab floor plan and build the velocity-command planner

Written 2026-11-09 by the simulation session. Use it as the first prompt of a fresh session started in `g1-sim-lab/`.
No hardware is involved: nothing in this task may publish `rt/lowcmd`, touch `robot_deploy/` code that runs on the robot, or use the `.env` robot credentials.

## Goal
Make a model of the lab (walls, doors, furniture; a door about 50 cm wide is the hard case) and a high-level planner that turns a goal point into the velocity
commands `(vx, vy, wz)` the deployed policies accept. Test it closed-loop with the real v11 and v12 policies, so that when the robot is available again the planner is already validated.

## What exists (do not rebuild)
- Policies: v11 (stand, turn in place, creep) and v12 (forward walking). Same interface: 78 obs, 13 actions, command `(vx, vy, wz)`, 50 Hz.
  Checkpoints in `logs/lab_checkpoints/{v11,v12}/`, Play tasks `G1-Walk-Flat-Inspire-V11-Play-v0` and `...V12-Play-v0`, loader `g1_rl/actor_io.py:load_actor`.
- Command shaping used in training and on the robot: deadband 0.05 (norm(vx,vy) and |wz|), then ramp 1 m/s^2 and 2 rad/s^2 (`robot_deploy/controller/commands.py`).
- Selector prototype (v11 below 0.3 m/s forward, v12 above, hysteresis 0.2): `scripts/evaluate_switch.py` (`--route forward --enter-walk 0.3 --exit-walk 0.2`).
- Evaluation results to take limits from: `logs/eval/grid/*.json` (command grid), `logs/eval/plant/*.json` (plant mismatch), `robot_deploy/docs/sim_answers_to_hardware.md`.
- Isaac Lab runs on the laptop (env `isaaclab`, 6 GB GPU, few envs) and on the lab desktop (`.env` LAB_*, needs the Tailscale SSH approval from the user each time). Training only on the lab.

## Command limits the planner must respect (from the stress tests)
| | vx | vy | wz |
|---|---|---|---|
| v11 (stand, turn, creep) | -0.3 to +0.5 | +-0.25 | +-0.8 (up to 1.0 later) |
| v12 (walk) | -0.2 to +0.8 (up to 1.0 later) | +-0.25 | +-0.5 |
- Clip first, then deadband, then ramp. Never feed a command that changes faster than the ramp.
- No sign reversal while moving: if the sign of vx flips with |vx| > 0.5 (v12) or 0.3 (v11), or wz flips with |wz| > 0.5, command zero first and wait until the speed is below 0.3 m/s. (Reversing 1.0-1.4 m/s at once falls 27-95% of the time in sim.)
- Never combine wz above 1.0 with backward or sideways motion (v11 falls there).
- v12 only for forward commands; below 0.3 m/s forward, near zero or backward use v11.
- Floor friction below about 0.4 of the trained values falls (v11 9%, v12 2% at x0.4; 55% / 32% at x0.2): treat slippery floors as unsupported.

## Inputs to ask the user for first (do not invent them)
1. A floor plan of the lab: dimensions of the room(s), wall positions, door positions and widths (the tightest door matters most), and where the robot starts.
2. Furniture and obstacles: position, footprint, height; whether they move.
3. Floor material and whether it is slippery; any thresholds or mats at doors.
4. The goal tasks: from where to where, how often the layout changes.
5. The robot footprint with arms at the sides is about 0.37 m wide (standing pose width unmeasured); ask whether the arms will be tucked.
6. What the robot has for localisation (the G1 has a lidar and a depth camera; nothing is wired up yet) and what the user is willing to add.

## Stages
1. **Floor plan to map.** Turn the plan into an occupancy grid (2D, 2-5 cm cells) and a USD scene for Isaac Lab (walls, doors, obstacles on a flat floor). Keep one source file (YAML or SVG) that produces both.
2. **Response model.** Identify how the base actually follows a command with v11 and v12 in Isaac Lab: velocity response (lag, overshoot, steady-state error), turning radius, and drift while turning in place (seen: planted turns drift under 10 cm). Fit a small kinematic model so the planner can be developed in a fast 2D sim (thousands of runs), then validate against Isaac Lab.
3. **Planner.** Global path on the inflated grid (robot radius plus margin, A* or similar), a smoothed path with a stop-and-turn at the door, and a local follower that outputs `(vx, vy, wz)`:
   - align the heading with the door before entering (turn in place with v11), cross at the creep speed (0.15-0.25 m/s), do not turn inside the door;
   - slow to a stop on arrival and when uncertain;
   - the limits and reversal rule above enforced in one place (a `CommandGuard`), with a unit test that fails if any limit is exceeded.
4. **Closed loop in Isaac Lab.** Put the scene and the real policies together with the selector, run start-to-goal tasks (including the 50 cm door, a corner and a tight turn), with observation noise, and report success rate, time, minimum wall clearance and falls. Repeat with the plant-mismatch settings that mattered (friction x0.5, gains x0.8).
5. **Localisation assumptions.** Start with ground-truth pose from the sim, then add noise and drift to the pose the planner sees (e.g. 2-5 cm and a few degrees) to learn how much error the door crossing tolerates. This gives the requirement for the real localisation.
6. **Interface for the robot.** The planner outputs only a command stream `(vx, vy, wz, timestamp)`; the existing `robot_deploy/controller/commands.py` shapes it. Do not edit `robot_deploy/controller*/`; write what the hardware session must add into a task file in `robot_deploy/docs/`.

## Deliverables
- `nav/` package (map builder, 2D sim with the fitted response model, planner, CommandGuard, scenarios) with tests.
- Scenario results (JSON and a short table) from stages 4-5 and a few recorded videos (`scripts/record_video.py` style) sent to the user's laptop.
- A requirements note: position accuracy needed at the door, update rate, sensors, and the open questions for the hardware session.

## Rules
- No robot access, no hardware commits mixed in, commit only your own files (`robot_deploy/` has the hardware session's uncommitted files).
- Commits end with the Co-Authored-By and Claude-Session lines of the session's attribution reminder. Do not push without asking.
- Kill processes by exact PID; run eval scripts with `python -u`; check logs for a traceback even when a `done` marker exists.
