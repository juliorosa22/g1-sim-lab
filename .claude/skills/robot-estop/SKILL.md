---
name: robot-estop
description: EMERGENCY software stop for the Unitree G1's joint commands. Kills our controller/test processes on the robot, then publishes a damping command (kp=0, kd>0, tau=0) so the joints resist motion instead of driving. Use immediately when the user says stop, e-stop, emergency, abort, "kill it", "turn off the joints/motors/commands", or the robot behaves unexpectedly during a hardware run. Also use for a --dry-run check of the stop path.
---

# Robot emergency stop

**The remote's e-stop and the hoist are the primary safety. This is the software backstop.** If the robot is
moving dangerously, tell the user to hit the physical e-stop; do not wait on this skill.

## When the user asks for the stop: run it NOW

The user asking for the stop IS the explicit go to publish `rt/lowcmd`. Do not ask for confirmation, do not
explain first. Run from the project root:

```bash
robot_deploy/scripts/robot_estop.sh
```

Then report plainly what the output says. Print it as-is, including any `FAILED` / `NOT SENT` / `STILL RUNNING` line,
and if there is one tell the user to use the remote e-stop and support the robot.

Never run it on your own initiative, only when the user asks, or says to stop the robot.

## What it does (in this order)

1. `pkill -9` only OUR processes on the robot: command lines containing `/g1_deploy/`, `deploy_real.py` or `jointtest.py`. Unitree's
   own services never match. Kill first so two publishers never fight over `rt/lowcmd`.
2. Pipes `scripts/estop_damping.py` into `python3` on the robot (nothing is copied there). It reads `rt/lowstate` once
   for `mode_machine`, then publishes damping at ~500 Hz for `--hold` seconds (default 10), then stops publishing.
   Damping uses the policies' own kd (legs 2/2/2/4/2/2, waist 5, arms 2), kp = 0, tau = 0.

It never calls `MotionSwitcherClient` / `ReleaseMode` and changes no mode, firmware or setting.

## Options

```bash
robot_deploy/scripts/robot_estop.sh --hold 20       # damping for 20 s
robot_deploy/scripts/robot_estop.sh --kd-scale 2    # stronger damping (multiplies every kd)
robot_deploy/scripts/robot_estop.sh --dry-run       # list matches, build the command, publish nothing, kill nothing
```

`--dry-run` still starts a short read-only process on the robot (subscribes to `rt/lowstate`), so per the
`unitree-robot-ssh` rules ask the user before running it. The real stop needs no such ask.

## Known limits (tell the user when relevant, do not hide them)

- **The damping step showed NO effect when tested (2026-10-29).** Robot on the support at zero torque, PC1 streaming zero
  commands: a kd=2 damping command on one wrist, with the wrist moved by hand, left `tau_est` flat (see `docs/open_issues.md`).
  Only the process-kill step is trustworthy. Do not tell the user the damping will stop the robot. Retest while our
  controller runs in Unitree's debug mode.
- **Damping is not a catch.** A standing robot still falls; damping only stops the joints driving. It removes
  commanded motion, it does not hold the robot up.
- **After `--hold` seconds publishing stops.** What the motors do then is unverified.
- **Needs the ssh link and `rt/lowstate`.** If either is down the script says so and does nothing; use the remote.
- **Another publisher IS active (measured 2026-10-29).** Even with nothing of ours running, `rt/lowcmd` carries ~800 msg/s of
  zero-torque commands (kp=kd=tau=0, mode_machine 6) that do not come from PC2. The likely source is PC1 (192.168.123.161,
  Unitree's motion computer; by elimination, not confirmed). The damping command shares the topic with that stream, so which one
  the motors obey is UNKNOWN. Do not assume damping takes effect until it is tested hoisted.
- **Not covered: the joystick START chain.** `unitree-start-joystick.service` (always running) launches
  `/home/unitree/hhhh/iniciar_robo.sh` when START is pressed, which starts `assistente.py` and `movimentos_militares.py`.
  Neither matches the kill pattern.
- **Only stops our processes.** If something else publishes `rt/lowcmd` (Unitree's `deploy_real` run by hand under another
  name, a stray script), it is not killed and will fight the damping command. Check with `pgrep -af` on the robot.
- The kd table is copied into `estop_damping.py` from `configs/g1_walk_stand.yaml` on purpose (no config parsing in an
  emergency). If `sim_kd` changes there, update it there too.
- If the kill pattern misses a process because the controller is run from somewhere other than `/g1_deploy/`, add its
  path to `PAT` in `robot_estop.sh`.
