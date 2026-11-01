# Bring-up checklist (staged, with safety gates)

Do the stages in order and do not skip. Each stage needs the user's explicit go-ahead in the session.
Procedure for the robot states follows Unitree's `deploy_real/README.md`.

## Stage 0: connection (no risk)
- [ ] Ethernet cable connected; laptop wired interface has a `192.168.123.x/24` address.
- [ ] `scripts/robot_check.sh` passes (ping, ssh, versions, deploy dir).

- [ ] `systemctl is-active unitree-start-joystick.service` (read-only). Must be stopped before any policy test (Stage 3
      onward); the RC's START button would otherwise launch `iniciar_robo.sh` / `movimentos_militares.py`. See CLAUDE.md rule 7.

## Stage 1: files only (no risk)
- [ ] `scripts/robot_sync.sh` (dry run), review the list, then `scripts/robot_sync.sh --apply`.
- [ ] On the robot, confirm the files and that Python 3.8 + torch import the exported policies.

## Stage 2: read-only on the robot (no motion)
- [ ] Robot powered, **hoisted**, in zero-torque mode. E-stop / remote in hand.
- [ ] Subscribe to `rt/lowstate` only; print joint positions in the default pose and the IMU readings.
- [ ] Compare joint zero offsets/signs with `configs/g1_walk_stand.yaml`; check IMU frame (gravity vector, gyro sign).
- [ ] Measure state-message rate and jitter (target: 50 Hz loop with margin).

## Stage 3: dry run of the controller (still no commands sent)
- [ ] Controller reads real state, builds observations, runs both policies and the selector, **logs** actions. Sends nothing.
- [ ] Inspect: observation ranges plausible, actions bounded, selector behaves with a scripted command sequence,
      loop time well inside 20 ms.

## Stage 4: hoisted, commands enabled, feet off the ground
- [ ] Debug mode as in Unitree's procedure (zero torque, then L2+R2). Start the controller; joints are limp.
- [ ] `start` -> default pose. Verify motion is smooth and torques are sane. Watchdog tested (kill the controller: robot must go limp/safe).
- [ ] Standing policy only, then walking policy only, at tiny commands, still hoisted.

- [ ] Plant identification (hoisted, one joint at a time, low gains): follow `docs/hoisted_joint_tests.md`; each run needs its own yes.
- [ ] Damping check in debug mode (stop sequence): `docs/damping_test_plan.md`, before Stage 2 of the v11 task.
- [ ] Pending requests from the simulation session (mass and hands; leg plant identification at operating gains): `docs/hardware_requests_from_sim.md`.

## Stage 5: feet touching down, hoist still taking weight
- [ ] Lower slowly, standing policy only. Watch for oscillation. Be ready to e-stop.

## Stage 6: free standing, then walking
- [ ] Standing, then small pushes by hand. Then walking at 0.1-0.3 m/s in an open area with a spotter. Selector transitions last.
- [ ] Only after all of the above: indoor use, doorways, planner integration.

Abort conditions at any stage: unexpected motion, oscillation, abnormal noise or heat, state messages stale, any doubt.
