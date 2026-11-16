# Update from the simulation session for the hardware session (2026-11-16)

What changed since `sim_answers_to_hardware.md`. Nothing here was run on the robot; every number is from simulation (Isaac Lab, 128-1024 envs, observation noise on unless noted).

## 1. New in the repo (all committed on feature/g1-walking-policy)
| what | where |
|---|---|
| v12 (walking) exported: ONNX, numpy weights, test vectors | `policies/walk_v12_20261116.{onnx,npz,_testvec.npz}`, entry in `policies/manifest.json` |
| v12 contract file (same 78 -> 13 interface as v11; trained range vx -0.3..1.0) | `configs/g1_v12_walk.json` |
| v12 contract test data and result | `tests/data/contract_v12.npz`; `python3 tests/contract_test.py --data tests/data/contract_v12.npz --npz policies/walk_v12_20261116.npz` passes (obs 1.5e-6, policy 1.9e-6, targets 0, command 4.8e-7) |
| v11/v12 selector (forward route, enter 0.3, exit 0.2, debounce 0.2/0.3 s, 0.3 s cross-fade) | `controller/selector.py`, `tests/test_selector.py` (all pass) |
| command-grid stress test results | `logs/eval/grid/*.json` (laptop only, not committed) |
| plant-mismatch sweep results | `logs/eval/plant/*.json` (laptop only, not committed) |

The selector is not wired into `run.py`; v11 alone is still what `run.py` loads. Wiring it (and loading v12) is yours.

## 2. Command limits (from the grid test: 210 cells v11, 120 cells v12, 8 envs each)
Inside the trained range, with the deployed ramp, with noise: 0% falls for both. Recommended limits for the planner/controller:
| | vx | vy | wz |
|---|---|---|---|
| v11 | -0.3 .. +0.5 | +-0.25 | +-0.8 (1.0 later) |
| v12 | -0.2 .. +0.8 (1.0 later) | +-0.25 | +-0.5 |
- Keep the ramp (1 m/s^2, 2 rad/s^2). Without it (abrupt steps) v11 and v12 still hold inside the range, but a sudden sign reversal falls: v12 27% from vx 1.0, 76% from 1.2, never up to 0.6; v11 11% from 0.6.
  Rule: if the sign of vx flips with |vx| > 0.5 (v12) / 0.3 (v11), or wz flips with |wz| > 0.5, command zero first and wait for speed < 0.3 m/s.
- Do not combine |wz| above 1.0 with backward or sideways motion (v11 falls there).

## 3. Plant mismatch: what matters (fall % over the in-range command box)
| change | v11 | v12 |
|---|---|---|
| baseline | 0.6 | 0.3 |
| friction x0.4 / x0.2 of trained | 9 / 55 | 2 / 32 |
| leg kp x0.6 / x1.4 (kd unchanged) | 18.5 / 17.5 | 26 / 10 |
| leg kd x0.5 / x1.6 | 8.5 / 1.3 | 3 / 15 |
| kp and kd together x0.7 / x1.3 | 0.8 / 6 | 3.3 / 1.4 |
| mass x0.85 / x1.15 / x1.3 | 3 / 0.6 / 1.3 | 0.3 / 1.1 / 4.2 |
| torque limit x0.9 / x0.75 / x0.6 | 1.3 / 1 / 3.8 | 0.3 / 0.3 / 5.6 |
| Coulomb joint friction 1 / 2 / 4 N*m on the legs | 1 / 0.6 / 0.8 | 0 / 0.3 / 0.6 |
| extra delay +10 / +20 / +30 ms | 1.3 / 2.3 / 0.4 | 0 / 0 / 0 |
| COM shift +-4 cm | 0.8 / 1.7 | 0 / 0.3 |
Differences up to 3 points are noise (8 envs per cell). Trained: friction 0.3-1.2, torso +-5 kg, leg gains +-20% (kp and kd together), delay 0-20 ms.

## 4. What I need measured on the robot (priority order)
1. **Floor friction** where the first ground turn will happen (a pull test with a spring scale on a foot gives mu = pull force / weight on it). Below about 0.4 of the trained range both policies start to slip and fall. Use a high-grip surface (rubber mat, dry) for the first ground runs. Never test on a polished or wet floor.
2. **Effective leg stiffness**: the data so far (tau_est vs kp*err, slope 0.92-0.99) says kp is within a few percent of nominal, which is harmless; keep logging it.
3. Total mass with the hands fitted (sim model includes the Inspire hands).

## 5. v11 next hoisted turn: limits to use (from `sim_answers_to_hardware.md`, restated)
- Tracking limit: at least 0.8 rad for turning (sim p99 knee error 0.44-0.53 rad, ankle pitch 0.50-0.76 rad; the first step can reach 0.5 rad). `--track-err 0.8`, bridge `--max-err 0.8`.
- Joint speed limit: 10 rad/s (the foot set-down reached 5.2; sim ankle speed up to ~23 rad/s is a spike, not sustained).
- Torque caps must exceed the sim demand while turning: knee p99 57-70 N*m (max 111), ankle pitch up to 50 N*m; the stage3 caps (hips/knee 40, ankle 25) will trip. Choose new caps for a turn stage and say which.
- Yaw command: start at +-0.2 rad/s, then 0.3; a turn needs a hoist that lets the robot rotate.

## 6. v12 on the robot: not ready for ground walking
The wiring of v12 and the selector into `run.py` (what to change, guards, tests, order of runs) is `docs/task_runpy_selector_wiring.md`.
Gating steps I recommend, each needing the user's explicit yes at the time:
1. Wire the selector and the v12 network into `run.py` as a dry run (computes, sends nothing); check the contract test and the loop timing on the Jetson (0.3 ms per policy call measured for v11).
2. Hoisted, feet in the air, forward 0.2-0.3 m/s: joint speeds and torques will exceed the standing-era trips more than the turn did; relax the trips with numbers from a new demand table (ask me: `scripts/evaluate_joint_demands.py` has a walking mode to add).
3. A gantry or harness that carries weight for the first ground steps; first ground command 0.2 m/s, then 0.3.
Delay budget for v12 (full table `docs/sim_delay_sweep_v12.md`): no extra falls up to +20 ms of action delay or observation age, at most 0.5-1% up to +30 ms, 2-8% at +40 ms, about half fall at +60 ms; late ticks up to 5 (100 ms) at 1% per tick cost at most 1.6%. Do not exceed +30 ms extra in total; the pipeline measured on the robot (about 10-15 ms) is inside the trained range.
Not yet done in sim: a wider-randomization retrain (friction down to 0.15, independent kp/kd, link masses) and the "trip on the ground" question (Request 2).

## 7. Open from the earlier requests
- Request 2 (behaviour after a trip while standing on the ground: fade, reduced gains, longer fade; time to fall, base speed at contact, peak torque): still to do in simulation.
- Security: `docs/task_shm_hardening.md` (owner and mode check of `/dev/shm/g1_bridge`).
