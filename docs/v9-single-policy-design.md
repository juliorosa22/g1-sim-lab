# Single-policy design: one policy that walks AND stands still  (now v11)

Status: DESIGN, not built. Written 2026-10-30. Interfaces with the real robot are the v8 ones.

**Numbering:** v9a/v9b (walking posture fix) and v10 (standing posture fix) are the runs in progress (see "Update 2026-10-30" below); this single policy becomes **v11**.

## Goal

A single policy (the v8 interface: 78 observations, 13 actions = 12 legs + waist yaw, arms observed, waist roll/pitch locked,
Unitree's real gains) that:
1. tracks velocity commands across the walking range,
2. stands without stepping at zero command (feet planted, centimetres of drift, arms free to move),
3. recovers from shoves by stepping only when needed and then settles again,
4. goes walk -> stop -> stand and stand -> walk cleanly when the command ramps, as the controller will send it.

It would replace the two-policy selector (v4 walks / v1 stands), which works in simulation but needs a handover.

## What the evidence says (so the design is not trial and error)

| Observation | Source |
|---|---|
| v1 stands planted (0 lifts/min, 1.3 cm drift) but is fragile: 38% fall in three shoves up to 0.7 m/s | `logs/eval/standing_v1_*` |
| v2-v4 step in place at zero command (~250 lifts/min, ~3 cm/s drift). v2 added shoves every 3-6 s up to 0.7 m/s (v1: every 8-12 s, 0.5) | evaluations, git diff 3028d1f..ff11ba2 |
| v5 collapsed: stand penalties at full strength made net reward per second negative while a fall cost only -4 (weights are x dt), so falling paid | training logs |
| v6 (warm start, ramped penalties, -40 per fall) was stable and stopped stepping but leaned ~15 deg, crept ~10 cm/s, and lost turning and sideways (risk aversion; no foot-slip penalty) | v6 evaluations |
| Unitree's recipe (the only one documented on a real G1): total reward clipped at >= 0, alive +0.15, no fall penalty, foot speed penalised while in contact, base-height term. It steps in place at zero command | `external_refs/unitree_rl_gym` |
| mjlab: clock-free; a tight whole-body pose reward when the command is below 0.05. playground: L1 pose pull at zero command and the stepping reward switched off when the robot is still | reference repos |
| Real robot: ideal-PD actuator, 3-8 ms command latency at 50 Hz, rest sensor noise 1-2 orders below the training noise | hardware report |
| v7 (v1's exact recipe on the real plant) steps in place (195 lifts/min, 56 cm drift): v1's planted standing is NOT explained by the recipe or by gentle shoves. Replica test (v1 recipe on v1's own plant, lab stack, seed 42) is running | `logs/eval/v78_stand_v7_*`, `logs/train_seed_v1r_s42.out` |
| Policies TRAINED on the real plant crouch: knee 0.97-1.17 rad (default 0.3), base height 0.69-0.73 m (others 0.79), knee torque 20 N*m (v4: 3-4). v1/v4 merely EVALUATED on the real gains keep 0.79 m. No earlier reward anchored the posture | `scripts/diagnose_posture.py` |
| v1's standing is also a knee-lock: knee -0.08 rad (the joint stop is -0.087) with 42 N*m commanded into it | same |
| v8 walks + v1 stands fails (63-73% survive the 78 s script): v1 cannot take over from v8's crouch | `logs/eval/v8v1_switch_*` |

Conclusions used below: (a) sampling balance is necessary but not sufficient, the reward must prefer stillness; (b) never let living become
net negative and never make a fall expensive enough to cause risk aversion; (c) a foot-slip term is what stops "standing" from turning into
shuffling; (d) shove strength and frequency must differ between standing and walking.

## 1. Command sampling (balanced scenarios)

A new command term draws a SCENARIO per env per interval, holds it, and ramps the command toward its target at the controller's
deceleration limits (1 m/s^2 linear, 2 rad/s^2 yaw), so training sees ramps like the deployed controller. Deadband 0.05 (smaller components are zero).

| Scenario | Share | Command | Dwell |
|---|---|---|---|
| STAND | 35% | all zero | 5-20 s, includes arm motion and shoves |
| CREEP | 10% | 0.05-0.2 m/s any direction, 0.05-0.3 rad/s | 3-8 s |
| WALK | 25% | forward -0.3..1.0, sideways +-0.5, yaw +-0.5 | 3-8 s |
| TURN | 15% | rotate in place and turn-while-walking, yaw +-1.0 | 3-8 s |
| STOP / START | 15% | walk command ramped to zero then held (stop), or zero ramped up (start) | 6-12 s |

Compare v1-v4: standing 10% for 3-10 s, every change an instant step, no ramps.

## 2. Disturbances per scenario
- STAND: gentle shoves 0.1-0.3 m/s every 8-15 s; a strong one (0.5-0.7) about every 30 s, so stepping is learned as recovery but planting stays the default.
- Other scenarios: 0.3-0.7 m/s every 4-8 s (v2-like).
- Arms: the v4/v8 `ArmPoseCommand`, plus corner poses (v5 piece, tested).

## 3. Rewards (one recipe, from the references)
- **Posture anchoring (new, from the crouch finding):** base height (Unitree, -10 x (z - 0.78)^2) and the mjlab whole-body pose reward
  (exp(-mean(err^2/std^2)), weight 1; std 0.05 at zero command, per-joint 0.1-0.35 while walking). Being tested as v9a/v9b/v10; keep what works.
- Total clipped at >= 0 per step (Unitree), alive +0.15, **no fall penalty** (Unitree, mjlab) or a small one (playground -100 x dt).
- Tracking exp kernels (sigma^2 0.25): linear 1.0, yaw 0.5 (Unitree). lin_vel_z -2, ang_vel_xy -0.05, orientation -1, base height -10 (target ~0.79),
  action rate -0.01, dof acc/vel as Unitree, dof_pos_limits -5, hip roll/yaw -1.
- Feet: contact speed (foot slip) -0.2 always; swing height -20 (0.08 m) while swinging; phase-contact +0.18 **only when moving or the base is moving**.
- Standing (active only at zero command), ramped in over ~150 iterations: L1 pull to the default pose (playground `stand_still`) and a narrow still
  bonus whose kernel starts wide; penalties on joint speed/foot lifts **exempt while recovering** (speed > 0.25 m/s, tilt > 9 deg, or 2 s after a shove).
- Gait clock frozen at zero command, restart at 0 on motion (v5 piece, tested).

## 4. Training plan
- Interface identical to v8, so v9a = warm start from v8, ~1500 iterations, 4096 envs; v9b = from scratch, 3000 iterations, 8192 envs. Run what the lab can hold.
- Needs v8 first for v9a (~3.4 h). v9b can start when v7 finishes (~1.5 h).

## 5. Acceptance gate (all in simulation, noise off AND on, arms moving)
- Standing: foot lifts <= 2/min, drift <= 3 cm / 17 s, tilt RMS < 3 deg, survive shoves 0.3/0.5/0.7 m/s >= 95%, calm again within 2 s >= 90%.
- Walking: tracking within 10% for 0.2-1.0 m/s forward, turn 0.5/1.0 rad/s, sideways 0.3 within 15%, no regression > 10% against v8.
- Stop/start: peak tilt < 6 deg through ramped stops; no falls in the 78 s switching script run as a single policy.
- Tools: `evaluate.py`, `evaluate_standing.py`, `evaluate_lowspeed.py`, `evaluate_switch.py` (single-policy mode to add).

## 6. To build
1. `ScenarioVelocityCommand` (scenario draw, ramping, deadband, exposes the scenario for the shove event and rewards).
2. Scenario-dependent shove event.
3. Reward terms: foot contact speed, swing height, base height, phase-contact mask; a reward manager that clips the sum at 0 (env subclass).
4. Unit checks like `tests/check_stand.py` for each piece, then a smoke test, then the lab runs.

## Open decisions
1. Gait clock: keep, frozen at zero command (tested), or remove it as mjlab does.
2. v9a (warm start) and v9b (scratch) both, or only one.
3. Fall penalty: none (Unitree/mjlab) or small (playground).
4. Standing share 35% and dwell 5-20 s are my guesses; the shove mix inside STAND is the most uncertain part.

## Update 2026-10-30: posture-fix runs (option A)
v9a = v8 + base height; v9b = v8 + base height + pose reward; v10 = v7 + both. Code: `g1_rl/v9_env_cfg.py`, `g1_rl/v10_env_cfg.py`,
`g1_rl/mdp/rewards.py: variable_posture`. Measured behaviour of the pose reward (`tests/check_posture_terms.py`): default pose 1.0; v8's
crouch scores 0.36 while walking and 0.00 while standing (the mean over 13 joints dilutes it). If the crouch survives in v9b, tighten the
walking tolerance or add a knee-specific term. Evaluate with `diagnose_posture.py`, `evaluate_standing.py`, `evaluate.py`, `evaluate_switch.py`.
