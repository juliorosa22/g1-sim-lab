# Task for the hardware session: wire v12 and the v11/v12 selector into `controller/run.py`

From the simulation session, 2026-11-16. Code only, laptop first: nothing here is run on the robot until the user says yes for that run (`CLAUDE.md` rules 2, 3, 7). `run.py` is yours; I did not touch it.
Ready for you: `policies/walk_v12_20261116.{npz,onnx}`, `configs/g1_v12_walk.json` (same 78 -> 13 interface, observation scales, joint order and gains as v11; only `name`, the notes and the command ranges differ),
`tests/data/contract_v12.npz` (contract test passes), `controller/selector.py` + `tests/test_selector.py` (17 checks). Background and numbers: `docs/sim_update_for_hardware_20261116.md`, `docs/sim_delay_sweep_v12.md`.

## What run.py does today (the points that change)
- One policy: `--config` default `g1_v11_stand_turn.json`, `--npz` default `stand_turn_v11_*.npz`; `policy = NumpyPolicy(...)` (line ~381); `policy.warmup()`.
- The request to the planner side is yaw only: `cmd = self.shaper.step([0.0, 0.0, self.request_wz(...)])` with `MAX_WZ = 0.5` and `--wz-script` / `--allow-turn`.
- Every tick: `obs = ob.build(gyro, grav, cmd, q27, dq27, self.last_action, t - t_pol0)`, `action = self.policy(obs)`, `self.last_action = action`, `target[act_motors] += alpha * act_scale * action` (alpha: BLEND ramps 0 -> `--alpha-max`, RUN holds it).
- Phases RAMP, HOLD, BLEND, RUN, STOP, DONE; the policy runs in every phase so `last_action` is always the previous raw output; the gait clock starts at BLEND.
- Limits: `Limits(tau_cap, tilt_deg)`, `--track-err`, `--speed-limit`, the bridge's `--max-err`, `--max-speed`, the per-tick step limit, `--stop-fade-s`.

## What to add
1. **Two policies, one observation builder.** Load v11 and v12 (`--stand-npz`, `--walk-npz`, defaulting to the newest `stand_turn_v11_*.npz` and `walk_v12_*.npz`). Both read the SAME 78-value observation from the same `ObservationBuilder`
   (the two contract files differ only in name and ranges: check `obs_layout`, `obs_scales`, `joints`, defaults and gains are identical, and fail at start if they are not). Keep `--config` for v11; add `--walk-config`.
2. **Selector in the loop.** `sel = PolicySelector(dt=0.02)` from `controller/selector.py`; each tick `mode, w = sel.update(cmd)` with `cmd` = the SHAPED command the policy sees (not the request). `need_stand, need_walk = sel.needs()`; compute only the nets needed (both during the 0.3 s blend);
   `action = sel.mix(a_stand, a_walk)`. **`self.last_action` must be the blended action** (that is what the simulation feeds back; the selector docstring says so). The gait clock stays one shared clock (it always runs in training).
   The selector acts on the shaped forward command only: sideways, backward, yaw-only and creeping up to 0.3 m/s stay with v11. Reset the selector with the shaper at the start of a run. Log `mode`, `w` and the switch times in the CSV.
3. **A translation request, with guards.** Today only yaw can be requested. Add `--cmd-script "vx:vy:wz:seconds,..."` (counted from the start of RUN; the old `--wz-script` can stay and map to it), with a `CommandGuard` in front of the shaper:
   - clip to the recommended limits: v11 vx -0.3..0.5, |vy| 0.25, |wz| 0.8; v12 vx -0.2..0.8, |vy| 0.25, |wz| 0.5 (first runs: vx up to 0.3, then 0.5; |wz| up to 0.5 as now); the effective limit is the one of the policy the selector will use (v12 only for forward commands);
   - no sign reversal while moving: if the sign of vx flips with |vx_shaped| > 0.5 (v12) or 0.3 (v11), or wz flips with |wz_shaped| > 0.5, insert a zero request until the shaped speed is below 0.3 m/s / 0.3 rad/s, then continue (sim: reversing from vx 1.0 falls 27%, from 1.2 76%, never from 0.6);
   - never |wz| > 1.0 together with backward or sideways motion;
   - the existing deadband (0.05) and ramp (1 m/s^2, 2 rad/s^2) stay in `CommandShaper`; the guard must not bypass them. A unit test that fails if the shaped command ever changes faster than the ramp or exceeds a limit.
   The nav session's `CommandGuard` (`g1-sim-lab/nav/`) may be reusable; read it first but do not depend on code that is not committed.
4. **Ending a run while walking.** The STOP phase (kp fade, then kd-only damping) is the emergency path, right for a trip, wrong for a normal end at speed. For `finished`: first request zero, wait until the shaped command is 0, the selector is back in stand and the measured base speed (IMU-integrated or the state's estimate) is low (v12's stop test settles in about 1.5-2 s from 0.5 m/s; allow 4 s), then the existing stop sequence. Trips keep the immediate fade.
5. **Limits for walking.** The standing-era limits will trip, as they did for the turn: tracking 0.30 rad (v11 turn needed up to 0.5 rad on the first step; use `--track-err 0.8`), joint speed 5 rad/s (`--speed-limit 10`), torque caps (stage3 hips/knee 40, ankle 25 are below the turn demand, knee 57-70 N*m typical, max 111), tilt 25 deg (v12 under stress in sim: tilt max about 5-13 deg).
   I have not produced a walking demand table; ask the simulation session for `evaluate_joint_demands` in walking mode before choosing caps (it is a small addition on my side, same method as the turn table), and say which command speeds to cover.
6. **The handover as a measured event.** Each switch: peak tilt and peak speed in the next 1.5 s, written to the log (sim: handover peak tilt is small with the forward route; a lurch on the robot would be new information). Abort criteria for a first run: tilt above 10 deg during a handover.

## Tests to add (laptop, no robot)
- `tests/contract_test.py --data tests/data/contract_v12.npz --npz policies/walk_v12_*.npz` stays green (already passes).
- `tests/test_runner.py`: with the fake clock and a toy plant, a script 0 -> 0.6 m/s -> 0 gives exactly 2 selector switches, `last_action` equal to the mixed action, the blend lasts 0.3 s, v12 is not computed outside the blend and walk mode; the guard cases above (clip, reversal inserts a stop, wz with backward motion refused); a normal end waits for stand and low speed before the stop sequence; a trip during walking goes straight to STOP.
- Mutation checks like the existing ones: dropping the guard, feeding the RAW instead of the blended action to `last_action`, switching on the request instead of the shaped command, each must be caught by a test.

## Order of runs (each needs the user's explicit yes at that moment)
1. Dry run on the robot (`--source dds`, nothing sent): check timing with two nets (v11 0.3 ms on the Jetson, v12 similar; cycle p99 must stay below 5 ms) and that a scripted request switches modes in the log.
2. Hoisted, feet in the air, `--cmd-script` forward 0.2 -> 0.3 m/s for a few seconds; relaxed limits as above; e-stop in hand; `unitree-start-joystick.service` stopped.
3. Ground steps only with a gantry or harness carrying the weight, first 0.2 m/s then 0.3, on a high-grip floor whose friction has been measured (the plant sweep: friction below about 0.4 of the trained range makes both policies fall).
Not before: the console test (`docs/task_hardware_console_test.md`) is useful for the take-over path but is not a prerequisite for the dry run.

## Report back in `docs/open_issues.md` (same day)
Cycle times with two nets; every switch (time, shaped vx, peak tilt, peak speed); tracking error, joint speeds and torques while walking; anything the guard clipped or delayed; the limits you ended up using.
