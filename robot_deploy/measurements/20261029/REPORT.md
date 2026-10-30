# Hardware session report, 2026-10-29

One Unitree G1 unit (onboard computer PC2 = Jetson, JetPack 5.1.1 / L4T 35.3.1, Python 3.8.10). Written by the hardware session for the
simulation/training session. Companion files: `docs/sim_parameter_proposal.md` (what to change in simulation), `docs/open_issues.md`
(chronological log with every correction), `joints.csv`, `imu.json`, `timing.json`, raw data under this folder.

**How to read the claims.** "Measured" = directly observed in a recording. "Estimated" = derived with a stated method and assumption.
"Unknown" = not measured. `tau_est` is the torque estimate reported by the motors; it may be derived from motor current or commands, so every
torque or friction number is conditional on it tracking the real torque. One unit, one day, one operator.

## 1. What was run and what changed on the robot

**Read-only (nothing published):** version checks (CUDA/TensorRT/torch), CPU inference benchmark, `rt/lowstate` + `rt/lowcmd` listens (zero torque,
standing, moving around, walking with the RC), timing runs, full-rate recorder (`scripts/record_walk.*`).

**Published `rt/lowcmd` (each run approved by the user, robot hoisted with feet off the ground, e-stop in hand):**
| Runs | What |
|---|---|
| 2 damping tests (left wrist yaw, kd 2, default zero-torque state) | no visible effect (PC1 was still streaming zero commands) |
| 4 `hold` runs (joint 21), incl. hand pushes | commands accepted; held joints pushed back like springs |
| `step` joint 21 and 28 at 200 Hz; joint 28, 25, 4, 3 at 50 Hz | response timing, friction |
| `ramp` joint 4 (left ankle pitch), joint 10 (right ankle pitch) | stick-slip |
All with low gains (kp 20-40) and amplitudes of 0.03-0.08 rad; held joints at 0.5 x onboard nominal gains. Two runs ended on the drift abort (held joint moved 0.15 rad under the user's hand), as designed.

**Changes made to the robot (all on the user's request):**
- Wi-Fi: NetworkManager profile "Julio-Net" added on `wlan0`, autoconnect (`sudo nmcli ...`). Remove with `sudo nmcli connection delete Julio-Net`.
- `unitree-start-joystick.service` stopped and restarted several times (`sudo systemctl`); it is currently **stopped**.
- `/home/unitree/hhhh/iniciar_robo.sh` patched so a missing Bluetooth speaker / voice stack no longer aborts the start-up chain (backup: `iniciar_robo.sh.bak_20261029`).
- Files added on the robot: `/home/unitree/g1_recordings/` (recordings, joint-test logs, tool copies). Nothing under `g1_deploy/`.
- The robot's clock was ~11 h off and jumped to correct time when it got internet over Wi-Fi.
- No firmware, mode-switch API (`ReleaseMode`), or motor-mode changes from software. The user entered Unitree's debug mode on the RC.

## 2. Answers to the questions in `hardware_identification_prompt.md`

1. **Joints.** 27 of 29 motors are live. **Waist roll and pitch (13, 14) are absent/locked**: mode 0, temperature 0, voltage 0, q = dq = tau = 0 in every state (zero torque, standing, walking, pushes, debug mode). Hands/Inspire: not visible in `rt/lowstate`; unknown.
2. **Zero offsets and signs.** Signs consistent with our default pose on all joints; legs within ~0.06 rad of it in the onboard standing pose. This is **not** an encoder-offset test (the robot was in its own pose). Unknown.
3. **IMU (pelvis).** Gyro bias ~[-0.0004, 0.0043, 0.0010] rad/s, std ~[0.0010, 0.0022, 0.0018] at rest; quaternion order w,x,y,z (inferred); projected gravity z = -1 upright, same convention as the simulation. Only a pelvis IMU exists in the message.
4. **Timing.** `rt/lowstate` ~700-1000 Hz (zero torque/standing ~780-810; debug mode ~1040), gap p99 2-6 ms, max 32 ms with the heavy recorder running; `tick` has ~4% repeats. Round trip command -> joint response **at 50 Hz: 3-8 ms** (wrist, elbow, ankle, knee; 1 kHz timestamps). At 200 Hz publish the response lagged **96-141 ms** (probable backlog in PC1/DDS; the earlier claim that this was a fixed latency was WRONG and is withdrawn). Do not publish faster than needed.
5. **Limits.** True limits not read (SDK/URDF/firmware not consulted this session): unknown. Observed `tau_est` 99th percentile while walking: ankle pitch 37-39, knee 26-32, hip roll 40-43, hip pitch 22-24, waist yaw 9 N*m; extremes (incl. manual pushes) ankle pitch 63-67, knee 101, hip roll 76. Peak joint speeds: knee 16 rad/s, ankle roll 28 rad/s (single spikes).
6. **Gains the robot itself uses (onboard controller, FSM 801).** Legs: kp 100/100/100/150/60/60, kd 4/4/4/5/3/3 (constant while nominal); waist 200/6; shoulders+elbow 60/3; wrists 40/3. Arms/waist gains are scheduled (drop to 14-35 / 0.9-1.2 during the user's gesture combos); legs dropped to ~40% for 62 s in one interval (cause unknown). Command is a pure PD: velocity target 0, torque feed-forward 0. In its damping state PC1 sends kp 0, kd 5 (hips/knees/waist yaw/shoulders), 0.2 (ankles), 0.01 (wrists).
7. **Mass.** Not measured. One hint: the hanging shank + foot load the left knee with ~0.85-1.15 N*m at 0.26 rad (estimated), comparable against the simulation's masses.

## 3. Plant findings

- **Actuator ~ ideal PD.** `tau_est` = 0.92-0.97 x kp*(q_target - q) + kd*(0 - dq) with correlation 0.97-1.00 (onboard walking, 12 leg joints; hand-pushed held joints 15, 19, 22, 23: stiffness within 5-15% of commanded; ramp run: 0.92-0.94). No saturation seen below 2.9 N*m in our runs.
- **Command path at 50 Hz is fast** (3-8 ms from our message to first joint response on 4 joints). Commands from PC2 are accepted once PC1 has released control (debug mode); in the default zero-torque state they have no effect.
- **Static friction (unloaded, hanging, estimated from residual error x kp in the two directions):**
  | Joint | Breakaway friction | Gravity-like offset | Notes |
  |---|---|---|---|
  | wrist yaw L / R | 0.22 / 0.21 N*m | ~0.11 (opposite signs) | mirror-image residuals |
  | elbow R | 0.4-0.6 | 0.05-0.2 | |
  | knee L | 0.6-0.65 | 0.85-1.15 | hanging shank + foot |
  | ankle pitch R | ~1.0 (up 0.69, down 1.32) | ~0.3 | 17 slips over a +-0.08 rad ramp, steady |
  | ankle pitch L | first slips 0.6-1.1, then rising to **2.3** | - | joint covered only 22% of the ramp; extra resistance **unexplained** (right side clean) |
  Loaded values on the ground are unmeasured and probably higher. Hand check by the user: both ankles need "a little more push" when disabled.
- **Not identified:** reflected inertia (armature), backlash, effective damping (fits of kd are 10-30% low and noisy), true torque limits, thermal limits. The `sine` profile was never run.

## 4. Walking / push data

`walk/walk10`: 537 s, full rate, but only **~74 s of clean walking** (25 segments, longest 4 s, forward stick mean 0.27); the rest is standing, RC gesture combos (which scheduled the arm/waist gains) and manual pushes at unrecorded times. Too little for behaviour cloning. Likely push responses (idle sticks): ~494 s (tilt 13.7 deg, gyro 2.0 rad/s), ~499 s, ~452 s, ~260 s. Walking cycle ~0.9-1.6 Hz (low confidence). Tilt mean 2.8 deg, p99 9.1.

## 5. Compute platform (for the deployment side)

Torch 2.4.1 on the Jetson is CPU-only (`cuda.is_available()` False). A 84-256-128-128-15 MLP forward pass took 6.5 ms mean (p99.9 45 ms) with the joystick service running, 1.5 ms mean (p50 1.1, p99 6.9, max 48 ms) after stopping it: **not fit for a 50 Hz loop as is** (single runs). TensorRT 8.5.2 + CUDA 11.4 are installed; no pycuda/cuda-python/onnx. Background load seen: `start_joystick.py` ~73% of a core, camera nodes.

## 6. Corrections made during the session (so nothing stale is trusted)

1. "Damping from PC2 stops the joints": **not shown**; no effect in the default state. The emergency-stop skill's damping step is unproven; only its process-kill part is trustworthy.
2. "~120 ms actuation latency": **withdrawn**; it was a 200 Hz publish-rate effect. At 50 Hz it is 3-8 ms.
3. "Ankle friction 0.7-0.9 N*m": first withdrawn after the left ramp, then partly restored using the clean right ankle (~1.0 N*m); the left ankle remains anomalous.
4. "Peak torques are walking loads": **no**, they include the user's pushes; use the 99th percentiles.
5. Hold tests could not show command acceptance by themselves (zero commanded torque); acceptance was established from the motor mode change and the hand-pushed held joints.

## 7. Recommendations (details in `docs/sim_parameter_proposal.md`)

Remove waist roll/pitch from the policy; decide on matching Unitree's leg gains (ankle kp 60, kd 4/4/4/5/3/3); keep the 0-20 ms actuation delay (50 Hz control); add a small static friction term (~0.2 wrist ... ~1 N*m ankle pitch) and test sensitivity; check the ankle torque limit (50 N*m vs observed peaks incl. pushes); treat sim observation noise as a robustness margin (rest noise is 1-2 orders smaller).

## 8. Open items

Left-ankle extra resistance (inspect for contact/strap/mechanics); loaded behaviour on the ground (Stage 5 with the hoist taking weight, needs the controller); hip/waist-yaw/other joint types; a `sine` identification; a longer, steadier walk (5+ min, no gesture combos) and push times; joint torque/speed limits from SDK/URDF/Unitree; mass; re-enable `unitree-start-joystick.service` when no longer needed (currently stopped).

## 9. Files

`joints.csv` (per-joint table incl. measured friction/latency where tested), `imu.json`, `timing.json`; `jointtests/jt_*` (test logs, `meta.json`); `walk/*` (full-rate recordings: `.bin` + `meta.json` with the numpy dtype); step A / standing / motion / damping CSVs; tools in `scripts/` (`record_walk.*`, `jointtest.*`, `robot_estop.sh`, `estop_damping.py`); procedure in `docs/hoisted_joint_tests.md`. Raw `.bin` and `*_raw.*` files are git-ignored; copies of the recorder runs also live on the robot in `/home/unitree/g1_recordings/`.
