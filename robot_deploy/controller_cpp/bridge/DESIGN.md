# g1_bridge: a small C++ process that owns all DDS traffic; the controller logic (Python today) talks to it through shared memory

Why (measured 2026-11-01): Python's DDS receive costs ~85% of a core and its publish call stalled up to 32 ms; the C++ publish path is 0.2 ms with no outliers. A dead or stalled Python controller currently means limp joints.
The bridge keeps everything risky about DDS in ~600 lines of C++ and adds a second, independent safety layer. The controller language stays free (Python now, anything later): it only reads and writes one small shared struct.

## Roles
| | bridge (C++) | controller (Python) |
|---|---|---|
| DDS | subscribes `rt/lowstate` (verifies the checksum), listens to `rt/lowcmd`, publishes `rt/lowcmd` | none |
| Logic | none: no observation, no policy | observation, policy, phases, its own safety checks (unchanged) |
| Safety | firewall on every command, state checks, dead-man's switch, stop sequence | trips and stop sequence as before |

## Shared memory: `/dev/shm/g1_bridge` (one file, one writer per block)
Header (32 bytes) then three seqlock blocks: STATE (bridge writes), CMD (controller writes), STATUS (bridge writes). Each block = `seq` (odd while being written) + `chk_xor` + `chk_sum` (XOR and wrapping sum of the payload's 64-bit words) + payload.
A reader accepts a copy only if `seq` was even and unchanged and both checksums match, so a torn read cannot be mistaken for data (the checksum also guards against memory-ordering surprises in Python).
Layout is fixed by `shm_protocol.h` (static_asserts) and verified against the Python mirror by `bridge --print-layout` in the tests. Time base for every timestamp: CLOCK_MONOTONIC seconds (C++ `steady_clock`, Python `time.perf_counter()`/`time.monotonic()`).
- STATE payload: arrival time of the newest `rt/lowstate`, message counters (received, bad checksum, foreign `rt/lowcmd`), tick, mode_machine, q, dq, tau_est, temperature, motor mode (29 each), gyro, quaternion.
- CMD payload: `counter` (must advance on every new command), `t_client`, `flags` (ENABLE, STOP = "run the bridge stop sequence", END = "clean end, publish nothing more"), target, kp, kd (29 each).
- STATUS payload: state (IDLE, ACTIVE, STOPPING, STOPPED, REFUSED), reason text, counters (published, commands seen, rejected, dead-man events), write-time statistics, heartbeat.

## Bridge state machine (50 Hz, absolute schedule)
- **IDLE**: waits for a fresh command with ENABLE. Accepts it only if the bridge has been up >= 1.5 s, the state is fresh and finite, **no other program has published on `rt/lowcmd`**, the robot is still (|dq| < 0.5 rad/s), no motor is above 60 C, and the first command passes the firewall *and* starts where the robot is (|target - q| <= 0.10 rad). Otherwise REFUSED (final).
- **ACTIVE**: each tick: state checks (stale > 50 ms, NaN, tilt > 25 deg, speed > 5 rad/s, torque caps, temperature > 70 C) -> STOPPING. Dead-man: no new command counter for `--dead-ms` (default 60 ms = 3 ticks) -> STOPPING. A fresh command is firewalled and published; a stale one (1-2 ticks) repeats the last good command. Three firewall rejections in a row -> STOPPING.
- **Firewall** (a gross-error catch, not a policy limiter): all values finite; 0 <= kp <= contract kp, 0 <= kd <= contract kd; waist roll/pitch (13, 14) kp = kd = 0; |target - measured q| <= 0.35 rad; per-tick target step <= 0.30 rad.
- **STOPPING** (also on SIGINT/SIGTERM/SIGHUP, CMD STOP): kp fades linearly to 0 over 0.5 s from the last command's stiffness, the target follows the measured pose, kd = max(last kd, contract kd), for a further 3 s; then STOPPED. Same sequence as `run.py` and `stoptest_probe`.
- **STOPPED / REFUSED**: publishes nothing (the motors fall back to mode 0, limp). **One run per bridge process**: it exits after `--linger-s`; a new run needs a new process and a new approval.
- **Shadow mode** (no `--publish`): the same machine runs but no publisher is created: the controller can do full dry runs through the bridge, and every firewall decision and stop is visible in STATUS.

## Publish timing: fixed tick (default) or `--on-arrival`
Fixed tick: the bridge runs its own 50 Hz clock; a command waits in shared memory until the next tick, i.e. a constant delay set by the phase between the two loops (0-20 ms, measured 5.3 ms on one hoisted run).
`--on-arrival` (ACTIVE only): the bridge wakes as soon as a new command counter appears (100 us polling) and publishes it at once (measured on the laptop stand-in: mean 0.16 ms), then re-phases its schedule to one period + 5 ms after that arrival. The 5 ms slack stops a controller that is a little late from getting a stale repeat first. A new command right after another new one (< 10 ms) waits for the schedule. When nothing arrives the schedule still fires: stale commands are repeated (1-2 ticks), the dead-man (default 60 ms) and every state check are unchanged. IDLE, STOPPING and the stop sequence stay on the fixed 50 Hz tick.

## Gating
`--publish --stage LABEL` plus `CONTROLLER_CONFIRM=yes-LABEL` in the bridge's environment (CLAUDE.md rule 2, per run). The controller keeps its own `--publish` gate: two independent keys. A second bridge cannot start while one holds the lock on the shared-memory file.

## Tests (laptop, no robot): `tests/test_bridge.py` against the stand-in SDK
layout equals the Python mirror; seqlock consistency under a 1 kHz writer; shadow mode sends nothing; publish gating refusals; normal run and clean END; client stop; **client killed (SIGKILL) -> dead-man stop within ~80 ms and the full fade/damp**; client stall; stale state; NaN / over-gain / huge step / bad start pose rejected; waist roll/pitch forced to zero; SIGTERM; a second enable after STOPPED refused; single-instance lock; mutation checks on the dead-man and the firewall.
