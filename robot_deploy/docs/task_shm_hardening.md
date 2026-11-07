# Task for the hardware session: harden the shared-memory file of the bridge

From the simulation session, 2026-11-07. Source: automated security review after the push of 8201f34 (finding MEDIUM, "insecure shared-memory creation").
Nothing was changed by the simulation session; `controller_cpp/` and `controller/bridge_client.py` are yours.

## The problem
- `controller_cpp/bridge/g1_bridge.cpp:173` opens `/dev/shm/g1_bridge` with `open(path, O_RDWR | O_CREAT, 0600)`.
- `/dev/shm` is world-writable (sticky). If another user, or a stale file, already owns `/dev/shm/g1_bridge`, the bridge opens it without checking.
  The file carries the command block the bridge publishes to the motors, so whoever controls the file controls the commands (the firewall, dead-man timer and limits still apply).
- `controller/bridge_client.py:55` opens the same path with `open(path, "r+b")` and also checks nothing.

## Risk here
Low: one user (`unitree`) on the Jetson, robot LAN only, all published runs so far attended. Worth fixing before unattended or walking runs.

## Requested change
1. Bridge, after `open`: add `O_NOFOLLOW`; `fstat(fd)`; refuse to start (`finish(2)`, clear message) unless `st_uid == geteuid()` and `(st_mode & 077) == 0` (`fchmod(fd, 0600)` for a file the bridge created itself).
   Optionally keep the file in a private directory (`/run/user/<uid>`) and make `--shm` default point there.
2. `bridge_client.py`, after opening: `os.fstat`; refuse the file unless the owner is the current user (or the expected bridge user) and the mode is 0600.
3. Tests: stand-in SDK suite (`controller_cpp/tests/`) gets a case for a file owned by another user / mode 0666 / a symlink: the bridge must refuse each one. The existing sections must still pass.
4. Rebuild on the robot (clock skew: force the rebuild, check the binary string or size) and rerun a shadow run before any published run.

Record the outcome in `open_issues.md`.
