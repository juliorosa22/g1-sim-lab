# robot_deploy

Everything deployed to the Unitree G1 and the tools to get it there. For a session working on the
hardware, start with [`CLAUDE.md`](CLAUDE.md) (context and safety rules).

## Quick start

```bash
cd g1-sim-lab
robot_deploy/scripts/robot_check.sh          # read-only: ping + ssh + versions (changes nothing)
robot_deploy/scripts/robot_ssh.sh 'uptime'   # run one command, or open a shell with no argument
robot_deploy/scripts/robot_sync.sh           # DRY RUN: what would be copied to the robot
robot_deploy/scripts/robot_sync.sh --apply   # copy configs/, controller/, policies/ to ~/g1_deploy
```

Credentials come from the project's `.env` (`ROBOT_HOST`, `ROBOT_USER`, `ROBOT_PASSWORD`,
`ROBOT_DEPLOY_DIR`); `.env` is git-ignored. The wired interface of the laptop needs a
`192.168.123.x/24` address (here `eno2`, 192.168.123.95).

## Contents

| Path | What |
|---|---|
| `configs/g1_v11_stand_turn.json`, `configs/g1_v12_walk.json` | Deployment contracts of the STANDARD policies v11 and v12 (`g1_walk_stand.yaml` is the legacy v4/v1 contract) |
| `controller/` | The code that runs on the robot (v11 runner `run.py`, observation, policy, commands, selector; v12 wiring pending) |
| `policies/` | Exported policies and their manifest (v11 and v12 are the standard ones) |
| `gui/` | Hardware console: camera, mode observer, guarded mode switching (mock-tested only) |
| `scripts/` | Connection and sync helpers |
| `docs/bringup_checklist.md` | Staged bring-up with safety gates |
| `docs/open_issues.md` | Known mismatches and unvalidated assumptions |

## Status

**Standard policies for the real robot: v11 (stand, turn in place, creep) and v12 (forward walking), switched by `controller/selector.py`** (see `CLAUDE.md`, "What is being deployed").
v11 has been through staged bring-up up to 30 s of free standing (2026-11-01); its turn in the air has not completed yet. v12 is exported and passes the contract test but has not run on the robot.
See `docs/open_issues.md` before any run.
