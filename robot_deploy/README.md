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
| `configs/g1_walk_stand.yaml` | Deployment contract: joint order and motor indices, gains, action scales, observation layout, policies, selector, limits |
| `controller/` | The code that runs on the robot (to be written) |
| `policies/` | Exported policies and their manifest (to be exported) |
| `scripts/` | Connection and sync helpers |
| `docs/bringup_checklist.md` | Staged bring-up with safety gates |
| `docs/open_issues.md` | Known mismatches and unvalidated assumptions |

## Status

Connection verified (2026-10-29). The controller and policy exports do not exist yet, and no policy
trained in this project has run on the robot. See `docs/open_issues.md` before any run.
