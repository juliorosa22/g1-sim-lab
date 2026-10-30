# robot_deploy: everything that is deployed to the Unitree G1

This folder is the **deploy surface** of the project. It holds what runs on (or is copied to) the
robot, plus the scripts and notes to get it there. Training, evaluation and simulation live in the
parent repo (`g1-sim-lab`); this folder consumes their outputs. A session started here should only
need this file and what it points to.

## Read this first: safety rules (they apply to you, the agent)

The target is a **real humanoid robot** (roughly 35 kg class) with torque-capable joints. Software mistakes here can hurt
people and break hardware.

1. **Read-only by default.** Connecting (`scripts/robot_check.sh`, `robot_ssh.sh 'read-only command'`),
   listing files, reading logs and *dry-run* syncing are fine without asking.
2. **Never send motor commands, start a controller, or run anything that publishes `rt/lowcmd`
   without the user explicitly confirming, in this session, for that specific run.** This includes
   Unitree's `deploy_real.py`, our controller, and any "quick test" script. Approval for one run
   is not approval for the next.
3. **Never** change robot modes, firmware, network settings or system services; never `sudo`, reboot
   or shut down the robot's computer unless the user asks for exactly that.
4. First time anything new runs: **dry-run mode** (compute actions, log them, send nothing), then a
   hoisted robot, then ground. See `docs/bringup_checklist.md`.
5. Don't copy secrets into this folder. Credentials live only in the project's `.env`
   (git-ignored). `scripts/robot_env.sh` loads just the `ROBOT_*` entries.
6. If a command's effect on the robot is unclear, ask first.
7. **Before ANY policy test, every session: stop `unitree-start-joystick.service`** (a custom service, always running after
   boot; the RC's START button launches `/home/unitree/hhhh/iniciar_robo.sh`, which starts `assistente.py` and
   `movimentos_militares.py`). Check with `systemctl is-active unitree-start-joystick.service`. Stopping it needs `sudo`, so
   do it only with the user's explicit ok that session (or let them run it via `!`). It comes back after every reboot
   (`Restart=always`), so re-check each time. Our tests use a separate USB joystick, not the RC.

## The hardware and the connection

- Onboard computer ("PC2"): `unitree@192.168.123.164` over the Ethernet cable, ssh password in `.env`
  (`ROBOT_HOST`, `ROBOT_USER`, `ROBOT_PASSWORD`, `ROBOT_DEPLOY_DIR`). The laptop's wired interface
  needs a `192.168.123.x/24` address; on this laptop it is `eno2` (192.168.123.95).
- Checked on 2026-10-29: hostname `ubuntu`, Linux 5.10.104-tegra **aarch64** (Jetson), **Python 3.8.10**,
  PyTorch 2.4.1, `unitree_sdk2py` at `/home/unitree/unitree_sdk2_python`, rsync present, 1.8 TB free.
  **Write Python 3.8-compatible code** (no `X | Y` runtime type unions, no `match`, no PEP 604/585 at runtime).
- Deploy target on the robot: `ROBOT_DEPLOY_DIR` (`/home/unitree/g1_deploy`), created by the first sync.
- Use the project skill `.claude/skills/unitree-robot-ssh` (scripts below) rather than hand-writing ssh.

## Layout

```
robot_deploy/
  CLAUDE.md                 this file (context for a dedicated session)
  README.md                 human overview and quick start
  configs/
    g1_walk_stand.yaml      THE deployment contract: joint order and motor indices, gains, defaults, action
                            scales, observation layout and scales, both policies, selector, command limits
  controller/               code that runs ON the robot (not written yet; see controller/README.md)
  policies/                 exported policies + manifest (see policies/README.md)
  scripts/                  robot_check.sh, robot_ssh.sh, robot_sync.sh (+ robot_env.sh, sourced by them)
  docs/
    bringup_checklist.md    staged bring-up with the safety gates
    open_issues.md          known mismatches and unvalidated assumptions: READ BEFORE ANY HARDWARE RUN
```

Only `configs/`, `controller/` and `policies/` are synced to the robot (`robot_sync.sh`); `README.md`
files, `docs/` and `scripts/` stay on the laptop.

## What is being deployed

Two policies behind a selector (the "switching controller", prototyped in the parent repo's
`scripts/evaluate_switch.py`):

| | walking policy ("v4") | standing policy ("v1") |
|---|---|---|
| obs to actions | 84 to 15 (legs + waist) | 47 to 12 (legs) |
| used when | `norm(vx, vy) + abs(wz)` above 0.08 | below 0.04 (hysteresis) |
| behaviour (simulation) | tracks commands; steps in place at zero command | stands without stepping; falls under hard shoves |

Both are plain MLPs (ELU, no input normalisation) running at **50 Hz** on a 0.8 s gait clock. Full detail
in `configs/g1_walk_stand.yaml`. **No g1_rl policy has run on hardware yet.** Unitree's own pretrained
policy is the only one with documented real-G1 deployment (`external_refs/unitree_rl_gym/deploy/deploy_real`).

## Where things come from (parent repo, `../`)

- Policy training / checkpoints: `g1_rl/`, `logs/rsl_rl/`; lab machine copies via the lab ssh in `.env`.
  Lab checkpoints are RSL-RL 5.x; the laptop reads 3.x: convert with `scripts/convert_checkpoint.py`.
- Evaluation: `scripts/evaluate*.py`, results in `logs/eval/`.
- The simulation-side definition of observations/rewards/gains: `g1_rl/walk_env_cfg.py`,
  `g1_rl/body_env_cfg.py`, `g1_rl/robot_cfg.py`.
- Reference deployment code (Unitree): `../external_refs/unitree_rl_gym/deploy/deploy_real/` and its README
  (zero-torque, then debug mode via L2+R2, then `start`, then `A`).
- Design notes: `../docs/architecture.md`, `../docs/walking-policy-training-plan.md`.

## Conventions

- `configs/g1_walk_stand.yaml` is the single source of truth for numbers. The controller reads it; don't
  hard-code gains, scales or joint order elsewhere. If you change a value, say why in `docs/open_issues.md`.
- Scripts are idempotent and non-destructive: dry-run first, never `--delete`, never start the robot.
- Any simulation/hardware difference you discover goes in `docs/open_issues.md` the same day.
- Commit this folder with the rest of the repo. Large binaries: list them in `policies/README.md`.
