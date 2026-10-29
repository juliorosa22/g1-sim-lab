---
name: unitree-robot-ssh
description: Connect to the Unitree G1's onboard computer over the Ethernet cable (ssh to 192.168.123.164) to check reachability, run read-only commands, inspect files, or sync the deploy folder. Use when the user asks to reach, check, ssh into, or copy files to the physical robot or "the hardware". Never moves the robot.
---

# Reaching the Unitree G1 over ssh

The robot's onboard computer is at `unitree@192.168.123.164`, reached over the Ethernet cable. The login
details are in the project `.env` (git-ignored): `ROBOT_HOST`, `ROBOT_USER`, `ROBOT_PASSWORD`,
`ROBOT_DEPLOY_DIR`. Use the helper scripts; they load only the `ROBOT_*` entries and pass the password to
`sshpass` through the environment, so it never shows up in a command line. Run them from the project root.

## Safety: this skill only connects

The target is a real humanoid robot. This skill is for **reaching and inspecting** it.

- Fine without asking: `robot_check.sh`, `robot_ssh.sh` with read-only commands (`ls`, `cat`, `ps`, `uptime`, `df`,
  `ip addr`, `python3 -c "import ..."`), and `robot_sync.sh` in its default **dry-run** mode.
- **Ask the user first, every time**, before: `robot_sync.sh --apply`, starting any process on the robot,
  anything that publishes motor commands (`rt/lowcmd`, Unitree's `deploy_real.py`, our controller), changing robot
  modes or system settings, `sudo`, reboot or shutdown. One approval covers one specific action only.
- Never write the password anywhere except `.env`.

## Commands

```bash
robot_deploy/scripts/robot_check.sh            # read-only health check: route, ping, ssh, versions, deploy dir
robot_deploy/scripts/robot_ssh.sh 'COMMAND'    # one command; no argument opens an interactive shell
robot_deploy/scripts/robot_sync.sh             # DRY RUN of the copy of configs/, controller/, policies/
robot_deploy/scripts/robot_sync.sh --apply     # actually copy (ask first); never deletes on the robot
```

## Reading the results

- `ip route get` shows the laptop's interface to the robot (currently `eno2`, 192.168.123.95); Unitree's
  `deploy_real.py` calls this the `net_interface`.
- Ping fails: check the cable and that the laptop's wired interface has a `192.168.123.x/24` address
  (a static address in that subnet is needed; the robot does not hand one out).
- Ping works but ssh fails: wrong `ROBOT_USER`/`ROBOT_PASSWORD` in `.env`, or ssh is not running on the robot.
- The first connection adds the robot to `known_hosts` (`StrictHostKeyChecking=accept-new`); if it later reports a
  changed host key, stop and tell the user rather than overriding it.

## Facts about the robot (checked 2026-10-29)

Hostname `ubuntu`, Linux 5.10.104-tegra aarch64 (Jetson), Python 3.8.10, PyTorch 2.4.1, `unitree_sdk2py` at
`/home/unitree/unitree_sdk2_python`, rsync available, 1.8 TB free. Python code for the robot must be 3.8-compatible.

## More context

Everything deployed to the robot, and the rules for a dedicated hardware session, live in `robot_deploy/`:
start with `robot_deploy/CLAUDE.md`, then `docs/open_issues.md` and `docs/bringup_checklist.md`.
