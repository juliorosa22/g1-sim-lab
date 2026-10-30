#!/usr/bin/env bash
# EMERGENCY software stop for the G1's joint commands. NOT a substitute for the remote's e-stop / the hoist.
#   robot_deploy/scripts/robot_estop.sh             kill OUR processes on the robot, then publish damping
#   robot_deploy/scripts/robot_estop.sh --dry-run   list what would be killed, build the command, publish NOTHING
#   robot_deploy/scripts/robot_estop.sh --hold 20   damping duration in seconds (default 10)
# Invoking this is the explicit go for the publish. It never calls ReleaseMode or touches robot modes.
source "$(dirname "${BASH_SOURCE[0]}")/robot_env.sh" || exit 1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_SSH_OPTS=(-o ConnectTimeout=3 -o ServerAliveInterval=2 -o ServerAliveCountMax=2
                -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no
                -o PreferredAuthentications=password,keyboard-interactive)

DRY=0; for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; done

# Only OUR processes: anything running from the deploy dir, or Unitree's deploy_real.py. The [x] bracket keeps
# the pattern from matching the shell that runs it. Unitree's own services never match.
PAT='([/ ][g]1_deploy/|[d]eploy_real\.py|[j]ointtest\.py)'

if [ "$DRY" = 1 ]; then
  echo "[dry-run] would kill:"; robot_ssh "pgrep -af '$PAT' || echo '  (nothing matches)'"
else
  echo "[estop] killing our controller processes"
  robot_ssh "pkill -9 -f '$PAT'; sleep 0.2; pgrep -af '$PAT' && echo 'STILL RUNNING' || echo 'none left'"
fi

echo "[estop] damping command"
robot_ssh 'python3 - '"$*" < "$HERE/estop_damping.py"
rc=$?
[ $rc -ne 0 ] && echo "[estop] FAILED (rc=$rc): USE THE REMOTE E-STOP / SUPPORT THE ROBOT" >&2
exit $rc
