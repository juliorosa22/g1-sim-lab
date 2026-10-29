#!/usr/bin/env bash
# Read-only reachability and health check of the robot's onboard computer. Changes nothing.
source "$(dirname "${BASH_SOURCE[0]}")/robot_env.sh" || exit 1
echo "== target: $ROBOT_USER@$ROBOT_HOST"
echo "== local route to the robot (the interface name is what Unitree's deploy_real calls net_interface)"
ip route get "$ROBOT_HOST" 2>&1 | head -1
echo "== ping"
ping -c 3 -W 2 "$ROBOT_HOST" 2>&1 | tail -3 || { echo "NOT REACHABLE. Check the cable and that this machine has a 192.168.123.x/24 address."; exit 2; }
echo "== ssh"
robot_ssh 'echo "hostname : $(hostname)"; echo "user     : $(whoami)"; echo "kernel   : $(uname -srm)";
  echo "uptime   : $(uptime -p)"; echo "disk     : $(df -h ~ | tail -1 | awk "{print \$4\" free of \"\$2}")";
  echo "python3  : $(python3 --version 2>&1)"; echo "rsync    : $(command -v rsync || echo missing)";
  echo "interfaces:"; ip -brief addr 2>/dev/null | sed "s/^/  /";
  echo "sdk      : $(python3 -c "import unitree_sdk2py,sys;print(unitree_sdk2py.__file__)" 2>&1 | tail -1)";
  echo "torch    : $(python3 -c "import torch;print(torch.__version__)" 2>&1 | tail -1)";
  echo "deploy dir '"$ROBOT_DEPLOY_DIR"': $( [ -d '"$ROBOT_DEPLOY_DIR"' ] && ls '"$ROBOT_DEPLOY_DIR"' | tr "\n" " " || echo absent)"' \
  || { echo "ssh FAILED (exit $?). Reachable by ping but login failed: check ROBOT_USER / ROBOT_PASSWORD in .env."; exit 3; }
