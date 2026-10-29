#!/usr/bin/env bash
# Sourced by the other scripts. Loads ONLY the ROBOT_* entries from the project .env
# (never the lab credentials) and prepares the ssh options.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$_ROOT/.env" ] || { echo "missing $_ROOT/.env" >&2; return 1 2>/dev/null || exit 1; }
set -a; source <(grep '^ROBOT_' "$_ROOT/.env"); set +a
: "${ROBOT_HOST:?ROBOT_HOST not set in .env}" "${ROBOT_USER:?ROBOT_USER not set in .env}" "${ROBOT_PASSWORD:?ROBOT_PASSWORD not set in .env}"
export SSHPASS="$ROBOT_PASSWORD"   # sshpass -e reads it, so the password never appears in a command line
ROBOT_DEPLOY_DIR="${ROBOT_DEPLOY_DIR:-/home/$ROBOT_USER/g1_deploy}"
ROBOT_SSH_OPTS=(-o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2
                -o StrictHostKeyChecking=accept-new -o PubkeyAuthentication=no
                -o PreferredAuthentications=password,keyboard-interactive)
robot_ssh() { sshpass -e ssh "${ROBOT_SSH_OPTS[@]}" "$ROBOT_USER@$ROBOT_HOST" "$@"; }
