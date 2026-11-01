#!/usr/bin/env bash
# Copy the deployable parts of robot_deploy/ (configs, controller, policies) to the robot.
# DRY RUN by default: prints what would be sent. Add --apply to really copy.
# Never deletes anything on the robot. Does not start anything. A dry run makes no change on the robot.
#   robot_deploy/scripts/robot_sync.sh            # dry run
#   robot_deploy/scripts/robot_sync.sh --apply
source "$(dirname "${BASH_SOURCE[0]}")/robot_env.sh" || exit 1
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPLY=0; [ "${1:-}" = "--apply" ] && APPLY=1
PARTS=(configs controller policies controller_cpp)   # controller_cpp: sources only; built on the robot
if robot_ssh 'command -v rsync >/dev/null' 2>/dev/null && command -v rsync >/dev/null; then
  FLAGS=(-av --exclude '__pycache__' --exclude '*.pyc' --exclude 'README.md' --exclude 'build' --exclude 'stubs' --exclude 'tests')
  [ "$APPLY" = 1 ] || FLAGS+=(-n)
  [ "$APPLY" = 1 ] && robot_ssh "mkdir -p '$ROBOT_DEPLOY_DIR'"  # only when really copying: a dry run touches nothing on the robot
  for p in "${PARTS[@]}"; do
    [ -d "$SRC/$p" ] || continue
    echo "== $p -> $ROBOT_USER@$ROBOT_HOST:$ROBOT_DEPLOY_DIR/$p $([ "$APPLY" = 1 ] || echo '(dry run)')"
    rsync "${FLAGS[@]}" -e "sshpass -e ssh ${ROBOT_SSH_OPTS[*]}" "$SRC/$p/" "$ROBOT_USER@$ROBOT_HOST:$ROBOT_DEPLOY_DIR/$p/"
  done
else
  echo "rsync not available on both ends: falling back to tar over ssh"
  for p in "${PARTS[@]}"; do [ -d "$SRC/$p" ] && echo "== would send: $p" && (cd "$SRC" && find "$p" -type f ! -name README.md ! -name '*.pyc' ! -path '*/__pycache__/*' ! -path '*/build/*' ! -path '*/stubs/*' ! -path '*/tests/*'); done
  if [ "$APPLY" = 1 ]; then
    (cd "$SRC" && tar czf - --exclude='__pycache__' --exclude='*.pyc' --exclude=README.md --exclude=build --exclude=stubs --exclude=tests "${PARTS[@]}") | robot_ssh "mkdir -p '$ROBOT_DEPLOY_DIR' && tar xzf - -C '$ROBOT_DEPLOY_DIR'"
    echo "sent."
  else echo "(dry run: nothing sent; use --apply)"; fi
fi
