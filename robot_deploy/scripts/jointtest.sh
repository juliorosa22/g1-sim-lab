#!/usr/bin/env bash
# Hoisted single-joint test wrapper. READ docs/hoisted_joint_tests.md first.
#   jointtest.sh check                       read-only preconditions on the robot (publishes nothing)
#   jointtest.sh dry JOINT PROFILE [opts]    preflight + print the plan (publishes nothing)
#   jointtest.sh run JOINT PROFILE [opts]    *** PUBLISHES rt/lowcmd ***  needs JOINTTEST_CONFIRM=yes-JOINT-PROFILE in the environment
#   jointtest.sh abort                       graceful abort of the running test (ramps to damping, then stops publishing)
#   jointtest.sh pull                        rsync the latest test run to measurements/<date>/jointtests/
# PROFILE: hold | step | sine | ramp.  opts are passed to jointtest.py (--kp --kd --amp --tau-cap --max-exc --vel-cap --hold-scale).
# Wi-Fi: ROBOT_HOST_OVERRIDE=10.173.210.67 robot_deploy/scripts/jointtest.sh ...
# The environment variable is not a safety system: it only forces a deliberate step. The yes must come from the user, in chat, for THIS run.
source "$(dirname "${BASH_SOURCE[0]}")/robot_env.sh" || exit 1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE=/home/$ROBOT_USER/g1_recordings/jointtests
cmd="${1:-}"; joint="${2:-}"; prof="${3:-}"; shift 3 2>/dev/null || shift $# 2>/dev/null

case "$cmd" in
  check) robot_ssh 'python3 - --check' < "$HERE/jointtest.py";;
  dry)   [ -n "$joint" ] && [ -n "$prof" ] || { sed -n 2,9p "$0"; exit 1; }
         robot_ssh "python3 - --joint $joint --profile $prof --dry-run $*" < "$HERE/jointtest.py";;
  run)   [ -n "$joint" ] && [ -n "$prof" ] || { sed -n 2,9p "$0"; exit 1; }
         if [ "${JOINTTEST_CONFIRM:-}" != "yes-$joint-$prof" ]; then
           echo "REFUSED: set JOINTTEST_CONFIRM=yes-$joint-$prof, only after the user said yes to this exact run" >&2; exit 1; fi
         name="jt_${joint}_${prof}_$(date +%H%M%S)"; dir="$BASE/$name"
         robot_ssh "mkdir -p $dir && echo $name > $BASE/current" && robot_ssh "cat > $dir/jointtest.py" < "$HERE/jointtest.py" || exit 1
         robot_ssh "cd $dir && setsid nohup python3 jointtest.py --joint $joint --profile $prof --run --outdir $dir $* > $dir/out.txt 2>&1 < /dev/null &"
         echo "started $name; waiting for it to finish (abort with: jointtest.sh abort)"
         for i in $(seq 1 60); do sleep 1; robot_ssh "pgrep -f '[j]ointtest.py' >/dev/null" || break; done
         robot_ssh "cat $dir/out.txt";;
  abort) robot_ssh "n=\$(cat $BASE/current); touch $BASE/\$n/STOP; echo STOP requested for \$n";;
  pull)  n="$(robot_ssh "cat $BASE/current" | tr -d '\r')"; dest="$HERE/../measurements/$(date +%Y%m%d)/jointtests/$n"; mkdir -p "$dest"
         sshpass -e rsync -av -e "ssh ${ROBOT_SSH_OPTS[*]}" "$ROBOT_USER@$ROBOT_HOST:$BASE/$n/" "$dest/" && echo "pulled to $dest";;
  *) sed -n 2,9p "$0"; exit 1;;
esac
