#!/usr/bin/env bash
# Full-rate READ-ONLY recorder of rt/lowstate + rt/lowcmd, detached on the robot (survives a dropped ssh/Wi-Fi link).
#   record_walk.sh start [NAME] [SECONDS]   copy the recorder to the robot and start it detached  (starts a process: ask first)
#   record_walk.sh status                    is it running, message counts, queue high-water mark
#   record_walk.sh stop                      ask it to finish (creates STOP), wait, print the summary
#   record_walk.sh pull [NAME]               rsync the run to robot_deploy/measurements/<date>/walk/<NAME>/  (read-only on the robot)
# Wi-Fi: ROBOT_HOST_OVERRIDE=10.173.210.67 robot_deploy/scripts/record_walk.sh ...
# Recordings live in /home/unitree/g1_recordings (deliberately NOT under g1_deploy/, so robot_estop.sh's kill pattern never matches it).
source "$(dirname "${BASH_SOURCE[0]}")/robot_env.sh" || exit 1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REC=/home/$ROBOT_USER/g1_recordings
CUR="$REC/current"        # holds the name of the active/latest run
cmd="${1:-status}"; name="${2:-}"

case "$cmd" in
  start)
    name="${name:-walk_$(date +%Y%m%d_%H%M%S)}"; secs="${3:-1800}"
    robot_ssh "mkdir -p $REC/$name && echo $name > $CUR && ! pgrep -f '[r]ecord_walk.py' >/dev/null || { echo 'a recorder is already running'; exit 1; }" || exit 1
    robot_ssh "cat > $REC/$name/record_walk.py" < "$HERE/record_walk.py"
    robot_ssh "cd $REC/$name && setsid nohup python3 record_walk.py $REC/$name --duration $secs > log.txt 2>&1 < /dev/null & sleep 3; pgrep -af '[r]ecord_walk.py' | cut -c1-120; tail -3 $REC/$name/log.txt"
    echo "recording to $REC/$name for up to ${secs}s (status / stop / pull)";;
  status)
    robot_ssh "n=\$(cat $CUR 2>/dev/null); echo run=\$n; pgrep -af '[r]ecord_walk.py' | cut -c1-100 || echo 'not running'; cat $REC/\$n/status.json 2>/dev/null; echo; df -h $REC | tail -1";;
  stop)
    robot_ssh "n=\$(cat $CUR); touch $REC/\$n/STOP; for i in \$(seq 1 60); do pgrep -f '[r]ecord_walk.py' >/dev/null || break; sleep 1; done; cat $REC/\$n/log.txt | tail -3; ls -l $REC/\$n";;
  pull)
    n="${name:-$(robot_ssh "cat $CUR" | tr -d '\r')}"; dest="$HERE/../measurements/$(date +%Y%m%d)/walk/$n"; mkdir -p "$dest"
    sshpass -e rsync -av -e "ssh ${ROBOT_SSH_OPTS[*]}" "$ROBOT_USER@$ROBOT_HOST:$REC/$n/" "$dest/" && echo "pulled to $dest";;
  *) sed -n 2,8p "$0"; exit 1;;
esac
