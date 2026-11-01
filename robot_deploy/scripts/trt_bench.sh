#!/usr/bin/env bash
# Policy-inference benchmark on the robot (numpy CPU vs TensorRT GPU). Pure compute: publishes nothing, reads no robot topics.
#   trt_bench.sh push                 copy the ONNX/npz/test vectors + trt_bench.py to /home/unitree/g1_recordings/trt_bench/ (a scratch dir)
#   trt_bench.sh build ROLE_VERSION   build an FP32 TensorRT engine ON the robot with trtexec (uses the GPU for ~1 min), e.g. walk_v4
#   trt_bench.sh bench MODE ROLE_VERSION [opts]   MODE numpy|trt; opts go to trt_bench.py (--iters --paced-s --hz)
#   trt_bench.sh pull                 rsync results to robot_deploy/measurements/<date>/trt/
# Wi-Fi: ROBOT_HOST_OVERRIDE=10.173.210.67 robot_deploy/scripts/trt_bench.sh ...
# Starts compute processes on the robot: ask the user first (CLAUDE.md / unitree-robot-ssh rules).
source "$(dirname "${BASH_SOURCE[0]}")/robot_env.sh" || exit 1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=/home/$ROBOT_USER/g1_recordings/trt_bench
STAMP=20261029
cmd="${1:-}"; a="${2:-}"; b="${3:-}"
case "$cmd" in
  push)  robot_ssh "mkdir -p $R"
         sshpass -e rsync -av -e "ssh ${ROBOT_SSH_OPTS[*]}" "$HERE/../policies/"*.onnx "$HERE/../policies/"*.npz "$HERE/trt_bench.py" "$ROBOT_USER@$ROBOT_HOST:$R/";;
  build) [ -n "$a" ] || { sed -n 2,7p "$0"; exit 1; }
         robot_ssh "cd $R && TRTX=/usr/src/tensorrt/bin/trtexec; (\$TRTX --onnx=${a}_$STAMP.onnx --saveEngine=${a}${ENG:-}.engine ${NOTF32:+--noTF32} --memPoolSize=workspace:256 --iterations=2000 --warmUp=500 --avgRuns=100 > build_${a}${ENG:-}.log 2>&1 || \$TRTX --onnx=${a}_$STAMP.onnx --saveEngine=${a}${ENG:-}.engine ${NOTF32:+--noTF32} --workspace=256 --iterations=2000 --warmUp=500 --avgRuns=100 > build_${a}${ENG:-}.log 2>&1); tail -n 25 build_${a}${ENG:-}.log | grep -E 'GPU Compute|Latency|Throughput|PASSED|FAILED|Error|error' ; ls -l ${a}${ENG:-}.engine";;
  bench) [ -n "$a" ] && [ -n "$b" ] || { sed -n 2,7p "$0"; exit 1; }
         shift 3
         if [ "$a" = numpy ]; then ARGS="numpy --npz ${b}_$STAMP.npz"; else ARGS="trt --engine ${b}${ENG:-}.engine"; fi
         ENVV=""; SUF="${ENG:-}"
         if [ "${THREADS1:-}" = 1 ]; then ENVV="OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"; SUF="${ENG:-}_1thr"; fi   # single-threaded BLAS, as a control loop would run
         robot_ssh "cd $R && $ENVV python3 trt_bench.py $ARGS --testvec ${b}_${STAMP}_testvec.npz --out res_${a}_${b}${SUF}.json $*" 2>&1 | grep -v 'take sample\|DeprecationWarning\|get_binding_name';;
  pull)  dest="$HERE/../measurements/$(date +%Y%m%d)/trt"; mkdir -p "$dest"
         sshpass -e rsync -av -e "ssh ${ROBOT_SSH_OPTS[*]}" --include='res_*.json' --include='build_*.log' --exclude='*' "$ROBOT_USER@$ROBOT_HOST:$R/" "$dest/" && echo "pulled to $dest";;
  *) sed -n 2,7p "$0"; exit 1;;
esac
