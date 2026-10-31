#!/usr/bin/env bash
# Run the arm GUI (scripts/play_arms.py) on this laptop with the v10 standing policy.
#   scripts/play_arms_local.sh [port]        then open http://localhost:<port>   (default 8765)
# Needs the isaaclab conda env, a free GPU (about 5 GB of the 6 GB RTX 3060; stop other Isaac jobs first)
# and the unitree_sim_isaaclab asset folder. The lab's rsl-rl 5 checkpoint is converted to the laptop's format once.
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${1:-8765}"
PY="${PY:-$HOME/miniconda3/envs/isaaclab/bin/python}"
export G1_USD_PATH="${G1_USD_PATH:-$HOME/Programming/UnitreeRobot/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd}"
SRC=logs/lab_checkpoints/v10/model_1499.pt
CKPT=logs/lab_checkpoints/v10/model_1499_rslrl3.pt
[ -f "$CKPT" ] || "$PY" scripts/convert_checkpoint.py "$SRC" "$CKPT"
exec "$PY" -u scripts/play_arms.py --checkpoint "$CKPT" --port "$PORT" --host 127.0.0.1
