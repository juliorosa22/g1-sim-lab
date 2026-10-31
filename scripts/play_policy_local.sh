#!/usr/bin/env bash
# Policy playground on this laptop:  scripts/play_policy_local.sh [port] [initial policy]    then open http://localhost:<port>
# Needs the isaaclab conda env, a free GPU (about 5 GB of the 6 GB RTX 3060: stop other Isaac jobs first) and the unitree_sim_isaaclab assets.
# Policies are offered when their checkpoint is in logs/lab_checkpoints/<name>/ (copy more from the lab with scp).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/miniconda3/envs/isaaclab/bin/python}"
export G1_USD_PATH="${G1_USD_PATH:-$HOME/Programming/UnitreeRobot/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd}"
exec "$PY" -u scripts/play_policy.py --port "${1:-8765}" --policy "${2:-v10}"
