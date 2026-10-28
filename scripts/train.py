#!/usr/bin/env python3
"""Train a g1_rl task with Isaac Lab's RSL-RL script.

Registers this repo's gym ids, then runs Isaac Lab's own train.py unchanged:

    python scripts/train.py --task G1-Walk-Flat-Inspire-v0 --headless --num_envs 2048

All of train.py's flags (--max_iterations, --seed, --resume, ...) pass through.
Needs the isaaclab conda env; ISAACLAB_PATH overrides the default ~/IsaacLab.
"""
import os
import runpy
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISAACLAB = os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab"))
RSL_RL_DIR = os.path.join(ISAACLAB, "scripts", "reinforcement_learning", "rsl_rl")

sys.path.insert(0, REPO_ROOT)  # g1_rl, g1_sim
sys.path.insert(0, RSL_RL_DIR)  # train.py's own `import cli_args`

import g1_rl  # noqa: E402,F401  (registers the gym ids)

runpy.run_path(os.path.join(RSL_RL_DIR, "train.py"), run_name="__main__")
