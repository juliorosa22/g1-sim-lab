#!/usr/bin/env python3
"""Checks the v6 (StandRamp) task: reward ramp, kernel anneal, fall penalty, recovery exemption.
    python tests/check_stand_ramp.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import g1_rl  # noqa: F401
from g1_rl.mdp.rewards import _stand_mask
from isaaclab_tasks.utils import parse_env_cfg

TASK = "G1-Walk-Flat-Inspire-StandRamp-Play-v0"
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=64)
env = gym.make(TASK, cfg=cfg)
base = env.unwrapped
env.reset()
rm, cm = base.reward_manager, base.curriculum_manager
names = ["stand_still_exp", "stand_speed_l1", "stand_feet_air", "stand_joint_vel_l2", "stand_pose_l1", "stand_action_rate_l2"]

def show(label):
    w = {n: round(rm.get_term_cfg(n).weight, 4) for n in names}
    std = rm.get_term_cfg("stand_still_exp").params["std"]
    print(f"CHECK {label:22s} std={std:.3f} weights={w}")

print("CHECK termination weight:", rm.get_term_cfg("termination_penalty").weight, "(x dt 0.02 = -40 per fall)")
for steps in (0, 600, 2400, 4200, 10000):
    base.common_step_counter = steps
    cm.compute(env_ids=torch.arange(base.num_envs, device=base.device))
    show(f"ramp at step {steps}")

# recovery exemption: kick half the envs to 0.6 m/s while commanded to stand
term = base.command_manager.get_term("base_velocity")
term.is_standing_env[:] = False
term.vel_command_b[:] = 0.0
robot = base.scene["robot"]
vel = robot.data.root_vel_w.clone()
vel[:32, 0] += 0.6
robot.write_root_velocity_to_sim(vel)
plain = _stand_mask(base, "base_velocity", 0.05, 0.0)
exempt = _stand_mask(base, "base_velocity", 0.05, 0.25)
print(f"CHECK standing mask, no exemption : kicked {plain[:32].mean().item():.2f}  calm {plain[32:].mean().item():.2f}  (expect 1 / 1)")
print(f"CHECK standing mask, recovery 0.25: kicked {exempt[:32].mean().item():.2f}  calm {exempt[32:].mean().item():.2f}  (expect 0 / 1)")
env.close()
app.close()
