#!/usr/bin/env python3
"""Checks the v3 (waist) and v4 (arms) tasks. Run with the isaaclab env:
    python tests/check_body.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import g1_rl  # noqa: F401
from g1_rl.robot_cfg import ARM_JOINT_NAMES
from isaaclab_tasks.utils import parse_env_cfg

TASK = "G1-Walk-Flat-Inspire-Arms-Play-v0"
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=64)
env = gym.make(TASK, cfg=cfg)
base = env.unwrapped
robot = base.scene["robot"]
env.reset()
zero = torch.zeros(env.action_space.shape, device="cuda:0")
print("CHECK action dim:", env.action_space.shape, "(expect 15)")

arm_ids = [robot.joint_names.index(n) for n in ARM_JOINT_NAMES]
finger_ids = [i for i, n in enumerate(robot.joint_names) if n[:2] in ("L_", "R_")]
term = base.command_manager.get_term("arm_pose")
span = torch.zeros(len(arm_ids), device="cuda:0")
lo = torch.full((len(arm_ids),), 9.0, device="cuda:0")
hi = torch.full((len(arm_ids),), -9.0, device="cuda:0")
maxstep = 0.0
env.step(zero)
prev = term.current.clone()
for _ in range(400):  # 8 s
    env.step(zero)
    fresh = base.episode_length_buf <= 1  # just reset: target jumped back to the default pose, not a rate violation
    q = robot.data.joint_pos[:, arm_ids]
    lo = torch.minimum(lo, q.min(0).values)
    hi = torch.maximum(hi, q.max(0).values)
    d = (term.current - prev).abs().max(dim=1).values
    maxstep = max(maxstep, d[~fresh].max().item())
    prev = term.current.clone()
print("CHECK arm joint range reached (rad), per joint:")
for n, a, b in zip(ARM_JOINT_NAMES, lo.tolist(), hi.tolist()):
    print(f"CHECK   {n:28s} [{a:6.2f}, {b:6.2f}]")
print(f"CHECK max arm target step per control step: {maxstep:.4f} rad (limit {term.cfg.max_velocity * base.step_dt:.4f})")
fing = (robot.data.joint_pos[:, finger_ids] - robot.data.default_joint_pos[:, finger_ids]).abs().max().item()
print(f"CHECK fingers held, max deviation: {fing:.3f} rad")
print("CHECK waist actuator:", type(robot.actuators["waist"]).__name__)
env.close()
app.close()
