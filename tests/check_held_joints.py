#!/usr/bin/env python3
"""Sanity check for g1_rl: with zero actions, do the joints the policy does not
control (arms, waist, hands) stay at their default pose, and does the robot
stay up? Run with the isaaclab env:

    python tests/check_held_joints.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import g1_rl  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("G1-Walk-Flat-Inspire-Play-v0", device="cuda:0", num_envs=16)
env = gym.make("G1-Walk-Flat-Inspire-Play-v0", cfg=cfg)
robot = env.unwrapped.scene["robot"]
obs, _ = env.reset()

held = [i for i, n in enumerate(robot.joint_names) if not any(
    n == l for l in g1_rl.robot_cfg.LEG_JOINT_NAMES)]
zero = torch.zeros(env.action_space.shape, device="cuda:0")
resets = 0
for _ in range(100):
    _, _, term, trunc, _ = env.step(zero)
    resets += int((term | trunc).sum())

dev = (robot.data.joint_pos[:, held] - robot.data.default_joint_pos[:, held]).abs()
worst = dev.max(0).values
names = [robot.joint_names[i] for i in held]
top = sorted(zip(worst.tolist(), names), reverse=True)[:5]
print("CHECK obs shape:", tuple(obs["policy"].shape), "critic:", tuple(obs["critic"].shape))
print("CHECK base z mean/min:", round(robot.data.root_pos_w[:, 2].mean().item(), 3),
      round(robot.data.root_pos_w[:, 2].min().item(), 3))
print("CHECK episode ends in 100 steps (16 envs):", resets)
print("CHECK worst held-joint deviation (rad):", [(n, round(v, 3)) for v, n in top])
env.close()
app.close()
