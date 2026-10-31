#!/usr/bin/env python3
"""Checks the v9c pieces:  python tests/check_v9c.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import g1_rl  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

TASK = "G1-Walk-Flat-Inspire-V9c-Play-v0"
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=512)
env = gym.make(TASK, cfg=cfg)
base = env.unwrapped
env.reset()
rm = base.reward_manager
print("CHECK reward terms:", {n: rm.get_term_cfg(n).weight for n in rm.active_terms if n in ("base_height", "pose", "track_lin_vel_xy_fine", "track_ang_vel_z_fine")})
print("CHECK pose std walking:", rm.get_term_cfg("pose").params["std_walking"])

term = base.command_manager.get_term("base_velocity")
ids = torch.arange(512, device="cuda:0")
samples = []
for _ in range(20):
    term._resample_command(ids)
    samples.append(term.vel_command_b.clone())
c = torch.cat(samples)
lin = c[:, :2].norm(dim=1)
yaw = c[:, 2].abs()
moving_l, moving_y = lin > 0, yaw > 0
print(f"CHECK n={len(c)}  lin>0: {moving_l.float().mean():.2f}  yaw>0: {moving_y.float().mean():.2f}")
print(f"CHECK lin norm   <0.05: {((lin < 0.05)).float().mean():.2f} (all exactly zero: {bool(((lin < 0.05) == (lin == 0)).all())})")
print(f"CHECK lin norm in (0.05,0.3]: {((lin > 0) & (lin <= 0.3)).float().mean():.2f}   > 0.3: {(lin > 0.3).float().mean():.2f}")
print(f"CHECK yaw in (0.05,0.5]: {((yaw > 0) & (yaw <= 0.5)).float().mean():.2f}   > 0.5: {(yaw > 0.5).float().mean():.2f}")
print(f"CHECK min nonzero lin {lin[lin > 0].min():.3f}  min nonzero yaw {yaw[yaw > 0].min():.3f}")

# the narrow kernels: standing still while commanded
import math
for name, std, cmd in (("lin fine @0.2", 0.15, 0.2), ("lin wide @0.2", 0.5, 0.2), ("yaw fine @0.5", 0.25, 0.5), ("yaw wide @0.5", 0.5, 0.5)):
    print(f"CHECK hover reward {name}: {math.exp(-cmd**2 / std**2):.3f}")
env.close()
app.close()
