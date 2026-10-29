#!/usr/bin/env python3
"""Checks the v5 (Stand) task. Run with the isaaclab env:  python tests/check_stand.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import g1_rl  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

TASK = "G1-Walk-Flat-Inspire-Stand-Play-v0"
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=4096)
env = gym.make(TASK, cfg=cfg)
base = env.unwrapped
term = base.command_manager.get_term("base_velocity")
env.reset()

# 1. command mix and deadband
term._resample_command(torch.arange(base.num_envs, device=base.device))
term.is_standing_env[:] = torch.rand(base.num_envs, device=base.device) < term.cfg.rel_standing_envs
term._update_command()
c = term.vel_command_b
lin, wz = c[:, :2].norm(dim=1), c[:, 2].abs()
zero = (lin == 0) & (wz == 0)
print(f"CHECK standing share (all-zero command): {zero.float().mean().item():.2f} (expect ~0.3 + mode overlap)")
print(f"CHECK commands in (0, 0.05): lin {((lin > 0) & (lin < 0.05)).sum().item()}  yaw {((wz > 0) & (wz < 0.05)).sum().item()}  (expect 0)")

# 2. gait clock: frozen at zero command, runs when moving, restarts at 0
zero_cmd = torch.zeros(3, device=base.device)
term.cfg.resampling_time_range = (1e9, 1e9)
env.reset()
act = torch.zeros(env.action_space.shape, device="cuda:0")
def gait():
    o = base.observation_manager.compute_group("critic")
    return o[0, -2:].tolist()
term.is_standing_env[:] = False  # standing envs get their command forced to zero every step
term.vel_command_b[:] = zero_cmd
for _ in range(10): env.step(act)
print("CHECK gait obs while standing  (expect [0, 1]):", [round(x, 3) for x in gait()])
term.vel_command_b[:] = torch.tensor([0.5, 0.0, 0.0], device=base.device)
seen = []
for _ in range(5):
    env.step(act); seen.append([round(x, 2) for x in gait()])
print("CHECK gait obs after moving starts (expect phase advancing from ~0):", seen[:3])
term.vel_command_b[:] = zero_cmd
for _ in range(3): env.step(act)
print("CHECK gait obs after stopping again (expect [0, 1]):", [round(x, 3) for x in gait()])

# 3. stand rewards are only active while standing
env.reset()
term.is_standing_env[:] = False
rm = base.reward_manager
names = [n for n in rm.active_terms if n.startswith("stand_")]
def term_means(cmd):
    term.vel_command_b[:] = torch.tensor(cmd, device=base.device)
    env.step(act)
    return {n: rm._step_reward[:, rm.active_terms.index(n)].abs().mean().item() for n in names}
standing = term_means([0.0, 0.0, 0.0])
walking = term_means([0.5, 0.0, 0.0])
print("CHECK stand reward terms |mean per-step| standing vs walking (walking should be 0):")
for n in names:
    print(f"CHECK   {n:22s} standing {standing[n]:.5f}   walking {walking[n]:.5f}")

# 4. corner arm poses
arm = base.command_manager.get_term("arm_pose")
arm._resample_command(torch.arange(base.num_envs, device=base.device))
at_end = ((arm.goal - arm.lo).abs() < 1e-6) | ((arm.goal - arm.hi).abs() < 1e-6)
corner = at_end.all(dim=1).float().mean().item()
print(f"CHECK arm goals with every joint at a range end: {corner:.2f} (expect ~0.2 x 0.8 = 0.16)")
env.close()
app.close()
