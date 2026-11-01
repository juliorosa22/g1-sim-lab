#!/usr/bin/env python3
"""Checks the v12 pieces:  python tests/check_v12.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import g1_rl  # noqa: F401
from g1_rl import mdp
from isaaclab_tasks.utils import parse_env_cfg

N = 512
cfg = parse_env_cfg("G1-Walk-Flat-Inspire-V12-v0", device="cuda:0", num_envs=N)
env = gym.make("G1-Walk-Flat-Inspire-V12-v0", cfg=cfg)
base = env.unwrapped
env.reset()
rm = base.reward_manager
names = ("action_rate_l2", "dof_acc_l2", "feet_slide", "dof_vel_l2", "feet_swing_height", "base_height", "pose", "stand_feet_air", "stand_speed_l1")
params = base.cfg.curriculum.smooth_ramp.params
print("CHECK reward terms:", [n for n in rm.active_terms if n in names])
for step in (0, 1200, 7000, 14400, 20000):
    base.common_step_counter = step
    f = mdp.ramp_reward_weights(base, None, **params)
    print(f"CHECK ramp step {step:6d} (iter {step // 24:4d}) f={f:.2f}: " + "  ".join(f"{n}={rm.get_term_cfg(n).weight:.2g}" for n in names)
          + f"  std_scale={rm.get_term_cfg('pose').params['std_scale']:.2f}")

# reward values are finite and non-zero on a live env while walking
term = base.command_manager.get_term("base_velocity")
term.is_standing_env[:] = False
term._target[:] = torch.tensor([0.6, 0.0, 0.0], device="cuda:0")
act = torch.zeros(N, base.action_manager.total_action_dim, device="cuda:0")
for _ in range(30):
    base.step(act)
vals = {n: rm.get_term_cfg(n).func(base, **rm.get_term_cfg(n).params) for n in ("feet_swing_height", "dof_vel_l2", "stand_speed_l1", "stand_feet_air")}
print("CHECK values while walking command (mean over envs):", {k: round(float(v.mean()), 5) for k, v in vals.items()}, " finite:", all(bool(torch.isfinite(v).all()) for v in vals.values()))

# command mix
ids = torch.arange(N, device="cuda:0")
tg = []
for _ in range(40):
    term._resample_command(ids)
    tg.append(term._target.clone())
t = torch.cat(tg)
lin, yaw = t[:, :2].norm(dim=1), t[:, 2].abs()
print(f"CHECK target mix: zero {((lin == 0) & (yaw == 0)).float().mean():.2f} (+{base.cfg.commands.base_velocity.rel_standing_envs:.2f} standing flag)  yaw-only {((lin == 0) & (yaw > 0)).float().mean():.2f}  "
      f"lin-only {((lin > 0) & (yaw == 0)).float().mean():.2f}  both {((lin > 0) & (yaw > 0)).float().mean():.2f}; |v| quantiles 25/50/75/95: "
      f"{[round(float(torch.quantile(lin[lin > 0], q)), 2) for q in (0.25, 0.5, 0.75, 0.95)]}")
print("CHECK resampling range:", base.cfg.commands.base_velocity.resampling_time_range)
env.close()
cfgp = parse_env_cfg("G1-Walk-Flat-Inspire-V12-Play-v0", device="cuda:0", num_envs=8)
print("CHECK play cfg ramp:", cfgp.commands.base_velocity.ramp_lin, cfgp.commands.base_velocity.ramp_yaw)
app.close()
