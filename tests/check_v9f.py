#!/usr/bin/env python3
"""Checks the v9f pieces:  python tests/check_v9f.py"""
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
cfg = parse_env_cfg("G1-Walk-Flat-Inspire-V9f-v0", device="cuda:0", num_envs=N)
env = gym.make("G1-Walk-Flat-Inspire-V9f-v0", cfg=cfg)
base = env.unwrapped
env.reset()
rm = base.reward_manager
names = ("base_height", "pose", "stand_feet_air", "track_lin_vel_xy_fine", "track_ang_vel_z_fine")
print("CHECK weights at start:", {n: rm.get_term_cfg(n).weight for n in names})
params = base.cfg.curriculum.posture_ramp.params
for step in (0, 1200, 7000, 14400, 20000):
    base.common_step_counter = step
    f = mdp.ramp_stand_rewards(base, None, **params)
    print(f"CHECK ramp at step {step:6d} (iter {step // 24:4d}): factor {f:.2f}  base_height {rm.get_term_cfg('base_height').weight:7.2f}  "
          f"pose {rm.get_term_cfg('pose').weight:.2f}  stand_feet_air {rm.get_term_cfg('stand_feet_air').weight:.2f}  std_scale {rm.get_term_cfg('pose').params['std_scale']:.2f}")

term = base.command_manager.get_term("base_velocity")
ids = torch.arange(N, device="cuda:0")
tg = []
for _ in range(40):
    term._resample_command(ids)
    tg.append(term._target.clone())
t = torch.cat(tg)
lin, yaw = t[:, :2].norm(dim=1), t[:, 2].abs()
print(f"CHECK target mix (n={len(t)}, before the standing flag): all-zero {((lin == 0) & (yaw == 0)).float().mean():.2f}  yaw-only {((lin == 0) & (yaw > 0)).float().mean():.2f}  "
      f"lin-only {((lin > 0) & (yaw == 0)).float().mean():.2f}  both {((lin > 0) & (yaw > 0)).float().mean():.2f}  | none in (0,0.05): {bool((((lin > 0) & (lin < 0.05)).sum() == 0) and (((yaw > 0) & (yaw < 0.05)).sum() == 0))}")
print(f"CHECK yaw-only |wz| quantiles 10/50/90%: {[round(float(torch.quantile(yaw[(lin == 0) & (yaw > 0)], q)), 2) for q in (0.1, 0.5, 0.9)]}")

# ramp: step the real env and look at how fast the command seen by the policy changes
env.reset()
term._target[:] = torch.tensor([1.0, 0.0, 1.0], device="cuda:0")
term.is_standing_env[:] = False
term.vel_command_b[:] = 0.0
act = torch.zeros(N, base.action_manager.total_action_dim, device="cuda:0")
prev, worst_lin, worst_yaw, seen = term.vel_command_b.clone(), 0.0, 0.0, []
for i in range(60):
    base.step(act)
    cur = term.vel_command_b
    worst_lin = max(worst_lin, float((cur[:, :2] - prev[:, :2]).norm(dim=1).max()))
    worst_yaw = max(worst_yaw, float((cur[:, 2] - prev[:, 2]).abs().max()))
    prev = cur.clone()
    if i in (0, 24, 49):
        seen.append((i + 1, float(cur[0, 0]), float(cur[0, 2])))
print(f"CHECK ramp: max change per step lin {worst_lin:.4f} (limit {1.0 * base.step_dt:.4f}), yaw {worst_yaw:.4f} (limit {2.0 * base.step_dt:.4f}); (step, vx, wz) {seen}")
env.close()

cfgp = parse_env_cfg("G1-Walk-Flat-Inspire-V9f-Play-v0", device="cuda:0", num_envs=8)
print("CHECK play cfg ramp:", cfgp.commands.base_velocity.ramp_lin, cfgp.commands.base_velocity.ramp_yaw)
app.close()
