#!/usr/bin/env python3
"""Checks that the g1_rl robustness changes actually take effect. Run with the
isaaclab env:  python tests/check_robustness.py
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

TASK = "G1-Walk-Flat-Inspire-v0"
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=4096)
env = gym.make(TASK, cfg=cfg)
base = env.unwrapped
robot = base.scene["robot"]
env.reset()

# 1. command mix: resample every env and classify
term = base.command_manager.get_term("base_velocity")
term._resample_command(torch.arange(base.num_envs, device=base.device))
term.is_standing_env[:] = False
c = term.vel_command_b
nz = c.abs() > 1e-6
fwd = nz[:, 0] & ~nz[:, 1] & ~nz[:, 2]
lat = ~nz[:, 0] & nz[:, 1] & ~nz[:, 2]
rot = ~nz[:, 0] & ~nz[:, 1] & nz[:, 2]
n = base.num_envs
print(f"CHECK command mix (4096 samples): forward-only {fwd.sum().item()/n:.2f}  lateral-only {lat.sum().item()/n:.2f}  "
      f"rotate-only {rot.sum().item()/n:.2f}  (expect ~0.20 / 0.10 / 0.20)")
print(f"CHECK command ranges: vx [{c[:,0].min():.2f}, {c[:,0].max():.2f}]  vy [{c[:,1].min():.2f}, {c[:,1].max():.2f}]  "
      f"wz [{c[:,2].min():.2f}, {c[:,2].max():.2f}]")

# 2. encoder bias reaches the actor's observation but not the critic's
bias = getattr(base, "encoder_bias", None)
obs_p = base.observation_manager.compute_group("policy")
obs_c = base.observation_manager.compute_group("critic")
true_q = robot.data.joint_pos[:, [robot.joint_names.index(j) for j in g1_rl.robot_cfg.LEG_JOINT_NAMES]] \
    - robot.data.default_joint_pos[:, [robot.joint_names.index(j) for j in g1_rl.robot_cfg.LEG_JOINT_NAMES]]
print("CHECK encoder_bias:", None if bias is None else (tuple(bias.shape), f"|max|={bias.abs().max().item():.4f}"))
print(f"CHECK critic joint_pos == true: {torch.allclose(obs_c[:, 12:24], true_q, atol=1e-5)}   "
      f"(critic obs width {obs_c.shape[1]}, policy {obs_p.shape[1]})")

# 3. ankle effort limit and randomized actuator gains
leg = robot.actuators["legs"]
names = [robot.joint_names[i] for i in leg.joint_indices]
ankle = [i for i, nme in enumerate(names) if "ankle" in nme]
print("CHECK ankle effort limit:", leg.effort_limit[0, ankle].tolist()[:2])
kp = leg.stiffness[:, [i for i, nme in enumerate(names) if nme == "left_knee_joint"]].flatten()
print(f"CHECK knee kp across envs: min {kp.min():.1f} max {kp.max():.1f} (nominal 150, expect ~120..180)")

# 4. push configuration
push = cfg.events.push_robot
print("CHECK push interval/velocity:", push.interval_range_s, push.params["velocity_range"])
env.close()
app.close()
