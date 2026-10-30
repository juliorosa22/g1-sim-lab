#!/usr/bin/env python3
"""Checks the v9/v10 posture terms:  python tests/check_posture_terms.py V9b|V10"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
which = sys.argv[1]

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import g1_rl  # noqa: F401
from g1_rl.robot_cfg import LEG_JOINT_NAMES
from isaaclab_tasks.utils import parse_env_cfg

TASK = f"G1-Walk-Flat-Inspire-{which}-Play-v0"
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=64)
env = gym.make(TASK, cfg=cfg)
base = env.unwrapped
robot = base.scene["robot"]
env.reset()
rm = base.reward_manager
print("CHECK reward terms:", [n for n in rm.active_terms if n in ("base_height", "pose")], " weights:",
      {n: rm.get_term_cfg(n).weight for n in ("base_height", "pose")})

term = base.command_manager.get_term("base_velocity")
term.is_standing_env[:] = False
ids = robot.find_joints(list(LEG_JOINT_NAMES), preserve_order=True)[0]
default = robot.data.default_joint_pos.clone()


def pose_reward(joint_pos, cmd):
    q = default.clone()
    q[:, ids] = joint_pos
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    term.vel_command_b[:] = torch.tensor(cmd, device="cuda:0")
    c = rm.get_term_cfg("pose")
    return c.func(base, **c.params).mean().item()


dflt = default[:, ids]
crouch = dflt.clone()
for j, name in enumerate(LEG_JOINT_NAMES):  # the posture v8 holds: knee ~1.0, hip pitch -0.41, ankle pitch -0.55
    if "knee" in name: crouch[:, j] = 0.97
    if "hip_pitch" in name: crouch[:, j] = -0.41
    if "ankle_pitch" in name: crouch[:, j] = -0.55
slight = dflt.clone()
for j, name in enumerate(LEG_JOINT_NAMES):  # a normal walking stride posture: knee 0.45
    if "knee" in name: slight[:, j] = 0.45
print("CHECK pose reward (1 = perfect):")
for label, pos in (("default pose", dflt), ("walking-like (knee 0.45)", slight), ("v8 crouch (knee 0.97)", crouch)):
    print(f"CHECK   {label:26s} standing cmd {pose_reward(pos, (0.0, 0.0, 0.0)):.3f}   walking cmd 0.5 m/s {pose_reward(pos, (0.5, 0.0, 0.0)):.3f}")

# base height term value at the standing height
env.reset()
c = rm.get_term_cfg("base_height")
val = c.func(base, **c.params)
print(f"CHECK base_height penalty at spawn height {robot.data.root_pos_w[:, 2].mean().item():.3f} m: {val.mean().item():.5f} (x weight {c.weight}); "
      f"at 0.69 m it would be {(0.69-0.78)**2:.4f}")
env.close()
app.close()
