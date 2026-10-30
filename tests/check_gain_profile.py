#!/usr/bin/env python3
"""Checks the G1_GAIN_PROFILE switch: applied gains, and that the simulation stays finite at the
Unitree gains (explicit PD at a 5 ms step is only stable while kd*dt/inertia < 2).
    G1_GAIN_PROFILE=unitree python tests/check_gain_profile.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import g1_rl  # noqa: F401
from g1_rl.robot_cfg import GAIN_PROFILE
from isaaclab_tasks.utils import parse_env_cfg

TASK = "G1-Walk-Flat-Inspire-Play-v0"
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=64)
env = gym.make(TASK, cfg=cfg)
base = env.unwrapped
robot = base.scene["robot"]
env.reset()
print(f"CHECK profile = {GAIN_PROFILE}")
want = ["left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "waist_yaw_joint", "left_shoulder_pitch_joint", "left_elbow_joint", "left_wrist_roll_joint"]
def gains(name):
    # explicit actuators (the legs) keep their gains on the actuator object; implicit ones in the sim drive
    for act in robot.actuators.values():
        if name in act.joint_names:
            j = list(act.joint_names).index(name)
            return act.stiffness[0, j].item(), act.damping[0, j].item()
    raise KeyError(name)


for name in want:
    kp, kd = gains(name)
    print(f"CHECK   {name:28s} kp {kp:6.1f}  kd {kd:5.1f}")

# finite and sane for 6 s with a standing command and the policy's default action (zeros)
act = torch.zeros(env.action_space.shape, device="cuda:0")
worst = 0.0
for _ in range(300):
    env.step(act)
    worst = max(worst, robot.data.joint_vel.abs().max().item())
    assert torch.isfinite(robot.data.joint_pos).all(), "non-finite joint positions: numerically unstable"
print(f"CHECK finite for 300 steps; largest joint speed seen {worst:.1f} rad/s (limits 20-37)")
env.close()
app.close()
