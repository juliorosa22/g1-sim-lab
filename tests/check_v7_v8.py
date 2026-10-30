#!/usr/bin/env python3
"""Checks the v7 / v8 real-plant tasks:  python tests/check_v7_v8.py V7|V8"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
which = sys.argv[1]

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import gymnasium as gym
import torch

import g1_rl  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

TASK = f"G1-Walk-Flat-Inspire-{which}-Play-v0"
cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=256)
env = gym.make(TASK, cfg=cfg)
base = env.unwrapped
robot = base.scene["robot"]
obs, _ = env.reset()
n_act = env.action_space.shape[-1]
print(f"CHECK {which}: actor obs {tuple(obs['policy'].shape)}  critic obs {tuple(obs['critic'].shape)}  actions {n_act}")


def mean_gain(name):
    for act in robot.actuators.values():
        if name in act.joint_names:
            j = list(act.joint_names).index(name)
            return act.stiffness[:, j].mean().item(), act.damping[:, j].mean().item()


for nm in ("left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "waist_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint"):
    kp, kd = mean_gain(nm)
    print(f"CHECK   mean gain {nm:26s} kp {kp:6.1f} kd {kd:5.2f}")
i = robot.joint_names.index("left_ankle_pitch_joint")
print("CHECK   ankle speed limit:", round(robot.actuators["legs"].velocity_limit[0, list(robot.actuators["legs"].joint_names).index("left_ankle_pitch_joint")].item(), 1))

# locked waist roll/pitch: largest excursion over 8 s, counting only envs that have not fallen or been reset
roll, pitch = robot.joint_names.index("waist_roll_joint"), robot.joint_names.index("waist_pitch_joint")
alive = torch.ones(256, dtype=torch.bool, device="cuda:0")
worst_alive = worst_all = 0.0
for _ in range(400):
    _, _, term, trunc, _ = env.step(torch.randn(env.action_space.shape, device="cuda:0") * 0.2)
    exc = robot.data.joint_pos[:, [roll, pitch]].abs().max(dim=1).values
    worst_all = max(worst_all, exc.max().item())
    alive &= ~(term | trunc)
    if alive.any():
        worst_alive = max(worst_alive, exc[alive].max().item())
print(f"CHECK   waist roll/pitch largest excursion over 8 s: {worst_alive:.3f} rad among envs still upright "
      f"({int(alive.sum())}/256), {worst_all:.3f} rad including fallen ones (locked: want ~0)")

cc = base.command_manager.get_term("base_velocity")
print(f"CHECK   command term: {type(cc).__name__}  heading_command={cc.cfg.heading_command}  rel_standing={cc.cfg.rel_standing_envs}  "
      f"resample={cc.cfg.resampling_time_range}  vx range={cc.cfg.ranges.lin_vel_x}  deadband={getattr(cc.cfg, 'deadband', 'n/a')}")
print("CHECK   push interval / velocity:", cfg.events.push_robot and (cfg.events.push_robot.interval_range_s, cfg.events.push_robot.params["velocity_range"]))
if which == "V8":
    yaw = robot.joint_names.index("waist_yaw_joint")
    q0 = robot.data.joint_pos[:, yaw].clone()
    a = torch.zeros(env.action_space.shape, device="cuda:0"); a[:, 12] = 2.0  # waist yaw action index
    for _ in range(25):
        env.step(a)
    print(f"CHECK   V8 waist yaw responds to action #12: mean change {(robot.data.joint_pos[:, yaw]-q0).mean().item():+.3f} rad")
    print("CHECK   V8 gyro bias drawn:", tuple(base.gyro_bias.shape), f"|max|={base.gyro_bias.abs().max().item():.4f}")
    print("CHECK   V8 arm command term present:", "arm_pose" in base.command_manager.active_terms)
env.close()
app.close()
