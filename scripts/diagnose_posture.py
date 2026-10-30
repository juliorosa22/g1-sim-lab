#!/usr/bin/env python3
"""What posture does a policy actually hold? Mean leg joint angles and base height while standing (zero command)
and walking (0.5 m/s), arms at rest, no pushes, no noise. Runs the actor directly (no RSL-RL), so it reads
checkpoints of either rsl-rl format.

    python scripts/diagnose_posture.py --task <Play task> --checkpoint <model.pt> --label v8
    G1_GAIN_PROFILE=unitree python scripts/diagnose_posture.py ...     # same policy on the real gains
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--label", default="policy")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seconds", type=float, default=10.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import torch
import torch.nn as nn

from isaaclab_tasks.utils import parse_env_cfg

import g1_rl  # noqa: F401
from g1_rl.robot_cfg import ARM_JOINT_NAMES, GAIN_PROFILE, LEG_JOINT_NAMES


def load_actor(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in ck:
        sd = {k[len("actor."):]: v for k, v in ck["model_state_dict"].items() if k.startswith("actor.")}
    else:
        sd = {k[len("mlp."):]: v for k, v in ck["actor_state_dict"].items() if k.startswith("mlp.")}
    idx = sorted({int(k.split(".")[0]) for k in sd})
    layers = []
    for n, i in enumerate(idx):
        w = sd[f"{i}.weight"]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if n < len(idx) - 1:
            layers.append(nn.ELU())
    net = nn.Sequential(*layers)
    net.load_state_dict(sd)
    return net.eval().to("cuda:0")


@torch.inference_mode()
def main():
    device = "cuda:0"
    cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    cc = cfg.commands.base_velocity
    cc.heading_command = False
    cc.mixed_sampling = False
    cc.rel_standing_envs = 0.0
    cc.deadband = 0.0
    cc.resampling_time_range = (1.0e9, 1.0e9)
    cfg.episode_length_s = args.seconds + 5.0
    cfg.events.push_robot = None
    if hasattr(cfg.commands, "arm_pose"):
        cfg.commands.arm_pose = None
    env = gym.make(args.task, cfg=cfg)
    net = load_actor(args.checkpoint)
    base = env.unwrapped
    robot = base.scene["robot"]
    term = base.command_manager.get_term("base_velocity")
    leg_ids = robot.find_joints(list(LEG_JOINT_NAMES), preserve_order=True)[0]
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    arm_default = robot.data.default_joint_pos[:, arm_ids].clone()
    waist = robot.find_joints(["waist_yaw_joint"])[0]
    steps = int(args.seconds / base.step_dt)
    w0 = int(3.0 / base.step_dt)
    default = robot.data.default_joint_pos[0, leg_ids].cpu()
    names = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]

    print(f"\n[posture] {args.label}  task={args.task}  gain profile={GAIN_PROFILE}  ({args.num_envs} envs, {args.seconds:.0f}s, mean after 3 s)")
    print(f"{'command':>12} {'z (m)':>6} " + " ".join(f"{n:>11}" for n in names) + f" {'waist_yaw':>9} {'knee tq':>7}")
    print(f"{'default pose':>12} {'':>6} " + " ".join(f"{default[i].item():11.3f}" for i in range(6)))
    for cmd in ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)):
        obs, _ = env.reset()
        acc_q, acc_z, acc_w, acc_tq, cnt = torch.zeros(12, device=device), 0.0, 0.0, 0.0, 0
        for i in range(steps):
            term.vel_command_b[:] = torch.tensor(cmd, device=device)
            robot.set_joint_position_target(arm_default, joint_ids=arm_ids)
            obs, _, term_, trunc, _ = env.step(net(obs["policy"]))
            if i >= w0:
                acc_q += robot.data.joint_pos[:, leg_ids].mean(0)
                acc_z += robot.data.root_pos_w[:, 2].mean().item()
                acc_w += robot.data.joint_pos[:, waist].mean().item()
                acc_tq += robot.data.applied_torque[:, leg_ids][:, [3, 9]].abs().mean().item()
                cnt += 1
        q = (acc_q / cnt).cpu()
        lr = [(q[i] + q[i + 6]).item() / 2 for i in range(6)]  # left/right average per joint type
        print(f"{str(cmd):>12} {acc_z / cnt:6.3f} " + " ".join(f"{v:11.3f}" for v in lr) + f" {acc_w / cnt:9.3f} {acc_tq / cnt:7.1f}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
