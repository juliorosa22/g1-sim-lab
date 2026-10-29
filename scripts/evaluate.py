#!/usr/bin/env python3
"""Evaluate a trained g1_rl walking policy at fixed velocity commands.

For each command [vx, vy, yaw_rate] the same batch of envs is reset, the command
is held constant, and the policy runs for --seconds of sim time. Reported per command:

  survival     fraction of envs that did not terminate early (fall / bad orientation)
  vx, vy, wz   mean achieved velocity (body frame / world yaw rate) before any termination
  err_xy, err_w   mean |achieved - commanded| for linear (m/s) and yaw (rad/s)
  z            mean base height
  d_act        mean |action_t - action_{t-1}| (smoothness; lower is better)
  torque       mean |applied torque| over the 12 leg joints (N*m)

Run with the isaaclab conda env:
    python scripts/evaluate.py                       # newest checkpoint, headless
    python scripts/evaluate.py --checkpoint logs/rsl_rl/g1_walk_flat_inspire/<run>/model_1499.pt
    python scripts/evaluate.py --push                # add random push disturbances
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="G1-Walk-Flat-Inspire-Play-v0")
parser.add_argument("--checkpoint", default=None, help="model_*.pt; default: newest under logs/rsl_rl/<experiment>")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seconds", type=float, default=20.0, help="sim time per command")
parser.add_argument("--push", action="store_true", help="enable random push disturbances")
parser.add_argument("--out", default=None, help="write results JSON here")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import g1_rl  # noqa: F401  (registers the gym ids)
from g1_rl.walk_env_cfg import EventCfg

COMMANDS = [
    (0.0, 0.0, 0.0),
    (0.3, 0.0, 0.0),
    (0.5, 0.0, 0.0),
    (0.8, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.3, 0.0),
    (0.0, 0.0, 0.5),
    (0.0, 0.0, 1.0),
    (0.5, 0.0, 0.5),
]


@torch.inference_mode()
def main():
    device = "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    # Fixed, repeatable commands: no heading controller, no standing envs, no resampling.
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.heading_command = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.episode_length_s = args.seconds + 1.0  # no time-out inside the measured window
    if args.push:
        env_cfg.events.push_robot = EventCfg().push_robot

    if args.checkpoint:
        ckpt = os.path.abspath(args.checkpoint)
    else:
        root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        ckpt = get_checkpoint_path(root, ".*", "model_.*.pt")
    print(f"[eval] checkpoint: {ckpt}")

    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = runner.alg.policy

    base = env.unwrapped
    robot = base.scene["robot"]
    cmd_term = base.command_manager.get_term("base_velocity")
    leg_ids = robot.find_joints(
        [".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"], preserve_order=False
    )[0]
    steps = int(args.seconds / base.step_dt)
    n = args.num_envs

    results = []
    for vx, vy, wz in COMMANDS:
        cmd_cfg.ranges.lin_vel_x = (vx, vx)
        cmd_cfg.ranges.lin_vel_y = (vy, vy)
        cmd_cfg.ranges.ang_vel_z = (wz, wz)
        cmd_term.cfg.ranges = cmd_cfg.ranges
        env.reset()
        obs = env.get_observations()

        alive = torch.ones(n, dtype=torch.bool, device=device)
        sums = {k: torch.zeros(n, device=device) for k in ("vx", "vy", "wz", "z", "d_act", "torque")}
        count = torch.zeros(n, device=device)
        prev_action = torch.zeros(n, base.action_manager.total_action_dim, device=device)

        for _ in range(steps):
            action = policy(obs)
            obs, _, dones, extras = env.step(action)
            policy_nn.reset(dones)
            # `alive` stays True only until an env's first termination; its stats stop there.
            died = dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
            w = alive.float()
            sums["vx"] += w * robot.data.root_lin_vel_b[:, 0]
            sums["vy"] += w * robot.data.root_lin_vel_b[:, 1]
            sums["wz"] += w * robot.data.root_ang_vel_w[:, 2]
            sums["z"] += w * robot.data.root_pos_w[:, 2]
            sums["d_act"] += w * (action - prev_action).abs().mean(dim=1)
            sums["torque"] += w * robot.data.applied_torque[:, leg_ids].abs().mean(dim=1)
            count += w
            prev_action = action.clone()
            alive &= ~died

        c = count.clamp(min=1.0)
        m = {k: (v / c).mean().item() for k, v in sums.items()}
        row = {
            "cmd": [vx, vy, wz],
            "survival": alive.float().mean().item(),
            "vx": m["vx"], "vy": m["vy"], "wz": m["wz"],
            "err_xy": ((sums["vx"] / c - vx) ** 2 + (sums["vy"] / c - vy) ** 2).sqrt().mean().item(),
            "err_w": (sums["wz"] / c - wz).abs().mean().item(),
            "z": m["z"], "d_act": m["d_act"], "torque": m["torque"],
        }
        results.append(row)

    print(f"\n[eval] {n} envs x {args.seconds:.0f}s per command, push={'on' if args.push else 'off'}")
    print(f"{'cmd (vx,vy,wz)':>18} {'surv':>6} {'vx':>6} {'vy':>6} {'wz':>6} {'err_xy':>7} {'err_w':>6} {'z':>6} {'d_act':>6} {'torque':>7}")
    for r in results:
        cmd = "({:.1f},{:.1f},{:.1f})".format(*r["cmd"])
        print(f"{cmd:>18} {r['survival']:6.2f} {r['vx']:6.2f} {r['vy']:6.2f} {r['wz']:6.2f} "
              f"{r['err_xy']:7.2f} {r['err_w']:6.2f} {r['z']:6.3f} {r['d_act']:6.2f} {r['torque']:7.2f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"checkpoint": ckpt, "push": args.push, "seconds": args.seconds, "results": results}, f, indent=2)
        print(f"[eval] wrote {args.out}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
