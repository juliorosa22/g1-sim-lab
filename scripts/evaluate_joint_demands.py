#!/usr/bin/env python3
"""What does the policy ask of each joint while it stands and while it turns in place?  (Request 3 of robot_deploy/docs/sim_requests_from_hardware.md)

Per scenario (stand at zero command; yaw-only commands +-0.2, +-0.3, +-0.5 rad/s): stand 3 s, then the yaw command ramped at 2 rad/s^2 and held,
then 3 s of hold counted from the moment the command is reached; the statistics are over the whole turning segment (ramp + 3 s hold) and over all
envs and control ticks. Arms at rest, no shoves.

  |target - measured q|   what a controller-side tracking trip sees: the target computed from the observation at tick k (default + scale * action)
                          against the joint position measured at the same tick k (before the target is applied)
  |torque|                the torque the simulated actuator applied (N*m), the analogue of the robot's tau_est
  max |joint speed|       rad/s

Reported per joint (legs and waist yaw, left/right pooled by name): p99 and max of each. Also the standing hip pitch / ankle pitch torques.

    python scripts/evaluate_joint_demands.py --task <Play task> --checkpoint <model.pt> --out <file.json> [--noise]
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="G1-Walk-Flat-Inspire-V11-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--yaws", type=float, nargs="+", default=[0.2, -0.2, 0.3, -0.3, 0.5, -0.5])
parser.add_argument("--hold", type=float, default=3.0)
parser.add_argument("--noise", action="store_true")
parser.add_argument("--out", default=None)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app = AppLauncher(args).app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import g1_rl  # noqa: F401
from g1_rl.actor_io import load_actor
from g1_rl.robot_cfg import ARM_JOINT_NAMES
from g1_rl.v8_env_cfg import ACTION_SCALE, CONTROLLED

STAND_S, YAW_RAMP, SETTLE_S = 3.0, 2.0, 1.5  # the stand run skips the first 1.5 s (the robot settling after it spawns)
SHORT = {  # pooled over left/right
    "hip_pitch": ("left_hip_pitch_joint", "right_hip_pitch_joint"), "hip_roll": ("left_hip_roll_joint", "right_hip_roll_joint"),
    "hip_yaw": ("left_hip_yaw_joint", "right_hip_yaw_joint"), "knee": ("left_knee_joint", "right_knee_joint"),
    "ankle_pitch": ("left_ankle_pitch_joint", "right_ankle_pitch_joint"), "ankle_roll": ("left_ankle_roll_joint", "right_ankle_roll_joint"),
    "waist_yaw": ("waist_yaw_joint",),
}


@torch.inference_mode()
def main():
    device = "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.heading_command = False
    if hasattr(cmd_cfg, "mixed_sampling"):
        cmd_cfg.mixed_sampling = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.episode_length_s = 60.0
    env_cfg.events.push_robot = None
    if hasattr(env_cfg.commands, "arm_pose"):
        env_cfg.commands.arm_pose = None
    env_cfg.observations.policy.enable_corruption = bool(args.noise)
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    actor = load_actor(os.path.abspath(args.checkpoint), device)
    base = env.unwrapped
    robot = base.scene["robot"]
    term = base.command_manager.get_term("base_velocity")
    n, dt = args.num_envs, base.step_dt
    ids = robot.find_joints(list(CONTROLLED), preserve_order=True)[0]  # 13 controlled joints, policy order
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    arm_default = robot.data.default_joint_pos[:, arm_ids].clone()
    default = robot.data.default_joint_pos[0, ids].clone()
    scale = torch.empty(len(ids), device=device)
    import re
    for i, name in enumerate(CONTROLLED):
        scale[i] = next(v for k, v in ACTION_SCALE.items() if re.fullmatch(k, name))
    col = {name: i for i, name in enumerate(CONTROLLED)}

    def pooled(arr, names):  # arr (T, n, 13) -> flat samples of the named joints
        return torch.cat([arr[..., col[nm]].reshape(-1) for nm in names])

    def stats(x):
        x = x.abs().float().flatten()
        return {"p99": float(torch.quantile(x[:: max(1, len(x) // 4_000_000)], 0.99)), "max": float(x.max())}

    def run(yaw):
        env.reset()
        obs = env.get_observations()
        steps = int(((STAND_S + abs(yaw) / YAW_RAMP + args.hold) if yaw else 10.0) / dt)
        cmd = 0.0
        err, tq, qd, alive = [], [], [], torch.ones(n, dtype=torch.bool, device=device)
        for i in range(steps):
            t = i * dt
            target_yaw = yaw if t >= STAND_S else 0.0
            cmd += max(-YAW_RAMP * dt, min(YAW_RAMP * dt, target_yaw - cmd))
            term.vel_command_b[:] = torch.tensor([0.0, 0.0, cmd], device=device)
            robot.set_joint_position_target(arm_default, joint_ids=arm_ids)
            q_meas = robot.data.joint_pos[:, ids].clone()
            action = actor(obs["policy"])
            target = default + scale * action
            obs, _, dones, extras = env.step(action)
            alive &= ~(dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool())
            if t >= (STAND_S if yaw else SETTLE_S):
                w = alive.float().unsqueeze(1)
                err.append(((target - q_meas) * w).unsqueeze(0))
                tq.append((robot.data.applied_torque[:, ids] * w).unsqueeze(0))
                qd.append((robot.data.joint_vel[:, ids] * w).unsqueeze(0))
        err, tq, qd = torch.cat(err), torch.cat(tq), torch.cat(qd)
        out = {"survival": float(alive.float().mean())}
        for short, names in SHORT.items():
            out[short] = {"target_minus_q": stats(pooled(err, names)), "torque": stats(pooled(tq, names)), "joint_speed": stats(pooled(qd, names))}
        return out

    results = {"stand": None}
    stand = run(0.0)  # stand: 'yaw' 0 -> the whole run counts
    results["stand"] = stand
    for yaw in args.yaws:
        results[f"yaw {yaw:+.1f}"] = run(yaw)
    print(f"\n[joint-demands] {os.path.basename(args.checkpoint)}  {n} envs  noise={'on' if args.noise else 'off'}; yaw: stand 3 s, ramp 2 rad/s^2, hold {args.hold:.0f} s")
    for scen, res in results.items():
        print(f"\n== {scen}  (survival {res['survival']:.2f})")
        print(f"{'joint':>12} | {'|tgt-q| p99':>11} {'max':>6} | {'|torque| p99':>12} {'max':>6} | {'|speed| p99':>11} {'max':>6}")
        for short in SHORT:
            r = res[short]
            print(f"{short:>12} | {r['target_minus_q']['p99']:11.3f} {r['target_minus_q']['max']:6.3f} | {r['torque']['p99']:12.1f} {r['torque']['max']:6.1f} | "
                  f"{r['joint_speed']['p99']:11.2f} {r['joint_speed']['max']:6.2f}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "noise": args.noise, "results": results}, f, indent=1)
        print(f"[joint-demands] wrote {args.out}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
