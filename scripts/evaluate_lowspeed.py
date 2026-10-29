#!/usr/bin/env python3
"""Low-speed evaluation of a g1_rl walking policy.

Before putting a standing policy behind a speed threshold we need to know where the
walking policy stops being usable: how it behaves for small commands (0.03-0.3 m/s,
0.05-0.3 rad/s) in every direction. For each command (constant, same for all envs) it
measures, over the window after the first 3 s:

  achieved   mean body-frame velocity (vx, vy) and yaw rate, averaged over envs
  ok%        share of envs that track the command: |achieved - cmd| < max(0.03, 30% of |cmd|)
             on every commanded axis, and are not sliding on an uncommanded one (< 0.03)
  jitter     temporal std of the speed along the commanded direction, averaged over envs
  lifts/s    foot lift-offs per second (a foot off the ground >= 0.1 s). Walking at the
             0.8 s gait clock gives ~2.5; 0 means the feet stay planted.
  foot cm/s  distance the feet travel per second
  tilt       RMS tilt away from vertical, degrees
  surv       share of envs that did not fall

Arms are held at the default pose by this script by default (--arms rest), so the result
is about the legs and waist only; --arms random leaves the task's own arm randomiser on.

    python scripts/evaluate_lowspeed.py --task <Play task> --checkpoint <model.pt> --label v4 [--noise]
"""
import argparse
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--label", default="policy")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seconds", type=float, default=20.0, help="per command")
parser.add_argument("--noise", action="store_true", help="enable observation noise")
parser.add_argument("--arms", choices=["rest", "random"], default="rest")
parser.add_argument("--out", default=None)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import g1_rl  # noqa: F401
from g1_rl.robot_cfg import ARM_JOINT_NAMES

WINDOW_START_S = 3.0
CONTACT_N = 10.0
MIN_AIR_S = 0.1
ABS_TOL = 0.03
REL_TOL = 0.30

# (vx, vy, wz): zero first as the reference row
COMMANDS = (
    [(0.0, 0.0, 0.0)]
    + [(v, 0.0, 0.0) for v in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30)]
    + [(-v, 0.0, 0.0) for v in (0.05, 0.10, 0.20)]
    + [(0.0, v, 0.0) for v in (0.05, 0.10, 0.20)]
    + [(0.0, 0.0, w) for w in (0.05, 0.10, 0.20, 0.30)]
    + [(0.10, 0.0, 0.10), (0.10, 0.0, -0.10)]
)


@torch.inference_mode()
def main():
    device = "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    try:
        import importlib.metadata as _metadata

        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _metadata.version("rsl-rl-lib"))
    except ImportError:
        pass

    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.heading_command = False
    cmd_cfg.mixed_sampling = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.deadband = 0.0  # tasks with a deadband would otherwise zero these small commands
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.episode_length_s = args.seconds + 5.0
    env_cfg.events.push_robot = None
    drive_arms = args.arms == "rest"
    if drive_arms and hasattr(env_cfg.commands, "arm_pose"):
        env_cfg.commands.arm_pose = None
    if args.noise:
        env_cfg.observations.policy.enable_corruption = True

    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(args.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    robot = base.scene["robot"]
    term = base.command_manager.get_term("base_velocity")
    n, dt = args.num_envs, base.step_dt
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    arm_default = robot.data.default_joint_pos[:, arm_ids].clone()
    foot_ids = robot.find_bodies(".*_ankle_roll_link")[0]
    sensor = base.scene["contact_forces"]
    sensor_foot_ids = sensor.find_bodies(".*_ankle_roll_link")[0]
    steps = int(args.seconds / dt)
    w0 = int(WINDOW_START_S / dt)
    min_air = max(int(MIN_AIR_S / dt), 1)

    rows = []
    for vx, vy, wz in COMMANDS:
        cmd_cfg.ranges.lin_vel_x = (vx, vx)
        cmd_cfg.ranges.lin_vel_y = (vy, vy)
        cmd_cfg.ranges.ang_vel_z = (wz, wz)
        term.cfg.ranges = cmd_cfg.ranges
        env.reset()
        obs = env.get_observations()
        cmd = torch.tensor([vx, vy, wz], device=device)

        alive = torch.ones(n, dtype=torch.bool, device=device)
        z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
        sum_v, sum_v2, count = z(n, 3), z(n), z(n)
        sum_along, sum_along2 = z(n), z(n)
        sum_tilt2, lifts, foot_path = z(n), z(n), z(n)
        air_run, prev_foot = z(n, 2), None
        direction = cmd / cmd.norm().clamp(min=1e-9)
        for i in range(steps):
            if drive_arms:
                robot.set_joint_position_target(arm_default, joint_ids=arm_ids)
            obs, _, dones, extras = env.step(policy(obs))
            died = dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
            alive &= ~died
            contact = sensor.data.net_forces_w[:, sensor_foot_ids, :].norm(dim=-1) > CONTACT_N
            air_run = torch.where(~contact, air_run + 1, torch.zeros_like(air_run))
            foot = robot.data.body_pos_w[:, foot_ids, :2]
            if i == w0:
                prev_foot = foot.clone()
            if i >= w0:
                w = alive.float()
                v = torch.stack(
                    (robot.data.root_lin_vel_b[:, 0], robot.data.root_lin_vel_b[:, 1], robot.data.root_ang_vel_w[:, 2]), dim=1
                )
                sum_v += w.unsqueeze(1) * v
                along = (v * direction).sum(1)
                sum_along += w * along
                sum_along2 += w * along**2
                g = robot.data.projected_gravity_b[:, :2].norm(dim=1).clamp(max=1.0)
                sum_tilt2 += w * torch.rad2deg(torch.asin(g)) ** 2
                lifts += w * (air_run == min_air).float().sum(1)
                foot_path += w * (foot - prev_foot).norm(dim=-1).sum(1)
                prev_foot = foot.clone()
                count += w
        c = count.clamp(min=1.0)
        mean_v = sum_v / c.unsqueeze(1)
        jitter = (sum_along2 / c - (sum_along / c) ** 2).clamp(min=0).sqrt()
        err = (mean_v - cmd).abs()
        tol = torch.maximum(torch.full_like(cmd, ABS_TOL), REL_TOL * cmd.abs())
        commanded = cmd.abs() > 0
        ok = alive & (err[:, commanded] < tol[commanded]).all(dim=1)
        if (~commanded).any():
            ok &= (mean_v[:, ~commanded].abs() < ABS_TOL).all(dim=1)
        secs = c * dt
        s = alive
        rows.append({
            "cmd": [vx, vy, wz],
            "survival": alive.float().mean().item(),
            "achieved": mean_v[s].mean(0).tolist() if s.any() else [float("nan")] * 3,
            "ok_pct": ok.float().mean().item(),
            "jitter": jitter[s].mean().item() if s.any() else float("nan"),
            "lifts_per_s": (lifts / secs)[s].mean().item() if s.any() else float("nan"),
            "foot_cm_s": (foot_path * 100 / secs)[s].mean().item() if s.any() else float("nan"),
            "tilt_deg": (sum_tilt2 / c).sqrt()[s].mean().item() if s.any() else float("nan"),
        })

    print(f"\n[lowspeed] {args.label}  checkpoint={os.path.basename(args.checkpoint)}  noise={'on' if args.noise else 'off'}  "
          f"arms={args.arms}  {n} envs x {args.seconds:.0f}s, window {WINDOW_START_S:.0f}s..end")
    print(f"[lowspeed] ok = achieved within max({ABS_TOL}, {int(REL_TOL*100)}% of cmd) on commanded axes, <{ABS_TOL} on others")
    print(f"{'cmd (vx, vy, wz)':>20} {'surv':>5} {'achieved (vx, vy, wz)':>24} {'ok%':>5} {'jitter':>6} {'lift/s':>6} {'foot cm/s':>9} {'tilt':>5}")
    for r in rows:
        c, a = r["cmd"], r["achieved"]
        print(f"({c[0]:5.2f},{c[1]:5.2f},{c[2]:5.2f}) {r['survival']:5.2f}   ({a[0]:6.3f},{a[1]:6.3f},{a[2]:6.3f}) "
              f"{r['ok_pct']*100:4.0f}% {r['jitter']:6.3f} {r['lifts_per_s']:6.2f} {r['foot_cm_s']:9.1f} {r['tilt_deg']:5.2f}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"label": args.label, "checkpoint": args.checkpoint, "noise": args.noise, "arms": args.arms, "rows": rows}, fh, indent=2)
        print(f"[lowspeed] wrote {args.out}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
