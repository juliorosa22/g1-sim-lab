#!/usr/bin/env python3
"""Does a policy turn in place on a yaw-only command and then settle back to a planted stand?

Per yaw rate: stand 3 s -> yaw-only command (vx = vy = 0) for 3 s -> zero command for 9 s, arms at rest, no shoves, no noise.
(A tiny vx, vy with a high yaw rate is the same thing once the controller's 0.05 deadband is applied, so it is not a separate case.)

  turn   (window 4-6 s)   achieved yaw rate, how far the base wandered (cm), foot lifts per minute
  settle                  seconds after the command returned to zero until the base is quiet (< 3 cm/s, < 0.1 rad/s) for 0.5 s
  after  (window 7.5-12)  foot lifts per minute, base displacement (cm), RMS speed (mm/s), max tilt
  planted                 share of envs that survive and, in the "after" window, lift no foot, move < 5 cm and stay under 30 mm/s

    python scripts/evaluate_turn_settle.py --task <Play task> --checkpoint <model.pt> --label v9e [--out file.json]
"""
import argparse
import json
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
parser.add_argument("--yaws", type=float, nargs="+", default=[0.3, 0.5, 1.0, -0.5])
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

T_TURN0, T_TURN1, T_END = 3.0, 6.0, 12.0
TURN_WIN = (4.0, 6.0)
AFTER_WIN = (7.5, 12.0)
CONTACT_N, MIN_AIR_S = 10.0, 0.1
QUIET_SPEED, QUIET_YAW, QUIET_HOLD_S = 0.03, 0.10, 0.5
PLANTED_DRIFT_M, PLANTED_SPEED = 0.05, 0.03


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
    env_cfg.episode_length_s = T_END + 5.0
    env_cfg.events.push_robot = None
    if hasattr(env_cfg.commands, "arm_pose"):
        env_cfg.commands.arm_pose = None
    env_cfg.observations.policy.enable_corruption = False

    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    actor = load_actor(os.path.abspath(args.checkpoint), device)
    base = env.unwrapped
    robot = base.scene["robot"]
    term = base.command_manager.get_term("base_velocity")
    n, dt = args.num_envs, base.step_dt
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    arm_default = robot.data.default_joint_pos[:, arm_ids].clone()
    sensor = base.scene["contact_forces"]
    foot_ids = sensor.find_bodies(".*_ankle_roll_link")[0]
    min_air = max(int(MIN_AIR_S / dt), 1)
    steps = int(T_END / dt)
    quiet_steps = int(QUIET_HOLD_S / dt)

    results = []
    for yaw in args.yaws:
        env.reset()
        obs = env.get_observations()
        alive = torch.ones(n, dtype=torch.bool, device=device)
        air_run = torch.zeros(n, 2, device=device)
        lifts_turn, lifts_after = torch.zeros(n, device=device), torch.zeros(n, device=device)
        wz_acc, cnt_turn = torch.zeros(n, device=device), 0
        sp2_after, cnt_after, tilt_max = torch.zeros(n, device=device), 0, torch.zeros(n, device=device)
        quiet_run = torch.zeros(n, device=device)
        settle = torch.full((n,), T_END - T_TURN1, device=device)
        settled = torch.zeros(n, dtype=torch.bool, device=device)
        xy_turn0 = xy_after0 = xy_turn1 = xy_after1 = None
        for i in range(steps):
            t = i * dt
            cmd = (0.0, 0.0, yaw) if T_TURN0 <= t < T_TURN1 else (0.0, 0.0, 0.0)
            term.vel_command_b[:] = torch.tensor(cmd, device=device)
            robot.set_joint_position_target(arm_default, joint_ids=arm_ids)
            obs, _, dones, extras = env.step(actor(obs["policy"]))
            died = dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
            alive &= ~died
            xy = robot.data.root_pos_w[:, :2]
            contact = sensor.data.net_forces_w[:, foot_ids, :].norm(dim=-1) > CONTACT_N
            air_run = torch.where(~contact, air_run + 1, torch.zeros_like(air_run))
            lift = (air_run == min_air).float().sum(1)
            speed = robot.data.root_lin_vel_w[:, :2].norm(dim=1)
            wz = robot.data.root_ang_vel_b[:, 2]
            if TURN_WIN[0] <= t < TURN_WIN[1]:
                if xy_turn0 is None:
                    xy_turn0 = xy.clone()
                xy_turn1 = xy.clone()
                lifts_turn += lift
                wz_acc += wz
                cnt_turn += 1
            if t >= T_TURN1:  # settle time
                quiet = (speed < QUIET_SPEED) & (wz.abs() < QUIET_YAW)
                quiet_run = torch.where(quiet, quiet_run + 1, torch.zeros_like(quiet_run))
                newly = (quiet_run >= quiet_steps) & ~settled
                settle[newly] = (t - T_TURN1) - QUIET_HOLD_S + dt
                settled |= newly
            if AFTER_WIN[0] <= t < AFTER_WIN[1]:
                if xy_after0 is None:
                    xy_after0 = xy.clone()
                xy_after1 = xy.clone()
                lifts_after += lift
                sp2_after += speed ** 2
                cnt_after += 1
                g = robot.data.projected_gravity_b[:, :2].norm(dim=1).clamp(max=1.0)
                tilt_max = torch.maximum(tilt_max, torch.rad2deg(torch.asin(g)))
        turn_min = TURN_WIN[1] - TURN_WIN[0]
        after_min = AFTER_WIN[1] - AFTER_WIN[0]
        drift_turn = (xy_turn1 - xy_turn0).norm(dim=1) * 100
        drift_after = (xy_after1 - xy_after0).norm(dim=1)
        speed_after = (sp2_after / max(cnt_after, 1)).sqrt()
        planted = alive & (lifts_after == 0) & (drift_after < PLANTED_DRIFT_M) & (speed_after < PLANTED_SPEED)
        a = alive
        row = {
            "yaw": yaw, "survival": a.float().mean().item(),
            "achieved_wz": (wz_acc[a] / max(cnt_turn, 1)).mean().item(),
            "turn_drift_cm": drift_turn[a].mean().item(), "turn_lifts_per_min": (lifts_turn[a] / turn_min * 60).mean().item(),
            "settle_s_median": settle[a].median().item(), "settled_share": settled[a].float().mean().item(),
            "after_lifts_per_min": (lifts_after[a] / after_min * 60).mean().item(),
            "after_drift_cm": (drift_after[a] * 100).mean().item(), "after_speed_mms": (speed_after[a] * 1000).mean().item(),
            "after_tilt_max": tilt_max[a].mean().item(), "planted": planted.float().mean().item(),
        }
        results.append(row)

    print(f"\n[turn-settle] {args.label}  checkpoint={os.path.basename(args.checkpoint)}  {n} envs; stand 3 s, yaw-only 3 s, zero command 9 s, arms at rest")
    print(f"{'yaw':>6} {'surv':>5} | {'wz':>6} {'wander cm':>9} {'lift/min':>8} | {'settle s':>8} {'settled':>7} | {'lift/min':>8} {'drift cm':>8} {'mm/s':>6} {'tilt max':>8} | {'planted':>7}")
    for r in results:
        print(f"{r['yaw']:6.2f} {r['survival']:5.2f} | {r['achieved_wz']:6.2f} {r['turn_drift_cm']:9.1f} {r['turn_lifts_per_min']:8.1f} | "
              f"{r['settle_s_median']:8.2f} {r['settled_share']*100:6.0f}% | {r['after_lifts_per_min']:8.1f} {r['after_drift_cm']:8.1f} "
              f"{r['after_speed_mms']:6.1f} {r['after_tilt_max']:8.1f} | {r['planted']*100:6.0f}%", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"label": args.label, "checkpoint": args.checkpoint, "results": results}, f, indent=1)
        print(f"[turn-settle] wrote {args.out}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
