#!/usr/bin/env python3
"""Walking quality and stopping: how smoothly does a walker walk, and how cleanly does it stop when the command goes to zero?

Per walking speed (forward command): walk 6 s, then the command goes to zero (ramped down at the deployed controller's 1 m/s^2 unless
--no-ramp), then 8 s of zero command. Arms at rest, no shoves, no noise.

  walk (window 3-6 s)     achieved speed, base height, tilt RMS, foot lifts/min
  smoothness (same)       action change per step (RMS of a_t - a_{t-1}), leg joint acceleration RMS (rad/s^2), leg torque RMS,
                          foot swing height (p90 of the ankle-roll link z while off the ground, m)
  stop (from the instant the command reaches zero)
                          peak tilt (deg), time until the base is under 3 cm/s for 0.5 s, speed 1.5 s after the stop (cm/s),
                          foot lifts per minute in the window 3-8 s after the stop, base displacement in that window (cm)
  settled                 share of envs that survive and, in the window 3-8 s after the stop, lift no foot, move < 5 cm and stay under 30 mm/s

    python scripts/evaluate_stop.py --task <Play task> --checkpoint <model.pt> --label v12 [--speeds 0.5 0.8] [--out file.json]
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
parser.add_argument("--speeds", type=float, nargs="+", default=[0.5, 0.8])
parser.add_argument("--no-ramp", action="store_true", help="drop the command to zero at once instead of ramping it down")
parser.add_argument("--noise", action="store_true", help="enable observation noise (the Play tasks disable it)")
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

T_WALK, WALK_WIN, AFTER_T = 6.0, (3.0, 6.0), 8.0
STOP_WIN = (3.0, 8.0)  # seconds after the command reaches zero
RAMP_DECEL = 1.0  # m/s^2
CONTACT_N, MIN_AIR_S = 10.0, 0.1
QUIET_SPEED, QUIET_HOLD_S = 0.03, 0.5
SETTLED_DRIFT_M, SETTLED_SPEED = 0.05, 0.03


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
    max_speed = max(args.speeds)
    env_cfg.episode_length_s = T_WALK + max_speed / RAMP_DECEL + AFTER_T + 5.0
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
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    arm_default = robot.data.default_joint_pos[:, arm_ids].clone()
    leg_ids = robot.find_joints([".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])[0]
    sensor = base.scene["contact_forces"]
    sensor_foot_ids = sensor.find_bodies(".*_ankle_roll_link")[0]
    foot_ids = robot.find_bodies(".*_ankle_roll_link")[0]
    min_air = max(int(MIN_AIR_S / dt), 1)
    quiet_steps = int(QUIET_HOLD_S / dt)

    results = []
    for speed in args.speeds:
        t_ramp = 0.0 if args.no_ramp else speed / RAMP_DECEL
        t_stop = T_WALK + t_ramp  # the command reaches zero
        steps = int((t_stop + AFTER_T) / dt)
        env.reset()
        obs = env.get_observations()
        alive = torch.ones(n, dtype=torch.bool, device=device)
        air_run = torch.zeros(n, 2, device=device)
        prev_action, prev_qd = None, None
        acc = {k: torch.zeros(n, device=device) for k in ("da2", "qdd2", "tq2", "tilt2", "sp", "z", "lifts_walk", "lifts_after", "sp2_after", "tilt_peak")}
        cnt_walk = cnt_after = 0
        swing_z = []
        quiet_run = torch.zeros(n, device=device)
        stop_time = torch.full((n,), AFTER_T, device=device)
        stopped = torch.zeros(n, dtype=torch.bool, device=device)
        sp_at_1p5 = torch.zeros(n, device=device)
        xy0 = xy1 = None
        for i in range(steps):
            t = i * dt
            vx = speed if t < T_WALK else max(speed - RAMP_DECEL * (t - T_WALK), 0.0)
            if args.no_ramp and t >= T_WALK:
                vx = 0.0
            term.vel_command_b[:] = torch.tensor([vx, 0.0, 0.0], device=device)
            robot.set_joint_position_target(arm_default, joint_ids=arm_ids)
            action = actor(obs["policy"])
            obs, _, dones, extras = env.step(action)
            alive &= ~(dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool())
            sp = robot.data.root_lin_vel_w[:, :2].norm(dim=1)
            g = robot.data.projected_gravity_b[:, :2].norm(dim=1).clamp(max=1.0)
            tilt = torch.rad2deg(torch.asin(g))
            contact = sensor.data.net_forces_w[:, sensor_foot_ids, :].norm(dim=-1) > CONTACT_N
            air_run = torch.where(~contact, air_run + 1, torch.zeros_like(air_run))
            lift = (air_run == min_air).float().sum(1)
            qd = robot.data.joint_vel[:, leg_ids]
            da = (action - prev_action).pow(2).mean(dim=1) if prev_action is not None else torch.zeros(n, device=device)
            qdd = ((qd - prev_qd) / dt).pow(2).mean(dim=1) if prev_qd is not None else torch.zeros(n, device=device)
            prev_action, prev_qd = action.clone(), qd.clone()
            if WALK_WIN[0] <= t < WALK_WIN[1]:
                acc["da2"] += da
                acc["qdd2"] += qdd
                acc["tq2"] += robot.data.applied_torque[:, leg_ids].pow(2).mean(dim=1)
                acc["tilt2"] += tilt ** 2
                acc["sp"] += sp
                acc["z"] += robot.data.root_pos_w[:, 2]
                acc["lifts_walk"] += lift
                cnt_walk += 1
                fz = robot.data.body_pos_w[:, foot_ids, 2]
                swing_z.append(fz[~contact])
            if t >= t_stop:
                acc["tilt_peak"] = torch.maximum(acc["tilt_peak"], tilt)
                quiet = sp < QUIET_SPEED
                quiet_run = torch.where(quiet, quiet_run + 1, torch.zeros_like(quiet_run))
                newly = (quiet_run >= quiet_steps) & ~stopped
                stop_time[newly] = (t - t_stop) - QUIET_HOLD_S + dt
                stopped |= newly
                if abs((t - t_stop) - 1.5) < dt / 2:
                    sp_at_1p5 = sp.clone()
                ts = t - t_stop
                if STOP_WIN[0] <= ts < STOP_WIN[1]:
                    if xy0 is None:
                        xy0 = robot.data.root_pos_w[:, :2].clone()
                    xy1 = robot.data.root_pos_w[:, :2].clone()
                    acc["lifts_after"] += lift
                    acc["sp2_after"] += sp ** 2
                    cnt_after += 1
        a = alive
        walk_s, after_s = WALK_WIN[1] - WALK_WIN[0], STOP_WIN[1] - STOP_WIN[0]
        drift = (xy1 - xy0).norm(dim=1)
        sp_after = (acc["sp2_after"] / max(cnt_after, 1)).sqrt()
        settled = alive & (acc["lifts_after"] == 0) & (drift < SETTLED_DRIFT_M) & (sp_after < SETTLED_SPEED)
        sz = torch.cat(swing_z) if swing_z else torch.zeros(1, device=device)
        row = {
            "speed_cmd": speed, "survival": a.float().mean().item(),
            "speed": (acc["sp"][a] / cnt_walk).mean().item(), "z": (acc["z"][a] / cnt_walk).mean().item(),
            "tilt_rms": (acc["tilt2"][a] / cnt_walk).sqrt().mean().item(), "lifts_per_min_walk": (acc["lifts_walk"][a] / walk_s * 60).mean().item(),
            "action_change_rms": (acc["da2"][a] / cnt_walk).sqrt().mean().item(), "joint_acc_rms": (acc["qdd2"][a] / cnt_walk).sqrt().mean().item(),
            "torque_rms": (acc["tq2"][a] / cnt_walk).sqrt().mean().item(), "swing_z_p90": float(torch.quantile(sz, 0.9)),
            "stop_peak_tilt": acc["tilt_peak"][a].mean().item(), "stop_peak_tilt_max": acc["tilt_peak"][a].max().item(),
            "stop_time_median": stop_time[a].median().item(), "stopped_share": stopped[a].float().mean().item(),
            "speed_at_1p5s_cms": sp_at_1p5[a].mean().item() * 100,
            "after_lifts_per_min": (acc["lifts_after"][a] / after_s * 60).mean().item(), "after_drift_cm": drift[a].mean().item() * 100,
            "settled": settled.float().mean().item(),
        }
        results.append(row)

    print(f"\n[stop] {args.label}  checkpoint={os.path.basename(args.checkpoint)}  {n} envs; walk 6 s, command {'dropped' if args.no_ramp else 'ramped'} to zero, 8 s after")
    print(f"{'cmd':>5} {'surv':>5} | {'speed':>5} {'z':>5} {'tilt':>5} {'lift/m':>6} | {'da rms':>6} {'qdd':>7} {'tq':>5} {'swing z':>7} | "
          f"{'peak tilt':>9} {'(max)':>6} {'stop s':>6} {'cm/s@1.5':>8} {'lift/m':>6} {'drift':>6} | {'settled':>7}")
    for r in results:
        print(f"{r['speed_cmd']:5.2f} {r['survival']:5.2f} | {r['speed']:5.2f} {r['z']:5.3f} {r['tilt_rms']:5.1f} {r['lifts_per_min_walk']:6.0f} | "
              f"{r['action_change_rms']:6.3f} {r['joint_acc_rms']:7.0f} {r['torque_rms']:5.1f} {r['swing_z_p90']:7.3f} | "
              f"{r['stop_peak_tilt']:9.1f} {r['stop_peak_tilt_max']:6.1f} {r['stop_time_median']:6.2f} {r['speed_at_1p5s_cms']:8.1f} "
              f"{r['after_lifts_per_min']:6.1f} {r['after_drift_cm']:6.1f} | {r['settled']*100:6.0f}%", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"label": args.label, "checkpoint": args.checkpoint, "ramped": not args.no_ramp, "results": results}, f, indent=1)
        print(f"[stop] wrote {args.out}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
