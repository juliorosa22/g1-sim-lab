#!/usr/bin/env python3
"""How much extra pipeline delay and how many late ticks can a policy tolerate?  (Request 1 of robot_deploy/docs/sim_requests_from_hardware.md)

What v11 was trained with (g1_rl/robot_cfg.py): the legs and waist yaw use a delayed-PD actuator. The joint position targets reach the PD law lagged by a
whole number of PHYSICS steps (5 ms each), drawn uniformly from 0..4 steps (0-20 ms) once per environment at every reset and fixed for that episode
(not per step). The action is also held for one control tick (20 ms) between policy calls. There is NO observation delay or noise-age in training
(observation noise yes, delay no). Arms are held by an implicit PD (no delay).

Sweeps (each on top of the trained model):
  A  extra ACTION delay   E ms (0, 10, 20, 30, 40, 60) = E/5 more physics steps added to every env's trained 0-4 step lag (lag in [E/5, E/5 + 4] steps)
  B  extra OBSERVATION age a ms (0, 10, 20, 30, 40, 60): the sensor part of the observation (gyro, gravity, joint pos and vel) is the linear interpolation
     between the observations of the last control ticks, a ms old; the command, the previous action and the gait clock stay current
  C  LATE ticks: at each tick, with probability p (1%) per env, the policy "stalls" for k consecutive ticks (k = 1, 2, 3, 5 = 20-100 ms): the last
     applied action is re-sent (PD keeps the last target and gains)
Scenarios (3 seeds x 64 envs each): stand at zero command 30 s; turn in place at 0.3 and 0.5 rad/s (stand 3 s, ramp 2 rad/s^2, hold 6 s, then zero
for 6 s); stand with a 0.3 m/s shove (random direction) at 4, 9 and 14 s (20 s).

    python scripts/evaluate_delay.py --task G1-Walk-Flat-Inspire-V11-Play-v0 --checkpoint <model.pt> --out logs/eval/delay_v11.json [--noise] [--quick]
"""
import argparse
import json
import math
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="G1-Walk-Flat-Inspire-V11-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seeds", type=int, default=3)
parser.add_argument("--noise", action="store_true")
parser.add_argument("--quick", action="store_true", help="1 seed, shorter stand: a smoke test")
parser.add_argument("--only", default=None, help="comma list of settings to run, e.g. action:0,obs:20,stall:3 (default: the whole sweep)")
parser.add_argument("--out", default=None)
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
from g1_rl.v8_env_cfg import CONTROLLED

PHYS_MS = 5.0
TRAINED_MAX_STEPS = 4
MAX_EXTRA_MS = 60
SENSOR_IDX = list(range(0, 6)) + list(range(9, 63))  # gyro, gravity, joint pos, joint vel in the 78-value observation
SETTINGS = (
    [("action", e) for e in (0, 10, 20, 30, 40, 60)] + [("obs", a) for a in (10, 20, 30, 40, 60)] + [("stall", k) for k in (1, 2, 3, 5)]
)
STALL_P = 0.01
YAW_RAMP, TURN_HOLD, TURN_AFTER, STAND_PRE = 2.0, 6.0, 6.0, 3.0
GROUPS = {"hip_pitch": ("left_hip_pitch_joint", "right_hip_pitch_joint"), "knee": ("left_knee_joint", "right_knee_joint"),
          "ankle_pitch": ("left_ankle_pitch_joint", "right_ankle_pitch_joint"), "waist_yaw": ("waist_yaw_joint",)}
STILL_SPEED, STILL_YAW, STILL_HOLD_S = 0.03, 0.10, 0.5


def scenarios(quick):
    return [("stand", 0.0, 10.0 if quick else 30.0, ()), ("turn0.3", 0.3, STAND_PRE + 0.3 / YAW_RAMP + TURN_HOLD + TURN_AFTER, ()),
            ("turn0.5", 0.5, STAND_PRE + 0.5 / YAW_RAMP + TURN_HOLD + TURN_AFTER, ()), ("push0.3", 0.0, 20.0, (4.0, 9.0, 14.0))]


@torch.inference_mode()
def main():
    device = "cuda:0"
    seeds = 1 if args.quick else args.seeds
    group = args.num_envs  # envs per group; the groups ("seeds") run side by side in ONE batch of group * seeds envs, which is much cheaper than sequential runs
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=group * seeds)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.heading_command = False
    if hasattr(cmd_cfg, "mixed_sampling"):
        cmd_cfg.mixed_sampling = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.episode_length_s = 120.0
    env_cfg.events.push_robot = None
    if hasattr(env_cfg.commands, "arm_pose"):
        env_cfg.commands.arm_pose = None
    env_cfg.observations.policy.enable_corruption = bool(args.noise)
    steps_cap = TRAINED_MAX_STEPS + int(MAX_EXTRA_MS / PHYS_MS)
    delayed = [name for name, a in env_cfg.scene.robot.actuators.items() if hasattr(a, "max_delay")]
    for name in delayed:  # room for the extra delay in the delay buffers (the trained range stays 0..4 steps)
        env_cfg.scene.robot.actuators[name].max_delay = steps_cap
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    actor = load_actor(os.path.abspath(args.checkpoint), device)
    base = env.unwrapped
    robot = base.scene["robot"]
    term = base.command_manager.get_term("base_velocity")
    sensor = base.scene["contact_forces"]
    foot_ids = sensor.find_bodies(".*_ankle_roll_link")[0]
    n, dt = group * seeds, base.step_dt
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    arm_default = robot.data.default_joint_pos[:, arm_ids].clone()
    jids = robot.find_joints(list(CONTROLLED), preserve_order=True)[0]
    col = {nm: i for i, nm in enumerate(CONTROLLED)}
    print(f"[delay] delayed actuators: {delayed}; trained lag 0..{TRAINED_MAX_STEPS} physics steps ({PHYS_MS:.0f} ms each); buffers sized for {steps_cap} steps", flush=True)

    def set_lags(extra_steps):
        for name in delayed:
            act = robot.actuators[name]
            lags = torch.randint(0, TRAINED_MAX_STEPS + 1, (n,), device=device, dtype=torch.int) + int(extra_steps)
            for buf in (act.positions_delay_buffer, act.velocities_delay_buffer, act.efforts_delay_buffer):
                buf.set_time_lag(lags)

    def run(kind, value, scen):
        name, yaw, duration, shoves = scen
        torch.manual_seed(17)
        env.reset()
        set_lags(value / PHYS_MS if kind == "action" else 0)
        obs = env.get_observations()["policy"]
        hist = [obs.clone() for _ in range(5)]
        age_ticks = (value / (dt * 1000.0)) if kind == "obs" else 0.0
        i0, frac = int(age_ticks), age_ticks - int(age_ticks)
        steps = int(duration / dt)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        stall_left = torch.zeros(n, dtype=torch.long, device=device)
        prev_action = torch.zeros(n, 13, device=device)
        cmd, wz_acc, wz_cnt = 0.0, torch.zeros(n, device=device), 0
        turn_end = STAND_PRE + yaw / YAW_RAMP + TURN_HOLD if yaw else None
        win0 = STAND_PRE
        tilt2, tilt_max, da2, cnt = torch.zeros(n, device=device), torch.zeros(n, device=device), torch.zeros(n, device=device), 0
        peak = {g: torch.zeros(n, device=device) for g in GROUPS}
        sq = {g: torch.zeros(n, device=device) for g in ("hip_pitch", "ankle_pitch")}
        xy_a = xy_b = None
        quiet_run = torch.zeros(n, device=device)
        settle = torch.full((n,), TURN_AFTER, device=device)
        settled = torch.zeros(n, dtype=torch.bool, device=device)
        shove_steps = {int(round(t / dt)) for t in shoves}
        turn_start_xy = turn_end_xy = None
        start_step = int(round((turn_end - 3.0) / dt)) if turn_end is not None else -1
        end_step = int(round(turn_end / dt)) if turn_end is not None else -1
        for i in range(steps):
            t = i * dt
            goal = yaw if (yaw and STAND_PRE <= t < turn_end) else 0.0
            cmd += max(-YAW_RAMP * dt, min(YAW_RAMP * dt, goal - cmd))
            term.vel_command_b[:] = torch.tensor([0.0, 0.0, cmd], device=device)
            robot.set_joint_position_target(arm_default, joint_ids=arm_ids)
            if i in shove_steps:
                ang = torch.rand(n, device=device) * 2 * math.pi
                vel = robot.data.root_vel_w.clone()
                vel[:, 0] += 0.3 * ang.cos()
                vel[:, 1] += 0.3 * ang.sin()
                robot.write_root_velocity_to_sim(vel)
            # observation as the policy gets it
            o = obs
            if kind == "obs" and value > 0:
                j = min(i0 + 1, len(hist) - 1)
                delayed_obs = (1.0 - frac) * hist[min(i0, len(hist) - 1)] + frac * hist[j]
                o = obs.clone()
                o[:, SENSOR_IDX] = delayed_obs[:, SENSOR_IDX]
            action = actor(o)
            if kind == "stall":
                start = (stall_left == 0) & (torch.rand(n, device=device) < STALL_P)
                stall_left[start] = value
                held = stall_left > 0
                action = torch.where(held.unsqueeze(1), prev_action, action)
                stall_left = (stall_left - 1).clamp_min(0)
            obs, _, dones, extras = env.step(action)
            obs = obs["policy"]
            hist = [obs.clone()] + hist[:-1]
            alive &= ~(dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool())
            w = alive.float()
            if t >= win0:
                g = robot.data.projected_gravity_b[:, :2].norm(dim=1).clamp(max=1.0)
                tilt = torch.rad2deg(torch.asin(g))
                tilt2 += (tilt ** 2) * w
                tilt_max = torch.maximum(tilt_max, tilt * w)
                da2 += (action - prev_action).pow(2).mean(dim=1) * w
                cnt += 1
                tq = robot.data.applied_torque[:, jids]
                for gname, names in GROUPS.items():
                    v = torch.stack([tq[:, col[nm]].abs() for nm in names]).amax(dim=0)
                    peak[gname] = torch.maximum(peak[gname], v * w)
                    if gname in sq:
                        sq[gname] += torch.stack([tq[:, col[nm]] for nm in names]).pow(2).mean(dim=0) * w
                if xy_a is None:
                    xy_a = robot.data.root_pos_w[:, :2].clone()
                xy_b = robot.data.root_pos_w[:, :2].clone()
            if turn_end is not None:
                if t >= turn_end - 3.0 and t < turn_end:
                    wz_acc += robot.data.root_ang_vel_b[:, 2] * w
                    wz_cnt += 1
                if i == start_step:
                    turn_start_xy = robot.data.root_pos_w[:, :2].clone()
                if i == end_step:
                    turn_end_xy = robot.data.root_pos_w[:, :2].clone()
                if t >= turn_end:
                    quiet = (robot.data.root_lin_vel_w[:, :2].norm(dim=1) < STILL_SPEED) & (robot.data.root_ang_vel_b[:, 2].abs() < STILL_YAW)
                    quiet_run = torch.where(quiet, quiet_run + 1, torch.zeros_like(quiet_run))
                    newly = (quiet_run >= int(STILL_HOLD_S / dt)) & ~settled
                    settle[newly] = (t - turn_end) - STILL_HOLD_S + dt
                    settled |= newly
            prev_action = action
        drift = (xy_b - xy_a).norm(dim=1) * 100
        outs = []
        for gi in range(seeds):  # one summary per group of envs (= per "seed")
            sl = slice(gi * group, (gi + 1) * group)
            a = alive[sl]
            ok = bool(a.any())
            nan = float("nan")
            res = {"fall_rate": float(1.0 - a.float().mean()),
                   "tilt_rms": float((tilt2[sl][a] / cnt).sqrt().mean()) if ok else nan,
                   "tilt_max": float(tilt_max[sl][a].mean()) if ok else nan, "drift_cm": float(drift[sl][a].mean()) if ok else nan,
                   "chatter": float((da2[sl][a] / cnt).sqrt().mean()) if ok else nan}
            for gname in GROUPS:
                res[f"peak_tq_{gname}"] = float(peak[gname][sl][a].mean()) if ok else nan
                res[f"peak_tq_{gname}_max"] = float(peak[gname][sl][a].max()) if ok else nan
            for gname in sq:
                res[f"rms_tq_{gname}"] = float((sq[gname][sl][a] / cnt).sqrt().mean()) if ok else nan
            if turn_end is not None:
                res["wz_reached"] = float((wz_acc[sl][a] / max(wz_cnt, 1)).mean()) if ok else nan
                res["turn_wander_cm"] = float((turn_end_xy - turn_start_xy).norm(dim=1)[sl][a].mean() * 100) if ok else nan
                res["settle_s"] = float(settle[sl][a].median()) if ok else nan
                res["settled_share"] = float(settled[sl][a].float().mean()) if ok else nan
            outs.append(res)
        return outs

    settings = SETTINGS if not args.only else [(k.split(":")[0], int(k.split(":")[1])) for k in args.only.split(",")]
    out = {}
    scen_list = scenarios(args.quick)
    for kind, value in settings:
        key = f"{kind}:{value}"
        out[key] = {}
        for scen in scen_list:
            per_seed = run(kind, value, scen)
            avg = {k: float(sum(r[k] for r in per_seed) / len(per_seed)) for k in per_seed[0]}
            avg["_per_seed_fall_rate"] = [r["fall_rate"] for r in per_seed]
            avg["_seed_std"] = {k: float(torch.tensor([r[k] for r in per_seed]).std()) if seeds > 1 else 0.0 for k in per_seed[0]}
            out[key][scen[0]] = avg
        print(f"[delay] {key:10s} " + "  ".join(f"{s[0]}: fall {out[key][s[0]]['fall_rate']:.2f} tilt {out[key][s[0]]['tilt_rms']:.1f}/{out[key][s[0]]['tilt_max']:.1f} "
                                                  f"drift {out[key][s[0]]['drift_cm']:.1f}" for s in scen_list), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "noise": args.noise, "envs_per_seed_group": group, "seeds": seeds, "results": out}, f, indent=1)
        print(f"[delay] wrote {args.out}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
