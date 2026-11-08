#!/usr/bin/env python3
"""Which velocity commands make the robot fall?  A grid over (vx, vy, wz), every cell run with several envs.

Per cell: stand 2 s, then the command goal (after the controller's 0.05 deadband) is applied and held. Modes:
  ramp     the goal is reached through the deployed ramp (1 m/s^2, 2 rad/s^2), held until 8 s
  step     the goal is applied at once, no ramp (what a glitching or missing ramp would do), held until 8 s
  reverse  ramp to the goal, hold until 8 s, then the goal flips sign AT ONCE (no ramp), hold 6 s more
Arms at rest, no shoves (add them with --shove), observation noise with --noise, extra pipeline delay with --extra-delay-ms.

Per cell (envs pooled): fall rate (a termination before the end), time to fall, max tilt, foot lifts per minute in the last 3 s of the hold, and the tracking
error of the body-frame velocity over the last 3 s of the first hold (|vx,vy - goal| in m/s, |wz - goal| in rad/s).

    python -u scripts/evaluate_command_grid.py --task <Play task> --checkpoint <model.pt> --preset v11 --mode ramp --out file.json [--noise]
"""
import argparse
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher

PRESETS = {  # beyond the trained ranges on purpose: v11 trained vx (-0.3, 0.6) vy +-0.3 wz +-1; v12 vx (-0.3, 1.0)
    "v11": dict(vx=[-0.5, -0.3, 0.0, 0.3, 0.6, 0.8], vy=[-0.5, -0.3, 0.0, 0.3, 0.5], wz=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]),
    "v12": dict(vx=[-0.5, -0.3, 0.0, 0.3, 0.6, 1.0, 1.2, 1.4], vy=[-0.3, 0.0, 0.3], wz=[-1.0, -0.5, 0.0, 0.5, 1.0]),
}

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--label", default="policy")
parser.add_argument("--preset", choices=list(PRESETS), default="v11")
parser.add_argument("--vx", type=float, nargs="+", default=None)
parser.add_argument("--vy", type=float, nargs="+", default=None)
parser.add_argument("--wz", type=float, nargs="+", default=None)
parser.add_argument("--mode", choices=["ramp", "step", "reverse"], default="ramp")
parser.add_argument("--reps", type=int, default=8, help="envs per cell")
parser.add_argument("--num_envs", type=int, default=1024, help="envs per wave (cells are run in waves)")
parser.add_argument("--noise", action="store_true")
parser.add_argument("--shove", action="store_true", help="a 0.3 m/s shove at 4 s and 6 s of each run")
parser.add_argument("--extra-delay-ms", type=float, default=0.0, help="extra action delay on top of the trained 0-20 ms")
parser.add_argument("--quick", action="store_true", help="a small grid, 2 reps: a smoke test")
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

T_STAND, T_HOLD1, T_HOLD2 = 2.0, 8.0, 14.0
RAMP_LIN, RAMP_YAW, DEADBAND = 1.0, 2.0, 0.05
PHYS_MS, TRAINED_MAX_STEPS = 5.0, 4
CONTACT_N, MIN_AIR_S = 10.0, 0.1
SHOVE_TIMES, SHOVE_SPEED = (4.0, 6.0), 0.3


@torch.inference_mode()
def main():
    device = "cuda:0"
    grid = dict(PRESETS[args.preset])
    for k in ("vx", "vy", "wz"):
        if getattr(args, k) is not None:
            grid[k] = getattr(args, k)
    reps = args.reps
    if args.quick:
        grid, reps = dict(vx=[0.0, 0.6], vy=[0.0], wz=[0.0, 1.0]), 2
    cells = [(vx, vy, wz) for wz in grid["wz"] for vx in grid["vx"] for vy in grid["vy"]]
    per_wave = max(1, args.num_envs // reps)
    n = per_wave * reps
    t_end = T_HOLD2 if args.mode == "reverse" else T_HOLD1

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=n)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.heading_command = False
    if hasattr(cmd_cfg, "mixed_sampling"):
        cmd_cfg.mixed_sampling = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.episode_length_s = t_end + 10.0
    env_cfg.events.push_robot = None
    if hasattr(env_cfg.commands, "arm_pose"):
        env_cfg.commands.arm_pose = None
    env_cfg.observations.policy.enable_corruption = bool(args.noise)
    extra_steps = int(round(args.extra_delay_ms / PHYS_MS))
    delayed = [name for name, a in env_cfg.scene.robot.actuators.items() if hasattr(a, "max_delay")]
    for name in delayed:
        env_cfg.scene.robot.actuators[name].max_delay = TRAINED_MAX_STEPS + extra_steps
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    actor = load_actor(os.path.abspath(args.checkpoint), device)
    base = env.unwrapped
    robot = base.scene["robot"]
    term = base.command_manager.get_term("base_velocity")
    sensor = base.scene["contact_forces"]
    foot_ids = sensor.find_bodies(".*_ankle_roll_link")[0]
    dt = base.step_dt
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    arm_default = robot.data.default_joint_pos[:, arm_ids].clone()
    min_air = max(int(MIN_AIR_S / dt), 1)
    print(f"[grid] {args.label} {os.path.basename(args.checkpoint)} mode={args.mode} noise={args.noise} shove={args.shove} extra_delay={args.extra_delay_ms}ms "
          f"cells={len(cells)} reps={reps} envs/wave={n}", flush=True)

    def deadband(g):
        g = g.clone()
        small = g[:, :2].norm(dim=1) < DEADBAND
        g[small, :2] = 0.0
        g[g[:, 2].abs() < DEADBAND, 2] = 0.0
        return g

    def run_wave(wave_cells):
        k = len(wave_cells)
        goal = torch.zeros(n, 3, device=device)
        for ci, c in enumerate(wave_cells):
            goal[ci * reps:(ci + 1) * reps] = torch.tensor(c, device=device)
        goal = deadband(goal)
        used = torch.zeros(n, dtype=torch.bool, device=device)
        used[: k * reps] = True
        for name in delayed:
            act = robot.actuators[name]
            lags = torch.randint(0, TRAINED_MAX_STEPS + 1, (n,), device=device, dtype=torch.int) + extra_steps
            for buf in (act.positions_delay_buffer, act.velocities_delay_buffer, act.efforts_delay_buffer):
                buf.set_time_lag(lags)
        env.reset()
        obs = env.get_observations()
        cur = torch.zeros(n, 3, device=device)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        fall_t = torch.full((n,), float("nan"), device=device)
        tilt_max = torch.zeros(n, device=device)
        air_run = torch.zeros(n, 2, device=device)
        lifts = torch.zeros(n, device=device)
        lift_win = (t_end - 3.0, t_end)
        trk_win = (T_HOLD1 - 3.0, T_HOLD1)
        vel_acc, cnt_trk = torch.zeros(n, 3, device=device), 0
        shove_i = 0
        for i in range(int(t_end / dt)):
            t = i * dt
            if t < T_STAND:
                target = torch.zeros_like(goal)
            elif args.mode == "reverse" and t >= T_HOLD1:
                target = -goal
            else:
                target = goal
            if args.mode == "step" or (args.mode == "reverse" and t >= T_HOLD1):
                cur = target.clone()
            else:
                d = target[:, :2] - cur[:, :2]
                nrm = d.norm(dim=1, keepdim=True).clamp_min(1e-9)
                d = d * torch.clamp(RAMP_LIN * dt / nrm, max=1.0)
                cur[:, :2] += d
                cur[:, 2] += torch.clamp(target[:, 2] - cur[:, 2], -RAMP_YAW * dt, RAMP_YAW * dt)
            term.vel_command_b[:] = cur
            if args.shove and shove_i < len(SHOVE_TIMES) and t >= SHOVE_TIMES[shove_i]:
                ang = torch.rand(n, device=device) * 2 * math.pi
                v = robot.data.root_lin_vel_w.clone()
                v[:, 0] += SHOVE_SPEED * torch.cos(ang)
                v[:, 1] += SHOVE_SPEED * torch.sin(ang)
                robot.write_root_velocity_to_sim(torch.cat([v, robot.data.root_ang_vel_w], dim=1))
                shove_i += 1
            robot.set_joint_position_target(arm_default, joint_ids=arm_ids)
            obs, _, dones, extras = env.step(actor(obs["policy"]))
            died = dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
            newly = died & alive
            fall_t[newly] = t
            alive &= ~died
            g = robot.data.projected_gravity_b[:, :2].norm(dim=1).clamp(max=1.0)
            tilt_max = torch.where(alive, torch.maximum(tilt_max, torch.rad2deg(torch.asin(g))), tilt_max)
            contact = sensor.data.net_forces_w[:, foot_ids, :].norm(dim=-1) > CONTACT_N
            air_run = torch.where(~contact, air_run + 1, torch.zeros_like(air_run))
            if lift_win[0] <= t < lift_win[1]:
                lifts += (air_run == min_air).float().sum(1)
            if trk_win[0] <= t < trk_win[1]:
                vel_acc += torch.cat([robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], dim=1)
                cnt_trk += 1
        vel = vel_acc / max(cnt_trk, 1)
        out = []
        for ci, c in enumerate(wave_cells):
            sl = slice(ci * reps, (ci + 1) * reps)
            a = alive[sl]
            fell = ~a
            gl = goal[sl]
            err_lin = (vel[sl][:, :2] - gl[:, :2]).norm(dim=1)
            err_yaw = (vel[sl][:, 2] - gl[:, 2]).abs()
            ok = a
            out.append({
                "vx": c[0], "vy": c[1], "wz": c[2], "fall_rate": float(fell.float().mean()),
                "fall_time_s": float(fall_t[sl][fell].mean()) if fell.any() else None,
                "tilt_max_deg": float(tilt_max[sl].max()), "tilt_mean_deg": float(tilt_max[sl].mean()),
                "lifts_per_min": float((lifts[sl][ok] / 3.0 * 60).mean()) if ok.any() else None,
                "err_lin": float(err_lin[ok].mean()) if ok.any() else None, "err_yaw": float(err_yaw[ok].mean()) if ok.any() else None,
            })
        return out

    results = []
    for w in range(0, len(cells), per_wave):
        results += run_wave(cells[w:w + per_wave])
        print(f"[grid] {len(results)}/{len(cells)} cells", flush=True)

    print(f"\n[grid] fall rate % (rows vx, columns vy), one table per wz; '*' = survives but tracking error > 0.25 m/s or 0.4 rad/s")
    by = {(r['vx'], r['vy'], r['wz']): r for r in results}
    for wz in grid["wz"]:
        print(f"\nwz = {wz:+.2f} rad/s" + "".join(f"{'vy=%+.1f' % vy:>9}" for vy in grid["vy"]))
        for vx in grid["vx"]:
            row = f"  vx={vx:+.1f}      "
            for vy in grid["vy"]:
                r = by[(vx, vy, wz)]
                bad = (r["err_lin"] or 0) > 0.25 or (r["err_yaw"] or 0) > 0.4
                row += f"{r['fall_rate'] * 100:7.0f}{'*' if bad and r['fall_rate'] < 1 else ' '} "
            print(row)
    worst = sorted(results, key=lambda r: (-r["fall_rate"], -r["tilt_max_deg"]))[:8]
    print("\n[grid] worst cells:")
    for r in worst:
        print(f"  vx={r['vx']:+.1f} vy={r['vy']:+.1f} wz={r['wz']:+.1f}  fall {r['fall_rate']*100:.0f}%  fall_t {r['fall_time_s']}  tilt max {r['tilt_max_deg']:.1f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"label": args.label, "checkpoint": args.checkpoint, "mode": args.mode, "noise": args.noise, "shove": args.shove,
                       "extra_delay_ms": args.extra_delay_ms, "reps": reps, "grid": grid, "results": results}, f, indent=1)
        print(f"[grid] wrote {args.out}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
