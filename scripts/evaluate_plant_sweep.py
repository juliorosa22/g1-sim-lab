#!/usr/bin/env python3
"""How far can the real robot differ from the simulated one before a policy falls?  A plant-mismatch sweep on top of the training randomisation.

The policy runs on a command box (ramp, observation noise, optional shoves): stand 2 s, ramp to each command of the box, hold until 8 s. Each SETTING changes
the plant after the environment's own startup randomisation (so it is relative to what was trained: friction 0.3-1.2, torso mass +-5 kg, torso COM +-3 cm,
leg gains +-20 %, armature x0.8-1.5, delay 0-20 ms):

  mass      all link masses (and inertias) x s
  torso     extra mass on the torso (kg), on top of the +-5 kg trained
  friction  the robot's contact friction x s (static and dynamic)
  kp / kd   leg + waist-yaw stiffness x s / damping x s
  torque    leg + waist-yaw torque limit x s (a weaker motor)
  jfric     Coulomb joint friction on the legs, N*m, tau -= f*tanh(qd/0.02)  (the real ankles show ~0.7-1 N*m breakaway)
  delay     extra action delay, ms, on top of the trained 0-20
  com       torso centre of mass shifted forward by x metres
  combined  several of the above together

    python -u scripts/evaluate_plant_sweep.py --task <Play task> --checkpoint <model.pt> --preset v11 --out file.json [--shove] [--quick]
"""
import argparse
import json
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher

BOXES = {  # the commands inside the trained range with the recommended margin
    "v11": dict(vx=[-0.3, 0.0, 0.3, 0.5], vy=[-0.25, 0.0, 0.25], wz=[-1.0, -0.5, 0.0, 0.5, 1.0]),
    "v12": dict(vx=[0.0, 0.3, 0.6, 0.8, 1.0], vy=[-0.25, 0.0, 0.25], wz=[-0.5, 0.0, 0.5]),
}
SETTINGS = [
    ("baseline", {}),
    ("mass x0.85", dict(mass=0.85)), ("mass x1.15", dict(mass=1.15)), ("mass x1.3", dict(mass=1.3)),
    ("torso +8 kg", dict(torso=8.0)), ("torso -8 kg", dict(torso=-8.0)),
    ("friction x0.4", dict(friction=0.4)), ("friction x0.2", dict(friction=0.2)), ("friction x1.5", dict(friction=1.5)),
    ("kp x0.6", dict(kp=0.6)), ("kp x1.4", dict(kp=1.4)), ("kd x0.5", dict(kd=0.5)), ("kd x1.6", dict(kd=1.6)),
    ("kp,kd x0.7", dict(kp=0.7, kd=0.7)), ("kp,kd x1.3", dict(kp=1.3, kd=1.3)),
    ("torque x0.9", dict(torque=0.9)), ("torque x0.75", dict(torque=0.75)), ("torque x0.6", dict(torque=0.6)),
    ("jfric 1 Nm", dict(jfric=1.0)), ("jfric 2 Nm", dict(jfric=2.0)), ("jfric 4 Nm", dict(jfric=4.0)),
    ("delay +10 ms", dict(delay=10.0)), ("delay +20 ms", dict(delay=20.0)), ("delay +30 ms", dict(delay=30.0)),
    ("com +4 cm", dict(com=0.04)), ("com -4 cm", dict(com=-0.04)),
    ("combined mild", dict(mass=1.1, friction=0.6, kp=0.85, kd=0.85, torque=0.9, jfric=1.0, delay=10.0)),
    ("combined hard", dict(mass=1.2, friction=0.4, kp=0.7, kd=0.7, torque=0.8, jfric=2.0, delay=20.0, com=0.03)),
]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--label", default="policy")
parser.add_argument("--preset", choices=list(BOXES), default="v11")
parser.add_argument("--reps", type=int, default=8)
parser.add_argument("--noise", action="store_true", default=True)
parser.add_argument("--no-noise", dest="noise", action="store_false")
parser.add_argument("--shove", action="store_true")
parser.add_argument("--only", default=None, help="comma list of setting names")
parser.add_argument("--quick", action="store_true")
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

T_STAND, T_END = 2.0, 8.0
RAMP_LIN, RAMP_YAW, DEADBAND = 1.0, 2.0, 0.05
PHYS_MS, TRAINED_MAX_STEPS, MAX_EXTRA_MS = 5.0, 4, 30.0
CONTACT_N, MIN_AIR_S = 10.0, 0.1
SHOVE_TIMES, SHOVE_SPEED = (4.0, 6.0), 0.3


@torch.inference_mode()
def main():
    device = "cuda:0"
    box = BOXES[args.preset]
    reps = 2 if args.quick else args.reps
    if args.quick:
        box = dict(vx=[0.0, 0.5], vy=[0.0], wz=[0.0, 1.0])
    cells = [(vx, vy, wz) for wz in box["wz"] for vx in box["vx"] for vy in box["vy"]]
    n = len(cells) * reps
    settings = SETTINGS if not args.only else [s for s in SETTINGS if s[0] in args.only.split(",")]
    if args.quick:
        settings = [SETTINGS[0], SETTINGS[1], SETTINGS[18], SETTINGS[26]]

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=n)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.heading_command = False
    if hasattr(cmd_cfg, "mixed_sampling"):
        cmd_cfg.mixed_sampling = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.episode_length_s = T_END + 10.0
    env_cfg.events.push_robot = None
    if hasattr(env_cfg.commands, "arm_pose"):
        env_cfg.commands.arm_pose = None
    env_cfg.observations.policy.enable_corruption = bool(args.noise)
    delayed = [name for name, a in env_cfg.scene.robot.actuators.items() if hasattr(a, "max_delay")]
    for name in delayed:
        env_cfg.scene.robot.actuators[name].max_delay = TRAINED_MAX_STEPS + int(MAX_EXTRA_MS / PHYS_MS)
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
    view = robot.root_physx_view
    all_env = torch.arange(n)  # physx views want CPU indices
    torso_idx = robot.find_bodies("torso_link")[0][0]

    # ---- originals, so every setting starts from the trained plant ----
    m0, i0 = view.get_masses().clone(), view.get_inertias().clone()
    mat0 = view.get_material_properties().clone()
    com0 = view.get_coms().clone()
    acts = {name: robot.actuators[name] for name in delayed}
    kp0 = {k: a.stiffness.clone() for k, a in acts.items()}
    kd0 = {k: a.damping.clone() for k, a in acts.items()}
    eff0 = {k: a.effort_limit.clone() for k, a in acts.items()}
    fr = {"nm": 0.0}
    for a in acts.values():  # Coulomb joint friction added on the actuator output
        orig = a.compute

        def wrapped(control_action, joint_pos, joint_vel, _orig=orig):
            ca = _orig(control_action, joint_pos, joint_vel)
            if fr["nm"] > 0:
                ca.joint_efforts = ca.joint_efforts - fr["nm"] * torch.tanh(joint_vel / 0.02)
            return ca
        a.compute = wrapped

    def apply(s):
        m, inr = m0.clone(), i0.clone()
        if "mass" in s:
            m *= s["mass"]
            inr *= s["mass"]
        if "torso" in s:
            ratio = (m[:, torso_idx] + s["torso"]) / m[:, torso_idx]
            m[:, torso_idx] *= ratio
            inr[:, torso_idx] *= ratio.unsqueeze(-1)
        view.set_masses(m, all_env)
        view.set_inertias(inr, all_env)
        mat = mat0.clone()
        if "friction" in s:
            mat[..., 0] *= s["friction"]
            mat[..., 1] *= s["friction"]
        view.set_material_properties(mat, all_env)
        com = com0.clone()
        if "com" in s:
            com[:, torso_idx, 0] += s["com"]
        view.set_coms(com, all_env)
        for k, a in acts.items():
            a.stiffness[:] = kp0[k] * s.get("kp", 1.0)
            a.damping[:] = kd0[k] * s.get("kd", 1.0)
            a.effort_limit[:] = eff0[k] * s.get("torque", 1.0)
        fr["nm"] = float(s.get("jfric", 0.0))
        extra = int(round(s.get("delay", 0.0) / PHYS_MS))
        for a in acts.values():
            lags = torch.randint(0, TRAINED_MAX_STEPS + 1, (n,), device=device, dtype=torch.int) + extra
            for buf in (a.positions_delay_buffer, a.velocities_delay_buffer, a.efforts_delay_buffer):
                buf.set_time_lag(lags)

    goal = torch.zeros(n, 3, device=device)
    for ci, c in enumerate(cells):
        goal[ci * reps:(ci + 1) * reps] = torch.tensor(c, device=device)
    small = goal[:, :2].norm(dim=1) < DEADBAND
    goal[small, :2] = 0.0
    goal[goal[:, 2].abs() < DEADBAND, 2] = 0.0

    def run(s):
        apply(s)
        env.reset()
        obs = env.get_observations()
        cur = torch.zeros(n, 3, device=device)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        fall_t = torch.full((n,), float("nan"), device=device)
        tilt_max = torch.zeros(n, device=device)
        air_run, lifts = torch.zeros(n, 2, device=device), torch.zeros(n, device=device)
        vel_acc, cnt = torch.zeros(n, 3, device=device), 0
        shove_i = 0
        for i in range(int(T_END / dt)):
            t = i * dt
            target = torch.zeros_like(goal) if t < T_STAND else goal
            d = target[:, :2] - cur[:, :2]
            nrm = d.norm(dim=1, keepdim=True).clamp_min(1e-9)
            cur[:, :2] += d * torch.clamp(RAMP_LIN * dt / nrm, max=1.0)
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
            fall_t[died & alive] = t
            alive &= ~died
            g = robot.data.projected_gravity_b[:, :2].norm(dim=1).clamp(max=1.0)
            tilt_max = torch.where(alive, torch.maximum(tilt_max, torch.rad2deg(torch.asin(g))), tilt_max)
            contact = sensor.data.net_forces_w[:, foot_ids, :].norm(dim=-1) > CONTACT_N
            air_run = torch.where(~contact, air_run + 1, torch.zeros_like(air_run))
            if t >= T_END - 3.0:
                lifts += (air_run == min_air).float().sum(1)
                vel_acc += torch.cat([robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], dim=1)
                cnt += 1
        vel = vel_acc / max(cnt, 1)
        fell = ~alive
        err_lin = (vel[:, :2] - goal[:, :2]).norm(dim=1)
        err_yaw = (vel[:, 2] - goal[:, 2]).abs()
        per_cell = [float(fell[ci * reps:(ci + 1) * reps].float().mean()) for ci in range(len(cells))]
        worst = sorted(range(len(cells)), key=lambda ci: -per_cell[ci])[:3]
        return {
            "fall_rate": float(fell.float().mean()),
            "fall_time_s": float(fall_t[fell].mean()) if fell.any() else None,
            "cells_with_fall": sum(1 for p in per_cell if p > 0), "cells": len(cells),
            "worst_cells": [{"cmd": cells[ci], "fall": per_cell[ci]} for ci in worst if per_cell[ci] > 0],
            "tilt_max_mean": float(tilt_max[alive].mean()) if alive.any() else None,
            "lifts_per_min": float((lifts[alive] / 3.0 * 60).mean()) if alive.any() else None,
            "err_lin": float(err_lin[alive].mean()) if alive.any() else None,
            "err_yaw": float(err_yaw[alive].mean()) if alive.any() else None,
        }

    print(f"[plant] {args.label} {os.path.basename(args.checkpoint)} box={args.preset} cells={len(cells)} reps={reps} envs={n} noise={args.noise} shove={args.shove}", flush=True)
    results = {}
    print(f"{'setting':>16} | {'fall %':>6} {'cells':>7} {'fall t':>6} | {'tilt':>5} {'lift/min':>8} {'err v':>6} {'err wz':>6} | worst cells")
    for name, s in settings:
        r = run(s)
        results[name] = {"setting": s, **r}
        w = "; ".join(f"({c['cmd'][0]:+.1f},{c['cmd'][1]:+.2f},{c['cmd'][2]:+.1f}) {c['fall']*100:.0f}%" for c in r["worst_cells"])
        f = lambda x, p: f"{x:{p}}" if x is not None else "   - "
        print(f"{name:>16} | {r['fall_rate']*100:6.1f} {r['cells_with_fall']:3d}/{r['cells']:<3d} {f(r['fall_time_s'], '6.1f')} | {f(r['tilt_max_mean'], '5.1f')} "
              f"{f(r['lifts_per_min'], '8.1f')} {f(r['err_lin'], '6.2f')} {f(r['err_yaw'], '6.2f')} | {w}", flush=True)
    if args.out:
        with open(args.out, "w") as fo:
            json.dump({"label": args.label, "checkpoint": args.checkpoint, "preset": args.preset, "noise": args.noise, "shove": args.shove, "reps": reps,
                       "box": box, "results": results}, fo, indent=1)
        print(f"[plant] wrote {args.out}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
