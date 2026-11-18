#!/usr/bin/env python3
"""What should the robot do after a trip while standing on the ground?  (Request 2 of robot_deploy/docs/sim_requests_from_hardware.md)

The controller's stop today: kp of every joint fades to 0 over 0.5 s, then kd-only damping. Hoisted that is benign; on the ground the legs stop supporting the robot.
This script trips a standing (or turning) v11 at a random moment and compares stop strategies for the legs and waist yaw (the explicit-PD joints; the arms keep their
implicit hold, a simplification: on the robot they fade too):

  none          no trip (the policy keeps running): the reference, and the case of a false alarm if the trip is ignored
  fade 0.5      kp 1 -> 0 over 0.5 s, then damping            (today)
  fade 1 / 2    the same over 1 s / 2 s
  hold X for H  the last target at X of the stiffness (X = 0.5, 0.25) for H seconds (1, 2), then damping
  stiff 1 + 1   full stiffness for 1 s, then fade over 1 s
  damp now      kp = 0 at once (the worst case)
Damping in the damping phase: kd = d x the trained kd (d = 0.5 by default, about what scripts/estop_damping.py sends: kd 2/2/2/4/2/2 against the trained 4/4/4/5/3/3).
The policy output is frozen at the trip (the target stays where it was). The trip time is random in [5.0, 5.8] s so it falls at every gait-clock phase; in the "turn" scenario
the robot is turning in place at 0.3 rad/s when it trips (the false alarm of Stage 5a).

Per strategy and scenario: share that fell (terminated within 6 s), time to fall, base speed and downward speed just before the fall (impact), share that sagged (alive, base below
0.6 m after 6 s) or still stands, and the peak joint torque after the trip (legs and waist yaw, p95 over envs).

    python -u scripts/evaluate_trip_stop.py --task <V11 Play task> --checkpoint <model.pt> --out file.json [--noise] [--d 0.5]
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
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--d", type=float, default=0.5, help="damping phase kd as a fraction of the trained kd")
parser.add_argument("--noise", action="store_true")
parser.add_argument("--quick", action="store_true")
parser.add_argument("--only", default=None, help="comma list of strategy names")
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
from g1_rl.v8_env_cfg import CONTROLLED

T_STAND, T_TRIP0, T_TRIP_SPREAD, T_AFTER = 3.0, 5.0, 0.8, 6.0
YAW_RAMP = 2.0
SAG_Z = 0.6
# name -> (kind, params): kp scale as a function of the time since the trip
STRATEGIES = [
    ("none", ("none",)),
    ("fade 0.5 (today)", ("fade", 0.5)), ("fade 1.0", ("fade", 1.0)), ("fade 2.0", ("fade", 2.0)),
    ("hold 50% 1 s", ("hold", 0.5, 1.0)), ("hold 50% 2 s", ("hold", 0.5, 2.0)),
    ("hold 25% 1 s", ("hold", 0.25, 1.0)), ("hold 25% 2 s", ("hold", 0.25, 2.0)),
    ("stiff 1 s + fade 1 s", ("stiff_fade", 1.0, 1.0)), ("damp now", ("fade", 0.0)),
]
SCENARIOS = [("stand", 0.0), ("turn 0.3", 0.3)]


def kp_scale(kind_params, tau):
    """tau: time since the trip per env (negative before). Returns kp multiplier per env."""
    kind = kind_params[0]
    after = tau >= 0
    if kind == "none":
        return torch.ones_like(tau)
    if kind == "fade":
        f = kind_params[1]
        s = torch.clamp(1.0 - tau / max(f, 1e-6), 0.0, 1.0) if f > 0 else torch.zeros_like(tau)
        return torch.where(after, s, torch.ones_like(tau))
    if kind == "hold":
        x, h = kind_params[1], kind_params[2]
        return torch.where(after, torch.where(tau < h, torch.full_like(tau, x), torch.zeros_like(tau)), torch.ones_like(tau))
    if kind == "stiff_fade":
        h, f = kind_params[1], kind_params[2]
        s = torch.where(tau < h, torch.ones_like(tau), torch.clamp(1.0 - (tau - h) / f, 0.0, 1.0))
        return torch.where(after, s, torch.ones_like(tau))
    raise ValueError(kind)


@torch.inference_mode()
def main():
    device = "cuda:0"
    n = 64 if args.quick else args.num_envs
    torch.manual_seed(args.seed)
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=n)
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
    dt = base.step_dt
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    arm_default = robot.data.default_joint_pos[:, arm_ids].clone()
    jids = robot.find_joints(list(CONTROLLED), preserve_order=True)[0]
    acts = [a for a in robot.actuators.values() if hasattr(a, "positions_delay_buffer")]
    kp0 = [a.stiffness.clone() for a in acts]
    kd0 = [a.damping.clone() for a in acts]
    strategies = STRATEGIES if not args.only else [s for s in STRATEGIES if s[0] in args.only.split(",")]
    if args.quick:
        strategies = [STRATEGIES[0], STRATEGIES[1], STRATEGIES[-1]]
    steps = int((T_TRIP0 + T_TRIP_SPREAD + T_AFTER) / dt)

    def run(strategy, yaw):
        name, kp = strategy
        for a, k0, d0 in zip(acts, kp0, kd0):  # every run starts from the trained gains (the previous strategy left them at 0)
            a.stiffness[:] = k0
            a.damping[:] = d0
        env.reset()
        obs = env.get_observations()
        g = torch.Generator(device=device).manual_seed(args.seed + 7)
        t_trip = T_TRIP0 + torch.rand(n, device=device, generator=g) * T_TRIP_SPREAD
        alive = torch.ones(n, dtype=torch.bool, device=device)
        fall_t = torch.full((n,), float("nan"), device=device)
        impact_speed, impact_vz = torch.zeros(n, device=device), torch.zeros(n, device=device)
        peak_tq = torch.zeros(n, device=device)
        cmd = 0.0
        held = torch.zeros(n, 13, device=device)
        tripped = torch.zeros(n, dtype=torch.bool, device=device)
        for i in range(steps):
            t = i * dt
            goal = yaw if t >= T_STAND else 0.0
            cmd += max(-YAW_RAMP * dt, min(YAW_RAMP * dt, goal - cmd))
            term.vel_command_b[:] = torch.tensor([0.0, 0.0, cmd], device=device)
            robot.set_joint_position_target(arm_default, joint_ids=arm_ids)
            tau = t - t_trip
            now_trip = (tau >= 0) & ~tripped
            action = actor(obs["policy"])
            if kp[0] != "none":
                held = torch.where(tripped.unsqueeze(1), held, action)  # freeze the output at the trip: the last pre-trip action
                held = torch.where(now_trip.unsqueeze(1), action, held)
                tripped = tripped | now_trip
                action = torch.where(tripped.unsqueeze(1), held, action)
                s = kp_scale(kp, tau)
                d = torch.where(tau >= 0, torch.full_like(tau, args.d), torch.ones_like(tau))
                for a, k0, d0 in zip(acts, kp0, kd0):
                    a.stiffness[:] = k0 * s.unsqueeze(1)
                    a.damping[:] = d0 * d.unsqueeze(1)
            pre_v = robot.data.root_lin_vel_w.clone()
            obs, _, dones, extras = env.step(action)
            died = dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool() & alive
            fall_t[died] = t
            impact_speed[died] = pre_v[died].norm(dim=1)
            impact_vz[died] = -pre_v[died][:, 2]
            alive &= ~died
            post = (tau >= 0) & alive
            tq = robot.data.applied_torque[:, jids].abs().amax(dim=1)
            peak_tq = torch.where(post, torch.maximum(peak_tq, tq), peak_tq)
        z = robot.data.root_pos_w[:, 2]
        fell = ~alive
        sag = alive & (z < SAG_Z)
        stand = alive & (z >= SAG_Z)
        f = lambda x: float(x.float().mean())
        return {
            "fell": f(fell), "sagged": f(sag), "stands": f(stand),
            "fall_time_after_trip_s": float((fall_t[fell] - t_trip[fell]).median()) if fell.any() else None,
            "impact_speed_ms": float(impact_speed[fell].mean()) if fell.any() else None,
            "impact_down_ms": float(impact_vz[fell].mean()) if fell.any() else None,
            "peak_torque_p95": float(torch.quantile(peak_tq[tripped | (kp[0] == "none")], 0.95)) if n else None,
            "peak_torque_max": float(peak_tq.max()),
        }

    results = {}
    print(f"[trip] {os.path.basename(args.checkpoint)} {n} envs, noise={args.noise}, damping kd x{args.d}; trip at 5.0-5.8 s, {T_AFTER:.0f} s observed", flush=True)
    for sc_name, yaw in SCENARIOS:
        print(f"\n== {sc_name}")
        print(f"{'strategy':>22} | {'fell':>5} {'sagged':>6} {'stands':>6} | {'t fall':>6} {'v impact':>8} {'v down':>6} | {'tau p95':>7} {'max':>5}")
        results[sc_name] = {}
        for strategy in strategies:
            r = run(strategy, yaw)
            results[sc_name][strategy[0]] = r
            fs = lambda x, p: ("%" + p) % x if x is not None else "    - "
            print(f"{strategy[0]:>22} | {r['fell']*100:4.0f}% {r['sagged']*100:5.0f}% {r['stands']*100:5.0f}% | {fs(r['fall_time_after_trip_s'], '6.2f')} "
                  f"{fs(r['impact_speed_ms'], '8.2f')} {fs(r['impact_down_ms'], '6.2f')} | {r['peak_torque_p95']:7.1f} {r['peak_torque_max']:5.1f}", flush=True)
    if args.out:
        with open(args.out, "w") as fo:
            json.dump({"checkpoint": args.checkpoint, "noise": args.noise, "d": args.d, "results": results}, fo, indent=1)
        print(f"[trip] wrote {args.out}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
