#!/usr/bin/env python3
"""Standing-still evaluation for a g1_rl walking policy.

Indoor navigation needs the robot to stay *put* when the velocity command is zero:
no drift, no stepping, no sway. This runs the policy at command (0, 0, 0) under a
set of arm-motion scenarios and (separately) shoves, and measures how still it is.

Arm scenarios (the arms are driven by this script identically for every policy, not
by the policy; any arm-randomising command term in the task is removed):
  rest         arms at the default pose
  pitch_neg    both shoulders held at pitch -1.0 rad   (static pose: shifts the centre of mass)
  pitch_pos    both shoulders held at pitch +1.0 rad
  roll_out     both arms held out to the sides
  wave_right   right arm waves (shoulder + elbow sinusoid, 1 Hz)
  swing_both   both arms swing in antiphase (0.8 Hz)
  human_slow   random reaches between poses at <= 1.0 rad/s with 0.5-1.5 s pauses
  flail_fast   random poses at 3 rad/s, no pauses (stress)

Metrics are taken over the window after the first 3 s, per env:
  drift        horizontal distance from where it was at 3 s to the end        (cm)
  speed        RMS horizontal base speed                                      (mm/s)
  yaw          heading change over the window                                 (deg)
  tilt rms/max base tilt away from vertical                                   (deg)
  jitter       RMS leg joint velocity                                         (rad/s)
  lifts/min    foot lift-offs per minute (a foot off the ground for >= 0.1 s; ignores force flicker)
  foot path    total distance the feet travelled over the window              (cm)
  settle       time until the base is quiet for 0.5 s                         (s)
  %still       share of envs that survive AND meet every STILL_* threshold below

Push scenarios (arms at rest, and with human_slow arms): shoves of 0.3 / 0.5 / 0.7 m/s
in random directions at 4 / 9 / 14 s. Per shove: share of envs that are "calm" again (base
under 10 cm/s for 0.5 s) within 4.5 s, time to calm, share that get fully "quiet" (under
3 cm/s), distance from the pre-shove spot 4.5 s later, and foot lift-offs.

    python scripts/evaluate_standing.py --task <Play task> --checkpoint <model.pt> --label v3 [--noise]

--noise turns observation noise on (the Play tasks disable it by default); that is the
realistic condition for hardware. Run it in the isaaclab conda env.
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
parser.add_argument("--seconds", type=float, default=20.0, help="per arm scenario")
parser.add_argument("--noise", action="store_true", help="enable observation noise")
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
from g1_rl.body_env_cfg import ARM_POSE_RANGES
from g1_rl.robot_cfg import ARM_JOINT_NAMES

# "Quiet / still" thresholds. These are proposals for indoor navigation, not standards.
STILL_DRIFT_M = 0.05
STILL_SPEED_MS = 0.03
STILL_TILT_DEG = 6.0
STILL_LIFTS = 0
WINDOW_START_S = 3.0
CONTACT_N = 10.0
QUIET_SPEED = 0.03
QUIET_YAW = 0.10
QUIET_HOLD_S = 0.5
CALM_SPEED = 0.10  # looser bar used for shove recovery: the robot is no longer being carried away
CALM_YAW = 0.30
MIN_AIR_S = 0.1  # a foot counts as lifted only if it stays off the ground this long (ignores force flicker)

ARM_SCENARIOS = ["rest", "pitch_neg", "pitch_pos", "roll_out", "wave_right", "swing_both", "human_slow", "flail_fast"]
PUSH_TIMES_S = (4.0, 9.0, 14.0)
PUSH_MAGS = (0.3, 0.5, 0.7)
PUSH_WINDOW_S = 4.5


class ArmDriver:
    """Produces absolute arm joint targets (n_envs x 14) for one scenario."""

    def __init__(self, kind, n, device, default, lo, hi):
        self.kind, self.n, self.device = kind, n, device
        self.default, self.lo, self.hi = default, lo, hi
        self.idx = {name: i for i, name in enumerate(ARM_JOINT_NAMES)}
        self.phase = torch.rand(n, device=device) * 2 * math.pi
        self.cur = default.clone()
        self.goal = default.clone()
        self.hold = torch.full((n,), 1.0, device=device)
        self.speed, self.pause = {"human_slow": (1.0, (0.5, 1.5)), "flail_fast": (3.0, (0.0, 0.0))}.get(kind, (0, (0, 0)))
        self.static = {}
        if kind == "pitch_neg":
            self.static = {"left_shoulder_pitch_joint": -1.0, "right_shoulder_pitch_joint": -1.0}
        elif kind == "pitch_pos":
            self.static = {"left_shoulder_pitch_joint": 1.0, "right_shoulder_pitch_joint": 1.0}
        elif kind == "roll_out":
            self.static = {"left_shoulder_roll_joint": 0.8, "right_shoulder_roll_joint": -0.8}

    def target(self, t, dt):
        d = self.default.clone()
        ramp = min(t / 1.5, 1.0)
        ramp = ramp * ramp * (3 - 2 * ramp)  # smoothstep, so static poses are reached gently
        if self.kind == "rest":
            return d
        if self.static:
            for name, val in self.static.items():
                i = self.idx[name]
                d[:, i] = self.default[:, i] + ramp * (min(max(val, self.lo[i].item()), self.hi[i].item()) - self.default[:, i])
            return d
        if self.kind == "wave_right":
            s = torch.sin(2 * math.pi * 1.0 * t + self.phase)
            c = torch.cos(2 * math.pi * 1.0 * t + self.phase)
            d[:, self.idx["right_shoulder_pitch_joint"]] += ramp * 0.6 * s
            d[:, self.idx["right_elbow_joint"]] += ramp * 0.4 * c
            return d
        if self.kind == "swing_both":
            s = torch.sin(2 * math.pi * 0.8 * t + self.phase)
            d[:, self.idx["left_shoulder_pitch_joint"]] += ramp * 0.7 * s
            d[:, self.idx["right_shoulder_pitch_joint"]] -= ramp * 0.7 * s
            return d
        # human_slow / flail_fast: speed-limited moves between random poses
        moving = self.hold <= 0
        delta = self.goal - self.cur
        arrived = moving & (delta.abs().max(dim=1).values < 1e-3)
        if arrived.any():
            k = int(arrived.sum())
            self.goal[arrived] = self.lo + torch.rand(k, len(self.lo), device=self.device) * (self.hi - self.lo)
            self.hold[arrived] = self.pause[0] + torch.rand(k, device=self.device) * (self.pause[1] - self.pause[0])
        self.hold = torch.where(self.hold > 0, self.hold - dt, self.hold)
        step = (self.goal - self.cur).clamp(-self.speed * dt, self.speed * dt)
        self.cur = self.cur + step * (self.hold <= 0).unsqueeze(1).float()
        return self.cur


def tilt_deg(robot):
    g = robot.data.projected_gravity_b[:, :2].norm(dim=1).clamp(max=1.0)
    return torch.rad2deg(torch.asin(g))


def yaw_of(robot):
    q = robot.data.root_quat_w  # w, x, y, z
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def stats(x):
    x = x.float()
    return {"mean": x.mean().item(), "p95": torch.quantile(x, 0.95).item(), "max": x.max().item()} if len(x) else {}


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
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    cmd_cfg.ranges.lin_vel_x = (0.0, 0.0)
    cmd_cfg.ranges.lin_vel_y = (0.0, 0.0)
    cmd_cfg.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.episode_length_s = args.seconds + 5.0
    env_cfg.events.push_robot = None
    if hasattr(env_cfg.commands, "arm_pose"):
        env_cfg.commands.arm_pose = None  # the arms are driven by this script, identically for every policy
    if args.noise:
        env_cfg.observations.policy.enable_corruption = True

    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(args.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    robot = base.scene["robot"]
    n, dt = args.num_envs, base.step_dt
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    leg_ids = robot.find_joints([".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])[0]
    foot_ids = robot.find_bodies(".*_ankle_roll_link")[0]
    sensor = base.scene["contact_forces"]
    sensor_foot_ids = sensor.find_bodies(".*_ankle_roll_link")[0]
    default = robot.data.default_joint_pos[:, arm_ids].clone()
    lo = torch.tensor([ARM_POSE_RANGES[nm][0] for nm in ARM_JOINT_NAMES], device=device)
    hi = torch.tensor([ARM_POSE_RANGES[nm][1] for nm in ARM_JOINT_NAMES], device=device)
    w0 = int(WINDOW_START_S / dt)
    quiet_hold = int(QUIET_HOLD_S / dt)

    def feet_contact():
        return sensor.data.net_forces_w[:, sensor_foot_ids, :].norm(dim=-1) > CONTACT_N  # (n, 2)

    def quiet_now(speed=QUIET_SPEED, yaw=QUIET_YAW):
        v = robot.data.root_lin_vel_w[:, :2].norm(dim=1)
        wz = robot.data.root_ang_vel_w[:, 2].abs()
        return (v < speed) & (wz < yaw)

    min_air = max(int(MIN_AIR_S / dt), 1)

    def lift_events(air_run, contact):
        """Updates the per-foot airborne run length; returns (new run, lift events this step per env)."""
        air_run = torch.where(~contact, air_run + 1, torch.zeros_like(air_run))
        return air_run, (air_run == min_air).float().sum(1)

    def start(kind):
        env.reset()
        driver = ArmDriver(kind, n, device, default, lo, hi)
        return driver, env.get_observations()

    def step(driver, t, obs, prev_action):
        robot.set_joint_position_target(driver.target(t, dt), joint_ids=arm_ids)
        action = policy(obs)
        obs, _, dones, extras = env.step(action)
        died = dones.bool() & ~extras.get("time_outs", torch.zeros_like(dones)).bool()
        return action, obs, died

    # ---------------- standing under arm motion ----------------
    arm_results = {}
    total_steps = int(args.seconds / dt)
    for kind in ARM_SCENARIOS:
        driver, obs = start(kind)
        alive = torch.ones(n, dtype=torch.bool, device=device)
        prev_action = torch.zeros(n, base.action_manager.total_action_dim, device=device)
        z = lambda: torch.zeros(n, device=device)  # noqa: E731
        acc = {k: z() for k in ("sp2", "tilt2", "jv2", "dact", "tq2", "foot_path", "lifts", "count")}
        tilt_max, settle, quiet_run = z(), torch.full((n,), args.seconds, device=device), z()
        start_xy = start_yaw = prev_foot = prev_contact = pos_last = yaw_last = None
        air_run = torch.zeros(n, 2, device=device)
        for i in range(total_steps):
            t = i * dt
            action, obs, died = step(driver, t, obs, prev_action)
            alive &= ~died
            w = alive.float()
            pos = robot.data.root_pos_w[:, :2].clone()
            q = quiet_now()
            quiet_run = torch.where(q, quiet_run + 1, torch.zeros_like(quiet_run))
            newly = (quiet_run >= quiet_hold) & (settle >= args.seconds)
            settle = torch.where(newly, torch.full_like(settle, max(t - QUIET_HOLD_S, 0.0)), settle)
            contact, foot = feet_contact(), robot.data.body_pos_w[:, foot_ids, :2].clone()
            air_run, lifted = lift_events(air_run, contact)
            if i == w0:
                start_xy, start_yaw = pos.clone(), yaw_of(robot)
                prev_foot, prev_contact = foot, contact
            if i >= w0:
                acc["sp2"] += w * robot.data.root_lin_vel_w[:, :2].pow(2).sum(1)
                tl = tilt_deg(robot)
                acc["tilt2"] += w * tl.pow(2)
                tilt_max = torch.maximum(tilt_max, tl * w)
                acc["jv2"] += w * robot.data.joint_vel[:, leg_ids].pow(2).mean(1)
                acc["dact"] += w * (action - prev_action).abs().mean(1)
                acc["tq2"] += w * robot.data.applied_torque[:, leg_ids].pow(2).mean(1)
                acc["foot_path"] += w * (foot - prev_foot).norm(dim=-1).sum(1)
                acc["lifts"] += w * lifted
                acc["count"] += w
                prev_foot, prev_contact = foot, contact
                pos_last, yaw_last = pos, yaw_of(robot)
            prev_action = action
        c = acc["count"].clamp(min=1.0)
        survived = alive
        drift = (pos_last - start_xy).norm(dim=1)
        yaw_drift = torch.rad2deg(torch.atan2(torch.sin(yaw_last - start_yaw), torch.cos(yaw_last - start_yaw)).abs())
        speed = (acc["sp2"] / c).sqrt()
        tilt_rms = (acc["tilt2"] / c).sqrt()
        minutes = (c * dt) / 60.0
        lifts_pm = acc["lifts"] / minutes.clamp(min=1e-6)
        still = survived & (drift < STILL_DRIFT_M) & (speed < STILL_SPEED_MS) & (tilt_max < STILL_TILT_DEG) & (acc["lifts"] <= STILL_LIFTS)
        s = survived
        arm_results[kind] = {
            "survival": survived.float().mean().item(),
            "drift_cm": stats(drift[s] * 100), "speed_mms": stats(speed[s] * 1000), "yaw_deg": stats(yaw_drift[s]),
            "tilt_rms_deg": stats(tilt_rms[s]), "tilt_max_deg": stats(tilt_max[s]),
            "jitter_rads": stats((acc["jv2"] / c).sqrt()[s]), "d_act": stats((acc["dact"] / c)[s]),
            "torque_rms": stats((acc["tq2"] / c).sqrt()[s]), "lifts_per_min": stats(lifts_pm[s]),
            "foot_path_cm": stats(acc["foot_path"][s] * 100), "settle_s": stats(settle[s]),
            "pct_still": still.float().mean().item(),
        }

    # ---------------- shoves while standing ----------------
    push_results = {}
    for kind in ("rest", "human_slow"):
        driver, obs = start(kind)
        total = int((PUSH_TIMES_S[-1] + PUSH_WINDOW_S + 0.5) / dt)
        push_steps = {int(round(tt / dt)): m for tt, m in zip(PUSH_TIMES_S, PUSH_MAGS)}
        alive = torch.ones(n, dtype=torch.bool, device=device)
        prev_action = torch.zeros(n, base.action_manager.total_action_dim, device=device)
        active = None  # (push index, step, pre-push position, quiet counter, recovery step, lifts)
        recs = []
        air_run = torch.zeros(n, 2, device=device)
        for i in range(total):
            t = i * dt
            if i in push_steps:
                ang = torch.rand(n, device=device) * 2 * math.pi
                vel = robot.data.root_vel_w.clone()
                vel[:, 0] += push_steps[i] * torch.cos(ang)
                vel[:, 1] += push_steps[i] * torch.sin(ang)
                robot.write_root_velocity_to_sim(vel)
                nan = torch.full((n,), float("nan"), device=device)
                active = {"mag": push_steps[i], "i0": i, "pos0": robot.data.root_pos_w[:, :2].clone(),
                          "quiet": torch.zeros(n, device=device), "calm": torch.zeros(n, device=device),
                          "rec_quiet": nan.clone(), "rec_calm": nan.clone(), "disp_end": nan.clone(),
                          "lifts": torch.zeros(n, device=device)}
                recs.append(active)
            action, obs, died = step(driver, t, obs, prev_action)
            alive &= ~died
            prev_action = action
            contact = feet_contact()
            air_run, lifted = lift_events(air_run, contact)
            for r in recs:
                last = r["i0"] + int(PUSH_WINDOW_S / dt)
                if i < r["i0"] or i > last:
                    continue
                r["lifts"] += lifted * alive.float()
                r["quiet"] = torch.where(quiet_now(), r["quiet"] + 1, torch.zeros_like(r["quiet"]))
                r["calm"] = torch.where(quiet_now(CALM_SPEED, CALM_YAW), r["calm"] + 1, torch.zeros_like(r["calm"]))
                t_rel = (i - r["i0"]) * dt - QUIET_HOLD_S
                r["rec_quiet"] = torch.where((r["quiet"] >= quiet_hold) & torch.isnan(r["rec_quiet"]) & alive,
                                             torch.full_like(r["rec_quiet"], t_rel), r["rec_quiet"])
                r["rec_calm"] = torch.where((r["calm"] >= quiet_hold) & torch.isnan(r["rec_calm"]) & alive,
                                            torch.full_like(r["rec_calm"], t_rel), r["rec_calm"])
                if i == last:
                    r["disp_end"] = (robot.data.root_pos_w[:, :2] - r["pos0"]).norm(dim=1)
        rows = []
        for r in recs:
            sv = alive
            calm = sv & ~torch.isnan(r["rec_calm"])
            quiet = sv & ~torch.isnan(r["rec_quiet"])
            rows.append({
                "push_ms": r["mag"],
                "calm_frac": calm.float().mean().item(), "calm_s": stats(r["rec_calm"][calm]),
                "quiet_frac": quiet.float().mean().item(),
                "disp_end_cm": stats(r["disp_end"][sv] * 100),
                "lifts": r["lifts"][sv].mean().item() if sv.any() else float("nan"),
            })
        push_results[kind] = {"survival": alive.float().mean().item(), "pushes": rows}

    # ---------------- report ----------------
    print(f"\n[standing] {args.label}  checkpoint={os.path.basename(args.checkpoint)}  noise={'on' if args.noise else 'off'}  "
          f"{n} envs x {args.seconds:.0f}s, window {WINDOW_START_S:.0f}s..end")
    print(f"[standing] still = survive & drift<{STILL_DRIFT_M*100:.0f}cm & speed<{STILL_SPEED_MS*1000:.0f}mm/s & "
          f"tilt<{STILL_TILT_DEG:.0f}deg & lifts<={STILL_LIFTS}")
    head = f"{'arms':>11} {'surv':>5} {'drift cm':>9} {'p95':>6} {'speed':>6} {'yaw':>5} {'tilt':>5} {'tmax':>5} {'jit':>5} {'lift/m':>6} {'foot cm':>7} {'settle':>6} {'%still':>6}"
    print(head)
    for k in ARM_SCENARIOS:
        r = arm_results[k]
        f = lambda key, sub="mean": r[key].get(sub, float("nan"))  # noqa: E731
        print(f"{k:>11} {r['survival']:5.2f} {f('drift_cm'):9.1f} {f('drift_cm','p95'):6.1f} {f('speed_mms'):6.1f} "
              f"{f('yaw_deg'):5.1f} {f('tilt_rms_deg'):5.2f} {f('tilt_max_deg','p95'):5.2f} {f('jitter_rads'):5.2f} "
              f"{f('lifts_per_min'):6.1f} {f('foot_path_cm'):7.1f} {f('settle_s'):6.1f} {r['pct_still']*100:5.0f}%")
    print(f"\n[standing] shoves at 4/9/14 s. calm = base <{CALM_SPEED*100:.0f}cm/s for 0.5 s within {PUSH_WINDOW_S}s; "
          f"quiet = <{QUIET_SPEED*100:.0f}cm/s; disp = distance from the pre-shove spot {PUSH_WINDOW_S}s later")
    print(f"{'arms':>11} {'push m/s':>8} {'surv':>5} {'calm%':>6} {'calm s':>6} {'quiet%':>6} {'disp cm':>8} {'p95':>6} {'lifts':>6}")
    for kind, pr in push_results.items():
        for row in pr["pushes"]:
            print(f"{kind:>11} {row['push_ms']:8.1f} {pr['survival']:5.2f} {row['calm_frac']*100:5.0f}% "
                  f"{row['calm_s'].get('mean', float('nan')):6.2f} {row['quiet_frac']*100:5.0f}% "
                  f"{row['disp_end_cm'].get('mean', float('nan')):8.1f} {row['disp_end_cm'].get('p95', float('nan')):6.1f} {row['lifts']:6.2f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"label": args.label, "checkpoint": args.checkpoint, "noise": args.noise, "seconds": args.seconds,
                       "thresholds": {"drift_m": STILL_DRIFT_M, "speed_ms": STILL_SPEED_MS, "tilt_deg": STILL_TILT_DEG, "lifts": STILL_LIFTS},
                       "arms": arm_results, "pushes": push_results}, fh, indent=2)
        print(f"[standing] wrote {args.out}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
