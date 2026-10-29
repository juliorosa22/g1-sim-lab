#!/usr/bin/env python3
"""Switching-controller test: a walking policy and a standing policy behind a selector.

  walking policy   v4 (legs + waist, 84 obs -> 15 actions)         used for |cmd| above the band
  standing policy  v1 (legs only,   47 obs -> 12 actions)          used for |cmd| below the band
  selector         hysteresis (walk above 0.08, stand below 0.04), debounce, 0.3 s action blend,
                   and a command ramp so a stop reaches the handover at low speed

Both policies run in the v4 task (so one simulation hosts both). v1's observation is rebuilt
from the simulation state exactly as its own task built it, and its 12 leg actions are padded
with a zero waist action (waist stays at the default pose while v1 is active).

All envs run the same scripted command sequence (below); the robot starts standing. Reported:
  * survival per segment, and in which mode each fall happened
  * standing quality in every stand segment (drift, speed, tilt, foot lift-offs)
  * walking accuracy in every walk segment (achieved vs commanded)
  * handover transients: peak tilt / base speed in the 1.5 s after each switch
  * switch count in the hysteresis tests (should be 0 inside the band)

--video records ONE env of the same script with captions (command, active policy, shoves).

    python scripts/evaluate_switch.py --walk-checkpoint A.pt --stand-checkpoint B.pt [--video] [--noise]

Checkpoints may be in the rsl-rl 3.x or 5.x format (only the actor weights are read).
"""
import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="G1-Walk-Flat-Inspire-Arms-Play-v0")
parser.add_argument("--walk-checkpoint", required=True)
parser.add_argument("--stand-checkpoint", required=True)
parser.add_argument("--walk-label", default="v4")
parser.add_argument("--stand-label", default="v1")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--noise", action="store_true", help="observation noise for both policies")
parser.add_argument("--arms", choices=["rest", "random"], default="rest")
parser.add_argument("--enter-walk", type=float, default=0.08)
parser.add_argument("--exit-walk", type=float, default=0.04)
parser.add_argument("--debounce-walk", type=float, default=0.2, help="s above enter-walk before switching to walking")
parser.add_argument("--debounce-stand", type=float, default=0.3, help="s below exit-walk before switching to standing")
parser.add_argument("--blend", type=float, default=0.3, help="s over which actions cross-fade at a switch")
parser.add_argument("--ramp-lin", type=float, default=1.0, help="command deceleration limit, m/s^2")
parser.add_argument("--ramp-yaw", type=float, default=2.0, help="rad/s^2")
parser.add_argument("--video", action="store_true", help="record one env with captions instead of measuring 128")
parser.add_argument("--out", default=None)
parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "videos", "switch"))
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max-seconds", type=float, default=None, help="stop early (smoke tests)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = bool(args.video)

app = AppLauncher(args).app

import gymnasium as gym
import torch
import torch.nn as nn

from isaaclab_tasks.utils import parse_env_cfg

import g1_rl  # noqa: F401
from g1_rl.robot_cfg import ARM_JOINT_NAMES, LEG_JOINT_NAMES

STAND, WALK = 0, 1
CONTACT_N, MIN_AIR_S = 10.0, 0.1

# (t_start, kind, command (vx, vy, wz), label, hard) -- the command stays until the next row.
# `hard` = the command jumps with no ramp (an emergency stop); otherwise it is ramped.
SCRIPT = [
    (0.0, "stand", (0.0, 0.0, 0.0), "Standing", False),
    (3.0, "walk", (0.5, 0.0, 0.0), "Walking forward  0.5 m/s", False),
    (9.0, "stand", (0.0, 0.0, 0.0), "Stop (ramped)", False),
    (14.0, "walk", (0.3, 0.0, 0.3), "Walking + turning", False),
    (20.0, "stand", (0.0, 0.0, 0.0), "Stop (ramped)", False),
    (25.0, "walk", (0.8, 0.0, 0.0), "Walking forward  0.8 m/s", False),
    (30.0, "stand", (0.0, 0.0, 0.0), "EMERGENCY stop (no ramp)", True),
    (35.0, "walk", (0.0, 0.2, 0.0), "Walking sideways  0.2 m/s", False),
    (40.0, "stand", (0.0, 0.0, 0.0), "Stop (ramped)", False),
    (45.0, "walk", (0.0, 0.0, 0.5), "Turning in place  0.5 rad/s", False),
    (50.0, "stand", (0.0, 0.0, 0.0), "Stop (ramped)", False),
    (55.0, "hover_stand", (0.06, 0.0, 0.0), "Command hovering in the band (standing)", False),
    (61.0, "walk", (0.2, 0.0, 0.0), "Walking forward  0.2 m/s", False),
    (67.0, "hover_walk", (0.06, 0.0, 0.0), "Command hovering in the band (walking)", False),
    (73.0, "stand", (0.0, 0.0, 0.0), "Stop (ramped)", False),
]
END_S = 78.0
SHOVES = [(11.0, 0.3), (22.0, 0.3)]  # (time s, m/s): while standing after a ramped stop
SETTLE_S = {"stand": 2.0, "walk": 1.5, "hover_stand": 1.0, "hover_walk": 1.0}


def load_actor(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in ck:  # rsl-rl 3.x
        sd = {k[len("actor."):]: v for k, v in ck["model_state_dict"].items() if k.startswith("actor.")}
    else:  # rsl-rl 5.x
        sd = {k[len("mlp."):]: v for k, v in ck["actor_state_dict"].items() if k.startswith("mlp.")}
    idx = sorted({int(k.split(".")[0]) for k in sd})
    layers = []
    for n, i in enumerate(idx):
        w = sd[f"{i}.weight"]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if n < len(idx) - 1:
            layers.append(nn.ELU())
    net = nn.Sequential(*layers)
    net.load_state_dict(sd)  # strict: Linear layers sit at the even indices, as in rsl-rl's MLP
    return net.eval().to("cuda:0")


def command_at(t):
    cmd, kind, hard, label, t0 = (0.0, 0.0, 0.0), "stand", False, "", 0.0
    for ts, k, c, lb, h in SCRIPT:
        if t >= ts:
            cmd, kind, label, t0, hard = c, k, lb, ts, h
    if kind.startswith("hover"):  # alternates inside the band every 0.5 s
        v = 0.05 if int((t - t0) / 0.5) % 2 == 0 else 0.07
        cmd = (v, 0.0, 0.0)
    return cmd, kind, label, t0, hard


def segment_of(t):
    seg = 0
    for i, row in enumerate(SCRIPT):
        if t >= row[0]:
            seg = i
    return seg


@torch.inference_mode()
def main():
    device = "cuda:0"
    n = 1 if args.video else args.num_envs
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=n)
    env_cfg.seed = args.seed
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.heading_command = False
    cmd_cfg.mixed_sampling = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.deadband = 0.0
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.episode_length_s = END_S + 30.0
    env_cfg.events.push_robot = None
    drive_arms = args.arms == "rest"
    if drive_arms and hasattr(env_cfg.commands, "arm_pose"):
        env_cfg.commands.arm_pose = None
    if args.noise:
        env_cfg.observations.policy.enable_corruption = True
    if args.video:
        pose = env_cfg.events.reset_base.params["pose_range"]
        pose["x"], pose["y"], pose["yaw"] = (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)
        v = env_cfg.viewer
        v.origin_type, v.asset_name, v.env_index = "asset_root", "robot", 0
        v.eye, v.lookat = (1.0, -3.6, 0.35), (0.3, 0.0, -0.1)
        v.resolution = (1280, 720)

    total_steps = int(round((args.max_seconds or END_S) / (env_cfg.decimation * env_cfg.sim.dt)))
    raw_dir = os.path.join(args.out_dir, "_raw")
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        shutil.rmtree(raw_dir, ignore_errors=True)
        os.makedirs(raw_dir)
        env = gym.wrappers.RecordVideo(env, video_folder=raw_dir, step_trigger=lambda s: s == 0,
                                       video_length=total_steps, disable_logger=True)

    walk_net, stand_net = load_actor(args.walk_checkpoint), load_actor(args.stand_checkpoint)
    base = env.unwrapped
    robot = base.scene["robot"]
    term = base.command_manager.get_term("base_velocity")
    dt = base.step_dt
    leg_ids = robot.find_joints(list(LEG_JOINT_NAMES), preserve_order=True)[0]
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    arm_default = robot.data.default_joint_pos[:, arm_ids].clone()
    leg_default = robot.data.default_joint_pos[:, leg_ids].clone()
    foot_ids = robot.find_bodies(".*_ankle_roll_link")[0]
    sensor = base.scene["contact_forces"]
    sensor_foot_ids = sensor.find_bodies(".*_ankle_roll_link")[0]
    min_air = max(int(MIN_AIR_S / dt), 1)
    cmd_scale = torch.tensor([2.0, 2.0, 0.25], device=device)

    def stand_obs(cmd_t):
        """v1's 47-value observation, built as its own task built it (see g1_rl/walk_env_cfg.py)."""
        w = robot.data.root_ang_vel_b
        g = robot.data.projected_gravity_b
        q = robot.data.joint_pos[:, leg_ids] - leg_default
        qd = robot.data.joint_vel[:, leg_ids]
        if args.noise:
            u = lambda x, a: x + (torch.rand_like(x) * 2 - 1) * a  # noqa: E731
            w, g, q, qd = u(w, 0.2), u(g, 0.05), u(q, 0.01), u(qd, 1.5)
        last = base.action_manager.action[:, :12]
        t = base.episode_length_buf.float() * dt
        ph = 2 * math.pi * ((t % 0.8) / 0.8)
        return torch.cat((w * 0.25, g, cmd_t * cmd_scale, q, qd * 0.05, last, torch.stack((ph.sin(), ph.cos()), 1)), dim=1)

    def flat(x):
        return torch.zeros(n, device=device) if x is None else x

    obs, _ = env.reset()
    z = lambda *s: torch.zeros(*s, device=device)  # noqa: E731
    alive = torch.ones(n, dtype=torch.bool, device=device)
    fall = {"t": torch.full((n,), float("nan"), device=device), "mode": torch.full((n,), -1, device=device),
            "seg": torch.full((n,), -1, device=device, dtype=torch.long)}
    cmd_shaped = z(3)
    mode, up, down, w_walk = STAND, 0, 0, 0.0
    n_up, n_down = int(args.debounce_walk / dt), int(args.debounce_stand / dt)
    events, switches_in_seg = [], {}
    pending = []  # transient trackers after a switch
    prev_action = z(n, 15)
    air_run = z(n, 2)
    seg_acc = {}  # per-segment accumulators

    def acc_for(seg):
        if seg not in seg_acc:
            seg_acc[seg] = {"count": z(n), "sp2": z(n), "tilt2": z(n), "tilt_max": z(n), "lifts": z(n),
                            "sum_v": z(n, 3), "start_xy": None, "end_xy": None}
        return seg_acc[seg]

    shove_steps = {int(round(t / dt)): m for t, m in SHOVES}
    for i in range(total_steps):
        t = i * dt
        cmd_in, kind, label, t0, hard = command_at(t)
        target = torch.tensor(cmd_in, device=device)
        if hard and abs(t - t0) < dt / 2:
            cmd_shaped = target.clone()
        else:
            lim = torch.tensor([args.ramp_lin, args.ramp_lin, args.ramp_yaw], device=device) * dt
            cmd_shaped = cmd_shaped + (target - cmd_shaped).clamp(-lim, lim)
        term.vel_command_b[:] = cmd_shaped
        if drive_arms:
            robot.set_joint_position_target(arm_default, joint_ids=arm_ids)
        if i in shove_steps:
            ang = torch.rand(n, device=device) * 2 * math.pi
            vel = robot.data.root_vel_w.clone()
            vel[:, 0] += shove_steps[i] * ang.cos()
            vel[:, 1] += shove_steps[i] * ang.sin()
            robot.write_root_velocity_to_sim(vel)
            events.append({"t": t, "type": "shove", "magnitude": shove_steps[i]})

        # selector
        s = float(cmd_shaped[:2].norm() + cmd_shaped[2].abs())
        if mode == STAND:
            up = up + 1 if s > args.enter_walk else 0
            if up >= n_up:
                mode, up, down = WALK, 0, 0
                events.append({"t": t, "type": "switch", "to": "walk"})
        else:
            down = down + 1 if s < args.exit_walk else 0
            if down >= n_down:
                mode, up, down = STAND, 0, 0
                events.append({"t": t, "type": "switch", "to": "stand"})
        if events and events[-1]["type"] == "switch" and events[-1]["t"] == t:
            seg = segment_of(t)
            switches_in_seg[seg] = switches_in_seg.get(seg, 0) + 1
            pending.append({"t": t, "to": events[-1]["to"], "until": i + int(1.5 / dt), "tilt": z(n), "speed": z(n), "seg": seg})
        step_w = dt / max(args.blend, dt)
        w_walk = min(1.0, w_walk + step_w) if mode == WALK else max(0.0, w_walk - step_w)

        a_walk = walk_net(obs["policy"])
        a_stand = torch.cat((stand_net(stand_obs(cmd_shaped.expand(n, 3))), z(n, 3)), dim=1)
        action = w_walk * a_walk + (1.0 - w_walk) * a_stand
        obs, _, terminated, truncated, _ = env.step(action)

        died = terminated.bool() & alive
        if died.any():
            fall["t"][died] = t
            fall["mode"][died] = mode
            fall["seg"][died] = segment_of(t)
            alive &= ~died
        w = alive.float()
        speed = robot.data.root_lin_vel_w[:, :2].norm(dim=1)
        tilt = torch.rad2deg(torch.asin(robot.data.projected_gravity_b[:, :2].norm(dim=1).clamp(max=1.0)))
        contact = sensor.data.net_forces_w[:, sensor_foot_ids, :].norm(dim=-1) > CONTACT_N
        air_run = torch.where(~contact, air_run + 1, torch.zeros_like(air_run))
        for p in list(pending):
            p["tilt"] = torch.maximum(p["tilt"], tilt * w)
            p["speed"] = torch.maximum(p["speed"], speed * w)
            if i >= p["until"]:
                pending.remove(p)
                p["alive"] = alive.clone()
                events.append({"t": p["t"], "type": "transient", "to": p["to"], "seg": p["seg"],
                               "peak_tilt_deg": {"mean": p["tilt"][p["alive"]].mean().item() if p["alive"].any() else None,
                                                 "max": p["tilt"][p["alive"]].max().item() if p["alive"].any() else None},
                               "peak_speed_ms": {"mean": p["speed"][p["alive"]].mean().item() if p["alive"].any() else None,
                                                 "max": p["speed"][p["alive"]].max().item() if p["alive"].any() else None}})
        # per-segment measurements after the settle time
        seg = segment_of(t)
        if t - t0 >= SETTLE_S[kind]:
            a = acc_for(seg)
            xy = robot.data.root_pos_w[:, :2].clone()
            if a["start_xy"] is None:
                a["start_xy"] = xy
            a["end_xy"] = xy
            a["count"] += w
            a["sp2"] += w * speed**2
            a["tilt2"] += w * tilt**2
            a["tilt_max"] = torch.maximum(a["tilt_max"], tilt * w)
            a["lifts"] += w * (air_run == min_air).float().sum(1)
            a["sum_v"] += w.unsqueeze(1) * torch.stack(
                (robot.data.root_lin_vel_b[:, 0], robot.data.root_lin_vel_b[:, 1], robot.data.root_ang_vel_w[:, 2]), dim=1)
        prev_action = action

    # ------------------------------------------------------------------ report
    rows = []
    for si, row in enumerate(SCRIPT):
        ts, kind, cmd, label, hard = row
        te = SCRIPT[si + 1][0] if si + 1 < len(SCRIPT) else END_S
        fell = fall["seg"] == si
        entry = {"segment": si, "t0": ts, "t1": te, "kind": kind, "cmd": cmd, "label": label, "hard": hard,
                 "fell": int(fell.sum().item()), "fell_in_stand_mode": int((fell & (fall["mode"] == STAND)).sum().item()),
                 "fell_in_walk_mode": int((fell & (fall["mode"] == WALK)).sum().item()),
                 "switches": switches_in_seg.get(si, 0)}
        a = seg_acc.get(si)
        if a is not None and a["count"].max() > 0:
            ok = a["count"] > 0.9 * a["count"].max()
            c = a["count"].clamp(min=1.0)
            secs = c * dt
            entry["drift_cm"] = ((a["end_xy"] - a["start_xy"]).norm(dim=1) * 100)[ok].mean().item() if ok.any() else None
            entry["speed_rms_mms"] = ((a["sp2"] / c).sqrt() * 1000)[ok].mean().item() if ok.any() else None
            entry["tilt_rms_deg"] = ((a["tilt2"] / c).sqrt())[ok].mean().item() if ok.any() else None
            entry["tilt_max_deg"] = a["tilt_max"][ok].max().item() if ok.any() else None
            entry["lifts_per_min"] = (a["lifts"] * 60.0 / secs)[ok].mean().item() if ok.any() else None
            entry["achieved"] = (a["sum_v"] / c.unsqueeze(1))[ok].mean(0).tolist() if ok.any() else None
        rows.append(entry)

    transients = [e for e in events if e["type"] == "transient"]
    summary = {
        "label": f"{args.walk_label} walks / {args.stand_label} stands", "noise": args.noise, "arms": args.arms, "envs": n,
        "thresholds": {"enter_walk": args.enter_walk, "exit_walk": args.exit_walk, "blend_s": args.blend,
                       "ramp_lin": args.ramp_lin, "ramp_yaw": args.ramp_yaw},
        "survived_total": alive.float().mean().item(), "segments": rows, "events": events,
    }
    print(f"\n[switch] {summary['label']}  noise={'on' if args.noise else 'off'}  arms={args.arms}  {n} envs  "
          f"survived whole script: {summary['survived_total']*100:.0f}%")
    print(f"[switch] walk when |cmd| > {args.enter_walk} (>{args.debounce_walk}s), stand when < {args.exit_walk} (>{args.debounce_stand}s), "
          f"blend {args.blend}s, ramp {args.ramp_lin} m/s^2")
    print(f"{'t0-t1':>9} {'kind':>11} {'command':>17} {'fell':>5} {'sw':>3}   {'drift cm':>8} {'speed':>6} {'tilt':>5} {'tmax':>5} {'lift/m':>6}   achieved (vx, vy, wz)")
    for r in rows:
        f = lambda k, fmt: (fmt % r[k]) if r.get(k) is not None else "-"  # noqa: E731
        ach = "(%.2f, %.2f, %.2f)" % tuple(r["achieved"]) if r.get("achieved") and r["kind"] in ("walk", "hover_walk") else ""
        c = r["cmd"]
        print(f"{r['t0']:4.0f}-{r['t1']:<4.0f} {r['kind']:>11} ({c[0]:4.2f},{c[1]:4.2f},{c[2]:4.2f}) "
              f"{r['fell']:5d} {r['switches']:3d}   {f('drift_cm','%8.1f'):>8} {f('speed_rms_mms','%6.0f'):>6} "
              f"{f('tilt_rms_deg','%5.2f'):>5} {f('tilt_max_deg','%5.1f'):>5} {f('lifts_per_min','%6.1f'):>6}   {ach}")
    print("[switch] handovers (peak over the 1.5 s after the switch, alive envs)")
    for e in transients:
        pt, ps = e["peak_tilt_deg"], e["peak_speed_ms"]
        if pt["mean"] is not None:
            print(f"   t={e['t']:5.1f}s -> {e['to']:5s}  peak tilt mean {pt['mean']:.1f}° (max {pt['max']:.1f}°)  "
                  f"peak speed mean {ps['mean']:.2f} (max {ps['max']:.2f}) m/s")
    print(f"[switch] shoves at {[t for t, _ in SHOVES]} s, {SHOVES[0][1]} m/s")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[switch] wrote {args.out}")

    env.close()  # finalises the recording
    if args.video:
        raws = sorted(glob.glob(os.path.join(raw_dir, "*.mp4")))
        if raws:
            label_video(raws[0], os.path.join(args.out_dir, "switch_test.mp4"), events)
        else:
            print("[switch] no recording produced", file=sys.stderr)
    app.close()  # exits the process, so it must come last


def label_video(src, dst, events):
    """30 fps h264 with the title, the scripted command, the ACTIVE policy and shove flashes."""
    font = next(iter(glob.glob("/usr/share/fonts/**/DejaVuSans-Bold.ttf", recursive=True)), None)
    ff = shutil.which("ffmpeg")
    if not (ff and font):
        shutil.copy(src, dst)
        print(f"[switch] ffmpeg/font missing, wrote unlabelled {dst}")
        return
    box = f"fontfile={font}:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10"
    title = f"Switching controller | {args.walk_label} walks, {args.stand_label} stands"
    filters = ["fps=30", f"drawtext={box}:fontsize=26:x=24:y=20:text='{title}'"]
    for si, (ts, kind, cmd, label, hard) in enumerate(SCRIPT):
        te = SCRIPT[si + 1][0] if si + 1 < len(SCRIPT) else END_S
        text = label.replace(":", "\\:").replace("'", "").replace("(", "").replace(")", "")
        filters.append(f"drawtext={box}:fontsize=32:x=24:y=h-70:enable='between(t,{ts:.2f},{te:.2f})':text='{text}'")
    sw = [(0.0, "stand")] + [(e["t"], e["to"]) for e in events if e["type"] == "switch"]
    for k, (ts, to) in enumerate(sw):
        te = sw[k + 1][0] if k + 1 < len(sw) else END_S
        name, color = (f"WALKING policy {args.walk_label}", "0x2e9e4f") if to == "walk" else (f"STANDING policy {args.stand_label}", "0x2b6cb0")
        filters.append(f"drawtext=fontfile={font}:fontcolor=white:box=1:boxcolor={color}@0.85:boxborderw=12:fontsize=30:"
                       f"x=w-text_w-24:y=20:enable='between(t,{ts:.2f},{te:.2f})':text='{name}'")
    for e in events:
        if e["type"] == "shove":
            filters.append(f"drawtext=fontfile={font}:fontcolor=red:fontsize=64:x=(w-text_w)/2:y=120:"
                           f"enable='between(t,{e['t']:.2f},{e['t'] + 0.8:.2f})':text='PUSH!'")
    cmd = [ff, "-y", "-loglevel", "error", "-i", src, "-vf", ",".join(filters),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", dst]
    subprocess.run(cmd, check=True)
    print(f"[switch] wrote {dst}")


if __name__ == "__main__":
    main()
