#!/usr/bin/env python3
"""Record a trained g1_rl walking policy to an mp4 (follow camera, on-screen labels).

Scenarios (a scripted sequence of velocity commands, one robot, flat ground):
  showcase   stand -> walk 0.5 -> 1.0 m/s -> walk-and-turn left/right -> slow -> stop
  push       walk at 0.5 m/s while being shoved sideways and forward, then a shove while standing

    python scripts/record_video.py --checkpoint <run>/model_1499.pt --scenario showcase
    python scripts/record_video.py --checkpoint <run>/model_1499.pt --scenario push

Writes <out-dir>/<scenario>.mp4 (30 fps, 1280x720 by default) plus the raw
recording. Needs the isaaclab conda env and ffmpeg on PATH for the labels.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="G1-Walk-Flat-Inspire-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--scenario", choices=["showcase", "push"], default="showcase")
parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "videos"))
parser.add_argument("--title", default="Unitree G1 | RL walking policy | Isaac Lab + PPO")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import g1_rl  # noqa: F401

# (duration_s, (vx, vy, yaw_rate), label)
SCENARIOS = {
    "showcase": [
        (2.5, (0.0, 0.0, 0.0), "Standing"),
        (4.0, (0.5, 0.0, 0.0), "Walking forward  0.5 m/s"),
        (4.0, (1.0, 0.0, 0.0), "Walking forward  1.0 m/s"),
        (5.0, (0.5, 0.0, 0.6), "Walking + turning left"),
        (5.0, (0.5, 0.0, -0.6), "Walking + turning right"),
        (3.0, (0.3, 0.0, 0.0), "Slow walk  0.3 m/s"),
        (2.5, (0.0, 0.0, 0.0), "Stop"),
    ],
    "push": [
        (12.0, (0.5, 0.0, 0.0), "Walking forward  0.5 m/s"),
        (3.0, (0.0, 0.0, 0.0), "Standing"),
    ],
}
# scripted shoves: (time_s, (dvx, dvy) added to the base velocity in m/s)
PUSHES = {
    "showcase": [],
    "push": [(3.0, (0.0, 0.8)), (6.0, (0.0, -0.8)), (9.0, (0.8, 0.0)), (13.0, (0.0, 0.6))],
}


@torch.inference_mode()
def main():
    device = "cuda:0"
    segments = SCENARIOS[args.scenario]
    pushes = PUSHES[args.scenario]
    total_s = sum(d for d, _, _ in segments)

    env_cfg = parse_env_cfg(args.task, device=device, num_envs=1)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    # Isaac Lab 0.54 / rsl-rl >= 4 expects a different runner-config layout than 0.48 / rsl-rl 3.x;
    # Isaac Lab ships a converter for it (absent on the older install, where the config is used as-is).
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
    env_cfg.episode_length_s = total_s + 5.0
    env_cfg.events.push_robot = None
    # start at the origin facing +x so the side-on camera frames the gait
    pose = env_cfg.events.reset_base.params["pose_range"]
    pose["x"], pose["y"], pose["yaw"] = (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)

    v = env_cfg.viewer
    v.origin_type, v.asset_name, v.env_index = "asset_root", "robot", 0
    # offsets are relative to the robot's root (pelvis, ~0.8 m up), not the world origin
    v.eye, v.lookat = (1.0, -3.6, 0.35), (0.3, 0.0, -0.1)
    v.resolution = (args.width, args.height)

    steps = int(round(total_s / (env_cfg.decimation * env_cfg.sim.dt)))
    raw_dir = os.path.join(args.out_dir, f"_raw_{args.scenario}")
    shutil.rmtree(raw_dir, ignore_errors=True)
    os.makedirs(raw_dir)

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env, video_folder=raw_dir, step_trigger=lambda s: s == 0, video_length=steps, disable_logger=True
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(os.path.abspath(args.checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base = env.unwrapped
    robot = base.scene["robot"]
    term = base.command_manager.get_term("base_velocity")
    dt = base.step_dt

    # command timeline in steps
    timeline, t0 = [], 0.0
    for dur, cmd, label in segments:
        timeline.append((t0, t0 + dur, cmd, label))
        t0 += dur
    push_steps = {int(round(t / dt)): d for t, d in pushes}

    env.reset()
    obs = env.get_observations()
    fell_at = None
    for i in range(steps):
        t = i * dt
        cmd = timeline[-1][2]
        for a, b, c, _ in timeline:
            if a <= t < b:
                cmd = c
        term.vel_command_b[:] = torch.tensor(cmd, device=device)
        if i in push_steps:
            vel = robot.data.root_vel_w.clone()
            vel[:, 0] += push_steps[i][0]
            vel[:, 1] += push_steps[i][1]
            robot.write_root_velocity_to_sim(vel)
        obs, _, dones, _ = env.step(policy(obs))
        if fell_at is None and bool(dones.any()) and i < steps - 1:
            fell_at = t
    print(f"[video] {steps} steps ({total_s:.1f}s); first early termination: {fell_at}")
    env.close()  # finalises the raw recording

    raws = sorted(glob.glob(os.path.join(raw_dir, "*.mp4")))
    if not raws:
        print("[video] no recording produced", file=sys.stderr)
    else:
        label_video(raws[0], os.path.join(args.out_dir, f"{args.scenario}.mp4"), timeline, pushes)
    app.close()  # exits the process, so it must come last


def label_video(src, dst, timeline, pushes):
    """Re-encode to 30 fps h264 and burn in a title, the active command and push flashes."""
    font = next((f for f in glob.glob("/usr/share/fonts/**/DejaVuSans-Bold.ttf", recursive=True)), None)
    ff = shutil.which("ffmpeg")
    if not (ff and font):
        shutil.copy(src, dst)
        print(f"[video] ffmpeg/font missing, wrote unlabelled {dst}")
        return
    ftxt = f"fontfile={font}:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10"
    filters = [f"fps=30", f"drawtext={ftxt}:fontsize=26:x=24:y=20:text='{args.title}'"]
    for a, b, _, label in timeline:
        text = label.replace(":", "\\:").replace("'", "")
        filters.append(f"drawtext={ftxt}:fontsize=34:x=24:y=h-70:enable='between(t,{a:.2f},{b:.2f})':text='{text}'")
    for t, _ in pushes:
        filters.append(
            f"drawtext=fontfile={font}:fontcolor=red:fontsize=64:x=(w-text_w)/2:y=120:"
            f"enable='between(t,{t:.2f},{t + 0.8:.2f})':text='PUSH!'"
        )
    cmd = [ff, "-y", "-loglevel", "error", "-i", src, "-vf", ",".join(filters),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium", dst]
    subprocess.run(cmd, check=True)
    print(f"[video] wrote {dst}")


if __name__ == "__main__":
    main()
