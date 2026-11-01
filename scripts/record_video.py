#!/usr/bin/env python3
"""Record a trained g1_rl walking policy to an mp4 (follow camera, on-screen labels).

Scenarios (a scripted sequence of velocity commands, one robot, flat ground):
  showcase   stand -> walk 0.5 -> 1.0 m/s -> walk-and-turn left/right -> slow -> stop
  push       walk at 0.5 m/s while being shoved sideways and forward, then a shove while standing
  stand      20 s at zero command, two shoves from 11 s: stands still? steps in place? recovers?

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
parser.add_argument("--scenario", choices=["showcase", "push", "stand", "arms_mimic", "compare", "walkstop", "turnsettle"], default="showcase")
parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "videos"))
parser.add_argument("--title", default="Unitree G1 | RL walking policy | Isaac Lab + PPO")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--noise", action="store_true", help="enable observation noise (the Play tasks disable it): the realistic condition")
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
    "arms_mimic": [  # zero command; the arms follow random human-like reaches at increasing speed (what a pose-imitation pipeline sends)
        (3.0, (0.0, 0.0, 0.0), "Standing, arms at rest"),
        (10.0, (0.0, 0.0, 0.0), "Zero command | arm reaches at 0.5 rad/s"),
        (10.0, (0.0, 0.0, 0.0), "Zero command | arm reaches at 1.0 rad/s"),
        (10.0, (0.0, 0.0, 0.0), "Zero command | arm reaches at 1.5 rad/s"),
        (10.0, (0.0, 0.0, 0.0), "Zero command | arm reaches at 2.0 rad/s"),
    ],
    "compare": [  # the commands the v9 variants differ on: stillness, slow motion, sideways, turning
        (3.0, (0.0, 0.0, 0.0), "Standing (zero command)"),
        (4.0, (0.2, 0.0, 0.0), "Slow forward  0.2 m/s"),
        (4.0, (0.5, 0.0, 0.0), "Forward  0.5 m/s"),
        (3.0, (0.0, 0.0, 0.0), "Stop (zero command)"),
        (4.0, (0.0, 0.0, 0.3), "Turn in place  0.3 rad/s"),
        (4.0, (0.0, 0.0, 0.5), "Turn in place  0.5 rad/s"),
        (4.0, (0.0, 0.2, 0.0), "Sideways  0.2 m/s"),
        (4.0, (0.5, 0.0, 0.5), "Forward 0.5 m/s + turn 0.5 rad/s"),
        (3.0, (0.0, 0.0, 0.0), "Stop (zero command)"),
    ],
    "turnsettle": [  # v11: yaw-only commands, each followed by zero command: does it turn, and does it settle planted again?
        (3.0, (0.0, 0.0, 0.0), "Standing (zero command)"),
        (3.0, (0.0, 0.0, 0.3), "Turn in place  0.3 rad/s"),
        (5.0, (0.0, 0.0, 0.0), "Zero command: settles"),
        (3.0, (0.0, 0.0, 0.5), "Turn in place  0.5 rad/s"),
        (5.0, (0.0, 0.0, 0.0), "Zero command: settles"),
        (3.0, (0.0, 0.0, -0.5), "Turn in place  -0.5 rad/s (other way)"),
        (5.0, (0.0, 0.0, 0.0), "Zero command: settles"),
        (3.0, (0.0, 0.0, 1.0), "Turn in place  1.0 rad/s"),
        (5.0, (0.0, 0.0, 0.0), "Zero command: settles"),
    ],
    "walkstop": [  # a walker alone: steady walking at several speeds, each followed by a stop (the command is ramped down like the controller does)
        (2.5, (0.0, 0.0, 0.0), "Standing"),
        (5.0, (0.4, 0.0, 0.0), "Forward  0.4 m/s"),
        (2.0, (0.2, 0.0, 0.0), "Slowing  (command ramps down)"),
        (3.5, (0.0, 0.0, 0.0), "Stop (zero command)"),
        (5.0, (0.8, 0.0, 0.0), "Forward  0.8 m/s"),
        (3.5, (0.4, 0.0, 0.0), "Slowing  (command ramps down)"),
        (3.5, (0.0, 0.0, 0.0), "Stop (zero command)"),
        (5.0, (1.0, 0.0, 0.0), "Forward  1.0 m/s"),
        (4.0, (0.5, 0.0, 0.0), "Slowing  (command ramps down)"),
        (3.5, (0.0, 0.0, 0.0), "Stop (zero command)"),
        (6.0, (0.5, 0.0, 0.5), "Forward 0.5 m/s + turn 0.5 rad/s"),
        (3.5, (0.0, 0.0, 0.0), "Stop (zero command)"),
    ],
    "stand": [  # zero command the whole time: does it stand still, and what does it do when shoved?
        (9.0, (0.0, 0.0, 0.0), "Standing (zero command)"),
        (11.0, (0.0, 0.0, 0.0), "Standing, being pushed"),
    ],
}
# scripted shoves: (time_s, (dvx, dvy) added to the base velocity in m/s)
PUSHES = {
    "turnsettle": [],
    "walkstop": [],
    "arms_mimic": [],
    "compare": [],
    "showcase": [],
    "push": [(3.0, (0.0, 0.8)), (6.0, (0.0, -0.8)), (9.0, (0.8, 0.0)), (13.0, (0.0, 0.6))],
    "stand": [(11.0, (0.0, 0.3)), (16.0, (0.4, 0.0))],
}


# arms_mimic: peak arm speed (rad/s) per segment, None = arms at rest. Targets are random reaches inside 60% of the arm range around
# the default pose with 0.3-1.0 s pauses (the "mimic_<v>" scenarios of scripts/evaluate_standing.py)
ARM_SPEEDS = {"arms_mimic": [None, 0.5, 1.0, 1.5, 2.0]}
MIMIC_RANGE = 0.6
PAUSE_S = (0.3, 1.0)


class MimicArms:
    def __init__(self, robot, device, dt):
        from g1_rl.body_env_cfg import ARM_POSE_RANGES
        from g1_rl.robot_cfg import ARM_JOINT_NAMES

        self.robot, self.dt, self.device = robot, dt, device
        self.ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
        d0 = robot.data.default_joint_pos[0, self.ids].clone()
        lo = torch.tensor([ARM_POSE_RANGES[n][0] for n in ARM_JOINT_NAMES], device=device)
        hi = torch.tensor([ARM_POSE_RANGES[n][1] for n in ARM_JOINT_NAMES], device=device)
        self.glo, self.ghi = d0 + MIMIC_RANGE * (lo - d0), d0 + MIMIC_RANGE * (hi - d0)
        self.default, self.cur, self.goal, self.hold = d0, d0.clone(), d0.clone(), 0.0

    def step(self, speed):
        if speed is not None:
            if self.hold > 0:
                self.hold -= self.dt
            elif (self.goal - self.cur).abs().max() < 1e-3:
                self.goal = self.glo + torch.rand_like(self.glo) * (self.ghi - self.glo)
                self.hold = PAUSE_S[0] + float(torch.rand(1)) * (PAUSE_S[1] - PAUSE_S[0])
            else:
                self.cur = self.cur + (self.goal - self.cur).clamp(-speed * self.dt, speed * self.dt)
        else:
            self.cur = self.cur + (self.default - self.cur).clamp(-0.5 * self.dt, 0.5 * self.dt)
        self.robot.set_joint_position_target(self.cur.unsqueeze(0), joint_ids=self.ids)


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

    if args.scenario in ARM_SPEEDS and hasattr(env_cfg.commands, "arm_pose"):
        env_cfg.commands.arm_pose = None  # the arms are driven by this script
    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.heading_command = False
    cmd_cfg.mixed_sampling = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.episode_length_s = total_s + 5.0
    env_cfg.events.push_robot = None
    if args.noise:
        env_cfg.observations.policy.enable_corruption = True
    # start at the origin facing +x so the side-on camera frames the gait
    pose = env_cfg.events.reset_base.params["pose_range"]
    pose["x"], pose["y"], pose["yaw"] = (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)

    v = env_cfg.viewer
    v.origin_type, v.asset_name, v.env_index = "asset_root", "robot", 0
    # offsets are relative to the robot's root (pelvis, ~0.8 m up), not the world origin
    v.eye, v.lookat = (1.0, -3.6, 0.35), (0.3, 0.0, -0.1)
    if args.scenario in ARM_SPEEDS:
        v.eye, v.lookat = (1.0, -2.6, 0.45), (0.1, 0.0, 0.1)  # closer, to read the arms
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
    arms = MimicArms(robot, device, dt) if args.scenario in ARM_SPEEDS else None

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
        if arms is not None:
            seg = next((k for k, (a, b, _, _) in enumerate(timeline) if a <= t < b), len(timeline) - 1)
            arms.step(ARM_SPEEDS[args.scenario][seg])
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
