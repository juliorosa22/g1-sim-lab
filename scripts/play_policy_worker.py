#!/usr/bin/env python3
"""Simulation worker for scripts/play_policy.py: one policy, one Isaac Lab env, a small HTTP API on localhost.

Not meant to be run by hand (play_policy.py starts and stops it so the policy can be swapped live):

    python scripts/play_policy_worker.py --task <Play task> --checkpoint <model.pt> --label v9e --port 18765

Every control step it (1) moves the arm joints toward the targets set through the API (rate limited), (2) writes the velocity command
(optionally ramped at the deployed controller's limits and deadbanded at 0.05), (3) runs the policy and steps the simulation, (4) every
few steps renders a camera frame, which the API serves as MJPEG. Policies are loaded with g1_rl.actor_io (either rsl-rl checkpoint format).
"""
import argparse
import io
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--label", default="policy")
parser.add_argument("--port", type=int, default=18765)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=600)
parser.add_argument("--stream-every", type=int, default=2, help="render one frame every N control steps (50 Hz / N)")
parser.add_argument("--jpeg-quality", type=int, default=80)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

app = AppLauncher(args).app

import gymnasium as gym
import torch
from PIL import Image

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import g1_rl  # noqa: F401
from g1_rl.actor_io import load_actor
from g1_rl.body_env_cfg import ARM_POSE_RANGES
from g1_rl.robot_cfg import ARM_JOINT_NAMES

RECOMMENDED_FRACTION = 0.6  # of the training arm range around the default pose
CONTACT_N = 10.0
MIN_AIR_S = 0.1
DEADBAND = 0.05  # same as the deployed controller / MixedVelocityCommand
CAMERAS = {  # eye, lookat: relative to the robot's pelvis
    "front": ((2.2, -1.1, 0.45), (0.0, 0.0, 0.1)),
    "side": ((0.3, -2.8, 0.4), (0.0, 0.0, 0.1)),
    "back": ((-2.2, -1.1, 0.5), (0.0, 0.0, 0.1)),
    "close": ((1.4, -0.7, 0.55), (0.0, 0.0, 0.2)),
    "far": ((0.5, -5.5, 1.2), (0.5, 0.0, 0.0)),
}


class Shared:
    """State shared between the simulation loop (main thread) and the HTTP handlers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame_cond = threading.Condition()
        self.jpeg = b""
        self.frame_id = 0
        self.target = {}  # arm joint -> absolute target (rad)
        self.speed = 1.0  # rad/s arm rate limit
        self.clamp = True
        self.random_reach = False
        self.random_speed = 1.0
        self.cmd_target = [0.0, 0.0, 0.0]
        self.cmd_applied = [0.0, 0.0, 0.0]
        self.ramp = False
        self.deadband = False
        self.requests = []
        self.telemetry = {}
        self.joints = []


shared = Shared()


def encode_jpeg(frame):
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=args.jpeg_quality)
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/state":
            with shared.lock:
                joints = [dict(j, target=shared.target[j["name"]]) for j in shared.joints]
                body = json.dumps({
                    "label": args.label, "joints": joints, "telemetry": shared.telemetry, "random_reach": shared.random_reach,
                    "clamp": shared.clamp, "speed": shared.speed,
                    "command": {"target": shared.cmd_target, "applied": shared.cmd_applied, "ramp": shared.ramp,
                                "deadband": shared.deadband},
                })
            return self._send(200, body)
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            try:
                while True:
                    with shared.frame_cond:
                        shared.frame_cond.wait_for(lambda: shared.frame_id != last, timeout=2.0)
                        data, last = shared.jpeg, shared.frame_id
                    if not data:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(data))
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self._send(404, "not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._send(400, "bad json", "text/plain")
        with shared.lock:
            if self.path == "/target":
                for name, v in body.get("values", {}).items():
                    if name in shared.target:
                        shared.target[name] = float(v)
                shared.random_reach = False
            elif self.path == "/config":
                if "speed" in body:
                    shared.speed = max(0.05, float(body["speed"]))
                if "random_speed" in body:
                    shared.random_speed = max(0.05, float(body["random_speed"]))
                if "clamp" in body:
                    shared.clamp = bool(body["clamp"])
                    if shared.clamp:  # re-locking pulls every stored target back into the safe band
                        for j in shared.joints:
                            shared.target[j["name"]] = min(max(shared.target[j["name"]], j["rec_lo"]), j["rec_hi"])
                if "random_reach" in body:
                    shared.random_reach = bool(body["random_reach"])
                if "ramp" in body:
                    shared.ramp = bool(body["ramp"])
                if "deadband" in body:
                    shared.deadband = bool(body["deadband"])
            elif self.path == "/command":
                for i, key in enumerate(("vx", "vy", "wz")):
                    if key in body:
                        shared.cmd_target[i] = float(body[key])
            elif self.path == "/action":
                shared.requests.append((body.get("name"), body))
            else:
                return self._send(404, "not found", "text/plain")
        self._send(200, "{}")


@torch.inference_mode()
def main():
    device = "cuda:0"
    env_cfg = parse_env_cfg(args.task, device=device, num_envs=1)
    env_cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    cmd_cfg = env_cfg.commands.base_velocity
    cmd_cfg.heading_command = False
    if hasattr(cmd_cfg, "mixed_sampling"):
        cmd_cfg.mixed_sampling = False
    cmd_cfg.rel_standing_envs = 0.0
    cmd_cfg.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.episode_length_s = 1.0e6
    env_cfg.events.push_robot = None
    if hasattr(env_cfg.commands, "arm_pose"):
        env_cfg.commands.arm_pose = None  # the arms are driven from the web page
    env_cfg.observations.policy.enable_corruption = False

    v = env_cfg.viewer
    v.origin_type, v.asset_name, v.env_index = "asset_root", "robot", 0
    v.eye, v.lookat = CAMERAS["front"]
    v.resolution = (args.width, args.height)

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    actor = load_actor(os.path.abspath(args.checkpoint), device)

    base = env.unwrapped
    robot = base.scene["robot"]
    term = base.command_manager.get_term("base_velocity")
    dt = base.step_dt
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    default = robot.data.default_joint_pos[0, arm_ids].clone()
    soft = robot.data.soft_joint_pos_limits[0, arm_ids].clone()
    lo_full = torch.tensor([ARM_POSE_RANGES[n][0] for n in ARM_JOINT_NAMES], device=device)
    hi_full = torch.tensor([ARM_POSE_RANGES[n][1] for n in ARM_JOINT_NAMES], device=device)
    rec_lo = default + RECOMMENDED_FRACTION * (lo_full - default)
    rec_hi = default + RECOMMENDED_FRACTION * (hi_full - default)
    sensor = base.scene["contact_forces"]
    foot_ids = sensor.find_bodies(".*_ankle_roll_link")[0]

    with shared.lock:
        for i, n in enumerate(ARM_JOINT_NAMES):
            shared.target[n] = float(default[i])
            shared.joints.append({"name": n, "lo": float(soft[i, 0]), "hi": float(soft[i, 1]), "default": float(default[i]),
                                  "rec_lo": float(rec_lo[i]), "rec_hi": float(rec_hi[i])})

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[worker] {args.label}: serving on 127.0.0.1:{args.port} (task {args.task}, checkpoint {args.checkpoint})", flush=True)

    cur = default.clone()
    goal = default.clone()
    cmd = [0.0, 0.0, 0.0]
    hold = 0.0
    obs = env.get_observations()
    start_xy = robot.data.root_pos_w[0, :2].clone()
    falls, lifts, air_run = 0, 0, torch.zeros(2, device=device)
    t_wall, steps_done, fps, step = time.time(), 0, 0.0, 0
    next_t = time.time()
    while app.is_running():
        with shared.lock:
            reqs, shared.requests = shared.requests, []
            speed, clamp = shared.speed, shared.clamp
            random_reach, rspeed = shared.random_reach, shared.random_speed
            tgt = torch.tensor([shared.target[n] for n in ARM_JOINT_NAMES], device=device)
            cmd_target, ramp, deadband = list(shared.cmd_target), shared.ramp, shared.deadband
        for name, body in reqs:
            if name == "reset":
                env.reset()
                obs = env.get_observations()
                cur = default.clone()
                cmd = [0.0, 0.0, 0.0]
                start_xy = robot.data.root_pos_w[0, :2].clone()
                falls, lifts = 0, 0
            elif name == "push":
                ang = float(torch.rand(1)) * 2 * math.pi
                vel = robot.data.root_vel_w.clone()
                vel[:, 0] += float(body.get("mag", 0.3)) * math.cos(ang)
                vel[:, 1] += float(body.get("mag", 0.3)) * math.sin(ang)
                robot.write_root_velocity_to_sim(vel)
            elif name == "camera" and body.get("view") in CAMERAS:
                eye, lookat = CAMERAS[body["view"]]
                try:
                    base.viewport_camera_controller.update_view_location(eye=eye, lookat=lookat)
                except Exception as exc:  # noqa: BLE001
                    print("[worker] camera change failed:", exc, flush=True)

        # ---- arms: random reaches inside the safe band, or the slider targets
        if random_reach:
            if hold > 0:
                hold -= dt
            elif (goal - cur).abs().max() < 1e-3:
                goal = rec_lo + torch.rand_like(rec_lo) * (rec_hi - rec_lo)
                hold = 0.3 + float(torch.rand(1)) * 0.7
            tgt = goal
            with shared.lock:
                for i, n in enumerate(ARM_JOINT_NAMES):
                    shared.target[n] = float(goal[i])
            rate = rspeed
        else:
            rate = speed
        if clamp:
            tgt = torch.maximum(torch.minimum(tgt, rec_hi), rec_lo)
        tgt = torch.maximum(torch.minimum(tgt, soft[:, 1]), soft[:, 0])
        cur = cur + (tgt - cur).clamp(-rate * dt, rate * dt)
        robot.set_joint_position_target(cur.unsqueeze(0), joint_ids=arm_ids)

        # ---- velocity command: deadband, then the controller's acceleration limits
        goal_cmd = list(cmd_target)
        if deadband:
            if math.hypot(goal_cmd[0], goal_cmd[1]) < DEADBAND:
                goal_cmd[0] = goal_cmd[1] = 0.0
            if abs(goal_cmd[2]) < DEADBAND:
                goal_cmd[2] = 0.0
        if ramp:
            dlin, dyaw = 1.0 * dt, 2.0 * dt
            dx, dy = goal_cmd[0] - cmd[0], goal_cmd[1] - cmd[1]
            n = math.hypot(dx, dy)
            if n > dlin:
                dx, dy = dx / n * dlin, dy / n * dlin
            cmd = [cmd[0] + dx, cmd[1] + dy, cmd[2] + max(-dyaw, min(dyaw, goal_cmd[2] - cmd[2]))]
        else:
            cmd = goal_cmd
        term.vel_command_b[:] = torch.tensor(cmd, device=device)

        # ---- policy and physics
        action = actor(obs["policy"])
        obs, _, dones, extras = env.step(action)
        died = bool(dones.any()) and not bool(extras.get("time_outs", torch.zeros_like(dones)).bool().any())
        if died:
            falls += 1
            cur = robot.data.joint_pos[0, arm_ids].clone()  # the robot respawned: continue from where the arms really are
            cmd = [0.0, 0.0, 0.0]
            start_xy = robot.data.root_pos_w[0, :2].clone()
        contact = sensor.data.net_forces_w[0, foot_ids, :].norm(dim=-1) > CONTACT_N
        air_run = torch.where(~contact, air_run + 1, torch.zeros_like(air_run))
        lifts += int((air_run == max(int(MIN_AIR_S / dt), 1)).sum())

        step += 1
        steps_done += 1
        if time.time() - t_wall > 1.0:
            fps, steps_done, t_wall = steps_done / (time.time() - t_wall), 0, time.time()
        if step % args.stream_every == 0:
            frame = base.render()
            if frame is not None:
                data = encode_jpeg(frame)
                with shared.frame_cond:
                    shared.jpeg, shared.frame_id = data, shared.frame_id + 1
                    shared.frame_cond.notify_all()
        g = robot.data.projected_gravity_b[0, :2].norm().clamp(max=1.0)
        vb = robot.data.root_lin_vel_b[0]
        with shared.lock:
            shared.cmd_applied = list(cmd)
            shared.telemetry = {
                "z": float(robot.data.root_pos_w[0, 2]),
                "tilt": float(torch.rad2deg(torch.asin(g))),
                "dist_cm": float((robot.data.root_pos_w[0, :2] - start_xy).norm() * 100),
                "speed": float(robot.data.root_lin_vel_w[0, :2].norm()),
                "vx": float(vb[0]), "vy": float(vb[1]), "wz": float(robot.data.root_ang_vel_b[0, 2]),
                "lifts": lifts, "falls": falls, "fps": fps,
            }
        next_t += dt  # real-time pacing at the control rate when the machine is fast enough
        delay = next_t - time.time()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.time()
    server.shutdown()
    env.close()
    app.close()


if __name__ == "__main__":
    main()
