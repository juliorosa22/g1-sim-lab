#!/usr/bin/env python3
"""Records raw robot states, shaped commands, observations and policy actions from the simulation, for robot_deploy/tests/contract_test.py.

    python scripts/dump_contract_data.py --checkpoint <v11 model.pt> --out robot_deploy/tests/data/contract_v11.npz

Runs the v11 TRAINING task (so the command ramp is the one used in training) with a few envs, observation noise off and the per-env gyro and
encoder biases set to zero (the controller cannot know them), the arms moving through random reaches, and a scripted sequence of planner
requests (stand, yaw-only at several rates, slow translation, tiny commands inside the deadband, mixed). Per step and env it saves the raw
quantities a real controller would have (gyro, orientation quaternion, 27 joint positions and velocities, previous action, time since start),
the planner's REQUEST, the command the policy saw, the observation Isaac Lab built, the processed joint targets, and the actor's output.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="G1-Walk-Flat-Inspire-V11-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import g1_rl  # noqa: F401
from g1_rl.actor_io import load_actor
from g1_rl.body_env_cfg import ARM_POSE_RANGES
from g1_rl.robot_cfg import ARM_JOINT_NAMES
from g1_rl.v8_env_cfg import CONTROLLED, OBSERVED

DEADBAND = 0.05
REQUESTS = [  # (vx, vy, wz): what a planner might send
    (0.0, 0.0, 0.0), (0.0, 0.0, 0.3), (0.0, 0.0, 0.5), (0.0, 0.0, -0.4), (0.0, 0.0, 1.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0),
    (0.03, 0.03, 0.02), (0.04, 0.0, 0.04), (0.3, 0.0, 0.3), (0.4, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, 0.0, 0.0),
]


@torch.inference_mode()
def main():
    device = "cuda:0"
    torch.manual_seed(args.seed)
    cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
    cfg.seed = args.seed
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    cfg.events.push_robot = None
    cfg.observations.policy.enable_corruption = False
    if hasattr(cfg.commands, "arm_pose"):
        cfg.commands.arm_pose = None
    cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    cfg.episode_length_s = 1.0e6
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=agent_cfg.clip_actions)
    actor = load_actor(os.path.abspath(args.checkpoint), device)
    base = env.unwrapped
    robot = base.scene["robot"]
    term = base.command_manager.get_term("base_velocity")
    act_term = base.action_manager.get_term("joint_pos")
    dt, n = base.step_dt, args.num_envs
    ids = robot.find_joints(list(OBSERVED), preserve_order=True)[0]
    arm_ids = robot.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)[0]
    default27 = robot.data.default_joint_pos[0, ids].cpu().numpy()
    arm_default = robot.data.default_joint_pos[0, arm_ids].clone()
    lo = torch.tensor([ARM_POSE_RANGES[nm][0] for nm in ARM_JOINT_NAMES], device=device)
    hi = torch.tensor([ARM_POSE_RANGES[nm][1] for nm in ARM_JOINT_NAMES], device=device)
    glo, ghi = arm_default + 0.6 * (lo - arm_default), arm_default + 0.6 * (hi - arm_default)

    env.reset()
    obs = env.get_observations()
    base.encoder_bias.zero_()  # the controller cannot know these two; the observation is checked without them
    base.gyro_bias.zero_()
    obs = env.get_observations()
    cur, goal = arm_default.repeat(n, 1), arm_default.repeat(n, 1)
    request_prev = torch.zeros(n, 3, device=device)
    req_idx = torch.zeros(n, dtype=torch.long)
    rec = {k: [] for k in ("obs", "gyro", "quat", "gravity", "q", "dq", "last_action", "t", "cmd_shaped", "request_prev", "processed", "action_next")}
    for step in range(args.steps):
        buf = base.episode_length_buf.clone()
        action_next = actor(obs["policy"])
        rec["obs"].append(obs["policy"].cpu().numpy())
        rec["gyro"].append(robot.data.root_ang_vel_b.cpu().numpy())
        rec["quat"].append(robot.data.root_quat_w.cpu().numpy())
        rec["gravity"].append(robot.data.projected_gravity_b.cpu().numpy())
        rec["q"].append(robot.data.joint_pos[:, ids].cpu().numpy())
        rec["dq"].append(robot.data.joint_vel[:, ids].cpu().numpy())
        rec["last_action"].append(base.action_manager.action.cpu().numpy())
        rec["t"].append((buf.float() * dt).cpu().numpy())
        rec["cmd_shaped"].append(term.vel_command_b.cpu().numpy())
        rec["request_prev"].append(request_prev.cpu().numpy())
        rec["processed"].append(act_term.processed_actions.cpu().numpy())
        rec["action_next"].append(action_next.cpu().numpy())
        # next planner request (changes every 2 s per env, staggered) and the arms' random reaches
        if step % 100 == 0:
            req_idx = torch.randint(0, len(REQUESTS), (n,))
        elif step % 100 == 50:
            req_idx = torch.randint(0, len(REQUESTS), (n,))
        req = torch.tensor([REQUESTS[i] for i in req_idx.tolist()], device=device)
        goal_cmd = req.clone()
        lin_small = goal_cmd[:, :2].norm(dim=1) < DEADBAND
        goal_cmd[lin_small, :2] = 0.0
        goal_cmd[goal_cmd[:, 2].abs() < DEADBAND, 2] = 0.0
        term._target[:] = goal_cmd  # the sim ramps vel_command_b toward this exactly as in training
        term.is_standing_env[:] = False
        request_prev = req
        if step % 75 == 0:
            goal = glo + torch.rand(n, len(ARM_JOINT_NAMES), device=device) * (ghi - glo)
        cur = cur + (goal - cur).clamp(-1.0 * dt, 1.0 * dt)
        robot.set_joint_position_target(cur, joint_ids=arm_ids)
        obs, _, dones, _ = env.step(action_next)
    out = {k: np.stack(v).astype(np.float32) for k, v in rec.items()}
    out["default27"] = default27.astype(np.float32)
    out["joint_names"] = np.array(OBSERVED)
    out["controlled_names"] = np.array(CONTROLLED)
    out["dt"] = np.float32(dt)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"[dump] wrote {args.out}: {args.steps} steps x {n} envs; obs {out['obs'].shape}, resets seen: "
          f"{int(((out['t'][1:] == 0).sum()))}")
    env.close()
    app.close()


if __name__ == "__main__":
    main()
