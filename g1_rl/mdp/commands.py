"""Velocity command term that covers the single-axis cases evenly.

Isaac Lab's UniformVelocityCommand draws vx, vy and yaw rate independently, so
"rotate in place" (vx = vy = 0) and "pure sideways" (vx = yaw = 0) almost never
occur -- the first policy trained on it could not turn in place or walk
sideways (scripts/evaluate.py: yaw-only 0.5 rad/s achieved 0.0, vy-only 0.3 m/s
achieved 0.12). This term mixes in dedicated single-axis modes.

Standing (all zero) stays with the parent's rel_standing_envs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    pass


class MixedVelocityCommand(UniformVelocityCommand):
    cfg: "MixedVelocityCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if not self.cfg.mixed_sampling:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        u = torch.rand(len(ids), device=self.device)
        c1 = self.cfg.p_forward_only
        c2 = c1 + self.cfg.p_lateral_only
        c3 = c2 + self.cfg.p_rotate_only
        forward = ids[u < c1]
        lateral = ids[(u >= c1) & (u < c2)]
        rotate = ids[(u >= c2) & (u < c3)]
        cmd = self.vel_command_b
        cmd[forward, 1] = 0.0
        cmd[forward, 2] = 0.0
        cmd[lateral, 0] = 0.0
        cmd[lateral, 2] = 0.0
        cmd[rotate, 0] = 0.0
        cmd[rotate, 1] = 0.0


@configclass
class MixedVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = MixedVelocityCommand

    mixed_sampling: bool = True
    """False falls back to plain independent sampling (scripts/evaluate.py uses this
    to hold an exact command)."""
    p_forward_only: float = 0.2
    p_lateral_only: float = 0.1
    p_rotate_only: float = 0.2
    """Probabilities of the single-axis modes; the remainder samples all three axes freely."""


class ArmPoseCommand(CommandTerm):
    """Drives the arm joints to randomized target poses, outside the policy.

    The walking policy does not control the arms (design choice B: they stay
    free for other tasks), so it has to stay balanced however they are posed
    and moved. Every few seconds each env gets a new goal pose, sampled
    uniformly inside cfg.pose_ranges (absolute joint angles); the target
    handed to the PD controllers moves toward the goal at most
    cfg.max_velocity rad/s so the arms sweep instead of teleporting. With
    probability cfg.p_default the goal is the default pose instead.

    This term writes joint position targets itself every control step; the
    policy never sees the goal, only the resulting joint positions.
    """

    cfg: "ArmPoseCommandCfg"

    def __init__(self, cfg: "ArmPoseCommandCfg", env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.joint_ids, names = self.robot.find_joints(list(cfg.joint_names), preserve_order=True)
        soft = self.robot.data.soft_joint_pos_limits[0, self.joint_ids]
        lo = torch.tensor([cfg.pose_ranges[n][0] for n in names], device=self.device)
        hi = torch.tensor([cfg.pose_ranges[n][1] for n in names], device=self.device)
        self.lo = torch.maximum(lo, soft[:, 0])
        self.hi = torch.minimum(hi, soft[:, 1])
        self.default = self.robot.data.default_joint_pos[0, self.joint_ids].clone()
        shape = (self.num_envs, len(names))
        self.goal = self.default.repeat(self.num_envs, 1).reshape(shape)
        self.current = self.goal.clone()

    @property
    def command(self) -> torch.Tensor:
        return self.current

    def reset(self, env_ids=None):
        extras = super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        # the arms start every episode at the default pose, and so do their targets
        self.current[ids] = self.default
        self.robot.set_joint_position_target(self.current[ids], joint_ids=self.joint_ids, env_ids=env_ids)
        return extras

    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        u = torch.rand(len(ids), len(self.joint_ids), device=self.device)
        goal = self.lo + u * (self.hi - self.lo)
        keep = torch.rand(len(ids), device=self.device) < self.cfg.p_default
        goal[keep] = self.default
        self.goal[ids] = goal

    def _update_command(self):
        step = self.cfg.max_velocity * self._env.step_dt
        self.current += (self.goal - self.current).clamp(-step, step)
        self.robot.set_joint_position_target(self.current, joint_ids=self.joint_ids)

    def _update_metrics(self):
        pass


@configclass
class ArmPoseCommandCfg(CommandTermCfg):
    class_type: type = ArmPoseCommand

    asset_name: str = "robot"
    joint_names: tuple = ()
    pose_ranges: dict = {}
    """joint name -> (low, high), absolute joint angles in rad."""
    max_velocity: float = 1.5
    """Max speed of the arm targets, rad/s."""
    p_default: float = 0.2
    """Probability that a new goal is just the default pose."""
