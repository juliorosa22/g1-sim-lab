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
