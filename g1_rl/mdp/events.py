"""Event terms specific to this task."""
from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def hold_joints_at_default(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """Sets the position targets of joints no action term controls (arms,
    waist, hands) to their default pose.

    Isaac Lab initialises joint position targets to zero and never touches the
    ones no action term writes -- left alone, the arms would be driven to 0
    rather than the standing pose. Targets persist between steps, so setting
    them at reset is enough.
    """
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    asset.set_joint_position_target(asset.data.default_joint_pos[env_ids][:, ids], joint_ids=ids, env_ids=env_ids)


def randomize_encoder_bias(
    env: ManagerBasedEnv, env_ids: torch.Tensor | None, bias_range: tuple[float, float], asset_cfg: SceneEntityCfg
):
    """Draws a fixed per-env, per-joint offset (rad) that
    g1_rl.mdp.observations.joint_pos_rel_biased adds to the joint positions the
    actor sees -- a miscalibrated encoder. Stored on the env; startup mode."""
    n = len(asset_cfg.joint_ids)
    env.encoder_bias = torch.empty(env.num_envs, n, device=env.device).uniform_(*bias_range)
