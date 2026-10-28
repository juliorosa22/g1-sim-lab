"""Event terms specific to this task."""
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
