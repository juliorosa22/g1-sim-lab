"""Reward terms specific to this task."""
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse


def body_flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared horizontal components of gravity in the given body's frame (0 when upright).

    Isaac Lab's flat_orientation_l2 looks at the root (pelvis). Once the waist is
    controlled the torso can tilt independently, and mjlab's `upright` reward is
    on the torso for the same reason.
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    gravity = torch.tensor([0.0, 0.0, -1.0], device=quat.device).expand(quat.shape[0], 3)
    g_b = quat_apply_inverse(quat, gravity)
    return torch.sum(torch.square(g_b[:, :2]), dim=1)
