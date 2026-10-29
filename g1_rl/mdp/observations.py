"""Observation terms specific to this task."""
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def gait_phase(env: ManagerBasedRLEnv, period_s: float) -> torch.Tensor:
    """[sin, cos] of the gait phase, shape (num_envs, 2).

    phase = (t mod period) / period with t = time since the episode started,
    the same clock g1_walking.observation.ObservationBuilder runs on hardware
    (it starts at 0 when the policy starts). Matches Unitree's obs[45:47].
    """
    t = env.episode_length_buf.to(torch.float32) * env.step_dt
    phase = (t % period_s) / period_s
    angle = 2.0 * torch.pi * phase
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def joint_pos_rel_biased(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions relative to default, plus the per-env encoder bias drawn by
    g1_rl.mdp.events.randomize_encoder_bias (zero until that startup event ran)."""
    asset = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
    bias = getattr(env, "encoder_bias", None)
    return q if bias is None else q + bias
