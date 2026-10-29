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


def gait_phase_gated(env: ManagerBasedRLEnv, period_s: float, command_name: str, deadband: float) -> torch.Tensor:
    """[sin, cos] of a gait phase that only runs while a velocity is commanded.

    At (near-)zero command the phase is frozen at 0, i.e. the observation is the
    constant [0, 1], so the policy gets no clock tick to step to; it restarts from
    0 whenever motion is commanded again (and at every episode start). The always-
    running clock of v2-v4 is the likely reason those policies step in place when
    told to stand (mujoco_playground carries the same idea as a commented-out option).

    The controller on the robot must apply the same rule: clock stopped while the
    command is inside the deadband, reset to 0 when it leaves it.
    """
    cmd = env.command_manager.get_command(command_name)
    moving = (cmd[:, :2].norm(dim=1) > deadband) | (cmd[:, 2].abs() > deadband)
    t = getattr(env, "_gait_time", None)
    if t is None or t.shape[0] != env.num_envs:
        t = torch.zeros(env.num_envs, device=env.device)
    # The policy and critic groups both call this once per step; advance only once per env step.
    if getattr(env, "_gait_time_step", -1) != env.common_step_counter:
        fresh = env.episode_length_buf == 0  # just reset
        t = torch.where(fresh | ~moving, torch.zeros_like(t), t + env.step_dt)
        env._gait_time = t
        env._gait_time_step = env.common_step_counter
    angle = 2.0 * torch.pi * ((t % period_s) / period_s)
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
