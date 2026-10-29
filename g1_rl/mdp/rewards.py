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


# --- stand-still terms ------------------------------------------------------------
# All are multiplied by a mask that is 1 only while the velocity command is inside the
# deadband (i.e. "stand still"), so they never fight the walking rewards.


def _standing(env: ManagerBasedRLEnv, command_name: str, deadband: float) -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return ((cmd[:, :2].norm(dim=1) < deadband) & (cmd[:, 2].abs() < deadband)).float()


def _stand_mask(env: ManagerBasedRLEnv, command_name: str, deadband: float, recovery_speed: float = 0.0) -> torch.Tensor:
    """_standing, minus moments of recovery: when the base is moving faster than
    `recovery_speed` or tilted past ~9 degrees (a shove just landed), stepping is the right
    response, so the penalties on the act of stepping are switched off. 0 = no exemption."""
    mask = _standing(env, command_name, deadband)
    if recovery_speed > 0.0:
        asset = env.scene["robot"]
        fast = asset.data.root_lin_vel_w[:, :2].norm(dim=1) > recovery_speed
        tilted = asset.data.projected_gravity_b[:, :2].norm(dim=1) > 0.15
        mask = mask * (~(fast | tilted)).float()
    return mask


def _base_motion(asset) -> tuple[torch.Tensor, torch.Tensor]:
    return asset.data.root_lin_vel_w[:, :2].norm(dim=1), asset.data.root_ang_vel_w[:, 2].abs()


def stand_speed_l1(env: ManagerBasedRLEnv, command_name: str, deadband: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Linear penalty on base speed (+ half the yaw rate) while standing. Unlike the Gaussian
    velocity-tracking reward (flat near zero), this keeps a gradient all the way down."""
    v, wz = _base_motion(env.scene[asset_cfg.name])
    return _standing(env, command_name, deadband) * (v + 0.5 * wz)


def stand_still_exp(
    env: ManagerBasedRLEnv, command_name: str, deadband: float, std: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Narrow bonus (std ~5 cm/s) for being truly motionless while standing."""
    v, wz = _base_motion(env.scene[asset_cfg.name])
    return _standing(env, command_name, deadband) * torch.exp(-(v**2 + wz**2) / std**2)


def stand_joint_vel_l2(
    env: ManagerBasedRLEnv, command_name: str, deadband: float, asset_cfg: SceneEntityCfg, recovery_speed: float = 0.0
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return _stand_mask(env, command_name, deadband, recovery_speed) * torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def stand_pose_l1(env: ManagerBasedRLEnv, command_name: str, deadband: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Pull the controlled joints toward the default pose while standing (mjlab's `pose` idea)."""
    asset = env.scene[asset_cfg.name]
    dev = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return _standing(env, command_name, deadband) * torch.sum(dev.abs(), dim=1)


def stand_action_rate_l2(
    env: ManagerBasedRLEnv, command_name: str, deadband: float, recovery_speed: float = 0.0
) -> torch.Tensor:
    """Extra action-rate penalty while standing (the global one is small)."""
    delta = env.action_manager.action - env.action_manager.prev_action
    return _stand_mask(env, command_name, deadband, recovery_speed) * torch.sum(torch.square(delta), dim=1)


def stand_feet_air(
    env: ManagerBasedRLEnv, command_name: str, deadband: float, sensor_cfg: SceneEntityCfg, recovery_speed: float = 0.0
) -> torch.Tensor:
    """Number of feet off the ground while standing (a lifted foot = a step)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    airborne = (forces < 1.0).float().sum(dim=1)
    return _stand_mask(env, command_name, deadband, recovery_speed) * airborne
