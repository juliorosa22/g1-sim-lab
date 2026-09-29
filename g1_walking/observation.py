"""Builds the 47-dim observation vector the walking policy expects, and the
quaternion math needed to build it.

Observation layout (transcribed verbatim from
unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco.py):
    obs[0:3]   base angular velocity (BODY frame) * ang_vel_scale
    obs[3:6]   gravity direction rotated into the body frame
    obs[6:9]   velocity command [vx, vy, yaw_rate] * cmd_scale
    obs[9:21]  12 leg joint positions - default_angles, * dof_pos_scale
    obs[21:33] 12 leg joint velocities * dof_vel_scale
    obs[33:45] previous action (raw, unscaled)
    obs[45:47] [sin(phase), cos(phase)] gait clock
"""
import numpy as np

from .config import WalkingPolicyConfig


def quat_rotate_inverse(quat_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotates a world-frame vector into the frame a unit quaternion
    represents (world -> body, when quat_wxyz is that body's orientation
    expressed in the world frame).

    Standard Hamilton-product vector rotation by a quaternion's conjugate
    (= inverse, for a unit quaternion). Scalar-first (w, x, y, z), matching
    Isaac Sim's convention throughout isaacsim.core (confirmed earlier this
    project from isaacsim.core.prims.impl.single_prim_wrapper).

    Algebraically verified equal to get_gravity_orientation(quat) below when
    v = [0, 0, -1], so the two are interchangeable; get_gravity_orientation
    is kept as its own function because it's transcribed literally from
    Unitree's source and is trivial to diff against it.
    """
    w = quat_wxyz[0]
    q_vec = quat_wxyz[1:4]
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * (2.0 * w)
    c = q_vec * (2.0 * np.dot(q_vec, v))
    return a - b + c


def get_gravity_orientation(quat_wxyz: np.ndarray) -> np.ndarray:
    """Gravity direction expressed in the body frame, given the body's
    orientation quaternion (world frame, scalar-first). Transcribed exactly
    from unitree_rl_gym's deploy_mujoco.py get_gravity_orientation.
    """
    qw, qx, qy, qz = quat_wxyz
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


class ObservationBuilder:
    def __init__(self, config: WalkingPolicyConfig):
        self.config = config
        self.previous_action = np.zeros(config.num_actions, dtype=np.float32)
        self._elapsed_s = 0.0

    def reset(self):
        self.previous_action[:] = 0.0
        self._elapsed_s = 0.0

    def build(
        self,
        base_quat_wxyz: np.ndarray,
        base_ang_vel_body: np.ndarray,
        leg_pos: np.ndarray,
        leg_vel: np.ndarray,
        dt_since_last_control: float,
    ) -> np.ndarray:
        cfg = self.config
        self._elapsed_s += dt_since_last_control
        n = cfg.num_actions

        obs = np.zeros(cfg.num_obs, dtype=np.float32)
        obs[0:3] = np.asarray(base_ang_vel_body) * cfg.ang_vel_scale
        obs[3:6] = get_gravity_orientation(np.asarray(base_quat_wxyz))
        obs[6:9] = np.asarray(cfg.cmd) * np.asarray(cfg.cmd_scale)
        obs[9 : 9 + n] = (np.asarray(leg_pos) - np.asarray(cfg.default_angles)) * cfg.dof_pos_scale
        obs[9 + n : 9 + 2 * n] = np.asarray(leg_vel) * cfg.dof_vel_scale
        obs[9 + 2 * n : 9 + 3 * n] = self.previous_action

        phase = (self._elapsed_s % cfg.gait_period_s) / cfg.gait_period_s
        obs[9 + 3 * n] = np.sin(2 * np.pi * phase)
        obs[9 + 3 * n + 1] = np.cos(2 * np.pi * phase)
        return obs

    def record_action(self, action: np.ndarray):
        self.previous_action[:] = action
