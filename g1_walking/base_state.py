"""Reads the robot's floating-base state directly from physics ground truth
(the articulation root's own pose/velocity) -- the same way unitree_rl_gym's
reference reads d.qpos[3:7]/d.qvel[3:6] straight from MuJoCo's physics state,
NOT from a simulated IMU sensor. g1_sim/sensors.py's ImuSensor is wired for
the perception pipeline (state estimation, SLAM); this policy expects the
same privileged ground-truth state it was trained against.

IMPORTANT -- a frame-convention correction this port makes that a literal
1:1 translation would miss:
    MuJoCo free joints store qvel[3:6] as angular velocity already expressed
    in the body's own local frame (a MuJoCo-specific consequence of how
    free-joint quaternion integration works), so deploy_mujoco.py can use it
    directly with no transform. PhysX rigid bodies -- what Isaac Sim's
    articulation root velocities come from -- report angular velocity in the
    WORLD frame instead. Feeding a world-frame angular velocity into a
    policy trained on body-frame angular velocity would silently corrupt the
    first 3 observation entries every step, so this module rotates it into
    the body frame with quat_rotate_inverse before handing it to
    ObservationBuilder.

    This is standard PhysX/Isaac Sim behaviour, but unlike every other API
    used in this repo, it has NOT been grep-verified against your installed
    isaacsim source (this dev container has no isaacsim install to check
    against). Please run verify_walking_apis.py once on your machine and
    paste the output back -- it prints SingleArticulation.get_world_pose,
    get_angular_velocity and SimulationContext.get_physics_dt's docstrings so
    we can confirm this before trusting the walking policy on it. A quick
    empirical check also works: once the robot is standing still under
    run_walking.py, obs[0:3] (base angular velocity) should read near zero at
    rest -- if it's instead large/nonzero while the robot is visibly still,
    the frame assumed here is likely wrong.
"""
import numpy as np

from .observation import quat_rotate_inverse


class BaseState:
    """Thin wrapper around a SingleArticulation's root state."""

    def __init__(self, articulation):
        self.articulation = articulation

    def read(self):
        """Returns (quat_wxyz, ang_vel_body) as float64 numpy arrays."""
        _, quat_wxyz = self.articulation.get_world_pose()
        ang_vel_world = self.articulation.get_angular_velocity()
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
        ang_vel_body = quat_rotate_inverse(quat_wxyz, np.asarray(ang_vel_world, dtype=np.float64))
        return quat_wxyz, ang_vel_body
