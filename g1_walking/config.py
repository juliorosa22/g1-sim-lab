"""Configuration for the G1 legs-only walking policy port.

Every numeric value here is transcribed verbatim from Unitree's own reference
deployment for this exact checkpoint:
    unitree_rl_gym/deploy/deploy_mujoco/configs/g1.yaml
    unitree_rl_gym/deploy/deploy_mujoco/deploy_mujoco.py

The checkpoint (checkpoint/motion.pt) is Unitree's pretrained 12-DOF
(legs-only) G1 locomotion policy: an LSTM actor ("policy_lstm_1") exported as
a self-contained TorchScript module. torch.jit.load() loads architecture +
weights together -- no model class needs reconstructing, and Isaac Gym (the
framework it was trained in) is not required at inference time, only
PyTorch. See checkpoint/README.md for licensing/provenance.

Joint order -- per leg: hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch,
ankle_roll. CONFIRMED directly from the real MJCF <joint>/<actuator> order in
unitree_rl_gym/resources/robots/g1_description/g1_12dof.xml. Earlier in this
project I mis-stated this order from a Python config dict's key-listing
order (which is incidental, not authoritative for qpos/qvel/action
indexing) -- this file reflects the corrected, source-verified order.

This module has no isaacsim/omni/pxr dependencies (same convention as
g1_sim/config.py), so it's safe to import before SimulationApp() exists.
"""
import os
from dataclasses import dataclass, field
from typing import Tuple

# Legs-only, in the exact order the policy's kps/kds/default_angles/action
# vector expect -- see this file's module docstring for how that was verified.
LEG_JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)


def _default_checkpoint_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint", "motion.pt")


@dataclass
class WalkingPolicyConfig:
    checkpoint_path: str = field(default_factory=_default_checkpoint_path)

    num_actions: int = 12
    num_obs: int = 47

    # (stiffness, damping) applied 1:1 per joint in LEG_JOINT_NAMES order --
    # transcribed verbatim from configs/g1.yaml's kps/kds.
    kps: Tuple[float, ...] = (100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40)
    kds: Tuple[float, ...] = (2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2)

    # Resting joint targets the policy's action is added to, LEG_JOINT_NAMES order.
    default_angles: Tuple[float, ...] = (
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    )

    ang_vel_scale: float = 0.25
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    action_scale: float = 0.25
    cmd_scale: Tuple[float, float, float] = (2.0, 2.0, 0.25)

    # Forward-walking velocity command [vx, vy, yaw_rate] (m/s, m/s, rad/s) --
    # fixed for now, matching deploy_mujoco.py's own cmd_init. Wiring this up
    # to a ROS2/teleop topic is a follow-up, not part of this port.
    cmd: Tuple[float, float, float] = (0.5, 0.0, 0.0)

    # Gait phase clock: sin/cos of (elapsed mod period)/period, the policy's
    # last 2 observation entries.
    gait_period_s: float = 0.8

    # The reference control rate is simulation_dt * control_decimation =
    # 0.002 * 10 = 50Hz. Isaac Sim's own physics dt generally won't be
    # MuJoCo's 0.002s, so run_walking.py derives the actual step count for
    # this control period from the stage's real physics dt at runtime rather
    # than assuming these two values directly translate.
    simulation_dt: float = 0.002
    control_decimation: int = 10
