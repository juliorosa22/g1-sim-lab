"""v9a / v9b: v8 with the posture terms the other recipes have and ours lacked.

v8 (the v4 recipe on the real plant) crouches: knee ~1.0-1.1 rad (default 0.3), base height 0.725 m (others 0.79 m), knee torque
20 N*m standing (v4: 3-4). Nothing in the reward set resisted it. Two controlled changes, nothing else:

  v9a  + base height   Unitree: -10 * (z - 0.78)^2
  v9b  + base height   (as above)
       + whole-body pose reward (mjlab variable_posture): exp(-mean(err^2/std^2)), weight 1.0,
         std walking: hip pitch 0.3, hip roll/yaw 0.15, knee 0.35, ankle pitch 0.25, ankle roll 0.1, waist yaw 0.2;
         std standing (command below 0.05): 0.05 for every joint
         Measured (tests/check_posture_terms.py): default pose 1.0; v8's crouch (knee 0.97, hip pitch -0.41, ankle pitch -0.55)
         scores 0.36 while walking and 0.00 while standing, because the mean over all 13 joints dilutes the crouched ones.
"""
import re

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .robot_cfg import LEG_JOINT_NAMES, WAIST_YAW_JOINT

BASE_HEIGHT_TARGET = 0.78  # Unitree's target; the default pose stands at 0.789 m
BASE_HEIGHT_WEIGHT = -10.0
POSE_STD_WALKING = {".*_hip_pitch_joint": 0.3, ".*_hip_roll_joint": 0.15, ".*_hip_yaw_joint": 0.15, ".*_knee_joint": 0.35,
                    ".*_ankle_pitch_joint": 0.25, ".*_ankle_roll_joint": 0.1, "waist_yaw_joint": 0.2}
POSE_STD_STANDING = {".*": 0.05}


def add_base_height(cfg):
    cfg.rewards.base_height = RewTerm(
        func=mdp.base_height_l2, weight=BASE_HEIGHT_WEIGHT,
        params={"target_height": BASE_HEIGHT_TARGET, "asset_cfg": SceneEntityCfg("robot")},
    )


def add_pose(cfg, joint_names):
    # keep only tolerance entries that match one of these joints (the leg-only v10 has no waist yaw)
    keep = lambda d: {k: v for k, v in d.items() if any(re.fullmatch(k, n) for n in joint_names)}  # noqa: E731
    cfg.rewards.pose = RewTerm(
        func=mdp.variable_posture, weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(joint_names), preserve_order=True),
            "command_name": "base_velocity",
            "std_standing": keep(POSE_STD_STANDING),
            "std_walking": keep(POSE_STD_WALKING),
            "walking_threshold": 0.05,
        },
    )


def _play(cfg):
    cfg.scene.num_envs = 50
    cfg.scene.env_spacing = 2.5
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None


WALK_JOINTS = tuple(LEG_JOINT_NAMES) + (WAIST_YAW_JOINT,)


from .v8_env_cfg import G1V8EnvCfg  # noqa: E402


@configclass
class G1V9aEnvCfg(G1V8EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_base_height(self)


@configclass
class G1V9aEnvCfg_PLAY(G1V9aEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


@configclass
class G1V9bEnvCfg(G1V8EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_base_height(self)
        add_pose(self, WALK_JOINTS)


@configclass
class G1V9bEnvCfg_PLAY(G1V9bEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
