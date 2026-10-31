"""v9c: v9b without its deadband.

v9b (v8 + base height + whole-body pose) stands upright and planted, but ignores small commands: forward below ~0.25 m/s,
sideways below ~0.3, yaw below ~0.3 rad/s (a 0.5 rad/s turn reaches 0.21). Training logs ruled out exploration collapse
(action std 0.30 at the end, same as v8) and unstable updates, so this is reward shaping:
  - the tracking kernel exp(-err^2/0.25) pays 0.85 for standing still at a 0.2 m/s command, so moving gains little
  - the pose reward pays ~0.88 per step for staying near the default pose, and feet air time fell from 0.062 to 0.018
Changes against v9b (warm start from its checkpoint):
  1. a second, narrow tracking pair (lin std 0.15, yaw std 0.25, weight 0.5): standing still at 0.2 m/s now pays 0.17, at 0.5 rad/s 0.02
  2. 25% of commands redrawn gentle (lin 0.06-0.3 m/s, yaw 0.1-0.5 rad/s), deadband 0.05 as in the controller
  3. looser walking pose tolerance on the joints a turn or sideways step needs: hip roll 0.25, hip yaw 0.3, ankle roll 0.15, waist yaw 0.3
Kept: base height, the standing std (0.05), interface (78 obs, 13 actions), real-plant gains.
"""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from . import mdp
from .v8_env_cfg import G1V8EnvCfg
from .v9_env_cfg import POSE_STD_WALKING, WALK_JOINTS, _play, add_base_height, add_pose

POSE_STD_WALKING_C = dict(POSE_STD_WALKING)
POSE_STD_WALKING_C.update({".*_hip_roll_joint": 0.25, ".*_hip_yaw_joint": 0.3, ".*_ankle_roll_joint": 0.15, "waist_yaw_joint": 0.3})
FINE_LIN_STD = 0.15
FINE_YAW_STD = 0.25
FINE_WEIGHT = 0.5
P_SLOW = 0.25
DEADBAND = 0.05


@configclass
class G1V9cEnvCfg(G1V8EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_base_height(self)
        add_pose(self, WALK_JOINTS, std_walking=POSE_STD_WALKING_C)
        self.rewards.track_lin_vel_xy_fine = RewTerm(
            func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=FINE_WEIGHT,
            params={"command_name": "base_velocity", "std": FINE_LIN_STD},
        )
        self.rewards.track_ang_vel_z_fine = RewTerm(
            func=mdp.track_ang_vel_z_world_exp, weight=FINE_WEIGHT,
            params={"command_name": "base_velocity", "std": FINE_YAW_STD},
        )
        cmd = self.commands.base_velocity
        cmd.p_slow = P_SLOW
        cmd.deadband = DEADBAND


@configclass
class G1V9cEnvCfg_PLAY(G1V9cEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
