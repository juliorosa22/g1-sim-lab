"""v10: v7 (the v1 recipe on the real plant, legs-only standing policy) + base height + the whole-body pose reward.
Same two terms as v9b, applied to the 12 leg joints (waist yaw is held by the environment here). See v9_env_cfg.py."""
from isaaclab.utils import configclass

from .robot_cfg import LEG_JOINT_NAMES
from .v7_env_cfg import G1V7EnvCfg
from .v9_env_cfg import _play, add_base_height, add_pose


@configclass
class G1V10EnvCfg(G1V7EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_base_height(self)
        add_pose(self, LEG_JOINT_NAMES)


@configclass
class G1V10EnvCfg_PLAY(G1V10EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
