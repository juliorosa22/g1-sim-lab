"""Flat-ground variant (phase 3, stage 1)."""
from isaaclab.utils import configclass

from .walk_env_cfg import G1WalkEnvCfg


@configclass
class G1WalkFlatEnvCfg(G1WalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None


@configclass
class G1WalkFlatEnvCfg_PLAY(G1WalkFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # no randomization or pushes when watching / evaluating
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
