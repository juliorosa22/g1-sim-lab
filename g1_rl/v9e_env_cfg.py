"""v9e: base height anchors the walking posture, a pose pull applies only at zero command.

v9b, v9c and v9d (whole-body pose, whole-body pose with looser tolerance, sagittal-only pose) all ended with the same
signature next to v8: yaw tracking 0.61-0.65 (v8 0.76), feet air time 0.02-0.03 (v8 0.062), turning at 0.5 rad/s ~0.2 (v8 0.48).
v9a (base height only) kept v8's yaw (0.70) and air time (0.069). My reading: a pose bonus paid on every step makes stepping a
net loss. v9e, warm-started from v8:
  * base height (-10 at 0.78 m), as v9a: the only posture term while walking
  * the whole-body pose reward (std 0.05) counts ONLY at zero command (walking_scale 0): holds the stance planted when still,
    pays nothing while moving, so a stride costs no pose reward (playground's stand_still / feet_phase masking works the same way)
  * both ramped in over iterations 50-350 (the v6 mechanism)
Interface unchanged (78 obs, 13 actions).
"""
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from . import mdp
from .v8_env_cfg import G1V8EnvCfg
from .v9_env_cfg import WALK_JOINTS, _play, add_base_height, add_pose
from .v9d_env_cfg import RAMP_END_ITERS, RAMP_START_ITERS, STEPS_PER_ITER


@configclass
class G1V9eEnvCfg(G1V8EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_base_height(self)
        add_pose(self, WALK_JOINTS, walking_scale=0.0)
        final = {"base_height": self.rewards.base_height.weight, "pose": self.rewards.pose.weight}
        for name in final:
            getattr(self.rewards, name).weight = 0.0  # the curriculum raises them
        self.curriculum.posture_ramp = CurrTerm(
            func=mdp.ramp_stand_rewards,
            params={
                "term_weights": final,
                "start_step": RAMP_START_ITERS * STEPS_PER_ITER,
                "end_step": RAMP_END_ITERS * STEPS_PER_ITER,
            },
        )


@configclass
class G1V9eEnvCfg_PLAY(G1V9eEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
