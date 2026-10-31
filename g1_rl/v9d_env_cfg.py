"""v9d: keep v8's turning, remove its crouch with the least interference.

v8 (the v4 recipe on the real plant) turns and tracks small commands well but crouches. v9b/v9c fixed the crouch with a
whole-body pose reward and lost yaw: that reward also penalises hip roll/yaw and ankle roll, the joints a turn uses.
v9d, warm-started from v8:
  * base height (-10 at 0.78 m), as v9a
  * the pose reward on the SAGITTAL joints only (hip pitch, knee, ankle pitch): the joints that define the crouch.
    Hip roll/yaw, ankle roll and waist yaw are left to v8's old deviation terms, so turning is not taxed.
    Standing std 0.05 on those joints also pins the stance at zero command; walking std as v9b (0.3 / 0.35 / 0.25).
  * both terms ramped in from zero over RAMP_ITERS iterations (the v6 mechanism), so tracking and turning are kept
    while the posture is pulled upright gradually.
Interface unchanged (78 obs, 13 actions). v9a (base height only, no ramp) is the controlled baseline.
"""
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from . import mdp
from .v8_env_cfg import G1V8EnvCfg
from .v9_env_cfg import _play, add_base_height, add_pose

SAGITTAL_JOINTS = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
)
STEPS_PER_ITER = 24  # num_steps_per_env of the PPO runner
RAMP_START_ITERS = 50
RAMP_END_ITERS = 350


@configclass
class G1V9dEnvCfg(G1V8EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_base_height(self)
        add_pose(self, SAGITTAL_JOINTS)
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
class G1V9dEnvCfg_PLAY(G1V9dEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
