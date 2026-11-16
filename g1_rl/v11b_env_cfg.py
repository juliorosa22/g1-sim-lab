"""v11b: v11 (stand + turn in place) retrained with the wider randomisation of g1_rl/wide_randomization.py.

Warm start from v11 (model_5997): the posture terms are at their FINAL weights and the final pose tolerance from the first iteration (the v11 ramp is
removed; it would pull the weights back to zero and risk losing the stand), everything else as v11.
"""
from isaaclab.utils import configclass

from .v11_env_cfg import G1V11EnvCfg, G1V11EnvCfg_PLAY
from .v9_env_cfg import _play
from .wide_randomization import widen_randomization


@configclass
class G1V11bEnvCfg(G1V11EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        final = self.curriculum.posture_ramp.params["term_weights"]
        for name, weight in final.items():
            getattr(self.rewards, name).weight = weight
        self.rewards.pose.params["std_scale"] = 1.0
        self.curriculum.posture_ramp = None
        widen_randomization(self)


@configclass
class G1V11bEnvCfg_PLAY(G1V11bEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
        self.commands.base_velocity.ramp_lin = 0.0
        self.commands.base_velocity.ramp_yaw = 0.0
