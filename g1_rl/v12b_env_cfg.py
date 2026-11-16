"""v12b: v12 (walking) retrained with the wider randomisation of g1_rl/wide_randomization.py.

Warm start from v12 (model_4998): all the smoothing / posture weights are at their FINAL values from the first iteration (the v12 ramp is removed).
"""
from isaaclab.utils import configclass

from .v12_env_cfg import SMOOTHING, G1V12EnvCfg
from .v9_env_cfg import _play
from .wide_randomization import widen_randomization


@configclass
class G1V12bEnvCfg(G1V12EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        for name, (_, w1) in SMOOTHING.items():
            getattr(self.rewards, name).weight = w1
        self.rewards.pose.params["std_scale"] = 1.0
        self.curriculum.smooth_ramp = None
        widen_randomization(self)


@configclass
class G1V12bEnvCfg_PLAY(G1V12bEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
        self.commands.base_velocity.ramp_lin = 0.0
        self.commands.base_velocity.ramp_yaw = 0.0
