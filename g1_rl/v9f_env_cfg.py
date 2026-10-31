"""v9f: a policy that stands planted like v10 AND turns in place, for tight spaces.

Evidence (see docs and the turn/settle evaluation): v8/v9a turn in place well (0.45-0.48 rad/s at 0.5) but step at zero command;
v9e holds a planted stand at zero command but crouches (base 0.76 m, knee 0.57 standing) and its slow yaw/sideways are weak;
the stand-only policies (v1, v7, v10) were never trained on yaw-only commands, so they step during a turn and do not settle.
Warm start from v9e (same 78 -> 13 interface), with:
  * commands the way the controller sends them: the policy sees the planner's command RAMPED at 1 m/s^2 and 2 rad/s^2, 0.05 deadband
  * a command mix built for this: 25% stand, ~27% rotate-in-place (so also stand -> turn -> stand sequences), 25% of all commands redrawn gentle
    (lin 0.06-0.3 m/s, yaw 0.1-0.5 rad/s), the rest walking
  * the pose pull at zero command only (v9e), now with an ANNEALED tolerance (6x -> 1x the 0.05 rad standing std over the ramp): an exp
    kernel at 0.05 gives a crouched stance zero reward and no gradient, which is why v9e could not climb out of its crouch
  * base height -30 (v9a/v9e -10 left it at 0.755-0.766 m)
  * a penalty on foot lifts while the command is zero (stand_feet_air, with the recovery exemption after a shove)
  * v9c's narrow tracking pair (lin 0.15, yaw 0.25, weight 0.5), so small and in-place commands pay
Posture terms ramp in over iterations 50-600. The Play variants switch the command ramp off (the evaluation scripts write the command directly).
"""
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .v8_env_cfg import G1V8EnvCfg
from .v9_env_cfg import WALK_JOINTS, _play, add_base_height, add_pose

BASE_HEIGHT_WEIGHT_F = -30.0
STAND_FEET_AIR_WEIGHT = -0.5
STD_SCALE = (6.0, 1.0)  # standing pose tolerance multiplier: wide at first, then the real 0.05 rad
FINE_LIN_STD, FINE_YAW_STD, FINE_WEIGHT = 0.15, 0.25, 0.5
STEPS_PER_ITER = 24
RAMP_START_ITERS, RAMP_END_ITERS = 50, 600
DEADBAND = 0.05
RAMP_LIN, RAMP_YAW = 1.0, 2.0
P_STAND, P_FORWARD, P_LATERAL, P_ROTATE, P_SLOW = 0.25, 0.15, 0.08, 0.27, 0.25


@configclass
class G1V9fEnvCfg(G1V8EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_base_height(self)
        self.rewards.base_height.weight = BASE_HEIGHT_WEIGHT_F
        add_pose(self, WALK_JOINTS, walking_scale=0.0)
        r = self.rewards
        r.stand_feet_air = RewTerm(
            func=mdp.stand_feet_air, weight=STAND_FEET_AIR_WEIGHT,
            params={"command_name": "base_velocity", "deadband": DEADBAND, "recovery_speed": 0.25,
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
        r.track_lin_vel_xy_fine = RewTerm(
            func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=FINE_WEIGHT,
            params={"command_name": "base_velocity", "std": FINE_LIN_STD},
        )
        r.track_ang_vel_z_fine = RewTerm(
            func=mdp.track_ang_vel_z_world_exp, weight=FINE_WEIGHT,
            params={"command_name": "base_velocity", "std": FINE_YAW_STD},
        )
        ramped = ("base_height", "pose", "stand_feet_air")
        final = {name: getattr(r, name).weight for name in ramped}
        for name in ramped:
            getattr(r, name).weight = 0.0  # the curriculum raises them
        r.pose.params["std_scale"] = STD_SCALE[0]
        self.curriculum.posture_ramp = CurrTerm(
            func=mdp.ramp_stand_rewards,
            params={
                "term_weights": final,
                "start_step": RAMP_START_ITERS * STEPS_PER_ITER,
                "end_step": RAMP_END_ITERS * STEPS_PER_ITER,
                "anneal": {"pose": ("std_scale", STD_SCALE[0], STD_SCALE[1])},
            },
        )
        cmd = self.commands.base_velocity
        cmd.rel_standing_envs = P_STAND
        cmd.p_forward_only, cmd.p_lateral_only, cmd.p_rotate_only = P_FORWARD, P_LATERAL, P_ROTATE
        cmd.p_slow = P_SLOW
        cmd.deadband = DEADBAND
        cmd.ramp_lin, cmd.ramp_yaw = RAMP_LIN, RAMP_YAW


@configclass
class G1V9fEnvCfg_PLAY(G1V9fEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
        self.commands.base_velocity.ramp_lin = 0.0  # evaluation / play scripts write the command directly (and ramp it themselves if asked)
        self.commands.base_velocity.ramp_yaw = 0.0
