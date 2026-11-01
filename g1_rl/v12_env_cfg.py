"""v12: the walking-only policy: smooth legs, a clean stop, upright at speed.

The plan (docs and the v11 results): v11 stands planted, turns in place and creeps up to ~0.3 m/s; walking at speed is v12's job and a
selector hands over at about 0.3 m/s. What v12 has to be good at, that earlier walkers (v8, v9a-e) were not:
  * smooth legs      action-rate and joint-acceleration penalties 10x / 5x stronger, leg joint speed penalised, a fixed swing-foot
                     clearance (Unitree's feet_swing_height) and no foot sliding, so the gait is steady instead of marching and flailing
  * a clean stop     the commanded velocity is RAMPED (1 m/s^2, 2 rad/s^2, 0.05 deadband, as the deployed controller does), 15% of the commands
                     are zero and the commands change every 2.5-6 s, so a stop from speed happens constantly. While the command is zero: the
                     whole-body pose pull (tight, with the annealed tolerance that worked in v11) plus penalties on base speed and foot lifts
                     bring the robot to an upright stand that matches v11's, so the handover does not find a lurching robot
  * upright at speed v9b's recipe: base height -30 and the whole-body pose reward in both modes (v9b walked upright, 0.787 m, and tracked
                     0.97 at 1.0 m/s; its flaw, ignoring commands below ~0.3 m/s and yaw-only, is v11's job now)
Commands: forward -0.3..1.0, sideways +-0.4, yaw +-1.0, with forward-only and free mixes (rotate-in-place only 5%). Arms are randomised as in v8.
Warm start from v9b (same 78 -> 13 interface). Every new or strengthened term ramps in over iterations 50-600 (the old weight moves to the new one,
the old penalty is never switched off).
"""
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .v8_env_cfg import G1V8EnvCfg
from .v9_env_cfg import WALK_JOINTS, _play, add_base_height, add_pose
from .v11_env_cfg import STEPS_PER_ITER

DEADBAND = 0.05
RAMP_LIN, RAMP_YAW = 1.0, 2.0
RAMP_START_ITERS, RAMP_END_ITERS = 50, 600
BASE_HEIGHT_WEIGHT = -30.0
STD_SCALE = (6.0, 1.0)  # standing pose tolerance multiplier (annealed, as in v11)
SWING_TARGET_Z = 0.09  # ankle-roll link origin height of a swing foot: 0.035 m flat on the ground, v9b swings to ~0.04-0.06 at 0.6 m/s, so ~5.5 cm of clearance
# name -> (old weight, new weight)
SMOOTHING = {
    "action_rate_l2": (-0.005, -0.05),
    "dof_acc_l2": (-1.0e-7, -5.0e-7),
    "feet_slide": (-0.1, -0.2),
    "dof_vel_l2": (0.0, -5.0e-4),
    "feet_swing_height": (0.0, -10.0),
    "base_height": (0.0, BASE_HEIGHT_WEIGHT),
    "pose": (0.0, 1.0),
    "stand_feet_air": (0.0, -0.5),
    "stand_speed_l1": (0.0, -1.0),
}
P_STAND, P_FORWARD, P_LATERAL, P_ROTATE = 0.15, 0.40, 0.06, 0.07  # of the non-standing envs: 40% forward-only, the rest mixed
VX_RANGE, VY_RANGE, WZ_RANGE = (-0.3, 1.0), (-0.4, 0.4), (-1.0, 1.0)


@configclass
class G1V12EnvCfg(G1V8EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        add_base_height(self)
        add_pose(self, WALK_JOINTS, walking_scale=1.0)  # v9b's pose reward in both modes
        r = self.rewards
        legs = SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])
        foot_sensor = SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")
        foot_body = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")
        std = {"command_name": "base_velocity", "deadband": DEADBAND}
        r.dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": legs})
        r.feet_swing_height = RewTerm(
            func=mdp.feet_swing_height, weight=0.0,
            params={**std, "sensor_cfg": foot_sensor, "asset_cfg": foot_body, "target_z": SWING_TARGET_Z},
        )
        r.stand_feet_air = RewTerm(func=mdp.stand_feet_air, weight=0.0, params={**std, "recovery_speed": 0.25, "sensor_cfg": foot_sensor})
        r.stand_speed_l1 = RewTerm(func=mdp.stand_speed_l1, weight=0.0, params={**std, "asset_cfg": SceneEntityCfg("robot")})
        for name, (w0, _) in SMOOTHING.items():
            getattr(r, name).weight = w0
        r.pose.params["std_scale"] = STD_SCALE[0]
        self.curriculum.smooth_ramp = CurrTerm(
            func=mdp.ramp_reward_weights,
            params={
                "term_weights": SMOOTHING,
                "start_step": RAMP_START_ITERS * STEPS_PER_ITER,
                "end_step": RAMP_END_ITERS * STEPS_PER_ITER,
                "anneal": {"pose": ("std_scale", STD_SCALE[0], STD_SCALE[1])},
            },
        )
        cmd = self.commands.base_velocity
        cmd.rel_standing_envs = P_STAND
        cmd.p_forward_only, cmd.p_lateral_only, cmd.p_rotate_only = P_FORWARD, P_LATERAL, P_ROTATE
        cmd.resampling_time_range = (2.5, 6.0)
        cmd.deadband = DEADBAND
        cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = VX_RANGE, VY_RANGE, WZ_RANGE
        cmd.ramp_lin, cmd.ramp_yaw = RAMP_LIN, RAMP_YAW


@configclass
class G1V12EnvCfg_PLAY(G1V12EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)
        self.commands.base_velocity.ramp_lin = 0.0  # evaluation / play scripts write the command directly (and ramp it themselves if asked)
        self.commands.base_velocity.ramp_yaw = 0.0
