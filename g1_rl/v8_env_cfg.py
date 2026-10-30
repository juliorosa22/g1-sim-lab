"""v8: the v4 recipe on the real plant.

Same recipe as v4 (g1_rl/body_env_cfg.py: G1WalkArmsEnvCfg): arms swept through random poses by an
ArmPoseCommand, the actor observes the arms, v2's randomisation and command mix. What changes is the
plant, from the first hardware session (robot_deploy/measurements/20261029):

  * waist roll and pitch are LOCKED on this unit: rigid in simulation, never acted on, never observed
  * waist YAW stays: it is an action (13 actions = 12 legs + waist yaw), as Unitree's own controller
    uses it while walking
  * Unitree's onboard gains (legs kp 100/100/100/150/60/60, kd 4/4/4/5/3/3; waist yaw 200/6; arms 60/3; wrists 40/3)
  * ankle speed limit 37 rad/s (Unitree's URDF)
  * a per-env gyro bias of +/-0.005 rad/s (measured pelvis bias about 0.004 rad/s) on what the actor sees

Observation: 3 + 3 + 3 + 27 (joint pos) + 27 (joint vel) + 13 (last action) + 2 (gait clock) = 78; the critic adds
base linear velocity (81). Everything else (rewards, shoves, command sampling) is exactly v4's, except that the
waist roll/pitch reward term is gone (those joints cannot move).
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .body_env_cfg import ARM_POSE_RANGES
from .flat_env_cfg import G1WalkFlatEnvCfg
from .robot_cfg import ARM_JOINT_NAMES, G1_REAL_WALK_CFG, LEG_JOINT_NAMES, WAIST_YAW_JOINT

CONTROLLED = tuple(LEG_JOINT_NAMES) + (WAIST_YAW_JOINT,)  # 13 actions
OBSERVED = CONTROLLED + tuple(ARM_JOINT_NAMES)  # 27 joints
ACTION_SCALE = {".*_hip_.*": 0.25, ".*_knee_joint": 0.25, ".*_ankle_.*": 0.25, "waist_yaw_joint": 0.2}
GYRO_BIAS_RANGE = (-0.005, 0.005)  # rad/s per axis


def _joints(names) -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(names), preserve_order=True)


@configclass
class G1V8EnvCfg(G1WalkFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_REAL_WALK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        act = self.actions.joint_pos
        act.joint_names = list(CONTROLLED)
        act.preserve_order = True
        act.scale = dict(ACTION_SCALE)

        for group in (self.observations.policy, self.observations.critic):
            group.joint_pos.params["asset_cfg"] = _joints(OBSERVED)
            group.joint_vel.params["asset_cfg"] = _joints(OBSERVED)
        # the actor's gyro carries a constant per-env bias (the critic sees the true value)
        self.observations.policy.base_ang_vel.func = mdp.base_ang_vel_biased

        ev = self.events
        ev.encoder_bias.params["asset_cfg"] = _joints(OBSERVED)
        ev.actuator_gains.params["asset_cfg"] = _joints(CONTROLLED)
        ev.joint_armature.params["asset_cfg"] = _joints(CONTROLLED)
        ev.reset_leg_joints.params["asset_cfg"] = _joints(CONTROLLED)
        ev.hold_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=["waist_roll_joint", "waist_pitch_joint", "^[LR]_.*"]
        )
        ev.gyro_bias = EventTerm(
            func=mdp.randomize_gyro_bias, mode="startup", params={"bias_range": GYRO_BIAS_RANGE}
        )

        r = self.rewards
        r.dof_torques_l2.params["asset_cfg"] = _joints(CONTROLLED)
        r.torso_orientation = RewTerm(
            func=mdp.body_flat_orientation_l2, weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
        )
        r.waist_deviation_yaw = RewTerm(
            func=mdp.joint_deviation_l1, weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[WAIST_YAW_JOINT])},
        )

        self.commands.arm_pose = mdp.ArmPoseCommandCfg(
            asset_name="robot",
            joint_names=tuple(ARM_JOINT_NAMES),
            pose_ranges=dict(ARM_POSE_RANGES),
            resampling_time_range=(2.0, 5.0),
            max_velocity=1.5,
            p_default=0.2,
            debug_vis=False,
        )


@configclass
class G1V8EnvCfg_PLAY(G1V8EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
