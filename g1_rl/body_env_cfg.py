"""Whole-body variants of the flat walking task.

  v3  G1WalkWaistEnvCfg   the policy drives legs + waist (15 joints). Arms and hands
                          are held at the standing pose.
  v4  G1WalkArmsEnvCfg    same 15 actions, but the actor also observes all 29 body
                          joints, and the arms are NOT held: an ArmPoseCommand
                          sweeps them through random poses during training, so the
                          policy learns to stay balanced however the arms are
                          posed (design choice B: arms stay free for other tasks).

Both inherit everything else (commands, randomization, rewards, terminations)
from the v2 flat task and change only what is listed in each __post_init__.
"""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .flat_env_cfg import G1WalkFlatEnvCfg
from .robot_cfg import ARM_JOINT_NAMES, G1_INSPIRE_WAIST_CFG, LEG_JOINT_NAMES, WAIST_JOINT_NAMES

CONTROLLED = tuple(LEG_JOINT_NAMES) + tuple(WAIST_JOINT_NAMES)  # 15 actions
OBSERVED_ALL = CONTROLLED + tuple(ARM_JOINT_NAMES)  # 29 body joints

# Action scale per joint (rad per unit action). Legs keep v2's 0.25. The waist gets
# 0.2: the sim config's waist torque limits (35-88 N*m) against kp=200 saturate
# long before a 0.25 rad offset. Placeholder -- revisit once v3 is trained.
ACTION_SCALE = {".*_hip_.*": 0.25, ".*_knee_joint": 0.25, ".*_ankle_.*": 0.25, "waist_.*": 0.2}

# Arm goal poses for v4: absolute joint angles (rad), (low, high). Shoulder roll is
# one-sided (outward only) so the arms are never driven into the torso, and the
# elbow stays bent. Placeholder ranges chosen by hand -- not checked against the
# real robot's self-collision envelope.
ARM_POSE_RANGES = {
    "left_shoulder_pitch_joint": (-1.0, 1.0), "right_shoulder_pitch_joint": (-1.0, 1.0),
    "left_shoulder_roll_joint": (0.0, 0.8), "right_shoulder_roll_joint": (-0.8, 0.0),
    "left_shoulder_yaw_joint": (-0.6, 0.6), "right_shoulder_yaw_joint": (-0.6, 0.6),
    "left_elbow_joint": (0.2, 1.6), "right_elbow_joint": (0.2, 1.6),
    "left_wrist_roll_joint": (-0.6, 0.6), "right_wrist_roll_joint": (-0.6, 0.6),
    "left_wrist_pitch_joint": (-0.4, 0.4), "right_wrist_pitch_joint": (-0.4, 0.4),
    "left_wrist_yaw_joint": (-0.4, 0.4), "right_wrist_yaw_joint": (-0.4, 0.4),
}


def _joints(names) -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(names), preserve_order=True)


def _control_and_observe(cfg, observed, held_regex):
    """Points every joint-dependent term at the new joint sets."""
    cfg.scene.robot = G1_INSPIRE_WAIST_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    act = cfg.actions.joint_pos
    act.joint_names = list(CONTROLLED)
    act.preserve_order = True
    act.scale = dict(ACTION_SCALE)

    for group in (cfg.observations.policy, cfg.observations.critic):
        group.joint_pos.params["asset_cfg"] = _joints(observed)
        group.joint_vel.params["asset_cfg"] = _joints(observed)

    ev = cfg.events
    ev.encoder_bias.params["asset_cfg"] = _joints(observed)
    ev.actuator_gains.params["asset_cfg"] = _joints(CONTROLLED)
    ev.joint_armature.params["asset_cfg"] = _joints(CONTROLLED)
    ev.reset_leg_joints.params["asset_cfg"] = _joints(CONTROLLED)
    ev.hold_joints.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=held_regex)

    cfg.rewards.dof_torques_l2.params["asset_cfg"] = _joints(CONTROLLED)
    # With a controllable waist the torso can tilt independently of the pelvis
    # (mjlab's `upright` reward is on the torso for this reason), and the waist
    # should stay near neutral unless it is needed (mjlab `pose` reward).
    cfg.rewards.torso_orientation = RewTerm(
        func=mdp.body_flat_orientation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    cfg.rewards.waist_deviation_rp = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_roll_joint", "waist_pitch_joint"])},
    )
    cfg.rewards.waist_deviation_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])},
    )


@configclass
class G1WalkWaistEnvCfg(G1WalkFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # observe exactly the joints the policy controls; hold arms + hands
        _control_and_observe(
            self, observed=CONTROLLED, held_regex=[".*shoulder.*", ".*elbow.*", ".*wrist.*", "^[LR]_.*"]
        )


@configclass
class G1WalkWaistEnvCfg_PLAY(G1WalkWaistEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class G1WalkArmsEnvCfg(G1WalkFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # observe all 29 joints; only the fingers are held, the arms follow the command term
        _control_and_observe(self, observed=OBSERVED_ALL, held_regex=["^[LR]_.*"])
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
class G1WalkArmsEnvCfg_PLAY(G1WalkArmsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
