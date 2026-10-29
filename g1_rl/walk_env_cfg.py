"""Base environment config for the G1 (Inspire) velocity-tracking walking task.

Inherits scene / commands / curriculum / sim settings (dt=0.005, decimation=4
-> 50 Hz control) from Isaac Lab's LocomotionVelocityRoughEnvCfg and replaces
the MDP pieces:

  * actions: 12 leg joints only, in the policy's joint order (preserve_order),
    scale 0.25 around the default pose -- same as g1_walking/config.py.
  * actor observations: exactly the 47-value proprioceptive layout of Unitree's
    policy and g1_walking.observation.ObservationBuilder, with the same scales.
  * critic observations: the actor's terms without noise, plus base linear
    velocity (privileged; the real robot cannot measure it).
  * rewards: Isaac Lab's stock G1 terms, joint names adapted to this USD.
  * events: domain randomization for sim-to-real (see EventCfg).

Flat vs rough lives in flat_env_cfg.py / (later) rough_env_cfg.py.
"""
import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from . import mdp
from .robot_cfg import G1_INSPIRE_CFG, LEG_JOINT_NAMES

GAIT_PERIOD_S = 0.8  # Unitree's reference gait clock (g1_walking/config.py)


def _legs() -> SceneEntityCfg:
    """Leg joints in policy order. A fresh instance per term (they are resolved in place)."""
    return SceneEntityCfg("robot", joint_names=list(LEG_JOINT_NAMES), preserve_order=True)


def _held() -> SceneEntityCfg:
    """Every joint the policy does not control."""
    held = [".*waist.*", ".*shoulder.*", ".*elbow.*", ".*wrist.*", "^[LR]_.*"]
    return SceneEntityCfg("robot", joint_names=held)


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(LEG_JOINT_NAMES),
        preserve_order=True,
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # Order and scales are the contract with g1_walking.observation.
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, scale=(2.0, 2.0, 0.25), params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_biased, params={"asset_cfg": _legs()}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, params={"asset_cfg": _legs()}, noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        actions = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period_s": GAIT_PERIOD_S})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, scale=(2.0, 2.0, 0.25), params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _legs()})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, params={"asset_cfg": _legs()})
        actions = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period_s": GAIT_PERIOD_S})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    # -- task (stock G1 values)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=-1.0e-7, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])}
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=-2.0e-6, params={"asset_cfg": _legs()}
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis", "torso_link"]), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


@configclass
class EventCfg:
    """Domain randomization. Ranges are conservative placeholders; widen once
    training is stable and record them with each run."""

    # startup (per-env, fixed for the run)
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.25, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    add_torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.02, 0.02)},
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": _legs(),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    encoder_bias = EventTerm(
        func=mdp.randomize_encoder_bias,
        mode="startup",
        params={"asset_cfg": _legs(), "bias_range": (-0.015, 0.015)},
    )
    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": _legs(),
            "armature_distribution_params": (0.8, 1.5),
            "operation": "scale",
        },
    )

    # reset. Order matters: all joints go to default first, then the legs get
    # a small random offset on top, then the held joints' targets are pinned.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        },
    )
    reset_all_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    reset_leg_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": _legs(), "position_range": (-0.1, 0.1), "velocity_range": (0.0, 0.0)},
    )
    hold_joints = EventTerm(func=mdp.hold_joints_at_default, mode="reset", params={"asset_cfg": _held()})

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.7, 0.7), "y": (-0.7, 0.7)}},
    )


@configclass
class G1WalkEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Isaac Lab's stock default is a ray-cast height scanner on "base"; this
        # USD's root is "pelvis" and the flat task has no terrain to scan.
        self.scene.height_scanner = None

        # Replaces the stock term: mixes in forward-only / sideways-only /
        # rotate-in-place commands (see mdp/commands.py), resamples every 3-8 s
        # (stock: 10 s, i.e. two commands per episode), allows slow backward
        # walking, and samples yaw rate directly instead of via a heading target.
        self.commands.base_velocity = mdp.MixedVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.1,
            heading_command=False,
            debug_vis=False,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.4, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
            ),
        )
