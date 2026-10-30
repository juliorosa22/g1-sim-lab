"""Articulation config for the Inspire-hand G1 used for walking-policy training.

Trimmed copy of unitree_sim_isaaclab's G129_CFG_WITH_INSPIRE_WHOLEBODY
(robots/unitree.py) -- same USD, effort/velocity limits, armature and
non-leg gains. Copied rather than imported because that module reads
PROJECT_ROOT from the environment at import time and defines ~20 other robots.

What differs from the original, deliberately:
  * Leg PD gains are Unitree's own deployment gains (unitree_rl_gym
    configs/g1.yaml kps/kds, the same values g1_walking/config.py uses), NOT
    the sim config's 200/5 hips and 20/2 ankles. A policy outputs targets that
    the motors' PD loop tracks with whatever gains the deployment sends, so
    training must use the gains hardware will run.
  * Legs use DelayedPDActuatorCfg (explicit PD + randomized command delay) so
    actuation latency is part of training. Everything else stays implicit.
  * Initial leg pose is the walking policy's default pose, not the standing
    pose in g1_sim.config.StandingPoseConfig.

No isaacsim/omni imports beyond isaaclab's config classes, so this stays
importable the same way the stock Isaac Lab robot configs are.
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from g1_sim.config import RobotConfig

# Order the policy's actions/observations use: per leg hip_pitch, hip_roll,
# hip_yaw, knee, ankle_pitch, ankle_roll. Duplicated from
# g1_walking/config.py::LEG_JOINT_NAMES on purpose (g1_rl does not import the
# runtime package); tests/test_obs_contract.py asserts the two stay equal.
LEG_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
)

# Ankle torque limit is 50 N*m: mjlab (2 x 25 N*m motors through the 4-bar
# linkage) and mujoco_playground (+-50) agree; the sim config's 35 was too low.

WAIST_JOINT_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# Max command delay in physics steps (dt=0.005 s -> 4 steps = 20 ms, i.e. one
# control step). Placeholder range, to be widened once training is stable.
MAX_ACTUATOR_DELAY_STEPS = 4

G1_INSPIRE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=RobotConfig().usd_path,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80),
        joint_pos={
            # Walking policy default pose (g1_walking/config.py default_angles).
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.3,
            ".*_ankle_pitch_joint": -0.2,
            # Arms held in the validated standing pose (StandingPoseConfig).
            ".*_elbow_joint": 0.87,
            "left_shoulder_roll_joint": 0.18,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.18,
            "right_shoulder_pitch_joint": 0.35,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint",
                ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 88.0, ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0, ".*_knee_joint": 139.0,
                ".*_ankle_pitch_joint": 50.0, ".*_ankle_roll_joint": 50.0,
            },
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0, ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0, ".*_knee_joint": 139.0,
                ".*_ankle_pitch_joint": 50.0, ".*_ankle_roll_joint": 50.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 32.0, ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0, ".*_knee_joint": 20.0,
                ".*_ankle_pitch_joint": 30.0, ".*_ankle_roll_joint": 30.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0, ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0, ".*_knee_joint": 20.0,
                ".*_ankle_pitch_joint": 30.0, ".*_ankle_roll_joint": 30.0,
            },
            # Unitree deployment gains (configs/g1.yaml kps/kds).
            stiffness={
                ".*_hip_yaw_joint": 100.0, ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0, ".*_knee_joint": 150.0,
                ".*_ankle_pitch_joint": 40.0, ".*_ankle_roll_joint": 40.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.0, ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0, ".*_knee_joint": 4.0,
                ".*_ankle_pitch_joint": 2.0, ".*_ankle_roll_joint": 2.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=MAX_ACTUATOR_DELAY_STEPS,
        ),
        # Held joints from here down: never commanded by the policy, kept at
        # their default pose by g1_rl.mdp.events.hold_joints_at_default.
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[".*waist.*"],
            effort_limit_sim={
                ".*waist_yaw_joint": 88.0, ".*waist_roll_joint": 35.0, ".*waist_pitch_joint": 35.0,
            },
            velocity_limit_sim={
                ".*waist_yaw_joint": 32.0, ".*waist_roll_joint": 30.0, ".*waist_pitch_joint": 30.0,
            },
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
        "shoulders": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness=100.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_yaw_joint", ".*_elbow_joint"],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness=50.0,
            damping=2.0,
            armature=0.01,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_.*"],
            effort_limit_sim={
                ".*_wrist_yaw_joint": 5.0, ".*_wrist_roll_joint": 25.0, ".*_wrist_pitch_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_wrist_yaw_joint": 22.0, ".*_wrist_roll_joint": 37.0, ".*_wrist_pitch_joint": 22.0,
            },
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_index_proximal_joint", ".*_index_intermediate_joint",
                ".*_middle_proximal_joint", ".*_middle_intermediate_joint",
                ".*_pinky_proximal_joint", ".*_pinky_intermediate_joint",
                ".*_ring_proximal_joint", ".*_ring_intermediate_joint",
                ".*_thumb_proximal_yaw_joint", ".*_thumb_proximal_pitch_joint",
                ".*_thumb_intermediate_joint", ".*_thumb_distal_joint",
            ],
            effort_limit=100.0,
            velocity_limit=50.0,
            stiffness=1000.0,
            damping=15.0,
            armature=0.0,
        ),
    },
)


# ---------------------------------------------------------------------------------------------
# Gain profiles. "sim" (default) = the gains v1-v6 were trained with. "unitree" = the gains Unitree's
# own onboard controller sends on THIS robot, recorded by the first hardware session
# (robot_deploy/measurements/20261029, docs/sim_parameter_proposal.md items 2-5):
#   legs kp 100/100/100/150/60/60, kd 4/4/4/5/3/3; waist 200/6; shoulders and elbow 60/3; wrists 40/3.
# Select with the environment variable G1_GAIN_PROFILE=unitree (used for the sensitivity evaluations of
# the existing policies); new tasks pick a profile explicitly.
# ---------------------------------------------------------------------------------------------
_UNITREE_LEG_KP = {".*_hip_yaw_joint": 100.0, ".*_hip_roll_joint": 100.0, ".*_hip_pitch_joint": 100.0,
                   ".*_knee_joint": 150.0, ".*_ankle_pitch_joint": 60.0, ".*_ankle_roll_joint": 60.0}
_UNITREE_LEG_KD = {".*_hip_yaw_joint": 4.0, ".*_hip_roll_joint": 4.0, ".*_hip_pitch_joint": 4.0,
                   ".*_knee_joint": 5.0, ".*_ankle_pitch_joint": 3.0, ".*_ankle_roll_joint": 3.0}
GAIN_PROFILES = {
    "sim": {},
    "unitree": {
        "legs": {"stiffness": _UNITREE_LEG_KP, "damping": _UNITREE_LEG_KD},
        "waist": {"stiffness": 200.0, "damping": 6.0},
        "shoulders": {"stiffness": 60.0, "damping": 3.0},
        "arms": {"stiffness": 60.0, "damping": 3.0},
        "wrist": {"stiffness": 40.0, "damping": 3.0},
    },
}


def with_gain_profile(cfg: ArticulationCfg, profile: str) -> ArticulationCfg:
    """Returns `cfg` with the actuator gains of the named profile (other actuator settings untouched)."""
    if profile not in GAIN_PROFILES:
        raise ValueError(f"unknown gain profile {profile!r}; choose from {sorted(GAIN_PROFILES)}")
    changes = GAIN_PROFILES[profile]
    if not changes:
        return cfg
    actuators = dict(cfg.actuators)
    for name, gains in changes.items():
        if name in actuators:
            actuators[name] = actuators[name].replace(**gains)
    return cfg.replace(actuators=actuators)


_G1_BASE_CFG = G1_INSPIRE_CFG  # the unmodified config, for the real-plant builders below
GAIN_PROFILE = os.environ.get("G1_GAIN_PROFILE", "sim")
G1_INSPIRE_CFG = with_gain_profile(G1_INSPIRE_CFG, GAIN_PROFILE)

# Variant for tasks where the policy also drives the waist (v3, v4): same gains as
# the held waist above, but explicit PD with randomized command delay like the legs.
G1_INSPIRE_WAIST_CFG = G1_INSPIRE_CFG.replace(
    actuators={
        **G1_INSPIRE_CFG.actuators,
        "waist": DelayedPDActuatorCfg(
            joint_names_expr=[".*waist.*"],
            effort_limit={".*waist_yaw_joint": 88.0, ".*waist_roll_joint": 35.0, ".*waist_pitch_joint": 35.0},
            effort_limit_sim={".*waist_yaw_joint": 88.0, ".*waist_roll_joint": 35.0, ".*waist_pitch_joint": 35.0},
            velocity_limit={".*waist_yaw_joint": 32.0, ".*waist_roll_joint": 30.0, ".*waist_pitch_joint": 30.0},
            velocity_limit_sim={".*waist_yaw_joint": 32.0, ".*waist_roll_joint": 30.0, ".*waist_pitch_joint": 30.0},
            stiffness=GAIN_PROFILES.get(GAIN_PROFILE, {}).get("waist", {}).get("stiffness", 200.0),
            damping=GAIN_PROFILES.get(GAIN_PROFILE, {}).get("waist", {}).get("damping", 5.0),
            armature=0.01,
            min_delay=0,
            max_delay=MAX_ACTUATOR_DELAY_STEPS,
        ),
    }
)


# ---------------------------------------------------------------------------------------------
# Real-plant variants (v7, v8). Built from the measurements of the first hardware session
# (robot_deploy/measurements/20261029, docs/sim_parameter_proposal.md):
#   * Unitree's onboard gains (legs kp 100/100/100/150/60/60, kd 4/4/4/5/3/3; waist yaw 200/6; arms 60/3; wrists 40/3)
#   * waist roll and pitch are LOCKED on this unit: they are held rigidly at 0 here, never acted on or observed
#   * ankle speed limit 37 rad/s (Unitree's URDF; the sim config had 30)
# Torque shortfall (3-7% measured) needs no separate term: the +/-20% stiffness/damping randomisation covers it.
# ---------------------------------------------------------------------------------------------
def make_real_cfg(drive_waist_yaw: bool) -> ArticulationCfg:
    """drive_waist_yaw=True: waist yaw is an explicit PD the policy commands (with delay), like the legs.
    False: waist yaw is held at its default by an implicit PD (policy does not touch it)."""
    cfg = with_gain_profile(_G1_BASE_CFG, "unitree")
    actuators = dict(cfg.actuators)

    legs = actuators["legs"]
    vel = dict(legs.velocity_limit)
    vel[".*_ankle_pitch_joint"] = 37.0
    vel[".*_ankle_roll_joint"] = 37.0
    actuators["legs"] = legs.replace(velocity_limit=vel, velocity_limit_sim=dict(vel))

    del actuators["waist"]
    if drive_waist_yaw:
        actuators["waist_yaw"] = DelayedPDActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit=88.0, effort_limit_sim=88.0, velocity_limit=32.0, velocity_limit_sim=32.0,
            stiffness=200.0, damping=6.0, armature=0.01, min_delay=0, max_delay=MAX_ACTUATOR_DELAY_STEPS,
        )
    else:
        actuators["waist_yaw"] = ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0, velocity_limit_sim=32.0, stiffness=200.0, damping=6.0, armature=0.01,
        )
    actuators["waist_locked"] = ImplicitActuatorCfg(
        joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
        effort_limit_sim=500.0, velocity_limit_sim=30.0, stiffness=2000.0, damping=100.0, armature=0.01,
    )
    return cfg.replace(actuators=actuators)


WAIST_YAW_JOINT = "waist_yaw_joint"
WAIST_LOCKED_JOINTS = ("waist_roll_joint", "waist_pitch_joint")
G1_REAL_STAND_CFG = make_real_cfg(drive_waist_yaw=False)  # v7: legs-only policy, waist yaw held
G1_REAL_WALK_CFG = make_real_cfg(drive_waist_yaw=True)    # v8: legs + waist yaw driven
