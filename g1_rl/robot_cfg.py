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
                ".*_ankle_pitch_joint": 35.0, ".*_ankle_roll_joint": 35.0,
            },
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0, ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0, ".*_knee_joint": 139.0,
                ".*_ankle_pitch_joint": 35.0, ".*_ankle_roll_joint": 35.0,
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
