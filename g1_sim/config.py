"""Central configuration for the G1 sensor-testing simulation.

Single source of truth for paths, mount points, and sensor parameters so nothing
is duplicated or hardcoded across modules. This module has no isaacsim/omni/pxr
dependencies, so it's safe to import before SimulationApp() is constructed --
useful for deciding e.g. headless mode from config before starting the app.
"""
import os
from dataclasses import dataclass, field
from typing import Tuple

# Per-machine override: this repo is shared between the desktop and the laptop
# (different usernames, different clone locations for unitree_sim_isaaclab), so
# the USD path can never safely be a single hardcoded string. Set this once per
# machine, e.g. in ~/.bashrc:
#   export G1_USD_PATH="/home/julio/Programming/UnitreeRobot/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd"
# Falls back to the desktop's path if the env var isn't set, so existing desktop
# runs keep working unchanged.
_DEFAULT_USD_PATH = (
    "/home/cesar/Programming/UnitreeG1/unitree_sim_isaaclab/assets/robots/"
    "g1-29dof_wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd"
)


@dataclass
class LidarConfig:
    prim_path: str = "/lidar_livox"
    mount_translation: Tuple[float, float, float] = (0.05, 0.0, 0.12)
    mount_orientation_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    num_channels: int = 32
    elevation_min_deg: float = -7.0
    elevation_max_deg: float = 52.0  # -7 + 59 deg total vertical span
    near_range_m: float = 0.1
    far_range_m: float = 40.0
    min_reflectance: float = 0.1
    min_reflection_range_m: float = 40.0
    report_rate_hz: int = 10240
    scan_rate_hz: int = 10
    range_accuracy_m: float = 0.03
    azimuth_error_std: float = 0.01
    elevation_error_std: float = 0.01
    wave_length_nm: float = 905.0
    topic: str = "/g1/lidar/points"
    frame_id: str = "lidar_frame"
    graph_path: str = "/Graph/ROS_Lidar"


@dataclass
class CameraConfig:
    prim_name: str = "camera_realsense"
    mount_translation: Tuple[float, float, float] = (0.06, 0.0, 0.10)
    mount_orientation_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    width: int = 1280
    height: int = 720
    horizontal_fov_deg: float = 69.0  # approximates the RealSense D435i RGB stream
    horizontal_aperture_mm: float = 20.955  # USD/Isaac Sim default film-back width
    near_clip_m: float = 0.1
    far_clip_m: float = 10.0  # D435i usable depth range is roughly 0.1-10m
    rgb_topic: str = "/g1/camera/color/image_raw"
    depth_topic: str = "/g1/camera/depth/image_raw"
    camera_info_topic: str = "/g1/camera/color/camera_info"
    color_frame_id: str = "camera_color_frame"
    depth_frame_id: str = "camera_depth_frame"
    graph_path: str = "/Graph/ROS_Camera"


@dataclass
class ImuConfig:
    prim_path: str = "/imu_livox"
    # Co-located with the lidar mount point -- approximates a Livox-style lidar
    # with a built-in IMU, which is what a tightly-coupled LIO pipeline (e.g.
    # FAST-LIO2) expects: lidar and IMU rigidly attached and time-synced. A
    # separate body-frame IMU (e.g. at the pelvis, for the robot's own balance
    # control) would be a second ImuSensor instance with a different parent_link.
    mount_translation: Tuple[float, float, float] = (0.05, 0.0, 0.12)
    mount_orientation_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    sensor_period: float = -1.0  # -1 = sample every physics step
    linear_acceleration_filter_size: int = 1
    angular_velocity_filter_size: int = 1
    orientation_filter_size: int = 1
    topic: str = "/g1/imu/data"
    frame_id: str = "imu_link"
    graph_path: str = "/Graph/ROS_Imu"


@dataclass
class RobotConfig:
    usd_path: str = field(
        default_factory=lambda: os.environ.get("G1_USD_PATH", _DEFAULT_USD_PATH)
    )
    prim_path: str = "/World/G1"
    head_link_name: str = "head_link"
    # x/y clear of Simple_Room's table (table footprint is x in [-1.59,1.59],
    # y in [-0.81,0.81] -- confirmed via RoomConfig.print_bounding_boxes).
    # z=0.80, NOT 0.0 -- this matches unitree_sim_isaaclab's own
    # G129_CFG_WITH_INSPIRE_WHOLEBODY init_state.pos for this exact USD
    # (robots/unitree.py); the bent-knee standing pose in StandingPoseConfig
    # needs that base height for the feet to land on the floor correctly.
    spawn_position: Tuple[float, float, float] = (2.0, 2.0, 0.80)
    # Upright, scalar-first (w,x,y,z) -- confirmed convention from
    # isaacsim.core.prims's SingleXFormPrim.set_world_pose (single_prim_wrapper.py).
    # Must be passed explicitly on every re-spawn: set_world_pose leaves
    # orientation UNCHANGED when this argument is omitted, which is why an
    # earlier fix that reset only position failed to correct a robot that had
    # already toppled during the pre-standing-pose physics window.
    spawn_orientation_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class RoomConfig:
    # Relative to Isaac Sim's assets root (get_assets_root_path())
    usd_relative_path: str = "/Isaac/Environments/Simple_Room/simple_room.usd"
    prim_path: str = "/World/Room"
    print_bounding_boxes: bool = False  # flip on when picking a new spawn point


@dataclass
class StandingPoseConfig:
    """Standing joint targets and PD gains, transcribed from unitree_sim_isaaclab's
    own G129_CFG_WITH_INSPIRE_WHOLEBODY (robots/unitree.py) -- the real Isaac Lab
    ArticulationCfg for this exact USD (g1_29dof_with_inspire_rev_1_0.usd). This
    is the same resting pose their own RL environments hold before any policy
    runs, so it's about as validated a "just stand there" pose as exists for
    this robot.

    Confirmed via diagnose_joints.py that the USD's own baked-in drive gains are
    a flat stiffness=100/damping=1 on every one of the 53 joints -- nowhere near
    enough to hold hip/knee/ankle against body weight (and, in the other
    direction, way too little for the tiny finger joints to matter, but way too
    much of a mismatch for the legs). These per-group values replace that flat
    default entirely.

    Isaac Lab's config expresses joints as regexes (e.g. ".*_hip_yaw_joint");
    these dicts are that same data pre-expanded to the 53 concrete joint names
    this specific G1 variant has (matches diagnose_joints.py's dof_names exactly
    -- legs 11 + feet 4 + shoulders 4 + arms 4 + wrist 6 + hands 24 = 53).
    """

    joint_pos_overrides: dict = field(default_factory=lambda: {
        "left_hip_pitch_joint": -0.20,
        "right_hip_pitch_joint": -0.20,
        "left_knee_joint": 0.42,
        "right_knee_joint": 0.42,
        "left_ankle_pitch_joint": -0.23,
        "right_ankle_pitch_joint": -0.23,
        "left_elbow_joint": 0.87,
        "right_elbow_joint": 0.87,
        "left_shoulder_roll_joint": 0.18,
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_roll_joint": -0.18,
        "right_shoulder_pitch_joint": 0.35,
        # Every other joint (hip_yaw, hip_roll, waist_*, ankle_roll,
        # shoulder_yaw, wrist_*, all finger joints) is 0.0 in the real config
        # too -- StandingController falls back to 0.0 for anything not listed
        # here, so they're omitted rather than spelled out redundantly.
    })

    # (stiffness, damping) per joint, transcribed from the real actuator groups.
    gains: dict = field(default_factory=lambda: {
        **{n: (150.0, 5.0) for n in (
            "left_hip_yaw_joint", "right_hip_yaw_joint",
            "left_hip_roll_joint", "right_hip_roll_joint",
        )},
        **{n: (200.0, 5.0) for n in (
            "left_hip_pitch_joint", "right_hip_pitch_joint",
            "left_knee_joint", "right_knee_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        )},
        **{n: (20.0, 2.0) for n in (
            "left_ankle_pitch_joint", "right_ankle_pitch_joint",
            "left_ankle_roll_joint", "right_ankle_roll_joint",
        )},
        **{n: (100.0, 2.0) for n in (
            "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
            "left_shoulder_roll_joint", "right_shoulder_roll_joint",
        )},
        **{n: (50.0, 2.0) for n in (
            "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
            "left_elbow_joint", "right_elbow_joint",
        )},
        **{n: (40.0, 2.0) for n in (
            "left_wrist_yaw_joint", "right_wrist_yaw_joint",
            "left_wrist_roll_joint", "right_wrist_roll_joint",
            "left_wrist_pitch_joint", "right_wrist_pitch_joint",
        )},
        **{
            f"{side}_{finger}_joint": (1000.0, 15.0)
            for side in ("L", "R")
            for finger in (
                "index_proximal", "index_intermediate",
                "middle_proximal", "middle_intermediate",
                "pinky_proximal", "pinky_intermediate",
                "ring_proximal", "ring_intermediate",
                "thumb_proximal_yaw", "thumb_proximal_pitch",
                "thumb_intermediate", "thumb_distal",
            )
        },
    })


@dataclass
class SimulationConfig:
    headless: bool = False
    robot: RobotConfig = field(default_factory=RobotConfig)
    room: RoomConfig = field(default_factory=RoomConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    standing: StandingPoseConfig = field(default_factory=StandingPoseConfig)
