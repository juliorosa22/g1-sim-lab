# g1-sim-lab

Minimal autonomous navigation stack for the Unitree G1 EDU humanoid
(29-DOF body + Inspire RH56DFTP hands), built and tested first in Isaac Sim
before deploying to hardware. Architecture: **Sense -> Plan -> Act**.

Current stage: **Sense** -- a G1 spawned in Isaac Sim's `Simple_Room` sample
scene, standing on a validated PD pose, publishing RTX lidar, RealSense-style
RGB-D camera, and IMU data over ROS2.

## Machines

This repo runs on two machines with different paths -- see "Per-machine
setup" below before running anything.

- **Desktop**: ROS2 Jazzy
- **Laptop**: ROS2 Humble, via Isaac Sim's bundled internal `rclpy` bridge

## Layout

```
run_sensors.py       entry point -- spawns the G1, attaches lidar+camera+IMU,
                      applies the standing pose, publishes over ROS2
diagnose_joints.py    standalone diagnostic: prints every joint's live
                      position/stiffness/damping/driveMode straight from the
                      simulation (not from any Isaac Lab config file)
g1_sim/
  config.py           single source of truth for every tunable value
  robot.py             G1Robot: load the USD, validate head_link, spawn
  environment.py       RoomEnvironment: load the Simple_Room scene
  standing.py           StandingController: real per-joint-group PD gains
                        + standing targets (see config.py's StandingPoseConfig
                        docstring for where the numbers come from)
  stage_utils.py        USD stage helpers (attribute setting, bbox printing)
  sensors/
    lidar.py            RTX Lidar + ROS2 point cloud publishing
    camera.py           RealSense-style RGB-D camera + ROS2 publishing
    imu.py               IMU + ROS2 publishing
```

## Per-machine setup

`RobotConfig.usd_path` (in `g1_sim/config.py`) can't be a single hardcoded
path since `unitree_sim_isaaclab` lives in a different location on each
machine. Set it via an environment variable instead of editing the file:

```bash
export G1_USD_PATH="/absolute/path/to/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd"
```

Find the real path on a given machine with:

```bash
find ~ -iname "g1_29dof_with_inspire_rev_1_0.usd" 2>/dev/null
```

If `G1_USD_PATH` isn't set, it falls back to the desktop's original path.

## Running

With the matching machine's environment active (`isaacenv`):

```bash
python run_sensors.py
```

Publishes:
- `/g1/lidar/points` (`sensor_msgs/PointCloud2`)
- `/g1/camera/color/image_raw` (`sensor_msgs/Image`, rgb8)
- `/g1/camera/depth/image_raw` (`sensor_msgs/Image`, 32FC1 depth)
- `/g1/camera/color/camera_info` (`sensor_msgs/CameraInfo`)
- `/g1/imu/data` (`sensor_msgs/Imu`)

To diagnose joint/gain issues directly from the live simulation rather than
guessing from Isaac Lab config files:

```bash
python diagnose_joints.py
```

## Next up

- Confirm the standing pose holds on both machines
- Re-verify all 5 topics reflect a genuinely standing, stationary robot
- Stage 2: state estimation / SLAM (likely FAST-LIO2-style, lidar+IMU tightly coupled)
- Stage 3: Nav2 navigation
