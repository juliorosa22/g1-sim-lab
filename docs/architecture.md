# Architecture: hierarchical planner + locomotion policy

Design reference for g1-sim-lab. Describes how the Sense -> Plan -> Act stack
is split into two layers, and the planned development order.

## Two layers

| | Planner (high level) | Locomotion policy (low level) |
|---|---|---|
| Rate | ~5-20 Hz (Nav2 controller ~20 Hz, global planning slower) | 50 Hz |
| Inputs | Lidar, camera, SLAM pose, goal | Proprioception only (IMU, joint encoders) |
| Output | `cmd_vel` = `[vx, vy, yaw_rate]` | 12 leg joint targets |
| Stack | ROS2 (SLAM, Nav2) | PyTorch, inside the sim loop |
| Code | (planned) | `g1_walking/` |

The interface between the layers is one message: `geometry_msgs/Twist` on
`/cmd_vel`. It maps onto `WalkingPolicyConfig.cmd`, which is currently fixed
at `(0.5, 0.0, 0.0)`.

```
 Lidar / Camera / IMU --> SLAM --> Nav2 --> /cmd_vel --+
   (g1_sim/sensors)                                    |
                                                       v
 joint encoders + IMU gyro/orientation --> observation (47) --> LSTM policy
                                                       |
                                                       v
                                      12 leg joint targets (PD) --> robot
```

## Walking policy observations (47)

Checkpoint: Unitree's `motion.pt` (unitree_rl_gym), a TorchScript LSTM,
47 obs -> 12 actions, legs only.

| Indices | Content | Scale | Real-robot source |
|---|---|---|---|
| 0-2 | Base angular velocity, body frame | x0.25 | IMU gyro |
| 3-5 | Gravity direction, body frame | none | IMU orientation |
| 6-8 | Velocity command `[vx, vy, yaw_rate]` | x(2, 2, 0.25) | planner / joystick |
| 9-20 | Leg joint positions - default angles | x1.0 | motor encoders |
| 21-32 | Leg joint velocities | x0.05 | motor encoders |
| 33-44 | Previous action | raw | software |
| 45-46 | Gait clock `[sin, cos]`, 0.8 s period | none | software |

All inputs are proprioceptive, so the policy needs no state estimator beyond
the IMU. In sim, base state is read from physics ground truth
(`g1_walking/base_state.py`) rather than from the simulated IMU; on hardware
the IMU provides the same quantities, noisier.

## Published ROS2 topics (current)

`/g1/lidar/points`, `/g1/camera/color/image_raw`,
`/g1/camera/depth/image_raw`, `/g1/camera/color/camera_info`, `/g1/imu/data`.
These serve the planner. The walking policy does not use them.

## Design rules

1. **Planner and policy are decoupled.** The planner can lag or fail without
   affecting balance, because the policy uses only proprioception.
2. **Hold the latest command between planner updates**, and zero it if no
   `/cmd_vel` arrives within a timeout (suggested 0.5 s) so a crashed planner
   cannot leave the robot walking.
3. **Clamp commands** to the velocity range the policy was trained on (check
   `unitree_rl_gym` training config; not yet verified here). Configure Nav2
   velocity limits to match.
4. **Nav2 needs a TF tree** (`map -> odom -> base_link -> sensor frames`) and
   odometry. Sensors currently publish frames (`lidar_frame`, `imu_link`,
   camera frames) but nothing connects them.
5. **Arms and waist** are not controlled by the policy; a separate controller
   (`StandingController` in sim) must hold them stable.

## Development order

1. Verify walking in isolation: run `verify_walking_apis.py` (confirms the
   world-vs-body angular velocity assumption) and check the robot walks at the
   fixed command.
2. Add the command interface: `/cmd_vel` subscriber with clamping and timeout
   driving `cfg.cmd`. Test with `ros2 topic pub` or teleop.
3. Publish missing ROS outputs: `/joint_states`, `/odom` (ground truth first),
   and the TF tree.
4. Merge standing/walking and sensors into one entry point
   (`run_sensors.py` stands still, `run_walking.py` has no sensors).
5. SLAM, then Nav2. Test Nav2 on ground-truth odometry first, then swap in
   SLAM, so faults can be isolated per piece.

## Known risks

- Sim-to-real gap: the policy was trained in Isaac Gym and is run in Isaac Sim
  (PhysX); IMU noise, latency and actuator differences apply on hardware.
- Gait motion of the walking body will appear in lidar/camera data; SLAM must
  tolerate it.
- Angular velocity frame (world vs body) in `base_state.py` is unverified
  until `verify_walking_apis.py` is run.
