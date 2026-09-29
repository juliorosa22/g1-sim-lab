#!/usr/bin/env python3
"""Entry point: G1 sensor suite (RTX Lidar + RealSense-style RGB-D camera + IMU)
publishing over ROS2, tested in Isaac Sim's Simple_Room sample environment.

Run with the matching machine's environment active first (isaacenv), then:
    python run_sensors.py

Every tunable value (paths, mount points, topics, FOVs, spawn position) lives in
g1_sim/config.py -- edit SimulationConfig's defaults there rather than in here.
"""
# config.py has zero isaacsim/omni/pxr dependencies, so it's safe to import
# before SimulationApp() exists -- used here just to decide headless mode.
from g1_sim.config import SimulationConfig

config = SimulationConfig()

from isaacsim import SimulationApp

sim = SimulationApp({"headless": config.headless})

# Everything below this line imports omni/pxr modules, which only work once
# SimulationApp has started the Kit process above -- keep it that way.
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.sensors.rtx")
enable_extension("isaacsim.sensors.physics")
enable_extension("isaacsim.ros2.bridge")
sim.update()

import numpy as np
import omni.timeline
from isaacsim.core.prims import SingleArticulation

from g1_sim.robot import G1Robot
from g1_sim.environment import FlatGroundEnvironment, RoomEnvironment
from g1_sim.sensors import LidarSensor, CameraSensor, ImuSensor
from g1_sim.standing import StandingController


def main():
    robot = G1Robot(config.robot)
    robot.load(sim)

    # config.use_flat_ground=True while validating standing (default) -- isolates
    # the standing problem from Simple_Room's own uninspected floor material.
    # Set it False in config.py once standing is confirmed, to go back to the room.
    if config.use_flat_ground:
        environment = FlatGroundEnvironment(config.ground)
    else:
        environment = RoomEnvironment(config.room)
    environment.load(sim)

    robot.spawn(sim)

    lidar = LidarSensor(config.lidar, parent_link=robot.head_link_path)
    lidar.create()
    lidar.build_ros_graph()

    camera = CameraSensor(config.camera, parent_link=robot.head_link_path)
    camera.create()
    camera.build_ros_graph()

    imu = ImuSensor(config.imu, parent_link=robot.head_link_path)
    imu.create()
    imu.build_ros_graph()

    # Start physics before touching the articulation -- SingleArticulation.initialize()
    # needs a live physics_sim_view, which only exists once play() has run a few ticks
    # (same requirement diagnose_joints.py ran into).
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        sim.update()

    articulation = SingleArticulation(prim_path=config.robot.prim_path)
    articulation.initialize()

    # Articulation root properties Isaac Lab's own config sets for this exact
    # USD (G129_CFG_WITH_INSPIRE_WHOLEBODY) but our raw-load script never has --
    # see RobotConfig for why these matter.
    articulation.set_solver_position_iteration_count(config.robot.solver_position_iteration_count)
    articulation.set_solver_velocity_iteration_count(config.robot.solver_velocity_iteration_count)
    articulation.set_enabled_self_collisions(config.robot.enable_self_collisions)

    # The robot may have already sagged/toppled during the physics ticks above,
    # under the USD's original weak gains -- re-snap the base before applying
    # the standing pose, so StandingController starts from a clean state rather
    # than correcting a fall already in progress.
    robot.spawn(sim)
    articulation.set_linear_velocity(np.zeros(3))
    articulation.set_angular_velocity(np.zeros(3))

    StandingController(articulation, config.standing).apply()

    print("Publishing:")
    print(f"  {config.lidar.topic:<32} (sensor_msgs/PointCloud2)")
    print(f"  {config.camera.rgb_topic:<32} (sensor_msgs/Image, rgb8)")
    print(f"  {config.camera.depth_topic:<32} (sensor_msgs/Image, 32FC1 depth)")
    print(f"  {config.camera.camera_info_topic:<32} (sensor_msgs/CameraInfo)")
    print(f"  {config.imu.topic:<32} (sensor_msgs/Imu)")

    try:
        while True:
            sim.update()
    except KeyboardInterrupt:
        pass
    finally:
        timeline.stop()
        sim.close()


if __name__ == "__main__":
    main()
