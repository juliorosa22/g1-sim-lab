#!/usr/bin/env python3
"""Entry point: G1 standing + walking using Unitree's pretrained legs-only
locomotion policy (unitree_rl_gym's motion.pt), ported from their MuJoCo
reference deployment (deploy/deploy_mujoco/deploy_mujoco.py) onto our Isaac
Sim setup.

Reuses g1_sim's robot loading, flat-ground environment, and standing
controller exactly as run_sensors.py does -- the new pieces all live in
g1_walking/: the policy wrapper, the observation builder, the base-state
reader, and the leg controller that commands the 12 leg joints while the
rest of the body holds the standing pose.

Run with the matching machine's environment active first (isaacenv), then:
    python run_walking.py

Before trusting this for real walking, please run verify_walking_apis.py
once -- one frame-convention assumption in g1_walking/base_state.py (world
vs. body frame for the base angular velocity) couldn't be checked against
your installed isaacsim source from this dev container.
"""
from g1_sim.config import SimulationConfig
from g1_walking.config import WalkingPolicyConfig

config = SimulationConfig()
walking_config = WalkingPolicyConfig()

from isaacsim import SimulationApp

sim = SimulationApp({"headless": config.headless})

# isaacsim/omni/pxr imports only work once SimulationApp has started the Kit
# process above -- keep it that way (same requirement as run_sensors.py).
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.sensors.physics")
enable_extension("isaacsim.ros2.bridge")
sim.update()

import numpy as np
import omni.timeline
from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import SingleArticulation

from g1_sim.environment import FlatGroundEnvironment, RoomEnvironment
from g1_sim.robot import G1Robot
from g1_sim.standing import StandingController
from g1_walking.base_state import BaseState
from g1_walking.leg_controller import LegController
from g1_walking.observation import ObservationBuilder
from g1_walking.policy import WalkingPolicy


def main():
    robot = G1Robot(config.robot)
    robot.load(sim)

    if config.use_flat_ground:
        environment = FlatGroundEnvironment(config.ground)
    else:
        environment = RoomEnvironment(config.room)
    environment.load(sim)

    robot.spawn(sim)

    # Start physics before touching the articulation -- same requirement
    # run_sensors.py and diagnose_joints.py both rely on.
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        sim.update()

    articulation = SingleArticulation(prim_path=config.robot.prim_path)
    articulation.initialize()

    articulation.set_solver_position_iteration_count(config.robot.solver_position_iteration_count)
    articulation.set_solver_velocity_iteration_count(config.robot.solver_velocity_iteration_count)
    articulation.set_enabled_self_collisions(config.robot.enable_self_collisions)

    # Re-snap base pose/velocity before applying the standing pose -- see
    # RobotConfig / run_sensors.py for why (the robot may have sagged during
    # the physics ticks above, under the USD's original weak gains).
    robot.spawn(sim)
    articulation.set_linear_velocity(np.zeros(3))
    articulation.set_angular_velocity(np.zeros(3))

    # Same validated standing pose as run_sensors.py -- gets the robot to a
    # stable, upright rest state before the walking policy ever runs.
    StandingController(articulation, config.standing).apply()

    leg_controller = LegController(articulation, walking_config, config.standing)
    leg_controller.install_gains()

    base_state = BaseState(articulation)
    obs_builder = ObservationBuilder(walking_config)
    policy = WalkingPolicy(walking_config.checkpoint_path)

    # Derive how many physics steps make up one 50Hz control period from the
    # stage's actual physics dt, rather than assuming it matches MuJoCo's
    # simulation_dt=0.002 the reference config was written for.
    try:
        physics_dt = SimulationContext.instance().get_physics_dt()
    except Exception as exc:  # pragma: no cover -- fallback for an unverified API path
        physics_dt = 1.0 / 60.0
        print(
            f"WARNING: could not read physics dt via SimulationContext ({exc}); "
            f"assuming Isaac Sim's default {physics_dt:.5f}s (60Hz). If that's wrong, "
            "the walking policy's control rate won't match its 50Hz training rate -- "
            "run verify_walking_apis.py to check."
        )

    control_period_s = walking_config.simulation_dt * walking_config.control_decimation
    steps_per_control = max(1, round(control_period_s / physics_dt))
    print(
        f"physics_dt={physics_dt:.5f}s -> policy control every {steps_per_control} physics "
        f"step(s) (~{1.0 / (physics_dt * steps_per_control):.1f} Hz; reference is 50 Hz)"
    )
    print(f"Starting walking policy control loop (cmd={walking_config.cmd} [vx, vy, yaw_rate])...")

    step_count = 0
    try:
        while True:
            sim.update()
            step_count += 1
            if step_count % steps_per_control == 0:
                quat_wxyz, ang_vel_body = base_state.read()
                leg_pos, leg_vel = leg_controller.read_leg_state()
                obs = obs_builder.build(
                    base_quat_wxyz=quat_wxyz,
                    base_ang_vel_body=ang_vel_body,
                    leg_pos=leg_pos,
                    leg_vel=leg_vel,
                    dt_since_last_control=control_period_s,
                )
                action = policy.act(obs)
                obs_builder.record_action(action)
                target = action * walking_config.action_scale + np.array(walking_config.default_angles)
                leg_controller.apply(target)
    except KeyboardInterrupt:
        pass
    finally:
        timeline.stop()
        sim.close()


if __name__ == "__main__":
    main()
