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
import argparse

from g1_sim.config import SimulationConfig
from g1_walking.config import WalkingPolicyConfig

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=None,
                    help="stop after this much sim time and print a pass/fail summary")
parser.add_argument("--headless", action="store_true", help="override config.headless")
parser.add_argument("--cmd", type=float, nargs=3, metavar=("VX", "VY", "YAW"), default=None,
                    help="override the velocity command")
args, _ = parser.parse_known_args()

config = SimulationConfig()
if args.headless:
    config.headless = True
walking_config = WalkingPolicyConfig()
if args.cmd is not None:
    walking_config.cmd = tuple(args.cmd)

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

# SimulationContext.instance() only returns something once a SimulationContext
# has actually been constructed in this process -- SimulationApp() alone
# doesn't create one, which is why the first version of this script saw
# instance() come back None and silently fell back to an assumed 60Hz.
# Constructing it here (before the robot/ground are loaded) also creates the
# stage's physics scene if one doesn't already exist yet.
sim_context = SimulationContext(
    physics_dt=walking_config.physics_dt, rendering_dt=walking_config.physics_dt
)

from g1_sim.environment import FlatGroundEnvironment, RoomEnvironment
from g1_sim.robot import G1Robot
from g1_sim.standing import StandingController
from g1_walking.base_state import BaseState
from g1_walking.leg_controller import LegController
from g1_walking.observation import ObservationBuilder, quat_rotate_inverse
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
    # simulation_dt=0.002 the reference config was written for. sim_context
    # was constructed at module load time (above) specifically so this
    # instance() call resolves instead of returning None.
    try:
        physics_dt = sim_context.get_physics_dt()
    except Exception as exc:  # pragma: no cover -- defensive fallback only
        physics_dt = 1.0 / 60.0
        print(
            f"WARNING: could not read physics dt from SimulationContext ({exc}); "
            f"assuming Isaac Sim's default {physics_dt:.5f}s (60Hz). If that's wrong, "
            "the walking policy's control rate won't match its 50Hz training rate."
        )

    control_period_s = walking_config.simulation_dt * walking_config.control_decimation
    steps_per_control = max(1, round(control_period_s / physics_dt))
    print(
        f"physics_dt={physics_dt:.5f}s -> policy control every {steps_per_control} physics "
        f"step(s) (~{1.0 / (physics_dt * steps_per_control):.1f} Hz; reference is 50 Hz)"
    )
    print(f"Starting walking policy control loop (cmd={walking_config.cmd} [vx, vy, yaw_rate])...")

    # Telemetry: a fall is the base dropping well below its standing height or
    # tilting far from upright (gravity-z in body frame is -1 when upright).
    FALL_HEIGHT_M = 0.4
    FALL_TILT_GZ = -0.5
    LOG_EVERY_S = 0.5
    pos0, _ = articulation.get_world_pose()
    start_z = float(pos0[2])
    start_xy = np.array(pos0[:2], dtype=np.float64)
    fell_at = None
    sim_time = 0.0
    next_log = 0.0
    max_tilt_gz = -1.0
    print(f"start base height z={start_z:.3f} m")
    print(f"{'t(s)':>6} {'z(m)':>6} {'gz':>6} {'v_fwd':>6} {'v_lat':>6} {'|w|':>6} {'dist':>6} {'|a|max':>6}")

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

                sim_time += control_period_s
                pos, _ = articulation.get_world_pose()
                gz = float(obs[5])
                max_tilt_gz = max(max_tilt_gz, gz)
                if fell_at is None and (pos[2] < FALL_HEIGHT_M or gz > FALL_TILT_GZ):
                    fell_at = sim_time
                    print(f"*** FALL detected at t={sim_time:.2f}s (z={pos[2]:.3f}, gz={gz:.2f})")
                if sim_time >= next_log:
                    next_log += LOG_EVERY_S
                    v_body = quat_rotate_inverse(quat_wxyz, np.asarray(articulation.get_linear_velocity(), dtype=np.float64))
                    dist = float(np.linalg.norm(np.array(pos[:2]) - start_xy))
                    print(f"{sim_time:6.2f} {pos[2]:6.3f} {gz:6.2f} {v_body[0]:6.2f} {v_body[1]:6.2f} "
                          f"{np.linalg.norm(ang_vel_body):6.2f} {dist:6.2f} {np.abs(action).max():6.2f}")
                if fell_at is not None and args.seconds is not None:
                    break
                if args.seconds is not None and sim_time >= args.seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        pos, _ = articulation.get_world_pose()
        dist = float(np.linalg.norm(np.array(pos[:2]) - start_xy))
        verdict = "FELL" if fell_at is not None else "STAYED UPRIGHT"
        print(f"=== {verdict} | sim_time={sim_time:.1f}s distance={dist:.2f}m "
              f"avg_speed={dist / max(sim_time, 1e-6):.2f}m/s (cmd vx={walking_config.cmd[0]}) ===")
        timeline.stop()
        sim.close()


if __name__ == "__main__":
    main()
