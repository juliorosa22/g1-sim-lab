#!/usr/bin/env python3
"""Diagnostic: read the G1's actual joint state and drive gains straight from the
live simulation, rather than assuming from unitree_sim_isaaclab's Isaac Lab
Python configs (G129_CFG_WITH_INSPIRE_WHOLEBODY etc. in robots/unitree.py) --
those configs are only applied by Isaac Lab's own scene/environment setup, which
our script never invokes. We load the robot via a raw add_reference_to_stage,
so whatever is baked directly into the USD's UsdPhysics.DriveAPI attributes is
the real ground truth here, and this is the only reliable way to know it.

Answers the question that decides how to fix the standing-pose problem:
  - If stiffness/damping are already non-zero (matching something like the repo's
    own tuned values), the fix is likely just picking a better default pose.
  - If stiffness/damping are zero (drives expecting external torque control, e.g.
    from an RL policy), the fix is setting gains ourselves before the robot can
    hold any pose at all.

Run with the matching machine's environment active first (isaacenv), then:
    python diagnose_joints.py
"""
from g1_sim.config import SimulationConfig

config = SimulationConfig()

from isaacsim import SimulationApp

sim = SimulationApp({"headless": config.headless})

# isaacsim/omni/pxr imports only work after SimulationApp() above.
import omni.timeline
from isaacsim.core.prims import SingleArticulation

from g1_sim.robot import G1Robot


def main():
    robot = G1Robot(config.robot)
    robot.load(sim)

    # SingleArticulation needs a physics_sim_view, which only exists once physics
    # has actually started -- a few ticks after play() is enough for it to
    # populate, and it's too little time for gravity to meaningfully move the
    # joints, so this doesn't interfere with reading the "as spawned" state.
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        sim.update()

    articulation = SingleArticulation(prim_path=config.robot.prim_path)
    articulation.initialize()

    names = articulation.dof_names
    props = articulation.dof_properties
    state = articulation.get_joints_state()

    print(f"{'joint':30s} {'pos(rad)':>10s} {'stiffness':>12s} {'damping':>10s} {'maxEffort':>10s} {'driveMode':>10s}")
    for i, name in enumerate(names):
        print(
            f"{name:30s} {state.positions[i]:10.4f} "
            f"{props['stiffness'][i]:12.2f} {props['damping'][i]:10.2f} "
            f"{props['maxEffort'][i]:10.2f} {props['driveMode'][i]:10d}"
        )

    stiffness = props["stiffness"]
    damping = props["damping"]
    max_effort = props["maxEffort"]
    print()
    print(f"num DOF: {len(names)}")
    print(f"stiffness: min={stiffness.min():.2f} max={stiffness.max():.2f} mean={stiffness.mean():.2f}")
    print(f"damping:   min={damping.min():.2f} max={damping.max():.2f} mean={damping.mean():.2f}")
    print(f"maxEffort: min={max_effort.min():.2f} max={max_effort.max():.2f} mean={max_effort.mean():.2f}")
    print(f"joints with zero stiffness: {(stiffness == 0).sum()} / {len(names)}")
    print(f"joints with maxEffort < 50 N*m (legs need up to 139 per the real Isaac Lab config): "
          f"{(max_effort < 50).sum()} / {len(names)}")

    timeline.stop()
    sim.close()


if __name__ == "__main__":
    main()
