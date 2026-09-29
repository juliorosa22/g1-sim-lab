"""Standing-pose controller.

Replaces the USD's flat, undersized joint gains (stiffness=100/damping=1 on
every one of the 53 joints, confirmed via diagnose_joints.py) with the real
per-joint-group values from unitree_sim_isaaclab's own Isaac Lab config, and
commands the validated standing joint targets. See StandingPoseConfig in
config.py for exactly where these numbers come from and why the flat default
isn't enough (legs need more, ankles need less, fingers need much more).
"""
import numpy as np

from .config import StandingPoseConfig


class StandingController:
    def __init__(self, articulation, config: StandingPoseConfig):
        """articulation: an already-initialized isaacsim.core.prims.SingleArticulation
        (physics must be playing -- SingleArticulation.initialize() requires a live
        physics_sim_view, same requirement diagnose_joints.py ran into).
        """
        self.articulation = articulation
        self.config = config

    def apply(self):
        """Teleport the robot into the standing pose, then set real PD gains to hold it.

        Between timeline.play() and this call, physics has already stepped a few
        times (required for SingleArticulation.initialize() to get a live
        physics_sim_view) with the USD's original flat, undersized gains
        (stiffness=100/damping=1, confirmed via diagnose_joints.py) and the
        default straight-leg joint pose -- which is enough for the robot to
        already sag/collapse before this method ever runs. Commanding the
        standing pose as a PD *target* at that point just pulls a PD drive
        from wherever the robot already fell to, arriving too late.

        Isaac Lab's own environments avoid this because their Articulation
        asset writes init_state.joint_pos straight into the simulation as a
        kinematic reset before the episode steps physics at all. We reproduce
        that here with SingleArticulation.set_joint_positions(), which the
        installed source documents as: "This method will immediately set
        (teleport) the affected joints to the indicated value" (as opposed to
        apply_action(), which drives toward a target gradually). Teleporting
        first erases whatever happened during the weak-gain window; setting
        gains and applying the same pose as the PD target afterward makes the
        drive hold exactly where we just placed it, instead of pulling it
        there from a fallen position.
        """
        from isaacsim.core.utils.types import ArticulationAction

        names = self.articulation.dof_names
        cfg = self.config

        missing = [n for n in names if n not in cfg.gains]
        if missing:
            print(f"WARNING: no standing-pose gains defined for {len(missing)} joint(s): {missing}")
            print("         these keep whatever gains the USD already had for them.")

        joint_positions = np.array([cfg.joint_pos_overrides.get(n, 0.0) for n in names])

        # Kinematic teleport -- instant, bypasses the PD drive entirely, erasing
        # whatever the robot already did under the old weak gains.
        self.articulation.set_joint_positions(joint_positions)
        self.articulation.set_joint_velocities(np.zeros_like(joint_positions))

        # Now install the real per-group gains and command the same pose as the
        # PD target, so the drive holds exactly where we just placed it.
        controller = self.articulation.get_articulation_controller()
        current_kps, current_kds = controller.get_gains()
        kps = np.array([cfg.gains[n][0] if n in cfg.gains else current_kps[i] for i, n in enumerate(names)])
        kds = np.array([cfg.gains[n][1] if n in cfg.gains else current_kds[i] for i, n in enumerate(names)])
        controller.set_gains(kps=kps, kds=kds)
        controller.apply_action(ArticulationAction(joint_positions=joint_positions))

        print(f"Standing pose applied: {len(names)} joints teleported to standing targets, "
              f"then given real per-group PD gains to hold there")
