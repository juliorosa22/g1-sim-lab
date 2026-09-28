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
        """Set real PD gains and command the standing joint pose, once.

        A PD position drive in PhysX holds its target continuously once set --
        this is a one-time call right after the articulation initializes, not
        something that needs to run every physics tick.
        """
        from isaacsim.core.utils.types import ArticulationAction

        names = self.articulation.dof_names
        cfg = self.config

        missing = [n for n in names if n not in cfg.gains]
        if missing:
            print(f"WARNING: no standing-pose gains defined for {len(missing)} joint(s): {missing}")
            print("         these keep whatever gains the USD already had for them.")

        controller = self.articulation.get_articulation_controller()
        current_kps, current_kds = controller.get_gains()

        kps = np.array([cfg.gains[n][0] if n in cfg.gains else current_kps[i] for i, n in enumerate(names)])
        kds = np.array([cfg.gains[n][1] if n in cfg.gains else current_kds[i] for i, n in enumerate(names)])
        controller.set_gains(kps=kps, kds=kds)

        joint_positions = np.array([cfg.joint_pos_overrides.get(n, 0.0) for n in names])
        controller.apply_action(ArticulationAction(joint_positions=joint_positions))

        print(f"Standing pose applied: {len(names)} joints given real per-group PD gains + standing targets")
