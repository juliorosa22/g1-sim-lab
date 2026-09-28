"""G1 robot loading, head_link validation, and spawn positioning."""
import os

import numpy as np
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage

from .config import RobotConfig


class G1Robot:
    """Loads the G1 USD asset and exposes its head_link path for sensor mounting."""

    def __init__(self, config: RobotConfig):
        self.config = config
        self.head_link_path = f"{config.prim_path}/{config.head_link_name}"

    def load(self, sim):
        """Add the G1 reference to the stage and validate head_link exists.

        Raises RuntimeError immediately if usd_path doesn't exist on this
        machine (set the G1_USD_PATH env var -- see config.py), since that
        failure otherwise looks identical to a wrong head_link_name: both
        leave /World/G1 with an empty child list.

        Otherwise raises RuntimeError with the actual child prim list printed
        if head_link_name doesn't match this USD's link naming -- fix
        RobotConfig.head_link_name using that list rather than guessing.
        """
        if not os.path.isfile(self.config.usd_path):
            raise RuntimeError(
                f"usd_path does not exist on this machine: {self.config.usd_path}\n"
                "Set the G1_USD_PATH environment variable to the real path for this "
                "machine (see config.py) rather than editing the default in place."
            )

        add_reference_to_stage(usd_path=self.config.usd_path, prim_path=self.config.prim_path)
        sim.update()

        stage = omni.usd.get_context().get_stage()
        head_prim = stage.GetPrimAtPath(self.head_link_path)
        if not head_prim.IsValid():
            children = stage.GetPrimAtPath(self.config.prim_path).GetChildren()
            print(f"'{self.head_link_path}' not found -- actual children of {self.config.prim_path}:")
            if not children:
                print("  (none -- the USD loaded but produced no child prims; usd_path may point "
                      "at the wrong/an empty stage even though the file exists)")
            for child in children:
                print("  -", child.GetPath())
            raise RuntimeError("Fix RobotConfig.head_link_name using the list printed above, then rerun.")

        return stage

    def spawn(self, sim, position=None):
        """Move the robot to a spawn point clear of room furniture.

        Uses SingleXFormPrim.set_world_pose rather than adding a raw xform op --
        it correctly handles a prim that may already carry xform ops from its
        source USD, instead of risking a duplicate/conflicting op.
        """
        from isaacsim.core.prims import SingleXFormPrim

        spawn_position = np.array(position if position is not None else self.config.spawn_position)
        SingleXFormPrim(self.config.prim_path).set_world_pose(position=spawn_position)
        sim.update()
        print(f"G1 spawned at: {spawn_position}")
