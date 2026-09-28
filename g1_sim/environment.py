"""Room/environment loading."""
import omni.usd
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage

from .config import RoomConfig
from .stage_utils import print_room_bounding_boxes


class RoomEnvironment:
    """Loads one of Isaac Sim's bundled sample environments (default: Simple_Room)."""

    def __init__(self, config: RoomConfig):
        self.config = config

    def load(self, sim):
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("WARNING: could not resolve Isaac Sim assets root (check network/Nucleus access).")
            return

        add_reference_to_stage(
            usd_path=assets_root_path + self.config.usd_relative_path,
            prim_path=self.config.prim_path,
        )
        sim.update()

        if self.config.print_bounding_boxes:
            stage = omni.usd.get_context().get_stage()
            print_room_bounding_boxes(stage, self.config.prim_path)
