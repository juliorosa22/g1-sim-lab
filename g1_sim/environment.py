"""Room/environment loading."""
import omni.usd
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage

from .config import GroundConfig, RoomConfig
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


class FlatGroundEnvironment:
    """A flat ground plane with fully-known friction/restitution, used in place
    of Simple_Room while validating the standing pose. See GroundConfig for why.

    Uses isaacsim.core.api.objects.GroundPlane + isaacsim.core.api.materials.
    PhysicsMaterial -- confirmed from the installed source
    (isaacsim.core.api/isaacsim/core/api/objects/ground_plane.py and
    .../materials/physics_material.py): GroundPlane(..., physics_material=)
    takes a PhysicsMaterial instance, and PhysicsMaterial(prim_path,
    static_friction=, dynamic_friction=, restitution=) creates a UsdShade
    material with UsdPhysics.MaterialAPI applied.
    """

    def __init__(self, config: GroundConfig):
        self.config = config

    def load(self, sim):
        from isaacsim.core.api.materials.physics_material import PhysicsMaterial
        from isaacsim.core.api.objects.ground_plane import GroundPlane

        cfg = self.config
        material = PhysicsMaterial(
            prim_path=cfg.material_prim_path,
            static_friction=cfg.static_friction,
            dynamic_friction=cfg.dynamic_friction,
            restitution=cfg.restitution,
        )
        GroundPlane(
            prim_path=cfg.prim_path,
            z_position=cfg.z_position,
            size=cfg.size,
            physics_material=material,
        )
        sim.update()
        print(
            f"Flat ground plane created at z={cfg.z_position} with "
            f"static_friction={cfg.static_friction}, dynamic_friction={cfg.dynamic_friction}, "
            f"restitution={cfg.restitution}"
        )
