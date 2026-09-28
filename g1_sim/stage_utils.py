"""Small USD/stage helpers shared across sensor modules."""


def set_usd_attr(prim, name, value):
    """Set a USD attribute by name, warning (not raising) if it doesn't exist.

    Isaac Sim's RTX lidar schema attributes are only guaranteed to exist once the
    sensor plugin has actually registered them on the prim; a missing attribute
    usually means a typo or an API/version mismatch, not something worth crashing
    the whole script over -- so we warn and continue.
    """
    attr = prim.GetAttribute(name)
    if not attr.IsValid():
        print(f"WARNING: attribute {name} not found, skipping")
        return
    attr.Set(value)


def print_room_bounding_boxes(stage, room_prim_path):
    """Print world-space AABBs of everything under the room prim.

    Use this to pick a robot spawn point that doesn't collide with furniture --
    see RobotConfig.spawn_position in config.py.
    """
    from pxr import UsdGeom

    bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_], useExtentsHint=True)
    room_prim = stage.GetPrimAtPath(room_prim_path)
    if not room_prim.IsValid():
        print(f"WARNING: room prim {room_prim_path} not found, skipping bbox printout")
        return

    print(f"=== {room_prim_path} child bounding boxes (world space) ===")
    for child in room_prim.GetChildren():
        rng = bbox_cache.ComputeWorldBound(child).ComputeAlignedRange()
        print(f"  {child.GetPath()}: min={rng.GetMin()} max={rng.GetMax()}")
