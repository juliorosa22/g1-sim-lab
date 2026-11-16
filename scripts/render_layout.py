#!/usr/bin/env python3
"""Render the nav layout (nav/layouts/*.yaml) in Isaac Sim: a top-down and two perspective views as PNG.

    ~/miniconda3/envs/isaaclab/bin/python -u scripts/render_layout.py --layout nav/layouts/three_rooms.yaml --out-dir outputs/nav_render
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--layout", default=os.path.join(REPO_ROOT, "nav/layouts/three_rooms.yaml"))
parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "outputs/nav_render"))
parser.add_argument("--width", type=int, default=1600)
parser.add_argument("--height", type=int, default=900)
parser.add_argument("--robot-usd", default=os.environ.get("G1_USD_PATH", os.path.expanduser("~/Programming/UnitreeRobot/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd")))
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import omni.replicator.core as rep
import omni.usd
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

from nav.map import load_layout, write_usda

os.makedirs(args.out_dir, exist_ok=True)
lay = load_layout(args.layout)
usda = os.path.join(args.out_dir, "layout.usda")
write_usda(lay, usda)

omni.usd.get_context().open_stage(usda)
for _ in range(10):
    app.update()
stage = omni.usd.get_context().get_stage()


def colour(prim, rgb):
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def cube(path, c, size, rgb):
    p = UsdGeom.Cube.Define(stage, path)
    p.GetSizeAttr().Set(1.0)
    UsdGeom.Xformable(p).AddTranslateOp().Set(Gf.Vec3d(*c))
    UsdGeom.Xformable(p).AddScaleOp().Set(Gf.Vec3d(*size))
    colour(p.GetPrim(), rgb)


colour(stage.GetPrimAtPath("/World/Floor"), (0.82, 0.70, 0.48))
for prim in stage.GetPrimAtPath("/World").GetChildren():
    if prim.GetName().startswith("Wall_"):
        colour(prim, (0.30, 0.38, 0.50))

rng = np.random.default_rng(args.seed)
t = lay.sample_target(rng)
cube("/World/Target", (t[0], t[1], 0.1), (0.2, 0.2, 0.2), (0.9, 0.1, 0.1))
sx, sy = lay.start_xy

def add_robot(path, xy, yaw_deg):
    """Reference the G1 model, stand it on the floor (bbox min z = 0) and report its size."""
    outer = UsdGeom.Xform.Define(stage, path)  # our placement; the referenced file may carry its own xform ops
    r = UsdGeom.Xform.Define(stage, path + "/G1")
    r.GetPrim().GetReferences().AddReference(args.robot_usd)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    b = cache.ComputeWorldBound(r.GetPrim()).ComputeAlignedRange()
    lo, hi = b.GetMin(), b.GetMax()
    print("robot bbox size (x, y, z) m:", [round(hi[i] - lo[i], 3) for i in range(3)], "min z", round(lo[2], 3), flush=True)
    outer.AddTranslateOp().Set(Gf.Vec3d(xy[0], xy[1], -lo[2] + 0.01))
    outer.AddRotateZOp().Set(yaw_deg)


add_robot("/World/RobotStart", lay.start_xy, 0.0)
add_robot("/World/RobotDoor", (2.5, 1.55), 90.0)
add_robot("/World/RobotRoomC", (t[0] - 0.9, t[1] + 0.3), -20.0)

dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
dome.CreateIntensityAttr(350.0)
sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
sun.CreateIntensityAttr(1800.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-55, 20, 0))

g = lay.grid
cx, cy = g.origin + np.array(g.shape[::-1]) * g.resolution / 2.0
views = {
    "top": dict(pos=(cx, cy, 24.0), look=(cx, cy, 0.0), focal=18.0),
    "oblique": dict(pos=(cx - 5.0, cy - 12.0, 11.0), look=(cx + 0.5, cy, 0.0), focal=20.0),
    "doorway_front": dict(pos=(2.5, 3.6, 1.3), look=(2.5, 1.55, 0.7), focal=12.0),
    "doorway_corridor": dict(pos=(2.5, 0.1, 1.2), look=(2.5, 1.55, 0.7), focal=10.0),
    "corridor_long": dict(pos=(0.1, 0.75, 1.0), look=(8.0, 0.75, 0.7), focal=10.0),
    "room_c_target": dict(pos=(t[0] + 1.2, t[1] - 1.3, 1.6), look=(t[0] - 0.5, t[1] + 0.1, 0.5), focal=11.0),
    "robot_door_side": dict(pos=(4.3, 1.55, 0.9), look=(2.5, 1.55, 0.65), focal=14.0),
}
for name, v in views.items():
    cam = rep.create.camera(position=v["pos"], look_at=v["look"], focal_length=v["focal"])
    rp = rep.create.render_product(cam, (args.width, args.height))
    ann = rep.AnnotatorRegistry.get_annotator("rgb")
    ann.attach([rp])
    print("camera", name, flush=True)
    for _ in range(30):
        app.update()
    img = np.asarray(ann.get_data())[..., :3]
    if name == "top":  # camera has +x pointing down the image; rotate so +x is right and +y is up
        img = np.rot90(img, 1)
    Image.fromarray(img.astype(np.uint8)).save(os.path.join(args.out_dir, f"{name}.png"))
    print("wrote", name, img.shape, "target", t, flush=True)
print("done", flush=True)
app.close()
