#!/usr/bin/env python3
"""Print world pose (default pose, robot standing on the floor) of the sensor frames in the G1 USD: d435, mid360, imu, head."""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--robot-usd", default=os.environ.get("G1_USD_PATH", os.path.expanduser("~/Programming/UnitreeRobot/unitree_sim_isaaclab/assets/robots/g1-29dof_wholebody_inspire/g1_29dof_with_inspire_rev_1_0.usd")))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np
from pxr import Gf, Usd, UsdGeom

stage = Usd.Stage.Open(args.robot_usd)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
zmin = cache.ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange().GetMin()[2]
pelvis = None
for prim in stage.Traverse():
    n = prim.GetName().lower()
    if prim.IsA(UsdGeom.Xformable) and any(k in n for k in ("d435", "mid360", "lidar", "imu", "head_link", "pelvis")) and prim.GetTypeName() in ("Xform", "Camera", ""):
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = m.ExtractTranslation()
        fwd = m.TransformDir(Gf.Vec3d(1, 0, 0))   # robot forward in this USD is +x
        look = m.TransformDir(Gf.Vec3d(0, 0, -1))  # USD camera looks down its -z
        print(f"{prim.GetPath()}  type={prim.GetTypeName()}  pos(x,y,z above floor)=({t[0]:.3f}, {t[1]:.3f}, {t[2] - zmin:.3f})  "
              f"local+x->world({fwd[0]:.2f},{fwd[1]:.2f},{fwd[2]:.2f})  local-z->world({look[0]:.2f},{look[1]:.2f},{look[2]:.2f})", flush=True)
print("done", flush=True)
app.close()
