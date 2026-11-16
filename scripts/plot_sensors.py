#!/usr/bin/env python3
"""Draw what the simulated lidar and camera produce at a few poses: python scripts/plot_sensors.py [--out file.png]"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from nav.map import load_layout
from nav.sim2d import CameraDetector, Lidar2D

parser = argparse.ArgumentParser()
parser.add_argument("--layout", default=os.path.join(REPO_ROOT, "nav/layouts/three_rooms.yaml"))
parser.add_argument("--out", default=os.path.join(REPO_ROOT, "outputs/nav_render/sensors.png"))
args = parser.parse_args()
os.makedirs(os.path.dirname(args.out), exist_ok=True)

lay = load_layout(args.layout)
g = lay.grid
lid, cam = Lidar2D(g), CameraDetector(g)
rng = np.random.default_rng(3)
cases = [  # (title, pose x, y, yaw, target x, y)
    ("Start, facing east (target in room A, out of view)", 0.75, 0.75, 0.0, 2.5, 3.4),
    ("Doorway of door A, target 1.2 m ahead: inside the floor patch", 2.5, 1.55, np.pi / 2, 2.5, 2.8),
    ("Same pose, target 2.8 m ahead: too far, above the view", 2.5, 1.55, np.pi / 2, 2.5, 4.4),
    ("Inside room A, turned 60 deg, target 1 m away", 2.2, 2.4, np.deg2rad(60), 2.5, 3.4),
]
fig, axes = plt.subplots(2, 2, figsize=(16, 9))
ext = [g.origin[0], g.origin[0] + g.shape[1] * g.resolution, g.origin[1], g.origin[1] + g.shape[0] * g.resolution]
for ax, (title, x, y, yaw, tx, ty) in zip(axes.ravel(), cases):
    target = np.array([tx, ty])
    ax.imshow(~g.occ, origin="lower", extent=ext, cmap="gray", vmin=-0.3, vmax=1.2)
    pose = (x, y, yaw)
    scan = lid.scan(pose, rng)
    pts = lid.points_world(pose, scan)
    o = np.array([x, y]) + np.array([np.cos(yaw), np.sin(yaw)]) * lid.p.offset[0]
    for p in pts[::3]:
        ax.plot([o[0], p[0]], [o[1], p[1]], color="tab:blue", lw=0.3, alpha=0.5)
    ax.scatter(pts[:, 0], pts[:, 1], s=6, color="tab:blue", zorder=3, label="lidar returns")
    ax.add_patch(Polygon(cam.footprint(pose), closed=True, color="tab:orange", alpha=0.35, label="camera: floor patch where the target is seen"))
    ok = cam.visible(pose, target)[0]
    ax.scatter(*target, marker="s", s=90, color="tab:red", zorder=4, label="target" + (" (SEEN)" if ok else " (not seen)"))
    ax.add_patch(plt.Circle((x, y), 0.19, color="tab:green", zorder=5))
    ax.arrow(x, y, 0.35 * np.cos(yaw), 0.35 * np.sin(yaw), width=0.04, color="darkgreen", zorder=6)
    ax.set_title(f"{title}\n{np.isfinite(scan).sum()} of {len(scan)} beams returned")
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-0.5, 5.2)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)
fig.suptitle("Lidar: horizontal slice at 1.25 m (sees walls only). Camera: forehead, 1.27 m high, 48 deg down, sees a floor patch about 0.45-2.3 m ahead. White = free, black = wall, green = robot")
fig.tight_layout()
fig.savefig(args.out, dpi=110)
print("wrote", args.out)
