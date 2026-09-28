"""RealSense-style RGB-D camera: prim creation and ROS2 publishing graph.

Uses a bare UsdGeom.Camera prim (not the high-level isaacsim.sensors.camera.Camera
class) -- we only need ROS2 publishing, not Python-side frame reads, so a plain
prim + IsaacCreateRenderProduct mirrors the lidar's approach and needs no
Replicator-managed render product.
"""
import math

import omni.graph.core as og
import omni.usd
from pxr import Gf, UsdGeom

from ..config import CameraConfig


class CameraSensor:
    def __init__(self, config: CameraConfig, parent_link: str):
        self.config = config
        self.parent_link = parent_link
        self.prim_path = f"{parent_link}/{config.prim_name}"

    def create(self):
        cfg = self.config
        stage = omni.usd.get_context().get_stage()

        UsdGeom.Camera.Define(stage, self.prim_path)
        camera_prim = stage.GetPrimAtPath(self.prim_path)
        camera_schema = UsdGeom.Camera(camera_prim)

        xformable = UsdGeom.Xformable(camera_prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*cfg.mount_translation))
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(*cfg.mount_orientation_wxyz))

        # Solve focal length for the desired horizontal FOV at the fixed aperture.
        focal_length_mm = (cfg.horizontal_aperture_mm / 2.0) / math.tan(math.radians(cfg.horizontal_fov_deg / 2.0))
        vertical_aperture_mm = cfg.horizontal_aperture_mm * (cfg.height / cfg.width)

        camera_schema.CreateFocalLengthAttr().Set(focal_length_mm)
        camera_schema.CreateHorizontalApertureAttr().Set(cfg.horizontal_aperture_mm)
        camera_schema.CreateVerticalApertureAttr().Set(vertical_aperture_mm)
        camera_schema.CreateClippingRangeAttr().Set(Gf.Vec2f(cfg.near_clip_m, cfg.far_clip_m))

        print(
            f"Camera created at: {self.prim_path}  "
            f"(focal_length={focal_length_mm:.2f}mm for {cfg.horizontal_fov_deg} deg HFOV)"
        )
        print("NOTE: this approximates the D435i's RGB FOV/resolution for BOTH the rgb and depth")
        print("      streams from one render product. The real D435i's depth module has a wider")
        print("      FOV (~87x58) and its own resolution (e.g. 848x480) via a separate imager --")
        print("      not modeled here. Good enough for SLAM/perception testing; revisit if you")
        print("      need metrically accurate depth FOV specifically.")

        return self.prim_path

    def build_ros_graph(self):
        """RunOnce -> RenderProduct -> fan out to {RGB, Depth, CameraInfo} helpers.

        Pattern copied from the real Ros2CameraGraph / camera-info builder in
        isaacsim.ros2.bridge.impl.og_shortcuts.og_rtx_sensors -- inputs:type's
        allowed tokens ("rgb", "depth", ...) confirmed via OgnROS2CameraHelper.rst.
        """
        cfg = self.config
        keys = og.Controller.Keys
        og.Controller.edit(
            {"graph_path": cfg.graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                    ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("RgbPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("DepthPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                    ("CameraInfoPublish", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ],
                keys.SET_VALUES: [
                    ("RenderProduct.inputs:cameraPrim", self.prim_path),
                    ("RenderProduct.inputs:width", cfg.width),
                    ("RenderProduct.inputs:height", cfg.height),
                    ("RgbPublish.inputs:type", "rgb"),
                    ("RgbPublish.inputs:topicName", cfg.rgb_topic),
                    ("RgbPublish.inputs:frameId", cfg.color_frame_id),
                    ("RgbPublish.inputs:nodeNamespace", ""),
                    ("RgbPublish.inputs:resetSimulationTimeOnStop", True),
                    ("DepthPublish.inputs:type", "depth"),
                    ("DepthPublish.inputs:topicName", cfg.depth_topic),
                    ("DepthPublish.inputs:frameId", cfg.depth_frame_id),
                    ("DepthPublish.inputs:nodeNamespace", ""),
                    ("DepthPublish.inputs:resetSimulationTimeOnStop", True),
                    ("CameraInfoPublish.inputs:topicName", cfg.camera_info_topic),
                    ("CameraInfoPublish.inputs:frameId", cfg.color_frame_id),
                    ("CameraInfoPublish.inputs:nodeNamespace", ""),
                    ("CameraInfoPublish.inputs:resetSimulationTimeOnStop", True),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
                    ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),
                    ("RenderProduct.outputs:execOut", "RgbPublish.inputs:execIn"),
                    ("RenderProduct.outputs:renderProductPath", "RgbPublish.inputs:renderProductPath"),
                    ("Context.outputs:context", "RgbPublish.inputs:context"),
                    ("RenderProduct.outputs:execOut", "DepthPublish.inputs:execIn"),
                    ("RenderProduct.outputs:renderProductPath", "DepthPublish.inputs:renderProductPath"),
                    ("Context.outputs:context", "DepthPublish.inputs:context"),
                    ("RenderProduct.outputs:execOut", "CameraInfoPublish.inputs:execIn"),
                    ("RenderProduct.outputs:renderProductPath", "CameraInfoPublish.inputs:renderProductPath"),
                    ("Context.outputs:context", "CameraInfoPublish.inputs:context"),
                ],
            },
        )
