"""RTX Lidar sensor: creation, custom Livox-style config, and ROS2 publishing graph."""
import omni.graph.core as og
import omni.kit.commands
import omni.usd
from pxr import Gf, Vt

from ..config import LidarConfig
from ..stage_utils import set_usd_attr


class LidarSensor:
    def __init__(self, config: LidarConfig, parent_link: str):
        self.config = config
        self.parent_link = parent_link
        self.prim_path = None

    def create(self):
        """Create the RTX Lidar prim and set its Livox-approximation attributes.

        Config values are set directly on the omni:sensor:Core:* USD attributes
        rather than via a named config string -- IsaacSensorCreateRtxLidar's
        config lookup (_add_reference in isaacsim.sensors.rtx.impl.commands) only
        matches Nucleus-hosted vendor configs by exact filename stem, so a custom
        local JSON config never actually loads (confirmed by direct attribute
        inspection after creation). Setting attributes directly bypasses that
        lookup entirely and is the only approach confirmed to work.
        """
        cfg = self.config
        _, lidar_prim = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=cfg.prim_path,
            parent=self.parent_link,
            translation=Gf.Vec3d(*cfg.mount_translation),
            orientation=Gf.Quatd(*cfg.mount_orientation_wxyz),
        )
        self.prim_path = str(lidar_prim.GetPath())

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(self.prim_path)

        n = cfg.num_channels
        elevation_deg = [
            round(cfg.elevation_min_deg + i * (cfg.elevation_max_deg - cfg.elevation_min_deg) / (n - 1), 2)
            for i in range(n)
        ]
        azimuth_deg = [0.0] * n
        channel_id = list(range(1, n + 1))
        fire_time_ns = [1525 + i * 3051 for i in range(n)]

        set_usd_attr(prim, "omni:sensor:Core:numberOfEmitters", n)
        set_usd_attr(prim, "omni:sensor:Core:numberOfChannels", n)
        set_usd_attr(prim, "omni:sensor:Core:nearRangeM", cfg.near_range_m)
        set_usd_attr(prim, "omni:sensor:Core:farRangeM", cfg.far_range_m)
        set_usd_attr(prim, "omni:sensor:Core:minReflectance", cfg.min_reflectance)
        set_usd_attr(prim, "omni:sensor:Core:minReflectionRangeM", cfg.min_reflection_range_m)
        set_usd_attr(prim, "omni:sensor:Core:reportRateBaseHz", cfg.report_rate_hz)
        set_usd_attr(prim, "omni:sensor:Core:scanRateBaseHz", cfg.scan_rate_hz)
        set_usd_attr(prim, "omni:sensor:Core:rangeAccuracyM", cfg.range_accuracy_m)
        set_usd_attr(prim, "omni:sensor:Core:azimuthErrorStd", cfg.azimuth_error_std)
        set_usd_attr(prim, "omni:sensor:Core:elevationErrorStd", cfg.elevation_error_std)
        set_usd_attr(prim, "omni:sensor:Core:waveLengthNm", cfg.wave_length_nm)
        set_usd_attr(prim, "omni:sensor:Core:emitterState:s001:azimuthDeg", Vt.FloatArray(azimuth_deg))
        set_usd_attr(prim, "omni:sensor:Core:emitterState:s001:elevationDeg", Vt.FloatArray(elevation_deg))
        set_usd_attr(prim, "omni:sensor:Core:emitterState:s001:fireTimeNs", Vt.UIntArray(fire_time_ns))
        set_usd_attr(prim, "omni:sensor:Core:emitterState:s001:channelId", Vt.UIntArray(channel_id))
        set_usd_attr(prim, "omni:sensor:tickRate", float(cfg.scan_rate_hz))

        return self.prim_path

    def build_ros_graph(self):
        """OnPlaybackTick -> RunOnce -> RenderProduct -> ROS2RtxLidarHelper.

        fullScan=True is required -- it defaults to False (confirmed via the real
        OgnROS2RtxLidarHelper.ogn schema), and without it the topic publishes
        partial per-tick slivers instead of full rotations.
        """
        if self.prim_path is None:
            raise RuntimeError("Call create() before build_ros_graph().")

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
                    ("PointCloudPublish", "isaacsim.ros2.bridge.ROS2RtxLidarHelper"),
                ],
                keys.SET_VALUES: [
                    ("RenderProduct.inputs:cameraPrim", self.prim_path),
                    ("PointCloudPublish.inputs:topicName", cfg.topic),
                    ("PointCloudPublish.inputs:type", "point_cloud"),
                    ("PointCloudPublish.inputs:frameId", cfg.frame_id),
                    ("PointCloudPublish.inputs:nodeNamespace", ""),
                    ("PointCloudPublish.inputs:fullScan", True),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
                    ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),
                    ("RenderProduct.outputs:execOut", "PointCloudPublish.inputs:execIn"),
                    ("RenderProduct.outputs:renderProductPath", "PointCloudPublish.inputs:renderProductPath"),
                    ("Context.outputs:context", "PointCloudPublish.inputs:context"),
                ],
            },
        )
