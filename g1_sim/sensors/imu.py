"""IMU sensor: creation and ROS2 publishing graph."""
import omni.graph.core as og
import omni.kit.commands
from pxr import Gf

from ..config import ImuConfig


class ImuSensor:
    def __init__(self, config: ImuConfig, parent_link: str):
        self.config = config
        self.parent_link = parent_link
        self.prim_path = None

    def create(self):
        cfg = self.config
        _, imu_prim = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path=cfg.prim_path,
            parent=self.parent_link,
            sensor_period=cfg.sensor_period,
            translation=Gf.Vec3d(*cfg.mount_translation),
            orientation=Gf.Quatd(*cfg.mount_orientation_wxyz),
            linear_acceleration_filter_size=cfg.linear_acceleration_filter_size,
            angular_velocity_filter_size=cfg.angular_velocity_filter_size,
            orientation_filter_size=cfg.orientation_filter_size,
        )
        self.prim_path = str(imu_prim.GetPath())
        print(f"IMU created at: {self.prim_path}")
        return self.prim_path

    def build_ros_graph(self):
        """OnPlaybackTick -> IsaacReadIMU -> ROS2PublishImu.

        Attribute names confirmed against OgnIsaacReadIMU.rst / OgnROS2PublishImu.rst
        -- no real Python graph-builder example for this exact pairing exists in
        the installed source (unlike lidar/camera/odometry), so this was wired
        directly from the verified .rst input/output names rather than copied
        from an existing example.
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
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("ReadImu", "isaacsim.sensors.physics.IsaacReadIMU"),
                    ("PublishImu", "isaacsim.ros2.bridge.ROS2PublishImu"),
                ],
                keys.SET_VALUES: [
                    ("ReadSimTime.inputs:resetOnStop", False),
                    ("ReadImu.inputs:imuPrim", self.prim_path),
                    ("ReadImu.inputs:readGravity", True),
                    ("ReadImu.inputs:useLatestData", False),
                    ("PublishImu.inputs:topicName", cfg.topic),
                    ("PublishImu.inputs:frameId", cfg.frame_id),
                    ("PublishImu.inputs:nodeNamespace", ""),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "ReadImu.inputs:execIn"),
                    ("ReadImu.outputs:execOut", "PublishImu.inputs:execIn"),
                    ("ReadImu.outputs:angVel", "PublishImu.inputs:angularVelocity"),
                    ("ReadImu.outputs:linAcc", "PublishImu.inputs:linearAcceleration"),
                    ("ReadImu.outputs:orientation", "PublishImu.inputs:orientation"),
                    ("Context.outputs:context", "PublishImu.inputs:context"),
                    ("ReadSimTime.outputs:simulationTime", "PublishImu.inputs:timeStamp"),
                ],
            },
        )
