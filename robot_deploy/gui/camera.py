"""Camera sources for the hardware GUI: JPEG frames from the robot's RealSense (or a test pattern).  Python 3.8.

  MockCamera   a moving test pattern (no robot)
  RosCamera    subscribes to the RealSense ROS2 topic the robot already publishes (default /camera/camera/color/image_raw, the topic used in
               UnitreeRobot/scripts/g1_camera_viewer.py). Run the server in a shell that has ROS2 Humble and the CycloneDDS settings of
               UnitreeRobot/scripts/init_ros_env.sh (system Python, not conda). It only SUBSCRIBES: the robot's CPU load does not change except for
               what the camera node already costs. Frames are downscaled and re-encoded here, on the laptop.
  UrlCamera    reads a multipart MJPEG stream from a URL (for a streamer running on the robot, if ever wanted)
Every source exposes .latest() -> (jpeg_bytes, timestamp) or (None, 0) and .stop().
"""
import threading
import time
import urllib.request


def _encode(frame_bgr, width, quality):
    import cv2
    h, w = frame_bgr.shape[:2]
    if width and w > width:
        frame_bgr = cv2.resize(frame_bgr, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


class _Base(object):
    def __init__(self):
        self._jpeg, self._t = None, 0.0
        self._lock = threading.Lock()

    def _set(self, jpeg):
        with self._lock:
            self._jpeg, self._t = jpeg, time.time()

    def latest(self):
        with self._lock:
            return self._jpeg, self._t

    def stop(self):
        pass


class MockCamera(_Base):
    def __init__(self, fps=10, width=640):
        _Base.__init__(self)
        self._run = True
        self.fps, self.width = fps, width
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        import cv2
        import numpy as np
        i = 0
        while self._run:
            img = np.zeros((360, 640, 3), np.uint8)
            img[:] = (40 + (i * 3) % 60, 60, 80)
            cv2.circle(img, (80 + (i * 7) % 480, 180), 40, (0, 200, 255), -1)
            cv2.putText(img, "MOCK CAMERA  %s" % time.strftime("%H:%M:%S"), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            self._set(_encode(img, self.width, 70))
            i += 1
            time.sleep(1.0 / self.fps)

    def stop(self):
        self._run = False


class RosCamera(_Base):
    def __init__(self, topic="/camera/camera/color/image_raw", fps=10, width=640, quality=70):
        _Base.__init__(self)
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        self.min_dt, self.width, self.quality, self._last = 1.0 / fps, width, quality, 0.0
        self._rclpy = rclpy
        rclpy.init()
        outer = self

        class _Node(Node):
            def __init__(self):
                Node.__init__(self, "g1_gui_camera")
                self.create_subscription(Image, topic, outer._on_image, qos_profile_sensor_data)

        self._node = _Node()
        self._thread = threading.Thread(target=lambda: rclpy.spin(self._node), daemon=True)
        self._thread.start()

    def _on_image(self, msg):
        now = time.time()
        if now - self._last < self.min_dt:
            return
        self._last = now
        import numpy as np
        import cv2
        ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(msg.encoding)
        if ch is None:
            return  # e.g. 16UC1 depth: not shown here
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // 1)[:, : msg.width * ch].reshape(msg.height, msg.width, ch)
        code = {"rgb8": cv2.COLOR_RGB2BGR, "rgba8": cv2.COLOR_RGBA2BGR, "bgra8": cv2.COLOR_BGRA2BGR, "mono8": cv2.COLOR_GRAY2BGR}.get(msg.encoding)
        frame = arr if code is None else cv2.cvtColor(arr, code)
        jpeg = _encode(frame, self.width, self.quality)
        if jpeg:
            self._set(jpeg)

    def stop(self):
        try:
            self._node.destroy_node()
            self._rclpy.shutdown()
        except Exception:
            pass


class UrlCamera(_Base):
    def __init__(self, url):
        _Base.__init__(self)
        self._run, self.url = True, url
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._run:
            try:
                with urllib.request.urlopen(self.url, timeout=5) as r:
                    buf = b""
                    while self._run:
                        chunk = r.read(4096)
                        if not chunk:
                            break
                        buf += chunk
                        a, b = buf.find(b"\xff\xd8"), buf.find(b"\xff\xd9")
                        if a != -1 and b != -1 and b > a:
                            self._set(buf[a:b + 2])
                            buf = buf[b + 2:]
            except Exception:
                time.sleep(1.0)

    def stop(self):
        self._run = False
