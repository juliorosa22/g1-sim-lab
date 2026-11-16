#!/usr/bin/env python3
"""Hardware GUI: the robot's RealSense view, the onboard/debug mode and the hand-over to Unitree's own controller.  Python 3.8, standard library + cv2/numpy.

    python3 robot_deploy/gui/server.py --mock                               # no robot: mock backend and test-pattern camera (default)
    python3 robot_deploy/gui/server.py --real --interface eth0 [--camera ros|none|URL]   # the robot. Needs the user's explicit go for that run.

then open http://localhost:8770. The server only READS until the page is armed (the red ARM switch); mode changes then go through gui/modes.py, which
refuses unsafe requests (see its docstring). It never publishes motor commands. The camera runs in a separate thread and can be switched off (--camera none).
The "damp" button works without arming when the onboard service runs. In debug mode use the controller's stop sequence or scripts/robot_estop.sh.
"""
import argparse
import hmac
import json
import os
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from modes import Backend, MockBackend, ModeError, ModeManager  # noqa: E402

MANAGER, CAMERA = None, None
TOKEN = secrets.token_urlsafe(32)  # per process; injected into the page, required on every POST (a web page from another origin cannot read it)
ALLOWED_HOSTS, ALLOWED_ORIGINS = set(), set()
LATEST = {"snap": None}


def poll_loop(period=0.2):
    while True:
        try:
            LATEST["snap"] = MANAGER.snapshot()
        except Exception as e:
            LATEST["snap"] = {"error": str(e), "time": time.time()}
        time.sleep(period)


def do_action(path, body):
    """POST /api/<action>: returns a JSON-able dict, raises ModeError with a message for the page."""
    if path == "arm":
        MANAGER.arm(bool(body.get("on")))
        return {"armed": MANAGER.armed}
    if path == "damp":
        return MANAGER.damp_now()
    if not body.get("confirm"):
        raise ModeError("this action needs the confirmation from the page")
    if path == "enter_debug":
        return MANAGER.enter_debug(bool(body.get("supported")))
    if path == "hand_back":
        return MANAGER.hand_back(int(body.get("then_fsm", 1)))
    if path == "select_name":
        return MANAGER.select_name(body.get("name", ""), int(body.get("then_fsm", 1)))
    if path == "fsm":
        return MANAGER.set_fsm(int(body["id"]), bool(body.get("supported")))
    raise ModeError("unknown action %r" % path)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, data):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            if self.headers.get("Host") not in ALLOWED_HOSTS:
                return self._send(403, "text/plain", b"forbidden host")
            with open(os.path.join(HERE, "static", "index.html"), "rb") as f:
                return self._send(200, "text/html; charset=utf-8", f.read().replace(b"__TOKEN__", TOKEN.encode()))
        if self.path == "/state":
            snap = dict(LATEST["snap"] or {})
            snap["log"] = [(t, m) for t, m in MANAGER.log[-40:]]
            snap["observed"] = MANAGER.observed[-40:]
            snap["camera"] = None if CAMERA is None else (time.time() - CAMERA.latest()[1] if CAMERA.latest()[0] else None)
            snap["backend"] = type(MANAGER.b).__name__
            return self._send(200, "application/json", json.dumps(snap, default=str).encode())
        if self.path.startswith("/camera.jpg"):
            jpeg = CAMERA.latest()[0] if CAMERA else None
            return self._send(200, "image/jpeg", jpeg) if jpeg else self._send(404, "text/plain", b"no frame")
        if self.path.startswith("/camera.mjpg"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = 0.0
            try:
                while CAMERA is not None:
                    jpeg, t = CAMERA.latest()
                    if jpeg and t != last:
                        last = t
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg) + jpeg + b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self._send(404, "text/plain", b"not found")

    def do_POST(self):
        if not self.path.startswith("/api/"):
            return self._send(404, "text/plain", b"not found")
        # Anything that can change the robot's mode must come from OUR page: JSON content type (forces a CORS preflight we never answer), our own
        # Host and Origin (blocks DNS rebinding and other sites), and the per-process token in a custom header.
        origin = self.headers.get("Origin")
        if (self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json" or self.headers.get("Host") not in ALLOWED_HOSTS
                or (origin and origin not in ALLOWED_ORIGINS) or not hmac.compare_digest(self.headers.get("X-GUI-Token", ""), TOKEN)):
            return self._send(403, "text/plain", b"forbidden")
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            res = {"ok": True, "result": do_action(self.path[len("/api/"):], body)}
        except ModeError as e:
            res = {"ok": False, "error": str(e)}
        except Exception as e:
            res = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
        MANAGER._note("%s -> %s" % (self.path, "ok" if res["ok"] else res["error"]))
        self._send(200, "application/json", json.dumps(res, default=str).encode())


def main():
    global MANAGER, CAMERA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mock", action="store_true", help="mock robot and test-pattern camera (default)")
    g.add_argument("--real", action="store_true", help="connect to the robot over DDS (reads only until armed)")
    ap.add_argument("--interface", default="eth0", help="network interface of the robot's DDS network")
    ap.add_argument("--camera", default=None, help="mock | ros | none | http://url/stream.mjpg (default: mock with --mock, ros with --real)")
    ap.add_argument("--topic", default="/camera/camera/color/image_raw")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    real = bool(args.real)
    if real:
        from modes import SdkBackend
        backend = SdkBackend(args.interface)
    else:
        backend = MockBackend()
    MANAGER = ModeManager(backend)
    cam = args.camera or ("ros" if real else "mock")
    if cam == "mock":
        from camera import MockCamera
        CAMERA = MockCamera()
    elif cam == "ros":
        from camera import RosCamera
        CAMERA = RosCamera(args.topic)
    elif cam.startswith("http"):
        from camera import UrlCamera
        CAMERA = UrlCamera(cam)
    threading.Thread(target=poll_loop, daemon=True).start()
    names = {"127.0.0.1", "localhost", args.host} - {"0.0.0.0", ""}
    ALLOWED_HOSTS.update("%s:%d" % (n, args.port) for n in names)
    ALLOWED_ORIGINS.update("http://%s:%d" % (n, args.port) for n in names)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("[g1-gui] %s backend, camera %s: http://%s:%d   (the page starts DISARMED)" % ("REAL" if real else "MOCK", cam, args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    if CAMERA:
        CAMERA.stop()


if __name__ == "__main__":
    main()
