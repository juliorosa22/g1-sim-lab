"""The console's web API refuses requests that do not come from its own page:  python3 robot_deploy/tests/test_gui_server.py  (mock backend, no robot)"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8779
FAILS = []


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        FAILS.append(name)


def post(path, body, headers):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path), data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None


proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "gui", "server.py"), "--mock", "--camera", "none", "--port", str(PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(50):
        try:
            html = urllib.request.urlopen("http://127.0.0.1:%d/" % PORT, timeout=1).read().decode()
            break
        except Exception:
            time.sleep(0.2)
    token = re.search(r"'X-GUI-Token':'([^']+)'", html).group(1)
    good = {"Content-Type": "application/json", "X-GUI-Token": token}
    check("the page carries a token", len(token) > 20 and "__TOKEN__" not in html)
    check("no token -> 403", post("/api/arm", {"on": True}, {"Content-Type": "application/json"})[0] == 403)
    check("wrong token -> 403", post("/api/arm", {"on": True}, dict(good, **{"X-GUI-Token": "x" * 43}))[0] == 403)
    check("text/plain (a cross-site simple request) -> 403", post("/api/arm", {"on": True}, {"Content-Type": "text/plain", "X-GUI-Token": token})[0] == 403)
    check("foreign Origin -> 403", post("/api/arm", {"on": True}, dict(good, Origin="http://evil.example"))[0] == 403)
    check("foreign Host (DNS rebinding) -> 403", post("/api/arm", {"on": True}, dict(good, Host="evil.example:%d" % PORT))[0] == 403)
    check("nothing was armed by the refused requests", json.loads(urllib.request.urlopen("http://127.0.0.1:%d/state" % PORT).read())["armed"] is False)
    for path in ("/", "/state", "/camera.jpg", "/camera.mjpg"):
        req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path), headers={"Host": "evil.example:%d" % PORT})
        try:
            urllib.request.urlopen(req, timeout=3)
            code_get = 200
        except urllib.error.HTTPError as e:
            code_get = e.code
        check("GET %s with a foreign Host -> 403" % path, code_get == 403)
    code, res = post("/api/arm", {"on": True}, dict(good, Origin="http://127.0.0.1:%d" % PORT))
    check("the page's own request works", code == 200 and res["ok"] and res["result"]["armed"] is True)
finally:
    proc.terminate()
    proc.wait(5)
print("RESULT: %s" % ("ALL PASS" if not FAILS else "FAILED: " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
