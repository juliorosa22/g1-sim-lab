#!/usr/bin/env python3
"""Laptop test of g1_bridge + controller/bridge_client.py against the stand-in SDK. Nothing touches a robot or a network.   python3 controller_cpp/tests/test_bridge.py
Needs g++ (the bridge is compiled with -Werror against stubs/), numpy. Each scenario starts its own bridge on a private shared-memory file."""
import json, os, signal, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CPP = os.path.dirname(HERE)
ROOT = os.path.dirname(CPP)
sys.path.insert(0, os.path.join(ROOT, "controller"))
import bridge_client as bc  # noqa: E402

FAILS = []
BIN = os.path.join(tempfile.gettempdir(), "g1_bridge_test_bin_%d" % os.getpid())
SHM = "/dev/shm/g1_bridge_test_%d" % os.getpid()


def check(cond, msg):
    print("  %s %s" % ("PASS" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


def build(src=None, out=None):
    src = src or os.environ.get("G1_BRIDGE_SRC") or os.path.join(CPP, "bridge", "g1_bridge.cpp")
    r = subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I" + os.path.join(CPP, "stubs"), "-I" + os.path.join(CPP, "bridge"), src, "-o", out or BIN, "-pthread"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[:2000]); sys.exit("COMPILE FAILED")


TABLES = None


class Bridge:
    def __init__(self, publish=True, extra=(), env=None, binary=None, linger=0.4, shm=None):
        self.shm = shm or SHM
        self.wlog = tempfile.mktemp(suffix=".wlog")
        e = dict(os.environ)
        e.update({"CONTROLLER_CONFIRM": "yes-t", "STUB_WRITE_LOG": self.wlog})
        e.update(env or {})
        args = [binary or BIN, "--shm", self.shm, "--linger-s", str(linger)] + (["--publish", "--stage", "t"] if publish else []) + list(extra)
        self.p = subprocess.Popen(args, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.t0 = time.monotonic()
        self.link = bc.BridgeLink(self.shm, wait_s=5)
        while self.link.latest() is None and time.monotonic() - self.t0 < 5:
            time.sleep(0.02)

    def wait_ready(self):
        while time.monotonic() - self.t0 < 1.7:   # the bridge refuses to enable before 1.5 s of uptime (the foreign-publisher check needs listening time)
            time.sleep(0.02)

    def wlines(self):
        try:
            return [l.split() for l in open(self.wlog).read().splitlines() if l.strip()]
        except FileNotFoundError:
            return []

    def stop(self, sig=None, wait=8.0):
        if sig and self.p.poll() is None:
            self.p.send_signal(sig)
        try:
            out, _ = self.p.communicate(timeout=wait)
        except subprocess.TimeoutExpired:
            self.p.kill(); out, _ = self.p.communicate()
        self.link.close()
        return out

    def status_until(self, names, timeout=6.0):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            s = self.link.bridge_status()
            if s and s["state_name"] in names:
                return s
            time.sleep(0.01)
        return self.link.bridge_status()


def contract():
    global TABLES
    if TABLES is None:
        TABLES = json.loads(subprocess.run([BIN, "--print-tables"], capture_output=True, text=True).stdout)
    return TABLES


def drive(link, seconds, scale=0.5, rate=50.0, fn=None):
    """Send commands at 50 Hz like the controller does: hold the measured pose with `scale` of the contract gains."""
    t = contract()
    kp0, kd0 = np.array(t["kp"]), np.array(t["kd"])
    nxt = time.perf_counter()
    for k in range(int(seconds * rate)):
        nxt += 1.0 / rate
        s = link.latest()
        tgt = s["q"].copy(); kp = kp0 * scale; kd = kd0 * scale
        tgt[13] = tgt[14] = 0.0
        if fn:
            r = fn(k, tgt, kp, kd)
            if r is not None:
                tgt, kp, kd = r
        link.send(tgt, kp, kd)
        d = nxt - time.perf_counter()
        if d > 0:
            time.sleep(d)


def main():
    build()
    t = contract()

    print("== 1. protocol layout: the C++ header equals the Python mirror")
    lay = json.loads(subprocess.run([BIN, "--print-layout"], capture_output=True, text=True).stdout)
    ok = (lay["off"]["state"], lay["off"]["cmd"], lay["off"]["status"], lay["total"]) == (bc.OFF_STATE, bc.OFF_CMD, bc.OFF_STATUS, bc.TOTAL)
    for blk, dt in (("state", bc.STATE_DT), ("cmd", bc.CMD_DT), ("status", bc.STATUS_DT)):
        ok &= lay[blk]["size"] == dt.itemsize
        for name, (fdt, off) in dt.fields.items():
            if name != "pad" and name in lay[blk]:
                ok &= lay[blk][name] == off
                if lay[blk][name] != off:
                    print("   mismatch", blk, name, lay[blk][name], off)
    check(ok, "block sizes, payload offsets and block offsets all match (state %d, cmd %d, status %d bytes)" % (bc.STATE_DT.itemsize, bc.CMD_DT.itemsize, bc.STATUS_DT.itemsize))

    print("== 2. the bridge's gain and cap tables equal the controller's contract")
    cfg = json.load(open(os.path.join(ROOT, "configs", "g1_v11_stand_turn.json")))
    kp = np.zeros(29); kd = np.zeros(29)
    for j in cfg["joints"]:
        kp[j["motor"]] = j["kp"]; kd[j["motor"]] = j["kd"]
    sys.path.insert(0, os.path.join(ROOT, "controller"))
    from safety import default_torque_caps
    check(np.allclose(kp, t["kp"]) and np.allclose(kd, t["kd"]), "kp and kd of all 29 motors match configs/g1_v11_stand_turn.json (waist roll/pitch 0)")
    cap = default_torque_caps(); cap[cap > 1e8] = 1e9
    check(np.allclose(cap, t["cap"]), "torque caps match controller/safety.py")

    print("== 3. publishing is refused without the confirmation; a second bridge cannot start")
    env = dict(os.environ); env.pop("CONTROLLER_CONFIRM", None)
    for extra, label in (([], "no --stage"), (["--stage", "t"], "no CONTROLLER_CONFIRM")):
        r = subprocess.run([BIN, "--shm", SHM + "x", "--publish"] + extra, env=env, capture_output=True, text=True, timeout=20)
        check(r.returncode == 2 and "REFUSED" in r.stderr, "refused: %s" % label)
    b = Bridge(publish=False)
    r = subprocess.run([BIN, "--shm", SHM], capture_output=True, text=True, timeout=20)
    check(r.returncode == 2 and "another g1_bridge" in r.stderr, "single instance: a second bridge on the same shared memory is refused")
    b.stop(signal.SIGTERM)

    print("== 4. seqlock: a reader never sees a torn state while the bridge writes at 1 kHz")
    b = Bridge(publish=False, env={"STUB_RAMP": "1"})
    base = 0.1 * np.sin(0.3 * np.arange(29))
    torn = nones = n = 0
    raw_fail = 0
    t_end = time.monotonic() + 3.0
    while time.monotonic() < t_end:
        if b.link._read(bc.OFF_STATE, bc.STATE_DT) is None:
            raw_fail += 1                    # a single attempt that had to give up (safe: it is detected, not used)
        s = b.link.latest()
        if s is None:
            nones += 1
            continue
        n += 1
        off = s["q"] - base
        if np.max(off) - np.min(off) > 1e-6:
            torn += 1
    check(n > 1000 and torn == 0, "%d reads in 3 s, %d torn (inconsistent data)" % (n, torn))
    check(nones == 0, "latest() always returned a sample (single-read give-ups %d of %d, absorbed by the retry)" % (raw_fail, n))
    b.stop(signal.SIGTERM)

    print("== 5. shadow mode (no --publish): the whole machine runs, nothing can be published")
    b = Bridge(publish=False)
    b.wait_ready()
    drive(b.link, 1.0)
    s = b.status_until(["ACTIVE"])
    check(s["state_name"] == "ACTIVE" and s["publishing"] == 0, "ACTIVE with publishing = 0 (%s)" % s["state_name"])
    b.link.end()
    s = b.status_until(["STOPPED"])
    check(s["state_name"] == "STOPPED" and "clean end" in s["reason"], "END from the controller -> STOPPED: %s" % s["reason"])
    out = b.stop()
    check(len(b.wlines()) == 0 and "SHADOW" in out, "no write reached the publisher")

    print("== 6. a normal published run and a clean END")
    b = Bridge()
    b.wait_ready()
    drive(b.link, 2.0)
    b.link.end()
    s = b.status_until(["STOPPED"])
    out = b.stop()
    w = b.wlines()
    check(s["state_name"] == "STOPPED" and "clean end" in s["reason"] and 95 <= len(w) <= 110, "run ended cleanly, %d writes for 2 s at 50 Hz, nothing after END" % len(w))
    check(all(x[3] == "1" for x in w) and all(float(x[4]) == 0.0 and float(x[5]) == 0.0 for x in w), "every write enables the motors and keeps waist roll/pitch at kp = kd = 0")
    check(max(float(x[0]) for x in w) <= 150.001, "no write above the contract knee kp of 150 (max %.1f)" % max(float(x[0]) for x in w))
    ws = out.split("write mean")[1].split("max")[1].split("|")[0].strip() if "write mean" in out else "?"
    print("   (stand-in write time max %s ms)" % ws)

    print("== 6b. the bridge log measures the controller-write -> publish delay")
    import csv as _csv
    logdir = tempfile.mkdtemp()
    b = Bridge(extra=["--out", logdir])
    b.wait_ready()
    drive(b.link, 3.0)
    b.link.end()
    b.status_until(["STOPPED"])
    b.stop()
    rows = list(_csv.DictReader(open(os.path.join(logdir, "bridge_log.csv"))))
    check("write_to_pub_ms" in rows[0] and "state_to_pub_ms" in rows[0], "bridge_log.csv has write_to_pub_ms and state_to_pub_ms")
    act = [float(x["write_to_pub_ms"]) for x in rows if x["state"] == "1" and float(x["write_to_pub_ms"]) >= 0]
    check(len(act) >= 100, "an active run logs the delay on every tick (%d values)" % len(act))
    check(min(act) >= 0.0 and max(act) < 45.0, "delay is between 0 and one command period plus jitter (min %.2f, mean %.2f, max %.2f ms)" % (min(act), sum(act) / len(act), max(act)))
    check(sum(act) / len(act) > 0.5, "and it is not the constant 0 of the old cmd_age_ms (mean %.2f ms)" % (sum(act) / len(act)))

    print("== 7. the controller asks for the stop sequence: kp fades over 0.5 s, then damping, then it ends")
    b = Bridge()
    b.wait_ready()
    drive(b.link, 1.0, scale=1.0)
    n_before = len(b.wlines())
    b.link.request_stop()
    s = b.status_until(["STOPPED"], timeout=8)
    out = b.stop()
    w = b.wlines()[n_before:]
    kpk = np.array([float(x[0]) for x in w]); kdk = np.array([float(x[1]) for x in w])
    check(s["state_name"] == "STOPPED", "ends in STOPPED (%s)" % s["reason"])
    check(168 <= len(w) <= 182, "%d writes in the stop sequence (0.5 s fade + 3 s damping = 175)" % len(w))
    check(abs(kpk[0] - 150) < 20 and kpk[-1] == 0.0 and np.all(np.diff(kpk) <= 1e-6), "kp falls monotonically from ~%.0f to 0" % kpk[0])
    check(np.all(kdk >= t["kd"][3] - 1e-6), "kd never drops below the contract damping (%.1f)" % t["kd"][3])

    print("== 8. DEAD-MAN: the controller is killed (SIGKILL) mid-run")
    b = Bridge()
    b.wait_ready()
    drv = subprocess.Popen([sys.executable, "-c", "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); import test_bridge as T; T.child_drive(%r, %r)" % (HERE, os.path.join(ROOT, "controller"), b.shm, BIN)])
    time.sleep(1.5)
    st0 = b.link.bridge_status()
    check(st0["state_name"] == "ACTIVE", "ACTIVE while the controller drives")
    n0 = len(b.wlines())
    drv.send_signal(signal.SIGKILL); t_kill = time.monotonic(); drv.wait()
    s = b.status_until(["STOPPING", "STOPPED"], timeout=2.0)
    dt_detect = time.monotonic() - t_kill
    check(s["state_name"] in ("STOPPING", "STOPPED") and "dead-man" in s["reason"], "bridge went to %s: %s" % (s["state_name"], s["reason"]))
    check(dt_detect < 0.25, "detected %.0f ms after the kill (limit 60 ms + a tick or two)" % (dt_detect * 1e3))
    s = b.status_until(["STOPPED"], timeout=8)
    out = b.stop()
    w = b.wlines()[n0:]
    kpk = np.array([float(x[0]) for x in w])
    check(s["state_name"] == "STOPPED" and s["n_deadman"] == 1, "full stop sequence completed, dead-man events = %d" % s["n_deadman"])
    check(kpk[-1] == 0.0 and 165 <= len(w) - 3 <= 190, "the bridge itself published the fade and the damping (%d writes after the kill)" % len(w))

    print("== 9. a stalled controller (200 ms without commands) trips the dead-man, and the run cannot be revived")
    b = Bridge()
    b.wait_ready()
    drive(b.link, 1.0)
    time.sleep(0.2)                       # the stall
    s = b.status_until(["STOPPING", "STOPPED"], timeout=1.0)
    check(s["state_name"] in ("STOPPING", "STOPPED") and "dead-man" in s["reason"], "stall detected: %s" % s["reason"])
    s = b.status_until(["STOPPED"], timeout=8)
    npub = s["n_published"]
    drive(b.link, 0.5)                    # the controller wakes up and keeps sending
    s2 = b.link.bridge_status()
    b.stop()
    check(s2["state_name"] == "STOPPED" and s2["n_published"] == npub, "after STOPPED nothing is published again (one run per bridge process)")

    print("== 10. stale state (the state feed stops) -> stop")
    pause = tempfile.mktemp()
    b = Bridge(env={"STUB_PAUSE_FILE": pause})
    b.wait_ready()
    drive(b.link, 0.6)
    open(pause, "w").close()
    t0 = time.monotonic()
    drive(b.link, 0.4, fn=lambda k, tg, kp, kd: (tg, kp, kd))
    s = b.status_until(["STOPPING", "STOPPED"], timeout=2.0)
    os.unlink(pause)
    b.stop(signal.SIGTERM)
    check(s["state_name"] in ("STOPPING", "STOPPED") and "stale" in s["reason"], "state check tripped: %s" % s["reason"])

    print("== 11. the firewall")
    def scenario(label, bad_fn, expect_reason, expect_state, n_bad=1, first=False, key=None):
        b = Bridge()
        b.wait_ready()
        if first:
            drive(b.link, 0.5, fn=lambda k, tg, kp, kd: bad_fn(tg, kp, kd))
        else:
            drive(b.link, 0.6)
            drive(b.link, n_bad / 50.0, fn=lambda k, tg, kp, kd: bad_fn(tg, kp, kd))
            drive(b.link, 0.4)
        s = b.status_until(["REFUSED", "STOPPING", "STOPPED"], timeout=1.5) if expect_state != "ACTIVE" else b.link.bridge_status()
        w = b.wlines()
        out_state = s["state_name"]
        b.stop(signal.SIGTERM)
        check(out_state in expect_state.split("|") and (expect_reason in s["reason"] if expect_reason else True), "%s -> %s (%s)" % (label, out_state, s["reason"][:70]))
        check(all(np.isfinite([float(v) for v in x[:3]]).all() and float(x[0]) <= 150.001 for x in w), "%s: nothing out of range ever reached the publisher (%d writes)" % (label, len(w)))
        return s

    def nan_cmd(tg, kp, kd): tg = tg.copy(); tg[2] = np.nan; return tg, kp, kd
    def hot_kp(tg, kp, kd): kp = kp.copy(); kp[3] = 999.0; return tg, kp, kd
    def big_step(tg, kp, kd): tg = tg.copy(); tg[4] += 0.34; return tg, kp, kd
    def waist(tg, kp, kd): kp = kp.copy(); kp[13] = 50.0; return tg, kp, kd
    def far_start(tg, kp, kd): tg = tg.copy(); tg[1] += 0.2; return tg, kp, kd
    scenario("one NaN command (held, run continues)", nan_cmd, "", "ACTIVE", n_bad=1)
    scenario("three NaN commands in a row", nan_cmd, "firewall", "STOPPING|STOPPED", n_bad=3)
    scenario("kp above the contract (3 in a row)", hot_kp, "firewall", "STOPPING|STOPPED", n_bad=3)
    scenario("waist roll with kp != 0 (3 in a row)", waist, "waist", "STOPPING|STOPPED", n_bad=3)
    scenario("a 0.34 rad target jump (3 in a row)", big_step, "firewall", "STOPPING|STOPPED", n_bad=3)
    scenario("first command 0.2 rad away from the robot", far_start, "from the measured pose", "REFUSED", first=True)

    print("== 12. enabling is refused while another program publishes on rt/lowcmd")
    b = Bridge(env={"STUB_FOREIGN": "1"})
    b.wait_ready()
    drive(b.link, 0.5)
    s = b.status_until(["REFUSED"], timeout=2.0)
    b.stop(signal.SIGTERM)
    check(s["state_name"] == "REFUSED" and "another publisher" in s["reason"] and len(b.wlines()) == 0, "REFUSED, nothing published: %s" % s["reason"][:70])

    print("== 13. SIGTERM to the bridge mid-run goes through the stop sequence")
    b = Bridge()
    b.wait_ready()
    drv = subprocess.Popen([sys.executable, "-c", "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); import test_bridge as T; T.child_drive(%r, %r)" % (HERE, os.path.join(ROOT, "controller"), b.shm, BIN)])
    time.sleep(1.5)
    n0 = len(b.wlines())
    b.p.send_signal(signal.SIGTERM)
    s = b.status_until(["STOPPING", "STOPPED"], timeout=1.0)
    check(s["state_name"] in ("STOPPING", "STOPPED") and "signal" in s["reason"], "signal -> %s" % s["reason"])
    s = b.status_until(["STOPPED"], timeout=8)
    drv.kill(); drv.wait()
    b.stop()
    w = b.wlines()[n0:]
    check(s["state_name"] == "STOPPED" and float(w[-1][0]) == 0.0 and len(w) >= 160, "stop sequence completed (%d writes)" % len(w))

    print("== 14. --on-arrival: a new command is published at once, and every protection still works")
    import csv as _csv, random as _random
    def rows_of(d):
        return list(_csv.DictReader(open(os.path.join(d, "bridge_log.csv"))))
    ld = tempfile.mkdtemp()
    b = Bridge(extra=["--on-arrival", "--out", ld])
    b.wait_ready()
    drive(b.link, 3.0)
    b.link.end(); b.status_until(["STOPPED"]); b.stop()
    rw = [x for x in rows_of(ld) if x["state"] == "1"]
    lat = np.array([float(x["write_to_pub_ms"]) for x in rw])[1:]     # the first tick (the enabling command) is still on the fixed schedule
    check(145 <= len(rw) <= 156 and lat.mean() < 2.5 and lat.max() < 6.0, "steady 50 Hz controller: delay mean %.2f max %.2f ms (the fixed tick gave ~16 on this stand-in), %d writes for 3 s" % (lat.mean(), lat.max(), len(rw)))
    cn = np.diff(np.array([int(x["cmd_counter"]) for x in rw]))
    check(set(cn.tolist()) <= {0, 1}, "no command skipped (counter steps %s)" % sorted(set(cn.tolist())))
    ld2 = tempfile.mkdtemp()
    b = Bridge(extra=["--on-arrival", "--out", ld2])
    b.wait_ready()
    link = b.link; tdat = contract(); kp0, kd0 = np.array(tdat["kp"]), np.array(tdat["kd"]); n_sent = 0
    _random.seed(7); t_end_drive = time.perf_counter() + 3.0
    while time.perf_counter() < t_end_drive:
        st = link.latest(); tg = st["q"].copy(); tg[13] = tg[14] = 0.0
        link.send(tg, kp0 * 0.5, kd0 * 0.5); n_sent += 1
        time.sleep(_random.uniform(0.013, 0.027))               # a controller with +-7 ms of jitter
    s = b.link.bridge_status(); link.end(); b.status_until(["STOPPED"]); b.stop()
    rw = [x for x in rows_of(ld2) if x["state"] == "1"]
    check(s["n_cmd_rejected"] == 0 and s["n_deadman"] == 0, "jittery controller: nothing rejected, no dead-man (%d sent, %d writes)" % (n_sent, len(rw)))
    check(n_sent <= len(rw) <= int(1.5 * n_sent) + 3, "each command is published once (and a late one is repeated at most): %d writes for %d commands" % (len(rw), n_sent))
    b = Bridge(extra=["--on-arrival"])
    b.wait_ready()
    drv = subprocess.Popen([sys.executable, "-c", "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); import test_bridge as T; T.child_drive(%r, %r)" % (HERE, os.path.join(ROOT, "controller"), b.shm, BIN)])
    time.sleep(1.5)
    n0 = len(b.wlines())
    drv.send_signal(signal.SIGKILL); t_kill = time.monotonic(); drv.wait()
    s = b.status_until(["STOPPING", "STOPPED"], timeout=2.0)
    check("dead-man" in s["reason"] and time.monotonic() - t_kill < 0.25, "dead-man still trips (%s) %.0f ms after the kill" % (s["reason"], (time.monotonic() - t_kill) * 1e3))
    s = b.status_until(["STOPPED"], timeout=8); b.stop(); w = b.wlines()[n0:]
    check(s["state_name"] == "STOPPED" and s["n_deadman"] == 1 and 165 <= len(w) - 3 <= 190 and float(w[-1][0]) == 0.0, "and the full fade + damping still follows (%d writes)" % len(w))
    b = Bridge(extra=["--on-arrival"])
    b.wait_ready()
    drive(b.link, 1.0)
    b.link.send(np.full(29, 9.0), np.zeros(29), np.zeros(29))      # a huge step: the firewall must still reject it
    s = b.link.bridge_status(); time.sleep(0.2); s = b.link.bridge_status(); b.stop()
    check(s["n_cmd_rejected"] >= 1, "the firewall still rejects a bad command (%d rejected)" % s["n_cmd_rejected"])

    print("== 15. --fade-s 2: the dead-man fade lasts 2 s, then 3 s of damping")
    b = Bridge(extra=["--on-arrival", "--fade-s", "2.0"])
    b.wait_ready()
    drv = subprocess.Popen([sys.executable, "-c", "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); import test_bridge as T; T.child_drive(%r, %r)" % (HERE, os.path.join(ROOT, "controller"), b.shm, BIN)])
    time.sleep(1.5)
    n0 = len(b.wlines())
    drv.send_signal(signal.SIGKILL); drv.wait()
    s = b.status_until(["STOPPED"], timeout=10)
    b.stop(); w = b.wlines()[n0:]
    kpk = np.array([float(x[0]) for x in w[3:]]); kdk = np.array([float(x[1]) for x in w[3:]])
    check(s["state_name"] == "STOPPED" and s["n_deadman"] == 1, "dead-man, then a complete stop sequence (%s)" % s["reason"])
    check(240 <= len(w) - 3 <= 262, "fade 2 s + damping 3 s = 250 writes (%d)" % (len(w) - 3))
    i1 = int(1.0 / 0.02)
    check(abs(kpk[i1] / kpk[0] - 0.5) < 0.1 and np.all(np.diff(kpk) <= 1e-6) and kpk[-1] == 0.0, "kp is about half after 1 s, never rises, ends at 0 (%.2f)" % (kpk[i1] / kpk[0]))
    check(np.all(kdk >= contract()["kd"][3] - 1e-6), "kd never drops below the contract damping")

    print("== 16. --max-err: the firewall's distance between a target and the measured pose")
    def offset_run(extra, off, secs=1.5):
        b = Bridge(extra=extra)
        b.wait_ready()
        drive(b.link, 0.6)                                                           # enabled and ACTIVE first
        def ramp(k, tg, kp, kd):
            tg = tg.copy(); tg[4] += min(off, 0.05 * (k + 1)); return tg, kp, kd     # reaches the offset in steps of 0.05 rad (the per-tick step limit stays 0.30)
        drive(b.link, secs, fn=ramp)
        s = b.link.bridge_status()
        b.stop(signal.SIGTERM)
        return s
    s = offset_run([], 0.6)
    check(s["state_name"] in ("STOPPING", "STOPPED") and "firewall" in s["reason"], "default 0.35 rad: a target 0.6 rad from the robot is rejected and stops the run (%s)" % s["reason"][:60])
    s = offset_run(["--max-err", "0.8"], 0.6)
    check(s["state_name"] == "ACTIVE" and s["n_cmd_rejected"] == 0, "--max-err 0.8: the same 0.6 rad offset is accepted (%s, %d rejected)" % (s["state_name"], s["n_cmd_rejected"]))
    s = offset_run(["--max-err", "0.8"], 0.9)
    check(s["state_name"] in ("STOPPING", "STOPPED") and "firewall" in s["reason"], "--max-err 0.8: 0.9 rad is still rejected (%s)" % s["reason"][:60])
    b = Bridge(extra=["--max-err", "0.05"], linger=0.1)
    try:
        b.p.wait(timeout=5)
        check(b.p.returncode == 2, "--max-err 0.05 (below the allowed 0.1) is refused at start (exit %s)" % b.p.returncode)
    except subprocess.TimeoutExpired:
        b.stop(signal.SIGTERM); check(False, "--max-err 0.05 is refused at start")
    s = None
    bf = Bridge(extra=["--max-err", "0.8"])
    bf.wait_ready()
    drive(bf.link, 0.5, fn=lambda k, tg, kp, kd: (tg + np.where(np.arange(29) == 1, 0.2, 0.0), kp, kd))   # the FIRST command 0.2 rad away
    s = bf.status_until(["REFUSED", "STOPPING", "STOPPED"], timeout=1.5)
    bf.stop(signal.SIGTERM)
    check(s["state_name"] == "REFUSED", "the first command is still limited to 0.10 rad whatever --max-err is (%s)" % s["reason"][:50])

    print("== 17. --max-speed: the joint-speed state check")
    def speed_run(extra, dq):
        f = tempfile.mktemp()
        b = Bridge(extra=extra, env={"STUB_DQ3_FILE": f})
        b.wait_ready()
        drive(b.link, 0.6)
        open(f, "w").write("%g\n" % dq)
        drive(b.link, 0.5)
        s = b.link.bridge_status()
        b.stop(signal.SIGTERM)
        try: os.unlink(f)
        except OSError: pass
        return s
    s = speed_run([], 7.0)
    check(s["state_name"] in ("STOPPING", "STOPPED") and "speed 7.0 rad/s > 5" in s["reason"], "default: a joint at 7 rad/s stops the run (%s)" % s["reason"][:60])
    s = speed_run(["--max-speed", "10"], 7.0)
    check(s["state_name"] == "ACTIVE", "--max-speed 10: 7 rad/s is accepted (%s)" % s["state_name"])
    s = speed_run(["--max-speed", "10"], 11.0)
    check(s["state_name"] in ("STOPPING", "STOPPED") and "> 10" in s["reason"], "--max-speed 10: 11 rad/s still stops the run (%s)" % s["reason"][:60])
    b = Bridge(extra=["--max-speed", "25"], linger=0.1)
    try:
        b.p.wait(timeout=5)
        check(b.p.returncode == 2, "--max-speed 25 (above the allowed 20) is refused at start (exit %s)" % b.p.returncode)
    except subprocess.TimeoutExpired:
        b.stop(signal.SIGTERM); check(False, "--max-speed 25 is refused at start")

    print()
    print("RESULT: %s" % ("ALL PASS" if not FAILS else "%d FAILED: %s" % (len(FAILS), FAILS)))
    return 0 if not FAILS else 1


def child_drive(path, binary):
    global BIN
    BIN = binary
    link = bc.BridgeLink(path)
    drive(link, 60.0)


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        for f in (BIN, SHM):
            try: os.unlink(f)
            except OSError: pass
    sys.exit(rc)
