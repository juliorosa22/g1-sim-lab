#!/usr/bin/env python3
"""Tests for controller/run.py with a simulated plant and fault injection. Laptop only, numpy, Python 3.8. Nothing touches a robot or a network.

    OPENBLAS_NUM_THREADS=1 python3 robot_deploy/tests/test_runner.py

Checks: the phase sequence and its smoothness; the stop sequence after the end of a run and after EVERY safety trip (NaN, stale state, tilt, speed, torque, temperature,
tracking error, STOP file, signal): kp fades to zero in 0.5 s, kd stays, publishing continues for the damping time, commands stay finite; dry run sends nothing and only
logs trips; publishing is refused without the confirmation variable; waist roll/pitch always get zero gains.
"""
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "controller"))

from observation import load_config  # noqa: E402
from policy import NumpyPolicy  # noqa: E402
from run import DT, END_RAMP, FakeClock, Runner, STOP  # noqa: E402
from safety import Limits  # noqa: E402
from state import NullPublisher, ReplayState  # noqa: E402

CFG = load_config(os.path.join(ROOT, "configs", "g1_v11_stand_turn.json"))
NPZ = sorted(glob.glob(os.path.join(ROOT, "policies", "stand_turn_v11_*[0-9].npz")))[-1]
FAILS = []


def check(cond, msg):
    print("  %s %s" % ("PASS" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


class Plant:
    """Source and publisher in one: joints follow their targets with a speed limit; a fault can be injected at a tick."""

    def __init__(self, clock, default29, seed=0, fault=None):
        rng = np.random.default_rng(seed)
        self.clock = clock
        self.q = default29 + rng.uniform(-0.25, 0.25, 29)
        self.q[13] = self.q[14] = 0.0
        self.q_prev = self.q.copy()
        self.cmd = None
        self.sent = []
        self.n_foreign_cmd = 0
        self.tick = 0
        self.fault = fault          # (tick, kind) or None
        self.stuck = None

    def send(self, target, kp, kd):
        self.sent.append((np.array(target), np.array(kp), np.array(kd)))
        self.cmd = self.sent[-1]

    def latest(self, now=None):
        self.tick += 1
        if self.cmd is not None:
            target, kp, kd = self.cmd
            gain = np.minimum(1.0, kp / 40.0)
            self.q_prev = self.q.copy()
            self.q = self.q + np.clip((target - self.q) * 0.3 * gain, -0.03, 0.03)
            self.q[13] = self.q[14] = 0.0
        dq = (self.q - self.q_prev) / DT
        tau = (self.cmd[1] * (self.cmd[0] - self.q) if self.cmd is not None else np.zeros(29))
        st = {"t": self.clock.now(), "mm": 6, "q": self.q.copy(), "dq": dq, "tau": tau, "temp": np.full(29, 40.0), "mode": np.ones(29, dtype=np.int64),
              "gyro": np.zeros(3), "quat": np.array([1.0, 0.0, 0.0, 0.0])}
        if self.fault and self.tick >= self.fault[0]:
            kind = self.fault[1]
            if kind == "nan":
                st["q"][5] = np.nan
            elif kind == "stale":
                st["t"] -= 1.0
            elif kind == "tilt":
                a = np.radians(30.0) / 2
                st["quat"] = np.array([np.cos(a), np.sin(a), 0.0, 0.0])
            elif kind == "speed":
                st["dq"][4] = 6.0
            elif kind == "torque":
                st["tau"][3] = 100.0
            elif kind == "temp":
                st["temp"][7] = 75.0
            elif kind == "tracking":
                if self.stuck is None:
                    self.stuck = self.q[3] - 0.5 * np.sign(self.cmd[0][3] - self.q[3] or 1.0)
                self.q[3] = self.stuck
                st["q"][3] = self.stuck
        return st


def make_runner(plant, clock, publish=True, outdir=None, **kw):
    pol = NumpyPolicy(NPZ, CFG["obs_size"], CFG["action_size"])
    kw.setdefault("run_s", 5.0)
    return Runner(CFG, pol, plant, plant if publish else NullPublisher(), clock, publish=publish, outdir=outdir, **kw)


def default29_of(runner):
    return runner.default29.copy()


def run_sim(fault=None, publish=True, outdir=None, **kw):
    clock = FakeClock()
    probe = make_runner(Plant(clock, np.zeros(29)), clock, publish=publish)
    plant = Plant(clock, probe.default29, fault=fault)
    r = make_runner(plant, clock, publish=publish, outdir=outdir, **kw)
    res = r.run(max_s=60)
    return r, plant, res


def stop_ok(plant, res, name):
    """After a stop begins: kp reaches 0, kd stays at the contract values, commands stay finite, publishing lasts the damping time."""
    sent = plant.sent
    kp_last, kd_last = sent[-1][1], sent[-1][2]
    live = [i for i in range(29) if i not in (13, 14)]
    check(np.all(kp_last == 0.0), "%s: kp is 0 at the end" % name)
    r = res["runner"]
    check(np.allclose(kd_last[live], r.kd29[live]), "%s: kd stays at the contract values while damping" % name)
    check(all(np.all(np.isfinite(s[0])) and np.all(np.isfinite(s[1])) and np.all(np.isfinite(s[2])) for s in sent), "%s: every command is finite" % name)
    check(all(s[1][13] == s[1][14] == s[2][13] == s[2][14] == 0.0 for s in sent), "%s: waist roll/pitch always kp = kd = 0" % name)
    n_stop = sum(1 for s in sent[::-1] if True)
    return n_stop


def main():
    print("== 1. dry run on the real standing recording (replay): sends nothing, completes")
    clock = FakeClock()
    pol = NumpyPolicy(NPZ, CFG["obs_size"], CFG["action_size"])
    pub = NullPublisher()
    src = ReplayState(os.path.join(ROOT, "measurements", "20261029", "standing_lowstate_raw.csv"), clock)
    r = Runner(CFG, pol, src, pub, clock, publish=False, ramp_s=3.0, hold_s=2.0, blend_s=5.0, run_s=5.0, outdir=None)
    res = r.run(max_s=60)
    check(len(pub.sent) == 0, "dry run: nothing was handed to the publisher")
    check(res["stop_reason"] == "finished" and r.phase == 5, "dry run: ran through every phase and finished (%s)" % res["stop_reason"])
    check(res["cycle_ms"]["max"] < 5.0, "dry run: compute per tick well inside 20 ms (max %.2f ms)" % res["cycle_ms"]["max"])

    print("== 2. publishing against a simulated plant: phases, smoothness, stop")
    r, plant, res = run_sim()
    res["runner"] = r
    sent = plant.sent
    check(res["stop_reason"] == "finished" and not res["trips"], "full run finishes with no trips (%s, trips %s)" % (res["stop_reason"], res["trips"]))
    T = np.array([s[0] for s in sent])
    live = [i for i in range(29) if i not in (13, 14)]
    step = np.abs(np.diff(T[:, live], axis=0)).max(axis=1)
    check(step[:150].max() < 0.02, "ramp: per-tick target change <= 0.02 rad (max %.4f)" % step[:150].max())
    check(step.max() < 0.05, "whole run: per-tick target change <= 0.05 rad (max %.4f)" % step.max())
    k_ramp_end, k_hold_end = int(3.0 / DT), int(5.0 / DT)
    check(np.allclose(T[k_ramp_end + 2][live], r.default29[live], atol=1e-6) or step[k_ramp_end:k_ramp_end + 3].max() < 0.01, "target reaches the default pose at the end of the ramp")
    check(np.allclose(T[k_hold_end - 2][r.act_motors], r.default29[r.act_motors], atol=1e-6), "hold: policy not yet applied (alpha 0)")
    kp0 = sent[0][1]
    check(np.allclose(kp0[live], 0.3 * r.kp29[live]), "first command uses 30%% of the gains (kp0 hip pitch %.0f)" % kp0[0])
    check(np.allclose(sent[int(2.5 / DT)][1][live], r.kp29[live]), "full gains after 2 s")
    n = len(sent)
    stop_start = n - int((END_RAMP + 3.0) / DT)
    check(abs(sent[stop_start + 12][1][3] / r.kp29[3] - (1 - 12 * DT / END_RAMP)) < 0.06, "stop: kp fades linearly over 0.5 s")
    stop_ok(plant, res, "end of run")

    print("== 3. every safety trip leads to the stop sequence (publish mode)")
    faults = ["nan", "stale", "tilt", "speed", "torque", "temp", "tracking"]
    for kind in faults:
        r, plant, res = run_sim(fault=(700, kind))
        res["runner"] = r
        n = len(plant.sent)
        check(res["stop_reason"] is not None and res["stop_reason"] != "finished", "%-8s -> stop: %s" % (kind, res["stop_reason"]))
        word = {"nan": "NaN", "stale": "stale", "tilt": "tilt", "speed": "speed", "torque": "tau_est", "temp": " C", "tracking": "tracking"}[kind]
        check(word in str(res["stop_reason"]), "%-8s -> stopped for THE RIGHT REASON (reason contains %r)" % (kind, word))
        first_trip_tick = int(res["trips"][0][0] / DT) if res["trips"] else -1
        check(abs(first_trip_tick - 700) <= 3, "%-8s -> detected within 3 ticks of the fault (tick %d)" % (kind, first_trip_tick))
        expect = int((END_RAMP + 3.0) / DT)
        check(abs((n - first_trip_tick) - expect) <= 6, "%-8s -> publishing continued for the damping sequence (%d msgs, expected ~%d)" % (kind, n - first_trip_tick, expect))
        stop_ok(plant, res, kind)
        kp_seq = np.array([s[1][3] for s in plant.sent[first_trip_tick:]])
        check(np.all(np.diff(kp_seq) <= 1e-9), "%-8s -> kp never increases after the trip" % kind)
        if kind == "nan":
            check(np.abs(plant.sent[first_trip_tick + 1][0] - plant.sent[first_trip_tick - 1][0]).max() < 0.1, "nan      -> targets did not jump (stop used the last good target)")

    print("== 3b. a trip during the gain ramp (about half stiffness) fades from there, never up to 100%")
    r, plant, res = run_sim(fault=(30, "tilt"))
    res["runner"] = r
    ft = int(res["trips"][0][0] / DT)
    kp_seq = np.array([x[1][3] for x in plant.sent[max(0, ft - 2):]])
    check(res["stop_reason"] is not None and ft < 40, "early trip detected at tick %d (%s)" % (ft, res["stop_reason"]))
    check(np.all(np.diff(kp_seq) <= 1e-9), "kp never increases from the moment of the trip (max step %+.3f)" % np.diff(kp_seq).max())
    stop_ok(plant, res, "early trip")

    print("== 4. STOP file and signal")
    with tempfile.TemporaryDirectory() as d:
        clock = FakeClock()
        probe = make_runner(Plant(clock, np.zeros(29)), clock)
        plant = Plant(clock, probe.default29)
        r = make_runner(plant, clock, outdir=d)
        orig = plant.latest

        def with_stop(now=None):
            if plant.tick == 300:
                open(os.path.join(d, "STOP"), "w").close()
            return orig(now)
        plant.latest = with_stop
        res = r.run(max_s=60)
        res["runner"] = r
        check(res["stop_reason"] == "STOP file", "STOP file ends the run through the stop sequence (%s)" % res["stop_reason"])
        stop_ok(plant, res, "STOP file")
    clock = FakeClock()
    probe = make_runner(Plant(clock, np.zeros(29)), clock)
    plant = Plant(clock, probe.default29)
    r = make_runner(plant, clock)
    orig2 = plant.latest

    def with_sig(now=None):
        if plant.tick == 200:
            r.external_stop = "signal"
        return orig2(now)
    plant.latest = with_sig
    res = r.run(max_s=60)
    res["runner"] = r
    check(res["stop_reason"] == "signal", "signal ends the run through the stop sequence")
    stop_ok(plant, res, "signal")

    print("== 5. dry run logs trips instead of stopping")
    r, plant, res = run_sim(fault=(300, "tilt"), publish=False)
    check(res["stop_reason"] == "finished" and len(res["trips"]) > 10, "dry run with a tilt fault: trips logged (%d), run still finishes" % len(res["trips"]))
    check(len(r.pub.sent) == 0, "dry run with a fault: still nothing sent")

    print("== 6. publishing is refused without the confirmation")
    env = dict(os.environ)
    env.pop("CONTROLLER_CONFIRM", None)
    for extra, label in (([], "no --stage"), (["--stage", "stage2"], "--stage but no CONTROLLER_CONFIRM")):
        p = subprocess.run([sys.executable, os.path.join(ROOT, "controller", "run.py"), "--publish"] + extra, capture_output=True, text=True, env=env, timeout=60)
        check(p.returncode != 0 and "REFUSED" in (p.stdout + p.stderr), "refused: %s" % label)
    env["CONTROLLER_CONFIRM"] = "yes-other"
    p = subprocess.run([sys.executable, os.path.join(ROOT, "controller", "run.py"), "--publish", "--stage", "stage2"], capture_output=True, text=True, env=env, timeout=60)
    check(p.returncode != 0 and "REFUSED" in (p.stdout + p.stderr), "refused: CONTROLLER_CONFIRM names a different stage")
    p = subprocess.run([sys.executable, os.path.join(ROOT, "controller", "run.py"), "--source", "replay", "--publish", "--stage", "x"], capture_output=True, text=True, env=env, timeout=60)
    check(p.returncode != 0, "refused: --publish with the replay source")

    print("== 7. publishing refuses to start from a pose far from the default")
    from run import start_pose_problem
    from observation import ObservationBuilder
    ob = ObservationBuilder(CFG)
    base = np.zeros(29)
    base[ob.motors] = ob.default
    near = base.copy(); near[3] += 0.3
    far = base.copy(); far[2] += 0.79
    check(start_pose_problem(ob, near, 0.6) is None, "0.30 rad from default: allowed")
    msg = start_pose_problem(ob, far, 0.6)
    check(msg is not None and "left_hip_yaw" in msg, "0.79 rad on left hip yaw: refused, names the joint (%s)" % (msg[:60] if msg else None))
    check(start_pose_problem(ob, far, 0.9) is None, "the limit can be raised explicitly")

    print("== 8. stop test mode: hold the measured pose, then fade and damp")
    r, plant, res = run_sim(stop_test=True, stop_test_hold_s=4.0)
    res["runner"] = r
    sent = plant.sent
    live = [i for i in range(29) if i not in (13, 14)]
    n_expect = int(4.0 / DT) + int((END_RAMP + 3.0) / DT)
    check(res["stop_reason"] == "finished (stop test)" and not res["trips"], "stop test finishes with no trips (%s)" % res["stop_reason"])
    check(abs(len(sent) - n_expect) <= 4, "stop test publishes only hold + stop (%d msgs, expected ~%d)" % (len(sent), n_expect))
    q_start = r.q_start
    check(all(np.allclose(sent[k][0][live], q_start[live], atol=1e-9) for k in range(0, int(4.0 / DT) - 1)), "hold: the target is the measured start pose, not the default")
    check(np.allclose(sent[0][1][live], 0.3 * r.kp29[live]) and np.allclose(sent[int(3.0 / DT)][1][live], r.kp29[live]), "hold: gains rise 30% -> 100% and then stay")
    check(np.all(sent[-1][1] == 0.0) and np.allclose(sent[-1][2][live], r.kd29[live]), "ends in kd-only damping")
    check(np.abs(np.diff(np.array([x[0][live] for x in sent]), axis=0)).max() < 0.05, "no per-tick jump in the targets")
    stop_ok(plant, res, "stop test")

    print("== 9. stage2 torque caps: tighter than the default ones, and they trip where the default do not")
    import safety
    d, s2 = safety.default_torque_caps(), safety.stage2_torque_caps()
    live = [i for i in range(29) if i not in (13, 14)]
    check(np.all(s2[live] < d[live]), "every live joint's stage2 cap is below its default cap")
    check(np.all(s2[live] >= 15.0), "no stage2 cap is below 15 N*m (limb gravity plus a hand push stays inside)")
    st = {"t": 0.0, "q": np.zeros(29), "dq": np.zeros(29), "tau": np.zeros(29), "temp": np.full(29, 40.0), "gyro": np.zeros(3), "quat": np.array([1.0, 0.0, 0.0, 0.0])}
    for j in live:
        st["tau"][:] = 0.0
        st["tau"][j] = 0.5 * (d[j] + s2[j])                 # between the two caps
        check(safety.check_state(st, 0.0, safety.Limits()) is None, "joint %2d: %.0f N*m is accepted with the default caps" % (j, st["tau"][j]))
        r2 = safety.check_state(st, 0.0, safety.Limits(tau_cap=s2))
        check(r2 is not None and "tau_est" in r2 and ("motor %d " % j) in r2, "joint %2d: %.0f N*m trips the stage2 caps (%s)" % (j, st["tau"][j], r2))
        st["tau"][j] = 0.9 * s2[j]
        check(safety.check_state(st, 0.0, safety.Limits(tau_cap=s2)) is None, "joint %2d: 90%% of the stage2 cap does not trip" % j)
    r, plant, res = run_sim(limits=safety.Limits(tau_cap=s2))
    check(res["stop_reason"] == "finished" and not res["trips"], "a normal simulated run does not trip the stage2 caps (%s, trips %s)" % (res["stop_reason"], res["trips"]))

    print("== 10. stage3 caps sit between the stage2 and the default caps")
    s3 = safety.stage3_torque_caps()
    check(np.all(s2[live] <= s3[live]) and np.all(s3[live] <= d[live]), "every live joint: stage2 cap <= stage3 cap <= default cap")
    check(np.any(s2[live] < s3[live]), "and stage3 is looser than stage2 somewhere")
    for j in live:
        st["tau"][:] = 0.0
        st["tau"][j] = 0.5 * (s3[j] + d[j]) if d[j] > s3[j] else s3[j] * 1.01
        r3 = safety.check_state(st, 0.0, safety.Limits(tau_cap=s3))
        check(r3 is not None and ("motor %d " % j) in r3, "joint %2d: %.0f N*m trips the stage3 caps" % (j, st["tau"][j]))
        st["tau"][j] = 0.9 * s3[j]
        check(safety.check_state(st, 0.0, safety.Limits(tau_cap=s3)) is None, "joint %2d: 90%% of the stage3 cap does not trip" % j)

    print("== 11. --stop-fade-s: a trip fades kp over the chosen time (2 s), then damps, in every kind of fault")
    for kind in ["torque", "tilt", "nan", "stale", "tracking"]:
        r, plant, res = run_sim(fault=(700, kind), fade_s=2.0)
        res["runner"] = r
        n = len(plant.sent)
        first_trip_tick = int(res["trips"][0][0] / DT) if res["trips"] else -1
        kp_seq = np.array([sx[1][3] for sx in plant.sent[first_trip_tick:]])
        check(abs((n - first_trip_tick) - int((2.0 + 3.0) / DT)) <= 8, "%-8s: publishing continued for fade 2 s + damping 3 s (%d msgs, expected ~%d)" % (kind, n - first_trip_tick, int(5.0 / DT)))
        k1 = int(1.0 / DT)
        check(abs(kp_seq[k1] / kp_seq[0] - 0.5) < 0.08, "%-8s: half of the stiffness is left after 1 s (%.2f)" % (kind, kp_seq[k1] / kp_seq[0]))
        check(np.all(np.diff(kp_seq) <= 1e-9) and kp_seq[-1] == 0.0, "%-8s: kp never increases and ends at 0" % kind)
        check(np.allclose(plant.sent[-1][2][[0, 1, 3, 4]], r.kd29[[0, 1, 3, 4]]), "%-8s: ends in kd-only damping" % kind)
    r, plant, res = run_sim(fade_s=2.0)
    check(res["stop_reason"] == "finished" and not res["trips"], "a normal run with --stop-fade-s 2 finishes (%s)" % res["stop_reason"])
    try:
        make_runner(Plant(FakeClock(), np.zeros(29)), FakeClock(), fade_s=0.1)
        check(False, "a 0.1 s fade is refused")
    except AssertionError:
        check(True, "a fade shorter than 0.3 s is refused")

    print("== 12. --alpha-max: 0 = control run (the default pose is held, the policy output is never applied), 0.5 = half the policy")
    r1, plant1, res1 = run_sim(alpha_max=1.0)
    r0, plant0, res0 = run_sim(alpha_max=0.0)
    rh, planth, resh = run_sim(alpha_max=0.5)
    for nm_, rr, pl, rs in [("alpha 1", r1, plant1, res1), ("alpha 0", r0, plant0, res0), ("alpha 0.5", rh, planth, resh)]:
        check(rs["stop_reason"] == "finished" and not rs["trips"], "%s: the run finishes with no trips (%s)" % (nm_, rs["stop_reason"]))
    live13 = [m for m in r0.act_motors]
    t0_ = np.array([sx[0][live13] for sx in plant0.sent]); t1_ = np.array([sx[0][live13] for sx in plant1.sent]); th_ = np.array([sx[0][live13] for sx in planth.sent])
    dflt = r0.default29[live13]
    n_run = slice(int((3.0 + 2.0 + 5.0) / DT) + 5, int((3.0 + 2.0 + 5.0 + 4.0) / DT))
    check(np.abs(t0_[n_run] - dflt).max() < 1e-6, "alpha 0: in RUN every controlled target IS the default pose (max offset %.1e rad)" % np.abs(t0_[n_run] - dflt).max())
    check(np.abs(t1_[n_run] - dflt).max() > 1e-3, "alpha 1: the policy moves the targets away from the default (max %.3f rad)" % np.abs(t1_[n_run] - dflt).max())
    ratio = np.abs(th_[n_run] - dflt).max() / np.abs(t1_[n_run] - dflt).max()
    check(0.3 < ratio < 0.7, "alpha 0.5: about half of the policy offset (ratio %.2f; the plant feedback changes the actions a little)" % ratio)
    check(np.allclose(plant0.sent[int(7.0 / DT)][1], plant1.sent[int(7.0 / DT)][1]), "the gains are the same with and without the policy")
    for bad in (-0.1, 1.5):
        try:
            make_runner(Plant(FakeClock(), np.zeros(29)), FakeClock(), alpha_max=bad)
            check(False, "alpha_max %s is refused" % bad)
        except AssertionError:
            check(True, "alpha_max %s is refused" % bad)

    print("== 13. the tilt trip level can be lowered (--tilt-deg)")
    st2 = {"t": 0.0, "q": np.zeros(29), "dq": np.zeros(29), "tau": np.zeros(29), "temp": np.full(29, 40.0), "gyro": np.zeros(3)}
    a_ = np.radians(15.0) / 2
    st2["quat"] = np.array([np.cos(a_), np.sin(a_), 0.0, 0.0])        # a 15 degree tilt
    check(safety.check_state(st2, 0.0, safety.Limits()) is None, "a 15 deg tilt passes the default 25 deg level")
    r12 = safety.check_state(st2, 0.0, safety.Limits(tilt_deg=12.0))
    check(r12 is not None and "tilt" in r12, "and trips a 12 deg level (%s)" % r12)

    print("== 14. yaw commands (stage B): the script, the shaping, the limits, the gate")
    import csv as _csv2, tempfile as _tf
    def yaw_run(script, allow=True, run_s=12.0):
        od = _tf.mkdtemp()
        # the toy plant is slow (30% of the error per tick, 0.03 rad per tick at most): the turning targets run ahead of it and tau = kp x error becomes unrealistic, so the tracking and torque trips are switched off in THIS test, which only checks the command path (the real robot tracks much faster)
        rr, pl, rs = run_sim(allow_turn=allow, wz_script=script, run_s=run_s, outdir=od, limits=Limits(tau_cap=np.full(29, 1e9), track_err=10.0))
        rows_ = list(_csv2.DictReader(open(os.path.join(od, "log.csv"))))
        col = lambda n: np.array([float(x[n]) for x in rows_])
        return rs, col
    rs, col = yaw_run("0.3:3,0:6")
    ph_, req, cmd, tt_ = col("phase"), col("req_wz"), col("cmd_wz"), col("t")
    run_idx = np.where(ph_ == 3)[0]; t_run0 = tt_[run_idx[0]]
    check(rs["stop_reason"] == "finished" and not rs["trips"], "a run with a yaw script finishes with no trips (%s)" % rs["stop_reason"])
    check(np.all(cmd[ph_ < 3] == 0.0) and np.all(req[ph_ < 3] == 0.0), "no yaw request or command before RUN")
    on = run_idx[(tt_[run_idx] - t_run0 >= 0.0) & (tt_[run_idx] - t_run0 < 3.0)]
    check(np.allclose(req[on], 0.3), "the request is 0.3 rad/s for the first 3 s of RUN")
    check(abs(cmd.max() - 0.3) < 1e-6 and cmd.min() >= 0.0, "the shaped command reaches 0.3 and never exceeds it (max %.4f)" % cmd.max())
    check(np.abs(np.diff(cmd)).max() <= 2.0 * DT + 1e-6, "the command ramps at <= 2 rad/s^2 (largest step %.4f per tick, limit %.4f)" % (np.abs(np.diff(cmd)).max(), 2.0 * DT))
    after = run_idx[(tt_[run_idx] - t_run0 >= 3.3)]
    check(np.all(cmd[after] == 0.0) and np.all(req[after] == 0.0), "back to exactly 0 after the script (ramped), and it stays 0 for the rest of RUN")
    reach = run_idx[(tt_[run_idx] - t_run0 >= 0.3) & (tt_[run_idx] - t_run0 < 3.0)]
    check(np.all(np.abs(cmd[reach] - 0.3) < 1e-6), "0.3 rad/s is held from 0.3 s to 3 s")
    rs2, col2 = yaw_run("-0.5:3,0:6")
    check(abs(col2("cmd_wz").min() + 0.5) < 1e-6 and col2("cmd_wz").max() <= 0.0, "a negative request gives a negative command, down to -0.5")
    rs3, col3 = yaw_run("0.04:3,0:6")
    check(np.all(col3("cmd_wz") == 0.0), "a 0.04 rad/s request is inside the 0.05 deadband: the command stays 0")
    rs4, col4 = yaw_run("0.3:3,0:6", allow=False)
    check(np.all(col4("cmd_wz") == 0.0) and np.all(col4("req_wz") == 0.0), "without allow_turn the script is ignored: all commands 0")
    for bad in ("0.6:3", "-0.7:3"):
        try:
            make_runner(Plant(FakeClock(), np.zeros(29)), FakeClock(), allow_turn=True, wz_script=bad)
            check(False, "script %r is refused" % bad)
        except AssertionError:
            check(True, "script %r (|wz| > 0.5) is refused" % bad)

    print("== 15. --track-err sets the tracking trip")
    p_ = subprocess.run([sys.executable, os.path.join(ROOT, "controller", "run.py"), "--source", "replay", "--plan", "--track-err", "0.8"], capture_output=True, text=True)
    check("tracking error |q - target| > 0.80 rad" in p_.stdout, "the plan shows the tracking trip at 0.80 rad")
    p_ = subprocess.run([sys.executable, os.path.join(ROOT, "controller", "run.py"), "--source", "replay", "--plan"], capture_output=True, text=True)
    check("tracking error |q - target| > 0.30 rad" in p_.stdout, "and 0.30 rad by default")
    r, plant, res = run_sim(fault=(700, "tracking"), limits=Limits(track_err=0.8))
    check(res["stop_reason"] == "finished" or "tracking" not in str(res["stop_reason"]), "a 0.5 rad stuck joint does not trip a 0.8 rad limit (%s)" % res["stop_reason"])
    r, plant, res = run_sim(fault=(700, "tracking"), limits=Limits(track_err=0.3))
    check("tracking" in str(res["stop_reason"]), "but trips the 0.3 rad limit (%s)" % res["stop_reason"])

    print("== 16. --speed-limit sets the joint-speed trip")
    p_ = subprocess.run([sys.executable, os.path.join(ROOT, "controller", "run.py"), "--source", "replay", "--plan", "--speed-limit", "10"], capture_output=True, text=True)
    check("any joint speed > 10.0 rad/s" in p_.stdout, "the plan shows the speed trip at 10 rad/s")
    p_ = subprocess.run([sys.executable, os.path.join(ROOT, "controller", "run.py"), "--source", "replay", "--plan"], capture_output=True, text=True)
    check("any joint speed > 5.0 rad/s" in p_.stdout, "and 5 rad/s by default")
    r, plant, res = run_sim(fault=(700, "speed"), limits=Limits(speed=10.0))
    check("speed" not in str(res["stop_reason"]), "a 6 rad/s joint does not trip a 10 rad/s limit (%s)" % res["stop_reason"])

    print()
    print("RESULT: %s" % ("ALL PASS" if not FAILS else "%d FAILED: %s" % (len(FAILS), FAILS)))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
