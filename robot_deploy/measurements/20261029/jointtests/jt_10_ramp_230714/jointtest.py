"""Hoisted single-joint identification test for the G1. Runs ON the robot (Python 3.8).

*** --run PUBLISHES rt/lowcmd. Only with the user's explicit yes for that specific run, robot hoisted with feet off the ground, ***
*** e-stop / power in the user's hand, robot in Unitree's debug (low-level) mode.  See docs/hoisted_joint_tests.md.             ***

  python3 jointtest.py --check                                   read-only preconditions, publishes nothing
  python3 jointtest.py --joint 21 --profile step --dry-run       preflight + print the plan, publishes nothing
  python3 jointtest.py --joint 21 --profile step --run --outdir DIR

One joint moves; every other joint is HELD at its starting position with half of the onboard nominal gains (gains ramped up over
2 s, so nothing jumps). Waist roll/pitch (13, 14) are absent on this unit and always get kp = kd = 0.
The run ends (or aborts) by ramping kp to 0 over 0.5 s with the target following the measured position, then damping for 2 s, then
publishing stops. Abort = file DIR/STOP, SIGINT/SIGTERM, or any safety trip below.
"""
import argparse, json, math, os, signal, subprocess, sys, time

N = 29
ABSENT = (13, 14)
LEAD, END_RAMP, END_DAMP = 2.0, 0.5, 2.0
NAMES = ("l_hip_pitch l_hip_roll l_hip_yaw l_knee l_ankle_pitch l_ankle_roll r_hip_pitch r_hip_roll r_hip_yaw r_knee "
         "r_ankle_pitch r_ankle_roll waist_yaw waist_roll waist_pitch l_shoulder_pitch l_shoulder_roll l_shoulder_yaw l_elbow "
         "l_wrist_roll l_wrist_pitch l_wrist_yaw r_shoulder_pitch r_shoulder_roll r_shoulder_yaw r_elbow r_wrist_roll "
         "r_wrist_pitch r_wrist_yaw").split()
# onboard nominal gains seen in the recordings (kp, kd)
NOMINAL = ([(100, 4)] * 3 + [(150, 5), (60, 3), (60, 3)]) * 2 + [(200, 6)] * 3 + [(60, 3)] * 4 + [(40, 3)] * 3 \
    + [(60, 3)] * 4 + [(40, 3)] * 3


def defaults(j):
    """(kp, kd, amp rad, torque cap N*m) for the tested joint: deliberately low gains and tiny amplitudes."""
    if j in (19, 20, 21, 26, 27, 28):
        return 20.0, 1.0, 0.05, 3.0           # wrists
    if 15 <= j <= 25:
        return 30.0, 1.5, 0.05, 4.0           # shoulders, elbows
    if j == 12:
        return 40.0, 2.0, 0.03, 6.0           # waist yaw
    if j in (3, 9):
        return 40.0, 2.0, 0.03, 8.0           # knees
    if j in (4, 5, 10, 11):
        return 30.0, 1.5, 0.03, 5.0           # ankles
    return 40.0, 2.0, 0.03, 6.0               # hips


def total_time(profile, amp, hold_s=4.0):
    if profile == "hold":
        return LEAD + hold_s
    if profile == "step":
        return LEAD + 9.0
    if profile == "sine":
        return LEAD + 12.0
    if profile == "ramp":
        return LEAD + 4.0 * amp / 0.02
    raise ValueError(profile)


def offset(profile, amp, t):
    """Target offset (rad) of the tested joint relative to its start position at time t (s)."""
    tt = t - LEAD
    if tt < 0 or profile == "hold":
        return 0.0
    if profile == "step":
        if tt >= 9.0:
            return 0.0
        u = tt % 4.5
        return amp if u < 1.5 else (-amp if u < 3.0 else 0.0)
    if profile == "sine":
        if tt >= 12.0:
            return 0.0
        f0, f1, T = 0.2, 1.5, 12.0
        env = max(0.0, min(1.0, tt / 1.0, (T - tt) / 1.0))      # 1 s fade in/out: starts and ends at 0 with no jump
        return env * amp * math.sin(2 * math.pi * (f0 * tt + 0.5 * (f1 - f0) / T * tt * tt))
    if profile == "ramp":
        v = 0.02
        a = amp / v
        if tt < a:
            return v * tt
        if tt < 3 * a:
            return amp - v * (tt - a)
        if tt < 4 * a:
            return -amp + v * (tt - 3 * a)
        return 0.0
    raise ValueError(profile)


def largest_command_error(profile, amp, max_exc):
    """Largest |q_target - q| the profile can ask for at once: a step goes +amp -> -amp in one tick."""
    return max(max_exc, 2.0 * amp if profile == "step" else amp)


def plan_ok(profile, kp, kd, amp, max_exc, vel_cap, tau_cap):
    """Refuse a plan whose commanded torque (or speed) could exceed the caps, before any runtime trip is needed."""
    if max_exc < amp:
        return "max excursion must be at least the amplitude"
    worst = kp * largest_command_error(profile, amp, max_exc)
    if worst > tau_cap:
        return "kp * largest command error = %.2f N*m exceeds torque cap %.2f" % (worst, tau_cap)
    if profile == "sine" and amp * 2 * math.pi * 1.5 > vel_cap:
        return "sine peak speed %.2f rad/s exceeds speed cap %.2f" % (amp * 2 * math.pi * 1.5, vel_cap)
    return None


def violation(st, q0, args, now):
    """Return a reason string if a safety limit is breached (st: dict of q, dq, tau, temp, t), else None."""
    j = args.joint
    if st is None or now - st["t"] > 0.05:
        return "stale or missing lowstate (> 50 ms)"
    for k in ("q", "dq", "tau"):
        if any(math.isnan(x) or math.isinf(x) for x in st[k]):
            return "NaN/inf in " + k
    if abs(st["q"][j] - q0[j]) > args.max_exc * 1.5:
        return "joint %d excursion %.3f rad > %.3f" % (j, st["q"][j] - q0[j], args.max_exc * 1.5)
    if abs(st["dq"][j]) > args.vel_cap:
        return "joint %d speed %.2f rad/s > %.2f" % (j, st["dq"][j], args.vel_cap)
    if abs(st["tau"][j]) > args.tau_cap * 1.5:
        return "joint %d |tau_est| %.2f > %.2f" % (j, st["tau"][j], args.tau_cap * 1.5)
    for i in range(N):
        if i in ABSENT:
            continue
        if abs(st["q"][i] - q0[i]) > 0.15 and i != j:
            return "held joint %d drifted %.3f rad" % (i, st["q"][i] - q0[i])
        if abs(st["dq"][i]) > 6.0:
            return "joint %d speed %.1f rad/s" % (i, st["dq"][i])
        if st["temp"][i] > 70:
            return "joint %d temperature %d C" % (i, st["temp"][i])
    return None


def iface():
    for l in subprocess.check_output(["ip", "-o", "-4", "addr", "show"]).decode().splitlines():
        p = l.split()
        if p[3].startswith("192.168.123.164/"):
            return p[1]
    raise RuntimeError("no interface carries 192.168.123.164")


class Link:
    """Subscribes to lowstate and lowcmd; keeps the latest state."""

    def __init__(self):
        sys.path.insert(0, "/home/unitree/unitree_sdk2_python")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
        self.mk_cmd, self.LowCmd_, self.CRC, self.Pub = unitree_hg_msg_dds__LowCmd_, LowCmd_, CRC, ChannelPublisher
        self.st, self.n_cmd_seen, self.mm = None, 0, None
        ChannelFactoryInitialize(0, iface())
        self.sub_s = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub_s.Init(self._on_state, 10)
        self.sub_c = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.sub_c.Init(self._on_cmd, 10)
        self.pub = None

    def _on_state(self, m):
        s = m.motor_state
        self.mm = m.mode_machine
        self.st = {"t": time.monotonic(), "q": [s[i].q for i in range(N)], "dq": [s[i].dq for i in range(N)],
                   "tau": [s[i].tau_est for i in range(N)], "temp": [s[i].temperature[0] for i in range(N)],
                   "mode": [int(s[i].mode) for i in range(N)]}

    def _on_cmd(self, m):
        self.n_cmd_seen += 1

    def start_publisher(self):
        self.pub = self.Pub("rt/lowcmd", self.LowCmd_)
        self.pub.Init()
        self.cmd = self.mk_cmd()
        self.crc = self.CRC()

    def send(self, q, kp, kd):
        c = self.cmd
        c.mode_pr, c.mode_machine = 0, self.mm
        for i in range(N):
            m = c.motor_cmd[i]
            m.mode, m.q, m.dq, m.tau = 1, float(q[i]), 0.0, 0.0
            m.kp, m.kd = float(kp[i]), float(kd[i])
        c.crc = self.crc.Crc(c)
        self.pub.Write(c)


def preflight(link, quiet=False):
    """Read-only checks. Returns (problems, warnings, q0)."""
    problems, warns = [], []
    t0 = time.monotonic()
    while link.st is None and time.monotonic() - t0 < 2.0:
        time.sleep(0.01)
    if link.st is None:
        return ["no rt/lowstate within 2 s"], warns, None
    time.sleep(1.5)                      # also listens for foreign lowcmd publishers
    qs = []
    for _ in range(20):
        qs.append(list(link.st["q"]))
        time.sleep(0.02)
    q0 = [sum(c) / len(c) for c in zip(*qs)]
    st = link.st
    if link.n_cmd_seen > 0:
        problems.append("another publisher is active on rt/lowcmd (%d msgs in ~1.5 s): the robot is not in debug/low-level mode" % link.n_cmd_seen)
    still = max(abs(x) for i, x in enumerate(st["dq"]) if i not in ABSENT)
    if still > 0.5:
        problems.append("robot is moving: max |dq| = %.2f rad/s" % still)
    bad = [i for i in range(N) if i not in ABSENT and st["mode"][i] != 1]
    if bad and link.n_cmd_seen == 0 and len(bad) >= 20:
        warns.append("%d motors report mode %s with no publisher: expected in debug mode (disabled until the first command with "
                     "mode=1 enables them). Limbs may be free to swing" % (len(bad), sorted(set(st["mode"][i] for i in bad))))
    elif bad:
        problems.append("motors not in mode 1: %s" % bad)
    hot = max(st["temp"][i] for i in range(N) if i not in ABSENT)
    if hot > 60:
        problems.append("a motor is at %d C (limit 60 to start)" % hot)
    elif hot > 50:
        warns.append("hottest motor %d C" % hot)
    return problems, warns, q0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--joint", type=int)
    ap.add_argument("--profile", choices=["hold", "step", "sine", "ramp"], default="step")
    ap.add_argument("--kp", type=float)
    ap.add_argument("--kd", type=float)
    ap.add_argument("--amp", type=float)
    ap.add_argument("--tau-cap", type=float)
    ap.add_argument("--max-exc", type=float)
    ap.add_argument("--vel-cap", type=float, default=2.0)
    ap.add_argument("--hold-scale", type=float, default=0.5)
    ap.add_argument("--rate", type=float, default=200.0, help="publish rate in Hz, 50..250 (Unitree's own deploy_real publishes at 50 Hz)")
    ap.add_argument("--hold-s", type=float, default=4.0, help="seconds to hold after the 2 s lead-in (profile hold only; max 30)")
    ap.add_argument("--outdir", default="/home/unitree/g1_recordings/jointtests/manual")
    args = ap.parse_args()

    link = Link()
    problems, warns, q0 = preflight(link)
    if args.check:
        print("PROBLEMS:", problems or "none")
        print("WARNINGS:", warns or "none")
        if q0:
            print("q0 (rad):", json.dumps([round(x, 3) for x in q0]))
        return 1 if problems else 0
    if args.joint is None or args.joint in ABSENT or not 0 <= args.joint < N:
        sys.exit("--joint required, 0..28 and not 13/14 (absent on this unit)")
    kp0, kd0, amp0, cap0 = defaults(args.joint)
    args.kp = args.kp if args.kp is not None else kp0
    args.kd = args.kd if args.kd is not None else kd0
    args.amp = args.amp if args.amp is not None else amp0
    args.tau_cap = args.tau_cap if args.tau_cap is not None else cap0
    args.max_exc = args.max_exc if args.max_exc is not None else min(0.10, 2 * args.amp)
    bad = plan_ok(args.profile, args.kp, args.kd, args.amp, args.max_exc, args.vel_cap, args.tau_cap)
    if not 50.0 <= args.rate <= 250.0:
        sys.exit("--rate must be between 50 and 250 Hz")
    args.hold_s = max(1.0, min(30.0, args.hold_s))
    T = total_time(args.profile, args.amp, args.hold_s)
    print("PLAN joint %d (%s) profile=%s amp=%.3f rad kp=%.1f kd=%.2f | tau cap %.1f N*m, max excursion %.3f rad (abort at x1.5), "
          "speed cap %.1f rad/s | others held at %.2f x nominal gains | publish rate %.0f Hz | duration %.1f s + %.1f s damping" % (
              args.joint, NAMES[args.joint], args.profile, args.amp, args.kp, args.kd, args.tau_cap, args.max_exc,
              args.vel_cap, args.hold_scale, args.rate, T, END_RAMP + END_DAMP))
    print("   start q = %.3f rad; target range [%.3f, %.3f]; worst commanded kp*err = %.2f N*m" % (
        q0[args.joint] if q0 else float("nan"), (q0[args.joint] if q0 else 0) - args.amp, (q0[args.joint] if q0 else 0) + args.amp,
        args.kp * largest_command_error(args.profile, args.amp, args.max_exc)))
    print("PREFLIGHT problems:", problems or "none", "| warnings:", warns or "none")
    if bad:
        print("PLAN REFUSED:", bad)
        return 2
    if args.dry_run or not args.run:
        print("DRY RUN: nothing published")
        return 1 if problems and args.run else 0
    if problems:
        print("NOT RUNNING: fix the problems above")
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    stop_file = os.path.join(args.outdir, "STOP")
    sig = {"abort": None}
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, lambda *_: sig.update(abort="signal"))
    q1 = list(link.st["q"]) if link.st else None
    if q1 is None or max(abs(q1[i] - q0[i]) for i in range(N) if i not in ABSENT) > 0.05:
        print("NOT RUNNING: joints moved more than 0.05 rad since the preflight; re-run the dry run so the start pose is current")
        return 1
    q0 = q1
    link.start_publisher()
    hold_kp = [0.0 if i in ABSENT else NOMINAL[i][0] * args.hold_scale for i in range(N)]
    hold_kd = [0.0 if i in ABSENT else NOMINAL[i][1] * args.hold_scale for i in range(N)]
    j = args.joint
    rows, t_stop, reason, dt = [], None, None, 1.0 / args.rate
    t_start = time.monotonic()
    nxt = t_start
    log_hdr = "t,phase,q_target,q,dq,tau_est,kp,kd," + ",".join("q_%d" % i for i in range(N)) + "," + ",".join("tau_%d" % i for i in range(N)) + "," + ",".join("mode_%d" % i for i in range(N))
    try:
        while True:
            now = time.monotonic()
            t = now - t_start
            st = link.st
            if t_stop is None:
                reason = sig["abort"] or (("STOP file" if os.path.exists(stop_file) else None)) or violation(st, q0, args, now)
                if reason:
                    t_stop = t
                elif t >= T:
                    t_stop, reason = t, "finished"
            if t_stop is None:
                ramp = max(0.3, min(1.0, t / LEAD))      # zero position error at the start, so 30% is benign and limits sag while enabling
                kp = [hold_kp[i] * ramp for i in range(N)]
                kd = [hold_kd[i] * ramp for i in range(N)]
                kp[j], kd[j] = args.kp * ramp, args.kd * ramp
                tgt = list(q0)
                tgt[j] = q0[j] + offset(args.profile, args.amp, t)
                phase = "run"
            else:
                u = t - t_stop
                if u > END_RAMP + END_DAMP:
                    break
                s = max(0.0, 1.0 - u / END_RAMP)
                kp = [hold_kp[i] * s for i in range(N)]
                kd = [hold_kd[i] for i in range(N)]
                kp[j], kd[j] = args.kp * s, args.kd
                tgt = list(st["q"]) if st else list(q0)          # follow the measured position: no jump while kp fades
                phase = "end"
            link.send(tgt, kp, kd)
            if st:
                rows.append([t, phase, tgt[j], st["q"][j], st["dq"][j], st["tau"][j], kp[j], kd[j]] + st["q"] + st["tau"] + st["mode"])
            nxt += dt
            if nxt < time.monotonic() - 0.05:          # fell behind: do not burst to catch up
                nxt = time.monotonic()
            time.sleep(max(0.0, nxt - time.monotonic()))
    finally:
        with open(os.path.join(args.outdir, "log.csv"), "w") as f:
            f.write(log_hdr + "\n")
            for r in rows:
                f.write(",".join(["%.5f" % r[0], r[1]] + ["%.6g" % v for v in r[2:]]) + "\n")
        json.dump({"args": vars(args), "q0": q0, "reason": reason, "t_stop": t_stop, "preflight_warnings": warns,
                   "wall": time.time()}, open(os.path.join(args.outdir, "meta.json"), "w"), indent=1)
    print("END reason: %s (t_stop %.2f s); rows %d; publishing has STOPPED" % (reason, t_stop or -1, len(rows)))
    return 0 if reason == "finished" else 3


if __name__ == "__main__":
    sys.exit(main())
