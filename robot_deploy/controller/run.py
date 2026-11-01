#!/usr/bin/env python3
"""Controller for the v11 stand(+turn) policy on the G1.  Python 3.8, numpy only. Runs ON the robot (the replay source also works on a laptop).

    python3 run.py --plan                                  preflight + print exactly what a run would do, publish nothing
    python3 run.py                                         DRY RUN (default): read state, build observations, run the policy, log the commands it WOULD send
    CONTROLLER_CONFIRM=yes-<stage> python3 run.py --publish --stage <stage>      *** publishes rt/lowcmd ***  (only after the user's yes for THIS run)

Safety (CLAUDE.md): publishing needs --publish AND the environment variable CONTROLLER_CONFIRM=yes-<stage> naming the same --stage; the env var is only a forced deliberate
step, the real approval is the user's yes in the session, per run. Robot hoisted, e-stop in hand, unitree-start-joystick.service stopped, robot in debug mode (entered on the RC).

Phases (50 Hz, publish at 50 Hz only):  RAMP (current pose -> default, gains 30% -> 100%)  HOLD (default)  BLEND (policy action x alpha, 0 -> 1)  RUN (policy)  STOP.
STOP (end of run, any safety trip, STOP file, SIGINT/SIGTERM): kp fades to 0 over 0.5 s while the target follows the measured position, then kd-only damping for --damp-s, then
publishing ends. NOTE: after publishing ends the motors fall back to mode 0 (limp); and damping from PC2 has not yet been verified in debug mode (docs/open_issues.md).
The policy runs every tick (also during RAMP/HOLD) so last_action is always the previous raw output; its gait clock starts when BLEND starts.
Dry run: the same phase machine runs and the commands are logged, but nothing is sent; trips are logged and counted instead of stopping the run.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):   # before numpy is imported anywhere: threaded BLAS stalled up to 168 ms on the Jetson
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from commands import CommandShaper  # noqa: E402
from observation import ObservationBuilder, load_config, projected_gravity_from_quat  # noqa: E402
from policy import NumpyPolicy  # noqa: E402
from safety import ABSENT, Limits, check_state, stage2_torque_caps, stage3_torque_caps, tilt_deg  # noqa: E402

DT = 0.02
END_RAMP = 0.5
PHASES = ["RAMP", "HOLD", "BLEND", "RUN", "STOP", "DONE"]
RAMP, HOLD, BLEND, RUN, STOP, DONE = range(6)
MAX_WZ = 0.5          # this task: yaw-only commands up to 0.5 rad/s


def smoothstep(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


class RealClock:
    def now(self):
        return time.perf_counter()

    def sleep_until(self, t):
        while True:
            dt = t - time.perf_counter()
            if dt <= 0:
                return
            time.sleep(dt - 0.0015 if dt > 0.003 else 0)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    def sleep_until(self, t):
        self.t = max(self.t, t)


class Runner:
    def __init__(self, cfg, policy, source, publisher, clock, publish, ramp_s=3.0, hold_s=2.0, blend_s=5.0, run_s=10.0, damp_s=3.0,
                 limits=None, allow_turn=False, wz_script="", outdir=None, stop_test=False, stop_test_hold_s=2.0, fade_s=END_RAMP, alpha_max=1.0):
        self.cfg, self.policy, self.source, self.pub, self.clock, self.publish = cfg, policy, source, publisher, clock, publish
        self.ramp_s, self.hold_s, self.blend_s, self.run_s, self.damp_s = ramp_s, hold_s, blend_s, run_s, damp_s
        self.stop_test, self.stop_test_hold_s = stop_test, stop_test_hold_s
        self.alpha_max = float(alpha_max)            # cap of the policy weight: 1.0 = the policy in full; 0.0 = CONTROL RUN, the default pose is held and the policy output is never applied
        assert 0.0 <= self.alpha_max <= 1.0, "--alpha-max must be between 0 and 1"
        self.fade_s = float(fade_s)                  # length of the kp fade at the start of the stop sequence (0.5 s; longer for the ground stages)
        assert 0.3 <= self.fade_s <= 5.0, "--stop-fade-s must be between 0.3 and 5 s"
        assert ramp_s >= 3.0 and blend_s >= 5.0 or not publish, "publishing needs ramp >= 3 s and blend >= 5 s (task stage 2)"
        self.lim = limits or Limits()
        self.ob = ObservationBuilder(cfg)
        self.shaper = CommandShaper(dt=DT, deadband=cfg["command"]["deadband"], ramp_lin=cfg["command"]["ramp_lin"], ramp_yaw=cfg["command"]["ramp_yaw"])
        self.n_act = self.ob.n_act
        m = self.ob.motors
        self.default29 = np.zeros(29)
        self.kp29, self.kd29 = np.zeros(29), np.zeros(29)
        kp27, kd27 = self.ob.gains()
        self.default29[m] = self.ob.default
        self.kp29[m], self.kd29[m] = kp27, kd27
        for i in ABSENT:
            self.kp29[i] = self.kd29[i] = 0.0
        self.act_motors = m[:self.n_act]
        self.last_action = np.zeros(self.n_act, dtype=np.float32)
        self.q_start = None
        self.t0 = None
        self.phase = RAMP
        self.t_stop = None
        self.stop_reason = None
        self.external_stop = None
        self.trips = []
        self.last_tick = None
        self.periods, self.cycles = [], []
        self.n = 0
        self.last_target = None
        self.scale_now, self.stop_scale = 1.0, 1.0
        self.send_ms, self.wake_late_ms = 0.0, 0.0
        self.allow_turn = allow_turn
        self.wz_segments = []
        if wz_script and allow_turn:
            for part in wz_script.split(","):
                v, d = part.split(":")
                v = float(v)
                assert abs(v) <= MAX_WZ + 1e-9, "this task allows |wz| <= %.1f rad/s" % MAX_WZ
                self.wz_segments.append((v, float(d)))
        self.outdir = outdir
        self.logf = None
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            self.logf = open(os.path.join(outdir, "log.csv"), "w")
            self.logf.write(self._header() + "\n")

    # ------------------------------------------------------------------ helpers
    def _header(self):
        c = ["t", "phase", "alpha", "gain_scale", "req_wz", "cmd_vx", "cmd_vy", "cmd_wz", "gyro_x", "gyro_y", "gyro_z", "grav_x", "grav_y", "grav_z", "tilt_deg"]
        c += ["q%d" % i for i in range(self.ob.n_joints)] + ["dq%d" % i for i in range(self.ob.n_joints)] + ["tau%d" % i for i in range(self.ob.n_joints)]
        c += ["action%d" % i for i in range(self.n_act)] + ["target%d" % i for i in range(self.n_act)] + ["obs%d" % i for i in range(self.ob.obs_size)]
        c += ["cycle_ms", "state_age_ms", "send_ms", "wake_late_ms", "trip"]
        return ",".join(c)

    def request_wz(self, t_run):
        if not self.wz_segments or t_run < 0:
            return 0.0
        for v, d in self.wz_segments:
            if t_run < d:
                return v
            t_run -= d
        return 0.0

    def plan_text(self, q_start=None):
        kp, kd = self.ob.gains()
        j = self.cfg["joints"]
        lines = []
        if self.stop_test:
            lines.append("  *** STOP TEST: hold the MEASURED pose (gains 30%%->100%%) for %.1f s, then the stop sequence (kp fades %.1f s, kd-only damping %.1f s), then publishing ends. No ramp, no policy. ***" % (self.stop_test_hold_s, self.fade_s, self.damp_s))
        lines += ["PLAN %s | %s | publish rate 50 Hz | mode: %s" % (self.cfg["name"], "PUBLISHING" if self.publish else "DRY RUN (nothing is sent)", "publish" if self.publish else "dry-run"),
                 "  phases: RAMP %.1f s (current pose -> default, gain scale 30%%->100%% over 2 s) | HOLD %.1f s | BLEND %.1f s (alpha 0->%g) | RUN %.1f s | STOP: kp fades over %.1f s, then kd-only damping %.1f s, then publishing ends" % (
                     self.ramp_s, self.hold_s, self.blend_s, self.alpha_max, self.run_s, self.fade_s, self.damp_s),
                 "  gains (kp/kd): legs hip p,r,y %g/%g %g/%g %g/%g, knee %g/%g, ankle p,r %g/%g %g/%g | waist yaw %g/%g | arms held at default: shoulders+elbow %g/%g, wrists %g/%g | waist roll/pitch 0/0, hands untouched" % (
                     kp[0], kd[0], kp[1], kd[1], kp[2], kd[2], kp[3], kd[3], kp[4], kd[4], kp[5], kd[5], kp[12], kd[12], kp[13], kd[13], kp[19], kd[19]),
                 "  action scale legs %g, waist yaw %g ; targets = default + scale * action * alpha" % (j[0]["action_scale"], j[12]["action_scale"]),
                 "  safety: " + self.lim.describe(),
                 "  command: zero%s" % (" ; yaw script %s (|wz| <= %.1f)" % (self.wz_segments, MAX_WZ) if self.wz_segments else "")]
        if q_start is not None:
            err = np.abs(self.ob.gather(q_start) - self.ob.default)
            lines.append("  start pose vs default: max |diff| %.3f rad at %s ; the RAMP moves every joint by that much over %.1f s (%.3f rad/s at most)" % (
                err.max(), self.ob.names[int(err.argmax())], self.ramp_s, 1.5 * err.max() / self.ramp_s))
        return "\n".join(lines)

    # ------------------------------------------------------------------ one control tick
    def begin_stop(self, reason, t):
        if self.phase < STOP:
            self.phase, self.t_stop, self.stop_reason = STOP, t, reason
            self.stop_scale = self.scale_now       # fade kp from the stiffness it had, never up to 100% first

    def step(self):
        tick_start = self.clock.now()
        st = self.source.latest()
        now = self.clock.now()
        self.n += 1
        if self.q_start is None:
            if st is None or not np.all(np.isfinite(st["q"])):
                return True                      # wait for the first valid state
            self.q_start, self.t0 = st["q"].copy(), now
        t = now - self.t0
        # ---- safety
        reason = None
        bs_fn = getattr(self.source, "bridge_status", None)
        if bs_fn is not None and self.phase < STOP:       # the C++ bridge is a second safety layer: if it stopped, or stopped talking, so do we
            b = bs_fn()
            if b is None:
                reason = "no bridge status"
            elif now - b["t_heartbeat"] > 0.2 and b["t_heartbeat"] > 0:
                reason = "bridge heartbeat lost (%.0f ms)" % ((now - b["t_heartbeat"]) * 1e3)
            elif b["state"] in (2, 3, 4) and self.publish:
                reason = "bridge %s: %s" % (b["state_name"], b["reason"])
        if reason:
            pass
        elif self.external_stop:
            reason = self.external_stop
        elif self.outdir and os.path.exists(os.path.join(self.outdir, "STOP")):
            reason = "STOP file"
        if self.phase < STOP:
            target_prev = self.last_target
            reason = reason or check_state(st, now, self.lim, target29=target_prev, enforce_tracking=self.publish and self.n > 5)
            if reason is None and self.last_tick is not None and now - self.last_tick > self.lim.period_s:
                reason = "loop period %.0f ms" % ((now - self.last_tick) * 1e3)
            if reason:
                self.trips.append((round(t, 3), reason))
                if self.publish or reason == self.external_stop or reason == "STOP file":
                    self.begin_stop(reason, t)
        if st is None or not np.all(np.isfinite(st["q"])) or not np.all(np.isfinite(st["quat"])):
            self.last_tick = now
            if self.phase == STOP and self.last_target is not None:
                u = t - self.t_stop              # state unusable: fade kp and damp from the last good target, never skip the stop sequence
                if u > self.fade_s + self.damp_s:
                    self.phase = DONE
                elif self.publish:
                    self.pub.send(self.last_target, self.kp29 * self.stop_scale * max(0.0, 1.0 - u / self.fade_s), self.kd29)
            return self.phase != DONE
        # ---- phase by time
        if self.phase < STOP and self.stop_test:
            if t < self.stop_test_hold_s:
                self.phase = HOLD                # hold the MEASURED pose: no ramp, no policy
            else:
                self.begin_stop("finished (stop test)", t)
        elif self.phase < STOP:
            if t < self.ramp_s:
                self.phase = RAMP
            elif t < self.ramp_s + self.hold_s:
                self.phase = HOLD
            elif t < self.ramp_s + self.hold_s + self.blend_s:
                self.phase = BLEND
            elif t < self.ramp_s + self.hold_s + self.blend_s + self.run_s:
                self.phase = RUN
            else:
                self.begin_stop("finished", t)
        t_pol0 = self.ramp_s + self.hold_s
        t_run0 = t_pol0 + self.blend_s
        # ---- observation and policy (every tick)
        grav = projected_gravity_from_quat(st["quat"])
        cmd = self.shaper.step([0.0, 0.0, self.request_wz(t - t_run0) if self.phase == RUN else 0.0])
        q27, dq27 = self.ob.gather(st["q"]), self.ob.gather(st["dq"])
        obs = self.ob.build(st["gyro"], grav, cmd, q27, dq27, self.last_action, max(0.0, t - t_pol0))
        action = self.policy(obs)
        self.last_action = action.astype(np.float32)
        # ---- commands
        alpha, scale = 0.0, 1.0
        if self.phase < STOP:
            if self.stop_test:
                target = self.q_start.copy()     # the pose the robot is in right now
                scale = max(0.3, min(1.0, t / 2.0))
            elif self.phase == RAMP:
                s = smoothstep(t / self.ramp_s)
                target = self.q_start + s * (self.default29 - self.q_start)
                scale = max(0.3, min(1.0, t / 2.0))
            else:
                target = self.default29.copy()
            if self.phase == BLEND:
                alpha = self.alpha_max * smoothstep((t - t_pol0) / self.blend_s)
            elif self.phase == RUN:
                alpha = self.alpha_max
            target[self.act_motors] += alpha * self.ob.act_scale * action
            kp, kd = self.kp29 * scale, self.kd29 * scale
            self.scale_now = scale
        else:
            u = t - self.t_stop
            if u > self.fade_s + self.damp_s:
                self.phase = DONE
            target = st["q"].copy()               # follow the measured position: no jump while kp fades
            kp = self.kp29 * self.stop_scale * max(0.0, 1.0 - u / self.fade_s)
            kd = self.kd29.copy()
            alpha = 0.0
        for i in ABSENT:
            target[i] = 0.0
        self.last_target = target.copy()
        self.send_ms = 0.0
        if self.phase != DONE and self.publish:
            t_send = self.clock.now()
            self.pub.send(target, kp, kd)         # dry run: nothing is sent, the command is only logged below
            self.send_ms = (self.clock.now() - t_send) * 1e3
        # ---- log
        cycle = (self.clock.now() - tick_start) * 1e3
        self.cycles.append(cycle)
        if self.last_tick is not None:
            self.periods.append((now - self.last_tick) * 1e3)
        self.last_tick = now
        if self.logf:
            row = [t, self.phase, alpha, scale, self.request_wz(t - t_run0), *cmd, *st["gyro"], *grav, tilt_deg(st["quat"]), *q27, *dq27, *self.ob.gather(st["tau"]),
                   *action, *self.ob.gather(target)[:self.n_act], *obs, cycle, (now - st["t"]) * 1e3, self.send_ms, self.wake_late_ms]
            self.logf.write(",".join("%.6g" % v for v in row) + "," + (self.trips[-1][1].replace(",", ";") if self.trips and self.trips[-1][0] == round(t, 3) else "") + "\n")
            if self.n % 50 == 0:
                self.logf.flush()
        return self.phase != DONE

    def run(self, max_s=600.0):
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(s, lambda *_: setattr(self, "external_stop", "signal"))
            except ValueError:
                pass                              # not the main thread (tests)
        nxt = self.clock.now()
        while True:
            if not self.step():
                break
            if hasattr(self.source, "exhausted") and self.source.exhausted():
                self.external_stop = "signal"     # replay ran out
            nxt += DT
            if nxt < self.clock.now() - 0.05:
                nxt = self.clock.now()            # fell behind: do not burst to catch up
            self.clock.sleep_until(nxt)
            self.wake_late_ms = (self.clock.now() - nxt) * 1e3     # how late the loop woke up (0 on the fake clock)
            if self.t0 is not None and self.clock.now() - self.t0 > max_s:
                self.external_stop = "max-s"
        if self.logf:
            self.logf.close()
        if self.publish and hasattr(self.pub, "end"):
            self.pub.end()                            # tell the bridge the run is over: it stops publishing at once
        return self.summary()

    def summary(self):
        cy, pe = np.array(self.cycles), np.array(self.periods) if self.periods else np.array([0.0])
        return {"ticks": self.n, "stop_reason": self.stop_reason, "trips": self.trips, "cycle_ms": {"mean": float(cy.mean()), "p99": float(np.percentile(cy, 99)), "max": float(cy.max())},
                "period_ms": {"mean": float(pe.mean()), "p99": float(np.percentile(pe, 99)), "max": float(pe.max())}, "published": len(getattr(self.pub, "sent", [])) if not self.publish else None}


# ---------------------------------------------------------------------- command line
def preflight(link, publish):
    """Read-only checks on the real robot. Returns (problems, warnings, q0)."""
    problems, warns = [], []
    t0 = time.monotonic()
    while link.latest() is None and time.monotonic() - t0 < 2.0:
        time.sleep(0.01)
    if link.latest() is None:
        return ["no rt/lowstate within 2 s"], warns, None
    time.sleep(1.5)                               # also listens for a foreign rt/lowcmd publisher
    st = link.latest()
    q0 = st["q"].copy()
    if publish and link.n_foreign_cmd > 0:
        problems.append("another publisher is active on rt/lowcmd (%d msgs): the robot is not in debug mode" % link.n_foreign_cmd)
    live = [i for i in range(29) if i not in ABSENT]
    if np.abs(st["dq"][live]).max() > 0.5:
        problems.append("robot is moving: max |dq| %.2f rad/s" % np.abs(st["dq"][live]).max())
    if st["temp"][live].max() > 60:
        problems.append("a motor is at %d C" % st["temp"][live].max())
    if int((st["mode"][live] != 1).sum()) >= 20 and link.n_foreign_cmd == 0:
        warns.append("motors report mode 0 (disabled until the first command): limbs may be free to swing")
    return problems, warns, q0


def start_pose_problem(ob, q29, limit):
    """Publishing refuses to start when a joint hangs farther than `limit` rad from its default: the ramp would swing it through that distance at full gains."""
    diff = np.abs(ob.gather(q29) - ob.default)
    i = int(diff.argmax())
    if diff[i] > limit:
        return "joint %s is %.2f rad from its default (limit %.2f): move the robot closer to the default pose by hand, or raise --max-start-diff after reviewing the plan" % (ob.names[i], diff[i], limit)
    return None


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dds", "replay", "bridge"], default="dds", help="dds: Python DDS (old path); bridge: the C++ g1_bridge over shared memory")
    ap.add_argument("--csv", default=os.path.join(root, "measurements", "20261029", "standing_lowstate_raw.csv"), help="replay source only")
    ap.add_argument("--config", default=os.path.join(root, "configs", "g1_v11_stand_turn.json"))
    ap.add_argument("--npz", default=None)
    ap.add_argument("--plan", action="store_true", help="preflight, print the plan, exit; publishes nothing")
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--stage", default="", help="label of the approved run, e.g. stage2-hoisted")
    ap.add_argument("--ramp-s", type=float, default=3.0)
    ap.add_argument("--hold-s", type=float, default=2.0)
    ap.add_argument("--blend-s", type=float, default=5.0)
    ap.add_argument("--run-s", type=float, default=10.0)
    ap.add_argument("--damp-s", type=float, default=3.0)
    ap.add_argument("--caps", choices=["default", "stage2", "stage3"], default="default", help="torque trip levels: stage2 = tight, for the hoisted run in the air; stage3 = for runs with part of the weight on the legs (safety.stage3_torque_caps)")
    ap.add_argument("--track-err", type=float, default=0.30, help="tracking trip: |target - measured q| in rad (default 0.30, set for standing; stepping needs more, see docs/open_issues.md Stage 5a)")
    ap.add_argument("--speed-limit", type=float, default=5.0, help="joint speed trip in rad/s (default 5, set for standing; a normal foot set-down exceeds it, see docs/open_issues.md Stage 5a)")
    ap.add_argument("--tilt-deg", type=float, default=25.0, help="tilt trip level in degrees (default 25; lower for a run that might lean, e.g. the alpha-0 control run)")
    ap.add_argument("--alpha-max", type=float, default=1.0, help="cap of the policy weight (default 1 = the policy in full). 0 = CONTROL RUN: the default pose is held at the normal gains and the policy output is not applied (to see what the policy adds)")
    ap.add_argument("--stop-fade-s", type=float, default=END_RAMP, help="length of the kp fade of the stop sequence before the kd-only damping (default 0.5 s; use 2 s on the ground, with --fade-s 2 on the bridge)")
    ap.add_argument("--max-start-diff", type=float, default=0.6, help="publishing refuses if any joint is farther than this (rad) from its default pose")
    ap.add_argument("--stop-test", action="store_true", help="test the stop sequence only: hold the measured pose, then fade and damp (publishes with --publish)")
    ap.add_argument("--stop-test-hold-s", type=float, default=4.0, help="2 s of gain ramp, then the rest at full gains")
    ap.add_argument("--allow-turn", action="store_true", help="enable yaw-only commands (stage B only)")
    ap.add_argument("--wz-script", default="", help='yaw request segments "value:seconds,..." counted from the start of RUN, |wz| <= 0.5')
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    npz = args.npz or sorted(glob.glob(os.path.join(root, "policies", "stand_turn_v11_*[0-9].npz")))[-1]
    policy = NumpyPolicy(npz, cfg["obs_size"], cfg["action_size"])
    policy.warmup()
    if args.publish:
        if args.source not in ("dds", "bridge"):
            sys.exit("--publish needs --source dds or bridge")
        if not args.stage or os.environ.get("CONTROLLER_CONFIRM") != "yes-" + args.stage:
            sys.exit("REFUSED: publishing needs --stage LABEL and CONTROLLER_CONFIRM=yes-LABEL, set only after the user's yes for this exact run")
    outdir = args.outdir or os.path.join(os.path.expanduser("~"), "g1_recordings", "controller", time.strftime("%Y%m%d_%H%M%S"))
    if args.plan:
        outdir = None
    if args.source in ("dds", "bridge"):
        if args.source == "dds":
            from state import DdsLink
            link = DdsLink()
        else:
            from bridge_client import BridgeLink
            link = BridgeLink()
        clock = RealClock()
        problems, warns, q0 = preflight(link, args.publish)
        source, pub = link, (link if args.publish else None)
    else:
        from state import NullPublisher, ReplayState
        clock = FakeClock()
        source, pub, problems, warns, q0 = ReplayState(args.csv, clock), NullPublisher(), [], [], None
    if pub is None:
        from state import NullPublisher
        pub = NullPublisher()
    runner = Runner(cfg, policy, source, pub, clock, publish=args.publish, ramp_s=args.ramp_s, hold_s=args.hold_s, blend_s=args.blend_s, run_s=args.run_s,
                    damp_s=args.damp_s, allow_turn=args.allow_turn, wz_script=args.wz_script, outdir=outdir, stop_test=args.stop_test, stop_test_hold_s=args.stop_test_hold_s,
                    limits=Limits(tau_cap={"stage2": stage2_torque_caps(), "stage3": stage3_torque_caps()}.get(args.caps), tilt_deg=args.tilt_deg, track_err=args.track_err, speed=args.speed_limit), fade_s=args.stop_fade_s, alpha_max=args.alpha_max)
    print(runner.plan_text(q0))
    if q0 is not None and not args.stop_test:
        sp = start_pose_problem(runner.ob, q0, args.max_start_diff)
        if sp and args.publish:
            problems.append(sp)
        elif sp:
            warns.append("a publishing run would be refused: " + sp)
    print("PREFLIGHT problems: %s | warnings: %s" % (problems or "none", warns or "none"))
    if args.plan or (args.publish and problems):
        print("PLAN ONLY: nothing published" if args.plan else "NOT RUNNING: fix the problems above")
        return 1 if (args.publish and problems) else 0
    if args.publish:
        source.start_publisher()
    res = runner.run()
    print("END", res)
    return 0 if not res["trips"] or str(res["stop_reason"]).startswith("finished") else 3


if __name__ == "__main__":
    sys.exit(main())
