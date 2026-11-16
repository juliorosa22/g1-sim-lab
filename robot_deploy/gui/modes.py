"""Robot mode switching for the hardware GUI.  Python 3.8, standard library only (unitree_sdk2py is imported lazily by SdkBackend).

Two worlds on the G1 (from unitree_sdk2py and the hardware session's logs):
  * ONBOARD mode   Unitree's motion service ("sport") runs on PC1 and owns rt/lowcmd. MotionSwitcherClient.CheckMode() returns {"name": <mode>, ...} with a
                   non-empty name. LocoClient (SetFsmId) only works here: FSM ids from the SDK are 0 zero torque, 1 damp, 3 sit, 500 start, 702 lie->stand,
                   706 squat<->stand; the robot was also seen in FSM 801 (balance) after the RC's F1+Up.
  * DEBUG mode     MotionSwitcherClient.ReleaseMode() until CheckMode() returns an empty name: the onboard service is gone and OUR rt/lowcmd is obeyed.
                   The hardware session entered it from the RC (L2+R2) so far, never from software. When publishing stops every motor falls back to mode 0 (limp).

This module never publishes motor commands. It can: read (CheckMode, GetFsmId, lowstate summary, rt/lowcmd rate) and, only through ModeManager with every
precondition met, call ReleaseMode / SelectMode / SetFsmId. Nothing here has been run on the robot: the sequences are UNVERIFIED hypotheses (see
docs/mode_switching_gui.md); MockBackend is what the tests and `server.py --mock` use.
"""
import threading
import time

FSM_ZERO_TORQUE, FSM_DAMP, FSM_SIT, FSM_START, FSM_LIE2STAND, FSM_SQUAT_STAND = 0, 1, 3, 500, 702, 706
FSM_NAMES = {0: "zero torque", 1: "damp", 3: "sit", 500: "start", 702: "lie to stand", 706: "squat <-> stand", 801: "balance (seen after F1+Up)"}
ALLOWED_FSM = (FSM_ZERO_TORQUE, FSM_DAMP, FSM_SIT, FSM_START, FSM_LIE2STAND, FSM_SQUAT_STAND)  # what the GUI may SET (801 is observed only)
SOFT_FSM = (FSM_ZERO_TORQUE, FSM_DAMP)  # the robot is not holding itself up: the only FSM states from which ReleaseMode is allowed


class ModeError(Exception):
    pass


class Backend(object):
    """What ModeManager needs from the robot. Every method may raise; ModeManager turns that into a ModeError."""

    def check_mode(self):  # -> name of the onboard mode, "" when released (debug mode), None when unknown
        raise NotImplementedError

    def release_mode(self):
        raise NotImplementedError

    def select_mode(self, name):
        raise NotImplementedError

    def get_fsm_id(self):  # -> int or None
        raise NotImplementedError

    def set_fsm_id(self, fsm_id):
        raise NotImplementedError

    def lowstate(self):  # -> dict(mode_machine, tilt_deg, age_s, max_abs_tau) or None
        raise NotImplementedError

    def lowcmd_rate(self):  # -> messages/s seen on rt/lowcmd over the last second (somebody is publishing), or None
        raise NotImplementedError


class MockBackend(Backend):
    """A tiny robot: enough behaviour to test the manager (and to click through the GUI without a robot)."""

    def __init__(self, name="ai", fsm=FSM_DAMP, foreign_cmd_hz=0.0):
        self.mode, self.fsm, self.foreign = name, fsm, foreign_cmd_hz
        self.calls = []
        self.fail_next = None

    def _maybe_fail(self, what):
        if self.fail_next == what:
            self.fail_next = None
            raise RuntimeError("mock failure in " + what)

    def check_mode(self):
        self._maybe_fail("check_mode")
        return self.mode

    def release_mode(self):
        self._maybe_fail("release_mode")
        self.calls.append(("release",))
        self.remembered, self.mode = self.mode, ""
        return 0

    def select_mode(self, name):
        self._maybe_fail("select_mode")
        self.calls.append(("select", name))
        self.mode, self.fsm = name, FSM_DAMP  # the onboard service starts in damp
        return 0

    def get_fsm_id(self):
        return self.fsm if self.mode else None

    def set_fsm_id(self, fsm_id):
        self.calls.append(("fsm", fsm_id))
        if not self.mode:
            raise RuntimeError("no onboard motion service")
        self.fsm = fsm_id
        return 0

    def lowstate(self):
        return {"mode_machine": 6, "tilt_deg": 1.2, "age_s": 0.002, "max_abs_tau": 3.0}

    def lowcmd_rate(self):
        return self.foreign


class SdkBackend(Backend):
    """The real thing, through unitree_sdk2py. Constructed only by `server.py --real`."""

    def __init__(self, interface, domain_id=0, timeout=5.0):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        ChannelFactoryInitialize(domain_id, interface)
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(timeout)
        self.msc.Init()
        self.loco = LocoClient()
        self.loco.SetTimeout(timeout)
        self.loco.Init()
        self._state, self._state_t, self._cmd_times = None, 0.0, []
        self._lock = threading.Lock()
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self._on_state, 10)
        sub_cmd = ChannelSubscriber("rt/lowcmd", LowCmd_)  # read only: is anybody publishing?
        sub_cmd.Init(self._on_cmd, 10)
        self._subs = (sub, sub_cmd)

    def _on_state(self, msg):
        import math
        q = msg.imu_state.quaternion  # w, x, y, z
        gz = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])  # world z axis in the body frame: cos(tilt)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, gz))))
        tau = max(abs(m.tau_est) for m in msg.motor_state[:29])
        with self._lock:
            self._state, self._state_t = {"mode_machine": int(msg.mode_machine), "tilt_deg": tilt, "max_abs_tau": tau}, time.time()

    def _on_cmd(self, msg):
        now = time.time()
        with self._lock:
            self._cmd_times.append(now)
            self._cmd_times = [t for t in self._cmd_times if now - t < 1.0]

    def check_mode(self):
        code, data = self.msc.CheckMode()
        if code != 0 or data is None:
            return None
        return data.get("name") or ""

    def release_mode(self):
        return self.msc.ReleaseMode()[0]

    def select_mode(self, name):
        return self.msc.SelectMode(name)[0]

    def get_fsm_id(self):
        code, fsm = self.loco.GetFsmId()
        return fsm if code == 0 else None

    def set_fsm_id(self, fsm_id):
        return self.loco.SetFsmId(int(fsm_id))

    def lowstate(self):
        with self._lock:
            if self._state is None:
                return None
            s = dict(self._state)
            s["age_s"] = time.time() - self._state_t
            return s

    def lowcmd_rate(self):
        now = time.time()
        with self._lock:
            return float(len([t for t in self._cmd_times if now - t < 1.0]))


class ModeManager(object):
    """Gate in front of the backend. Every actuating call needs `armed` and its own preconditions; everything is logged."""

    def __init__(self, backend, wait_s=6.0, poll_s=0.2, sleep=time.sleep, clock=time.time):
        self.b, self.wait_s, self.poll_s, self._sleep, self._clock = backend, wait_s, poll_s, sleep, clock
        self.armed = False
        self.remembered_name = None  # the onboard mode name seen before ReleaseMode: what hand_back selects again
        self.log = []
        self.observed = []  # read-only discovery table: every change of (mode name, fsm id, mode_machine)
        self._last_obs = None
        self._lock = threading.RLock()

    # ---- bookkeeping ----
    def _note(self, msg):
        with self._lock:
            self.log.append((self._clock(), msg))
            del self.log[:-200]

    def arm(self, on):
        self.armed = bool(on)
        self._note("GUI %s" % ("ARMED (mode changes allowed)" if self.armed else "disarmed"))

    def _require_armed(self):
        if not self.armed:
            raise ModeError("the GUI is not armed")

    # ---- read only ----
    def snapshot(self):
        """Everything the page shows; safe to call at any rate. Also feeds the discovery table."""
        snap = {"armed": self.armed, "remembered_name": self.remembered_name, "time": self._clock()}
        try:
            name = self.b.check_mode()
        except Exception as e:
            name, snap["mode_error"] = None, str(e)
        fsm = None
        if name:
            try:
                fsm = self.b.get_fsm_id()
            except Exception as e:
                snap["fsm_error"] = str(e)
        ls = self.b.lowstate()
        snap.update({"mode_name": name, "world": "unknown" if name is None else ("debug" if name == "" else "onboard"), "fsm_id": fsm,
                     "fsm_text": FSM_NAMES.get(fsm) if fsm is not None else None, "lowstate": ls, "lowcmd_hz": self.b.lowcmd_rate()})
        key = (name, fsm, ls["mode_machine"] if ls else None)
        if key != self._last_obs:
            self._last_obs = key
            self.observed.append({"t": snap["time"], "mode_name": name, "fsm_id": fsm, "mode_machine": key[2]})
            del self.observed[:-500]
            self._note("observed: mode=%r fsm=%r mode_machine=%r" % key)
        return snap

    # ---- actuating, each with its preconditions ----
    def enter_debug(self, supported):
        """ReleaseMode until the onboard service is gone. Refused unless: armed, robot declared supported (hoisted), onboard FSM is zero torque or damp."""
        with self._lock:
            self._require_armed()
            if not supported:
                raise ModeError("declare the robot hoisted / supported first")
            name = self.b.check_mode()
            if name is None:
                raise ModeError("cannot read the current mode")
            if name == "":
                raise ModeError("already in debug mode")
            fsm = self.b.get_fsm_id()
            if fsm not in SOFT_FSM:
                raise ModeError("onboard FSM is %r (%s): release only from zero torque (0) or damp (1); the robot is holding itself up" % (fsm, FSM_NAMES.get(fsm, "?")))
            self.remembered_name = name
            self._note("ReleaseMode (onboard mode %r, fsm %r)" % (name, fsm))
            deadline = self._clock() + self.wait_s
            while True:
                self.b.release_mode()
                self._sleep(self.poll_s)
                if self.b.check_mode() == "":
                    break
                if self._clock() > deadline:
                    raise ModeError("the onboard mode did not release in %.0f s (still %r)" % (self.wait_s, self.b.check_mode()))
            self._note("debug mode reached (the robot may be limp until our controller publishes)")
            return {"remembered_name": self.remembered_name}

    def hand_back(self, then_fsm=FSM_DAMP):
        """Give the robot back to Unitree's controller: SelectMode(remembered name), wait, then SetFsmId(then_fsm) (damp by default, never a stand/walk).
        Refused if somebody still publishes rt/lowcmd (our controller): stop it first (a second publisher would fight the onboard service)."""
        with self._lock:
            self._require_armed()
            if self.remembered_name is None:
                raise ModeError("no onboard mode name was remembered (enter_debug in this session first, or pass a name with select_name)")
            return self._select(self.remembered_name, then_fsm)

    def select_name(self, name, then_fsm=FSM_DAMP):
        """Like hand_back for a name typed by the user (e.g. when the GUI started in debug mode and nothing was remembered)."""
        with self._lock:
            self._require_armed()
            if not name or not str(name).strip():
                raise ModeError("a mode name is required")
            return self._select(str(name).strip(), then_fsm)

    def _select(self, name, then_fsm):
        hz = self.b.lowcmd_rate()
        if hz is None or hz > 0:
            raise ModeError("rt/lowcmd has %s messages/s: stop every publisher (our controller) before handing back" % ("an unknown number of" if hz is None else "%.0f" % hz))
        if then_fsm not in SOFT_FSM:
            raise ModeError("after the hand-back the robot goes to zero torque or damp only (got %r)" % then_fsm)
        self._note("SelectMode(%r)" % name)
        self.b.select_mode(name)
        deadline = self._clock() + self.wait_s
        while self.b.check_mode() != name:
            if self._clock() > deadline:
                raise ModeError("SelectMode(%r) did not take effect in %.0f s (mode is %r)" % (name, self.wait_s, self.b.check_mode()))
            self._sleep(self.poll_s)
        self.b.set_fsm_id(then_fsm)
        self._note("onboard mode %r active, FSM %d (%s)" % (name, then_fsm, FSM_NAMES.get(then_fsm)))
        return {"mode_name": name, "fsm_id": then_fsm}

    def set_fsm(self, fsm_id, supported=False):
        """Onboard FSM change (only while the onboard service runs). Soft states are always allowed when armed; anything else needs the robot declared supported."""
        with self._lock:
            self._require_armed()
            fsm_id = int(fsm_id)
            if fsm_id not in ALLOWED_FSM:
                raise ModeError("FSM id %d is not in the allowed list %r" % (fsm_id, ALLOWED_FSM))
            if not self.b.check_mode():
                raise ModeError("debug mode: there is no onboard motion service to command")
            if fsm_id not in SOFT_FSM and not supported:
                raise ModeError("FSM %d (%s) makes the robot hold itself up or move: declare it hoisted / supported first" % (fsm_id, FSM_NAMES.get(fsm_id)))
            self._note("SetFsmId(%d) %s" % (fsm_id, FSM_NAMES.get(fsm_id)))
            self.b.set_fsm_id(fsm_id)
            return {"fsm_id": fsm_id}

    def damp_now(self):
        """The one action that never needs arming: onboard FSM damp, when the onboard service is running. (In debug mode our own
        controller's stop sequence / scripts/estop_damping.py does the damping; the page says so.)"""
        with self._lock:
            name = self.b.check_mode()
            if not name:
                raise ModeError("debug mode: damp through the controller's stop sequence or scripts/robot_estop.sh, not through this call")
            self._note("DAMP (FSM 1)")
            self.b.set_fsm_id(FSM_DAMP)
            return {"fsm_id": FSM_DAMP}
