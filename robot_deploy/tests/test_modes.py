"""Tests of gui/modes.py with the mock robot (no robot, no SDK):  python3 robot_deploy/tests/test_modes.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gui"))
from modes import FSM_DAMP, FSM_START, MockBackend, ModeError, ModeManager  # noqa: E402

FAILS = []


def check(name, ok, info=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + info if info else ""))
    if not ok:
        FAILS.append(name)


def refuses(fn, *a, **k):
    try:
        fn(*a, **k)
    except ModeError as e:
        return str(e)
    return None


def manager(backend):
    t = [0.0]

    def clock():
        t[0] += 0.05
        return t[0]
    return ModeManager(backend, wait_s=2.0, poll_s=0.1, sleep=lambda s: None, clock=clock)


# 1. nothing actuating without arming
b = MockBackend()
m = manager(b)
check("enter_debug refused when not armed", refuses(m.enter_debug, True) is not None and not b.calls)
check("hand_back refused when not armed", refuses(m.hand_back) is not None)
check("set_fsm refused when not armed", refuses(m.set_fsm, FSM_DAMP) is not None and not b.calls)

# 2. enter debug: needs 'supported' and a soft FSM
m.arm(True)
check("enter_debug refused unless the robot is declared supported", refuses(m.enter_debug, False) is not None and not b.calls)
b.fsm = FSM_START
check("enter_debug refused from a holding FSM (500)", "holding" in (refuses(m.enter_debug, True) or "") and not b.calls)
b.fsm = 801
check("enter_debug refused from balance FSM 801", refuses(m.enter_debug, True) is not None and not b.calls)
b.fsm = FSM_DAMP
r = m.enter_debug(True)
check("enter_debug from damp releases and remembers the name", b.mode == "" and m.remembered_name == "ai" and r["remembered_name"] == "ai" and ("release",) in b.calls)
check("enter_debug twice refused", refuses(m.enter_debug, True) is not None)

# 3. hand back: refused while somebody publishes rt/lowcmd, then selects the remembered name and ends in damp (never a stand)
b.foreign = 480.0
check("hand_back refused while rt/lowcmd is being published", "rt/lowcmd" in (refuses(m.hand_back) or "") and ("select", "ai") not in b.calls)
b.foreign = None
check("hand_back refused when the lowcmd rate is unknown", refuses(m.hand_back) is not None)
b.foreign = 0.0
check("hand_back refuses a stand/walk target", refuses(m.hand_back, FSM_START) is not None)
r = m.hand_back()
check("hand_back selects the remembered mode and damps", b.mode == "ai" and b.fsm == FSM_DAMP and ("select", "ai") in b.calls and ("fsm", FSM_DAMP) in b.calls)

# 4. fsm changes: soft states ok when armed, stand/walk need 'supported', debug mode has no service
check("set_fsm damp allowed", m.set_fsm(FSM_DAMP)["fsm_id"] == FSM_DAMP)
check("set_fsm start (500) refused unless supported", refuses(m.set_fsm, FSM_START) is not None and b.fsm == FSM_DAMP)
check("set_fsm start (500) allowed when supported", m.set_fsm(FSM_START, supported=True)["fsm_id"] == FSM_START)
check("set_fsm refuses ids outside the list (801, 4, 9999)", all(refuses(m.set_fsm, i, True) is not None for i in (801, 4, 9999)))

# 5. damp_now never needs arming, works in the onboard world only
m.arm(False)
b.fsm = FSM_START
m.damp_now()
check("damp_now works when disarmed", b.fsm == FSM_DAMP)
b.mode = ""
check("damp_now in debug mode points to the controller's stop", "debug" in (refuses(m.damp_now) or ""))

# 6. a mode that does not switch is reported, not ignored
b2 = MockBackend()
m2 = manager(b2)
m2.arm(True)
b2.select_mode = lambda name: 0  # the service never comes back
b2.mode = ""
m2.remembered_name = "ai"
check("hand_back reports a SelectMode that does not take effect", "did not take effect" in (refuses(m2.hand_back) or ""))

# 7. snapshot: the read-only discovery table records every change once
b3 = MockBackend(fsm=FSM_DAMP)
m3 = manager(b3)
for _ in range(3):
    m3.snapshot()
b3.fsm = 801
m3.snapshot()
m3.snapshot()
check("observer logs each change once", [o["fsm_id"] for o in m3.observed] == [FSM_DAMP, 801], str([o["fsm_id"] for o in m3.observed]))
s = m3.snapshot()
check("snapshot names the world and the FSM", s["world"] == "onboard" and s["fsm_id"] == 801 and "balance" in s["fsm_text"])

# 8. a failing backend call becomes a clear error path, not a state change
b4 = MockBackend()
m4 = manager(b4)
m4.arm(True)
b4.fail_next = "check_mode"
try:
    m4.enter_debug(True)
    raised = False
except Exception:
    raised = True
check("a backend failure aborts enter_debug before any release", raised and ("release",) not in b4.calls)

print("RESULT: %s" % ("ALL PASS" if not FAILS else "FAILED: " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
