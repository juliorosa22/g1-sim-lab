"""Tests of controller/selector.py (numpy only, Python 3.8):  python3 robot_deploy/tests/test_selector.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "controller"))
from commands import CommandShaper  # noqa: E402
from selector import STAND, WALK, PolicySelector  # noqa: E402

FAILS = []


def check(name, ok, info=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("  " + info if info else ""))
    if not ok:
        FAILS.append(name)


def run(requests_by_second, seconds):
    sel, shaper = PolicySelector(), CommandShaper()
    log = []
    for i in range(int(seconds / 0.02)):
        t = i * 0.02
        req = requests_by_second(t)
        mode, w = sel.update(shaper.step(req))
        log.append((t, mode, w, float(shaper.cur[0])))
    return sel, log


# 1. stand at zero command: never walks
sel, log = run(lambda t: (0.0, 0.0, 0.0), 10)
check("zero command stays in stand, w = 0", all(m == STAND and w == 0 for _, m, w, _ in log) and sel.switches == 0)

# 2. yaw only, lateral, backward and creep: stay with the stander
for name, req in [("yaw 1.0", (0.0, 0.0, 1.0)), ("lateral 0.5", (0.0, 0.5, 0.0)), ("backward -0.5", (-0.5, 0.0, 0.0)), ("creep 0.25", (0.25, 0.0, 0.0)), ("forward 0.29 (below enter; 0.3 exactly sits on the threshold)", (0.29, 0.0, 0.0))]:
    sel, log = run(lambda t, r=req: r, 8)
    check("route forward: %s stays with v11" % name, sel.switches == 0 and log[-1][1] == STAND)

# 3. forward 0.6 m/s: switches to walk once vx passes 0.3 (after the 1 m/s^2 ramp, 0.3 s, and a debounce of 0.2 s), blend 0.3 s, no flapping
sel, log = run(lambda t: (0.6, 0.0, 0.0) if t >= 1.0 else (0.0, 0.0, 0.0), 6)
t_switch = next(t for t, m, _, _ in log if m == WALK)
check("forward 0.6: switches to walk once", sel.switches == 1 and log[-1][1] == WALK, "t_switch = %.2f s" % t_switch)
check("walk starts after ramp (0.3 s) + debounce (0.2 s)", 1.45 <= t_switch <= 1.65)
ws = [w for _, _, w, _ in log]
check("w rises monotonically from 0 to 1 over about 0.3 s", all(b >= a for a, b in zip(ws, ws[1:])) and ws[-1] == 1.0)
t_full = next(t for t, _, w, _ in log if w >= 1.0)
check("blend takes 0.3 s", abs((t_full - t_switch) - 0.3) < 0.05, "%.2f s" % (t_full - t_switch))

# 4. hysteresis: vx between exit (0.2) and enter (0.3) while walking does not switch back
sel, log = run(lambda t: (0.6, 0.0, 0.0) if t < 3 else (0.25, 0.0, 0.0), 8)
check("0.25 m/s while walking stays in walk (inside the band)", sel.switches == 1 and log[-1][1] == WALK)

# 5. stop: back to stand after vx stays below 0.2 for 0.3 s
sel, log = run(lambda t: (0.6, 0.0, 0.0) if t < 3 else (0.0, 0.0, 0.0), 8)
back = [t for t, m, _, _ in log if t > 3 and m == STAND]
check("stop returns to stand, 2 switches in total", sel.switches == 2 and log[-1][1] == STAND and log[-1][2] == 0.0)

# 6. a noisy request around the threshold does not flap faster than the debounce allows
rng = np.random.RandomState(0)
sel, log = run(lambda t: (0.25 + 0.15 * rng.randn() * 0.5, 0.0, 0.0), 30)
check("noisy request around 0.25 m/s: at most a couple of switches in 30 s", sel.switches <= 2, "switches = %d" % sel.switches)

# 7. mix(): w = 0 / 1 return the pure action, in between the blend; needs() says which nets run
sel = PolicySelector()
a_s, a_w = np.zeros(13, np.float32), np.ones(13, np.float32)
check("mix at w = 0 is the stander", np.allclose(sel.mix(a_s, None), 0.0) and sel.needs() == (True, False))
sel.mode, sel.w = WALK, 0.5
check("mix at w = 0.5 is the average and both nets run", np.allclose(sel.mix(a_s, a_w), 0.5) and sel.needs() == (True, True))
sel.w = 1.0
check("mix at w = 1 is the walker", np.allclose(sel.mix(None, a_w), 1.0) and sel.needs() == (False, True))

# 8. bad construction
try:
    PolicySelector(enter=0.2, exit=0.3)
    check("enter must be above exit", False)
except ValueError:
    check("enter must be above exit", True)

print("RESULT: %s" % ("ALL PASS" if not FAILS else "FAILED: " + ", ".join(FAILS)))
sys.exit(1 if FAILS else 0)
