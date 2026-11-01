"""Toy closed-loop check of the v11 controller on a hoisted robot (laptop only, nothing is sent anywhere).
The plant is a first-order lag: q += clip((target - q) * lag, +-0.03) per 50 Hz tick; ramp to default, hold, blend, run, as in controller/run.py.
It reproduced the real Stage 2 run (action RMS 0.09, max 0.21, chatter 0.0046; real: 0.09 / 0.21 / 0.0037). Usage: python3 scripts/closed_loop_toy.py LOG.csv [LOG2.csv ...]
(takes the mean q of the RUN phase of each controller log.csv as the START pose; use it to see whether a dry run's restless actions settle once the pose can follow its targets)."""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "controller"))
import observation as O  # noqa: E402
from policy import NumpyPolicy  # noqa: E402

cfg = O.load_config(os.path.join(ROOT, "configs", "g1_v11_stand_turn.json"))
ob = O.ObservationBuilder(cfg)
pol = NumpyPolicy(os.path.join(ROOT, "policies", "stand_turn_v11_20261101.npz"), cfg["obs_size"], cfg["action_size"])
default = np.asarray(ob.default, np.float32)


def start_pose(f):
    q = list(csv.DictReader(open(f)))
    c = lambda n: np.array([float(x[n]) for x in q])  # noqa: E731
    m = c("phase") == 3
    return np.array([c("q%d" % k) for k in range(27)]).T[m].mean(0).astype(np.float32)


def sim(q0, lag=0.3, ramp=150, hold=100, blend=250, run=500, gyro_noise=0.004, seed=0):
    rng = np.random.default_rng(seed)
    q = q0.copy()
    a = np.zeros(13, np.float32)
    acts, qs = [], []
    for k in range(ramp + hold + blend + run):
        tgt = q0 + (default - q0) * (k + 1) / ramp if k < ramp else default.copy()
        al = 0.0 if k < ramp else min(1.0, max(0.0, (k - ramp - hold) / blend))
        o = ob.build(rng.normal(0, gyro_noise, 3), np.array([0, 0, -1.0]), np.zeros(3), q, np.zeros(27), a, k * 0.02)
        a = np.asarray(pol(o), np.float32)
        if k >= ramp:
            tgt[:13] = default[:13] + ob.act_scale * a * al if np.isscalar(ob.act_scale) else default[:13] + np.asarray(ob.act_scale)[:13] * a * al
        q = q + np.clip((tgt - q) * lag, -0.03, 0.03)
        if k >= ramp + hold + blend:
            acts.append(a.copy())
            qs.append(q.copy())
    return np.array(acts), np.array(qs)


if __name__ == "__main__":
    for f in sys.argv[1:]:
        for lag in (0.3, 0.1):
            A, Q = sim(start_pose(f), lag=lag)
            print("%s lag %.1f | RUN action RMS %.2f max %.2f chatter %.4f | leg q range %.3f rad" % (f, lag, np.sqrt((A ** 2).mean()), np.abs(A).max(), np.sqrt((np.diff(A, axis=0) ** 2).mean()), np.ptp(Q[:, :12], axis=0).max()))
