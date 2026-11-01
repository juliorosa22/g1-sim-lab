#!/usr/bin/env python3
"""Stage 0b: feed a REAL rt/lowstate recording through the controller's observation code and the v11 network. Laptop only, numpy, Python 3.8.

    OPENBLAS_NUM_THREADS=1 python3 robot_deploy/tests/real_state_check.py [--csv measurements/20261029/standing_lowstate_raw.csv]

The CSV is the step-A listener format (scripts' listen output: t, quaternion q0..q3 (w,x,y,z), gyro g*, accel a*, q_i, dq_i, tau_i for the 29 motors).
Checks: no NaN, gravity near [0, 0, -1], observation ranges, action size, and how far the policy's targets are from the measured pose (the step the robot would
make when the policy is blended in). Nothing is published anywhere.
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "controller"))

from observation import ObservationBuilder, load_config, projected_gravity_from_quat  # noqa: E402
from policy import NumpyPolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "measurements", "20261029", "standing_lowstate_raw.csv"))
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "g1_v11_stand_turn.json"))
    ap.add_argument("--npz", default=None)
    args = ap.parse_args()
    npz = args.npz or sorted(glob.glob(os.path.join(ROOT, "policies", "stand_turn_v11_*[0-9].npz")))[-1]
    cfg = load_config(args.config)
    ob, pol = ObservationBuilder(cfg), NumpyPolicy(npz, cfg["obs_size"], cfg["action_size"])
    d = np.genfromtxt(args.csv, delimiter=",", names=True)
    t = d["t"] - d["t"][0]
    # 50 Hz subsample: the first sample at or after each 20 ms tick
    ticks = np.arange(0.0, t[-1], 0.02)
    idx = np.unique(np.searchsorted(t, ticks))
    q_all = np.stack([d["q_%d" % i] for i in range(29)], axis=1)
    dq_all = np.stack([d["dq_%d" % i] for i in range(29)], axis=1)
    quat = np.stack([d["q0"], d["q1"], d["q2"], d["q3"]], axis=1)
    gyro = np.stack([d["gx"], d["gy"], d["gz"]], axis=1)

    last = np.zeros(cfg["action_size"], dtype=np.float32)
    obs_l, act_l, tgt_gap, grav_l, tt = [], [], [], [], []
    for k, i in enumerate(idx):
        g = projected_gravity_from_quat(quat[i])
        q27, dq27 = ob.gather(q_all[i]), ob.gather(dq_all[i])
        o = ob.build(gyro[i], g, [0.0, 0.0, 0.0], q27, dq27, last, k * 0.02)
        a = pol(o)
        tg = ob.action_to_targets(a)
        obs_l.append(o); act_l.append(a); grav_l.append(g)
        tgt_gap.append(tg - q27[:cfg["action_size"]])
        last = a.astype(np.float32)           # open-loop chain: the robot does NOT follow the policy here
    O, A, G, GAP = map(np.array, (obs_l, act_l, grav_l, tgt_gap))
    names = [j["name"] for j in cfg["joints"]]
    print("rows %d (50 Hz subsample of %.1f s), recording %s" % (len(idx), t[-1], os.path.basename(args.csv)))
    print("NaN/inf in observations: %s ; in actions: %s" % (not np.all(np.isfinite(O)), not np.all(np.isfinite(A))))
    print("gravity mean %s, min z %.4f ; tilt from vertical: mean %.2f deg, max %.2f deg" % (
        G.mean(0).round(4), G[:, 2].min(), np.degrees(np.arccos(np.clip(-G[:, 2], -1, 1))).mean(), np.degrees(np.arccos(np.clip(-G[:, 2], -1, 1))).max()))
    sl = {"gyro x0.25": (0, 3), "gravity": (3, 6), "command": (6, 9), "joint_pos-default": (9, 36), "joint_vel x0.05": (36, 63), "last_action": (63, 76), "gait clock": (76, 78)}
    for k, (a_, b_) in sl.items():
        print("  obs %-18s min %+8.3f max %+8.3f  max|.| %.3f" % (k, O[:, a_:b_].min(), O[:, a_:b_].max(), np.abs(O[:, a_:b_]).max()))
    print("joint_pos - default (rad), mean over the recording (policy joint order):")
    jp = O[:, 9:36].mean(0)
    for n, v in zip(names, jp):
        print("   %-26s %+.3f" % (n, v))
    print("action (raw), first step / mean / max|.| over the recording, per controlled joint:")
    for n, a0, am, ax in zip(names[:cfg["action_size"]], A[0], A.mean(0), np.abs(A).max(0)):
        print("   %-26s %+7.3f  %+7.3f  %6.3f" % (n, a0, am, ax))
    print("target - measured position at the first step (rad): max |gap| %.3f at %s ; legs+waist gaps: %s" % (
        np.abs(GAP[0]).max(), names[int(np.abs(GAP[0]).argmax())], np.round(GAP[0], 3).tolist()))
    print("policy call time on this machine: %.3f ms (last)" % (pol.last_call_s * 1e3))


if __name__ == "__main__":
    main()
