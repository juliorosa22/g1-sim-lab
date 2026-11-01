#!/usr/bin/env python3
"""Contract test: do the controller's observation / command / policy / target code reproduce what Isaac Lab did?  numpy only, Python 3.8.

    python3 robot_deploy/tests/contract_test.py [--data robot_deploy/tests/data/contract_v11.npz] [--npz robot_deploy/policies/stand_turn_v11_*.npz]

The data file comes from scripts/dump_contract_data.py (simulation). Checks, per recorded step of every env (an env is dropped from its first reset on):
  1. observation   ObservationBuilder.build(raw state) == the simulation's 78-value observation        (tolerance 1e-5)
  2. gravity       projected_gravity_from_quat(IMU quaternion) == the simulation's projected gravity     (1e-5)
  3. command       CommandShaper.step(planner request) == the command the policy saw (deadband + ramp)  (1e-5)
  4. policy        NumpyPolicy(obs) == the torch actor's output                                          (1e-4)
  5. targets       action_to_targets(last action) == Isaac Lab's processed joint targets                 (1e-5)
  6. config        joint defaults and order in the contract file == the simulation's
Exit status 0 only if every check passes.
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "controller"))

from commands import CommandShaper  # noqa: E402
from observation import ObservationBuilder, load_config, projected_gravity_from_quat  # noqa: E402
from policy import NumpyPolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "contract_v11.npz"))
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "g1_v11_stand_turn.json"))
    ap.add_argument("--npz", default=None, help="exported policy weights (default: newest policies/stand_turn_v11_*.npz)")
    args = ap.parse_args()
    npz = args.npz or sorted(glob.glob(os.path.join(ROOT, "policies", "stand_turn_v11_*[0-9].npz")))[-1]

    cfg = load_config(args.config)
    ob = ObservationBuilder(cfg)
    pol = NumpyPolicy(npz, cfg["obs_size"], cfg["action_size"])
    d = np.load(args.data)
    steps, n = d["obs"].shape[0], d["obs"].shape[1]
    ok = True

    def report(name, err, tol, extra=""):
        nonlocal ok
        passed = bool(err <= tol)
        ok &= passed
        print("%-10s max |diff| %.2e  (tolerance %.0e)  %s %s" % (name, err, tol, "PASS" if passed else "FAIL", extra))

    # 6. config vs simulation
    names_ok = list(d["joint_names"]) == ob.names
    default_err = float(np.abs(d["default27"] - ob.default).max())
    ok &= names_ok
    print("%-10s joint order matches the simulation: %s" % ("config", "PASS" if names_ok else "FAIL"))
    report("config", default_err, 1e-6, "(joint defaults)")

    # valid mask: an env is used until its first reset (t == 0 after step 0)
    valid = np.ones((steps, n), dtype=bool)
    for e in range(n):
        resets = np.nonzero(d["t"][1:, e] == 0)[0]
        if len(resets):
            valid[resets[0] + 1:, e] = False
    print("used %d of %d env-steps (%d envs, %d steps; envs are dropped from their first reset)" % (valid.sum(), steps * n, n, steps))

    e_obs = e_grav = e_cmd = e_pol = e_tgt = 0.0
    times = []
    for e in range(n):
        shaper = CommandShaper(float(d["dt"]), cfg["command"]["deadband"], cfg["command"]["ramp_lin"], cfg["command"]["ramp_yaw"])
        for k in range(steps):
            if not valid[k, e]:
                break
            shaped = shaper.step(d["request_prev"][k, e]) if k > 0 else np.zeros(3, dtype=np.float32)
            e_cmd = max(e_cmd, float(np.abs(shaped - d["cmd_shaped"][k, e]).max()))
            g = projected_gravity_from_quat(d["quat"][k, e])
            e_grav = max(e_grav, float(np.abs(g - d["gravity"][k, e]).max()))
            # the observation is built from the controller's own gravity and command, not the simulation's
            o = ob.build(d["gyro"][k, e], g, shaped, d["q"][k, e], d["dq"][k, e], d["last_action"][k, e], float(d["t"][k, e]))
            e_obs = max(e_obs, float(np.abs(o - d["obs"][k, e]).max()))
            a = pol(o)
            times.append(pol.last_call_s)
            e_pol = max(e_pol, float(np.abs(a - d["action_next"][k, e]).max()))
            if k > 0:
                tg = ob.action_to_targets(d["last_action"][k, e])
                e_tgt = max(e_tgt, float(np.abs(tg - d["processed"][k, e]).max()))
    report("gravity", e_grav, 1e-5, "(IMU quaternion -> projected gravity)")
    report("command", e_cmd, 1e-5, "(deadband + ramp)")
    report("observation", e_obs, 1e-5, "(78 values)")
    report("policy", e_pol, 1e-4, "(numpy network vs torch actor)")
    report("targets", e_tgt, 1e-5, "(default + scale * action)")
    print("policy call: median %.3f ms, max %.3f ms (this machine)" % (1e3 * np.median(times), 1e3 * max(times)))
    print("RESULT:", "ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
