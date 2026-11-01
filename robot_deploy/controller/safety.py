"""Safety limits and checks for the controller.  Python 3.8, numpy only.

Every check returns a reason string (the controller then goes to the STOP sequence) or None. The limits are printed in the run plan; they are
proposals to be reviewed by the user before the first run that publishes. Motors 13 and 14 (waist roll/pitch) are absent on this unit and ignored.
"""
import math

import numpy as np

from observation import projected_gravity_from_quat

ABSENT = (13, 14)


def default_torque_caps():
    """|tau_est| trip level per motor (N*m), 29 values. Proposals below the simulation limits (88 / 139 / 139 / 50 ...); to be reviewed per run."""
    cap = np.full(29, 25.0)                      # arms
    for side in (0, 6):
        cap[side + 0] = 60.0                     # hip pitch
        cap[side + 1] = 80.0                     # hip roll
        cap[side + 2] = 60.0                     # hip yaw
        cap[side + 3] = 80.0                     # knee
        cap[side + 4] = 40.0                     # ankle pitch
        cap[side + 5] = 40.0                     # ankle roll
    cap[12] = 50.0                               # waist yaw
    cap[13] = cap[14] = 1e9                      # absent
    return cap


def stage2_torque_caps():
    """Tighter |tau_est| trip levels for the hoisted run (Stage 2): the only loads are limb gravity, so anything near these is abnormal. Ground stages use default_torque_caps()."""
    cap = np.full(29, 15.0)                      # arms
    for side in (0, 6):
        cap[side + 0] = 30.0                     # hip pitch
        cap[side + 1] = 30.0                     # hip roll
        cap[side + 2] = 30.0                     # hip yaw
        cap[side + 3] = 35.0                     # knee
        cap[side + 4] = 20.0                     # ankle pitch
        cap[side + 5] = 20.0                     # ankle roll
    cap[12] = 20.0                               # waist yaw
    cap[13] = cap[14] = 1e9                      # absent
    return cap


def stage3_torque_caps():
    """|tau_est| trip levels for the runs with part of the weight on the legs (Stage 3b): above the Stage 2 levels, well below the defaults. Full standing needs a few N*m per joint in simulation (knee 3.5)."""
    cap = np.full(29, 15.0)                      # arms
    for side in (0, 6):
        cap[side + 0] = 40.0                     # hip pitch
        cap[side + 1] = 40.0                     # hip roll
        cap[side + 2] = 40.0                     # hip yaw
        cap[side + 3] = 40.0                     # knee
        cap[side + 4] = 25.0                     # ankle pitch
        cap[side + 5] = 25.0                     # ankle roll
    cap[12] = 25.0                               # waist yaw
    cap[13] = cap[14] = 1e9                      # absent
    return cap


class Limits:
    def __init__(self, tau_cap=None, tilt_deg=25.0, speed=5.0, track_err=0.30, stale_s=0.05, temp_c=70.0, period_s=0.040):
        self.tau_cap = default_torque_caps() if tau_cap is None else np.asarray(tau_cap, dtype=np.float64)
        self.tilt_deg, self.speed, self.track_err, self.stale_s, self.temp_c, self.period_s = tilt_deg, speed, track_err, stale_s, temp_c, period_s

    def describe(self):
        return ("stale state > %.0f ms | NaN/inf | tilt > %.0f deg | any joint speed > %.1f rad/s | |tau_est| above the per-motor cap (e.g. hip pitch %.0f, knee %.0f, "
                "ankle %.0f, waist yaw %.0f, arms %.0f N*m) | tracking error |q - target| > %.2f rad | motor > %.0f C | loop period > %.0f ms | STOP file / signal" % (
                    self.stale_s * 1e3, self.tilt_deg, self.speed, self.tau_cap[0], self.tau_cap[3], self.tau_cap[4], self.tau_cap[12], self.tau_cap[15],
                    self.track_err, self.temp_c, self.period_s * 1e3))


def tilt_deg(quat_wxyz):
    g = projected_gravity_from_quat(quat_wxyz)
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(g[2])))))


def check_state(st, now, lim, target29=None, enforce_tracking=False):
    """st: dict from a state source (t, q, dq, tau, temp, gyro, quat, 29-motor arrays). Returns a reason string or None."""
    if st is None:
        return "no state yet"
    if now - st["t"] > lim.stale_s:
        return "stale state (%.0f ms old)" % ((now - st["t"]) * 1e3)
    for k in ("q", "dq", "tau", "gyro", "quat"):
        if not np.all(np.isfinite(st[k])):
            return "NaN/inf in " + k
    live = [i for i in range(29) if i not in ABSENT]
    if tilt_deg(st["quat"]) > lim.tilt_deg:
        return "tilt %.1f deg > %.0f" % (tilt_deg(st["quat"]), lim.tilt_deg)
    dq = np.abs(st["dq"][live])
    if dq.max() > lim.speed:
        return "joint %d speed %.1f rad/s > %.1f" % (live[int(dq.argmax())], dq.max(), lim.speed)
    over = np.abs(st["tau"]) > lim.tau_cap
    if over.any():
        i = int(np.argmax(over))
        return "motor %d |tau_est| %.1f N*m > cap %.0f" % (i, abs(st["tau"][i]), lim.tau_cap[i])
    if st["temp"][live].max() > lim.temp_c:
        return "motor %d at %d C" % (live[int(np.argmax(st["temp"][live]))], int(st["temp"][live].max()))
    if enforce_tracking and target29 is not None:
        err = np.abs(st["q"] - target29)[live]
        if err.max() > lim.track_err:
            return "joint %d tracking error %.2f rad > %.2f" % (live[int(err.argmax())], err.max(), lim.track_err)
    return None
