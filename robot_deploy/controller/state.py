"""State sources and the command publisher.  Python 3.8.

DdsLink            real robot: subscribes to rt/lowstate (and rt/lowcmd, only to detect a foreign publisher); publishes rt/lowcmd. Imports the Unitree SDK lazily,
                   so everything else in this folder runs on a laptop.
ReplayState        replays a recorded lowstate CSV (laptop tests).
NullPublisher      remembers what WOULD be sent (dry-run, tests); never touches a network.

A state is a dict: t (monotonic seconds at arrival), q, dq, tau (29 motors), temp (29), mode (29), gyro (3, body frame), quat (4, w x y z), mm (mode_machine).
"""
import subprocess
import sys
import time

import numpy as np

N = 29


def robot_iface():
    for l in subprocess.check_output(["ip", "-o", "-4", "addr", "show"]).decode().splitlines():
        p = l.split()
        if p[3].startswith("192.168.123.164/"):
            return p[1]
    raise RuntimeError("no interface carries 192.168.123.164 (run this ON the robot)")


def state_from_msg(m, t):
    """Convert one rt/lowstate message into the state dict. Called once per control tick (50 Hz), NOT in the DDS callback (about 1000 Hz): converting in the
    callback starved the control loop of Python's global lock and made ticks late by 25-45 ms (measured 2026-11-01)."""
    s, im = m.motor_state, m.imu_state
    return {"t": t, "mm": int(m.mode_machine),
            "q": np.array([s[i].q for i in range(N)], dtype=np.float64), "dq": np.array([s[i].dq for i in range(N)], dtype=np.float64),
            "tau": np.array([s[i].tau_est for i in range(N)], dtype=np.float64), "temp": np.array([s[i].temperature[0] for i in range(N)], dtype=np.float64),
            "mode": np.array([s[i].mode for i in range(N)], dtype=np.int64),
            "gyro": np.array(im.gyroscope, dtype=np.float64), "quat": np.array(im.quaternion, dtype=np.float64)}


class DdsLink:
    def __init__(self):
        sys.path.insert(0, "/home/unitree/unitree_sdk2_python")
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber, ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC
        self._mk_cmd, self._LowCmd, self._CRC, self._Pub = unitree_hg_msg_dds__LowCmd_, LowCmd_, CRC, ChannelPublisher
        self._raw, self._raw_t, self._cache, self._cache_id = None, 0.0, None, None
        self.st = None
        self.n_foreign_cmd = 0
        self.mm = 0
        ChannelFactoryInitialize(0, robot_iface())
        self._sub_s = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub_s.Init(self._on_state, 10)
        self._sub_c = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self._sub_c.Init(self._on_cmd, 10)
        self._pub = None
        self.published = 0

    def _on_state(self, m):
        self._raw, self._raw_t = m, time.monotonic()      # store only: no conversion in the callback

    def _on_cmd(self, m):
        self.n_foreign_cmd += 1

    def latest(self, now=None):
        m, t = self._raw, self._raw_t
        if m is None:
            return None
        if self._cache_id is not id(m):
            self._cache, self._cache_id = state_from_msg(m, t), id(m)
        self.st = self._cache
        self.mm = self._cache["mm"]
        return self._cache

    def start_publisher(self):
        self._pub = self._Pub("rt/lowcmd", self._LowCmd)
        self._pub.Init()
        self._cmd = self._mk_cmd()
        self._crc = self._CRC()

    def send(self, target29, kp29, kd29):
        """Mode 1 (enabled), position target + kp/kd, velocity target 0, torque feed-forward 0: the same pure PD the onboard controller sends."""
        c = self._cmd
        c.mode_pr, c.mode_machine = 0, self.mm
        for i in range(N):
            m = c.motor_cmd[i]
            m.mode, m.q, m.dq, m.tau = 1, float(target29[i]), 0.0, 0.0
            m.kp, m.kd = float(kp29[i]), float(kd29[i])
        c.crc = self._crc.Crc(c)
        self._pub.Write(c)
        self.published += 1


class NullPublisher:
    """Dry-run: stores the last command that WOULD have been sent."""

    def __init__(self):
        self.sent = []

    def send(self, target29, kp29, kd29):
        self.sent.append((np.array(target29), np.array(kp29), np.array(kd29)))


class ReplayState:
    """Replays a recorded lowstate CSV (columns t, q0..q3 quaternion, gx..gz, q_i, dq_i, tau_i) at 50 Hz against a (fake or real) clock."""

    def __init__(self, csv_path, clock):
        d = np.genfromtxt(csv_path, delimiter=",", names=True)
        t = d["t"] - d["t"][0]
        self.idx = np.unique(np.searchsorted(t, np.arange(0.0, t[-1], 0.02)))
        self.d, self.clock, self.k = d, clock, 0
        self.n_foreign_cmd = 0
        self.stale_next = False

    def latest(self, now=None):
        i = self.idx[min(self.k, len(self.idx) - 1)]
        self.k += 1
        d = self.d
        return {"t": self.clock.now() - (1.0 if self.stale_next else 0.0), "mm": 6,
                "q": np.array([d["q_%d" % j][i] for j in range(N)]), "dq": np.array([d["dq_%d" % j][i] for j in range(N)]),
                "tau": np.array([d["tau_%d" % j][i] for j in range(N)]), "temp": np.full(N, 40.0), "mode": np.ones(N, dtype=np.int64),
                "gyro": np.array([d["gx"][i], d["gy"][i], d["gz"][i]]), "quat": np.array([d["q0"][i], d["q1"][i], d["q2"][i], d["q3"][i]])}

    def exhausted(self):
        return self.k >= len(self.idx)
