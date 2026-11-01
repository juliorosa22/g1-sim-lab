"""Python side of the shared-memory protocol with the C++ g1_bridge (controller_cpp/bridge/DESIGN.md).  Python 3.8, numpy only.

BridgeLink has the same calls as state.DdsLink (latest(), n_foreign_cmd, start_publisher(), send()), so run.py can use either. The bridge owns all DDS traffic;
this module only reads one shared struct (robot state) and writes another (the command). Layout mirrors controller_cpp/bridge/shm_protocol.h and is checked against
`g1_bridge --print-layout` by controller_cpp/tests/test_bridge.py. All timestamps are CLOCK_MONOTONIC seconds (time.perf_counter() on Linux).
"""
import mmap
import os
import struct
import time

import numpy as np

N = 29
MAGIC = 0x52423147
VERSION = 1
CMD_ENABLE, CMD_STOP, CMD_END = 1, 2, 4
ST_IDLE, ST_ACTIVE, ST_STOPPING, ST_STOPPED, ST_REFUSED = range(5)
ST_NAMES = ["IDLE", "ACTIVE", "STOPPING", "STOPPED", "REFUSED"]

STATE_DT = np.dtype([("seq", "<u8"), ("chk_xor", "<u8"), ("chk_sum", "<u8"), ("t_arrival", "<f8"), ("n_msgs", "<u8"), ("n_crc_bad", "<u8"), ("n_foreign_cmd", "<u8"),
                     ("tick", "<u4"), ("mode_machine", "<u4"), ("q", "<f8", (N,)), ("dq", "<f8", (N,)), ("tau", "<f8", (N,)), ("temp", "<f8", (N,)),
                     ("mode", "<i4", (N + 1,)), ("gyro", "<f8", (3,)), ("quat", "<f8", (4,))])
CMD_DT = np.dtype([("seq", "<u8"), ("chk_xor", "<u8"), ("chk_sum", "<u8"), ("counter", "<u8"), ("t_client", "<f8"), ("flags", "<u4"), ("pad", "<u4"),
                   ("target", "<f8", (N,)), ("kp", "<f8", (N,)), ("kd", "<f8", (N,))])
STATUS_DT = np.dtype([("seq", "<u8"), ("chk_xor", "<u8"), ("chk_sum", "<u8"), ("state", "<u4"), ("reason_code", "<u4"), ("reason", "S96"), ("n_published", "<u8"),
                      ("n_cmd_seen", "<u8"), ("n_cmd_rejected", "<u8"), ("n_deadman", "<u8"), ("t_started", "<f8"), ("t_heartbeat", "<f8"), ("t_last_publish", "<f8"),
                      ("last_write_ms", "<f8"), ("max_write_ms", "<f8"), ("cmd_age_ms", "<f8"), ("publishing", "<u4"), ("pad", "<u4")])
HEADER = struct.Struct("<8I")
PAYLOAD_OFF = 24   # seq + two checksums


def _align64(x):
    return (x + 63) & ~63


OFF_STATE = _align64(HEADER.size)
OFF_CMD = OFF_STATE + _align64(STATE_DT.itemsize)
OFF_STATUS = OFF_CMD + _align64(CMD_DT.itemsize)
TOTAL = OFF_STATUS + _align64(STATUS_DT.itemsize)


def checksum(payload_bytes):
    w = np.frombuffer(payload_bytes, dtype=np.uint64)
    return int(np.bitwise_xor.reduce(w)), int(w.sum(dtype=np.uint64))


class BridgeLink:
    def __init__(self, path="/dev/shm/g1_bridge", wait_s=5.0):
        t0 = time.monotonic()
        while not os.path.exists(path) or os.path.getsize(path) < TOTAL:
            if time.monotonic() - t0 > wait_s:
                raise RuntimeError("no g1_bridge shared memory at %s (is the bridge running?)" % path)
            time.sleep(0.02)
        self.f = open(path, "r+b")
        self.mm = mmap.mmap(self.f.fileno(), TOTAL)
        magic, version, n, hs, o1, o2, o3, tot = HEADER.unpack_from(self.mm, 0)
        if magic != MAGIC or version != VERSION or n != N or (o1, o2, o3, tot) != (OFF_STATE, OFF_CMD, OFF_STATUS, TOTAL):
            raise RuntimeError("shared memory header does not match this client (magic %x version %d layout %s)" % (magic, version, (o1, o2, o3, tot)))
        self.counter = 0
        self._last_cmd = None
        self._status_cache = None
        self._state_cache = None

    # ---- seqlock reads
    def _read(self, off, dt):
        for _ in range(8):
            raw = bytes(self.mm[off:off + dt.itemsize])
            s_after = struct.unpack_from("<Q", self.mm, off)[0]
            arr = np.frombuffer(raw, dtype=dt)[0]
            s1 = int(arr["seq"])
            if s1 & 1 or s1 != s_after or s1 == 0:
                continue
            x, s = checksum(raw[PAYLOAD_OFF:])
            if x == int(arr["chk_xor"]) and s == int(arr["chk_sum"]):
                return arr
        return None

    def latest(self, now=None):
        a = self._read(OFF_STATE, STATE_DT)
        for _ in range(30):                      # the writer is inside its update for a few microseconds at a time: wait it out
            if a is not None:
                break
            time.sleep(0.0001)
            a = self._read(OFF_STATE, STATE_DT)
        if a is None:
            return self._state_cache             # still nothing consistent: the last good sample (the controller judges its age); None only before the first one
        self._state_cache = self._state_dict(a)
        return self._state_cache

    @staticmethod
    def _state_dict(a):
        return {"t": float(a["t_arrival"]), "mm": int(a["mode_machine"]), "q": a["q"].astype(np.float64), "dq": a["dq"].astype(np.float64), "tau": a["tau"].astype(np.float64),
                "temp": a["temp"].astype(np.float64), "mode": a["mode"][:N].astype(np.int64), "gyro": a["gyro"].astype(np.float64), "quat": a["quat"].astype(np.float64),
                "n_foreign_cmd": int(a["n_foreign_cmd"]), "n_crc_bad": int(a["n_crc_bad"]), "n_msgs": int(a["n_msgs"])}

    @property
    def n_foreign_cmd(self):
        a = self._read(OFF_STATE, STATE_DT)
        return 0 if a is None else int(a["n_foreign_cmd"])

    def bridge_status(self):
        a = self._read(OFF_STATUS, STATUS_DT)
        if a is None:
            return self._status_cache
        self._status_cache = {"state": int(a["state"]), "state_name": ST_NAMES[min(int(a["state"]), 4)], "reason_code": int(a["reason_code"]),
                              "reason": bytes(a["reason"]).split(b"\0")[0].decode("utf-8", "replace"), "n_published": int(a["n_published"]), "n_cmd_seen": int(a["n_cmd_seen"]),
                              "n_cmd_rejected": int(a["n_cmd_rejected"]), "n_deadman": int(a["n_deadman"]), "t_heartbeat": float(a["t_heartbeat"]),
                              "last_write_ms": float(a["last_write_ms"]), "max_write_ms": float(a["max_write_ms"]), "cmd_age_ms": float(a["cmd_age_ms"]), "publishing": int(a["publishing"])}
        return self._status_cache

    # ---- command writes (single writer: this process)
    def start_publisher(self):
        pass                      # the bridge is the publisher; the first send() enables it

    def _write_cmd(self, target, kp, kd, flags):
        self.counter += 1
        c = np.zeros(1, dtype=CMD_DT)
        c["counter"] = self.counter
        c["t_client"] = time.perf_counter()
        c["flags"] = flags
        c["target"][0] = np.asarray(target, dtype=np.float64)
        c["kp"][0] = np.asarray(kp, dtype=np.float64)
        c["kd"][0] = np.asarray(kd, dtype=np.float64)
        raw = c.tobytes()
        x, s = checksum(raw[PAYLOAD_OFF:])
        seq = struct.unpack_from("<Q", self.mm, OFF_CMD)[0]
        struct.pack_into("<Q", self.mm, OFF_CMD, seq + 1)                           # odd: the bridge will not trust a read now
        self.mm[OFF_CMD + 8:OFF_CMD + CMD_DT.itemsize] = struct.pack("<QQ", x, s) + raw[PAYLOAD_OFF:]
        struct.pack_into("<Q", self.mm, OFF_CMD, seq + 2)                           # even again: consistent
        self._last_cmd = (np.array(target, dtype=np.float64), np.array(kp, dtype=np.float64), np.array(kd, dtype=np.float64))

    def send(self, target, kp, kd):
        """Same call as state.DdsLink.send: a new command, with ENABLE set (the bridge enables on the first valid one)."""
        self._write_cmd(target, kp, kd, CMD_ENABLE)

    def request_stop(self):
        """Ask the bridge to run its stop sequence now (kp fade, then damping)."""
        t, kp, kd = self._last_cmd if self._last_cmd else (np.zeros(N), np.zeros(N), np.zeros(N))
        self._write_cmd(t, kp, kd, CMD_ENABLE | CMD_STOP)

    def end(self):
        """Clean end of the run: the bridge stops publishing at once (use after the controller's own stop sequence has finished)."""
        t, kp, kd = self._last_cmd if self._last_cmd else (np.zeros(N), np.zeros(N), np.zeros(N))
        self._write_cmd(t, kp, kd, CMD_ENABLE | CMD_END)

    def close(self):
        try:
            self.mm.close()
            self.f.close()
        except Exception:
            pass
