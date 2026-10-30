"""Full-rate READ-ONLY recorder for rt/lowstate and rt/lowcmd. Publishes nothing. Runs ON the robot, Python 3.8.

  python3 record_walk.py OUTDIR [--duration SECONDS]      stop early by creating OUTDIR/STOP

Every message is stored (no decimation) as packed binary numpy records, written in ~1 s batches by a writer thread:
  state.bin  dtype in meta.json["dtype_state"]     cmd.bin  dtype in meta.json["dtype_cmd"]
  status.json (heartbeat every 5 s), meta.json (at the end).  t = time.monotonic_ns() at arrival; wall_start_ns in meta.json.
Read back with np.fromfile(path, dtype=np.dtype(...)); the dtypes are rebuilt by load_walk() in the laptop-side analysis.
"""
import argparse, json, os, queue, socket, subprocess, sys, threading, time
import numpy as np

sys.path.insert(0, "/home/unitree/unitree_sdk2_python")
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

N = 29
DT_STATE = [("t", "<i8"), ("tick", "<u4"), ("mm", "u1"), ("quat", "<f4", 4), ("gyro", "<f4", 3), ("acc", "<f4", 3),
            ("rpy", "<f4", 3), ("rc", "u1", 40), ("q", "<f4", N), ("dq", "<f4", N), ("tau", "<f4", N),
            ("temp", "i1", N), ("mmode", "u1", N)]
DT_CMD = [("t", "<i8"), ("mode_pr", "u1"), ("mm", "u1"), ("q", "<f4", N), ("dq", "<f4", N), ("tau", "<f4", N),
          ("kp", "<f4", N), ("kd", "<f4", N), ("mode", "u1", N)]
dts, dtc = np.dtype(DT_STATE), np.dtype(DT_CMD)


def pack_state(batch):
    a = np.zeros(len(batch), dtype=dts)
    for k, (t, m) in enumerate(batch):
        im, s = m.imu_state, m.motor_state
        a[k]["t"], a[k]["tick"], a[k]["mm"] = t, m.tick, m.mode_machine
        a[k]["quat"], a[k]["gyro"], a[k]["acc"], a[k]["rpy"] = im.quaternion, im.gyroscope, im.accelerometer, im.rpy
        a[k]["rc"] = np.frombuffer(bytes(m.wireless_remote)[:40].ljust(40, b"\0"), dtype="u1")
        a[k]["q"] = [s[i].q for i in range(N)]
        a[k]["dq"] = [s[i].dq for i in range(N)]
        a[k]["tau"] = [s[i].tau_est for i in range(N)]
        a[k]["temp"] = [max(-127, min(127, int(s[i].temperature[0]))) for i in range(N)]
        a[k]["mmode"] = [int(s[i].mode) for i in range(N)]
    return a


def pack_cmd(batch):
    a = np.zeros(len(batch), dtype=dtc)
    for k, (t, m) in enumerate(batch):
        c = m.motor_cmd
        a[k]["t"], a[k]["mode_pr"], a[k]["mm"] = t, m.mode_pr, m.mode_machine
        for f in ("q", "dq", "tau", "kp", "kd"):
            a[k][f] = [getattr(c[i], f) for i in range(N)]
        a[k]["mode"] = [int(c[i].mode) for i in range(N)]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--duration", type=float, default=1800.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    stop_file = os.path.join(args.outdir, "STOP")
    iface = [l.split()[1] for l in subprocess.check_output(["ip", "-o", "-4", "addr", "show"]).decode().splitlines()
             if l.split()[3].startswith("192.168.123.164/")][0]

    qs, qc = queue.SimpleQueue(), queue.SimpleQueue()
    seen = {"s": 0, "c": 0}
    written = {"s": 0, "c": 0}
    qmax = {"s": 0, "c": 0}
    done = threading.Event()

    def on_state(m):
        seen["s"] += 1
        qs.put((time.monotonic_ns(), m))

    def on_cmd(m):
        seen["c"] += 1
        qc.put((time.monotonic_ns(), m))

    def writer():
        fs, fc = open(os.path.join(args.outdir, "state.bin"), "wb"), open(os.path.join(args.outdir, "cmd.bin"), "wb")
        last_hb = 0.0
        while True:
            finished = done.is_set()
            for key, q, f, pack in (("s", qs, fs, pack_state), ("c", qc, fc, pack_cmd)):
                qmax[key] = max(qmax[key], q.qsize())
                batch = []
                while True:
                    try:
                        batch.append(q.get_nowait())
                    except queue.Empty:
                        break
                if batch:
                    pack(batch).tofile(f)
                    written[key] += len(batch)
            now = time.time()
            if now - last_hb > 5.0:
                fs.flush(); fc.flush()
                json.dump({"t": now, "seen": dict(seen), "written": dict(written), "qmax": dict(qmax)},
                          open(os.path.join(args.outdir, "status.json"), "w"))
                last_hb = now
            if finished:
                break
            time.sleep(1.0)
        fs.close(); fc.close()

    ChannelFactoryInitialize(0, iface)
    a = ChannelSubscriber("rt/lowstate", LowState_); a.Init(on_state, 10)
    b = ChannelSubscriber("rt/lowcmd", LowCmd_); b.Init(on_cmd, 10)
    wall0, mono0 = time.time_ns(), time.monotonic_ns()
    th = threading.Thread(target=writer, daemon=True)
    th.start()
    t_end = time.time() + args.duration
    while time.time() < t_end and not os.path.exists(stop_file):
        time.sleep(0.2)
    done.set()
    th.join(timeout=60)
    meta = {"host": socket.gethostname(), "wall_start_ns": wall0, "mono_start_ns": mono0, "seconds": time.time() - wall0 / 1e9,
            "seen": seen, "written": written, "queue_max": qmax, "dtype_state": DT_STATE, "dtype_cmd": DT_CMD,
            "note": "t = time.monotonic_ns() at arrival; read-only recording, nothing was published"}
    json.dump(meta, open(os.path.join(args.outdir, "meta.json"), "w"), indent=1)
    print("done", json.dumps({"seen": seen, "written": written, "queue_max": qmax}))


if __name__ == "__main__":
    main()
