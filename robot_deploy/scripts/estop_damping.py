"""Emergency damping: publish kp=0, kd>0, tau=0 on rt/lowcmd for all 29 G1 joints.

Runs ON the robot (piped through ssh by robot_estop.sh; nothing is copied there). Python 3.8.
It only reads rt/lowstate (for mode_machine) and publishes rt/lowcmd. It does NOT call
MotionSwitcherClient / ReleaseMode or touch any robot mode.

  python3 - [--hold SECONDS] [--kd-scale X] [--dry-run]

--dry-run builds the command and prints it, publishes nothing.
"""
import argparse
import subprocess
import sys
import time

sys.path.insert(0, "/home/unitree/unitree_sdk2_python")

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

NUM_MOTOR = 29
ROBOT_IP = "192.168.123.164"
# Same kd as the policies are trained with (configs/g1_walk_stand.yaml sim_kd), so no new gain numbers.
# Copied here on purpose: this script must not depend on parsing a config during an emergency.
KD = [2, 2, 2, 4, 2, 2,  2, 2, 2, 4, 2, 2,   # legs
      5, 5, 5,                                # waist
      2, 2, 2, 2, 2, 2, 2,                    # left arm
      2, 2, 2, 2, 2, 2, 2]                    # right arm


def robot_interface():
    for line in subprocess.check_output(["ip", "-o", "-4", "addr", "show"]).decode().splitlines():
        parts = line.split()
        if parts[3].startswith(ROBOT_IP + "/"):
            return parts[1]
    raise RuntimeError("no interface carries " + ROBOT_IP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=float, default=10.0)
    ap.add_argument("--kd-scale", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--joints", default="all", help="comma list of motor indices to damp (default all); the rest get "
                    "kp=kd=tau=0, mode=1, the same as the onboard zero-torque stream")
    args = ap.parse_args()
    damp = set(range(NUM_MOTOR)) if args.joints == "all" else set(int(j) for j in args.joints.split(","))
    assert all(0 <= j < NUM_MOTOR for j in damp), "joint index out of range"

    ChannelFactoryInitialize(0, robot_interface())

    state = {}

    def on_state(msg):
        state["mode_machine"] = msg.mode_machine

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)
    t0 = time.time()
    while "mode_machine" not in state and time.time() - t0 < 2.0:
        time.sleep(0.01)
    if "mode_machine" not in state:
        print("ESTOP DAMPING NOT SENT: no rt/lowstate within 2 s, cannot fill mode_machine. USE THE REMOTE E-STOP.")
        return 2

    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = 0  # PR, as in Unitree's low-level example
    cmd.mode_machine = state["mode_machine"]
    for i in range(NUM_MOTOR):
        m = cmd.motor_cmd[i]
        m.mode = 1
        m.q = 0.0
        m.dq = 0.0
        m.kp = 0.0
        m.kd = float(KD[i]) * args.kd_scale if i in damp else 0.0
        m.tau = 0.0
    crc = CRC()
    cmd.crc = crc.Crc(cmd)

    print("mode_machine=%d kp=0 tau=0 kd(leg/waist/arm)=%s/%s/%s hold=%.1fs" % (
        cmd.mode_machine, [cmd.motor_cmd[i].kd for i in range(6)], cmd.motor_cmd[12].kd,
        cmd.motor_cmd[15].kd, args.hold))
    if args.dry_run:
        print("DRY RUN: nothing published")
        return 0

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    t0 = time.time()
    n = 0
    while time.time() - t0 < args.hold:
        pub.Write(cmd)
        n += 1
        time.sleep(0.002)
    print("damping published: %d messages over %.1f s. Publishing has now STOPPED." % (n, args.hold))
    return 0


if __name__ == "__main__":
    sys.exit(main())
