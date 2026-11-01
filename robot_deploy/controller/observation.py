"""Builds the policy's observation from the robot's state, and turns the policy's action into joint targets.  Python 3.8, numpy only.

Contract: configs/g1_v11_stand_turn.json (the same 78 -> 13 interface as v8 / v9 / v11 / v12). Every number here comes from that file; nothing is
hard-coded. tests/contract_test.py replays recorded simulation states through this module and requires the observation and the targets to match
what Isaac Lab produced.

Observation, 78 values:
    gyro x 0.25 (3) | projected gravity (3) | command x [2, 2, 0.25] (3) | joint pos - default, 27 joints (27) |
    joint vel x 0.05 (27) | last action, raw (13) | gait clock [sin, cos] (2)
The 27 joints are 12 legs, waist yaw, then the 14 arm joints; the action is the first 13 of them (legs + waist yaw).
Frames: gyro and gravity in the pelvis (IMU) frame, gravity = unit vector pointing DOWN in that frame ([0, 0, -1] when upright).
Do not remove a gyro bias: training added a per-run bias of +-0.005 rad/s on purpose.
"""
import json
import math

import numpy as np


def load_config(path):
    with open(path) as f:
        return json.load(f)


def projected_gravity_from_quat(quat_wxyz):
    """Gravity direction in the body frame from the IMU orientation quaternion (w, x, y, z), body -> world: g_body = R^T [0, 0, -1]."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    # third row of R (world z axis expressed in the body frame), negated
    return np.array([-2.0 * (x * z - w * y), -2.0 * (y * z + w * x), -(1.0 - 2.0 * (x * x + y * y))], dtype=np.float32)


class ObservationBuilder:
    def __init__(self, config):
        self.cfg = config
        joints = config["joints"]
        self.names = [j["name"] for j in joints]
        self.motors = np.array([j["motor"] for j in joints], dtype=np.int64)  # index of each policy joint in the 29-motor arrays
        self.default = np.array([j["default"] for j in joints], dtype=np.float32)
        self.n_joints = len(joints)
        self.n_act = int(config["action_size"])
        self.act_scale = np.array([j["action_scale"] for j in joints[:self.n_act]], dtype=np.float32)
        s = config["obs_scales"]
        self.gyro_scale = float(s["gyro"])
        self.cmd_scale = np.array(s["command"], dtype=np.float32)
        self.vel_scale = float(s["joint_vel"])
        self.period = float(config["gait_period_s"])
        self.obs_size = int(config["obs_size"])
        expected = sum(n for _, n in config["obs_layout"])
        assert expected == self.obs_size, "obs_layout does not add up to obs_size"
        assert self.obs_size == 3 + 3 + 3 + 2 * self.n_joints + self.n_act + 2, "observation size does not match the joint counts"

    # ---- state -> policy order
    def gather(self, motor_values):
        """A 29-motor array (rt/lowstate order) -> the 27 policy joints."""
        return np.asarray(motor_values, dtype=np.float32)[self.motors]

    def gait_clock(self, t):
        """t = seconds since the policy started (0 at its first step); the clock always runs, also at zero command."""
        angle = 2.0 * math.pi * ((t % self.period) / self.period)
        return np.array([math.sin(angle), math.cos(angle)], dtype=np.float32)

    def build(self, gyro, gravity, command, q, dq, last_action, t):
        """gyro (3, rad/s), gravity (3), command (3, already deadbanded and ramped: commands.CommandShaper), q and dq (27, absolute),
        last_action (13, the raw previous policy output), t (s). Returns float32 (78,)."""
        q = np.asarray(q, dtype=np.float32)
        dq = np.asarray(dq, dtype=np.float32)
        assert q.shape == (self.n_joints,) and dq.shape == (self.n_joints,), "joint arrays must hold the 27 policy joints"
        last_action = np.asarray(last_action, dtype=np.float32)
        assert last_action.shape == (self.n_act,)
        obs = np.concatenate([
            np.asarray(gyro, dtype=np.float32) * self.gyro_scale,
            np.asarray(gravity, dtype=np.float32),
            np.asarray(command, dtype=np.float32) * self.cmd_scale,
            q - self.default,
            dq * self.vel_scale,
            last_action,
            self.gait_clock(t),
        ]).astype(np.float32)
        assert obs.shape == (self.obs_size,)
        return obs

    # ---- action -> targets
    def action_to_targets(self, action):
        """Raw policy output (13) -> absolute position targets for the 13 controlled joints: default + scale * action."""
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (self.n_act,)
        return (self.default[:self.n_act] + self.act_scale * action).astype(np.float32)

    def hold_targets(self):
        """Targets for the joints the policy does not drive (the 14 arm joints): their default pose."""
        return self.default[self.n_act:].copy()

    def gains(self):
        """(kp, kd) per policy joint, in policy order (what the policy was trained with)."""
        j = self.cfg["joints"]
        return np.array([x["kp"] for x in j], dtype=np.float32), np.array([x["kd"] for x in j], dtype=np.float32)
