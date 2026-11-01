"""Velocity-command shaping: what the planner asks for -> what the policy sees.  Python 3.8, numpy only.

Policies v11 / v12 were trained on a command that is (1) dead-banded at 0.05 and (2) ramped toward its target at 1 m/s^2 (vector) and
2 rad/s^2 (yaw), because the deployed controller does the same to the planner's output. Training did exactly this inside
g1_rl/mdp/commands.py: MixedVelocityCommand._update_command; tests/contract_test.py compares this class with the simulation step by step.
"""
import numpy as np


class CommandShaper:
    def __init__(self, dt=0.02, deadband=0.05, ramp_lin=1.0, ramp_yaw=2.0):
        self.dt, self.deadband, self.ramp_lin, self.ramp_yaw = float(dt), float(deadband), float(ramp_lin), float(ramp_yaw)
        self.cur = np.zeros(3, dtype=np.float64)

    def reset(self):
        """A new run starts from rest (the simulation does the same at every episode start)."""
        self.cur[:] = 0.0

    def apply_deadband(self, request):
        goal = np.array(request, dtype=np.float64)
        if np.hypot(goal[0], goal[1]) < self.deadband:
            goal[0] = goal[1] = 0.0
        if abs(goal[2]) < self.deadband:
            goal[2] = 0.0
        return goal

    def step(self, request):
        """One control step: returns the shaped command [vx, vy, wz] (float32, 3)."""
        goal = self.apply_deadband(request)
        d = goal[:2] - self.cur[:2]
        n = max(float(np.hypot(d[0], d[1])), 1e-9)
        if self.ramp_lin > 0.0:
            d = d * min(1.0, self.ramp_lin * self.dt / n)
        self.cur[:2] += d
        dw = goal[2] - self.cur[2]
        self.cur[2] += float(np.clip(dw, -self.ramp_yaw * self.dt, self.ramp_yaw * self.dt)) if self.ramp_yaw > 0.0 else dw
        return self.cur.astype(np.float32)
