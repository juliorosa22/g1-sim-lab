"""Policy selector: v11 (stand, turn in place, creep) or v12 (forward walking).  Python 3.8, numpy only.

The rule is the one evaluated in simulation (scripts/evaluate_switch.py --route forward --enter-walk 0.3 --exit-walk 0.2):
  * the selector looks at the SHAPED forward command vx only (sideways, backward, yaw-only and creeping stay with the stander);
  * walk is chosen after vx stays above `enter` for `debounce_walk` seconds, stand after it stays below `exit` for `debounce_stand` seconds
    (hysteresis band between exit and enter: no switching inside it);
  * the two actions are cross-faded over `blend` seconds, linearly: action = w * walk + (1 - w) * stand.
Both policies read the same 78-value observation; the previous action in it is the BLENDED action that was applied (as in simulation).
While w is 0 (pure stand) the walker need not run, while w is 1 the stander need not run; both run in between and for one tick after a switch.
"""
import numpy as np

STAND, WALK = "stand", "walk"


class PolicySelector:
    def __init__(self, dt=0.02, enter=0.3, exit=0.2, debounce_walk=0.2, debounce_stand=0.3, blend=0.3):
        if not exit < enter:
            raise ValueError("exit must be below enter (hysteresis band)")
        self.dt, self.enter, self.exit = float(dt), float(enter), float(exit)
        self.n_up = max(1, int(round(debounce_walk / dt)))
        self.n_down = max(1, int(round(debounce_stand / dt)))
        self.blend_step = dt / max(float(blend), dt)
        self.reset()

    def reset(self):
        self.mode, self.w, self.up, self.down, self.switches = STAND, 0.0, 0, 0, 0

    def update(self, cmd_shaped):
        """One control step with the shaped command [vx, vy, wz]. Returns (mode, w_walk)."""
        s = float(cmd_shaped[0])
        if self.mode == STAND:
            self.up = self.up + 1 if s > self.enter else 0
            if self.up >= self.n_up:
                self.mode, self.up, self.down = WALK, 0, 0
                self.switches += 1
        else:
            self.down = self.down + 1 if s < self.exit else 0
            if self.down >= self.n_down:
                self.mode, self.up, self.down = STAND, 0, 0
                self.switches += 1
        if self.mode == WALK:
            self.w = min(1.0, self.w + self.blend_step)
        else:
            self.w = max(0.0, self.w - self.blend_step)
        return self.mode, self.w

    def needs(self):
        """Which networks must run this tick: (stand, walk)."""
        return self.w < 1.0, self.w > 0.0 or self.mode == WALK

    def mix(self, a_stand, a_walk):
        """Cross-faded action; a network that does not need to run may be passed as None."""
        if self.w <= 0.0:
            return np.asarray(a_stand, dtype=np.float32)
        if self.w >= 1.0:
            return np.asarray(a_walk, dtype=np.float32)
        return (self.w * np.asarray(a_walk) + (1.0 - self.w) * np.asarray(a_stand)).astype(np.float32)
