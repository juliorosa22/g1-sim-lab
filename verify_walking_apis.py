#!/usr/bin/env python3
"""One-off verification for the Isaac Sim APIs g1_walking/base_state.py and
run_walking.py depend on, which this dev container couldn't check directly
(no isaacsim install here -- see base_state.py's module docstring for why it
matters). Run this once on your machine, in the same isaacenv as everything
else, and paste the output back before trusting the walking policy on it:

    python verify_walking_apis.py

It doesn't spawn anything or touch a robot -- just imports the classes and
prints whether these methods exist and what their docstrings say about
frame convention / units.
"""
from isaacsim import SimulationApp

sim = SimulationApp({"headless": True})

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.api import SimulationContext


def show(cls, name):
    print(f"--- {cls.__name__}.{name} ---")
    fn = getattr(cls, name, None)
    if fn is None:
        print("  NOT FOUND on this class")
        print()
        return
    doc = (fn.__doc__ or "").strip()
    print(doc if doc else "  (no docstring)")
    print()


for name in ("get_world_pose", "get_angular_velocity", "get_linear_velocity"):
    show(SingleArticulation, name)

show(SimulationContext, "get_physics_dt")

sim.close()
