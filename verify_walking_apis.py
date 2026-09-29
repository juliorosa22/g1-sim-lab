#!/usr/bin/env python3
"""One-off verification for the Isaac Sim APIs g1_walking/base_state.py and
run_walking.py depend on, which this dev container couldn't check directly
(no isaacsim install here -- see base_state.py's module docstring for why it
matters). Run this once on your machine, in the same isaacenv as everything
else, and paste the output back before trusting the walking policy on it:

    python verify_walking_apis.py

It doesn't spawn a robot, but it DOES construct a real SimulationContext (so
get_physics_dt() below returns an actual measured value, not documentation
boilerplate -- the previous version of this script only printed the
docstring's own `>>> ...` example output, which is illustrative text, not a
live reading, and got mistaken for one).
"""
import inspect

from isaacsim import SimulationApp

sim = SimulationApp({"headless": True})

from isaacsim.core.api import SimulationContext
from isaacsim.core.prims import SingleArticulation


def show_doc(cls, name):
    print(f"--- {cls.__name__}.{name} (docstring) ---")
    fn = getattr(cls, name, None)
    if fn is None:
        print("  NOT FOUND on this class")
        print()
        return
    doc = (fn.__doc__ or "").strip()
    print(doc if doc else "  (no docstring)")
    print()


def show_source(cls, name):
    print(f"--- {cls.__name__}.{name} (source, for frame-convention checking) ---")
    fn = getattr(cls, name, None)
    if fn is None:
        print("  NOT FOUND on this class")
        print()
        return
    try:
        print(f"  file: {inspect.getsourcefile(fn)}")
        print(inspect.getsource(fn))
    except (OSError, TypeError) as exc:
        print(f"  could not read source: {exc}")
    print()


for name in ("get_world_pose", "get_angular_velocity", "get_linear_velocity"):
    show_doc(SingleArticulation, name)

# get_angular_velocity's docstring doesn't state world vs. body frame -- the
# implementation itself (likely a one-line delegation into an
# ArticulationView/PhysX tensor call) is the only way to actually confirm it.
show_source(SingleArticulation, "get_angular_velocity")

# Real measured physics dt, from an actual SimulationContext -- not the
# docstring's example value.
sim_context = SimulationContext()
print(f"--- SimulationContext().get_physics_dt() -- ACTUAL measured value ---")
print(f"  {sim_context.get_physics_dt()}")

sim.close()
