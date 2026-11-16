"""Wider domain randomisation for v11b / v12b, from the plant-mismatch sweep (logs/eval/plant, docs/sim_update_for_hardware_20261116.md).

What the sweep found (fall % over the in-range command box, v11 / v12): floor friction x0.4 9 / 2 and x0.2 55 / 32; leg stiffness alone x0.6 18 / 26,
x1.4 18 / 10; damping x1.6 1 / 15; mass x1.3 1 / 4; torque x0.6 4 / 6. Everything else (delay to 30 ms, COM +-4 cm, joint friction to 4 N*m) was harmless.
Trained before: friction 0.3-1.2 (static) / 0.25-1.0 (dynamic), torso mass +-5 kg, torso COM +-3 cm, leg + waist-yaw stiffness and damping x0.8-1.2
(drawn independently), armature x0.8-1.5, delay 0-20 ms.

Now: friction down to 0.15, stiffness and damping x0.6-1.4 (independent), ALL link masses x0.8-1.3 on top of the torso +-5 kg, torso COM +-5 cm,
torque limit x0.7-1.0. Delay, armature and the rest are unchanged.
"""
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from . import mdp

FRICTION_STATIC, FRICTION_DYNAMIC = (0.15, 1.2), (0.12, 1.0)
GAIN_RANGE = (0.6, 1.4)
LINK_MASS_SCALE = (0.8, 1.3)
COM_RANGE = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.03, 0.03)}
TORQUE_SCALE = (0.7, 1.0)


def widen_randomization(cfg):
    ev = cfg.events
    ev.physics_material.params["static_friction_range"] = FRICTION_STATIC
    ev.physics_material.params["dynamic_friction_range"] = FRICTION_DYNAMIC
    ev.actuator_gains.params["stiffness_distribution_params"] = GAIN_RANGE
    ev.actuator_gains.params["damping_distribution_params"] = GAIN_RANGE
    ev.torso_com.params["com_range"] = dict(COM_RANGE)
    ev.link_mass_scale = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!torso_link$).*"),  # the torso keeps its own +-5 kg (this event would overwrite it)
            "mass_distribution_params": LINK_MASS_SCALE,
            "operation": "scale",
        },
    )
    ev.torque_limit_scale = EventTerm(
        func=mdp.randomize_effort_limit_scale,
        mode="startup",
        params={"scale_range": TORQUE_SCALE, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
