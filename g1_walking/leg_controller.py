"""Applies the walking policy's 12-DOF leg targets to the articulation while
holding every other joint (waist, arms, hands) at the standing pose from
g1_sim.config.StandingPoseConfig.

The policy only ever knew about 12 leg joints (it was trained legs-only), so
the rest of the body needs its own controller -- the standing controller's
gains/targets are the best-validated "just hold still" reference we already
have for them (see g1_sim/standing.py, confirmed to hold the full 53-DOF
robot upright earlier this project).
"""
import numpy as np

from g1_sim.config import StandingPoseConfig

from .config import LEG_JOINT_NAMES, WalkingPolicyConfig


class LegController:
    def __init__(self, articulation, walking_config: WalkingPolicyConfig, standing_config: StandingPoseConfig):
        self.articulation = articulation
        self.walking_config = walking_config
        self.standing_config = standing_config

        self.names = list(articulation.dof_names)
        # leg_indices[i] is this articulation's dof index for LEG_JOINT_NAMES[i].
        self.leg_indices = [self.names.index(n) for n in LEG_JOINT_NAMES]

    def install_gains(self):
        """Sets per-joint PD gains once: the walking policy's own gains on
        the 12 leg joints, standing gains on everything else that has one,
        and whatever the USD/articulation already had for the rest -- same
        gap-filling approach as StandingController.apply().
        """
        controller = self.articulation.get_articulation_controller()
        current_kps, current_kds = controller.get_gains()
        kps = np.array(current_kps, dtype=np.float64)
        kds = np.array(current_kds, dtype=np.float64)

        standing_gains = self.standing_config.gains
        for i, name in enumerate(self.names):
            if name in standing_gains:
                kps[i], kds[i] = standing_gains[name]

        wcfg = self.walking_config
        for leg_pos, dof_i in enumerate(self.leg_indices):
            kps[dof_i] = wcfg.kps[leg_pos]
            kds[dof_i] = wcfg.kds[leg_pos]

        controller.set_gains(kps=kps, kds=kds)

    def read_leg_state(self):
        """Returns (leg_pos, leg_vel), each length 12, in LEG_JOINT_NAMES order."""
        state = self.articulation.get_joints_state()
        leg_pos = np.array([state.positions[i] for i in self.leg_indices])
        leg_vel = np.array([state.velocities[i] for i in self.leg_indices])
        return leg_pos, leg_vel

    def apply(self, leg_target_positions: np.ndarray):
        """Commands new position targets for the 12 leg joints; every other
        joint is re-commanded to its standing-pose target (0.0 for anything
        not in joint_pos_overrides) so it keeps holding still under its own
        installed gains.
        """
        from isaacsim.core.utils.types import ArticulationAction

        overrides = self.standing_config.joint_pos_overrides
        full_targets = np.array([overrides.get(n, 0.0) for n in self.names], dtype=np.float64)
        for leg_pos, dof_i in enumerate(self.leg_indices):
            full_targets[dof_i] = leg_target_positions[leg_pos]

        controller = self.articulation.get_articulation_controller()
        controller.apply_action(ArticulationAction(joint_positions=full_targets))
