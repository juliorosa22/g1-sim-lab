"""Curriculum terms specific to this task."""
import torch

from isaaclab.envs import ManagerBasedRLEnv


def ramp_stand_rewards(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    term_weights: dict,
    start_step: int,
    end_step: int,
    anneal: dict | None = None,
) -> float:
    """Phases the stand-still rewards in so they cannot dominate early training.

    Every term in `term_weights` (name -> final weight) is scaled by a factor that is 0 up to
    `start_step`, rises linearly, and is 1 from `end_step` on (steps are env steps: one PPO
    iteration is num_steps_per_env = 24 of them). `anneal` (name -> (param, start, end)) moves a
    reward parameter, e.g. the width of the still-bonus kernel, along the same ramp.

    v5 switched the stand penalties on at full strength from step 0; net reward per second
    went negative, falling over (a -4 penalty) became cheaper than staying up, and training
    collapsed. Called on every reset; idempotent.
    """
    f = (env.common_step_counter - start_step) / max(end_step - start_step, 1)
    f = float(min(max(f, 0.0), 1.0))
    for name, weight in term_weights.items():
        cfg = env.reward_manager.get_term_cfg(name)
        cfg.weight = weight * f
        env.reward_manager.set_term_cfg(name, cfg)
    for name, (param, v0, v1) in (anneal or {}).items():
        cfg = env.reward_manager.get_term_cfg(name)
        cfg.params[param] = v0 + (v1 - v0) * f
        env.reward_manager.set_term_cfg(name, cfg)
    return f
