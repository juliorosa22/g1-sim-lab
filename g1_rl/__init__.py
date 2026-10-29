"""Isaac Lab task package: G1 (Inspire) velocity-tracking walking.

Importing this package registers the gym ids. Entry points are strings, so
nothing from isaaclab is imported until a task is actually made -- same
pattern as isaaclab_tasks, and safe to import before the app launches.
"""
import gymnasium as gym

from . import agents

gym.register(
    id="G1-Walk-Flat-Inspire-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1WalkFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkFlatPPORunnerCfg",
    },
)

gym.register(
    id="G1-Walk-Flat-Inspire-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1WalkFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1WalkFlatPPORunnerCfg",
    },
)

for _name, _env, _agent in (
    ("Waist", "G1WalkWaistEnvCfg", "G1WalkWaistPPORunnerCfg"),  # v3: legs + waist
    ("Arms", "G1WalkArmsEnvCfg", "G1WalkArmsPPORunnerCfg"),  # v4: + arm-randomised, all 29 joints observed
    ("Stand", "G1WalkStandEnvCfg", "G1WalkStandPPORunnerCfg"),  # v5: v4 + stand still, no stepping
):
    for _suffix, _cfg_suffix in (("", ""), ("-Play", "_PLAY")):
        gym.register(
            id=f"G1-Walk-Flat-Inspire-{_name}{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.body_env_cfg:{_env}{_cfg_suffix}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_agent}",
            },
        )
