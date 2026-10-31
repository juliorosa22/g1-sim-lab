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
    ("Stand", "G1WalkStandEnvCfg", "G1WalkStandPPORunnerCfg"),  # v5: v4 + stand still, no stepping (collapsed)
    ("StandRamp", "G1WalkStandRampEnvCfg", "G1WalkStandRampPPORunnerCfg"),  # v6: v5 made trainable, warm-started from v4
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

# v7 / v8: the v1 and v4 recipes on the real plant (hardware session 2026-10-29): Unitree gains, waist roll/pitch locked.
for _name, _mod, _env, _agent in (
    ("V7", "v7_env_cfg", "G1V7EnvCfg", "G1V7PPORunnerCfg"),  # standing, legs only (12 actions), v1's recipe
    ("V8", "v8_env_cfg", "G1V8EnvCfg", "G1V8PPORunnerCfg"),  # walking, legs + waist yaw (13 actions), v4's recipe
    ("V1R", "v7_env_cfg", "G1V1ReplicaEnvCfg", "G1V1ReplicaPPORunnerCfg"),  # v1's recipe on v1's original plant (replication test)
    ("V9a", "v9_env_cfg", "G1V9aEnvCfg", "G1V9aPPORunnerCfg"),  # v8 + base height (posture fix, controlled)
    ("V9b", "v9_env_cfg", "G1V9bEnvCfg", "G1V9bPPORunnerCfg"),  # v8 + base height + whole-body pose reward
    ("V9c", "v9c_env_cfg", "G1V9cEnvCfg", "G1V9cPPORunnerCfg"),  # v9b without the small-command deadband (warm start)
    ("V9d", "v9d_env_cfg", "G1V9dEnvCfg", "G1V9dPPORunnerCfg"),  # v8 + base height + sagittal-only pose, ramped (warm start from v8)
    ("V10", "v10_env_cfg", "G1V10EnvCfg", "G1V10PPORunnerCfg"),  # v7 + base height + pose reward (standing)
):
    for _suffix, _cfg_suffix in (("", ""), ("-Play", "_PLAY")):
        gym.register(
            id=f"G1-Walk-Flat-Inspire-{_name}{_suffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.{_mod}:{_env}{_cfg_suffix}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{_agent}",
            },
        )
