from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1WalkFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Stock G1-flat PPO hyperparameters, with an asymmetric critic."""

    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "g1_walk_flat_inspire"
    # actor sees only "policy" (proprioceptive, deployable); critic sees "critic" (privileged).
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1WalkWaistPPORunnerCfg(G1WalkFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_walk_flat_inspire_waist"
        self.max_iterations = 3000


@configclass
class G1WalkArmsPPORunnerCfg(G1WalkFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_walk_flat_inspire_arms"
        self.max_iterations = 3000
