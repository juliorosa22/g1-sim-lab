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


@configclass
class G1WalkStandPPORunnerCfg(G1WalkFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_walk_flat_inspire_stand"
        self.max_iterations = 3000


@configclass
class G1WalkStandRampPPORunnerCfg(G1WalkFlatPPORunnerCfg):
    """Warm start from v4: logs/rsl_rl/g1_walk_flat_inspire_stand6/warm_start_v4/model_2999.pt must exist
    (a copy of v4's final checkpoint, in the format of the installed rsl-rl).

    NOTE: Isaac Lab's train.py overwrites `resume` with its CLI value (default False), so the
    fields below are not enough -- launch with `--resume --load_run warm_start_v4 --checkpoint model_2999.pt`.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_walk_flat_inspire_stand6"
        self.max_iterations = 500
        self.resume = True
        self.load_run = "warm_start_v4"
        self.load_checkpoint = "model_2999.pt"


@configclass
class G1V7PPORunnerCfg(G1WalkFlatPPORunnerCfg):
    """v7 = v1's recipe on the real plant: v1's PPO settings (1500 iterations; run it with 2048 envs like v1)."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_v7_stand_real"
        self.max_iterations = 1500


@configclass
class G1V8PPORunnerCfg(G1WalkFlatPPORunnerCfg):
    """v8 = v4's recipe on the real plant: v4's PPO settings (3000 iterations; run it with 8192 envs like v4)."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_v8_walk_real"
        self.max_iterations = 3000


@configclass
class G1V1ReplicaPPORunnerCfg(G1WalkFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_v1_replica"
        self.max_iterations = 1500


@configclass
class G1V9aPPORunnerCfg(G1WalkFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_v9a_walk_height"
        self.max_iterations = 3000


@configclass
class G1V9bPPORunnerCfg(G1WalkFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_v9b_walk_pose"
        self.max_iterations = 3000


@configclass
class G1V10PPORunnerCfg(G1WalkFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_v10_stand_pose"
        self.max_iterations = 1500


@configclass
class G1V9cPPORunnerCfg(G1WalkFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_v9c_walk_slow"
        self.max_iterations = 1500  # warm start from v9b (--resume --load_run ... --checkpoint ...)


@configclass
class G1V9dPPORunnerCfg(G1WalkFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_v9d_walk_sagittal"
        self.max_iterations = 1500  # warm start from v8


@configclass
class G1V9ePPORunnerCfg(G1WalkFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_v9e_walk_standpull"
        self.max_iterations = 1500  # warm start from v8


@configclass
class G1V9fPPORunnerCfg(G1WalkFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "g1_v9f_stand_turn"
        self.max_iterations = 1500  # warm start from v9e
