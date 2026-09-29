# motion.pt -- provenance

Vendored unmodified from Unitree Robotics' `unitree_rl_gym` repository:

    https://github.com/unitreerobotics/unitree_rl_gym
    deploy/pre_train/g1/motion.pt

A TorchScript-exported (`torch.jit.save`) LSTM locomotion policy
(internal module name `policy_lstm_1`), 12 actions (legs only), 47
observations. Trained in Isaac Gym; loadable anywhere with `torch.jit.load`
at inference time, no Isaac Gym dependency required. See
`g1_walking/config.py` and `g1_walking/policy.py` for how this codebase
loads and runs it, and `g1_walking/observation.py` for the exact
observation contract it expects.

`unitree_rl_gym` is BSD-3-Clause licensed
(Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. / Unitree
Robotics). Redistributing this file here, with this attribution, satisfies
that license's requirements.
