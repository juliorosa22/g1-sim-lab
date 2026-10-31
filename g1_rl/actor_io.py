"""Builds a policy's actor network from an RSL-RL checkpoint of either format.

rsl-rl 3.x (laptop) keeps `model_state_dict` with `actor.*` keys; rsl-rl 5.x (lab) keeps `actor_state_dict` with `mlp.*` keys.
Reading the weights directly means one script works on both machines without converting checkpoints, and any observation /
action size (47 -> 12, 78 -> 13, 84 -> 15, ...) is taken from the file. The actors are plain ELU MLPs with no observation
normaliser (the loader refuses a checkpoint that has one).
"""
import torch
import torch.nn as nn


def load_actor(path, device="cuda:0"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in ck:  # rsl-rl 3.x
        sd = {k[len("actor."):]: v for k, v in ck["model_state_dict"].items() if k.startswith("actor.")}
    else:  # rsl-rl 5.x
        sd = {k[len("mlp."):]: v for k, v in ck["actor_state_dict"].items() if k.startswith("mlp.")}
        if any("normalizer" in k and k.endswith(("mean", "std", "_mean", "_var")) for k in ck["actor_state_dict"]):
            raise ValueError(f"{path}: the actor has an observation normaliser, which this loader does not apply")
    idx = sorted({int(k.split(".")[0]) for k in sd})
    layers = []
    for n, i in enumerate(idx):
        w = sd[f"{i}.weight"]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if n < len(idx) - 1:
            layers.append(nn.ELU())
    net = nn.Sequential(*layers)
    net.load_state_dict(sd)  # strict: the Linear layers sit at the even indices, as in rsl-rl's MLP
    return net.eval().to(device)
