#!/usr/bin/env python3
"""Convert RSL-RL checkpoints between the 5.x format (lab machine, rsl-rl-lib 5.0.1)
and the 3.x format (laptop, rsl-rl-lib 3.1.2).

    python scripts/convert_checkpoint.py in.pt out.pt          # 5.x -> 3.x (default)
    python scripts/convert_checkpoint.py in.pt out.pt --to 5   # 3.x -> 5.x

5.x:  actor_state_dict {distribution.std_param, mlp.*}, critic_state_dict {mlp.*}, optimizer_state_dict, iter
3.x:  model_state_dict {std, actor.*, critic.*},        optimizer_state_dict, iter, infos

Pure key renaming; weights are untouched. Refuses (instead of dropping data) if
the checkpoint holds anything else, e.g. an observation normalizer.
Needs only torch.
"""
import sys

import torch


def convert_to_5(src: str, dst: str) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    if "actor_state_dict" in ckpt:
        print(f"{src}: already in the 5.x format, copying unchanged")
        torch.save(ckpt, dst)
        return
    model = ckpt["model_state_dict"]
    actor = {"distribution.std_param": model["std"]}
    critic = {}
    for key, value in model.items():
        if key == "std":
            continue
        if key.startswith("actor."):
            actor["mlp." + key[len("actor."):]] = value
        elif key.startswith("critic."):
            critic["mlp." + key[len("critic."):]] = value
        else:
            raise SystemExit(f"unexpected key {key!r} -- not converting")
    torch.save(
        {
            "actor_state_dict": actor,
            "critic_state_dict": critic,
            "optimizer_state_dict": ckpt["optimizer_state_dict"],
            "iter": ckpt["iter"],
            "infos": ckpt.get("infos"),
        },
        dst,
    )
    print(f"{src} -> {dst}  (3.x -> 5.x, iter {ckpt['iter']})")


def convert(src: str, dst: str) -> None:
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    if "model_state_dict" in ckpt:
        print(f"{src}: already in the 3.x format, copying unchanged")
        torch.save(ckpt, dst)
        return
    actor, critic = ckpt["actor_state_dict"], ckpt["critic_state_dict"]

    model = {"std": actor["distribution.std_param"]}
    for key, value in actor.items():
        if key == "distribution.std_param":
            continue
        if not key.startswith("mlp."):
            raise SystemExit(f"unexpected actor key {key!r} (normalizer?) -- not converting")
        model["actor." + key[len("mlp."):]] = value
    for key, value in critic.items():
        if not key.startswith("mlp."):
            raise SystemExit(f"unexpected critic key {key!r} (normalizer?) -- not converting")
        model["critic." + key[len("mlp."):]] = value

    torch.save(
        {
            "model_state_dict": model,
            "optimizer_state_dict": ckpt["optimizer_state_dict"],
            "iter": ckpt["iter"],
            "infos": ckpt.get("infos"),
        },
        dst,
    )
    print(f"{src} -> {dst}  (iter {ckpt['iter']}, {len(model)} tensors)")


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if a != "--to" and a != "5"]
    if len(argv) != 2:
        raise SystemExit(__doc__)
    (convert_to_5 if "--to" in sys.argv else convert)(argv[0], argv[1])
