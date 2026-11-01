"""Export a trained actor (rsl-rl 3.x or 5.x checkpoint) to ONNX + a dependency-free .npz, with reference test vectors.

Run on the laptop in an environment that has torch, onnx and onnxruntime (e.g. the `isaaclab` conda env):
  ~/miniconda3/envs/isaaclab/bin/python robot_deploy/scripts/export_policy.py CKPT.pt --role walk --version v4 --out robot_deploy/policies

Writes <role>_<version>_<yyyymmdd>.onnx / .npz / _testvec.npz and appends to manifest.json. The actor is the deterministic mean:
MLP in-256-128-128-out, ELU, no observation normaliser (same network as scripts/evaluate_switch.py: load_actor()).
ONNX: fixed batch 1, opset 13 (TensorRT 8.5 on the robot supports up to 17), float32.
The export is verified against the torch actor on random and extreme inputs, through onnxruntime and through the .npz weights in numpy.
"""
import argparse, datetime, hashlib, json, os, sys
import numpy as np
import torch
import torch.nn as nn


def load_actor(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" in ck:  # rsl-rl 3.x
        sd = {k[len("actor."):]: v for k, v in ck["model_state_dict"].items() if k.startswith("actor.")}
        fmt = "rsl-rl 3.x"
    else:  # rsl-rl 5.x
        sd = {k[len("mlp."):]: v for k, v in ck["actor_state_dict"].items() if k.startswith("mlp.")}
        fmt = "rsl-rl 5.x"
    idx = sorted({int(k.split(".")[0]) for k in sd})
    layers = []
    for n, i in enumerate(idx):
        w = sd[f"{i}.weight"]
        layers.append(nn.Linear(w.shape[1], w.shape[0]))
        if n < len(idx) - 1:
            layers.append(nn.ELU())
    net = nn.Sequential(*layers)
    net.load_state_dict(sd)
    return net.eval(), sd, idx, fmt, ck.get("iter")


def np_forward(ws, bs, x):
    """Reference numpy forward: x (n, in) -> (n, out). ELU between layers."""
    for i, (w, b) in enumerate(zip(ws, bs)):
        x = x @ w.T + b
        if i < len(ws) - 1:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
    return x


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--role", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", default="robot_deploy/policies")
    ap.add_argument("--n-test", type=int, default=2000)
    args = ap.parse_args()

    net, sd, idx, fmt, it = load_actor(args.ckpt)
    n_in, n_out = sd[f"{idx[0]}.weight"].shape[1], sd[f"{idx[-1]}.weight"].shape[0]
    stamp = datetime.date.today().strftime("%Y%m%d")
    base = os.path.join(args.out, f"{args.role}_{args.version}_{stamp}")
    os.makedirs(args.out, exist_ok=True)

    # ONNX
    dummy = torch.zeros(1, n_in)
    torch.onnx.export(net, dummy, base + ".onnx", input_names=["obs"], output_names=["action"], opset_version=13,
                      dynamo=False, do_constant_folding=True)
    # numpy weights
    ws = [sd[f"{i}.weight"].numpy().astype(np.float32) for i in idx]
    bs = [sd[f"{i}.bias"].numpy().astype(np.float32) for i in idx]
    np.savez(base + ".npz", **{f"w{k}": w for k, w in enumerate(ws)}, **{f"b{k}": b for k, b in enumerate(bs)})
    # test vectors: typical-scale, wide, and extreme inputs
    rng = np.random.default_rng(0)
    typ = rng.normal(0, 0.5, (args.n_test, n_in))
    wide = rng.normal(0, 2.0, (args.n_test // 2, n_in))
    ext = rng.choice([-5.0, 5.0], size=(args.n_test // 10, n_in)) * rng.random((args.n_test // 10, n_in))
    x = np.concatenate([typ, wide, ext]).astype(np.float32)
    with torch.no_grad():
        y = net(torch.from_numpy(x)).numpy()
    np.savez(base + "_testvec.npz", obs=x, action=y)

    # verification
    import onnxruntime as ort
    sess = ort.InferenceSession(base + ".onnx", providers=["CPUExecutionProvider"])
    y_ort = np.concatenate([sess.run(None, {"obs": x[i:i + 1]})[0] for i in range(0, len(x), 1)])
    y_np = np_forward(ws, bs, x)
    rep = {"max_abs_err_onnxruntime_vs_torch": float(np.abs(y_ort - y).max()), "max_abs_err_numpy_vs_torch": float(np.abs(y_np - y).max()),
           "max_abs_action": float(np.abs(y).max()), "n_test_vectors": int(len(x))}
    entry = {"file": os.path.basename(base), "role": args.role, "version": args.version, "checkpoint": args.ckpt, "checkpoint_sha256": sha(args.ckpt),
             "checkpoint_format": fmt, "checkpoint_iter": it, "obs_size": int(n_in), "action_size": int(n_out),
             "layers": [list(w.shape) for w in ws], "activation": "ELU", "exported": datetime.datetime.now().isoformat(timespec="seconds"),
             "onnx_sha256": sha(base + ".onnx"), "npz_sha256": sha(base + ".npz"), "verification": rep,
             "torch": torch.__version__, "onnx_opset": 13}
    mpath = os.path.join(args.out, "manifest.json")
    m = json.load(open(mpath)) if os.path.exists(mpath) else {"policies": []}
    m["policies"] = [p for p in m["policies"] if p["file"] != entry["file"]] + [entry]
    json.dump(m, open(mpath, "w"), indent=1)
    print(json.dumps(entry, indent=1))
    bad = rep["max_abs_err_onnxruntime_vs_torch"] > 1e-4 or rep["max_abs_err_numpy_vs_torch"] > 1e-4
    if bad:
        sys.exit("EXPORT VERIFICATION FAILED")


if __name__ == "__main__":
    main()
