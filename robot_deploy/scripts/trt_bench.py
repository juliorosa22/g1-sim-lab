"""Policy-inference latency benchmark: numpy (CPU) vs TensorRT (GPU). Runs on the robot (Python 3.8) or the laptop (numpy mode).
Never touches the DDS/robot topics: pure compute on random/recorded observation vectors.

  python3 trt_bench.py numpy --npz walk_v4_20261029.npz --testvec walk_v4_20261029_testvec.npz --out res_numpy.json
  python3 trt_bench.py trt   --engine walk_v4.engine     --testvec walk_v4_20261029_testvec.npz --out res_trt.json

Each mode: (1) correctness on the first 1000 test vectors against the torch reference, (2) back-to-back latency (warm-up discarded),
(3) a paced 50 Hz loop (the real regime: the GPU idles between calls and its clock can drop) with wake-up lateness and the GPU clock
before/after. TensorRT mode times host->device copy + execute + device->host copy + stream sync, with pinned host memory, via libcudart/ctypes.
"""
import argparse, ctypes, glob, json, os, sys, time
import numpy as np


def pct(a, p):
    return float(np.percentile(a, p))


def stats(ns):
    us = np.asarray(ns, dtype=np.float64) / 1e3
    return {"n": int(len(us)), "mean_us": float(us.mean()), "p50_us": pct(us, 50), "p99_us": pct(us, 99), "p99.9_us": pct(us, 99.9),
            "max_us": float(us.max()), "over_1ms": int((us > 1000).sum()), "over_5ms": int((us > 5000).sum())}


def gpu_freq():
    out = {}
    for p in glob.glob("/sys/devices/gpu.0/devfreq/*/cur_freq") + glob.glob("/sys/devices/platform/*gpu*/devfreq/*/cur_freq"):
        try:
            out[p.split("/")[-2]] = int(open(p).read().strip()) // 1_000_000
        except Exception:
            pass
    return out


# ------------------------------------------------------------------ numpy backend
class NumpyMLP:
    def __init__(self, npz):
        d = np.load(npz)
        n = len([k for k in d.files if k.startswith("w")])
        self.ws = [np.ascontiguousarray(d["w%d" % i].T, dtype=np.float32) for i in range(n)]   # (in, out)
        self.bs = [d["b%d" % i].astype(np.float32) for i in range(n)]
        self.n_in, self.n_out = self.ws[0].shape[0], self.ws[-1].shape[1]

    def __call__(self, x):
        for i, (w, b) in enumerate(zip(self.ws, self.bs)):
            x = x @ w + b
            if i < len(self.ws) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        return x


# ------------------------------------------------------------------ TensorRT backend (ctypes cudart)
class TRTMLP:
    def __init__(self, engine_path, n_in, n_out):
        import tensorrt as trt
        self.cu = ctypes.CDLL("libcudart.so")
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        rt = trt.Runtime(logger)
        self.engine = rt.deserialize_cuda_engine(open(engine_path, "rb").read())
        assert self.engine is not None, "could not deserialize engine (built with a different TensorRT/GPU?)"
        self.ctx = self.engine.create_execution_context()
        self.n_in, self.n_out = n_in, n_out
        self.stream = ctypes.c_void_p()
        self._ck(self.cu.cudaStreamCreate(ctypes.byref(self.stream)))
        self.d_in, self.d_out = ctypes.c_void_p(), ctypes.c_void_p()
        self._ck(self.cu.cudaMalloc(ctypes.byref(self.d_in), n_in * 4))
        self._ck(self.cu.cudaMalloc(ctypes.byref(self.d_out), n_out * 4))
        h_in, h_out = ctypes.c_void_p(), ctypes.c_void_p()
        self._ck(self.cu.cudaHostAlloc(ctypes.byref(h_in), n_in * 4, 0))
        self._ck(self.cu.cudaHostAlloc(ctypes.byref(h_out), n_out * 4, 0))
        self.h_in_ptr, self.h_out_ptr = h_in.value, h_out.value
        self.h_in = np.frombuffer((ctypes.c_float * n_in).from_address(h_in.value), dtype=np.float32)
        self.h_out = np.frombuffer((ctypes.c_float * n_out).from_address(h_out.value), dtype=np.float32)
        self.bindings = [self.d_in.value, self.d_out.value]
        # binding order must be (obs, action)
        assert self.engine.get_binding_name(0) == "obs" and self.engine.get_binding_name(1) == "action", "unexpected binding order"

    def _ck(self, rc):
        assert rc == 0, "CUDA runtime error %d" % rc

    def __call__(self, x):
        self.h_in[:] = x
        cu = self.cu
        cu.cudaMemcpyAsync(self.d_in, ctypes.c_void_p(self.h_in_ptr), self.n_in * 4, 1, self.stream)      # H2D
        ok = self.ctx.execute_async_v2(self.bindings, self.stream.value)
        cu.cudaMemcpyAsync(ctypes.c_void_p(self.h_out_ptr), self.d_out, self.n_out * 4, 2, self.stream)   # D2H
        cu.cudaStreamSynchronize(self.stream)
        assert ok
        return self.h_out.copy()


def run(f, args, label):
    tv = np.load(args.testvec)
    obs, ref = tv["obs"].astype(np.float32), tv["action"].astype(np.float32)
    res = {"label": label, "platform": os.uname().machine, "python": sys.version.split()[0]}
    n = min(1000, len(obs))
    out = np.stack([f(obs[i]) for i in range(n)]).reshape(n, -1)
    res["max_abs_err_vs_torch"] = float(np.abs(out - ref[:n]).max())
    res["max_abs_ref_action"] = float(np.abs(ref[:n]).max())
    # back-to-back
    x = obs[0].copy()
    for _ in range(500):
        f(x)
    t = []
    for i in range(args.iters):
        x = obs[i % len(obs)]
        a = time.perf_counter_ns(); f(x); t.append(time.perf_counter_ns() - a)
    res["back_to_back"] = stats(t)
    # paced loop at args.hz
    period = int(1e9 / args.hz)
    g0 = gpu_freq()
    lat, late = [], []
    t0 = time.perf_counter_ns() + 20_000_000
    for k in range(int(args.paced_s * args.hz)):
        target = t0 + k * period
        while True:
            now = time.perf_counter_ns()
            if now >= target:
                break
            if target - now > 2_000_000:
                time.sleep((target - now - 1_500_000) / 1e9)
        start = time.perf_counter_ns()
        f(obs[k % len(obs)])
        end = time.perf_counter_ns()
        late.append(start - target); lat.append(end - start)
    res["paced"] = {"hz": args.hz, "seconds": args.paced_s, "latency": stats(lat), "wakeup_lateness": stats(late),
                    "gpu_mhz_before": g0, "gpu_mhz_after": gpu_freq()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["numpy", "trt"])
    ap.add_argument("--npz")
    ap.add_argument("--engine")
    ap.add_argument("--testvec", required=True)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--paced-s", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tv = np.load(args.testvec)
    n_in, n_out = tv["obs"].shape[1], tv["action"].shape[1]
    if args.mode == "numpy":
        f = NumpyMLP(args.npz)
        assert (f.n_in, f.n_out) == (n_in, n_out)
        res = run(f, args, "numpy cpu")
    else:
        f = TRTMLP(args.engine, n_in, n_out)
        res = run(f, args, "tensorrt gpu")
    res["load_avg"] = os.getloadavg()
    json.dump(res, open(args.out, "w"), indent=1)
    b, p = res["back_to_back"], res["paced"]["latency"]
    print("%s | err %.2e | back-to-back mean %.0f us p50 %.0f p99 %.0f p99.9 %.0f max %.0f | paced %g Hz: mean %.0f p99 %.0f max %.0f us, "
          "wakeup lateness p99 %.0f max %.0f us | loadavg %.2f" % (
              res["label"], res["max_abs_err_vs_torch"], b["mean_us"], b["p50_us"], b["p99_us"], b["p99.9_us"], b["max_us"], args.hz,
              p["mean_us"], p["p99_us"], p["max_us"], res["paced"]["wakeup_lateness"]["p99_us"], res["paced"]["wakeup_lateness"]["max_us"], res["load_avg"][0]))


if __name__ == "__main__":
    main()
