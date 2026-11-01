"""Runs the exported policy network with numpy only (the Jetson's torch build is CPU-only and slow to start; numpy is ~0.3 ms per call).  Python 3.8.

The network is the actor mean: MLP in-256-128-128-out, ELU, no observation normaliser (see scripts/export_policy.py, which writes the .npz).

IMPORTANT: set the thread environment BEFORE numpy is imported anywhere (the controller entry point does it), otherwise BLAS threads
stall on the busy Jetson (measured stalls up to 168 ms):
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"""
import time

import numpy as np


class NumpyPolicy:
    def __init__(self, npz_path, obs_size=None, action_size=None):
        data = np.load(npz_path)
        n = len([k for k in data.files if k.startswith("w")])
        self.ws = [data["w%d" % k].astype(np.float32) for k in range(n)]
        self.bs = [data["b%d" % k].astype(np.float32) for k in range(n)]
        self.obs_size, self.action_size = self.ws[0].shape[1], self.ws[-1].shape[0]
        if obs_size is not None and obs_size != self.obs_size:
            raise ValueError("policy takes %d observations, the contract says %d" % (self.obs_size, obs_size))
        if action_size is not None and action_size != self.action_size:
            raise ValueError("policy gives %d actions, the contract says %d" % (self.action_size, action_size))
        self.last_call_s = 0.0

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        if x.shape[-1] != self.obs_size:
            raise ValueError("observation has %d values, expected %d" % (x.shape[-1], self.obs_size))
        if not np.all(np.isfinite(x)):
            raise ValueError("non-finite value in the observation")
        t0 = time.perf_counter()
        for i, (w, b) in enumerate(zip(self.ws, self.bs)):
            x = x @ w.T + b
            if i < len(self.ws) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0))).astype(np.float32)  # ELU
        self.last_call_s = time.perf_counter() - t0
        if not np.all(np.isfinite(x)):
            raise ValueError("non-finite value in the action")
        return x

    def warmup(self, n=20):
        z = np.zeros(self.obs_size, dtype=np.float32)
        for _ in range(n):
            self(z)
