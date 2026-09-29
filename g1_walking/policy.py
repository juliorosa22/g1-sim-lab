"""Loads and runs Unitree's pretrained TorchScript locomotion policy.

torch.jit.load() produces a self-contained module (architecture + weights
both embedded in the .pt archive), so no model class needs reconstructing
here. Isaac Gym (the framework the policy was originally trained in) is not
required at inference time -- only PyTorch. See checkpoint/README.md for
provenance and the internal module name ("policy_lstm_1", an LSTM actor).
"""
import numpy as np


class WalkingPolicy:
    def __init__(self, checkpoint_path: str):
        import torch  # deferred: only needed once we actually load a policy

        self._torch = torch
        self.model = torch.jit.load(checkpoint_path)
        self.model.eval()

    def act(self, obs: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
            action = self.model(obs_tensor).detach().numpy().squeeze()
        return action
