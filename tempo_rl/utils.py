"""Small numerical helpers: schedules, reward scaling, the HL-Gauss value head and
colored (pink) exploration noise."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


class LinearSchedule:
    """``f(progress_remaining)``: ``start`` at the beginning of training, ``end`` at the end.

    JSON-serialisable (saved with the model), unlike a lambda.
    """

    def __init__(self, start: float, end: float):
        self.start, self.end = float(start), float(end)

    def __call__(self, progress_remaining: float) -> float:
        return self.end + (self.start - self.end) * progress_remaining

    def to_json(self) -> dict:
        return {"linear_schedule": [self.start, self.end]}

    def __repr__(self) -> str:
        return f"LinearSchedule({self.start}, {self.end})"


def schedule_from_json(value):
    if isinstance(value, dict) and "linear_schedule" in value:
        return LinearSchedule(*value["linear_schedule"])
    return value


def schedule_to_json(value):
    if isinstance(value, LinearSchedule):
        return value.to_json()
    if isinstance(value, np.generic):          # np.float32(3e-4) is not a Python float
        value = value.item()
    return value if isinstance(value, (int, float, str)) else None


class RewardScaler:
    """Scale-only reward normalisation from running discounted-return statistics.

    ``scale = max(std(G), max|G| / v_max)`` where ``G`` is the per-environment
    discounted return, so that scaled returns stay within ``[-v_max, v_max]``
    (the SimbaV2 rule). Rewards are never centred, which would change the
    optimal policy of episodic tasks.
    """

    def __init__(self, n_envs: int, gamma: float, v_max: float = 5.0, eps: float = 1e-8):
        self.gamma, self.v_max, self.eps = gamma, v_max, eps
        self.ret = np.zeros(n_envs, dtype=np.float64)
        self.count, self.mean, self.m2 = 0, 0.0, 0.0
        self.max_abs = 0.0
        #: multiplier of the scale (research option, Screen 7b): 1.0 = the published rule
        self.kappa = 1.0

    def update(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        self.ret = self.ret * self.gamma + np.asarray(rewards, dtype=np.float64)
        for g in self.ret:
            self.count += 1
            d = g - self.mean
            self.mean += d / self.count
            self.m2 += d * (g - self.mean)
        self.max_abs = max(self.max_abs, float(np.abs(self.ret).max()))
        self.ret[np.asarray(dones, dtype=bool)] = 0.0

    @property
    def scale(self) -> float:
        if self.count < 2 or self.max_abs == 0.0:   # no reward signal yet (sparse tasks)
            return 1.0
        std = math.sqrt(self.m2 / (self.count - 1))
        return max(std, self.max_abs / self.v_max, self.eps) * self.kappa

    def state(self) -> dict:
        return dict(ret=self.ret.tolist(), count=self.count, mean=self.mean, m2=self.m2,
                    max_abs=self.max_abs, v_max=self.v_max, gamma=self.gamma, kappa=self.kappa)

    def load_state(self, s: dict) -> None:
        self.ret = np.asarray(s["ret"], dtype=np.float64)
        self.count, self.mean, self.m2 = int(s["count"]), float(s["mean"]), float(s["m2"])
        self.max_abs = float(s["max_abs"])
        self.kappa = float(s.get("kappa", 1.0))


class HLGauss:
    """Histogram loss with Gaussian smoothing (Imani & White 2018; Farebrother et al. 2024).

    The critic outputs logits over ``n_bins`` evenly spaced bins on
    ``[-v_max, v_max]``; its value is the expectation under the softmax. A scalar
    target ``y`` becomes the probability mass a Gaussian ``N(y, sigma^2)`` puts
    on each bin, with ``sigma = sigma_ratio * bin_width``.
    """

    def __init__(self, n_bins: int = 101, v_max: float = 5.0, sigma_ratio: float = 0.75,
                 device="cpu"):
        self.n_bins, self.v_max = n_bins, v_max
        self.edges = torch.linspace(-v_max, v_max, n_bins + 1, device=device)
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2
        self.sigma = sigma_ratio * (2 * v_max / n_bins)

    def to(self, device) -> "HLGauss":
        self.edges, self.centers = self.edges.to(device), self.centers.to(device)
        return self

    def value(self, logits: torch.Tensor) -> torch.Tensor:
        return (F.softmax(logits, -1) * self.centers).sum(-1)

    def target_probs(self, y: torch.Tensor) -> torch.Tensor:
        y = y.clamp(-self.v_max, self.v_max).unsqueeze(-1)
        cdf = torch.special.erf((self.edges - y) / (math.sqrt(2.0) * self.sigma))
        mass = cdf[..., 1:] - cdf[..., :-1]
        return mass / mass.sum(-1, keepdim=True).clamp_min(1e-12)

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cross-entropy, averaged over every leading dimension."""
        return -(self.target_probs(y) * F.log_softmax(logits, -1)).sum(-1).mean()


class ColoredNoise:
    """Per-environment, per-dimension ``1/f^beta`` Gaussian noise sequences.

    ``beta = 0`` is white noise, ``beta = 1`` pink noise (Eberhard et al., 2023, "Pink
    Noise Is All You Need"). Each sequence has unit marginal variance; a fresh sequence is
    drawn at every episode start, so consecutive actions within an episode share
    low-frequency components and explore coherently. TEMPO uses ``beta = 1`` with
    ``length`` = the episode time limit (``exploration_noise="pink"``, the default).
    """

    def __init__(self, n_envs: int, dim: int, beta: float = 1.0, length: int = 1000,
                 seed: int | None = None):
        self.n_envs, self.dim, self.beta, self.length = n_envs, dim, beta, max(int(length), 16)
        self.rng = np.random.default_rng(seed)
        self.seqs = np.zeros((n_envs, self.length, dim), dtype=np.float32)
        self.t = np.zeros(n_envs, dtype=np.int64)
        for e in range(n_envs):
            self.reset(e)

    def _sample(self) -> np.ndarray:
        n = self.length
        f = np.fft.rfftfreq(n)
        f[0] = f[1]                                     # no infinite power at DC
        amp = f ** (-self.beta / 2.0)
        spec = (self.rng.standard_normal((len(f), self.dim))
                + 1j * self.rng.standard_normal((len(f), self.dim))) * amp[:, None]
        x = np.fft.irfft(spec, n=n, axis=0)
        x = (x - x.mean(0)) / x.std(0).clip(min=1e-8)
        return x.astype(np.float32)

    def reset(self, env_index: int) -> None:
        self.seqs[env_index] = self._sample()
        self.t[env_index] = 0

    def next(self) -> np.ndarray:
        """``[n_envs, dim]`` noise for the current step of every environment."""
        out = self.seqs[np.arange(self.n_envs), self.t % self.length]
        self.t += 1
        return out
