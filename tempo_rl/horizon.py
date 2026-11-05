"""Horizon audition: choose the planning horizon of a task from its own undiscounted returns.

The automatic discount ``gamma_A = clip(1 - 5/T, 0.95, 0.995)`` plans over a fifth of an
episode. Most tasks prefer it (a long horizon makes falling look catastrophic, so agents on
Hopper-like tasks stay standing), but some are scored on structure that only pays off over the
whole episode (Swimmer's swimming gait). A single discount cannot serve both, so TEMPO can
*audition* a second horizon:

* a shadow learner with ``gamma_L = 1 - 1/T`` (effective horizon = the episode) trains on the
  same replay buffer, with its own networks, optimisers, temperature, novelty estimator and a
  reward scale made consistent with bootstrapping through time limits
  (:func:`~tempo_rl.env_util.horizon_kappa`);
* it collects a fraction ``share`` of the training episodes;
* at the end of the tempering phase (the anneal fraction of the entropy schedule) the
  undiscounted returns of recent training episodes decide: the run continues with the shadow
  only if it is clearly better (Welch margin and a relative margin over the random warm-up
  return); otherwise the shadow is dropped and the run continues as the automatic-horizon
  learner, which never saw a difference in its own update rule.

The shadow uses private random streams (torch, buffer sampling, exploration noise), so with
``share = 0`` the main learner's parameters are bit-identical to a run without an audition.
"""
from __future__ import annotations

import contextlib
import copy
import math

import numpy as np
import torch

from .env_util import episode_horizon, horizon_kappa


class _BufferView:
    """The shared replay buffer as seen by the shadow: its own discount and sampling RNG."""

    def __init__(self, buffer, gamma: float, rng: np.random.Generator):
        self._buffer, self._gamma, self._rng = buffer, gamma, rng

    def sample(self, batch_size: int, rng=None, with_next1: bool = False) -> dict:
        return self._buffer.sample(batch_size, rng=self._rng, with_next1=with_next1,
                                   gamma=self._gamma)

    def __getattr__(self, name):
        return getattr(self._buffer, name)


class _TorchStream:
    """A private torch random stream (CPU and, when used, MPS / CUDA)."""

    def __init__(self, seed: int, device: torch.device):
        self.device = device
        with self._saved():
            torch.manual_seed(seed)
            self.state = self._get()

    def _get(self):
        st = {"cpu": torch.get_rng_state()}
        if self.device.type == "mps":
            st["mps"] = torch.mps.get_rng_state()
        elif self.device.type == "cuda":
            st["cuda"] = torch.cuda.get_rng_state(self.device)
        return st

    def _set(self, st) -> None:
        torch.set_rng_state(st["cpu"])
        if "mps" in st:
            torch.mps.set_rng_state(st["mps"])
        if "cuda" in st:
            torch.cuda.set_rng_state(st["cuda"], self.device)

    @contextlib.contextmanager
    def _saved(self):
        outer = self._get()
        try:
            yield
        finally:
            self._set(outer)

    @contextlib.contextmanager
    def active(self):
        """Run the block on this stream; the caller's stream is untouched."""
        with self._saved():
            self._set(self.state)
            try:
                yield
            finally:
                self.state = self._get()


class HorizonAudition:
    """Second (long-horizon) learner and the rule that may hand the run over to it."""

    #: fraction of training episodes the shadow collects during the audition
    SHARE = 0.25
    #: the decision uses training episodes that ended in this last fraction of the window
    RECENT = 0.2
    #: decision rule: Welch margin (in standard errors) and relative margin over random
    Z = 3.0
    REL = 0.25
    MIN_EPISODES = 3
    #: evidence-driven allocation (Screen 7d): the shadow's share of new episodes rises from
    #: SHARE to SHARE_MAX as its lead over the automatic learner grows from Z_START to Z_FULL
    #: standard errors (time-matched: episodes that ended in the last WINDOW of the budget)
    ADAPTIVE = False
    SHARE_MAX = 0.75
    Z_START, Z_FULL = 1.0, 3.0
    WINDOW = 0.1
    #: progress fractions before the end of tempering at which a clear lead commits early
    CHECKPOINTS: tuple = ()

    def __init__(self, main, gamma_long: float, seed: int | None):
        self.main = main
        self.gamma_long = float(gamma_long)
        self.decided = False
        self.choice = "auto"
        self.episodes: list[tuple[int, float, float]] = []   # (arm, return, progress at end)
        self.random_returns: list[float] = []
        base = 0 if seed is None else int(seed)
        self.rng = np.random.default_rng(base + 4242)        # arm assignment
        self.arm = np.zeros(main.n_envs, dtype=np.int64)       # 0 = automatic, 1 = long
        self.shadow = self._make_shadow(base)
        self.log: dict = {}
        for k, v in (getattr(main, "_audition_opts", None) or {}).items():
            if not hasattr(type(self), k):
                raise ValueError(f"unknown audition option {k!r}")
            setattr(self, k, tuple(v) if isinstance(v, list) else v)
        self._checked: set = set()
        self.trace: list = []          # (progress, share, z) about every 1 % of the budget

    # ------------------------------------------------------------------ construction
    def _make_shadow(self, base: int):
        from .tempo import TEMPO

        m = self.main
        horizon = episode_horizon(m.env)
        kw = dict(learning_rate=m.learning_rate, buffer_size=m.buffer_size,
                  learning_starts=m.learning_starts, batch_size=m.batch_size, tau=m.tau,
                  gamma=self.gamma_long, train_freq=m.train_freq, gradient_steps=m.gradient_steps,
                  n_step=m.n_step, policy_delay=m.policy_delay, ent_coef=m.ent_coef,
                  target_entropy=m.target_entropy,
                  entropy_schedule=getattr(m, "_long_entropy_schedule", None) or m.entropy_schedule,
                  critic_loss=m.critic_loss, reward_scaling=m.reward_scaling,
                  representation_lr=m.representation_lr, exploration_bonus=m.exploration_bonus,
                  exploration_noise=m.exploration_noise, max_grad_norm=m.max_grad_norm,
                  policy_kwargs=copy.deepcopy(m.policy_kwargs), device=m.device, verbose=0,
                  seed=base + 1000,
                  # the shadow trains on the main learner's host buffer (the classic engine: an
                  # audition never runs on the phase engine)
                  gpu_replay=False, act_on_cpu=False, pipeline=False,
                  # speed / batch-size options of the main learner, so that the shadow (and the run
                  # after adopt_shadow) uses the same Batch-Renorm momentum and warm-up, SALE refresh,
                  # Polyak tau, Adam betas, learning rates, autocast and Batch-Renorm kernel; the
                  # policy's fused_critics follows brn_impl as in the main learner
                  fused_adam=m.fused_adam, amp=m.amp, invariance=m.invariance, lr_rule=m.lr_rule,
                  ref_batch=m.ref_batch, brn_impl=m.brn_impl)
        # everything the shadow's construction draws comes from saved-and-restored streams
        py_state, np_state = __import__("random").getstate(), np.random.get_state()
        self.torch_stream = _TorchStream(base + 2000, m.device)
        with self.torch_stream.active():
            sh = TEMPO(m.policy_class, env=None, _init_setup_model=False, **kw)
            sh.observation_space = copy.deepcopy(m.observation_space)
            sh.action_space = copy.deepcopy(m.action_space)
            sh.n_envs = m.n_envs
            sh._setup_model()
        __import__("random").setstate(py_state)
        np.random.set_state(np_state)
        sh.env = None
        if sh.reward_scaler is not None:
            sh.reward_scaler.kappa = horizon_kappa(self.gamma_long, horizon)
        sh.replay_buffer = _BufferView(m.replay_buffer, self.gamma_long,
                                       np.random.default_rng(base + 3000))
        sh._learning_starts_resolved = m._learning_starts_resolved
        sh._colored_horizon = horizon or 1000
        return sh

    # ------------------------------------------------------------------ helpers
    def _sync(self) -> None:
        sh, m = self.shadow, self.main
        sh.num_timesteps = m.num_timesteps
        sh._total_timesteps = m._total_timesteps
        sh._current_progress_remaining = m._current_progress_remaining
        sh._learning_starts_resolved = m._learning_starts_resolved

    @property
    def commit_progress(self) -> float:
        frac = self.main._schedule[2]
        return float(frac) if frac > 0 else 0.5

    # ------------------------------------------------------------------ data collection
    def actions(self, stored, main_buffer_actions, main_env_actions):
        """Replace the actions of the environments whose episode belongs to the shadow."""
        if self.decided or not self.arm.any() or \
                self.main.num_timesteps < self.main._learning_starts_resolved:
            return main_buffer_actions, main_env_actions
        self._sync()
        sh = self.shadow
        sh.env = self.main.env                     # pink-noise horizon only
        with self.torch_stream.active():
            buf, env = sh._sample_actions(stored)
        sh.env = None
        idx = np.flatnonzero(self.arm)
        main_buffer_actions = np.array(main_buffer_actions, copy=True)
        main_env_actions = np.array(main_env_actions, copy=True)
        main_buffer_actions[idx], main_env_actions[idx] = buf[idx], env[idx]
        return main_buffer_actions, main_env_actions

    def observe(self, new_obs, rewards, terms, dones, infos) -> None:
        """Keep the shadow's statistics in step and record finished episodes per arm."""
        sh, m = self.shadow, self.main
        if sh.reward_scaler is not None:
            sh.reward_scaler.update(rewards, dones)
        sh._update_normalizer(new_obs)
        if sh._colored is not None:
            for i in np.flatnonzero(dones):
                sh._colored.reset(int(i))
        from .vec_env import episode_info

        progress = 1.0 - m._current_progress_remaining
        for i, info in enumerate(infos):
            if not dones[i]:
                continue
            ep = episode_info(info)
            if ep is not None:
                if m.num_timesteps <= m._learning_starts_resolved:
                    self.random_returns.append(ep[0])
                else:
                    self.episodes.append((int(self.arm[i]), ep[0], progress))
            self.arm[i] = int(self.rng.random() < self.current_share())

    def _window(self, lo: float, hi: float = 2.0):
        ra = np.array([r for a, r, p in self.episodes if a == 0 and lo <= p <= hi], np.float64)
        rl = np.array([r for a, r, p in self.episodes if a == 1 and lo <= p <= hi], np.float64)
        return ra, rl

    def lead_z(self, lo: float) -> float:
        """Lead of the shadow over the automatic learner in standard errors (0 if unknown)."""
        ra, rl = self._window(lo)
        if min(len(ra), len(rl)) < self.MIN_EPISODES:
            return 0.0
        se = math.sqrt(ra.var(ddof=1) / len(ra) + rl.var(ddof=1) / len(rl))
        return float((rl.mean() - ra.mean()) / se) if se > 0 else 0.0

    def current_share(self) -> float:
        if not self.ADAPTIVE:
            return self.SHARE
        p = 1.0 - self.main._current_progress_remaining
        z = self.lead_z(p - self.WINDOW)
        w = min(1.0, max(0.0, (z - self.Z_START) / (self.Z_FULL - self.Z_START)))
        share = self.SHARE + (self.SHARE_MAX - self.SHARE) * w
        if not self.trace or p - self.trace[-1][0] >= 0.01:
            self.trace.append((round(p, 4), round(share, 3), round(z, 2)))
        return share

    # ------------------------------------------------------------------ training
    def train(self, gradient_steps: int, batch_size: int) -> None:
        self._sync()
        with self.torch_stream.active():
            self.shadow.train(gradient_steps, batch_size)

    # ------------------------------------------------------------------ decision
    def due(self) -> bool:
        if self.decided:
            return False
        p = 1.0 - self.main._current_progress_remaining
        if p >= self.commit_progress:
            return True
        for c in self.CHECKPOINTS:                 # early commit on a clear lead only
            if c <= p and c not in self._checked:
                self._checked.add(c)
                if self._evaluate(p)[0] == "long":
                    return True
        return False

    def decide(self) -> str:
        """``"long"`` if the shadow's recent undiscounted returns are clearly better."""
        p = 1.0 - self.main._current_progress_remaining
        choice, log = self._evaluate(min(p, self.commit_progress) if p < self.commit_progress
                                     else self.commit_progress)
        log["trace"] = self.trace
        self.log, self.choice, self.decided = log, choice, True
        return choice

    def _evaluate(self, end: float):
        start = max(0.0, end - self.RECENT)
        ra, rl = self._window(start)
        if min(len(ra), len(rl)) < self.MIN_EPISODES:      # too few recent episodes: use all
            ra, rl = self._window(0.0)
        r_rand = (float(np.mean(self.random_returns)) if self.random_returns
                  else (float(min(ra.min(), rl.min())) if len(ra) and len(rl) else 0.0))
        log = dict(n_auto=int(len(ra)), n_long=int(len(rl)), random_return=r_rand,
                   gamma_auto=float(self.main.gamma), gamma_long=self.gamma_long,
                   progress=float(1.0 - self.main._current_progress_remaining))
        choice = "auto"
        if min(len(ra), len(rl)) >= self.MIN_EPISODES:
            ma, ml = float(ra.mean()), float(rl.mean())
            se = math.sqrt(ra.var(ddof=1) / len(ra) + rl.var(ddof=1) / len(rl))
            margin = max(self.Z * se, self.REL * max(ma - r_rand, 0.0))
            log.update(mean_auto=ma, mean_long=ml, se=se, margin=margin)
            if ml - ma > margin:
                choice = "long"
        log["choice"] = choice
        return choice, log

    def adopt_shadow(self) -> None:
        """Hand the run over to the long-horizon learner."""
        m, sh = self.main, self.shadow
        for name in ("policy", "actor_optimizer", "critic_optimizer", "rnd_optimizer",
                     "rep_optimizer", "log_ent_coef", "ent_coef_optimizer", "reward_scaler",
                     "_novelty_stats", "_rep_updates", "_colored", "_schedule", "entropy_schedule",
                     "_n_updates"):
            setattr(m, name, getattr(sh, name))
        m.gamma = self.gamma_long
        m.replay_buffer.gamma = self.gamma_long
        m.policy.train(False)
