"""TEMPO: Tempered-Entropy Multi-step Policy Optimization.

A maximum-entropy off-policy actor-critic with one default configuration for
discrete and continuous actions and for vector, image and dictionary
observations::

    from tempo_rl import TEMPO

    model = TEMPO("MlpPolicy", "Humanoid-v5", verbose=1)
    model.learn(total_timesteps=1_000_000)
    model.save("tempo_humanoid")

Algorithm (one gradient step, default configuration)::

    y      = (1/c) sum_{j<k} gamma^j r_{t+j} + b_t
             + gamma^k (1 - d) [min_i Q_i(s_{t+k}, a') - (alpha/c) log pi(a'|s_{t+k})]
    b_t    = beta(progress) (1 - gamma) e(s_{t+1}) / std(e)    (tempered novelty bonus, RND error;
             beta decays linearly from exploration_bonus to 0 over the entropy anneal fraction)
    critic : HL-Gauss cross-entropy of each Q_i(s_t, a_t) against y     (k <= n_step, c = reward scale)
             -- Batch-Renormalised critics (2 x 1024 units), one joint forward pass over
                (s_t, a_t) and (s_{t+k}, a'), no target network (CrossQ)
             -- inputs: encoder features and TD7's SALE embeddings z_s, z_sa (fixed copy)
    SALE   : g(z_s, a_t) regresses z_{s_{t+1}}; fixed copy refreshed every 250 steps of batch 256
             (re-counted for the default batch 1024: every 62 updates)
    actor  : min  E[alpha log pi(a|s) - min_i Q_i(s, a)]   every policy_delay steps
             (exact expectation for discrete actions)
    alpha  : min  log(alpha) * (H[pi] - H_target(progress))   (tempered target entropy)

Data collection draws the continuous policy's Gaussian sample from pink (1/f) noise,
one sequence per episode, and ``gamma="auto"`` sets the discount from the time limit
``T`` as ``clip(1 - 5/T, 0.95, 0.995)``. Continuous actions live in unit coordinates
``[-1, 1]^d`` inside the agent, so the entropy target means the same thing whatever
the environment's action bounds; ``Box`` vector observations are normalised with
running statistics. The default values are in the "Default recipe" table of the
documentation; the recipe is compute-heavy and runs on a GPU when one is available.

Speed (default since 2026-10-23: the fast recipe, formerly the variant ``fast4x``). The update maths
above is recipe C1's (the Screen-6 winner, the default from 2026-09-30 to 2026-10-22); the default runs it
on the fast engine: 4 environments, one update of batch 1024 per vector step (still 256 samples processed
per environment step, 4x fewer sequential updates), batch-size invariance rules with Adam's square-root
rule, the replay buffer on the training device, acting with a CPU copy of the actor pipelined one phase
behind the device, fused Adam, bfloat16 critic MLPs on a GPU, Batch Renormalization with a closed-form
backward and the twin critics as one batched MLP (:class:`TEMPO`, "Speed options"). One run alone on an
Apple M5 Pro GPU (Hopper-v5): ~507 environment steps per second, 5.3x the previous default's 94.8. It is NOT
bit-identical to C1 and its scores have not been screened; ``benchmarks/variants.py`` keeps C1 as the
named configuration ``"c1"`` (the C1 engine with every speed option off, bit for bit the previous default).

Under every engine the loop never waits for the device without need: host->device copies are
non-blocking, losses are logged as device tensors read only when a log line is written, and the
finiteness check of the critic loss is read back later. A diverging run on MPS / CUDA therefore ends
with the non-finite critic loss reported at the next action or phase boundary (or the end of
``learn``), after that update has been applied, and NaN actor outputs do not raise
``torch.distributions``' ``ValueError`` (see :mod:`tempo_rl._sync`).

A five-value ``entropy_schedule`` adds a second, late entropy anneal (research option; see :class:`TEMPO`).
"""
from __future__ import annotations

import math
import pathlib
import pickle
import platform
import random
import sys
import contextlib
import copy
import time
import warnings
from collections import deque
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces

from . import _sync
from .buffers import DeviceReplay, NStepReplayBuffer
from .callbacks import BaseCallback, make_callback
from .env_util import auto_gamma, episode_horizon, make_vec_env, wrap_env
from .logger import Logger, configure_logger, mean_of, scalar_of
from .policies import POLICY_ALIASES, TempoPolicy
from .preprocessing import check_action_space
from .save_util import load_from_zip, save_to_zip, space_from_json, space_to_json, spaces_match
from .networks import BatchRenorm, HostRunningNorm, StackedBRNMLP, project_weights
from .utils import ColoredNoise, HLGauss, RewardScaler, schedule_from_json, schedule_to_json
from .vec_env import SB3VecEnvAdapter, TempoVecEnv, episode_info
from .version import __version__

Schedule = Callable[[float], float]

#: Tempered target-entropy schedule (start, end, anneal fraction).
#: Continuous: nats per action dimension in unit coordinates.
#: Discrete: fraction of the maximum entropy ln|A|.
DEFAULT_ENTROPY_SCHEDULE = {"continuous": (0.0, -0.5, 0.5), "discrete": (0.7, 0.05, 0.5)}


def get_device(device: str | torch.device = "auto", has_images: bool = False,
               widest_layer: int = 0) -> torch.device:
    """``"auto"``: CUDA if available; else Apple MPS for image observations or wide
    networks (a layer of 512 units or more); else CPU.

    ``widest_layer`` is the largest actor or critic hidden width (``net_arch``); the
    default 1024-unit critics therefore run on MPS on Apple silicon. Small MLPs are as
    fast or faster on CPU than on MPS; CNNs and wide critics are several times faster
    on MPS. ``"mps"`` falls back to the CPU when MPS is unavailable.
    """
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (has_images or widest_layer >= 512) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _constant(value: float) -> Schedule:
    return lambda _: float(value)


def _opt_state_to_heads(sd: dict, st, lead: int) -> dict:
    """Adam state dict of fused critics (``lead`` leading parameters, then the stacked ones) -> the
    layout of the per-head critics (``lead`` parameters, then head 0's, head 1's, ...)."""
    names = st.head_param_names()
    P, state = len(names), sd["state"]
    group = dict(sd["param_groups"][0])
    shapes = dict(st.named_parameters())
    out = {i: state[i] for i in range(lead) if i in state}
    for p, name in enumerate(names):
        s = state.get(lead + p)
        if s is None:
            continue
        for i in range(st.n):
            out[lead + i * P + p] = {
                k: (st.to_head(name, v, i) if torch.is_tensor(v) and v.shape == shapes[name].shape
                    else (v.clone() if torch.is_tensor(v) else v)) for k, v in s.items()}
    group["params"] = list(range(lead + st.n * P))
    return {"state": out, "param_groups": [group] + [dict(g) for g in sd["param_groups"][1:]]}


def _opt_state_from_heads(sd: dict, st, lead: int) -> dict:
    """Inverse of :func:`_opt_state_to_heads` (a state dict already in the fused layout is returned)."""
    names = st.head_param_names()
    P, state = len(names), sd["state"]
    group = dict(sd["param_groups"][0])
    if len(group["params"]) == lead + P and st.n != 1:
        return sd
    out = {i: state[i] for i in range(lead) if i in state}
    for p, name in enumerate(names):
        heads = [state.get(lead + i * P + p) for i in range(st.n)]
        if any(h is None for h in heads):
            continue
        out[lead + p] = {
            k: (st.from_heads(name, [h[k] for h in heads]) if torch.is_tensor(v) and v.dim() > 0
                else (v.clone() if torch.is_tensor(v) else v)) for k, v in heads[0].items()}
    group["params"] = list(range(lead + P))
    return {"state": out, "param_groups": [group] + [dict(g) for g in sd["param_groups"][1:]]}


class TEMPO:
    """Tempered-Entropy Multi-step Policy Optimization.

    :param policy: ``"MlpPolicy"``, ``"CnnPolicy"``, ``"MultiInputPolicy"`` or ``"auto"``;
        the network is always built from the observation space.
    :param env: environment id, ``gymnasium.Env``, TEMPO vector env, or SB3 ``VecEnv``.
    :param learning_rate: Adam learning rate (float or ``f(progress_remaining)``).
    :param buffer_size: replay buffer capacity in transitions.
    :param learning_starts: uniform-random steps before learning; ``None`` picks
        2% of the first ``learn`` budget, clipped to [1000, 10000].
    :param batch_size: minibatch size per gradient step (default 1024 = 4 x ``ref_batch``; with
        ``invariance="samples"`` one update of 1024 stands for 4 updates of 256).
    :param tau: Polyak coefficient of the target critics (used only with
        ``policy_kwargs=dict(batch_norm=False)``; the default critics have none).
    :param gamma: ``"auto"`` (default): ``clip(1 - 5/T, 0.95, 0.995)`` from the
        environment's time limit ``T`` (the TD-MPC2 heuristic; 0.99 without a limit), or a
        fixed discount factor.
    :param train_freq: environment steps (per env) between training phases. ``None`` (default):
        ``max(1, round(batch_size / (ref_batch * n_envs)))``, so that about ``ref_batch`` = 256 samples
        are processed per environment step: 1 with the default 4 environments, 4 with one environment
        (a ``gymnasium.Env`` instance or a one-environment vector env).
    :param gradient_steps: gradient steps per training phase (-1: as many as env steps).
    :param n_step: horizon of the multi-step return (1 recovers SAC's target).
    :param policy_delay: actor and temperature update every ``policy_delay`` critic steps.
    :param ent_coef: ``"auto"`` (learned, initial 1.0), ``"auto_<init>"``, or a fixed float.
    :param target_entropy: ``"auto"`` (tempered schedule), a fixed float (nats), or
        ``f(progress_remaining) -> nats``.
    :param entropy_schedule: ``(start, end, anneal_fraction)`` for the tempered schedule;
        per-dimension nats in unit coordinates (continuous) or fractions of ln|A|
        (discrete). ``None`` uses TEMPO's defaults. A five-value schedule
        ``(start, mid, frac, end, frac2)`` (research option) adds a second, late anneal:
        ``start -> mid`` over the first ``frac`` of the budget (as the three-value form), then
        ``mid -> end`` linearly between ``frac`` and ``frac2``, constant afterwards; the novelty bonus
        still decays over ``frac``. Example: ``(0.0, -0.5, 0.5, -1.0, 0.8)`` keeps lowering the
        continuous entropy floor after 50 % instead of freezing it at -0.5 nats per dimension.
    :param critic_loss: ``"mse"`` (regression critic) or ``"hl_gauss"`` (categorical
        critic over 101 bins on [-5, 5] trained with the HL-Gauss cross-entropy).
    :param reward_scaling: divide rewards by a running discounted-return scale
        (never centred). ``None``: on for the categorical critic, off otherwise.
    :param exploration_bonus: weight ``beta_0`` of a *tempered novelty bonus* (default 5): a
        Random Network Distillation error of the next observation, divided by its running
        standard deviation and multiplied by ``1 - gamma``, added to the scaled n-step return
        with a weight that decays linearly to zero over the anneal fraction of the entropy
        schedule (the first half of training by default). ``0`` disables it. It reads the
        non-image part of the observation and is disabled, with a ``UserWarning``, for
        image-only observation spaces.
    :param exploration_noise: ``"pink"`` (default) or ``"white"``: with ``"pink"`` the
        standard-normal sample of the continuous policy during data collection is replaced by
        1/f noise that is correlated across the steps of an episode (Eberhard et al., 2023).
        Training losses, ``predict`` and evaluation are unchanged; discrete actions ignore it.
    :param horizon_audition: audition a second planning horizon (research option, off by
        default): a shadow learner with ``gamma = 1 - 1/T`` (T = the time limit) trains on the
        same replay data and collects a quarter of the episodes; at the end of the tempering
        phase the run continues with it only if its undiscounted training returns are clearly
        better. See :mod:`tempo_rl.horizon`. It runs on the classic engine (see "Fallbacks" below).
    :param representation_lr: Adam learning rate of the SALE representation
        (``policy_kwargs=dict(representation="sale")``, the default); TD7's value by default.
    :param max_grad_norm: optional gradient-norm clip (``None``: off).
    :param n_envs: number of environments when ``env`` is an id string (default 4). A
        ``gymnasium.Env`` instance is one environment; a vector env brings its own number.
    :param env_kwargs: keyword arguments for ``gymnasium.make`` when ``env`` is an id.
    :param stats_window_size: episodes averaged for ``rollout/`` logging.
    :param tensorboard_log: TensorBoard log directory (``None``: off).
    :param policy_kwargs: forwarded to :class:`~tempo_rl.policies.TempoPolicy`
        (``net_arch``, ``normalize_observations``, ``critic_layer_norm``, ...).
    :param verbose: 0 silent, 1 training table (every ``log_interval`` episodes).
    :param seed: random seed for torch, numpy, python and the environments.
    :param device: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``. ``"auto"``: CUDA if
        available; else MPS for image observations or a hidden layer of 512 units or more
        (the default critics); else CPU (``tempo_rl.tempo.get_device``).

    **Speed options (on by default since 2026-10-23: the fast recipe).** Each of them changes the
    numbers, so the default is NOT bit-identical to recipe C1, which every speed option switched off
    reproduces bit for bit (``n_envs=1, batch_size=256, train_freq=1, gpu_replay=False,
    act_on_cpu=False, pipeline=False, fused_adam=False, amp=None, invariance="none", lr_rule="none",
    brn_impl="autograd", policy_kwargs=dict(fused_critics=False)``; the configuration ``"c1"`` of
    ``benchmarks/variants.py``). Learning intensity is counted as SPE = samples processed per
    environment step = ``batch_size * U`` with ``U = gradient_steps / (n_envs * train_freq)`` updates
    per environment step: 256 for the default (1 update of 1024 per vector step of 4 environments) and
    for C1 (1 update of 256 per environment step). Measured for one run alone (Apple M5 Pro, MPS,
    Hopper-v5, 2026-10-23): ~507 environment steps per second, C1 94.8.

    :param gpu_replay: phase engine with the replay buffer on the training device
        (:class:`~tempo_rl.buffers.DeviceReplay`): a phase collects ``train_freq`` steps of
        ``n_envs`` environments, sends them with one non-blocking copy, then queues
        ``gradient_steps`` updates without any host-device synchronisation (minibatch indices
        from a device generator, n-step windows gathered on the device, the novelty bonus's
        running standard deviation merged on the device in float32 (Chan et al.) instead of a
        float64 host Welford loop, the observation normaliser kept on the host and pushed once
        per phase, the finiteness check of the critic loss read back when the phase completes).
        Vector, discrete and dictionary-of-vector observations; no ``horizon_audition``. ``None``
        (default): on, except for image observations and ``horizon_audition=True`` (see "Fallbacks").
    :param act_on_cpu: (``gpu_replay`` only) act with a CPU copy of the normaliser, the SALE
        encoder and the actor, refreshed once per phase by one non-blocking device->host copy;
        ``predict`` uses a CPU copy refreshed from the device when it is called. ``None`` (default):
        the resolved ``gpu_replay``.
    :param pipeline: (``gpu_replay`` and ``act_on_cpu`` only) the device runs phase p's updates
        while the CPU collects phase p+1 with the actor of phase p-1 (one phase of extra policy
        lag; without it acting uses the newest actor). The lag is fixed by construction, not by
        timing, so a seeded run is reproducible. ``None`` (default): the resolved ``act_on_cpu``.
    :param fused_adam: fused Adam kernels (``torch.optim.Adam(fused=True)``): same algorithm,
        fewer kernel launches, different rounding. Default True.
    :param amp: ``"auto"`` (default), ``"bf16"`` or ``None``: bfloat16 autocast around the critic MLPs
        only (weights, gradient accumulation and optimiser state stay float32). ``"auto"``: bfloat16 on
        MPS and on CUDA devices that support it, float32 on the CPU (bfloat16 matrix products are about
        100x slower there). An explicit ``"bf16"`` is ignored, with a warning, on the CPU.
    :param invariance: ``"samples"`` (default) or ``"none"``: batch-size invariance rules. With
        ``kappa = batch_size / ref_batch``, these quantities counted in updates are re-counted in
        samples, so that one update of ``256 kappa`` stands for ``kappa`` updates of 256:
        Batch-Renorm momentum ``m -> 1 - (1 - m)^kappa`` and warm-up ``W -> W / kappa`` passes, the
        Polyak coefficient of target critics (``batch_norm=False`` only; the default critics have
        none) ``tau -> 1 - (1 - tau)^kappa``, SALE fixed-copy refresh
        ``250 -> max(1, round(250 / kappa))`` updates, and the temperature learning rate ``x 1 / U``
        (its sign-like Adam steps keep their speed per environment step). NOT re-counted:
        ``policy_delay`` (the actor and the temperature still step every ``policy_delay`` updates,
        i.e. every ``policy_delay / U`` environment steps: 12 for the default, 3 for C1),
        ``learning_starts`` and the learning-rate schedule (both are counted in environment steps
        already), ``n_step``. Identity at ``kappa = 1`` and ``U = 1``.
    :param lr_rule: ``None`` (default: ``"sqrt"`` with ``invariance="samples"``, else ``"none"``),
        ``"sqrt"`` or ``"none"``: Adam's square-root scaling rule (Malladi et al., 2022) for the
        critic, actor, SALE and novelty networks: ``lr x sqrt(kappa)``, ``1 - beta2 -> kappa (1 - beta2)``,
        ``1 - beta1 -> kappa (1 - beta1)``, each beta clamped at 0. With CrossQ's critic / actor betas
        ``(0.5, 0.999)`` and ``kappa = 4`` (the default) the rule gives ``1 - 4 x 0.5 = -1``, clamped to
        0: those two optimisers run Adam WITHOUT momentum, ``(0.0, 0.996)``, at learning rate 2e-3
        (outside the small-``kappa (1 - beta)`` range where the SDE argument holds; the exact
        exponential re-count ``beta1^kappa`` would give 0.0625); SALE and novelty
        ``(0.9, 0.999) -> (0.6, 0.996)``. ``"sqrt"`` needs ``invariance="samples"``.
    :param ref_batch: the batch size the default hyperparameters were tuned for (256).
    :param brn_impl: ``"closed_form"`` (default) or ``"autograd"``: the training-mode kernel of every
        Batch-Renorm layer of the actor and the critics. ``"closed_form"`` computes the function
        ``y = x a + c`` (``a = w / sqrt(s2 + eps)``, ``c = b - m a`` with the warm-up / renormalised
        ``(m, s2)``) with one ``addcmul`` and differentiates it in closed form (two reductions, summed
        in float32, and two ``addcmul`` over the activations: :class:`tempo_rl.networks._BRNClosedForm`)
        instead of autograd's chain of elementwise kernels (``"autograd"``, C1's kernel); the
        statistics and the running-statistic updates are the same. Exact in real arithmetic,
        different rounding; evaluation mode is the same for both. With ``"closed_form"`` and
        Batch-Renormalised critics without ``weight_norm``, ``policy_kwargs`` ``fused_critics``
        defaults to True (the twin critics as one batched MLP,
        :class:`~tempo_rl.policies.TempoPolicy`); pass ``policy_kwargs=dict(fused_critics=False)`` for
        per-head critics. ``fused_critics=True`` needs ``brn_impl="closed_form"``.

    **Fallbacks** (where a part of the fast recipe cannot run; at most one ``UserWarning`` each):

    * a ``gymnasium.Env`` *instance* cannot be copied into 4 environments: the model collects with
      that one environment and ``train_freq`` resolves to 4 (one update of 1024 every 4 environment
      steps: the same 256 samples per environment step and the same ``U = 1/4``, so every invariance
      rule is unchanged; fewer environment steps per second than 4 environments). A warning says so;
      an explicit ``train_freq`` silences it. Pass the id (or a vector env) for 4 environments.
    * the CPU: everything as above except bfloat16 (``amp="auto"`` resolves to float32 there; no
      warning). The phase engine runs on the CPU too (the "device" is the CPU; the pipeline's one-phase
      lag is kept, so the numbers are those of the same recipe on the CPU).
    * image observations (``CnnPolicy`` inputs) and ``horizon_audition=True``: the phase engine
      (``gpu_replay``, ``act_on_cpu``, ``pipeline``) does not support them, so ``gpu_replay=None``
      resolves to False: the classic engine (replay buffer on the host, acting on the training device)
      with every other default unchanged (4 environments, batch 1024, invariance rules, fused Adam,
      bfloat16, closed-form Batch Renorm, fused critics). A warning says so; ``gpu_replay=False``
      silences it. An explicit ``gpu_replay=True`` raises instead.
    * dictionary observations of vectors run the fast recipe unchanged (images inside a dictionary:
      as image observations above).
    """

    policy_aliases = {name: TempoPolicy for name in POLICY_ALIASES}

    def __init__(self, policy: str | type = "MlpPolicy", env=None,
                 learning_rate: float | Schedule = 1e-3, buffer_size: int = 1_000_000,
                 learning_starts: int | None = None, batch_size: int = 1024, tau: float = 0.005,
                 gamma: float | str = "auto", train_freq: int | None = None, gradient_steps: int = 1,
                 n_step: int = 3, policy_delay: int = 3, ent_coef: str | float = "auto",
                 target_entropy: str | float | Schedule = "auto",
                 entropy_schedule: tuple[float, float, float] | None = None,
                 critic_loss: str = "hl_gauss", reward_scaling: bool | None = None,
                 representation_lr: float = 3e-4, exploration_bonus: float = 5.0,
                 exploration_noise: str = "pink", horizon_audition: bool = False,
                 max_grad_norm: float | None = None, n_envs: int = 4,
                 env_kwargs: dict | None = None, stats_window_size: int = 100,
                 tensorboard_log: str | None = None, policy_kwargs: dict | None = None,
                 verbose: int = 0, seed: int | None = None, device: str | torch.device = "auto",
                 gpu_replay: bool | None = None, act_on_cpu: bool | None = None,
                 pipeline: bool | None = None, fused_adam: bool = True, amp: str | None = "auto",
                 invariance: str = "samples", lr_rule: str | None = None, ref_batch: int = 256,
                 brn_impl: str = "closed_form",
                 _init_setup_model: bool = True):
        if isinstance(policy, str) and policy not in self.policy_aliases:
            raise ValueError(f"unknown policy {policy!r}; use one of {POLICY_ALIASES}")
        self.policy_class = self.policy_aliases[policy] if isinstance(policy, str) else policy
        self.policy_name = policy if isinstance(policy, str) else getattr(policy, "__name__", "custom")
        self.learning_rate = learning_rate
        self.buffer_size, self.learning_starts, self.batch_size = buffer_size, learning_starts, batch_size
        self.tau, self.gamma = tau, gamma
        if train_freq is not None and int(train_freq) < 1:
            raise ValueError(f"train_freq must be None or a positive integer, got {train_freq!r}")
        #: the requested train_freq (None: resolved from batch_size and n_envs, :meth:`_resolve_train_freq`)
        self._train_freq_request = None if train_freq is None else int(train_freq)
        self.train_freq, self.gradient_steps = self._train_freq_request or 1, int(gradient_steps)
        self.n_step, self.policy_delay = int(n_step), int(policy_delay)
        self.ent_coef, self.target_entropy = ent_coef, target_entropy
        self.entropy_schedule = entropy_schedule
        if critic_loss not in ("mse", "hl_gauss"):
            raise ValueError(f"critic_loss must be 'mse' or 'hl_gauss', got {critic_loss!r}")
        self.critic_loss, self.reward_scaling = critic_loss, reward_scaling
        self.representation_lr = float(representation_lr)
        self.exploration_bonus = float(exploration_bonus)
        if exploration_noise not in ("white", "pink"):
            raise ValueError(f"exploration_noise must be 'white' or 'pink', got {exploration_noise!r}")
        self.exploration_noise = exploration_noise
        self.horizon_audition = bool(horizon_audition)
        #: research hook: entropy schedule of the long-horizon learner (None: the same as this one)
        self._long_entropy_schedule = None
        self._audition = None                  # HorizonAudition while it runs
        self._horizon_choice = None            # "auto" / "long" once decided
        self._audition_log: dict = {}
        self._colored = None
        if self.exploration_bonus < 0:
            raise ValueError("exploration_bonus must be >= 0")
        self.max_grad_norm = max_grad_norm
        self.env_kwargs = env_kwargs
        self.stats_window_size = stats_window_size
        self.tensorboard_log = tensorboard_log
        self.policy_kwargs = dict(policy_kwargs or {})
        self.verbose, self.seed = verbose, seed
        self._device_arg = device
        # ---- speed options (the fast recipe by default; see the class docstring)
        if amp not in (None, "bf16", "auto"):
            raise ValueError(f"amp must be 'auto', 'bf16' or None, got {amp!r}")
        if invariance not in ("none", "samples"):
            raise ValueError(f"invariance must be 'none' or 'samples', got {invariance!r}")
        if lr_rule is None:                    # default: Adam's square-root rule with the invariance rules
            lr_rule = "sqrt" if invariance == "samples" else "none"
        if lr_rule not in ("none", "sqrt"):
            raise ValueError(f"lr_rule must be None, 'none' or 'sqrt', got {lr_rule!r}")
        if lr_rule != "none" and invariance != "samples":
            raise ValueError("lr_rule='sqrt' needs invariance='samples'")
        if int(ref_batch) <= 0:
            raise ValueError("ref_batch must be positive")
        # phase engine: None = automatic (resolved in _setup_model, :meth:`_resolve_engine`); explicit
        # combinations that cannot work are refused here
        if (act_on_cpu or pipeline) and gpu_replay is False:
            raise ValueError("act_on_cpu and pipeline are options of the phase engine: set "
                             "gpu_replay=True")
        if pipeline and act_on_cpu is False:
            raise ValueError("pipeline=True needs act_on_cpu=True (the CPU acts while the device "
                             "trains)")
        if gpu_replay and horizon_audition:
            raise ValueError("gpu_replay does not support horizon_audition")
        self._engine_request = (gpu_replay, act_on_cpu, pipeline)
        self.gpu_replay = None if gpu_replay is None else bool(gpu_replay)
        self.act_on_cpu = None if act_on_cpu is None else bool(act_on_cpu)
        self.pipeline = None if pipeline is None else bool(pipeline)
        self.fused_adam, self.amp = bool(fused_adam), amp
        self._amp = None                 # the resolved autocast dtype name ("bf16" or None), _setup_model
        self.invariance, self.lr_rule, self.ref_batch = invariance, lr_rule, int(ref_batch)
        self._host = self._host_eval = self._host_norm = None   # phase engine (gpu_replay)
        self._host_fetched_at = None
        self._host_loads = None          # policy state loads the host copies have seen (phase engine)
        if brn_impl not in ("autograd", "closed_form"):
            raise ValueError(f"brn_impl must be 'autograd' or 'closed_form', got {brn_impl!r}")
        self.brn_impl = brn_impl
        if (self.policy_kwargs.get("fused_critics") and brn_impl != "closed_form"):
            raise ValueError("policy_kwargs fused_critics=True runs the closed-form Batch-Renorm kernel: "
                             "pass brn_impl='closed_form' with it")

        self.env = None
        self.n_envs = n_envs
        self.observation_space: spaces.Space | None = None
        self.action_space: spaces.Space | None = None
        if env is not None:
            self.env = self._wrap_env(env, n_envs, seed, env_kwargs)
            self.n_envs = self.env.num_envs
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
            if (isinstance(env, gym.Env) and self._train_freq_request is None and int(n_envs) > 1
                    and self.n_envs == 1):
                warnings.warn(
                    f"TEMPO's default collects with {int(n_envs)} copies of the environment, but a gymnasium.Env "
                    "instance cannot be copied: this model uses the one environment with train_freq="
                    f"{self._auto_train_freq()} (one update of batch {self.batch_size} every "
                    f"{self._auto_train_freq()} environment steps, the same samples per environment step), which "
                    "collects fewer environment steps per second. Pass the environment id or a vector env for "
                    f"{int(n_envs)} environments, or train_freq explicitly to silence this warning.",
                    UserWarning, stacklevel=2)
        self._resolve_train_freq()

        self.num_timesteps = 0
        self._total_timesteps = 0
        self._episode_num = 0
        self._n_updates = 0
        self._current_progress_remaining = 1.0
        self._last_obs = None
        self._learning_starts_resolved: int | None = None
        self.ep_info_buffer: deque = deque(maxlen=stats_window_size)
        self.replay_buffer: NStepReplayBuffer | None = None
        self._logger: Logger | None = None
        self._start_time = time.time_ns()
        self._num_timesteps_at_start = 0
        #: Offset added to log-probabilities in the entropy terms. 0 means unit
        #: coordinates (TEMPO's default); research code sets it to -sum(log scale)
        #: to reproduce the physical-coordinate convention of the first version.
        self._entropy_log_offset = 0.0
        #: research switch (Screen 7, 2026-10-01): False keeps the reward scaler's running return
        #: through time-limit truncations, matching the critic, which bootstraps through them
        self._scale_resets_on_truncation = True

        self.gamma_request = gamma
        if isinstance(gamma, str):
            if gamma != "auto":
                raise ValueError(f"gamma must be a float or 'auto', got {gamma!r}")
            self.gamma = auto_gamma(episode_horizon(self.env)) if self.env is not None else 0.99
        else:
            self.gamma = float(gamma)

        if _init_setup_model:
            if self.observation_space is None:
                raise ValueError("an environment is required unless _init_setup_model=False")
            self._setup_model()

    # ================================================================ setup
    @staticmethod
    def _wrap_env(env, n_envs: int, seed: int | None, env_kwargs: dict | None = None):
        if isinstance(env, str):
            return make_vec_env(env, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs)
        if isinstance(env, (TempoVecEnv, SB3VecEnvAdapter)):
            return env
        if SB3VecEnvAdapter.is_sb3_vec_env(env):
            return SB3VecEnvAdapter(env)
        if isinstance(env, gym.Env):
            wrapped = wrap_env(env)
            venv = TempoVecEnv([lambda: wrapped])
            venv.seed(seed)
            return venv
        raise TypeError(f"unsupported environment {env!r}")

    def _auto_train_freq(self) -> int:
        """``train_freq=None``: ``max(1, round(batch_size / (ref_batch * n_envs)))`` (about ``ref_batch``
        samples processed per environment step)."""
        return max(1, int(round(self.batch_size / (self.ref_batch * max(1, int(self.n_envs))))))

    def _resolve_train_freq(self) -> None:
        """``self.train_freq`` from the request (an explicit value, or :meth:`_auto_train_freq`)."""
        self.train_freq = (self._auto_train_freq() if self._train_freq_request is None
                           else int(self._train_freq_request))

    def _resolve_engine(self) -> None:
        """Resolve ``gpu_replay`` / ``act_on_cpu`` / ``pipeline`` left at ``None`` (the default): the phase
        engine unless the observations contain images or ``horizon_audition`` is on (one warning then)."""
        g, a, p = getattr(self, "_engine_request", (self.gpu_replay, self.act_on_cpu, self.pipeline))
        if g is None:
            from .preprocessing import ObservationSpec
            reason = ("image observations" if ObservationSpec(self.observation_space).has_images else
                      "horizon_audition=True" if self.horizon_audition else None)
            g = reason is None
            if reason is not None:
                warnings.warn(f"TEMPO's default phase engine (gpu_replay, act_on_cpu, pipeline) does not support "
                              f"{reason}: this model uses the classic engine (replay buffer on the host, acting on "
                              "the training device); every other default setting is unchanged. Pass "
                              "gpu_replay=False to silence this warning.", UserWarning, stacklevel=4)
        if a is None:
            a = g
        if p is None:
            p = a
        if (a or p) and not g:
            raise ValueError("act_on_cpu and pipeline are options of the phase engine (gpu_replay=True), which "
                             "this model does not use")
        if p and not a:
            raise ValueError("pipeline=True needs act_on_cpu=True (the CPU acts while the device trains)")
        self.gpu_replay, self.act_on_cpu, self.pipeline = bool(g), bool(a), bool(p)
        self._engine_request = (self.gpu_replay, self.act_on_cpu, self.pipeline)

    def _resolve_amp(self) -> str | None:
        """The autocast dtype of the critic MLPs on this model's device (``amp``; ``"auto"``: bfloat16 on
        MPS and on CUDA devices that support it; always float32, i.e. None, on the CPU)."""
        if self.device.type == "cpu" or self.amp is None:
            return None
        if self.amp != "auto":
            return self.amp
        if self.device.type == "mps":
            newer = getattr(torch.backends.mps, "is_macos_or_newer", None)
            return "bf16" if newer is None or newer(14, 0) else None
        if self.device.type == "cuda":
            try:
                return "bf16" if torch.cuda.is_bf16_supported() else None
            except Exception:                                   # noqa: BLE001
                return None
        return None

    def _setup_model(self) -> None:
        check_action_space(self.action_space)
        self.discrete = isinstance(self.action_space, spaces.Discrete)
        self._resolve_engine()
        if self.seed is not None:
            self.set_random_seed(self.seed)     # before the networks are initialised
        pk = dict(self.policy_kwargs)
        opt_kwargs = dict(pk.pop("optimizer_kwargs", None) or {})
        if ("fused_critics" not in pk and self.brn_impl == "closed_form" and isinstance(self.policy_class, type)
                and issubclass(self.policy_class, TempoPolicy)):
            # default: the twin critics as one batched MLP where the policy can build them
            pk["fused_critics"] = bool(pk.get("batch_norm", True)) and not pk.get("weight_norm", False)
        if self.critic_loss == "hl_gauss":
            pk["n_bins"] = 101
        if self.exploration_bonus > 0:
            from .preprocessing import ObservationSpec
            if ObservationSpec(self.observation_space).flat_dim > 0:
                pk["novelty"] = True
            else:   # the RND novelty estimator reads the vector part of the observation
                import warnings
                warnings.warn("exploration_bonus needs a non-image observation component; it is "
                              "disabled for this image-only observation space", UserWarning,
                              stacklevel=3)                 # points at the TEMPO(...) call
                self.exploration_bonus = 0.0
        self.policy: TempoPolicy = self.policy_class(self.observation_space, self.action_space, **pk)
        if self.gpu_replay and self.policy.spec.has_images:
            raise ValueError("gpu_replay supports vector, discrete and dictionary-of-vector "
                             "observations, not images")
        widest = max([*self.policy.net_arch["pi"], *self.policy.net_arch["qf"], 0])
        self.device = get_device(self._device_arg, self.policy.spec.has_images, widest)
        self._amp = self._resolve_amp()
        if self.amp == "bf16" and self.device.type == "cpu":
            import warnings                   # (a local name in this function, see above)
            warnings.warn(f"amp={self.amp!r} is ignored on the CPU (bfloat16 matrix products are about 100x slower "
                          "there than float32); the critics run in float32", UserWarning, stacklevel=3)
        self.policy.to(self.device)
        if self.brn_impl != "autograd":              # kernel of every Batch-Renorm layer (training mode; default closed_form)
            for m in self.policy.modules():
                if isinstance(m, BatchRenorm):
                    m.impl = self.brn_impl
        self.hl_gauss = (HLGauss(101, 5.0, device=self.device) if self.critic_loss == "hl_gauss"
                         else None)
        use_scaling = (self.critic_loss == "hl_gauss") if self.reward_scaling is None \
            else bool(self.reward_scaling)
        self.reward_scaler = RewardScaler(self.n_envs, self.gamma) if use_scaling else None
        self.policy.train(False)
        self.learning_rate = schedule_from_json(self.learning_rate)
        self.lr_schedule = (self.learning_rate if callable(self.learning_rate)
                            else _constant(self.learning_rate))
        lr = self.lr_schedule(1.0)
        if getattr(self.policy, "batch_norm", False):
            opt_kwargs.setdefault("betas", (0.5, 0.999))   # CrossQ's Adam setting
        self._configure_invariance()
        # Adam variants of the speed options (fused kernels, square-root rule; both on by default, off for C1)
        fused = {"fused": True} if self.fused_adam else {}
        if self.lr_rule == "sqrt":
            opt_kwargs["betas"] = self._scaled_betas(*opt_kwargs.get("betas", (0.9, 0.999)))
            aux = {"betas": self._scaled_betas(0.9, 0.999)}
        else:
            aux = {}
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=lr, **opt_kwargs,
                                                **fused)
        self.critic_optimizer = torch.optim.Adam(self.policy.critic.parameters(), lr=lr,
                                                 **opt_kwargs, **fused)
        self.rnd_optimizer = (torch.optim.Adam(self.policy.rnd.predictor.parameters(),
                                               lr=1e-4 * self._lr_mult, **aux, **fused)
                              if self.policy.rnd is not None else None)
        self._novelty_stats = [0, 0.0, 0.0]          # count, mean, M2 of raw RND errors
        self._novelty_dev = None                     # the same statistic on the device (gpu_replay)
        self.rep_optimizer = (torch.optim.Adam(self.policy.rep.parameters(),
                                               lr=self.representation_lr * self._lr_mult, **aux,
                                               **fused)
                              if self.policy.rep is not None else None)
        self._rep_updates = 0

        self.auto_entropy = isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto")
        if self.auto_entropy:
            init = float(self.ent_coef.split("_")[1]) if "_" in self.ent_coef else 1.0
            if init <= 0:
                raise ValueError("the initial ent_coef must be positive")
            self.log_ent_coef = torch.tensor([math.log(init)], device=self.device, requires_grad=True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=lr, **fused)
        else:
            if float(self.ent_coef) < 0:
                raise ValueError("ent_coef must be non-negative")
            # log(0) = -inf -> alpha = 0 (no entropy bonus), as SB3 allows
            self.log_ent_coef = torch.tensor([float(self.ent_coef)], device=self.device).log()
            self.ent_coef_optimizer = None

        kind = "discrete" if self.discrete else "continuous"
        self._schedule = tuple(self.entropy_schedule or DEFAULT_ENTROPY_SCHEDULE[kind])
        self._entropy_unit = (math.log(self.action_space.n) if self.discrete
                              else float(self.action_space.shape[0]))

    # ======================================================= speed options
    @property
    def kappa(self) -> float:
        """``batch_size / ref_batch`` with ``invariance="samples"``, else 1."""
        return self.batch_size / self.ref_batch if self.invariance == "samples" else 1.0

    @property
    def updates_per_env_step(self) -> float:
        """``U = gradient_steps / (n_envs * train_freq)`` (``gradient_steps=-1``: 1)."""
        g = self.gradient_steps if self.gradient_steps >= 0 else self.train_freq * self.n_envs
        return g / float(self.train_freq * self.n_envs)

    @property
    def samples_per_env_step(self) -> float:
        """Learning intensity SPE = ``batch_size * U`` (the default recipe: 256)."""
        return self.batch_size * self.updates_per_env_step

    def _scaled_betas(self, b1: float, b2: float) -> tuple[float, float]:
        k = self.kappa
        return max(0.0, 1.0 - k * (1.0 - b1)), max(0.0, 1.0 - k * (1.0 - b2))

    def _configure_invariance(self) -> None:
        """Learning-rate multipliers and Batch-Renorm rules of ``invariance`` / ``lr_rule``
        (all identities by default)."""
        k = self.kappa
        self._lr_mult = math.sqrt(k) if self.lr_rule == "sqrt" else 1.0
        if self.invariance == "samples" and k != 1.0:
            for m in self.policy.modules():
                if isinstance(m, (BatchRenorm, StackedBRNMLP)):
                    m.momentum = 1.0 - (1.0 - m.momentum) ** k
                    m.warmup_steps = int(round(m.warmup_steps / k))

    def _polyak_tau(self) -> float:
        """Polyak coefficient per update of the target critics (``batch_norm=False`` only): ``tau``,
        or ``1 - (1 - tau)^kappa`` under ``invariance="samples"`` (``kappa`` updates of 256 in one)."""
        k = self.kappa
        if self.invariance == "samples" and k != 1.0:
            return 1.0 - (1.0 - self.tau) ** k
        return self.tau

    def _alpha_lr_mult(self) -> float:
        """Learning-rate multiplier of the temperature: ``1 / U`` with ``invariance="samples"``, else 1."""
        u = self.updates_per_env_step
        return 1.0 / u if (self.invariance == "samples" and u > 0) else 1.0

    def _rep_refresh_every(self) -> int:
        if self.invariance == "samples" and self.kappa != 1.0:
            return max(1, int(round(self.REP_REFRESH_EVERY / self.kappa)))
        return self.REP_REFRESH_EVERY

    def scaling_summary(self) -> dict:
        """The resolved speed settings: N, K, G, B, kappa, U, SPE, learning-rate multipliers,
        Adam betas, Batch-Renorm momentum / warm-up, SALE refresh period, engine options."""
        brn = next((m for m in self.policy.modules() if isinstance(m, BatchRenorm)), None)
        return dict(N=self.n_envs, K=self.train_freq, K_auto=self._train_freq_request is None,
                    G=self.gradient_steps, B=self.batch_size,
                    kappa=self.kappa, updates_per_env_step=self.updates_per_env_step,
                    samples_per_env_step=self.samples_per_env_step, lr_mult=self._lr_mult,
                    alpha_lr_mult=self._alpha_lr_mult(),
                    adam_betas=list(self.critic_optimizer.param_groups[0]["betas"]),
                    brn_momentum=None if brn is None else brn.momentum,
                    brn_warmup=None if brn is None else brn.warmup_steps,
                    rep_refresh_every=self._rep_refresh_every(), policy_delay=self.policy_delay,
                    polyak_tau=self._polyak_tau() if self.policy.critic_target is not None else None,
                    gpu_replay=self.gpu_replay, act_on_cpu=self.act_on_cpu, pipeline=self.pipeline,
                    fused_adam=self.fused_adam, amp=self.amp, amp_resolved=self._amp,
                    brn_impl=self.brn_impl, fused_critics=bool(getattr(self.policy, "fused_critics", False)))

    # ============================================================ entropy
    def current_target_entropy(self) -> float:
        """Target entropy (nats) at the current training progress."""
        te = self.target_entropy
        if callable(te):
            return float(te(self._current_progress_remaining))
        if not isinstance(te, str):
            return float(te)
        done = 1.0 - self._current_progress_remaining
        if len(self._schedule) == 5:       # research option (Screen 7b): a second, later anneal
            start, mid, frac, end, frac2 = self._schedule
            if done > frac and frac2 > frac:
                w2 = min(1.0, (done - frac) / (frac2 - frac))
                return (mid + (end - mid) * w2) * self._entropy_unit
            end = mid
        else:
            start, end, frac = self._schedule
        w = 1.0 if frac <= 0 else min(1.0, max(0.0, done / frac))
        return (start + (end - start) * w) * self._entropy_unit

    # ============================================================== logging
    @property
    def logger(self) -> Logger:
        if self._logger is None:
            self._logger = Logger(None, ())
        return self._logger

    def set_logger(self, logger: Logger) -> None:
        """Use ``logger`` for every later ``learn`` call (instead of building one)."""
        self._logger = logger
        self._custom_logger = True

    def _dump_logs(self) -> None:
        elapsed = max((time.time_ns() - self._start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / elapsed)
        lg = self.logger
        lg.record("time/episodes", self._episode_num)
        if self.ep_info_buffer:
            lg.record("rollout/ep_rew_mean", float(np.mean([e[0] for e in self.ep_info_buffer])))
            lg.record("rollout/ep_len_mean", float(np.mean([e[1] for e in self.ep_info_buffer])))
        lg.record("time/fps", fps)
        lg.record("time/time_elapsed", int(elapsed))
        lg.record("time/total_timesteps", self.num_timesteps)
        lg.dump(step=self.num_timesteps)

    # ============================================================== learning
    def _setup_learn(self, total_timesteps: int, callback, reset_num_timesteps: bool,
                     tb_log_name: str, progress_bar: bool):
        if self.env is None:
            raise ValueError("no environment: pass one to TEMPO(...) or call set_env()")
        self._start_time = time.time_ns()
        if self.gpu_replay:
            self._prepare_phase_engine()
        if reset_num_timesteps or self._last_obs is None:
            if reset_num_timesteps:
                self.num_timesteps = 0
                self._episode_num = 0
                self.ep_info_buffer = deque(maxlen=self.stats_window_size)
            if self.replay_buffer is not None:
                self.replay_buffer.close_open_episodes()
            self._last_obs = self.env.reset_all()
            self._update_normalizer(self._last_obs)
        if not reset_num_timesteps:
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps
        if self.learning_starts is None:
            if self._learning_starts_resolved is None:
                self._learning_starts_resolved = int(np.clip(0.02 * total_timesteps, 1000, 10_000))
        else:
            self._learning_starts_resolved = int(self.learning_starts)

        # + train_freq: the last collection phase may overshoot total_timesteps
        rows_needed = (math.ceil(max(total_timesteps - self.num_timesteps, 1) / self.n_envs)
                       + self.train_freq + 1)
        max_rows = math.ceil(self.buffer_size / self.n_envs) + 1
        if self.replay_buffer is None and self.gpu_replay:
            self.replay_buffer = DeviceReplay(
                min(rows_needed, max_rows), self.policy.spec,
                () if self.discrete else self.action_space.shape, self.discrete,
                n_envs=self.n_envs, n_step=self.n_step, gamma=self.gamma, device=self.device,
                seed=None if self.seed is None else self.seed + 12345)
        elif self.replay_buffer is None:
            self.replay_buffer = NStepReplayBuffer(
                min(rows_needed, max_rows), self.policy.spec,
                action_shape=() if self.discrete else self.action_space.shape,
                action_dtype=np.int64 if self.discrete else np.float32, n_envs=self.n_envs,
                n_step=self.n_step, gamma=self.gamma, device=self.device,
                seed=None if self.seed is None else self.seed + 12345)
        else:
            self.replay_buffer.ensure_capacity(self.replay_buffer.n_added + rows_needed, max_rows)
        self._maybe_start_audition()

        if not getattr(self, "_custom_logger", False):
            self._logger = configure_logger(self.verbose, self.tensorboard_log, tb_log_name,
                                        reset_num_timesteps)
        callback = make_callback(callback, progress_bar)
        callback.init_callback(self)
        return total_timesteps, callback

    def _maybe_start_audition(self) -> None:
        if (not self.horizon_audition or self._horizon_choice is not None
                or self._audition is not None or self.num_timesteps > 0):
            return
        horizon = episode_horizon(self.env)
        gamma_long = 1.0 - 1.0 / horizon if horizon else None
        if gamma_long is None or gamma_long <= self.gamma:
            self._horizon_choice = "auto"          # nothing longer to audition
            return
        from .horizon import HorizonAudition

        self._audition = HorizonAudition(self, gamma_long, self.seed)
        self._audition.shadow._update_normalizer(self._last_obs)
        self._audition.arm[:] = (self._audition.rng.random(self.n_envs)
                                 < self._audition.SHARE).astype(np.int64)

    def _audition_step(self, steps: int) -> None:
        aud = self._audition
        aud.train(steps, self.batch_size)
        if aud.due():
            choice = aud.decide()
            if choice == "long":
                aud.adopt_shadow()
            self._horizon_choice, self._audition_log, self._audition = choice, aud.log, None
            if self.verbose >= 1:
                print(f"horizon audition: {choice} {aud.log}", flush=True)

    def learn(self, total_timesteps: int, callback=None, log_interval: int = 4,
              tb_log_name: str = "TEMPO", reset_num_timesteps: bool = True,
              progress_bar: bool = False) -> "TEMPO":
        """Train for ``total_timesteps`` environment steps. Returns ``self``.

        ``log_interval`` counts episodes between log dumps (``verbose >= 1``).
        """
        total_timesteps, callback = self._setup_learn(total_timesteps, callback,
                                                      reset_num_timesteps, tb_log_name,
                                                      progress_bar)
        callback.on_training_start(locals(), globals())
        if self.gpu_replay:
            self._learn_phases(total_timesteps, callback, log_interval)
            callback.on_training_end()
            return self
        while self.num_timesteps < total_timesteps:
            callback.on_rollout_start()
            cont = self._collect(callback, log_interval)
            callback.on_rollout_end()
            if not cont:
                break
            if self.num_timesteps > self._learning_starts_resolved and self.replay_buffer.size > 0:
                steps = self.gradient_steps if self.gradient_steps >= 0 else self.train_freq * self.n_envs
                if steps > 0:
                    self.train(steps, self.batch_size)
                    if self._audition is not None:
                        self._audition_step(steps)
        if self.device.type != "cpu":     # deferred checks of the last updates (one blocking read)
            _sync.full_sync(self.device)
        callback.on_training_end()
        return self

    def _update_normalizer(self, obs_list) -> None:
        norm = self.policy.normalizer
        if norm is None:
            return
        vec = self.policy.spec.vector_part(self.policy.spec.stack(obs_list))
        if self._host_norm is not None:            # phase engine: host mirror, pushed per phase
            self._host_norm.update(vec)
            h = self._host.normalizer
            h.mean.copy_(torch.as_tensor(self._host_norm.mean, dtype=torch.float32))
            h.var.copy_(torch.as_tensor(self._host_norm.var, dtype=torch.float32))
            h.count.fill_(self._host_norm.count)
            return
        norm.update(_sync.to_device(vec, self.device))

    def _sample_actions(self, stored_obs):
        """(buffer actions, environment actions) for the current observations."""
        if self.num_timesteps < self._learning_starts_resolved:
            env_actions = np.array([self.action_space.sample() for _ in range(self.n_envs)])
            buf = env_actions if self.discrete else self.policy.scale_action(env_actions)
            return buf, env_actions
        tensors = {k: _sync.to_device(v, self.device) for k, v in stored_obs.items()}
        noise = None
        if self.exploration_noise == "pink" and not self.discrete:
            if self._colored is None or self._colored.n_envs != self.n_envs:
                horizon = episode_horizon(self.env) or 1000
                self._colored = ColoredNoise(self.n_envs, int(self.action_space.shape[0]), 1.0,
                                             horizon, None if self.seed is None else self.seed + 777)
            noise = _sync.to_device(self._colored.next(), self.device)
        a = self.policy.act(tensors, deterministic=False, noise=noise).cpu().numpy()
        if self.device.type != "cpu":
            # the .cpu() above waited for the whole device queue: release the host buffers of
            # earlier non-blocking copies and raise if a deferred finiteness check failed
            _sync.synced(self.device)
        out = (a, a) if self.discrete else (a, self.policy.unscale_action(a))
        if self._audition is not None:
            out = self._audition.actions(stored_obs, *out)
        return out

    def _collect(self, callback: BaseCallback, log_interval: int) -> bool:
        spec = self.policy.spec
        self.policy.train(False)
        for _ in range(self.train_freq):
            stored = spec.stack(self._last_obs)
            buffer_actions, env_actions = self._sample_actions(stored)
            new_obs, rewards, terms, truncs, infos, final_obs = self.env.step_all(env_actions)
            dones = terms | truncs
            self.num_timesteps += self.n_envs
            if self._colored is not None:
                for i in np.flatnonzero(dones):
                    self._colored.reset(int(i))
            if self.reward_scaler is not None:
                self.reward_scaler.update(rewards, dones if self._scale_resets_on_truncation else terms)
            self._update_normalizer(new_obs)
            if self._audition is not None:
                self._audition.observe(new_obs, rewards, terms, dones, infos)
            final_stored = {i: {k: v[0] for k, v in spec.to_storage(o, batched=False).items()}
                            for i, o in final_obs.items()}
            self.replay_buffer.add(stored, buffer_actions, rewards, terms, truncs,
                                   spec.stack(new_obs), final_stored)
            self._last_obs = new_obs
            self._current_progress_remaining = 1.0 - self.num_timesteps / max(1, self._total_timesteps)
            finished = 0
            for info in infos:
                ep = episode_info(info)
                if ep is not None:
                    self.ep_info_buffer.append(ep)
            for d in dones:
                if d:
                    self._episode_num += 1
                    finished += 1
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            if finished and log_interval and self._episode_num % log_interval == 0:
                self._dump_logs()
        return True

    # ======================================================== phase engine
    # TEMPO(gpu_replay=True): collect a phase of train_freq x n_envs steps on the host, send it to
    # the device replay with one non-blocking copy, queue the phase's updates, and wait for the
    # device only at the phase boundary (pipeline=True: one phase later).
    def _prepare_phase_engine(self) -> None:
        """(Re)build the host copies used by the phase engine from the device networks."""
        _sync.full_sync(self.device, check=False)
        self._host = copy.deepcopy(self.policy).to("cpu")
        self._host.train(False)
        self._host.critic.qs = torch.nn.ModuleList()     # acting never reads the critic MLPs
        if getattr(self._host.critic, "stacked", None) is not None:
            self._host.critic.stacked = None             # fused_critics
        if self._host.rnd is not None:
            self._host.rnd = None
        self._host_eval = None
        self._host_fetched_at = None
        self._host_loads = self._policy_loads()
        self._host_norm = (HostRunningNorm(self.policy.normalizer)
                           if self.policy.normalizer is not None else None)

        def floats(mod):
            return [t for t in list(mod.parameters()) + list(mod.buffers()) if t.is_floating_point()]
        pol, host = self.policy, self._host
        self._sync_src = floats(pol.actor) + (floats(pol.rep_fixed.f) if pol.rep_fixed is not None else [])
        self._sync_dst = floats(host.actor) + (floats(host.rep_fixed.f) if host.rep_fixed is not None else [])
        self._sync_sizes = [t.numel() for t in self._sync_dst]
        self._stage = {k: [] for k in ("obs", "next_obs", "actions", "rewards", "terms", "truncs")}

    @torch.no_grad()
    def _snapshot_policy(self) -> torch.Tensor:
        """Queue a non-blocking device->host copy of the acting parameters (one flat buffer),
        valid once a device event recorded after this call has completed."""
        flat = torch.cat([t.reshape(-1) for t in self._sync_src])
        host = torch.empty(flat.shape, dtype=flat.dtype)
        host.copy_(flat, non_blocking=True)
        return host

    @torch.no_grad()
    def _load_snapshot(self, flat: torch.Tensor, dst=None) -> None:
        dst = self._sync_dst if dst is None else dst
        for t, chunk in zip(dst, torch.split(flat, self._sync_sizes)):
            t.copy_(chunk.view_as(t))

    def _push_normalizer(self, non_blocking: bool = True):
        """Host statistics -> device normaliser; returns the host tensor to keep alive."""
        if self._host_norm is None:
            return None
        return self._host_norm.write_to(self.policy.normalizer, non_blocking=non_blocking)

    def _sync_device_normalizer(self) -> None:
        """Blocking push of the host normaliser statistics (end of learn, save)."""
        if self._host_norm is not None:
            self._push_normalizer(non_blocking=False)
            _sync.full_sync(self.device, check=False)

    def _sample_actions_phase(self, stored):
        if self.num_timesteps < self._learning_starts_resolved:
            env_actions = np.array([self.action_space.sample() for _ in range(self.n_envs)])
            buf = env_actions if self.discrete else self.policy.scale_action(env_actions)
            return buf, env_actions
        noise = None
        if self.exploration_noise == "pink" and not self.discrete:
            if self._colored is None or self._colored.n_envs != self.n_envs:
                horizon = episode_horizon(self.env) or 1000
                self._colored = ColoredNoise(self.n_envs, int(self.action_space.shape[0]), 1.0,
                                             horizon, None if self.seed is None else self.seed + 777)
            noise = self._colored.next()
        if self.act_on_cpu:
            tensors = {k: torch.as_tensor(v) for k, v in stored.items()}
            a = self._host.act(tensors, deterministic=False,
                               noise=None if noise is None else torch.as_tensor(noise)).numpy()
        else:
            tensors = {k: _sync.to_device(v, self.device) for k, v in stored.items()}
            nz = None if noise is None else _sync.to_device(noise, self.device)
            a = self.policy.act(tensors, deterministic=False, noise=nz).cpu().numpy()
            _sync.synced(self.device, check=False)
        return (a, a) if self.discrete else (a, self.policy.unscale_action(a))

    def _collect_phase(self, callback: BaseCallback, log_interval: int) -> bool:
        spec = self.policy.spec
        st = self._stage
        for _ in range(self.train_freq):
            stored = spec.stack(self._last_obs)
            buffer_actions, env_actions = self._sample_actions_phase(stored)
            new_obs, rewards, terms, truncs, infos, final_obs = self.env.step_all(env_actions)
            dones = terms | truncs
            self.num_timesteps += self.n_envs
            if self._colored is not None:
                for i in np.flatnonzero(dones):
                    self._colored.reset(int(i))
            if self.reward_scaler is not None:
                self.reward_scaler.update(rewards, dones if self._scale_resets_on_truncation else terms)
            self._update_normalizer(new_obs)
            genuine = list(new_obs)
            for i, o in final_obs.items():
                genuine[int(i)] = o
            st["obs"].append(stored)
            st["next_obs"].append(spec.stack(genuine))
            st["actions"].append(np.asarray(buffer_actions))
            st["rewards"].append(rewards)
            st["terms"].append(terms)
            st["truncs"].append(truncs)
            self._last_obs = new_obs
            self._current_progress_remaining = 1.0 - self.num_timesteps / max(1, self._total_timesteps)
            finished = 0
            for info in infos:
                ep = episode_info(info)
                if ep is not None:
                    self.ep_info_buffer.append(ep)
            for d in dones:
                if d:
                    self._episode_num += 1
                    finished += 1
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            if finished and log_interval and self._episode_num % log_interval == 0:
                self._dump_logs()
        return True

    def _flush(self) -> list:
        """Queue the staged transitions into the device replay (non-blocking); returns the host
        tensors that must stay alive until the phase's device event completes."""
        st = self._stage
        if not st["rewards"]:
            return []
        keys = st["obs"][0].keys()
        obs = {k: np.stack([o[k] for o in st["obs"]]) for k in keys}
        nxt = {k: np.stack([o[k] for o in st["next_obs"]]) for k in keys}
        keep = self.replay_buffer.add_rows(obs, np.stack(st["actions"]), np.stack(st["rewards"]),
                                           np.stack(st["terms"]), np.stack(st["truncs"]), nxt)
        for v in st.values():
            v.clear()
        return keep

    def _learn_phases(self, total_timesteps: int, callback: BaseCallback, log_interval: int) -> None:
        dev = self.device
        inflight: deque = deque()     # (device event, actor snapshot or None, host buffers, flag)

        def retire(item) -> None:
            ev, snap, _keep, flag = item
            if ev is not None:
                ev.synchronize()
            if snap is not None:
                self._load_snapshot(snap)
            if flag is not None and bool(flag):
                nf = self.__dict__.get("_nonfinite")
                if nf is not None:            # reported once (a caught error does not fire again)
                    nf.zero_()
                raise FloatingPointError(_sync.NONFINITE_MESSAGE)

        try:
            while self.num_timesteps < total_timesteps:
                if self._host_loads != self._policy_loads():   # policy.load_state_dict in a callback
                    self._reseed_host_mirrors()
                callback.on_rollout_start()
                cont = self._collect_phase(callback, log_interval)
                callback.on_rollout_end()
                keep = self._flush()
                snap = flag = None
                if not cont:                  # stopped by a callback: the flushed rows still land
                    ev = _sync.new_event(dev)
                    if ev is not None:
                        ev.record()
                    inflight.append((ev, None, keep, None))
                    break
                keep.append(self._push_normalizer())     # host statistics -> device, once per phase
                if self.num_timesteps > self._learning_starts_resolved and self.replay_buffer.size > 0:
                    steps = self.gradient_steps if self.gradient_steps >= 0 else self.train_freq * self.n_envs
                    if steps > 0:
                        self.train(steps, self.batch_size)
                        if self.act_on_cpu:
                            snap = self._snapshot_policy()
                        nf = self.__dict__.get("_nonfinite")
                        if nf is not None:
                            flag = torch.empty((), dtype=torch.bool)
                            flag.copy_(nf, non_blocking=True)
                ev = _sync.new_event(dev)
                if ev is not None:
                    ev.record()
                inflight.append((ev, snap, keep, flag))
                # pipeline: the next phase acts with the actor of the previous phase while the device
                # runs this phase's updates; otherwise wait now (acting then uses the newest actor)
                depth = 1 if self.pipeline else 0
                while len(inflight) > depth:
                    retire(inflight.popleft())
            while inflight:
                retire(inflight.popleft())
        except BaseException:
            # an error (non-finite flag, callback, KeyboardInterrupt) leaves queued copies that still
            # read (staged rows, normaliser statistics) or write (actor snapshots, flags) host buffers
            # held by `inflight` / `keep`: wait for the device before this frame releases them
            _sync.full_sync(dev, check=False)
            raise
        self._sync_device_normalizer()

    def _policy_loads(self) -> int:
        """How many times the policy's state has been loaded (``TempoPolicy._load_from_state_dict``)."""
        return self.policy.__dict__.get("_load_count", 0)

    @torch.no_grad()
    def _reseed_host_mirrors(self) -> None:
        """The policy's state was loaded from outside the phase engine (``set_parameters``,
        ``policy.load_state_dict``): copy the device actor, SALE encoder and normaliser statistics
        into the host copies (blocking), so that acting (``act_on_cpu``), the normaliser pushed to the
        device and ``predict`` use the loaded state. With ``pipeline=True`` a snapshot queued before
        the load can still overwrite the acting copy once (one phase, the pipeline's usual lag)."""
        if self._host is None:
            return
        flat = torch.cat([t.reshape(-1) for t in self._sync_src]).cpu()
        _sync.synced(self.device, check=False)        # .cpu() waited for the whole queue
        self._load_snapshot(flat)
        if self._host_norm is not None:
            self._host_norm = HostRunningNorm(self.policy.normalizer)
            self._host.normalizer.load_state_dict(self.policy.normalizer.state_dict())
        self._host_fetched_at = None
        self._host_loads = self._policy_loads()

    @torch.no_grad()
    def _host_predictor(self):
        """CPU copy of the acting networks for ``predict`` (act_on_cpu), refreshed from the
        device when the model has changed since the last call (trained, or its policy state loaded
        with ``set_parameters`` / ``policy.load_state_dict``; one blocking copy)."""
        if self._host_loads != self._policy_loads():
            self._reseed_host_mirrors()
        if self._host_eval is None:
            self._host_eval = copy.deepcopy(self._host)
            self._host_eval_dst = (
                [t for t in list(self._host_eval.actor.parameters()) + list(self._host_eval.actor.buffers())
                 if t.is_floating_point()]
                + ([t for t in list(self._host_eval.rep_fixed.f.parameters())
                    + list(self._host_eval.rep_fixed.f.buffers()) if t.is_floating_point()]
                   if self._host_eval.rep_fixed is not None else []))
            self._host_fetched_at = None
        stamp = (self.num_timesteps, self._n_updates)
        if self._host_fetched_at != stamp:
            flat = torch.cat([t.reshape(-1) for t in self._sync_src]).cpu()
            _sync.synced(self.device, check=False)    # .cpu() waited for the whole queue
            self._load_snapshot(flat, self._host_eval_dst)
            if self._host_eval.normalizer is not None:
                self._host_eval.normalizer.load_state_dict(self._host.normalizer.state_dict())
            self._host_fetched_at = stamp
        return self._host_eval

    # ============================================================== training
    def _clip(self, params) -> None:
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

    def train(self, gradient_steps: int, batch_size: int = 256) -> None:
        """Perform ``gradient_steps`` TEMPO updates from the replay buffer.

        Networks stay in eval mode except inside the forward passes that are
        meant to update Batch-Renormalization statistics. Nothing here waits for the device
        (losses are logged as device tensors; see :mod:`tempo_rl._sync`).
        """
        self.policy.train(False)
        lr = self.lr_schedule(self._current_progress_remaining)
        for opt in (self.actor_optimizer, self.critic_optimizer):
            for g in opt.param_groups:
                g["lr"] = lr * self._lr_mult if self._lr_mult != 1.0 else lr
        if self.ent_coef_optimizer is not None:
            alpha_mult = self._alpha_lr_mult()
            for g in self.ent_coef_optimizer.param_groups:
                g["lr"] = lr * alpha_mult if alpha_mult != 1.0 else lr
        if self.rep_optimizer is not None:            # not scheduled; a loaded/set value wins
            for g in self.rep_optimizer.param_groups:
                g["lr"] = (self.representation_lr * self._lr_mult if self._lr_mult != 1.0
                           else self.representation_lr)
        h_target = self.current_target_entropy()
        scale = self.reward_scaler.scale if self.reward_scaler is not None else 1.0
        # The temperature acts on *unscaled* rewards: in scaled units the entropy
        # bonus is alpha / scale, so alpha's dynamics do not depend on the scaling.
        self._reward_scale = scale
        on_device_replay = isinstance(self.replay_buffer, DeviceReplay)
        overlap = self.device.type != "cpu" and not on_device_replay
        tau = self._polyak_tau()                      # self.tau itself unless invariance="samples"
        stats: dict[str, list] = {}
        for _ in range(gradient_steps):
            batch = self.replay_buffer.sample(
                batch_size, with_next1=self.policy.rep is not None or self.policy.rnd is not None)
            if scale != 1.0:
                batch["returns"] = batch["returns"] / scale
            finish = None
            if self.policy.rnd is not None:
                bonus_w = self.current_exploration_bonus()
                if bonus_w > 0:
                    if on_device_replay:
                        batch["returns"] = batch["returns"] + self._novelty_bonus_device(batch, bonus_w)
                    elif overlap:     # host statistic read once the critic's forward is queued
                        finish = self._novelty_bonus_start(batch, bonus_w)
                    else:
                        batch["returns"] = batch["returns"] + self._novelty_bonus(batch, bonus_w)
            self._n_updates += 1
            update_actor = self._n_updates % max(1, self.policy_delay) == 0
            rep_loss = self._train_representation(batch) if self.policy.rep is not None else None
            if self.discrete:
                out = self._train_discrete(batch, update_actor, finish)
            else:
                out = self._train_continuous(batch, update_actor, finish)
            if update_actor and self.auto_entropy:
                ent_loss = (self.log_ent_coef * (out["entropy_t"] - h_target)).mean()
                self.ent_coef_optimizer.zero_grad(set_to_none=True)
                ent_loss.backward()
                self.ent_coef_optimizer.step()
                out["ent_coef_loss"] = ent_loss.detach()
            out.pop("entropy_t", None)
            if rep_loss is not None:
                out["representation_loss"] = rep_loss
            if self.policy.critic_target is not None:
                with torch.no_grad():
                    for p, pt in zip(self.policy.critic.parameters(),
                                     self.policy.critic_target.parameters()):
                        pt.lerp_(p, tau)
                    if self.policy.rep_fixed_target is not None:
                        for p, pt in zip(self.policy.rep_fixed.parameters(),
                                         self.policy.rep_fixed_target.parameters()):
                            pt.lerp_(p, tau)
            for k, v in out.items():
                stats.setdefault(k, []).append(v)
        lg = self.logger
        lg.record("train/n_updates", self._n_updates)
        lg.record("train/learning_rate", lr)
        lg.record("train/target_entropy", h_target)
        lg.record("train/ent_coef", scalar_of(self.log_ent_coef.detach().exp()))
        if self.reward_scaler is not None:
            lg.record("train/reward_scale", scale)
        if self.policy.rnd is not None:
            lg.record("train/exploration_bonus", self.current_exploration_bonus())
        for k, v in stats.items():
            lg.record(f"train/{k}", mean_of(v))
        if self.device.type != "cpu" and not on_device_replay and self.__dict__.get("_nonfinite") is not None:
            _sync.queue_check(self._nonfinite)      # read after the next action (no wait here)

    def current_exploration_bonus(self) -> float:
        """Current weight of the tempered novelty bonus.

        ``exploration_bonus * max(0, 1 - progress / f)``, where ``f`` is the anneal fraction
        of the entropy schedule and ``progress`` the fraction of the ``learn`` budget done;
        0 when the bonus is off.
        """
        if self.exploration_bonus <= 0:
            return 0.0
        frac = self._schedule[2]
        done = 1.0 - self._current_progress_remaining
        return self.exploration_bonus * (0.0 if frac <= 0 else max(0.0, 1.0 - done / frac))

    def _novelty_error(self, b: dict) -> torch.Tensor:
        """Train the RND predictor on ``s_{t+1}``; returns the raw (pre-step) errors."""
        pol = self.policy
        flat, _ = pol.preprocess(b["next1_obs"])
        err = pol.rnd.error(flat)
        loss = err.mean()
        self.rnd_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.rnd_optimizer.step()
        return err.detach()

    def _welford(self, values) -> float:
        """Merge raw errors into the running statistics (sequential Welford, float64); returns
        the running standard deviation."""
        n, mean, m2 = self._novelty_stats
        for x in values:                                # Welford over raw errors
            n += 1
            d = x - mean
            mean += d / n
            m2 += d * (x - mean)
        self._novelty_stats = [n, mean, m2]
        return math.sqrt(m2 / (n - 1)) if n > 1 else 1.0

    def _novelty_bonus(self, b: dict, weight: float) -> torch.Tensor:
        """RND bonus for the first step of each window, in the critic's (scaled) units.

        The raw prediction error of ``s_{t+1}`` is divided by its running standard
        deviation and multiplied by ``weight * (1 - gamma)``, so the discounted sum of
        bonuses stays of order ``weight`` on the critic's value scale. The predictor is
        trained on the same observations.
        """
        raw = self._novelty_error(b)
        std = self._welford(raw.cpu().numpy().tolist())
        return weight * (1.0 - self.gamma) * raw / max(std, 1e-8)

    def _novelty_bonus_start(self, b: dict, weight: float):
        """:meth:`_novelty_bonus` in two halves (accelerators): the predictor update and a
        non-blocking copy of the raw errors are queued now; the returned ``finish(batch)`` waits
        for that copy only (the critic's forward pass queued in between keeps the device busy),
        merges the statistics on the host and adds the bonus to ``batch["returns"]``. Same values
        as :meth:`_novelty_bonus`."""
        raw = self._novelty_error(b)
        host = torch.empty(raw.shape, dtype=raw.dtype)
        host.copy_(raw, non_blocking=True)
        # the device writes `host` later: keep it alive until the next full sync even if an error
        # drops `finish` before it has waited for the copy
        _sync.keep(host, self.device)
        ev = self.__dict__.get("_novelty_event")
        if ev is None:
            ev = self._novelty_event = _sync.new_event(self.device)
        ev.record()

        def finish(batch: dict) -> None:
            ev.synchronize()
            std = self._welford(host.numpy().tolist())
            batch["returns"] = batch["returns"] + weight * (1.0 - self.gamma) * raw / max(std, 1e-8)
        return finish

    def _novelty_bonus_device(self, b: dict, weight: float) -> torch.Tensor:
        """:meth:`_novelty_bonus` with the running statistics on the device (``gpu_replay``):
        Chan et al.'s parallel merge of the batch's mean and variance in float32 (no host read)."""
        raw = self._novelty_error(b)
        st = self._novelty_dev
        if st is None:
            n, mean, m2 = self._novelty_stats
            st = self._novelty_dev = [torch.tensor(float(v), device=self.device)
                                      for v in (n, mean, m2)]
        cnt, mean, m2 = st
        nb = float(raw.numel())
        bm, bv = raw.mean(), raw.var(unbiased=False)
        tot = cnt + nb
        delta = bm - mean
        mean = mean + delta * (nb / tot)
        m2 = m2 + bv * nb + delta.pow(2) * cnt * nb / tot
        self._novelty_dev = [tot, mean, m2]
        std = torch.where(tot > 1, (m2 / (tot - 1).clamp_min(1.0)).sqrt(),
                          torch.ones_like(tot)).clamp_min(1e-8)
        return weight * (1.0 - self.gamma) * raw / std

    def _novelty_stats_host(self) -> list:
        """``[count, mean, M2]`` of the raw RND errors (reads the device copy under gpu_replay)."""
        if self._novelty_dev is not None:
            n, mean, m2 = (float(t) for t in self._novelty_dev)
            return [int(round(n)), mean, m2]
        return list(self._novelty_stats)

    def _ent_coef(self) -> torch.Tensor:
        """Temperature in the critic's (possibly scaled) reward units."""
        return self.log_ent_coef.detach().exp() / getattr(self, "_reward_scale", 1.0)

    @staticmethod
    def _cat_obs(flat, images, nflat, nimages):
        jflat = None if flat is None else torch.cat([flat, nflat], 0)
        jimages = {k: torch.cat([images[k], nimages[k]], 0) for k in images}
        return jflat, jimages

    def _values(self, critic_out: torch.Tensor) -> torch.Tensor:
        """Critic output -> scalar Q values (expectation over bins for the categorical critic)."""
        return critic_out if self.hl_gauss is None else self.hl_gauss.value(critic_out)

    def _critic_loss(self, q_out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Sum over critics of the per-critic loss against the shared target ``y`` [B]."""
        n_critics = q_out.shape[0]
        if self.hl_gauss is None:
            return F.mse_loss(q_out, y.expand_as(q_out), reduction="none").mean(1).sum()
        return n_critics * self.hl_gauss.loss(q_out, y.expand(n_critics, -1))

    def _critic_out(self, features: torch.Tensor, action=None, target: bool = False):
        """Critic (or target critic) output on precomputed features; with bfloat16 (``amp``, resolved for
        the device) the critic MLPs run under bfloat16 autocast and the output is cast back to float32."""
        net = self.policy.critic_target if target else self.policy.critic
        if self._amp is None or self.device.type == "cpu":
            if action is None:
                return net(None, None, features=features)
            return net(None, None, action, features=features)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            out = (net(None, None, features=features) if action is None
                   else net(None, None, action, features=features))
        return out.float()

    def _critic_step(self, critic_loss: torch.Tensor) -> None:
        if self.device.type == "cpu":
            if not torch.isfinite(critic_loss):
                raise FloatingPointError(_sync.NONFINITE_MESSAGE)
        else:
            # accelerators: the check stays on the device and is read later (_sync), so the
            # backward pass is queued without waiting for the forward pass to finish. A non-finite
            # loss is therefore reported at the next action (or the end of learn), after this update
            # has been applied (on the CPU: raised here, before the update)
            flag = self.__dict__.get("_nonfinite")
            if flag is None or flag.device != critic_loss.device:
                flag = self._nonfinite = torch.zeros((), dtype=torch.bool, device=critic_loss.device)
            flag.logical_or_(~torch.isfinite(critic_loss.detach()))
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip(self.policy.critic.parameters())
        self.critic_optimizer.step()
        if self.policy.weight_norm:
            project_weights(self.policy.critic)

    def _actor_step(self, actor_loss: torch.Tensor) -> None:
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._clip(self.policy.actor.parameters())
        self.actor_optimizer.step()
        if self.policy.weight_norm:
            project_weights(self.policy.actor)

    @contextlib.contextmanager
    def _critic_frozen(self):
        """Critic parameters without gradient during the actor update: the critic's weight
        gradients of the actor loss were never used (the next critic step resets them), the
        actor's gradient is the same."""
        params = [p for p in self.policy.critic.parameters() if p.requires_grad]
        for p in params:
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p in params:
                p.requires_grad_(True)

    #: gradient steps between refreshes of the fixed embedding copy (TD7)
    REP_REFRESH_EVERY = 250

    def _train_representation(self, b: dict) -> torch.Tensor:
        """SALE step: ``g(f(s), a)`` regresses the next state's ``f(s')`` (one step ahead)."""
        pol = self.policy
        with torch.no_grad():
            enc = pol.critic.encoder(*pol.preprocess(b["obs"]))
            target = pol.rep.zs(pol.critic.encoder(*pol.preprocess(b["next1_obs"])))
        pred = pol.rep.zsa(pol.rep.zs(enc), pol.rep_action(b["actions"]))
        loss = F.mse_loss(pred, target)
        self.rep_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.rep_optimizer.step()
        self._rep_updates += 1
        if self._rep_updates % self._rep_refresh_every() == 0:
            pol.rep_fixed.load_state_dict(pol.rep.state_dict())
        return loss.detach()

    def _train_continuous(self, b: dict, update_actor: bool, finish=None) -> dict:
        pol = self.policy
        alpha = self._ent_coef()
        off = self._entropy_log_offset
        flat, images = pol.preprocess(b["obs"], augment=True)
        nflat, nimages = pol.preprocess(b["next_obs"], augment=True)
        with torch.no_grad():
            nenc = pol.critic.encoder(nflat, nimages)
            nzs = pol.state_embedding(nenc)
            nx, nlogp = pol.actor(pol.actor_input(nenc, nzs))
        if pol.batch_norm:
            # CrossQ: one joint pass so the normalisation statistics cover the
            # mixture of current and bootstrap state-actions; no target network.
            jflat, jimages = self._cat_obs(flat, images, nflat, nimages)
            ja = torch.cat([b["actions"], nx], 0)
            pol.critic.train(True)
            jenc = pol.critic.encoder(jflat, jimages)
            q_all = self._critic_out(pol.critic_input(jenc, pol.state_embedding(jenc), ja), ja)
            pol.critic.train(False)
            n = b["actions"].shape[0]
            q, q_next = q_all[:, :n], self._values(q_all[:, n:].detach()).min(0).values
        else:
            with torch.no_grad():
                nenc_t = pol.critic_target.encoder(nflat, nimages)
                nzs_t = pol.state_embedding(nenc_t, target=True)
                q_next = self._values(self._critic_out(
                    pol.critic_input(nenc_t, nzs_t, nx, target=True), nx, target=True)
                ).min(0).values
            enc = pol.critic.encoder(flat, images)
            q = self._critic_out(pol.critic_input(enc, pol.state_embedding(enc), b["actions"]),
                                 b["actions"])
        if finish is not None:                    # novelty bonus whose host read was deferred
            finish(b)
        y = (b["returns"] + b["discounts"] * (q_next - alpha * (nlogp + off))).detach()
        critic_loss = self._critic_loss(q, y)
        self._critic_step(critic_loss)
        out = {"critic_loss": critic_loss.detach(), "q_mean": self._values(q.detach()).mean()}
        if not update_actor:
            return out
        with self._critic_frozen():
            enc = pol.critic.encoder(flat, images).detach()
            zs = pol.state_embedding(enc)
            pol.actor.train(True)
            x, logp = pol.actor(pol.actor_input(enc, zs))
            pol.actor.train(False)
            q_pi = self._values(self._critic_out(pol.critic_input(enc, zs, x), x)).min(0).values
            logp = logp + off
            actor_loss = (alpha * logp - q_pi).mean()
            self._actor_step(actor_loss)
        entropy = -logp.detach().mean()
        out.update(actor_loss=actor_loss.detach(), entropy=entropy, entropy_t=entropy)
        return out

    def _train_discrete(self, b: dict, update_actor: bool, finish=None) -> dict:
        pol = self.policy
        alpha = self._ent_coef()
        flat, images = pol.preprocess(b["obs"], augment=True)
        nflat, nimages = pol.preprocess(b["next_obs"], augment=True)
        with torch.no_grad():
            nenc = pol.critic.encoder(nflat, nimages)
            nzs = pol.state_embedding(nenc)
            nlogp = F.log_softmax(pol.actor(pol.actor_input(nenc, nzs)), -1)
        if pol.batch_norm:
            jflat, jimages = self._cat_obs(flat, images, nflat, nimages)
            pol.critic.train(True)
            jenc = pol.critic.encoder(jflat, jimages)
            q_joint = self._critic_out(pol.critic_input(jenc, pol.state_embedding(jenc)))
            pol.critic.train(False)
            n = b["actions"].shape[0]
            q_all, q_next = q_joint[:, :n], self._values(q_joint[:, n:].detach()).min(0).values
        else:
            with torch.no_grad():
                nenc_t = pol.critic_target.encoder(nflat, nimages)
                nzs_t = pol.state_embedding(nenc_t, target=True)
                q_next = self._values(self._critic_out(
                    pol.critic_input(nenc_t, nzs_t, target=True), target=True)).min(0).values
            enc = pol.critic.encoder(flat, images)
            q_all = self._critic_out(pol.critic_input(enc, pol.state_embedding(enc)))
        if finish is not None:                    # novelty bonus whose host read was deferred
            finish(b)
        v_next = (nlogp.exp() * (q_next - alpha * nlogp)).sum(-1)   # exact soft value
        y = (b["returns"] + b["discounts"] * v_next).detach()
        idx = b["actions"].long().reshape(1, -1, 1)
        if self.hl_gauss is None:
            q = q_all.gather(2, idx.expand(q_all.shape[0], -1, 1)).squeeze(2)
        else:
            idx = idx.unsqueeze(-1).expand(q_all.shape[0], -1, 1, q_all.shape[-1])
            q = q_all.gather(2, idx).squeeze(2)
        critic_loss = self._critic_loss(q, y)
        self._critic_step(critic_loss)
        out = {"critic_loss": critic_loss.detach(), "q_mean": self._values(q.detach()).mean()}
        if not update_actor:
            return out
        with torch.no_grad():
            enc = pol.critic.encoder(flat, images)
            zs = pol.state_embedding(enc)
            q_min = self._values(self._critic_out(pol.critic_input(enc, zs))).min(0).values
        pol.actor.train(True)
        logp = F.log_softmax(pol.actor(pol.actor_input(enc, zs)), -1)
        pol.actor.train(False)
        pi = logp.exp()
        actor_loss = (pi * (alpha * logp - q_min)).sum(-1).mean()   # exact expectation
        self._actor_step(actor_loss)
        entropy = -(pi * logp).sum(-1).detach().mean()
        out.update(actor_loss=actor_loss.detach(), entropy=entropy, entropy_t=entropy)
        return out

    # ============================================================ inference
    def predict(self, observation, state=None, episode_start=None, deterministic: bool = False):
        """Action(s) for one observation or a batch: returns ``(actions, state)``.

        With ``act_on_cpu=True`` (after ``learn``) a CPU copy of the acting networks answers,
        refreshed from the device whenever the model has trained or its policy state has been loaded
        (``set_parameters``, ``policy.load_state_dict``) since the last call. Parameters changed in
        place by other means (e.g. ``p.data.copy_(...)``) are not detected: call ``set_parameters``."""
        if self.act_on_cpu and self._host is not None:
            return self._host_predictor().predict(observation, state, episode_start, deterministic)
        return self.policy.predict(observation, state, episode_start, deterministic)

    # ------------------------------------------- critic optimiser state in the per-head format
    def _fused_layout(self):
        """``(stacked module, number of critic parameters before it)`` with ``fused_critics``, else None."""
        st = getattr(self.policy.critic, "stacked", None)
        if st is None:
            return None
        params = list(self.policy.critic.parameters())
        first = next(iter(st.parameters()))
        lead = next(i for i, p in enumerate(params) if p is first)
        return st, lead

    def _critic_optimizer_state(self) -> dict:
        """The critic optimiser's state dict; with ``fused_critics`` converted to the per-head layout of
        the default critics (files and ``get_parameters`` keep one format)."""
        sd = self.critic_optimizer.state_dict()
        lay = self._fused_layout()
        return sd if lay is None else _opt_state_to_heads(sd, *lay)

    def _load_critic_optimizer(self, sd: dict) -> None:
        lay = self._fused_layout()
        if lay is not None:
            sd = _opt_state_from_heads(sd, *lay)
        self.critic_optimizer.load_state_dict(sd)

    # ================================================================= envs
    def get_env(self):
        return self.env

    def get_vec_normalize_env(self):
        """SB3 compatibility (used by SB3's ``EvalCallback``): TEMPO has no ``VecNormalize``."""
        return None

    def set_env(self, env, force_reset: bool = True) -> None:
        venv = self._wrap_env(env, self.n_envs, None, self.env_kwargs)
        if not spaces_match(venv.observation_space, self.observation_space):
            raise ValueError("observation space of the new env does not match the model's")
        if not spaces_match(venv.action_space, self.action_space):
            raise ValueError("action space of the new env does not match the model's")
        if self.replay_buffer is not None and venv.num_envs != self.n_envs:
            raise ValueError("the number of environments cannot change once the buffer exists")
        self.env = venv
        self.n_envs = venv.num_envs
        self._resolve_train_freq()              # train_freq=None follows the number of environments
        if self.reward_scaler is not None and len(self.reward_scaler.ret) != self.n_envs:
            self.reward_scaler.ret = np.zeros(self.n_envs, dtype=np.float64)
        if force_reset:
            self._last_obs = None

    def set_random_seed(self, seed: int | None = None) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.action_space is not None:
            self.action_space.seed(seed)
        if self.env is not None:
            self.env.seed(seed)

    # ========================================================== persistence
    _HYPERPARAMS = ("policy_name", "buffer_size", "learning_starts", "batch_size", "tau", "gamma",
                    "train_freq", "gradient_steps", "n_step", "policy_delay", "ent_coef",
                    "target_entropy", "entropy_schedule", "critic_loss", "reward_scaling",
                    "representation_lr", "exploration_bonus", "exploration_noise",
                    "horizon_audition",
                    "max_grad_norm", "n_envs",
                    "stats_window_size", "verbose", "seed",
                    # speed options (2026-10-22 / 2026-10-23); files saved before them load with
                    # _LEGACY_DEFAULTS
                    "gpu_replay", "act_on_cpu", "pipeline", "fused_adam", "amp", "invariance",
                    "lr_rule", "ref_batch", "brn_impl")

    #: The values of the speed options before they became the default (2026-10-23): the C1 engine. A
    #: file that does not record one of them was saved by an older library, whose default was this value.
    _LEGACY_DEFAULTS = dict(gpu_replay=False, act_on_cpu=False, pipeline=False, fused_adam=False, amp=None,
                            invariance="none", lr_rule="none", ref_batch=256, brn_impl="autograd")
    #: Options removed on 2026-10-23 (research options of Screen 8 that were never adopted) and the value
    #: that meant "off". A file saved with one of them active cannot be reproduced and is refused;
    #: ``eval_ema`` never changed training, so its average is dropped with a warning instead.
    _REMOVED_OPTIONS = dict(recent_replay=None, actor_pessimism=1.0, alpha_lr_mult=None, eval_ema=None)

    #: policy settings whose defaults have changed (or may change): saved resolved, so a file
    #: always rebuilds the architecture its weights belong to (``fused_critics``: files saved before
    #: 2026-10-23 do not record it and load with per-head critics, their default then)
    _RESOLVED_POLICY = ("net_arch", "representation", "batch_norm", "weight_norm", "zsa_norm",
                        "fused_critics")

    def _resolved_policy_json(self) -> dict:
        pol = self.policy
        out = {k: getattr(pol, k) for k in self._RESOLVED_POLICY if hasattr(pol, k)}
        if isinstance(out.get("net_arch"), dict):
            out["net_arch"] = {k: [int(w) for w in v] for k, v in out["net_arch"].items()}
        if getattr(pol, "rep", None) is not None:
            out["zs_dim"] = int(pol.rep.zs_dim)
        return out

    @staticmethod
    def _infer_policy_kwargs(pk: dict, policy_state: dict) -> dict:
        """Architecture of a file saved before the resolved settings were recorded (2026-09-30).

        Such files store only the ``policy_kwargs`` the user passed; the rest were the defaults of
        their time, which have changed since. The hidden widths are read from the weight shapes
        and the representation from the stored parameters, so the model is rebuilt as saved.
        """
        def widths(prefix: str) -> list[int]:
            layers = []
            for k, v in policy_state.items():
                idx = k[len(prefix):].split(".")[0]
                if k.startswith(prefix) and k.endswith(".weight") and v.ndim == 2 and idx.isdigit():
                    layers.append((int(idx), int(v.shape[0])))
            return [w for _, w in sorted(layers)[:-1]]

        pk = dict(pk)
        if "net_arch" not in pk:
            qf, pi = widths("critic.qs.0."), widths("actor.net.")
            if qf and pi:
                pk["net_arch"] = dict(pi=pi, qf=qf)
        if "representation" not in pk:
            pk["representation"] = "sale" if any(k.startswith("rep.") for k in policy_state) else None
        return pk

    def _policy_kwargs_json(self) -> dict:
        out = {}
        for k, v in self.policy_kwargs.items():
            if k == "activation_fn":
                out[k] = v.__name__
            else:
                out[k] = v
        return out

    def save(self, path, exclude=None, include=None) -> None:
        """Save to ``path.zip`` (networks, optimisers, temperature, obs statistics, hyperparameters)."""
        self._sync_device_normalizer()                 # phase engine: newest host statistics
        data: dict[str, Any] = {k: getattr(self, k) for k in self._HYPERPARAMS}
        data["train_freq"] = self._train_freq_request           # None: resolved again for the env of load
        data["train_freq_resolved"] = self.train_freq
        data["learning_rate"] = schedule_to_json(self.learning_rate)
        data["target_entropy"] = schedule_to_json(self.target_entropy)
        for key, value in (("learning_rate", data["learning_rate"]),
                           ("target_entropy", data["target_entropy"])):
            if value is None:                          # a lambda / arbitrary callable
                data[f"{key}_not_saved"] = True
                warnings.warn(f"{key} is a callable and is not saved; TEMPO.load will need "
                              f"custom_objects={{'{key}': ...}} (or use "
                              "tempo_rl.utils.LinearSchedule)", UserWarning)
        data.update(
            algorithm="TEMPO", tempo_version=__version__,
            observation_space=space_to_json(self.observation_space),
            action_space=space_to_json(self.action_space),
            policy_kwargs=self._policy_kwargs_json(),
            resolved_policy_kwargs=self._resolved_policy_json(),
            num_timesteps=self.num_timesteps, n_updates=self._n_updates,
            episode_num=self._episode_num, total_timesteps=self._total_timesteps,
            progress_remaining=self._current_progress_remaining,
            learning_starts_resolved=self._learning_starts_resolved,
            entropy_log_offset=self._entropy_log_offset,
            # an audition interrupted by save() cannot resume (the shadow is not saved): "auto"
            horizon_choice=self._horizon_choice or ("auto" if self.horizon_audition else None),
            audition_log=self._audition_log,
            reward_scaler=None if self.reward_scaler is None else self.reward_scaler.state(),
            system_info=dict(python=platform.python_version(), torch=torch.__version__,
                             gymnasium=gym.__version__, numpy=np.__version__,
                             platform=platform.platform()))
        for k in list(exclude or []):
            data.pop(k, None)
        tempo_state = {"log_ent_coef": self.log_ent_coef.detach().cpu(),
                       "actor_optimizer": self.actor_optimizer.state_dict(),
                       "critic_optimizer": self._critic_optimizer_state()}
        if self.ent_coef_optimizer is not None:
            tempo_state["ent_coef_optimizer"] = self.ent_coef_optimizer.state_dict()
        if self.rnd_optimizer is not None:
            tempo_state["rnd_optimizer"] = self.rnd_optimizer.state_dict()
            tempo_state["novelty_stats"] = torch.tensor(self._novelty_stats_host(), dtype=torch.float64)
        if self.rep_optimizer is not None:
            tempo_state["rep_optimizer"] = self.rep_optimizer.state_dict()
            tempo_state["rep_updates"] = torch.tensor(self._rep_updates)
        save_to_zip(path, data, self.policy.state_dict(), tempo_state, __version__)

    @classmethod
    def load(cls, path, env=None, device: str | torch.device = "auto",
             custom_objects: dict | None = None, print_system_info: bool = False,
             force_reset: bool = True, **kwargs) -> "TEMPO":
        """Load a model saved with :meth:`save`. Pass ``env`` to continue training."""
        data, policy_state, tempo_state = load_from_zip(path, device="cpu")
        if print_system_info:
            print("System info of the saved model:", data.get("system_info"))
        ent_coef_overridden = "ent_coef" in kwargs or "ent_coef" in (custom_objects or {})
        for key in ("learning_rate", "target_entropy"):
            if (data.get(f"{key}_not_saved") and key not in kwargs
                    and key not in (custom_objects or {})):
                warnings.warn(f"{key} was a callable that cannot be saved; pass it again with "
                              f"TEMPO.load(..., custom_objects={{'{key}': ...}})", UserWarning)
        data.update(custom_objects or {})
        for key, off in cls._REMOVED_OPTIONS.items():     # options removed on 2026-10-23
            value = data.get(key, off)
            if value is None and off is None or value is not None and off is not None and float(value) == off:
                continue
            if key == "eval_ema":
                warnings.warn(f"this model was saved with eval_ema={value!r}, an evaluation-only option removed "
                              "on 2026-10-23: its averaged policy is not loaded (training and predict are "
                              "unaffected)", UserWarning)
                continue
            raise ValueError(f"this model was saved with {key}={value!r}, a research option removed from TEMPO on "
                             "2026-10-23 (benchmarks/PROTOCOL.md); this version cannot reproduce it. Load it with "
                             "the library it was trained with.")
        pk = dict(kwargs.pop("policy_kwargs", None) or data.get("policy_kwargs") or {})
        if "activation_fn" in pk and isinstance(pk["activation_fn"], str):
            pk["activation_fn"] = getattr(torch.nn, pk["activation_fn"])
        # settings left to the defaults when the model was saved: as they resolved then
        for k, v in (data.get("resolved_policy_kwargs") or {}).items():
            pk.setdefault(k, v)
        pk = cls._infer_policy_kwargs(pk, policy_state)
        pk.setdefault("fused_critics", False)          # files before 2026-10-23: per-head critics
        hyper = {k: data[k] for k in cls._HYPERPARAMS if k in data and k != "policy_name"}
        # files from before the exploration options existed explored with white noise, no bonus
        hyper.setdefault("exploration_noise", "white")
        hyper.setdefault("exploration_bonus", 0.0)
        # files from before the speed options became the default (2026-10-23) ran the C1 engine
        for k, v in cls._LEGACY_DEFAULTS.items():
            hyper.setdefault(k, v)
        if isinstance(hyper.get("entropy_schedule"), list):
            hyper["entropy_schedule"] = tuple(hyper["entropy_schedule"])
        hyper["target_entropy"] = schedule_from_json(hyper.get("target_entropy"))
        hyper.update(kwargs)
        lr = hyper.pop("learning_rate", data.get("learning_rate"))
        policy = data.get("policy_name", "MlpPolicy")
        if policy not in cls.policy_aliases:          # saved from TEMPO(SomePolicyClass, ...)
            policy = (custom_objects or {}).get("policy_class", TempoPolicy)
        model = cls(policy, env=None,
                    learning_rate=1e-3 if lr is None else lr, policy_kwargs=pk, device=device,
                    _init_setup_model=False, **{k: v for k, v in hyper.items()
                                                if k not in ("learning_rate",)})
        if hyper.get("target_entropy") is None:
            model.target_entropy = "auto"
        model.observation_space = space_from_json(data["observation_space"])
        model.action_space = space_from_json(data["action_space"])
        model.n_envs = int(data.get("n_envs", 1))
        model._resolve_train_freq()
        model._setup_model()
        model.policy.load_state_dict(policy_state)
        if not ent_coef_overridden:                  # an explicit ent_coef beats the saved one
            with torch.no_grad():
                model.log_ent_coef.copy_(tempo_state["log_ent_coef"].to(model.device))
        model.actor_optimizer.load_state_dict(tempo_state["actor_optimizer"])
        model._load_critic_optimizer(tempo_state["critic_optimizer"])
        if model.ent_coef_optimizer is not None and "ent_coef_optimizer" in tempo_state:
            model.ent_coef_optimizer.load_state_dict(tempo_state["ent_coef_optimizer"])
        if model.rnd_optimizer is not None and "rnd_optimizer" in tempo_state:
            model.rnd_optimizer.load_state_dict(tempo_state["rnd_optimizer"])
            st = tempo_state["novelty_stats"].tolist()
            model._novelty_stats = [int(st[0]), float(st[1]), float(st[2])]
        if model.rep_optimizer is not None and "rep_optimizer" in tempo_state:
            model.rep_optimizer.load_state_dict(tempo_state["rep_optimizer"])
            model._rep_updates = int(tempo_state.get("rep_updates", torch.tensor(0)))
        model.num_timesteps = int(data.get("num_timesteps", 0))
        model._n_updates = int(data.get("n_updates", 0))
        model._episode_num = int(data.get("episode_num", 0))
        model._total_timesteps = int(data.get("total_timesteps", 0))
        model._current_progress_remaining = float(data.get("progress_remaining", 1.0))
        model._learning_starts_resolved = data.get("learning_starts_resolved")
        model._entropy_log_offset = float(data.get("entropy_log_offset", 0.0))
        model._horizon_choice = data.get("horizon_choice")
        model._audition_log = data.get("audition_log") or {}
        if model.reward_scaler is not None and data.get("reward_scaler"):
            model.reward_scaler.load_state(data["reward_scaler"])
        if env is not None:
            model.set_env(env, force_reset=force_reset)
            if model.gamma_request == "auto" and model._horizon_choice != "long":
                # resolve against the env given to load (a chosen long horizon is kept)
                model.gamma = auto_gamma(episode_horizon(model.env))
                if model.reward_scaler is not None:
                    model.reward_scaler.gamma = model.gamma
        return model

    def get_parameters(self) -> dict[str, dict]:
        params = {"policy": self.policy.state_dict(),
                  "actor.optimizer": self.actor_optimizer.state_dict(),
                  "critic.optimizer": self._critic_optimizer_state()}
        if self.ent_coef_optimizer is not None:
            params["ent_coef_optimizer"] = self.ent_coef_optimizer.state_dict()
        if self.rep_optimizer is not None:
            params["rep.optimizer"] = self.rep_optimizer.state_dict()
        if self.rnd_optimizer is not None:
            params["rnd.optimizer"] = self.rnd_optimizer.state_dict()
        return params

    def set_parameters(self, load_path_or_dict, exact_match: bool = True, device="auto") -> None:
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
            if "policy" in params:
                self.policy.load_state_dict(params["policy"], strict=exact_match)
            if "actor.optimizer" in params:
                self.actor_optimizer.load_state_dict(params["actor.optimizer"])
            if "critic.optimizer" in params:
                self._load_critic_optimizer(params["critic.optimizer"])
            if "ent_coef_optimizer" in params and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.load_state_dict(params["ent_coef_optimizer"])
            if "rep.optimizer" in params and self.rep_optimizer is not None:
                self.rep_optimizer.load_state_dict(params["rep.optimizer"])
            if "rnd.optimizer" in params and self.rnd_optimizer is not None:
                self.rnd_optimizer.load_state_dict(params["rnd.optimizer"])
        else:
            _, policy_state, _ = load_from_zip(load_path_or_dict, device="cpu")
            self.policy.load_state_dict(policy_state, strict=exact_match)
        if self._host is not None and self._host_loads != self._policy_loads():
            self._reseed_host_mirrors()          # phase engine: act / predict with the loaded weights

    def save_replay_buffer(self, path) -> None:
        path = pathlib.Path(path)
        if path.suffix != ".pkl":
            path = pathlib.Path(str(path) + ".pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.replay_buffer, f)

    def load_replay_buffer(self, path) -> None:
        """Load a replay buffer written by :meth:`save_replay_buffer` (uses pickle: trusted files only)."""
        path = pathlib.Path(path)
        if not path.exists():
            path = pathlib.Path(str(path) + ".pkl")
        with open(path, "rb") as f:
            buf = pickle.load(f)
        if isinstance(buf, DeviceReplay) != bool(self.gpu_replay):
            raise TypeError(f"a {type(buf).__name__} cannot be used with gpu_replay={self.gpu_replay}")
        buf.__dict__.pop("recent", None)          # sampling option of buffers pickled before 2026-10-23
        self.replay_buffer = buf
        if isinstance(buf, DeviceReplay):
            buf.to(self.device)
        else:
            self.replay_buffer.device = self.device
        self.replay_buffer.close_open_episodes()   # its frontier does not continue the current env

    def __repr__(self) -> str:
        return (f"TEMPO(policy={self.policy_name!r}, n_step={self.n_step}, "
                f"num_timesteps={self.num_timesteps}, device={getattr(self, 'device', None)})")
