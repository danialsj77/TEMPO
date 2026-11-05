"""The TEMPO policy: observation preprocessing, actor, twin critics, learned embeddings.

It also holds the novelty estimator of the exploration bonus when :class:`~tempo_rl.TEMPO`
asks for one, and target critics only in the LayerNorm configuration (``batch_norm=False``).

``MlpPolicy``, ``CnnPolicy`` and ``MultiInputPolicy`` are accepted as names for
compatibility with Stable-Baselines3; TEMPO builds the right feature extractor
from the observation space in every case (``"auto"`` works too).
"""
from __future__ import annotations

import copy
import warnings

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from .networks import (RND, SALE, CategoricalActor, ContinuousCritic, DiscreteCritic, Encoder,
                       GaussianActor, RunningNorm, avg_l1_norm, fuse_critic_heads, project_weights,
                       random_shift)
from ._sync import synced, to_device
from .preprocessing import ObservationSpec

POLICY_ALIASES = ("MlpPolicy", "CnnPolicy", "MultiInputPolicy", "auto")

#: default widths: a 256-wide actor and 1024-wide critics (CrossQ's critic width)
DEFAULT_ARCH = dict(pi=[256, 256], qf=[1024, 1024])


class TempoPolicy(nn.Module):
    """Actor + ``n_critics`` critics (+ Polyak targets when ``batch_norm=False``) over any supported observation space.

    :param net_arch: hidden layers, as a list shared by actor and critics (e.g.
        ``[256, 256]``) or ``dict(pi=[...], qf=[...])``. ``None`` uses ``DEFAULT_ARCH``:
        a 256-unit actor and 1024-unit critics.
    :param normalize_observations: normalise ``Box`` vector observations with
        running statistics (saved with the model, frozen at prediction).
    :param critic_layer_norm: LayerNorm after every critic hidden layer (ignored with
        ``batch_norm=True``, the default).
    :param actor_layer_norm: LayerNorm after every actor hidden layer.
    :param augment_images: DrQ random-shift augmentation of image observations.
    :param batch_norm: Batch-Renormalization layers in actor and critic (CrossQ
        layout). The critic is then trained with a joint forward pass over current
        and bootstrap state-actions and **without target networks**.
    :param n_bins: if set, critics output logits over ``n_bins`` value bins
        (categorical HL-Gauss critic); set by :class:`~tempo_rl.TEMPO` from
        ``critic_loss``, not by hand.
    :param representation: ``"sale"`` (default) or ``None``: learn state and state-action
        embeddings by predicting the next state's embedding (TD7's SALE) and feed them
        to the actor (``zs``) and the critic (``zs`` and, for continuous actions,
        ``zsa``).
    :param zs_dim: embedding width for ``representation="sale"``.
    :param novelty: build a Random-Network-Distillation novelty estimator over the
        vector part of the observation; set by :class:`~tempo_rl.TEMPO` from
        ``exploration_bonus``, not by hand.
    :param zsa_norm: apply AvgL1Norm to the state-action embedding fed to the critic
        (off by default, as in TD7, whose ablation found it harmful).
    :param weight_norm: project every hidden linear layer to unit-norm weight rows
        after each optimiser step (XQC-style); intended for several gradient steps per
        environment step. Off by default.
    :param fused_critics: (speed option, 2026-10-23; off by default here, but :class:`~tempo_rl.TEMPO`
        turns it on whenever its ``brn_impl="closed_form"`` (its default) and the critics are
        Batch-Renormalised without ``weight_norm``, unless ``policy_kwargs`` sets it) evaluate the
        ``n_critics`` Batch-Renormalised critic MLPs as one batched MLP
        (:class:`~tempo_rl.networks.StackedBRNMLP`: ``baddbmm`` over ``[n, in, out]`` weights, one
        pass instead of ``n`` plus a ``torch.stack``) with the closed-form Batch-Renorm kernel in
        training mode and the folded form ``y = x a + c`` of the evaluation function. The per-head modules are built first exactly as without the option (same
        random draws) and copied, so the starting weights are the same; the state dict keeps the
        per-head format (``critic.qs.{i}.{j}.*``). Same function, different rounding. Needs
        ``batch_norm=True``; not with ``weight_norm``. :class:`~tempo_rl.TEMPO` requires
        ``brn_impl="closed_form"`` with it.
    """

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
                 net_arch=None, activation_fn=nn.ReLU, n_critics: int = 2,
                 normalize_observations: bool = True, critic_layer_norm: bool = True,
                 actor_layer_norm: bool = False, cnn_features_dim: int = 256,
                 augment_images: bool = True, obs_clip: float = 10.0, batch_norm: bool = True,
                 n_bins: int | None = None, representation: str | None = "sale",
                 zs_dim: int = 256, weight_norm: bool = False, zsa_norm: bool = False,
                 novelty: bool = False, fused_critics: bool = False):
        super().__init__()
        if fused_critics and not batch_norm:
            raise ValueError("fused_critics=True needs batch_norm=True (the CrossQ critics)")
        if fused_critics and weight_norm:
            raise ValueError("fused_critics=True does not support weight_norm=True")
        self.observation_space, self.action_space = observation_space, action_space
        self.spec = ObservationSpec(observation_space)
        self.discrete = isinstance(action_space, spaces.Discrete)
        arch = DEFAULT_ARCH if net_arch is None else net_arch
        if isinstance(arch, (list, tuple)):
            arch = dict(pi=list(arch), qf=list(arch))
        self.net_arch = dict(pi=list(arch["pi"]), qf=list(arch["qf"]))
        self.augment_images = augment_images
        self.batch_norm = batch_norm
        self.n_bins = n_bins
        self.weight_norm = weight_norm
        if representation not in (None, "sale"):
            raise ValueError(f"representation must be None or 'sale', got {representation!r}")
        if weight_norm and not batch_norm:
            warnings.warn("weight_norm=True without batch_norm: LayerNorm / unnormalised layers are "
                          "not invariant to per-row rescaling, so the projection changes the "
                          "function", UserWarning, stacklevel=2)
        self.zsa_norm = zsa_norm
        if novelty and self.spec.flat_dim == 0:
            raise ValueError("exploration_bonus needs a non-image observation component")
        self.rnd = RND(self.spec.flat_dim) if novelty else None
        self.representation = representation
        if batch_norm:
            critic_layer_norm = False

        self.normalizer = (RunningNorm(self.spec.vector_dim, clip=obs_clip)
                           if normalize_observations and self.spec.vector_dim > 0 else None)
        images = {c.key: c.image_shape for c in self.spec.components if c.kind == "image"}
        encoder = Encoder(self.spec.flat_dim, images, cnn_features_dim)
        fdim = encoder.features_dim
        n_act_in = int(action_space.n) if self.discrete else int(action_space.shape[0])
        extra_actor = extra_critic = 0
        self.rep = self.rep_fixed = self.rep_fixed_target = None
        if representation == "sale":
            self.rep = SALE(fdim, n_act_in, zs_dim)
            self.rep_fixed = copy.deepcopy(self.rep)     # periodically refreshed copy (TD7)
            self.rep_fixed.requires_grad_(False)
            extra_actor = zs_dim
            extra_critic = zs_dim if self.discrete else 2 * zs_dim
        if self.discrete:
            n = int(action_space.n)
            self.critic = DiscreteCritic(encoder, n, self.net_arch["qf"], n_critics,
                                         critic_layer_norm, activation_fn, batch_norm, n_bins,
                                         extra_dim=extra_critic)
            self.actor = CategoricalActor(fdim + extra_actor, n, self.net_arch["pi"], activation_fn,
                                          actor_layer_norm, batch_norm)
            self.act_dim = n
        else:
            d = int(action_space.shape[0])
            self.critic = ContinuousCritic(encoder, d, self.net_arch["qf"], n_critics,
                                           critic_layer_norm, activation_fn, batch_norm, n_bins,
                                           extra_dim=extra_critic)
            self.actor = GaussianActor(fdim + extra_actor, d, self.net_arch["pi"], activation_fn,
                                       actor_layer_norm, batch_norm)
            low = np.asarray(action_space.low, dtype=np.float32)
            high = np.asarray(action_space.high, dtype=np.float32)
            self.register_buffer("action_scale", torch.as_tensor((high - low) / 2.0))
            self.register_buffer("action_bias", torch.as_tensor((high + low) / 2.0))
            self.act_dim = d
        if weight_norm:
            project_weights(self.critic)
            project_weights(self.actor)
        # fused_critics: the critic heads as one batched MLP (built from the per-head modules above, so the
        # random draws and the starting values are those of per-head critics; no draw happens here)
        self.fused_critics = bool(fused_critics)
        if self.fused_critics:
            fuse_critic_heads(self.critic)
        # Batch-renormalised critics bootstrap from themselves (no target network).
        self.critic_target = None if batch_norm else copy.deepcopy(self.critic)
        if self.critic_target is not None:
            self.critic_target.requires_grad_(False)
            if self.rep_fixed is not None:
                # the Polyak target critics read an embedding that lags at the same rate
                self.rep_fixed_target = copy.deepcopy(self.rep_fixed)

    # ------------------------------------------------------------ plumbing
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def obs_to_tensor(self, observation):
        """Raw observation(s) -> (storage tensors on device, batched flag)."""
        batched = self.spec.is_batched(observation)
        stored = self.spec.to_storage(observation, batched)
        dev = self.device
        return {k: to_device(v, dev) for k, v in stored.items()}, batched

    def preprocess(self, stored, augment: bool = False):
        flat, images = self.spec.to_tensors(stored, self.normalizer)
        if augment and self.augment_images and images:
            images = {k: random_shift(v) for k, v in images.items()}
        return flat, images

    def actor_features(self, flat, images) -> torch.Tensor:
        """Actor input: detached encoder features (plus ``zs`` with a representation)."""
        enc = self.critic.encoder(flat, images).detach()
        return self.actor_input(enc, self.state_embedding(enc))

    # ------------------------------------------------------- representation
    def rep_action(self, action: torch.Tensor) -> torch.Tensor:
        """Action as fed to the state-action embedding (one-hot for discrete actions)."""
        if self.discrete:
            return torch.nn.functional.one_hot(action.long().reshape(-1), self.act_dim).float()
        return action

    def _fixed(self, target: bool):
        return self.rep_fixed_target if target and self.rep_fixed_target is not None \
            else self.rep_fixed

    @torch.no_grad()
    def state_embedding(self, enc: torch.Tensor, target: bool = False):
        """``zs`` from the fixed embedding (``None`` without a representation).

        ``target=True`` uses the copy that lags with the Polyak target critics.
        """
        return None if self.rep_fixed is None else self._fixed(target).zs(enc.detach())

    def actor_input(self, enc: torch.Tensor, zs) -> torch.Tensor:
        enc = enc.detach()
        return enc if zs is None else torch.cat([enc, zs], -1)

    def critic_input(self, enc: torch.Tensor, zs, action=None, target: bool = False) -> torch.Tensor:
        """Critic features: encoder output (+ ``zs`` and, continuous, ``zsa``; ``AvgL1Norm(zsa)``
        with ``zsa_norm=True``)."""
        if zs is None:
            return enc
        if self.discrete:
            return torch.cat([enc, zs], -1)
        # rep_fixed has no trainable parameters: gradients reach only the action (as in TD7)
        zsa = self._fixed(target).zsa(zs, action)
        if self.zsa_norm:
            zsa = avg_l1_norm(zsa)
        return torch.cat([enc, zs, zsa], -1)

    def _host_action_affine(self):
        """``(scale, bias)`` as host arrays: the buffers' values, read once and cached (reading a
        device buffer at every step would wait for the device; ``load_state_dict`` refreshes it)."""
        cached = self.__dict__.get("_action_affine_np")
        if cached is None:
            cached = (self.action_scale.cpu().numpy(), self.action_bias.cpu().numpy())
            self.__dict__["_action_affine_np"] = cached
        return cached

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.__dict__.pop("_action_affine_np", None)
        # number of state loads (load_state_dict on this policy): TEMPO's phase engine compares it to
        # refresh its CPU copies of the actor and the normaliser (act_on_cpu, predict)
        self.__dict__["_load_count"] = self.__dict__.get("_load_count", 0) + 1

    def unscale_action(self, unit: np.ndarray) -> np.ndarray:
        """Unit action ``[-1, 1]^d`` -> environment action."""
        scale, bias = self._host_action_affine()
        return np.clip(unit * scale + bias, self.action_space.low, self.action_space.high)

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """Environment action -> unit action ``[-1, 1]^d``."""
        scale, bias = self._host_action_affine()
        return np.clip((np.asarray(action, dtype=np.float32) - bias) / scale, -1.0, 1.0)

    # -------------------------------------------------------------- acting
    @torch.no_grad()
    def act(self, stored, deterministic: bool = False, noise=None) -> torch.Tensor:
        """Unit actions (continuous) or action indices (discrete) for a batch."""
        flat, images = self.preprocess(stored, augment=False)
        enc = self.critic.encoder(flat, images)
        feats = self.actor_input(enc, self.state_embedding(enc))
        if self.discrete:
            logits = self.actor(feats)
            if deterministic:
                return logits.argmax(-1)
            # validate_args=False: same sample without the argument checks (no device sync)
            return torch.distributions.Categorical(logits=logits, validate_args=False).sample()
        x, _ = self.actor(feats, deterministic=deterministic, with_log_prob=False, noise=noise)
        return x

    def predict(self, observation, state=None, episode_start=None, deterministic: bool = False):
        """SB3-compatible ``predict``: returns ``(actions, state)``.

        Accepts one observation or a batch; returns environment actions (bounded
        ``Box`` actions are rescaled from the policy's unit cube).
        """
        was_training = self.training
        self.train(False)
        try:
            stored, batched = self.obs_to_tensor(observation)
            dev = self.device
            actions = self.act(stored, deterministic=deterministic).cpu().numpy()
            if dev.type != "cpu":           # the .cpu() above waited for every copy queued on dev
                synced(dev, check=False)
        finally:
            self.train(was_training)
        if not self.discrete:
            actions = self.unscale_action(actions)
        if not batched:
            actions = actions.squeeze(axis=0)     # SB3: 0-d array (Discrete) / (d,) (Box)
        return actions, state
