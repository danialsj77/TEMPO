"""Network building blocks: observation normaliser, encoders, critics and actors.

Conventions
-----------
* Continuous actions live in **unit coordinates** ``x = tanh(u) in [-1, 1]^d``
  everywhere inside the agent (replay buffer, critic input, entropy). They are
  mapped to the environment's bounds only when an action is sent to the
  environment. This makes the critic input and the entropy target independent
  of the action bounds.
* Critics output ``[n_critics, B]`` (continuous) or ``[n_critics, B, |A|]``
  (discrete). Image encoders belong to the critic; the actor reads detached
  encoder features (the SAC-AE / DrQ convention).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


# --------------------------------------------------------------------------- utils
class BatchRenorm(nn.Module):
    """Batch Renormalization (Ioffe, 2017) as used by CrossQ (Bhatt et al., 2024).

    Behaves like BatchNorm for ``warmup_steps`` training forward passes, then
    normalises with running statistics corrected by clipped ``r``/``d`` factors,
    which keeps the layer stable when a minibatch is unrepresentative. In eval
    mode the running statistics are used and not updated. Adapted from the
    MIT-licensed implementation in sb3-contrib / SBX.
    """

    #: training-mode kernel: ``"autograd"`` (the layer's default: the expression below, differentiated by
    #: autograd) or ``"closed_form"`` (same function, closed-form backward: :meth:`_forward_closed_form`); set
    #: by ``TEMPO(brn_impl=...)``, whose default is ``"closed_form"``. Evaluation mode always uses the
    #: expression below.
    impl = "autograd"

    def __init__(self, num_features: int, eps: float = 1e-3, momentum: float = 0.01,
                 warmup_steps: int = 100_000, rmax: float = 3.0, dmax: float = 5.0):
        super().__init__()
        self.register_buffer("ra_mean", torch.zeros(num_features))
        self.register_buffer("ra_var", torch.ones(num_features))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long))
        self._steps: int | None = None   # host-side mirror of `steps` (avoids a device sync)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps, self.momentum, self.warmup_steps = eps, momentum, warmup_steps
        self.rmax, self.dmax = rmax, dmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.impl == "closed_form":
            return self._forward_closed_form(x)
        if self.training:
            mean, var = x.mean(0), x.var(0)
            if self._steps is None:
                self._steps = int(self.steps)
            if self._steps > self.warmup_steps:
                ra_std = (self.ra_var + self.eps).sqrt()
                r = ((var + self.eps).sqrt() / ra_std).detach().clamp(1 / self.rmax, self.rmax)
                d = ((mean - self.ra_mean) / ra_std).detach().clamp(-self.dmax, self.dmax)
                use_mean, use_var = mean - d * var.sqrt() / r, var / r.pow(2)
            else:
                use_mean, use_var = mean, var
            with torch.no_grad():
                self.ra_mean += self.momentum * (mean.detach() - self.ra_mean)
                self.ra_var += self.momentum * (var.detach() - self.ra_var)
                self.steps += 1
            self._steps += 1
        else:
            use_mean, use_var = self.ra_mean, self.ra_var
        return self.weight * (x - use_mean) / (use_var + self.eps).sqrt() + self.bias

    def _forward_closed_form(self, x: torch.Tensor) -> torch.Tensor:
        """Training-mode forward of ``impl="closed_form"`` (``TEMPO(brn_impl="closed_form")``).

        The same function as the expression in :meth:`forward` (and the same statistics and running-
        statistic updates), written as ``y = x a + c`` with per-feature float32 vectors
        ``a = w / sqrt(s2 + eps)``, ``c = b - m a`` and differentiated in closed form
        (:class:`_BRNClosedForm`): two reductions and two fused multiply-adds over the ``[N, F]``
        activations instead of autograd's chain of elementwise kernels. Only the rounding differs.
        """
        with torch.no_grad():
            var, mean = torch.var_mean(x, 0)                    # correction 1, as x.var(0)
            if self._steps is None:
                self._steps = int(self.steps)
            r = d = None
            if self._steps > self.warmup_steps:
                ra_std = (self.ra_var + self.eps).sqrt()
                r = ((var + self.eps).sqrt() / ra_std).clamp(1 / self.rmax, self.rmax)
                d = ((mean - self.ra_mean) / ra_std).clamp(-self.dmax, self.dmax)
            self.ra_mean += self.momentum * (mean.detach() - self.ra_mean)
            self.ra_var += self.momentum * (var.detach() - self.ra_var)
            self.steps += 1
        self._steps += 1
        dt = self.weight.dtype
        return _BRNClosedForm.apply(x, self.weight, self.bias, mean.to(dt), var.to(dt),
                                    None if r is None else r.to(dt), None if d is None else d.to(dt),
                                    self.eps)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._steps = None


class _BRNClosedForm(torch.autograd.Function):
    """Batch Renormalization in training mode with a closed-form backward (``brn_impl="closed_form"``).

    ``x`` is ``[..., N, F]``; the batch statistics ``(mu, v)`` (variance with correction 1) are taken
    over dim -2 by the caller and passed in, as float32 vectors ``[..., F]`` like ``w``, ``b``, ``r``
    and ``d`` (``r = d = None`` during the warm-up). With ``r`` and ``d`` constants (Ioffe's stop-
    gradient):

        warm-up : m = mu,                  s2 = v
        after   : m = mu - d sqrt(v) / r,  s2 = v / r^2
        y = x a + c,   a = w / sqrt(s2 + eps),   c = b - m a      (a, c cast to x's dtype)

    Backward with ``G = dL/dy`` (sums accumulated in float32 for bfloat16 / float16 ``G``):

        A = sum G,  B = sum G x,  C = B - A m;   dL/dw = C / sqrt(s2 + eps),  dL/db = A
        g_m = -A a,  g_s2 = -C a / (2 (s2 + eps))
        g_mu = g_m;  g_v = g_s2 (warm-up),  -g_m d / (2 r max(sqrt v, 1e-30)) + g_s2 / r^2 (after)
        dL/dx = x c1 + c0 + G a,   c1 = 2 g_v / (N - 1),   c0 = g_mu / N - c1 mu

    (``dL/dx`` is computed only when ``x`` requires a gradient.) Exact in real arithmetic; a feature
    whose batch variance is 0 after the warm-up gets a finite ``dL/dx`` where autograd's ``sqrt``
    backward gives NaN."""

    @staticmethod
    def forward(ctx, x, weight, bias, mean, var, r, d, eps):
        if r is None:
            m, s2 = mean, var
        else:
            m, s2 = mean - d * var.sqrt() / r, var / (r * r)
        inv = torch.rsqrt(s2 + eps)
        a = weight * inv
        c = bias - m * a
        ctx.save_for_backward(x, mean, var, r, d, m, s2, inv, a)
        ctx.eps = eps
        return torch.addcmul(c.to(x.dtype).unsqueeze(-2), x, a.to(x.dtype).unsqueeze(-2))

    @staticmethod
    def backward(ctx, G):
        x, mean, var, r, d, m, s2, inv, a = ctx.saved_tensors
        n = x.shape[-2]
        acc = torch.float32 if G.dtype in (torch.bfloat16, torch.float16) else G.dtype
        A = G.sum(-2, dtype=acc).to(a.dtype)
        B = (G * x).sum(-2, dtype=acc).to(a.dtype)
        C = B - A * m
        gw = C * inv if ctx.needs_input_grad[1] else None
        gb = A if ctx.needs_input_grad[2] else None
        gx = None
        if ctx.needs_input_grad[0]:
            g_m = -A * a
            g_s2 = -C * a / (2.0 * (s2 + ctx.eps))
            if r is None:
                g_v = g_s2
            else:
                g_v = -g_m * d / (2.0 * r * var.sqrt().clamp_min(1e-30)) + g_s2 / (r * r)
            c1 = 2.0 * g_v / (n - 1)
            c0 = g_m / n - c1 * mean
            gx = torch.addcmul(c0.to(x.dtype).unsqueeze(-2), x, c1.to(x.dtype).unsqueeze(-2))
            gx.addcmul_(G, a.to(G.dtype).unsqueeze(-2))
        return gx, gw, gb, None, None, None, None, None


#: the same Function over dim -2 of ``[n, N, F]`` (per-head statistics of :class:`StackedBRNMLP`)
_BRNClosedFormND = _BRNClosedForm


class StackedBRNMLP(nn.Module):
    """``n`` Batch-Renormalised MLPs of identical shape evaluated as ONE batched MLP
    (``TempoPolicy(fused_critics=True)``; the twin critics of the CrossQ layout).

    Built from the per-head ``create_mlp(..., batch_norm=True)`` modules
    (``[BRN -> Linear -> act] * (L-1) -> BRN -> Linear``), whose values it copies: Linear weights as
    ``[n, in, out]`` (applied with ``baddbmm``), biases ``[n, 1, out]``, Batch-Renorm parameters and
    running statistics ``[n, F]``, one step counter (every Batch-Renorm layer of every head sees the
    same training passes). ``forward(x [N, d]) -> [n, N, out]`` without a ``torch.stack``.

    Training mode uses the closed-form Batch-Renorm kernel (:class:`_BRNClosedFormND`, float32 sums)
    with each head's own statistics: the input layer's batch statistics are computed once from the
    shared input and each head applies its own running-statistic correction (``r``, ``d``) and affine
    parameters, so it is exact (in real arithmetic) for any running statistics. Evaluation mode computes
    :class:`BatchRenorm`'s evaluation function per head in the folded form ``y = x a + c`` (one ``addcmul``;
    same function, other rounding). The parent critic maps the state dict to and
    from the per-head layout (``critic.qs.{i}.{j}.*``), so checkpoints keep the default format.
    """

    def __init__(self, mlps):
        super().__init__()
        seqs = [list(s) for s in mlps]
        self.n = len(seqs)
        if self.n < 1:
            raise ValueError("StackedBRNMLP needs at least one MLP")
        kinds = [type(m) for m in seqs[0]]
        L = sum(1 for k in kinds if k is nn.Linear)
        expect = [BatchRenorm, nn.Linear, None] * (L - 1) + [BatchRenorm, nn.Linear]
        ok = len(kinds) == len(expect) and all(
            (k is e) if e is not None else (k not in (BatchRenorm, nn.Linear, nn.LayerNorm))
            for k, e in zip(kinds, expect))
        if not ok or any([type(m) for m in s] != kinds for s in seqs):
            raise ValueError("StackedBRNMLP needs identical [BRN -> Linear -> act]* -> BRN -> Linear MLPs "
                             "(batch_norm=True, no LayerNorm, no output activation)")
        brns = [[m for m in s if isinstance(m, BatchRenorm)] for s in seqs]
        lins = [[m for m in s if isinstance(m, nn.Linear)] for s in seqs]
        self.L = L
        ref = brns[0][0]
        for b in (b for row in brns for b in row):
            if (b.eps, b.momentum, b.warmup_steps, b.rmax, b.dmax) != \
                    (ref.eps, ref.momentum, ref.warmup_steps, ref.rmax, ref.dmax):
                raise ValueError("StackedBRNMLP: Batch-Renorm settings differ between layers")
        steps = {int(b.steps) for row in brns for b in row}
        if len(steps) != 1:
            raise ValueError("StackedBRNMLP: Batch-Renorm step counters differ between layers")
        self.eps, self.momentum, self.warmup_steps = ref.eps, ref.momentum, ref.warmup_steps
        self.rmax, self.dmax = ref.rmax, ref.dmax
        self.acts = nn.ModuleList([m for m in seqs[0] if type(m) not in (BatchRenorm, nn.Linear)])
        with torch.no_grad():
            for k in range(L):
                # registration order = the per-head parameter order (BRN weight, bias, Linear weight, bias)
                self.register_parameter(f"bn_weight{k}", nn.Parameter(
                    torch.stack([b[k].weight.detach() for b in brns]).clone()))
                self.register_parameter(f"bn_bias{k}", nn.Parameter(
                    torch.stack([b[k].bias.detach() for b in brns]).clone()))
                self.register_parameter(f"weight{k}", nn.Parameter(
                    torch.stack([l[k].weight.detach().t() for l in lins]).contiguous()))
                self.register_parameter(f"bias{k}", nn.Parameter(
                    torch.stack([l[k].bias.detach() for l in lins]).unsqueeze(1).clone()))
                self.register_buffer(f"ra_mean{k}", torch.stack([b[k].ra_mean for b in brns]).clone())
                self.register_buffer(f"ra_var{k}", torch.stack([b[k].ra_var for b in brns]).clone())
        self.register_buffer("steps", ref.steps.detach().clone())
        self._steps: int | None = None

    # ------------------------------------------------------------ per-head layout
    def head_param_names(self) -> list[str]:
        """Stacked parameter names in registration order; position ``p`` corresponds to the ``p``-th
        parameter of each per-head MLP."""
        return [f"{kind}{k}" for k in range(self.L) for kind in ("bn_weight", "bn_bias", "weight", "bias")]

    @staticmethod
    def to_head(name: str, t: torch.Tensor, i: int) -> torch.Tensor:
        """Head ``i``'s tensor (per-head layout) of the stacked tensor ``t`` named ``name``."""
        if name.startswith("weight"):
            return t[i].t().contiguous()
        if name.startswith("bias"):
            return t[i, 0].clone()
        return t[i].clone()

    @staticmethod
    def from_heads(name: str, ts) -> torch.Tensor:
        """The stacked tensor ``name`` from the per-head tensors ``ts`` (inverse of :meth:`to_head`)."""
        if name.startswith("weight"):
            return torch.stack([t.t() for t in ts]).contiguous()
        if name.startswith("bias"):
            return torch.stack(list(ts)).unsqueeze(1)
        return torch.stack(list(ts))

    # ------------------------------------------------------------ forward
    def _norm(self, k: int, x: torch.Tensor, post: bool) -> torch.Tensor:
        """Batch-Renorm layer ``k`` of every head: ``x`` is ``[N, F]`` (shared input, ``k = 0``) or
        ``[n, N, F]``; returns ``[n, N, F]``."""
        ra_mean, ra_var = getattr(self, f"ra_mean{k}"), getattr(self, f"ra_var{k}")
        w, b = getattr(self, f"bn_weight{k}"), getattr(self, f"bn_bias{k}")
        if not self.training:
            # BatchRenorm's evaluation function w (x - mu_run) / sqrt(v_run + eps) + b, folded per head into
            # y = x a + c (a = w / sqrt(v_run + eps), c = b - mu_run a; cast to x's dtype): one addcmul instead of four
            # broadcast kernels over [n, N, F]. The actor step's critic pass at batch 1024 in bf16 (MPS, 2026-10-23):
            # 1.5 ms, vs 4.0 ms with the four-kernel form and 3.1 ms for the per-head critics
            a = w * torch.rsqrt(ra_var + self.eps)
            c = b - ra_mean * a
            return torch.addcmul(c.to(x.dtype).unsqueeze(-2), x, a.to(x.dtype).unsqueeze(-2))
        shared = x.dim() == 2
        with torch.no_grad():
            var, mean = torch.var_mean(x, -2)                   # [F] (shared input) or [n, F]
            if shared:
                var, mean = var.expand_as(ra_var), mean.expand_as(ra_mean)
            r = d = None
            if post:
                ra_std = (ra_var + self.eps).sqrt()
                r = ((var + self.eps).sqrt() / ra_std).clamp(1 / self.rmax, self.rmax)
                d = ((mean - ra_mean) / ra_std).clamp(-self.dmax, self.dmax)
            ra_mean += self.momentum * (mean.detach() - ra_mean)
            ra_var += self.momentum * (var.detach() - ra_var)
        dt = w.dtype
        xs = x.expand(self.n, *x.shape) if shared else x
        return _BRNClosedFormND.apply(xs, w, b, mean.to(dt), var.to(dt),
                                      None if r is None else r.to(dt), None if d is None else d.to(dt),
                                      self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        post = False
        if self.training:
            if self._steps is None:
                self._steps = int(self.steps)
            post = self._steps > self.warmup_steps
        h = x
        for k in range(self.L):
            h = self._norm(k, h, post)
            h = torch.baddbmm(getattr(self, f"bias{k}"), h, getattr(self, f"weight{k}"))
            if k < self.L - 1:
                h = self.acts[k](h)
        if self.training:
            with torch.no_grad():
                self.steps += 1
            self._steps += 1
        return h

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._steps = None


def _stacked_state_dict_hook(critic, state_dict, prefix, local_metadata):
    """Critic state-dict hook (``fused_critics``): the stacked tensors as per-head keys
    ``{prefix}qs.{i}.{j}.*`` in the order of the per-head critic (the default checkpoint format)."""
    st = critic.__dict__["_modules"].get("stacked")
    if st is None:
        return state_dict
    sp = prefix + "stacked."
    vals = {k[len(sp):]: state_dict.pop(k).detach() for k in [k for k in state_dict if k.startswith(sp)]}
    for i in range(st.n):
        for k in range(st.L):
            j = 3 * k                                           # index of BRN k / Linear k in create_mlp
            base, lin = f"{prefix}qs.{i}.{j}.", f"{prefix}qs.{i}.{j + 1}."
            state_dict[base + "weight"] = st.to_head("bn_weight", vals[f"bn_weight{k}"], i)
            state_dict[base + "bias"] = st.to_head("bn_bias", vals[f"bn_bias{k}"], i)
            state_dict[base + "ra_mean"] = vals[f"ra_mean{k}"][i].clone()
            state_dict[base + "ra_var"] = vals[f"ra_var{k}"][i].clone()
            state_dict[base + "steps"] = vals["steps"].clone()
            state_dict[lin + "weight"] = st.to_head("weight", vals[f"weight{k}"], i)
            state_dict[lin + "bias"] = st.to_head("bias", vals[f"bias{k}"], i)
    return state_dict


def _stacked_load_pre_hook(critic, state_dict, prefix, local_metadata, strict, missing_keys,
                           unexpected_keys, error_msgs):
    """Critic load pre-hook (``fused_critics``): per-head keys ``{prefix}qs.{i}.{j}.*`` (the default
    checkpoint format) -> the stacked tensors."""
    st = critic.__dict__["_modules"].get("stacked")
    qp = prefix + "qs."
    if st is None or not any(k.startswith(qp) for k in state_dict):
        return
    sp = prefix + "stacked."
    try:
        steps = set()
        for k in range(st.L):
            j = 3 * k
            heads = [(f"{qp}{i}.{j}.", f"{qp}{i}.{j + 1}.") for i in range(st.n)]
            state_dict[f"{sp}bn_weight{k}"] = st.from_heads("bn_weight", [state_dict.pop(b + "weight") for b, _ in heads])
            state_dict[f"{sp}bn_bias{k}"] = st.from_heads("bn_bias", [state_dict.pop(b + "bias") for b, _ in heads])
            state_dict[f"{sp}ra_mean{k}"] = torch.stack([state_dict.pop(b + "ra_mean") for b, _ in heads])
            state_dict[f"{sp}ra_var{k}"] = torch.stack([state_dict.pop(b + "ra_var") for b, _ in heads])
            steps.update(int(state_dict.pop(b + "steps")) for b, _ in heads)
            state_dict[f"{sp}weight{k}"] = st.from_heads("weight", [state_dict.pop(l + "weight") for _, l in heads])
            state_dict[f"{sp}bias{k}"] = st.from_heads("bias", [state_dict.pop(l + "bias") for _, l in heads])
    except KeyError as e:
        error_msgs.append(f"fused critics: missing per-head key {e} in the state dict")
        return
    if len(steps) != 1:
        error_msgs.append(f"fused critics: the per-head Batch-Renorm step counters differ ({sorted(steps)})")
        return
    state_dict[f"{sp}steps"] = torch.tensor(steps.pop(), dtype=torch.long)
    leftover = [k for k in state_dict if k.startswith(qp)]
    for k in leftover:                                          # e.g. more heads than this model has
        unexpected_keys.append(k)
        state_dict.pop(k)


def fuse_critic_heads(critic: nn.Module) -> "StackedBRNMLP":
    """Replace ``critic.qs`` (per-head MLPs) by one :class:`StackedBRNMLP` holding the same values
    (``TempoPolicy(fused_critics=True)``); the state dict keeps the per-head format."""
    st = StackedBRNMLP(list(critic.qs))
    p = next(critic.qs.parameters())
    st.to(device=p.device)
    critic.qs = nn.ModuleList()
    critic.stacked = st
    critic._register_state_dict_hook(_stacked_state_dict_hook)
    critic._register_load_state_dict_pre_hook(_stacked_load_pre_hook, with_module=True)
    return st


def create_mlp(in_dim: int, arch, out_dim: int, activation=nn.ReLU, layer_norm: bool = False,
               output_activation=None, batch_norm: bool = False) -> nn.Sequential:
    """``in -> [Linear -> (LayerNorm) -> act] * len(arch) -> Linear -> out``.

    With ``batch_norm=True`` every Linear is preceded by :class:`BatchRenorm`
    (the CrossQ layout: ``[BRN -> Linear -> act] * len(arch) -> BRN -> Linear``).
    """
    layers, d = [], in_dim
    for h in arch:
        if batch_norm:
            layers.append(BatchRenorm(d))
        lin = nn.Linear(d, h)
        lin._tempo_hidden = True          # eligible for weight normalisation
        layers.append(lin)
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(activation())
        d = h
    if batch_norm:
        layers.append(BatchRenorm(d))
    layers.append(nn.Linear(d, out_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


@torch.no_grad()
def project_weights(module: nn.Module) -> None:
    """Rescale every hidden linear layer's weight rows to unit L2 norm.

    Used with ``weight_norm=True`` after each optimiser step (the XQC / SimbaV2
    recipe for keeping the effective learning rate constant). Only hidden layers are
    projected; output layers are left free. With Batch Renormalization before every
    linear layer (the default) the next layer's normalisation absorbs the row scale;
    without it the projection changes the function (a warning is issued).
    """
    for m in module.modules():
        if isinstance(m, nn.Linear) and getattr(m, "_tempo_hidden", False):
            m.weight.div_(m.weight.norm(dim=1, keepdim=True).clamp_min(1e-8))


def avg_l1_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """AvgL1Norm (Fujimoto et al., 2023): divide by the mean absolute value."""
    return x / x.abs().mean(-1, keepdim=True).clamp_min(eps)


class SALE(nn.Module):
    """State-Action Learned Embeddings (TD7; Fujimoto et al., 2023).

    ``zs = AvgL1Norm(f(s))`` and ``zsa = g(zs, a)`` are trained so that ``zsa``
    predicts the next state's ``zs``. TEMPO (by default) feeds ``zs`` to the actor and
    ``(zs, zsa)`` to the critic, alongside the raw features; ``zsa`` is unnormalised as
    in TD7 unless ``zsa_norm=True``.
    """

    def __init__(self, features_dim: int, action_dim: int, zs_dim: int = 256, hidden: int = 256):
        super().__init__()
        self.zs_dim = zs_dim
        self.f = nn.Sequential(nn.Linear(features_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden),
                               nn.ELU(), nn.Linear(hidden, zs_dim))
        self.g = nn.Sequential(nn.Linear(zs_dim + action_dim, hidden), nn.ELU(),
                               nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, zs_dim))

    def zs(self, features: torch.Tensor) -> torch.Tensor:
        return avg_l1_norm(self.f(features))

    def zsa(self, zs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.g(torch.cat([zs, action], -1))


class RND(nn.Module):
    """Random Network Distillation (Burda et al., 2019) novelty estimator.

    A fixed, randomly initialised target network embeds observations; a predictor is
    trained to match it. The prediction error is high on rarely seen observations and
    serves as an exploration bonus: TEMPO's tempered novelty bonus
    (``TEMPO(exploration_bonus=...)``) reads the non-image part of the observation,
    clipped to ``[-5, 5]``, and trains the predictor with Adam (learning rate 1e-4).
    """

    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.target = nn.Sequential(nn.Linear(in_dim, hidden), nn.LeakyReLU(),
                                    nn.Linear(hidden, hidden), nn.LeakyReLU(),
                                    nn.Linear(hidden, out_dim))
        self.predictor = nn.Sequential(nn.Linear(in_dim, hidden), nn.LeakyReLU(),
                                       nn.Linear(hidden, hidden), nn.LeakyReLU(),
                                       nn.Linear(hidden, hidden), nn.LeakyReLU(),
                                       nn.Linear(hidden, out_dim))
        self.target.requires_grad_(False)

    def error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample squared prediction error (differentiable in the predictor)."""
        x = x.clamp(-5.0, 5.0)
        with torch.no_grad():
            t = self.target(x)
        return (self.predictor(x) - t).pow(2).mean(-1)


class RunningNorm(nn.Module):
    """Running mean / variance observation normaliser (no gradient).

    Statistics are updated explicitly from collected observations with the
    parallel Welford merge and stored as buffers, so they are saved with the
    model and applied identically at training and prediction time.
    """

    def __init__(self, dim: int, clip: float = 10.0, eps: float = 1e-8):
        super().__init__()
        self.clip, self.eps = clip, eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, self.mean.shape[0]).to(self.mean.device, torch.float32)
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), float(x.shape[0])
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * (b_n / tot)
        m2 = self.var * self.count + b_var * b_n + delta.pow(2) * self.count * b_n / tot
        self.var.copy_(m2 / tot)
        self.count.copy_(tot)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)


class HostRunningNorm:
    """Host (numpy, float64) mirror of :class:`RunningNorm` statistics (``TEMPO(gpu_replay=True)``).

    Same parallel-Welford merge as :meth:`RunningNorm.update`, updated at every environment step on
    the host and pushed to the device module once per training phase with one non-blocking copy.
    """

    def __init__(self, module: RunningNorm):
        self.mean = module.mean.detach().cpu().double().numpy().copy()
        self.var = module.var.detach().cpu().double().numpy().copy()
        self.count = float(module.count.detach().cpu())

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.mean.shape[0])
        b_mean, b_var, b_n = x.mean(0), x.var(0), float(x.shape[0])
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean = self.mean + delta * (b_n / tot)
        m2 = self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot
        self.var = m2 / tot
        self.count = tot

    def packed(self) -> torch.Tensor:
        """``[mean, var, count]`` as one float32 CPU tensor."""
        return torch.from_numpy(np.concatenate([self.mean, self.var, [self.count]]).astype(np.float32))

    @torch.no_grad()
    def write_to(self, module: RunningNorm, non_blocking: bool = True) -> torch.Tensor:
        """Copy into ``module``; returns the host tensor, which must stay alive until the device
        has run the (non-blocking) copy."""
        packed = self.packed()
        d = self.mean.shape[0]
        module.mean.copy_(packed[:d], non_blocking=non_blocking)
        module.var.copy_(packed[d:2 * d], non_blocking=non_blocking)
        module.count.copy_(packed[2 * d], non_blocking=non_blocking)
        return packed


def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """DrQ augmentation: replicate-pad and take one random crop per sample."""
    n, _, h, w = x.shape
    x = F.pad(x, (pad,) * 4, mode="replicate")
    eps_h, eps_w = 1.0 / (h + 2 * pad), 1.0 / (w + 2 * pad)
    ar_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h + 2 * pad, device=x.device)[:h]
    ar_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w + 2 * pad, device=x.device)[:w]
    gy, gx = torch.meshgrid(ar_h, ar_w, indexing="ij")
    base = torch.stack([gx, gy], -1).unsqueeze(0).expand(n, h, w, 2)
    shift = torch.randint(0, 2 * pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
    shift = shift * torch.tensor([2.0 / (w + 2 * pad), 2.0 / (h + 2 * pad)], device=x.device)
    return F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)


class NatureCNN(nn.Module):
    """Nature-DQN convolutional trunk -> LayerNorm'd, tanh-bounded feature vector."""

    def __init__(self, image_shape, features_dim: int = 256):
        super().__init__()
        c, h, w = image_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.fc = nn.Sequential(nn.Linear(n_flat, features_dim), nn.LayerNorm(features_dim),
                                nn.Tanh())
        self.features_dim = features_dim

    def forward(self, x):
        return self.fc(self.conv(x))


class Encoder(nn.Module):
    """Concatenates the flat observation part with one CNN embedding per image."""

    def __init__(self, flat_dim: int, image_shapes: dict, cnn_features_dim: int = 256):
        super().__init__()
        self.flat_dim = flat_dim
        self.cnns = nn.ModuleDict({k: NatureCNN(s, cnn_features_dim)
                                   for k, s in image_shapes.items()})
        self.features_dim = flat_dim + cnn_features_dim * len(self.cnns)

    def forward(self, flat, images):
        parts = [] if flat is None else [flat]
        parts += [self.cnns[k](images[k]) for k in self.cnns]
        return torch.cat(parts, -1) if len(parts) > 1 else parts[0]


# ------------------------------------------------------------------------- critics
class ContinuousCritic(nn.Module):
    """``n_critics`` independent Q(s, x) networks over encoder features and unit actions.

    Returns ``[n_critics, B]`` values, or ``[n_critics, B, n_bins]`` logits when
    ``n_bins`` is set (categorical HL-Gauss critic).
    """

    def __init__(self, encoder: Encoder, act_dim: int, arch, n_critics: int = 2,
                 layer_norm: bool = True, activation=nn.ReLU, batch_norm: bool = False,
                 n_bins: int | None = None, extra_dim: int = 0):
        super().__init__()
        self.encoder = encoder
        self.n_bins = n_bins
        d = encoder.features_dim + extra_dim + act_dim
        self.qs = nn.ModuleList([create_mlp(d, arch, n_bins or 1, activation, layer_norm,
                                            batch_norm=batch_norm) for _ in range(n_critics)])
        self.stacked = None          # StackedBRNMLP after fuse_critic_heads (fused_critics=True)

    @property
    def n_critics(self) -> int:
        """Number of critic heads (per-head MLPs or the heads of the fused MLP)."""
        return self.stacked.n if self.stacked is not None else len(self.qs)

    def forward(self, flat, images, action, features=None):
        f = self.encoder(flat, images) if features is None else features
        x = torch.cat([f, action], -1)
        if self.stacked is not None:
            out = self.stacked(x)
            return out if self.n_bins else out.squeeze(-1)
        out = torch.stack([q(x) for q in self.qs])
        return out if self.n_bins else out.squeeze(-1)


class DiscreteCritic(nn.Module):
    """``n_critics`` independent Q(s, .) networks returning all action values.

    Returns ``[n_critics, B, |A|]`` values, or ``[n_critics, B, |A|, n_bins]`` logits.
    """

    def __init__(self, encoder: Encoder, n_actions: int, arch, n_critics: int = 2,
                 layer_norm: bool = True, activation=nn.ReLU, batch_norm: bool = False,
                 n_bins: int | None = None, extra_dim: int = 0):
        super().__init__()
        self.encoder = encoder
        self.n_actions, self.n_bins = n_actions, n_bins
        self.qs = nn.ModuleList([create_mlp(encoder.features_dim + extra_dim, arch,
                                            n_actions * (n_bins or 1),
                                            activation, layer_norm, batch_norm=batch_norm)
                                 for _ in range(n_critics)])
        self.stacked = None          # StackedBRNMLP after fuse_critic_heads (fused_critics=True)

    @property
    def n_critics(self) -> int:
        """Number of critic heads (per-head MLPs or the heads of the fused MLP)."""
        return self.stacked.n if self.stacked is not None else len(self.qs)

    def forward(self, flat, images, features=None):
        f = self.encoder(flat, images) if features is None else features
        if self.stacked is not None:
            out = self.stacked(f)
            if self.n_bins:
                out = out.reshape(*out.shape[:-1], self.n_actions, self.n_bins)
            return out
        out = torch.stack([q(f) for q in self.qs])
        if self.n_bins:
            out = out.reshape(*out.shape[:-1], self.n_actions, self.n_bins)
        return out


# -------------------------------------------------------------------------- actors
class GaussianActor(nn.Module):
    """tanh-squashed diagonal Gaussian over unit actions ``x in [-1, 1]^d``.

    ``log_prob`` is the density of ``x`` (Gaussian density plus the tanh
    Jacobian), i.e. it never depends on the environment's action bounds.
    """

    def __init__(self, features_dim: int, act_dim: int, arch, activation=nn.ReLU,
                 layer_norm: bool = False, batch_norm: bool = False):
        super().__init__()
        self.net = create_mlp(features_dim, arch, 2 * act_dim, activation, layer_norm,
                              batch_norm=batch_norm)
        self.act_dim = act_dim

    def dist(self, features):
        mean, log_std = self.net(features).chunk(2, -1)
        return mean, torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

    @staticmethod
    def tanh_log_det(u: torch.Tensor) -> torch.Tensor:
        """``sum_i log(1 - tanh(u_i)^2)``, numerically stable for large ``|u|``."""
        return (2.0 * (math.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)

    def forward(self, features, deterministic: bool = False, with_log_prob: bool = True,
                noise: torch.Tensor | None = None):
        """``noise``: optional standard-normal sample to use instead of fresh white noise
        (e.g. a temporally correlated pink-noise sequence for exploration)."""
        mean, log_std = self.dist(features)
        if deterministic:
            u = mean
        else:
            eps = torch.randn_like(mean) if noise is None else noise.to(mean)
            u = mean + log_std.exp() * eps
        x = torch.tanh(u)
        if not with_log_prob:
            return x, None
        # validate_args=False: the same density without the argument checks, whose
        # tensor -> bool reads stall an accelerator (no numerical difference)
        logp = (Normal(mean, log_std.exp(), validate_args=False).log_prob(u).sum(-1)
                - self.tanh_log_det(u))
        return x, logp


class CategoricalActor(nn.Module):
    def __init__(self, features_dim: int, n_actions: int, arch, activation=nn.ReLU,
                 layer_norm: bool = False, batch_norm: bool = False):
        super().__init__()
        self.net = create_mlp(features_dim, arch, n_actions, activation, layer_norm,
                              batch_norm=batch_norm)

    def forward(self, features):
        return self.net(features)  # logits
