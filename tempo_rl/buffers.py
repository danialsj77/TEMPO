"""Replay buffer with n-step returns computed at sample time.

Storage is a ring of rows, one row per vectorised environment step, holding
``[n_envs, ...]`` arrays. Each observation is stored **once**: the observation
that follows row ``i`` is row ``i + 1`` (written provisionally as soon as it is
known), except at episode ends, where the genuine final observation is kept in
a small side table. This halves memory against storing ``(obs, next_obs)``
pairs and makes n-step windows exact:

* a window never crosses an episode end or the write frontier;
* a true terminal stops bootstrapping, a time-limit truncation bootstraps from
  the genuine final observation;
* the discount of the bootstrap uses the realised horizon ``k <= n``.

Minibatch rows are drawn uniformly from the whole history.
"""
from __future__ import annotations

import numpy as np
import torch

from .preprocessing import ObservationSpec


class NStepReplayBuffer:
    def __init__(self, capacity: int, obs_spec: ObservationSpec, action_shape: tuple,
                 action_dtype, n_envs: int = 1, n_step: int = 3, gamma: float = 0.99,
                 device: torch.device | str = "cpu", seed: int | None = None):
        if capacity < 2:
            raise ValueError("replay buffer capacity must be at least 2 rows")
        self.spec = obs_spec
        self.n_envs, self.n_step, self.gamma = n_envs, n_step, gamma
        self.action_shape, self.action_dtype = tuple(action_shape), action_dtype
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.capacity = 0
        self._allocate(capacity)
        self.pos = 0          # next row to write (holds the provisional observation)
        self.n_added = 0      # rows written so far
        self.final_obs: dict[tuple[int, int], dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------ storage
    def _allocate(self, capacity: int, keep_rows: int = 0):
        def grow(old, shape, dtype):
            new = np.zeros((capacity, self.n_envs, *shape), dtype=dtype)
            if old is not None and keep_rows:
                new[:keep_rows] = old[:keep_rows]
            return new

        old = getattr(self, "obs", None)
        self.obs = {c.key: grow(None if old is None else old[c.key], c.storage_shape,
                                c.storage_dtype) for c in self.spec.components}
        self.actions = grow(getattr(self, "actions", None), self.action_shape, self.action_dtype)
        self.rewards = grow(getattr(self, "rewards", None), (), np.float32)
        self.terminated = grow(getattr(self, "terminated", None), (), bool)
        self.truncated = grow(getattr(self, "truncated", None), (), bool)
        self.capacity = capacity

    def ensure_capacity(self, rows: int, max_rows: int) -> None:
        """Grow the ring to ``min(rows, max_rows)`` rows, keeping every stored step.

        A ring that has already wrapped is first rotated into chronological order
        (oldest complete row first, provisional row last).
        """
        target = min(max(rows, 2), max_rows)
        if target <= self.capacity:
            return
        if self.n_added >= self.capacity:          # wrapped: rotate to chronological order
            cap = self.capacity
            order = (self.pos + 1 + np.arange(cap)) % cap   # oldest ... newest, provisional last
            new_index = {int(old): i for i, old in enumerate(order)}
            for k in self.obs:
                self.obs[k] = self.obs[k][order]
            self.actions, self.rewards = self.actions[order], self.rewards[order]
            self.terminated, self.truncated = self.terminated[order], self.truncated[order]
            self.final_obs = {(new_index[r], e): v for (r, e), v in self.final_obs.items()
                              if new_index[r] != cap - 1}
            self.pos, self.n_added = cap - 1, cap - 1
        self._allocate(target, keep_rows=self.n_added + 1)

    def close_open_episodes(self) -> None:
        """Mark every episode still open at the write frontier as time-limit truncated.

        Call before the environments are reset outside :meth:`add` (a new
        ``learn`` call, a new env, a reloaded buffer): the provisional
        observation at the frontier -- the genuine successor of the last stored
        step -- becomes that step's final observation, so the step still
        bootstraps correctly and no n-step window crosses into the next episode.
        """
        if self.n_added == 0:
            return
        last = (self.pos - 1) % self.capacity
        for e in range(self.n_envs):
            if not (self.terminated[last, e] or self.truncated[last, e]):
                self.truncated[last, e] = True
                self.final_obs[(last, e)] = {k: v[self.pos, e].copy() for k, v in self.obs.items()}

    def nbytes(self) -> int:
        arrays = [*self.obs.values(), self.actions, self.rewards, self.terminated, self.truncated]
        return int(sum(a.nbytes for a in arrays))

    @property
    def size(self) -> int:
        """Number of complete transitions available for sampling (per env)."""
        return min(self.n_added, self.capacity - 1)

    def add(self, obs: dict, action: np.ndarray, reward: np.ndarray, terminated: np.ndarray,
            truncated: np.ndarray, next_obs: dict, final_obs: dict[int, dict]) -> None:
        """Store one vectorised step.

        ``next_obs`` is what the environments returned (the *reset* observation
        for environments whose episode just ended); ``final_obs`` maps those
        environments' indices to their genuine last observation.
        """
        i, n = self.pos, self.capacity
        for k, v in obs.items():
            self.obs[k][i] = v
        self.actions[i] = np.asarray(action).reshape(self.n_envs, *self.action_shape)
        self.rewards[i] = reward
        self.terminated[i] = terminated
        self.truncated[i] = truncated
        for e in range(self.n_envs):
            self.final_obs.pop((i, e), None)
        for e, fo in final_obs.items():
            self.final_obs[(i, int(e))] = {k: np.array(v) for k, v in fo.items()}
        nxt = (i + 1) % n
        for k, v in next_obs.items():
            self.obs[k][nxt] = v
        self.pos = nxt
        self.n_added += 1

    # ----------------------------------------------------------- sampling
    def _sample_rows(self, batch_size: int, rng):
        if self.n_added < self.capacity:
            rows = rng.integers(0, self.n_added, size=batch_size)
        else:
            rows = (self.pos + 1 + rng.integers(0, self.capacity - 1, size=batch_size)) % self.capacity
        envs = rng.integers(0, self.n_envs, size=batch_size)
        return rows, envs

    def sample(self, batch_size: int, rng: np.random.Generator | None = None,
               with_next1: bool = False, gamma: float | None = None) -> dict:
        """Returns tensors: obs, actions, returns (n-step), discounts, next_obs, horizon.

        ``with_next1=True`` adds ``next1_obs``: the observation one step later (the
        genuine final observation at an episode end), for one-step model losses.
        ``gamma`` overrides the buffer's discount for this sample (a second learner with
        another horizon can share the buffer).
        """
        g = self.gamma if gamma is None else gamma
        if self.size < 1:
            raise RuntimeError("cannot sample from an empty replay buffer")
        rng = rng or self.rng
        rows, envs = self._sample_rows(batch_size, rng)
        n = self.capacity
        term_, trunc_ = self.terminated, self.truncated

        def done(r):                                # episode end at (r, envs): only where read
            return term_[r, envs] | trunc_[r, envs]

        newer = (self.pos - 1 - rows) % n          # complete rows written after `rows`

        ret = self.rewards[rows, envs].astype(np.float64)
        horizon = np.ones(batch_size, dtype=np.int64)
        last = rows.copy()
        done_rows = done(rows)
        alive = ~done_rows
        for j in range(1, self.n_step):
            can = alive & (newer >= j)
            nxt = (rows + j) % n
            ret += np.where(can, g ** j * self.rewards[nxt, envs], 0.0)
            horizon += can
            last = np.where(can, nxt, last)
            alive = can & ~done(nxt)

        term = self.terminated[last, envs]
        discounts = (g ** horizon) * (1.0 - term)
        boot = (last + 1) % n
        next_obs = {k: v[boot, envs] for k, v in self.obs.items()}
        ended = np.flatnonzero(done(last))
        for b in ended:
            fo = self.final_obs.get((int(last[b]), int(envs[b])))
            if fo is not None:
                for k in next_obs:
                    next_obs[k][b] = fo[k]

        out = {}
        if with_next1:
            b1 = (rows + 1) % n
            next1 = {k: v[b1, envs] for k, v in self.obs.items()}
            for b in np.flatnonzero(done_rows):
                fo = self.final_obs.get((int(rows[b]), int(envs[b])))
                if fo is not None:
                    for k in next1:
                        next1[k][b] = fo[k]
            out["next1_obs"] = next1
        out.update(obs={k: v[rows, envs] for k, v in self.obs.items()},
                   actions=self.actions[rows, envs], returns=ret, discounts=discounts,
                   next_obs=next_obs, horizon=horizon)
        return _to_tensors(out, self.device)


#: dtype of each sampled field on the device (observations and actions keep their storage dtype)
_FLOAT_FIELDS = ("returns", "discounts", "horizon")


def _to_tensors(fields: dict, device: torch.device) -> dict:
    """Sampled numpy fields -> tensors on ``device``.

    CPU: ``torch.as_tensor`` per field. Accelerator: every field is packed byte-for-byte into ONE
    host buffer, copied with one non-blocking transfer, and viewed back with its dtype on the
    device (the values are the same bits as separate ``torch.as_tensor`` copies; one transfer
    instead of seven, and no wait for the device queue).
    """
    if device.type == "cpu":
        def t(x, dtype=None):
            return torch.as_tensor(x, device=device, dtype=dtype)
        out = {k: ({kk: t(vv) for kk, vv in v.items()} if isinstance(v, dict)
                   else t(v, torch.float32 if k in _FLOAT_FIELDS else None))
               for k, v in fields.items()}
        return out
    from . import _sync

    flat, spec, off = [], [], 0
    for k, v in fields.items():
        items = v.items() if isinstance(v, dict) else [(None, v)]
        for kk, a in items:
            a = np.ascontiguousarray(a, dtype=np.float32 if k in _FLOAT_FIELDS else None)
            pad = (-off) % 8                         # 8-byte aligned views (int64 fields)
            if pad:
                flat.append(np.zeros(pad, np.uint8))
                off += pad
            spec.append((k, kk, off, a.dtype, a.shape))
            flat.append(a.reshape(-1).view(np.uint8))
            off += a.nbytes
    host = torch.from_numpy(np.concatenate(flat))
    staged = torch.empty(host.shape, dtype=torch.uint8, device=device)
    staged.copy_(host, non_blocking=True)
    _sync.keep(host, device)
    out: dict = {}
    for k, kk, o, dt, shape in spec:
        nbytes = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
        tensor = staged[o:o + nbytes].view(_TORCH_DTYPE[dt.type]).view(shape)
        if kk is None:
            out[k] = tensor
        else:
            out.setdefault(k, {})[kk] = tensor
    return out


_TORCH_DTYPE = {np.float32: torch.float32, np.float64: torch.float64, np.int64: torch.int64,
                np.int32: torch.int32, np.uint8: torch.uint8, np.bool_: torch.bool}


class DeviceReplay:
    """Replay ring that lives on the training device (``TEMPO(gpu_replay=True)``).

    One row per vector step holds ``[n_envs, ...]`` complete transitions: observation, action,
    reward, ``terminated``, ``truncated`` and the *genuine* next observation (the final observation
    at an episode end). Rows arrive in blocks with one packed non-blocking host->device copy
    (:meth:`add_rows`); minibatch indices come from a device random generator and the n-step
    windows are gathered on the device with exactly the rules of :class:`NStepReplayBuffer`: a
    window never crosses an episode end or the write frontier, a terminal stops bootstrapping, a
    truncation bootstraps from the genuine final observation, and the bootstrap discount uses the
    realised horizon ``k <= n``. Not bit-identical to :class:`NStepReplayBuffer`: other random
    indices (device generator) and float32 instead of float64 n-step sums.
    """

    #: ``torch.randint`` on MPS recompiles its kernel for every new ``high`` (~1.7 ms per call as
    #: the buffer grows), so indices are drawn from the fixed range [0, 2^31 - 1) and reduced
    #: modulo the current size (relative bias <= size / 2^31, i.e. <= 0.05 % at 1M rows).
    _RNG_HIGH = (1 << 31) - 1

    def __init__(self, capacity: int, obs_spec: ObservationSpec, action_shape: tuple, discrete: bool,
                 n_envs: int = 1, n_step: int = 3, gamma: float = 0.99,
                 device: torch.device | str = "cpu", seed: int | None = None):
        self.spec = obs_spec
        self.n_envs, self.n_step, self.gamma = int(n_envs), int(n_step), float(gamma)
        self.action_shape, self.discrete = tuple(action_shape), bool(discrete)
        self.device = torch.device(device)
        self.capacity = 0
        self.pos = 0
        self.n_added = 0
        self._allocate(max(int(capacity), 2))
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(0 if seed is None else int(seed))
        if seed is None:
            self.gen.seed()
        self._gpow = None

    # ------------------------------------------------------------ storage
    _TDT = {np.float32: torch.float32, np.int64: torch.int64, np.uint8: torch.uint8}

    def _fields(self):
        yield from (("obs", k) for k in self.obs)
        yield from (("next_obs", k) for k in self.next_obs)
        yield from (("actions", None), ("rewards", None), ("terminated", None), ("truncated", None))

    def _allocate(self, capacity: int, keep_order=None) -> None:
        dev, n = self.device, self.n_envs

        def z(shape, dtype):
            return torch.zeros((capacity, n, *shape), dtype=dtype, device=dev)

        old = None if self.capacity == 0 else dict(
            obs=self.obs, next_obs=self.next_obs, actions=self.actions, rewards=self.rewards,
            terminated=self.terminated, truncated=self.truncated)
        comps = self.spec.components
        self.obs = {c.key: z(c.storage_shape, self._TDT[c.storage_dtype]) for c in comps}
        self.next_obs = {c.key: z(c.storage_shape, self._TDT[c.storage_dtype]) for c in comps}
        self.actions = (z((), torch.int64) if self.discrete
                        else z(self.action_shape, torch.float32))
        self.rewards = z((), torch.float32)
        self.terminated = z((), torch.bool)
        self.truncated = z((), torch.bool)
        if old is not None and keep_order is not None and len(keep_order):
            idx = torch.as_tensor(keep_order, device=dev)
            k = len(keep_order)
            for name in ("obs", "next_obs"):
                for key, v in old[name].items():
                    getattr(self, name)[key][:k] = v[idx]
            for name in ("actions", "rewards", "terminated", "truncated"):
                getattr(self, name)[:k] = old[name][idx]
        self.capacity = capacity

    def ensure_capacity(self, rows: int, max_rows: int) -> None:
        """Grow the ring to ``min(rows, max_rows)`` rows, keeping every stored transition in
        chronological order."""
        target = min(max(int(rows), 2), int(max_rows))
        if target <= self.capacity:
            return
        if self.n_added > self.capacity:                 # wrapped: oldest row is at pos
            order = (self.pos + np.arange(self.capacity)) % self.capacity
        else:
            order = np.arange(self.n_added)
        self._allocate(target, keep_order=order)
        self.pos = len(order) % target
        self.n_added = len(order)

    def close_open_episodes(self) -> None:
        """Mark the episodes still open at the write frontier as time-limit truncated (their last
        row already holds the genuine next observation, so it still bootstraps correctly)."""
        if self.n_added == 0:
            return
        last = (self.pos - 1) % self.capacity
        self.truncated[last].logical_or_(~self.terminated[last])

    def nbytes(self) -> int:
        ts = [*self.obs.values(), *self.next_obs.values(), self.actions, self.rewards,
              self.terminated, self.truncated]
        return int(sum(t.numel() * t.element_size() for t in ts))

    @property
    def size(self) -> int:
        """Number of complete transitions available for sampling (per env)."""
        return min(self.n_added, self.capacity)

    def add_rows(self, obs: dict, actions, rewards, terms, truncs, next_obs: dict) -> list:
        """Append ``k`` rows given as host arrays of shape ``[k, n_envs, ...]``.

        Every field is packed into ONE float32 host array and sent with one *non-blocking* copy
        (integer, boolean and uint8 fields are exact in float32). The returned host tensors must
        stay alive until a device event recorded after this call has completed.
        """
        k = int(np.asarray(rewards).shape[0])
        start = self.pos
        dsts, cols = [], []
        for key in obs:
            dsts += [self.obs[key], self.next_obs[key]]
            cols += [obs[key], next_obs[key]]
        dsts += [self.actions, self.rewards, self.terminated, self.truncated]
        cols += [actions, rewards, terms, truncs]
        widths = [int(np.prod(d.shape[2:], dtype=np.int64)) for d in dsts]
        packed = np.concatenate([np.asarray(c, dtype=np.float32).reshape(k, self.n_envs, w)
                                 for c, w in zip(cols, widths)], axis=-1)
        host = torch.from_numpy(np.ascontiguousarray(packed))
        staged = torch.empty(host.shape, dtype=torch.float32, device=self.device)
        staged.copy_(host, non_blocking=True)
        rows = (start + torch.arange(k, device=self.device)) % self.capacity if k > self.capacity \
            else None
        first = min(k, self.capacity - start)
        off = 0
        for dst, w in zip(dsts, widths):
            part = staged[..., off:off + w].reshape(k, *dst.shape[1:])
            if dst.dtype == torch.bool:
                part = part > 0.5
            elif dst.dtype != torch.float32:
                part = part.round().to(dst.dtype)
            if rows is not None:                         # block longer than the ring: keep the last
                dst[rows[-self.capacity:]] = part[-self.capacity:]
            else:
                dst[start:start + first].copy_(part[:first])
                if first < k:
                    dst[0:k - first].copy_(part[first:])
            off += w
        self.pos = int((self.pos + k) % self.capacity)
        self.n_added += k
        return [host, staged]

    # ----------------------------------------------------------- sampling
    def _powers(self, gamma: float) -> torch.Tensor:
        if self._gpow is None or self._gpow[0] != gamma:
            self._gpow = (gamma, torch.tensor([gamma ** j for j in range(self.n_step + 1)],
                                              dtype=torch.float32, device=self.device))
        return self._gpow[1]

    def sample_indices(self, batch_size: int):
        size = self.size
        r = torch.randint(0, self._RNG_HIGH, (batch_size,), device=self.device, generator=self.gen)
        if self.n_added <= self.capacity:
            rows = r % size
        else:                                            # wrapped: the oldest row is at pos
            rows = (self.pos + r % self.capacity) % self.capacity
        envs = torch.randint(0, self.n_envs, (batch_size,), device=self.device, generator=self.gen)
        return rows, envs

    def gather(self, rows: torch.Tensor, envs: torch.Tensor, with_next1: bool = False,
               gamma: float | None = None) -> dict:
        g = self.gamma if gamma is None else float(gamma)
        gpow = self._powers(g)
        n = self.capacity
        term_, trunc_ = self.terminated, self.truncated
        newer = (self.pos - 1 - rows) % n                 # complete rows written after `rows`
        ret = self.rewards[rows, envs].clone()
        horizon = torch.ones_like(ret)
        last = rows.clone()
        alive = ~(term_[rows, envs] | trunc_[rows, envs])
        for j in range(1, self.n_step):
            can = alive & (newer >= j)
            nxt = (rows + j) % n
            ret = ret + torch.where(can, gpow[j] * self.rewards[nxt, envs], 0.0)
            horizon = horizon + can.float()
            last = torch.where(can, nxt, last)
            alive = can & ~(term_[nxt, envs] | trunc_[nxt, envs])
        term = term_[last, envs]
        discounts = torch.pow(g, horizon) * (~term).float()
        out = dict(obs={k: v[rows, envs] for k, v in self.obs.items()},
                   actions=self.actions[rows, envs], returns=ret, discounts=discounts,
                   next_obs={k: v[last, envs] for k, v in self.next_obs.items()},
                   horizon=horizon)
        if with_next1:
            out["next1_obs"] = {k: v[rows, envs] for k, v in self.next_obs.items()}
        return out

    def sample(self, batch_size: int, rng=None, with_next1: bool = False,
               gamma: float | None = None) -> dict:
        """Same fields as :meth:`NStepReplayBuffer.sample`, as device tensors (``rng`` is
        ignored: the indices come from the buffer's device generator)."""
        if self.size < 1:
            raise RuntimeError("cannot sample from an empty replay buffer")
        if self.gen is None:                             # unpickled and not moved yet
            self.to(self.device)
        rows, envs = self.sample_indices(batch_size)
        return self.gather(rows, envs, with_next1, gamma)

    # --------------------------------------------------------- persistence
    def to(self, device) -> "DeviceReplay":
        """Move the ring (and its random generator state) to ``device``."""
        device = torch.device(device)
        for name in ("obs", "next_obs"):
            setattr(self, name, {k: v.to(device) for k, v in getattr(self, name).items()})
        for name in ("actions", "rewards", "terminated", "truncated"):
            setattr(self, name, getattr(self, name).to(device))
        state = self.__dict__.pop("_gen_state", None)
        old = self.__dict__.get("gen")
        if state is None and old is not None:
            state = old.get_state()
        self.gen = torch.Generator(device=device)
        if state is not None:
            try:
                self.gen.set_state(state)
            except (RuntimeError, TypeError, ValueError):     # state of another generator type
                self.gen.manual_seed(int(state.to(torch.int64).sum()) % (1 << 31))
        self.device = device
        self._gpow = None
        return self

    def __getstate__(self):
        st = dict(self.__dict__)
        for name in ("obs", "next_obs"):
            st[name] = {k: v.cpu() for k, v in st[name].items()}
        for name in ("actions", "rewards", "terminated", "truncated"):
            st[name] = st[name].cpu()
        st["_gen_state"] = self.gen.get_state()
        st["gen"] = None
        st["_gpow"] = None
        st["device"] = torch.device("cpu")
        return st

    def __setstate__(self, st):
        self.__dict__.update(st)       # CPU tensors; TEMPO.load_replay_buffer calls .to(device)
