"""Observation-space handling.

TEMPO accepts every observation space Gymnasium's built-in environments use:
``Box`` vectors, ``Box`` images (uint8, channel-first or channel-last), ``Discrete``,
``MultiDiscrete``, ``MultiBinary``, and ``Dict``/``Tuple`` combinations of them.

Two representations are involved:

* **storage** -- what the replay buffer keeps: a dict of numpy arrays, one per
  component, in the most compact dtype (uint8 images, int64 indices, float32
  vectors);
* **network input** -- what :func:`ObservationSpec.to_tensors` produces: every
  non-image component flattened to float (one-hot for discrete components) and
  concatenated into one ``flat`` tensor; every image component as a float
  ``[B, C, H, W]`` tensor in ``[0, 1]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces


def is_image_space(space: spaces.Space) -> bool:
    """A ``Box`` with uint8 dtype, three dimensions and bounds [0, 255]."""
    if not isinstance(space, spaces.Box) or space.dtype != np.uint8 or len(space.shape) != 3:
        return False
    return bool(np.all(space.low == 0) and np.all(space.high == 255))


def image_is_channels_first(shape) -> bool:
    """Heuristic used by SB3 as well: the channel axis is the smallest one."""
    c_first, c_last = shape[0], shape[-1]
    if c_first in (1, 3, 4) and c_last not in (1, 3, 4):
        return True
    if c_last in (1, 3, 4) and c_first not in (1, 3, 4):
        return False
    return c_first < c_last


@dataclass
class Component:
    key: str
    kind: str             # "vector" | "image" | "discrete" | "multidiscrete" | "multibinary"
    space: spaces.Space
    storage_shape: tuple
    storage_dtype: type
    flat_dim: int = 0     # contribution to the flat vector (0 for images)
    image_shape: tuple = ()  # channel-first (C, H, W) for images


def _component(key: str, space: spaces.Space) -> Component:
    if isinstance(space, spaces.Box):
        if is_image_space(space):
            shape = tuple(space.shape)
            chw = shape if image_is_channels_first(shape) else (shape[2], shape[0], shape[1])
            return Component(key, "image", space, shape, np.uint8, 0, chw)
        dim = int(np.prod(space.shape))
        return Component(key, "vector", space, (dim,), np.float32, dim)
    if isinstance(space, spaces.Discrete):
        return Component(key, "discrete", space, (), np.int64, int(space.n))
    if isinstance(space, spaces.MultiDiscrete):
        nvec = np.asarray(space.nvec).ravel()
        return Component(key, "multidiscrete", space, (len(nvec),), np.int64, int(nvec.sum()))
    if isinstance(space, spaces.MultiBinary):
        dim = int(np.prod(space.shape))
        return Component(key, "multibinary", space, (dim,), np.float32, dim)
    raise NotImplementedError(
        f"TEMPO does not support observation space {space!r}. Supported: Box, Discrete, "
        "MultiDiscrete, MultiBinary, and Dict/Tuple combinations of them.")


class ObservationSpec:
    """Describes how one observation space is stored and fed to the networks."""

    def __init__(self, observation_space: spaces.Space):
        self.observation_space = observation_space
        self.is_dict = isinstance(observation_space, (spaces.Dict, spaces.Tuple))
        if isinstance(observation_space, spaces.Dict):
            items = list(observation_space.spaces.items())
        elif isinstance(observation_space, spaces.Tuple):
            items = [(str(i), s) for i, s in enumerate(observation_space.spaces)]
        else:
            items = [("obs", observation_space)]
        for key, sub in items:
            if isinstance(sub, (spaces.Dict, spaces.Tuple)):
                raise NotImplementedError("nested Dict/Tuple observation spaces are not supported")
        self.components = [_component(k, s) for k, s in items]
        self.vector_dim = sum(c.flat_dim for c in self.components if c.kind == "vector")
        self.flat_dim = sum(c.flat_dim for c in self.components)
        self.image_keys = [c.key for c in self.components if c.kind == "image"]
        self.has_images = bool(self.image_keys)

    # ----------------------------------------------------------------- numpy
    def _items(self, obs):
        if isinstance(self.observation_space, spaces.Dict):
            return [(c, obs[c.key]) for c in self.components]
        if isinstance(self.observation_space, spaces.Tuple):
            return [(c, obs[int(c.key)]) for c in self.components]
        return [(self.components[0], obs)]

    def is_batched(self, obs) -> bool:
        """True if ``obs`` carries a leading batch dimension."""
        comp, value = self._items(obs)[0]
        value = np.asarray(value)
        base = comp.space.shape if comp.kind != "discrete" else ()
        return value.ndim == len(base) + 1

    def to_storage(self, obs, batched: bool) -> dict[str, np.ndarray]:
        """Raw environment observation(s) -> dict of arrays with a batch dimension."""
        out = {}
        for comp, value in self._items(obs):
            arr = np.asarray(value)
            if not batched:
                arr = arr[None]
            n = arr.shape[0]
            if comp.kind == "image":
                arr = arr.astype(np.uint8, copy=False)
            else:
                arr = arr.reshape(n, *comp.storage_shape).astype(comp.storage_dtype, copy=False)
            out[comp.key] = arr
        return out

    def stack(self, obs_list) -> dict[str, np.ndarray]:
        """List of single raw observations (one per environment) -> storage dict."""
        parts = [self.to_storage(o, batched=False) for o in obs_list]
        return {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}

    # ----------------------------------------------------------------- torch
    def to_tensors(self, stored: dict[str, torch.Tensor], normalizer=None):
        """Storage tensors -> (flat float tensor or None, {image key: float [B,C,H,W] in [0,1]}).

        ``normalizer`` (a :class:`tempo_rl.networks.RunningNorm`) is applied to the
        concatenated ``Box`` vector components only; one-hot and binary parts are
        already on a unit scale.
        """
        vec, other, images = [], [], {}
        for comp in self.components:
            x = stored[comp.key]
            if comp.kind == "vector":
                vec.append(x.float().reshape(x.shape[0], -1))
            elif comp.kind == "discrete":
                start = int(comp.space.start)
                other.append(F.one_hot(x.long().reshape(-1) - start, int(comp.space.n)).float())
            elif comp.kind == "multidiscrete":
                nvec = np.asarray(comp.space.nvec).ravel()
                start = np.asarray(comp.space.start).ravel()
                x = x.long().reshape(x.shape[0], -1)
                other.extend(F.one_hot(x[:, i] - int(start[i]), int(n)).float()
                             for i, n in enumerate(nvec))
            elif comp.kind == "multibinary":
                other.append(x.float().reshape(x.shape[0], -1))
            else:  # image
                img = x if x.shape[1:] == torch.Size(comp.image_shape) else x.permute(0, 3, 1, 2)
                images[comp.key] = img.float() / 255.0
        flat = None
        if vec:
            v = torch.cat(vec, -1) if len(vec) > 1 else vec[0]
            if normalizer is not None:
                v = normalizer(v)
            other = [v] + other
        if other:
            flat = torch.cat(other, -1) if len(other) > 1 else other[0]
        return flat, images

    def vector_part(self, stored: dict[str, np.ndarray]) -> np.ndarray | None:
        """Concatenated float ``Box`` vector components (what the normalizer tracks)."""
        parts = [stored[c.key].reshape(stored[c.key].shape[0], -1)
                 for c in self.components if c.kind == "vector"]
        if not parts:
            return None
        return np.concatenate(parts, -1) if len(parts) > 1 else parts[0]


def check_action_space(action_space: spaces.Space) -> None:
    if isinstance(action_space, spaces.Discrete):
        if int(action_space.start) != 0:
            raise NotImplementedError(
                f"TEMPO supports Discrete action spaces starting at 0, got start={action_space.start}; "
                "wrap the environment in a gymnasium.ActionWrapper that adds the offset")
        return
    if isinstance(action_space, spaces.Box):
        if len(action_space.shape) != 1:
            raise NotImplementedError("TEMPO supports 1-D Box action spaces only")
        if not (np.all(np.isfinite(action_space.low)) and np.all(np.isfinite(action_space.high))):
            raise ValueError("TEMPO needs a bounded Box action space (finite low/high): wrap the "
                             "environment in a gymnasium.ActionWrapper whose action_space declares "
                             "the range the environment accepts")
        return
    raise NotImplementedError(
        f"TEMPO supports Discrete and Box action spaces, got {action_space!r}")
