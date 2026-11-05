"""Zip archive format for saved TEMPO models.

``model.zip``
    ``data.json``          hyperparameters, spaces (as JSON), counters, versions
    ``policy.pth``         policy state dict (networks, target networks, obs statistics)
    ``tempo_state.pth``    optimiser states and the temperature
    ``_tempo_version.txt`` package version

Everything tensor-valued is loaded with ``torch.load(weights_only=True)``; no
arbitrary Python objects are unpickled when loading a model.
"""
from __future__ import annotations

import io
import json
import pathlib
import zipfile

import numpy as np
import torch
from gymnasium import spaces


# --------------------------------------------------------------------- spaces
def space_to_json(space: spaces.Space) -> dict:
    if isinstance(space, spaces.Box):
        return dict(type="Box", low=np.asarray(space.low).tolist(),
                    high=np.asarray(space.high).tolist(), shape=list(space.shape),
                    dtype=str(space.dtype))
    if isinstance(space, spaces.Discrete):
        return dict(type="Discrete", n=int(space.n), start=int(space.start))
    if isinstance(space, spaces.MultiDiscrete):
        return dict(type="MultiDiscrete", nvec=np.asarray(space.nvec).tolist(),
                    start=np.asarray(space.start).tolist())
    if isinstance(space, spaces.MultiBinary):
        n = space.n
        return dict(type="MultiBinary", n=n if isinstance(n, int) else list(n))
    if isinstance(space, spaces.Dict):
        return dict(type="Dict", spaces={k: space_to_json(s) for k, s in space.spaces.items()})
    if isinstance(space, spaces.Tuple):
        return dict(type="Tuple", spaces=[space_to_json(s) for s in space.spaces])
    raise NotImplementedError(f"cannot serialise space {space!r}")


def _inf(x):
    return np.asarray([float(v) for v in np.ravel(x)]) if isinstance(x, list) else np.asarray(x)


def space_from_json(d: dict) -> spaces.Space:
    t = d["type"]
    if t == "Box":
        dtype = np.dtype(d["dtype"])
        shape = tuple(d["shape"])
        low = np.asarray(d["low"], dtype=np.float64).reshape(shape).astype(dtype)
        high = np.asarray(d["high"], dtype=np.float64).reshape(shape).astype(dtype)
        return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    if t == "Discrete":
        return spaces.Discrete(d["n"], start=d.get("start", 0))
    if t == "MultiDiscrete":
        return spaces.MultiDiscrete(d["nvec"], start=d.get("start"))
    if t == "MultiBinary":
        return spaces.MultiBinary(d["n"])
    if t == "Dict":
        return spaces.Dict({k: space_from_json(s) for k, s in d["spaces"].items()})
    if t == "Tuple":
        return spaces.Tuple([space_from_json(s) for s in d["spaces"]])
    raise NotImplementedError(f"unknown space type {t!r}")


def spaces_match(a: spaces.Space, b: spaces.Space) -> bool:
    return space_to_json(a) == space_to_json(b)


# ------------------------------------------------------------------- archive
def _torch_bytes(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def _json_default(o):
    """numpy scalars/arrays (e.g. hyperparameters from a numpy sweep) -> plain Python."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def save_to_zip(path, data: dict, policy_state: dict, tempo_state: dict, version: str) -> pathlib.Path:
    path = pathlib.Path(path)
    if path.suffix != ".zip":
        path = path.with_suffix(path.suffix + ".zip") if path.suffix else path.with_suffix(".zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, default=_json_default)   # fail before touching the file
    tmp = path.with_name(path.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("data.json", payload)
            z.writestr("policy.pth", _torch_bytes(policy_state))
            z.writestr("tempo_state.pth", _torch_bytes(tempo_state))
            z.writestr("_tempo_version.txt", version)
    except BaseException:
        tmp.unlink(missing_ok=True)                  # never leave a half-written archive behind
        raise
    tmp.replace(path)                                  # atomic: never leaves a half-written model
    return path


def load_from_zip(path, device="cpu"):
    path = pathlib.Path(path)
    if not path.exists() and pathlib.Path(str(path) + ".zip").exists():
        path = pathlib.Path(str(path) + ".zip")         # what save_to_zip writes for this name
    with zipfile.ZipFile(path, "r") as z:
        data = json.loads(z.read("data.json"))
        policy_state = torch.load(io.BytesIO(z.read("policy.pth")), map_location=device,
                                  weights_only=True)
        tempo_state = torch.load(io.BytesIO(z.read("tempo_state.pth")), map_location=device,
                                 weights_only=True)
    return data, policy_state, tempo_state
