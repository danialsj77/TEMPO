"""A small Stable-Baselines3-style logger: stdout table, CSV and TensorBoard."""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np


class _HumanOutput:
    def __init__(self, stream=None):
        # resolved when the logger is built (every learn() call), not at import time
        self.stream = sys.stdout if stream is None else stream

    def write(self, values: dict, step: int) -> None:
        if not values:
            return
        groups = defaultdict(list)
        for key in sorted(values):
            head, _, tail = key.rpartition("/")
            groups[head].append((tail, values[key]))
        rows = []
        for head in sorted(groups):
            if head:
                rows.append((f"{head}/", ""))
            for tail, v in groups[head]:
                rows.append((f"    {tail}", _fmt(v)))
        kw = max(len(k) for k, _ in rows)
        vw = max(max(len(v) for _, v in rows), 8)
        line = "-" * (kw + vw + 7)
        out = [line] + [f"| {k.ljust(kw)} | {v.ljust(vw)} |" for k, v in rows] + [line]
        self.stream.write("\n".join(out) + "\n")
        self.stream.flush()

    def close(self):
        pass


def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:<8.3g}"
    return str(v)


class _CSVOutput:
    def __init__(self, path: str):
        self.path = path
        self.keys: list[str] = []
        self.file = open(path, "w+t", newline="")

    def write(self, values: dict, step: int) -> None:
        new = [k for k in values if k not in self.keys]
        if new:
            self.keys.extend(new)
            self.file.seek(0)
            old_rows = list(csv.reader(self.file))[1:]
            self.file.seek(0)
            self.file.truncate()
            w = csv.writer(self.file)
            w.writerow(self.keys)
            for r in old_rows:
                w.writerow(r + [""] * (len(self.keys) - len(r)))
        w = csv.writer(self.file)
        w.writerow([values.get(k, "") for k in self.keys])
        self.file.flush()

    def close(self):
        self.file.close()


class _TensorBoardOutput:
    def __init__(self, folder: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError("tensorboard_log needs the `tensorboard` package: "
                              "pip install tensorboard") from e
        self.writer = SummaryWriter(log_dir=folder)

    def write(self, values: dict, step: int) -> None:
        for k, v in values.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                self.writer.add_scalar(k, float(v), step)
        self.writer.flush()

    def close(self):
        self.writer.close()


class Deferred:
    """A logged value read from the device only when the logger writes it (or someone reads it).

    TEMPO records its training statistics (losses, entropy, temperature) as device tensors and
    converts them to Python floats at :meth:`Logger.dump` time, so an update never waits for the
    device just to log. Reading ``Logger.values`` resolves them (a plain float comes back), and a
    ``Deferred`` that escapes (e.g. through ``dict(logger.values)``) still behaves like its number in
    ``float()``, ``int()``, ``round()``, formatting, arithmetic and order comparisons. ``==`` and
    hashing stay those of the object, and ``json`` cannot encode a ``Deferred``: use ``float(v)``.
    """

    __slots__ = ("_fn", "_value", "_done")

    def __init__(self, fn):
        self._fn, self._value, self._done = fn, None, False

    def resolve(self):
        if not self._done:
            self._value, self._fn, self._done = self._fn(), None, True
        return self._value

    def __float__(self) -> float:
        return float(self.resolve())

    def __int__(self) -> int:
        return int(self.resolve())

    def __bool__(self) -> bool:
        return bool(self.resolve())

    def __round__(self, ndigits=None):
        return round(float(self), ndigits)

    def __format__(self, spec: str) -> str:
        return format(float(self), spec)

    def __repr__(self) -> str:
        return repr(self.resolve())

    # numeric protocol: the resolved value (a Python float) does the work
    def __neg__(self):
        return -float(self)

    def __abs__(self):
        return abs(float(self))

    def __add__(self, o):
        return float(self) + o

    def __radd__(self, o):
        return o + float(self)

    def __sub__(self, o):
        return float(self) - o

    def __rsub__(self, o):
        return o - float(self)

    def __mul__(self, o):
        return float(self) * o

    def __rmul__(self, o):
        return o * float(self)

    def __truediv__(self, o):
        return float(self) / o

    def __rtruediv__(self, o):
        return o / float(self)

    def __lt__(self, o):
        return float(self) < o

    def __le__(self, o):
        return float(self) <= o

    def __gt__(self, o):
        return float(self) > o

    def __ge__(self, o):
        return float(self) >= o


class _ResolvingDict(dict):
    """``Logger.values``: a ``dict`` whose reads return resolved values. A :class:`Deferred` entry is
    read from the device when it is first accessed (``d[k]``, ``get``, ``items``, ``values``, ``copy``,
    ``pop``) and stored back as its float, so user code (callbacks, ``json.dumps``, ``np.mean``) sees
    plain numbers. Writing and the logger's own bookkeeping never resolve anything."""

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        if isinstance(value, Deferred):
            value = value.resolve()
            dict.__setitem__(self, key, value)
        return value

    def get(self, key, default=None):
        return self[key] if key in self else default

    def items(self):
        return [(k, self[k]) for k in list(dict.keys(self))]

    def values(self):
        return [self[k] for k in list(dict.keys(self))]

    def copy(self):
        return {k: self[k] for k in list(dict.keys(self))}

    def pop(self, key, *default):
        if key in self:
            value = self[key]
            dict.pop(self, key)
            return value
        return dict.pop(self, key, *default)


def mean_of(values: list) -> Deferred:
    """``float(np.mean([float(v) for v in values]))`` of device scalars, computed lazily (one
    device->host copy for the whole list)."""
    def fn():
        try:
            import torch

            if values and all(isinstance(v, torch.Tensor) for v in values):
                host = torch.stack([v.detach().reshape(()) for v in values]).cpu().tolist()
                return float(np.mean([float(x) for x in host]))
        except ImportError:                                  # pragma: no cover
            pass
        return float(np.mean([float(v) for v in values]))
    return Deferred(fn)


def scalar_of(value) -> Deferred:
    """``float(value)`` of one device tensor, computed lazily."""
    return Deferred(lambda: float(value))


def resolve(value):
    return value.resolve() if isinstance(value, Deferred) else value


class Logger:
    """``record(key, value)`` then ``dump(step)``; mirrors SB3's logger API.

    Recorded values may be :class:`Deferred` (device statistics read at ``dump``). ``values`` (the
    values recorded since the last ``dump``) returns them resolved, as Python floats, when it is read,
    so a callback that inspects ``model.logger.values`` gets numbers (the read waits for the device
    once). Since 2026-10-22 TEMPO records device statistics as :class:`Deferred` (before: Python numbers);
    reading ``values`` gives numbers as before."""

    def __init__(self, folder: str | None = None, output_formats=("stdout",)):
        self.folder = folder
        self.values: dict = _ResolvingDict()
        self._means: dict = defaultdict(list)
        self.outputs = []
        for fmt in output_formats:
            if fmt == "stdout":
                self.outputs.append(_HumanOutput())
            elif fmt == "csv":
                os.makedirs(folder, exist_ok=True)
                self.outputs.append(_CSVOutput(os.path.join(folder, "progress.csv")))
            elif fmt == "tensorboard":
                os.makedirs(folder, exist_ok=True)
                self.outputs.append(_TensorBoardOutput(folder))
            else:
                raise ValueError(f"unknown log format {fmt!r}")

    def record(self, key: str, value, exclude=None) -> None:
        self.values[key] = value

    def record_mean(self, key: str, value, exclude=None) -> None:
        self._means[key].append(value)

    def dump(self, step: int = 0) -> None:
        if self.outputs:                     # nothing is read from the device without an output
            values = {k: resolve(v) for k, v in self.values.items()}
            values.update({k: float(np.mean([resolve(x) for x in v]))
                           for k, v in self._means.items() if v})
            for out in self.outputs:
                out.write(values, step)
        self.values.clear()
        self._means.clear()

    def close(self) -> None:
        for out in self.outputs:
            out.close()


def configure_logger(verbose: int, tensorboard_log: str | None, tb_log_name: str,
                     reset_num_timesteps: bool) -> Logger:
    formats = ["stdout"] if verbose >= 1 else []
    folder = None
    if tensorboard_log is not None:
        os.makedirs(tensorboard_log, exist_ok=True)
        run = 1
        existing = [d for d in os.listdir(tensorboard_log) if d.startswith(tb_log_name + "_")]
        ids = [int(d.rsplit("_", 1)[1]) for d in existing if d.rsplit("_", 1)[1].isdigit()]
        if ids:
            run = max(ids) + (1 if reset_num_timesteps else 0)
        folder = os.path.join(tensorboard_log, f"{tb_log_name}_{run}")
        formats.append("tensorboard")
    return Logger(folder, formats)
