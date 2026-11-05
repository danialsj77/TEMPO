"""Host <-> device transfers without needless synchronisation (they never change a number).

On an accelerator (Apple MPS, CUDA) every blocking transfer or ``float(tensor)`` makes the CPU wait
until the whole device queue has run, so the device idles while Python prepares the next kernels.
TEMPO therefore

* copies host arrays to the device with *non-blocking* copies (:func:`to_device`). A non-blocking
  copy reads the host memory only when the device reaches it, so the host array is kept alive
  (:func:`keep`) until the next full synchronisation of THAT device (:func:`synced`, called after
  every blocking read such as the ``.cpu()`` of an action). The kept buffers are filed per device,
  so a synchronisation of one device (or a CPU model's ``predict``) never releases the buffers
  another device has not read yet (two models on different devices in one process);
* checks the finiteness of the critic loss on the device and reads the result later: a
  non-blocking copy of the flag is queued after each training phase (:func:`queue_check`) and
  inspected at the next full synchronisation of that device, which raises the same
  ``FloatingPointError`` and clears the device flag (a caught error is not reported again unless a
  later loss is non-finite too, as on the CPU).

On the CPU nothing is deferred: :func:`to_device` is ``torch.as_tensor`` and the finiteness check
stays immediate (see ``TEMPO._critic_step``).

Behaviour of a diverging run on an accelerator (finite runs are bit-for-bit unaffected): a
non-finite critic loss is reported at the next action (or at the end of ``learn``), i.e. after that
update and the ones queued with it have been applied, on the same environment step as before the
deferral (the action is computed but never reaches the environment). The policy's distributions are
built with ``validate_args=False`` (no device read), so NaN actor outputs no longer raise
``torch.distributions``' ``ValueError``; the deferred critic check reports the divergence instead.
"""
from __future__ import annotations

import numpy as np
import torch

#: per device key (:func:`_key`): host tensors of queued non-blocking copies, alive until that
#: device's next full synchronisation
_KEEP: dict = {}
#: per device key: ``(host copy, device flag)`` of queued non-finite flags; the host copy is valid
#: after that device's next full synchronisation
_CHECKS: dict = {}
#: a full synchronisation is forced when this many host buffers are waiting on one device
_MAX_KEEP = 4096

NONFINITE_MESSAGE = ("TEMPO: non-finite critic loss; lower the learning rate or check the "
                     "environment's rewards")


def _key(device) -> str:
    """One key per device queue: ``"cpu"``, ``"mps"`` (``mps`` and ``mps:0`` are the same
    device), ``"cuda:<index>"``."""
    device = torch.device(device)
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        return f"cuda:{index}"
    return device.type


def _device_sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def to_device(array, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    """``torch.as_tensor(array, device=device, dtype=dtype)`` with a non-blocking copy on an
    accelerator (same values: the dtype conversion happens on the host exactly as ``as_tensor``
    does it, round-to-nearest-even)."""
    if device.type == "cpu":
        return torch.as_tensor(array, device=device, dtype=dtype)
    host = torch.as_tensor(np.ascontiguousarray(array), dtype=dtype)
    out = torch.empty(host.shape, dtype=host.dtype, device=device)
    out.copy_(host, non_blocking=True)
    keep(host, device)
    return out


def keep(host, device: torch.device) -> None:
    """Keep ``host`` alive until the next full synchronisation of ``device`` (the device that reads
    or writes it with a queued non-blocking copy)."""
    kept = _KEEP.setdefault(_key(device), [])
    kept.append(host)
    if len(kept) > _MAX_KEEP:                    # bounded memory: wait for the device once
        full_sync(device, check=False)


def queue_check(flag: torch.Tensor) -> None:
    """Queue a non-blocking device->host copy of a non-finite flag (read at the next full sync of
    the flag's device)."""
    host = torch.empty((), dtype=torch.bool)
    host.copy_(flag, non_blocking=True)
    checks = _CHECKS.setdefault(_key(flag.device), [])
    checks.append((host, flag))
    if len(checks) > _MAX_KEEP:                  # train() called many times without acting
        full_sync(flag.device, check=True)


def synced(device: torch.device, check: bool = True) -> None:
    """Call right after a blocking read from ``device`` (every copy queued on it has completed):
    release the host buffers kept for that device and, with ``check``, raise if one of its queued
    non-finite flags was set (the device flags are cleared first, so the error is reported once).
    Nothing is queued for the CPU: a CPU ``device`` is a no-op."""
    k = _key(device)
    if k == "cpu":
        return
    _KEEP.pop(k, None)
    if not check:
        return
    checks = _CHECKS.pop(k, None)
    if not checks:
        return
    if any(bool(host) for host, _ in checks):
        for flag in {id(f): f for _, f in checks}.values():
            flag.zero_()
        raise FloatingPointError(NONFINITE_MESSAGE)


def full_sync(device: torch.device, check: bool = True) -> None:
    """Blocking synchronisation of ``device`` followed by :func:`synced`."""
    _device_sync(device)
    synced(device, check=check)


def new_event(device: torch.device):
    """A device event (MPS / CUDA) or ``None`` on the CPU (everything is synchronous there)."""
    if device.type == "mps":
        return torch.mps.event.Event()
    if device.type == "cuda":
        return torch.cuda.Event()
    return None
