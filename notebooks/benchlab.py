"""Helpers shared by the benchmark notebooks in this folder.

The notebooks drive the existing benchmark harness; nothing here trains a model:

* ``benchmarks.plans``      builds job lists (``custom_jobs``, ``option_a``, ``option_b``, ...);
* ``benchmarks.scheduler``  runs a job list with per-device concurrency limits, skips jobs
                            whose result file exists (restart-safe) and writes
                            ``<logs>/<label>.status``;
* ``benchmarks.run_one``    trains one (algorithm, variant, task, seed) and writes
                            ``<results>/<suite>/<env>/<algo>/<variant>/seed<k>.json``.

This module adds what a notebook needs around them: result/log directories (real or
smoke), device assignment and wall-clock estimates, launching the scheduler in the
background (it survives the notebook kernel), a status table, a safe stop, the
pre-registered selection rules (Screen 3: against CrossQ; Screen 5: against the best SB3
baseline of every task) and learning-curve plots.

The last section ("per-environment notebooks") backs ``notebooks/environments/NN_<EnvId>.ipynb``
(written by ``notebooks/make_env_notebooks.py``): one end-to-end 5-seed benchmark of one task,
TEMPO against the tuned SB3 baselines, with the shared result locations of suite ``final``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass


def find_root(start: pathlib.Path | None = None) -> pathlib.Path:
    """The project root: the first parent that holds both ``benchmarks/`` and ``tempo_rl/``."""
    p = (start or pathlib.Path(__file__).resolve().parent).resolve()
    for q in (p, *p.parents):
        if (q / "benchmarks").is_dir() and (q / "tempo_rl").is_dir():
            return q
    raise RuntimeError(f"cannot find the project root above {p}")


ROOT = find_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCH = ROOT / "benchmarks"
PY = sys.executable

#: Five MuJoCo locomotion tasks of the option A / option B screens.
LOCO = ("Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Walker2d-v5", "Hopper-v5")

# --------------------------------------------------------------------------- flags / dirs


def env_flag(name: str, default: bool) -> bool:
    """Boolean override from the environment (``1/true/yes/on`` or ``0/false/no/off``)."""
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return bool(default)
    return v in ("1", "true", "yes", "on")


def scratch_root() -> pathlib.Path:
    """Where smoke runs write: ``$TEMPO_NOTEBOOK_SCRATCH`` or ``<tmp>/tempo_notebooks``."""
    p = os.environ.get("TEMPO_NOTEBOOK_SCRATCH", "").strip()
    return pathlib.Path(p) if p else pathlib.Path(tempfile.gettempdir()) / "tempo_notebooks"


@dataclass
class Dirs:
    results: pathlib.Path
    logs: pathlib.Path
    figures: pathlib.Path
    tables: pathlib.Path
    smoke: bool

    def show(self) -> None:
        mode = "SMOKE (temporary directories, tiny runs)" if self.smoke else "REAL (benchmark directories)"
        print(f"mode    : {mode}")
        for k in ("results", "logs", "figures", "tables"):
            print(f"{k:8s}: {getattr(self, k)}")


def label_for(name: str, smoke: bool) -> str:
    """Scheduler label; smoke runs get their own (``smoke_<name>``) so that a smoke notebook can
    never find, report on or stop a real run of the same notebook."""
    return f"smoke_{name}" if smoke else name


def bench_dirs(smoke: bool, name: str) -> Dirs:
    """Result/log/figure/table directories for a notebook.

    Real mode uses ``benchmarks/results``, ``benchmarks/logs``, ``benchmarks/figures`` and
    ``benchmarks/tables``. Smoke mode uses ``<scratch>/<name>/...`` so tiny test runs never mix
    with real results. ``TEMPO_BENCH_RESULTS`` / ``TEMPO_BENCH_LOGS`` are set in this kernel
    as well (they are also passed explicitly to every subprocess).
    """
    if smoke:
        base = scratch_root() / name
        d = Dirs(base / "results", base / "logs", base / "figures", base / "tables", True)
    else:
        d = Dirs(BENCH / "results", BENCH / "logs", BENCH / "figures", BENCH / "tables", False)
    for p in (d.results, d.logs, d.logs / "locks"):
        p.mkdir(parents=True, exist_ok=True)
    os.environ["TEMPO_BENCH_RESULTS"] = str(d.results)
    os.environ["TEMPO_BENCH_LOGS"] = str(d.logs)
    return d


# --------------------------------------------------------------------------- jobs


def job_id(j: dict) -> str:
    """Same id as ``benchmarks.scheduler.job_id`` (log and lock file names)."""
    return f"{j.get('suite', 'main')}__{j['env']}__{j['algo']}__{j.get('variant', 'default')}__s{j['seed']}"


def result_path(j: dict, results: pathlib.Path) -> pathlib.Path:
    """Same layout as ``benchmarks.run_one.result_path`` but for an explicit results root."""
    return (pathlib.Path(results) / j.get("suite", "main") / j["env"] / j["algo"]
            / j.get("variant", "default") / f"seed{j['seed']}.json")


#: Compute-relevant settings of TEMPO variants resolved by instantiating the model with the code
#: that will run (:func:`tempo_run_config`); :func:`tempo_config` prefers them when present.
_RESOLVED: dict = {}
_LIVE_DEFAULTS: dict | None = None


def _live_defaults() -> dict:
    """Constructor defaults of the live library that decide a run's compute and its exploration:
    the critic architecture (``tempo_rl.policies.DEFAULT_ARCH`` / ``TempoPolicy(net_arch=...)``),
    the SALE representation, weight norm, ``TEMPO(gradient_steps=...)``, the novelty bonus
    (``exploration_bonus``), the action noise (``exploration_noise``) and the discount (``gamma``).
    Read from the signatures, so a change of the library defaults is picked up without editing
    this file."""
    global _LIVE_DEFAULTS
    if _LIVE_DEFAULTS is None:
        try:
            import inspect

            from tempo_rl import policies as P
            from tempo_rl.tempo import TEMPO

            t = {k: p.default for k, p in inspect.signature(TEMPO.__init__).parameters.items()}
            pp = {k: p.default for k, p in inspect.signature(P.TempoPolicy.__init__).parameters.items()}
            _LIVE_DEFAULTS = dict(arch=pp.get("net_arch") or P.DEFAULT_ARCH,
                                  gradient_steps=t.get("gradient_steps", 1),
                                  representation=pp.get("representation"),
                                  weight_norm=pp.get("weight_norm", False),
                                  exploration_bonus=float(t.get("exploration_bonus", 0.0) or 0.0),
                                  exploration_noise=t.get("exploration_noise", "white"),
                                  gamma=t.get("gamma", 0.99))
        except Exception as e:                        # noqa: BLE001 - keep the planning helpers usable
            print(f"note: cannot read the tempo_rl defaults ({type(e).__name__}: {e}); assuming 256-wide critics")
            _LIVE_DEFAULTS = dict(arch=dict(pi=[256, 256], qf=[256, 256]), gradient_steps=1,
                                  representation=None, weight_norm=False, exploration_bonus=0.0,
                                  exploration_noise="white", gamma=0.99)
    return _LIVE_DEFAULTS


def tempo_config(variant: str) -> dict:
    """Compute-relevant settings of a TEMPO variant: critic width, gradient steps per env step,
    SALE, weight norm and the novelty bonus (``novelty`` = ``exploration_bonus > 0``: the RND
    networks are trained at every update), plus the action noise and the discount request.

    Settings a variant does not fix come from the live library's defaults (:func:`_live_defaults`);
    a variant resolved by :func:`tempo_run_config` in this kernel uses the resolved values."""
    from benchmarks.variants import VARIANTS

    if variant in _RESOLVED:
        return dict(_RESOLVED[variant])
    cfg = VARIANTS[variant]
    lib = _live_defaults()
    pk = dict(cfg.get("policy_kwargs") or {})
    arch = pk.get("net_arch") or lib["arch"]
    if isinstance(arch, dict):
        qf = list(arch["qf"])
    else:
        qf = list(arch)
    bonus = float(cfg.get("exploration_bonus", lib["exploration_bonus"]) or 0.0)
    return dict(width=int(max(qf)), gradient_steps=int(cfg.get("gradient_steps", lib["gradient_steps"])),
                sale=pk.get("representation", lib["representation"]) == "sale",
                weight_norm=bool(pk.get("weight_norm", lib["weight_norm"])),
                novelty=bonus > 0, exploration_bonus=bonus,
                exploration_noise=cfg.get("exploration_noise", lib["exploration_noise"]),
                gamma=cfg.get("gamma", lib["gamma"]))


#: The previous library default (the cheap 256-wide recipe of 2026-09-26, pinned in
#: ``benchmarks/variants.py``): the reference for "relative compute" and the cheap alternative
#: ``TEMPO_VARIANT`` of the per-environment notebooks.
PREVIOUS_DEFAULT = "bn256_e05_hl"

#: Rough env-steps/s of ONE job on this Mac (18-core Apple silicon) while the machine is busy
#: with a full schedule. Planning numbers (updated 2026-09-30 with the Screen-6 timings); edit freely.
#:
#: * TEMPO on the CPU (1 thread per job): 256-wide critic ~55, 512-wide ~35. A 1024-wide
#:   critic on the CPU is ~8 (a short timing on 2026-09-28 put it 6.7x slower than the 256-wide
#:   recipe), which is why wide critics go to the Apple GPU.
#: * TEMPO on MPS with 4-8 concurrent MPS jobs: ~40 for critics up to 1024 wide (the GPU is
#:   bound by kernel-launch latency, not by the width).
#: * SALE (learned representations) adds a fixed cost per update: ~25 ms on MPS (1024-wide +
#:   SALE = ``tall_gauto`` measured at ~20-21 env-steps/s per run with six concurrent MPS runs in
#:   Screen 5, benchmarks/PROTOCOL.md) and ~25 ms on the CPU (256-wide + SALE took 2.3x the time
#:   of the 256-wide recipe).
#: * The tempered novelty bonus (``exploration_bonus > 0``: an RND target and predictor network
#:   trained at every update) adds ~11 ms per update on MPS and ~12 ms on the CPU. Screen 6
#:   (2026-09-29/30) measured C1 = ``tall_gauto`` + pink noise + bonus 5, today's library default,
#:   at ~16-18 env-steps/s per run on MPS with eight concurrent GPU runs (~4.7-5.2 h per
#:   300k-step run) and ~4.6-6.2 on the CPU (1 thread); the model below gives ~16.4 on MPS and
#:   ~6.2 on the CPU. Pink noise itself costs nothing measurable.
#: * SB3 baselines (1 CPU thread): SAC ~70, TD3 ~150, PPO ~800, DQN ~500; CrossQ (1024-wide
#:   critics) ~30 on MPS, ~8 on the CPU. These are conservative: the finished final-suite runs
#:   had medians SAC 200, TD3 190, PPO ~3000, DQN ~1600 and CrossQ (MPS) 28 env-steps/s.
TEMPO_RATES = {"cpu": {256: 55.0, 512: 35.0, 1024: 8.0},
               "mps": {256: 40.0, 512: 40.0, 1024: 40.0}}
SALE_EXTRA_S = {"cpu": 0.025, "mps": 1 / 20 - 1 / 40}
NOVELTY_EXTRA_S = {"cpu": 0.012, "mps": 0.011}
SB3_RATES = {"SAC": {"cpu": 70.0}, "TD3": {"cpu": 150.0}, "PPO": {"cpu": 800.0},
             "DQN": {"cpu": 500.0}, "CrossQ": {"mps": 30.0, "cpu": 8.0}}
#: Overhead of the weight-norm projection (3000-step Hopper runs, 2026-09-26: wn_utd2 took
#: 2.1x the time of the default, i.e. two updates x 1.05).
WN_FACTOR = 1.05
#: Fixed per-job overhead (imports, environment creation, final save), seconds.
JOB_OVERHEAD_S = 15.0
#: Steps per run in smoke mode (3 evaluations; learning starts after 1,000 steps). Smaller than
#: ``plans.custom_jobs``' default of 3,000 so that the wide-critic notebooks finish in minutes.
SMOKE_STEPS = 1500


def _seconds_per_update(table: dict, width: int, device: str) -> float:
    """Seconds per env step of a TEMPO critic of ``width`` from a {width: env-steps/s} table
    (linear interpolation of the time; beyond the table: x (w/w_max)^2 on the CPU, x w/w_max on MPS)."""
    ws = sorted(table)
    if width <= ws[0]:
        return 1.0 / table[ws[0]]
    if width >= ws[-1]:
        f = width / ws[-1]
        return (1.0 / table[ws[-1]]) * (f if pool(device) == "mps" else f * f)
    lo = max(x for x in ws if x <= width)
    hi = min(x for x in ws if x >= width)
    if lo == hi:
        return 1.0 / table[lo]
    t_lo, t_hi = 1.0 / table[lo], 1.0 / table[hi]
    return t_lo + (t_hi - t_lo) * (width - lo) / (hi - lo)


def job_rate(j: dict, device: str | None = None) -> float:
    """Rough env-steps/s of one job on ``device`` (default: the job's own device); see
    :data:`TEMPO_RATES`, :data:`SALE_EXTRA_S`, :data:`NOVELTY_EXTRA_S` and :data:`SB3_RATES`."""
    dev = pool(device or j.get("device", "cpu"))
    if j["algo"] != "TEMPO":
        r = SB3_RATES.get(j["algo"], {})
        return float(r.get(dev) or r.get("cpu") or 100.0)
    c = tempo_config(j.get("variant", "default"))
    per_update = _seconds_per_update(TEMPO_RATES[dev], c["width"], dev)
    if c["sale"]:
        per_update += SALE_EXTRA_S[dev]
    if c.get("novelty"):
        per_update += NOVELTY_EXTRA_S[dev]
    per_step = c["gradient_steps"] * per_update * (WN_FACTOR if c["weight_norm"] else 1.0)
    return 1.0 / per_step


def job_seconds(j: dict, device: str | None = None) -> float:
    return j["steps"] / job_rate(j, device) + JOB_OVERHEAD_S


def mps_ok(j: dict, min_width: int = 512) -> bool:
    """Jobs that benefit from the GPU: CrossQ (1024-wide critics) and wide TEMPO critics."""
    if j["algo"] == "CrossQ":
        return True
    return j["algo"] == "TEMPO" and tempo_config(j.get("variant", "default"))["width"] >= min_width


def first_seed_only(jobs: list[dict]) -> list[dict]:
    """Smoke mode: keep only the jobs of the lowest seed (every variant and task still runs once)."""
    if not jobs:
        return jobs
    s0 = min(j["seed"] for j in jobs)
    return [j for j in jobs if j["seed"] == s0]


def shrink(jobs: list[dict], steps: int = SMOKE_STEPS) -> list[dict]:
    """Smoke mode: ``steps`` env steps per run with three evaluations (smoke suites only)."""
    out = []
    for j in jobs:
        assert str(j.get("suite", "")).startswith("smoke_"), "shrink() is for smoke suites only"
        out.append(dict(j, steps=int(steps), eval_freq=max(1, int(steps) // 3), n_eval=3))
    return out


#: Screen 5 as amended on 2026-09-28 17:20 (benchmarks/PROTOCOL.md): candidate ``tall`` was dropped
#: from every task where it had not started; its four runs already under way (Humanoid-v5 and
#: Ant-v5, seeds 6-7) continue and are reported as a partial check of gamma="auto" versus 0.99 at
#: equal width. The selection rule is applied to the candidates with results on every task.
SCREEN5_DROPPED = {"tall": ("Humanoid-v5", "Ant-v5")}


def screen5_candidates() -> tuple:
    """The Screen-5 candidates that take part in the selection (``plans.BEAT_ALL_VARIANTS`` minus the
    dropped ones), in the order of the plan."""
    from benchmarks import plans

    return tuple(v for v in plans.BEAT_ALL_VARIANTS if v not in SCREEN5_DROPPED)


def beat_all_jobs(variants, tasks, seeds, smoke: bool = False, partial: dict | None = None) -> list[dict]:
    """Jobs of the Screen-5 plan ``benchmarks.plans.beat_all`` (suite ``screen5``; 1024-wide critics
    on MPS, the rest on the CPU) for ``variants`` x ``tasks`` x ``seeds``, plus ``partial``
    (``{variant: tasks}``, e.g. :data:`SCREEN5_DROPPED`) on those tasks only. Tasks outside
    ``plans.BEAT_ALL_TASKS`` are added with their suite budgets and the same device rule."""
    from benchmarks import plans

    tasks = list(tasks)
    unknown = [e for e in tasks if e not in plans.SUITE and e not in plans.TOY_TEXT]
    if unknown:
        raise ValueError(f"unknown tasks {unknown}; known: {sorted(plans.SUITE)}")

    def build(vs, envs):
        out = [j for j in plans.beat_all(smoke=smoke, variants=tuple(vs), seeds=tuple(seeds)) if j["env"] in envs]
        extra = [e for e in envs if e not in plans.BEAT_ALL_TASKS]
        if extra:
            more = plans.custom_jobs("screen5", variants=tuple(vs), envs=extra, seeds=tuple(seeds), smoke=smoke)
            for j in more:
                j["device"] = "mps" if tempo_config(j["variant"])["width"] >= 1024 else "cpu"
            out += more
        return out

    jobs = build(variants, tasks)
    order_v = list(variants)
    for v, envs in (partial or {}).items():
        if v in variants:
            continue
        jobs += build((v,), [e for e in envs if e in tasks])
        order_v.append(v)
    order = {e: i for i, e in enumerate(tasks)}
    return sorted(jobs, key=lambda j: (order[j["env"]], order_v.index(j["variant"]), j["seed"]))


def pin_devices(jobs: list[dict], cpu_workers: int, mps_workers: int, cpu_threads: int = 1,
                results: pathlib.Path | None = None) -> list[dict]:
    """Keep the device each job already has (e.g. ``benchmarks.plans.beat_all`` puts every
    1024-wide critic on MPS and the rest on the CPU); set threads and the estimated duration as
    ``cost`` (longest first). Without MPS workers the MPS jobs fall back to the CPU (much slower)."""
    jobs = [dict(j) for j in jobs]
    todo = [j for j in jobs if results is None or not result_path(j, results).exists()]
    if mps_workers <= 0 and any(pool(j.get("device")) == "mps" for j in todo):
        n = sum(pool(j.get("device")) == "mps" for j in todo)
        print(f"note: MPS_WORKERS = 0: {n} jobs planned for the Apple GPU run on the CPU instead "
              "(expect them to take several times longer)")
        for j in jobs:
            j["device"] = "cpu"
    if cpu_workers <= 0 and any(j.get("device", "cpu") == "cpu" for j in todo):
        raise ValueError("some jobs run on the CPU: set CPU_WORKERS > 0")
    for j in jobs:
        j["device"] = j.get("device", "cpu")
        j["threads"] = int(cpu_threads) if j["device"] == "cpu" else 1
        j["cost"] = round(job_seconds(j))
    return jobs


def assign_devices(jobs: list[dict], cpu_workers: int, mps_workers: int, cpu_threads: int = 1,
                   results: pathlib.Path | None = None) -> list[dict]:
    """Put each job on ``cpu`` or ``mps``, spreading both devices evenly over the variants.

    Only jobs for which :func:`mps_ok` holds may go to MPS; the others stay on the CPU. The
    MPS share of the eligible work is chosen so that the estimated load per worker is about
    equal on both devices (with each device's own speed, :func:`job_rate`), and inside every
    (algorithm, variant) group the MPS jobs are spaced evenly over tasks and seeds (rotated from
    group to group). So no variant runs only on one device, and a device difference cannot
    masquerade as a variant difference.
    ``cost`` is set to the estimated duration in seconds, so the scheduler starts the longest
    jobs first. Jobs that already have a result are left out. The device does not change a
    job's result path, so a relaunch with different worker counts still skips finished jobs.
    """
    jobs = [dict(j) for j in jobs]
    todo = [j for j in jobs if results is None or not result_path(j, results).exists()]
    for j in jobs:
        j["device"] = "cpu"
    elig = [j for j in todo if mps_ok(j)]
    if cpu_workers <= 0 and any(not mps_ok(j) for j in todo):
        raise ValueError("some jobs can only run on the CPU: set CPU_WORKERS > 0")
    if mps_workers > 0 and elig:
        # fraction f of the eligible jobs on MPS such that
        #   f * mps_seconds(elig) / n_mps == (cpu_seconds(cpu-only) + (1 - f) * cpu_seconds(elig)) / n_cpu
        t_cpu_only = sum(job_seconds(j, "cpu") for j in todo if not mps_ok(j))
        c_elig = sum(job_seconds(j, "cpu") for j in elig)
        m_elig = sum(job_seconds(j, "mps") for j in elig)
        if cpu_workers <= 0:
            share = 1.0
        else:
            share = ((t_cpu_only + c_elig) / cpu_workers) / (m_elig / mps_workers + c_elig / cpu_workers)
            share = min(1.0, max(0.0, share))
        groups: dict = {}
        for j in elig:
            groups.setdefault((j["algo"], j.get("variant", "default")), []).append(j)
        order = {e: i for i, e in enumerate(dict.fromkeys(j["env"] for j in jobs))}
        for g, (_, members) in enumerate(sorted(groups.items())):
            members.sort(key=lambda j: (order[j["env"]], j["seed"]))
            n = len(members)
            k = min(n, int(math.floor(share * n + 0.5 + 1e-9)))
            for t in range(k):
                members[(int((t + 0.5) * n / k) + g) % n]["device"] = "mps"
    for j in jobs:
        j["threads"] = int(cpu_threads) if j["device"] == "cpu" else 1
        j["cost"] = round(job_seconds(j))
    return jobs


def estimate(jobs: list[dict], cpu_workers: int, mps_workers: int,
             results: pathlib.Path | None = None, verbose: bool = True) -> dict:
    """Wall-clock estimate: list-schedule the unfinished jobs per device, longest first
    (what ``benchmarks.scheduler`` does), with the rough per-device rates of :func:`job_rate`."""
    todo = [j for j in jobs if results is None or not result_path(j, results).exists()]
    caps = {"cpu": cpu_workers, "mps": mps_workers}
    span, job_h = {}, 0.0
    for dev in ("cpu", "mps"):
        dj = sorted([j for j in todo if pool(j.get("device")) == dev], key=lambda j: -job_seconds(j))
        if dj and caps[dev] <= 0:
            raise ValueError(f"{len(dj)} jobs need device {dev!r} but it has 0 workers")
        free = [0.0] * max(1, caps[dev])
        for j in dj:
            k = min(range(len(free)), key=free.__getitem__)
            free[k] += job_seconds(j)
            job_h += job_seconds(j) / 3600
        span[dev] = max(free) / 3600 if dj else 0.0
    out = dict(n_jobs=len(jobs), n_done=len(jobs) - len(todo), n_todo=len(todo), job_hours=job_h,
               wall_hours=max(span.values()) if span else 0.0, per_device_hours=span)
    if verbose:
        print(f"{out['n_jobs']} jobs: {out['n_done']} already finished, {out['n_todo']} to run "
              f"({job_h:.1f} job-hours)")
        print(f"estimated wall-clock with {cpu_workers} CPU + {mps_workers} MPS workers: "
              f"~{_fmt_hours(out['wall_hours'])}  (cpu {_fmt_hours(span.get('cpu', 0))}, "
              f"mps {_fmt_hours(span.get('mps', 0))})")
        print("rough: per-job planning rates (benchlab.TEMPO_RATES / SALE_EXTRA_S / NOVELTY_EXTRA_S / SB3_RATES; "
              "e.g. TEMPO 256-wide ~55 env-steps/s on the CPU; on MPS 1024-wide ~40, + SALE ~20, the library "
              "default C1 (+ SALE + novelty bonus) ~16). Real times can be ~50% longer when every core is busy; "
              "the STATUS cell shows the measured speeds.")
    return out


def _fmt_hours(h: float) -> str:
    return f"{h * 60:.0f} min" if h < 1 else f"{h:.1f} h"


def summarize_jobs(jobs: list[dict]):
    """Counts per (variant or algorithm, device) with the planning rate and the estimated hours
    of the longest job of each group, as a small DataFrame."""
    import pandas as pd

    df = pd.DataFrame([dict(algo=j["algo"], variant=j.get("variant", "default"), env=j["env"],
                            seed=j["seed"], steps=j["steps"], device=j.get("device", "cpu"),
                            rate=round(job_rate(j), 1), est_hours=round(job_seconds(j) / 3600, 2))
                       for j in jobs])
    if df.empty:
        return df
    return (df.groupby(["algo", "variant", "device"])
              .agg(jobs=("env", "size"), tasks=("env", "nunique"), seeds=("seed", "nunique"),
                   steps_per_job=("steps", "max"), env_steps_per_s=("rate", "first"),
                   est_hours_longest_job=("est_hours", "max"))
              .reset_index())


def write_jobs(jobs: list[dict], label: str, dirs: Dirs) -> pathlib.Path:
    p = dirs.logs / f"{label}.jobs.json"
    p.write_text(json.dumps(jobs, indent=1))
    return p


# --------------------------------------------------------------------------- processes


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _cmdline(pid: int) -> str:
    try:
        return subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(pid)], capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _processes() -> list[tuple[int, int, int, str]]:
    """(pid, ppid, pgid, command) of every process of this user."""
    out = subprocess.run(["ps", "-x", "-ww", "-o", "pid=,ppid=,pgid=,command="], capture_output=True,
                         text=True, timeout=20).stdout
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) == 4 and all(x.isdigit() for x in parts[:3]):
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3]))
    return rows


def _lock_pid(path: pathlib.Path) -> int | None:
    """The pid in a scheduler lock file (``<logs>/locks/<job-id>.pid``: ``"<pid>"`` or, since
    2026-09-28, ``"<pid> <device>"``), or None."""
    try:
        return int(pathlib.Path(path).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _pidfile(label: str, dirs: Dirs) -> pathlib.Path:
    return dirs.logs / f"{label}.pid"


def _is_scheduler(cmd: str, label: str) -> bool:
    return "benchmarks.scheduler" in cmd and re.search(rf"(^|\s){re.escape(label)}(\s|$)", cmd) is not None


def scheduler_info(label: str, dirs: Dirs) -> dict | None:
    """The live scheduler for ``label`` (from ``<logs>/<label>.pid`` or the process table)."""
    f = _pidfile(label, dirs)
    if f.exists():
        try:
            info = json.loads(f.read_text())
            if _alive(info["pid"]) and _is_scheduler(_cmdline(info["pid"]), label):
                return info
        except (ValueError, KeyError):
            pass
    for pid, _, pgid, cmd in _processes():      # e.g. started from a terminal
        if _is_scheduler(cmd, label) and pid != os.getpid():
            return dict(pid=pid, pgid=pgid, label=label, cmd=cmd, started="?")
    return None


def launch(label: str, jobs: list[dict], dirs: Dirs, cpu_workers: int, mps_workers: int,
           background: bool = True, hint: str = "Use the STATUS cell.") -> dict | int | None:
    """Start ``python -m benchmarks.scheduler <label> --jobs-file ...``.

    ``background=True``: detached (own session), output appended to ``<logs>/<label>.out``;
    the notebook stays usable and the runs survive a kernel restart or closing JupyterLab.
    ``background=False``: runs in the foreground and streams the scheduler's output (smoke mode).
    Re-running is safe: finished jobs are skipped, and a second scheduler for the same label is
    never started while one is alive. ``hint`` names the notebook cell that shows the progress.
    """
    running = scheduler_info(label, dirs)
    if running:
        print(f"scheduler '{label}' is already running (pid {running['pid']}, started "
              f"{running.get('started', '?')}); not starting another one. {hint}")
        return running
    jobs_file = write_jobs(jobs, label, dirs)
    todo = [j for j in jobs if not result_path(j, dirs.results).exists()]
    if not todo:
        print(f"all {len(jobs)} jobs of '{label}' already have results; nothing to launch.")
        return None
    cmd = [PY, "-m", "benchmarks.scheduler", label, "--jobs-file", str(jobs_file),
           "--cpu", str(int(cpu_workers)), "--mps", str(int(mps_workers))]
    env = {**os.environ, "TEMPO_BENCH_RESULTS": str(dirs.results), "TEMPO_BENCH_LOGS": str(dirs.logs),
           "PYTHONUNBUFFERED": "1"}
    out = dirs.logs / f"{label}.out"
    if background:
        with open(out, "a") as fh:
            fh.write(f"\n=== launched {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(cmd)}\n")
            fh.flush()
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
        info = dict(pid=p.pid, pgid=p.pid, label=label, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                    cmd=cmd, jobs_file=str(jobs_file), results=str(dirs.results), logs=str(dirs.logs))
        _pidfile(label, dirs).write_text(json.dumps(info, indent=1))
        time.sleep(3)
        if p.poll() is not None:
            print(f"the scheduler exited immediately (code {p.returncode}); last output:")
            print(tail(out, 30))
            return p.returncode
        print(f"started scheduler '{label}' in the background: pid {p.pid}, {len(todo)} jobs to run")
        print(f"  output : {out}")
        print(f"  status : {dirs.logs / (label + '.status')}")
        print(f"You can close this notebook; the runs continue. Re-open it later. {hint}")
        return info
    # foreground (smoke mode): stream the output, stop everything on interrupt
    t0 = time.time()
    with open(out, "a") as fh:
        fh.write(f"\n=== foreground {time.strftime('%Y-%m-%d %H:%M:%S')}: {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True, text=True, bufsize=1)
        _pidfile(label, dirs).write_text(json.dumps(dict(pid=p.pid, pgid=p.pid, label=label, cmd=cmd,
                                                          started=time.strftime("%Y-%m-%d %H:%M:%S"))))
        try:
            for line in p.stdout:
                fh.write(line)
                fh.flush()
                print(line, end="", flush=True)
            p.wait()
        except KeyboardInterrupt:
            print("interrupted: stopping the scheduler and its jobs")
            _kill_tree(p.pid)
            raise
    print(f"scheduler '{label}' exited with code {p.returncode} after {(time.time() - t0) / 60:.1f} min")
    return p.returncode


def _kill_tree(pid: int, timeout: float = 15.0) -> None:
    """Terminate a scheduler and every process it started (its jobs), nothing else.

    A scheduler started by :func:`launch` leads its own process group, but one started from a
    terminal may share its group with other schedulers (or with the shell), so the processes
    are collected by parent pid instead of killing the whole group. The scheduler is stopped
    first so that it cannot start new jobs while its running jobs are being stopped.
    """
    procs = _processes()
    children: dict = {}
    for p, pp, _, _ in procs:
        children.setdefault(pp, []).append(p)
    tree, todo = [], [pid]
    while todo:
        q = todo.pop()
        tree.append(q)
        todo.extend(children.get(q, []))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for q in tree:
            try:
                os.kill(q, sig)
            except (ProcessLookupError, PermissionError):
                pass
        t = time.time()
        while time.time() - t < timeout:
            if not any(_alive(q) for q in tree):
                return
            time.sleep(0.5)


def _run_one_procs() -> list[tuple[int, int, dict]]:
    """(pid, parent pid, job) of every running ``benchmarks.run_one`` process."""
    out = []
    for pid, ppid, _, cmd in _processes():
        if "benchmarks.run_one" not in cmd:
            continue
        i = cmd.find("{")
        if i < 0:
            continue
        try:
            job = json.loads(cmd[i:])
        except ValueError:
            continue
        if isinstance(job, dict) and {"env", "algo", "seed"} <= set(job):
            out.append((pid, ppid, job))
    return out


def _run_one_jobs() -> list[tuple[int, dict]]:
    """(pid, job) of every running ``benchmarks.run_one`` process."""
    return [(pid, job) for pid, _, job in _run_one_procs()]


def stop(label: str, dirs: Dirs, jobs: list[dict] | None = None, only_orphans: bool = False) -> None:
    """Stop the scheduler ``label`` and every job it started; then also any stray
    ``benchmarks.run_one`` process running one of ``jobs`` (e.g. left by a killed kernel).
    ``only_orphans=True`` limits the strays to processes whose scheduler is gone (parent pid 1),
    so a job of the same list that another live scheduler runs is left alone.

    Finished results are never touched. Jobs that were interrupted leave only a partial log;
    they restart from scratch on the next launch (results are written only at the end).
    """
    info = scheduler_info(label, dirs)
    if info:
        print(f"stopping scheduler '{label}' (pid {info['pid']}) and its jobs ...")
        _kill_tree(int(info["pid"]))
    else:
        print(f"no live scheduler '{label}'")
    ids = {job_id(j) for j in (jobs or [])}
    strays = [(pid, j) for pid, ppid, j in _run_one_procs()
              if job_id(j) in ids and (not only_orphans or ppid == 1 or not _alive(ppid))]
    for pid, j in strays:
        print(f"stopping stray job {job_id(j)} (pid {pid})")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if strays:
        time.sleep(3)
        for pid, _ in strays:
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)
    n = 0
    for f in (dirs.logs / "locks").glob("*.pid"):
        if ids and f.stem not in ids:
            continue
        pid = _lock_pid(f)
        if pid is None or not _alive(pid):
            f.unlink(missing_ok=True)
            n += 1
    _pidfile(label, dirs).unlink(missing_ok=True)
    left = [job_id(j) for _, j in _run_one_jobs() if job_id(j) in ids]
    print(f"removed {n} stale lock file(s); jobs of this notebook still running: {len(left)}")


# --------------------------------------------------------------------------- status

_EVAL = re.compile(r"step\s+(\d+)\s+eval\s+(\S+)\s+\(\s*(\d+)s\)")


def tail(path: pathlib.Path, n: int = 20) -> str:
    try:
        return "\n".join(pathlib.Path(path).read_text(errors="replace").splitlines()[-n:])
    except FileNotFoundError:
        return f"(no file {path})"


def last_eval(log: pathlib.Path) -> tuple[int, float, float] | None:
    """(step, mean eval return, seconds since start) of the latest evaluation in a job log."""
    try:
        text = pathlib.Path(log).read_text(errors="replace")
    except FileNotFoundError:
        return None
    m = None
    for m in _EVAL.finditer(text):
        pass
    if m is None:
        return None
    try:
        ret = float(m.group(2))
    except ValueError:
        ret = math.nan
    return int(m.group(1)), ret, float(m.group(3))


def final_of(path: pathlib.Path) -> float:
    """Final score of a finished run: mean of the last three evaluations (30 episodes),
    evaluations after the step budget discarded (as in benchmarks/report.py)."""
    import numpy as np

    r = json.loads(pathlib.Path(path).read_text())
    keep = [i for i, s in enumerate(r["steps"]) if s <= r["job"]["steps"]]
    eps = [x for i in keep[-3:] for x in r["returns"][i]]
    return float(np.mean(eps)) if eps else math.nan


def status(label: str, jobs: list[dict], dirs: Dirs, show: bool = True):
    """Per-job table: done / running / failed / pending, latest evaluation, progress."""
    import pandas as pd

    sched = scheduler_info(label, dirs)
    stat_file = dirs.logs / f"{label}.status"
    st = json.loads(stat_file.read_text()) if stat_file.exists() else {}
    failed = set(st.get("failed", []))
    rows = []
    for j in jobs:
        jid = job_id(j)
        res = result_path(j, dirs.results)
        lock = dirs.logs / "locks" / f"{jid}.pid"
        ev = last_eval(dirs.logs / f"{jid}.log")
        lock_pid = _lock_pid(lock)
        if res.exists():
            state = "done"
        elif lock_pid is not None and _alive(lock_pid):
            state = "running"
        elif jid in failed:
            state = "failed"
        elif ev is not None:
            state = "stopped (restarts from 0)"
        else:
            state = "pending"
        row = dict(env=j["env"], algo=j["algo"], variant=j.get("variant", "default"), seed=j["seed"],
                   device=j.get("device", "cpu"), state=state,
                   progress=f"{min(1.0, (ev[0] if ev else 0) / j['steps']):.0%}" if state != "done" else "100%",
                   last_step=ev[0] if ev else None, last_eval=round(ev[1], 1) if ev else None,
                   minutes=round(ev[2] / 60, 1) if ev else None,
                   steps_per_s=round(ev[0] / ev[2], 1) if ev and ev[2] > 0 else None,
                   eta_min=(round((j["steps"] - ev[0]) / (ev[0] / ev[2]) / 60)
                            if state == "running" and ev and ev[0] > 0 and ev[2] > 0 else None),
                   final=round(final_of(res), 1) if state == "done" else None)
        rows.append(row)
    df = pd.DataFrame(rows)
    if show:
        if sched:
            print(f"scheduler '{label}': RUNNING (pid {sched['pid']}, started {sched.get('started', '?')})")
        else:
            print(f"scheduler '{label}': not running")
        if st:
            print(f"last status write: {st.get('time', '?')} (a stopped scheduler leaves a stale file)")
        counts = df["state"].value_counts().to_dict() if not df.empty else {}
        print("jobs:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), f"(total {len(df)})")
        todo = [j for j, r in zip(jobs, rows) if r["state"] != "done"]
        if todo and sched:
            left = sum(job_seconds(j) * (1 - (r["last_step"] or 0) / j["steps"])
                       for j, r in zip(jobs, rows) if r["state"] != "done")
            print(f"remaining work ~{left / 3600:.1f} job-hours at the planning rates (divide by the "
                  "number of workers for wall-clock)")
            meas = [r["steps_per_s"] for r in rows if r["state"] == "running" and r["steps_per_s"]]
            if meas:
                print(f"measured speed of the running jobs: median {sorted(meas)[len(meas) // 2]:.0f} "
                      f"env-steps/s per job (includes evaluation time); eta_min = minutes left per job")
        for j, r in zip(jobs, rows):
            if r["state"] == "failed":
                print(f"\n--- FAILED {job_id(j)}; last lines of its log:")
                print(tail(dirs.logs / f"{job_id(j)}.log", 12))
    return df


def watch(label: str, jobs: list[dict], dirs: Dirs, minutes: float = 0, every: float = 120) -> None:
    """Refresh the status table every ``every`` seconds for ``minutes`` minutes (0 = once)."""
    from IPython.display import clear_output, display

    t_end = time.time() + 60 * minutes
    while True:
        clear_output(wait=True)
        df = status(label, jobs, dirs)
        display(df[df.state != "done"] if (df.state != "done").any() else df)
        if time.time() >= t_end or not scheduler_info(label, dirs):
            break
        time.sleep(every)


# --------------------------------------------------------------------------- analysis


def load_finals(results: pathlib.Path, suite: str, algo: str | None = None) -> dict:
    """{env: {variant: {seed: Run}}} of a suite (``benchmarks.report.load_suite``)."""
    from benchmarks.report import load_suite

    out: dict = {}
    for env, rows in load_suite(pathlib.Path(results), suite).items():
        for runs in rows.values():
            for r in runs:
                if algo is None or r.algo == algo:
                    out.setdefault(env, {}).setdefault(r.variant if algo else r.label, {})[r.seed] = r
    return out


def randoms() -> dict:
    return json.loads((BENCH / "random_scores.json").read_text())


def crossq_reference(tasks=LOCO, results: pathlib.Path | None = None, seeds=(10, 11, 12, 13, 14)) -> dict:
    """{env: [final per seed]} of the finished sb3-contrib CrossQ runs (suite ``final``)."""
    runs = load_finals(results or BENCH / "results", "final", algo="CrossQ")
    return {e: [runs[e]["default"][s].final for s in seeds if s in runs.get(e, {}).get("default", {})]
            for e in tasks}


def selection_rule(variants, tempo_results: pathlib.Path, suite: str, tasks=LOCO, tie: float = 0.02,
                   crossq_results: pathlib.Path | None = None, seeds=None, show: bool = True,
                   select: bool = True) -> dict:
    """The pre-registered rule (benchmarks/PROTOCOL.md, Screen 3), applied mechanically.

    Per task, normalise each variant's mean final score as (x - random) / (CrossQ - random),
    CrossQ = mean final score of the sb3-contrib CrossQ runs (seeds 10-14). Select the variant
    with the highest mean normalised score over the tasks; ties (< ``tie``) go to the variant
    listed first in ``variants``.
    """
    import numpy as np
    import pandas as pd

    rnd = randoms()
    cq = crossq_reference(tasks, crossq_results)
    runs = load_finals(tempo_results, suite, algo="TEMPO")
    raw, norm, nseeds = {}, {}, {}
    for v in variants:
        raw[v], norm[v], nseeds[v] = {}, {}, {}
        for e in tasks:
            got = {s: r for s, r in runs.get(e, {}).get(v, {}).items() if seeds is None or s in seeds}
            nseeds[v][e] = len(got)
            if not got or not cq.get(e):
                continue
            x = float(np.mean([r.final for r in got.values()]))
            c = float(np.mean(cq[e]))
            raw[v][e] = x
            norm[v][e] = (x - rnd[e]) / (c - rnd[e])
    need = len(seeds) if seeds else 1      # every seed of every task before the selection is final
    complete = {v: all(e in norm[v] and nseeds[v][e] >= need for e in tasks) for v in variants}
    score = {v: float(np.mean(list(norm[v].values()))) if norm[v] else math.nan for v in variants}
    rows = []
    for v in variants:
        row = {"variant": v}
        for e in tasks:
            row[e] = (f"{raw[v][e]:.0f} ({norm[v][e]:.2f}) n={nseeds[v][e]}" if e in norm[v] else "-")
        row["mean normalised"] = round(score[v], 3) if not math.isnan(score[v]) else None
        row["tasks"] = f"{len(norm[v])}/{len(tasks)}"
        rows.append(row)
    ref = {"variant": "CrossQ (seeds 10-14)"}
    for e in tasks:
        ref[e] = f"{np.mean(cq[e]):.0f} (1.00) n={len(cq[e])}" if cq.get(e) else "-"
    ref["mean normalised"], ref["tasks"] = 1.0, f"{sum(bool(cq.get(e)) for e in tasks)}/{len(tasks)}"
    rows.append(ref)
    table = pd.DataFrame(rows)
    eligible = [v for v in variants if not math.isnan(score[v])]
    selected, final = None, all(complete.values()) and bool(eligible)
    if eligible:
        best = max(score[v] for v in eligible)
        selected = next(v for v in variants if v in eligible and best - score[v] < tie)
    lines = [f"tasks: {', '.join(tasks)}; cells: mean final score over the seeds (normalised score)",
             f"normalisation: (x - random) / (CrossQ - random); tie margin {tie}; tie order: {list(variants)}"]
    if not select:
        lines = lines[:1] + ["normalisation: (x - random) / (CrossQ - random); "
                             "mean normalised score over the tasks with results: "
                             + ", ".join(f"{v} {score[v]:.3f}" for v in variants if not math.isnan(score[v]))]
    elif selected is None:
        lines.append("no results yet: nothing to select")
    else:
        lines.append(("SELECTED: " if final else "PROVISIONAL (some runs are not finished): ")
                     + f"{selected}  (mean normalised {score[selected]:.3f}; CrossQ = 1.000)")
        beats = [e for e in tasks if norm[selected].get(e, -1) > 1]
        lines.append(f"{selected} above the CrossQ mean on {len(beats)}/{len(tasks)} tasks: {beats}")
    if show:
        from IPython.display import display

        display(table)
        print("\n".join(lines))
    return dict(table=table, scores=score, normalised=norm, raw=raw, selected=selected, final=final,
                complete=complete, text="\n".join(lines))


_PALETTE = ["#e34948", "#2a78d6", "#1baf7a", "#eda100", "#008300", "#52514e", "#898781", "#383835"]
_CROSSQ = dict(color="#4a3aa7", ls=(0, (6, 1.5, 1.5, 1.5)))


def plot_vs_crossq(variants, tempo_results: pathlib.Path, suite: str, tasks=LOCO,
                   crossq_results: pathlib.Path | None = None, path: pathlib.Path | None = None,
                   title: str | None = None):
    """Learning curves (mean over seeds, min-max band) of TEMPO variants vs. CrossQ."""
    import matplotlib.pyplot as plt
    import numpy as np

    runs = load_finals(tempo_results, suite, algo="TEMPO")
    cq = load_finals(crossq_results or BENCH / "results", "final", algo="CrossQ")
    n = len(tasks)
    ncol = min(3, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow), squeeze=False)

    def band(ax, rs, color, ls, label, lw):
        L = min(len(r.curve) for r in rs)
        if L == 0:
            return
        ys = np.stack([r.curve[:L] for r in rs])
        x = rs[0].steps[:L]
        ax.plot(x, ys.mean(0), color=color, ls=ls, lw=lw, label=f"{label} (n={len(rs)})")
        if len(rs) > 1:
            ax.fill_between(x, ys.min(0), ys.max(0), color=color, alpha=0.12, lw=0)

    for k, e in enumerate(tasks):
        ax = axes[k // ncol][k % ncol]
        c = list(cq.get(e, {}).get("default", {}).values())
        if c:
            band(ax, c, _CROSSQ["color"], _CROSSQ["ls"], "CrossQ", 1.3)
        for i, v in enumerate(variants):
            rs = list(runs.get(e, {}).get(v, {}).values())
            if rs:
                band(ax, rs, _PALETTE[i % len(_PALETTE)], "-", f"TEMPO[{v}]", 1.8 if i == 0 else 1.3)
        ax.set_title(e, fontsize=10, loc="left")
        ax.grid(alpha=0.3, lw=0.5)
        ax.set_xlabel("environment steps", fontsize=8)
        ax.tick_params(labelsize=8)
        if k % ncol == 0:
            ax.set_ylabel("evaluation return", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(title or f"TEMPO variants ({suite}) vs. CrossQ (final suite, seeds 10-14)", fontsize=11,
                 x=0.01, ha="left")
    fig.tight_layout()
    if path is not None:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- screen 5: beat every baseline

FINAL_SEEDS = (10, 11, 12, 13, 14)


def _task_budgets() -> dict:
    from benchmarks.plans import SUITE, TOY_TEXT

    return {e: v[0] for e, v in {**SUITE, **TOY_TEXT}.items()}


def best_baselines(tasks, results: pathlib.Path | None = None, suite: str = "final",
                   seeds=FINAL_SEEDS) -> dict:
    """Per task, the SB3 baseline with the highest mean final score (``benchmarks/report.py``'s
    "best baseline": per run, the mean of the last three evaluations at or before the budget;
    per algorithm, the mean over its finished seeds).

    Returns ``{env: dict(algo, mean, n, means, n_seeds, missing, short)}``: ``missing`` lists the
    baselines the benchmark plans on that task but that have no finished run yet, ``short`` those
    with fewer finished seeds than ``seeds`` (the best baseline can still change there).
    """
    import numpy as np
    from benchmarks.report import BASELINES, expected_pool, load_suite

    data = load_suite(pathlib.Path(results or BENCH / "results"), suite)
    out = {}
    for e in tasks:
        rows = data.get(e, {})
        means, ns = {}, {}
        for b in BASELINES:
            rs = [r for r in rows.get(b, []) if seeds is None or r.seed in seeds]
            if rs:
                means[b] = float(np.mean([r.final for r in rs]))
                ns[b] = len(rs)
        best = max(means, key=means.get) if means else None
        planned = [b for b in BASELINES if b in expected_pool(e)]
        out[e] = dict(algo=best, mean=means.get(best), n=ns.get(best, 0), means=means, n_seeds=ns,
                      missing=[b for b in planned if b not in means],
                      short=[b for b in means if seeds is not None and ns[b] < len(seeds)])
    return out


def compute_key(variant: str) -> tuple:
    """Order of 'lower compute' for the screen-5 tie-break: critic width, then updates per step,
    then SALE, then weight norm (so ``t512_sale_gauto`` < ``tall`` = ``tall_gauto``)."""
    c = tempo_config(variant)
    return (c["width"], c["gradient_steps"], int(c["sale"]), int(c["weight_norm"]))


def discount_of(variant: str, env_id: str) -> float:
    """The discount a TEMPO variant uses on a task: a rule (``gamma="auto"``, ``"limit"`` or ``"fifth"``) is resolved from
    the time limit; for ``"limit"`` this is the value before the decision of the warm-up (episodes that end early)."""
    import gymnasium as gym

    from benchmarks import plans
    from benchmarks.variants import tempo_kwargs
    from tempo_rl import env_util

    g = tempo_kwargs(variant, env_id).get("gamma", _live_defaults()["gamma"])   # unset: the library default
    if isinstance(g, str):                       # the time limit the benchmark uses (ENV_KWARGS may add one)
        kw = plans.ENV_KWARGS.get(env_id) or {}
        limit = kw.get("max_episode_steps", gym.spec(env_id).max_episode_steps)
        rule = getattr(env_util, "horizon_gamma", None) if g in ("auto", "limit") else None
        return (rule or env_util.auto_gamma)(limit)   # "fifth" (and libraries before 2026-11-13): clip(1 - 5/T, 0.95, 0.995)
    return float(g)


def candidate_table(variants, tasks):
    """Critic width, SALE, updates per step and discount of each candidate (one row each);
    ``relative_compute`` = planning time per CPU update relative to the 256-wide recipe
    :data:`PREVIOUS_DEFAULT` (the library default when the screens were run)."""
    import pandas as pd

    rows = []
    for v in variants:
        c = tempo_config(v)
        gammas = sorted({discount_of(v, e) for e in tasks})
        rows.append(dict(variant=v, critic=f"{c['width']}x{c['width']}", sale=c["sale"],
                         weight_norm=c["weight_norm"], updates_per_step=c["gradient_steps"],
                         discount=(f"{gammas[0]:g}" if len(gammas) == 1 else
                                   f"auto ({gammas[0]:.3g}-{gammas[-1]:.4g})"),
                         relative_compute=round(job_rate(dict(algo="TEMPO", variant=PREVIOUS_DEFAULT), "cpu")
                                                / job_rate(dict(algo="TEMPO", variant=v), "cpu"), 1)))
    return pd.DataFrame(rows).set_index("variant")


def discount_table(variants, tasks):
    """The discount each candidate uses on each task (``gamma="auto"`` resolved)."""
    import pandas as pd

    return pd.DataFrame({v: {e: round(discount_of(v, e), 4) for e in tasks} for v in variants})


def beat_all_rule(variants, tempo_results: pathlib.Path, suite: str, tasks, seeds=None,
                  baseline_results: pathlib.Path | None = None, baseline_seeds=FINAL_SEEDS,
                  select: bool = True, show: bool = True, smoke: bool = False) -> dict:
    """The pre-registered Screen-5 rule (benchmarks/PROTOCOL.md, "Screen 5"), applied mechanically.

    For each candidate and task, the candidate's mean final score (mean over ``seeds`` of the
    per-run final score: the mean of the last three evaluations at or before the budget, as in
    ``benchmarks/report.py``) is compared with the best SB3 baseline's mean final score on that
    task (final seeds 10-14). WIN = strictly higher, LOSS = lower (an exact tie is shown as TIE and
    is not a win). The candidate that beats the best baseline on the most tasks is selected; ties
    are broken by the mean normalised score (x - random) / (best baseline - random) over the tasks
    (no margin), then by lower compute (:func:`compute_key`), then by the order of ``variants``.
    """
    import numpy as np
    import pandas as pd

    rnd = randoms()
    base = best_baselines(tasks, baseline_results, "final", baseline_seeds)
    runs = load_finals(pathlib.Path(tempo_results), suite, algo="TEMPO")
    need = len(seeds) if seeds else 1
    budgets = _task_budgets()
    res, wins, score, n_tasks, complete, budgets_ok = {}, {}, {}, {}, {}, {}
    for v in variants:
        res[v] = {}
        for e in tasks:
            got = {s: r for s, r in runs.get(e, {}).get(v, {}).items() if seeds is None or s in seeds}
            b = base[e]
            if not got:
                res[v][e] = dict(n=0)
                continue
            x = float(np.mean([r.final for r in got.values()]))
            if b["mean"] is None:
                verdict = "n/a"
            else:
                verdict = "WIN" if x > b["mean"] else "TIE" if x == b["mean"] else "LOSS"
            r0 = rnd.get(e)
            ok = (b["mean"] is not None and r0 is not None
                  and (b["mean"] - r0) > 1e-6 * max(1.0, abs(r0)))
            planned = budgets.get(e)
            res[v][e] = dict(n=len(got), mean=x, result=verdict,
                             norm=(x - r0) / (b["mean"] - r0) if ok else math.nan,
                             budget=min(r.budget for r in got.values()),
                             full_budget=planned is None or all(r.budget == planned for r in got.values()))
        done = [e for e in tasks if res[v][e]["n"] > 0]
        wins[v] = sum(res[v][e]["result"] == "WIN" for e in done)
        norms = [res[v][e]["norm"] for e in done if not math.isnan(res[v][e]["norm"])]
        score[v] = float(np.mean(norms)) if norms else math.nan
        n_tasks[v] = len(done)
        complete[v] = all(res[v][e]["n"] >= need for e in tasks)
        budgets_ok[v] = all(res[v][e].get("full_budget", True) for e in done)
    eligible = [v for v in variants if n_tasks[v] > 0]
    ranking = sorted(eligible, key=lambda v: (-wins[v], -(score[v] if not math.isnan(score[v]) else -1e9),
                                              compute_key(v), list(variants).index(v)))
    selected = ranking[0] if ranking else None
    base_ok = all(base[e]["algo"] is not None for e in tasks)
    base_full = all(not base[e]["missing"] and not base[e]["short"] for e in tasks)
    final = bool(selected) and all(complete.values()) and base_ok

    # ---- table: one row per task, one column per candidate
    rows = []
    for e in tasks:
        b = base[e]
        row = {"task": e}
        if b["algo"]:
            flag = " *" if (b["missing"] or b["short"]) else ""
            row["best baseline (seeds 10-14)"] = f"{b['algo']} {b['mean']:.1f} (n={b['n']}){flag}"
        else:
            row["best baseline (seeds 10-14)"] = "none finished"
        for v in variants:
            c = res[v][e]
            if c["n"] == 0:
                row[v] = "-"
            else:
                nz = f", {c['norm']:.2f}" if not math.isnan(c["norm"]) else ""
                row[v] = f"{c['result']} {c['mean']:.1f} (n={c['n']}{nz})"
        rows.append(row)
    summary = {"task": "tasks won", "best baseline (seeds 10-14)": ""}
    summary.update({v: f"{wins[v]}/{len(tasks)} (ran {n_tasks[v]})" for v in variants})
    summary2 = {"task": "mean normalised", "best baseline (seeds 10-14)": "1.00"}
    summary2.update({v: (f"{score[v]:.3f}" if not math.isnan(score[v]) else "-") for v in variants})
    table = pd.DataFrame(rows + [summary, summary2]).set_index("task")

    lines = [f"cells: WIN/LOSS against the best SB3 baseline, candidate mean final score (n = seeds, "
             f"normalised score (x - random)/(best - random); > 1 = above the best baseline)"]
    if smoke:
        lines.append("SMOKE MODE: candidate runs are a few thousand steps long; the verdicts below only "
                     "show that the pipeline works and mean nothing about the algorithm.")
    notes = []
    for e in tasks:
        b = base[e]
        if b["missing"]:
            notes.append(f"{e}: {', '.join(b['missing'])} not finished")
        if b["short"]:
            notes.append(f"{e}: " + ", ".join(f"{a} n={b['n_seeds'][a]}" for a in b["short"]))
    if notes:
        lines.append("* baselines not complete on seeds 10-14 (the best baseline may still change): "
                     + "; ".join(notes))
    if not select:
        pass
    elif selected is None:
        lines.append("no candidate results yet: nothing to select")
    else:
        head = "SELECTED" if final else "PROVISIONAL (some candidate runs are not finished)"
        lines.append(f"{head}: {selected}  (beats the best baseline on {wins[selected]}/{len(tasks)} tasks; "
                     f"mean normalised {score[selected]:.3f})")
        lines.append("ranking: " + " > ".join(f"{v} ({wins[v]} wins, {score[v]:.3f})" for v in ranking))
        lost = [e for e in tasks if res[selected][e].get("result") in ("LOSS", "TIE")]
        if lost:
            lines.append(f"{selected} does not beat the best baseline on: {lost}")
        if not base_full:
            lines.append("note: re-run this cell after the missing baseline runs finish (notebook 03 runs them)")
        if not all(budgets_ok[v] for v in eligible) and not smoke:
            lines.append("WARNING: some candidate runs used a budget different from the suite's: "
                         "they are not comparable with the baselines")
    text = "\n".join(lines)
    if show:
        from IPython.display import display

        display(table)
        print(text)
    return dict(table=table, results=res, wins=wins, scores=score, ranking=ranking, selected=selected,
                final=final, complete=complete, baselines=base, text=text)


def _mean_curve(runs):
    """Mean learning curve over runs (on the first run's evaluation steps, truncated to the
    shortest run) with the min / max over runs."""
    import numpy as np

    L = min(len(r.curve) for r in runs)
    if L == 0:
        return None
    ys = np.stack([r.curve[:L] for r in runs])
    return runs[0].steps[:L], ys.mean(0), ys.min(0), ys.max(0)


#: TEMPO candidates: TEMPO's red at three lightness steps, each with its own marker and dash, so
#: they read as one family and stay distinguishable without colour; baselines keep the report's
#: colours and dashes (benchmarks/report.py STYLE); the 256-wide default is a neutral reference.
_CAND = [dict(color="#e34948", ls="-", marker="o"), dict(color="#9e1f1e", ls=(0, (5, 1.5)), marker="s"),
         dict(color="#f09a97", ls=(0, (1.5, 1)), marker="^"), dict(color="#5e0f0f", ls="-.", marker="D")]
_DEFAULT_REF = dict(color="#898781", ls=(0, (3, 2)), lw=1.4)


def plot_vs_baselines(variants, tempo_results: pathlib.Path, suite: str, tasks, seeds=None,
                      baseline_results: pathlib.Path | None = None, baseline_seeds=FINAL_SEEDS,
                      show_default: bool = True, path: pathlib.Path | None = None, title: str | None = None):
    """Learning curves per task: TEMPO candidates (mean over ``seeds``, min-max band) against every
    SB3 baseline (mean over seeds 10-14) and, as a reference, the 256-wide default (seeds 10-14)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    from benchmarks.report import BASELINES, STYLE

    cands = load_finals(pathlib.Path(tempo_results), suite, algo="TEMPO")
    final = load_finals(pathlib.Path(baseline_results or BENCH / "results"), "final")
    n = len(tasks)
    ncol = min(3, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.0 * nrow), squeeze=False)
    handles: dict = {}
    kfmt = FuncFormatter(lambda x, _: f"{x / 1000:g}k" if x else "0")
    for k, e in enumerate(tasks):
        ax = axes[k // ncol][k % ncol]
        for b in BASELINES:
            rs = [r for s, r in final.get(e, {}).get(b, {}).items() if baseline_seeds is None or s in baseline_seeds]
            mc = _mean_curve(rs) if rs else None
            if mc is None:
                continue
            st = STYLE[b]
            ax.plot(mc[0], mc[1], color=st["color"], ls=st["ls"], lw=1.2, zorder=3)
            handles.setdefault(b, Line2D([], [], color=st["color"], ls=st["ls"], lw=1.5, label=b))
        if show_default:
            rs = [r for s, r in final.get(e, {}).get("TEMPO", {}).items()
                  if baseline_seeds is None or s in baseline_seeds]
            mc = _mean_curve(rs) if rs else None
            if mc is not None:
                ax.plot(mc[0], mc[1], zorder=2, **_DEFAULT_REF)
                handles.setdefault("default", Line2D([], [], label="TEMPO default (256-wide)", **_DEFAULT_REF))
        for i, v in enumerate(variants):
            rs = [r for s, r in cands.get(e, {}).get(v, {}).items() if seeds is None or s in seeds]
            mc = _mean_curve(rs) if rs else None
            if mc is None:
                continue
            st = _CAND[i % len(_CAND)]
            x, m, lo, hi = mc
            if len(rs) > 1:
                ax.fill_between(x, lo, hi, color=st["color"], alpha=0.13, lw=0, zorder=4)
            ax.plot(x, m, color=st["color"], ls=st["ls"], lw=2.0, marker=st["marker"], markersize=4,
                    markevery=max(1, len(x) // 6), zorder=6 + i)
            handles.setdefault(v, Line2D([], [], color=st["color"], ls=st["ls"], lw=2.0, marker=st["marker"],
                                         markersize=5, label=f"TEMPO[{v}]"))
        ax.set_title(e, fontsize=10, loc="left")
        ax.grid(alpha=0.3, lw=0.5)
        ax.xaxis.set_major_formatter(kfmt)
        ax.tick_params(labelsize=8)
        if k // ncol == nrow - 1 or k + ncol >= n:
            ax.set_xlabel("environment steps", fontsize=8)
        if k % ncol == 0:
            ax.set_ylabel("evaluation return", fontsize=8)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    order = [v for v in variants if v in handles] + [b for b in BASELINES if b in handles] + \
        (["default"] if "default" in handles else [])
    fig.legend(handles=[handles[h] for h in order], loc="lower center", ncol=min(len(order), 5), fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title or f"TEMPO candidates ({suite}) vs. every SB3 baseline (final suite, seeds 10-14)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05 if nrow > 1 else 0.1, 1, 1))
    if path is not None:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- misc


def run(cmd: list[str], cwd: pathlib.Path | None = None, env: dict | None = None,
        check: bool = True, timeout: float | None = None, quiet: bool = False) -> subprocess.CompletedProcess:
    """Run a command, print its output, raise on failure when ``check``."""
    print("$", " ".join(str(c) for c in cmd))
    p = subprocess.run([str(c) for c in cmd], cwd=cwd or ROOT, env={**os.environ, **(env or {})},
                       capture_output=True, text=True, timeout=timeout)
    text = (p.stdout or "") + (p.stderr or "")
    if not quiet or p.returncode != 0:
        print(text.rstrip())
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed with exit code {p.returncode}: {' '.join(map(str, cmd))}")
    return p


def protocol_section(heading: str) -> str:
    """The text of one ``## ...`` section of benchmarks/PROTOCOL.md (whose title contains ``heading``)."""
    if not (BENCH / "PROTOCOL.md").exists():
        return "(benchmarks/PROTOCOL.md is not part of this copy)"
    text = (BENCH / "PROTOCOL.md").read_text()
    parts = re.split(r"(?m)^## ", text)
    for p in parts[1:]:
        if heading.lower() in p.splitlines()[0].lower():
            return "## " + p.rstrip()
    return f"(no section '{heading}' in PROTOCOL.md)"


# =========================================================================== per-environment notebooks
#
# ``notebooks/environments/NN_<EnvId>.ipynb`` run the final 5-seed benchmark of ONE task end to end:
# TEMPO (library defaults or a named variant) and the tuned SB3 baselines of that task, with the
# result locations of suite ``final`` (shared with notebooks 03 and 04), then the statistics, the
# learning curves and a per-task table. Every notebook calls the helpers below in the same order:
#
#   EXP  = env_bench(...)            settings of the run (directories, label, workers)
#   CFG  = tempo_run_config(EXP)     the exact TEMPO configuration, instantiated from the code that runs
#   JOBS = env_jobs(EXP, CFG)        TEMPO x seeds + the baselines of the task, devices assigned
#   env_status / env_estimate        plan table and wall-clock estimate
#   env_launch / env_watch           background scheduler + progress table; env_stop stops it
#   env_summary / plot_env / save_env   results, learning curves, per-task table (.md / .json)

#: The suite of the final benchmark (seeds 10-14); smoke runs use ``smoke_final`` in a scratch folder.
FINAL_SUITE = "final"
#: Where the per-task figure and tables go (under ``benchmarks/figures`` and ``benchmarks/tables``).
ENV_SUBDIR = "environments"
#: TEMPO critics at least this wide run on the Apple GPU (MPS); narrower ones on the CPU.
GPU_MIN_WIDTH = 512
#: Default number of parallel Apple-GPU jobs (machine-wide limit) when MPS works.
AUTO_MPS_WORKERS = 6
#: Bootstrap resamples for the per-task confidence intervals (as ``benchmarks/report.py``).
N_BOOT = 10_000

#: Plain names of the task families of ``benchmarks.plans.SUITE`` / ``TOY_TEXT``.
FAMILY_TITLES = {"classic-discrete": "classic control, discrete actions",
                 "classic-continuous": "classic control, continuous actions",
                 "box2d-discrete": "Box2D, discrete actions", "box2d-continuous": "Box2D, continuous actions",
                 "mujoco": "MuJoCo", "toy-text": "toy text (tabular)"}


def task_info(env_id: str) -> dict:
    """Budget, evaluation frequency, family, baselines and reference scores of a benchmark task
    (``benchmarks.plans``, ``benchmarks/random_scores.json``, the gymnasium registry)."""
    import gymnasium as gym

    from benchmarks import plans
    from benchmarks.report import BASELINES, expected_pool

    table = {**plans.SUITE, **plans.TOY_TEXT}
    if env_id in table:
        steps, freq, fam = table[env_id]
        budget, freq, n_evals = int(steps), int(freq), int(steps // freq)
    elif env_id in plans.ADDED:       # added to the test folders (2026-11-03): no earlier budget, runlab sets the protocol
        budget, freq, n_evals, fam = None, None, None, plans.ADDED[env_id]
    else:
        raise ValueError(f"{env_id!r} is not a benchmark task; known: {[*table, *plans.ADDED]}")
    spec = gym.spec(env_id)
    kw = plans.ENV_KWARGS.get(env_id) or {}
    return dict(env=env_id, budget=budget, eval_freq=freq, n_evals=n_evals, family=fam,
                family_title=FAMILY_TITLES.get(fam, fam), toy_text=env_id in plans.TOY_TEXT_ALL,
                discrete=env_id in plans.DISCRETE,
                baselines=tuple(b for b in BASELINES if b in expected_pool(env_id)),
                env_kwargs=kw or None, time_limit=kw.get("max_episode_steps", spec.max_episode_steps),
                threshold=spec.reward_threshold, random=randoms().get(env_id))


_TOY_CACHE: dict = {}


def _toy_text_values(env_id: str):
    """Finite-horizon value iteration on a toy-text task's transition table ``P`` within its time limit:
    ``(q, initial_state_distribution)`` with ``q[k]`` = optimal action values with ``k + 1`` steps left."""
    if env_id not in _TOY_CACHE:
        import gymnasium as gym
        import numpy as np

        from benchmarks import plans

        env = gym.make(env_id, **(plans.ENV_KWARGS.get(env_id) or {}))
        u, horizon = env.unwrapped, int(env.spec.max_episode_steps)
        n_s, n_a = u.observation_space.n, u.action_space.n
        v, qs = np.zeros(n_s), []
        for _ in range(horizon):
            q = np.zeros((n_s, n_a))
            for s in range(n_s):
                for a in range(n_a):
                    q[s, a] = sum(p * (r + (0.0 if done else v[s2])) for p, s2, r, done in u.P[s][a])
            qs.append(q)
            v = q.max(1)
        _TOY_CACHE[env_id] = (qs, np.asarray(u.initial_state_distrib, float))
        env.close()
    return _TOY_CACHE[env_id]


def _blackjack(env_id: str):
    """``(optimum, policy)`` of Blackjack-v1 by exact dynamic programming: gymnasium's rules with an infinite deck (each
    draw is 1-9 with probability 1/13, a 10-valued card with 4/13), the dealer draws to 17 (soft 17 stands), with the
    task's ``sab`` / ``natural`` flags (registered default sab=True: a natural blackjack wins unless the dealer also has
    one). Blackjack has no transition table ``P`` and no time limit, so :func:`_toy_text_values` cannot handle it.
    ``policy[(player sum, dealer card, usable ace)]`` = optimal action (1 hit, 0 stick) for every reachable observation;
    the observation is a sufficient statistic (an ace counted as 1 never becomes usable again)."""
    if (env_id, "blackjack") not in _TOY_CACHE:
        from functools import lru_cache

        import gymnasium as gym

        from benchmarks import plans

        u = gym.make(env_id, **(plans.ENV_KWARGS.get(env_id) or {})).unwrapped
        sab, natural = bool(u.sab), bool(u.natural)
        card = {c: (4 / 13 if c == 10 else 1 / 13) for c in range(1, 11)}

        def total(raw, ace):                         # hand value with a usable ace counted as 11
            return raw + 10 if ace and raw + 10 <= 21 else raw

        @lru_cache(None)
        def dealer(raw, ace):                        # {final score: probability}, 0 = bust
            s = total(raw, ace)
            if s > 21 or s >= 17:
                return {0 if s > 21 else s: 1.0}
            out: dict = {}
            for c, p in card.items():
                for k, q in dealer(raw + c, ace or c == 1).items():
                    out[k] = out.get(k, 0.0) + p * q
            return out

        final = {}                                   # dealer card -> {final score: probability} over the hidden card
        for up in card:
            final[up] = {}
            for h, p in card.items():
                for k, q in dealer(up + h, up == 1 or h == 1).items():
                    final[up][k] = final[up].get(k, 0.0) + p * q

        def stick(s, up):
            return sum(q * ((s > k) - (s < k)) for k, q in final[up].items())

        @lru_cache(None)
        def value(raw, ace, up):                     # optimal value of a hand that is not a natural
            if total(raw, ace) > 21:
                return -1.0
            return max(stick(total(raw, ace), up), sum(p * value(raw + c, ace or c == 1, up) for c, p in card.items()))

        def best(raw, ace, up):
            hit = sum(p * value(raw + c, ace or c == 1, up) for c, p in card.items())
            return 1 if hit > stick(total(raw, ace), up) else 0

        optimum = 0.0
        for up, pu in card.items():
            p_dealer_natural = 4 / 13 if up == 1 else 1 / 13 if up == 10 else 0.0
            for c1, p1 in card.items():
                for c2, p2 in card.items():
                    if sorted((c1, c2)) == [1, 10]:  # a natural: stick (worth more than hitting soft 21)
                        if sab:
                            v = 1.0 - p_dealer_natural
                        else:
                            win = sum(q for k, q in final[up].items() if k != 21)
                            v = win * (1.5 if natural else 1.0)
                        v = max(v, sum(p * value(11 + c, True, up) for c, p in card.items()))
                    else:
                        v = value(c1 + c2, c1 == 1 or c2 == 1, up)
                    optimum += pu * p1 * p2 * v
        policy = {(total(raw, ace), up, int(ace and raw + 10 <= 21)): best(raw, ace, up)
                  for raw in range(2, 22) for ace in (False, True) for up in card if total(raw, ace) <= 21}
        _TOY_CACHE[(env_id, "blackjack")] = (float(optimum), policy)
    return _TOY_CACHE[(env_id, "blackjack")]


def toy_text_optimum(env_id: str) -> float:
    """Exact optimal expected return of a toy-text task within its time limit (finite-horizon value
    iteration on the environment's transition table ``P``, from its initial-state distribution;
    Blackjack: exact dynamic programming over the infinite deck, :func:`_blackjack`)."""
    import numpy as np

    if env_id.startswith("Blackjack"):
        return _blackjack(env_id)[0]
    qs, init = _toy_text_values(env_id)
    return float(np.dot(init, qs[-1].max(1)))


def toy_text_optimal_eval(env_id: str, n_eval: int = 10) -> float:
    """Mean return of an optimal policy (the finite-horizon value-iteration policy of
    :func:`toy_text_optimum`) on the benchmark's evaluation episodes, played exactly like
    ``benchmarks.run_one.evaluate``: episode *i* reset with the held-out seed 100000 + *i*. Every
    evaluation of every run replays these same episodes, so on a stochastic task (FrozenLake) a run's
    final score is a success rate over ``n_eval`` fixed episodes and can differ from the exact
    expectation in either direction, even for an optimal policy."""
    import gymnasium as gym
    import numpy as np

    from benchmarks import plans
    from benchmarks.run_one import EVAL_SEED_BASE

    key = (env_id, "eval", int(n_eval))
    if key not in _TOY_CACHE:
        if env_id.startswith("Blackjack"):           # observation (player sum, dealer card, usable ace), no time limit
            policy = _blackjack(env_id)[1]

            def act(obs, t):
                return policy[(int(obs[0]), int(obs[1]), int(obs[2]))]
        else:
            qs, _ = _toy_text_values(env_id)

            def act(obs, t):
                return int(np.argmax(qs[len(qs) - 1 - t][int(obs)]))
        env = gym.make(env_id, **(plans.ENV_KWARGS.get(env_id) or {}))
        rets = []
        for i in range(int(n_eval)):
            obs, _ = env.reset(seed=EVAL_SEED_BASE + i)
            done, total, t = False, 0.0, 0
            while not done:
                obs, r, term, trunc, _ = env.step(act(obs, t))
                total, t, done = total + float(r), t + 1, term or trunc
            rets.append(total)
        env.close()
        _TOY_CACHE[key] = float(np.mean(rets))
    return _TOY_CACHE[key]


def pool(device) -> str:
    """Worker pool of a device: "cpu", or "mps" for any GPU (Apple "mps", NVIDIA "cuda" / "cuda:<i>"); the GPU limits are
    the ``mps_workers`` / ``MPS_WORKERS`` ones, and planning rates of a CUDA GPU are guessed from the Apple GPU's."""
    return "cpu" if str(device or "cpu") == "cpu" else "mps"


def gpu_available() -> bool:
    """A GPU torch can use: NVIDIA CUDA or the Apple GPU (MPS)."""
    try:
        import torch

        return bool(torch.cuda.is_available() or torch.backends.mps.is_available())
    except Exception:                                   # noqa: BLE001
        return False


def mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:                                   # noqa: BLE001
        return False


def check_kernel() -> None:
    """Print the Python / PyTorch / Apple-GPU / package setup of this kernel (the benchmark jobs run
    with the same interpreter) and warn about anything that would make the runs fail."""
    import importlib
    import platform

    print(f"kernel     : Python {platform.python_version()} ({sys.executable})")
    try:
        import torch

        mps = torch.backends.mps.is_available()
        ok = ""
        if mps:
            x = torch.ones(256, 256, device="mps")
            ok = " (matmul check " + ("ok" if float((x @ x).mean().cpu()) == 256.0 else "FAILED") + ")"
        print(f"PyTorch    : {torch.__version__}; Apple GPU (MPS): {'available' + ok if mps else 'not available'}; "
              f"NVIDIA GPU (CUDA): {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}; "
              f"CPU cores: {os.cpu_count()}")
    except Exception as e:                              # noqa: BLE001
        print(f"PyTorch    : MISSING ({type(e).__name__}: {e})")
    vers, missing = [], []
    for mod in ("gymnasium", "mujoco", "Box2D", "stable_baselines3", "sb3_contrib", "scipy", "pandas", "matplotlib"):
        try:
            vers.append(f"{mod} {getattr(importlib.import_module(mod), '__version__', '?')}")
        except Exception:                               # noqa: BLE001
            missing.append(mod)
    print("packages   : " + ", ".join(vers))
    if missing:
        print(f"MISSING    : {', '.join(missing)} (tasks that need them will fail)")
    import tempo_rl

    src = pathlib.Path(tempo_rl.__file__).resolve().parent
    print(f"tempo_rl   : {tempo_rl.__version__} imported from {src}")
    if not src.is_relative_to(ROOT):
        print("             WARNING: not this project's tempo_rl (an installed copy shadows it?)")


def describe_env(env_id: str) -> dict:
    """Print the task's spaces and time limit (read live from gymnasium), the budget and the
    evaluation schedule of the benchmark, the reference scores and the baselines."""
    import gymnasium as gym

    info = task_info(env_id)
    env = gym.make(env_id, **(info["env_kwargs"] or {}))
    info.update(observation_space=str(env.observation_space), action_space=str(env.action_space))
    env.close()
    print(f"task              : {env_id}  ({info['family_title']})")
    print(f"observation space : {info['observation_space']}")
    print(f"action space      : {info['action_space']}")
    print(f"time limit        : {info['time_limit']} steps per episode"
          + ("  (added by the benchmark: benchmarks.plans.ENV_KWARGS)" if info["env_kwargs"] else "")
          + ("  (none: an episode ends after a few cards)" if info["time_limit"] is None else ""))
    if info["threshold"] is not None:
        print(f"solved at         : {info['threshold']:g}  (gymnasium reward_threshold)")
    if info["budget"] is not None:
        print(f"budget            : {info['budget']:,} environment steps per run; evaluation every "
              f"{info['eval_freq']:,} steps ({info['n_evals']} evaluations of 10 episodes)")
    else:
        print("budget            : (not part of the earlier benchmark: the run notebooks' protocol)")
    if info["random"] is not None:
        print(f"random policy     : {info['random']:{',.0f' if abs(info['random']) >= 1e4 else '.4g'}}  "
              "(mean return, benchmarks/random_scores.json)")
    if info["toy_text"]:
        info["optimum"] = toy_text_optimum(env_id)
        print(f"known optimum     : {info['optimum']:.4g}  (exact, "
              + ("dynamic programming over the infinite deck)" if env_id.startswith("Blackjack")
                 else "value iteration within the time limit)"))
        print("baselines         : none (no tuned RL-Zoo baselines for toy text): TEMPO is compared with "
              "the optimum and the random policy")
    else:
        print(f"baselines         : {', '.join(info['baselines'])}  (Stable-Baselines3 / sb3-contrib, "
              "RL-Baselines3-Zoo hyperparameters)")
    return info


# --------------------------------------------------------------------------- the code that runs


def code_root_for(suite: str) -> pathlib.Path:
    """The directory whose ``tempo_rl`` a run of ``suite`` imports (``benchmarks.run_one.use_snapshot``):
    the frozen copy ``benchmarks/snapshots/<suite>`` if it exists, else the live library (project root)."""
    snap = BENCH / "snapshots" / suite
    return snap if (snap / "tempo_rl").is_dir() else ROOT


def source_hash(code_root: pathlib.Path) -> str:
    """Hash of ``<code_root>/tempo_rl/*.py``: the ``tempo_source`` that ``benchmarks.run_one``
    records in every TEMPO result (same bytes, same order)."""
    h = hashlib.sha256()
    for p in sorted((pathlib.Path(code_root) / "tempo_rl").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def freeze_code(suite: str = FINAL_SUITE) -> pathlib.Path:
    """Copy the live ``tempo_rl/`` to ``benchmarks/snapshots/<suite>/tempo_rl`` unless a snapshot
    exists. ``benchmarks.run_one`` then imports that copy for every job of the suite, so all runs of
    the final benchmark use the same code even if the library is edited while they run. Delete the
    folder to go back to the live library. Never used for smoke suites."""
    if suite.startswith("smoke_"):
        raise ValueError("smoke suites are never frozen")
    snap = BENCH / "snapshots" / suite
    if (snap / "tempo_rl").is_dir():
        return snap
    tmp = BENCH / "snapshots" / f".{suite}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(ROOT / "tempo_rl", tmp / "tempo_rl", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (tmp / "FROZEN.json").write_text(json.dumps(dict(
        frozen=time.strftime("%Y-%m-%d %H:%M:%S"), source_hash=source_hash(tmp), frozen_from=str(ROOT / "tempo_rl"),
        by="notebooks/environments (benchlab.freeze_code)"), indent=1))
    try:
        os.rename(tmp, snap)                   # atomic: two notebooks launched at once freeze once
    except OSError:
        shutil.rmtree(tmp, ignore_errors=True)
    return snap


def _tempo_source(path: pathlib.Path) -> str | None:
    try:
        return json.loads(pathlib.Path(path).read_text()).get("tempo_source")
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- settings of one task


@dataclass
class EnvExperiment:
    """Settings of one per-environment benchmark notebook (built by :func:`env_bench`)."""
    env: str
    variant: str
    seeds: tuple
    smoke: bool
    cpu_workers: int
    mps_workers: int
    run_baselines: bool
    rerun_outdated: bool
    freeze_code: bool
    dirs: Dirs
    label: str
    suite: str
    info: dict
    figures: pathlib.Path
    tables: pathlib.Path

    @property
    def tempo_label(self) -> str:
        """Column name of this TEMPO in the report: ``TEMPO`` (defaults) or ``TEMPO[<variant>]``."""
        return "TEMPO" if self.variant == "default" else f"TEMPO[{self.variant}]"

    @property
    def baselines(self) -> tuple:
        return self.info["baselines"]

    @property
    def code_root(self) -> pathlib.Path:
        return code_root_for(self.suite)

    def code_hash(self) -> str:
        """Source hash of the tempo_rl the runs use (the snapshot once frozen, else the live library)."""
        return source_hash(self.code_root)


def env_bench(env_id: str, variant: str = "default", seeds=FINAL_SEEDS, smoke: bool = False,
              cpu_workers: int | None = None, mps_workers: int | None = None, run_baselines: bool = True,
              rerun_outdated: bool = True, freeze: bool = True, show: bool = True) -> EnvExperiment:
    """Resolve the settings of a per-environment notebook.

    * ``mps_workers=None``: 6 when the Apple GPU works (else 0); ``cpu_workers=None``: the CPU cores
      minus the MPS workers minus 2 (10 on an 18-core Mac). Both are machine-wide limits: the
      scheduler counts the jobs of every other scheduler on the device toward them, so several
      environment notebooks can run at once without oversubscribing the machine.
    * ``smoke=True``: first seed only, 1,500-step runs (suite ``smoke_final``), results, logs, figures
      and tables in ``<scratch>/env_<task>/``, at most 2 CPU + 1 MPS workers, scheduler label
      ``smoke_env_<task>``; nothing is frozen and nothing under ``benchmarks/`` is written.
    """
    from benchmarks.variants import VARIANTS

    info = task_info(env_id)
    if variant not in VARIANTS:
        raise ValueError(f"unknown TEMPO_VARIANT {variant!r}; known (benchmarks/variants.py): {sorted(VARIANTS)}")
    seeds = tuple(int(s) for s in seeds)
    if not seeds:
        raise ValueError("SEEDS is empty")
    gpu = mps_available()
    if mps_workers is None:
        mps_workers = AUTO_MPS_WORKERS if gpu else 0
    elif mps_workers and not gpu:
        print("note: the Apple GPU (MPS) is not available here: MPS_WORKERS set to 0")
        mps_workers = 0
    if cpu_workers is None:
        cpu_workers = max(1, (os.cpu_count() or 8) - int(mps_workers) - 2)
    if smoke:
        seeds = seeds[:1]
        cpu_workers, mps_workers = min(int(cpu_workers), 2), min(int(mps_workers), 1)
    name = f"env_{env_id}"
    dirs = bench_dirs(smoke, name)
    exp = EnvExperiment(env=env_id, variant=variant, seeds=seeds, smoke=smoke, cpu_workers=int(cpu_workers),
                        mps_workers=int(mps_workers), run_baselines=bool(run_baselines),
                        rerun_outdated=bool(rerun_outdated), freeze_code=bool(freeze) and not smoke,
                        dirs=dirs, label=label_for(name, smoke), suite=("smoke_" if smoke else "") + FINAL_SUITE,
                        info=info, figures=dirs.figures / ENV_SUBDIR, tables=dirs.tables / ENV_SUBDIR)
    if show:
        steps = SMOKE_STEPS if smoke else info["budget"]
        print(f"task      : {env_id}; seeds {list(seeds)}; {steps:,} steps per run"
              + ("  [SMOKE: pipeline check only; the numbers mean nothing]" if smoke else ""))
        print(f"TEMPO     : variant '{variant}'" + ("  (= TEMPO(\"MlpPolicy\", env) with the library defaults)"
                                                    if variant == "default" else "  (benchmarks/variants.py)"))
        print(f"baselines : {', '.join(info['baselines']) or 'none (toy text)'}"
              + (f"  (missing runs are {'run' if run_baselines else 'NOT run'}; finished runs are reused)"
                 if info["baselines"] else ""))
        print(f"results   : {dirs.results / exp.suite / env_id}/<algorithm>/<variant>/seed<k>.json")
        print(f"logs      : {dirs.logs}  (scheduler label '{exp.label}')")
        print(f"figure    : {exp.figures / env_id}.png / .pdf")
        print(f"tables    : {exp.tables / env_id}.md / .json")
        print(f"workers   : {exp.cpu_workers} CPU + {exp.mps_workers} Apple-GPU (machine-wide limits, "
              "shared with every other running scheduler)")
    return exp


# --------------------------------------------------------------------------- TEMPO configuration

_CONFIG_SCRIPT = r'''
import inspect, json, math, sys, warnings
warnings.filterwarnings("ignore")
a = json.loads(sys.argv[1])
sys.path[:0] = list(dict.fromkeys([a["code_root"], a["root"]]))   # like benchmarks.run_one.use_snapshot
import numpy as np
import torch
torch.set_num_threads(1)
import tempo_rl
from tempo_rl import TEMPO
from tempo_rl import policies as P
import tempo_rl.tempo as T
from benchmarks.variants import tempo_kwargs

def enc(v):
    if v is inspect.Parameter.empty:
        return "(required)"
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [enc(x) for x in v]
    if isinstance(v, dict):
        return {str(k): enc(x) for k, x in v.items()}
    if isinstance(v, type):
        return v.__name__
    return repr(v)

def get(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default

kw = tempo_kwargs(a["variant"], a["env"])
phys = kw.pop("_physical_entropy", False)
pk_var = dict(kw.get("policy_kwargs") or {})
harness = dict(seed=a["seed"], device=a["device"], env_kwargs=a["env_kwargs"])
skip = {"self", "policy", "env", "_init_setup_model", "tensorboard_log", "verbose", "stats_window_size"}
rows = []
for k, p in inspect.signature(TEMPO.__init__).parameters.items():
    if k in skip or k == "policy_kwargs":
        continue
    if k in kw:
        rows.append(["TEMPO argument", k, enc(kw[k]), "variant"])
    elif k in harness:
        rows.append(["TEMPO argument", k, enc(harness[k]), "benchmark harness"])
    else:
        rows.append(["TEMPO argument", k, enc(p.default), "library default"])
pskip = {"self", "observation_space", "action_space", "n_bins", "novelty"}
for k, p in inspect.signature(P.TempoPolicy.__init__).parameters.items():
    if k in pskip:
        continue
    if k in pk_var:
        rows.append(["policy_kwargs", k, enc(pk_var[k]), "variant"])
    elif k == "net_arch" and p.default is None:
        rows.append(["policy_kwargs", k, "None -> " + json.dumps(enc(P.DEFAULT_ARCH)), "library default"])
    else:
        rows.append(["policy_kwargs", k, enc(p.default), "library default"])
for k in pk_var:
    if k not in {r[1] for r in rows if r[0] == "policy_kwargs"}:
        rows.append(["policy_kwargs", k, enc(pk_var[k]), "variant"])

def build(kwargs):
    """TEMPO exactly as benchmarks.run_one builds it (on the CPU: only the configuration is read)."""
    return TEMPO("MlpPolicy", a["env"], seed=a["seed"], device="cpu", env_kwargs=a["env_kwargs"], **kwargs)

def fingerprint(m):
    """Every resolved setting that decides what a run does, plus the network shapes and a hash of the
    initial weights: two models with equal fingerprints train identically for the same seed."""
    import hashlib
    p = m.policy
    lr_ = get(m, "lr_schedule")
    fp = {k: enc(get(m, k)) for k in ("gamma", "gamma_request", "n_step", "policy_delay", "gradient_steps",
                                      "train_freq", "batch_size", "tau", "buffer_size", "learning_starts",
                                      "critic_loss", "representation_lr", "exploration_bonus",
                                      "exploration_noise", "max_grad_norm", "ent_coef", "_schedule",
                                      "_entropy_unit", "_entropy_log_offset", "policy_average")}
    fp.update(target_entropy=enc(get(m, "target_entropy")) if isinstance(get(m, "target_entropy"), (str, float)) else "custom",
              reward_scaling=get(m, "reward_scaler") is not None,
              learning_rate=[lr_(1.0), lr_(0.5), lr_(0.0)] if lr_ is not None else None)
    fp.update({f"policy.{k}": enc(get(p, k)) for k in ("net_arch", "batch_norm", "n_bins", "weight_norm", "zsa_norm",
                                                       "representation", "augment_images")})
    fp["policy.shapes"] = {k: list(v.shape) for k, v in p.state_dict().items()}
    h = hashlib.sha256()
    for k, v in sorted(p.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().double().numpy().tobytes())
    fp["policy.initial_weights_sha256"] = h.hexdigest()[:16]
    return fp

model = build(kw)
pol = model.policy
arch = get(pol, "net_arch", {}) or {}
bn = bool(get(pol, "batch_norm", False))
n_bins = get(pol, "n_bins")
rep = get(pol, "representation")
discrete = bool(get(model, "discrete", False))
sched = tuple(get(model, "_schedule", ()) or ())
unit = float(get(model, "_entropy_unit", float("nan")))
bonus = float(get(model, "exploration_bonus", 0.0) or 0.0)
noise = get(model, "exploration_noise", "white")
nparams = lambda m: int(sum(x.numel() for x in m.parameters())) if m is not None else 0
fp_run = fingerprint(model)                       # before the learning-starts probe below changes the model

# ---- the recipe of this run, resolved on this task (shown above the table, one line per ingredient)
n_q = len(get(pol.critic, "qs", [])) or get(pol.critic, "n_critics", "?")
critic_txt = (f"{n_q} x MLP {arch.get('qf')}, "
              + ("Batch Renormalization, joint forward pass, no target network" if bn else "LayerNorm + Polyak target network")
              + (f"; categorical HL-Gauss critic over {n_bins} bins" if n_bins else "; MSE regression critic"))
actor_txt = f"MLP {arch.get('pi')}" + (" (categorical policy)" if discrete else " (tanh-squashed Gaussian policy)")
rep_txt = (f"SALE learned state / state-action embeddings (dim {get(get(pol, 'rep'), 'zs_dim', '?')})"
           if rep == "sale" else "none (raw observations)")
gam = get(model, "gamma")
req = get(model, "gamma_request", gam)
try:
    from tempo_rl.env_util import episode_horizon
    horizon = episode_horizon(model.env)
except Exception:
    horizon = None
_rules = {"auto": "clip(1 - 5/T, 0.99, 0.995)", "fifth": "clip(1 - 5/T, 0.95, 0.995)",
          "limit": "clip(1 - 5/T, 0.99, 0.995), or clip(1 - 1/T, 0.99, 0.999) if the warm-up shows that only the time limit "
                   "ends an episode (decided when training starts)"}
if isinstance(req, str) and not hasattr(T, "LEGACY_ENTROPY_SCHEDULE"):    # a library from before 2026-11-13
    _rules["auto"] = _rules["fifth"]
gamma_txt = f"{gam:.6g}" + (f"  (gamma={req!r}: {_rules.get(req, req)} with the time limit T = {horizon})"
                            if isinstance(req, str) else "  (fixed)")
if discrete:
    noise_txt = (f"exploration_noise={noise!r} has no effect here: discrete actions are sampled from the "
                 "categorical policy (pink noise applies to continuous actions only)")
elif noise == "pink":
    noise_txt = "pink (1/f) Gaussian noise, correlated across the steps of an episode (data collection only)"
else:
    noise_txt = f"{noise}: independent Gaussian samples at every step"
frac = sched[2] if len(sched) in (3, 5) else 0.5
bonus_txt = (f"weight {bonus:g} -> 0, decaying linearly over the first {frac:.0%} of training: RND prediction "
             "error normalised to unit scale, added to the reward" if bonus > 0 else "off")
lr = get(model, "lr_schedule")
lr_txt = f"{lr(1.0):.3g}" + ("" if lr is None or lr(1.0) == lr(0.0) else f" -> {lr(0.0):.3g} (schedule)")
if len(sched) == 3 and isinstance(get(model, "target_entropy"), str):
    s0, s1, f = sched
    if discrete:
        ent_txt = (f"{s0:g} -> {s1:g} x ln|A| over the first {f:.0%} of training "
                   f"({s0 * unit:.3g} -> {s1 * unit:.3g} nats)")
    else:
        ent_txt = (f"{s0:+g} -> {s1:+g} nats per action dimension over the first {f:.0%} of training "
                   f"({s0 * unit:+.3g} -> {s1 * unit:+.3g} nats in total)")
elif len(sched) == 5 and isinstance(get(model, "target_entropy"), str):
    s0, s1, f, s2, f2 = sched                     # a second, late anneal (the exploitation phase)
    if discrete:
        ent_txt = (f"{s0:g} -> {s1:g} x ln|A| over the first {f:.0%} of training, then -> {s2:g} x ln|A| by {f2:.0%} "
                   f"({s0 * unit:.3g} -> {s1 * unit:.3g} -> {s2 * unit:.3g} nats)")
    else:
        ent_txt = (f"{s0:+g} -> {s1:+g} nats per action dimension over the first {f:.0%} of training, then -> {s2:+g} "
                   f"by {f2:.0%} ({s0 * unit:+.3g} -> {s1 * unit:+.3g} -> {s2 * unit:+.3g} nats in total)")
else:
    ent_txt = repr(get(model, "target_entropy"))
_n = get(model, "n_step")
n_txt = (f"{_n[0]}-step returns for the first {frac:.0%} of training, then {_n[1]}-step" if isinstance(_n, (tuple, list))
         else f"n = {_n}-step returns")
_avg = get(model, "policy_average")
_phase = bool(get(model, "gpu_replay") and get(model, "act_on_cpu"))     # the averaged policy needs the phase engine
avg_txt = (f"predict / evaluation use an average of the actor with a half-life of {int(_avg):,} environment steps"
           if _avg and _phase else
           f"policy_average={int(_avg)} is not used on this engine (predict / evaluation use the newest actor)"
           if _avg else "off (predict / evaluation use the newest actor)")
recipe = [["critic", critic_txt], ["actor", actor_txt], ["representation", rep_txt],
          ["updates", f"{n_txt}; lr {lr_txt}; policy delay {get(model, 'policy_delay')}; "
                      f"{int(get(model, 'gradient_steps', 1))} update(s) per step; reward scaling "
                      + ("on" if get(model, "reward_scaler") is not None else "off")],
          ["target entropy", ent_txt], ["discount gamma", gamma_txt], ["action noise", noise_txt],
          ["novelty bonus", bonus_txt], ["averaged policy", avg_txt]]

# ---- the table: resolved settings first, then every constructor argument with its source
res = []
def r(k, v):
    res.append(["resolved for this task", k, enc(v), "instantiated model"])
r("observation / action space", f"{model.observation_space} / {model.action_space}")
r("discount gamma", gamma_txt)
r("target entropy", ent_txt)
r("learning rate", lr_txt)
ls = get(model, "learning_starts")
if ls is None:              # "auto": let the library resolve it for this budget (no rule copied here)
    try:
        model.buffer_size = 64                     # throwaway model: a tiny buffer is enough to read the value
        model._setup_learn(a["budget"], None, True, "config", False)
        ls = f"auto: {int(model._learning_starts_resolved):,} steps for the {a['budget']:,}-step budget"
    except Exception as e:
        ls = f"auto (resolved by the library when training starts; not readable here: {type(e).__name__})"
r("learning starts", ls)
gs, tf = int(get(model, "gradient_steps", 1)), int(get(model, "train_freq", 1))
r("updates per environment step", f"{gs}/{tf}")
r("multi-step return", n_txt)
r("averaged policy", avg_txt)
r("critic", critic_txt)
r("actor", actor_txt)
r("representation", rep_txt)
r("reward scaling", "on (running return scale)" if get(model, "reward_scaler") is not None else "off")
r("observation normalisation", "on (running mean / std)" if get(pol, "normalizer") is not None else "off")
r("action noise (data collection)", noise_txt)
r("novelty bonus", bonus_txt)
if phys:
    r("entropy coordinates", "physical action units (variant flag _physical_entropy)")
r("parameters", f"actor {nparams(get(pol, 'actor')):,}; critic {nparams(get(pol, 'critic')):,}"
  + (f"; representation {nparams(get(pol, 'rep')):,}" if get(pol, "rep") is not None else "")
  + (f"; novelty (RND) {nparams(get(pol, 'rnd')):,}" if get(pol, "rnd") is not None else ""))

# ---- does this configuration equal a named reference recipe (e.g. the Screen-6 winner "c1")?
match = None
if a.get("reference") and a["reference"] != a["variant"]:
    try:
        fp_ref = fingerprint(build(tempo_kwargs(a["reference"], a["env"])))
        diff = sorted(k for k in set(fp_run) | set(fp_ref) if fp_run.get(k) != fp_ref.get(k))
        match = dict(reference=a["reference"], equal=not diff, differences=diff[:12])
    except Exception as e:
        match = dict(reference=a["reference"], equal=None, error=f"{type(e).__name__}: {e}")

qf = list(arch.get("qf") or [256])
out = dict(rows=res + rows, recipe=recipe, match=match, tempo_rl_file=tempo_rl.__file__,
           version=get(tempo_rl, "__version__", "?"), kwargs=enc(kw), width=int(max(qf)), gradient_steps=gs,
           sale=rep == "sale", weight_norm=bool(get(pol, "weight_norm", False)), novelty=bonus > 0,
           exploration_bonus=bonus, exploration_noise=noise, discrete=discrete, gamma=float(gam),
           gamma_request=enc(req), horizon=horizon,
           defaults=dict(arch=enc(P.DEFAULT_ARCH), entropy_schedule=enc(get(T, "DEFAULT_ENTROPY_SCHEDULE"))))
print("@@CONFIG@@" + json.dumps(out))
'''


def reference_recipe(variant: str) -> str | None:
    """The named recipe a variant should equal: for ``"default"`` the final recipe of the screens
    (``benchmarks.variants.FINAL_RECIPE``, the Screen-6 winner ``"c1"``), else ``None``."""
    from benchmarks import variants as V

    ref = getattr(V, "FINAL_RECIPE", None)
    return ref if variant == "default" and ref in V.VARIANTS else None


def tempo_run_config(exp: EnvExperiment) -> dict:
    """The exact TEMPO configuration of this task's runs.

    Instantiates ``TEMPO("MlpPolicy", env, seed=..., **tempo_kwargs(variant, env))`` exactly as
    ``benchmarks.run_one`` does, in a fresh subprocess that imports the ``tempo_rl`` the runs will
    import (the frozen snapshot of the suite if there is one, else the live library), and returns
    every constructor argument with its source (library default / variant / benchmark harness), the
    settings resolved for this task (discount, entropy target, exploration, network sizes, ...) and,
    for ``"default"``, whether it equals the final recipe of the screens (:func:`reference_recipe`:
    same resolved settings, same network shapes, identical initial weights for the same seed). The
    compute-relevant values (critic width, SALE, novelty bonus, updates per step) also drive the
    device choice and the time estimate (:func:`tempo_config`)."""
    root = code_root_for(exp.suite)
    width_guess = tempo_config(exp.variant)["width"]
    device = "mps" if (width_guess >= GPU_MIN_WIDTH and exp.mps_workers > 0) else "cpu"
    args = dict(code_root=str(root), root=str(ROOT), env=exp.env, variant=exp.variant, seed=exp.seeds[0],
                env_kwargs=exp.info["env_kwargs"], budget=SMOKE_STEPS if exp.smoke else exp.info["budget"],
                device=device, reference=reference_recipe(exp.variant))
    p = subprocess.run([PY, "-c", _CONFIG_SCRIPT, json.dumps(args)], cwd=ROOT, capture_output=True, text=True,
                       timeout=600, env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONWARNINGS": "ignore"})
    line = next((ln for ln in reversed(p.stdout.splitlines()) if ln.startswith("@@CONFIG@@")), None)
    if p.returncode != 0 or line is None:
        raise RuntimeError("could not instantiate TEMPO for the configuration check:\n"
                           + "\n".join((p.stdout + p.stderr).splitlines()[-25:]))
    cfg = json.loads(line[len("@@CONFIG@@"):])
    _RESOLVED[exp.variant] = {k: cfg[k] for k in ("width", "gradient_steps", "sale", "weight_norm", "novelty",
                                                  "exploration_bonus", "exploration_noise")}
    _RESOLVED[exp.variant]["gamma"] = cfg["gamma_request"]
    cfg.update(env=exp.env, variant=exp.variant, code_root=str(root), snapshot=root != ROOT,
               source_hash=source_hash(root), live_hash=source_hash(ROOT),
               device="mps" if (cfg["width"] >= GPU_MIN_WIDTH and exp.mps_workers > 0) else "cpu")
    return cfg


def show_tempo_config(cfg: dict, exp: EnvExperiment | None = None) -> None:
    """Print where the TEMPO code comes from, the device, the recipe resolved on this task (one line
    per ingredient), whether it equals the final recipe of the screens, and the full configuration
    table of :func:`tempo_run_config`."""
    import pandas as pd
    from IPython.display import display

    what = ('TEMPO("MlpPolicy", env) with no keyword arguments: the library defaults'
            if cfg["variant"] == "default" else f"variant '{cfg['variant']}' of benchmarks/variants.py")
    print(f"TEMPO on {cfg['env']}: {what}")
    # -- which tempo_rl code the runs import
    if cfg["snapshot"]:
        frozen = pathlib.Path(cfg["code_root"]) / "FROZEN.json"
        when = json.loads(frozen.read_text()).get("frozen", "?") if frozen.exists() else "?"
        print(f"code    : frozen snapshot {cfg['code_root']}/tempo_rl (frozen {when}; source {cfg['source_hash']})")
        if cfg["live_hash"] != cfg["source_hash"]:
            print(f"          NOTE: the live library (source {cfg['live_hash']}) differs from the snapshot; the runs "
                  "use the SNAPSHOT. Delete that snapshot folder to run the live library instead (then every "
                  "TEMPO result of the final suite made with the snapshot counts as older code).")
    else:
        print(f"code    : live library {cfg['tempo_rl_file'].rsplit('/', 1)[0]} (tempo_rl {cfg['version']}, "
              f"source {cfg['source_hash']})")
        if exp is not None and exp.freeze_code:
            print("          FREEZE_CODE: the first real launch copies this code to benchmarks/snapshots/final; every "
                  "later final-suite run (all environment notebooks, notebook 03) then imports that same copy")
        elif exp is not None and exp.smoke:
            print("          smoke mode: nothing is frozen")
    # -- where the TEMPO runs go, and the planning speed there
    rate = job_rate(dict(algo="TEMPO", variant=cfg["variant"]), cfg["device"])
    if cfg["device"] == "mps":
        why = f"critic width {cfg['width']} >= {GPU_MIN_WIDTH} and Apple-GPU workers are available"
    elif cfg["width"] >= GPU_MIN_WIDTH:
        why = f"critic width {cfg['width']}, but no Apple-GPU workers: expect it to be several times slower"
    else:
        why = f"critic width {cfg['width']} < {GPU_MIN_WIDTH}: small networks are as fast on the CPU"
    print(f"device  : {cfg['device']} ({why}); planning rate ~{rate:.0f} env-steps/s per run")
    # -- the recipe, resolved on this task by the library itself
    for name, text in cfg.get("recipe") or []:
        print(f"{name:15s}: {text}")
    m = cfg.get("match")
    if m:
        if m.get("equal"):
            print(f"check          : identical to the recipe '{m['reference']}' of benchmarks/variants.py (same "
                  "resolved settings, same networks, same initial weights for this seed)")
        elif m.get("equal") is None:
            print(f"check          : could not compare with '{m['reference']}' ({m.get('error')})")
        else:
            print(f"check          : WARNING: differs from the recipe '{m['reference']}' of benchmarks/variants.py in "
                  + ", ".join(m.get("differences") or []))
    print()
    # -- every constructor argument with its source
    df = pd.DataFrame(cfg["rows"], columns=["group", "setting", "value", "source"])
    df["value"] = df["value"].map(lambda v: v if isinstance(v, str) else repr(v))
    df = df.set_index(["group", "setting"])
    with pd.option_context("display.max_rows", 300, "display.max_colwidth", 160):
        try:                                   # left-aligned text columns in JupyterLab
            display(df.style.set_properties(**{"text-align": "left"}).set_table_styles(
                [dict(selector="th", props=[("text-align", "left")])]))
        except Exception:                      # noqa: BLE001 - no jinja2: plain table
            display(df)


# --------------------------------------------------------------------------- jobs, plan, run


def env_jobs(exp: EnvExperiment, cfg: dict | None = None) -> list[dict]:
    """TEMPO x seeds and every baseline of the task x seeds (``benchmarks.plans``), suite ``final``
    (smoke: ``smoke_final``, 1,500 steps). Devices: TEMPO critics >= 512 wide (from ``cfg``, the
    instantiated configuration) and CrossQ on the Apple GPU when there are MPS workers, the rest on
    the CPU; ``cost`` = estimated seconds (the scheduler starts the longest jobs first)."""
    from benchmarks import plans
    from benchmarks.report import BASELINES

    jobs = plans.custom_jobs(FINAL_SUITE, variants=(exp.variant,), envs=[exp.env], seeds=exp.seeds,
                             smoke=exp.smoke, smoke_steps=SMOKE_STEPS)
    if exp.baselines:
        base = plans.baseline_jobs(algos=exp.baselines, envs={exp.env}, seeds=exp.seeds)
        if exp.smoke:
            base = shrink([dict(j, suite=exp.suite) for j in base])
        jobs += sorted(base, key=lambda j: (BASELINES.index(j["algo"]), j["seed"]))
    width = cfg["width"] if cfg else tempo_config(exp.variant)["width"]
    for j in jobs:
        gpu = j["algo"] == "CrossQ" or (j["algo"] == "TEMPO" and width >= GPU_MIN_WIDTH)
        j["device"] = "mps" if (gpu and exp.mps_workers > 0) else "cpu"
        j["threads"] = 1
        j["cost"] = round(job_seconds(j))
    return jobs


def _launchable(exp: EnvExperiment, j: dict) -> bool:
    return j["algo"] == "TEMPO" or exp.run_baselines


def _env_rows(exp: EnvExperiment, jobs: list[dict]) -> list[dict]:
    """Raw state of every job: done / older code / running / failed / to run, with progress."""
    code = exp.code_hash()
    st_file = exp.dirs.logs / f"{exp.label}.status"
    try:
        failed = set(json.loads(st_file.read_text()).get("failed", [])) if st_file.exists() else set()
    except ValueError:
        failed = set()
    rows = []
    for j in jobs:
        jid = job_id(j)
        res = result_path(j, exp.dirs.results)
        lock_pid = _lock_pid(exp.dirs.logs / "locks" / f"{jid}.pid")
        ev, final = None, None
        if res.exists():
            old = j["algo"] == "TEMPO" and _tempo_source(res) != code
            state = ("older code: re-run" if exp.rerun_outdated else "done (older code)") if old else "done"
            final = final_of(res)
        elif lock_pid is not None and _alive(lock_pid):
            state = "running"
            ev = last_eval(exp.dirs.logs / f"{jid}.log")
        elif not _launchable(exp, j):
            state = "missing (RUN_BASELINES = False)"
        elif jid in failed:
            state = "failed: re-runs at the next launch"
        else:
            state = "to run"
        frac = 1.0 if state.startswith("done") else (min(1.0, ev[0] / j["steps"]) if ev else 0.0)
        rate = ev[0] / ev[2] if ev and ev[2] > 0 and ev[0] > 0 else None
        rows.append(dict(job=j, id=jid, algorithm="TEMPO" if j["algo"] == "TEMPO" else j["algo"],
                         seed=j["seed"], device=j["device"], status=state, frac=frac,
                         last_step=ev[0] if ev else None, last_eval=ev[1] if ev else None, rate=rate,
                         eta_min=((j["steps"] - ev[0]) / rate / 60) if rate else None, final=final))
    return rows


_TODO = ("to run", "older code: re-run", "failed: re-runs at the next launch")


def env_status(exp: EnvExperiment, jobs: list[dict]):
    """Table of the task's jobs: algorithm, seed, device, status (done / running / to run / ...),
    progress, latest evaluation of a running job, final score of a finished one."""
    import pandas as pd

    rows = _env_rows(exp, jobs)
    df = pd.DataFrame([dict(algorithm=r["algorithm"] if r["algorithm"] != "TEMPO" else exp.tempo_label,
                            seed=r["seed"], device=r["device"], status=r["status"],
                            progress=f"{r['frac']:.0%}",
                            **{"last eval (step)": (f"{r['last_eval']:.1f} ({r['last_step']:,})"
                                                    if r["last_step"] else ""),
                               "env-steps/s": round(r["rate"], 1) if r["rate"] else None,
                               "min left": round(r["eta_min"]) if r["eta_min"] is not None else None,
                               "final": round(r["final"], 2) if r["final"] is not None else ""})
                       for r in rows])
    return df.astype(object).where(df.notna(), "").set_index(["algorithm", "seed"])


def env_estimate(exp: EnvExperiment, jobs: list[dict], verbose: bool = True) -> dict:
    """Wall-clock estimate of the runs still to do: running jobs keep their slot for their remaining
    time, the others are list-scheduled longest first on their device (as the scheduler does) with
    the planning rates of :func:`job_rate` (``TEMPO_RATES`` / ``SB3_RATES``)."""
    rows = _env_rows(exp, jobs)
    todo = [r for r in rows if r["status"] in _TODO]
    running = [r for r in rows if r["status"] == "running"]
    caps = {"cpu": exp.cpu_workers, "mps": exp.mps_workers}
    span = {}
    for dev in ("cpu", "mps"):
        dj = [r for r in todo if r["device"] == dev]
        if dj and caps[dev] <= 0:
            raise ValueError(f"{len(dj)} jobs need the {dev.upper()} but it has 0 workers: set "
                             f"{'CPU' if dev == 'cpu' else 'MPS'}_WORKERS > 0")
        free = [0.0] * max(1, caps[dev])
        for r in running:
            if r["device"] == dev:
                k = min(range(len(free)), key=free.__getitem__)
                free[k] += job_seconds(r["job"]) * (1 - r["frac"])
        for r in sorted(dj, key=lambda r: -job_seconds(r["job"])):
            k = min(range(len(free)), key=free.__getitem__)
            free[k] += job_seconds(r["job"])
        span[dev] = max(free) / 3600 if (dj or any(r["device"] == dev for r in running)) else 0.0
    wall = max(span.values())
    out = dict(n_jobs=len(rows), n_done=sum(r["status"].startswith("done") for r in rows), n_todo=len(todo),
               n_running=len(running), wall_hours=wall, per_device_hours=span)
    if verbose:
        n_t = sum(r["algorithm"] == "TEMPO" for r in todo)
        print(f"{out['n_done']} of {len(rows)} runs finished; {len(running)} running; {len(todo)} to run "
              f"({n_t} TEMPO, {len(todo) - n_t} baseline)")
        if todo or running:
            print(f"estimated wall-clock: ~{_fmt_hours(wall)} with {exp.cpu_workers} CPU + {exp.mps_workers} "
                  f"Apple-GPU workers (CPU {_fmt_hours(span['cpu'])}, GPU {_fmt_hours(span['mps'])}), if no "
                  "other scheduler holds the workers")
            seen = []
            for r in todo + running:
                key = (r["algorithm"], r["device"])
                if key not in seen:
                    seen.append(key)
            print("planning rates per job: " + "; ".join(
                f"{a} ~{job_rate(next(r['job'] for r in todo + running if (r['algorithm'], r['device']) == (a, d))):.0f} "
                f"env-steps/s on the {'CPU' if d == 'cpu' else 'Apple GPU'}" for a, d in seen))
            print("Real runs can take up to ~30% longer when the machine is busy with other work; the Run cell "
                  "shows the measured speed and the minutes left of every running job.")
        else:
            print("nothing left to run: go to the Results cell")
    return out


def _archive_outdated(exp: EnvExperiment, jobs: list[dict]) -> list:
    """Move TEMPO results made with other tempo_rl code (``tempo_source`` != the code that runs now)
    to ``<results>/_outdated/<time>/...`` so that the scheduler re-runs them. Nothing is deleted."""
    code, stamp, moved = exp.code_hash(), time.strftime("%Y%m%d-%H%M%S"), []
    for j in jobs:
        p = result_path(j, exp.dirs.results)
        if j["algo"] != "TEMPO" or not p.exists() or _tempo_source(p) == code:
            continue
        dest = exp.dirs.results / "_outdated" / stamp / p.relative_to(exp.dirs.results)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest))
        moved.append(dest)
    if moved:
        print(f"moved {len(moved)} TEMPO result(s) made with older tempo_rl code to "
              f"{exp.dirs.results / '_outdated' / stamp} (they are re-run now)")
    return moved


def env_launch(exp: EnvExperiment, jobs: list[dict]):
    """Start this notebook's scheduler in the background (label ``env_<task>``).

    Before launching: freeze the tempo_rl code for the final suite (``FREEZE_CODE``, first launch
    only) and, with ``RERUN_OUTDATED_TEMPO``, move TEMPO results made with older code out of the way.
    Nothing is started when every job has a result or the scheduler is already running; jobs that
    another scheduler is running (e.g. notebook 03's) are left to it. The runs survive closing the
    notebook or restarting the kernel."""
    running = scheduler_info(exp.label, exp.dirs)
    if running:
        print(f"scheduler '{exp.label}' is already running (pid {running['pid']}, started "
              f"{running.get('started', '?')}): watching it")
        return running
    if exp.freeze_code:
        existed = (BENCH / "snapshots" / FINAL_SUITE / "tempo_rl").is_dir()
        snap = freeze_code(FINAL_SUITE)
        print(("code is frozen in " if existed else "froze the live tempo_rl into ") + f"{snap}/tempo_rl "
              f"(source {source_hash(snap)})")
    if exp.rerun_outdated:
        _archive_outdated(exp, jobs)
    run = [j for j in jobs if _launchable(exp, j)]
    todo = [j for j in run if not result_path(j, exp.dirs.results).exists()]
    if any(j["device"] == "cpu" for j in todo) and exp.cpu_workers <= 0:
        raise ValueError("some jobs run on the CPU: set CPU_WORKERS > 0")
    return launch(exp.label, run, exp.dirs, exp.cpu_workers, exp.mps_workers, background=True,
                  hint="Run the Run cell again to watch the progress.")


def env_watch(exp: EnvExperiment, jobs: list[dict], every: float = 120) -> None:
    """Show a compact progress table every ``every`` seconds until no job of this task is running
    and this notebook's scheduler has exited. Interrupting (the stop button) only stops watching:
    the runs continue in the background; run the cell again to resume."""
    from IPython.display import clear_output, display

    try:
        while True:
            df = env_status(exp, jobs)
            sched = scheduler_info(exp.label, exp.dirs)
            n_run = int((df.status == "running").sum())
            clear_output(wait=True)
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {exp.env}: scheduler '{exp.label}' "
                  + (f"RUNNING (pid {sched['pid']})" if sched else "not running")
                  + f"; output: {exp.dirs.logs / (exp.label + '.out')}")
            print("jobs: " + ", ".join(f"{k} {v}" for k, v in df.status.value_counts().items()))
            display(df)
            if not sched and n_run == 0:
                break
            print(f"refreshing every {every:g} s; interrupt to stop watching (the runs continue)")
            time.sleep(every)
    except KeyboardInterrupt:
        print("Stopped watching. The runs continue in the background; run this cell again to resume watching.")
        return
    left = df[~df.status.str.startswith("done")]
    if left.empty:
        print(f"all {len(df)} runs of {exp.env} are finished.")
        return
    print(f"{len(left)} run(s) without a result:")
    for (alg, seed), row in left.iterrows():
        print(f"  {alg} seed {seed}: {row.status}")
    for j in jobs:
        if any(s.startswith("failed") for s in left.status) and not result_path(j, exp.dirs.results).exists() \
                and (exp.dirs.logs / f"{job_id(j)}.log").exists() and _launchable(exp, j):
            print(f"\n--- last lines of {job_id(j)}.log")
            print(tail(exp.dirs.logs / f"{job_id(j)}.log", 12))


def env_stop(exp: EnvExperiment, jobs: list[dict]) -> None:
    """Stop this notebook's scheduler and the jobs it started (plus orphaned jobs of this task left by
    a killed scheduler). Jobs of other schedulers and all finished results are left alone."""
    stop(exp.label, exp.dirs, jobs, only_orphans=True)


# --------------------------------------------------------------------------- results


def _fmt(x: float, d: int | None = None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if d is None:
        d = 0 if abs(x) >= 1000 else 1 if abs(x) >= 10 else 2 if abs(x) >= 1 else 3
    return f"{x:.{d}f}"


def env_summary(exp: EnvExperiment, n_boot: int = N_BOOT) -> dict:
    """Per-seed final scores and AUC of every algorithm on the task, mean and 95% percentile-bootstrap
    CI over seeds, TEMPO vs. each baseline (two-sided Welch t-test, alpha 0.05: win / tie / loss) and
    the verdict line.

    Final score of a run = mean return of the last three evaluations at or before the budget
    (30 episodes); AUC = mean return over all evaluations (``benchmarks/report.py``). Verdict: YES when
    TEMPO's mean final score is higher than the mean final score of every baseline (the criterion of
    the Screen-5 rule in ``benchmarks/PROTOCOL.md``); the Welch tests say which differences are
    significant. Toy text: TEMPO against the exact optimum and the random policy. TEMPO results made
    with other tempo_rl code are left out when ``rerun_outdated`` is set (they are re-run)."""
    import numpy as np
    import pandas as pd

    from benchmarks.report import ALPHA, boot_mean_ci, load_suite, outcome

    info, env, tl = exp.info, exp.env, exp.tempo_label
    data = load_suite(exp.dirs.results, exp.suite).get(env, {})
    code = exp.code_hash()
    tempo_all = [r for r in data.get(tl, []) if r.seed in exp.seeds]
    outdated = [r for r in tempo_all if r.tempo_source != code]
    runs = {tl: [r for r in tempo_all if not (exp.rerun_outdated and r.tempo_source != code)]}
    for b in exp.baselines:
        runs[b] = [r for r in data.get(b, []) if r.seed in exp.seeds]
    labels = [l for l in runs if runs[l]]
    stats = {}
    for l in labels:
        finals, aucs = [r.final for r in runs[l]], [r.auc for r in runs[l]]
        stats[l] = dict(finals=finals, aucs=aucs, seeds=[r.seed for r in runs[l]],
                        final=boot_mean_ci(finals, ("final", env, l), n_boot),
                        auc=boot_mean_ci(aucs, ("auc", env, l), n_boot),
                        hours=float(np.nanmean([r.wall_h for r in runs[l]])),
                        budget=min(r.budget for r in runs[l]))
    # ---- per-seed table ("older code" = a TEMPO result of other tempo_rl code, left out and re-run)
    old_seeds = {r.seed for r in outdated} if exp.rerun_outdated else set()
    rows = []
    for l in [tl] + list(exp.baselines):
        s = stats.get(l)
        row = {"algorithm": l}
        by_seed = dict(zip(s["seeds"], s["finals"])) if s else {}
        row.update({f"seed {k}": _fmt(by_seed.get(k)) if k in by_seed else
                    ("older code" if l == tl and k in old_seeds else "pending") for k in exp.seeds})
        if s:
            m, lo, hi = s["final"]
            am, alo, ahi = s["auc"]
            row.update({"n": len(s["finals"]), "mean final": _fmt(m), "95% CI": f"[{_fmt(lo)}, {_fmt(hi)}]",
                        "AUC": _fmt(am), "AUC 95% CI": f"[{_fmt(alo)}, {_fmt(ahi)}]", "h / run": round(s["hours"], 2)})
        else:
            row.update({"n": 0, "mean final": "-", "95% CI": "-", "AUC": "-", "AUC 95% CI": "-", "h / run": None})
        rows.append(row)
    per_seed = pd.DataFrame(rows).set_index("algorithm")
    # ---- comparisons / reference
    comp, reference, details, raw = None, None, [], []
    t = stats.get(tl)
    if info["toy_text"]:
        opt, rnd = toy_text_optimum(env), info["random"]
        n_ev = 3 if exp.smoke else 10
        opt_eval = toy_text_optimal_eval(env, n_ev)
        reference = pd.DataFrame([dict(reference="known optimum (value iteration)", score=_fmt(opt)),
                                  dict(reference=f"optimal policy on the {n_ev} evaluation episodes",
                                       score=_fmt(opt_eval)),
                                  dict(reference="random policy (benchmarks/random_scores.json)", score=_fmt(rnd))]
                                 + ([dict(reference="gymnasium reward_threshold", score=_fmt(info["threshold"]))]
                                    if info["threshold"] is not None else [])).set_index("reference")
        head = f"TEMPO beats all baselines on {env}: n/a (toy text: no tuned baselines)"
        if t:
            m, lo, hi = t["final"]
            hi_ = hi if not math.isnan(hi) else m
            reach = hi_ >= opt - 1e-9
            share = (m - rnd) / (opt - rnd) if rnd is not None and opt != rnd else math.nan
            verdict = f"{head}; {tl} reaches the known optimum {opt:.4g}: {'YES' if reach else 'NO'}"
            details.append(f"{tl} mean final {m:.4g}, 95% CI [{_fmt(lo)}, {_fmt(hi)}], n={len(t['finals'])} (YES = the "
                           f"CI reaches the optimum); {share:.0%} of the way from the random policy ({rnd:.4g}) to "
                           "the optimum")
            if info["threshold"] is not None:
                details.append(f"solved (mean final >= gymnasium threshold {info['threshold']:g}): "
                               f"{'YES' if m >= info['threshold'] else 'NO'}")
        else:
            verdict = f"{head}; {tl} reaches the known optimum {opt:.4g}: PENDING (no finished TEMPO run yet)"
        if abs(opt_eval - opt) > 1e-9:
            details.append(f"{env} is stochastic and every evaluation replays the same {n_ev} held-out episodes, so a "
                           f"run's final score is measured on {n_ev} fixed episodes: an optimal policy scores "
                           f"{opt_eval:.4g} on them (exact expectation {opt:.4g}). The CI over seeds does not include "
                           "this episode-level noise; read the verdict with it in mind.")
    else:
        crow, raw, base_means = [], [], {}
        for b in exp.baselines:
            s = stats.get(b)
            if not s:
                continue
            base_means[b] = s["final"][0]
            if t:
                res_f, p_f = outcome(t["finals"], s["finals"])
                res_a, p_a = outcome(t["aucs"], s["aucs"])
                raw.append(dict(baseline=b, n=len(s["finals"]), baseline_mean_final=s["final"][0],
                                tempo_mean_final=t["final"][0], difference=t["final"][0] - s["final"][0],
                                welch_p_final=p_f, result_final=res_f, baseline_mean_auc=s["auc"][0],
                                tempo_mean_auc=t["auc"][0], welch_p_auc=p_a, result_auc=res_a))
                crow.append({"baseline": b, "n": len(s["finals"]), "baseline mean final": _fmt(s["final"][0]),
                             f"{tl} mean final": _fmt(t["final"][0]),
                             "difference": _fmt(t["final"][0] - s["final"][0]),
                             "Welch p (final)": _fmt(p_f, 4), "result (final)": res_f.upper(),
                             "Welch p (AUC)": _fmt(p_a, 4), "result (AUC)": res_a.upper()})
        comp = pd.DataFrame(crow).set_index("baseline") if crow else None
        missing = [b for b in exp.baselines if b not in base_means]
        if not t:
            verdict = f"TEMPO beats all baselines on {env}: PENDING (no finished TEMPO run yet)"
        elif not base_means:
            verdict = f"TEMPO beats all baselines on {env}: n/a (no finished baseline run)"
        else:
            best = max(base_means, key=base_means.get)
            tm = t["final"][0]
            if tm > base_means[best]:
                verdict = f"TEMPO beats all baselines on {env}: YES"
            elif tm == base_means[best]:
                verdict = f"TEMPO beats all baselines on {env}: NO (tie with {best}: equal mean final scores)"
            else:
                verdict = f"TEMPO beats all baselines on {env}: NO"
            details.append(f"{tl} mean final {tm:.4g} vs. best baseline {best} {base_means[best]:.4g} "
                           "(YES = higher mean final score than every baseline)")
            if comp is not None:
                # the significance belongs on the verdict line itself: a higher mean alone is not a proven win
                res = list(comp["result (final)"])
                sig = f"significantly better than {res.count('WIN')} of {len(res)} baselines"
                if res.count("LOSS"):
                    sig += f", significantly worse than {res.count('LOSS')}"
                if res.count("N/A") == len(res):
                    sig = "not testable yet (fewer than 2 seeds)"
                verdict += f"  [Welch t-test, alpha {ALPHA}: {sig}]"
                parts = []
                for word, key in (("significantly better than", "WIN"), ("statistically tied with", "TIE"),
                                  ("significantly worse than", "LOSS"), ("not testable (n < 2) against", "N/A")):
                    who = list(comp.index[comp["result (final)"] == key])
                    if who:
                        parts.append(f"{word} {', '.join(who)}")
                details.append(f"Welch t-test on the final scores (alpha {ALPHA}): " + "; ".join(parts))
        if missing:
            details.append(f"baselines without finished runs: {', '.join(missing)} (the verdict may change)")
    short = {l: len(stats[l]["finals"]) for l in stats if len(stats[l]["finals"]) < len(exp.seeds)}
    if t and short:
        verdict += "  (PROVISIONAL: not every seed has finished)"
        details.append("provisional: fewer finished seeds than requested for "
                       + ", ".join(f"{l} (n={n})" for l, n in short.items()))
    if outdated:
        details.append(f"{len(outdated)} TEMPO result(s) of this task come from other tempo_rl code (source "
                       f"{', '.join(sorted({r.tempo_source for r in outdated}))}; current {code}): "
                       + ("left out and re-run by the Run cell" if exp.rerun_outdated else "INCLUDED (RERUN_OUTDATED_TEMPO = False)"))
    if t and any(r.budget != info["budget"] for r in runs[tl]) and not exp.smoke:
        details.append("WARNING: some TEMPO runs used a budget different from the suite's")
    if exp.smoke:
        details.append(f"SMOKE MODE: {SMOKE_STEPS:,}-step runs of one seed; these numbers only show that the "
                       "pipeline works and mean nothing about the algorithms")
    return dict(env=env, variant=exp.variant, tempo_label=tl, seeds=list(exp.seeds), smoke=exp.smoke,
                code_hash=code, stats=stats, runs=runs, per_seed=per_seed, comparison=comp, comparison_raw=raw,
                reference=reference, verdict=verdict, details=details, info=info)


def show_env_summary(S: dict) -> None:
    """Display the tables of :func:`env_summary` and print the verdict."""
    from IPython.display import display

    print(f"Final score per seed (mean of the last three evaluations = {9 if S['smoke'] else 30} episodes), mean and "
          "95% bootstrap CI over seeds, AUC (mean over all evaluations):")
    display(S["per_seed"])
    if S["comparison"] is not None:
        print(f"{S['tempo_label']} vs. each baseline (two-sided Welch t-test on the per-seed scores, alpha 0.05):")
        display(S["comparison"])
    if S["reference"] is not None:
        display(S["reference"])
    print()
    print(S["verdict"])
    for d in S["details"]:
        print("  - " + d)


_RC = {"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
       "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": "#c3c2b7",
       "axes.linewidth": 0.8, "axes.labelcolor": "#52514e", "axes.titlecolor": "#0b0b0b",
       "xtick.color": "#898781", "ytick.color": "#898781", "xtick.labelcolor": "#52514e",
       "ytick.labelcolor": "#52514e", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
       "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
       "lines.solid_capstyle": "round", "lines.dash_capstyle": "butt"}


def plot_env(exp: EnvExperiment, S: dict, n_boot: int = 2000):
    """Learning curves of the task: mean evaluation return over seeds with a 95% bootstrap band per
    algorithm (the report's colours and dash patterns; TEMPO thicker and on top), x = environment
    steps; reference lines for the gymnasium threshold (and, for toy text, the optimum and the random
    policy). Saved as ``<figures>/environments/<task>.png`` and ``.pdf``; returns the figure."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    from benchmarks.report import STYLE, _steps_formatter, curve_band

    info, tl = S["info"], S["tempo_label"]
    labels = [l for l in [tl] + list(exp.baselines) if S["runs"].get(l)]
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.6, 3.9))
        handles = []
        for lab in labels:
            st = STYLE["TEMPO"] if lab == tl else STYLE[lab]
            lw = 2.4 if lab == tl else 1.3
            x, m, lo, hi = curve_band(S["runs"][lab], n_boot, (exp.env, lab))
            ok = np.isfinite(lo)
            if ok.any():
                ax.fill_between(x, lo, hi, where=ok, color=st["color"], alpha=0.18 if lab == tl else 0.10, lw=0,
                                zorder=st["z"] - 0.5)
            ax.plot(x, m, color=st["color"], ls=st["ls"], lw=lw, zorder=st["z"])
            handles.append(Line2D([], [], color=st["color"], ls=st["ls"], lw=lw + 0.3,
                                  label=f"{lab} (n={len(S['runs'][lab])})"))
        refs = []
        if info["toy_text"]:
            refs += [(toy_text_optimum(exp.env), "optimum"), (info["random"], "random policy")]
        if info["threshold"] is not None:
            refs.append((info["threshold"], "gymnasium threshold"))
        budget = max([r.budget for rs in S["runs"].values() for r in rs] or [info["budget"]])
        for y, name in refs:
            if y is None:
                continue
            ax.axhline(y, color="#898781", lw=0.8, ls=(0, (1, 2)), zorder=1)
            ax.text(budget, y, f" {name} {y:.4g}", color="#52514e", fontsize=7.5, va="bottom", ha="right")
        ax.set_xlim(0, budget)
        ax.xaxis.set_major_formatter(_steps_formatter())
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Evaluation return (mean over episodes)")
        seeds = ", ".join(str(s) for s in S["seeds"])
        ax.set_title(f"{exp.env}: mean and 95% bootstrap band over seed{'s' if len(S['seeds']) > 1 else ''} {seeds}"
                     + ("  [SMOKE]" if S["smoke"] else ""), loc="left")
        if handles:
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
                      handlelength=2.6)
        else:
            ax.text(0.5, 0.5, "no finished runs yet", transform=ax.transAxes, ha="center", color="#52514e")
        fig.tight_layout()
        exp.figures.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(exp.figures / f"{exp.env}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {exp.figures / exp.env}.png and .pdf")
    return fig


def _json_clean(o):
    """JSON-safe copy: numpy scalars to Python numbers, NaN / inf to null."""
    if isinstance(o, dict):
        return {str(k): _json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            o = o.item()
        except (TypeError, ValueError):
            return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o if o is None or isinstance(o, (bool, int, float, str)) else str(o)


def save_env(exp: EnvExperiment, S: dict, cfg: dict | None = None) -> list[pathlib.Path]:
    """Write ``<tables>/environments/<task>.md`` (tables + verdict) and ``.json`` (per-seed scores,
    summary statistics, comparisons, verdict, protocol and code version)."""
    from benchmarks.report import md_table

    exp.tables.mkdir(parents=True, exist_ok=True)
    info = S["info"]

    def df_md(df, index_name):
        if df is None:
            return ""
        header = [index_name] + [str(c) for c in df.columns]
        rows = [[str(i)] + ["" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v) for v in r]
                for i, r in zip(df.index, df.itertuples(index=False))]
        return md_table(header, rows)

    lines = [f"# {exp.env}: {S['tempo_label']} vs. "
             + ("the known optimum and a random policy" if info["toy_text"] else "tuned baselines"), "",
             f"Suite `{exp.suite}`, seeds {S['seeds']}, budget {info['budget'] if not exp.smoke else SMOKE_STEPS:,} "
             f"environment steps, evaluation every {info['eval_freq'] if not exp.smoke else SMOKE_STEPS // 3:,} steps "
             f"({'3' if exp.smoke else '10'} deterministic episodes, held-out seeds 100000+i). Final score = mean of the "
             "last three evaluations at or before the budget; AUC = mean over all evaluations; 95% percentile-"
             "bootstrap CIs over seeds; two-sided Welch t-tests, alpha 0.05.",
             f"TEMPO: `{exp.variant}` ({'library defaults' if exp.variant == 'default' else 'benchmarks/variants.py'}), "
             f"tempo_rl source `{S['code_hash']}`. Generated {time.strftime('%Y-%m-%d %H:%M')} by "
             f"notebooks/environments.", "", "## Final score per seed", "", df_md(S["per_seed"], "algorithm")]
    if S["comparison"] is not None:
        lines += ["## TEMPO vs. each baseline", "", df_md(S["comparison"], "baseline")]
    if S["reference"] is not None:
        lines += ["## References", "", df_md(S["reference"], "reference")]
    lines += ["## Verdict", "", f"**{S['verdict']}**", ""] + [f"- {d}" for d in S["details"]] + [""]
    md = exp.tables / f"{exp.env}.md"
    md.write_text("\n".join(lines))
    stats = {l: dict(seeds=s["seeds"], finals=s["finals"], aucs=s["aucs"], mean_final=s["final"][0],
                     ci_final=list(s["final"][1:]), mean_auc=s["auc"][0], ci_auc=list(s["auc"][1:]),
                     hours_per_run=s["hours"], budget=s["budget"]) for l, s in S["stats"].items()}
    comp = S.get("comparison_raw") or None
    blob = dict(env=exp.env, suite=exp.suite, variant=exp.variant, tempo_label=S["tempo_label"], seeds=S["seeds"],
                smoke=exp.smoke, budget=info["budget"] if not exp.smoke else SMOKE_STEPS,
                eval_freq=info["eval_freq"] if not exp.smoke else SMOKE_STEPS // 3, tempo_source=S["code_hash"],
                baselines=list(exp.baselines), stats=stats, comparison=comp, verdict=S["verdict"],
                details=S["details"], generated=time.strftime("%Y-%m-%d %H:%M:%S"),
                reference=dict(random=info["random"], threshold=info["threshold"],
                               optimum=toy_text_optimum(exp.env) if info["toy_text"] else None,
                               optimal_policy_on_eval_episodes=(toy_text_optimal_eval(exp.env, 3 if exp.smoke else 10)
                                                                if info["toy_text"] else None)),
                tempo_config=None if cfg is None else dict(
                    kwargs=cfg.get("kwargs"), width=cfg.get("width"), code_root=cfg.get("code_root"),
                    source_hash=cfg.get("source_hash"), device=cfg.get("device"), gamma=cfg.get("gamma"),
                    recipe=dict(cfg.get("recipe") or []), matches_reference=cfg.get("match")))
    js = exp.tables / f"{exp.env}.json"
    js.write_text(json.dumps(_json_clean(blob), indent=1))
    print(f"wrote {md}\nwrote {js}")
    return [md, js]
