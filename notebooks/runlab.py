"""Helpers of the ``Baselines_test/`` and ``TEMPO_test/`` notebooks (one notebook per task, 24 each: the earlier
benchmark's 20 plus CarRacing-v3, HumanoidStandup-v5, Blackjack-v1 and Taxi-v4, added on 2026-11-03).

Both folders hold the same kind of notebook: it RUNS one task with the protocol of the two
folders (:data:`TEST_BUDGET` = 500,000 environment steps for EVERY task, an evaluation every
:data:`TEST_EVAL_FREQ` = 10,000 steps = 50 evaluations of :data:`N_EVAL` = 10 episodes, seeds
10-14; ``run_one.py`` evaluation, ``scheduler.py`` parallel jobs) and STORES everything inside
its own folder:

    Baselines_test/   the tuned Stable-Baselines3 baselines of the task (SAC, TD3, CrossQ, PPO, DQN;
                      RL-Baselines3-Zoo hyperparameters from ``benchmarks/zoo.py``), no TEMPO,
                      no comparison.  kind="baselines", suite ``baselines_test``
    TEMPO_test/       TEMPO (library defaults = the fast recipe since 2026-10-23, or any variant of
                      ``benchmarks/variants.py``, e.g. "c1", the previous default).  kind="tempo",
                      suite ``tempo_test``

    <folder>/results/<suite>/<env>/<algo>/<variant>/seed<k>.json   exactly the ``run_one.py`` format
    <folder>/logs/      scheduler output: <job-id>.log per run, <label>.status / .out / .pid, locks/
    <folder>/figures/<task>.png|pdf      learning curves
    <folder>/tables/<task>.md|json       per-seed scores, statistics, references, protocol, versions

Suites are named ``baselines_test`` / ``tempo_test`` on purpose, never ``final``: a frozen
``benchmarks/snapshots/final`` (made by the main notebooks) can then never change which
``tempo_rl`` code TEMPO_test runs. TEMPO_test always runs the LIVE ``tempo_rl`` and every result
records its source hash (``tempo_source``); a result made with other code is reported as "older
code" and, with ``rerun_outdated=True``, MOVED (never deleted) to ``<results>/_outdated/<time>/``
and re-run. Nothing is ever frozen.

Budget. Evaluation (10 deterministic episodes on the held-out seeds 100000 + i), seeds (10-14),
scores and job fields are those of the main benchmark; the BUDGET is not. The earlier benchmark
(``benchmarks/plans.py``, results in ``benchmarks/results/final``; :func:`earlier_protocol`) ran
each task for 20,000-300,000 steps with 20-30 evaluations. These folders run every task for
:data:`TEST_BUDGET` steps with an evaluation every :data:`TEST_EVAL_FREQ` steps: the single source
of truth is this module (``make_lab(budget=TEST_BUDGET, eval_freq=TEST_EVAL_FREQ)``, the BUDGET /
EVAL_FREQ lines of every Parameters cell); every job, estimate, plan, summary, figure, table and
protocol check uses the lab's values, never the budgets of ``benchmarks/plans.py`` (which stays
unchanged as the record of the earlier benchmark). Both folders must use the same values, or the
comparison is not like for like. Results of the earlier budgets are therefore never reused by
default: :func:`import_previous` only finds them when BUDGET / EVAL_FREQ are set back to the
earlier values, and a result file of another budget found in a results folder is of ``other
protocol`` (below): left out, moved to ``_outdated/`` and run again at the lab's budget.

Every step of a per-task notebook is one short call; the calls, in notebook order::

    import runlab as rl
    rl.check_kernel()                                    # Python, PyTorch, Apple GPU, packages
    SMOKE = rl.smoke_flag(SMOKE)                         # env TEMPO_NOTEBOOK_SMOKE wins over the cell
    LAB = rl.make_lab("baselines", ENV_ID, SEEDS, SMOKE, algos=ALGOS, budget=BUDGET,
                      eval_freq=EVAL_FREQ, ...)          # settings + folders (BUDGET = TEST_BUDGET)
    rl.show_task(LAB)                                    # spaces, time limit, budget, references
    rl.show_zoo(LAB)                                     # exact hyperparameters of every baseline
    CFG = rl.tempo_config(LAB); rl.show_tempo_config(LAB, CFG)    # (tempo) the configuration that runs
    JOBS = rl.make_jobs(LAB, CFG)                        # job list, devices assigned
    rl.plan(LAB, JOBS)                                   # one row per run + wall-clock estimate
    rl.import_previous(LAB)                              # (baselines, opt-in) reuse the earlier SB3 runs
    rl.launch(LAB, JOBS); rl.watch(LAB, JOBS)            # background scheduler, progress table
    S = rl.summary(LAB, JOBS); rl.show_summary(S)        # scores, CIs, AUC, references
    rl.plot_curves(LAB, S); rl.save(LAB, S, CFG)         # figure (PNG + PDF) and tables (.md + .json)

Notebook convention: ``plan``, ``plot_curves``, ``save``, ``import_previous``, ``launch`` and the other functions that
print or display their own results also return something (a table, the saved paths, the scheduler info); end such a
cell with ``;`` so the return value is not shown a second time (``watch`` and ``show_*`` return nothing).

Smoke mode (``smoke=True`` or env ``TEMPO_NOTEBOOK_SMOKE=1``, which wins; accepted values 1/true/yes/on and
0/false/no/off, unset or empty = the cell, anything else is an error): first seed only, 1,500-step
runs with 3 evaluations of 3 episodes (``benchlab.shrink``), at most 2 CPU + 1 Apple-GPU worker,
everything (results, logs, figures, tables) under ``benchlab.scratch_root()/<folder>_<task>/``
(``$TEMPO_NOTEBOOK_SCRATCH``), suite ``smoke_<suite>``, scheduler label ``smoke_<label>``. Real mode
writes inside the folder. Importing this module creates no file; folders are created by ``make_lab``.

Device policy (override with ``devices=``): TEMPO -> ``benchmarks.plans.default_tempo_device()`` (the
Apple GPU for the wide-critic default when available, else the CPU; a named variant: GPU when its
critic is >= 512 wide) with ALL seeds of a task on the same device; CrossQ -> the Apple GPU when
available, else the CPU; SAC, TD3, PPO, DQN -> the CPU. Without Apple-GPU workers GPU jobs fall back
to the CPU (much slower for wide critics).

Worker limits (``cpu_workers`` / ``mps_workers``, ``None`` = automatic as in ``benchlab.env_bench``:
6 GPU workers when the GPU works, the remaining cores minus two for the CPU) are enforced by the
scheduler PER LOGS FOLDER: it counts the jobs it sees through the lock files in ``<logs>/locks``.
All notebooks of one folder share their limits; Baselines_test, TEMPO_test and the main notebooks
(``benchmarks/logs``) do NOT see each other, so running two of them at once can use up to the sum
of their limits (:func:`launch` warns when jobs of another folder are running).

Robustness (what the pre-existing ``benchlab`` / ``benchmarks.scheduler`` do not do, done here without changing
them): a scheduler is recognised by its label AND this lab's jobs file (another lab with the same label in another
logs folder is not "already running" and is not stopped by :func:`stop`); a lock file whose process id was recycled
by an unrelated process counts as stale (:func:`launch` and :func:`stop` remove it); the liveness test knows that
a killed scheduler which is a child of the kernel is a zombie (``stop`` returns at once instead of after 30 s);
two launches of one folder at the same instant are serialised; a TEMPO result whose run overlapped an edit of
``tempo_rl`` is reported as "older code" (``tempo_source`` is computed when a run ENDS, not when it starts).

Existing results are checked, not just found (benchlab and the scheduler count any file at a run's path as finished): a
result file at a planned run's path is ``unreadable`` (not valid JSON, no job or curve, fewer evaluations at or before the
budget than the schedule, or evaluations with another number of episodes; evaluations AFTER the budget are normal for PPO)
or of ``other protocol`` (budget, evaluation schedule, episodes, seed, algorithm, task, variant or env_kwargs of its job
differ from the planned run, e.g. a run of the earlier 50,000-step CartPole budget, or, for the baselines, the
hyperparameters it recorded differ from what ``benchmarks/run_one.py`` builds now). Such files are never used: the plan
shows the state, :func:`summary` leaves them out with a WARNING naming the file and what differs (the figure and tables are
then provisional; result files on disk that are not among the planned runs, e.g. of an algorithm not in ALGOS, are checked
against the lab's protocol as well) and :func:`launch` moves them (never deletes) to
``<results>/_outdated/<time>/<suite>/<env>/<algo>/<variant>/seed<k>.json``, with the logs of their runs
(``_outdated/<time>/logs/``, so a re-run waiting in the queue is not shown as "interrupted" with the old run's progress),
and runs them again, whatever ``rerun_outdated`` says; while the scheduler of the task is still running it only lists them
(the next launch after it has finished moves them). A result file that is not at the path of the run its record
describes (a Finder copy ``seed10 copy.json``, a seed-10 record renamed ``seed99.json``) is never used either (WARNING in the
summary; not a planned run's file, so it is never moved: the user removes it). A figure / tables of the task whose table
records another budget are moved to ``<figures|tables>/_outdated/<time>/`` by the launch, or before the next figure / table
is written, instead of staying next to (and later being overwritten by) the new runs' files. A result whose device,
threads or library versions differ from the planned run / the installed packages is kept and used, with a NOTE in the plan,
the summary and the saved tables. ``TEMPO_NOTEBOOK_SMOKE`` is parsed strictly (an unknown value raises before anything is
created, an override of the cell value is announced once). :func:`import_previous` updates its manifest under a lock (two
notebooks importing at once keep every entry), writes the entry of a copy before the copy (an import killed half-way leaves
no copy without its entry) and also rejects records of another device, thread count or python / torch / gymnasium / numpy
version.
"""
from __future__ import annotations

import contextlib
import json
import math
import numbers
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:                      # benchlab.py lives next to this file
    sys.path.insert(0, str(_HERE))

import benchlab as bl                               # noqa: E402  (puts the project root on sys.path)
from benchmarks import plans, report                # noqa: E402

ROOT = bl.ROOT
BENCH = bl.BENCH

__all__ = ["Lab", "make_lab", "tasks", "smoke_flag", "check_kernel", "show_task", "show_zoo", "zoo_info", "tempo_config",
           "show_tempo_config", "device_for", "baseline_jobs", "tempo_jobs", "make_jobs", "status_table", "estimate", "plan",
           "launch", "watch", "stop", "show_log", "archive_outdated", "reference_scores", "references", "summary",
           "show_summary", "plot_curves", "save", "import_previous", "earlier_protocol", "tempo_milestones", "KINDS",
           "FOLDERS", "SUITES", "FINAL_SEEDS", "TEST_SEEDS", "DEVICE_POLICY", "TEST_BUDGET", "TEST_EVAL_FREQ", "N_EVAL"]

#: The two kinds of folder: results are stored in ``<ROOT>/<FOLDERS[kind]>``.
KINDS = ("baselines", "tempo")
FOLDERS = {"baselines": "Baselines_test", "tempo": "TEMPO_test"}
#: Suite names (``smoke_`` + name in smoke mode). Deliberately not ``final``.
SUITES = {"baselines": "baselines_test", "tempo": "tempo_test"}
#: Scheduler labels are ``<prefix>_<task>`` (``smoke_`` + label in smoke mode); the main notebooks use ``env_<task>``.
LABEL_PREFIX = {"baselines": "baselines", "tempo": "tempo"}
SMOKE_ENV = "TEMPO_NOTEBOOK_SMOKE"
FINAL_SEEDS = tuple(plans.FINAL_SEEDS)
#: Training seeds of Baselines_test / TEMPO_test: the five final seeds 10-14 of the earlier benchmark (since
#: 2026-11-05, for the final run; from 2026-10-23 to 2026-11-04 the first three, 10-12, whose runs stay valid). ``make_lab``
#: defaults to these and complab expects them; ``FINAL_SEEDS`` stays the earlier benchmark's record.
TEST_SEEDS = (10, 11, 12, 13, 14)
#: The protocol of Baselines_test / TEMPO_test (fixed on 2026-10-21): 500,000 environment steps per run for
#: EVERY task, in both folders, with an evaluation every 10,000 steps (50 evaluations) of :data:`N_EVAL` deterministic
#: episodes on the held-out seeds 100000 + i. The single source of truth: ``make_lab`` defaults to these, the notebooks'
#: BUDGET / EVAL_FREQ lines repeat them, ``Comparison/`` reads them. ``benchmarks/plans.py`` keeps the budgets of the earlier
#: benchmark (:func:`earlier_protocol`). Changing them in one folder only breaks the comparison.
TEST_BUDGET = 500_000
TEST_EVAL_FREQ = 10_000
#: Deterministic evaluation episodes per evaluation (``run_one``; smoke mode: 3).
N_EVAL = 10
#: Tasks evaluated on more episodes than :data:`N_EVAL` (both folders; :func:`n_eval_for`), decided 2026-11-03
#: (benchmarks/PROTOCOL.md): Blackjack's returns are -1 / 0 / +1 per hand and every evaluation replays the same held-out
#: deals, so 10 hands cannot tell policies apart (the optimal policy scores +0.30 on the first 10, random play -0.70; the
#: exact optimum is -0.043). 1,000 hands (about 1,400 steps per evaluation) put the standard error near 0.03.
N_EVAL_BY_TASK = {"Blackjack-v1": 1000}
SMOKE_STEPS = bl.SMOKE_STEPS
N_BOOT = bl.N_BOOT
#: Baselines that can act in continuous / discrete action spaces.
CONTINUOUS_ALGOS = ("SAC", "TD3", "CrossQ", "PPO")
DISCRETE_ALGOS = ("PPO", "DQN")
#: Baselines of the toy-text tasks (FrozenLake, CliffWalking, Blackjack, Taxi): PPO and DQN. RL-Baselines3-Zoo has no entry
#: for them; benchmarks/zoo.py gives them the grid-search winners of benchmarks/tuned_toy_text.json (flagged tuned).
#: ``benchmarks.report.expected_pool`` (the plan of the main benchmark, which ran TEMPO only on toy text) is not changed.
TOY_TEXT_ALGOS = DISCRETE_ALGOS

DEVICE_POLICY = (
    "TEMPO -> benchmarks.plans.default_tempo_device() (the GPU for the wide-critic default if available, else "
    "CPU; a named variant: GPU if its critic is >= 512 wide), all seeds of a task on the same device; "
    "CrossQ -> the GPU if available, else CPU; SAC / TD3 / PPO / DQN -> CPU, except SAC / TD3 on the pixel task "
    "CarRacing (convolutional networks: the GPU if available). The GPU is NVIDIA CUDA when available, else the Apple GPU "
    "(benchmarks.plans.gpu_device). Override with devices=; without GPU workers GPU jobs fall back to the CPU.")
#: Automatic Apple-GPU worker limit of a task's notebook (``make_lab(mps_workers=None)``), where the default
#: (:data:`benchlab.AUTO_MPS_WORKERS` = 6) would not fit in memory. CarRacing at full replay (100,000 pixel transitions,
#: ``benchmarks.plans.REPLAY_CAPACITY``): SB3 SAC / TD3 ~6.5 GB per run (observations stored twice), TEMPO ~4 GB per run.
TASK_MPS_WORKERS = {"CarRacing-v3": {"baselines": 2, "tempo": 3}}
#: Planning rates (env-steps/s of one run while its folder runs a full schedule) of the slow pixel task, where the
#: per-algorithm rates of benchlab / this module are far too optimistic. Measured 2026-11-03 for one run alone (Apple M5
#: Pro): TEMPO default 17.6 (MPS; classic engine, image observations), SAC 13.9 and TD3 21.9 (MPS, CnnPolicy), PPO 52.9
#: (CPU; RL-Zoo settings, 8 environments); the simulator itself renders ~325 frames/s = ~81 agent steps/s (frame-skip 4),
#: and every evaluation plays 10 episodes of 250 agent steps. Planned with the GPU shared by 2-3 runs and evaluations
#: included: an ESTIMATE, not timed under a full schedule.
TASK_RATES = {"CarRacing-v3": {"TEMPO": {"mps": 9.0, "cpu": 2.0}, "SAC": {"mps": 8.0, "cpu": 1.4},
                               "TD3": {"mps": 12.0, "cpu": 2.0}, "PPO": {"cpu": 40.0, "mps": 45.0}}}

#: Result files at a planned run's path that are never used: :func:`launch` moves them to ``_outdated/`` and runs them again.
_STATES_BAD = ("unreadable", "other protocol")
_STATES_TODO = ("to run", "older code", "failed", "interrupted", *_STATES_BAD)

#: Values of ``TEMPO_NOTEBOOK_SMOKE`` (stripped, case-insensitive); anything else is an error, not silently "real runs".
_SMOKE_TRUE = ("1", "true", "yes", "on")
_SMOKE_FALSE = ("0", "false", "no", "off")
#: (value, cell value) of the overrides :func:`smoke_flag` has announced in this process (make_lab then does not repeat them).
_SMOKE_ANNOUNCED: set = set()


# =========================================================================== TEMPO planning rates
#: Planning rates of TEMPO runs (env-steps/s of ONE run while the machine runs a full schedule: the convention of
#: ``benchlab.TEMPO_RATES``, i.e. all :data:`benchlab.AUTO_MPS_WORKERS` Apple-GPU slots busy). benchlab's model
#: (calibrated on Screen 6: stock C1 ran at 16-18 env-steps/s per run with eight GPU runs at once) describes the C1
#: engine of the library BEFORE 2026-10-22. This module applies the measured speed-ups on top of it:
#:
#: * ``TEMPO_VARIANT_RATES["default"]``: the library default since 2026-10-23, the fast recipe (C1's update maths on
#:   the phase engine with 4 environments, batch 1024, bfloat16 critics, closed-form Batch Renorm and fused critics;
#:   the former variant ``fast4x``; ``"fast"`` is the same recipe spelled out). One run alone, measured on Hopper
#:   (Apple M5 Pro, MPS): 506.7 env-steps/s (2026-10-23; 516.5 / 528.7 when re-timed after the default changed), two
#:   runs at once 321.8 per run. With the six GPU slots busy it is planned at 45 x 2.2 = 99 env-steps/s per run: the
#:   stage-1 prototype's measured aggregate GPU throughput of the 4-environment engine shared by six slots (45) times
#:   the measured update-time speed-up of the closed-form kernel + fused critics (2.2). That is an ESTIMATE, not
#:   timed with six runs at once. On the CPU (no Apple GPU) the default is planned like benchlab's C1 model (not
#:   measured under a schedule; one 1,500-step run alone on 4 CPU threads: ~54 env-steps/s after learning starts).
#: * ``TEMPO_ENGINE_SPEEDUP``: every configuration WITHOUT ``gpu_replay`` (``"c1"``, the previous default, and every
#:   other pinned screening variant) runs the C1 engine, which since 2026-10-22 waits for the device only where a number
#:   needs it (bit-for-bit the same training as before; :data:`TEMPO_SPEED_FACTS`). Measured on the Apple GPU: 1.61x
#:   for one run alone, 1.50x per run with two runs at once; with all GPU slots busy the GPU itself becomes the limit
#:   and the engine does almost the same GPU work as before, so the planning factor 1.2 is an ESTIMATE between the
#:   two-run measurement and saturation. CPU: 1.01-1.04x measured (4 threads), planned as 1.0.
#: * any other ``gpu_replay`` variant: benchlab's model x ``GPU_REPLAY_SPEEDUP`` (a guess from the 2026-10-22
#:   prototype; no such variant is defined).
TEMPO_ENGINE_SPEEDUP = {"mps": 1.2, "cpu": 1.0}
TEMPO_VARIANT_RATES = {"default": {"mps": 45.0 * 2.2}, "fast": {"mps": 45.0 * 2.2}}
#: planning factors on the C1 engine's rate for variants WITHOUT ``gpu_replay`` (none since 2026-10-23)
TEMPO_VARIANT_FACTORS: dict = {}
#: Where each entry of :data:`TEMPO_VARIANT_RATES` comes from (printed by the configuration cell's NOTE).
TEMPO_VARIANT_RATE_SOURCES = {
    v: ("an ESTIMATE, not timed with six runs at once: 45 x 2.2 = 99, i.e. the stage-1 prototype's measured aggregate "
        "GPU throughput of the 4-environment engine shared by the 6 GPU slots (45) times the measured update-time "
        "speed-up of the closed-form Batch-Renorm kernel + fused critics (2.2; 2026-10-23, Hopper: 16.74 -> 7.53 ms per "
        "update). Measured: one run alone ~507 env-steps/s, two runs at once ~322 per run")
    for v in ("default", "fast")}
GPU_REPLAY_SPEEDUP = {"mps": 31.0 / 16.4, "cpu": 1.0}
#: The measurements behind the factors above (Apple M5 Pro, Hopper-v5, budget 500k so the novelty bonus is active;
#: 2026-10-22: seeds 20/21, 15-25 s windows after learning starts; 2026-10-23: learn(500,000) stopped at 5,000 steps):
#: speed of one run relative to the library before 2026-10-22 (stock C1: 59.4 env-steps/s alone, 46.0 per run with two
#: runs at once). ``c1_*``: the C1 engine of the current library (the previous default); ``default_*``: the fast
#: recipe (the default since 2026-10-23).
TEMPO_SPEED_FACTS = dict(c1_single=95.4 / 59.4, c1_pair=69.0 / 46.0, c1_single_rate=95.4, stock_single=59.4,
                         default_single=506.7 / 59.4, default_single_rate=506.7, default_pair_rate=321.8,
                         default_vs_c1_single=506.7 / 94.8)
_BL_JOB_RATE = getattr(bl.job_rate, "_benchlab_original", bl.job_rate)


def _variant_kwargs(variant: str) -> dict:
    try:
        from benchmarks.variants import tempo_kwargs

        return tempo_kwargs(variant, "Hopper-v5")
    except Exception:                                   # noqa: BLE001 - unknown variant: benchlab's model
        return {}


def tempo_job_rate(j: dict, device: str | None = None) -> float:
    """Planning env-steps/s of one TEMPO job (see :data:`TEMPO_ENGINE_SPEEDUP` / :data:`TEMPO_VARIANT_RATES`)."""
    dev = bl.pool(device or j.get("device", "cpu"))
    variant = j.get("variant", "default")
    if dev in TEMPO_VARIANT_RATES.get(variant, {}):
        return float(TEMPO_VARIANT_RATES[variant][dev])
    base = _BL_JOB_RATE(j, device)
    if _variant_kwargs(variant).get("gpu_replay"):
        return base * GPU_REPLAY_SPEEDUP[dev]
    return base * TEMPO_ENGINE_SPEEDUP[dev] * TEMPO_VARIANT_FACTORS.get(variant, {}).get(dev, 1.0)


def _job_rate(j: dict, device: str | None = None) -> float:
    """``benchlab.job_rate`` with the TEMPO planning rates of this module (other algorithms: benchlab's) and the per-task
    rates of :data:`TASK_RATES`."""
    task = TASK_RATES.get(j.get("env"), {}).get(j.get("algo"))
    if task:
        dev = device or j.get("device", "cpu")
        return float(task.get(bl.pool(dev)) or task.get("cpu") or min(task.values()))
    if j.get("algo") == "TEMPO":
        return tempo_job_rate(j, device)
    return _BL_JOB_RATE(j, device)


_job_rate._benchlab_original = _BL_JOB_RATE
# benchlab.job_seconds / estimate and make_env_notebooks.run_hours look job_rate up in benchlab at call time: every
# estimate of the notebooks that import this module uses the rates above.
bl.job_rate = _job_rate


def tasks() -> list[str]:
    """The 24 tasks in notebook order: the earlier benchmark's 20 (01 CartPole-v1 ... 20 CliffWalking-v1), then the 4
    added on 2026-11-03 (``benchmarks.plans.ADDED``: 21 CarRacing-v3, 22 HumanoidStandup-v5, 23 Blackjack-v1,
    24 Taxi-v4)."""
    return [*plans.SUITE, *plans.TOY_TEXT, *plans.ADDED]


def earlier_protocol(env_id: str) -> tuple[int, int] | None:
    """``(budget, eval_freq)`` of the EARLIER benchmark on ``env_id`` (``benchmarks/plans.py``: 20,000-300,000 steps, 20-30
    evaluations; the protocol of ``benchmarks/results/final``). Not the protocol of these folders (:data:`TEST_BUDGET`).
    ``None`` for the tasks added on 2026-11-03 (``benchmarks.plans.ADDED``): the earlier benchmark never ran them."""
    table = {**plans.SUITE, **plans.TOY_TEXT}
    if env_id in plans.ADDED:
        return None
    if env_id not in table:
        raise ValueError(f"{env_id!r} is not a benchmark task; known: {[*table, *plans.ADDED]}")
    steps, freq, _ = table[env_id]
    return int(steps), int(freq)


def _optimum_method(env_id: str) -> str:
    """How the exact toy-text optimum of ``env_id`` is computed (``benchlab.toy_text_optimum``)."""
    return ("dynamic programming over the infinite deck" if env_id.startswith("Blackjack")
            else "value iteration within the time limit")


def n_eval_for(env_id: str) -> int:
    """Deterministic episodes per evaluation on ``env_id`` in real mode: :data:`N_EVAL` = 10, except the tasks of
    :data:`N_EVAL_BY_TASK` (smoke mode: 3 everywhere, ``benchlab.shrink``)."""
    return int(N_EVAL_BY_TASK.get(env_id, N_EVAL))


def _check_protocol(budget, eval_freq) -> tuple[int, int]:
    """Validated ``(budget, eval_freq)``: positive whole numbers of environment steps, ``eval_freq`` divides ``budget``
    (the last evaluation falls on the budget) and there are at least 3 evaluations (the final score is the mean of the
    last three). Raises ValueError naming the Parameters-cell names."""
    for name, v in (("BUDGET (budget=)", budget), ("EVAL_FREQ (eval_freq=)", eval_freq)):
        if isinstance(v, bool) or not isinstance(v, numbers.Integral) or int(v) <= 0:
            raise ValueError(f"{name} must be a positive whole number of environment steps, not {v!r}")
    budget, eval_freq = int(budget), int(eval_freq)
    if budget % eval_freq:
        raise ValueError(f"EVAL_FREQ = {eval_freq:,} does not divide BUDGET = {budget:,}: the evaluations must end exactly at "
                         f"the budget (e.g. EVAL_FREQ = {TEST_EVAL_FREQ:,} with BUDGET = {TEST_BUDGET:,})")
    if budget // eval_freq < 3:
        raise ValueError(f"BUDGET / EVAL_FREQ = {budget // eval_freq} evaluation(s): at least 3 are needed (the final score is "
                         "the mean of the last three evaluations)")
    return budget, eval_freq


def _smoke_from_env(default) -> tuple[bool, str | None]:
    """(smoke, normalised value of ``TEMPO_NOTEBOOK_SMOKE`` or None when it is unset / empty). Strict: a value that is not
    in :data:`_SMOKE_TRUE` / :data:`_SMOKE_FALSE` raises instead of meaning "real runs" (``benchlab.env_flag`` reads
    '2' or 'smoke' as False, which turned a smoke check into real training)."""
    raw = os.environ.get(SMOKE_ENV)
    if raw is None or not raw.strip():
        return bool(default), None
    v = raw.strip().lower()
    if v in _SMOKE_TRUE:
        return True, v
    if v in _SMOKE_FALSE:
        return False, v
    raise ValueError(f"environment variable {SMOKE_ENV}={raw!r} is not understood: accepted values are "
                     f"{'/'.join(_SMOKE_TRUE)} (smoke mode) and {'/'.join(_SMOKE_FALSE)} (real runs), case-insensitive; unset "
                     "or empty = the SMOKE value of the Parameters cell. Fix or unset the variable (nothing was created).")


def _smoke_note(value: str, default, smoke: bool) -> str:
    return (f"the environment variable {SMOKE_ENV}={value!r} overrides SMOKE = {bool(default)} of the Parameters cell: "
            + ("SMOKE mode (pipeline check in the scratch folder)" if smoke else "REAL runs"))


def smoke_flag(default: bool = False) -> bool:
    """``default`` unless the environment variable ``TEMPO_NOTEBOOK_SMOKE`` is set, which wins: 1/true/yes/on = True,
    0/false/no/off = False (stripped, case-insensitive), unset or empty = ``default``, any other value raises ValueError
    (before any folder is created). When the variable changes the cell's value, one NOTE line says so. For the Parameters
    cell: ``SMOKE = rl.smoke_flag(SMOKE)``."""
    smoke, value = _smoke_from_env(default)
    if value is not None and smoke != bool(default):
        print("NOTE: " + _smoke_note(value, default, smoke))
        _SMOKE_ANNOUNCED.add((value, bool(default)))
    return smoke


def check_kernel() -> None:
    """Print the Python / PyTorch / Apple-GPU / package setup of this kernel (the runs use the same interpreter) and
    warn about anything that would make them fail.

    Unlike ``benchlab.check_kernel`` it does not import ``stable_baselines3`` (which loads OpenCV), ``sb3_contrib``,
    ``mujoco`` or ``Box2D`` into the kernel: package versions come from the installed metadata, and a broken binary
    install is detected in short throw-away subprocesses (OpenCV and Box2D / pygame are never loaded into one
    process, so the notebook kernel stays free of the duplicate-SDL warnings those cause)."""
    import importlib.metadata as md
    import importlib.util
    import platform
    import subprocess

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
    for mod, dists in (("gymnasium", ("gymnasium",)), ("mujoco", ("mujoco",)), ("Box2D", ("box2d-py", "Box2D", "box2d")),
                       ("pygame", ("pygame", "pygame-ce")),
                       ("stable_baselines3", ("stable-baselines3",)), ("sb3_contrib", ("sb3-contrib",)),
                       ("scipy", ("scipy",)), ("pandas", ("pandas",)), ("matplotlib", ("matplotlib",))):
        try:
            present = importlib.util.find_spec(mod) is not None
        except Exception:                               # noqa: BLE001
            present = False
        if not present:
            missing.append(mod)
            continue
        ver = next((v for v in (_dist_version(md, d) for d in dists) if v), "?")
        vers.append(f"{mod} {ver}")
    print("packages   : " + ", ".join(vers))
    if missing:
        print(f"MISSING    : {', '.join(missing)} (tasks that need them will fail)")
    # the binary packages really import (each group in its own process: OpenCV never meets Box2D / pygame)
    groups = [g for g in (("Box2D", "mujoco", "pygame"), ("stable_baselines3", "sb3_contrib")) if not set(g) & set(missing)]
    procs = [(g, subprocess.Popen([sys.executable, "-c", "import " + ", ".join(g)], stdout=subprocess.DEVNULL,
                                  stderr=subprocess.PIPE, text=True)) for g in groups]
    for g, p in procs:
        try:
            err = p.communicate(timeout=120)[1]
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            err = "timed out"
        if p.returncode != 0 or err == "timed out":
            last = (err or "").strip().splitlines()[-1:] or ["?"]
            print(f"BROKEN     : import {', '.join(g)} fails in a fresh process ({last[0]}); tasks that need them will fail")
    import tempo_rl

    src = pathlib.Path(tempo_rl.__file__).resolve().parent
    print(f"tempo_rl   : {tempo_rl.__version__} imported from {src}")
    if not src.is_relative_to(ROOT):
        print("             WARNING: not this project's tempo_rl (an installed copy shadows it?)")


def _dist_version(md, dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


# =========================================================================== settings


@dataclass
class Lab:
    """Settings of one per-task notebook (built by :func:`make_lab`); every function takes it first.

    ``kind`` "baselines" | "tempo"; ``env`` task id; ``seeds``; ``smoke``; ``variant`` ("default" for
    baselines); ``algos`` the algorithms that run (TEMPO: ``("TEMPO",)``); ``suite``; ``folder`` the
    notebook folder; ``dirs`` results / logs / figures / tables (benchlab.Dirs); ``label`` scheduler
    label; ``cpu_workers`` / ``mps_workers`` limits per logs folder; ``devices`` explicit device
    overrides {algo: "cpu" | "mps"}; ``rerun_outdated`` (TEMPO) move results of other tempo_rl code
    away and re-run; ``info`` = ``benchlab.task_info`` with ``budget`` / ``eval_freq`` / ``n_evals`` replaced by
    this lab's protocol (``make_lab(budget=, eval_freq=)``, default :data:`TEST_BUDGET` / :data:`TEST_EVAL_FREQ`),
    which is what the tempo_config subprocess and every helper read.

    Protocol attributes: ``real_budget`` / ``real_eval_freq`` = BUDGET / EVAL_FREQ of the lab (also in smoke mode);
    ``budget`` / ``eval_freq`` / ``n_eval`` = what the runs of this lab use (smoke: 1,500 steps, 3 evaluations of 3
    episodes); ``earlier_budget`` / ``earlier_eval_freq`` = the earlier benchmark's (``benchmarks/plans.py``).
    """
    kind: str
    env: str
    seeds: tuple
    smoke: bool
    variant: str
    algos: tuple
    suite: str
    folder: pathlib.Path
    dirs: bl.Dirs
    label: str
    cpu_workers: int
    mps_workers: int
    devices: dict
    rerun_outdated: bool
    info: dict
    notes: list = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"Lab({self.kind}, {self.env}, seeds={list(self.seeds)}, suite={self.suite!r}, "
                f"{'SMOKE' if self.smoke else 'REAL'}, results={str(self.dirs.results)!r})")

    # -- names
    @property
    def tempo_label(self) -> str:
        """Column name of this TEMPO in tables and figures: ``TEMPO`` or ``TEMPO[<variant>]``."""
        return report.label_of("TEMPO", self.variant)

    @property
    def labels(self) -> list[str]:
        """Labels of the algorithms of this notebook, in table order."""
        return [self.tempo_label] if self.kind == "tempo" else [report.label_of(a, "default") for a in self.algos]

    @property
    def folder_name(self) -> str:
        return FOLDERS[self.kind]

    # -- protocol of the runs of this lab (smoke: shrunk)
    @property
    def budget(self) -> int:
        return SMOKE_STEPS if self.smoke else self.info["budget"]

    @property
    def eval_freq(self) -> int:
        return SMOKE_STEPS // 3 if self.smoke else self.info["eval_freq"]

    @property
    def n_eval(self) -> int:
        return 3 if self.smoke else n_eval_for(self.env)

    @property
    def real_budget(self) -> int:
        """BUDGET of this lab (``make_lab(budget=)``), also in smoke mode, where the runs are shrunk to 1,500 steps."""
        return int(self.info["budget"])

    @property
    def real_eval_freq(self) -> int:
        """EVAL_FREQ of this lab (``make_lab(eval_freq=)``), also in smoke mode."""
        return int(self.info["eval_freq"])

    @property
    def earlier_budget(self) -> int | None:
        """Budget of the EARLIER benchmark on this task (``benchmarks/plans.py``, ``benchmarks/results/final``); None for
        the tasks added on 2026-11-03 (``benchmarks.plans.ADDED``), which were never part of it."""
        ep = earlier_protocol(self.env)
        return ep[0] if ep else None

    @property
    def earlier_eval_freq(self) -> int | None:
        ep = earlier_protocol(self.env)
        return ep[1] if ep else None

    def earlier_text(self) -> str:
        """'the earlier benchmark ran this task for N steps, evaluating every F' (or that it never ran it)."""
        if self.earlier_budget is None:
            return "this task was not part of the earlier benchmark (added to the test folders on 2026-11-03)"
        return (f"the earlier benchmark ran this task for {self.earlier_budget:,} steps, evaluating every "
                f"{self.earlier_eval_freq:,}")

    # -- locations
    @property
    def figures(self) -> pathlib.Path:
        return self.dirs.figures

    @property
    def tables(self) -> pathlib.Path:
        return self.dirs.tables

    @property
    def suite_dir(self) -> pathlib.Path:
        """``<results>/<suite>``: what ``benchmarks.report.load_suite(results, suite)`` reads."""
        return self.dirs.results / self.suite

    @property
    def task_dir(self) -> pathlib.Path:
        return self.suite_dir / self.env

    @property
    def freeze_code(self) -> bool:
        """Never: these folders always run the live library."""
        return False

    # -- code
    def code_root(self) -> pathlib.Path:
        """Where the runs import ``tempo_rl`` from: the live project root (a snapshot named like the
        suite would win in ``benchmarks.run_one.use_snapshot``; :func:`launch` refuses that)."""
        return bl.code_root_for(self.suite)

    def code_hash(self) -> str:
        """Source hash of the ``tempo_rl`` the runs import (the ``tempo_source`` of every TEMPO result)."""
        return bl.source_hash(self.code_root())

    def show(self) -> None:
        """Print the settings: mode, task, seeds, suite, algorithms and devices, folders, workers."""
        steps = f"{self.budget:,}"
        what = ("TEMPO" if self.kind == "tempo" else
                {"none": "Stable-Baselines3 baselines: none selected", "all": "tuned Stable-Baselines3 baselines",
                 "no": "Stable-Baselines3 baselines at library defaults, untuned", "some": "Stable-Baselines3 baselines, some untuned"}[_tuning(self)])
        print(f"folder    : {self.folder_name} ({what}); suite '{self.suite}'")
        print(f"task      : {self.env}; seeds {list(self.seeds)}; {steps} steps per run, evaluation every "
              f"{self.eval_freq:,} steps ({self.n_eval} episodes)"
              + ("  [SMOKE: pipeline check only; the numbers mean nothing]" if self.smoke else ""))
        same = (self.real_budget, self.real_eval_freq) == (TEST_BUDGET, TEST_EVAL_FREQ)
        print(f"protocol  : BUDGET {self.real_budget:,} steps, EVAL_FREQ {self.real_eval_freq:,} "
              f"({self.real_budget // self.real_eval_freq} evaluations)"
              + (" = runlab.TEST_BUDGET / TEST_EVAL_FREQ, the same for every task and both folders" if same else
                 f" (NOT runlab's {TEST_BUDGET:,} / {TEST_EVAL_FREQ:,}: comparable only with runs of the same values)")
              + (" [smoke runs are shrunk]" if self.smoke else "")
              + f"; {self.earlier_text()}")
        if self.kind == "tempo":
            what = ('TEMPO("MlpPolicy", env) with the library defaults (the fast recipe)' if self.variant == "default"
                    else f"variant '{self.variant}' of benchmarks/variants.py")
            print(f"TEMPO     : {what}; runs the LIVE tempo_rl (source {self.code_hash()}), never frozen")
            print(f"device    : {device_for(self, 'TEMPO')}")
        else:
            if self.algos:
                from benchmarks.zoo import zoo_hyperparams
                parts = []
                for a in self.algos:
                    tuned = zoo_hyperparams(a, self.env)[1]
                    parts.append(f"{a} on {device_for(self, a)}" + ("" if tuned else " (UNTUNED: library defaults)"))
                print("baselines : " + ", ".join(parts))
            else:
                print("baselines : none selected (ALGOS = ()); ALGOS = None runs the baselines of this task")
        print("mode      : " + ("SMOKE (scratch directories, tiny runs)" if self.smoke else f"REAL (inside {self.dirs.results.parent})"))
        for k in ("results", "logs", "figures", "tables"):
            print(f"{k:10s}: {getattr(self.dirs, k)}")
        print(f"runs      : {self.suite_dir}/{self.env}/<algorithm>/<variant>/seed<k>.json")
        print(f"files     : {self.figures.name}/{self.env}.png|pdf, {self.tables.name}/{self.env}.md|json; "
              f"scheduler label '{self.label}' (output {self.label}.out in the logs folder)")
        print(f"workers   : {self.cpu_workers} CPU + {self.mps_workers} Apple-GPU; the limits are per logs folder: "
              "schedulers of other folders do not count toward them")
        for n in self.notes:
            print("note      : " + n)


def _norm_devices(devices, algos) -> dict:
    if devices is None:
        return {}
    if isinstance(devices, str):
        devices = {a: devices for a in algos}
    out = {}
    for a, d in dict(devices).items():
        if not (d in ("cpu", "mps") or str(d).startswith("cuda")):
            raise ValueError(f"devices[{a!r}] = {d!r}: must be 'cpu', 'mps' or 'cuda'")
        if a not in algos:
            raise ValueError(f"devices names {a!r}, which does not run in this notebook; algorithms: {list(algos)}")
        out[a] = d
    return out


def _check_algos(env_id: str, info: dict, algos) -> tuple:
    """Validated baseline algorithms in canonical order (``None`` = the pool of the task; toy text: :data:`TOY_TEXT_ALGOS`)."""
    if algos is None:
        return tuple(info["baselines"]) or (TOY_TEXT_ALGOS if info["toy_text"] else ())
    if isinstance(algos, str):
        algos = (algos,)
    algos = tuple(algos)
    unknown = [a for a in algos if a not in report.BASELINES]
    if unknown:
        raise ValueError(f"unknown baseline algorithm(s) {unknown}; known: {list(report.BASELINES)}")
    if len(set(algos)) != len(algos):
        raise ValueError(f"ALGOS lists an algorithm twice: {list(algos)}")
    ok = DISCRETE_ALGOS if info["discrete"] else CONTINUOUS_ALGOS
    bad = [a for a in algos if a not in ok]
    if bad:
        raise ValueError(f"{bad} cannot act in {env_id} ({'discrete' if info['discrete'] else 'continuous'} actions); "
                         f"algorithms for this task: {list(ok)}")
    if env_id in plans.IMAGE and "CrossQ" in algos:
        raise ValueError(f"CrossQ cannot run on {env_id}: its observations are images and sb3-contrib's CrossQ has no "
                         f"CNN policy (MlpPolicy only); algorithms for this task: {[a for a in ok if a != 'CrossQ']}")
    return tuple(a for a in report.BASELINES if a in algos)


def make_lab(kind: str, env_id: str, seeds=TEST_SEEDS, smoke: bool = False, algos=None, variant: str = "default",
             cpu_workers: int | None = None, mps_workers: int | None = None, devices=None,
             rerun_outdated: bool = True, out_dir=None, show: bool = True, export_env: bool = True,
             budget: int = TEST_BUDGET, eval_freq: int = TEST_EVAL_FREQ) -> Lab:
    """Resolve the settings of a per-task notebook and create its (real-mode) output folders.

    * ``kind``: ``"baselines"`` (Baselines_test) or ``"tempo"`` (TEMPO_test); ``env_id`` one of :func:`tasks`.
    * ``seeds``: default :data:`TEST_SEEDS` = 10-14 (the final seeds of the main benchmark).
    * ``budget`` / ``eval_freq``: environment steps per run and between two evaluations (the BUDGET / EVAL_FREQ lines of
      the Parameters cell); default :data:`TEST_BUDGET` = 500,000 / :data:`TEST_EVAL_FREQ` = 10,000 for every task and
      both folders. Validated (positive whole numbers, ``eval_freq`` divides ``budget``, at least 3 evaluations) before
      anything is created. Every job, estimate, plan, summary, figure, table and protocol check of the lab uses them;
      ``benchmarks/plans.py`` (the earlier benchmark, :func:`earlier_protocol`) is not used for the budget. The values of
      the earlier benchmark make :func:`import_previous` find its results again. Smoke mode still shrinks the runs to
      1,500 steps.
    * ``smoke``: tiny pipeline-check mode (see the module docstring); ``TEMPO_NOTEBOOK_SMOKE`` wins (parsed as by
      :func:`smoke_flag`: an unknown value raises; an override not yet announced by smoke_flag becomes a note). Smoke
      keeps the first seed only and caps the workers at 2 CPU + 1 GPU.
    * ``algos`` (baselines): ``None`` = the pool of the task (``report.expected_pool``; on toy text, which has no
      RL-Zoo entry, PPO and DQN with the grid-search winners of ``benchmarks/tuned_toy_text.json``: :data:`TOY_TEXT_ALGOS`); a tuple = only those;
      ``()`` = no baselines.
    * ``variant`` (tempo): ``"default"`` = ``TEMPO("MlpPolicy", env)`` or a name in ``benchmarks.variants.VARIANTS``.
    * ``cpu_workers`` / ``mps_workers``: ``None`` = automatic; limits PER LOGS FOLDER (module docstring).
    * ``devices``: ``None`` or ``{"CrossQ": "cpu", ...}`` / ``"cpu"`` overriding :data:`DEVICE_POLICY`.
    * ``rerun_outdated`` (tempo): move TEMPO results of other ``tempo_rl`` code to ``<results>/_outdated/`` at launch.
    * ``out_dir``: write results / logs / figures / tables below this folder instead (tests, relocation).
    * ``export_env``: also set ``TEMPO_BENCH_RESULTS`` / ``TEMPO_BENCH_LOGS`` in this process (like
      ``benchlab.bench_dirs``), so that a ``benchmarks.run_one`` / ``scheduler`` started from the kernel by hand lands in
      this lab's folders and never in ``benchmarks/results``. The scheduler :func:`launch` starts gets them explicitly
      either way. Pass ``False`` for a throw-away lab (a temporary ``out_dir`` that is deleted afterwards).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    info = dict(bl.task_info(env_id))                   # ValueError for an unknown task
    budget, eval_freq = _check_protocol(budget, eval_freq)
    info.update(budget=budget, eval_freq=eval_freq, n_evals=budget // eval_freq)   # this lab's protocol, not plans.py's
    notes: list = []
    if (budget, eval_freq) != (TEST_BUDGET, TEST_EVAL_FREQ):
        notes.append(f"BUDGET / EVAL_FREQ = {budget:,} / {eval_freq:,} differ from runlab.TEST_BUDGET / TEST_EVAL_FREQ = "
                     f"{TEST_BUDGET:,} / {TEST_EVAL_FREQ:,}, the protocol of both folders: these runs are comparable only with "
                     "runs made with the same values (set them in the other folder too)")
    cell = smoke
    smoke, value = _smoke_from_env(cell)                # same strict parsing as smoke_flag (ValueError before any folder)
    if value is not None and smoke != bool(cell) and (value, bool(cell)) not in _SMOKE_ANNOUNCED:
        notes.append(_smoke_note(value, cell, smoke))   # printed by show(); not repeated after smoke_flag announced it
    seeds = tuple(int(s) for s in seeds)
    if not seeds:
        raise ValueError("SEEDS is empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"SEEDS lists a seed twice: {list(seeds)}")
    if kind == "baselines":
        if variant != "default":
            raise ValueError("the baselines have no variants: variant must stay 'default'")
        algos = _check_algos(env_id, info, algos)
    else:
        from benchmarks.variants import VARIANTS
        if algos not in (None, ("TEMPO",), ["TEMPO"], "TEMPO"):
            raise ValueError("TEMPO_test runs only TEMPO: do not pass algos")
        if variant not in VARIANTS:
            raise ValueError(f"unknown TEMPO variant {variant!r}; known (benchmarks/variants.py): {sorted(VARIANTS)}")
        algos = ("TEMPO",)
    if smoke:
        seeds = seeds[:1]
    gpu = bl.gpu_available()                            # NVIDIA CUDA or the Apple GPU: the MPS_WORKERS limits apply to either
    cpu_base = None                                     # GPU slots the automatic CPU limit leaves free
    if mps_workers is None:
        mps_workers = TASK_MPS_WORKERS.get(env_id, {}).get(kind, bl.AUTO_MPS_WORKERS) if gpu else 0
        cpu_base = bl.AUTO_MPS_WORKERS if gpu else 0     # a per-task GPU cut (memory) does not hand its slots to the CPU
        if gpu and env_id in TASK_MPS_WORKERS and not smoke:
            notes.append(f"MPS_WORKERS = None: at most {mps_workers} Apple-GPU runs at once on {env_id} (memory: its replay "
                         "buffers hold images; runlab.TASK_MPS_WORKERS). Do not run the CarRacing notebooks of both "
                         "folders at the same time, or set MPS_WORKERS lower")
    elif mps_workers and not gpu:
        notes.append("the Apple GPU (MPS) is not available here: MPS_WORKERS set to 0 (GPU jobs run on the CPU)")
        mps_workers = 0
    if cpu_workers is None:
        cpu_workers = max(1, (os.cpu_count() or 8) - (int(mps_workers) if cpu_base is None else cpu_base) - 2)
    if int(cpu_workers) < 0 or int(mps_workers) < 0:
        raise ValueError("worker limits must be >= 0")
    if smoke:
        cpu_workers, mps_workers = min(int(cpu_workers), 2), min(int(mps_workers), 1)
    folder = ROOT / FOLDERS[kind]
    if out_dir is not None:
        base = pathlib.Path(out_dir)
    elif smoke:
        base = bl.scratch_root() / f"{folder.name}_{env_id}"
    else:
        base = folder
    dirs = bl.Dirs(base / "results", base / "logs", base / "figures", base / "tables", smoke)
    suite = ("smoke_" if smoke else "") + SUITES[kind]
    lab = Lab(kind=kind, env=env_id, seeds=seeds, smoke=smoke, variant=variant, algos=tuple(algos), suite=suite,
              folder=folder, dirs=dirs, label=bl.label_for(f"{LABEL_PREFIX[kind]}_{env_id}", smoke),
              cpu_workers=int(cpu_workers), mps_workers=int(mps_workers), devices=_norm_devices(devices, algos),
              rerun_outdated=bool(rerun_outdated), info=info, notes=notes)
    if kind == "tempo" and (BENCH / "snapshots" / suite / "tempo_rl").is_dir():
        lab.notes.append(f"WARNING: benchmarks/snapshots/{suite}/tempo_rl exists; benchmarks.run_one would import that "
                         "frozen copy instead of the live library. TEMPO_test never freezes: launch() refuses until "
                         "that folder is removed or renamed.")
    slow = [a for a in algos if _wanted_device(lab, a) != "cpu" and not (gpu and lab.mps_workers > 0)]
    if slow:
        lab.notes.append(f"{', '.join(slow)} would use the Apple GPU but "
                         + ("it is not available" if not gpu else "mps_workers is 0") + ": they run on the CPU (much slower)")
    elif kind == "tempo" and device_for(lab, "TEMPO") == "cpu" and bl.tempo_config(variant)["width"] >= bl.GPU_MIN_WIDTH:
        lab.notes.append(f"TEMPO's {bl.tempo_config(variant)['width']}-wide critic runs on the CPU (~"
                         f"{bl.job_rate(dict(algo='TEMPO', variant=variant), 'cpu'):.0f} env-steps/s instead of ~"
                         f"{bl.job_rate(dict(algo='TEMPO', variant=variant), 'mps'):.0f} on the Apple GPU)")
    for p in (dirs.results, dirs.logs, dirs.logs / "locks"):        # only now: a rejected argument creates nothing
        p.mkdir(parents=True, exist_ok=True)
    if export_env:
        os.environ["TEMPO_BENCH_RESULTS"] = str(dirs.results)     # like benchlab.bench_dirs; the scheduler
        os.environ["TEMPO_BENCH_LOGS"] = str(dirs.logs)           # gets them explicitly as well
    if show:
        lab.show()
    return lab


# =========================================================================== devices and jobs


def _wanted_device(lab: Lab, algo: str, cfg: dict | None = None) -> str:
    """The device :data:`DEVICE_POLICY` (or ``lab.devices``) asks for, before the worker limits are applied."""
    want = lab.devices.get(algo)
    if want is not None:
        return want
    gpu = plans.gpu_device() or "mps"                   # "cuda" or "mps"; without a GPU, device_for falls back to the CPU
    if algo == "TEMPO":
        if lab.variant == "default":
            return plans.default_tempo_device()
        width = cfg["width"] if cfg else bl.tempo_config(lab.variant)["width"]
        return gpu if width >= bl.GPU_MIN_WIDTH else "cpu"
    if algo in ("SAC", "TD3") and lab.env in plans.IMAGE:   # CnnPolicy updates: ~10x slower on one CPU thread
        return gpu
    return gpu if algo == "CrossQ" else "cpu"


def device_for(lab: Lab, algo: str, cfg: dict | None = None) -> str:
    """The device of ``algo`` in this lab (:data:`DEVICE_POLICY`); ``cfg`` = :func:`tempo_config` result
    (gives the critic width of a named variant; without it the width is read from the live defaults).
    GPU jobs (NVIDIA CUDA, else the Apple GPU) fall back to the CPU when there is no GPU or ``mps_workers`` is 0."""
    want = _wanted_device(lab, algo, cfg)
    if want != "cpu" and not (bl.gpu_available() and lab.mps_workers > 0):
        return "cpu"
    return want


def _job(lab: Lab, algo: str, variant: str, seed: int, device: str) -> dict:
    """A full-protocol job of this lab: ``steps`` / ``eval_freq`` = the lab's BUDGET / EVAL_FREQ (``lab.info``), never the
    budget of ``benchmarks/plans.py``; the other fields as in ``benchmarks.plans`` (smoke: shrunk by :func:`_finish_jobs`)."""
    return dict(suite=lab.suite, algo=algo, variant=variant, env=lab.env, seed=int(seed), steps=lab.real_budget,
                eval_freq=lab.real_eval_freq, n_eval=n_eval_for(lab.env), device=device, threads=1, cost=0,
                env_kwargs=plans.ENV_KWARGS.get(lab.env))


def _finish_jobs(lab: Lab, jobs: list[dict]) -> list[dict]:
    """Smoke: shrink (suite is ``smoke_*``); then ``cost`` = estimated seconds (longest first)."""
    if lab.smoke:
        jobs = bl.shrink(bl.first_seed_only(jobs))
    for j in jobs:
        j["cost"] = round(bl.job_seconds(j))
    return jobs


def baseline_jobs(lab: Lab) -> list[dict]:
    """One job per (baseline in ``lab.algos``, seed): variant ``default``, hyperparameters from
    ``benchmarks/zoo.py`` through ``run_one`` (never re-specified), the lab's budget / evaluation
    schedule (default :data:`TEST_BUDGET` / :data:`TEST_EVAL_FREQ`), suite ``baselines_test`` (smoke:
    first seed, 1,500 steps)."""
    if lab.kind != "baselines":
        raise ValueError("baseline_jobs is for the baselines folder; use tempo_jobs")
    jobs = [_job(lab, a, "default", s, device_for(lab, a)) for a in lab.algos for s in lab.seeds]
    return _finish_jobs(lab, jobs)


def tempo_jobs(lab: Lab, cfg: dict | None = None) -> list[dict]:
    """One TEMPO job per seed (variant ``lab.variant``, ``env_kwargs`` from ``plans.ENV_KWARGS``, e.g.
    CliffWalking ``max_episode_steps=200``, the lab's budget / evaluation schedule), all seeds on the
    same device, suite ``tempo_test``."""
    if lab.kind != "tempo":
        raise ValueError("tempo_jobs is for the TEMPO folder; use baseline_jobs")
    dev = device_for(lab, "TEMPO", cfg)
    return _finish_jobs(lab, [_job(lab, "TEMPO", lab.variant, s, dev) for s in lab.seeds])


def make_jobs(lab: Lab, cfg: dict | None = None) -> list[dict]:
    """:func:`baseline_jobs` or :func:`tempo_jobs` according to ``lab.kind``."""
    return baseline_jobs(lab) if lab.kind == "baselines" else tempo_jobs(lab, cfg)


# =========================================================================== information printers


def show_task(lab: Lab) -> dict:
    """Print the task (spaces, time limit, solved threshold, this lab's budget and evaluation schedule with the earlier
    benchmark's for reference, random policy, toy-text optimum; ``benchlab.describe_env``) and what this lab runs on it;
    returns the info (``budget`` / ``eval_freq`` / ``n_evals`` = this lab's)."""
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):               # describe_env prints; its "baselines" / "budget" lines: main benchmark
        info = bl.describe_env(lab.env)
    info.update(budget=lab.real_budget, eval_freq=lab.real_eval_freq, n_evals=lab.real_budget // lab.real_eval_freq)
    for line in buf.getvalue().splitlines():
        if line.startswith("baselines"):
            line = ("runs here         : TEMPO (compared with the exact optimum and the random policy)" if info["toy_text"]
                    else "runs here         : TEMPO only (the baselines run in Baselines_test; Comparison/ compares them)") \
                if lab.kind == "tempo" else _baselines_line(lab)
        elif line.startswith("budget"):                 # this lab's protocol, not benchmarks/plans.py's
            line = (f"budget            : {lab.real_budget:,} environment steps per run; evaluation every "
                    f"{lab.real_eval_freq:,} steps ({lab.real_budget // lab.real_eval_freq} evaluations of "
                    f"{n_eval_for(lab.env):,} episodes)"
                    + (" = runlab.TEST_BUDGET / TEST_EVAL_FREQ, the same for every task and both folders"
                       if (lab.real_budget, lab.real_eval_freq) == (TEST_BUDGET, TEST_EVAL_FREQ) else
                       " (BUDGET / EVAL_FREQ of this notebook)")
                    + (f"\nepisodes          : {n_eval_for(lab.env):,} per evaluation on this task instead of {N_EVAL} "
                       "(runlab.N_EVAL_BY_TASK: one hand is worth -1 / 0 / +1, so 10 hands cannot tell policies apart)"
                       if n_eval_for(lab.env) != N_EVAL else "")
                    + (f"\nearlier benchmark : {lab.earlier_budget:,} steps, evaluation every {lab.earlier_eval_freq:,} "
                       "(benchmarks/plans.py, benchmarks/results/final): another protocol, not used here"
                       if lab.earlier_budget is not None else
                       "\nearlier benchmark : none (task added to the test folders on 2026-11-03)"))
        elif line.startswith("observation space") and lab.env in plans.IMAGE:
            line += ("\n                    every algorithm sees uint8 frames (4, 84, 84) instead: tempo_rl.env_util.make_env "
                     "greys and crops each frame to 84x84, repeats each action for 4 frames (rewards summed) and stacks "
                     "the last 4: one environment step = 4 simulator frames")
        elif line.startswith("time limit") and lab.env in plans.IMAGE:
            line += "  (simulator frames: at most 250 agent steps per episode with frame-skip 4)"
        print(line)
    if lab.env in plans.REPLAY_CAPACITY:
        print(f"replay capacity   : {plans.REPLAY_CAPACITY[lab.env]:,} transitions for every off-policy learner (TEMPO, SAC, TD3; "
              "benchmarks.plans.REPLAY_CAPACITY): the 1M default of 500k image observations does not fit in memory")
    if lab.smoke:
        print(f"SMOKE mode        : {lab.budget:,} steps, {lab.n_eval} episodes per evaluation, seed {lab.seeds[0]} only")
    print(f"seeds             : {list(lab.seeds)}  (the final seeds of the main benchmark)")
    return info


def _zoo_commit() -> str:
    """The rl-baselines3-zoo commit ``benchmarks/zoo.py`` was copied from (read from its docstring)."""
    from benchmarks import zoo

    m = re.search(r"master @ ([0-9a-f]{7,})", zoo.__doc__ or "")
    return m.group(1) if m else "?"


def _tuning(lab: Lab) -> str:
    """``"none"`` (no baseline runs here), ``"all"`` / ``"some"`` / ``"no"`` baselines have an RL-Zoo entry for this task."""
    if not lab.algos:
        return "none"
    n = sum(d["tuned"] for d in zoo_info(lab).values())
    return "all" if n == len(lab.algos) else "no" if n == 0 else "some"


def _baselines_line(lab: Lab) -> str:
    """The ``baselines`` line of :func:`show_task` in the Baselines folder: what THIS notebook runs and whether the zoo has
    an entry for the task (``benchlab.describe_env`` always says 'RL-Baselines3-Zoo hyperparameters' or 'none')."""
    if not lab.algos:
        return "baselines         : none selected (ALGOS = ()); ALGOS = None runs the baselines of this task"
    z = zoo_info(lab)
    untuned = [a for a in lab.algos if not z[a]["tuned"]]
    names = ", ".join(lab.algos)
    if not untuned:
        from benchmarks.zoo import hp_source
        src = {a: hp_source(a, lab.env, short=True) for a in lab.algos}
        how = ("RL-Baselines3-Zoo hyperparameters" if set(src.values()) == {"zoo entry"} else
               "tuned: " + "; ".join(f"{a} {s}" for a, s in src.items()))
    else:
        lib = "library defaults" + (" plus the zoo's `default` entry for TD3" if "TD3" in untuned and _zoo_default("TD3") else "")
        how = (f"UNTUNED: no RL-Baselines3-Zoo entry for this task, {lib}" if len(untuned) == len(lab.algos) else
               f"RL-Baselines3-Zoo hyperparameters, except {', '.join(untuned)}: UNTUNED, {lib}")
    return f"baselines         : {names}  (Stable-Baselines3 / sb3-contrib, {how})"


def _zoo_default(algo: str) -> bool:
    """Whether ``benchmarks/zoo.py`` has a ``default`` entry for ``algo`` (TD3 only), i.e. what an untuned run gets on top of the
    library defaults (a harness setting such as the pixel replay cap is not one)."""
    from benchmarks.zoo import ZOO

    return bool(ZOO[algo].get("default"))


def _hp_text(v) -> str:
    """Readable value of a zoo hyperparameter (classes by name, schedules explained)."""
    if isinstance(v, str) and v.startswith("lin_"):
        return f"{v}  (linear schedule from {v[4:]} to 0 over the run)"
    if isinstance(v, type):
        return v.__name__
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_hp_text(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_hp_text(x) for x in v) + "]"
    return repr(v)


def _hp_json(v):
    if isinstance(v, type):
        return v.__name__
    if isinstance(v, dict):
        return {str(k): _hp_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_hp_json(x) for x in v]
    return v if v is None or isinstance(v, (bool, int, float, str)) else repr(v)


def zoo_info(lab: Lab) -> dict:
    """``{algo: dict(tuned, policy, device, n_envs, normalize, noise_type, noise_std, hyperparameters)}`` of
    every baseline of the lab, resolved exactly as ``benchmarks/run_one.run_sb3`` does
    (``zoo_hyperparams``; ``n_envs`` / ``normalize`` / noise split off the SB3 arguments); JSON-safe."""
    from benchmarks.zoo import zoo_hyperparams

    out = {}
    for a in lab.algos:
        hp, tuned = zoo_hyperparams(a, lab.env)
        hp.pop("n_timesteps", None)
        n_envs, normalize = hp.pop("n_envs", 1), hp.pop("normalize", False)
        nt, ns = hp.pop("noise_type", None), hp.pop("noise_std", None)
        out[a] = dict(tuned=bool(tuned), policy="CnnPolicy" if lab.env in plans.IMAGE else "MlpPolicy",
                      device=device_for(lab, a), n_envs=int(n_envs),
                      normalize=normalize if isinstance(normalize, dict) else bool(normalize), noise_type=nt,
                      noise_std=ns, hyperparameters=_hp_json(hp), _raw=hp)
    return out


def show_zoo(lab: Lab) -> dict:
    """Print the exact hyperparameters each baseline gets on this task (``benchmarks.zoo.zoo_hyperparams``:
    TUNED = an RL-Baselines3-Zoo entry for this task exists; UNTUNED = the zoo's ``default`` entry /
    library defaults), with ``n_envs``, observation / reward normalisation and action noise. Returns
    :func:`zoo_info`. Everything not listed is the Stable-Baselines3 / sb3-contrib 2.9.0 default."""
    if lab.kind != "baselines":
        raise ValueError("show_zoo is for the baselines folder; use show_tempo_config")
    z = zoo_info(lab)
    if not z:
        print(f"{lab.env}: no baselines selected (algos=()); algos=None selects the baselines of this task")
        return z
    from benchmarks.zoo import hp_source
    tuned_any = any(d["tuned"] for d in z.values())
    src = (f"benchmarks/zoo.py = RL-Baselines3-Zoo master @ {_zoo_commit()}"
           + ("; MuJoCo v4 entries (PPO partly v2) applied to v5" if lab.info["family"] == "mujoco" else "")
           + ("; toy text: grid-search winners, benchmarks/tuned_toy_text.json"
              if any(hp_source(a, lab.env, short=True) == "grid search" for a in z) else "") if tuned_any else
           "no RL-Baselines3-Zoo entry for this task: library defaults")
    print(f"hyperparameters of the baselines on {lab.env} ({src}; budget {lab.real_budget:,} steps for every algorithm, the "
          "zoo's n_timesteps is not used):")
    for a, d in z.items():
        flag = (f"TUNED ({hp_source(a, lab.env)})" if d["tuned"] else
                "UNTUNED (no zoo entry for this task: library defaults" + (" + the zoo's 'default' entry)" if _zoo_default(a) else ")"))
        noise = "none" if d["noise_type"] is None else f"{d['noise_type']} (std {d['noise_std']})"
        print(f"\n{a}: {flag}")
        print(f"  policy {d['policy']} | n_envs {d['n_envs']} | normalize "
              f"{'(obs + reward) ' + str(d['normalize']) if not isinstance(d['normalize'], dict) else d['normalize']} | "
              f"action noise {noise} | device {d['device']}")
        for k, v in d["_raw"].items():
            print(f"  {k:18s} = {_hp_text(v)}")
        if not d["_raw"]:
            print("  (no argument changed: SB3 defaults)")
    print("\nEvery other argument is the Stable-Baselines3 / sb3-contrib 2.9.0 default. Evaluation of every algorithm: "
          f"{n_eval_for(lab.env):,} deterministic episodes every eval_freq steps on held-out seeds 100000 + i.")
    if lab.env in plans.REPLAY_CAPACITY:
        print(f"buffer_size {plans.REPLAY_CAPACITY[lab.env]:,} on this task is the harness's memory limit for every off-policy "
              "learner (benchmarks.plans.REPLAY_CAPACITY), not an RL-Zoo hyperparameter.")
    if lab.env in plans.IMAGE:
        print("CnnPolicy = SB3's NatureCNN on the uint8 4x84x84 frames; CrossQ does not run here (sb3-contrib has no CNN "
              "policy for it).")
    if any(isinstance(d["normalize"], dict) for d in z.values()):
        print("normalize {norm_obs: False, norm_reward: True}: VecNormalize scales the rewards only (the zoo's setting; "
              "normalising image observations would break the CNN).")
    return z


#: ``benchlab._CONFIG_SCRIPT`` builds the model with ``TEMPO(..., **kwargs)``; the research variants whose kwargs carry the
#: private harness flags ``_scale_through_truncation`` / ``_kappa_scale`` / ``_long_entropy_schedule`` / ``_audition_opts``
#: would crash there (``benchmarks.run_one.run_tempo`` pops and applies them first). This is that ``build`` function.
_BUILD_OLD = '''def build(kwargs):
    """TEMPO exactly as benchmarks.run_one builds it (on the CPU: only the configuration is read)."""
    return TEMPO("MlpPolicy", a["env"], seed=a["seed"], device="cpu", env_kwargs=a["env_kwargs"], **kwargs)
'''
_BUILD_NEW = '''def build(kwargs):
    """TEMPO exactly as benchmarks.run_one.run_tempo builds it (on the CPU: only the configuration is read): the private
    harness flags of a variant are popped first and set on the model afterwards, never passed to TEMPO()."""
    kwargs = dict(kwargs)
    sct = kwargs.pop("_scale_through_truncation", False)
    kappa = kwargs.pop("_kappa_scale", False)
    long_s = kwargs.pop("_long_entropy_schedule", None)
    aud = kwargs.pop("_audition_opts", None)
    m = TEMPO("MlpPolicy", a["env"], seed=a["seed"], device="cpu", env_kwargs=a["env_kwargs"], **kwargs)
    if sct:
        m._scale_resets_on_truncation = False
    if aud:
        m._audition_opts = dict(aud)
    if long_s is not None and not m.discrete:
        m._long_entropy_schedule = tuple(long_s)
    if kappa and m.reward_scaler is not None:
        from tempo_rl.env_util import episode_horizon, horizon_kappa
        m.reward_scaler.kappa = horizon_kappa(m.gamma, episode_horizon(m.env))
    return m
'''


#: benchlab's ``fingerprint`` ("two models with equal fingerprints train identically for the same seed") predates the
#: speed options (``TEMPO(gpu_replay=..., ...)``, 2026-10-22/23); the swaps below add them (and the policy's
#: ``fused_critics``), so a variant that differs from the reference only in one of them is reported as different (a
#: library without them reads None on both sides). ``_amp`` is the autocast dtype resolved for the device.
_FP_OLD = '"_entropy_unit", "_entropy_log_offset")}'
_FP_NEW = ('"_entropy_unit", "_entropy_log_offset", "gpu_replay", "act_on_cpu", "pipeline", "fused_adam", "amp", '
           '"_amp", "invariance", "lr_rule", "ref_batch", "brn_impl", "n_envs")}')
_FPP_OLD = '"representation", "augment_images")}'
_FPP_NEW = '"representation", "augment_images", "fused_critics")}'
#: ... and it hands the resolved speed settings (``TEMPO.scaling_summary``: N, K, G, B, kappa, updates and samples per
#: environment step, learning-rate multipliers, Adam betas, Batch-Renorm rules) to :func:`tempo_config`.
_OUT_OLD = 'print("@@CONFIG@@" + json.dumps(out))'
_OUT_NEW = '''try:
    out["scaling"] = enc(model.scaling_summary()) if hasattr(model, "scaling_summary") else None
except Exception:
    out["scaling"] = None
print("@@CONFIG@@" + json.dumps(out))'''


@contextlib.contextmanager
def _harness_flags_in_config_script(env: str | None = None):
    """Run ``benchlab.tempo_run_config`` with a configuration script whose ``build`` handles the private variant flags,
    whose fingerprint covers the speed options and which also returns ``TEMPO.scaling_summary()``, and with
    :func:`default_reference_recipe` as the reference of ``"default"`` (benchlab itself stays untouched; the swaps last
    for the call). A part whose expected text is no longer in benchlab's script is left as it is."""
    old = bl._CONFIG_SCRIPT
    new = old
    for a, b in ((_BUILD_OLD, _BUILD_NEW), (_FP_OLD, _FP_NEW), (_FPP_OLD, _FPP_NEW), (_OUT_OLD, _OUT_NEW)):
        if new.count(a) == 1:
            new = new.replace(a, b)
    bl._CONFIG_SCRIPT = new
    old_ref = bl.reference_recipe
    bl.reference_recipe = (default_reference_recipe if env is None else
                           (lambda variant: default_reference_recipe(variant, env)))
    try:
        yield
    finally:
        bl._CONFIG_SCRIPT = old
        bl.reference_recipe = old_ref


def default_reference_recipe(variant: str, env: str | None = None) -> str | None:
    """The named recipe the configuration check compares a variant with: for ``"default"`` the documented default
    recipe ``benchmarks.variants.DEFAULT_RECIPE`` (``"fast"``: the fast recipe spelled out; benchlab's own
    :func:`benchlab.reference_recipe` still names the Screen-6 recipe ``"c1"``, the default until 2026-10-22), on the
    pixel tasks of ``benchmarks.plans.IMAGE`` ``DEFAULT_RECIPE_IMAGE`` (``"fast_image"``: the fast recipe on the classic
    engine, as the default resolves it for image observations), else ``None``."""
    from benchmarks import variants as V

    ref = getattr(V, "DEFAULT_RECIPE_IMAGE" if env in plans.IMAGE else "DEFAULT_RECIPE", None)
    return ref if variant == "default" and ref in V.VARIANTS else None


def _scaling_identity(sc: dict) -> bool:
    """True when ``TEMPO.scaling_summary()`` describes C1's update scheme (one environment, one update of batch 256 per
    step, no re-counting): the configuration display then stays as before."""
    return (sc.get("N", 1) == 1 and sc.get("K", 1) == 1 and sc.get("G", 1) == 1 and sc.get("kappa", 1.0) == 1.0
            and sc.get("lr_mult", 1.0) == 1.0 and sc.get("alpha_lr_mult", 1.0) == 1.0)


def _apply_scaling(cfg: dict) -> None:
    """Make the 'updates' line and rows say what one update is when a configuration does not update once per
    environment step (``n_envs`` > 1, ``train_freq`` > 1, ``gradient_steps`` != 1) or re-counts its settings
    (``invariance`` / ``lr_rule`` / ``alpha_lr_mult``), from ``TEMPO.scaling_summary()``."""
    sc = cfg.get("scaling")
    if not isinstance(sc, dict) or _scaling_identity(sc):
        return
    n, k, g, b = int(sc["N"]), int(sc["K"]), int(sc["G"]), int(sc["B"])
    u, spe = float(sc["updates_per_env_step"]), float(sc["samples_per_env_step"])
    per = (f"{k} vector steps" if k > 1 else "vector step") + (f" of {n} environments" if n > 1 else "")
    upd = (f"{g} update(s) of batch {b} per {per} = {u:g} updates per environment step ({spe:g} samples processed "
           "per environment step)")
    for item in cfg.get("recipe") or []:
        if item[0] == "updates":
            item[1] = re.sub(r"-?\d+ update\(s\) per step", upd, item[1])
    rows = cfg.get("rows") or []
    at = len(rows)
    for i, row in enumerate(rows):
        if row[:2] == ["resolved for this task", "updates per environment step"]:
            row[2] = f"{g}/({n} x {k}) = {u:g} (batch {b}; {spe:g} samples per environment step)"
            at = i + 1
    betas = sc.get("adam_betas") or [None, None]
    parts = [f"kappa = batch / ref_batch = {float(sc['kappa']):g}", f"learning rate x {float(sc['lr_mult']):g}",
             f"critic / actor Adam betas ({betas[0]:g}, {betas[1]:g})" if None not in betas else None,
             (f"Batch-Renorm momentum {float(sc['brn_momentum']):.6g}, warm-up {sc['brn_warmup']} passes"
              if sc.get("brn_momentum") is not None else None),
             f"SALE fixed copy refreshed every {sc['rep_refresh_every']} updates",
             f"temperature learning rate x {float(sc['alpha_lr_mult']):g}",
             f"actor / temperature every {sc['policy_delay']} update(s)",
             f"Polyak tau {float(sc['polyak_tau']):.6g}" if sc.get("polyak_tau") is not None else None]
    rows.insert(at, ["resolved for this task", "update scaling (TEMPO.scaling_summary)",
                     "; ".join(x for x in parts if x), "instantiated model"])


def tempo_config(lab: Lab) -> dict:
    """The exact TEMPO configuration of this lab's runs.

    ``benchlab.tempo_run_config`` instantiates ``TEMPO("MlpPolicy", env, ...)`` exactly as
    ``benchmarks.run_one`` does in a fresh subprocess (the notebook kernel stays light) and returns every
    constructor argument with its source, the settings resolved on this task and, for ``"default"``, whether
    it equals the documented default recipe (``benchmarks.variants.DEFAULT_RECIPE`` = ``"fast"``). Adds ``device`` (this lab's device policy), ``code_hash`` (source hash
    of the live ``tempo_rl``) and ``variant``. It also makes the planning rates use the resolved widths."""
    if lab.kind != "tempo":
        raise ValueError("tempo_config is for the TEMPO folder; use show_zoo for the baselines")
    exp = types.SimpleNamespace(env=lab.env, variant=lab.variant, seeds=lab.seeds, smoke=lab.smoke,
                                mps_workers=lab.mps_workers, info=lab.info, suite=lab.suite)
    with _harness_flags_in_config_script(lab.env):
        cfg = bl.tempo_run_config(exp)
    _apply_scaling(cfg)
    for k, v in (cfg.get("kwargs") or {}).items():          # private flags of a research variant: shown, not hidden
        if k.startswith("_"):
            cfg["rows"].append(["benchmark harness flag", k, v if isinstance(v, str) else repr(v),
                                "variant (set on the model by benchmarks/run_one.py after TEMPO(...), not a TEMPO argument)"])
    cfg["device"] = device_for(lab, "TEMPO", cfg)
    cfg["code_hash"] = lab.code_hash()
    for row in cfg.get("rows", []):                          # the table must show the device this lab really uses
        if row[:2] == ["TEMPO argument", "device"]:
            row[2], row[3] = cfg["device"], "this notebook's device policy"
    cfg["milestones"] = tempo_milestones(lab, cfg)
    return cfg


def tempo_milestones(lab: Lab, cfg: dict) -> dict:
    """The budget-relative quantities of this lab's TEMPO runs, from the configuration the LIVE library resolved
    (:func:`tempo_config`, instantiated for ``lab.budget``): ``learning_starts`` (the library's own value for this budget,
    the ``learning starts`` row; None if it cannot be read), ``frac`` (anneal fraction of the entropy schedule: the
    variant's ``entropy_schedule`` or the library's ``DEFAULT_ENTROPY_SCHEDULE``), ``tempering_end`` (step at which the
    entropy target reaches its final value; ``second_anneal_end`` for a 5-value schedule), ``novelty_end`` (step at which
    the novelty bonus has decayed to 0; None without a bonus) and ``audition_at`` (step at which a ``horizon_audition``
    decides: the end of tempering; None without an audition). ``budget`` = the steps of the runs (smoke: 1,500)."""
    budget = int(lab.budget)
    ls = None
    for row in cfg.get("rows", []):
        if row[:2] == ["resolved for this task", "learning starts"]:
            v = row[2]
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                ls = int(v)
            else:
                m = re.search(r"auto:\s*([\d,]+)\s*steps", str(v))
                ls = int(m.group(1).replace(",", "")) if m else None
    kw = cfg.get("kwargs") or {}
    kind = "discrete" if cfg.get("discrete") else "continuous"
    sched = kw.get("entropy_schedule") or ((cfg.get("defaults") or {}).get("entropy_schedule") or {}).get(kind)
    try:
        frac = float(sched[2])
    except (TypeError, IndexError, ValueError):
        frac = None
    second = float(sched[4]) if frac is not None and len(sched) == 5 else None
    at = (lambda f: int(round(f * budget)) if f is not None and f > 0 else None)
    bonus = float(cfg.get("exploration_bonus") or 0.0)
    return dict(budget=budget, learning_starts=ls, frac=frac, tempering_end=at(frac),
                second_anneal_end=at(second) if second and frac is not None and second > frac else None,
                novelty_end=at(frac) if bonus > 0 else None, exploration_bonus=bonus,
                audition_at=(at(frac) if frac and frac > 0 else int(round(0.5 * budget))) if kw.get("horizon_audition") else None)


def _milestones_text(ms: dict) -> str:
    """One sentence on the budget-relative quantities of :func:`tempo_milestones`."""
    b = ms["budget"]
    parts = [f"learning starts after {ms['learning_starts']:,} steps of uniform-random actions (the library's value for "
             f"this budget)" if ms.get("learning_starts") is not None else "learning starts: not readable"]
    if ms.get("tempering_end") is not None:
        parts.append(f"the entropy target is tempered over the first {ms['frac']:.0%} = {ms['tempering_end']:,} steps"
                     + (f", a second anneal ends at step {ms['second_anneal_end']:,}" if ms.get("second_anneal_end") else ""))
    if ms.get("novelty_end") is not None:
        parts.append(f"the novelty bonus decays from {ms['exploration_bonus']:g} to 0 by step {ms['novelty_end']:,}")
    if ms.get("audition_at") is not None:
        parts.append(f"the horizon audition decides at step {ms['audition_at']:,}")
    return f"{b:,} steps per run: " + "; ".join(parts)


def _cost_note(lab: Lab) -> str | None:
    """Words for planning rates that are estimates or lower bounds: the fast recipe (the default since 2026-10-23, and
    any ``gpu_replay`` variant) is planned from measured single-run speed-ups, not timed under a full schedule;
    ``benchlab.job_rate`` models the critic width, SALE, weight norm and the novelty bonus, not the second (shadow)
    learner of ``horizon_audition``."""
    if lab.kind != "tempo":
        return None
    if lab.variant == "default" and lab.env in plans.IMAGE:
        return ("the default on image observations runs the classic engine (the phase engine does not support images: "
                "replay buffer on the host, acting on the GPU; 4 environments, batch 1024, bfloat16 critics and the rest of "
                "the fast recipe unchanged) and no novelty bonus (TEMPO disables it on image-only observations); its "
                "planning rate is runlab.TASK_RATES (measured alone: 17.6 env-steps/s on the Apple GPU, planned at "
                f"{TASK_RATES[lab.env]['TEMPO']['mps']:g} with the GPU shared and evaluations included: an ESTIMATE)")
    if lab.variant == "default":
        kw = {"gpu_replay": True, "pipeline": True, "n_envs": 4, "batch_size": 1024, "amp": "auto"}
        audition = False
        what = "the default (the fast recipe since 2026-10-23"
    else:
        try:
            from benchmarks.variants import tempo_kwargs

            kw = tempo_kwargs(lab.variant, lab.env)
            audition = bool(kw.get("horizon_audition"))
        except Exception:                                   # noqa: BLE001
            return None
        what = f"variant '{lab.variant}' (TEMPO's phase engine"
    if kw.get("gpu_replay"):
        return (f"{what}: gpu_replay=True"
                + (", pipelined CPU acting" if kw.get("pipeline") else "")
                + (f", {kw['n_envs']} environments, batch {kw.get('batch_size', 256)}" if kw.get("n_envs", 1) > 1 else "")
                + (", bfloat16 critics on the GPU" if kw.get("amp") else "")
                + ") is NOT bit-identical to the previous default 'c1' (recipe C1) and its scores have not been "
                "screened; its planning rate is "
                + (f"{TEMPO_VARIANT_RATE_SOURCES.get(lab.variant, 'from')} (runlab.TEMPO_VARIANT_RATES; Apple GPU, on "
                   "the CPU benchlab's C1 model)"
                   if lab.variant in TEMPO_VARIANT_RATES else
                   "benchlab's model x runlab.GPU_REPLAY_SPEEDUP: a guess"))
    if audition:
        return (f"variant '{lab.variant}' uses horizon_audition: a second (shadow) learner trains on the same replay buffer, which "
                "the planning rates do not include: the real runs are slower than planned, so read every estimate here "
                "as a LOWER BOUND")
    return None


def show_tempo_config(lab: Lab, cfg: dict) -> None:
    """Print which code runs (live ``tempo_rl`` with its source hash), the device, the budget-relative quantities at
    this lab's budget (:func:`tempo_milestones`: learning starts, end of tempering, novelty-bonus decay, audition), the
    recipe resolved on this task (one line per ingredient: critic, actor, representation, updates, entropy target,
    discount, action noise, novelty bonus), the check against the screening recipe, and the table of every TEMPO /
    policy argument with its source (library default / variant / benchmark harness)."""
    import pandas as pd
    from IPython.display import display

    what = ('TEMPO("MlpPolicy", env) with no keyword arguments: the library defaults'
            if cfg["variant"] == "default" else f"variant '{cfg['variant']}' of benchmarks/variants.py")
    if cfg["variant"] == "default" and lab.env in plans.REPLAY_CAPACITY:
        what = (f'TEMPO("MlpPolicy", env, buffer_size={plans.REPLAY_CAPACITY[lab.env]:_}): the library defaults except the '
                "replay capacity, the harness's memory limit for every off-policy learner on this pixel task "
                "(benchmarks.plans.REPLAY_CAPACITY)")
    print(f"TEMPO on {cfg['env']}: {what}")
    print(f"code    : LIVE library {cfg['tempo_rl_file'].rsplit('/', 1)[0]} (tempo_rl {cfg['version']}, source "
          f"{cfg['source_hash']}); never frozen. Every result records its source hash; a result made with other "
          "code is flagged 'older code'"
          + (" and moved to results/_outdated/ at the next launch." if lab.rerun_outdated else " (kept: rerun_outdated=False)."))
    if cfg.get("snapshot"):
        print(f"          WARNING: {cfg['code_root']} is a frozen snapshot; the runs would import it, not the live library")
    rate = bl.job_rate(dict(algo="TEMPO", variant=cfg["variant"], env=lab.env), cfg["device"])   # runlab.TASK_RATES apply
    w = cfg["width"]
    why = ("explicit devices= override" if "TEMPO" in lab.devices else
           f"critic width {w} >= {bl.GPU_MIN_WIDTH}: GPU ({cfg['device']})" if cfg["device"] != "cpu" else
           f"critic width {w} >= {bl.GPU_MIN_WIDTH} would use the Apple GPU, but it is unavailable or mps_workers is 0: "
           "CPU, several times slower" if w >= bl.GPU_MIN_WIDTH else f"critic width {w} < {bl.GPU_MIN_WIDTH}: small networks "
           "are as fast on the CPU")
    print(f"device  : {cfg['device']} ({why}); all seeds on this device; planning rate ~{rate:.0f} env-steps/s per run")
    if _cost_note(lab):
        print("          NOTE: " + _cost_note(lab))
    ms = cfg.get("milestones") or tempo_milestones(lab, cfg)
    print(f"budget  : {_milestones_text(ms)}"
          + (f" [SMOKE; the real runs: BUDGET {lab.real_budget:,}]" if lab.smoke else
             f" (BUDGET {lab.real_budget:,}, evaluation every {lab.real_eval_freq:,})"))
    for name, text in cfg.get("recipe") or []:
        print(f"{name:15s}: {text}")
    m = cfg.get("match")
    if m:
        if m.get("equal"):
            print(f"check          : identical to the recipe '{m['reference']}' of benchmarks/variants.py (same resolved "
                  "settings, same networks, same initial weights for this seed)")
        elif m.get("equal") is None:
            print(f"check          : could not compare with '{m['reference']}' ({m.get('error')})")
        else:
            print(f"check          : WARNING: differs from the recipe '{m['reference']}' of benchmarks/variants.py in "
                  + ", ".join(m.get("differences") or []))
    print()
    df = pd.DataFrame(cfg["rows"], columns=["group", "setting", "value", "source"])
    df["value"] = df["value"].map(lambda v: v if isinstance(v, str) else repr(v))
    df = df.set_index(["group", "setting"])
    with pd.option_context("display.max_rows", 300, "display.max_colwidth", 160):
        try:
            display(df.style.set_properties(**{"text-align": "left"}).set_table_styles(
                [dict(selector="th", props=[("text-align", "left")])]))
        except Exception:                                   # noqa: BLE001 - no jinja2: plain table
            display(df)


# =========================================================================== status, plan, estimate


def _failed_ids(lab: Lab) -> set:
    f = lab.dirs.logs / f"{lab.label}.status"
    try:
        return set(json.loads(f.read_text()).get("failed", [])) if f.exists() else set()
    except (OSError, ValueError):
        return set()


def _manifest_path(lab: Lab) -> pathlib.Path:
    return lab.suite_dir / "_imported.json"


def _read_manifest(lab: Lab) -> list[dict]:
    """The entries of ``<suite>/_imported.json``; ``[]`` when it is missing, corrupt or partially written (entries that are
    not records are dropped)."""
    try:
        data = json.loads(_manifest_path(lab).read_text())
    except (OSError, ValueError):
        return []
    return [m for m in data if isinstance(m, dict)] if isinstance(data, list) else []


def _write_manifest(lab: Lab, entries: list[dict]) -> None:
    """Write the manifest atomically (temporary file + ``os.replace``): a reader never sees half a file."""
    p = _manifest_path(lab)
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(json.dumps(entries, indent=1))
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def _manifest_lock(lab: Lab, timeout: float = 120.0):
    """Exclusive ``flock`` on ``<results>/<suite>/.imported.lock`` for a read-manifest -> copy / move -> write step. Two
    notebooks of one folder importing at the same moment each rewrote the whole manifest and lost the other's entries.
    Raises TimeoutError instead of writing without the lock."""
    import fcntl

    lab.suite_dir.mkdir(parents=True, exist_ok=True)
    fh = open(lab.suite_dir / ".imported.lock", "a")
    try:
        t0 = time.time()
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"another process holds {lab.suite_dir / '.imported.lock'} for more than {timeout:.0f} s "
                                       "(an import of another notebook of this folder?); nothing was changed: run the cell again")
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    finally:
        fh.close()


def _imported(lab: Lab) -> dict:
    """{(env, algo, seed): sha256} of results copied by :func:`import_previous` (from ``<suite>/_imported.json``; a missing,
    corrupt or partially written manifest = nothing imported). Entries of copies that :func:`launch` moved away
    (``moved_to``) do not count."""
    out = {}
    for m in _read_manifest(lab):
        if m.get("moved_to"):
            continue
        try:
            out[(m["env"], m["algo"], int(m["seed"]))] = m.get("sha256")
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _sha256(path: pathlib.Path) -> str | None:
    import hashlib

    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


# ---- existing result files: can they be used for the planned run?

#: Job fields that define a run's protocol (a result whose job differs in one of them is of 'other protocol'). The suite is
#: not one of them (imported results say 'final'); device and threads are provenance.
_PROTOCOL_FIELDS = ("steps", "eval_freq", "n_eval", "seed", "algo", "env", "variant", "env_kwargs")
_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _installed_versions() -> dict:
    """The versions ``benchmarks.run_one`` records, of this interpreter's packages (installed metadata; nothing imported):
    python / torch / gymnasium / numpy (``versions`` of a result), stable_baselines3 / sb3_contrib (``info``)."""
    import importlib.metadata as md
    import platform

    out = dict(python=platform.python_version())
    for key, dist in (("torch", "torch"), ("gymnasium", "gymnasium"), ("numpy", "numpy"),
                      ("stable_baselines3", "stable-baselines3"), ("sb3_contrib", "sb3-contrib")):
        out[key] = _dist_version(md, dist)
    return out


def _neq(name: str, got, want, what: str = "") -> str:
    """``"<name> <got> != <what><want>"`` for a message about a value that differs. When the two print the same (the string
    '1' recorded where the number 1 is planned) both are shown as reprs, so the message never reads 'threads 1 != 1'."""
    g, w = (repr(got), repr(want)) if str(got) == str(want) else (got, want)
    return f"{name} {g} != {what}{w}"


def _version_differences(rec: dict, algo: str, now: dict | None = None) -> list[str]:
    """Library versions a run recorded that differ from the installed ones: python / torch / gymnasium / numpy for every
    run, Stable-Baselines3 for every baseline and sb3-contrib for CrossQ. ``now`` = :func:`_installed_versions`."""
    now = now or _installed_versions()
    vers = rec.get("versions") if isinstance(rec.get("versions"), dict) else {}
    info = rec.get("info") if isinstance(rec.get("info"), dict) else {}
    out = [_neq(k, vers.get(k), now[k], "installed ") for k in ("python", "torch", "gymnasium", "numpy")
           if now.get(k) is not None and vers.get(k) != now[k]]
    if algo != "TEMPO":
        for key, inst in (("sb3_version", "stable_baselines3"), ("sb3_contrib_version", "sb3_contrib")):
            if key == "sb3_contrib_version" and algo != "CrossQ":
                continue
            if now.get(inst) is not None and info.get(key) != now[inst]:
                out.append(_neq(key, info.get(key), now[inst], "installed "))
    return out


def _mask_addresses(v):
    """``v`` with the memory addresses in reprs masked (``<function ... at 0x10a2b3c40>``), recursively: every run of the
    same code records its schedule functions under another address."""
    if isinstance(v, str):
        return _ADDRESS.sub(" at 0x?", v)
    if isinstance(v, dict):
        return {k: _mask_addresses(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_mask_addresses(x) for x in v]
    return v


def _hp_differences(algo: str, env: str, info, budget: int) -> list[str]:
    """How the hyperparameters a baseline run recorded (``info`` of its result: ``zoo_tuned``, ``n_envs``, ``normalize`` and
    ``config``, the SB3 arguments as ``benchmarks.run_one.run_sb3`` stores them) differ from what ``run_sb3`` builds now
    from ``benchmarks/zoo.py``. ``lin_X`` schedules are stored as the repr of a function and compared with the memory
    address masked; the action noise by class and sigma. Empty = identical."""
    from benchmarks.run_one import _schedule
    from benchmarks.zoo import zoo_hyperparams

    info = info if isinstance(info, dict) else {}
    diff: list[str] = []
    hp, tuned = zoo_hyperparams(algo, env)
    hp.pop("n_timesteps", None)
    n_envs, normalize = hp.pop("n_envs", 1), hp.pop("normalize", False)
    noise_type, noise_std = hp.pop("noise_type", None), hp.pop("noise_std", None)
    if info.get("zoo_tuned") != bool(tuned):
        diff.append(_neq("zoo_tuned", info.get("zoo_tuned"), bool(tuned)))
    if info.get("n_envs") != n_envs:
        diff.append(_neq("n_envs", info.get("n_envs"), n_envs))
    normalize = normalize if isinstance(normalize, dict) else bool(normalize)   # run_sb3 records the zoo's dict form as is
    if info.get("normalize") != normalize:
        diff.append(_neq("normalize", info.get("normalize"), normalize))
    rcfg = dict(info["config"]) if isinstance(info.get("config"), dict) else {}
    action_noise = rcfg.pop("action_noise", None)
    if (action_noise is not None) != (noise_type is not None):
        diff.append(f"action noise {'recorded' if action_noise is not None else 'absent'}, zoo says {noise_type}")
    elif action_noise is not None:
        cls = "NormalActionNoise" if noise_type == "normal" else "OrnsteinUhlenbeckActionNoise"    # as run_sb3 picks it
        m = re.search(r"sigma=\[\s*([0-9.eE+-]+)", str(action_noise))
        if not str(action_noise).startswith(cls + "("):
            diff.append(f"action noise {str(action_noise).split('(')[0]} != {cls}")
        elif m and noise_std is not None and abs(float(m.group(1)) - float(noise_std)) > 1e-9:
            diff.append(f"action noise sigma {m.group(1)} != {noise_std}")
    for k in ("learning_rate", "clip_range"):                  # run_sb3 turns 'lin_X' into a schedule function
        if k in hp:
            hp[k] = _schedule(hp[k], budget)
    want = {k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else repr(v)) for k, v in hp.items()}
    want = _mask_addresses(json.loads(json.dumps(want, default=repr)))      # what the result file would hold
    rpk = rcfg.get("policy_kwargs")
    if isinstance(rpk, dict) and "use_sde" not in (hp.get("policy_kwargs") or {}):
        rcfg["policy_kwargs"] = {k: v for k, v in rpk.items() if k != "use_sde"}      # sb3-contrib adds it to the dict in place
    rcfg = _mask_addresses(rcfg)
    bad = sorted(k for k in set(want) | set(rcfg) if want.get(k, "<absent>") != rcfg.get(k, "<absent>"))
    if bad:
        diff.append("config differs from benchmarks/zoo.py in " + ", ".join(bad[:6]))
    return diff


def _curve_problem(rec: dict, steps: int, eval_freq: int, n_eval: int) -> str | None:
    """Why the learning curve of a result is incomplete for the schedule (``steps // eval_freq`` evaluations at or before the
    budget, ``n_eval`` episodes each), else None. Evaluations AFTER the budget are normal (PPO with many parallel
    environments overshoots the budget by up to one rollout) and are not checked."""
    s, r = rec.get("steps"), rec.get("returns")
    if not isinstance(s, list) or not isinstance(r, list):
        return "no learning curve (steps / returns)"
    if len(s) != len(r):
        return "curve: steps and returns differ in length"
    try:
        keep = [i for i, st in enumerate(s) if st <= steps]
        want = int(steps) // int(eval_freq)
        if len(keep) < want:
            return f"curve: {len(keep)} evaluations within the budget, the protocol has {want}"
        if any(len(r[i]) != n_eval for i in keep):
            return f"curve: an evaluation has other than {n_eval} episodes"
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for i in keep for x in r[i]):
            return "curve: an evaluation holds values that are not numbers"
    except (TypeError, ValueError, ZeroDivisionError):
        return "curve: malformed steps / returns"
    return None


def _job_differences(job: dict, rj: dict) -> list[str]:
    """Protocol fields (:data:`_PROTOCOL_FIELDS`) in which a result's job record differs from the planned job (a missing key
    counts as None; empty env_kwargs = None, as ``make_env`` treats them)."""
    out = []
    for f in _PROTOCOL_FIELDS:
        want, got = job.get(f), rj.get(f)
        if f == "env_kwargs":
            want, got = want or None, got or None
        if got != want:
            out.append(f"{f} {got!r} != planned {want!r}")
    return out


def _check_record(job: dict, rec, now: dict | None = None) -> tuple:
    """``(state, why, provenance)`` of a parsed result file at the path of the planned ``job``:

    * state ``"unreadable"``: no job record / steps / returns, or the curve is incomplete (:func:`_curve_problem`);
    * state ``"other protocol"``: the job differs in a protocol field (:func:`_job_differences`) or, for a baseline, the
      recorded hyperparameters differ from what ``run_one`` builds now (:func:`_hp_differences`);
    * state ``None``: usable; ``provenance`` lists what differs without changing the protocol (device, threads, library
      versions): such a run is kept and used, with a NOTE.

    In smoke mode the planned (shrunk) job is the reference."""
    if not isinstance(rec, dict) or not isinstance(rec.get("job"), dict):
        return "unreadable", ["no job record"], []
    missing = [k for k in ("steps", "returns") if k not in rec]
    if missing:
        return "unreadable", ["no " + " / ".join(missing)], []
    rj = rec["job"]
    diff = _job_differences(job, rj)
    if diff:
        return "other protocol", diff, []
    curve = _curve_problem(rec, job["steps"], job["eval_freq"], job["n_eval"])
    if curve:
        return "unreadable", [curve], []
    if job["algo"] != "TEMPO":
        hp = _hp_differences(job["algo"], job["env"], rec.get("info"), job["steps"])
        if hp:
            return "other protocol", hp, []
    prov = [_neq(f, rj.get(f, d), job.get(f, d), "planned ") for f, d in (("device", "cpu"), ("threads", 1))
            if rj.get(f, d) != job.get(f, d)]                    # run_one's defaults for a missing key
    return None, [], prov + _version_differences(rec, job["algo"], now)


def _inspect(job: dict, path: pathlib.Path, now: dict | None = None) -> dict | None:
    """Read and check the result file of a planned job: ``dict(state, why, provenance, rec)`` (:func:`_check_record`; a file
    that cannot be read or parsed is ``"unreadable"`` with the parser's message), or None when there is no file."""
    try:
        text = pathlib.Path(path).read_text()
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as e:
        return dict(state="unreadable", why=[f"cannot be read ({type(e).__name__}: {e})"], provenance=[], rec=None)
    try:
        rec = json.loads(text)
    except ValueError as e:
        return dict(state="unreadable", why=[f"not valid JSON ({e})"], provenance=[], rec=None)
    state, why, prov = _check_record(job, rec, now)
    return dict(state=state, why=why, provenance=prov, rec=rec)


def _reference_job(lab: Lab, algo: str, variant: str, seed: int) -> dict:
    """The job this lab WOULD plan for a result file on disk that is not one of its planned runs (an algorithm not in ALGOS,
    another TEMPO variant): the lab's protocol (smoke: shrunk) and the device the policy gives that algorithm / variant.
    :func:`summary` checks such files against it before they enter the statistics."""
    dev = lab.devices.get(algo)
    if dev is None and algo == "TEMPO" and variant != lab.variant:
        try:
            want = (plans.default_tempo_device() if variant == "default" else
                    (plans.gpu_device() or "mps") if bl.tempo_config(variant)["width"] >= bl.GPU_MIN_WIDTH else "cpu")
        except Exception:                                   # noqa: BLE001 - an unknown variant name on disk
            want = "cpu"
        dev = want if want == "cpu" or (bl.gpu_available() and lab.mps_workers > 0) else "cpu"
    elif dev is None or (dev != "cpu" and not (bl.gpu_available() and lab.mps_workers > 0)):
        dev = device_for(lab, algo)
    j = _job(lab, algo, variant, seed, dev)
    if lab.smoke:
        j = bl.shrink([j])[0]
    return j


def _provenance_note(items) -> str:
    """One NOTE line for ``[(label, seed, differences)]`` of runs that are kept although their provenance differs."""
    return (f"NOTE: {len(items)} run(s) are kept and used, but were recorded with other settings than this notebook's planned "
            "runs / installed packages (the protocol is the same): "
            + "; ".join(f"{l} seed {k}: {', '.join(d)}" for l, k, d in items))


# ---- processes: liveness that survives recycled pids and zombies

#: Marks of the processes a lock file may legitimately point to: a run, or (for the few milliseconds between fork and exec) the
#: scheduler that is about to become one.
_JOB_PROCESS_MARKS = ("benchmarks.run_one", "benchmarks.scheduler")


def _alive(pid: int) -> bool:
    """``os.kill(pid, 0)``, except that a zombie child of THIS process (a killed scheduler that nobody has reaped) is not
    alive. ``benchlab._alive`` counts it as alive, which made :func:`stop` wait 30 s."""
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    try:
        if os.waitpid(int(pid), os.WNOHANG)[0] == int(pid):         # exited child: reaped now
            return False
    except OSError:                                                 # not our child (or already reaped)
        pass
    return True


def _job_pids() -> set | None:
    """pids of the live ``benchmarks.run_one`` (and, briefly, ``benchmarks.scheduler``) processes of this user; ``None`` if
    the process table cannot be read. A lock file whose pid is not in this set is stale even though ``kill(pid, 0)``
    succeeds (the pid was recycled by an unrelated process)."""
    try:
        return {pid for pid, _, _, cmd in bl._processes() if any(m in cmd for m in _JOB_PROCESS_MARKS)}
    except Exception:                                               # noqa: BLE001
        return None


def _lock_live(pid, procs: set | None) -> bool:
    """Is the process of a lock file a live job? (plain liveness when the process table is unavailable)"""
    if pid is None:
        return False
    return int(pid) in procs if procs is not None else _alive(pid)


def _is_unrelated(pid: int) -> bool:
    """True only if ``pid`` is a live process that is NOT a zombie (a finished job whose scheduler has not reaped it yet still
    answers ``kill(pid, 0)``; its lock belongs to that job and the scheduler removes it)."""
    try:
        out = subprocess.run(["ps", "-o", "stat=,command=", "-p", str(pid)], capture_output=True, text=True, timeout=10).stdout
    except Exception:                                               # noqa: BLE001
        return False
    out = out.strip()
    return bool(out) and not out.startswith("Z") and not any(m in out for m in _JOB_PROCESS_MARKS)


def _clean_stale_locks(lab: Lab, jobs: list[dict] | None = None, verbose: bool = True) -> list:
    """Remove lock files of this lab's logs folder whose process id is alive but belongs to an UNRELATED process (a recycled
    pid; after a crash or reboot many locks are left behind). The scheduler and :func:`status_table` would treat such a job as
    running forever. With ``jobs`` only the locks of those jobs. Locks of dead processes are left to the scheduler (it
    overwrites them) and to :func:`stop`. Returns the removed files."""
    procs = _job_pids()
    if procs is None:
        return []
    ids = {bl.job_id(j) for j in jobs} if jobs else None
    gone = []
    for f in sorted((lab.dirs.logs / "locks").glob("*.pid")):
        if ids is not None and f.stem not in ids:
            continue
        pid = bl._lock_pid(f)
        if pid is not None and _alive(pid) and pid not in procs and _is_unrelated(pid):
            f.unlink(missing_ok=True)
            gone.append(f)
    if gone and verbose:
        print(f"removed {len(gone)} stale lock file(s) whose process id now belongs to an unrelated process: "
              + ", ".join(f.stem for f in gone[:3]) + (" ..." if len(gone) > 3 else ""))
    return gone


def _jobs_file_of(info: dict) -> pathlib.Path | None:
    """The ``--jobs-file`` of a scheduler (from its pidfile record or its command line; the path may contain spaces)."""
    if info.get("jobs_file"):
        return pathlib.Path(info["jobs_file"])
    cmd = info.get("cmd")
    cmd = " ".join(map(str, cmd)) if isinstance(cmd, (list, tuple)) else str(cmd or "")
    m = re.search(r"--jobs-file (.+?\.jobs\.json)", cmd)
    return pathlib.Path(m.group(1)) if m else None


def _scheduler_info(lab: Lab) -> dict | None:
    """The live scheduler of THIS lab: ``benchlab.scheduler_info`` with the process-table fallback restricted to schedulers
    whose ``--jobs-file`` is this lab's (otherwise any lab with the same label in another logs folder, e.g. another
    scratch folder or a copy of the project, would be told 'already running' and its :func:`stop` would kill the foreign
    scheduler and its jobs)."""
    f = bl._pidfile(lab.label, lab.dirs)
    if f.exists():
        try:
            info = json.loads(f.read_text())
            if _alive(info["pid"]) and bl._is_scheduler(bl._cmdline(info["pid"]), lab.label):
                return info
        except (ValueError, KeyError, TypeError):
            pass
    want = str(lab.dirs.logs / f"{lab.label}.jobs.json")
    for pid, _, pgid, cmd in bl._processes():                       # e.g. started from a terminal
        if bl._is_scheduler(cmd, lab.label) and pid != os.getpid() and re.search(
                rf"--jobs-file {re.escape(want)}(\s|$)", cmd):
            return dict(pid=pid, pgid=pgid, label=lab.label, cmd=cmd, started="?")
    return None


@contextlib.contextmanager
def _scoped(lab: Lab):
    """While ``benchlab.launch`` / ``benchlab.stop`` run, make their scheduler lookup this lab's (:func:`_scheduler_info`) and
    their liveness test zombie-aware (:func:`_alive`). benchlab itself is not modified; the swap is undone afterwards."""
    keep = bl.scheduler_info, bl._alive
    bl.scheduler_info = lambda label, dirs: _scheduler_info(lab) if label == lab.label else keep[0](label, dirs)
    bl._alive = _alive
    try:
        yield
    finally:
        bl.scheduler_info, bl._alive = keep


# ---- TEMPO results made while tempo_rl was being edited

def _code_mtime(lab: Lab) -> float:
    """Newest modification time of the ``tempo_rl/*.py`` files the runs import."""
    try:
        return max(p.stat().st_mtime for p in (lab.code_root() / "tempo_rl").glob("*.py"))
    except (OSError, ValueError):
        return 0.0


def _code_mtimes(lab: Lab) -> tuple:
    """Modification times of the ``tempo_rl/*.py`` files the runs import (one per file)."""
    try:
        return tuple(p.stat().st_mtime for p in (lab.code_root() / "tempo_rl").glob("*.py"))
    except OSError:
        return ()


def _started_at(finished: str, wall_s) -> float | None:
    """Epoch second a run started (``finished`` minus its wall-clock), from the fields of its result file."""
    try:
        return time.mktime(time.strptime(str(finished), "%Y-%m-%d %H:%M:%S")) - float(wall_s)
    except (ValueError, TypeError, OverflowError):
        return None


def _stale_reason(lab: Lab, source, started, code: str, mtime, ended=None, source_start=None) -> str | None:
    """Why a TEMPO result does not count as made with the live code: ``"other code"`` (its ``tempo_source`` differs from the
    live hash) or ``"edited during the run"`` (same hash, but a ``tempo_rl`` file was modified while the run was going:
    ``benchmarks.run_one`` computes ``tempo_source`` when the run ENDS, while the process imported the code when it began,
    so the stamp can describe code the run never ran). ``mtime``: the files' modification times (:func:`_code_mtimes`; a single
    newest time is accepted too); ``ended``: the run's end epoch. A touch or a content-preserving save after the run ended does
    not count. ``source_start``: the hash ``benchmarks.run_one`` recorded when the run began (results since 2026-11-05); when it
    differs from the end hash ``source``, the code changed during the run whatever the modification times say. ``None``: current."""
    if source != code:
        return "other code"
    if source_start is not None and source_start != source:
        return "edited during the run"
    if started is not None:
        hi = ended + 1.0 if ended is not None else float("inf")
        if any(started + 1.0 < m <= hi for m in (mtime if isinstance(mtime, (tuple, list)) else (mtime,))):
            return "edited during the run"
    return None


def _result_stamp(res: pathlib.Path) -> tuple:
    """(tempo_source, start epoch) of a result file."""
    try:
        r = json.loads(pathlib.Path(res).read_text())
    except (OSError, ValueError):
        return None, None
    return r.get("tempo_source"), _started_at(r.get("finished"), r.get("wall_s"))


def _rows(lab: Lab, jobs: list[dict]) -> list[dict]:
    """Raw state of every job: done / older code / unreadable / other protocol / running / failed / interrupted / to run,
    with progress. A result file is checked against its planned job (:func:`_check_record`): ``problem`` says why an
    unusable one cannot be used, ``provenance`` how a usable one differs in device / threads / library versions."""
    code = lab.code_hash() if lab.kind == "tempo" else None
    mtime = _code_mtimes(lab) if lab.kind == "tempo" else ()
    failed, imported, now = _failed_ids(lab), _imported(lab), _installed_versions()
    procs: list = []                                         # the process table, read once and only if a lock file needs it
    rows = []
    for j in jobs:
        jid = bl.job_id(j)
        res = bl.result_path(j, lab.dirs.results)
        lock_pid = bl._lock_pid(lab.dirs.logs / "locks" / f"{jid}.pid")
        ev, final, problem, prov = None, None, "", []
        chk = _inspect(j, res, now) if res.exists() else None
        if chk is not None and chk["state"]:
            state, problem = chk["state"], "; ".join(chk["why"])      # never used; launch() moves it away and re-runs it
        elif chk is not None:
            rec, prov, old = chk["rec"], chk["provenance"], False
            if j["algo"] == "TEMPO" and code is not None:
                src, started = rec.get("tempo_source"), _started_at(rec.get("finished"), rec.get("wall_s"))
                old = _stale_reason(lab, src, started, code, mtime, _started_at(rec.get("finished"), 0),
                                    rec.get("tempo_source_start")) is not None
            state = ("older code" if lab.rerun_outdated else "done (older code)") if old else "done"
            key = (j["env"], j["algo"], j["seed"])
            if state == "done" and key in imported and imported[key] and _sha256(res) == imported[key]:
                state = "done (imported)"                    # only while the file is still the byte copy that was imported
            try:
                final = bl.final_of(res)
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                final = None
        else:
            ev = bl.last_eval(lab.dirs.logs / f"{jid}.log")
            if lock_pid is not None and not procs:
                procs.append(_job_pids())
            if lock_pid is not None and _lock_live(lock_pid, procs[0]):
                state = "running"
            elif jid in failed:
                state = "failed"
            elif ev is not None:
                state = "interrupted"
            else:
                state = "to run"
        frac = 1.0 if state.startswith(("done", "older")) else (min(1.0, ev[0] / j["steps"]) if ev else 0.0)
        rate = ev[0] / ev[2] if ev and ev[2] > 0 and ev[0] > 0 else None
        rows.append(dict(job=j, id=jid, algorithm=report.label_of(j["algo"], j.get("variant", "default")), algo=j["algo"],
                         variant=j.get("variant", "default"), seed=j["seed"], device=j.get("device", "cpu"),
                         status=state, frac=frac, last_step=ev[0] if ev else None, last_eval=ev[1] if ev else None,
                         rate=rate, eta_min=((j["steps"] - ev[0]) / rate / 60) if (rate and state == "running") else None,
                         plan_min=bl.job_seconds(j) / 60, final=final, path=res, problem=problem, provenance=prov))
    return rows


def status_table(lab: Lab, jobs: list[dict]):
    """One row per run (algorithm, variant, seed, device, status, progress, latest evaluation, speed, ETA,
    final score). Status: ``done`` / ``done (imported)`` / ``running`` / ``to run`` / ``failed`` (re-runs at the next
    launch) / ``interrupted`` (a log without result: restarts from step 0) / ``older code`` (TEMPO result of other
    tempo_rl code: re-run; ``done (older code)`` when ``rerun_outdated=False``) / ``unreadable`` (a result file that is
    not valid JSON or whose curve is incomplete) / ``other protocol`` (a result of another budget, evaluation schedule,
    episodes, seed, variant, env_kwargs or, for the baselines, other hyperparameters): these two are never used and the
    next launch moves them (and their logs) to ``_outdated/`` and runs them again. ETA = measured minutes left
    for a running job, ``~N`` = planning-rate estimate of a full run otherwise."""
    import pandas as pd

    out = []
    for r in _rows(lab, jobs):
        done = r["status"].startswith("done")
        eta = "" if done else (f"{r['eta_min']:.0f} (measured)" if r["eta_min"] is not None else f"~{r['plan_min']:.0f} (plan)")
        out.append({"algorithm": r["algorithm"], "variant": r["variant"], "seed": r["seed"], "device": r["device"],
                    "status": r["status"], "progress": f"{r['frac']:.0%}",
                    "last eval (step)": f"{r['last_eval']:.1f} ({r['last_step']:,})" if r["last_step"] else "",
                    "env-steps/s": round(r["rate"], 1) if r["rate"] else "",
                    "ETA (min)": eta, "final": round(r["final"], 2) if r["final"] is not None and not math.isnan(r["final"]) else ""})
    df = pd.DataFrame(out, columns=["algorithm", "variant", "seed", "device", "status", "progress", "last eval (step)",
                                    "env-steps/s", "ETA (min)", "final"])
    return df.set_index(["algorithm", "variant", "seed"]) if len(df) else df


def estimate(lab: Lab, jobs: list[dict], verbose: bool = True) -> dict:
    """Wall-clock estimate of the runs still to do: running jobs keep their slot for their remaining time,
    the others are list-scheduled longest first per device (as ``benchmarks.scheduler`` does) with benchlab's
    planning rates (``TEMPO_RATES`` / ``SB3_RATES``). Returns the numbers; prints them when ``verbose``."""
    rows = _rows(lab, jobs)
    todo = [r for r in rows if r["status"] in _STATES_TODO]
    running = [r for r in rows if r["status"] == "running"]
    caps = {"cpu": lab.cpu_workers, "mps": lab.mps_workers}
    span = {}
    for dev in ("cpu", "mps"):                          # "mps" = the GPU pool (Apple or CUDA: bl.pool)
        dj = [r for r in todo if bl.pool(r["device"]) == dev]
        if dj and caps[dev] <= 0:
            raise ValueError(f"{len(dj)} jobs need the {'CPU' if dev == 'cpu' else 'GPU'} but it has 0 workers: set "
                             f"{'cpu_workers' if dev == 'cpu' else 'mps_workers'} > 0")
        free = [0.0] * max(1, caps[dev])
        for r in running:
            if bl.pool(r["device"]) == dev:
                k = min(range(len(free)), key=free.__getitem__)
                free[k] += bl.job_seconds(r["job"]) * (1 - r["frac"])
        for r in sorted(dj, key=lambda r: -bl.job_seconds(r["job"])):
            k = min(range(len(free)), key=free.__getitem__)
            free[k] += bl.job_seconds(r["job"])
        span[dev] = max(free) / 3600 if (dj or any(bl.pool(r["device"]) == dev for r in running)) else 0.0
    out = dict(n_jobs=len(rows), n_done=sum(r["status"].startswith("done") for r in rows), n_todo=len(todo),
               n_running=len(running), wall_hours=max(span.values()), per_device_hours=span,
               job_hours=sum(bl.job_seconds(r["job"]) for r in todo) / 3600)
    if verbose:
        print(f"{out['n_done']} of {len(rows)} runs finished; {len(running)} running; {len(todo)} to run")
        if todo or running:
            print(f"estimated wall-clock: ~{bl._fmt_hours(out['wall_hours'])} with {lab.cpu_workers} CPU + {lab.mps_workers} "
                  f"Apple-GPU workers (CPU {bl._fmt_hours(span['cpu'])}, GPU {bl._fmt_hours(span['mps'])}; "
                  f"{out['job_hours']:.1f} job-hours), if no other scheduler of this folder holds the workers")
            seen: dict = {}
            for r in todo + running:
                seen.setdefault((r["algorithm"], r["device"]), r["job"])
            print("planning rates per job: " + "; ".join(
                f"{a} ~{bl.job_rate(j):.0f} env-steps/s on the {'CPU' if d == 'cpu' else 'Apple GPU'}"
                for (a, d), j in seen.items()))
            print("Real runs can take up to ~30% longer on a busy machine; watch() shows the measured speeds."
                  if lab.kind == "tempo" else
                  "The planning rates of the baselines are rough: for some (TD3 and CrossQ on Ant / Humanoid, ...) the earlier "
                  "benchmark took up to twice the planned time. watch() shows the measured speeds and ETA.")
            if _cost_note(lab):
                print("NOTE: " + _cost_note(lab))
        else:
            print("nothing left to run")
    return out


def plan(lab: Lab, jobs: list[dict]):
    """Display the :func:`status_table` and print the :func:`estimate`; returns the table. Result files that the next
    :func:`launch` would move to ``_outdated/`` (``unreadable`` / ``other protocol``, and TEMPO ``older code``) are announced
    first, each unusable file with the reason; runs that are kept although their device / threads / library versions
    differ get a NOTE, and so does a figure / tables of this task made with another budget (:func:`_archive_outputs`)."""
    from IPython.display import display

    if not jobs:
        print(f"no runs planned on {lab.env}: " + ("no baselines selected (ALGOS = ())" if lab.kind == "baselines"
                                                   else "empty job list"))
        return status_table(lab, jobs)
    df = status_table(lab, jobs)
    display(df)
    rows = _rows(lab, jobs)
    bad = [r for r in rows if r["status"] in _STATES_BAD]
    busy = (" (if the scheduler of this task is still running, the launch cell only lists them; the next launch after it "
            "has finished moves them)")
    if bad:
        hours = sum(bl.job_seconds(r["job"]) for r in bad) / 3600
        print(f"WARNING: {len(bad)} result file(s) cannot be used and are left out of the results. The launch cell MOVES them, "
              f"with the logs of their runs, to {lab.dirs.results / '_outdated'}/<time>/ (kept, never deleted) and runs them "
              f"again (~{hours:.1f} job-hours){busy}:")
        for r in bad:
            print(f"  {r['algorithm']} seed {r['seed']}: {r['status']}: {r['problem']} ({r['path']})")
    prov = [(r["algorithm"], r["seed"], r["provenance"]) for r in rows if r["provenance"]]
    if prov:
        print(_provenance_note(prov))
    old = [r for r in rows if r["status"] == "older code"] if lab.kind == "tempo" else []
    if old:
        hours = sum(bl.job_seconds(r["job"]) for r in old) / 3600
        print(f"NOTICE: {len(old)} finished TEMPO result(s) were made with other tempo_rl code (or while it was being edited). "
              f"The launch cell MOVES them to {lab.dirs.results / '_outdated'}/<time>/ (kept, never deleted) and runs them again "
              f"(~{hours:.1f} job-hours){busy}. To keep them and run only what is missing, set RERUN_OUTDATED_TEMPO = False in "
              "the Parameters cell and run from there again.")
    stale = _outputs_of_other_protocol(lab)
    if stale:
        print(f"NOTE: the figure / tables of {lab.env} in this folder were made with another protocol than this notebook's "
              f"(BUDGET {lab.budget:,} / EVAL_FREQ {lab.eval_freq:,}, {lab.n_eval} episodes): {_outputs_note(stale)}. The launch "
              f"cell (or else the next figure or table this notebook writes) moves them to {lab.figures / '_outdated'}/<time>/ "
              f"and {lab.tables / '_outdated'}/<time>/ (kept, never deleted).")
    estimate(lab, jobs)
    return df


# =========================================================================== launch, watch, stop, log


def _stale_results(lab: Lab, jobs: list[dict]) -> list:
    """TEMPO only: ``(job, path, reason, tempo_source)`` of every finished result that does not count as made with the live
    ``tempo_rl`` (:func:`_stale_reason`)."""
    if lab.kind != "tempo":
        return []
    code, mtime, out = lab.code_hash(), _code_mtimes(lab), []
    for j in jobs:
        p = bl.result_path(j, lab.dirs.results)
        if j["algo"] != "TEMPO" or not p.exists():
            continue
        chk = _inspect(j, p)
        if chk is None or chk["state"]:                    # unusable files are moved as such (_unusable_results)
            continue
        rec = chk["rec"]
        src = rec.get("tempo_source")
        why = _stale_reason(lab, src, _started_at(rec.get("finished"), rec.get("wall_s")), code, mtime, _started_at(rec.get("finished"), 0),
                            rec.get("tempo_source_start"))
        if why:
            out.append((j, p, why, src))
    return out


def _outdated_dest(lab: Lab, p: pathlib.Path, stamp: str) -> pathlib.Path:
    """``<results>/_outdated/<stamp>/<suite>/<env>/<algo>/<variant>/seed<k>.json`` for the result ``p`` (``seed<k>.<n>.json``
    when that name is taken)."""
    dest = lab.dirs.results / "_outdated" / stamp / p.relative_to(lab.dirs.results)
    k = 1
    while dest.exists():
        dest = dest.with_name(f"{p.stem}.{k}{p.suffix}")
        k += 1
    return dest


def _unusable_results(lab: Lab, jobs: list[dict]) -> list:
    """``(job, path, state, why)`` of every result file at a planned job's path that is ``unreadable`` or of ``other
    protocol`` (:func:`_check_record`); only paths inside this lab's results folder, outside ``_outdated/``."""
    root, now, seen, out = lab.dirs.results.resolve(), _installed_versions(), set(), []
    for j in jobs:
        p = bl.result_path(j, lab.dirs.results)
        if p in seen or not p.exists():
            continue
        seen.add(p)
        try:
            inside = p.resolve().is_relative_to(root) and "_outdated" not in p.resolve().relative_to(root).parts
        except (OSError, ValueError):
            inside = False
        chk = _inspect(j, p, now)
        if chk is not None and chk["state"]:
            if inside:
                out.append((j, p, chk["state"], "; ".join(chk["why"])))
            else:                                       # e.g. a symbolic link to a file elsewhere: never moved from here
                print(f"NOT moved: {p} is {chk['state']} but is not a file inside {lab.dirs.results} (a link?); remove or "
                      "replace it yourself, the scheduler skips a run whose result path exists")
    return out


def _move_log(lab: Lab, job: dict, stamp: str) -> pathlib.Path | None:
    """Move the log of ``job`` (``<logs>/<job-id>.log``, if there is one) to ``<results>/_outdated/<stamp>/logs/``, next to
    the result it belongs to (never deleted; ``<job-id>.<n>.log`` when the name is taken). Left in place, it would make
    :func:`status_table` / :func:`watch` show the re-run that waits in the queue as ``interrupted`` with the old run's
    progress (a 50,000-step log reads 10% of a 500,000-step budget) until the scheduler starts the job. Returns the
    destination, None when there is no log."""
    log = lab.dirs.logs / f"{bl.job_id(job)}.log"
    if not log.is_file():
        return None
    dest = lab.dirs.results / "_outdated" / stamp / "logs" / log.name
    k = 1
    while dest.exists():
        dest = dest.with_name(f"{log.stem}.{k}{log.suffix}")
        k += 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(log), str(dest))
    return dest


def _move_unusable(lab: Lab, bad: list, stamp: str | None = None) -> list:
    """Move the files of :func:`_unusable_results` and the logs of their runs (never delete) to
    ``<results>/_outdated/<stamp>/...`` (the scheme of :func:`archive_outdated`; logs in ``_outdated/<stamp>/logs/``), print
    every move, and mark the import-manifest entries of their paths as moved (under the manifest lock), so that a copy made
    by :func:`import_previous` no longer counts as imported. If the manifest lock cannot be had, the moves stand and a NOTE
    says so (harmless: an entry only counts while the file at its path has its SHA-256). Returns the destinations."""
    if not bad:
        return []
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    moved, gone, logs = [], [], 0
    for j, p, state, why in bad:
        dest = _outdated_dest(lab, p, stamp)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest))
        moved.append(dest)
        gone.append((j, dest))
        log = _move_log(lab, j, stamp)
        logs += log is not None
        print(f"moved {p}\n   -> {dest}\n      ({state}: {why})" + (f"\n      and its log -> {log}" if log else ""))
    hours = sum(bl.job_seconds(j) for j, *_ in bad) / 3600
    print(f"moved {len(moved)} result file(s) that cannot be used" + (f" and {logs} log(s) of their runs" if logs else "")
          + f" to {lab.dirs.results / '_outdated' / stamp} (kept, never deleted); they are run again now at BUDGET "
          f"{lab.budget:,} (~{hours:.1f} job-hours at the planning rates)")
    if _manifest_path(lab).exists():
        try:
            with _manifest_lock(lab):
                entries, changed = _read_manifest(lab), False
                for j, dest in gone:                     # the copy at this path is gone, whatever its bytes were
                    for m in entries:
                        if (not m.get("moved_to") and m.get("env") == j["env"] and m.get("algo") == j["algo"]
                                and str(m.get("seed")) == str(j["seed"]) and j.get("variant", "default") == "default"):
                            m.update(moved_to=str(dest), moved=time.strftime("%Y-%m-%d %H:%M:%S"))
                            changed = True
                if changed:
                    _write_manifest(lab, entries)
        except TimeoutError:
            print(f"NOTE: the {len(moved)} file(s) above ARE moved, but the import manifest {_manifest_path(lab)} could not "
                  "be updated (another notebook of this folder holds its lock). This is harmless: an entry only counts as "
                  "imported while the file at its path has the recorded SHA-256, and these files are gone. The runs start "
                  "now.")
    return moved


def _outputs_of_other_protocol(lab: Lab) -> list:
    """``[(stem, (budget, eval_freq, n_eval), [files])]``: this task's figure and tables, complete (``<task>.*``) and
    provisional (``<task>.partial.*``), whose table (``<tables>/<stem>.json``, ``protocol``) records another budget,
    evaluation frequency or number of episodes than this lab's (e.g. the files of a run of this notebook at the earlier
    50,000-step budget). The files are ``<tables>/<stem>.json|md`` and ``<figures>/<stem>.png|pdf``. A figure without its
    table, or a table that cannot be read, cannot be dated and is left alone (it is overwritten as before)."""
    out = []
    for stem in (lab.env, f"{lab.env}.partial"):
        js = lab.tables / f"{stem}.json"
        try:
            prot = json.loads(js.read_text())["protocol"]
            got = (prot["budget"], prot["eval_freq"], prot.get("n_eval", lab.n_eval))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if got == (lab.budget, lab.eval_freq, lab.n_eval):
            continue
        files = [f for f in (js, lab.tables / f"{stem}.md", lab.figures / f"{stem}.png", lab.figures / f"{stem}.pdf")
                 if f.is_file()]
        out.append((stem, got, files))
    return out


def _outputs_note(stale: list) -> str:
    """'tables/X.json, ..., figures/X.pdf (BUDGET 50,000 / EVAL_FREQ 2,500, 10 episodes)' per stem of
    :func:`_outputs_of_other_protocol` (whatever types the table recorded: never raises)."""
    def num(v) -> str:
        return f"{v:,}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)

    return "; ".join(f"{', '.join(f.parent.name + '/' + f.name for f in files)} (BUDGET {num(b)} / EVAL_FREQ {num(e)}, "
                     f"{num(n)} episodes)" for _, (b, e, n), files in stale)


def _archive_outputs(lab: Lab, stamp: str | None = None) -> list:
    """Move (never delete) the figure and tables of :func:`_outputs_of_other_protocol` to ``<figures>/_outdated/<stamp>/``
    and ``<tables>/_outdated/<stamp>/`` and print what moved. Called by :func:`launch` (with the stamp of the results it
    moves) and by :func:`plot_curves` / :func:`save` before they write, so that a complete figure or table of another
    budget is neither left next to the new runs' files nor overwritten without a copy. Returns the destinations."""
    stale = _outputs_of_other_protocol(lab)
    if not stale:
        return []
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    moved = []
    for _, _, files in stale:
        for f in files:
            dest = f.parent / "_outdated" / stamp / f.name
            k = 1
            while dest.exists():
                dest = dest.with_name(f"{f.stem}.{k}{f.suffix}")
                k += 1
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest))
            moved.append(dest)
    print(f"moved the figure / tables of {lab.env} made with another protocol than this notebook's (BUDGET {lab.budget:,} / "
          f"EVAL_FREQ {lab.eval_freq:,}, {lab.n_eval} episodes): {_outputs_note(stale)} -> "
          f"{lab.figures / '_outdated' / stamp} and {lab.tables / '_outdated' / stamp} (kept, never deleted)")
    return moved


def archive_outdated(lab: Lab, jobs: list[dict], dry_run: bool = False, stamp: str | None = None) -> list:
    """TEMPO only. Move TEMPO results that do not count as made with the live ``tempo_rl`` (``tempo_source`` != the live
    hash, or ``tempo_rl`` changed while the run was going, between its start and its end: :func:`_stale_reason`), and the logs of their runs, to
    ``<results>/_outdated/<stamp>/<suite>/...`` (logs: ``_outdated/<stamp>/logs/``; ``stamp`` default: now) so that the
    scheduler re-runs them. Nothing is deleted; ``dry_run`` only lists. Returns the destination (or, dry run, the source)
    paths of the results. (Unusable files, ``unreadable`` / ``other protocol``, are moved by :func:`launch` as such.)"""
    if lab.kind != "tempo":
        return []
    code, stamp, moved, logs = lab.code_hash(), stamp or time.strftime("%Y%m%d-%H%M%S"), [], 0
    stale = _stale_results(lab, jobs)
    for j, p, why, src in stale:
        dest = _outdated_dest(lab, p, stamp)
        if dry_run:
            moved.append(p)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest))
        moved.append(dest)
        logs += _move_log(lab, j, stamp) is not None
    if moved:
        other = sorted({str(src) for _, _, why, src in stale if why == "other code"})
        edited = sum(why != "other code" for _, _, why, _ in stale)
        hours = sum(bl.job_seconds(j) for j, *_ in stale) / 3600
        print(("would move " if dry_run else "moved ") + f"{len(moved)} TEMPO result(s) made with older tempo_rl code "
              + (f"(source {', '.join(other)} -> current {code}" if other else f"(current source {code}")
              + (f"; {edited} of them overlapped an edit of tempo_rl" if edited else "")
              + ")" + (f" and {logs} log(s) of their runs" if logs else "") + f" to {lab.dirs.results / '_outdated' / stamp}"
              + ("" if dry_run else f" (kept, never deleted; ~{hours:.1f} job-hours are re-run now; "
                 "RERUN_OUTDATED_TEMPO = False keeps such results instead)"))
    return moved


@contextlib.contextmanager
def _launch_lock(lab: Lab, enabled: bool = True, timeout: float = 60.0):
    """Serialise the 'is a scheduler running? / start one' step of all launches in one logs folder (an exclusive ``flock`` on
    ``<logs>/.launch.lock``). Two schedulers that start at the same instant would each count the other's jobs as not yet
    running and exceed the worker limits; one launch lasts about three seconds."""
    if not enabled:
        yield
        return
    import fcntl

    lab.dirs.logs.mkdir(parents=True, exist_ok=True)
    fh = open(lab.dirs.logs / ".launch.lock", "a")
    locked = False
    try:
        t0 = time.time()
        while not locked:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError:
                if time.time() - t0 > timeout:
                    print("(another launch of this folder is taking long: continuing without waiting for it)")
                    break
                time.sleep(0.2)
        yield
    finally:
        try:
            if locked:
                fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()


def _other_folders_running(lab: Lab) -> dict:
    """``{folder: {device: n}}`` of live jobs in the lock directories of the OTHER run folders (Baselines_test, TEMPO_test,
    the main benchmark's ``benchmarks/logs``), whose worker limits do not count toward this lab's."""
    procs = _job_pids()
    if procs is None:
        return {}
    own = (lab.dirs.logs / "locks").resolve()
    out: dict = {}
    for name, logs in (("Baselines_test", ROOT / "Baselines_test" / "logs"), ("TEMPO_test", ROOT / "TEMPO_test" / "logs"),
                       ("benchmarks", BENCH / "logs")):
        d = logs / "locks"
        if not d.is_dir() or d.resolve() == own:
            continue
        for f in d.glob("*.pid"):
            try:
                parts = f.read_text().split()
                pid = int(parts[0])
            except (OSError, ValueError, IndexError):
                continue
            if pid in procs:
                dev = parts[1] if len(parts) > 1 else "cpu"
                out.setdefault(name, {})
                out[name][dev] = out[name].get(dev, 0) + 1
    return out


def _warn_other_folders(lab: Lab) -> None:
    if lab.smoke:                                       # smoke runs are tiny and live in their own scratch folder
        return
    other = _other_folders_running(lab)
    if not other:
        return
    cpu = sum(d.get("cpu", 0) for d in other.values())
    gpu = sum(d.get("mps", 0) for d in other.values())
    what = "; ".join(f"{k}: " + ", ".join(f"{n} {'GPU' if dev == 'mps' else 'CPU'}" for dev, n in sorted(d.items()))
                     for k, d in other.items())
    print(f"WARNING: jobs of other run folders are running right now ({what}). The worker limits count per folder, so together "
          f"with this one's limits ({lab.cpu_workers} CPU + {lab.mps_workers} Apple-GPU) up to {cpu + lab.cpu_workers} CPU + "
          f"{gpu + lab.mps_workers} GPU jobs can run at once: every run is slower and the wall-clock hours stored in the "
          "results are not comparable with runs made on an idle machine.")


def _note_job_list(lab: Lab, running: dict, jobs: list[dict]) -> None:
    """Tell the user when the scheduler that is already running works on a job list other than this notebook's (e.g.
    ``ALGOS``, ``SEEDS`` or ``TEMPO_VARIANT`` were changed after launching)."""
    jf = _jobs_file_of(running)
    try:
        old = {bl.job_id(j) for j in json.loads(jf.read_text())} if jf else None
    except (OSError, ValueError, KeyError, TypeError):
        old = None
    new = {bl.job_id(j) for j in jobs}
    if old is not None and old != new:
        extra = sorted(new - old)
        print(f"WARNING: the scheduler that is running works on a DIFFERENT job list than this notebook ({len(old)} jobs; this "
              f"notebook has {len(new)}" + (f", {len(extra)} of them not in it, e.g. {extra[0]}" if extra else "") + "). Settings "
              "(ALGOS, SEEDS, TEMPO_VARIANT) changed after the launch only affect a new scheduler: wait until it has finished "
              "(watch()) or stop() it, then run the launch cell again. One task runs one job list at a time (the scheduler "
              "label is the task, not the variant).")


#: Processes whose load would end up in the wall-clock time of benchmark runs (which the comparison's rank score uses): the
#: baseline hyperparameter search of notebooks/tunelab.py and the timing measurements of Comparison/timelab.py.
_COMPETING_MARKS = ("benchmarks.tune_one", "tunelab.py _pool", "timelab.py _measure")


def _competing_processes() -> list[tuple[int, str]]:
    """Running processes of :data:`_COMPETING_MARKS`: ``[(pid, command)]``."""
    import subprocess

    ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    out = []
    for line in ps.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        if pid.isdigit() and os.path.basename(cmd.split()[0] if cmd.split() else "").startswith("python") \
                and any(m in cmd for m in _COMPETING_MARKS):
            out.append((int(pid), cmd.strip()[:140]))
    return out


def launch(lab: Lab, jobs: list[dict], background: bool = True):
    """Start this notebook's scheduler (``python -m benchmarks.scheduler <label> --jobs-file ...``).

    Safe to re-run: finished jobs are skipped, jobs another scheduler of this folder is running are left to
    it, and a second scheduler for the same label is never started while one is alive (the existing one is
    reported, with a warning when it works on a different job list). Lock files of dead or recycled process ids
    are removed first. Result files of the planned jobs that are ``unreadable`` or of ``other protocol`` are moved
    (never deleted) to ``<results>/_outdated/<time>/``, with the logs of their runs (``_outdated/<time>/logs/``; every move
    printed), and run again, whatever ``rerun_outdated`` says. TEMPO: results of other ``tempo_rl`` code are first moved to
    ``_outdated/`` (with their logs) when ``rerun_outdated``. A figure / tables of this task made with another budget are
    moved to ``<figures|tables>/_outdated/<time>/`` (:func:`_archive_outputs`). None of these moves happens while the
    scheduler of this task is still running: then the cell only lists the unusable files, and the next launch after that
    scheduler has finished moves them. Jobs of other run folders that are running are reported (worker limits are per folder).
    ``background=True`` (default) detaches the scheduler (it survives the kernel and the
    notebook; output in ``<logs>/<label>.out``); ``False`` runs it in the foreground and streams its output.
    Returns the scheduler info dict, the exit code (foreground) or ``None`` (nothing to launch)."""
    if not jobs:
        print("no runs planned: nothing to launch")
        return None
    competing = [] if lab.smoke else _competing_processes()
    if competing:
        print("Not launched: a baseline hyperparameter search (notebooks/tunelab.py) or a timing measurement "
              "(Comparison/timelab.py) is running. Runs started next to it would carry its load in their wall-clock time, "
              "which the comparison's rank score uses. Run this cell again when it has finished:")
        for pid, cmd in competing[:6]:
            print(f"  pid {pid}: {cmd}")
        return None
    if lab.kind == "tempo" and (BENCH / "snapshots" / lab.suite / "tempo_rl").is_dir():
        raise RuntimeError(f"benchmarks/snapshots/{lab.suite}/tempo_rl exists: benchmarks.run_one would run that frozen "
                           "copy instead of the live library. TEMPO_test never freezes code; remove or rename that folder.")
    with _launch_lock(lab, background):
        running = _scheduler_info(lab)
        if running:
            _note_job_list(lab, running, jobs)
            print(f"scheduler '{lab.label}' is already running (pid {running['pid']}, started {running.get('started', '?')}): "
                  "not starting another one; use watch() to follow it")
            bad = _unusable_results(lab, jobs)
            if bad:
                print(f"{len(bad)} result file(s) cannot be used ({', '.join(sorted({s for _, _, s, _ in bad}))}; see the plan): "
                      "they are left out of the results and are moved and run again by the next launch after this "
                      "scheduler has finished")
            return running
        _clean_stale_locks(lab)
        bad = _unusable_results(lab, jobs)
        bad_ids = {bl.job_id(j) for j, *_ in bad}
        stale = _stale_results(lab, jobs) if lab.kind == "tempo" and lab.rerun_outdated else []
        stale_ids = {bl.job_id(j) for j, *_ in stale}
        todo = [j for j in jobs if bl.job_id(j) in stale_ids | bad_ids or not bl.result_path(j, lab.dirs.results).exists()]
        for dev, cap in (("cpu", lab.cpu_workers), ("mps", lab.mps_workers)):          # before anything is moved
            n = sum(j.get("device", "cpu") == dev for j in todo)
            if n and cap <= 0:
                raise ValueError(f"{n} jobs need the {dev.upper()} but it has 0 workers")
        stamp = time.strftime("%Y%m%d-%H%M%S")              # one _outdated/<stamp>/ for everything this launch moves
        moved = _move_unusable(lab, bad, stamp)
        moved += archive_outdated(lab, jobs, stamp=stamp) if stale else []
        _archive_outputs(lab, stamp)                        # a figure / tables of another BUDGET: not left next to the new runs
        _warn_other_folders(lab)
        with _scoped(lab):
            out = bl.launch(lab.label, jobs, lab.dirs, lab.cpu_workers, lab.mps_workers, background=background,
                            hint="Use watch() to follow the progress.")
    if isinstance(out, int) and out != 0 and moved:
        print(f"The scheduler failed after {len(moved)} result file(s) were moved into {lab.dirs.results / '_outdated'}: they "
              "are all still there (nothing is deleted); fix the problem above and run the launch cell again.")
    elif isinstance(out, int) and out == 0 and background:
        procs = _job_pids()
        held = [bl.job_id(j) for j in todo if _lock_live(bl._lock_pid(lab.dirs.logs / "locks" / f"{bl.job_id(j)}.pid"), procs)]
        if held:
            print(f"Exit code 0 is not an error here: the scheduler found nothing to start, because {len(held)} job(s) are already "
                  "running in other processes (e.g. a scheduler that was killed). watch() follows them; stop() ends them.")
    return out


def _running_now(lab: Lab, jobs: list[dict]) -> int:
    """Number of jobs of the list that have a live process (cheap: lock files and one process-table read)."""
    procs = _job_pids()
    n = 0
    for j in jobs:
        pid = bl._lock_pid(lab.dirs.logs / "locks" / f"{bl.job_id(j)}.pid")
        if pid is not None and not bl.result_path(j, lab.dirs.results).exists() and _lock_live(pid, procs):
            n += 1
    return n


def watch(lab: Lab, jobs: list[dict], every: float | None = None):
    """Show the progress table every ``every`` seconds (default 10 in smoke mode, else 120) until this lab's
    scheduler has exited and no job of the list is running, i.e. until every run is done or has failed (the end is
    noticed within five seconds, not at the next refresh).
    Interrupting (the stop button) only stops watching: the runs continue; run it again to resume.
    Failed runs are listed with the last lines of their logs. TEMPO: warns when ``tempo_rl`` is edited while watching.
    Returns ``None`` (the table is displayed)."""
    from IPython.display import clear_output, display

    if not jobs:
        print(f"no runs planned on {lab.env}: nothing to watch")
        return None
    every = (10 if lab.smoke else 120) if every is None else every
    df = status_table(lab, jobs)
    code0 = lab.code_hash() if lab.kind == "tempo" else None
    try:
        while True:
            df = status_table(lab, jobs)
            sched = _scheduler_info(lab)
            n_run = int((df.status == "running").sum()) if len(df) else 0
            clear_output(wait=True)
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {lab.env}: scheduler '{lab.label}' "
                  + (f"RUNNING (pid {sched['pid']})" if sched else "not running")
                  + f"; output: {lab.dirs.logs / (lab.label + '.out')}")
            if code0 is not None and lab.code_hash() != code0:
                print("WARNING: tempo_rl was edited while this notebook is watching. Runs that are already going keep the code they "
                      "imported, but a result is stamped with the source at the moment it is written; results whose run "
                      "overlapped an edit are shown as 'older code' and re-run by the next launch.")
            if len(df):
                print("jobs: " + ", ".join(f"{k} {v}" for k, v in df.status.value_counts().items()))
                display(df)
            if not sched and n_run == 0:
                break
            print(f"refreshing every {every:g} s; interrupt to stop watching (the runs continue)")
            t_end = time.time() + every
            while t_end - time.time() > 0:
                time.sleep(min(5.0, max(0.0, t_end - time.time())))
                if time.time() < t_end and not _scheduler_info(lab) and not _running_now(lab, jobs):
                    break                               # everything ended: redraw now instead of at the next refresh
    except KeyboardInterrupt:
        print("Stopped watching. The runs continue in the background; run this cell again to resume watching.")
        return None
    left = df[~df.status.str.startswith("done")] if len(df) else df
    if len(df) and left.empty:
        print(f"all {len(df)} runs of {lab.env} are finished.")
    elif len(left):
        print(f"{len(left)} run(s) without a result:")
        for (alg, var, seed), row in left.iterrows():
            print(f"  {alg} seed {seed}: {row.status}")
        if left.status.isin(list(_STATES_TODO)).any():
            print("No scheduler is running, so nothing will start these runs by itself: run the launch cell again (failed and "
                  "interrupted runs restart from step 0; unreadable results and results of another protocol are moved to "
                  "_outdated/ first).")
        for j in jobs:
            if not bl.result_path(j, lab.dirs.results).exists() and (lab.dirs.logs / f"{bl.job_id(j)}.log").exists():
                print(f"\n--- last lines of {bl.job_id(j)}.log")
                print(bl.tail(lab.dirs.logs / f"{bl.job_id(j)}.log", 12))


def stop(lab: Lab, jobs: list[dict]) -> None:
    """Stop this notebook's scheduler and the runs it started (plus orphaned runs of these jobs left by a
    killed scheduler). Finished results are kept; interrupted runs restart from step 0 at the next launch.
    Lock files of these jobs whose process id was recycled by an unrelated process are removed as well."""
    with _scoped(lab):
        bl.stop(lab.label, lab.dirs, jobs, only_orphans=True)
    _clean_stale_locks(lab, jobs)


def _log_text(path: pathlib.Path, n: int) -> str:
    """Last ``n`` lines of a log, without the ``objc[pid]: Class ... is implemented in both ...`` warnings that macOS prints when
    pygame's and OpenCV's copies of the SDL library are loaded into one process (a Box2D task whose run imports
    Stable-Baselines3 first); they are counted, not hidden silently."""
    try:
        lines = pathlib.Path(path).read_text(errors="replace").splitlines()
    except FileNotFoundError:
        return f"(no file {path})"
    keep = [ln for ln in lines if not ln.startswith("objc[")]
    text = "\n".join(keep[-n:])
    if len(keep) < len(lines):
        text += (f"\n({len(lines) - len(keep)} line(s) 'objc[pid]: Class ... is implemented in both ...' hidden: two copies of the "
                 "SDL library are loaded in the run's process; the benchmark opens no window, so it does not affect the run)")
    return text


def show_log(lab: Lab, jobs: list[dict], algo: str | None = None, seed: int | None = None, n: int = 30) -> str:
    """Print (and return) the last ``n`` lines of the log of the runs matching ``algo`` (algorithm name or table
    label) and ``seed``; without either, of the running jobs, else of the most recently written log."""
    pick = [j for j in jobs if (algo is None or algo in (j["algo"], report.label_of(j["algo"], j.get("variant", "default"))))
            and (seed is None or j["seed"] == seed)]
    if algo is None and seed is None:
        state = {r["id"]: r["status"] for r in _rows(lab, jobs)}
        run = [j for j in pick if state[bl.job_id(j)] == "running"]
        logs = [j for j in pick if (lab.dirs.logs / f"{bl.job_id(j)}.log").exists()]
        pick = run or sorted(logs, key=lambda j: (lab.dirs.logs / f"{bl.job_id(j)}.log").stat().st_mtime)[-1:]
    if not pick:
        print("no log: no matching run has been started (launch() first)")
        return ""
    text = []
    for j in pick:
        f = lab.dirs.logs / f"{bl.job_id(j)}.log"
        text.append(f"--- {f} (last {n} lines)\n{_log_text(f, n)}")
    out = "\n\n".join(text)
    print(out)
    return out


# =========================================================================== references (toy text)

_REF_CACHE: dict = {}


def _random_policy_eval(env_id: str, n_eval: int) -> float:
    """Mean return of the uniform-random policy on the benchmark's held-out evaluation episodes (reset seed
    100000 + i), played by ``benchmarks.run_one.evaluate`` itself; the action stream is seeded like
    ``benchmarks/random_scores.py`` (``action_space.seed(0)``)."""
    import numpy as np

    from benchmarks.run_one import evaluate
    from tempo_rl.env_util import make_env

    kw = plans.ENV_KWARGS.get(env_id)
    space = make_env(env_id, kw).action_space
    space.seed(0)
    rets, _ = evaluate(lambda obs: space.sample(), env_id, int(n_eval), lambda: make_env(env_id, kw))
    return float(np.mean(rets))


def reference_scores(lab: Lab, write: bool | None = None) -> dict:
    """Reference scores of the task on the benchmark's held-out evaluation episodes (episode i reset with seed
    100000 + i, ``lab.n_eval`` episodes, exactly like every evaluation of a run):

    ``random`` (benchmarks/random_scores.json, 3 x n_eval episodes: 30, Blackjack 3,000), ``random_policy_eval`` (random policy on the
    evaluation episodes), ``threshold`` (gymnasium solved score), and for toy text ``optimum`` (exact, value
    iteration within the time limit) and ``optimal_policy_eval`` (the optimal policy on the evaluation
    episodes: on stochastic FrozenLake a run's final score is a success rate over these fixed episodes, so even
    an optimal policy can score above or below the exact expectation).

    ``write`` (default: toy text only) stores them in ``<results>/<suite>/<env>/reference.json``, which
    ``benchmarks.report.load_suite`` ignores. Works with zero runs."""
    info = lab.info
    toy = info["toy_text"]
    ref = dict(env=lab.env, n_eval=lab.n_eval, eval_seed_base=100_000, random=info["random"], threshold=info["threshold"],
               optimum=None, optimal_policy_eval=None, random_policy_eval=_random_policy_eval(lab.env, lab.n_eval))
    if toy:
        ref["optimum"] = bl.toy_text_optimum(lab.env)
        ref["optimal_policy_eval"] = bl.toy_text_optimal_eval(lab.env, lab.n_eval)
    ref["generated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if write is None:
        write = toy
    if write:
        lab.task_dir.mkdir(parents=True, exist_ok=True)
        f = lab.task_dir / "reference.json"
        f.write_text(json.dumps(bl._json_clean(ref), indent=1))
        print(f"wrote {f}")
    _REF_CACHE[(lab.env, lab.n_eval)] = ref
    return ref


def references(lab: Lab) -> dict:
    """The reference scores for summaries and figures. Toy text: ``reference.json`` of the results folder if it
    matches, else computed in memory (not written); other tasks: random policy + threshold from the files
    (no episode is played)."""
    info = lab.info
    if not info["toy_text"]:
        return dict(env=lab.env, n_eval=lab.n_eval, random=info["random"], threshold=info["threshold"], optimum=None,
                    optimal_policy_eval=None, random_policy_eval=None)
    f = lab.task_dir / "reference.json"
    try:
        ref = json.loads(f.read_text())
        if ref.get("n_eval") == lab.n_eval and ref.get("optimum") is not None:
            return ref
    except (OSError, ValueError):
        pass
    key = (lab.env, lab.n_eval)
    return _REF_CACHE[key] if key in _REF_CACHE else reference_scores(lab, write=False)


# =========================================================================== summary, figure, tables


def _n_imported(lab: Lab, runs: dict) -> int:
    """How many of the runs are still the byte copies :func:`import_previous` made (manifest + SHA-256)."""
    imp = _imported(lab)
    if not imp:
        return 0
    return sum(1 for rs in runs.values() for r in rs
               if imp.get((r.env, r.algo, r.seed)) and _sha256(r.path) == imp[(r.env, r.algo, r.seed)])


def _ci(m, lo, hi) -> str:
    return "-" if m is None or (isinstance(m, float) and math.isnan(m)) else f"[{bl._fmt(lo)}, {bl._fmt(hi)}]"


def _run_of(f: pathlib.Path, r: dict):
    """The ``benchmarks.report.Run`` of one parsed result file, built with exactly the expressions of
    ``report.load_suite`` (None where load_suite skips the file: no evaluation at or before the budget)."""
    import numpy as np

    j = r["job"]
    steps = np.asarray(r["steps"], int)
    keep = steps <= int(j["steps"])
    if not keep.any():
        return None
    return report.Run(env=j["env"], algo=j["algo"], variant=j.get("variant", "default"), seed=int(j["seed"]),
                      budget=int(j["steps"]), eval_freq=int(j["eval_freq"]), n_eval=int(j["n_eval"]),
                      steps=steps[keep], episodes=[r["returns"][i] for i in np.flatnonzero(keep)],
                      n_dropped=int((~keep).sum()), raw_last_step=int(steps[-1]),
                      wall_h=float(r.get("wall_s", np.nan)) / 3600, device=str(j.get("device", "cpu")),
                      path=f, info=r.get("info") or {}, tempo_source=r.get("tempo_source", ""),
                      code_root=r.get("code_root", ""), versions=r.get("versions") or {},
                      finished=str(r.get("finished", "")))


def _load_task(lab: Lab, skip=frozenset()) -> tuple[dict, list, list]:
    """``({label: [Run sorted by seed]}, [(path, reason)], [(path, label, seed, run path)])``: the runs of this task's result
    files (``<results>/<suite>/<env>/*/*/seed*.json``) as ``benchmarks.report.load_suite`` builds them (:func:`_run_of`),
    leaving out ``skip``. A file that cannot be read or turned into a run is listed with the reason (second list) instead of
    stopping the summary (``load_suite`` raises on the first bad file of the whole suite, without naming it). A file that is
    not at the path of the run its record describes (``<suite>/<env>/<algo>/<variant>/seed<k>.json`` of the record's job:
    a Finder copy ``seed10 copy.json``, a record of seed 10 renamed ``seed99.json``, a record of another task) is never a run
    (third list, with the path its record belongs at): it would add a duplicate seed or a run of another protocol that no
    check sees."""
    out: dict = {}
    bad, misplaced = [], []
    for f in sorted(lab.task_dir.glob("*/*/seed*.json")):
        if f in skip:
            continue
        try:
            rec = json.loads(f.read_text())
            j = rec["job"]
            home = (lab.suite_dir / str(j["env"]) / str(j["algo"]) / str(j.get("variant", "default"))
                    / f"seed{int(j['seed'])}.json")
            if f != home:
                misplaced.append((f, report.label_of(str(j["algo"]), str(j.get("variant", "default"))), int(j["seed"]), home))
                continue
            run = _run_of(f, rec)
            if run is not None:
                run.final, run.auc                      # noqa: B018 - the scores must be computable
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as e:
            bad.append((f, f"{type(e).__name__}: {e}"))
            continue
        if run is not None and run.env == lab.env:
            out.setdefault(run.label, []).append(run)
    for rs in out.values():
        rs.sort(key=lambda x: x.seed)
    return dict(sorted(out.items(), key=lambda kv: report.label_key(kv[0]))), bad, misplaced


def summary(lab: Lab, jobs: list[dict] | None = None, n_boot: int = N_BOOT) -> dict:
    """Statistics of every finished run of this task in ``<results>/<suite>`` (works with zero, partial or
    failed runs; ``jobs`` lets the table say running / failed instead of pending).

    Scores are the project's (``benchmarks.report``): final = mean return of the last three evaluations at or
    before the budget (30 episodes); AUC = mean over all evaluations; per algorithm mean and 95% percentile-
    bootstrap CI over seeds (``boot_mean_ci``). Algorithms found on disk but not selected are included. TEMPO:
    results of other ``tempo_rl`` code are left out of the statistics when ``rerun_outdated`` (shown as ``older
    code``). Baselines: a ranking by mean final score, normalised as (x - random)/(best - random).

    The result file of every planned run (``jobs``, default :func:`make_jobs`) is checked first (:func:`_check_record`),
    and so is every other result file of the task that would enter the statistics (seeds of the lab, another algorithm or
    TEMPO variant), against the job the lab would plan for it (:func:`_reference_job`, the lab's budget): an
    ``unreadable`` one or one of ``other protocol`` (e.g. of the earlier 50,000-step budget) is left out with a WARNING
    naming the file and what differs (the summary is then provisional); one whose device / threads / library versions
    differ is used, with a NOTE. A file that is not at the path of the run its record describes (``seed10 copy.json``, a
    seed-10 record renamed ``seed99.json``: state ``misplaced``, :func:`_load_task`) is never used either, with a WARNING;
    it is not a planned run's file, so it does not make the summary provisional and :func:`launch` does not move it. A file
    that cannot be read never stops the summary.

    Returns a dict: ``labels``, ``runs`` {label: [Run]}, ``stats`` {label: ...}, ``per_seed`` (DataFrame),
    ``ranking`` (DataFrame | None), ``references``, ``reference_table`` (DataFrame), ``pending`` / ``failed``
    [(label, seed)], ``notes``, ``excluded`` / ``provenance`` (the checks above, one dict per file; ``state`` of an
    excluded file: ``unreadable``, ``other protocol`` or ``misplaced``), plus ``env``,
    ``kind``, ``suite``, ``seeds``, ``smoke``, ``budget``, ``code_hash``."""
    import numpy as np
    import pandas as pd

    planned = list(jobs) if jobs is not None else make_jobs(lab)
    refs = {}                                               # result path -> (reference job, is it a planned run?)
    for j in planned:
        refs.setdefault(bl.result_path(j, lab.dirs.results), (j, True))
    ok_algos = ("TEMPO",) if lab.kind == "tempo" else report.BASELINES
    for f in sorted(lab.task_dir.glob("*/*/seed*.json")):   # files on disk that would enter the statistics but are not planned
        m = re.fullmatch(r"seed(\d+)", f.stem)
        algo, variant = f.parent.parent.name, f.parent.name
        if f not in refs and m and int(m.group(1)) in lab.seeds and algo in ok_algos:
            refs[f] = (_reference_job(lab, algo, variant, int(m.group(1))), False)
    now, checks = _installed_versions(), {}
    for p, (j, is_planned) in refs.items():                 # every such file checked once against its (reference) job
        chk = _inspect(j, p, now)
        if chk is not None:
            checks[p] = (j, chk, is_planned)
    excluded = [dict(path=str(p), algorithm=report.label_of(j["algo"], j.get("variant", "default")), seed=j["seed"],
                     state=c["state"], why=c["why"], planned=pl) for p, (j, c, pl) in checks.items() if c["state"]]
    provenance = [dict(path=str(p), algorithm=report.label_of(j["algo"], j.get("variant", "default")), seed=j["seed"],
                       differences=c["provenance"]) for p, (j, c, _) in checks.items() if not c["state"] and c["provenance"]]
    data, unreadable, misplaced = _load_task(lab, skip={p for p, (_, c, _) in checks.items() if c["state"]})
    excluded += [dict(path=str(f), algorithm=l, seed=k, state="misplaced", planned=False,
                      why=[f"its record is the run {home.relative_to(lab.dirs.results)}, not this file (a copy or a renamed "
                           "file?)"]) for f, l, k, home in misplaced]
    code = lab.code_hash()
    mtime = _code_mtimes(lab) if lab.kind == "tempo" else ()
    own = lab.labels
    ok_base = ("TEMPO",) if lab.kind == "tempo" else report.BASELINES          # other variants / algorithms on disk
    labels = own + sorted((l for l in data if l not in own and l.split("[")[0] in ok_base), key=report.label_key)
    rows = _rows(lab, jobs) if jobs else []
    state = {(r["algorithm"], r["seed"]): r for r in rows}
    runs, outdated, notes = {}, [], []
    for l in labels:
        rs = [r for r in data.get(l, []) if r.seed in lab.seeds]
        if lab.kind == "tempo" and l == lab.tempo_label:
            old = [r for r in rs if _stale_reason(lab, r.tempo_source, _started_at(r.finished, r.wall_h * 3600), code, mtime,
                                                  _started_at(r.finished, 0))]
            outdated += old
            if lab.rerun_outdated:
                gone = {id(r) for r in old}
                rs = [r for r in rs if id(r) not in gone]
        runs[l] = rs
    ref = references(lab)
    rnd, thr, opt = ref.get("random"), ref.get("threshold"), ref.get("optimum")
    stats = {}
    for l in labels:
        if not runs[l]:
            continue
        finals, aucs = [r.final for r in runs[l]], [r.auc for r in runs[l]]
        stats[l] = dict(finals=finals, aucs=aucs, seeds=[r.seed for r in runs[l]],
                        final=report.boot_mean_ci(finals, ("final", lab.env, l), n_boot),
                        auc=report.boot_mean_ci(aucs, ("auc", lab.env, l), n_boot),
                        std=float(np.std(finals, ddof=1)) if len(finals) > 1 else math.nan,
                        hours=float(np.nanmean([r.wall_h for r in runs[l]])), budget=min(r.budget for r in runs[l]),
                        devices=sorted({r.device for r in runs[l]}), tempo_sources=sorted({r.tempo_source for r in runs[l]}),
                        solved=(int(sum(f >= thr for f in finals)) if thr is not None else None))
    # ---- per-seed table
    old_seeds = {(r.label, r.seed) for r in outdated} if lab.rerun_outdated else set()
    bad_seeds = {(e["algorithm"], e["seed"]): e["state"] for e in excluded if e["state"] in _STATES_BAD}
    table, pending, failed = [], [], []
    for l in labels:
        s = stats.get(l)
        by_seed = dict(zip(s["seeds"], s["finals"])) if s else {}
        row = {"algorithm": l}
        for k in lab.seeds:
            if k in by_seed:
                row[f"seed {k}"] = bl._fmt(by_seed[k])
            elif (l, k) in bad_seeds:                       # left out: launch() moves the file away and re-runs it
                row[f"seed {k}"] = bad_seeds[(l, k)]
            elif (l, k) in old_seeds:
                row[f"seed {k}"] = "older code"
            else:
                st = state.get((l, k))
                if st and st["status"] == "running":
                    row[f"seed {k}"] = f"running {st['frac']:.0%}"
                elif st and st["status"] == "failed":
                    row[f"seed {k}"] = "failed"
                    failed.append((l, k))
                else:
                    row[f"seed {k}"] = "pending"
                if l in own:
                    pending.append((l, k))
        if s:
            m, lo, hi = s["final"]
            am, alo, ahi = s["auc"]
            row.update({"n": len(s["finals"]), "mean final": bl._fmt(m), "95% CI": _ci(m, lo, hi), "AUC": bl._fmt(am),
                        "AUC 95% CI": _ci(am, alo, ahi), "h / run": round(s["hours"], 3)})
            if thr is not None:
                row["seeds >= threshold"] = f"{s['solved']}/{len(s['finals'])}"
            if opt is not None and rnd is not None and opt != rnd:
                row["% random to optimum"] = f"{(m - rnd) / (opt - rnd):.0%}"
        else:
            row.update({"n": 0, "mean final": "-", "95% CI": "-", "AUC": "-", "AUC 95% CI": "-", "h / run": "-"})
            if thr is not None:
                row["seeds >= threshold"] = "-"
            if opt is not None and rnd is not None and opt != rnd:
                row["% random to optimum"] = "-"
        table.append(row)
    if table:
        per_seed = pd.DataFrame(table).set_index("algorithm")
    else:                                                   # no baselines selected (ALGOS = ()) and no runs: references only
        per_seed = pd.DataFrame(columns=["algorithm", *[f"seed {k}" for k in lab.seeds], "n", "mean final", "95% CI", "AUC",
                                         "AUC 95% CI", "h / run"]).set_index("algorithm")
    # ---- baselines: ranking by mean final score
    ranking = None
    if lab.kind == "baselines" and stats:
        order = sorted(stats, key=lambda l: -stats[l]["final"][0])
        best = stats[order[0]]["final"][0]
        ok = rnd is not None and (best - rnd) > 1e-6 * max(1.0, abs(rnd))
        rk = []
        for i, l in enumerate(order, 1):
            s = stats[l]
            m, lo, hi = s["final"]
            am, alo, ahi = s["auc"]
            tuned = next((r.info.get("zoo_tuned") for r in runs[l] if "zoo_tuned" in r.info), None)
            rk.append({"rank": i, "algorithm": l, "n": len(s["finals"]), "mean final": bl._fmt(m), "95% CI": _ci(m, lo, hi),
                       "std": bl._fmt(s["std"]), "AUC": bl._fmt(am), "AUC 95% CI": _ci(am, alo, ahi),
                       "normalised": f"{(m - rnd) / (best - rnd):.2f}" if ok else "-",
                       "hyperparameters": "-" if tuned is None else ("tuned" if tuned else "untuned (defaults)"),
                       "h / run": round(s["hours"], 3)})
        ranking = pd.DataFrame(rk).set_index("rank")
    # ---- references
    rt = [(f"random policy (benchmarks/random_scores.json, {3 * n_eval_for(lab.env):,} episodes)", rnd)]
    if ref.get("random_policy_eval") is not None:
        rt.append((f"random policy on the {ref['n_eval']} evaluation episodes", ref["random_policy_eval"]))
    if thr is not None:
        rt.append(("gymnasium reward_threshold (solved)", thr))
    if opt is not None:
        rt.append((f"known optimum (exact, {_optimum_method(lab.env)})", opt))
        rt.append((f"optimal policy on the {ref['n_eval']} evaluation episodes", ref["optimal_policy_eval"]))
    ref_table = pd.DataFrame([dict(reference=k, score=bl._fmt(v)) for k, v in rt if v is not None]).set_index("reference")
    # ---- notes
    short = {l: len(stats[l]["finals"]) for l in own if l in stats and len(stats[l]["finals"]) < len(lab.seeds)}
    missing = [l for l in own if l not in stats]
    if missing:
        notes.append(f"no finished run yet: {', '.join(missing)}")
    if short:
        notes.append("provisional: fewer finished seeds than requested for " + ", ".join(f"{l} (n={n})" for l, n in short.items()))
    other, edited = [r for r in outdated if r.tempo_source != code], [r for r in outdated if r.tempo_source == code]
    verdict = "left out and re-run by launch()" if lab.rerun_outdated else "INCLUDED (rerun_outdated=False)"
    if other:
        notes.append(f"{len(other)} TEMPO result(s) of this task come from other tempo_rl code (source "
                     f"{', '.join(sorted({r.tempo_source for r in other}))}; current {code}): {verdict}")
    if edited:
        notes.append(f"{len(edited)} TEMPO result(s) of this task were still running when a tempo_rl file was modified (a run's "
                     "tempo_source is written when the run ends, so it may describe code the run never ran): " + verdict)
    for e in excluded:
        if e["state"] == "misplaced":
            notes.append(f"WARNING: {e['path']} is never used: {'; '.join(e['why'])}. Counting it would add a second "
                         f"{e['algorithm']} seed {e['seed']} or a run that no protocol check has seen. The launch cell only "
                         "moves files at the path of a planned run, so move or delete this one yourself")
            continue
        notes.append(f"WARNING: {e['path']} is {e['state']} ({'; '.join(e['why'])}): left out of the statistics; "
                     + ("the launch cell moves it to _outdated/ (kept, never deleted) and runs it again at BUDGET "
                        f"{lab.budget:,}" if e["planned"] else
                        "it is not one of this notebook's planned runs (ALGOS / SEEDS / TEMPO_VARIANT), so the launch cell "
                        "does not touch it"))
    for f, why in unreadable:
        notes.append(f"WARNING: {f} cannot be read ({why}): skipped")
    if provenance:
        notes.append(_provenance_note([(e["algorithm"], e["seed"], e["differences"]) for e in provenance]))
    n_imp = _n_imported(lab, runs)
    if n_imp:
        notes.append(f"{n_imp} of the {sum(len(rs) for rs in runs.values())} runs used here are byte copies of earlier runs from "
                     "benchmarks/results/final (import_previous), not runs made by this notebook; their stored job record "
                     "still says suite 'final' and their tempo_source / library versions are those of the earlier benchmark")
    if any(s["budget"] != lab.budget for s in stats.values()):
        notes.append(f"WARNING: some runs have a budget other than this notebook's {lab.budget:,} steps "
                     "(a smoke run in a real folder, or another BUDGET)")
    if opt is not None and ref.get("optimal_policy_eval") is not None and abs(ref["optimal_policy_eval"] - opt) > 1e-9:
        notes.append(f"{lab.env} " + ("starts each episode in a random state" if lab.env.startswith("Taxi") else "is stochastic")
                     + f" and every evaluation replays the same {ref['n_eval']} held-out episodes: an "
                     f"optimal policy scores {ref['optimal_policy_eval']:.4g} on them (exact expectation {opt:.4g}), so "
                     "'% random to optimum' can exceed 100%; the CI over seeds does not include this episode-level noise")
    if lab.kind == "baselines" and not lab.algos:
        notes.append("no baselines selected (ALGOS = ()): " + ("finished runs found in the results folder are shown"
                                                                if stats else "only the reference scores are shown"))
    if lab.kind == "baselines" and any(not zoo_info(lab)[a]["tuned"] for a in lab.algos):
        notes.append("untuned: " + ", ".join(a for a in lab.algos if not zoo_info(lab)[a]["tuned"])
                     + " (no RL-Zoo entry for this task: library defaults)")
    if lab.smoke:
        notes.append(f"SMOKE MODE: {SMOKE_STEPS:,}-step runs of one seed; these numbers only show that the pipeline "
                     "works and mean nothing about the algorithms")
    provisional = bool(missing or short or pending or failed or any(e["planned"] for e in excluded))
    return dict(provisional=provisional, env=lab.env, kind=lab.kind, suite=lab.suite, folder=lab.folder_name, seeds=list(lab.seeds), smoke=lab.smoke,
                budget=lab.budget, eval_freq=lab.eval_freq, n_eval=lab.n_eval, tempo_label=lab.tempo_label, code_hash=code,
                labels=labels, runs=runs, stats=stats, per_seed=per_seed, ranking=ranking, references=ref,
                reference_table=ref_table, pending=pending, failed=failed, notes=notes, outdated=outdated, info=lab.info,
                excluded=excluded, provenance=provenance)


def show_summary(S: dict) -> None:
    """Display the tables of :func:`summary` (per-seed scores, ranking, references) and print the notes. Row labels are
    shown in full (pandas would cut long ones with '...' in a text output)."""
    import pandas as pd
    from IPython.display import display

    n = 3 * S["n_eval"]
    with pd.option_context("display.max_colwidth", None):
        if len(S["per_seed"]):
            print(f"{S['env']} ({S['folder']}, suite {S['suite']}): final score per seed (mean of the last three evaluations = "
                  f"{n} episodes), mean and 95% bootstrap CI over seeds, AUC (mean over all evaluations):")
            display(S["per_seed"])
        else:
            print(f"{S['env']} ({S['folder']}, suite {S['suite']}): no runs on this task, only the reference scores:")
        if S["ranking"] is not None:
            print("Ranking by mean final score (normalised = (x - random) / (best - random)):")
            display(S["ranking"])
        if len(S["reference_table"]):
            print("References:")
            display(S["reference_table"])
    for d in S["notes"]:
        print("  - " + d)


def _seed_text(seeds) -> str:
    """``10-14`` for consecutive seeds, else ``10, 12, 14``."""
    s = sorted(int(k) for k in seeds)
    if len(s) > 2 and s == list(range(s[0], s[-1] + 1)):
        return f"{s[0]}-{s[-1]}"
    return ", ".join(str(k) for k in s)


def _suite_tempo_labels(lab: Lab, labels=()) -> list[str]:
    """Every TEMPO label of this lab's results (all tasks of ``<results>/<suite>``, plus ``labels`` and the lab's own),
    so that :func:`figstyle.tempo_style` gives a TEMPO variant the same colour on every task, as the Comparison does."""
    found = {lab.tempo_label} | {l for l in labels if report.is_tempo(l)}
    try:
        for d in lab.suite_dir.glob("*/TEMPO/*"):
            if d.is_dir() and any(d.glob("seed*.json")):
                found.add(report.label_of("TEMPO", d.name))
    except OSError:                                         # unreadable folder: the labels of this task only
        pass
    return sorted(found, key=report.label_key)


def plot_curves(lab: Lab, S: dict, n_boot: int = 2000, band=None, seeds: bool = False) -> list:
    """Learning curves of the task in the academic style of ``notebooks/figstyle.py`` (one single-column panel, 3.5 in):
    per algorithm ONE bold line = the mean evaluation return over the seeds (each evaluation = the mean of ``lab.n_eval``
    deterministic episodes) and ONE shaded band in the same colour = the spread of the seeds. ``band``: ``None`` = figstyle's
    default ``"minmax"`` (range across seeds), ``"std"`` (mean +- 1 standard deviation) or ``"ci95"`` (95% bootstrap CI of
    the mean, ``benchmarks.report.curve_band`` with ``n_boot`` resamples, the previous figures' band); one seed gives a
    line without a band; ``seeds=True`` adds the individual seeds as thin lines. Seeds are aligned on their common
    evaluation steps; only evaluations at or before the budget are used (the runs of :func:`summary`). Band opacity
    follows the number of banded curves (``figstyle.band_alpha``: lighter when 3 or more overlap). The y axis is fitted
    to the mean lines and the reference lines (``figstyle.fit_ylim``): a band is shown in full unless one seed lies far
    outside them (more than half their span beyond), then it is clipped at the axes and the caption says so. Reference
    lines (random policy, gymnasium solved threshold, toy-text optimum) in neutral grey with their own dash patterns,
    labelled with their values in the legend. Colours and dashes are ``benchmarks.report.STYLE``; this lab's TEMPO
    (``lab.tempo_label``) is ``STYLE["TEMPO"]``, drawn bolder, on top and first in the legend; every other TEMPO variant
    of the folder (plain TEMPO too when the notebook runs another variant) gets its own red / purple colour and dash
    pattern by ``figstyle.tempo_style`` over all TEMPO labels of the folder, the rule of the Comparison figures. The
    panel title is the task name; the explanation is the caption printed under the figure. Saves ``<figures>/<task>.pdf`` (vector, embedded
    TrueType fonts) and ``.png`` (300 dpi), shows the saved PNG inline and returns ``[png, pdf]`` (end the notebook cell
    with ``;``). The x axis spans the lab's budget (``lab.budget``) unless a run went further. Works with no finished run
    (empty axes with the reference lines). When the summary is provisional (``S["provisional"]``: runs pending, running or
    failed) the title and the caption say so and the files are ``<task>.partial.png`` / ``.pdf``, so an earlier complete
    figure is never replaced by a partial one. A figure and tables of this task whose table records another budget /
    evaluation schedule are first moved to ``<figures|tables>/_outdated/<time>/`` (:func:`_archive_outputs`; never
    deleted)."""
    import matplotlib.pyplot as plt

    import figstyle as fs

    band = fs.check_band(band)
    labels = [l for l in S["labels"] if S["runs"].get(l)]
    ref = S["references"]
    budget = max([r.budget for rs in S["runs"].values() for r in rs] or [lab.budget])
    prov = bool(S.get("provisional"))
    no_baselines = lab.kind == "baselines" and not lab.algos
    ns = {len(S["runs"][l]) for l in labels}
    n_common = next(iter(ns)) if len(ns) == 1 else None    # None: the curves have different numbers of seeds
    headline = lab.tempo_label
    tempo_labels = _suite_tempo_labels(lab, labels)
    alpha = fs.band_alpha(sum(len(S["runs"][l]) > 1 for l in labels))
    with fs.paper():
        fig, ax = fs.subplots(width=fs.SINGLE, height=2.75)          # one column; the legend sits below the axes
        curves = []
        for l in labels:
            tempo = lab.kind == "tempo" and l == headline
            st = fs.tempo_style(l, headline, tempo_labels)  # headline: STYLE["TEMPO"]; other TEMPO: a variant slot
            rs = S["runs"][l]
            steps, Y = fs.align(rs, budget=budget)
            name = l if n_common is not None else f"{l} (n={len(rs)})"
            c = fs.curve(ax, steps, Y, name, style=st, tempo=tempo, band=band, seeds=seeds, key=(lab.env, l),
                         n_boot=n_boot, alpha=alpha)
            if c is not None:
                curves.append(c)
        base = lambda c: c.label.split(" (n=")[0]          # noqa: E731
        curves.sort(key=lambda c: (base(c) != headline, report.label_key(base(c))))   # this notebook's TEMPO first
        refs, ref_text, ref_y = [], [], []
        for kind, y, what in (("optimum", ref.get("optimum"), f"known optimum ({_optimum_method(lab.env)})"),
                              ("solved", ref.get("threshold"), "solved threshold (gymnasium reward_threshold)"),
                              ("random", ref.get("random"), "random policy (benchmarks/random_scores.json)")):
            if y is None:
                continue
            refs.append(fs.reference_line(ax, y, kind, label=f"{fs.REFERENCE[kind]['label']} "
                                                             f"({fs.num(y, ',.0f' if abs(y) >= 1e4 else '.4g')})"))
            ref_text.append(f"{what} {y:,.0f}" if abs(y) >= 1e4 else f"{what} {y:.4g}")
            ref_y.append(y)
        clipped = fs.fit_ylim(ax, *[c.mean for c in curves], ref_y, bands=curves)    # one far-out seed: band clipped
        fs.steps_axis(ax, budget)
        fs.value_axis(ax, fs.RETURN_LABEL)
        tags = (["smoke test"] if lab.smoke else []) + (["provisional"] if prov else [])
        ax.set_title(lab.env + (f" ({', '.join(tags)})" if tags else ""))
        if not curves:
            ax.text(0.5, 0.5, "Reference scores only (no baselines run)" if no_baselines else "No finished run yet",
                    transform=ax.transAxes, ha="center", va="center", color=fs.INK2, fontsize=fs.FONT_SIZE, zorder=5,
                    bbox=dict(facecolor=fs.SURFACE, edgecolor="none", pad=1.5))
        fs.legend(fig, curves, refs)
        _archive_outputs(lab)                               # a figure / tables of another BUDGET is moved, not overwritten
        stem = f"{lab.env}.partial" if prov else lab.env       # a provisional figure never replaces a complete one
        paths = fs.save(fig, lab.figures / stem, formats=("png", "pdf"))
    plt.close(fig)
    fs.show(paths[0])                                       # the saved PNG: the notebook shows exactly the file
    cap = [f"Figure. {lab.env}, {lab.folder_name}" + (" (SMOKE: pipeline check, the numbers mean nothing)" if lab.smoke else "")
           + f": evaluation return (mean of {lab.n_eval} deterministic episodes) against environment steps; {budget:,} steps, "
           f"an evaluation every {lab.eval_freq:,}."]
    if curves:
        ids = _seed_text(S["seeds"]) if n_common == len(S["seeds"]) else None
        cap.append(fs.band_caption(band, n_common, seeds, ids=ids, clipped=clipped))
        others = [base(c) for c in curves if report.is_tempo(base(c)) and base(c) != headline]
        if lab.kind == "tempo" and others:
            cap.append(f"Red, bold, on top: {headline} (this notebook's TEMPO); the other TEMPO variants of this folder "
                       f"({', '.join(others)}) have their own colours and dash patterns.")
    else:
        cap.append("No finished run: only the reference lines." if not no_baselines else
                   "No baselines are run on this task: only the reference lines.")
    if ref_text:
        cap.append("Grey lines: " + "; ".join(ref_text) + ".")
    if prov:
        cap.append(f"PROVISIONAL: not every planned run has finished ({len(S['seeds'])} seeds requested), so the files are "
                   "named .partial and a complete figure of this task is not overwritten.")
    if curves and n_common != 1:
        other = {"minmax": 'band="minmax" (range across seeds, the default)', "std": 'band="std" (mean \u00b1 1 standard '
                 'deviation)', "ci95": 'band="ci95" (95% bootstrap CI of the mean)'}
        cap.append("Other bands: rl.plot_curves(LAB, S, ...) with " + " or ".join(v for k, v in other.items() if k != band)
                   + ("" if seeds else "; seeds=True adds the individual seeds") + ".")
    print(fs.caption(" ".join(cap)))
    print(f"saved {paths[0]} and .pdf")
    return paths


def _versions(lab: Lab) -> dict:
    import importlib.metadata as md
    import platform

    def v(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    out = dict(python=platform.python_version(), platform=platform.platform())
    for key, pkg in (("torch", "torch"), ("gymnasium", "gymnasium"), ("numpy", "numpy"), ("stable_baselines3", "stable-baselines3"),
                     ("sb3_contrib", "sb3-contrib"), ("mujoco", "mujoco"), ("pandas", "pandas"), ("scipy", "scipy")):
        out[key] = v(pkg)
    try:
        import tempo_rl
        out["tempo_rl"] = tempo_rl.__version__
    except Exception:                                       # noqa: BLE001
        out["tempo_rl"] = None
    out["tempo_rl_source"] = lab.code_hash()
    return out


def _df_md(df, index_name: str) -> str:
    """Markdown table of a DataFrame (index as the first column; text columns left-aligned, numbers right-aligned)."""
    if df is None or not len(df):
        return ""
    header = [index_name] + [str(c) for c in df.columns]
    rows = [[str(i)] + ["" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v) for v in r]
            for i, r in zip(df.index, df.itertuples(index=False))]
    left = {"algorithm", "hyperparameters", "reference"}
    align = "".join("l" if h in left or (k == 0 and index_name in left) else "r" for k, h in enumerate(header))
    return report.md_table(header, rows, align)


def save(lab: Lab, S: dict, cfg: dict | None = None) -> list:
    """Write ``<tables>/<task>.md`` (readable tables) and ``.json`` (per-seed scores, summary statistics, ranking,
    references, protocol, library versions: python / torch / gymnasium / stable-baselines3 / sb3-contrib / numpy,
    tempo_rl version + source hash, and the zoo hyperparameters (baselines) or the TEMPO configuration ``cfg``
    (tempo); the notes, with the result files left out and the runs kept despite another provenance, also as
    ``excluded`` / ``provenance``; the protocol with the lab's BUDGET / EVAL_FREQ, runlab's TEST values and the earlier
    benchmark's). Returns the two paths (end the notebook cell with ``;``). A provisional summary (``S["provisional"]``)
    is written as ``<task>.partial.md`` / ``.json`` so that complete tables are never overwritten by partial ones. Tables (and
    the figure) of this task that record another budget / evaluation schedule are first moved to ``<tables|figures>/
    _outdated/<time>/`` (:func:`_archive_outputs`; never deleted)."""
    _archive_outputs(lab)
    lab.tables.mkdir(parents=True, exist_ok=True)
    same = (lab.real_budget, lab.real_eval_freq) == (TEST_BUDGET, TEST_EVAL_FREQ)
    protocol = dict(budget=lab.budget, eval_freq=lab.eval_freq, n_eval=lab.n_eval, evaluations=lab.budget // lab.eval_freq,
                    eval_seed_base=100_000,
                    final="mean return of the last three evaluations at or before the budget",
                    auc="mean return over all evaluations", ci="95% percentile bootstrap over seeds (10,000 resamples)",
                    env_kwargs=lab.info["env_kwargs"], suite=lab.suite, seeds=list(lab.seeds),
                    lab_budget=lab.real_budget, lab_eval_freq=lab.real_eval_freq,
                    test_budget=TEST_BUDGET, test_eval_freq=TEST_EVAL_FREQ, same_as_test_protocol=same,
                    earlier_budget=lab.earlier_budget, earlier_eval_freq=lab.earlier_eval_freq,
                    replay_capacity=plans.REPLAY_CAPACITY.get(lab.env),
                    source=("budget / eval_freq: make_lab(budget=, eval_freq=) = the Parameters cell's BUDGET / EVAL_FREQ "
                            f"(runlab.TEST_BUDGET / TEST_EVAL_FREQ = {TEST_BUDGET:,} / {TEST_EVAL_FREQ:,}, the same for every "
                            "task and both folders); evaluation, seeds and job fields: benchmarks/run_one.py and "
                            "benchmarks/plans.py as in the main benchmark, whose budget for this task (earlier_budget) was "
                            "different"))
    zoo = {a: {k: v for k, v in d.items() if k != "_raw"} for a, d in zoo_info(lab).items()} if lab.kind == "baselines" else None
    first = {l: next((r for r in rs), None) for l, rs in S["runs"].items()}
    run_info = {l: dict(info=bl._json_clean(r.info), device=r.device, tempo_source=r.tempo_source, versions=r.versions,
                        finished=r.finished) for l, r in first.items() if r is not None}
    tcfg = None
    if cfg is not None:
        tcfg = dict(variant=cfg.get("variant"), kwargs=cfg.get("kwargs"), width=cfg.get("width"), device=cfg.get("device"),
                    gamma=cfg.get("gamma"), source_hash=cfg.get("source_hash"), code_root=cfg.get("code_root"),
                    recipe=dict(cfg.get("recipe") or []), matches_reference=cfg.get("match"),
                    arguments=[dict(group=g, setting=s, value=v, source=src) for g, s, v, src in cfg.get("rows", [])])
    stats = {l: dict(seeds=s["seeds"], finals=s["finals"], aucs=s["aucs"], mean_final=s["final"][0], ci_final=list(s["final"][1:]),
                     std_final=s["std"], mean_auc=s["auc"][0], ci_auc=list(s["auc"][1:]), hours_per_run=s["hours"],
                     budget=s["budget"], devices=s["devices"], tempo_sources=s["tempo_sources"], seeds_at_or_above_threshold=s["solved"])
             for l, s in S["stats"].items()}
    ranking = None if S["ranking"] is None else [dict(rank=int(i), **row) for i, row in S["ranking"].iterrows()]
    blob = dict(task=lab.env, kind=lab.kind, folder=lab.folder_name, suite=lab.suite, smoke=lab.smoke, seeds=list(lab.seeds),
                algorithms=list(S["labels"]), protocol=protocol, references=S["references"], stats=stats, ranking=ranking,
                pending=[list(p) for p in S["pending"]], failed=[list(p) for p in S["failed"]], notes=S["notes"],
                excluded=S.get("excluded", []), provenance=S.get("provenance", []),
                versions=_versions(lab), tempo_source=S["code_hash"], zoo=zoo, tempo_config=tcfg, run_info=run_info,
                generated=time.strftime("%Y-%m-%d %H:%M:%S"))
    prov = bool(S.get("provisional"))
    stem = f"{lab.env}.partial" if prov else lab.env             # a provisional table never replaces a complete one
    blob["provisional"] = prov
    js = lab.tables / f"{stem}.json"
    js.write_text(json.dumps(bl._json_clean(blob), indent=1))
    what = (S["tempo_label"] if lab.kind != "baselines" else
            {"none": ("baselines found in the results folder (none selected: ALGOS = ())" if S["stats"] else
                      "reference scores (no baselines run)"), "all": "tuned baselines",
             "no": "baselines at library defaults (untuned: no RL-Zoo entry for this task)",
             "some": "baselines (some tuned, some at library defaults)"}[_tuning(lab)])
    lines = [f"# {lab.env}: {what} ({lab.folder_name})" + (" [PROVISIONAL: not every run has finished]" if prov else ""), "",
             f"Suite `{lab.suite}`, seeds {list(lab.seeds)}, budget {lab.budget:,} environment steps, evaluation every "
             f"{lab.eval_freq:,} steps ({lab.budget // lab.eval_freq} evaluations of {lab.n_eval} deterministic episodes, "
             "held-out seeds 100000+i)"
             + ("" if lab.smoke else
                " = runlab.TEST_BUDGET / TEST_EVAL_FREQ, the same for every task and both folders" if same else
                f" (NOT runlab.TEST_BUDGET / TEST_EVAL_FREQ = {TEST_BUDGET:,} / {TEST_EVAL_FREQ:,})")
             + f"; {lab.earlier_text()}. Final score = mean of the "
             "last three evaluations at or before the budget; AUC = mean over all evaluations; 95% percentile-bootstrap CIs "
             f"over seeds. tempo_rl source `{S['code_hash']}`. Generated {time.strftime('%Y-%m-%d %H:%M')} by notebooks/runlab.py.",
             ""]
    if len(S["per_seed"]):
        lines += ["## Final score per seed", "", _df_md(S["per_seed"], "algorithm")]
    if S["ranking"] is not None:
        lines += ["## Ranking", "", _df_md(S["ranking"], "rank")]
    if len(S["reference_table"]):
        lines += ["## References", "", _df_md(S["reference_table"], "reference")]
    if zoo:
        lines += ["## Hyperparameters (benchmarks/zoo.py)", ""]
        for a, d in zoo.items():
            lines.append(f"* **{a}** ({'tuned' if d['tuned'] else 'UNTUNED: library defaults'}): n_envs {d['n_envs']}, normalize "
                         f"{d['normalize']}, noise {d['noise_type'] or 'none'}, device {d['device']}; "
                         f"`{json.dumps(d['hyperparameters'])}`")
        lines.append("")
    if tcfg:
        lines += ["## TEMPO configuration", ""] + [f"* **{k}**: {v}" for k, v in tcfg["recipe"].items()]
        lines += [f"* device {tcfg['device']}, source `{tcfg['source_hash']}`", ""]
    if S["notes"]:
        lines += ["## Notes", ""] + [f"- {d}" for d in S["notes"]] + [""]
    md = lab.tables / f"{stem}.md"
    md.write_text("\n".join(lines))
    print(f"wrote {md}\nwrote {js}"
          + (" (PROVISIONAL: not every run has finished, so the files are named .partial and complete tables of this task are "
             "not overwritten)" if prov else ""))
    return [md, js]


# =========================================================================== reuse the earlier SB3 results


def _record_differences(lab: Lab, algo: str, rec: dict, now: dict | None = None) -> list[str]:
    """How an earlier result record differs from what this lab would run now, beyond the job fields: the suite, the length
    and shape of the curve (every evaluation of the budget present, 10 episodes each), the hyperparameters the run
    recorded (``zoo_tuned``, ``n_envs``, ``normalize``, ``config`` = the zoo arguments as ``run_one.run_sb3`` stored them;
    :func:`_hp_differences`) against today's ``benchmarks/zoo.py``, and the library versions it recorded (python, torch,
    gymnasium, numpy, Stable-Baselines3, sb3-contrib for CrossQ; :func:`_version_differences`) against the installed
    ones. Empty = protocol-identical."""
    diff: list[str] = []
    j = rec.get("job") or {}
    if j.get("suite", "main") != "final":
        diff.append(f"suite {j.get('suite', 'main')!r} != 'final'")
    curve = _curve_problem(rec, lab.real_budget, lab.real_eval_freq, N_EVAL)
    if curve:
        diff.append(curve)
    diff += _hp_differences(algo, lab.env, rec.get("info"), lab.real_budget)
    diff += _version_differences(rec, algo, now)
    return diff


def import_previous(lab: Lab, algos=None, seeds=None, dry_run: bool = True, source=None):
    """Baselines only, OPT-IN. Copy (never move, never overwrite) earlier protocol-identical SB3 results
    ``benchmarks/results/final/<env>/<algo>/default/seed<k>.json`` into this lab's
    ``<results>/baselines_test/<env>/<algo>/default/seed<k>.json`` so they need not be re-run.

    ``dry_run=True`` (default) only prints what would be copied or skipped; call again with ``dry_run=False`` to
    copy. A file is copied only if its job matches this task's protocol (algorithm, variant ``default``, budget,
    ``eval_freq``, ``n_eval`` 10, env_kwargs, seed, suite ``final``) and this notebook's run (device =
    :func:`device_for`, threads 1), its curve is complete (every evaluation of the budget, 10 episodes each), the
    hyperparameters and the python / torch / gymnasium / numpy / Stable-Baselines3 / sb3-contrib versions it recorded equal
    today's ``benchmarks/zoo.py`` and the installed libraries (``_record_differences``), and the destination does not exist.
    Copies are byte-exact; a manifest ``<suite>/_imported.json`` (ignored by ``load_suite``) records source and SHA-256, and
    the plan table shows ``done (imported)`` while the file is still that copy. The manifest is re-read, extended and written
    (atomically) under an exclusive lock (``<suite>/.imported.lock``), so notebooks importing at the same time keep every
    entry; the entry of a copy is written BEFORE the copy appears at its path (a journal), so an import killed half-way
    leaves no copy without its entry (an entry without its file is harmless: only entries whose SHA-256 matches the file
    count), and the temporary files of a killed import are removed by the next one. Not available in smoke mode (tiny
    smoke runs must not mix with full results).
    The budget and evaluation schedule are the LAB's (BUDGET / EVAL_FREQ, default :data:`TEST_BUDGET` /
    :data:`TEST_EVAL_FREQ`), and the earlier benchmark ran every task at another budget (:func:`earlier_protocol`), so by
    default nothing matches: then only one line is printed ("no earlier result matches BUDGET=... / EVAL_FREQ=...: the
    earlier benchmark ran this task for <budget> steps, evaluating every <freq>; ...") and nothing is copied. With
    ``make_lab(budget=, eval_freq=)`` set to the earlier values the import works exactly as before.
    ``source``: another results root (default ``benchmarks/results``). The earlier logs no longer exist. Returns a
    DataFrame with one row per (algorithm, seed) and its action."""
    import pandas as pd

    if lab.kind != "baselines":
        raise ValueError("import_previous is for the baselines folder")
    cols = ["algorithm", "seed", "action", "reason"]
    if lab.smoke:
        print("SMOKE mode: nothing imported (smoke runs are tiny and must not mix with full-protocol results)")
        return pd.DataFrame(columns=cols)
    if lab.earlier_budget is None:
        print(f"nothing to import: {lab.env} was not part of the earlier benchmark (added to the test folders on 2026-11-03)")
        return pd.DataFrame(columns=cols)
    algos = _check_algos(lab.env, lab.info, algos) if algos is not None else lab.algos
    seeds = tuple(int(s) for s in (seeds if seeds is not None else lab.seeds))
    src_root = pathlib.Path(source) if source is not None else BENCH / "results"
    want = dict(steps=lab.real_budget, eval_freq=lab.real_eval_freq, n_eval=N_EVAL, env=lab.env)
    kw = plans.ENV_KWARGS.get(lab.env) or None
    now = _installed_versions()
    plan_rows, copies = [], []
    for a in algos:
        dev = device_for(lab, a)
        for k in seeds:
            s = src_root / "final" / lab.env / a / "default" / f"seed{k}.json"
            d = lab.suite_dir / lab.env / a / "default" / f"seed{k}.json"
            if d.exists():
                act, why = "skip", "destination exists (never overwritten)"
            elif not s.exists():
                act, why = "skip", "no earlier result"
            else:
                try:
                    rec = json.loads(s.read_text())
                    j = rec["job"]
                    diff = [_neq(f, j.get(f), v) for f, v in want.items() if j.get(f) != v]
                    if j.get("algo") != a or j.get("variant", "default") != "default" or int(j.get("seed", -1)) != k:
                        diff.append("job identity differs")
                    if (j.get("env_kwargs") or None) != kw:
                        diff.append(_neq("env_kwargs", j.get("env_kwargs"), kw))
                    if j.get("device", "cpu") != dev:               # run_one's defaults for a missing key
                        diff.append(_neq("device", j.get("device", "cpu"), dev) + f" (this notebook's device for {a})")
                    if j.get("threads", 1) != 1:
                        diff.append(_neq("threads", j.get("threads", 1), 1))
                    diff += _record_differences(lab, a, rec, now)
                    act, why = ("skip", "protocol differs: " + "; ".join(diff)) if diff else ("copy", "protocol-identical")
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
                    act, why = "skip", f"unreadable ({type(e).__name__})"
            plan_rows.append(dict(algorithm=a, seed=k, action=act if dry_run or act == "skip" else "copy", reason=why))
            if act == "copy":
                copies.append((a, k, s, d))
    df = pd.DataFrame(plan_rows, columns=cols)
    eb, ef = lab.earlier_budget, lab.earlier_eval_freq
    if not copies and (lab.real_budget, lab.real_eval_freq) != (eb, ef):
        print(f"no earlier result matches BUDGET={lab.real_budget:,} / EVAL_FREQ={lab.real_eval_freq:,}: the earlier benchmark "
              f"ran this task for {eb:,} steps, evaluating every {ef:,}; set BUDGET/EVAL_FREQ to those values to reuse them")
        return df
    if not dry_run and copies:
        with _manifest_lock(lab):                           # re-read -> copy -> write as one step (other notebooks wait)
            manifest = _read_manifest(lab)
            for a in algos:                                  # temporary copies of an import that was killed: only imports
                for k in seeds:                              # write these names, and they hold the lock we hold now
                    for t in (lab.suite_dir / lab.env / a / "default").glob(f".seed{k}.json.tmp-*"):
                        t.unlink(missing_ok=True)
            for a, k, s, d in copies:
                d.parent.mkdir(parents=True, exist_ok=True)
                tmp = d.with_name(f".{d.name}.tmp-{os.getpid()}")
                shutil.copyfile(s, tmp)
                entry = dict(env=lab.env, algo=a, seed=k, source=str(s), sha256=_sha256(tmp),
                             copied=time.strftime("%Y-%m-%d %H:%M:%S"))
                same = (lab.env, a, str(k))                  # an older entry of this path (not moved away) is stale:
                keep = [m for m in manifest if m.get("moved_to")          # the destination did not exist
                        or (m.get("env"), m.get("algo"), str(m.get("seed"))) != same]
                _write_manifest(lab, keep + [entry])         # journal first: a kill after the link cannot lose the entry
                try:
                    os.link(tmp, d)                          # atomic and fails if the destination appeared meanwhile
                    done = True
                except FileExistsError:
                    done = False
                except OSError:                              # filesystem without hard links: still never overwrite
                    done = not d.exists()
                    if done:
                        os.replace(tmp, d)
                finally:
                    tmp.unlink(missing_ok=True)
                idx = df.index[(df.algorithm == a) & (df.seed == k)][0]
                if done:
                    df.loc[idx, ["action", "reason"]] = ["copied", f"from {s}"]
                    manifest = keep + [entry]
                else:
                    df.loc[idx, ["action", "reason"]] = ["skip", "destination appeared meanwhile (never overwritten)"]
            _write_manifest(lab, manifest)                   # final state: drops the journal entry of a copy not made
    n_copy = int((df.action.isin(["copy", "copied"])).sum()) if len(df) else 0
    from IPython.display import display
    if len(df):
        display(df.set_index(["algorithm", "seed"]))
    if dry_run:
        print(f"DRY RUN: {n_copy} file(s) would be copied from {src_root / 'final'} to {lab.suite_dir}; "
              "nothing was copied. import_previous(LAB, dry_run=False) (IMPORT_PREVIOUS = True in the notebooks) copies them.")
    else:
        print(f"copied {n_copy} file(s) to {lab.suite_dir}; skipped {int((df.action == 'skip').sum())}. Originals untouched.")
    return df
