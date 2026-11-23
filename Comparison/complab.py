"""Analysis helpers of ``Comparison/01_tempo_vs_baselines.ipynb``: TEMPO against the earlier RL algorithms.

The notebook reads two result folders written by the run notebooks

* ``Baselines_test/results/baselines_test``  SAC, TD3, CrossQ, PPO, DQN (tuned RL-Zoo hyperparameters),
* ``TEMPO_test/results/tempo_test``           TEMPO (library defaults = the fast recipe since 2026-10-23, the previous
                                              default C1 is the variant ``"c1"``; or other named variants),

each laid out as ``<root>/<suite>/<env>/<algo>/<variant>/seed<k>.json`` (``benchmarks/run_one.py``), and
compares them on 23 tasks: the 24 of the test folders (``report.TASK_ORDER``: the earlier benchmark's 20 and CarRacing-v3,
HumanoidStandup-v5, Blackjack-v1 and Taxi-v4, added on 2026-11-03) without CarRacing-v3, which the run notebooks still run but
the comparison leaves out unless ``TASKS`` names it (:data:`LEFT_OUT`). Nothing here trains a model or writes into a result
folder.

Everything statistical is the project's own code, ``benchmarks/report.py`` (``load_suite``, ``Run``,
``per_task``, ``normalisation``, ``aggregate_ci``, ``performance_profile``, ``prob_improvement``,
``outcome`` / ``win_tie_loss``, ``STYLE``), so the definitions are identical to the project's report:

* final score of a run = mean of the last three evaluations at or before the budget (3 x the episodes per evaluation: 30,
  Blackjack-v1 3,000),
  AUC = mean return over all evaluations, evaluations after the budget are discarded;
* per task: mean over seeds with a 95% percentile-bootstrap CI; Welch t-test with alpha 0.05;
* normalised score = (x - random) / (best baseline mean final - random), IQM / median / mean with
  stratified-bootstrap CIs, performance profiles, probability of improvement (Agarwal et al., 2021).

Additions of this module on top of ``report.py`` (each one is switchable or listed in the output): a normalisation is
only used when the best baseline covers at least ``min_gap`` (10%) of the reference range above the random policy
(otherwise the task is "reference too weak" and left out of every normalised statistic); toy-text tasks are never
normalised; Welch tests need ``min_n_test`` (3) seeds per side; aggregates are also given for like-for-like task sets
(discrete-action, continuous-action, tuned-baseline tasks) because an algorithm that ran on other tasks is not comparable;
cross-task rank statistics (mean rank, Friedman, Wilcoxon with Holm correction).

The two roots are loaded separately and merged; missing tasks, algorithms and seeds are reported as
``pending`` (never silently dropped), unreadable or off-protocol result files are listed, not used.

Protocol. The expected protocol of EVERY task is the test protocol of the run notebooks, defined once in
``notebooks/runlab.py`` (imported here, side-effect free): ``TEST_BUDGET`` environment steps, an evaluation every
``TEST_EVAL_FREQ`` steps with ``N_EVAL`` deterministic episodes (more on the tasks of ``runlab.N_EVAL_BY_TASK``, read per task
through ``runlab.n_eval_for``: Blackjack-v1 1,000), seeds 10-14 (``runlab.TEST_SEEDS``); in smoke mode the schedule
``benchlab.shrink`` makes of it (:func:`expected_protocol`; 3 episodes on every task). An environment step of the pixel task
CarRacing-v3 is an agent step of 4 simulator frames (:data:`FRAME_SKIP`). The protocol audit raises an ERROR when the algorithms of
a task differ from each other (budget, evaluation frequency, episodes) and a WARN when a run differs from the expected
protocol. Budgets shown in titles, axes, budget shares and steps-to-level come from the runs themselves. The earlier
benchmark (``benchmarks/plans.py`` budgets of 20k-300k steps, results in ``benchmarks/results/final``) is another
protocol: ``USE_PREVIOUS_BASELINES`` only fills empty cells with earlier runs that match the expected protocol.

Figures. Every figure is drawn with ``notebooks/figstyle.py`` (the academic style shared with the run notebooks: serif print
fonts, paper widths, one shared legend, a caption printed under each figure, vector PDF + 300-dpi PNG). Return curves show per
algorithm ONE bold line = the mean over seeds and ONE shaded band = the spread of the seeds, chosen by ``CURVE_BAND``:
"minmax" (default, range across seeds), "std" (mean +- 1 standard deviation) or "ci95" (95% bootstrap CI of the mean,
``report.curve_band``), its opacity ``figstyle.band_alpha`` of the number of bands in the panel; the y axis shows the mean lines
and reference lines in full and clips a band that one outlying seed stretches far beyond them (``figstyle.fit_ylim``; the caption
says so). Colours and dashes: ``figstyle.tempo_style`` = ``report.STYLE`` for the algorithms and the headline TEMPO,
``figstyle.VARIANT_STYLES`` for the other TEMPO variants (the same rule as the run notebooks).

Layout: 1 configuration and loading, 2 inventory and protocol audit, 3 figures, 4 score tables,
5 aggregate statistics, 6 sample efficiency, 7 stability, 8 compute cost, 9 TEMPO variants,
10 exports and the executive summary.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import pathlib
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np


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

from benchmarks import plans as P  # noqa: E402
from benchmarks import report as R  # noqa: E402

_NOTEBOOKS = ROOT / "notebooks"                     # runlab.py / benchlab.py: helpers of the run notebooks
if str(_NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS))
import benchlab as BL  # noqa: E402
import figstyle as fs  # noqa: E402  (academic figure style shared with runlab; changes no matplotlib state on import)
import runlab as RL  # noqa: E402  (creates no file and prints nothing on import)

#: The test protocol of Baselines_test / TEMPO_test (single source of truth: notebooks/runlab.py), expected on every task.
TEST_BUDGET = int(RL.TEST_BUDGET)
TEST_EVAL_FREQ = int(RL.TEST_EVAL_FREQ)
TEST_N_EVAL = int(RL.N_EVAL)
BASELINES = tuple(R.BASELINES)                      # SAC, TD3, CrossQ, PPO, DQN
BEST = R.BEST                                       # composite "Best baseline" (per task, by mean final)
ALPHA = R.ALPHA
FINAL_SEEDS = tuple(RL.TEST_SEEDS)                # the test folders' seeds (runlab.TEST_SEEDS: 10-14 since 2026-11-05)
#: Coverage of a 95% percentile-bootstrap CI of a mean over n seeds (normal data, simulated 2026-11-05: 75.7% with 3 seeds,
#: 84.2% with 5; with skewed data less).
_BOOT_COVERAGE = {3: 76, 5: 84}


def _coverage_text(n: int) -> str:
    c = _BOOT_COVERAGE.get(int(n))
    return (f"with {n} seeds a percentile bootstrap covers the true mean only about {c}% of the time" if c else
            f"with {n} seeds a percentile bootstrap covers the true mean less often than 95%")


def expected_pool(env: str) -> set:
    """Algorithms the final run plans on ``env``: ``report.expected_pool`` (the earlier benchmark's plan) plus, on toy text, the PPO /
    DQN references that Baselines_test runs on every seed (``runlab.TOY_TEXT_ALGOS``, tuned by grid search since 2026-11-04), so that
    completeness, pending cells and the seed audit cover them too. Toy text stays outside verdicts, win / tie / loss and aggregates."""
    return set(R.expected_pool(env)) | (set(RL.TOY_TEXT_ALGOS) if env in P.TOY_TEXT_ALL else set())


TASK_INFO = R.TASK_INFO                             # env: (budget, eval_freq, family) of the earlier benchmark's 20 tasks; here
                                                    # only those budgets (the EARLIER benchmark's, not the expected protocol)
TASK_ORDER = list(R.TASK_ORDER)                     # the 24 tasks of the test folders in report (family) order
#: Tasks of the test folders that the comparison leaves out unless ``TASKS`` names them (since 2026-11-04):
#: CarRacing-v3 takes days to run (pixel observations; planned ~46 h for its baselines, ~15 h for TEMPO), so the comparison does
#: not wait for it. Its notebooks in Baselines_test / TEMPO_test still run it.
LEFT_OUT = ("CarRacing-v3",)
#: The default ``TASKS``: :data:`TASK_ORDER` without :data:`LEFT_OUT` (23 tasks).
DEFAULT_TASKS = [t for t in TASK_ORDER if t not in LEFT_OUT]
#: An environment step of these pixel tasks is an agent step of k simulator frames: the pipeline that every algorithm and every
#: evaluation uses (tempo_rl/env_util.py, CarRacingPreprocess: frame-skip 4 with the rewards summed, 4-frame stack of 84x84 grey
#: images), so a 500,000-step budget is 2,000,000 frames and an episode (1,000 frames) has at most 250 steps.
FRAME_SKIP = {"CarRacing-v3": 4}
#: report.py's toy-text label says "TEMPO only" (the plan of the main benchmark); Baselines_test runs PPO / DQN references there.
FAMILY_NAMES = {**R.FAMILY_NAMES, "toy-text": "Toy text (judged against the exact optimum)"}
DEFAULT_BASELINES_SUITE = "baselines_test"
DEFAULT_TEMPO_SUITE = "tempo_test"
PREVIOUS_SUITE = "final"                            # benchmarks/results/final: the earlier SB3 runs
LEVELS = (0.5, 0.75, 0.9, 1.0)                      # sample-efficiency levels (share of the best baseline)

#: grid / figure groups (several families per figure group so that panels fill complete rows)
GROUPS = {"classic-discrete": "Classic control", "classic-continuous": "Classic control",
          "box2d-discrete": "Box2D", "box2d-continuous": "Box2D", "mujoco": "MuJoCo",
          "toy-text": "Toy text", "other": "Other"}
SOLVED_NOTE = "gymnasium reward_threshold"
#: the shaded band of every return-curve figure (CURVE_BAND; computed by notebooks/figstyle.py; "none" = lines only)
CURVE_BAND_TEXT = {"minmax": "range across seeds (lowest to highest seed)", "std": "mean +- 1 standard deviation across seeds",
                   "ci95": "95% bootstrap CI of the mean over seeds", "none": "nothing (lines only)"}


# --------------------------------------------------------------------------- environment flags
def env_flag(name: str, default: bool) -> bool:
    """Boolean override from the environment, strict like ``runlab.smoke_flag``: unset or empty = ``default``,
    ``1/true/yes/on`` = True, ``0/false/no/off`` = False (stripped, case-insensitive); any other value raises ValueError
    naming the variable (``benchlab.env_flag`` reads a typo such as ``2`` or ``y`` as False without a word)."""
    raw = os.environ.get(name)
    v = "" if raw is None else raw.strip().lower()
    if not v:
        return bool(default)
    if v in RL._SMOKE_TRUE:
        return True
    if v in RL._SMOKE_FALSE:
        return False
    raise ValueError(f"environment variable {name}={raw!r} is not understood: accepted values are "
                     f"{'/'.join(RL._SMOKE_TRUE)} (True) and {'/'.join(RL._SMOKE_FALSE)} (False), case-insensitive; unset or "
                     "empty = the value of the Parameters cell. Fix or unset the variable (nothing was read or written).")


def smoke_from_env(default: bool) -> tuple[bool, str | None]:
    """``(smoke, note)``: the notebook's ``SMOKE`` unless ``TEMPO_NOTEBOOK_SMOKE`` is set, which wins, parsed exactly as
    by the run notebooks (``runlab._smoke_from_env``: 1/true/yes/on, 0/false/no/off, unset or empty = ``default``, any
    other value raises ValueError before anything is read or written). ``note`` says when the variable changes the cell's
    value (None otherwise)."""
    smoke, value = RL._smoke_from_env(default)
    if value is None or smoke == bool(default):
        return smoke, None
    return smoke, (f"the environment variable {RL.SMOKE_ENV}={value!r} overrides SMOKE = {bool(default)} of the Parameters "
                   "cell: " + (f"SMOKE mode (reads the smoke folders of the run notebooks under {scratch_root()})"
                               if smoke else "REAL mode (reads the result folders of Baselines_test / TEMPO_test)"))


def scratch_root() -> pathlib.Path:
    """Where smoke runs write (``$TEMPO_NOTEBOOK_SCRATCH`` or ``<tmp>/tempo_notebooks``), as benchlab."""
    p = os.environ.get("TEMPO_NOTEBOOK_SCRATCH", "").strip()
    return pathlib.Path(p) if p else pathlib.Path(tempfile.gettempdir()) / "tempo_notebooks"


def _env_path(name: str) -> pathlib.Path | None:
    v = os.environ.get(name, "").strip()
    return pathlib.Path(v).expanduser() if v else None


def expected_protocol(smoke: bool = False, env: str | None = None) -> tuple:
    """(budget, eval_freq, n_eval) expected of every run on ``env``: the test protocol of the run notebooks
    (``runlab.TEST_BUDGET`` / ``TEST_EVAL_FREQ``, ``runlab.n_eval_for(env)`` episodes = ``N_EVAL`` except on the tasks of
    ``runlab.N_EVAL_BY_TASK``; ``env=None``: ``N_EVAL``); in smoke mode the schedule ``benchlab.shrink`` makes of it (3 episodes on
    every task)."""
    n = TEST_N_EVAL if env is None else int(RL.n_eval_for(env))
    j = dict(suite="smoke_protocol", steps=TEST_BUDGET, eval_freq=TEST_EVAL_FREQ, n_eval=n)
    if smoke:
        j = BL.shrink([j])[0]
    return int(j["steps"]), int(j["eval_freq"]), int(j["n_eval"])


def n_eval_exceptions(smoke: bool = False, tasks=None) -> dict:
    """{task: episodes per evaluation} of the tasks (default: all of :data:`TASK_ORDER`) that expect another number than ``N_EVAL``
    (``runlab.N_EVAL_BY_TASK``: Blackjack-v1 1,000; none in smoke mode)."""
    n = expected_protocol(smoke)[2]
    return {e: k for e in (TASK_ORDER if tasks is None else tasks) for k in [expected_protocol(smoke, e)[2]] if k != n}


def protocol_text(smoke: bool = False, tasks=None) -> str:
    """``500,000 steps, evaluation every 10,000 steps with 10 episodes (Blackjack-v1: 1,000), seeds 10-14`` (from
    :func:`expected_protocol`; the bracket names the tasks of ``tasks``, default all, with another number of episodes)."""
    b, f, n = expected_protocol(smoke)
    odd = n_eval_exceptions(smoke, tasks)
    ep = f"{n} episodes" + (" (" + ", ".join(f"{e}: {k:,}" for e, k in odd.items()) + ")" if odd else "")
    seeds = f"seed {FINAL_SEEDS[0]}" if smoke else f"seeds {FINAL_SEEDS[0]}-{FINAL_SEEDS[-1]}"
    return f"{b:,} steps, evaluation every {f:,} steps with {ep}, {seeds}"


def final_episodes_text(smoke: bool = False, tasks=None) -> str:
    """Episodes behind a final score (the last three evaluations): ``30 episodes``, or ``30 episodes; Blackjack-v1: 3,000`` when
    tasks of ``tasks`` (default all) expect another number per evaluation (:func:`n_eval_exceptions`)."""
    n = expected_protocol(smoke)[2]
    return f"{3 * n} episodes" + "".join(f"; {e}: {3 * k:,}" for e, k in n_eval_exceptions(smoke, tasks).items())


# --------------------------------------------------------------------------- configuration
@dataclass
class Config:
    """Everything the analysis needs to know; built by :func:`make_config`."""
    baseline_sources: list            # [(results_root, suite)] of the baseline runs
    tempo_sources: list               # [(results_root, suite)] of the TEMPO runs
    previous_source: tuple | None     # (results_root, suite) used to fill missing baseline cells, or None
    tasks: list                       # selected task names (benchmark order)
    baselines: list                   # selected baseline algorithms
    tempo_variants: list | None       # None = every TEMPO variant found
    main_variant: str | None          # None = "default" if present, else the variant with most runs
    seeds: tuple | None               # seeds counted (None = every seed found); expected = FINAL_SEEDS if None
    n_boot: int
    formats: tuple
    out_dir: pathlib.Path
    save: bool
    smoke: bool
    use_previous: bool
    agg_tasks: str = "tempo"          # aggregates over the tasks TEMPO has run on ("tempo") or every task ("all")
    level_plot: float = 0.9           # sample-efficiency level drawn in the dot plot
    reference: str = "best_baseline"  # target of the sample-efficiency levels: best_baseline | solved
    fail_level: float = 0.5           # a seed "fails" when its normalised final score is below this
    collapse_level: float = 0.5       # collapse: smoothed peak - final > this share of (best baseline - random)
    ma_window: int = 3                # evaluations in the moving average (matches the 3-evaluation final score)
    show_info_checks: bool = False
    min_gap: float = 0.10             # the best baseline must cover this share of the reference range above the random policy
    min_n_test: int = 3               # seeds needed on each side of a Welch test (fewer = "untested")
    ci_method: str = "bootstrap"      # per-task CI: "bootstrap" (report.py) or "t" (t-interval)
    only_live_code: bool = False      # True: drop TEMPO runs made with another tempo_rl source than the live library
    archive_stale: bool = True        # move earlier outputs that this (complete) run did not rewrite to <out>/_outdated/
    curve_band: str = fs.DEFAULT_BAND # shaded band of the return curves: "minmax" (range across seeds), "std", "ci95" (figstyle)
    notes: list = field(default_factory=list)

    @property
    def n_curve_boot(self) -> int:
        return int(min(self.n_boot, 1000))

    @property
    def n_expected(self) -> int:
        return len(self.seeds) if self.seeds else len(FINAL_SEEDS)

    @property
    def ci_label(self) -> str:
        return ("t-interval" if self.ci_method == "t" else "percentile-bootstrap CI")

    @property
    def protocol(self) -> tuple:
        """(budget, eval_freq, n_eval) expected of a run with ``N_EVAL`` episodes (:func:`expected_protocol`; per task: :meth:`protocol_of`)."""
        return expected_protocol(self.smoke)

    def protocol_of(self, env: str) -> tuple:
        """(budget, eval_freq, n_eval) expected of every run on ``env`` (``runlab.n_eval_for``: Blackjack-v1 1,000 episodes)."""
        return expected_protocol(self.smoke, env)

    @property
    def fig_dir(self) -> pathlib.Path:
        return self.out_dir / "figures"

    @property
    def table_dir(self) -> pathlib.Path:
        return self.out_dir / "tables"

    def show(self) -> None:
        """Print where the data come from and where outputs go."""
        mode = "SMOKE (scratch folders, tiny runs)" if self.smoke else "REAL"
        print(f"mode              : {mode}")
        for name, src in (("baseline results", self.baseline_sources), ("TEMPO results", self.tempo_sources)):
            if not src:
                print(f"{name:18s}: (none found)")
            for i, (root, suite) in enumerate(src[:4]):
                print(f"{name if i == 0 else '':18s}: {root / suite}" + ("" if (root / suite).is_dir() else "   [folder does not exist yet]"))
            if len(src) > 4:
                print(f"{'':18s}  ... and {len(src) - 4} more folders")
        print(f"expected protocol : {protocol_text(self.smoke, self.tasks)} on every task"
              + (" (SMOKE: the shrunk schedule of the run notebooks)" if self.smoke else " (runlab.TEST_BUDGET / TEST_EVAL_FREQ / n_eval_for)"))
        if self.previous_source:
            print(f"previous baselines: {self.previous_source[0] / self.previous_source[1]}  (read-only, fills EMPTY baseline cells with "
                  "earlier runs at the expected protocol only, marked with +)")
        else:
            print("previous baselines: not used (USE_PREVIOUS_BASELINES = False)")
        print(f"tasks             : {len(self.tasks)} ({', '.join(self.tasks[:3])}{', ...' if len(self.tasks) > 3 else ''})")
        print(f"baselines         : {', '.join(self.baselines) or '-'}")
        print(f"TEMPO variants    : {'all found' if self.tempo_variants is None else ', '.join(self.tempo_variants)}; "
              f"headline variant: {self.main_variant or 'automatic'}")
        print(f"seeds             : {self.seeds if self.seeds else 'every seed found'}; n_boot = {self.n_boot}")
        print(f"guards            : normalise only if the best baseline covers >= {self.min_gap:.0%} of the reference range (MIN_BASELINE_GAP); "
              f"Welch tests need >= {self.min_n_test} seeds per side (MIN_SEEDS_TEST); per-task CI: {self.ci_label}")
        print(f"return curves     : bold line = mean over seeds; shaded = {CURVE_BAND_TEXT[self.curve_band]} (CURVE_BAND = "
              f"{self.curve_band!r})")
        print(f"outputs           : {self.out_dir}" + ("" if self.save else "   [SAVE_OUTPUTS = False: nothing is written]"))
        for n in self.notes:
            print("note              : " + n)


def make_config(baselines_results=None, baselines_suite=DEFAULT_BASELINES_SUITE, tempo_results=None,
                tempo_suite=DEFAULT_TEMPO_SUITE, tasks=None, baselines=None, tempo_variants=None,
                main_variant=None, seeds=FINAL_SEEDS, n_boot=2000, formats=("png", "pdf"),
                use_previous_baselines=False, previous_results=None, out_dir=None, save=True, smoke=False,
                agg_tasks="tempo", level_plot=0.9, reference="best_baseline", fail_level=0.5,
                collapse_level=0.5, show_info_checks=False, min_gap=0.10, min_n_test=3, ci_method="bootstrap",
                only_live_code=False, archive_stale=True, curve_band=fs.DEFAULT_BAND) -> Config:
    """Resolve the notebook's parameters (and the ``COMPARISON_*`` / ``TEMPO_NOTEBOOK_SMOKE`` overrides).

    ``curve_band`` (``CURVE_BAND``) = the shaded band of every return-curve figure around the bold mean over seeds: "minmax"
    (default: range across seeds), "std" (mean +- 1 standard deviation) or "ci95" (95% bootstrap CI of the mean,
    ``benchmarks.report.curve_band``: the band of the earlier figures); "none" = lines only (``figstyle.check_band``).

    Environment variables (used by the tests; they win over the arguments):
    ``COMPARISON_BASELINES_DIR`` / ``COMPARISON_TEMPO_DIR`` = a results root (the folder that holds
    ``<suite>/``), ``COMPARISON_OUT_DIR`` = output folder (``figures/`` and ``tables/`` go below it),
    ``COMPARISON_PREVIOUS_DIR`` / ``COMPARISON_USE_PREVIOUS`` = the read-only previous-baselines root
    and its switch, ``COMPARISON_N_BOOT``, ``COMPARISON_TASKS`` (comma separated), ``COMPARISON_ONLY_LIVE_CODE``,
    ``COMPARISON_CI_METHOD``, ``COMPARISON_CURVE_BAND`` and
    ``TEMPO_NOTEBOOK_SMOKE`` (read the smoke folders of the run notebooks under the scratch root). The boolean ones
    (``TEMPO_NOTEBOOK_SMOKE``, ``COMPARISON_USE_PREVIOUS``, ``COMPARISON_ONLY_LIVE_CODE``) are parsed strictly, as in the
    run notebooks: 1/true/yes/on, 0/false/no/off or unset / empty; any other value raises ValueError before anything is
    read or written (a typo must not turn a smoke check into a real run that rewrites ``Comparison/``)."""
    notes = []
    smoke, smoke_note = smoke_from_env(smoke)
    if smoke_note:
        notes.append(smoke_note)
    d_b, d_t, d_o, d_p = (_env_path(n) for n in ("COMPARISON_BASELINES_DIR", "COMPARISON_TEMPO_DIR",
                                                 "COMPARISON_OUT_DIR", "COMPARISON_PREVIOUS_DIR"))
    use_previous = env_flag("COMPARISON_USE_PREVIOUS", use_previous_baselines)
    only_live_code = env_flag("COMPARISON_ONLY_LIVE_CODE", only_live_code)
    if os.environ.get("COMPARISON_CI_METHOD", "").strip():
        ci_method = os.environ["COMPARISON_CI_METHOD"].strip()
        notes.append(f"COMPARISON_CI_METHOD sets CI_METHOD = {ci_method!r}")
    if os.environ.get("COMPARISON_CURVE_BAND", "").strip():
        curve_band = os.environ["COMPARISON_CURVE_BAND"].strip()
        notes.append(f"COMPARISON_CURVE_BAND sets CURVE_BAND = {curve_band!r}")
    try:
        curve_band = fs.check_band(curve_band)
    except ValueError as e:
        raise ValueError(f"CURVE_BAND: {e}") from None
    if os.environ.get("COMPARISON_N_BOOT", "").strip():
        n_boot = int(os.environ["COMPARISON_N_BOOT"])
        notes.append(f"COMPARISON_N_BOOT sets n_boot = {n_boot}")
    if os.environ.get("COMPARISON_TASKS", "").strip():
        tasks = [t.strip() for t in os.environ["COMPARISON_TASKS"].split(",") if t.strip()]
        notes.append(f"COMPARISON_TASKS selects {len(tasks)} tasks")
    for name, v in (("COMPARISON_BASELINES_DIR", d_b), ("COMPARISON_TEMPO_DIR", d_t), ("COMPARISON_OUT_DIR", d_o),
                    ("COMPARISON_PREVIOUS_DIR", d_p)):
        if v is not None:
            notes.append(f"{name} = {v}")

    def sources(explicit, param, folder, suite):
        sname = "smoke_" + suite if smoke and not suite.startswith("smoke_") else suite
        root = explicit if explicit is not None else (pathlib.Path(param).expanduser() if param else None)
        if root is not None:                                      # an explicit folder wins, also in smoke mode
            return [(root, sname)]
        if smoke:                                                 # one results folder per task under the scratch root
            found = [(d / "results", sname) for d in sorted(scratch_root().glob(f"{folder}_*"))
                     if (d / "results" / sname).is_dir()]
            if not found:
                notes.append(f"SMOKE: no smoke results under {scratch_root()}/{folder}_*/results/{sname} yet")
            return found
        return [(ROOT / folder / "results", suite)]

    base_src = sources(d_b, baselines_results, "Baselines_test", baselines_suite)
    tempo_src = sources(d_t, tempo_results, "TEMPO_test", tempo_suite)
    prev = None
    if use_previous:
        prev = ((d_p or (pathlib.Path(previous_results).expanduser() if previous_results else ROOT / "benchmarks" / "results")),
                PREVIOUS_SUITE)
    if smoke and seeds is not None and tuple(seeds) == FINAL_SEEDS:      # the run notebooks' smoke mode: first seed only
        seeds = (FINAL_SEEDS[0],)
        notes.append(f"SMOKE: only seed {FINAL_SEEDS[0]} is expected (first seed of the smoke runs)")
    all_tasks = list(DEFAULT_TASKS)
    if not tasks:
        notes.append(f"left out of the comparison (complab.LEFT_OUT): {', '.join(LEFT_OUT)}, which the run notebooks still run; "
                     "TASKS = [...] naming it includes it")
    tasks = all_tasks if not tasks else list(tasks)
    bad = [t for t in tasks if t not in TASK_ORDER]
    if bad:
        notes.append(f"tasks not in the benchmark plan (kept, family 'other'): {', '.join(bad)}")
    tasks = sorted(tasks, key=R.task_key)
    baselines = list(BASELINES) if baselines is None else [b for b in BASELINES if b in baselines]
    if str(ci_method) not in ("bootstrap", "t"):
        raise ValueError(f"CI_METHOD must be 'bootstrap' or 't', not {ci_method!r}")
    if out_dir is not None or d_o is not None:
        out = pathlib.Path(d_o or out_dir).expanduser()
    elif smoke:
        out = scratch_root() / "Comparison"
    else:
        out = ROOT / "Comparison"
    return Config(baseline_sources=base_src, tempo_sources=tempo_src, previous_source=prev, tasks=tasks,
                  baselines=baselines, tempo_variants=None if tempo_variants is None else list(tempo_variants),
                  main_variant=main_variant, seeds=None if seeds is None else tuple(int(s) for s in seeds),
                  n_boot=int(n_boot), formats=tuple(formats), out_dir=out, save=bool(save), smoke=bool(smoke),
                  use_previous=bool(use_previous), agg_tasks=agg_tasks, level_plot=level_plot, reference=reference,
                  fail_level=fail_level, collapse_level=collapse_level, show_info_checks=show_info_checks,
                  min_gap=float(min_gap), min_n_test=int(min_n_test), ci_method=str(ci_method), only_live_code=bool(only_live_code),
                  archive_stale=bool(archive_stale), curve_band=curve_band, notes=notes)


# --------------------------------------------------------------------------- loading
def _check_result(r) -> str | None:
    """Why a parsed result file cannot be used (None = fine)."""
    if not isinstance(r, dict):
        return "not a JSON object"
    j = r.get("job")
    if not isinstance(j, dict):
        return "no 'job' record"
    for k in ("env", "algo", "steps", "eval_freq", "n_eval", "seed"):
        if k not in j:
            return f"job has no '{k}'"
    try:
        int(j["steps"]), int(j["eval_freq"]), int(j["n_eval"]), int(j["seed"])
    except (TypeError, ValueError):
        return "job fields are not numbers"
    steps, rets = r.get("steps"), r.get("returns")
    if not isinstance(steps, list) or not isinstance(rets, list) or not steps:
        return "no learning curve"
    if len(steps) != len(rets):
        return f"{len(steps)} evaluation steps but {len(rets)} return lists"
    try:
        flat = np.concatenate([np.asarray(e, float).ravel() for e in rets])
    except (TypeError, ValueError):
        return "returns are not numbers"
    if flat.size == 0 or any(np.asarray(e).size == 0 for e in rets):
        return "an evaluation has no episodes"
    if not np.all(np.isfinite(flat)):
        return "non-finite returns"
    return None


def _misplaced(f: pathlib.Path, suite_dir: pathlib.Path, job: dict) -> str | None:
    """Why a valid result file is not at the path of the run its record describes (None = it is), as
    ``runlab._load_task`` rules: ``<suite>/<env>/<algo>/<variant>/seed<k>.json`` of the record's job. A Finder copy
    (``seed10 copy.json``) or a renamed record (seed 10 saved as ``seed99.json``) would otherwise enter the statistics as
    a duplicate seed, and could even replace the real run of that seed."""
    home = (suite_dir / str(job["env"]) / str(job["algo"]) / str(job.get("variant") or "default")
            / f"seed{int(job['seed'])}.json")
    if f == home:
        return None
    return f"not at the path of its own run ({home.relative_to(suite_dir)}): a copy or renamed file"


@dataclass
class Scan:
    """What one result folder holds."""
    source: tuple
    runs: dict                                   # {env: {label: [Run]}} (benchmarks.report.load_suite)
    n_files: int = 0
    bad: list = field(default_factory=list)      # [(path, reason)] unreadable / invalid files (not used)
    skipped: list = field(default_factory=list)  # [(path, reason)] valid files without an evaluation inside the budget
    failed: list = field(default_factory=list)   # [dict] logs with a traceback and no result


def _failed_logs(results: pathlib.Path, suite: str) -> list:
    """Best effort: logs next to the results (``<root>/../logs/<suite>__<env>__<algo>__<variant>__s<k>.log``)
    that end in a traceback although no result exists (the scheduler leaves no result for a failed run)."""
    logs = results.parent / "logs"
    out = []
    if not logs.is_dir():
        return out
    for lf in sorted(logs.glob(f"{suite}__*__s*.log")):
        m = re.fullmatch(re.escape(suite) + r"__(.+?)__(.+?)__(.+?)__s(\d+)\.log", lf.name)
        if not m:
            continue
        env, algo, variant, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
        if (results / suite / env / algo / variant / f"seed{seed}.json").exists():
            continue
        try:
            with open(lf, "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 6000))
                tail = fh.read().decode("utf-8", "replace")
        except OSError:
            continue
        if "Traceback" in tail:
            last = [ln for ln in tail.strip().splitlines() if ln.strip()][-1][:160]
            out.append(dict(env=env, algo=algo, variant=variant, seed=seed, log=lf, error=last))
    return out


def scan_source(results: pathlib.Path, suite: str) -> Scan:
    """Load one ``<results>/<suite>`` folder with ``benchmarks.report.load_suite``.

    Files that cannot be used (truncated JSON, no curve, non-finite returns) are listed in ``Scan.bad``;
    when there are any, ``load_suite`` is run on a temporary symlink view of the good files so that
    the project's loader stays the only code that builds ``Run`` objects. Valid files that
    ``load_suite`` skips (every evaluation after the budget) are listed in ``Scan.skipped``."""
    root = pathlib.Path(results)
    sc = Scan(source=(root, suite), runs={})
    d = root / suite
    files = sorted(d.glob("*/*/*/seed*.json")) if d.is_dir() else []
    sc.n_files = len(files)
    good = []
    for f in files:
        try:
            rec = json.loads(f.read_text())
            reason = _check_result(rec) or _misplaced(f, d, rec["job"])
        except (OSError, ValueError) as e:
            reason = f"unreadable ({type(e).__name__})"
        (sc.bad.append((f, reason)) if reason else good.append(f))
    if not files:
        sc.failed = _failed_logs(root, suite)
        return sc
    if not sc.bad:
        sc.runs = R.load_suite(root, suite)
    else:
        with tempfile.TemporaryDirectory(prefix="complab_") as tmp:
            for f in good:
                link = pathlib.Path(tmp) / f.relative_to(root)
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(f.resolve())
            sc.runs = R.load_suite(pathlib.Path(tmp), suite) if good else {}
            for rows in sc.runs.values():
                for runs in rows.values():
                    for r in runs:
                        r.path = pathlib.Path(r.path).resolve()
    loaded = {pathlib.Path(r.path).resolve() for rows in sc.runs.values() for runs in rows.values() for r in runs}
    sc.skipped = [(f, "no evaluation at or before the budget") for f in good if f.resolve() not in loaded]
    sc.failed = _failed_logs(root, suite)
    return sc


def _t_interval(x) -> tuple:
    """Mean and 95% t-interval over seeds (CI is NaN for n < 2): the ``CI_METHOD = "t"`` alternative to the percentile bootstrap."""
    x = np.asarray(x, float)
    if x.size == 0:
        return (math.nan,) * 3
    m = float(x.mean())
    if x.size < 2:
        return m, math.nan, math.nan
    from scipy import stats
    h = float(stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / math.sqrt(x.size))
    return m, m - h, m + h


@dataclass
class Data:
    """The merged runs of both result folders and everything derived from them once."""
    cfg: Config
    runs: dict                        # {env: {label: [Run]}}, seeds filtered, may hold empty envs
    origin: dict                      # {(env, label): "baselines_test" | "tempo_test" | "previous"}
    excluded: list                    # [dict(reason, ...)] result files that were read but not used
    scans: list                       # [Scan]
    tasks: list
    main: str                         # label of the headline TEMPO configuration
    tempo_labels: list                # main first, then the other variants found
    labels: list                      # columns: TEMPO labels then the selected baselines
    randoms: dict
    cells: dict                       # {env: {label: report.Cell}} for cells with runs
    norm: dict                        # report.normalisation
    written: list = field(default_factory=list)
    #: every TEMPO label of the TEMPO result folders on disk (loaded or not; as runlab collects them), sorted by report.label_key:
    #: the label set of figstyle.tempo_style, so a variant gets the same colour slot here as in the TEMPO_test figures
    palette_labels: list = field(default_factory=list)
    _curves: dict = field(default_factory=dict)
    _extra: dict = field(default_factory=dict)

    # ---- access
    def rows(self, env: str) -> dict:
        return {l: rs for l, rs in self.runs.get(env, {}).items() if rs}

    def n(self, env: str, label: str) -> int:
        return len(self.runs.get(env, {}).get(label, []))

    @property
    def has_data(self) -> bool:
        return any(rs for rows in self.runs.values() for rs in rows.values())

    @property
    def has_main(self) -> bool:
        return any(self.main in self.rows(e) for e in self.tasks)

    def expected_labels(self, env: str) -> list:
        """Columns the plan expects on a task: the headline TEMPO and the baselines of its action space."""
        pool = expected_pool(env)
        return [self.main] + [b for b in self.cfg.baselines if b in pool]

    def status(self, env: str, label: str) -> str:
        """complete | partial | pending (expected, no run) | extra (run that the plan does not expect) | n/a | none."""
        k = self.n(env, label)
        if label in self.expected_labels(env):
            return "pending" if k == 0 else "complete" if k >= self.cfg.n_expected else "partial"
        if label in BASELINES:
            return "extra" if k else "n/a"
        return "extra" if k else "none"

    def is_previous(self, env: str, label: str) -> bool:
        return self.origin.get((env, label)) == "previous"

    def untuned_baselines(self, env: str) -> list:
        """Baselines of the task whose runs record ``zoo_tuned = False`` (they ran untuned, with library defaults)."""
        return [l for l, rs in self.rows(env).items() if l in BASELINES
                and any(r.info.get("zoo_tuned") is False for r in rs)]

    def is_untuned(self, env: str) -> bool:
        return bool(self.untuned_baselines(env))

    def task_name(self, env: str) -> str:
        """Task name for tables: ``<task> (untuned)`` when some or all of its baselines ran with library defaults."""
        return env + (" (untuned)" if self.is_untuned(env) else "")

    def disp(self, env: str, label: str) -> str:
        return label + ("+" if self.is_previous(env, label) else "")

    def curve(self, env: str, label: str):
        """(steps, mean, lo, hi) of the learning curve: report.curve_band, computed once."""
        key = (env, label)
        if key not in self._curves:
            self._curves[key] = R.curve_band(self.runs[env][label], self.cfg.n_curve_boot, key)
        return self._curves[key]

    def family(self, env: str) -> str:
        return R.family(env)

    def budgets(self, env: str) -> list:
        """The distinct budgets of the task's runs (several = protocol mismatch); the expected budget when it has none."""
        b = sorted({int(r.budget) for rs in self.rows(env).values() for r in rs})
        return b or [self.cfg.protocol[0]]

    def budget(self, env: str) -> int:
        """The task's budget from its runs (the largest when they differ; x-axis limit, budget shares, steps-to-level)."""
        return self.budgets(env)[-1]

    def eval_freq(self, env: str) -> int:
        f = [int(r.eval_freq) for rs in self.rows(env).values() for r in rs]
        return max(f) if f else self.cfg.protocol[1]

    def budget_text(self, env: str) -> str:
        """``500k``, or ``300k/500k`` when the runs of the task have different budgets."""
        return "/".join(R.fmt_steps(b) for b in self.budgets(env))


def pick_main(found: Counter, main_variant: str | None) -> str:
    """Label of the headline TEMPO: the requested variant; else ``TEMPO`` (variant default) when it has
    runs or nothing else exists; else the variant with the most runs."""
    if main_variant:
        return R.label_of("TEMPO", main_variant)
    if not found or "TEMPO" in found:
        return "TEMPO"
    return found.most_common(1)[0][0]


def load_data(cfg: Config, verbose: bool = True) -> Data:
    """Read the baseline and TEMPO folders (and, if asked, the previous baselines), merge, filter and
    compute the per-task statistics once. Never raises on missing or partial data."""
    runs = defaultdict(lambda: defaultdict(list))
    origin, excluded, scans, seen = {}, [], [], set()
    skipped_tasks = Counter()

    def add(r, org, role):
        label = r.label
        why = None
        if role == "baseline" and (r.algo not in BASELINES or r.variant != "default"):
            why = "not a baseline of the protocol (expected algorithm in SAC/TD3/CrossQ/PPO/DQN with variant default)"
        elif role == "tempo" and r.algo != "TEMPO":
            why = "not a TEMPO run (found in the TEMPO folder)"
        elif cfg.tasks and r.env not in cfg.tasks:
            skipped_tasks[r.env] += 1
            return
        elif role in ("baseline", "previous") and r.algo not in cfg.baselines:
            why = "baseline not selected (BASELINES)"
        elif role == "tempo" and cfg.tempo_variants is not None and r.variant not in cfg.tempo_variants:
            why = "TEMPO variant not selected (TEMPO_VARIANTS)"
        elif (role == "tempo" and cfg.only_live_code and live_source_hash() and r.tempo_source
              and r.tempo_source != live_source_hash()):
            why = (f"made with older tempo_rl code (hash {r.tempo_source}) than the live library ({live_source_hash()}); "
                   "ONLY_LIVE_CODE = True")
        elif cfg.seeds is not None and r.seed not in cfg.seeds:
            why = f"seed {r.seed} not in SEEDS {tuple(cfg.seeds)}"
        elif (r.env, label, r.seed) in seen:
            why = "duplicate (same task, algorithm, variant and seed already loaded)"
        if why:
            excluded.append(dict(reason=why, env=r.env, label=label, seed=r.seed, path=r.path))
            return
        seen.add((r.env, label, r.seed))
        runs[r.env][label].append(r)
        origin[(r.env, label)] = org

    for roles, srcs, org in (("baseline", cfg.baseline_sources, "baselines_test"),
                             ("tempo", cfg.tempo_sources, "tempo_test")):
        for root, suite in srcs:
            sc = scan_source(root, suite)
            scans.append(sc)
            for rows in sc.runs.values():
                for rs in rows.values():
                    for r in rs:
                        add(r, org, roles)
    prev_note = None
    if cfg.previous_source:
        root, suite = cfg.previous_source
        sc = scan_source(root, suite)
        scans.append(sc)
        have = {(e, l) for e, rows in runs.items() for l, rs in rows.items() if rs}
        used, other = 0, []                                  # earlier runs at another protocol never fill a cell
        for rows in sc.runs.values():
            for l, rs in rows.items():
                for r in rs:
                    if r.algo in BASELINES and r.variant == "default" and (r.env, l) not in have \
                            and r.algo in R.expected_pool(r.env) and (not cfg.tasks or r.env in cfg.tasks):
                        if (r.budget, r.eval_freq, r.n_eval) != cfg.protocol_of(r.env):
                            other.append(r)
                            continue
                        used += 1
                        add(r, "previous", "previous")
        src = _short(root / suite)
        if other:
            bs, fs = sorted({r.budget for r in other}), sorted({r.eval_freq for r in other})
            rng = lambda v: R.fmt_steps(v[0]) + ("" if len(v) == 1 else "-" + R.fmt_steps(v[-1]))      # noqa: E731
            prev_note = ((f"USE_PREVIOUS_BASELINES: {used} earlier run(s) match and fill empty cells; " if used else
                          f"USE_PREVIOUS_BASELINES: no earlier result in {src} matches ")
                         + f"the expected protocol ({protocol_text(cfg.smoke, cfg.tasks)}): the earlier benchmark ran these tasks for "
                         f"{rng(bs)} steps, evaluating every {rng(fs)}, so {len(other)} earlier run(s) "
                         + ("are" if used else "were") + " not used" + ("" if used else " and no cell is filled") + ".")
        elif not used:
            prev_note = f"USE_PREVIOUS_BASELINES: {src} has no earlier baseline run for an empty cell; nothing filled."
    tasks = list(cfg.tasks)
    randoms = R.load_randoms(R.BENCH / "random_scores.json")
    found = Counter()
    for rows in runs.values():                       # runs per TEMPO label summed over all tasks
        for l, rs in rows.items():
            if R.is_tempo(l):
                found[l] += len(rs)
    main = pick_main(found, cfg.main_variant)
    tempo_labels = [main] + sorted((l for l in found if l != main), key=R.label_key)
    labels = tempo_labels + list(cfg.baselines)
    ordered = {e: {l: sorted(rs, key=lambda x: x.seed) for l, rs in sorted(runs[e].items(), key=lambda kv: R.label_key(kv[0]))}
               for e in sorted(runs, key=R.task_key)}
    nonempty = {e: {l: rs for l, rs in rows.items() if rs} for e, rows in ordered.items()}
    nonempty = {e: rows for e, rows in nonempty.items() if rows}
    cells = R.per_task(nonempty, cfg.n_boot) if nonempty else {}
    if cfg.ci_method == "t":
        for row in cells.values():
            for c in row.values():
                c.final, c.auc = _t_interval(c.finals), _t_interval(c.aucs)
    norm = R.normalisation(cells, randoms) if cells else {}
    for env, nz in norm.items():                 # guards on top of report.normalisation (see the module docstring)
        if not nz["valid"]:
            continue
        if env in P.TOY_TEXT_ALL:
            nz.update(valid=False, reason="toy-text task: judged against the exact optimum, never normalised")
            continue
        means = [c.final[0] for c in cells[env].values()]
        ok, why = reference_check(randoms, env, nz["best_baseline"], cfg.min_gap, max(means) if means else None)
        if not ok:
            nz.update(valid=False, reason=why)
    on_disk = set(tempo_labels)                      # + TEMPO variants whose runs were not loaded (filtered, old code, ...)
    for root, suite in cfg.tempo_sources:
        with contextlib.suppress(OSError):
            for d in (pathlib.Path(root) / suite).glob("*/TEMPO/*"):
                if d.is_dir() and any(d.glob("seed*.json")):
                    on_disk.add(R.label_of("TEMPO", d.name))
    D = Data(cfg=cfg, runs=ordered, origin=origin, excluded=excluded, scans=scans, tasks=tasks, main=main,
             tempo_labels=tempo_labels, labels=labels, randoms=randoms, cells=cells, norm=norm,
             palette_labels=sorted(on_disk, key=R.label_key))
    D._extra["skipped_tasks"] = dict(skipped_tasks)
    D._extra["tempo_counts"] = dict(found)
    D._extra["previous_note"] = prev_note
    if cfg.main_variant and main not in found:
        D._extra.setdefault("notes", []).append(f"headline TEMPO variant {main!r} has no runs yet")
    if verbose:
        print(f"loaded {sum(len(rs) for rows in ordered.values() for rs in rows.values())} runs "
              f"from {sum(1 for s in scans if s.n_files)} result folder(s); headline TEMPO label: {main}")
        if prev_note:
            print(prev_note)
    return D


# --------------------------------------------------------------------------- live code, small helpers
_LIVE: dict = {}


def live_source_hash():
    """Hash of the live ``tempo_rl/*.py`` (what a TEMPO run started now would record as ``tempo_source``); None if unavailable."""
    if "h" not in _LIVE:
        h = None
        try:
            nb = str(ROOT / "notebooks")
            if nb not in sys.path:
                sys.path.insert(0, nb)
            import benchlab
            h = str(benchlab.source_hash(ROOT))
        except Exception:                                   # noqa: BLE001
            try:
                import hashlib
                m = hashlib.sha256()
                for q in sorted((ROOT / "tempo_rl").glob("*.py")):
                    m.update(q.name.encode())
                    m.update(q.read_bytes())
                h = m.hexdigest()[:16]
            except Exception:                               # noqa: BLE001
                h = None
        _LIVE["h"] = h
    return _LIVE["h"]


def _pl(n: int, word: str, plural: str | None = None) -> str:
    """``1 task`` / ``2 tasks``."""
    return f"{n} {word if n == 1 else (plural or word + 's')}"


def action_space(env: str):
    """'discrete' / 'continuous' for the non-toy tasks of :data:`TASK_ORDER` (CarRacing-v3 and HumanoidStandup-v5: continuous), None
    for toy text (``plans.TOY_TEXT_ALL``: FrozenLake, CliffWalking, Blackjack, Taxi) and unknown tasks."""
    if env in P.TOY_TEXT_ALL or env not in TASK_ORDER:
        return None
    return "discrete" if env in P.DISCRETE else "continuous"


def _partly_untuned(D: Data, envs=None) -> dict:
    """{task: untuned baselines} of the tasks (default: the selected ones) where only SOME planned baselines are untuned (SAC and TD3
    on HumanoidStandup-v5 and CarRacing-v3: no RL-Zoo entry there, unlike CrossQ / PPO); :meth:`Data.is_untuned` marks the whole task.
    Decided from the plan (``benchmarks.zoo``), not from the baselines that already have runs, so that a task whose tuned baselines
    are still pending is not reported as wholly untuned."""
    from benchmarks.zoo import zoo_hyperparams

    out = {}
    for e in (D.tasks if envs is None else envs):
        planned = [l for l in BASELINES if l in expected_pool(e)]         # the task's whole pool, whatever BASELINES selects
        un = [l for l in planned if not zoo_hyperparams(l, e)[1]]
        if un and len(un) < len(planned) and D.untuned_baselines(e):
            out[e] = un
    return out


def _untuned_names(D: Data, envs) -> list:
    """The untuned tasks of ``envs``, a partly untuned one with its untuned baselines: ``HumanoidStandup-v5 (SAC, TD3)``."""
    part = _partly_untuned(D, envs)
    return [e + (f" ({', '.join(part[e])})" if e in part else "") for e in envs if D.is_untuned(e)]


def _partly_note(D: Data, envs=None) -> str:
    """'' or the sentence that explains a partly untuned task (its other baselines use RL-Zoo hyperparameters, but one untuned baseline
    marks the whole task: '(untuned)', dagger, outside the tuned-baselines scope)."""
    part = _partly_untuned(D, envs)
    return (" Partly untuned: " + "; ".join(f"{e} ({', '.join(v)} untuned, the other baselines tuned)" for e, v in part.items())
            + "; one untuned baseline marks the whole task.") if part else ""


def _na_text(D: Data, envs=None) -> str:
    """What ``n/a`` means: not run on this action space, and CrossQ on the pixel tasks (``plans.IMAGE``) among ``envs`` (default: the
    selected tasks), where sb3-contrib CrossQ cannot run (no CNN policy)."""
    img = [e for e in (D.tasks if envs is None else envs) if e in P.IMAGE] if "CrossQ" in D.cfg.baselines else []
    return "not run on this action space" + (f" (CrossQ on the pixel task {', '.join(img)}: sb3-contrib CrossQ has no CNN policy)"
                                             if img else "")


def _frame_skip_text(D: Data, env: str) -> str:
    """What an environment step of a task of :data:`FRAME_SKIP` is (without the task name)."""
    k, limit = FRAME_SKIP[env], ""
    try:                                                # the time limit counts frames: at most limit / k agent steps per episode
        import gymnasium as gym
        t = gym.spec(env).max_episode_steps
        limit = f" and an episode (time limit {t:,} frames) has at most {t // k:,} steps" if t else ""
    except Exception:                                   # noqa: BLE001
        pass
    return (f"one environment step = {k} simulator frames (frame-skip of the pixel pipeline that every algorithm and every evaluation "
            f"uses, rewards summed), so its {D.budget(env):,}-step budget is {k * D.budget(env):,} frames{limit}")


def _frame_skip_note(D: Data, envs) -> str:
    """'' or the sentence on the tasks of ``envs`` whose environment steps are agent steps of several frames (:data:`FRAME_SKIP`)."""
    fk = [e for e in envs if e in FRAME_SKIP]
    return ("; ".join(f"{e}: {_frame_skip_text(D, e)}" for e in fk) + ".") if fk else ""


def reference_check(D_randoms: dict, env: str, best: float, min_gap: float, best_any: float | None = None, use_threshold: bool = True):
    """(ok, reason): is the best baseline far enough above the random policy to be a usable normalisation reference?

    The reference range is the largest of |random|, solved threshold - random (gymnasium) and the best mean final score
    of ANY algorithm on the task - random; the best baseline has to cover ``min_gap`` of it. Otherwise a baseline that
    barely beats the random policy makes every normalised score astronomically large."""
    rnd = D_randoms.get(env)
    if rnd is None or min_gap <= 0:
        return True, ""
    scales = [abs(float(rnd))]
    thr = solved_threshold(env) if use_threshold else None
    if thr is not None and thr > rnd:
        scales.append(float(thr) - float(rnd))
    if best_any is not None and np.isfinite(best_any) and best_any > rnd:
        scales.append(float(best_any) - float(rnd))
    scale = max(scales)
    gap = float(best) - float(rnd)
    if scale > 0 and gap < min_gap * scale:
        return False, (f"reference too weak: the best baseline ({best:.4g}) covers only {max(gap, 0.0) / scale:.1%} of the reference range "
                       f"{scale:.4g} above the random policy ({rnd:.4g}); needs {min_gap:.0%} (MIN_BASELINE_GAP)")
    return True, ""


def holm(pvals) -> list:
    """Holm-Bonferroni adjusted p-values (NaN stay NaN)."""
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    idx = [i for i in np.argsort(p) if np.isfinite(p[i])]
    m = len(idx)
    run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * p[i]))
        out[i] = run
    return out.tolist()


# --------------------------------------------------------------------------- reference scores
_REF_CACHE: dict = {}


def solved_threshold(env: str):
    """gymnasium's ``reward_threshold`` of the task (None if it has none or gymnasium is missing)."""
    if env not in _REF_CACHE:
        try:
            import gymnasium as gym
            t = gym.spec(env).reward_threshold
            _REF_CACHE[env] = None if t is None else float(t)
        except Exception:                                   # noqa: BLE001
            _REF_CACHE[env] = None
    return _REF_CACHE[env]


_OPT_CACHE: dict = {}


def toy_optimum(env: str):
    """Exact optimal return of a toy-text task (``notebooks/benchlab.toy_text_optimum``), None if unavailable."""
    if env not in P.TOY_TEXT_ALL:
        return None
    if env not in _OPT_CACHE:
        try:
            nb = str(ROOT / "notebooks")
            if nb not in sys.path:
                sys.path.insert(0, nb)
            import benchlab
            _OPT_CACHE[env] = float(benchlab.toy_text_optimum(env))
        except Exception:                                   # noqa: BLE001
            _OPT_CACHE[env] = None
    return _OPT_CACHE[env]


_OPTEVAL_CACHE: dict = {}


def toy_optimal_eval(env: str, n_eval: int):
    """Mean return of an optimal policy on the benchmark's fixed evaluation episodes (``benchlab.toy_text_optimal_eval``)."""
    if env not in P.TOY_TEXT_ALL:
        return None
    if (env, n_eval) not in _OPTEVAL_CACHE:
        try:
            nb = str(ROOT / "notebooks")
            if nb not in sys.path:
                sys.path.insert(0, nb)
            import benchlab
            _OPTEVAL_CACHE[(env, n_eval)] = float(benchlab.toy_text_optimal_eval(env, int(n_eval)))
        except Exception:                                   # noqa: BLE001
            _OPTEVAL_CACHE[(env, n_eval)] = None
    return _OPTEVAL_CACHE[(env, n_eval)]


def references(D: Data, env: str) -> list:
    """[(value, short name, kind)] reference scores of a task: random policy, solved threshold, optimum."""
    out = []
    rnd = D.randoms.get(env)
    if rnd is not None:
        out.append((float(rnd), "random", "random"))
    thr = solved_threshold(env)
    if thr is not None:
        out.append((thr, "solved", "solved"))
    opt = toy_optimum(env)
    if opt is not None:
        out.append((opt, "optimum", "optimum"))
    return out


# =========================================================================== display helpers
def _in_ipython() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:                                       # noqa: BLE001
        return False


def show(obj) -> None:
    """Display a pandas object / figure / Markdown in the notebook (all rows and columns); print it otherwise."""
    try:
        import pandas as pd
        ctx = pd.option_context("display.max_rows", 500, "display.max_columns", 60, "display.width", 220,
                                "display.max_colwidth", 140)
    except Exception:                                       # noqa: BLE001
        ctx = contextlib.nullcontext()
    with ctx:
        if _in_ipython():
            from IPython.display import display
            display(obj)
        else:
            print(obj.data if type(obj).__name__ == "Styler" else obj)      # a styled table: its data outside a notebook


def show_md(text: str) -> None:
    """Render Markdown in the notebook (plain text elsewhere)."""
    if _in_ipython():
        from IPython.display import Markdown, display
        display(Markdown(text))
    else:
        print(text)


def _pct(a: int, b: int) -> str:
    return f"{a}/{b}" + (f" ({100 * a / b:.0f}%)" if b else "")


def _seed_text(seeds) -> str:
    s = sorted(int(x) for x in seeds)
    if not s:
        return "-"
    if len(s) > 2 and s == list(range(s[0], s[-1] + 1)):
        return f"{s[0]}-{s[-1]}"
    return ",".join(map(str, s))


# =========================================================================== 2. inventory and audit
def completeness(D: Data) -> dict:
    """How much of the planned comparison exists: runs found / expected for the headline TEMPO and for the
    baselines of every task, and the tasks that are complete (all expected cells have all seeds)."""
    n_exp = D.cfg.n_expected
    out = dict(main=[0, 0], baselines=[0, 0], tasks_complete=[], tasks_main_complete=[], tasks_main_any=[],
               tasks_baselines_complete=[], per_task={})
    for env in D.tasks:
        row = {}
        for lab in D.expected_labels(env):
            k = min(D.n(env, lab), n_exp)
            row[lab] = (k, n_exp)
            tgt = out["main"] if lab == D.main else out["baselines"]
            tgt[0] += k
            tgt[1] += n_exp
        out["per_task"][env] = row
        mk = row[D.main][0]
        base = [v for l, v in row.items() if l != D.main]
        if mk >= n_exp:
            out["tasks_main_complete"].append(env)
        if mk > 0:
            out["tasks_main_any"].append(env)
        if base and all(k >= n for k, n in base):
            out["tasks_baselines_complete"].append(env)
        if mk >= n_exp and all(k >= n for k, n in base):
            out["tasks_complete"].append(env)
    return out


def inventory_matrix(D: Data):
    """Task x algorithm table of seeds found / expected ('+' = baseline filled from the previous results)."""
    import pandas as pd

    n_exp = D.cfg.n_expected
    rows = []
    for env in D.tasks:
        r = {"task": env, "family": GROUPS.get(R.family(env), "Other"), "budget": D.budget_text(env)}
        for lab in D.labels:
            st, k = D.status(env, lab), D.n(env, lab)
            if st in ("complete", "partial"):
                r[lab] = f"{k}/{n_exp}" + (" +" if D.is_previous(env, lab) else "")
            elif st == "extra":
                r[lab] = f"{k}*"
            else:
                r[lab] = st if st in ("pending", "n/a") else "-"
        rows.append(r)
    return pd.DataFrame(rows).set_index("task")


def style_inventory(df):
    """Colour the inventory matrix (green complete, amber partial, red pending, grey n/a); plain df on failure."""
    try:
        def colour(v):
            v = str(v)
            if v == "pending":
                return "background-color: rgba(220,60,60,0.28)"
            if v in ("n/a", "-"):
                return "color: #898781"
            m = re.match(r"(\d+)/(\d+)", v)
            if m:
                return ("background-color: rgba(46,160,67,0.30)" if m.group(1) == m.group(2)
                        else "background-color: rgba(237,161,0,0.35)")
            return ""
        cols = [c for c in df.columns if c not in ("family", "budget")]
        return df.style.map(colour, subset=cols)
    except Exception:                                       # noqa: BLE001
        return df


def inventory_details(D: Data):
    """One row per (task, algorithm) with runs: seeds, folder of origin, devices, protocol, dates, wall-clock."""
    import pandas as pd

    rows = []
    for env in D.tasks:
        for lab, rs in D.rows(env).items():
            fin = sorted(r.finished for r in rs if r.finished)
            rows.append({
                "task": env, "algorithm": lab, "runs": len(rs), "seeds": _seed_text(r.seed for r in rs),
                "from": {"baselines_test": "Baselines_test", "tempo_test": "TEMPO_test",
                         "previous": "benchmarks/results/final (previous)"}.get(D.origin.get((env, lab)), "?"),
                "device": "+".join(sorted({r.device for r in rs})),
                "budget": "/".join(R.fmt_steps(b) for b in sorted({r.budget for r in rs})),
                "eval_freq": "/".join(R.fmt_steps(f) for f in sorted({r.eval_freq for r in rs})),
                "n_eval": "/".join(str(n) for n in sorted({r.n_eval for r in rs})),
                "evaluations": "/".join(sorted({str(len(r.steps)) for r in rs})),
                "first finished": fin[0][:10] if fin else "?", "last finished": fin[-1][:10] if fin else "?",
                "wall h/run": round(float(np.nanmean([r.wall_h for r in rs])), 3)})
    return pd.DataFrame(rows)


def code_versions(D: Data) -> dict:
    """Versions of the code that produced the results: TEMPO source hashes, SB3 versions, Python stack."""
    import pandas as pd

    t_rows, b_rows, stack = [], [], defaultdict(Counter)
    for env in D.tasks:
        for lab, rs in D.rows(env).items():
            for r in rs:
                for k in ("python", "torch", "gymnasium", "numpy"):
                    if r.versions.get(k):
                        stack[k][str(r.versions[k])] += 1
            if R.is_tempo(lab):
                for src in sorted({r.tempo_source for r in rs}):
                    sub = [r for r in rs if r.tempo_source == src]
                    live = live_source_hash()
                    t_rows.append(dict(task=env, label=lab, tempo_source=src or "?",
                                       live="" if not (live and src) else "yes" if src == live else "NO (older code)", runs=len(sub),
                                       code_root=", ".join(sorted({pathlib.Path(r.code_root).name or "?" for r in sub})),
                                       finished=max((r.finished for r in sub), default="")[:10]))
            else:
                for r in rs:
                    b_rows.append(dict(algorithm=lab, sb3=r.info.get("sb3_version", "?"),
                                       sb3_contrib=r.info.get("sb3_contrib_version", "-"),
                                       zoo_tuned=r.info.get("zoo_tuned", "?")))
    t = pd.DataFrame(t_rows)
    if len(t):
        t = (t.groupby(["label", "tempo_source", "live", "code_root"], as_index=False)
             .agg(tasks=("task", "nunique"), runs=("runs", "sum"), last_finished=("finished", "max")))
        t = t.rename(columns={"live": "same as live tempo_rl"})
    b = pd.DataFrame(b_rows)
    if len(b):
        b = b.groupby(["algorithm", "sb3", "sb3_contrib", "zoo_tuned"], as_index=False).size().rename(columns={"size": "runs"})
    s = pd.DataFrame([dict(package=k, versions=", ".join(f"{v} ({n} runs)" for v, n in c.most_common()))
                      for k, c in stack.items()])
    return dict(tempo=t, baselines=b, stack=s)


def protocol_audit(D: Data) -> list:
    """Cross-checks of the protocol per task. Returns [dict(severity, task, message)], ERROR first.

    ERROR: algorithms of one task differ in budget / evaluation frequency / episodes per evaluation (the
    comparison would not be like for like). WARN: a budget, evaluation schedule or number of episodes that differs
    from the expected protocol of the task (:meth:`Config.protocol_of`: the test protocol ``runlab.TEST_BUDGET`` /
    ``TEST_EVAL_FREQ`` with ``runlab.n_eval_for(task)`` episodes, the shrunk schedule in smoke mode), a TEMPO label that mixes
    tempo_rl sources, runs with evaluations after the budget (PPO overshoot; those evaluations are dropped like the
    project's report). TEMPO runs made with another tempo_rl source than the live library (older code) are a WARN too.
    INFO: smoke-sized runs, toy-text tasks, baselines taken from the previous results, differing seed sets, untuned baselines,
    frame-skipped pixel tasks."""
    out = []
    live = live_source_hash()

    def add(sev, env, msg):
        out.append(dict(severity=sev, task=env, message=msg))

    for env in D.tasks:
        rows = D.rows(env)
        if not rows:
            continue
        exp_b, exp_f, exp_n = D.cfg.protocol_of(env)
        trip = {l: sorted({(r.budget, r.eval_freq, r.n_eval) for r in rs}) for l, rs in rows.items()}
        allt = sorted({t for v in trip.values() for t in v})
        if len(allt) > 1:
            detail = "; ".join(f"{l}: " + " / ".join(f"budget {b:,}, every {f:,}, {n} ep" for b, f, n in v)
                               for l, v in trip.items())
            add("ERROR", env, f"PROTOCOL MISMATCH between runs of this task (budget / evaluation frequency / "
                              f"episodes differ): {detail}")
        if D.cfg.smoke:
            t_b, t_f, t_n = expected_protocol(False, env)
            add("INFO", env, "smoke-sized run(s) (" + "; ".join(f"budget {b:,}, evaluation every {f:,}, {n} episodes" for b, f, n in allt)
                + f"); the test protocol is {t_b:,} / {t_f:,} / {t_n}")
        for (b, f, n) in allt:
            labs = ", ".join(l for l, v in trip.items() if (b, f, n) in v)
            if b != exp_b or f != exp_f:
                add("WARN", env, f"{labs}: budget {b:,} / evaluation every {f:,} differs from the expected protocol "
                                 f"({exp_b:,} / {exp_f:,}" + (", the smoke schedule)" if D.cfg.smoke else
                                                              "; runlab.TEST_BUDGET / TEST_EVAL_FREQ)"))
            if n != exp_n:
                add("WARN", env, f"{labs}: {n} episodes per evaluation (expected protocol: {exp_n})")
        seeds = {l: {r.seed for r in rs} for l, rs in rows.items() if l in D.expected_labels(env)}
        stray = sorted({s for v in seeds.values() for s in v} - set(FINAL_SEEDS))
        if stray:
            add("WARN", env, f"seeds {stray} are not among the benchmark's seeds {FINAL_SEEDS[0]}-{FINAL_SEEDS[-1]} (SEEDS = None lets them in)")
        if len(seeds) > 1 and not set.intersection(*seeds.values()):
            add("WARN", env, "the algorithms share NO seed: " + "; ".join(f"{l} {_seed_text(s)}" for l, s in seeds.items())
                + " (the protocol uses the same seeds for every algorithm)")
        elif len({frozenset(s) for s in seeds.values()}) > 1:
            add("INFO", env, "seed sets differ: " + "; ".join(f"{l} {_seed_text(s)}" for l, s in seeds.items())
                + " (runs still missing, or a different number of seeds; the tests use whatever seeds exist per algorithm)")
        for lab, rs in rows.items():
            drop = [r for r in rs if r.n_dropped]
            if drop:
                add("WARN", env, f"{lab}: {len(drop)} of {len(rs)} runs ran past the budget (up to step "
                                 f"{max(r.raw_last_step for r in drop):,} for a budget of {max(r.budget for r in drop):,}); "
                                 f"{sum(r.n_dropped for r in drop)} evaluation(s) after the budget dropped (as the project's report does)")
            if R.is_tempo(lab) and len({r.tempo_source for r in rs}) > 1:
                add("WARN", env, f"{lab}: runs come from {len({r.tempo_source for r in rs})} different tempo_rl sources "
                                 f"({', '.join(sorted({r.tempo_source or '?' for r in rs}))})")
            if R.is_tempo(lab) and live:
                old = [r for r in rs if r.tempo_source and r.tempo_source != live]
                if old:
                    add("WARN", env, f"{lab}: {len(old)} of {len(rs)} runs were made with older tempo_rl code "
                                     f"({', '.join(sorted({r.tempo_source for r in old}))}) than the live library ({live})")
            if len({r.device for r in rs}) > 1:
                add("INFO", env, f"{lab}: seeds ran on different devices ({', '.join(sorted({r.device for r in rs}))})")
            vers = {(r.info.get("sb3_version"), r.info.get("sb3_contrib_version")) for r in rs if "sb3_version" in r.info}
            if len(vers) > 1:
                add("WARN", env, f"{lab}: mixed SB3 versions {sorted(vers)}")
            if D.is_previous(env, lab):
                add("INFO", env, f"{lab}: taken from the previous results (benchmarks/results/final), not from Baselines_test")
        if env in P.TOY_TEXT_ALL:
            add("INFO", env, "toy-text task: TEMPO is judged against the optimum (the exact one, or an optimal policy's lower score on "
                             "the same evaluation episodes) and the random policy; its baselines are references, not tested")
            bl = [l for l in rows if l in BASELINES]
            if bl:
                add("INFO", env, f"{'untuned ' if D.is_untuned(env) else ''}baseline runs exist here ({', '.join(bl)}): they are shown as references in the per-task tables "
                                 "and curves; toy-text tasks stay out of verdicts, win / tie / loss and every normalised statistic")
        elif env in _partly_untuned(D, [env]):
            add("INFO", env, f"baselines partly untuned ({', '.join(D.untuned_baselines(env))} ran with library defaults on this "
                             "task; the other baselines are tuned): the task counts as untuned ('(untuned)', dagger, outside the "
                             "tuned-baselines scope); a win here is a weaker claim than on a tuned task")
        elif D.is_untuned(env):
            add("INFO", env, f"baselines untuned (library defaults on this task: {', '.join(D.untuned_baselines(env))}); "
                             "a win here is a weaker claim than on a tuned task")
        if env in FRAME_SKIP:
            add("INFO", env, _frame_skip_text(D, env))
        nz = D.norm.get(env)
        if nz and env not in P.TOY_TEXT_ALL and not nz["valid"] and any(b in D.cells.get(env, {}) for b in BASELINES):
            add("INFO" if D.cfg.smoke else "WARN", env, "left out of every normalised statistic: " + str(nz.get("reason") or "not normalisable"))
    cnt = D._extra.get("tempo_counts", {})
    if cnt.get(D.main, 0) and len(cnt) > 1:
        top, k_top = max(cnt.items(), key=lambda kv: kv[1])
        if top != D.main and k_top >= 2 * cnt[D.main]:
            add("WARN", "(all)", f"the headline {D.main} has {_pl(cnt[D.main], 'run')} but {top} has {k_top}: set MAIN_VARIANT = "
                                 f"{top.split('[')[-1].rstrip(']')!r} to make that configuration the headline")
    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    return sorted(out, key=lambda x: (order[x["severity"]], R.task_key(x["task"]) if x["task"] in TASK_ORDER else (len(TASK_ORDER), x["task"])))


def automatic_checks(D: Data) -> list:
    """The project's heuristic scan for failed / diverged / suspicious runs (``report.data_checks``) on the
    merged data, without the lines that :func:`protocol_audit` already reports."""
    if not D.cells:
        return []
    nonempty = {e: D.rows(e) for e in D.cells}
    try:
        checks = R.data_checks(nonempty, D.cells, D.norm, D.randoms, "comparison")
    except Exception as e:                                  # noqa: BLE001
        return [("warn", f"automatic data checks failed: {type(e).__name__}: {e}")]
    skip = ("evaluation(s) after the budget", "different tempo_rl sources", "mixed SB3 versions")
    out = []
    for s, m in checks:
        if any(k in m for k in skip):
            continue
        if s == "warn" and "excluded from the aggregates (toy-text task" in m:
            s = "info"                                      # by design (toy text is never normalised), not a problem
        out.append((s, m))
    return out


def excluded_table(D: Data):
    """Result files that were read but not used (and why)."""
    import pandas as pd

    rows = [dict(reason=x["reason"], file=_short(x["path"])) for x in D.excluded]
    for sc in D.scans:
        rows += [dict(reason="unusable file: " + why, file=_short(f)) for f, why in sc.bad]
        rows += [dict(reason="file skipped: " + why, file=_short(f)) for f, why in sc.skipped]
        rows += [dict(reason=f"FAILED run (log ends in a traceback, no result): {x['error']}", file=_short(x["log"]))
                 for x in sc.failed]
    if D._extra.get("skipped_tasks"):
        rows.append(dict(reason="results of tasks that are not selected / not in the plan: "
                                + ", ".join(f"{e} ({n})" for e, n in D._extra["skipped_tasks"].items()), file=""))
    return pd.DataFrame(rows)


def _short(p) -> str:
    p = pathlib.Path(p)
    for base in (ROOT, scratch_root()):
        try:
            return str(p.relative_to(base))
        except ValueError:
            continue
    return str(p)


def dropped_table(D: Data):
    """Evaluations after the budget that were dropped (PPO with parallel environments overshoots the budget)."""
    import pandas as pd

    rows = []
    for env in D.tasks:
        for lab, rs in D.rows(env).items():
            for r in rs:
                if r.n_dropped:
                    rows.append(dict(task=env, algorithm=lab, seed=r.seed, budget=r.budget, last_step_run=r.raw_last_step,
                                     dropped_evaluations=r.n_dropped))
    return pd.DataFrame(rows)


def show_inventory(D: Data) -> dict:
    """Section 2: what data exist, where they come from, the protocol audit and the automatic checks."""
    _mark(D, "inventory")
    cp = completeness(D)
    n_exp = D.cfg.n_expected
    print(f"headline TEMPO label : {D.main}   (other TEMPO variants found: "
          f"{', '.join(l for l in D.tempo_labels if l != D.main) or 'none'})")
    print(f"TEMPO runs           : {_pct(*cp['main'])} expected runs found ({len(cp['tasks_main_any'])} of {len(D.tasks)} tasks "
          f"with runs, {len(cp['tasks_main_complete'])} with all {n_exp} seeds)")
    print(f"baseline runs        : {_pct(*cp['baselines'])} expected runs found ({len(cp['tasks_baselines_complete'])} of "
          f"{len(D.tasks)} tasks with every expected baseline complete)")
    print(f"complete comparisons : {len(cp['tasks_complete'])} of {len(D.tasks)} tasks (TEMPO and every expected baseline with all {n_exp} seeds; on toy text the PPO / DQN references count too)")
    if not D.has_data:
        print("\nNO RESULT FILES FOUND YET: every cell below is 'pending'. Run the notebooks in Baselines_test/ and "
              "TEMPO_test/, then run this notebook again.")
    inv = inventory_matrix(D)
    print(f"\nSeeds found / expected per task and algorithm  (pending = planned, no finished run; n/a = {_na_text(D)}; "
          "* = runs outside the plan; + = from the previous results):")
    show(style_inventory(inv))
    det = inventory_details(D)
    if len(det):
        print("\nOrigin, device, protocol and dates of every (task, algorithm) with runs:")
        show(det)
    cv = code_versions(D)
    for key, title in (("tempo", "TEMPO source versions (tempo_rl hash recorded in every result)"),
                       ("baselines", "Baseline library versions"), ("stack", "Python stack")):
        if len(cv[key]):
            print(f"\n{title}:")
            show(cv[key])
    audit = protocol_audit(D)
    errs = [a for a in audit if a["severity"] == "ERROR"]
    print(f"\nProtocol audit (budget, evaluation frequency, episodes per evaluation, seeds; expected on every task: "
          f"{protocol_text(D.cfg.smoke, D.tasks)}):")
    if errs:
        print("  >>> " + f"{len(errs)} PROTOCOL MISMATCH(ES): the affected tasks do not compare like with like <<<")
    for a in audit:
        if a["severity"] != "INFO":
            print(f"  [{a['severity']}] {a['task']}: {a['message']}")
    infos = [a for a in audit if a["severity"] == "INFO"]
    if infos:
        print(f"  {len(infos)} informational note(s):")
        for a in infos:
            print(f"  [INFO] {a['task']}: {a['message']}")
    if not audit:
        print("  nothing to report" if D.has_data else "  no data to audit")
    dr = dropped_table(D)
    if len(dr):
        print("\nEvaluations after the budget that were dropped (like the project's report):")
        show(dr)
    ex = excluded_table(D)
    if len(ex):
        print(f"\nFiles read but NOT used ({len(ex)}):")
        show(ex)
    checks = automatic_checks(D)
    shown = [(s, m) for s, m in checks if s != "info" or D.cfg.show_info_checks]
    print(f"\nAutomatic data checks (benchmarks.report.data_checks): {sum(s == 'error' for s, _ in checks)} error, "
          f"{sum(s == 'warn' for s, _ in checks)} warn, {sum(s == 'info' for s, _ in checks)} info"
          + ("" if D.cfg.show_info_checks else " (info lines hidden: SHOW_INFO_CHECKS = True shows them)"))
    for s, m in shown[:60]:
        print(f"  [{s}] {m}")
    if len(shown) > 60:
        print(f"  ... {len(shown) - 60} more")
    D._extra["audit"], D._extra["checks"], D._extra["completeness"] = audit, checks, cp
    return dict(completeness=cp, audit=audit, checks=checks)


# =========================================================================== 3. figures
# Every figure is drawn in the academic style of notebooks/figstyle.py (the single source of truth of the figures of Baselines_test,
# TEMPO_test and Comparison: serif print fonts at 8-9 pt, paper widths SINGLE / DOUBLE, thin axes without top/right spines, a light
# horizontal grid, one shared frameless legend outside the panels, vector PDF with TrueType fonts + 300-dpi PNG). Algorithm colours,
# dashes and widths are benchmarks.report.STYLE; explanations go in the caption printed under each figure, not in the panels.
INK, INK2, MUTED, GRID = fs.INK, fs.INK2, fs.MUTED, fs.GRID
REF_STYLE = fs.REFERENCE                            # random policy / solved threshold / known optimum: neutral grey, own dashes
#: Reference levels figstyle has no kind for (same neutral grey, their own patterns): the best baseline's level (solid, thin; 1 on
#: the normalised scale) and the chance level of a probability of improvement.
LEVEL_STYLE = {"best": dict(color=MUTED, ls="-", lw=0.8, label="Best baseline"),
               "chance": dict(color=MUTED, ls=(0, (3, 1.5)), lw=0.8, label="Chance level (0.5)")}
_REF_ORDER = ("random", "solved", "optimum", "best", "chance")
#: Colours of the TEMPO variants are NOT defined here: style_of asks figstyle.tempo_style (figstyle.VARIANT_STYLES, the one rule that
#: runlab uses too), so a variant has the same colour and dash pattern in TEMPO_test and in this comparison.
_CI_LW, _CI_ALPHA = 2.6, 0.45                       # interval marks of dot / interval plots (thin translucent bar)
_TICK_MS, _TICK_MEW = 8.0, 1.4                      # the point estimate ("|") of dot / interval plots
_BAR_ALPHA = 0.6                                    # the few remaining bars (thin, translucent: no saturated blocks)
_NOTE_SIZE = fs.FONT_SIZE                           # in-panel notes ("pending", "n=3", values): the 8 pt print size


def _plt():
    import matplotlib
    if not _in_ipython():
        with contextlib.suppress(Exception):
            matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


@contextlib.contextmanager
def figure_style(extra: dict | None = None):
    """The academic style of every figure (``figstyle.paper()``; ``extra`` = rcParams overrides); yields pyplot."""
    plt = _plt()
    with fs.paper(**(extra or {})):
        yield plt


def style_of(D: Data, label: str) -> dict:
    """Colour / dash / width / z of an algorithm: ``figstyle.tempo_style`` (the rule of every folder): ``report.STYLE`` for the six
    algorithms, the headline TEMPO (``D.main``) always in TEMPO's red, every other TEMPO variant a slot of ``figstyle.VARIANT_STYLES``
    by its position among all TEMPO labels of the TEMPO result folders (``D.palette_labels``, the label set runlab passes too), so
    it keeps its colour from task to task and matches the TEMPO_test figures."""
    return fs.tempo_style(label, D.main, D.palette_labels or D.tempo_labels)


def _colour_ok(D: Data, label: str) -> bool:
    """False for a TEMPO variant whose style is not one of the slots of ``figstyle.VARIANT_STYLES`` (a variant past the last slot,
    which figstyle draws in a repeated or lightened hue): figures whose marks are told apart by colour alone leave it out (it is
    drawn in section 9 and in the labelled dot / interval rows)."""
    if not R.is_tempo(label) or label == D.main:
        return True
    st = style_of(D, label)
    return any(st.get("color") == v["color"] and st.get("ls") == v["ls"] for v in fs.VARIANT_STYLES)


def _no_colour_note(where: str, labels, see: str = "section 9") -> str:
    """Caption sentence for the TEMPO variants that a figure telling its marks apart by colour alone leaves out (:func:`_colour_ok`)."""
    return (f"{where} (TEMPO variants past the {len(fs.VARIANT_STYLES)} variant colours of figstyle.VARIANT_STYLES, so not told apart "
            f"by colour; see {see}): " + ", ".join(labels) + ".")


def _headline_only(D: Data):
    """``only_fn`` of the curve grids: the baselines and the headline TEMPO (other TEMPO variants are compared in section 9)."""
    return lambda env: [l for l in D.rows(env) if not R.is_tempo(l) or l == D.main]


_VARIANT_NOTE = "Other TEMPO variants: section 9."


def _line_width(D: Data, label: str) -> float:
    """Width of an algorithm's line outside :func:`figstyle.curve` (performance profiles, legend keys): the rule of figstyle.curve."""
    if label == D.main:
        return fs.TEMPO_LINE_WIDTH
    return min(max(fs.LINE_WIDTH, float(style_of(D, label).get("lw", fs.LINE_WIDTH))), fs.TEMPO_LINE_WIDTH - 0.3)


def _ref_style(kind: str) -> dict:
    return REF_STYLE[kind] if kind in REF_STYLE else LEVEL_STYLE[kind]


def ref_line(ax, value: float, kind: str, label: str | None = None, vertical: bool = False):
    """A labelled reference line in neutral grey: ``kind`` = random / solved / optimum (``figstyle.reference_line``) or best / chance
    (:data:`LEVEL_STYLE`); ``vertical=True`` for dot / interval plots whose values run along x."""
    if kind in REF_STYLE and not vertical:
        return fs.reference_line(ax, value, kind, label)
    st = _ref_style(kind)
    f = ax.axvline if vertical else ax.axhline
    return f(float(value), color=st["color"], ls=st["ls"], lw=st["lw"], zorder=1.2, label=st["label"] if label is None else label)


def _ref_handles(kinds, labels: dict | None = None) -> list:
    """Legend entries of the reference lines drawn (``kinds``), in a fixed order; ``labels`` = {kind: text} overrides."""
    from matplotlib.lines import Line2D

    out = []
    for k in _REF_ORDER:
        if k in kinds:
            st = _ref_style(k)
            out.append((Line2D([], [], color=st["color"], ls=st["ls"], lw=st["lw"]), (labels or {}).get(k, st["label"])))
    return out


def _legend_handles(D: Data, labels, kind: str = "line", with_n: dict | None = None) -> list:
    """[(handle, label)] per algorithm for figures whose marks are not figstyle curves: ``kind`` = line (dash pattern),
    marker (dot) or patch (bar)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    out = []
    for l in labels:
        st = style_of(D, l)
        name = ("Best baseline*" if l == BEST else l) + (f" (n={with_n[l]})" if with_n and l in with_n else "")
        if kind == "marker":
            h = Line2D([], [], ls="", marker="o", ms=fs.MARKER_SIZE + (1.5 if l == D.main else 0.5), color=st["color"],
                       mec="white", mew=0.4)
        elif kind == "patch":
            h = Patch(facecolor=st["color"], alpha=_BAR_ALPHA, lw=0)
        else:
            h = Line2D([], [], color=st["color"], ls=st["ls"], lw=_line_width(D, l))
        out.append((h, name))
    return out


def _mark_handles(ci_text: str) -> list:
    """Legend key of the dot plots: a seed, the mean, the confidence interval (neutral ink)."""
    from matplotlib.lines import Line2D

    return [(Line2D([], [], ls="", marker="o", ms=fs.MARKER_SIZE, color=INK2, mec="white", mew=0.4), "Seed"),
            (Line2D([], [], ls="", marker="|", ms=_TICK_MS, mew=_TICK_MEW, color=INK2), "Mean over seeds"),
            (Line2D([], [], color=INK2, lw=_CI_LW, alpha=_CI_ALPHA, solid_capstyle="butt"), ci_text)]


def _finish(D: Data, fig, stem: str, subdir: str = "", display: bool = True, caption: str | None = None) -> list:
    """Save (``figstyle.save``: vector PDF with TrueType fonts + 300-dpi PNG, only when there are data and saving is on), show the
    saved PNG inline (exactly the file; a figure that is not saved is rendered the same way), print the caption, close."""
    plt = _plt()
    paths = []
    if D.cfg.save and D.has_data:
        folder = D.cfg.fig_dir / subdir if subdir else D.cfg.fig_dir
        paths = fs.save(fig, folder / stem, formats=D.cfg.formats)
        D.written.extend(paths)
    if display:
        png = next((p for p in paths if p.suffix == ".png"), None)
        fs.show(png if png is not None else fig)
        if caption:
            print(fs.caption(caption))
    plt.close(fig)
    return paths


def _balanced_cols(n: int, max_cols: int = 4) -> int:
    """Columns of a grid of ``n`` panels: at most ``max_cols`` (a DOUBLE-width panel stays >= ~1.6 in wide with 4 columns, ~1.4 in
    with 5) and rows as full as possible (5 panels: 3 + 2 rather than 4 + 1)."""
    if n <= 0:
        return 1
    return math.ceil(n / math.ceil(n / max_cols))


def _grid_shape(n: int, ncols: int) -> tuple:
    ncols = max(1, min(ncols, n))
    return math.ceil(n / ncols), ncols


def _grid_axes(n: int, max_cols: int = 4, panel_aspect: float = 0.72, extra: float = 0.8, **kw):
    """``(fig, axes, ncols)``: a constrained-layout grid of ``n`` panels at a paper width (SINGLE for one panel, DOUBLE otherwise),
    panels of ``panel_aspect`` (height / width); unused cells are removed. Call inside :func:`figure_style`."""
    nrows, ncols = _grid_shape(n, _balanced_cols(n, max_cols))
    width = fs.SINGLE if ncols == 1 else fs.DOUBLE
    _, h = fs.grid_figsize(nrows, ncols, width, panel_aspect, extra)
    fig, axes = fs.subplots(nrows, ncols, width=width, height=h, squeeze=False, **kw)
    for ax in axes.flat[n:]:
        ax.remove()
    return fig, axes, ncols


def _grid_labels(fig, axes, n: int, ncols: int, xlabel: str, ylabel: str, last_row_only: bool = False) -> None:
    """x label under the lowest panel of every column (``last_row_only``: under the panels of the last row only, e.g. when the
    legend fills the rest of that row), one y label for the figure (one panel: on the axes)."""
    if n == 1:
        axes.flat[0].set_xlabel(xlabel)
        axes.flat[0].set_ylabel(ylabel)
        return
    for j in range(ncols):
        col = [axes.flat[i] for i in range(j, n, ncols)]
        if col and (not last_row_only or (n - 1) // ncols == (j + (len(col) - 1) * ncols) // ncols):
            col[-1].set_xlabel(xlabel)
    fig.supylabel(ylabel, fontsize=fs.LABEL_SIZE)


def _norm_ticks(axes, axis: str = "y", nbins: int = 4) -> None:
    """One tick rule for the normalised-score (or fraction) axes of a figure, called once the limits are final: at most ``nbins``
    intervals of 1 / 2 / 5 x 10^k and the SAME number of decimals on every panel (at least one: 0.0, 0.5, 1.0; 0.00, 0.05, ... only
    when a panel needs it), with a true minus sign."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    axs = [a for a in np.ravel(np.asarray(axes, dtype=object)) if a is not None]
    dec = 1
    for ax in axs:
        a = ax.yaxis if axis == "y" else ax.xaxis
        loc = MaxNLocator(nbins, steps=[1, 2, 5, 10])
        a.set_major_locator(loc)
        lo, hi = sorted(a.get_view_interval())
        t = [v for v in loc.tick_values(lo, hi) if lo - 1e-9 <= v <= hi + 1e-9]
        if len(t) > 1:
            dec = max(dec, -math.floor(math.log10(min(np.diff(t))) + 1e-9))
    fmt = FuncFormatter(lambda v, _pos: fs.num(0.0 if abs(v) < 1e-12 else v, f".{dec}f"))
    for ax in axs:
        (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def best_baseline(D: Data, env: str, metric: str = "final"):
    """(label, mean) of the baseline with the highest mean ``metric`` ('final' or 'auc') on the task. Toy-text tasks
    have none: baseline runs there (PPO / DQN) are references that stay out of every comparison."""
    row = D.cells.get(env, {})
    if env in P.TOY_TEXT_ALL:
        return None, math.nan
    c = {b: getattr(row[b], metric)[0] for b in BASELINES if b in row}
    if not c:
        return None, math.nan
    b = max(c, key=c.get)
    return b, c[b]


def outcome(D: Data, a, b):
    """``report.outcome`` (Welch, alpha 0.05) with ``cfg.min_n_test`` seeds required on each side ('n/a' otherwise)."""
    if len(a) < D.cfg.min_n_test or len(b) < D.cfg.min_n_test:
        return "n/a", math.nan
    return R.outcome(a, b)


def main_vs_best(D: Data, env: str, metric: str = "final"):
    """Welch outcome of the headline TEMPO against the best baseline on the metric:
    (best label, 'win'|'tie'|'loss'|'n/a'|None, p, tempo mean, best mean)."""
    row = D.cells.get(env, {})
    b, bm = best_baseline(D, env, metric)
    if b is None or D.main not in row:
        return b, None, math.nan, math.nan, bm
    f = "finals" if metric == "final" else "aucs"
    res, p = outcome(D, getattr(row[D.main], f), getattr(row[b], f))
    return b, res, p, getattr(row[D.main], metric)[0], bm


def _outcome_text(D: Data, env: str, metric: str = "final") -> str:
    """'win vs CrossQ (p = 0.012)' or a pending / n/a note (for captions)."""
    row = D.rows(env)
    if D.main not in row:
        return f"{D.main} pending"
    if env in P.TOY_TEXT_ALL:
        return "toy text: baselines not tested (judged against the exact optimum)"
    b, res, p, _, _ = main_vs_best(D, env, metric)
    if b is None:
        return "baselines pending"
    un = ", untuned baselines" if D.is_untuned(env) else ""
    if res == "n/a":
        return f"vs {b} untested (fewer than {D.cfg.min_n_test} seeds){un}"
    ps = "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
    return f"{res} vs {b} ({ps}){un}"


def _pending_note(D: Data, env: str) -> str:
    miss = [l for l in D.expected_labels(env) if D.n(env, l) == 0]
    return ", ".join(miss)


def _budget_flag(D: Data, env: str) -> str:
    """'' when the task's runs share the expected budget; else a short warning (see the protocol audit)."""
    bs = D.budgets(env)
    if len(bs) > 1:
        return "budgets differ"
    if D.rows(env) and bs[0] != D.cfg.protocol[0]:
        return f"budget {R.fmt_steps(bs[0])}, expected {R.fmt_steps(D.cfg.protocol[0])}"
    return ""


#: Font weight of the task-name panel titles (bold task names in every comparison figure)
TASK_TITLE_WEIGHT = "bold"


def _task_title(D: Data, env: str, short: bool = False) -> str:
    """Panel title: the task name (``short``: without the version suffix, for narrow panels), with a dagger when its budget is not
    the expected one (explained in the caption); drawn in :data:`TASK_TITLE_WEIGHT`."""
    return (re.sub(r"-v\d+$", "", env) if short else env) + (" †" if _budget_flag(D, env) else "")


def _band_choice(D: Data, band) -> str:
    return D.cfg.curve_band if band is None else fs.check_band(band)


def _join(items, limit: int = 12) -> str:
    items = list(items)
    return "; ".join(items[:limit]) + (f"; and {len(items) - limit} more" if len(items) > limit else "")


def _band_sentence(D: Data, pairs, band: str, seeds: bool, clipped=()) -> str:
    """The caption sentence of the line and the band (``figstyle.band_caption``) for the (task, label) curves drawn, naming the
    curves with fewer seeds than the others. ``clipped``: the panels (task names) where :func:`draw_curves` clipped a band at the axes
    (True for a one-panel figure): figstyle's clipping sentence, plus the panels of a grid."""
    ns = {k: D.n(*k) for k in pairs}
    if not ns:
        return ""
    count = Counter(ns.values())
    n_ref = max(count, key=lambda k: (count[k], k))
    ids = sorted({r.seed for (e, l) in pairs for r in D.runs[e][l]})
    same = all(sorted(r.seed for r in D.runs[e][l]) == ids for (e, l) in pairs)
    s = fs.band_caption(band, n_ref, seeds, _seed_text(ids) if same else None, clipped=bool(clipped))
    if clipped and not isinstance(clipped, bool):
        s += " Clipped on: " + _join(clipped) + "."
    odd = [f"{e} {l} ({k}{', line only' if k == 1 and band != 'none' else ''})" for (e, l), k in ns.items() if k != n_ref]
    if odd:
        s += " Other numbers of seeds: " + _join(odd) + "."
    return s


def _band_switch(D: Data, band: str) -> str:
    other = [b for b in fs.BANDS if b != band]
    return (f"Band: CURVE_BAND = {band!r}" + (" (default)" if band == fs.DEFAULT_BAND else "")
            + f"; set {' or '.join(repr(b) for b in other)} in the Parameters cell to switch.")


_REF_WORDS = {"random": "random policy (dotted)", "solved": "gymnasium solved threshold (dashed)",
              "optimum": "known optimum of the toy-text task (dash-dot)"}


def _ref_sentence(kinds, where: bool = True) -> str:
    k = [_REF_WORDS[x] for x in ("random", "solved", "optimum") if x in kinds]
    return ("Grey lines: " + ", ".join(k) + (", drawn where they lie near the data." if where else ".")) if k else ""


def _common_notes(D: Data, envs) -> list:
    """Caption sentences shared by the per-task figures: budget daggers, previous baselines, untuned baselines, smoke."""
    out = []
    flags = [f"{e} ({_budget_flag(D, e)})" for e in envs if _budget_flag(D, e)]
    if flags:
        out.append(f"† Budget not the expected {D.cfg.protocol[0]:,} steps (protocol audit, section 2): " + _join(flags) + ".")
    if any(D.is_previous(e, l) for e in envs for l in D.rows(e)):
        out.append("+ = baseline filled from the previous results (benchmarks/results/final).")
    un = [e for e in envs if D.is_untuned(e)]
    if un:
        out.append("Baselines run with library defaults (untuned) on: " + ", ".join(_untuned_names(D, un)) + "."
                   + (" One untuned baseline marks the whole task." if _partly_untuned(D, un) else ""))
    fk = _frame_skip_note(D, [e for e in envs if D.rows(e)])
    if fk:
        out.append(fk)
    if D.cfg.smoke:
        out.append("SMOKE-sized runs: the numbers mean nothing.")
    return out


# ---- learning curves ---------------------------------------------------------------
WALLCLOCK_LABEL = "Wall-clock time"                 # x label of the wall-clock grid: the unit (s / min / h) is on each panel's last tick
_TIME_UNITS = (("s", 1 / 3600.0), ("min", 1 / 60.0), ("h", 1.0))      # (unit, hours per unit)


def _time_axis(ax, hours_max: float, nbins: int = 4) -> str:
    """x axis of wall-clock time from 0 to ``hours_max`` hours in the unit that suits the panel (seconds below one minute, minutes
    below one hour, else hours), at most ``nbins`` intervals of 1 / 2 / 2.5 / 5 x 10^k units, the unit written after the last tick
    label ("0  10  20  30 min"). Returns the unit."""
    from matplotlib.ticker import MaxNLocator

    hours_max = float(hours_max) if np.isfinite(hours_max) and hours_max > 0 else 1.0
    unit, per = next((u, s) for u, s in _TIME_UNITS if u == "h" or hours_max < 60.0 * s)
    vmax = hours_max / per
    locs = [float(v) for v in MaxNLocator(nbins, steps=[1, 2, 2.5, 5, 10]).tick_values(0.0, vmax) if -1e-9 <= v <= vmax * (1 + 1e-9)]
    labels = [fs.num(v, "g") for v in locs]
    if labels:
        labels[-1] += f" {unit}"
    ax.set_xlim(0.0, hours_max)
    ax.set_xticks([v * per for v in locs], labels)
    return unit


def draw_curves(D: Data, ax, env: str, show_seeds: bool = False, only=None, band=None, x: str = "steps",
                ref_values: bool = False, step_bins: int = 5) -> tuple:
    """Draw the task's learning curves on ``ax`` with ``figstyle.curve``: per algorithm ONE bold line = the mean evaluation return over
    the seeds (seeds aligned on their common evaluation steps, evaluations after the budget dropped) and ONE shaded band in the same
    colour = the spread of the seeds (``band``, default ``CURVE_BAND``: "minmax" = range across seeds, "std" = mean +- 1 standard
    deviation, "ci95" = 95% bootstrap CI of the mean, ``report.curve_band``; one seed = line only); the headline TEMPO bolder and on top;
    ``show_seeds`` adds the single seeds as thin lines. Reference lines (random policy, solved threshold, toy-text optimum) in neutral
    grey where they lie near the data (``ref_values``: their values in the legend labels). ``only`` = just these algorithms;
    ``x = "hours"``: x = share of the budget x the algorithm's mean wall-clock hours per run on the task (constant time per step),
    ticked in seconds / minutes / hours as suits the panel (:func:`_time_axis`). ``step_bins``: at most this many intervals on the
    step axis (5: 0, 100k, ..., 500k on a wide panel; 2: 0, 250k, 500k on the narrow panels of a grid).
    Band opacity: ``figstyle.band_alpha`` of the number of curves with a band in the panel (fainter as more bands overlap). y limits:
    ``figstyle.fit_ylim`` with the mean lines and reference lines in full and the bands robust (a band that one outlying or collapsed
    seed stretches far beyond the mean lines is clipped at the axes instead of squashing the other curves).
    Returns ``({label: Curve}, {reference kind: Line2D}, clipped)``: ``clipped`` = a band (or seed line) was cut at the axes (say so in
    the caption: ``figstyle.band_caption(..., clipped=True)``)."""
    band = _band_choice(D, band)
    rows = {l: rs for l, rs in D.rows(env).items() if only is None or l in only}
    budget = D.budget(env)
    todo, xmax = [], 0.0
    for lab, runs in sorted(rows.items(), key=lambda kv: style_of(D, kv[0])["z"]):
        steps, Y = fs.align(runs, budget=budget)
        if not Y.size or not np.isfinite(Y).any():
            continue
        xs = steps
        if x == "hours":
            h = float(np.nanmean([r.wall_h for r in runs])) if runs else math.nan
            if not np.isfinite(h) or h <= 0:
                continue
            xs = steps / budget * h
            xmax = max(xmax, h)
        todo.append((lab, xs, Y))
    n_banded = 0 if band == "none" else sum(1 for _, _, Y in todo if (np.isfinite(Y).sum(0) >= 2).any())
    alpha = fs.band_alpha(n_banded)                         # 0.2 with 1-2 bands, 0.14 with 3-4, 0.10 with 5 or more
    curves = {}
    for lab, xs, Y in todo:
        c = fs.curve(ax, xs, Y, D.disp(env, lab), style=style_of(D, lab), tempo=lab == D.main, band=band, alpha=alpha,
                     seeds=show_seeds, key=(env, lab), n_boot=D.cfg.n_curve_boot)
        if c is not None:
            curves[lab] = c
    means = [c.mean for c in curves.values()]
    bands = list(curves.values())
    refs, refv = {}, []
    if curves:
        fs.fit_ylim(ax, *means, bands=bands)                # limits of the data alone: which reference lines lie near it
        lo, hi = ax.get_ylim()
        near = 0.3 * (hi - lo)
        for val, _name, kind in references(D, env):
            if lo - near <= val <= hi + near:
                refs[kind] = fs.reference_line(ax, val, kind, f"{REF_STYLE[kind]['label']} ({fs.num(val)})" if ref_values else None)
                refv.append(val)
    clipped = fs.fit_ylim(ax, *means, *refv, bands=bands) if curves else False
    if x == "hours":
        _time_axis(ax, xmax or 1.0, nbins=4)
    else:
        fs.steps_axis(ax, budget, label=False, nbins=step_bins)
    fs.value_axis(ax, None)
    return curves, refs, clipped


def _curve_figure(D: Data, envs, title_fn, show_seeds: bool, band, x: str = "steps", only_fn=None, max_cols: int = 4,
                  legend_in_grid: bool = False) -> tuple:
    """Grid of learning-curve panels (one per task) with ONE shared legend below; returns (fig, drawn (task, label) pairs, ref kinds,
    tasks whose panel clipped a band at the axes). Panels of a grid with three or more columns (< 2.4 in wide) tick the step axis at 0,
    half the budget and the budget. ``legend_in_grid``: the legend fills the empty cells of the last row instead when there are at
    least two (23 tasks on 5 columns: row 5, columns 4 and 5), so the colour of every algorithm is read inside the grid."""
    nrows, cols = _grid_shape(len(envs), _balanced_cols(len(envs), max_cols))
    free = nrows * cols - len(envs) if legend_in_grid else 0
    fig, axes, ncols = _grid_axes(len(envs), max_cols, extra=0.45 if free >= 2 else 0.8)
    first, kinds, pairs, clipped = {}, set(), [], []
    step_bins = 2 if ncols >= 3 else 5
    for ax, env in zip(axes.flat, envs):
        curves, refs, cut = draw_curves(D, ax, env, show_seeds=show_seeds, band=band, x=x, only=only_fn(env) if only_fn else None,
                                        step_bins=step_bins)
        if cut:
            clipped.append(env)
        kinds |= set(refs)
        for lab, c in curves.items():
            pairs.append((env, lab))
            if lab not in first or (first[lab].band is None and c.band is not None):
                first[lab] = c                                  # legend key: a curve with a band if there is one
        ax.set_title(title_fn(env), fontweight=TASK_TITLE_WEIGHT)
    _grid_labels(fig, axes, len(envs), ncols, WALLCLOCK_LABEL if x == "hours" else fs.STEPS_LABEL, fs.RETURN_LABEL,
                 last_row_only=free >= 2)
    labs = [l for l in D.labels if l in first] + [l for l in first if l not in D.labels]
    if free >= 2:                                           # the empty cells of the last row hold the legend
        lax = fig.add_subplot(axes.flat[0].get_subplotspec().get_gridspec()[nrows - 1, ncols - free:])
        lax.set_axis_off()
        fs.legend(fig, [(first[l].legend_handle, l) for l in labs], _ref_handles(kinds), ax=lax,
                  width=0.92 * fig.get_figwidth() * free / ncols)
    else:
        fs.legend(fig, [(first[l].legend_handle, l) for l in labs], _ref_handles(kinds))
    return fig, pairs, kinds, clipped


def _n_eval_text(D: Data, envs) -> str:
    """Episodes per evaluation of the runs of ``envs``: ``10 deterministic episodes``, or ``10 deterministic episodes (Blackjack-v1:
    1,000)`` when tasks differ (the most common number, the tasks with another one named)."""
    per = {e: sorted({r.n_eval for rs in D.rows(e).values() for r in rs}) for e in envs}
    n = sorted({k for v in per.values() for k in v})
    if len(n) <= 1:
        return ("/".join(map(str, n)) or str(D.cfg.protocol[2])) + " deterministic episodes"
    cnt = Counter(k for v in per.values() for k in v)
    main = max(n, key=lambda k: (cnt[k], -k))
    odd = [f"{e}: {'/'.join(f'{k:,}' for k in v)}" for e, v in per.items() if v and v != [main]]
    return f"{main} deterministic episodes ({', '.join(odd)})"


def plot_curve_grid(D: Data, tasks=None, ncols: int = 5, show_seeds: bool = False, band=None, stem: str = "curves_all_tasks"):
    """Learning curves of every task with results (family order; at most ``ncols`` panels per row at full text width: the 23 default
    tasks fill 5 rows of 5, classic control the first row and toy text the last; panels of a grid with 5 or more columns are titled
    without the version suffix): per algorithm a bold mean line over the seeds and a shaded band of the seeds' spread (``band`` / ``CURVE_BAND``), TEMPO
    bolder and on top; one shared legend with the colour of every algorithm, in the grid's last row where two or more cells are free
    (23 tasks: row 5, columns 4 and 5), else below; the caption names tasks without runs, pending algorithms and curves with fewer
    seeds. Saved as PNG + PDF."""
    if not D.has_data:
        print("pending: no results yet, nothing to draw.")
        return []
    _mark(D, "curves")
    band = _band_choice(D, band)
    tasks = list(tasks or D.tasks)
    only = _headline_only(D)
    envs = [e for e in tasks if only(e)]
    if not envs:
        print("pending: no task has runs of the baselines or of the headline TEMPO yet, nothing to draw (other TEMPO variants: section 9).")
        return []
    short = _balanced_cols(len(envs), ncols) >= 5
    with figure_style():
        fig, pairs, kinds, cut = _curve_figure(D, envs, lambda e: _task_title(D, e, short), show_seeds, band, max_cols=ncols, only_fn=only,
                                               legend_in_grid=True)
        with_main = sum(1 for e in envs if D.main in D.rows(e))
        empty = [e for e in tasks if not D.rows(e)]
        var_only = [e for e in tasks if D.rows(e) and not only(e)]
        pend = [f"{e}: {_pending_note(D, e)}" for e in envs if _pending_note(D, e)]
        cap = [f"Figure. Evaluation return (mean of {_n_eval_text(D, envs)}) against environment steps on "
               f"the {_pl(len(envs), 'task')} with results; {D.main} (bold red, drawn on top) has runs on {with_main} of them.",
               _band_sentence(D, pairs, band, show_seeds, cut), _ref_sentence(kinds)]
        cap += _common_notes(D, envs)
        if empty:
            cap.append("No run yet: " + ", ".join(empty) + ".")
        if var_only:
            cap.append("Only other TEMPO variants so far: " + ", ".join(var_only) + ".")
        if pend:
            cap.append("Pending (expected, no run yet): " + _join(pend) + ".")
        if len(D.tempo_labels) > 1:
            cap.append(_VARIANT_NOTE)
        cap.append(_band_switch(D, band))
        return _finish(D, fig, stem, caption=" ".join(c for c in cap if c))


def plot_task(D: Data, env: str, show_seeds: bool = False, display: bool = True, band=None):
    """One task at full text width: the learning curves (bold mean, shaded spread of the seeds; ``show_seeds=True`` adds the single
    seeds) beside the per-seed final-return and AUC dot plots. Saved as ``figures/tasks/<task>.png|pdf``. Call
    ``cl.plot_task(D, 'Hopper-v5')`` to look at any task."""
    if env not in D.tasks and env not in TASK_ORDER:
        raise ValueError(f"unknown task {env!r}")
    rows = D.rows(env)
    if not rows:
        if display:
            print(f"{env}: pending (no results yet)")
        return []
    band = _band_choice(D, band)
    keep = _headline_only(D)(env)                           # other TEMPO variants: section 9
    others = [l for l in rows if l not in keep]
    if not keep:
        if display:
            print(f"{env}: only other TEMPO variants so far ({', '.join(others)}): see section 9")
        return []
    labs = [l for l in D.labels if l in keep] + [l for l in keep if l not in D.labels]
    with figure_style() as plt:
        fig = plt.figure(figsize=(fs.DOUBLE, 2.2 + 0.12 * len(labs)), layout="constrained")
        fig.get_layout_engine().set(wspace=0.05)            # room between the dot panels: their end tick labels never touch
        gs = fig.add_gridspec(1, 3, width_ratios=[2.1, 1.0, 1.0])
        ax0 = fig.add_subplot(gs[0])
        curves, refs, cut = draw_curves(D, ax0, env, show_seeds=show_seeds, band=band, ref_values=True, only=keep)
        kinds = set(refs)
        ax0.set_xlabel(fs.STEPS_LABEL)
        ax0.set_ylabel(fs.RETURN_LABEL)
        ax0.set_title(_task_title(D, env), fontweight=TASK_TITLE_WEIGHT)
        ns = {l: len(rows[l]) for l in labs}
        with_n = ns if len(set(ns.values())) > 1 else None
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2], sharey=ax1)
        dk = _dots(ax1, D, env, "final", labs, ylabels=True, show_n=False)
        dk |= _dots(ax2, D, env, "auc", labs, ylabels=False, show_n=False)
        ax1.set_title("Final return")
        ax2.set_title("AUC")
        ax1.set_xlabel(fs.RETURN_LABEL)
        ax2.set_xlabel("Mean return over training")
        algos = [(curves[l].legend_handle, D.disp(env, l) + (f" (n={ns[l]})" if with_n else "")) for l in labs if l in curves]
        b, _ = best_baseline(D, env, "final")
        fs.legend(fig, algos, [refs[k] for k in _REF_ORDER if k in refs]
                  + _ref_handles(dk - kinds, {"best": "Best baseline (mean)"}), _mark_handles(f"95% {D.cfg.ci_label}"))
        fam = FAMILY_NAMES.get(R.family(env), R.family(env))
        fsq = " / ".join(f"{f:,}" for f in sorted({r.eval_freq for rs in rows.values() for r in rs}))
        ref_text = [f"{w.split(' (')[0]} {fs.num(v)}" for v, _n, k in references(D, env) if k in kinds for w in [_REF_WORDS[k]]]
        cap = [f"Figure. {D.task_name(env)}: {fam}; {D.budget_text(env)} steps, an evaluation every {fsq} steps. Left: evaluation return "
               f"against environment steps. {_band_sentence(D, [(env, l) for l in curves], band, show_seeds, cut)}",
               ("Grey lines: " + ", ".join(ref_text) + ".") if ref_text else "",
               "Middle and right: final return (mean of the last three evaluations) and AUC (mean return over all evaluations) of every "
               f"seed (dots), the mean over seeds (vertical bar) and its 95% {D.cfg.ci_label} (horizontal bar; two or more seeds)"
               + ("; solid grey: mean final return / AUC of the best baseline" if b else "") + ".",
               f"Welch test of {D.main} against the best baseline (final return, alpha 0.05): {_outcome_text(D, env, 'final')}."]
        cap += _common_notes(D, [env])
        note = _pending_note(D, env)
        if note:
            cap.append(f"Pending (expected, no run yet): {note}.")
        if others:
            cap.append(f"Other TEMPO variants on this task ({', '.join(others)}): section 9.")
        cap.append(_band_switch(D, band) + ("" if show_seeds else " show_seeds=True adds the single seeds."))
        return _finish(D, fig, env, "tasks", display=display, caption=" ".join(c for c in cap if c))


#: In-row note of a dot-plot row without a run (the markers of the heatmap): status -> text.
_EMPTY_ROW = {"pending": "pending", "n/a": "n/a", "none": "–"}
#: Caption key of those notes.
_EMPTY_ROW_TEXT = {"pending": "pending = expected, no run yet", "n/a": "n/a = not run on this action space",
                   "–": "– = not run (a TEMPO variant that is not part of this task)"}


def _dots(ax, D: Data, env: str, metric: str, labels, ylabels: bool = True, show_n: bool = False, marks: set | None = None) -> set:
    """Horizontal dot plot, one row per algorithm (top to bottom in ``labels`` order): every seed as a dot, the mean over seeds as a
    vertical bar and its 95% CI as a thin translucent bar; vertical reference lines: the best baseline's mean (solid grey), the
    random policy (final return) and the toy-text optimum. A row without a run says why in muted text (the heatmap's markers):
    ``pending`` (expected, no run yet), ``n/a`` (not run on this action space), ``–`` (a TEMPO variant not part of the task); the
    markers written are added to ``marks``. Returns the reference kinds drawn."""
    from matplotlib.transforms import blended_transform_factory

    tr = blended_transform_factory(ax.transAxes, ax.transData)
    cells = D.cells.get(env, {})
    pos = {l: len(labels) - 1 - i for i, l in enumerate(labels)}
    for lab in labels:
        y, st, c = pos[lab], style_of(D, lab), cells.get(lab)
        if c is None:
            txt = _EMPTY_ROW.get(D.status(env, lab), "–")
            ax.text(0.03, y, txt, transform=tr, va="center", fontsize=_NOTE_SIZE, color=MUTED,
                    style="normal" if txt == "–" else "italic")
            if marks is not None:
                marks.add(txt)
            continue
        vals = np.asarray(c.finals if metric == "final" else c.aucs, float)
        m, lo, hi = c.final if metric == "final" else c.auc
        jit = np.linspace(-0.2, 0.2, vals.size) if vals.size > 1 else np.zeros(1)        # seeds in order, top to bottom
        ax.scatter(vals, y + jit, s=fs.MARKER_SIZE ** 2, color=st["color"], alpha=0.75, zorder=3, edgecolor="white", linewidth=0.4)
        if np.isfinite(lo):
            ax.plot([lo, hi], [y, y], color=st["color"], lw=_CI_LW, alpha=_CI_ALPHA, solid_capstyle="butt", zorder=2)
        ax.plot([m], [y], marker="|", ms=_TICK_MS, mew=_TICK_MEW, color=st["color"], zorder=4)
        if show_n and c.n < D.cfg.n_expected:
            ax.text(0.995, y + 0.34, f"n={c.n}", transform=tr, ha="right", va="bottom", fontsize=_NOTE_SIZE, color=MUTED)
    kinds = set()
    b, bm = best_baseline(D, env, metric)
    if b is not None and np.isfinite(bm):
        ref_line(ax, bm, "best", vertical=True)
        kinds.add("best")
    rnd = D.randoms.get(env)
    if metric == "final" and rnd is not None:
        lo_, hi_ = ax.get_xlim()
        if lo_ - 0.1 * (hi_ - lo_) <= rnd <= hi_ + 0.1 * (hi_ - lo_):
            ref_line(ax, rnd, "random", vertical=True)
            kinds.add("random")
    if metric == "final":
        for val, _name, kind in references(D, env):
            if kind == "optimum":
                lo_, hi_ = ax.get_xlim()
                ref_line(ax, val, "optimum", vertical=True)
                kinds.add("optimum")
                span = (max(hi_, val) - min(lo_, val)) or 1.0
                ax.set_xlim(min(lo_, val - 0.04 * span), max(hi_, val + 0.04 * span))
    lo_, hi_ = ax.get_xlim()
    pad = 0.03 * ((hi_ - lo_) or 1.0)                      # keep the outermost dots off the axes edge
    ax.set_xlim(lo_ - pad, hi_ + pad)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_yticks([pos[l] for l in labels])
    ax.set_yticklabels([D.disp(env, l) for l in labels])
    ax.tick_params(axis="y", length=0, labelleft=ylabels)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x")
    fs.value_axis(ax, None, nbins=4, axis="x")
    for tick, lab in zip(ax.get_yticklabels(), labels):
        if lab == D.main:
            tick.set_fontweight("bold")
    return kinds


def plot_metric_groups(D: Data, metric: str = "final", ncols: int = 4):
    """Per task group (classic control, Box2D, MuJoCo, toy text): one dot plot per task of the per-seed ``metric`` ('final' return
    or 'auc') with mean and 95% CI per algorithm. Saved as ``figures/final_return_<group>`` / ``figures/auc_<group>``."""
    if not D.has_data:
        print("pending: no results yet, nothing to draw.")
        return []
    xlabel = {"final": "Final return", "auc": "AUC (mean return)"}[metric]
    stem = {"final": "final_return", "auc": "auc"}[metric]
    groups = defaultdict(list)
    for env in D.tasks:
        groups[GROUPS.get(R.family(env), "Other")].append(env)
    out = []
    for g, envs in groups.items():
        envs = [e for e in envs if D.rows(e)]
        if not envs:
            print(f"{g}: pending (no results yet)")
            continue
        labs = [l for l in D.labels if any(D.n(e, l) for e in envs)]
        with figure_style():
            nrows, nc = _grid_shape(len(envs), _balanced_cols(len(envs), ncols))
            width = fs.SINGLE if nc == 1 else fs.DOUBLE
            fig, axes = fs.subplots(nrows, nc, width=width, height=nrows * (0.15 * len(labs) + 0.62) + 0.55, squeeze=False)
            fig.get_layout_engine().set(wspace=0.04)        # end tick labels of neighbouring panels never touch
            for ax in axes.flat[len(envs):]:
                ax.remove()
            kinds, marks = set(), set()
            for k, (ax, env) in enumerate(zip(axes.flat, envs)):
                kinds |= _dots(ax, D, env, metric, labs, ylabels=(k % nc == 0), show_n=True, marks=marks)
                ax.set_title(_task_title(D, env), fontweight=TASK_TITLE_WEIGHT)
            for j in range(nc):
                col = [axes.flat[i] for i in range(j, len(envs), nc)]
                col[-1].set_xlabel(xlabel)
            ci = f"95% {D.cfg.ci_label}"
            fs.legend(fig, _mark_handles(ci), _ref_handles(kinds, {"best": "Best baseline (mean)"}))
            toy = all(e in P.TOY_TEXT_ALL for e in envs)
            words = {**_EMPTY_ROW_TEXT, "n/a": "n/a = " + _na_text(D, envs)}
            empty = [words[m] for m in ("pending", "n/a", "–") if m in marks]
            what = "Final return (mean of the last three evaluations)" if metric == "final" else "AUC (mean return over all evaluations)"
            cap = [f"Figure. {what} on the {g} tasks: every seed (dot), the mean over seeds (vertical bar) and its {ci} (horizontal bar; "
                   "two or more seeds). n=k: fewer seeds than expected."
                   + (" Rows without a run: " + "; ".join(empty) + "." if empty else "")]
            if toy:
                cap.append("Toy-text tasks are judged against the optimum: their baselines are references, so there is no best-baseline line and no test.")
            else:
                cap.append(f"Solid grey: mean of the best baseline. Welch test of {D.main} against the best baseline (alpha 0.05, at least "
                           f"{D.cfg.min_n_test} seeds per side): " + "; ".join(f"{e}: {_outcome_text(D, e, metric)}" for e in envs) + ".")
            cap.append(_ref_sentence(kinds - {"best"}, where=False))
            cap += _common_notes(D, envs)
            out += _finish(D, fig, f"{stem}_{g.lower().replace(' ', '_')}", caption=" ".join(c for c in cap if c))
    return out


# ---- normalised heatmap and family overlays -----------------------------------------
def _f2(v: float) -> str:
    """Two decimals without a negative zero ('-0.00' -> '0.00')."""
    t = f"{v:.2f}"
    return "0.00" if t == "-0.00" else t


def _sdec(v: float, d: int) -> str:
    """Signed number with ``d`` decimals, a plain zero instead of '-0' / '+0.00'."""
    t = f"{v:+.{d}f}"
    return t[1:] if float(t) == 0 else t


def _s2(v: float) -> str:
    """Signed two decimals without a negative zero."""
    t = f"{v:+.2f}"
    return "0.00" if t in ("-0.00", "+0.00") else t


def norm_matrix(D: Data, tasks=None):
    """(tasks, labels, matrix): mean normalised final score (random 0, best baseline 1) per task x algorithm;
    NaN where there is no run or the task cannot be normalised."""
    tasks = [e for e in (tasks or D.tasks) if D.norm.get(e, {}).get("valid")]
    labs = list(D.labels)
    M = np.full((len(tasks), len(labs)), np.nan)
    for i, env in enumerate(tasks):
        nz = D.norm[env]
        for j, l in enumerate(labs):
            c = D.cells.get(env, {}).get(l)
            if c is not None:
                M[i, j] = (c.final[0] - nz["random"]) / (nz["best_baseline"] - nz["random"])
    return tasks, labs, M


def plot_heatmap(D: Data, stem: str = "heatmap_normalised_final"):
    """Task x algorithm heatmap of the mean normalised final score (0 = random policy, 1 = best baseline of the task), single-hue
    sequential scale (light = low, dark = high); 'pending' / 'n/a' cells are marked."""
    tasks, labs, M = norm_matrix(D)
    if not tasks:
        print("pending: no task has a baseline result and a random score yet, nothing to normalise.")
        return []
    return _heatmap_figure(D, tasks, labs, M, "Normalised final score", "Normalised final score\n(0 = random policy, 1 = best baseline)",
                           stem, "Mean normalised final score over seeds per task and algorithm: (final - random policy) / (best baseline "
                           "mean final - random policy).")


def _heatmap_cmap():
    """figstyle's single-hue sequential ramp without its near-white start, so that the lowest value stays visibly tinted next to
    the white 'pending' cells."""
    from matplotlib.colors import ListedColormap

    return ListedColormap(fs.sequential_cmap()(np.linspace(0.1, 1.0, 256)), name="complab_seq")


def _heatmap_figure(D: Data, tasks, labs, M, title: str, cbar_label: str, stem: str, what: str):
    """Heatmap of a task x algorithm matrix on a single-hue sequential scale (light -> dark; 1 = best baseline marked on the colour
    bar), every cell annotated, pending / n/a cells white, families separated by a gap."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    fin = M[np.isfinite(M)]
    lo = min(0.0, float(fin.min())) if fin.size else 0.0
    hi = max(1.0, min(float(fin.max()), 2.0)) if fin.size else 1.0
    extend = "max" if fin.size and fin.max() > hi else "neither"
    norm, cmap = Normalize(lo, hi), _heatmap_cmap()
    nr, nc = len(tasks), len(labs)
    width = min(fs.DOUBLE, max(fs.SINGLE, 2.5 + 0.72 * nc))
    with figure_style({"axes.grid": False}):
        fig, ax = fs.subplots(width=width, height=0.22 * nr + 1.05)
        partial = previous = dash = False
        for i, env in enumerate(tasks):
            for j, l in enumerate(labs):
                v = M[i, j]
                if np.isfinite(v):
                    rgba = cmap(norm(v))
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc=rgba, ec="white", lw=1.2))
                    k = D.n(env, l)
                    partial |= k < D.cfg.n_expected
                    previous |= D.is_previous(env, l)
                    t = (_f2(v).replace("-", "\u2212") + (f" ({k})" if k < D.cfg.n_expected else "")
                         + ("+" if D.is_previous(env, l) else ""))
                    ax.text(j, i, t, ha="center", va="center", fontsize=fs.FONT_SIZE, color=fs.text_on(rgba))
                else:
                    st = D.status(env, l)
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="white", ec=GRID, lw=0.6))
                    txt = "pending" if st == "pending" else "n/a" if st == "n/a" else "–"   # en dash: not run, not expected
                    dash |= txt == "–"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=_NOTE_SIZE, color=MUTED,
                            style="normal" if txt == "–" else "italic")
        prev = None
        for i, env in enumerate(tasks):                    # a white gap between task families
            if prev is not None and R.family(env) != prev:
                ax.axhline(i - 0.5, color="white", lw=4.0, zorder=5)
            prev = R.family(env)
        ax.set_xlim(-0.5, nc - 0.5)
        ax.set_ylim(nr - 0.5, -0.5)
        ax.set_xticks(range(nc))
        ax.set_xticklabels(labs)
        ax.set_yticks(range(nr))
        ax.set_yticklabels([D.task_name(e) for e in tasks])
        ax.xaxis.tick_top()
        ax.tick_params(axis="both", length=0, labeltop=True, labelbottom=False)
        long = max((len(l) for l in labs), default=0) * fs.FONT_SIZE * 0.5 / 72 > (width - 2.4) / max(nc, 1) * 0.95
        for tl in ax.get_xticklabels():
            if long:
                tl.set_rotation(30)
                tl.set_ha("left")
                tl.set_rotation_mode("anchor")
            if tl.get_text() == D.main:
                tl.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.05, pad=0.02, extend=extend, aspect=max(12, 1.6 * nr))
        cb.set_label(cbar_label, fontsize=fs.FONT_SIZE)
        cb.outline.set_linewidth(fs.AXIS_WIDTH)
        cb.outline.set_edgecolor(INK2)
        cb.ax.tick_params(labelsize=fs.FONT_SIZE, width=fs.AXIS_WIDTH, length=2.5)
        cb.ax.axhline(1.0, color=INK, lw=0.8)
        ax.set_title(title, pad=4)
        cap = [f"Figure. {what} Single-hue scale from light (low) to dark (high); the line on the colour bar marks 1 (best baseline)"
               + (f"; values above {hi:g} have the darkest colour" if extend == "max" else "") + ". Rows grouped by task family. "
               "Not shown: tasks that cannot be normalised (toy text; no baseline result; reference too weak: the best baseline covers "
               f"less than {D.cfg.min_gap:.0%} of the range above the random policy). pending = expected, no run yet; n/a = "
               f"{_na_text(D, tasks)}."]
        if dash:
            cap.append("– = not run (variant not part of this task).")
        if partial:
            cap.append("(k) = only k seeds, fewer than expected.")
        if previous:
            cap.append("+ = previous baseline run.")
        if any(D.is_untuned(e) for e in tasks):
            cap.append("(untuned) = some or all baselines ran with library defaults." + _partly_note(D, tasks))
        if D.cfg.smoke:
            cap.append("SMOKE-sized runs.")
        return _finish(D, fig, stem, caption=" ".join(cap))


def aggregate_scope(D: Data, family: str | None = None, require_main: bool | None = None) -> list:
    """Tasks that enter the aggregate statistics: valid normalisation (a baseline result above the random
    policy), restricted to the tasks the headline TEMPO ran on (``agg_tasks = 'tempo'``; all tasks when TEMPO
    has no result at all) and optionally to one family."""
    tasks = [e for e in D.tasks if D.norm.get(e, {}).get("valid")]
    if require_main is None:
        require_main = D.cfg.agg_tasks == "tempo"
    if require_main and D.has_main:
        tasks = [e for e in tasks if D.main in D.cells.get(e, {})]
    if family:
        tasks = [e for e in tasks if R.family(e) == family]
    return tasks


def normalised_curves(D: Data, env: str, label: str, grid: np.ndarray):
    """Per-seed learning curves of ``label`` on the task, normalised like the final score and resampled onto
    fractions of the budget: array (n_seeds, len(grid))."""
    nz = D.norm[env]
    scale = nz["best_baseline"] - nz["random"]
    out = []
    for r in D.runs[env][label]:
        y = np.interp(grid * r.budget, r.steps, r.curve)
        out.append((y - nz["random"]) / scale)
    return np.asarray(out)


def _task_mean_band(Y: list, band: str, n_boot: int, key) -> tuple:
    """(mean, lo, hi) of curves averaged over tasks with equal weight; ``Y`` = one (n_seeds x n_points) array per task. Line = mean over
    tasks of the per-task mean over seeds. Band: "minmax" / "std" = the mean over tasks of the per-task band (lowest / highest seed,
    mean -+ 1 standard deviation; a one-seed task has a zero-width band); "ci95" = 95% bootstrap of the task-averaged mean, seeds
    resampled within every task. No band (NaN) when no task has two seeds or ``band = "none"``."""
    mean = np.mean([y.mean(0) for y in Y], axis=0)
    nan = np.full(mean.shape, np.nan)
    if band == "none" or not any(len(y) > 1 for y in Y):
        return mean, nan, nan.copy()
    if band == "minmax":
        return mean, np.mean([y.min(0) for y in Y], axis=0), np.mean([y.max(0) for y in Y], axis=0)
    if band == "std":
        sd = np.mean([y.std(0, ddof=1) if len(y) > 1 else np.zeros(y.shape[1]) for y in Y], axis=0)
        return mean, mean - sd, mean + sd
    rng = R._rng(*key)
    boots = np.mean([y[rng.integers(0, len(y), size=(n_boot, len(y)))].mean(1) for y in Y], axis=0)
    return mean, np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


def _curve_from_band(D: Data, ax, x, mean, lo, hi, label: str, alpha_line: float = 1.0, alpha: float = fs.BAND_ALPHA):
    """Draw a precomputed (mean, lo, hi) like ``figstyle.curve`` (bold line in the algorithm's style, band without edge in the same
    colour at the band opacity ``alpha``, i.e. ``figstyle.band_alpha`` of the number of bands in the panel): figstyle.curve computes
    the band from the seeds itself, an average over tasks needs this. Returns the :class:`figstyle.Curve` (``.alpha`` = the band's
    opacity, so its legend key matches; legend key with the band when one was drawn)."""
    c = fs.curve(ax, x, np.asarray(mean, float)[None, :], label, style=style_of(D, label), tempo=label == D.main, band="none")
    if c is None:
        return None
    if alpha_line < 1.0:
        c.line.set_alpha(alpha_line)
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any() and alpha_line == 1.0:                      # figstyle's band layer: above the reference lines, below every line
        z = 1.5 + float(style_of(D, label).get("z", 2)) / 100.0 + (0.05 if label == D.main else 0.0)
        c.band = ax.fill_between(x, lo, hi, where=ok, color=c.line.get_color(), alpha=alpha, lw=0, edgecolor="none", zorder=z)
        c.lo, c.hi, c.alpha = np.asarray(lo, float), np.asarray(hi, float), float(alpha)
    return c


_ACTION_TITLES = {"discrete": "Discrete-action tasks", "continuous": "Continuous-action tasks"}


def plot_family_overlays(D: Data, n_grid: int = 60, band=None, stem: str = "curves_normalised_by_family"):
    """Per task family, per action space and over all tasks: normalised learning curves (return normalised per task: 0 = random policy,
    1 = best baseline's final mean; x = share of the budget), averaged over the panel's tasks with equal weight: a bold line per
    algorithm and a shaded band (``band`` / ``CURVE_BAND``: "minmax" = the per-task range across seeds averaged over the tasks, "std" =
    the same for mean -+ 1 standard deviation, "ci95" = 95% bootstrap of the mean with seeds resampled within tasks). Only tasks the
    headline TEMPO ran on are used (when it has runs). An algorithm that ran on only some of the tasks of a panel is drawn faded (no
    band); one that covers fewer than half of them is not drawn (its curve would average other tasks). Band opacity:
    ``figstyle.band_alpha`` of the number of bands in the panel; y limits: ``figstyle.fit_ylim`` with the lines and the reference levels
    0 and 1 in full and the bands robust (clipped at the axes when they reach far beyond the lines; the caption names those panels)."""
    band = _band_choice(D, band)
    scope_all = aggregate_scope(D)
    if not scope_all:
        print("pending: no task with a baseline result (and TEMPO runs) yet, nothing to overlay.")
        return []
    fams = [f for f in dict.fromkeys(R.family(e) for e in scope_all)]
    panels = [(FAMILY_NAMES.get(f, f), [e for e in scope_all if R.family(e) == f]) for f in fams]
    kinds = {a: [e for e in scope_all if action_space(e) == a] for a in ("discrete", "continuous")}
    if kinds["discrete"] and kinds["continuous"]:
        panels += [(_ACTION_TITLES[a], kinds[a]) for a in ("discrete", "continuous")]
    if len(fams) > 1:
        panels.append(("All tasks", scope_all))
    notes, first, clipped = [], {}, []
    nseeds = Counter(D.n(e, l) for e in scope_all for l in [D.main] + list(D.cfg.baselines) if l in D.cells.get(e, {}))
    with figure_style():
        fig, axes, ncols = _grid_axes(len(panels), 3, panel_aspect=0.68)
        for ax, (title, tasks) in zip(axes.flat, panels):
            start = max(D.eval_freq(e) / D.budget(e) for e in tasks)
            grid = np.linspace(min(start * 1.001, 1.0), 1.0, n_grid)
            todo = []
            for lab in [D.main] + list(D.cfg.baselines):
                ts = [e for e in tasks if lab in D.cells.get(e, {})]
                if not ts:
                    continue
                cover = len(ts) / len(tasks)
                if cover < 0.5:
                    notes.append(f"{title}: {lab} not drawn (ran on {len(ts)} of {len(tasks)} tasks)")
                    continue
                faded = cover < 1.0
                Y = [normalised_curves(D, e, lab, grid) for e in ts]
                mean, lo, hi = _task_mean_band(Y, band, min(D.cfg.n_curve_boot, 400), ("overlay", title, lab))
                todo.append((lab, faded, len(ts), mean, lo, hi))
                if faded:
                    notes.append(f"{title}: {lab} faded (ran on {len(ts)} of {len(tasks)} tasks)")
            n_banded = sum(1 for _, faded, _, _, lo, hi in todo if not faded and (np.isfinite(lo) & np.isfinite(hi)).any())
            alpha = fs.band_alpha(n_banded)                 # fainter as more bands overlap in the panel
            means, bands = [0.0, 1.0], []
            for lab, faded, _, mean, lo, hi in todo:
                c = _curve_from_band(D, ax, grid, mean, lo, hi, lab, alpha_line=0.45 if faded else 1.0, alpha=alpha)
                if c is None:
                    continue
                means.append(mean)
                if c.band is not None:
                    bands.append((c.lo, c.hi))
                if lab not in first or (first[lab].band is None and c.band is not None):
                    first[lab] = c
            ref_line(ax, 1.0, "best")
            fs.reference_line(ax, 0.0, "random")
            if fs.fit_ylim(ax, *means, bands=bands):        # lines and 0 / 1 in full; a band far beyond them is clipped
                clipped.append(title)
            ax.set_xlim(0.0, 1.0)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.xaxis.set_major_formatter(lambda x, _: f"{100 * x:.0f}")
            ax.set_title(f"{title} ({_pl(len(tasks), 'task')})")
        _norm_ticks(list(axes.flat[:len(panels)]), "y", nbins=5)   # same ticks / decimals on every panel
        _grid_labels(fig, axes, len(panels), ncols, "Share of the step budget (%)", fs.NORMALISED_LABEL)
        labs = [l for l in [D.main] + list(D.cfg.baselines) if l in first]
        fs.legend(fig, [(first[l].legend_handle, l) for l in labs],
                  _ref_handles({"best", "random"}, {"best": "Best baseline (final mean) = 1", "random": "Random policy = 0"}))
        btxt = {"minmax": "the per-task range across seeds (lowest to highest seed), averaged over the tasks",
                "std": "the per-task mean ± 1 standard deviation across seeds, averaged over the tasks",
                "ci95": "95% bootstrap confidence interval of the task-averaged mean (seeds resampled within every task)",
                "none": ""}[band]
        cap = ["Figure. Normalised learning curves: return normalised per task (0 = random policy, 1 = final mean of the best baseline), "
               "x = share of each run's budget; every task of a panel has equal weight. Lines: mean over the tasks of the per-task mean "
               "over seeds" + (f"; shaded: {btxt}" if btxt else "") + f". {D.main} bold red, on top."
               + (" Bands clipped at the axes (they reach far beyond the lines) on: " + _join(clipped) + "." if clipped else ""),
               "Each algorithm is averaged over the tasks of the panel it ran on: faded (no band) = only some of them, not drawn = fewer "
               "than half (DQN runs on discrete-action tasks only, SAC / TD3 / CrossQ on continuous ones"
               + "".join(f", CrossQ not on the pixel task {e}" for e in scope_all if e in P.IMAGE) + ": compare them in the "
               "action-space panels)." + (" Panel notes: " + _join(dict.fromkeys(notes)) + "." if notes else ""),
               "Other TEMPO variants: section 9."]
        if len(nseeds) > 1:
            cap.append("Seeds per task and algorithm: " + ", ".join(f"{k} ({v} task-algorithm pairs)" for k, v in sorted(nseeds.items())) + ".")
        if D.cfg.smoke:
            cap.append("SMOKE-sized runs.")
        cap.append(_band_switch(D, band))
        return _finish(D, fig, stem, caption=" ".join(cap))


def plot_all_task_figures(D: Data, show_seeds: bool = False, band=None) -> list:
    """Save one figure per task (``figures/tasks/<task>.png|pdf``) without showing them; look at one with plot_task."""
    paths = []
    only = _headline_only(D)
    for env in D.tasks:
        if only(env):
            paths += plot_task(D, env, show_seeds=show_seeds, display=False, band=band)
    n = sum(1 for e in D.tasks if only(e))
    print(f"{n} per-task figures" + (f" saved in {D.cfg.fig_dir / 'tasks'}" if paths else
                                      (" drawn but not saved (SAVE_OUTPUTS = False)" if n else " (none: no results yet)")))
    return paths


# =========================================================================== 4. score tables
def _tables(D: Data) -> dict:
    return D._extra.setdefault("tables", {})


def _mark(D: Data, section: str) -> None:
    """Remember which sections ran (the export archives outdated files only after a complete run)."""
    D._extra.setdefault("sections", set()).add(section)


def register(D: Data, name: str, title: str, df=None, md: str | None = None, tex: str | None = None) -> None:
    """Remember a table for the export section (markdown, LaTeX, CSV)."""
    _tables(D)[name] = dict(title=title, df=df, md=md, tex=tex)


def _num(x, d: int = 2) -> str:
    return "nan" if x is None or not np.isfinite(x) else R._num(float(x), d)


def _pf(p) -> str:
    return "n/a" if p is None or not np.isfinite(p) else R._pfmt(p)


def _g(v) -> str:
    """Compact number: no exponents for the return scales met here (0.003 ... 99999)."""
    if v is None or not np.isfinite(v):
        return ""
    a = abs(v)
    return f"{v:.0f}" if a >= 1000 else f"{v:.1f}" if a >= 100 else f"{v:.2f}" if a >= 1 else f"{v:.3f}" if a >= 0.001 else f"{v:.2g}" if a else "0"


def df_to_md(df, index: bool = False) -> str:
    """Markdown table of a DataFrame (``benchmarks.report.md_table``); numbers keep 4 significant digits."""
    d = df.reset_index() if index else df
    rows = [[_g(v) if isinstance(v, (float, np.floating)) else str(v) for v in r] for r in d.itertuples(index=False)]
    align = "".join("r" if str(t) in ("int64", "float64", "Int64") else "l" for t in d.dtypes)
    return R.md_table([str(c) for c in d.columns], rows, align=align or None)


def df_to_tex(df, caption: str, label: str, index: bool = False) -> str:
    """booktabs LaTeX table of a DataFrame (numbers right-aligned, text escaped)."""
    d = df.reset_index() if index else df

    def cell(v):
        if isinstance(v, (float, np.floating)):
            s = _g(v)
        else:
            s = str(v)
        s = R._tex_escape(s).replace("<", "$<$").replace(">", "$>$")
        return re.sub(r"(?<![\w.])-(?=\d)", "$-$", s)

    align = "".join("r" if str(t) in ("int64", "float64", "Int64") else "l" for t in d.dtypes)
    wide = d.shape[1] > 7
    lines = ["% Generated by Comparison/complab.py; requires \\usepackage{booktabs" + (",graphicx" if wide else "") + "}.", r"\begin{table}[tbp]",
             r"\centering\small", rf"\caption{{{R._tex_escape(caption)}}}", rf"\label{{tab:{label}}}"]
    lines += [r"\resizebox{\textwidth}{!}{%"] if wide else []
    lines += [rf"\begin{{tabular}}{{@{{}}{align}@{{}}}}", r"\toprule", " & ".join(R._tex_escape(str(c)) for c in d.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(cell(v) for v in r) + r" \\" for r in d.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}" + ("}" if wide else ""), r"\end{table}", ""]
    return "\n".join(lines)


# ---- per-task score tables -----------------------------------------------------------
def _score_cell_md(D: Data, env: str, lab: str, metric: str, bold: set, d: int) -> str:
    c = D.cells.get(env, {}).get(lab)
    if c is None:
        st = D.status(env, lab)
        return "pending" if st == "pending" else "n/a" if st == "n/a" else "-"
    m, lo, hi = getattr(c, metric)
    ci = f"[{_num(lo, d)}, {_num(hi, d)}]" if np.isfinite(lo) else ""
    nk = "" if c.n >= D.cfg.n_expected else f"(n={c.n})"
    mm = _num(m, d)
    mm = f"**{mm}**" if lab in bold else mm
    return " ".join(x for x in (mm, ci, nk) if x) + ("+" if D.is_previous(env, lab) else "")


def _bold_labels(D: Data, env: str, metric: str, labels) -> tuple:
    """(set of labels in bold, best label): the best mean and every label whose CI overlaps the best one's."""
    row = D.cells.get(env, {})
    st = {l: getattr(row[l], metric) for l in labels if l in row}
    if not st:
        return set(), None
    best = max(st, key=lambda l: st[l][0])
    return {l for l in st if R._overlaps(st[l], st[best])}, best


def score_table_md(D: Data, metric: str = "final") -> str:
    """Markdown: mean [95% CI] (n if fewer seeds than expected) per task x algorithm; bold = best mean and every
    algorithm whose CI overlaps it (as the project's report); 'pending' = expected but no run yet."""
    head = ["Task", "Budget"] + list(D.labels) + ["Highest mean"]
    rows = []
    for env in D.tasks:
        bold, best = _bold_labels(D, env, metric, D.labels)
        d = R._decimals([getattr(D.cells[env][l], metric)[0] for l in D.labels if l in D.cells.get(env, {})])
        rows.append([D.task_name(env), D.budget_text(env)] + [_score_cell_md(D, env, l, metric, bold, d) for l in D.labels] + [best or "-"])
    what = (f"Final return (mean of the last three evaluations, {final_episodes_text(D.cfg.smoke, D.tasks)})" if metric == "final"
            else "Area under the learning curve (mean return over all evaluations)")
    note = (f"{what}: mean over seeds [95% {D.cfg.ci_label}" + ("; " + _coverage_text(D.cfg.n_expected) if D.cfg.ci_method != "t" else "")
            + f"]. Bold: best mean and every algorithm whose CI overlaps the best "
            f"one's. (n=k): fewer than the {D.cfg.n_expected} expected seeds. pending = expected, no finished run; n/a = {_na_text(D)}; "
            "- = not run. + = baseline from the previous results (benchmarks/results/final). Highest mean = the algorithm "
            "with the highest mean among all columns, TEMPO included. (untuned) = some or all baselines ran with library defaults."
            + _partly_note(D))
    return R.md_table(head, rows, align="ll" + "r" * (len(head) - 2)) + f"\n{note}\n"


def score_table_tex(D: Data, metric: str = "final") -> str:
    """LaTeX version of :func:`score_table_md` (booktabs, family blocks)."""
    labels = list(D.labels)
    body, prev = [], None
    for env in D.tasks:
        bold, _ = _bold_labels(D, env, metric, labels)
        d = R._decimals([getattr(D.cells[env][l], metric)[0] for l in labels if l in D.cells.get(env, {})])
        tex = []
        for l in labels:
            c = D.cells.get(env, {}).get(l)
            if c is None:
                st = D.status(env, l)
                tex.append(r"\textit{pending}" if st == "pending" else r"\textemdash")
                continue
            m, lo, hi = getattr(c, metric)
            mt = R._num(m, d, True)
            mt = rf"\textbf{{{mt}}}" if l in bold else mt
            ci = f"[{R._num(lo, d, True)}, {R._num(hi, d, True)}]" if np.isfinite(lo) else ""
            nk = "" if c.n >= D.cfg.n_expected else rf"($n{{=}}{c.n}$)"
            small = " ".join(x for x in (ci, nk) if x)
            tex.append(mt + (rf" {{\scriptsize {small}}}" if small else "") + (r"$^{+}$" if D.is_previous(env, l) else ""))
        fam = R.family(env)
        if fam != prev:
            if prev is not None:
                body.append(r"\midrule")
            body.append(rf"\multicolumn{{{len(labels) + 2}}}{{@{{}}l}}{{\emph{{{FAMILY_NAMES.get(fam, fam)}}}}} \\")
            prev = fam
        body.append(" & ".join([R._tex_escape(D.task_name(env)), D.budget_text(env)] + tex) + r" \\")
    what = "Final return (mean of the last three evaluations)" if metric == "final" else "AUC (mean return over all evaluations)"
    cap = (rf"{what}: mean over seeds and 95\,\% {'$t$-interval' if D.cfg.ci_method == 't' else 'percentile-bootstrap CI'}. Bold: best mean and every algorithm whose CI overlaps it. "
           rf"($n{{=}}k$): fewer than {D.cfg.n_expected} seeds; \textit{{pending}}: no finished run; \textemdash: not run. "
           r"$^{+}$: baseline from the previous results. (untuned): some or all baselines ran with library defaults."
           + R._tex_escape(_partly_note(D)))
    return "\n".join(["% Generated by Comparison/complab.py; requires \\usepackage{booktabs,graphicx}.", r"\begin{table*}[t]", r"\centering",
                      rf"\caption{{{cap}}}", rf"\label{{tab:{metric}-scores}}", r"\resizebox{\textwidth}{!}{%",
                      r"\begin{tabular}{@{}ll" + "c" * len(labels) + r"@{}}", r"\toprule",
                      " & ".join(["Task", "Budget"] + [R._tex_escape(l) for l in labels]) + r" \\", r"\midrule", *body,
                      r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])


# ---- rank of every algorithm on every task (the paper's rank table) ----------------------
RANK_GROUPS = {"classic": "Classic control", "box2d": "Box2D", "mujoco": "MuJoCo", "toy": "Toy text"}
#: circled podium ranks (TikZ; rank 1 bold with a heavier circle), defined only where the document has not defined them
RANK_TEX_MACROS = "\n".join([
    r"\providecommand{\rankcirc}[2]{\tikz[baseline=(r.base)]\node[circle,draw,line width=#1,inner sep=0pt,minimum size=1.3em](r){#2};}",
    r"\providecommand{\rankone}{\rankcirc{0.9pt}{\textbf{1}}}",
    r"\providecommand{\ranktwo}{\rankcirc{0.4pt}{2}}",
    r"\providecommand{\rankthree}{\rankcirc{0.4pt}{3}}"])


def task_ranks(D: Data, env: str, metric: str = "final") -> dict:
    """{label: rank} on one task by the mean ``metric`` over seeds (1 = highest) of the headline TEMPO and every baseline with a
    result; equal means share the best rank (1, 1, 3), unlike the averaged ranks of the Friedman statistics (:func:`rank_tests`)."""
    from scipy import stats

    row = D.cells.get(env, {})
    labs = [l for l in [D.main] + list(D.cfg.baselines) if l in row]
    if not labs:
        return {}
    return {l: int(r) for l, r in zip(labs, stats.rankdata([-getattr(row[l], metric)[0] for l in labs], method="min"))}


#: the criteria of the rank table in strict priority order (benchmarks/PROTOCOL.md, 2026-11-13 and 2026-11-15): sample
#: efficiency (the fewest steps to reach the highest return and stay there), then the final return, which only separates algorithms with equal
#: steps (:func:`priority_ranks`). The training time is not a criterion: the runs shared one machine, so their wall-clock
#: times are not comparable
RANK_ORDER = ("steps", "final")
#: share of the best algorithm's gain over the random policy that counts as reaching the highest return: the steps of the first
#: criterion are counted to this level, and only algorithms whose mean final return reaches it compete on them
RANK_REACH = 0.95
#: decimals at which two algorithms count as equal on a criterion (steps as a share of the budget: 1e-4 of 500,000 steps = 50
#: steps, far below one evaluation interval; final return as a share of the best gain)
RANK_DECIMALS = dict(steps=4, final=3)


def priority_ranks(recs) -> list:
    """Rank (1 = best; equal algorithms share the best rank) of the algorithms of ONE task in the strict priority order of
    :data:`RANK_ORDER`; ``recs`` = that task's rows of :func:`overall_scores` (dicts with top, speed, final and, optionally,
    share = the final return's share of the best gain before clipping, which then orders instead of ``final``).

    1. Algorithms that reach the highest return (``top``) come before those that do not.
    2. Among them: the fewest steps to the highest return (``speed`` = share of the budget left, mean over the seeds); if equal, the
       higher final return.
    3. The others follow by their final return (their steps do not count: they do not hold the level at the end of training).

    The final return is compared as its share of the best gain, unclipped where the rows give it (``share``), so two
    algorithms below the random policy are still ordered by their return."""
    def key(r) -> tuple:
        return (0 if r["top"] else 1, -round(r["speed"], RANK_DECIMALS["steps"]) if r["top"] else 0.0,
                -round(r.get("share", r["final"]), RANK_DECIMALS["final"]))

    keys = [key(r) for r in recs]
    return [1 + sum(k2 < k for k2 in keys) for k in keys]


def overall_scores(D: Data, reach: float = RANK_REACH):
    """One row per task and algorithm (the headline TEMPO and every baseline with a result): the two criteria of the rank table
    and the ``rank`` (1 = best; :func:`priority_ranks`). With ``gain`` = best mean final return of the task minus the random
    policy's return, an algorithm reaches the highest return (``top``) when its mean final return is at least ``reach`` of ``gain``
    above the random policy. Strict priority order: among the ``top`` algorithms the fewest ``steps`` to the highest return, then
    the higher final return; the others follow by their final return.

    * ``steps``: per seed, the environment steps until the moving average of the return reaches ``reach`` of ``gain`` and stays
      there until the end of training (:func:`steps_to_hold`: a curve that reaches the level and falls back is counted from its
      last return to the level); a seed that is below the level at the end counts as the full budget; mean over the seeds.
      ``speed`` = the same as the share of the budget left (1 - steps / budget), ``reached`` = number of seeds that hold the level;
    * ``final``: mean final return over seeds as a share of ``gain`` above the random policy (clipped to [0, 1]).

    Tasks without a random-policy return, or whose best final return is not above it, are left out."""
    import pandas as pd

    rows = []
    for env in D.tasks:
        cells, rnd = D.cells.get(env, {}), D.randoms.get(env)
        labs = [l for l in [D.main] + list(D.cfg.baselines) if l in cells]
        if not labs or rnd is None:
            continue
        gain = max(cells[l].final[0] for l in labs) - rnd
        if not gain > 0:
            continue
        level = rnd + reach * gain
        recs = []
        for l in labs:
            runs = D.rows(env)[l]
            steps = [steps_to_hold(r, level, D.cfg.ma_window) for r in runs]
            final = float(np.clip((cells[l].final[0] - rnd) / gain, 0.0, 1.0))
            recs.append(dict(task=env, algorithm=l, n=len(runs), final_return=float(cells[l].final[0]),
                             steps=float(np.mean([min(st, r.budget) if np.isfinite(st) else r.budget for st, r in zip(steps, runs)])),
                             reached=int(np.isfinite(steps).sum()),
                             top=bool(final >= reach - 1e-9), final=final, share=float((cells[l].final[0] - rnd) / gain),
                             speed=float(np.mean([1.0 - st / r.budget if np.isfinite(st) else 0.0 for st, r in zip(steps, runs)]))))
        for r, rk in zip(recs, priority_ranks(recs)):
            r["rank"] = int(rk)
        rows += recs
    cols = ["task", "algorithm", "n", "final_return", "steps", "reached", "top", "final", "speed", "rank"]
    return pd.DataFrame(rows, columns=cols)


def _rank_steps(x: float) -> str:
    """Environment steps for a table cell: 42k, 1.5k, 800."""
    return f"{x / 1000:.0f}k" if x >= 9950 else f"{x / 1000:.1f}k" if x >= 1000 else f"{x:.0f}"


def _rank_share(f: float) -> str:
    """Final return as a share of the best gain for a table cell of an algorithm below the level of :data:`RANK_REACH`: whole per
    cent, but one decimal, rounded down, where the whole number would show the level itself (94.8%, never 95%)."""
    return f"{100 * f:.0f}%" if round(100 * f) < 100 * RANK_REACH else f"{math.floor(1000 * f + 1e-9) / 10:.1f}%"


def rank_podium(D: Data) -> tuple:
    """(markdown, LaTeX) table of the overall rank of every algorithm on every task with a TEMPO result (:func:`overall_scores`:
    strict priority order of steps to the highest return, then final return; family order, as the curve grid), the
    toy-text tasks (PPO and DQN references) included; a markdown cell gives the rank and in
    brackets the mean steps to the highest return of an algorithm that reaches it, else its final return as a share of the best gain;
    a cell with fewer than the expected seeds is marked (n=k) in the markdown and listed in a comment of the LaTeX; a star marks the highest mean
    final return of the task (:func:`task_ranks`, ties included). LaTeX: ranks 1-3 circled (rank 1 bold), lower ranks plain, a dash
    where the algorithm does not apply to the action space; the last rows count, over the tasks each algorithm ran on, its first
    places and its stars. The LaTeX is a bare ``tabular`` for ``\\input`` inside a table environment (booktabs, tikz)."""
    labels = [D.main] + list(D.cfg.baselines)
    sc = overall_scores(D)
    sc = sc[sc["task"].isin([e for e in D.tasks if D.main in D.cells.get(e, {})])] if len(sc) else sc
    if not len(sc):
        return "No task has a finished TEMPO run yet.\n", None
    first, best, ran = Counter(), Counter(), Counter()
    circled = {1: r"\rankone", 2: r"\ranktwo", 3: r"\rankthree"}
    md_rows, body, prev, partial = [], [], None, []
    for env, g in sc.groupby("task", sort=False):
        rk = dict(zip(g["algorithm"], g["rank"]))
        note = {r.algorithm: _rank_steps(r.steps) if r.top else _rank_share(r.final) for r in g.itertuples()}
        nseeds = dict(zip(g["algorithm"], g["n"]))
        partial += [f"{re.sub(r'-v[0-9]+$', '', env)} {l} (n={nseeds[l]})" for l in rk if nseeds[l] < D.cfg.n_expected]
        star = {l for l, r in task_ranks(D, env).items() if r == 1}
        for l, r in rk.items():
            ran[l] += 1
            first[l] += r == 1
            best[l] += l in star
        name, untuned = re.sub(r"-v\d+$", "", env), D.is_untuned(env)
        miss = {l: D.status(env, l) for l in labels if l not in rk}
        md_rows.append([name + (" †" if untuned else "")]
                       + [((f"**{rk[l]}**" if rk[l] == 1 else str(rk[l])) + f" ({note[l]})" + ("*" if l in star else "")
                           + (f" (n={nseeds[l]})" if nseeds[l] < D.cfg.n_expected else "")) if l in rk
                          else "pending" if miss[l] == "pending" else "-" for l in labels])
        group = RANK_GROUPS.get(R.family(env).split("-")[0], R.family(env))
        if group != prev:
            if prev is not None:
                body.append(r"\midrule")
            body.append(rf"\multicolumn{{{len(labels) + 1}}}{{@{{}}l}}{{\emph{{{group}}}}} \\")
            prev = group
        cells = [(circled.get(rk[l], str(rk[l])) + (r"$^\star$" if l in star else r"\phantom{$^\star$}")) if l in rk
                 else r"\textit{pending}" if miss[l] == "pending" else r"\textemdash" for l in labels]
        body.append(" & ".join([R._tex_escape(name) + (r"$^\dagger$" if untuned else "")] + cells) + r" \\")
    wins = [f"{first[l]}/{ran[l]}" if ran[l] else "-" for l in labels]
    stars = [f"{best[l]}/{ran[l]}" if ran[l] else "-" for l in labels]
    md = (R.md_table(["Task"] + labels, md_rows + [["First places / tasks run"] + wins, ["Highest final return (*) / tasks run"] + stars],
                     align="l" + "c" * len(labels))
          + "\nOverall rank (1 = best) in a strict priority order: (1) the fewest environment steps to reach the highest return and stay there, (2) the "
          "higher final return, which only separates algorithms with equal steps (the training time is not a criterion). "
          f"An algorithm reaches the highest return when its mean final return is within {100 * (1 - RANK_REACH):g}% "
          "of the best algorithm's gain over the random policy. In brackets: for such an algorithm the steps until the "
          f"{D.cfg.ma_window}-evaluation moving average of its return reaches that level and no longer falls below it (mean over the "
          "seeds; a seed that is below it at the end counts as the full budget); for the others, which follow ordered by their final return, the final "
          "return as a share of the best gain (`overall_scores`, `priority_ranks`; every criterion in "
          "`tables/rank_overall_scores.csv`). * = highest mean final return of the task (ties "
          f"included). - = {_na_text(D, list(dict.fromkeys(sc['task'])))}; pending = expected, no finished run."
          + (" † = some or all baselines ran with library defaults (untuned)."
             if any(D.is_untuned(t) for t in dict.fromkeys(sc["task"])) else "")
          + (f" (n=k): fewer than the {D.cfg.n_expected} expected seeds: {', '.join(partial)}." if partial else "")
          + _partly_note(D, list(dict.fromkeys(sc["task"]))) + "\n")
    tex = "\n".join(["% Generated by Comparison/complab.py (rank_podium); \\input inside a table environment; requires \\usepackage{booktabs,tikz}.",
                     "% Rank: strict priority order (fewest steps to the highest return, then final return).",
                     *([f"% PARTIAL cells (fewer than {D.cfg.n_expected} seeds): {', '.join(partial)}"] if partial else []),
                     RANK_TEX_MACROS, r"\begin{tabular}{@{}l" + "c" * len(labels) + r"@{}}", r"\toprule",
                     " & ".join(["Task"] + [R._tex_escape(l) for l in labels]) + r" \\", r"\midrule", *body, r"\midrule",
                     " & ".join(["First places"] + wins) + r" \\", " & ".join([r"Highest return ($\star$)"] + stars) + r" \\",
                     r"\bottomrule", r"\end{tabular}", ""])
    return md, tex


def isolated_hours(D: Data | None = None) -> tuple:
    """``({(task, label): hours per run}, [(task, label) of D left with their wall-clock])`` from the training-time measurements of
    ``Comparison/timing`` (:mod:`timelab`): ``total_h``, the derived wall-clock of one full run measured alone (training and
    evaluations; the median over repeats, as timelab's table), read from every measurement file timelab counts as current ('done':
    today's settings, code and libraries), not from the saved CSV. A cell of ``D`` whose runs were not made with the timed
    configuration (other hyperparameters, other TEMPO code, another protocol) keeps its wall-clock and is listed. Used by the rank's
    robustness check (benchmarks/PROTOCOL.md, 2026-11-04 rule 8). Empty while nothing is measured."""
    import io

    here = str(pathlib.Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import timelab as tl

    smoke = bool(D is not None and D.cfg.smoke)
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = tl.make_config(smoke=smoke)                    # states and values from one timing folder (smoke: the scratch one)
        reps = [int(p.stem[3:]) for p in cfg.out.glob("*/*/*/rep*.json") if p.stem[3:].isdigit()]
        if not smoke and max(reps, default=1) > 1:
            cfg = tl.make_config(repeats=max(reps))          # the median over every repeat, as timelab's table
        items = tl.measurements(cfg)
    vals = defaultdict(list)
    for i in items:
        if i.state == "done":
            r = json.loads(i.path.read_text())
            vals[(i.task, i.label)].append(tl.derive(r["measured"], r["protocol"], r.get("time_limit"))["total_h"])
    out = {k: float(np.median(v)) for k, v in vals.items() if np.all(np.isfinite(v)) and float(np.median(v)) > 0}
    missing = []
    if D is not None:
        live = live_source_hash()
        for e in D.tasks:
            for l in [D.main, *D.cfg.baselines]:
                rs = D.rows(e).get(l)
                if not rs or l not in D.cells.get(e, {}):
                    continue
                same = all((r.budget, r.eval_freq, r.n_eval) == D.cfg.protocol_of(e) for r in rs) and (
                    all(r.tempo_source == live for r in rs) if R.is_tempo(l)
                    else all(not RL._hp_differences(l, e, r.info, r.budget) for r in rs))
                if (e, l) not in out or not same:
                    out.pop((e, l), None)
                    missing.append((e, l))
    return out, missing


def score_long(D: Data):
    """Tidy table: one row per (task, algorithm) with n, mean and CI of final return and AUC (the CSV export)."""
    import pandas as pd

    rows = []
    for env in D.tasks:
        for lab in D.labels:
            c = D.cells.get(env, {}).get(lab)
            st = D.status(env, lab)
            r = dict(task=env, family=R.family(env), algorithm=lab, status=st, origin=D.origin.get((env, lab), ""),
                     n=0 if c is None else c.n, task_baselines_untuned=D.is_untuned(env))
            for metric in ("final", "auc"):
                m, lo, hi = getattr(c, metric) if c is not None else (math.nan,) * 3
                r.update({f"{metric}_mean": m, f"{metric}_ci_lo": lo, f"{metric}_ci_hi": hi})
            rows.append(r)
    return pd.DataFrame(rows)


# ---- verdicts and TEMPO vs the best baseline -------------------------------------------
def verdict(D: Data, env: str) -> dict:
    """'TEMPO beats all baselines on <task>: YES / NO / PENDING' for the headline TEMPO.

    YES = its mean final score is strictly higher than that of every finished baseline (the criterion of the
    project's screens, ``benchmarks/PROTOCOL.md``); the same line says against how many baselines the Welch test
    (alpha 0.05, ``min_n_test`` seeds per side) calls the difference significant. PENDING = no finished TEMPO or no
    finished baseline run. PROVISIONAL = not every seed of TEMPO or an expected baseline has finished. Toy text: no
    tuned baselines exist (untuned runs are listed as references, not tested); the verdict is whether TEMPO's CI reaches
    the optimum, i.e. the lower of the exact optimum and an optimal policy's mean on the same fixed evaluation episodes
    (:func:`toy_optimal_eval`; every evaluation replays them, so an optimal policy scores 7.7 on Taxi's 10 episodes against an
    exact 7.93 and would otherwise get NO)."""
    name = D.main
    row = D.cells.get(env, {})
    out = dict(task=env, status="PENDING", provisional=False, significant_all=None, text="", beats_mean=None, wins=0, losses=0,
               ties=0, n_base=0, untuned=D.is_untuned(env))
    if env in P.TOY_TEXT_ALL:
        out["status"] = "N/A"
        head = f"{name} beats all baselines on {env}: n/a (toy text: judged against the exact optimum)"
        opt, rnd = toy_optimum(env), D.randoms.get(env)
        if D.main not in row:
            out["text"] = head + "; reaches the known optimum: PENDING (no finished TEMPO run yet)"
            out["status"] = "PENDING"
        else:
            m, lo, hi = row[D.main].final
            hi_ = hi if np.isfinite(hi) else m
            n_ev = row[D.main].n and D.runs[env][D.main][0].n_eval
            oe = toy_optimal_eval(env, n_ev) if n_ev else None
            if opt is None:
                out["text"] = head + f"; mean final {m:.4g} (optimum unavailable)"
            else:
                tgt = opt if oe is None else min(opt, oe)
                reach = hi_ >= tgt - 1e-9
                way = ("no random-policy score" if rnd is None else f"random policy {rnd:.4g}" if opt == rnd
                       else f"{(m - rnd) / (opt - rnd):.0%} of the way from the random policy {rnd:.4g}")
                rule = (f"the 95% CI reaches {tgt:.4g} = min(exact optimum, an optimal policy's mean on these {n_ev:,} evaluation episodes)"
                        if oe is not None else "the 95% CI reaches the exact optimum")
                aim = (f"the known optimum {opt:.4g}" if abs(tgt - opt) <= 1e-9 else
                       f"the optimum on its evaluation episodes {tgt:.4g} (exact optimum {opt:.4g})")
                out["text"] = (head + f"; reaches {aim}: {'YES' if reach else 'NO'} (mean final {m:.4g}, "
                               f"{way}; YES = {rule})")
                out["reaches_optimum"] = bool(reach)
            refs = sorted(((b, row[b].final[0]) for b in D.cfg.baselines if b in row), key=lambda x: -x[1])
            if refs:
                out["text"] += (("; untuned baselines" if D.is_untuned(env) else "; baselines") + " (references, not tested): "
                                + ", ".join(f"{b} {v:.4g}" for b, v in refs))
            if oe is not None and opt is not None and abs(oe - opt) > 1e-9:
                out["text"] += (f"  [every evaluation replays the same {n_ev:,} held-out episodes: an optimal policy scores {oe:.4g} on them, "
                                f"the exact expectation is {opt:.4g}]")
            if row[D.main].n < D.cfg.n_expected:
                out["text"] += f"  (PROVISIONAL: n={row[D.main].n} of {D.cfg.n_expected} seeds)"
                out["provisional"] = True
        return out
    base = {b: row[b] for b in D.cfg.baselines if b in row}
    if D.main not in row:
        out["text"] = f"{name} beats all baselines on {env}: PENDING (no finished {name} run yet)"
        return out
    if not base:
        out["text"] = f"{name} beats all baselines on {env}: PENDING (no finished baseline run yet)"
        return out
    t = row[D.main]
    bm = {b: c.final[0] for b, c in base.items()}
    best = max(bm, key=bm.get)
    res = {b: outcome(D, t.finals, c.finals) for b, c in base.items()}
    wins = [b for b, (r, _) in res.items() if r == "win"]
    losses = [b for b, (r, _) in res.items() if r == "loss"]
    ties = [b for b, (r, _) in res.items() if r == "tie"]
    untested = [b for b, (r, _) in res.items() if r == "n/a"]
    beats = t.final[0] > bm[best]
    out.update(status="YES" if beats else "NO", beats_mean=bool(beats), wins=len(wins), losses=len(losses), ties=len(ties),
               n_base=len(base), best=best, significant_all=(len(wins) == len(base)) if not untested else None)
    txt = f"{name} beats all baselines on {env}: {'YES' if beats else 'NO'}"
    out["tie_best"] = bool(t.final[0] == bm[best])
    if out["tie_best"]:
        txt += f" (tie with {best}: equal mean final scores)"
    if len(untested) == len(base):
        sig = f"not testable yet (fewer than {D.cfg.min_n_test} seeds)"
    else:
        sig = f"significantly better than {len(wins)} of {len(base)} baselines" + (f", significantly worse than {len(losses)}" if losses else "")
    txt += f"  [Welch t-test, alpha {R.ALPHA}: {sig}]"
    if out["untuned"]:
        part = _partly_untuned(D, [env]).get(env)
        txt += (f"  [baselines partly untuned: {', '.join(part)} ran with library defaults on this task]" if part
                else "  [baselines untuned: library defaults on this task]")
    why = []
    if t.n < D.cfg.n_expected:
        why.append(f"{name} n={t.n} of {D.cfg.n_expected} seeds")
    short = [f"{b} n={c.n}" for b, c in base.items() if c.n < D.cfg.n_expected]
    if short:
        why.append("baselines with fewer seeds: " + ", ".join(short))
    pend = [b for b in D.expected_labels(env) if b != D.main and b not in base]
    if pend:
        why.append("baselines pending: " + ", ".join(pend))
    if why:
        txt += "  (PROVISIONAL: " + "; ".join(why) + ")"
        out["provisional"] = True
    out["text"] = txt
    return out


def verdict_tag(v: dict) -> str:
    """Short qualifier of a YES / NO verdict: sig. (Welch-significant against every baseline) / n.s. / sig. worse, prov."""
    parts = []
    if v["status"] == "YES":
        sa = v.get("significant_all")
        parts.append("sig." if sa else "n.s." if sa is False else "untested")
    elif v["status"] == "NO":
        parts.append("tie" if v.get("tie_best") else "sig. worse" if v.get("losses") else "n.s.")
    if v.get("provisional"):
        parts.append("prov.")
    return ", ".join(parts)


def verdicts(D: Data) -> list:
    return [verdict(D, e) for e in D.tasks]


def vs_best_frame(D: Data):
    """Per task: headline TEMPO against the best baseline (by mean final score, and by AUC): means, difference,
    the difference on the normalised scale, Welch p (and the Holm-adjusted p over the tasks) and outcome; plus the verdict."""
    import pandas as pd

    rows = []
    for env in D.tasks:
        v = verdict(D, env)
        r = dict(task=env, tempo_n=D.n(env, D.main), verdict=v["status"], provisional=v["provisional"], verdict_tag=verdict_tag(v),
                 baselines_untuned=D.is_untuned(env))
        row = D.cells.get(env, {})
        nz = D.norm.get(env, {})
        b, res, p, tm, bm = main_vs_best(D, env, "final")
        tm = row[D.main].final[0] if D.main in row else math.nan
        r.update(tempo_final=row[D.main].final[0] if D.main in row else math.nan, best_baseline=b or "", baseline_n=(row[b].n if b else 0),
                 baseline_final=bm, diff=(tm - bm) if D.main in row and b else math.nan,
                 diff_normalised=((tm - bm) / (nz["best_baseline"] - nz["random"])
                                  if D.main in row and b and nz.get("valid") else math.nan),
                 welch_p=p, result=(res or "").upper())
        ba, resa, pa, ta, bma = main_vs_best(D, env, "auc")
        r.update(tempo_auc=ta if D.main in row else math.nan, best_auc_baseline=ba or "", baseline_auc=bma,
                 auc_diff=(ta - bma) if D.main in row and ba else math.nan, auc_welch_p=pa, auc_result=(resa or "").upper())
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["welch_p_holm"] = holm(df["welch_p"].to_numpy(float))
        df["auc_welch_p_holm"] = holm(df["auc_welch_p"].to_numpy(float))
    return df


VS_BEST_HEAD = ["Task", "TEMPO", "Best baseline", "Diff", "Diff (norm.)", "p", "Final", "AUC diff", "p (AUC)", "AUC", "Verdict"]


def _vs_best_rows(df) -> list:
    rows = []
    for r in df.itertuples(index=False):
        d = R._decimals([r.tempo_final, r.baseline_final])
        task = r.task + (" (untuned)" if getattr(r, "baselines_untuned", False) else "")
        if not np.isfinite(r.tempo_final):
            rows.append([task, "pending", "-", "", "", "", "", "", "", "", r.verdict])
            continue
        if not r.best_baseline:
            none = "not compared (toy text)" if r.task in P.TOY_TEXT_ALL else "pending"
            rows.append([task, f"{_num(r.tempo_final, d)} (n={r.tempo_n})", none, "", "", "", "", "", "", "", r.verdict])
            continue
        da = R._decimals([r.tempo_auc, r.baseline_auc])
        rows.append([task, f"{_num(r.tempo_final, d)} (n={r.tempo_n})", f"{r.best_baseline} {_num(r.baseline_final, d)} (n={r.baseline_n})",
                     _sdec(r.diff, d), "" if not np.isfinite(r.diff_normalised) else _s2(r.diff_normalised), _pf(r.welch_p), r.result or "n/a",
                     f"{_sdec(r.auc_diff, da)} vs {r.best_auc_baseline}", _pf(r.auc_welch_p), r.auc_result or "n/a",
                     r.verdict + (f" ({r.verdict_tag})" if r.verdict_tag else "")])
    return rows


def vs_best_md(D: Data, df=None) -> str:
    df = vs_best_frame(D) if df is None else df
    rows = _vs_best_rows(df)
    note = ("Best baseline = highest mean final return on the task (AUC columns: highest mean AUC). Diff (norm.) = difference divided by "
            "(best baseline - random policy); empty where the task cannot be normalised. p = two-sided Welch t-test on the per-seed scores, "
            f"alpha 0.05, needs {D.cfg.min_n_test} seeds per side, NOT corrected for the number of tasks (Holm-adjusted p: CSV); Final / AUC = WIN, TIE, LOSS "
            "of TEMPO. Verdict YES = TEMPO's mean final score is higher than every finished baseline's; in brackets sig. = significantly better than every "
            "baseline, n.s. = at least one difference is not significant, sig. worse = significantly worse than a baseline, tie = equal mean to the best baseline, "
            "prov. = not every seed / baseline has finished. (untuned) = some or all baselines ran with library defaults." + _partly_note(D))
    return R.md_table(VS_BEST_HEAD, rows, align="lrrrrrlrrll") + f"\n{note}\n"


def vs_best_tex(D: Data, df=None) -> str:
    """Paper table: TEMPO against the best baseline per task (final return, Welch p and outcome, AUC outcome, verdict)."""
    df = vs_best_frame(D) if df is None else df
    rows = _vs_best_rows(df)
    keep = [0, 1, 2, 4, 5, 6, 9, 10]                 # task, TEMPO, best baseline, diff (norm.), p, final outcome, AUC outcome, verdict
    head = [VS_BEST_HEAD[i] for i in keep]

    def cell(v):
        return re.sub(r"(?<![\w.])-(?=\d)", "$-$", R._tex_escape(v)).replace("<", "$<$")
    body = [" & ".join(cell(r[i]) for i in keep) + r" \\" for r in rows]
    cap = ("TEMPO against the best baseline of each task. Diff (norm.): difference of the mean final scores divided by (best baseline - random policy); "
           r"$p$: two-sided Welch $t$-test on the per-seed final scores, $\alpha=0.05$, not corrected for multiple tasks; Final / AUC: WIN, TIE or LOSS of TEMPO; "
           "Verdict YES: TEMPO's mean final "
           "score is higher than every finished baseline's (sig.: significantly better than every baseline, n.s.: not significant, prov.: not every seed or baseline "
           "has finished).")
    return "\n".join(["% Generated by Comparison/complab.py; requires \\usepackage{booktabs,graphicx}.", r"\begin{table*}[t]", r"\centering\small",
                      rf"\caption{{{cap}}}", r"\label{tab:tempo-vs-best}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}lrrrrlll@{}}",
                      r"\toprule", " & ".join(head) + r" \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])


# ---- win / tie / loss ------------------------------------------------------------------
def wtl_records(D: Data, tasks=None, metric: str = "final") -> list:
    """Per task with a headline-TEMPO run: Welch outcome of TEMPO against every baseline and the per-task best (as
    ``report.win_tie_loss``, but with ``min_n_test`` seeds per side and without toy-text tasks, which are judged against the exact optimum)."""
    f = "finals" if metric == "final" else "aucs"
    out = []
    for env in (D.tasks if tasks is None else tasks):
        row = D.cells.get(env, {})
        if D.main not in row or env in P.TOY_TEXT_ALL:
            continue
        t = row[D.main]
        rec = dict(env=env, tempo_mean=getattr(t, metric)[0], tempo_n=t.n, vs={})
        best, _ = best_baseline(D, env, metric)
        for b in list(BASELINES) + [BEST]:
            src = best if b == BEST else b
            if not src or src not in row:
                continue
            res, p = outcome(D, getattr(t, f), getattr(row[src], f))
            rec["vs"][b] = dict(algo=src, mean=getattr(row[src], metric)[0], n=row[src].n, p=p, result=res)
        out.append(rec)
    return out


def _wtl_counts(D: Data, metric: str, tasks) -> dict:
    """{opponent: dict(win, tie, loss, untested, tasks)} over ``tasks`` (opponents: each baseline and the per-task best)."""
    recs = wtl_records(D, tasks, metric)
    out = {}
    for b in list(BASELINES) + [BEST]:
        res = [rec["vs"][b]["result"] for rec in recs if b in rec["vs"]]
        if res:
            out[b] = dict(win=res.count("win"), tie=res.count("tie"), loss=res.count("loss"), untested=res.count("n/a"), tasks=len(res))
    return out


def wtl_frame(D: Data):
    """Win / tie / loss of the headline TEMPO against every baseline, overall and per task family, for final return and AUC."""
    import pandas as pd

    scopes = [("All tasks", list(D.tasks))] + [(FAMILY_NAMES.get(f, f), [e for e in D.tasks if R.family(e) == f])
                                              for f in dict.fromkeys(R.family(e) for e in D.tasks)]
    rows = []
    for scope, tasks in scopes:
        if scope != "All tasks" and len(scopes) == 2:
            continue
        for metric in ("final", "auc"):
            for b, c in _wtl_counts(D, metric, tasks).items():
                rows.append(dict(scope=scope, metric=metric, opponent="Best baseline (per task)" if b == BEST else b, win=c["win"],
                                 tie=c["tie"], loss=c["loss"], untested=c["untested"], tasks=c["tasks"]))
    return pd.DataFrame(rows, columns=["scope", "metric", "opponent", "win", "tie", "loss", "untested", "tasks"])


def wtl_family_md(w) -> str:
    """Per task family, final return: 'win/tie/loss' of TEMPO against every opponent (the inline version of the CSV)."""
    g = w[(w["scope"] != "All tasks") & (w["metric"] == "final")]
    if not len(g):
        return ""
    fams = list(dict.fromkeys(g["scope"]))
    opps = list(dict.fromkeys(g["opponent"]))
    rows = []
    for o in opps:
        r = [o]
        for f in fams:
            x = g[(g["scope"] == f) & (g["opponent"] == o)]
            r.append("-" if not len(x) else f"{int(x['win'].iloc[0])}/{int(x['tie'].iloc[0])}/{int(x['loss'].iloc[0])}"
                     + (f" ({int(x['untested'].iloc[0])} untested)" if int(x["untested"].iloc[0]) else ""))
        rows.append(r)
    return R.md_table(["Opponent (win/tie/loss)"] + fams, rows, align="l" + "c" * len(fams))


def wtl_task_md(D: Data) -> str:
    """Per task: outcome (WIN / TIE / LOSS and p) of the headline TEMPO against every baseline (final return)."""
    per_task = wtl_records(D, None, "final")
    head = ["Task", "TEMPO"] + list(D.cfg.baselines)
    rows = []
    for rec in per_task:
        if not rec["vs"]:
            continue
        d = R._decimals([rec["tempo_mean"]] + [v["mean"] for v in rec["vs"].values()])
        r = [D.task_name(rec["env"]), f"{_num(rec['tempo_mean'], d)} (n={rec['tempo_n']})"]
        for b in D.cfg.baselines:
            v = rec["vs"].get(b)
            r.append("" if v is None else f"{v['result'].upper()} (p{_pf(v['p']) if _pf(v['p']).startswith('<') else '=' + _pf(v['p'])})"
                     if v["result"] != "n/a" else f"n/a (n<{D.cfg.min_n_test})")
        rows.append(r)
    return R.md_table(head, rows, align="lr" + "c" * len(D.cfg.baselines)) if rows else "No task has a finished TEMPO run yet.\n"


def show_scores(D: Data) -> None:
    """Section 4: score tables, TEMPO vs the best baseline, win/tie/loss, verdicts."""
    if not D.has_data:
        print("pending: no results yet, no table to show.")
        return
    _mark(D, "scores")
    for metric, title in (("final", "Final return per task"), ("auc", "AUC (learning speed) per task")):
        md, tex = score_table_md(D, metric), score_table_tex(D, metric)
        register(D, f"{metric}_scores", title, md=md, tex=tex)
        show_md(f"#### {title}\n\n" + md)
    pmd, ptex = rank_podium(D)
    register(D, "rank_podium", "Overall rank of every algorithm on every task (strict priority order: steps to the highest return, "
             "then final return)", md=pmd, tex=ptex)
    register(D, "rank_overall_scores", "The criteria of the overall rank per task: steps to the highest return, final return",
             df=overall_scores(D).round(4))
    show_md("#### Overall rank of every algorithm on every task (fewest steps to the highest return, then final return; "
            "* = highest final return)\n\n" + pmd)
    register(D, "scores_long", "Final return and AUC, one row per task and algorithm", df=score_long(D))
    df = vs_best_frame(D)
    md = vs_best_md(D, df)
    register(D, "tempo_vs_best_baseline", "TEMPO against the best baseline per task", df=df, md=md, tex=vs_best_tex(D, df))
    show_md("#### TEMPO against the best baseline on every task\n\n" + md)
    w = wtl_frame(D)
    register(D, "win_tie_loss", "Win / tie / loss of TEMPO against every baseline", df=w)
    tm = wtl_task_md(D)
    register(D, "win_tie_loss_per_task", "Per task: TEMPO vs each baseline (Welch)", md=tm)
    if len(w):
        show_md(f"#### Win / tie / loss of TEMPO (Welch t-test on per-seed scores, alpha 0.05, uncorrected; untested = fewer than {D.cfg.min_n_test} seeds on a side)\n")
        for (scope, metric), g in w.groupby(["scope", "metric"], sort=False):
            if scope == "All tasks":
                show_md(f"**{scope}, {'final return' if metric == 'final' else 'AUC'}**\n\n" + df_to_md(g.drop(columns=["scope", "metric"])))
        fm = wtl_family_md(w)
        if fm:
            show_md("**Per task family, final return** (win/tie/loss; the AUC version is in `tables/win_tie_loss.csv`)\n\n" + fm)
        show_md("**Per task, final return**\n\n" + tm)
    vs = verdicts(D)
    D._extra["verdicts"] = vs
    print("\nVerdict per task (YES = TEMPO's mean final score is higher than every finished baseline's; sig./n.s./prov. in the executive summary):")
    for v in vs:
        print("  " + v["text"])
    c = Counter(v["status"] for v in vs)
    print(f"\n{c.get('YES', 0)} YES, {c.get('NO', 0)} NO, {c.get('PENDING', 0)} PENDING, {c.get('N/A', 0)} n/a (toy text).")


# =========================================================================== 5. aggregate statistics
ALL_SCOPE = "All tasks"
ACTION_SCOPES = ("Discrete-action tasks", "Continuous-action tasks")
TUNED_SCOPE = "Tasks with tuned baselines"


def optimality_gap_ci(scores: dict, n_boot: int, key, target: float = 1.0) -> list:
    """Optimality gap (Agarwal et al., 2021): mean over tasks of the mean shortfall max(target - score, 0) below the
    target (1 = the best baseline), with a stratified bootstrap (runs resampled within each task). 0 = never below it."""
    rng = R._rng("og", *key)
    point = float(np.mean([np.maximum(target - v, 0).mean() for v in scores.values()]))
    res = [v[rng.integers(0, v.size, size=(n_boot, v.size))] for v in scores.values()]
    boots = np.mean([np.maximum(target - r, 0).mean(1) for r in res], axis=0)
    return [point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def _all_single(scores: dict) -> bool:
    """True when every task of the row has a single run: resampling runs within tasks then varies nothing (a zero-width 'CI')."""
    return all(v.size < 2 for v in scores.values())


def _agg_ci(scores: dict, n_boot: int, key) -> dict:
    """``report.aggregate_ci`` with NaN intervals when no task has two or more runs."""
    a = R.aggregate_ci(scores, n_boot, key)
    if _all_single(scores):
        a = {k: [v[0], math.nan, math.nan] for k, v in a.items()}
    return a


_NULL_CACHE: dict = {}


def _best_of_null(nt: int, k: int, nb: int, reps: int = 20000) -> float:
    """P(a TEMPO run beats a run of the post-hoc best of ``k`` baselines) when TEMPO and the baselines are equally good (iid
    normal scores, ``nt`` TEMPO seeds, ``nb`` seeds per baseline, the best baseline picked by its mean on the same runs)."""
    if k <= 1 or nt < 1 or nb < 1:
        return 0.5
    key = (nt, k, nb)
    if key not in _NULL_CACHE:
        rng = R._rng("null-best", *key)
        t = rng.standard_normal((reps, nt))
        B = rng.standard_normal((reps, k, nb))
        sel = B[np.arange(reps), B.mean(2).argmax(1)]
        _NULL_CACHE[key] = float((t[:, :, None] > sel[:, None, :]).mean())
    return _NULL_CACHE[key]


def best_selection_null(D: Data, tasks) -> float:
    """Chance level of P(TEMPO > best baseline) given that the best baseline is picked per task on the same seeds (mean over tasks)."""
    vals = []
    for e in tasks:
        row = D.cells.get(e, {})
        bs = [row[b].n for b in BASELINES if b in row]
        if D.main in row and bs:
            vals.append(_best_of_null(row[D.main].n, len(bs), int(round(float(np.mean(bs))))))
    return float(np.mean(vals)) if vals else 0.5


def aggregates(D: Data, tasks: list, scope_key: str = "all") -> dict | None:
    """Normalised-score aggregates over ``tasks`` (random 0, best baseline 1, as report.py): IQM / median / mean /
    optimality gap with stratified-bootstrap CIs per algorithm, performance profiles, probability of improvement of the
    headline TEMPO over every baseline (with the chance level of the post-hoc best baseline), and TEMPO vs each baseline
    on the tasks both ran on. Intervals are NaN where no task has two runs; ``cover[l]`` = (tasks of l, tasks of the scope)."""
    if not tasks:
        return None
    cells = {e: D.cells[e] for e in tasks}
    algos = [l for l in D.labels if any(l in cells[e] for e in tasks)] + [BEST]
    scores = {l: R.normalised_scores(cells, D.norm, l) for l in algos}
    scores = {l: s for l, s in scores.items() if s}
    if not scores:
        return None
    nb = D.cfg.n_boot
    pooled = np.concatenate([v for s in scores.values() for v in s.values()])
    tau_max = float(np.clip(math.ceil(np.percentile(pooled, 97) * 4) / 4, 1.25, 3.0))
    taus = np.linspace(0.0, tau_max, 241)
    agg, prof = {}, {}
    for l, s in scores.items():
        agg[l] = dict(tasks=sorted(s, key=R.task_key), n_runs=int(sum(v.size for v in s.values())), single=sum(1 for v in s.values() if v.size < 2),
                      **_agg_ci(s, nb, (l, scope_key)))
        og = optimality_gap_ci(s, nb, (l, scope_key))
        pr = R.performance_profile(s, taus, min(nb, 1000), (l, scope_key))
        if _all_single(s):
            og = [og[0], math.nan, math.nan]
            pr = (pr[0], np.full(pr[0].shape, np.nan), np.full(pr[0].shape, np.nan))
        agg[l]["OG"] = og
        prof[l] = pr
    raw = {l: R.raw_finals(cells, D.norm, l) for l in algos}
    poi, pairwise = {}, {}
    if raw.get(D.main):
        for b in list(D.cfg.baselines) + [BEST]:
            if raw.get(b):
                r = R.prob_improvement(raw[D.main], raw[b], nb, (D.main, b, scope_key))
                if r:
                    if all(raw[D.main][t].size < 2 and raw[b][t].size < 2 for t in r["tasks"]):
                        r["lo"] = r["hi"] = math.nan
                    r["separated"] = bool(np.isfinite(r["lo"]) and r["hi"] - r["lo"] < 1e-9)
                    poi[b] = r
            common = {t: v for t, v in scores.get(D.main, {}).items() if t in scores.get(b, {})}
            if common:
                pairwise[b] = dict(tasks=sorted(common, key=R.task_key),
                                   tempo=_agg_ci(common, nb, (D.main, "common", b, scope_key)),
                                   other=_agg_ci({t: scores[b][t] for t in common}, nb, (b, "common", D.main, scope_key)))
    null = {BEST: best_selection_null(D, poi[BEST]["tasks"])} if BEST in poi else {}
    cover = {l: (len(a["tasks"]), len(tasks)) for l, a in agg.items()}
    return dict(scope=scope_key, tasks=tasks, scores=scores, agg=agg, profiles=prof, taus=taus, poi=poi, poi_null=null,
                pairwise=pairwise, cover=cover)


def agg_scopes(D: Data) -> list:
    """[(name, tasks)]: all tasks in the aggregate scope, then the like-for-like task sets (discrete-action and
    continuous-action tasks when both exist; the tasks with tuned baselines when some are untuned), then each family."""
    allt = aggregate_scope(D)
    if not allt:
        return []
    out = [(ALL_SCOPE, allt)]
    disc = [e for e in allt if action_space(e) == "discrete"]
    cont = [e for e in allt if action_space(e) == "continuous"]
    if disc and cont:
        out += [(ACTION_SCOPES[0], disc), (ACTION_SCOPES[1], cont)]
    tuned = [e for e in allt if not D.is_untuned(e)]
    if tuned and len(tuned) < len(allt):
        out.append((TUNED_SCOPE, tuned))
    fams = list(dict.fromkeys(R.family(e) for e in allt))
    if len(fams) > 1:
        out += [(FAMILY_NAMES.get(f, f), [e for e in allt if R.family(e) == f]) for f in fams]
    return out


def _ci3(v) -> str:
    if not (np.isfinite(v[1]) and np.isfinite(v[2])):
        return f"{v[0]:.2f} [n/a]"
    return f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"


def aggregate_frame(results: dict):
    """Tidy table of all scopes: IQM, median, mean, optimality gap (point, CI) per algorithm."""
    import pandas as pd

    rows = []
    for scope, A in results.items():
        for l, a in A["agg"].items():
            r = dict(scope=scope, algorithm="Best baseline (per task)" if l == BEST else l, tasks=len(a["tasks"]), scope_tasks=len(A["tasks"]),
                     single_seed_tasks=a["single"], runs=a["n_runs"])
            for k, name in (("IQM", "IQM"), ("median", "median"), ("mean", "mean"), ("OG", "optimality_gap")):
                r.update({name: a[k][0], f"{name}_lo": a[k][1], f"{name}_hi": a[k][2]})
            rows.append(r)
    return pd.DataFrame(rows)


def _poi_text(v: dict, d: int = 3) -> str:
    if not np.isfinite(v["lo"]):
        return f"{v['p']:.{d}f} [n/a]"
    t = f"{v['p']:.{d}f} [{v['lo']:.{d}f}, {v['hi']:.{d}f}]"
    return t + (" (complete separation: the bootstrap interval is degenerate)" if v.get("separated") else "")


def aggregate_md(results: dict, D: Data) -> str:
    """Overall: IQM / median / mean / optimality gap and P(TEMPO > X) per algorithm; per scope: IQM and P(TEMPO > best) matrices
    (all numbers of every scope are in the CSV)."""
    out = []
    first_name, A = next(iter(results.items()))
    out.append(f"**{first_name}** ({_pl(len(A['tasks']), 'task')}: {', '.join(A['tasks'])})\n")

    def tasks_cell(l, a):
        k, n = A["cover"][l]
        return (f"{k} of {n}" if k < n else str(k)) + (f" ({a['single']} with 1 run)" if a["single"] else "")
    rows = [["Best baseline* (per task)" if l == BEST else l, tasks_cell(l, a), str(a["n_runs"]), _ci3(a["IQM"]), _ci3(a["median"]),
             _ci3(a["mean"]), _ci3(a["OG"])] for l, a in A["agg"].items()]
    out.append(R.md_table(["Algorithm", "Tasks", "Runs", "IQM", "Median", "Mean", "Optimality gap"], rows))
    notes = []
    if any(k < n for k, n in A["cover"].values()):
        notes.append("**Rows with fewer tasks than the scope (k of N) ran on other tasks, so they are not comparable with the rows over all N tasks** "
                     "(for example DQN only runs on discrete-action tasks, SAC / TD3 / CrossQ only on continuous ones): compare them in the "
                     "like-for-like task sets below.")
    if any(np.isnan(a["IQM"][1]) for a in A["agg"].values()):
        notes.append("[n/a]: every task of the row has a single run, so there is no resampling variance and no interval.")
    if any(a["single"] for a in A["agg"].values()):
        notes.append("Tasks with a single run add no resampling variance: intervals of rows with such tasks are too narrow.")
    if BEST in A["agg"]:
        notes.append("*Best baseline is picked per task on the same runs: its score is 1 by construction and its row is selection-biased (flattered), "
                     "and so is the comparison with it: see the chance level below.")
    if notes:
        out.append("\n".join(notes) + "\n")
    if A["poi"]:
        null = A.get("poi_null", {})
        out.append(R.md_table(["X", "Tasks", f"P({D.main} > X) [95% CI]", "chance level (equal skill)"],
                              [["Best baseline* (per task)" if b == BEST else b, str(len(v["tasks"])), _poi_text(v),
                                f"{null[b]:.2f} (post-hoc best of the baselines)" if b in null else "0.50"] for b, v in A["poi"].items()]))
    fam = {k: v for k, v in results.items() if k != first_name}
    if fam:
        labs = list(dict.fromkeys(l for v in fam.values() for l in v["agg"]))
        head = ["IQM (tasks)"] + list(fam)
        rows = []
        for l in labs:
            r = ["Best baseline*" if l == BEST else l]
            for v in fam.values():
                a = v["agg"].get(l)
                r.append(f"{_ci3(a['IQM'])} ({len(a['tasks'])})" if a else "-")
            rows.append(r)
        out.append("**Like-for-like task sets and families: IQM of the normalised score (number of tasks)**\n")
        out.append(R.md_table(head, rows))
        rows = []
        for b in dict.fromkeys(b for v in fam.values() for b in v["poi"]):
            r = ["Best baseline*" if b == BEST else b]
            for v in fam.values():
                p = v["poi"].get(b)
                r.append(("-" if not p else f"{p['p']:.2f} [n/a]" if not np.isfinite(p["lo"]) else f"{p['p']:.2f} [{p['lo']:.2f}, {p['hi']:.2f}]"
                          + (" deg." if p.get("separated") else "")))
            rows.append(r)
        if rows:
            out.append(f"**Like-for-like task sets and families: P({D.main} > X)** (deg. = complete separation, degenerate interval; chance level 0.50, "
                       "for the post-hoc Best baseline* lower, see its chance level above)\n")
            out.append(R.md_table(["X"] + list(fam), rows))
    return "\n".join(out)


def _interval_panel(ax, D: Data, items, xref, xlabel: str, ylabels: bool = True, names=None, faded=(), marks=None,
                    ref_kind: str = "best") -> None:
    """items: [(label, point, lo, hi)] top to bottom; thin translucent bar = 95% CI, vertical bar = point estimate; ``faded`` labels
    (an algorithm that ran on only some of the tasks) are drawn pale; ``marks`` = {label: x} short dotted ticks (chance levels);
    ``xref`` = a vertical reference line of kind ``ref_kind`` (best baseline level or chance)."""
    for i, (lab, p, lo, hi) in enumerate(items):
        st = style_of(D, lab)
        y = len(items) - 1 - i
        fa = lab in faded
        if np.isfinite(lo):
            ax.plot([lo, hi], [y, y], color=st["color"], lw=_CI_LW, alpha=0.2 if fa else _CI_ALPHA, solid_capstyle="butt", zorder=2)
        ax.plot([p], [y], marker="|", ms=_TICK_MS, mew=_TICK_MEW, color=st["color"], alpha=0.35 if fa else 1.0, zorder=3)
        if marks and lab in marks:
            ax.plot([marks[lab]] * 2, [y - 0.38, y + 0.38], color=INK, lw=1.0, ls=(0, (1, 1.2)), zorder=4)
    if xref is not None:
        ref_line(ax, xref, ref_kind, vertical=True)
    ax.set_ylim(-0.6, len(items) - 0.4)
    ax.set_yticks(range(len(items))[::-1])
    names = names or [i[0] for i in items]
    ax.set_yticklabels(names)
    ax.tick_params(axis="y", length=0, labelleft=ylabels)
    for tick, (lab, *_r) in zip(ax.get_yticklabels(), items):
        if lab == D.main:
            tick.set_fontweight("bold")
        elif lab in faded:
            tick.set_color(MUTED)
    ax.grid(False, axis="y")
    ax.grid(True, axis="x")
    ax.set_xlabel(xlabel)


def _profile_panel(ax, D: Data, A: dict, title: str, skip=(), faded=(), ylabel: str = r"Fraction of runs with score $>\tau$") -> None:
    """Performance profiles (step lines in the algorithms' styles, TEMPO bolder) with their 95% bootstrap band."""
    for l in A["profiles"]:
        if l in skip:
            continue
        st = style_of(D, l)
        p, lo, hi = A["profiles"][l]
        fa = l in faded
        if np.isfinite(lo).any() and not fa:
            ax.fill_between(A["taus"], lo, hi, color=st["color"], alpha=0.12, lw=0, step="post", zorder=1.5 + st["z"] / 100)
        ax.step(A["taus"], p, where="post", color=st["color"], ls=st["ls"], lw=_line_width(D, l), alpha=0.4 if fa else 1.0,
                zorder=4.5 if l == D.main else 3 + st["z"] / 20, label="Best baseline*" if l == BEST else l)
    ref_line(ax, 1.0, "best", vertical=True)
    ax.set_xlim(A["taus"][0], A["taus"][-1])
    ax.set_ylim(0, 1.02)
    _norm_ticks([ax], "x", nbins=5)
    _norm_ticks([ax], "y", nbins=4)
    ax.set_xlabel(r"Normalised score $\tau$")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def _figure_algos(A: dict) -> tuple:
    """(drawn labels, faded labels, omitted labels): an algorithm that ran on fewer than half of the scope's tasks is not drawn,
    one that ran on only some of them is faded (its numbers average other tasks than the rest)."""
    shown, faded, omitted = [], set(), []
    for l in A["agg"]:
        k, n = A["cover"][l]
        if k / n < 0.5:
            omitted.append(f"{l} ({k} of {n} tasks)")
            continue
        shown.append(l)
        if k < n:
            faded.add(l)
    return shown, faded, omitted


def _agg_name(A: dict, l: str) -> str:
    k, n = A["cover"][l]
    return f"{'Best baseline*' if l == BEST else l} ({k} of {n})" if k < n else f"{'Best baseline*' if l == BEST else l} ({k})"


def plot_aggregate(D: Data, A: dict, stem: str = "aggregate_overall", single: bool = False):
    """Overall aggregate figure: IQM, median, mean and optimality gap of the normalised score with 95% stratified-bootstrap
    CIs per algorithm, the performance profile and the probability of improvement P(TEMPO > X). Algorithms that ran on fewer than
    half of the tasks are left out of the panels (listed in the caption), those on only some of them are drawn pale.
    ``single`` = the one-column layout for a two-column paper (the four statistics in a 2 x 2 block, the profile and the
    probability of improvement below them)."""
    labels, faded, omitted = _figure_algos(A)
    names = [_agg_name(A, l) for l in labels]
    no_colour = {l for l in labels if not _colour_ok(D, l)}    # told apart by colour alone in the profile: left out there
    h_top = 0.2 * len(labels) + 0.75
    with figure_style() as plt:
        if single:
            h_row = 0.15 * len(labels) + 0.5                  # compact rows: the figure fills most of one column
            fig = plt.figure(figsize=(fs.SINGLE, 3 * h_row + 1.45 + 0.75), layout="constrained")
            gs = fig.add_gridspec(4, 2, height_ratios=[h_row, h_row, 1.45, h_row])
            cells_top = [gs[0, 0], gs[0, 1], gs[1, 0], gs[1, 1]]
            cell_profile, cell_poi = gs[2, :], gs[3, :]
        else:
            fig = plt.figure(figsize=(fs.DOUBLE, h_top + 2.35 + 0.45), layout="constrained")
            gs = fig.add_gridspec(2, 4, height_ratios=[h_top, 2.35])
            cells_top = [gs[0, k] for k in range(4)]
            cell_profile, cell_poi = gs[1, :3], gs[1, 3]
        stats = (("IQM", "IQM", fs.NORMALISED_LABEL), ("median", "Median", fs.NORMALISED_LABEL), ("mean", "Mean", fs.NORMALISED_LABEL),
                 ("OG", "Optimality gap", "Gap (lower is better)"))
        top = []
        for k, (stat, ttl, xl) in enumerate(stats):
            ax = fig.add_subplot(cells_top[k], sharey=top[0] if top else None)
            top.append(ax)
            _interval_panel(ax, D, [(l,) + tuple(A["agg"][l][stat]) for l in labels], 0.0 if stat == "OG" else 1.0, xl,
                            ylabels=(k % 2 == 0 if single else k == 0),
                            names=names, faded=faded)
            ax.set_title(ttl)
        _norm_ticks(top, "x", nbins=4)                     # one tick rule / decimals on the four normalised panels
        axp = fig.add_subplot(cell_profile)
        _profile_panel(axp, D, A, "Performance profile", skip=(set(A["agg"]) - set(labels)) | no_colour, faded=faded)
        axq = fig.add_subplot(cell_poi)
        poi = {b: v for b, v in A["poi"].items() if len(v["tasks"]) / len(A["tasks"]) >= 0.5}
        null = A.get("poi_null", {}).get(BEST)
        if poi:
            items = [(b, v["p"], v["lo"], v["hi"]) for b, v in poi.items()]
            pf = {b for b, v in poi.items() if len(v["tasks"]) < len(A["tasks"])}
            _interval_panel(axq, D, items, 0.5, f"P({D.main} > X)", faded=pf, marks=A.get("poi_null"), ref_kind="chance",
                            names=[f"{'Best baseline*' if b == BEST else b} ({len(poi[b]['tasks'])})" for b in poi])
            if not single:
                axq.yaxis.tick_right()
            axq.set_xlim(0, 1)
            axq.set_xticks([0, 0.5, 1])
            axq.set_title("Probability of improvement")
        else:
            axq.text(0.5, 0.5, f"no {D.main} runs yet", ha="center", va="center", color=INK2, transform=axq.transAxes)
            axq.set_xticks([])
            axq.set_yticks([])
        from matplotlib.lines import Line2D

        extra = [(Line2D([], [], color=INK, lw=1.0, ls=(0, (1, 1.2))), "Chance vs Best baseline*")] if poi and null else []
        fs.legend(fig, _legend_handles(D, labels), _ref_handles({"best"} | ({"chance"} if poi else set()),
                                                                {"best": "Best baseline level (score 1, gap 0)"}) + extra)
        cap = [f"Figure. Aggregate normalised final score over {_pl(len(A['tasks']), 'task')}: (x - random policy) / (best baseline mean - "
               "random policy), so 0 = random policy and 1 = best baseline. Top: IQM (25% trimmed mean over all runs), median and mean over "
               "tasks, optimality gap (mean shortfall below the best baseline); vertical bar = point estimate, horizontal bar = 95% "
               "stratified-bootstrap CI (runs resampled within tasks). Bottom left: performance profile (fraction of runs above a score, "
               "shaded = 95% bootstrap band); bottom right: probability that a random run of "
               f"{D.main} beats a random run of X on a common task (chance 0.5).",
               f"(k) = tasks the algorithm ran on among the {len(A['tasks'])} in this scope ({D.cfg.agg_tasks!r}: "
               + ("tasks where TEMPO has runs" if D.cfg.agg_tasks == "tempo" and D.has_main else "all tasks with a baseline result")
               + "); pale = ran on only some of them (not comparable with the full rows: see the like-for-like figure)"
               + (f"; not drawn: {', '.join(omitted)}" if omitted else "") + ".",
               "*Best baseline = the baseline with the highest mean final score on each task, picked on the same runs (it scores 1 by "
               "construction and is flattered by the selection)"
               + (f"; dotted tick = chance level of P({D.main} > post-hoc best baseline) with equal skill, {null:.2f}" if poi and null else "")
               + "."]
        if no_colour:
            cap.append(_no_colour_note("Not in the performance profile", sorted(no_colour)))
        if D.cfg.smoke:
            cap.append("SMOKE-sized runs.")
        return _finish(D, fig, stem, caption=" ".join(cap))


def plot_aggregate_by_family(D: Data, results: dict, stem: str = "aggregate_by_family", names=None, label: str = "families"):
    """Per scope (task family by default), one row each: IQM interval of the normalised score (left) and the performance profile
    (right)."""
    fam = ({k: v for k, v in results.items() if k in names} if names is not None else
           {k: v for k, v in results.items() if k != ALL_SCOPE and k not in ACTION_SCOPES and k != TUNED_SCOPE})
    if not fam:
        print(f"No {label} to compare in the aggregate scope: see the overall figure.")
        return []
    rows_h = [max(1.3, 0.17 * len(_figure_algos(A)[0]) + 0.6) for A in fam.values()]
    with figure_style() as plt:
        fig = plt.figure(figsize=(fs.DOUBLE, sum(rows_h) + 0.45), layout="constrained")
        gs = fig.add_gridspec(len(fam), 2, height_ratios=rows_h, width_ratios=[1.0, 1.35])
        allab, notes, iqm_axes, prof_axes, no_colour = [], [], [], [], set()
        for j, (scope, A) in enumerate(fam.items()):
            labels, faded, omitted = _figure_algos(A)
            allab += [l for l in labels if l not in allab]
            no_colour |= {l for l in labels if not _colour_ok(D, l)}
            ax = fig.add_subplot(gs[j, 0])
            iqm_axes.append(ax)
            _interval_panel(ax, D, [(l,) + tuple(A["agg"][l]["IQM"]) for l in labels], 1.0,
                            "IQM normalised score" if j == len(fam) - 1 else "", ylabels=True, names=[_agg_name(A, l) for l in labels],
                            faded=faded)
            ax.set_title(f"{scope} ({_pl(len(A['tasks']), 'task')})", loc="left")
            axp = fig.add_subplot(gs[j, 1])
            prof_axes.append(axp)
            _profile_panel(axp, D, A, "", skip=(set(A["agg"]) - set(labels)) | no_colour, faded=faded, ylabel=r"Fraction $>\tau$")
            if j != len(fam) - 1:
                axp.set_xlabel("")
            if omitted:
                notes.append(f"{scope}: not drawn {', '.join(omitted)}")
        _norm_ticks(iqm_axes, "x", nbins=4)                # same tick rule and decimals on every row
        _norm_ticks(prof_axes, "x", nbins=5)
        _norm_ticks(prof_axes, "y", nbins=4)
        fs.legend(fig, _legend_handles(D, [l for l in D.labels + [BEST] if l in allab]),
                  _ref_handles({"best"}, {"best": "Best baseline level (score 1)"}))
        cap = [f"Figure. Normalised final score per {'task family' if names is None else 'like-for-like task set'} "
               "(0 = random policy, 1 = best baseline of the task). Left: IQM with 95% stratified-bootstrap CI ((k) = tasks the algorithm "
               "ran on); right: performance profile (fraction of runs scoring above tau, shaded = 95% bootstrap band). Scopes with a "
               "single task are noisy; *Best baseline is picked on the same runs (flattered)."
               + (" " + _join(notes) + "." if notes else "")]
        if no_colour:
            cap.append(_no_colour_note("Not in the performance profiles", sorted(no_colour)))
        if D.cfg.smoke:
            cap.append("SMOKE-sized runs.")
        return _finish(D, fig, stem, caption=" ".join(cap))


# ---- cross-task rank statistics (M01) and pairwise comparison (M03) -----------------------
def rank_sets(D: Data) -> list:
    """[(name, tasks, algos)]: task sets (all tasks; discrete-action; continuous-action) with the algorithms that have a result on EVERY
    task of the set: complete blocks for rank statistics (the headline TEMPO and the baselines, not the other TEMPO variants). A pixel
    task (``plans.IMAGE``: CarRacing-v3, where sb3-contrib CrossQ cannot run) would drop CrossQ from the whole set; it is left out of
    the set instead when that keeps more algorithms (``D._extra["rank_left_out"]``, named under the rank tables)."""
    out, left = [], {}
    scopes = dict(agg_scopes(D))
    for name in (ALL_SCOPE, *ACTION_SCOPES):
        tasks = scopes.get(name)
        if not tasks:
            continue
        full = lambda ts: [l for l in [D.main] + list(D.cfg.baselines) if all(l in D.cells.get(e, {}) for e in ts)]  # noqa: E731
        algos, rest = full(tasks), [e for e in tasks if e not in P.IMAGE]
        if rest and len(rest) < len(tasks) and len(full(rest)) > len(algos):
            left[name] = [e for e in tasks if e not in rest]
            tasks, algos = rest, full(rest)
        if D.main in algos and len(algos) >= 2:
            out.append((name, tasks, algos))
    D._extra["rank_left_out"] = left
    return out


def rank_tests(D: Data) -> dict:
    """Mean rank per algorithm (rank 1 = highest mean final score per task, ties averaged), Friedman test over the tasks, and the Wilcoxon
    signed-rank test of the headline TEMPO against every other algorithm on the per-task mean normalised scores (Holm-corrected over the
    opponents of a set). The Wilcoxon test needs at least 6 tasks (with fewer, no p below 0.05 exists); Friedman needs 3 tasks and 3 algorithms."""
    import pandas as pd
    from scipy import stats

    per_rows, sum_rows, wil_rows = [], [], []
    for name, tasks, algos in rank_sets(D):
        M = np.array([[D.cells[e][l].final[0] for l in algos] for e in tasks], float)
        ranks = np.vstack([stats.rankdata(-row) for row in M])
        fried = math.nan
        if len(algos) >= 3 and len(tasks) >= 3:
            try:
                fried = float(stats.friedmanchisquare(*[M[:, j] for j in range(len(algos))]).pvalue)
            except ValueError:
                pass
        for e, rk in zip(tasks, ranks):
            per_rows.append(dict(scope=name, task=e, **{l: float(r) for l, r in zip(algos, rk)}))
        for j, l in enumerate(algos):
            sum_rows.append(dict(scope=name, algorithm=l, tasks=len(tasks), algorithms=len(algos), mean_rank=float(ranks[:, j].mean()),
                                 first_place=int((ranks[:, j] == 1).sum()), friedman_p=fried))
        sc = {l: R.normalised_scores({e: D.cells[e] for e in tasks}, D.norm, l) for l in algos}
        recs = []
        for b in algos:
            if b == D.main:
                continue
            d = np.array([sc[D.main][e].mean() - sc[b][e].mean() for e in tasks if e in sc[D.main] and e in sc[b]])
            p = math.nan
            if d.size >= 6:
                if not np.any(d != 0):
                    p = 1.0
                else:
                    try:    # zero differences dropped; exact null distribution up to 25 tasks (scipy's 'auto' switches to the
                        nz = d[d != 0]  # normal approximation as soon as one difference is zero)
                        p = float(stats.wilcoxon(nz, alternative="two-sided", method="exact").pvalue if nz.size <= 25 else
                                  stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="asymptotic").pvalue)
                    except ValueError:
                        pass
            recs.append(dict(scope=name, opponent=b, tasks=int(d.size), tempo_better=int((d > 0).sum()), tempo_worse=int((d < 0).sum()),
                             median_diff_normalised=float(np.median(d)) if d.size else math.nan, wilcoxon_p=p))
        for r, ph in zip(recs, holm([r["wilcoxon_p"] for r in recs])):
            r["wilcoxon_p_holm"] = ph
        wil_rows += recs
    return dict(per_task=pd.DataFrame(per_rows), summary=pd.DataFrame(sum_rows), wilcoxon=pd.DataFrame(wil_rows))


def rank_md(D: Data, rt: dict) -> str:
    out = []
    summ, wil = rt["summary"], rt["wilcoxon"]
    for name in dict.fromkeys(summ["scope"]) if len(summ) else []:
        g = summ[summ["scope"] == name].sort_values("mean_rank")
        t0 = g.iloc[0]
        fr = t0["friedman_p"]
        gone = D._extra.get("rank_left_out", {}).get(name)
        out.append(f"**{name}** ({_pl(int(t0['tasks']), 'task')}, {int(t0['algorithms'])} algorithms with a result on every one of them"
                   + (f"; left out: {', '.join(gone)}, where CrossQ cannot run (pixel observations), so that CrossQ stays in" if gone else "")
                   + ")\n")
        out.append(R.md_table(["Algorithm", "Mean rank (1 = best)", "Tasks ranked first"],
                              [[r.algorithm, f"{r.mean_rank:.2f}", str(r.first_place)] for r in g.itertuples()]))
        out.append("Friedman test (all algorithms equal): " + ("n/a (needs 3 tasks and 3 algorithms)" if not np.isfinite(fr) else f"p = {_pf(fr)}") + "\n")
        w = wil[wil["scope"] == name]
        if len(w):
            out.append(R.md_table([f"{D.main} vs", "Tasks", f"{D.main} better / worse", "Median diff (norm.)", "Wilcoxon p", "p (Holm)"],
                                  [[r.opponent, str(r.tasks), f"{r.tempo_better} / {r.tempo_worse}", _s2(r.median_diff_normalised),
                                    _pf(r.wilcoxon_p) if np.isfinite(r.wilcoxon_p) else "n/a (needs >= 6 tasks)",
                                    _pf(r.wilcoxon_p_holm) if np.isfinite(r.wilcoxon_p_holm) else "n/a"] for r in w.itertuples()]))
    return "\n".join(out)


def pairwise_matrix(D: Data) -> tuple:
    """P(row > column) for every pair of algorithms of each rank set (mean over the tasks of the probability that a random run of the row beats
    a random run of the column; 0.5 = indistinguishable). Returns (long DataFrame with 95% CIs, Markdown)."""
    import pandas as pd

    rows, md = [], []
    nb = min(D.cfg.n_boot, 500)
    for name, tasks, algos in rank_sets(D):
        raw = {l: {e: np.asarray(D.cells[e][l].finals, float) for e in tasks} for l in algos}
        P_ = {}
        for a in algos:
            for b in algos:
                if a != b:
                    r = R.prob_improvement(raw[a], raw[b], nb, ("pair", name, a, b))
                    P_[(a, b)] = r
                    rows.append(dict(scope=name, row=a, column=b, tasks=len(tasks), p=r["p"], lo=r["lo"], hi=r["hi"]))
        table = [[a] + ["-" if a == b else f"{P_[(a, b)]['p']:.2f}" for b in algos] for a in algos]
        gone = D._extra.get("rank_left_out", {}).get(name)
        md.append(f"**{name}** ({_pl(len(tasks), 'task')}"
                  + (f"; {', '.join(gone)} left out: CrossQ cannot run on pixel observations" if gone else "") + ")\n\n"
                  + R.md_table(["P(row > column)"] + algos, table, align="l" + "r" * len(algos)))
    return pd.DataFrame(rows), "\n".join(md)


_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def _norm_at(D: Data, env: str, label: str) -> np.ndarray:
    """Mean over seeds of the normalised (trailing 3-evaluation average) return at 25 / 50 / 75 / 100% of the budget."""
    nz = D.norm[env]
    scale = nz["best_baseline"] - nz["random"]
    vals = []
    for r in D.runs[env][label]:
        st, y = smoothed(r, D.cfg.ma_window)
        vals.append((np.interp(np.asarray(_FRACTIONS) * r.budget, st, y) - nz["random"]) / scale)
    return np.mean(vals, axis=0)


def budget_fraction_frame(D: Data):
    """Normalised score at 25 / 50 / 75 / 100% of the budget per algorithm and scope (mean over the scope's tasks the algorithm ran on)."""
    import pandas as pd

    rows = []
    for name, tasks in agg_scopes(D):
        for lab in [D.main] + list(D.cfg.baselines):
            ts = [e for e in tasks if lab in D.cells.get(e, {})]
            if not ts:
                continue
            M = np.array([_norm_at(D, e, lab) for e in ts])
            rows.append(dict(scope=name, algorithm=lab, tasks=len(ts), scope_tasks=len(tasks),
                             **{f"at {int(f * 100)}% of budget": float(M[:, i].mean()) for i, f in enumerate(_FRACTIONS)}))
    return pd.DataFrame(rows)


def show_aggregates(D: Data) -> dict:
    """Section 5: aggregate statistics over tasks (overall, like-for-like task sets, per family), rank tests, figures and tables."""
    _mark(D, "aggregates")
    scopes = agg_scopes(D)
    if not scopes:
        print("pending: no task has both a baseline result and a usable normalisation yet, nothing to aggregate"
              + ("." if D.has_main else " (and TEMPO has no runs yet)."))
        weak = [(e, D.norm[e]["reason"]) for e in D.tasks if e in D.cells and not D.norm[e]["valid"] and any(b in D.cells[e] for b in BASELINES)
                and e not in P.TOY_TEXT_ALL]
        for e, why in weak:
            print(f"  left out: {e}: {why}")
        return {}
    if not D.has_main:
        print(f"note: {D.main} has no runs yet, so only the baselines are aggregated (over all tasks with a baseline result).")
    elif D.cfg.agg_tasks == "tempo":
        n_all = len([e for e in D.tasks if D.norm.get(e, {}).get("valid")])
        print(f"Aggregates cover the {_pl(len(scopes[0][1]), 'task')} where {D.main} has runs (of {_pl(n_all, 'task')} that can be normalised; "
              "AGG_TASKS = 'all' uses every task).")
    bad = [e for e in D.tasks if e in D.cells and not D.norm[e]["valid"] and any(b in D.cells[e] for b in BASELINES) and e not in P.TOY_TEXT_ALL]
    for e in bad:
        print(f"Not normalisable, left out: {e}: {D.norm[e]['reason']}")
    results = {name: aggregates(D, tasks, name) for name, tasks in scopes}
    results = {k: v for k, v in results.items() if v}
    D._extra["aggregates"] = results
    fr = aggregate_frame(results)
    md = aggregate_md(results, D)
    register(D, "aggregate_statistics", "Aggregate normalised scores (IQM, median, mean, optimality gap) and probability of improvement", df=fr, md=md)
    first = next(iter(results.values()))
    plot_aggregate(D, first)
    plot_aggregate(D, first, stem="aggregate_overall_single", single=True)   # the same figure for one column of a paper
    plot_aggregate_by_family(D, results, stem="aggregate_like_for_like", names=set(ACTION_SCOPES) | {TUNED_SCOPE},
                             label="like-for-like task sets (one action space or one kind of baseline tuning)")
    plot_aggregate_by_family(D, results)
    show_md("#### Aggregate statistics (normalised final score; 95% stratified-bootstrap CI)\n\n" + md)
    pw = first["pairwise"]
    if pw:
        rows = [["Best baseline* (per task)" if b == BEST else b, str(len(v["tasks"])), _ci3(v["tempo"]["IQM"]), _ci3(v["other"]["IQM"]),
                 _ci3(v["tempo"]["mean"]), _ci3(v["other"]["mean"])] for b, v in pw.items()]
        t = R.md_table(["X", "Tasks", f"{D.main} IQM", "X IQM", f"{D.main} mean", "X mean"], rows)
        register(D, "aggregate_pairwise_common_tasks", f"{D.main} vs X on the tasks both ran on", md=t)
        show_md(f"#### {D.main} against each baseline on the tasks both ran on (normalised score)\n\n" + t)
    # cross-task rank statistics
    rt = rank_tests(D)
    D._extra["rank_tests"] = rt
    if len(rt["summary"]):
        rmd = rank_md(D, rt)
        register(D, "rank_summary", "Mean rank of each algorithm over tasks (and Friedman test)", df=rt["summary"])
        register(D, "rank_per_task", "Rank of each algorithm on every task (1 = highest mean final score)", df=rt["per_task"])
        register(D, "wilcoxon_vs_tempo", f"Wilcoxon signed-rank test of {D.main} against each algorithm over tasks (Holm-corrected)", df=rt["wilcoxon"])
        register(D, "rank_tests", "Cross-task rank statistics", md=rmd)
        show_md("#### Ranks and cross-task tests (mean final score per task; like-for-like sets: only algorithms with a result on every task of the set)\n\n"
                + rmd + "\n*Wilcoxon signed-rank test on the per-task mean normalised scores, two-sided, Holm-corrected over the opponents of a set; "
                "it needs at least 6 tasks. The Friedman test needs 3 algorithms with a result on every task. Rank 1 = highest mean.*")
        pm, pmd = pairwise_matrix(D)
        if len(pm):
            register(D, "pairwise_win_probability", "P(row > column) for every pair of algorithms (like-for-like task sets)", df=pm.round(4), md=pmd)
            show_md("#### Probability that a random run of the row beats a random run of the column (all pairs; 95% CIs in the CSV)\n\n" + pmd)
    else:
        print("Rank statistics: pending (they need TEMPO and at least one baseline with a result on every task of a like-for-like set).")
    bf = budget_fraction_frame(D)
    if len(bf):
        register(D, "normalised_score_at_budget_fractions", "Normalised score at 25 / 50 / 75 / 100% of the budget", df=bf.round(4))
        rows = []
        for r in bf.to_dict("records"):
            rows.append([r["scope"], r["algorithm"], f"{r['tasks']} of {r['scope_tasks']}" if r["tasks"] < r["scope_tasks"] else str(r["tasks"])]
                        + [_f2(r[f"at {int(f * 100)}% of budget"]) for f in _FRACTIONS])
        show_md("#### Normalised score at fixed shares of the budget (trailing 3-evaluation average; mean over the tasks of the scope)\n\n"
                + R.md_table(["Scope", "Algorithm", "Tasks", "25%", "50%", "75%", "100%"], rows, align="llr" + "r" * 4))
    return results


# =========================================================================== 6. sample efficiency
def level_target(D: Data, env: str, level: float):
    """Return that counts as ``level`` of the way from the random policy to the reference score of the task: the best
    baseline's mean final score (default) or gymnasium's solved threshold (``reference = 'solved'``). None if unknown."""
    rnd = D.randoms.get(env)
    if D.cfg.reference == "solved":
        ref = solved_threshold(env)
    else:
        nz = D.norm.get(env)
        ref = nz["best_baseline"] if nz and nz["valid"] else None
    if rnd is None or ref is None or ref <= rnd:
        return None
    return float(rnd + level * (ref - rnd))


def smoothed(run, window: int) -> tuple:
    """(steps, moving average of the evaluation returns over ``window`` evaluations, trailing, like the 3-evaluation final score)."""
    c = run.curve
    if len(c) < window:
        return run.steps, c
    return run.steps[window - 1:], np.convolve(c, np.ones(window) / window, mode="valid")


def steps_to_reach(run, target: float, window: int = 3) -> float:
    """First evaluation step at which the moving average of the evaluation return reaches ``target`` (inf = never)."""
    st, y = smoothed(run, window)
    hit = np.flatnonzero(y >= target)
    return float(st[hit[0]]) if hit.size else math.inf


def steps_to_hold(run, target: float, window: int = 3) -> float:
    """First evaluation step from which the moving average of the evaluation return stays at or above ``target`` until the end
    of the run (inf = it is below ``target`` at the end): reaching the level and falling back does not count."""
    st, y = smoothed(run, window)
    ok = y >= target
    if not ok.size or not ok[-1]:
        return math.inf
    below = np.flatnonzero(~ok)
    return float(st[below[-1] + 1]) if below.size else float(st[0])


def _median_steps(st) -> float:
    """Median first-hit step over seeds; inf ('never') unless MORE than half of the seeds reach the level."""
    st = np.asarray(st, float)
    return math.inf if int(np.isfinite(st).sum()) * 2 <= st.size else float(np.median(st))


def reach_stats(D: Data) -> dict:
    """{(task, label, level): dict(steps=[per seed], median, n_reached, n, target)} for every level in LEVELS (cached)."""
    if "reach" in D._extra:
        return D._extra["reach"]
    out = {}
    for env in D.tasks:
        for lab, runs in D.rows(env).items():
            for lv in LEVELS:
                tg = level_target(D, env, lv)
                if tg is None:
                    continue
                st = [steps_to_reach(r, tg, D.cfg.ma_window) for r in runs]
                out[(env, lab, lv)] = dict(steps=st, median=_median_steps(st), n_reached=int(np.isfinite(st).sum()), n=len(st), target=tg)
    D._extra["reach"] = out
    return out


def _steps_text(x: float, reached: int, n: int) -> str:
    return ("never" if not np.isfinite(x) else R.fmt_steps(int(round(x)))) + f" ({reached}/{n})"


def reach_frame(D: Data):
    import pandas as pd

    rows = []
    for (env, lab, lv), s in reach_stats(D).items():
        rows.append(dict(task=env, algorithm=lab, level=lv, target_return=s["target"], n=s["n"], n_reached=s["n_reached"],
                         median_steps=s["median"] if np.isfinite(s["median"]) else math.nan,
                         median_budget_share=s["median"] / D.budget(env) if np.isfinite(s["median"]) else math.nan))
    return pd.DataFrame(rows, columns=["task", "algorithm", "level", "target_return", "n", "n_reached", "median_steps", "median_budget_share"])


def reach_table_md(D: Data, level: float) -> str:
    """Per task x algorithm: median environment steps to reach ``level`` of the way to the reference (k/n seeds that do)."""
    rs = reach_stats(D)
    head = ["Task", "Target"] + list(D.labels)
    rows = []
    for env in D.tasks:
        tg = level_target(D, env, level)
        if tg is None or not D.rows(env):
            continue
        d = R._decimals([tg])
        r = [D.task_name(env), _num(tg, d)]
        for l in D.labels:
            s = rs.get((env, l, level))
            r.append(_steps_text(s["median"], s["n_reached"], s["n"]) if s else (D.status(env, l) if D.status(env, l) in ("pending", "n/a") else "-"))
        rows.append(r)
    return R.md_table(head, rows, align="lr" + "r" * len(D.labels)) if rows else "No task has a reference score and a run yet.\n"


def _ratio(base_steps: float, tempo_steps: float) -> float:
    """baseline steps / TEMPO steps (> 1: TEMPO faster). inf: only TEMPO reaches the level; 0: only the baseline; NaN: neither."""
    if np.isfinite(tempo_steps) and np.isfinite(base_steps):
        return base_steps / tempo_steps
    if np.isfinite(tempo_steps):
        return math.inf
    if np.isfinite(base_steps):
        return 0.0
    return math.nan


def speedup_frame(D: Data):
    """Per task and level: median steps of the headline TEMPO to reach the level, and of two baselines: the best baseline by
    final score and the FASTEST baseline (fewest median steps; the best-final one if none reaches it); speed-up = baseline
    steps / TEMPO steps (> 1: TEMPO faster). inf: only TEMPO reaches it; 0: only the baseline."""
    import pandas as pd

    rs = reach_stats(D)
    rows = []
    for env in D.tasks:
        b, _ = best_baseline(D, env, "final")
        if b is None or D.main not in D.rows(env):
            continue
        for lv in LEVELS:
            t, o = rs.get((env, D.main, lv)), rs.get((env, b, lv))
            if not t or not o:
                continue
            cand = [(rs[(env, x, lv)]["median"], x) for x in D.rows(env) if x in BASELINES and (env, x, lv) in rs]
            fin = [c for c in cand if np.isfinite(c[0])]
            fb = min(fin)[1] if fin else b
            f = rs[(env, fb, lv)]
            rows.append(dict(task=env, level=lv, best_baseline=b, tempo_steps=t["median"], tempo_reached=f"{t['n_reached']}/{t['n']}",
                             baseline_steps=o["median"], baseline_reached=f"{o['n_reached']}/{o['n']}", speedup=_ratio(o["median"], t["median"]),
                             fastest_baseline=fb, fastest_steps=f["median"], fastest_reached=f"{f['n_reached']}/{f['n']}",
                             speedup_fastest=_ratio(f["median"], t["median"]), budget=D.budget(env)))
    return pd.DataFrame(rows, columns=["task", "level", "best_baseline", "tempo_steps", "tempo_reached", "baseline_steps", "baseline_reached",
                                       "speedup", "fastest_baseline", "fastest_steps", "fastest_reached", "speedup_fastest", "budget"])


def _sp_text(s: float) -> str:
    return "n/a" if np.isnan(s) else "only TEMPO" if np.isinf(s) else "only baseline" if s == 0 else f"{s:.2f}x"


def _reached(txt: str) -> tuple:
    a, b = txt.split("/")
    return int(a), int(b)


def speedup_md(D: Data, df=None) -> str:
    """Steps to the levels and speed-up of TEMPO against the best baseline by final score."""
    df = speedup_frame(D) if df is None else df
    if not len(df):
        return "No task has a finished headline-TEMPO run and a finished baseline run with a reference score yet.\n"
    head = ["Task", "Best baseline"] + [f"{int(lv * 100)}%: TEMPO / baseline / speed-up" for lv in LEVELS]
    rows = []
    for env, g in df.groupby("task", sort=False):
        r = [D.task_name(env), g["best_baseline"].iloc[0]]
        for lv in LEVELS:
            x = g[g["level"] == lv]
            if not len(x):
                r.append("-")
                continue
            x = x.iloc[0]
            r.append(f"{_steps_text(x['tempo_steps'], *_reached(x['tempo_reached']))} / "
                     f"{_steps_text(x['baseline_steps'], *_reached(x['baseline_reached']))} / {_sp_text(x['speedup'])}")
        rows.append(r)
    note = ("Median over seeds of the first evaluation step whose trailing 3-evaluation average reaches the level (level = share of the way from the "
            "random policy to the " + ("solved threshold" if D.cfg.reference == "solved" else "best baseline's mean final score")
            + "; (k/n) = seeds that reach it; never = at most half of the seeds reach it). Speed-up = baseline steps / TEMPO steps, vs the best baseline "
            "BY FINAL SCORE (which is not always the fastest one: see the next table); > 1x = TEMPO is faster.")
    return R.md_table(head, rows, align="ll" + "r" * len(LEVELS)) + f"\n{note}\n"


def speedup_fastest_md(D: Data, df=None) -> str:
    """Speed-up of TEMPO against the FASTEST baseline of each task (the one with the fewest median steps to the level)."""
    df = speedup_frame(D) if df is None else df
    if not len(df):
        return ""
    head = ["Task"] + [f"{int(lv * 100)}%: fastest baseline, steps / speed-up" for lv in LEVELS]
    rows = []
    for env, g in df.groupby("task", sort=False):
        r = [D.task_name(env)]
        for lv in LEVELS:
            x = g[g["level"] == lv]
            if not len(x):
                r.append("-")
                continue
            x = x.iloc[0]
            r.append(f"{x['fastest_baseline']} {_steps_text(x['fastest_steps'], *_reached(x['fastest_reached']))} / {_sp_text(x['speedup_fastest'])}")
        rows.append(r)
    note = ("Same as above, but against the baseline that reaches the level in the fewest median steps (the best-final baseline when none reaches it): "
            "the demanding comparison for a sample-efficiency claim.")
    return R.md_table(head, rows, align="l" + "r" * len(LEVELS)) + f"\n{note}\n"


def _geo(x) -> float:
    x = np.asarray(x, float)
    return float(np.exp(np.log(x).mean())) if x.size else math.nan


def speedup_summary(D: Data, df=None) -> dict:
    """Per level over tasks, against the best-final baseline and against the fastest baseline: tasks where both reach the level
    (geometric-mean speed-up, median, tasks where TEMPO is faster), tasks where only TEMPO / only the baseline reaches it, and the
    CENSORED estimate that counts 'never' as the full budget (an upper bound when only baselines fail to be matched, a lower bound when only
    TEMPO reaches; it avoids the survivorship bias of averaging only the tasks where both reach)."""
    df = speedup_frame(D) if df is None else df
    out = {}
    for lv in LEVELS:
        g = df[df["level"] == lv] if len(df) else df
        if not len(g):
            continue
        s = dict(tasks=len(g))
        for tag, bcol, scol in (("", "baseline_steps", "speedup"), ("fastest_", "fastest_steps", "speedup_fastest")):
            sp = g[scol]
            both = g[np.isfinite(sp) & (sp > 0)]
            cens = []
            for r in g.itertuples():
                b_, t_ = getattr(r, bcol), r.tempo_steps
                if np.isfinite(b_) or np.isfinite(t_):
                    cens.append((b_ if np.isfinite(b_) else r.budget) / (t_ if np.isfinite(t_) else r.budget))
            n_ot, n_ob = int(np.isinf(sp).sum()), int((sp == 0).sum())
            s.update({f"{tag}both": len(both), f"{tag}geo_mean": _geo(both[scol]) if len(both) else math.nan,
                      f"{tag}median": float(both[scol].median()) if len(both) else math.nan,
                      f"{tag}only_tempo": n_ot, f"{tag}only_baseline": n_ob,
                      f"{tag}faster": int((both[scol] > 1).sum()) if len(both) else 0,
                      f"{tag}baseline_reaches": int(np.isfinite(g[bcol]).sum()),
                      f"{tag}geo_mean_censored": _geo(cens), f"{tag}censored_tasks": len(cens),
                      f"{tag}bound": ("upper" if n_ob and not n_ot else "lower" if n_ot and not n_ob else "mixed" if n_ot and n_ob else "exact")})
        out[lv] = s
    return out


def speedup_line(s: dict, tag: str = "") -> str:
    """One sentence for a level of :func:`speedup_summary` (``tag`` '' = best-final baseline, 'fastest_' = fastest baseline)."""
    t = f"TEMPO reaches it on {s[tag + 'both']} of the {_pl(s[tag + 'baseline_reaches'], 'task')} where this baseline does"
    if s[tag + "both"]:
        t += f"; geometric-mean speed-up {s[tag + 'geo_mean']:.2f}x over {'those ' + str(s[tag + 'both']) if s[tag + 'both'] > 1 else 'that task'} (faster on {s[tag + 'faster']})"
    if s[tag + "only_tempo"] or s[tag + "only_baseline"]:
        t += f"; only TEMPO reaches it on {s[tag + 'only_tempo']}, only the baseline on {s[tag + 'only_baseline']}"
        if np.isfinite(s[tag + "geo_mean_censored"]):
            bound = {"upper": "an upper bound for TEMPO", "lower": "a lower bound for TEMPO", "mixed": "neither bound", "exact": "exact"}[s[tag + "bound"]]
            t += (f"; counting 'never' as the full budget: {s[tag + 'geo_mean_censored']:.2f}x over {_pl(s[tag + 'censored_tasks'], 'task')} ({bound})")
    return t


def norm_auc_scores(D: Data, tasks=None) -> dict:
    """{label: {task: per-seed normalised AUC}}: (AUC - random) / (best baseline mean AUC - random), so that the best baseline by AUC
    scores 1 on every task (the learning-speed analogue of the normalised final score)."""
    out = defaultdict(dict)
    for env in (tasks or D.tasks):
        b, bm = best_baseline(D, env, "auc")
        rnd = D.randoms.get(env)
        if b is None or rnd is None or bm - rnd <= 1e-9 * max(1.0, abs(rnd)):
            continue
        if not reference_check(D.randoms, env, bm, D.cfg.min_gap, max(c.auc[0] for c in D.cells[env].values()), use_threshold=False)[0]:
            continue                                   # best baseline barely above the random policy: no usable AUC reference
        for l, c in D.cells.get(env, {}).items():
            out[l][env] = (np.asarray(c.aucs, float) - rnd) / (bm - rnd)
    return dict(out)


def norm_auc_matrix(D: Data):
    sc = norm_auc_scores(D)
    tasks = [e for e in D.tasks if any(e in sc.get(l, {}) for l in D.labels)]
    M = np.full((len(tasks), len(D.labels)), np.nan)
    for i, e in enumerate(tasks):
        for j, l in enumerate(D.labels):
            if e in sc.get(l, {}):
                M[i, j] = float(np.mean(sc[l][e]))
    return tasks, list(D.labels), M


def plot_auc_intervals(D: Data):
    """Normalised AUC (learning speed; 0 = random policy, 1 = best baseline by AUC) per task group as horizontal dot-and-interval rows:
    the task names on the y axis, under each task one row per algorithm that ran on it (dot = mean over seeds, thin line = 95%
    bootstrap CI over seeds; the algorithms in legend order, top to bottom). One panel at the column width, two side by side at the
    full text width for more than five tasks. Saved as ``figures/auc_normalised_<group>``."""
    from matplotlib.lines import Line2D

    sc = norm_auc_scores(D)
    if not sc:
        print("pending: no task has a baseline result yet, nothing to normalise.")
        return []
    groups = defaultdict(list)
    for env in D.tasks:
        groups[GROUPS.get(R.family(env), "Other")].append(env)
    out = []
    pitch, gap = 0.105, 0.9                                 # in per algorithm row; gap between tasks in rows
    for g, envs in groups.items():
        envs = [e for e in envs if any(e in sc.get(l, {}) for l in D.labels)]
        if not envs:
            continue
        labs = [l for l in D.labels if any(e in sc.get(l, {}) for e in envs) and _colour_ok(D, l)]
        present = {e: [l for l in labs if e in sc.get(l, {})] for e in envs}
        envs = [e for e in envs if present[e]]
        if not envs:
            continue
        stats = {(e, l): R.boot_mean_ci(sc[l][e], ("nauc", e, l), D.cfg.n_boot) for e in envs for l in present[e]}
        ncol = 1 if len(envs) <= 5 else 2
        per = math.ceil(len(envs) / ncol)
        chunks = [envs[i:i + per] for i in range(0, len(envs), per)]
        rows = max(sum(len(present[e]) for e in ch) + gap * (len(ch) - 1) for ch in chunks)
        with figure_style():
            fig, axes = fs.subplots(1, ncol, width=fs.SINGLE if ncol == 1 else fs.DOUBLE, height=rows * pitch + 1.0, squeeze=False,
                                    sharex=True)
            vals = [0.0, 1.0]
            for ax, ch in zip(axes.flat, chunks):
                y, ticks = rows, []
                for e in ch:
                    top = y
                    for l in present[e]:
                        y -= 1
                        m, lo, hi = stats[(e, l)]
                        st = style_of(D, l)
                        if np.isfinite(lo):
                            ax.plot([lo, hi], [y, y], color=st["color"], lw=fs.ERRORBAR_WIDTH, solid_capstyle="butt", zorder=2)
                            vals += [lo, hi]
                        ax.plot([m], [y], ls="", marker="o", ms=fs.MARKER_SIZE + (1.0 if l == D.main else 0.0), color=st["color"],
                                mec="white", mew=0.4, zorder=4.5 if l == D.main else 3)
                        vals.append(m)
                    ticks.append(((top - 1 + y) / 2.0, e))
                    if e != ch[-1]:                             # thin rule between tasks: which rows belong to which task
                        ax.axhline(y - (gap + 1) / 2.0, color=GRID, lw=0.8, zorder=0.5)
                    y -= gap
                ref_line(ax, 1.0, "best", vertical=True)
                ref_line(ax, 0.0, "random", vertical=True)
                ax.set_ylim(-0.6, rows - 0.4)                   # the same row pitch in both panels (rows run from rows - 1 down to >= 0)
                ax.set_yticks([t for t, _ in ticks])
                ax.set_yticklabels([D.task_name(e) for _, e in ticks])
                ax.tick_params(axis="y", length=0)
                ax.grid(False, axis="y")
                ax.grid(True, axis="x")
                ax.set_xlabel("Normalised AUC")
            v = np.asarray(vals, float)
            v = v[np.isfinite(v)]
            span = float(v.max() - v.min()) or 1.0
            axes.flat[0].set_xlim(float(v.min()) - 0.04 * span, float(v.max()) + 0.04 * span)
            _norm_ticks(list(axes.flat), "x", nbins=4)
            if ncol == 1:
                axes.flat[0].set_title(g)
            else:
                fig.suptitle(g, fontsize=fs.LABEL_SIZE)
            marks = [(Line2D([], [], ls="", marker="o", ms=fs.MARKER_SIZE, color=INK2, mec="white", mew=0.4), "Mean over seeds"),
                     (Line2D([], [], color=INK2, lw=fs.ERRORBAR_WIDTH, solid_capstyle="butt"), f"95% {D.cfg.ci_label}")]
            fs.legend(fig, _legend_handles(D, labs, "marker"),
                      marks + _ref_handles({"best", "random"}, {"best": "Best baseline by AUC = 1", "random": "Random policy = 0"}))
            cap = [f"Figure. Learning speed on the {g} tasks: normalised AUC = (AUC - random policy) / (best baseline's mean AUC - random "
                   "policy), AUC = mean return over all evaluations. Under each task one row per algorithm that ran on it, in legend order: "
                   f"dot = mean over seeds, line = 95% {D.cfg.ci_label} over seeds (two or more seeds). Right of 1 = learned faster than "
                   "every baseline on average over the run."]
            hidden = [l for l in D.labels if any(e in sc.get(l, {}) for e in envs) and not _colour_ok(D, l)]
            if hidden:
                cap.append(_no_colour_note("Not drawn", hidden, "section 9 and the AUC heatmap"))
            un = [e for e in envs if D.is_untuned(e)]
            if un:
                cap.append("Baselines run with library defaults (untuned) on: " + ", ".join(_untuned_names(D, un)) + ".")
            if D.cfg.smoke:
                cap.append("SMOKE-sized runs.")
            out += _finish(D, fig, f"auc_normalised_{g.lower().replace(' ', '_')}", caption=" ".join(cap))
    return out


plot_auc_bars = plot_auc_intervals                    # the earlier name (bars), kept for old calls


def plot_sample_efficiency(D: Data, stem: str = "sample_efficiency"):
    """Left: share of the budget at which each algorithm first reaches ``level_plot`` of the way to the reference (median over
    seeds; x = reached by at most half of the seeds). Right: speed-up of TEMPO over the best baseline by final score (bar) and
    over the fastest baseline (diamond), log scale."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    lv = D.cfg.level_plot
    rs = reach_stats(D)
    tasks = [e for e in D.tasks if D.rows(e) and (e, next(iter(D.rows(e))), lv) in rs]
    if not tasks:
        print("pending: no task has a reference score and a run yet.")
        return []
    sp = speedup_frame(D)
    sp = sp[sp["level"] == lv] if len(sp) else sp
    with figure_style() as plt:
        fig = plt.figure(figsize=(fs.DOUBLE, 0.19 * len(tasks) + 1.25), layout="constrained")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.0])
        ax = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharey=ax)
        ys = {e: len(tasks) - 1 - i for i, e in enumerate(tasks)}
        max_never = max([sum(1 for l in D.rows(e) if (s := rs.get((e, l, lv))) and not np.isfinite(s["median"]) and _colour_ok(D, l))
                         for e in tasks] or [0])
        used, never = [], False
        for env in tasks:
            y = ys[env]
            k_never = 0
            for lab in D.rows(env):
                s = rs.get((env, lab, lv))
                if not s or not _colour_ok(D, lab):            # marks told apart by colour only
                    continue
                st = style_of(D, lab)
                is_main = lab == D.main
                z = 4.5 if is_main else 3 + st["z"] / 20
                if np.isfinite(s["median"]):
                    ax.scatter([s["median"] / D.budget(env)], [y], s=(fs.MARKER_SIZE + (1.5 if is_main else 0.5)) ** 2, marker="o",
                               color=st["color"], zorder=z, edgecolor="white", linewidth=0.5)
                else:
                    never = True                    # several algorithms that never get there: crosses side by side
                    ax.scatter([1.04 + 0.022 * k_never], [y], s=(fs.MARKER_SIZE + 0.5) ** 2, marker="x", color=st["color"], zorder=z,
                               linewidth=1.0)
                    k_never += 1
                if lab not in used:
                    used.append(lab)
        ax.set_xlim(0, max(1.09, 1.06 + 0.022 * max_never))
        ax.axvline(1.0, color=GRID, lw=0.8, zorder=0.5)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "25", "50", "75", "100"])
        ax.set_xlabel(f"Budget share to reach {int(lv * 100)}% of the reference (%)")
        ax.set_yticks([ys[e] for e in tasks])
        ax.set_yticklabels(tasks)
        ax.set_ylim(-0.7, len(tasks) - 0.3)
        ax.tick_params(axis="y", length=0)
        ax.set_title("Steps to the level")
        # right panel: speed-up bars (neutral: the direction is the side of 1x)
        ax2.set_xscale("log")
        for env in tasks:
            g = sp[sp["task"] == env] if len(sp) else sp
            if not len(g):
                continue
            s = g["speedup"].iloc[0]
            sf = g["speedup_fastest"].iloc[0]
            y = ys[env]
            if np.isfinite(sf) and sf > 0 and g["fastest_baseline"].iloc[0] != g["best_baseline"].iloc[0]:
                ax2.scatter([min(max(sf, 0.11), 11)], [y], marker="D", s=(fs.MARKER_SIZE) ** 2, color=INK, zorder=6, edgecolor="white",
                            linewidth=0.5)
            if np.isnan(s):
                ax2.text(1.0, y, "neither reaches it", ha="center", va="center", fontsize=_NOTE_SIZE, color=MUTED, style="italic")
            elif np.isinf(s):
                ax2.barh(y, 10 - 1, left=1, height=0.5, color=INK2, alpha=0.35, lw=0)
                ax2.text(1.12, y, f"only {D.main}", va="center", fontsize=_NOTE_SIZE, color=INK)
            elif s == 0:
                ax2.barh(y, 1 - 0.1, left=0.1, height=0.5, color=INK2, alpha=0.35, lw=0)
                ax2.text(0.112, y, "only baseline", va="center", fontsize=_NOTE_SIZE, color=INK)
            else:
                ax2.barh(y, s - 1, left=1, height=0.5, color=INK2, alpha=0.6, lw=0)
                ax2.text(max(s, 1) * 1.08 if s >= 1 else 1.08, y, f"{s:.2f}×", va="center", fontsize=_NOTE_SIZE, color=INK2)
        ax2.axvline(1.0, color=INK2, lw=fs.AXIS_WIDTH)
        ax2.set_xlim(0.1, 12)
        ax2.set_xticks([0.2, 0.5, 1, 2, 5, 10])
        ax2.set_xticklabels(["0.2×", "0.5×", "1×", "2×", "5×", "10×"])
        ax2.minorticks_off()
        ax2.set_xlabel(f"Speed-up (baseline / {D.main} steps)")
        ax2.tick_params(axis="y", length=0, labelleft=False)
        ax2.grid(False, axis="y")
        ax2.set_title("Speed-up")
        keys = ([(Line2D([], [], ls="", marker="x", ms=fs.MARKER_SIZE + 1, color=INK2, mew=1.0), "Not reached by most seeds")] if never else [])
        keys += [(Patch(facecolor=INK2, alpha=0.6, lw=0), "vs best-final baseline"),
                 (Line2D([], [], ls="", marker="D", ms=fs.MARKER_SIZE, color=INK, mec="white", mew=0.5), "vs fastest baseline")]
        fs.legend(fig, _legend_handles(D, [l for l in D.labels if l in used], "marker"), keys)
        cap = ["Figure. Left: median over seeds of the first evaluation whose trailing 3-evaluation average reaches the level, as a share "
               "of the task's budget; the level is the share of the way from the random policy to the "
               + ("solved threshold" if D.cfg.reference == "solved" else "best baseline's mean final score")
               + "; a cross right of 100% = at most half of the seeds reach it. Right: speed-up = baseline steps / "
               f"{D.main} steps (log scale; right of 1× = {D.main} faster), against the baseline with the highest final score (bar) "
               "and against the baseline that gets there fastest (diamond, drawn when it is another baseline); 'TEMPO is faster' is only "
               "established by the diamond. only TEMPO / only baseline: one side never reaches the level."]
        hidden = [l for l in D.tempo_labels if not _colour_ok(D, l) and any(l in D.rows(e) for e in tasks)]
        if hidden:
            cap.append(_no_colour_note("Not drawn on the left", hidden))
        if D.cfg.smoke:
            cap.append("SMOKE-sized runs.")
        return _finish(D, fig, stem, caption=" ".join(cap))


def show_efficiency(D: Data) -> None:
    """Section 6: environment steps to reach 50 / 75 / 90 / 100% of the reference, speed-up, normalised AUC."""
    if not D.has_data:
        print("pending: no results yet.")
        return
    _mark(D, "efficiency")
    plot_sample_efficiency(D)
    sp = speedup_frame(D)
    md = speedup_md(D, sp)
    mdf = speedup_fastest_md(D, sp)
    register(D, "sample_efficiency_speedup", "Steps to reach 50/75/90/100% of the best baseline and speed-up of TEMPO", df=sp, md=md)
    show_md("#### Steps to reach the levels, TEMPO against the best baseline (by final score)\n\n" + md)
    if mdf:
        register(D, "sample_efficiency_speedup_fastest", "Speed-up of TEMPO against the fastest baseline of each task", md=mdf)
        show_md("#### Steps to reach the levels, TEMPO against the FASTEST baseline of each task\n\n" + mdf)
    summ = speedup_summary(D, sp)
    if summ:
        print("Speed-up summary over tasks (geometric mean over the tasks where both reach the level; 'never' counted as the full budget in the censored estimate):")
        for lv, s in summ.items():
            print(f"  {int(lv * 100):3d}% ({_pl(s['tasks'], 'task')}):")
            print("      vs the best-final baseline: " + speedup_line(s))
            print("      vs the fastest baseline   : " + speedup_line(s, "fastest_"))
    D._extra["speedup_summary"] = summ
    for lv in LEVELS:
        register(D, f"steps_to_{int(lv * 100)}pct", f"Median steps to reach {int(lv * 100)}% of the reference, per algorithm", md=reach_table_md(D, lv))
    register(D, "steps_to_level_long", "Steps to reach each level, one row per task, algorithm and level", df=reach_frame(D))
    show_md(f"#### Steps to reach {int(D.cfg.level_plot * 100)}% of the reference, every algorithm\n\n" + reach_table_md(D, D.cfg.level_plot))
    tasks, labs, M = norm_auc_matrix(D)
    if tasks:
        import pandas as pd

        fr = pd.DataFrame(M, index=tasks, columns=labs).round(3)
        register(D, "normalised_auc", "Normalised AUC per task and algorithm (best baseline by AUC = 1)", df=fr.reset_index().rename(columns={"index": "task"}))
        _heatmap_figure(D, tasks, labs, M, "Normalised AUC", "Normalised AUC\n(0 = random policy, 1 = best baseline by AUC)",
                        "heatmap_normalised_auc", "Mean normalised AUC (learning speed) over seeds per task and algorithm: (AUC - random "
                        "policy) / (mean AUC of the best baseline by AUC - random policy), AUC = mean return over all evaluations.")
        scope = [e for e in aggregate_scope(D) if e in tasks]
        sc = norm_auc_scores(D, scope)
        sc = {l: v for l, v in sc.items() if v}
        if scope and sc:
            rows = []
            for l, s in sc.items():
                a = _agg_ci(s, D.cfg.n_boot, ("auc", l))
                rows.append([l, str(len(s)), _ci3(a["IQM"]), _ci3(a["median"]), _ci3(a["mean"])])
            t = R.md_table(["Algorithm", "Tasks", "IQM", "Median", "Mean"], rows)
            register(D, "normalised_auc_aggregate", "Aggregate normalised AUC", md=t)
            show_md(f"#### Aggregate normalised AUC over {len(scope)} tasks (95% stratified-bootstrap CI)\n\n" + t)


# =========================================================================== 7. stability
def stability_frame(D: Data):
    """Per task and algorithm: mean normalised score and across-seed spread of the final score (std, IQR, worst seed; also on the normalised scale),
    share of failing seeds (final below the random policy / below ``fail_level`` of the way to the best baseline),
    collapse (smoothed peak minus final above ``collapse_level`` of that scale) and curve roughness (mean absolute step-to-step
    change of the normalised curve; lower = smoother)."""
    import pandas as pd

    rows = []
    for env in D.tasks:
        nz = D.norm.get(env)
        rnd = D.randoms.get(env)
        scale = (nz["best_baseline"] - nz["random"]) if nz and nz["valid"] else None
        for lab, runs in D.rows(env).items():
            f = np.array([r.final for r in runs])
            r_ = dict(task=env, family=R.family(env), algorithm=lab, n=len(runs), mean=float(f.mean()), mean_norm=math.nan,
                      std=float(f.std(ddof=1)) if f.size > 1 else math.nan,
                      iqr=float(np.percentile(f, 75) - np.percentile(f, 25)) if f.size > 1 else math.nan, worst=float(f.min()), best=float(f.max()),
                      std_norm=math.nan, iqr_norm=math.nan, worst_norm=math.nan, fail_random=math.nan, fail_level=math.nan, collapse=math.nan,
                      roughness=math.nan)
            if rnd is not None:
                r_["fail_random"] = float((f < rnd).mean())
            if scale:
                nf = (f - rnd) / scale
                coll, rough = [], []
                for r in runs:
                    st, y = smoothed(r, D.cfg.ma_window)
                    coll.append((float(y.max()) - r.final) / scale > D.cfg.collapse_level)
                    rough.append(float(np.mean(np.abs(np.diff((r.curve - rnd) / scale)))) if r.curve.size > 1 else math.nan)
                r_.update(mean_norm=float(nf.mean()), std_norm=float(nf.std(ddof=1)) if f.size > 1 else math.nan,
                          iqr_norm=float(np.percentile(nf, 75) - np.percentile(nf, 25)) if f.size > 1 else math.nan,
                          worst_norm=float(nf.min()), fail_level=float((nf < D.cfg.fail_level).mean()), collapse=float(np.mean(coll)),
                          roughness=float(np.nanmean(rough)))
            rows.append(r_)
    return pd.DataFrame(rows)


def stability_summary(D: Data, df=None):
    """Per algorithm, averaged over the aggregate-scope tasks (every task equal weight)."""
    import pandas as pd

    df = stability_frame(D) if df is None else df
    scope = set(aggregate_scope(D))
    d = df[df["task"].isin(scope)] if len(df) else df
    rows = []
    for lab in D.labels:
        g = d[d["algorithm"] == lab] if len(d) else d
        if not len(g):
            continue
        ms, sd = g["mean_norm"].mean(), g["std_norm"].mean()
        rows.append({"algorithm": lab, "tasks": len(g), "runs": int(g["n"].sum()), "mean score (norm.)": ms, "std (norm.)": sd,
                     "relative std": sd / ms if np.isfinite(ms) and ms > 0.05 else math.nan,
                     "IQR (norm.)": g["iqr_norm"].mean(), "worst seed (norm.)": g["worst_norm"].mean(),
                     "below random %": 100 * g["fail_random"].mean(), f"below {int(D.cfg.fail_level * 100)}% of best %": 100 * g["fail_level"].mean(),
                     "collapsed %": 100 * g["collapse"].mean(), "roughness": g["roughness"].mean()})
    return pd.DataFrame(rows)


def plot_stability(D: Data, df=None, stem: str = "stability"):
    """Pooled normalised final scores per run (spread across seeds and tasks) and failure / collapse / roughness per algorithm."""
    from matplotlib.lines import Line2D

    df = stability_frame(D) if df is None else df
    scope = aggregate_scope(D)
    cells = {e: D.cells[e] for e in scope}
    labs = [l for l in D.labels if any(l in cells[e] for e in scope)]
    if not labs:
        print("pending: no task with a baseline result yet, nothing to plot.")
        return []
    summ = stability_summary(D, df).set_index("algorithm")
    with figure_style() as plt:
        fig = plt.figure(figsize=(fs.DOUBLE, 0.22 * len(labs) + 1.35), layout="constrained")
        gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1.0, 1.0, 1.0])
        axes = [fig.add_subplot(gs[0])]
        axes += [fig.add_subplot(gs[k], sharey=axes[0]) for k in (1, 2, 3)]
        pos = {l: len(labs) - 1 - i for i, l in enumerate(labs)}
        rng = np.random.default_rng(0)
        ax = axes[0]
        for l in labs:
            vals = np.concatenate([(np.asarray(cells[e][l].finals) - D.norm[e]["random"]) / (D.norm[e]["best_baseline"] - D.norm[e]["random"])
                                   for e in scope if l in cells[e]])
            st = style_of(D, l)
            ax.scatter(vals, pos[l] + rng.uniform(-0.25, 0.25, vals.size), s=(fs.MARKER_SIZE * 0.8) ** 2, color=st["color"], alpha=0.5,
                       edgecolor="none", zorder=2)
            q = np.percentile(vals, [25, 50, 75])
            ax.plot([q[0], q[2]], [pos[l]] * 2, color=st["color"], lw=_CI_LW, alpha=0.6, solid_capstyle="butt", zorder=3)
            ax.plot([q[1]], [pos[l]], marker="|", ms=_TICK_MS, mew=_TICK_MEW, color=st["color"], zorder=4)
        ref_line(ax, 1.0, "best", vertical=True)
        ref_line(ax, 0.0, "random", vertical=True)
        ax.set_xlabel("Normalised final score per run")
        ax.set_title("Final scores")
        ax.set_yticks([pos[l] for l in labs])
        ax.set_yticklabels(labs)
        for tick, l in zip(ax.get_yticklabels(), labs):
            if l == D.main:
                tick.set_fontweight("bold")
        # failure / collapse
        ax = axes[1]
        fl = int(D.cfg.fail_level * 100)
        marks = [("below random %", "o", "Below the random policy"), (f"below {fl}% of best %", "s", f"Below {fl}% of the best baseline"),
                 ("collapsed %", "^", "Collapsed after peaking")]
        for l in labs:
            st = style_of(D, l)
            for col, mk, _ in marks:
                v = summ.loc[l, col]
                if np.isfinite(v):
                    ax.scatter([v], [pos[l]], marker=mk, s=(fs.MARKER_SIZE + 0.8) ** 2, color=st["color"], edgecolor="white", linewidth=0.5,
                               zorder=3, alpha=0.9)
        ax.set_xlabel("Share of runs (%)")
        ax.set_xlim(-4, 104)
        ax.set_title("Failures and collapse")
        for k, (col, title, xl) in enumerate((("std (norm.)", "Spread over seeds", "Std (normalised)"),
                                              ("roughness", "Curve roughness", "Mean |change|"))):
            ax = axes[2 + k]
            for l in labs:                                  # dot on a thin stem from 0 (no saturated blocks), value beside it
                v = summ.loc[l, col]
                if np.isfinite(v):
                    c = style_of(D, l)["color"]
                    ax.plot([0, v], [pos[l]] * 2, color=c, lw=fs.ERRORBAR_WIDTH, alpha=_BAR_ALPHA, solid_capstyle="butt", zorder=2)
                    ax.plot([v], [pos[l]], ls="", marker="o", ms=fs.MARKER_SIZE + (1.0 if l == D.main else 0.5), color=c, mec="white",
                            mew=0.5, zorder=3)
                    ax.annotate(f"{v:.2f}", (v, pos[l]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=_NOTE_SIZE,
                                color=INK2)
            ax.set_xlabel(xl)
            ax.set_title(title)
            ax.set_xlim(left=0, right=ax.get_xlim()[1] * 1.3)
        for ax in axes:
            ax.tick_params(axis="y", length=0)
            ax.grid(False, axis="y")
            ax.grid(True, axis="x")
        for ax in axes[1:]:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_ylim(-0.6, len(labs) - 0.4)
        _norm_ticks([axes[0]], "x", nbins=4)
        fs.legend(fig, [(Line2D([], [], ls="", marker=mk, ms=fs.MARKER_SIZE + 0.8, color=INK2, mec="white", mew=0.5), lab) for _, mk, lab in marks],
                  _ref_handles({"best", "random"}, {"best": "Best baseline = 1", "random": "Random policy = 0"}))
        ms = [f"{l} {summ.loc[l, 'mean score (norm.)']:.2f}" for l in labs if np.isfinite(summ.loc[l, "mean score (norm.)"])]
        cap = [f"Figure. Stability over the {_pl(len(scope), 'aggregate-scope task')} (every task equal weight; normalised: 0 = random policy, "
               "1 = best baseline). Left: every run (dot), interquartile range (bar) and median (vertical bar) of the normalised final "
               f"score. Second: share of runs below the random policy, below {fl}% of the best baseline and collapsing (smoothed peak - final > "
               f"{int(D.cfg.collapse_level * 100)}% of best baseline - random). Third: standard deviation over seeds (normalised); right: "
               "curve roughness, the mean absolute change of the normalised curve between evaluations (lower = smoother); both as a dot "
               "on a thin line from 0, the value beside it.",
               "A low spread or roughness at a low mean score is a floor effect (a uniformly poor algorithm is not steady): read them with "
               "the mean normalised scores" + (": " + ", ".join(ms) if ms else "") + "."]
        if D.cfg.smoke:
            cap.append("SMOKE-sized runs.")
        return _finish(D, fig, stem, caption=" ".join(cap))


def show_stability(D: Data) -> None:
    """Section 7: across-seed spread, failures, collapse, curve smoothness."""
    if not D.has_data:
        print("pending: no results yet.")
        return
    _mark(D, "stability")
    df = stability_frame(D)
    summ = stability_summary(D, df)
    register(D, "stability_per_task", "Stability per task and algorithm", df=df.round(4))
    register(D, "stability_summary", "Stability per algorithm (task-balanced mean over the aggregate scope)", df=summ.round(4))
    plot_stability(D, df)
    if len(summ):
        show_md("#### Stability per algorithm (mean over the tasks in the aggregate scope; normalised scale: 0 = random, 1 = best baseline)\n\n"
                + df_to_md(summ.round(3))
                + "\nrelative std = std / mean score (scale-free; blank when the mean score is below 0.05). A small std at a low mean score is a floor "
                  "effect, not steadiness: compare spreads only between algorithms with comparable mean scores.\n")
    d = df[df["algorithm"] == D.main] if len(df) else df
    if len(d):
        cols = ["task", "n", "mean", "mean_norm", "std", "iqr", "worst", "std_norm", "worst_norm", "fail_random", "fail_level", "collapse", "roughness"]
        show_md(f"#### {D.main}, per task\n\n" + df_to_md(d[cols].round(3)))
    flagged = df[(df["fail_level"] > 0) | (df["collapse"] > 0)] if len(df) else df
    if len(flagged):
        print("Runs below the failure level or collapsing, per (task, algorithm): see tables/stability_per_task.csv "
              f"({len(flagged)} of {len(df)} rows have at least one failing or collapsing seed).")


# =========================================================================== 8. compute cost
def compute_frame(D: Data):
    """One row per run: device, wall-clock hours, environment steps per second, normalised final score and the hours the run needed to
    reach ``level_plot`` (default 90%) of the way to the reference (steps to the level / budget x wall-clock, i.e. constant time per step)."""
    import pandas as pd

    rs = reach_stats(D)
    lv = D.cfg.level_plot if D.cfg.level_plot in LEVELS else 0.9
    rows = []
    for env in D.tasks:
        nz = D.norm.get(env)
        for lab, runs in D.rows(env).items():
            st = rs.get((env, lab, lv))
            for i, r in enumerate(runs):
                sc = ((r.final - nz["random"]) / (nz["best_baseline"] - nz["random"])) if nz and nz["valid"] else math.nan
                h = r.wall_h
                hit = st["steps"][i] if st else math.inf
                ok = bool(st) and np.isfinite(hit) and h and np.isfinite(h)
                rows.append(dict(task=env, family=R.family(env), algorithm=lab, seed=r.seed, device=r.device, wall_h=h, budget=r.budget,
                                 steps_per_s=r.budget / (h * 3600) if h and np.isfinite(h) and h > 0 else math.nan, norm_final=sc,
                                 reference_known=bool(st), reached_level=bool(st) and bool(np.isfinite(hit)),
                                 hours_to_level=float(hit / r.budget * h) if ok else math.nan))
    return pd.DataFrame(rows, columns=["task", "family", "algorithm", "seed", "device", "wall_h", "budget", "steps_per_s", "norm_final",
                                       "reference_known", "reached_level", "hours_to_level"])


def _task_hours(g) -> float:
    """Median hours to the level over the seeds of one (task, algorithm): inf unless more than half of the seeds reach it."""
    arr = np.where(g["reached_level"].to_numpy(bool) & np.isfinite(g["hours_to_level"].to_numpy(float)), g["hours_to_level"].to_numpy(float), math.inf)
    return _median_steps(arr)


def compute_summary(D: Data, df=None):
    """Per algorithm: runs, devices, hours per run, throughput, the normalised score over the aggregate-scope tasks (with their number) and the
    wall-clock hours to reach the level (geometric mean over the tasks where more than half of the seeds reach it)."""
    import pandas as pd

    df = compute_frame(D) if df is None else df
    scope = set(aggregate_scope(D))
    lv = D.cfg.level_plot if D.cfg.level_plot in LEVELS else 0.9
    tm_tasks = set(df[df["algorithm"] == D.main]["task"]) if len(df) else set()
    rows = []
    for lab in D.labels:
        g = df[df["algorithm"] == lab] if len(df) else df
        if not len(g):
            continue
        per_task = g.groupby("task").agg(h=("wall_h", "mean"), sc=("norm_final", "mean"))
        sc_scope = per_task[per_task.index.isin(scope)]
        ref = g[g["reference_known"]]
        if tm_tasks:
            ref = ref[ref["task"].isin(tm_tasks)]                  # the same tasks for every algorithm
        th = [_task_hours(x) for _, x in ref.groupby("task")]
        fin = [x for x in th if np.isfinite(x) and x > 0]
        rows.append({"algorithm": lab, "runs": len(g), "tasks": g["task"].nunique(), "device": "+".join(sorted(g["device"].unique())),
                     "mean h/run": g["wall_h"].mean(), "total h": g["wall_h"].sum(), "median steps/s": g["steps_per_s"].median(),
                     "norm. score": sc_scope["sc"].mean() if len(sc_scope) else math.nan, "score tasks": len(sc_scope),
                     f"hours to {int(lv * 100)}% (geo. mean)": _geo(fin) if fin else math.nan,
                     f"reaches {int(lv * 100)}% on": f"{len(fin)} of {len(th)} {D.main if tm_tasks else ''} tasks".replace("  ", " ") if th else "-"})
    return pd.DataFrame(rows)


def hours_table_md(D: Data, df=None) -> str:
    """Per task x algorithm: median wall-clock hours to reach the level (k/n seeds that do)."""
    df = compute_frame(D) if df is None else df
    if not len(df) or not df["reference_known"].any():
        return "No task has a reference score and a run yet.\n"
    head = ["Task"] + list(D.labels)
    rows = []
    for env in D.tasks:
        g = df[(df["task"] == env) & df["reference_known"]]
        if not len(g):
            continue
        r = [D.task_name(env)]
        for l in D.labels:
            x = g[g["algorithm"] == l]
            if not len(x):
                r.append("-")
                continue
            m = _task_hours(x)
            r.append(("never" if not np.isfinite(m) else f"{m:.2f} h" if m >= 0.01 else "<0.1 min" if m * 60 < 0.1 else f"{m * 60:.1f} min")
                     + f" ({int(x['reached_level'].sum())}/{len(x)})")
        rows.append(r)
    lv = int((D.cfg.level_plot if D.cfg.level_plot in LEVELS else 0.9) * 100)
    note = (f"Median wall-clock hours to reach {lv}% of the way to the reference = steps to the level / budget x wall-clock of the run (assumes a constant time "
            "per step; the wall-clock includes the evaluation episodes); (k/n) = seeds that reach it; never = at most half do. Devices differ between algorithms.")
    return R.md_table(head, rows, align="l" + "r" * len(D.labels)) + f"\n{note}\n"


def wallclock_speedup(D: Data, df=None) -> dict:
    """Hours to the level of the best-final baseline and of the baseline that needs the fewest hours, against the headline TEMPO:
    geometric mean of baseline hours / TEMPO hours over the tasks where both reach it (> 1: TEMPO gets there in less wall-clock time)."""
    df = compute_frame(D) if df is None else df
    out = dict(tasks=0, both=0, geo_best=math.nan, geo_fastest=math.nan, tempo_faster_best=0, tempo_faster_fastest=0)
    if not len(df) or not df["reference_known"].any():
        return out
    rb, rf = [], []
    for env in D.tasks:
        g = df[(df["task"] == env) & df["reference_known"]]
        b, _ = best_baseline(D, env, "final")
        if b is None or not (g["algorithm"] == D.main).any():
            continue
        out["tasks"] += 1
        t = _task_hours(g[g["algorithm"] == D.main])
        hrs = {l: _task_hours(g[g["algorithm"] == l]) for l in g["algorithm"].unique() if l in BASELINES}
        fin = {l: h for l, h in hrs.items() if np.isfinite(h) and h > 0}
        if not np.isfinite(t) or t <= 0:
            continue
        if b in fin:
            rb.append(fin[b] / t)
        if fin:
            rf.append(min(fin.values()) / t)
    out.update(both=len(rb), geo_best=_geo(rb) if rb else math.nan, geo_fastest=_geo(rf) if rf else math.nan,
               tempo_faster_best=int(sum(x > 1 for x in rb)), tempo_faster_fastest=int(sum(x > 1 for x in rf)), both_fastest=len(rf))
    return out


def device_text(df) -> str:
    """'TEMPO, CrossQ: Apple GPU (MPS); SAC, TD3: CPU' from the devices recorded in the runs."""
    names = {"cpu": "CPU", "mps": "Apple GPU (MPS)", "cuda": "CUDA GPU"}
    by = defaultdict(list)
    for lab, g in df.groupby("algorithm", sort=False):
        by["/".join(names.get(d, d) for d in sorted(g["device"].unique()))].append(lab)
    return "; ".join(f"{', '.join(v)}: {k}" for k, v in by.items())


def compute_by_family(D: Data, df=None):
    """Mean wall-clock hours per run by algorithm and task family (run-weighted within a task, every task equal weight)."""
    import pandas as pd

    df = compute_frame(D) if df is None else df
    if not len(df):
        return pd.DataFrame()
    per_task = df.groupby(["algorithm", "family", "task"])["wall_h"].mean().reset_index()
    tab = per_task.groupby(["algorithm", "family"])["wall_h"].mean().unstack("family")
    tab = tab.reindex([l for l in D.labels if l in tab.index])
    tab = tab[[f for f in FAMILY_NAMES if f in tab.columns]].rename(columns=FAMILY_NAMES)
    return tab.round(3).reset_index()


def cost_vs_frame(D: Data, other: str = "CrossQ"):
    """Per task: wall-clock of the headline TEMPO against ``other`` (default CrossQ, the closest baseline in cost), ratio TEMPO / other."""
    import pandas as pd

    rows = []
    for env in D.tasks:
        row = D.rows(env)
        if D.main in row and other in row:
            t, o = row[D.main], row[other]
            ht, ho = float(np.nanmean([r.wall_h for r in t])), float(np.nanmean([r.wall_h for r in o]))
            rows.append(dict(task=env, tempo_h=ht, other_h=ho, ratio=ht / ho if ho > 0 else math.nan,
                             tempo_device="+".join(sorted({r.device for r in t})), other_device="+".join(sorted({r.device for r in o})),
                             same_device=sorted({r.device for r in t}) == sorted({r.device for r in o})))
    return pd.DataFrame(rows, columns=["task", "tempo_h", "other_h", "ratio", "tempo_device", "other_device", "same_device"])


_DEVICE_MARKERS = {"cpu": ("o", "CPU"), "mps": ("s", "Apple GPU (MPS)"), "cuda": ("D", "CUDA GPU")}


def plot_compute(D: Data, df=None, stem: str = "compute_cost"):
    """Wall-clock hours per run and environment steps per second per algorithm (per-task means; log axis), and the
    normalised final score against hours per run for every (task, algorithm)."""
    from matplotlib.lines import Line2D

    df = compute_frame(D) if df is None else df
    labs = [l for l in D.labels if len(df) and (df["algorithm"] == l).any()]
    if not labs or df["wall_h"].notna().sum() == 0:
        print("pending: no run with a recorded wall-clock yet.")
        return []
    pos = {l: len(labs) - 1 - i for i, l in enumerate(labs)}
    rng = np.random.default_rng(1)
    devs_seen = set()
    with figure_style() as plt:
        fig = plt.figure(figsize=(fs.DOUBLE, max(2.3, 0.22 * len(labs) + 1.35)), layout="constrained")
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
        axes = [fig.add_subplot(gs[0])]
        axes += [fig.add_subplot(gs[1], sharey=axes[0]), fig.add_subplot(gs[2])]
        for k, (col, xl, ttl) in enumerate((("wall_h", "Wall-clock hours per run", "Cost per run"),
                                            ("steps_per_s", "Environment steps per second", "Throughput"))):
            ax = axes[k]
            for l in labs:
                st = style_of(D, l)
                g = df[df["algorithm"] == l].groupby(["task", "device"])[col].mean().reset_index()
                g = g[np.isfinite(g[col]) & (g[col] > 0)]
                if not len(g):
                    continue
                for dev, gg in g.groupby("device"):
                    mk = _DEVICE_MARKERS.get(dev, ("^", dev))[0]
                    devs_seen.add(dev)
                    ax.scatter(gg[col], pos[l] + rng.uniform(-0.22, 0.22, len(gg)), s=fs.MARKER_SIZE ** 2, marker=mk, color=st["color"],
                               alpha=0.75, edgecolor="white", linewidth=0.4, zorder=3)
                gm = float(np.exp(np.log(g[col]).mean()))
                ax.plot([gm], [pos[l]], marker="|", ms=_TICK_MS + 2, mew=_TICK_MEW, color=st["color"], zorder=4)
            ax.set_xscale("log")
            ax.set_yticks([pos[l] for l in labs])
            ax.set_yticklabels(labs)
            ax.tick_params(axis="y", length=0, labelleft=(k == 0))
            ax.grid(False, axis="y")
            ax.grid(True, axis="x")
            ax.set_xlabel(xl)
            ax.set_title(ttl)
        for tick, l in zip(axes[0].get_yticklabels(), labs):
            if l == D.main:
                tick.set_fontweight("bold")
        axes[0].set_ylim(-0.6, len(labs) - 0.4)
        ax = axes[2]
        g = df.groupby(["task", "algorithm"]).agg(h=("wall_h", "mean"), sc=("norm_final", "mean")).reset_index()
        g = g[np.isfinite(g["h"]) & (g["h"] > 0) & np.isfinite(g["sc"])]
        if len(g):
            for l in [l for l in labs if _colour_ok(D, l)]:      # one dot per task, told apart by colour only
                gg = g[g["algorithm"] == l]
                st = style_of(D, l)
                ax.scatter(gg["h"], gg["sc"], s=(fs.MARKER_SIZE + (1.5 if l == D.main else 0.5)) ** 2, color=st["color"], alpha=0.85,
                           edgecolor="white", linewidth=0.5, zorder=4.5 if l == D.main else 3)
            ax.set_xscale("log")
            ref_line(ax, 1.0, "best")
            ax.set_ylabel(fs.NORMALISED_LABEL)
            _norm_ticks([ax], "y", nbins=4)
            ax.set_xlabel("Wall-clock hours per run")
        else:
            ax.text(0.5, 0.5, "pending: needs a baseline result\nto normalise the score", ha="center", va="center", transform=ax.transAxes,
                    color=MUTED, style="italic")
            ax.set_xticks([])
            ax.set_yticks([])
        ax.set_title("Score against cost")
        dev_keys = [(Line2D([], [], ls="", marker=_DEVICE_MARKERS.get(d, ("^", d))[0], ms=fs.MARKER_SIZE + 0.5, color=INK2, mec="white",
                            mew=0.4), _DEVICE_MARKERS.get(d, ("^", d))[1]) for d in sorted(devs_seen)]
        geo = [(Line2D([], [], ls="", marker="|", ms=_TICK_MS, mew=_TICK_MEW, color=INK2), "Geometric mean over tasks")]
        fs.legend(fig, _legend_handles(D, labs, "marker"), dev_keys + geo + (_ref_handles({"best"}, {"best": "Best baseline = 1"}) if len(g) else []))
        devs = device_text(df)
        cap = ["Figure. Compute cost. Left: wall-clock hours per run (mean per task, one dot per task and device; vertical bar = geometric "
               "mean over tasks); middle: environment steps per second; right: normalised final score (0 = random policy, 1 = best "
               "baseline) against hours per run, one dot per task and algorithm. Wall-clock includes the evaluation episodes and contention "
               "from other jobs. Devices recorded in these runs: " + devs
               + ("; they differ between algorithms, so costs compare only roughly." if ";" in devs else ".")]
        hidden = [l for l in labs if not _colour_ok(D, l)]
        if hidden and len(g):
            cap.append(_no_colour_note("Not drawn on the right", hidden))
        if D.cfg.smoke:
            cap.append("SMOKE-sized runs: times are not meaningful.")
        return _finish(D, fig, stem, caption=" ".join(cap))


def plot_wallclock_curves(D: Data, ncols: int = 4, band=None, stem: str = "curves_vs_wallclock"):
    """Learning-curve grid with wall-clock time on the x axis (bold mean and shaded spread of the seeds, as the step-based grid): each
    algorithm's curve is stretched to its mean hours per run on the task (x = share of the budget x mean hours per run, i.e. constant
    time per step); every panel in seconds, minutes or hours, the unit on its last tick. The headline TEMPO only (other variants:
    section 9). An algorithm that wins per step can lose per hour."""
    only = _headline_only(D)
    tasks = [e for e in D.tasks if only(e) and any(np.isfinite(r.wall_h) for l in only(e) for r in D.rows(e)[l])]
    if not tasks:
        print("pending: no run with a recorded wall-clock yet.")
        return []
    band = _band_choice(D, band)
    with figure_style():
        fig, pairs, kinds, cut = _curve_figure(D, tasks, lambda e: _task_title(D, e), False, band, x="hours", max_cols=ncols, only_fn=only)
        cap = [f"Figure. Evaluation return against wall-clock time on the {_pl(len(tasks), 'task')} with a recorded wall-clock: x = share "
               "of the budget x the algorithm's mean wall-clock per run on the task (constant time per step assumed; times include the "
               "evaluation episodes and machine load); each panel in seconds, minutes or hours, the unit written after its last tick "
               "label.", _band_sentence(D, pairs, band, False, cut), _ref_sentence(kinds),
               "Devices: " + device_text(compute_frame(D)) + "."]
        cap += _common_notes(D, tasks)
        if len(D.tempo_labels) > 1:
            cap.append(_VARIANT_NOTE)
        if D.cfg.smoke:
            cap.append("Times of SMOKE-sized runs are not meaningful.")
        cap.append(_band_switch(D, band))
        return _finish(D, fig, stem, caption=" ".join(c for c in cap if c))


def _compute_tables(D: Data, cells: dict, labels: list) -> tuple:
    """``report.compute_tables`` (Markdown, LaTeX) with its Budget column taken from the runs: report.py prints the
    earlier benchmark's budgets of ``benchmarks/plans.py``, which are not the budgets of the test folders."""
    md, tex = R.compute_tables(cells, labels)
    for env in cells:
        old, new = (R.fmt_steps(TASK_INFO[env][0]) if env in TASK_INFO else ""), D.budget_text(env)
        md = md.replace(f"| {env} | {old} |", f"| {env} | {new} |", 1)
        tex = tex.replace(f"{R._tex_escape(env)} & {old} &", f"{R._tex_escape(env)} & {new} &", 1)
    return md, tex


def show_compute(D: Data) -> None:
    """Section 8: wall-clock, throughput, hours to a level, TEMPO vs CrossQ cost."""
    if not D.has_data:
        print("pending: no results yet.")
        return
    _mark(D, "compute")
    df = compute_frame(D)
    summ = compute_summary(D, df)
    register(D, "compute_per_run", "Wall-clock, throughput and hours to the level of every run", df=df.round(4))
    register(D, "compute_summary", "Compute cost per algorithm", df=summ.round(4))
    plot_compute(D, df)
    if len(summ):
        show_md("#### Compute cost per algorithm\n\n" + df_to_md(summ.round(3))
                + "\nnorm. score = mean normalised final score over the `score tasks` tasks of the aggregate scope that the algorithm ran on (so a row with fewer "
                  "tasks than another averages other tasks); hours to the level = wall-clock hours until the run first reaches the level (steps to the level / budget x "
                  f"wall-clock), geometric mean over the tasks on which more than half of the seeds reach it, taken over the tasks where {D.main} has runs for every algorithm "
                  "(a task where the algorithm never gets there is not in the mean: read it with the 'reaches' column).\n")
    hm = hours_table_md(D, df)
    register(D, "hours_to_level", "Median wall-clock hours to reach the level, per task and algorithm", md=hm)
    show_md(f"#### Wall-clock hours to reach {int((D.cfg.level_plot if D.cfg.level_plot in LEVELS else 0.9) * 100)}% of the reference\n\n" + hm)
    wc = wallclock_speedup(D, df)
    D._extra["wallclock"] = wc
    if wc["both"]:
        print(f"Wall-clock to the level, {D.main} against the best-final baseline: geometric-mean ratio baseline hours / {D.main} hours = {wc['geo_best']:.2f} "
              f"over {_pl(wc['both'], 'task')} where both reach it ({D.main} needs fewer hours on {wc['tempo_faster_best']}); against the baseline needing the fewest hours: "
              f"{wc['geo_fastest']:.2f} over {_pl(wc['both_fastest'], 'task')} ({D.main} fewer hours on {wc['tempo_faster_fastest']}). > 1 = {D.main} is cheaper to the same level.")
    plot_wallclock_curves(D)
    fam = compute_by_family(D, df)
    if len(fam):
        register(D, "compute_by_family", "Mean wall-clock hours per run by algorithm and task family", df=fam)
        show_md("#### Mean wall-clock hours per run by task family\n\n" + df_to_md(fam))
    lab = [l for l in D.labels if (df["algorithm"] == l).any()] if len(df) else []
    cells = {e: D.cells[e] for e in D.cells}
    if cells and lab:
        md, tex = _compute_tables(D, cells, lab)
        register(D, "compute_hours_per_task", "Mean wall-clock hours per run, per task", md=md, tex=tex)
        show_md("#### Mean wall-clock hours per run and task (device in parentheses when not CPU)\n\n" + md)
    cq = cost_vs_frame(D, "CrossQ")
    register(D, "cost_tempo_vs_crossq", "Wall-clock of TEMPO against CrossQ", df=cq.round(4))
    if len(cq):
        gm = float(np.exp(np.log(cq["ratio"].dropna()).mean())) if cq["ratio"].notna().any() else math.nan
        same = cq[cq["same_device"]]
        show_md(f"#### {D.main} against CrossQ (wall-clock hours per run)\n\n" + df_to_md(cq.round(3)))
        print(f"Geometric-mean wall-clock ratio {D.main} / CrossQ over {len(cq)} tasks: {gm:.2f}"
              + (f"; on the {len(same)} tasks where both ran on the same device: "
                 f"{float(np.exp(np.log(same['ratio']).mean())):.2f}" if len(same) else "; no task where both used the same device"))
        D._extra["cost_ratio"] = gm
    else:
        print(f"{D.main} against CrossQ: pending (needs finished runs of both on the same task).")
    print("Caveat: wall-clock depends on the device (CPU vs GPU), on contention from other jobs on the machine and on the library versions; the runs of "
          "Baselines_test and TEMPO_test were made at different times. Read ratios as orders of magnitude, not as benchmarks of the algorithms.")


# =========================================================================== 9. TEMPO variants
def variant_frame(D: Data):
    """Each non-headline TEMPO variant against the headline one, task by task (final return, AUC; Welch, alpha 0.05)."""
    import pandas as pd

    rows = []
    for v in [l for l in D.tempo_labels if l != D.main]:
        for env in D.tasks:
            row = D.cells.get(env, {})
            if D.main in row and v in row:
                a, b = row[v], row[D.main]
                res, p = outcome(D, a.finals, b.finals)
                resa, pa = outcome(D, a.aucs, b.aucs)
                nz = D.norm.get(env, {})
                rows.append(dict(variant=v, task=env, variant_n=a.n, main_n=b.n, variant_final=a.final[0], main_final=b.final[0],
                                 diff=a.final[0] - b.final[0],
                                 diff_normalised=(a.final[0] - b.final[0]) / (nz["best_baseline"] - nz["random"]) if nz.get("valid") else math.nan,
                                 welch_p=p, result=res.upper(), auc_diff=a.auc[0] - b.auc[0], auc_welch_p=pa, auc_result=resa.upper()))
    return pd.DataFrame(rows, columns=["variant", "task", "variant_n", "main_n", "variant_final", "main_final", "diff", "diff_normalised",
                                       "welch_p", "result", "auc_diff", "auc_welch_p", "auc_result"])


def variant_summary(D: Data, df=None):
    import pandas as pd

    df = variant_frame(D) if df is None else df
    rows = []
    for v, g in (df.groupby("variant", sort=False) if len(df) else []):
        tasks = list(g["task"])
        ok = [e for e in tasks if D.norm.get(e, {}).get("valid")]
        r = {"variant": v, "tasks in common": len(tasks), "better (Welch)": int((g["result"] == "WIN").sum()), "tie": int((g["result"] == "TIE").sum()),
             "worse (Welch)": int((g["result"] == "LOSS").sum()), "untested": int((g["result"] == "N/A").sum())}
        if ok:
            sv = R.normalised_scores({e: D.cells[e] for e in ok}, D.norm, v)
            sm = R.normalised_scores({e: D.cells[e] for e in ok}, D.norm, D.main)
            common = [e for e in ok if e in sv and e in sm]
            if common:
                av = R.aggregate_ci({e: sv[e] for e in common}, D.cfg.n_boot, (v, "variant"))
                am = R.aggregate_ci({e: sm[e] for e in common}, D.cfg.n_boot, (D.main, "variant", v))
                r.update({"IQM variant": _ci3(av["IQM"]), f"IQM {D.main}": _ci3(am["IQM"]), "mean variant": f"{av['mean'][0]:.2f}", f"mean {D.main}": f"{am['mean'][0]:.2f}"})
        rows.append(r)
    return pd.DataFrame(rows)


def plot_variants(D: Data, band=None, stem: str = "tempo_variants"):
    """Learning curves of every TEMPO configuration and of the task's best baseline (bold mean, shaded spread of the seeds) on the
    tasks that have a non-headline variant."""
    tasks = [e for e in D.tasks if D.main in D.rows(e) and any(v in D.rows(e) for v in D.tempo_labels if v != D.main)]
    if not tasks:
        print("No task has runs of the headline TEMPO and of another variant: no curves to compare.")
        return []
    band = _band_choice(D, band)

    def only(env):
        b, _ = best_baseline(D, env, "final")
        return [l for l in D.tempo_labels if l in D.rows(env)] + ([b] if b else [])

    with figure_style():
        fig, pairs, kinds, cut = _curve_figure(D, tasks, lambda e: _task_title(D, e), False, band, only_fn=only, max_cols=3)
        best = [f"{e}: {best_baseline(D, e, 'final')[0]}" for e in tasks if best_baseline(D, e, "final")[0]]
        cap = [f"Figure. Evaluation return of every TEMPO configuration ({D.main} = headline, bold red; every other variant in its own "
               "colour and dash pattern of figstyle.VARIANT_STYLES, the same as in the TEMPO_test figures, named in the legend) and of "
               "the task's best baseline (highest mean final score)" + (" (" + "; ".join(best) + ")" if best else "") + ".",
               _band_sentence(D, pairs, band, False, cut), _ref_sentence(kinds)]
        cap += _common_notes(D, tasks)
        cap.append(_band_switch(D, band))
        return _finish(D, fig, stem, caption=" ".join(c for c in cap if c))


def show_variants(D: Data) -> None:
    """Section 9: every other TEMPO variant against the headline one."""
    _mark(D, "variants")
    others = [l for l in D.tempo_labels if l != D.main]
    if not others:
        print(f"Only one TEMPO configuration has runs ({D.main}): nothing to compare. Run a named variant (benchmarks/variants.py) in TEMPO_test "
              "and run this notebook again.")
        return
    df = variant_frame(D)
    summ = variant_summary(D, df)
    register(D, "tempo_variants", f"TEMPO variants against {D.main}, per task", df=df.round(4))
    register(D, "tempo_variants_summary", f"TEMPO variants against {D.main}, summary", df=summ)
    print(f"{len(others)} further TEMPO configuration(s): {', '.join(others)}; compared with {D.main} on the tasks both ran on.")
    plot_variants(D)
    if len(summ):
        show_md("#### Variants against the headline configuration (normalised IQM on the common tasks)\n\n" + df_to_md(summ))
    if len(df):
        show_md("#### Per task (diff = variant - headline; Welch alpha 0.05)\n\n" + df_to_md(df.round(3)))
    else:
        print("No task has runs of both the headline TEMPO and a variant yet.")


# =========================================================================== 10. exports and executive summary
def runs_table(D: Data):
    """Per-run score table (every run used in the comparison): scores, protocol, device, dates and code versions."""
    import pandas as pd

    rows = []
    for env in D.tasks:
        nz = D.norm.get(env)
        for lab, runs in D.rows(env).items():
            for r in runs:
                sc = ((r.final - nz["random"]) / (nz["best_baseline"] - nz["random"])) if nz and nz["valid"] else math.nan
                rows.append(dict(task=env, family=R.family(env), algorithm=r.algo, variant=r.variant, label=lab, seed=r.seed,
                                 origin=D.origin.get((env, lab), ""), final=r.final, auc=r.auc, peak_eval=float(r.curve.max()),
                                 norm_final=sc, budget=r.budget, eval_freq=r.eval_freq, n_eval=r.n_eval, evaluations=len(r.steps),
                                 dropped_evaluations=r.n_dropped, device=r.device, wall_h=r.wall_h, finished=r.finished,
                                 tempo_source=r.tempo_source, sb3_version=r.info.get("sb3_version", ""), file=_short(r.path)))
    return pd.DataFrame(rows)


def _json_clean(o):
    if isinstance(o, dict):
        return {str(k): _json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o if o is None or isinstance(o, (bool, int, float, str)) else str(o)


def headline(D: Data) -> dict:
    """The headline numbers (also what the executive summary prints)."""
    cp = D._extra.get("completeness") or completeness(D)
    vs = verdicts(D)
    yes, no = [v for v in vs if v["status"] == "YES"], [v for v in vs if v["status"] == "NO"]
    out = dict(main=D.main, completeness=dict(main_found=cp["main"][0], main_expected=cp["main"][1], baselines_found=cp["baselines"][0],
                                              baselines_expected=cp["baselines"][1], tasks_main_any=len(cp["tasks_main_any"]),
                                              tasks_main_complete=len(cp["tasks_main_complete"]), tasks_complete=len(cp["tasks_complete"]),
                                              tasks=len(D.tasks)),
               verdicts=dict(Counter(v["status"] for v in vs)), verdict_tasks={s: [v["task"] for v in vs if v["status"] == s] for s in ("YES", "NO")},
               verdict_tags={v["task"]: verdict_tag(v) for v in vs if v["status"] in ("YES", "NO")},
               verdict_detail=dict(yes_significant=sum(v.get("significant_all") is True for v in yes), yes_provisional=sum(bool(v["provisional"]) for v in yes),
                                   no_significantly_worse=sum(bool(v.get("losses")) for v in no), no_provisional=sum(bool(v["provisional"]) for v in no)))
    w = wtl_frame(D)
    if len(w):
        g = w[(w["scope"] == "All tasks") & (w["opponent"] == "Best baseline (per task)")]
        out["wtl_vs_best"] = {r.metric: dict(win=int(r.win), tie=int(r.tie), loss=int(r.loss), untested=int(r.untested), tasks=int(r.tasks)) for r in g.itertuples()}
    res = D._extra.get("aggregates")
    if res is None and aggregate_scope(D):
        res = {n: a for n, t in agg_scopes(D) for a in [aggregates(D, t, n)] if a}
        D._extra["aggregates"] = res
    if res:
        A = next(iter(res.values()))
        out["aggregate"] = dict(scope_tasks=len(A["tasks"]), main_tasks=len(A["agg"][D.main]["tasks"]) if D.main in A["agg"] else 0,
                                main=A["agg"].get(D.main), best=A["agg"].get(BEST), poi_vs_best=A["poi"].get(BEST), poi_null=A.get("poi_null", {}).get(BEST))
        out["aggregate_like_for_like"] = {n: dict(tasks=len(a["tasks"]), main=a["agg"].get(D.main), best=a["agg"].get(BEST))
                                          for n, a in res.items() if n in ACTION_SCOPES or n == TUNED_SCOPE}
    rt = D._extra.get("rank_tests") or rank_tests(D)
    D._extra["rank_tests"] = rt
    if len(rt["summary"]):
        out["rank"] = {}
        for name in dict.fromkeys(rt["summary"]["scope"]):
            g = rt["summary"][rt["summary"]["scope"] == name]
            m = g[g["algorithm"] == D.main]
            ww = rt["wilcoxon"][rt["wilcoxon"]["scope"] == name]
            if len(m):
                out["rank"][name] = dict(tasks=int(m["tasks"].iloc[0]), algorithms=int(m["algorithms"].iloc[0]), main_mean_rank=float(m["mean_rank"].iloc[0]),
                                         friedman_p=float(m["friedman_p"].iloc[0]),
                                         wilcoxon_p_holm={r.opponent: r.wilcoxon_p_holm for r in ww.itertuples()})
    out["speedup"] = speedup_summary(D)
    st = stability_summary(D)
    if len(st):
        out["stability"] = {r["algorithm"]: {k: (None if not np.isfinite(v) else float(v)) for k, v in r.items() if k != "algorithm" and isinstance(v, (int, float, np.floating, np.integer))}
                            for r in st.to_dict("records")}
    cs = compute_summary(D)
    if len(cs):
        out["compute"] = {r["algorithm"]: dict(mean_h_per_run=r["mean h/run"], median_steps_per_s=r["median steps/s"]) for r in cs.to_dict("records")}
    out["wallclock_to_level"] = D._extra.get("wallclock") or wallclock_speedup(D)
    cq = cost_vs_frame(D)
    if len(cq) and cq["ratio"].notna().any():
        out["cost_ratio_tempo_over_crossq"] = float(np.exp(np.log(cq["ratio"].dropna()).mean()))
    return out


def _warn_kind(msg: str) -> str | None:
    """Short label of a protocol WARN, None for the kinds that the caveats report separately."""
    for key, kind in (("older tempo_rl code", None), ("left out of every normalised", None), ("headline", None),
                      ("differs from the expected protocol", "budget / evaluation schedule differs from the expected protocol"),
                      ("episodes per evaluation", "episodes per evaluation differ from the expected protocol"),
                      ("NO seed", "algorithms share no seed"), ("not among the benchmark's seeds", "seeds outside the benchmark's"),
                      ("ran past the budget", "runs past the budget (the extra evaluations are dropped)"),
                      ("different tempo_rl sources", None), ("mixed SB3", "mixed SB3 versions")):
        if key in msg:
            return kind
    return "other"


def old_code_runs(D: Data) -> dict:
    """{label: n runs} of TEMPO runs made with another tempo_rl source than the live library (empty if the live hash is unknown)."""
    live = live_source_hash()
    out = Counter()
    if live:
        for e in D.tasks:
            for l, rs in D.rows(e).items():
                if R.is_tempo(l):
                    out[l] += sum(1 for r in rs if r.tempo_source and r.tempo_source != live)
    return {l: n for l, n in out.items() if n}


def caveats(D: Data) -> list:
    """Reasons the numbers must be read with care (printed in the executive summary)."""
    cp = D._extra.get("completeness") or completeness(D)
    out = []
    if D.cfg.smoke:
        out.append("SMOKE MODE: tiny runs of the pipeline check; the numbers say nothing about the algorithms.")
    if not D.has_data:
        return out + ["No result files exist yet: everything is pending."]
    if not D.has_main:
        out.append(f"{D.main} has no finished run: all TEMPO statistics are pending.")
    if cp["main"][1] and cp["main"][0] < cp["main"][1]:
        out.append(f"INCOMPLETE: {cp['main'][1] - cp['main'][0]} expected {D.main} runs and {cp['baselines'][1] - cp['baselines'][0]} expected baseline runs are missing; "
                   "verdicts marked provisional can change.")
    elif cp["baselines"][0] < cp["baselines"][1]:
        out.append(f"{cp['baselines'][1] - cp['baselines'][0]} expected baseline runs are missing.")
    vs = verdicts(D)
    nv = [v for v in vs if v["status"] in ("YES", "NO")]
    if nv:
        out.append(f"YES/NO verdicts: {len(nv)}; provisional (not every seed or baseline has finished): {sum(bool(v['provisional']) for v in nv)}; not statistically "
                   f"significant or untested: {sum(1 for v in nv if verdict_tag(v).split(',')[0] in ('n.s.', 'untested'))}. A YES is a higher mean, not a proven win, "
                   "unless it is marked sig.")
    prev = sorted({(e, l) for (e, l), o in D.origin.items() if o == "previous"})
    if prev:
        out.append(f"{len(prev)} baseline cell(s) come from the previous results (benchmarks/results/final), not from Baselines_test: "
                   + ", ".join(f"{e}/{l}" for e, l in prev[:6]) + (" ..." if len(prev) > 6 else ""))
    audit = D._extra.get("audit") or protocol_audit(D)
    errs = [a for a in audit if a["severity"] == "ERROR"]
    if errs:
        out.append(f"PROTOCOL MISMATCH on {len(errs)} task(s): " + ", ".join(a["task"] for a in errs) + " (budget / evaluation schedule differ between algorithms).")
    kinds = Counter()
    ktasks = defaultdict(set)
    for a in audit:
        if a["severity"] == "WARN":
            k = _warn_kind(a["message"])
            if k:
                kinds[k] += 1
                ktasks[k].add(a["task"])
    if kinds:
        out.append("Protocol warnings (section 2): " + "; ".join(f"{k} ({_pl(len(ktasks[k]), 'task')})" for k, _ in kinds.most_common(4))
                   + (f"; and {len(kinds) - 4} more kind(s)" if len(kinds) > 4 else "") + ".")
    n_ex = len(excluded_table(D))
    if n_ex:
        out.append(f"{n_ex} result file(s) / log(s) were read but not used (corrupt, off-protocol, failed): see the inventory.")
    ts = {r.tempo_source for e in D.tasks for l, rs in D.rows(e).items() if R.is_tempo(l) for r in rs}
    if len(ts) > 1:
        out.append(f"TEMPO results come from {len(ts)} different tempo_rl sources ({', '.join(sorted(ts))}): runs made with different code are mixed.")
    old = old_code_runs(D)
    if old:
        out.append(f"{sum(old.values())} TEMPO run(s) were made with older tempo_rl code than the live library ({live_source_hash()}); the TEMPO_test notebooks "
                   "re-run them at the next launch (ONLY_LIVE_CODE = True leaves them out of this comparison).")
    cnt = D._extra.get("tempo_counts", {})
    if cnt.get(D.main, 0) and len(cnt) > 1:
        top, k_top = max(cnt.items(), key=lambda kv: kv[1])
        if top != D.main and k_top >= 2 * cnt[D.main]:
            out.append(f"The headline {D.main} has {_pl(cnt[D.main], 'run')} but {top} has {k_top}: the headline numbers describe the former; set MAIN_VARIANT to change that.")
    left = [e for e in D.tasks if e in D.cells and e not in P.TOY_TEXT_ALL and not D.norm[e]["valid"] and any(b in D.cells[e] for b in BASELINES)]
    if left:
        out.append(f"{_pl(len(left), 'task')} left out of the normalised aggregates, sample efficiency and stability (best baseline not usable as a reference): "
                   + ", ".join(left) + ". The 'vs best baseline' tables and verdicts still count them.")
    un = [e for e in D.tasks if D.is_untuned(e) and e not in P.TOY_TEXT_ALL]      # toy text: no verdict at all (caveat below)
    if un:
        out.append(f"{_pl(len(un), 'task')} with untuned baselines (library defaults: {', '.join(_untuned_names(D, un))}"
                   + ("; in brackets the only untuned baselines of a task, which still counts as untuned" if _partly_untuned(D, un) else "")
                   + "): a win there is a weaker claim than on a tuned task; see the 'Tasks with tuned baselines' scope.")
    if D.cfg.ci_method != "t":
        out.append(f"The 95% per-task CIs are percentile bootstraps over (at most) {D.cfg.n_expected} seeds: {_coverage_text(D.cfg.n_expected)} "
                   "(CI_METHOD = \"t\" gives t-intervals); p-values are uncorrected for the number of tasks.")
    else:
        out.append("p-values are uncorrected for the number of tasks.")
    out.append("Wall-clock comparisons depend on devices (CPU vs GPU) and machine load.")
    if any(e in P.TOY_TEXT_ALL for e in D.tasks):
        toy_untuned = any(D.is_untuned(e) for e in D.tasks if e in P.TOY_TEXT_ALL)
        out.append("Toy-text tasks are judged against the exact optimum and are excluded from the verdicts and the normalised aggregates (their baseline runs are "
                   + ("untuned " if toy_untuned else "") + "PPO / DQN references that appear in the per-task tables and curves).")
    return out


def _agg_phrase(m: dict) -> str:
    return f"IQM {m['IQM'][0]:.2f} " + ("[n/a]" if not np.isfinite(m["IQM"][1]) else f"[{m['IQM'][1]:.2f}, {m['IQM'][2]:.2f}]")


def executive_summary(D: Data) -> str:
    """One-screen summary: data completeness, verdicts, win/tie/loss, aggregates, speed-up, stability, cost, caveats."""
    h = headline(D)
    c = h["completeness"]
    L = [f"EXECUTIVE SUMMARY  ({time.strftime('%Y-%m-%d %H:%M')}; {'SMOKE' if D.cfg.smoke else 'real'} mode; headline TEMPO label: {D.main})", "=" * 78,
         f"Data     : {D.main} {c['main_found']}/{c['main_expected']} expected runs on {c['tasks_main_any']}/{c['tasks']} tasks "
         f"({c['tasks_main_complete']} with all {D.cfg.n_expected} seeds); baselines {c['baselines_found']}/{c['baselines_expected']} expected runs; "
         f"{_pl(c['tasks_complete'], 'task')} fully comparable."]
    off = sorted({a["task"] for a in (D._extra.get("audit") or protocol_audit(D)) if a["severity"] == "ERROR"
                  or (a["severity"] == "WARN" and _warn_kind(a["message"]) in ("budget / evaluation schedule differs from the expected protocol",
                                                                              "episodes per evaluation differ from the expected protocol"))},
                 key=R.task_key)
    L.append(f"Protocol : expected on every task: {protocol_text(D.cfg.smoke, D.tasks)}; "
             + (f"the budget / evaluation schedule of {_pl(len(off), 'task')} deviates from it or differs between algorithms "
                f"({', '.join(off[:6])}{' ...' if len(off) > 6 else ''}; section 2)."
                if off else ("every run has this budget and evaluation schedule." if D.has_data else "no runs yet.")))
    vc, vd, tg = h["verdicts"], h["verdict_detail"], h["verdict_tags"]
    L.append(f"Verdicts : 'TEMPO beats all baselines' YES on {vc.get('YES', 0)} (significantly better than every baseline on {vd['yes_significant']}; provisional on "
             f"{vd['yes_provisional']}), NO on {vc.get('NO', 0)} (significantly worse than a baseline on {vd['no_significantly_worse']}; provisional on {vd['no_provisional']}), "
             f"PENDING on {vc.get('PENDING', 0)} tasks"
             + (f" ({_pl(vc.get('N/A', 0), 'toy-text task')}: no verdict, judged against the exact optimum)" if vc.get("N/A") else "") + ".")
    for st_, key in (("YES", "YES"), ("NO", "NO ")):
        if h["verdict_tasks"][st_]:
            L.append(f"   {key} : " + ", ".join(t + (f" ({tg[t]})" if tg.get(t) else "") for t in h["verdict_tasks"][st_]))
    if vc.get("YES", 0) or vc.get("NO", 0):
        L.append("   (sig. = Welch-significant against every baseline; n.s. = at least one difference not significant; sig. worse = significantly worse than a baseline; "
                 "tie = equal mean to the best baseline; prov. = not every seed / baseline finished)")
    for metric, name in (("final", "final return"), ("auc", "AUC")):
        w = h.get("wtl_vs_best", {}).get(metric)
        if w:
            L.append(f"Vs best  : {name}: win / tie / loss = {w['win']} / {w['tie']} / {w['loss']} against the best baseline of each task "
                     f"({_pl(w['tasks'], 'task')} with both; Welch, alpha {R.ALPHA}" + (f"; {w['untested']} untested (n<{D.cfg.min_n_test})" if w["untested"] else "") + ").")
    if not h.get("wtl_vs_best"):
        L.append("Vs best  : pending (needs finished TEMPO and baseline runs on the same task).")
    a = h.get("aggregate")
    if a and a.get("main"):
        m = a["main"]
        scope = (_pl(a["main_tasks"], "task") if a["main_tasks"] == a["scope_tasks"]
                 else f"{_pl(a['main_tasks'], 'task')} (of {a['scope_tasks']} in the scope)")
        s = (f"Aggregate: normalised score over {scope} (random 0, best baseline 1): {_agg_phrase(m)}, "
             f"mean {m['mean'][0]:.2f}, optimality gap {m['OG'][0]:.2f}")
        if a.get("poi_vs_best"):
            p = a["poi_vs_best"]
            s += (f"; P({D.main} > best baseline) = {p['p']:.2f} " + ("[n/a]" if not np.isfinite(p["lo"]) else f"[{p['lo']:.2f}, {p['hi']:.2f}]")
                  + (f" (chance level for a post-hoc best baseline: {a['poi_null']:.2f}, not 0.50)" if a.get("poi_null") else ""))
        L.append(s + ".")
        for name, x in h.get("aggregate_like_for_like", {}).items():
            if x.get("main") and x.get("best"):
                L.append(f"           {name} ({x['tasks']}): {D.main} {_agg_phrase(x['main'])} vs best baseline {_agg_phrase(x['best'])}, gap {x['main']['OG'][0]:.2f}.")
    elif a:
        L.append("Aggregate: baselines only (no TEMPO runs yet).")
    else:
        L.append("Aggregate: pending (needs finished baseline runs and, for the comparison, TEMPO runs on the same tasks).")
    for name, x in h.get("rank", {}).items():
        wp = {k: v for k, v in x["wilcoxon_p_holm"].items() if np.isfinite(v)}
        L.append(f"Ranks    : {name} ({_pl(x['tasks'], 'task')}, {x['algorithms']} algorithms): {D.main} mean rank {x['main_mean_rank']:.2f}; Friedman "
                 + ("n/a" if not np.isfinite(x["friedman_p"]) else f"p = {_pf(x['friedman_p'])}") + "; Wilcoxon (Holm) vs "
                 + (", ".join(f"{k} p = {_pf(v)}" for k, v in wp.items()) if wp else "n/a (needs >= 6 tasks)")
                 + (f"; {', '.join(gone)} left out (CrossQ cannot run on pixel observations)"
                    if (gone := D._extra.get("rank_left_out", {}).get(name)) else "") + ".")
    sp = h.get("speedup", {})
    if 0.9 in sp and (sp[0.9]["both"] or sp[0.9]["baseline_reaches"]):
        s9 = sp[0.9]
        L.append(f"Efficiency: steps to reach 90% of the best baseline's final score ({_pl(s9['tasks'], 'task')}): vs the best-final baseline: " + speedup_line(s9) + ".")
        L.append("            vs the fastest baseline of each task: " + speedup_line(s9, "fastest_") + ".")
    elif sp:
        L.append("Efficiency: no task where both TEMPO and the best baseline reach 90% yet.")
    else:
        L.append("Efficiency: pending (needs finished TEMPO and baseline runs on the same task).")
    stb = h.get("stability", {})
    if D.main in stb:
        sm = stb[D.main]
        bs = {k: v for k, v in stb.items() if k in BASELINES and v.get("std (norm.)") is not None}
        std = sm.get("std (norm.)")
        ms = sm.get("mean score (norm.)")
        tail = ""
        if bs:
            comp = {k: v for k, v in bs.items() if (v.get("mean score (norm.)") or 0) >= 0.5 and v.get("relative std") is not None
                    and v.get("tasks", 0) >= 0.5 * (sm.get("tasks") or 1)}
            if comp:
                lo = min(comp, key=lambda k: comp[k]["relative std"])
                tail = (f"; the steadiest baseline among those scoring at least half of the best baseline ({lo}): std {comp[lo]['std (norm.)']:.2f}, "
                        f"mean score {comp[lo]['mean score (norm.)']:.2f}")
            else:
                tail = "; no baseline scores half of the best baseline on average, so spreads are not ranked (a low std at a low score is a floor effect)"
        L.append(f"Stability: across-seed std {'n/a (one seed)' if std is None else f'{std:.2f}'} (normalised; mean score "
                 f"{'n/a' if ms is None else f'{ms:.2f}'}){tail}; "
                 f"{sm.get('below random %') or 0:.0f}% of {D.main} runs below the random policy, {sm.get('collapsed %') or 0:.0f}% collapsed.")
    cs = h.get("compute", {})
    if D.main in cs:
        s = f"Cost     : {D.main} {cs[D.main]['mean_h_per_run']:.2f} h/run"
        if h.get("cost_ratio_tempo_over_crossq"):
            s += f"; {h['cost_ratio_tempo_over_crossq']:.2f}x CrossQ's wall-clock (geometric mean over tasks)"
        wc = h.get("wallclock_to_level") or {}
        if wc.get("both_fastest"):
            s += (f"; wall-clock to 90% of the reference, baseline hours / {D.main} hours = {wc['geo_fastest']:.2f} against the baseline needing the fewest hours "
                  f"({_pl(wc['both_fastest'], 'task')}; > 1 = {D.main} cheaper)")
        L.append(s + " (devices differ).")
    L += ["", "Caveats:"] + [f"  - {x}" for x in caveats(D)]
    return "\n".join(L)


_OUTPUT_SUFFIXES = {".png", ".pdf", ".md", ".tex", ".csv", ".json", ".txt"}
_FULL_RUN = {"inventory", "curves", "scores", "aggregates", "efficiency", "stability", "compute", "variants"}


def archive_stale(D: Data) -> list:
    """Earlier outputs (figures / tables) that this run did not rewrite, e.g. the figure of a TEMPO variant that no longer exists. After a COMPLETE
    run (every section executed) they are moved to ``<out>/_outdated/<time>/`` (never deleted); after a partial run they are only listed."""
    if not (D.cfg.save and D.has_data):
        return []
    mine = {pathlib.Path(p).resolve() for p in D.written}
    out_root = D.cfg.out_dir.resolve()
    stale = []
    for base in (D.cfg.fig_dir, D.cfg.table_dir):
        if base.is_dir():
            for f in sorted(base.rglob("*")):
                if f.is_file() and f.suffix.lower() in _OUTPUT_SUFFIXES and f.resolve() not in mine and "_outdated" not in f.relative_to(out_root).parts:
                    stale.append(f)
    if not stale:
        return []
    done = D._extra.get("sections", set())
    if D.cfg.archive_stale and _FULL_RUN <= done:
        import shutil
        dest = out_root / "_outdated" / time.strftime("%Y%m%d-%H%M%S")
        moved = []
        for f in stale:
            try:
                tgt = dest / f.resolve().relative_to(out_root)
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(tgt))
                moved.append(tgt)
            except (OSError, ValueError) as e:                    # noqa: BLE001
                print(f"  could not move {f}: {e}")
        if moved:
            print(f"{len(moved)} file(s) from earlier runs were not rewritten by this complete run (their content is outdated); moved to {dest}:")
            print("  " + ", ".join(sorted(str(m.relative_to(dest)) for m in moved)[:12]) + (" ..." if len(moved) > 12 else ""))
        return moved
    why = ("ARCHIVE_STALE_OUTPUTS = False" if not D.cfg.archive_stale else
           "this run did not execute every section (" + ", ".join(sorted(_FULL_RUN - done)) + " missing)")
    print(f"{len(stale)} file(s) in the output folders were not rewritten by this run and may be outdated ({why}); kept: "
          + ", ".join(str(f.relative_to(out_root)) for f in stale[:8]) + (" ..." if len(stale) > 8 else ""))
    return []


def export_all(D: Data) -> list:
    """Write every registered table (Markdown, LaTeX, CSV), the per-run score table, the inventory and ``summary.json`` into
    ``<out>/tables``. Figures are saved when they are drawn (``<out>/figures``). Nothing is written without data."""
    if not D.cfg.save:
        print("SAVE_OUTPUTS = False: nothing written.")
        return []
    if not D.has_data:
        print("No results yet: nothing to export (no figures, tables or folders are created).")
        return []
    tdir = D.cfg.table_dir
    tdir.mkdir(parents=True, exist_ok=True)
    paths = []

    def w(name, text):
        p = tdir / name
        p.write_text(text)
        paths.append(p)

    register(D, "inventory", "Seeds found / expected per task and algorithm", df=inventory_matrix(D).reset_index())
    register(D, "runs", "Every run used in the comparison (per-run score table)", df=runs_table(D))
    import pandas as pd

    vdf = pd.DataFrame([dict(task=v["task"], status=v["status"], provisional=v["provisional"], qualifier=verdict_tag(v), baselines_untuned=v["untuned"], text=v["text"])
                        for v in verdicts(D)])
    register(D, "verdicts", "Verdict per task", df=vdf, tex=df_to_tex(vdf[["task", "status", "provisional", "qualifier"]], "Verdict per task: TEMPO beats all baselines", "verdicts"))
    for name, t in _tables(D).items():
        df = t["df"]
        w(f"{name}.md", f"# {t['title']}\n\n" + (t["md"] if t["md"] is not None else df_to_md(df)))
        if df is not None:
            df.to_csv(tdir / f"{name}.csv", index=False)
            paths.append(tdir / f"{name}.csv")
        tex = t["tex"] if t["tex"] is not None else (df_to_tex(df, t["title"], name.replace("_", "-")) if df is not None and len(df) <= 80 else None)
        if tex is not None:
            w(f"{name}.tex", tex)
    summary = headline(D)
    eb, ef, en = D.cfg.protocol
    summary.update(generated=time.strftime("%Y-%m-%d %H:%M:%S"), smoke=D.cfg.smoke, n_boot=D.cfg.n_boot, caveats=caveats(D),
                   expected_protocol=dict(budget=eb, eval_freq=ef, n_eval=en, n_eval_by_task=n_eval_exceptions(D.cfg.smoke, D.tasks),
                                          seeds=list(D.cfg.seeds or FINAL_SEEDS)),
                   guards=dict(min_gap=D.cfg.min_gap, min_n_test=D.cfg.min_n_test, ci_method=D.cfg.ci_method, only_live_code=D.cfg.only_live_code),
                   sources=dict(baselines=[str(r / s) for r, s in D.cfg.baseline_sources], tempo=[str(r / s) for r, s in D.cfg.tempo_sources]))
    w("summary.json", json.dumps(_json_clean(summary), indent=1))
    w("executive_summary.txt", executive_summary(D) + "\n")
    D.written.extend(paths)
    return paths


def show_exports(D: Data) -> None:
    """Section 10: write the tables, list everything written, archive outdated earlier outputs, print the executive summary."""
    _mark(D, "export")
    paths = export_all(D)
    if paths:
        print(f"wrote {len(paths)} table files to {D.cfg.table_dir} and {len([p for p in D.written if p.suffix in ('.png', '.pdf')])} figure files "
              f"to {D.cfg.fig_dir}")
        names = sorted({p.stem for p in paths})
        print("tables: " + ", ".join(names))
        archive_stale(D)
    print()
    print(executive_summary(D))
