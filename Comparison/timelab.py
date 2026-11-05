"""Training time of every algorithm of the benchmark: the logic of ``Comparison/02_training_time.ipynb``.

Why not ``wall_s``? Every result file of ``benchmarks/run_one.py`` records ``wall_s``, the wall-clock of the whole run. It
includes the 50 evaluations (500 deterministic episodes) and it was measured while the scheduler ran up to a dozen other
runs on the same machine, so it mixes the algorithm's cost with the machine's load at that time. Here each
(task, algorithm, variant) is timed ALONE, one measurement at a time, in a fresh process, with the job and device of the
benchmark, and training and evaluation are timed separately.

What runs: the SAME code the benchmark runs. A measurement builds the job of ``notebooks/runlab.py`` (the test protocol:
budget 500,000 steps, an evaluation every 10,000 steps, 10 episodes, ``runlab.n_eval_for``: 1,000 on Blackjack-v1;
``learning_starts``, learning-rate schedules and every other budget-dependent setting follow from that budget; device from
``runlab.device_for``; one thread with
``OMP_NUM_THREADS=1`` as the scheduler sets it) and calls ``benchmarks.run_one.run_sb3`` / ``run_tempo`` UNMODIFIED in a
fresh subprocess. What differs from a benchmark run, all inside that subprocess only:

* the job's ``eval_freq`` is set beyond any step count (:data:`NO_EVAL_FREQ`), so run_one's own evaluation callback never
  fires during training (counted: ``evaluations_in_training`` must be 0, or the measurement fails);
* the algorithm class's ``learn`` is wrapped: it passes run_one's callback together with a timing callback, which stops
  training once the timed window is complete. After ``learn`` returns, run_one's callback is triggered ONCE (exactly
  the code of a benchmark evaluation: ``run_one.evaluate`` with run_one's ``predict``, incl. VecNormalize) and timed;
* TEMPO with its novelty bonus only: during a second timed window the bonus is patched to 0 (see below).

Per measurement (one result JSON):

* ``setup_s``: from where ``run_one.main`` starts its clock (library import, environment and model construction, buffer
  allocation) to the first environment step;
* warm-up: the steps before learning starts (uniformly random actions) and their rate;
* training rate: env-steps/s over a window of whole update cycles (SB3 off-policy: ``train_freq`` steps; PPO: one
  rollout of ``n_steps x n_envs``; TEMPO: one phase of ``train_freq x n_envs``) that starts ``SETTLE_STEPS`` after
  learning starts and spans at least ``TRAIN_STEPS`` steps; the GPU is synchronised at both window boundaries, so queued
  asynchronous work is counted; the window holds exactly one update per cycle it spans;
* TEMPO with its novelty bonus (``exploration_bonus`` > 0: ``default`` and ``c1``, except on the image observations of CarRacing,
  where TEMPO switches the bonus off): the bonus weight anneals to 0 at
  ``bonus_end_step`` = the entropy schedule's anneal fraction x the run's steps (half the run), and from then on
  ``train`` skips the RND branch (predictor update + bonus), so later updates cost less. The window above lies in the
  bonus part; a second window of the same length follows it with ``current_exploration_bonus`` patched to 0 on the model
  (exactly its value after the anneal; set before the update of the phase that ends the first window, GPU synchronised
  at the boundary): ``train_rate_after_bonus``;
* one evaluation (the task's ``runlab.n_eval_for`` episodes: ``N_EVAL``, 1,000 on Blackjack-v1) with the trained model, plus its
  per-step cost;
* derived full-run numbers at the protocol budget (:func:`derive`): training hours = setup + learning_starts / warm-up
  rate + (budget - learning_starts) / training rate (the budget rounded up to whole cycles, as the algorithms overshoot
  it; TEMPO with its bonus: the training rate up to ``bonus_end_step``, ``train_rate_after_bonus`` after it); evaluation
  hours = (budget / eval_freq) x one evaluation; their sum.

Approximations (said in the notebook): apart from TEMPO's bonus anneal (timed separately, above), the cost per training
step is assumed constant over the run (the SB3 algorithms' updates cost the same at every step, and so do TEMPO's within
each bonus regime; an environment's own cost can drift a little as the behaviour changes), and one evaluation is timed
with a policy trained only for the windows, whose episodes can be shorter than a trained policy's (``eval_h_at_limit`` =
every episode running to the time limit, an upper bound).

Fairness: measurements run strictly one at a time; before each one :func:`guard` refuses when ``benchmarks.run_one`` /
``benchmarks.scheduler`` (or another timing, or a baseline hyperparameter search of ``notebooks/tunelab.py``) processes are
running and prints the machine load.

Files: ``Comparison/timing/<task>/<algorithm>/<variant>/rep<k>.json`` (one per measurement; finished ones are skipped, so a
run can be interrupted and resumed), logs in ``Comparison/timing/logs/``, tables in ``Comparison/timing/tables/`` and the
figure in ``Comparison/timing/figures/``. Outputs deliberately do NOT go to ``Comparison/tables`` / ``Comparison/figures``:
``complab.archive_stale`` moves every file there that the comparison notebook did not write itself to ``_outdated/``.
Smoke mode (``TEMPO_NOTEBOOK_SMOKE=1``): everything under ``$TEMPO_NOTEBOOK_SCRATCH/Comparison_timing``, never the project.

API (notebook order)::

    import timelab as tl
    SMOKE = tl.smoke_flag(SMOKE)                        # TEMPO_NOTEBOOK_SMOKE wins (runlab.smoke_flag)
    CFG = tl.make_config(tasks=None, algos=None, tempo_variants=("default", "c1"), train_steps=5000, repeats=1, smoke=SMOKE)
    CFG.show()
    PLAN = tl.plan(CFG)                                 # one row per measurement + estimated duration
    tl.run(CFG, PLAN)                                   # sequential, resumable, interrupt-safe
    RES = tl.results(CFG)                               # loads the JSONs; RES.show("train_h") ... RES.show("benchmark")
    tl.plot(CFG, RES)                                   # figure (saved when drawn)
    tl.save(CFG, RES)                                   # tables .md / .csv / .tex

CLI (one measurement in a fresh subprocess, same guards and files as the notebook)::

    python Comparison/timelab.py time Hopper-v5 SAC
    python Comparison/timelab.py time Hopper-v5 TEMPO --variant c1 --repeat 2 --train-steps 5000
    python Comparison/timelab.py _measure '<spec json>'      # internal: the measurement process itself
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORMAT = "timelab-2"
#: suite name of the timing jobs (``smoke_timing`` in smoke mode); never a suite of the benchmark
SUITE = "timing"
#: env steps of the timed training window (rounded up to whole update cycles)
TRAIN_STEPS = 5_000
#: env steps between the start of learning and the start of the window (rounded up to whole cycles)
SETTLE_STEPS = 1_000
TEMPO_VARIANTS = ("default", "c1")
#: one task per family (classic discrete / continuous, Box2D discrete / continuous, MuJoCo, toy text)
FAMILY_SUBSET = ("CartPole-v1", "Pendulum-v1", "LunarLander-v3", "LunarLanderContinuous-v3", "Hopper-v5", "FrozenLake-v1")
#: Tasks of the test folders that the comparison leaves out unless ``TASKS`` names them (since 2026-11-04; the same
#: list as complab.LEFT_OUT): CarRacing-v3 takes days to run, so neither notebook of Comparison/ waits for it.
LEFT_OUT = ("CarRacing-v3",)
SMOKE_TASKS = ("CartPole-v1", "Pendulum-v1")
SMOKE_TRAIN_STEPS = 200
SMOKE_SETTLE_STEPS = 50
#: per measurement; the slowest is CarRacing when TASKS names it (2026-11-03, one run alone, Apple M5 Pro): TEMPO ~9 min = setup + learning_starts
#: 10,000 random steps at the simulator's ~81 env-steps/s (~2 min) + 1,000 settle and >= 5,000 timed steps at 17.6 env-steps/s
#: (~6 min; its novelty bonus is off on image observations, so no second window) + one evaluation of 10 episodes (~35 s); SAC ~8 min
#: (6,100 steps at 13.9 env-steps/s), TD3 and PPO less
TIMEOUT_S = 30 * 60
SMOKE_TIMEOUT_S = 120
#: eval_freq of the timed job: beyond any step count, so run_one's evaluation callback never fires during training
NO_EVAL_FREQ = 10 ** 12
#: processes that make a timing unfair (they share the machine); the last one is another timing
BUSY_MARKS = ("benchmarks.run_one", "benchmarks.scheduler", "timelab.py _measure", "benchmarks.tune_one", "tunelab.py _pool")
#: 1-minute load average above which the progress line warns (the measurement still runs)
LOAD_WARN = 3.0
BASELINES = ("SAC", "TD3", "CrossQ", "PPO", "DQN")
CONTINUOUS_ALGOS = ("SAC", "TD3", "CrossQ", "PPO")
DISCRETE_ALGOS = ("PPO", "DQN")
#: Stable-Baselines3 2.9 constructor defaults, used ONLY by the plan's estimate (a measurement reads the model itself)
SB3_DEFAULTS = {"SAC": dict(learning_starts=100, train_freq=1), "TD3": dict(learning_starts=100, train_freq=1),
                "CrossQ": dict(learning_starts=100, train_freq=1), "DQN": dict(learning_starts=100, train_freq=4),
                "PPO": dict(n_steps=2048)}
#: guesses of the plan's estimate: random warm-up steps per second, seconds per evaluation step, steps per evaluation episode of a
#: task without a time limit (Blackjack: 1-3 steps per hand; any other such task: 1,000)
WARMUP_RATE_GUESS = 1000.0
EVAL_STEP_GUESS_S = 1e-3
EPISODE_STEPS_GUESS = {"Blackjack-v1": 2}
#: An environment step of these pixel tasks is an agent step of k simulator frames (tempo_rl/env_util.py, CarRacingPreprocess:
#: frame-skip 4, the pipeline of every algorithm and every evaluation), while the gymnasium time limit counts frames: a measurement's
#: time limit (``eval_h_at_limit``, table 3's bound) is the time limit / k agent steps (CarRacing: 1,000 frames = 250 steps).
FRAME_SKIP = {"CarRacing-v3": 4}
#: the run folders whose finished runs give the in-benchmark wall-clock (table 5): kind -> (folder, suite)
BENCH_FOLDERS = {"baselines": ("Baselines_test", "baselines_test"), "tempo": ("TEMPO_test", "tempo_test")}
MARKERS = {"TEMPO": "o", "SAC": "s", "TD3": "^", "CrossQ": "D", "PPO": "v", "DQN": "P"}

TABLES = {"train_h": "Training time per run (hours)", "rate": "Training speed (env-steps/s)",
          "eval_h": "Evaluation time per run (hours)", "speedup": "TEMPO speed-up over each baseline (training time)",
          "benchmark": "In-benchmark wall-clock of finished runs (hours; includes evaluations and concurrent load)"}


# =========================================================================== helpers shared by both processes
def _now() -> float:
    return time.perf_counter()


def _versions() -> dict:
    import importlib.metadata as md

    out = {"python": platform.python_version()}
    for name, dists in (("torch", ("torch",)), ("stable_baselines3", ("stable-baselines3",)),
                        ("sb3_contrib", ("sb3-contrib",)), ("gymnasium", ("gymnasium",)), ("numpy", ("numpy",)),
                        ("mujoco", ("mujoco",)), ("box2d", ("box2d-py", "Box2D", "box2d"))):
        for d in dists:
            try:
                out[name] = md.version(d)
                break
            except md.PackageNotFoundError:
                continue
    return out


def _sysctl(key: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:                                   # noqa: BLE001 - not macOS
        return ""


def _linux_info(key: str, path: str = "/proc/cpuinfo") -> str:
    """The value of the first ``key:`` line of a /proc file ('' when absent, e.g. on macOS)."""
    try:
        for line in pathlib.Path(path).read_text().splitlines():
            if line.startswith(key):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "Apple GPU" if torch.backends.mps.is_available() else None
    except Exception:                                   # noqa: BLE001
        return None


def machine_info() -> dict:
    """Chip, GPU, cores, memory, OS and package versions of this machine (recorded in every result; macOS or Linux)."""
    mem = _sysctl("hw.memsize")
    perf, eff = _sysctl("hw.perflevel0.physicalcpu"), _sysctl("hw.perflevel1.physicalcpu")
    kb = _linux_info("MemTotal", "/proc/meminfo").split(" ")[0]
    return dict(chip=_sysctl("machdep.cpu.brand_string") or _linux_info("model name") or platform.processor() or platform.machine(),
                gpu=_gpu_name(), cpu_cores=os.cpu_count(), performance_cores=int(perf) if perf.isdigit() else None,
                efficiency_cores=int(eff) if eff.isdigit() else None,
                memory_gb=round(int(mem) / 2 ** 30, 1) if mem.isdigit() else round(int(kb) / 2 ** 20, 1) if kb.isdigit() else None,
                os=platform.platform(), machine=platform.machine(), versions=_versions())


def derive(m: dict, protocol: dict, time_limit=None) -> dict:
    """Full-run numbers at the protocol budget from the measured ones (``m``: the ``measured`` dict of a result).

    The run's environment steps are the budget rounded up to whole update cycles (SB3 and TEMPO finish the cycle they are
    in); the warm-up covers the steps up to ``learning_starts`` at the measured warm-up rate (0 when learning starts with
    the first step), the rest runs at the training rate. TEMPO with its novelty bonus: the training rate holds while the
    bonus is on (up to ``bonus_end_step``, half the run), the rate measured with the bonus at 0 after that. Evaluations:
    one every ``eval_freq`` steps (run_one's callbacks evaluate at every multiple of ``eval_freq`` the run reaches), each
    costing the measured evaluation."""
    budget, freq, n_eval = int(protocol["budget"]), int(protocol["eval_freq"]), int(protocol["n_eval"])
    c = max(1, int(m["cycle"]))
    steps = c * math.ceil(budget / c)
    first = int(m["first_step"])
    ls = min(max(int(m["learning_starts"]), first), steps)
    rate_w = m.get("warmup_rate")
    warm_s = (ls - first) / rate_w if rate_w and ls > first else 0.0
    bonus_end, rate_off = m.get("bonus_end_step"), m.get("train_rate_after_bonus")
    if bonus_end is not None and rate_off and int(bonus_end) < steps:
        e = max(int(bonus_end), ls)
        train_s = (e - ls) / float(m["train_rate"]) + (steps - e) / float(rate_off)
    else:
        train_s = (steps - ls) / float(m["train_rate"])
    run_rate = (steps - ls) / train_s if train_s > 0 else None
    train_h = (float(m["setup_s"]) + warm_s + train_s) / 3600.0
    n_evals = steps // freq
    eval_h = n_evals * float(m["eval_s"]) / 3600.0
    limit_h = None
    if time_limit and m.get("eval_s_per_step"):
        limit_h = max(eval_h, n_evals * n_eval * int(time_limit) * float(m["eval_s_per_step"]) / 3600.0)
    return dict(steps=int(steps), n_evals=int(n_evals), warmup_s=warm_s, train_s=train_s, run_rate=run_rate,
                train_h=train_h,
                eval_h=eval_h, total_h=train_h + eval_h, eval_h_at_limit=limit_h,
                total_h_at_limit=None if limit_h is None else train_h + limit_h)


# =========================================================================== the measurement process (child)
def _sync_fn(device):
    import torch

    kind = getattr(device, "type", str(device))
    if kind == "mps":
        return torch.mps.synchronize
    if kind == "cuda":
        return torch.cuda.synchronize
    return lambda: None


def _updates(model):
    v = getattr(model, "_n_updates", None)
    return int(v) if isinstance(v, (int, float)) else None


class _Window:
    """State of the timing callback: marks (env steps, time) at the first step, at learning start, at the window start
    ``a`` and end ``b``; stops training at the window end. TEMPO with its novelty bonus: a second window ``b``-``b2`` of
    the same length with the bonus at 0 (its value after the anneal), see :meth:`step`."""

    def __init__(self, train_steps: int, settle_steps: int, t_start: float, counters: dict):
        self.train_steps, self.settle = int(train_steps), int(settle_steps)
        self.t_start, self.counters = t_start, counters
        self.marks: dict = {}
        self.model = None
        self.eval_callback = None
        self.sync = lambda: None
        self.bonus_end = None           # TEMPO: first env step at which the novelty bonus is 0 (None: no bonus)
        self.b2 = None                  # end of the bonus-off window (None: no second window)
        self.patched: list = []         # models whose current_exploration_bonus is patched to 0

    @property
    def last(self) -> str:
        """The mark that completes the measurement."""
        return "b" if self.b2 is None else "b2"

    def unpatch(self) -> None:
        for m in self.patched:
            m.__dict__.pop("current_exploration_bonus", None)
        self.patched = []

    def begin(self, model, on_policy: bool) -> None:
        self.model = model
        self.sync = _sync_fn(model.device)
        self.t_learn = _now()
        n_envs = int(getattr(model, "n_envs", 1) or 1)
        if hasattr(model, "_learning_starts_resolved"):          # TEMPO: learning_starts resolved in _setup_learn
            self.ls = int(model._learning_starts_resolved or 0)
            self.cycle, self.cycle_kind = int(model.train_freq) * n_envs, "phase (train_freq x n_envs)"
            if getattr(model.policy, "rnd", None) is not None and float(model.exploration_bonus) > 0:
                # TEMPO.current_exploration_bonus: exploration_bonus * max(0, 1 - progress / frac), progress =
                # num_timesteps / _total_timesteps, frac = _schedule[2]; > 0 while num_timesteps < frac x total
                self.bonus_end = int(math.ceil(float(model._schedule[2]) * int(model._total_timesteps)))
        elif on_policy:                                         # PPO: one rollout, then its update epochs
            self.ls = 0
            self.cycle, self.cycle_kind = int(model.n_steps) * n_envs, "rollout (n_steps x n_envs)"
        else:                                                   # SB3 off-policy
            self.ls = int(getattr(model, "learning_starts", 0) or 0)
            tf = model.train_freq
            unit = getattr(getattr(tf, "unit", None), "value", "step")
            if unit == "step":
                self.cycle, self.cycle_kind = int(tf.frequency) * n_envs, "train_freq x n_envs"
            else:                                               # episode-based train_freq: no fixed cycle
                self.cycle, self.cycle_kind = n_envs, f"train_freq in {unit}s (window not cycle-aligned)"
        c = max(1, self.cycle)
        self.a = c * math.ceil((self.ls + max(1, self.settle)) / c)
        self.b = self.a + c * max(1, math.ceil(self.train_steps / c))
        if self.bonus_end is not None and self.bonus_end > self.b:  # the window [a, b] lies in the bonus part
            self.b2 = self.b + (self.b - self.a)

    def _mark(self, name: str, n: int) -> None:
        self.sync()
        self.marks[name] = dict(step=int(n), t=_now(), cpu=time.process_time(), updates=_updates(self.model),
                                evaluations=len(self.counters["evaluate"]), records=len(self.counters["records"]))

    def step(self, model) -> bool:
        n = int(model.num_timesteps)
        if "first" not in self.marks:
            self._mark("first", n)
        if "ls" not in self.marks and n >= self.ls:
            self._mark("ls", n)
        if "a" not in self.marks:
            if n >= self.a:
                self._mark("a", n)
            return True
        if "b" not in self.marks:
            if n < self.b:
                return True
            self._mark("b", n)                                  # synchronised: the bonus-on window ends here
            if self.b2 is None:
                return False                                    # stop training: the window is complete
            # TEMPO after its bonus anneal: current_exploration_bonus() == 0 skips the RND branch of train(). Set
            # here, at the last collection step of the phase that ends at b, i.e. before that phase's update.
            for m in (model, getattr(getattr(model, "_audition", None), "shadow", None)):
                if m is not None:
                    m.current_exploration_bonus = lambda: 0.0
                    self.patched.append(m)
            return True
        if n >= self.b2:
            self._mark("b2", n)
            self.unpatch()
            return False                                        # stop training: both windows are complete
        return True


def _timer_class(base):
    class Timer(base):
        def __init__(self, ctl: _Window, on_policy: bool):
            super().__init__()
            self.ctl, self.on_policy = ctl, on_policy

        def _on_training_start(self) -> None:
            self.ctl.begin(self.model, self.on_policy)

        def _on_step(self) -> bool:
            return self.ctl.step(self.model)

    return Timer


def _measure(spec: dict) -> dict:
    """Time one (task, algorithm, variant) as specified (runs in its own process; see the module docstring)."""
    import resource

    t_proc = _now()
    load_before = os.getloadavg()
    job = dict(spec["job"])
    algo = job["algo"]
    protocol = dict(budget=int(job["steps"]), eval_freq=int(job["eval_freq"]), n_eval=int(job["n_eval"]))
    used_freq = int(spec.get("test_eval_freq") or NO_EVAL_FREQ)
    run_job = dict(job, eval_freq=used_freq)                    # the dict run_one's callbacks read
    import numpy as np
    import torch

    from benchmarks import run_one

    torch.set_num_threads(int(job.get("threads", 1)))           # as run_one.main
    counters = {"phase": "training", "evaluate": [], "records": []}
    ctl = _Window(spec["train_steps"], spec["settle_steps"], 0.0, counters)
    orig_evaluate = run_one.evaluate

    def counting_evaluate(predict, env_id, n_eval, make_env):     # run_one's callbacks look `evaluate` up at call time
        t0 = _now()
        out = orig_evaluate(predict, env_id, n_eval, make_env)
        model = ctl.model
        counters["evaluate"].append(dict(phase=counters["phase"], s=_now() - t0,
                                         step=int(model.num_timesteps) if model is not None else -1))
        return out

    def record(step, rets, lens, extra=None):
        counters["records"].append(dict(phase=counters["phase"], step=int(step), returns=[float(r) for r in rets],
                                        lengths=[int(x) for x in lens]))

    import_s = _now() - t_proc
    t_start = _now()                                            # == run_one.main's t0 (just before the runner)
    ctl.t_start = t_start
    if algo == "TEMPO":
        import tempo_rl.tempo
        from tempo_rl.callbacks import BaseCallback

        cls, on_policy, runner = tempo_rl.tempo.TEMPO, False, run_one.run_tempo
    else:
        if job["env"].startswith("CarRacing"):   # as benchmarks.run_one.run_sb3: keep OpenCV out of a pygame process
            sys.modules.setdefault("cv2", None)
        import sb3_contrib
        import stable_baselines3 as sb3
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

        cls = {"SAC": sb3.SAC, "TD3": sb3.TD3, "PPO": sb3.PPO, "DQN": sb3.DQN, "CrossQ": sb3_contrib.CrossQ}[algo]
        on_policy, runner = issubclass(cls, OnPolicyAlgorithm), run_one.run_sb3
    Timer = _timer_class(BaseCallback)
    orig_learn = cls.learn
    had_own_learn = "learn" in cls.__dict__
    evaluation: dict = {}

    def learn(self, total_timesteps, callback=None, **kwargs):
        ctl.eval_callback = callback
        try:
            out = orig_learn(self, total_timesteps, callback=[callback, Timer(ctl, on_policy)], **kwargs)
        finally:
            ctl.unpatch()                                       # no-op unless training ended inside window 2
        if ctl.last not in ctl.marks:
            end = ctl.b if ctl.b2 is None else ctl.b2
            raise RuntimeError(f"training ended at {self.num_timesteps:,} steps before the timing window "
                               f"{ctl.a:,}-{end:,} was complete")
        # ---- one evaluation, exactly as the benchmark does it: run_one's own callback, triggered once
        counters["phase"] = "evaluation"
        n_rec, n_ev = len(counters["records"]), len(counters["evaluate"])
        ctl.sync()
        t0 = _now()
        if algo == "TEMPO":                                     # Eval: `num_timesteps % job["eval_freq"] < n_envs`
            run_job["eval_freq"] = 1
            try:
                callback._on_step()
            finally:
                run_job["eval_freq"] = used_freq
        else:                                                   # Eval: `while num_timesteps >= next_eval`
            callback.next_eval = callback.num_timesteps
            callback._on_step()
        ctl.sync()
        evaluation["s"] = _now() - t0
        if len(counters["records"]) != n_rec + 1 or len(counters["evaluate"]) != n_ev + 1:
            raise RuntimeError("triggering benchmarks.run_one's evaluation callback did not run exactly one evaluation: "
                               "run_one.py changed, update Comparison/timelab.py")
        counters["phase"] = "after"
        return out

    cls.learn = learn
    run_one.evaluate = counting_evaluate
    try:
        info = runner(run_job, record)
    finally:
        run_one.evaluate = orig_evaluate
        if had_own_learn:
            cls.learn = orig_learn
        else:
            del cls.learn
    t_end = _now()

    mk = ctl.marks
    first, ls_m, a, b = mk["first"], mk["ls"], mk["a"], mk["b"]
    ev_rec = [r for r in counters["records"] if r["phase"] == "evaluation"][0]
    ev_call = [e for e in counters["evaluate"] if e["phase"] == "evaluation"][0]
    in_training = [e for e in counters["evaluate"] if e["phase"] == "training"]
    rec_training = [r for r in counters["records"] if r["phase"] == "training"]
    eval_steps = int(sum(ev_rec["lengths"]))
    warm_steps, warm_s = ls_m["step"] - first["step"], ls_m["t"] - first["t"]
    window_steps, window_s = b["step"] - a["step"], b["t"] - a["t"]
    b2 = mk.get("b2")                                           # TEMPO with its novelty bonus: the bonus-off window
    last = b2 or b
    w2 = {} if b2 is None else dict(
        window2_start=b["step"], window2_end=b2["step"], window2_steps=b2["step"] - b["step"],
        window2_s=b2["t"] - b["t"], window2_cpu_s=b2["cpu"] - b["cpu"],
        window2_updates=None if b["updates"] is None or b2["updates"] is None else b2["updates"] - b["updates"],
        evaluations_in_window2=b2["evaluations"] - b["evaluations"])
    measured = dict(
        import_s=import_s, setup_s=first["t"] - t_start, construct_s=ctl.t_learn - t_start,
        learn_setup_s=first["t"] - ctl.t_learn, first_step=first["step"], learning_starts=ctl.ls, cycle=ctl.cycle,
        cycle_kind=ctl.cycle_kind, warmup_steps=warm_steps, warmup_s=warm_s,
        warmup_rate=warm_steps / warm_s if warm_steps > 0 and warm_s > 0 else None,
        settle_steps=a["step"] - ls_m["step"], settle_s=a["t"] - ls_m["t"],
        window_start=a["step"], window_end=b["step"], window_steps=window_steps, window_s=window_s,
        train_rate=window_steps / window_s,
        bonus_end_step=ctl.bonus_end,
        train_rate_after_bonus=None if b2 is None else (b2["step"] - b["step"]) / (b2["t"] - b["t"]),
        **w2,
        window_cpu_s=b["cpu"] - a["cpu"],
        window_updates=None if a["updates"] is None or b["updates"] is None else b["updates"] - a["updates"],
        evaluations_in_training=len(in_training), records_in_training=len(rec_training),
        evaluations_in_window=b["evaluations"] - a["evaluations"], records_in_window=b["records"] - a["records"],
        evaluation_steps_in_training=[e["step"] for e in in_training][:50],
        eval_s=evaluation["s"], eval_evaluate_s=ev_call["s"], eval_episodes=len(ev_rec["lengths"]),
        eval_steps=eval_steps, eval_s_per_step=evaluation["s"] / eval_steps if eval_steps else None,
        eval_lengths=ev_rec["lengths"], eval_returns=ev_rec["returns"], eval_at_step=ev_rec["step"],
        steps_simulated=last["step"], process_s=t_end - t_proc)
    if measured["evaluations_in_training"] and not spec.get("test_eval_freq"):
        raise RuntimeError(f"{measured['evaluations_in_training']} evaluation(s) ran during training: the timing is invalid")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / 2 ** 20 if sys.platform == "darwin" else rss / 2 ** 10
    mps = bool(torch.backends.mps.is_available())
    tempo_source = run_one.source_hash()
    return dict(
        format=FORMAT, task=job["env"], algo=algo, variant=job.get("variant", "default"), label=spec["label"],
        repeat=int(spec["repeat"]), seed=int(job["seed"]), device=job.get("device", "cpu"),
        train_steps=int(spec["train_steps"]), settle_steps=int(spec["settle_steps"]), time_limit=spec.get("time_limit"),
        positive_control=bool(spec.get("test_eval_freq")), protocol=protocol, job=job, timed_job_eval_freq=used_freq,
        measured=measured, derived=derive(measured, protocol, spec.get("time_limit")), info=info,
        tempo_source=tempo_source, code_root=str(ROOT),
        machine=dict(machine_info(), device=job.get("device", "cpu"), mps_available=mps,
                     torch_threads=torch.get_num_threads(), omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
                     peak_rss_mb=round(rss_mb, 1), numpy_version=np.__version__),
        load_before=list(load_before), load_after=list(os.getloadavg()),
        finished=time.strftime("%Y-%m-%d %H:%M:%S"))


def _child_main(spec_text: str) -> None:
    spec = json.loads(spec_text)
    out = pathlib.Path(spec["out"])
    res = _measure(spec)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, default=repr))
    os.replace(tmp, out)
    m, d = res["measured"], res["derived"]
    off = (f" (bonus off {m['window2_start']:,}-{m['window2_end']:,}: {m['train_rate_after_bonus']:,.1f})"
           if m.get("train_rate_after_bonus") else "")
    print(f"saved {out}: setup {m['setup_s']:.1f} s, warm-up {m['warmup_steps']:,} steps, window "
          f"{m['window_start']:,}-{m['window_end']:,} at {m['train_rate']:,.1f} env-steps/s{off}, one evaluation "
          f"{m['eval_s']:.2f} s ({m['eval_steps']:,} steps); per run: train {d['train_h']:.3f} h, eval {d['eval_h']:.3f} h",
          flush=True)


# =========================================================================== notebook side (parent)
def _rl():
    nb = ROOT / "notebooks"
    if str(nb) not in sys.path:
        sys.path.insert(0, str(nb))
    import runlab                                               # creates no file and prints nothing on import

    return runlab


def _report():
    from benchmarks import report

    return report


def smoke_flag(default: bool = False) -> bool:
    """``runlab.smoke_flag``: the environment variable ``TEMPO_NOTEBOOK_SMOKE`` (1/true/yes/on, 0/false/no/off) wins over
    the cell's value; any other value raises before anything is created."""
    return _rl().smoke_flag(default)


@dataclass
class Config:
    """Resolved parameters (:func:`make_config`)."""
    tasks: tuple
    algos: tuple | None
    tempo_variants: tuple
    train_steps: int
    settle_steps: int
    repeats: int
    smoke: bool
    out: pathlib.Path                   # result JSONs: <out>/<task>/<algo>/<variant>/rep<k>.json, logs in <out>/logs
    tables: pathlib.Path
    figures: pathlib.Path
    timeout: float | None
    rerun_outdated: bool
    budget: int
    eval_freq: int
    n_eval: int
    seeds: tuple
    bench_root: pathlib.Path            # holds Baselines_test/ and TEMPO_test/ (table 5, read only)
    notes: list = field(default_factory=list)

    @property
    def suite(self) -> str:
        return ("smoke_" if self.smoke else "") + SUITE

    def n_eval_of(self, task: str) -> int:
        """Episodes per evaluation of ``task`` in the test protocol (``runlab.n_eval_for``: :attr:`n_eval`, more on the tasks of
        ``runlab.N_EVAL_BY_TASK``); the timing jobs are always real-protocol jobs, also in smoke mode."""
        return int(_rl().n_eval_for(task))

    def n_eval_text(self) -> str:
        """``10 episodes``, or ``10 episodes (Blackjack-v1: 1,000)`` when tasks of this configuration expect another number."""
        odd = {t: self.n_eval_of(t) for t in self.tasks if self.n_eval_of(t) != self.n_eval}
        return f"{self.n_eval} episodes" + (" (" + ", ".join(f"{t}: {k:,}" for t, k in odd.items()) + ")" if odd else "")

    def show(self) -> None:
        what = ("the task's tuned baselines + TEMPO" if self.algos is None else ", ".join(self.algos))
        tv = ", ".join(self.tempo_variants) or "none"
        print("mode       : " + ("SMOKE (pipeline check in the scratch folder; the numbers mean nothing)" if self.smoke
                                 else "REAL"))
        print(f"tasks      : {len(self.tasks)} ({', '.join(self.tasks[:4])}{' ...' if len(self.tasks) > 4 else ''})")
        print(f"algorithms : {what}; TEMPO variants: {tv} ('default' = the library default, the fast recipe; "
              "'c1' = the previous default)")
        print(f"window     : learning starts + {self.settle_steps:,} settle steps, then >= {self.train_steps:,} timed env "
              "steps (whole update cycles); GPU synchronised at both ends; then one evaluation")
        print(f"protocol   : {self.budget:,} steps per run, an evaluation every {self.eval_freq:,} steps "
              f"({self.budget // self.eval_freq} x {self.n_eval_text()}) = runlab.TEST_BUDGET / TEST_EVAL_FREQ / n_eval_for")
        seeds = ", ".join(str(self.seeds[(k - 1) % len(self.seeds)]) for k in range(1, self.repeats + 1))
        limit = f"timeout {self.timeout / 60:.0f} min each" if self.timeout else "no timeout"
        print(f"repeats    : {self.repeats} (seed{'s' if self.repeats > 1 else ''} {seeds}); strictly one measurement "
              f"at a time, each in a fresh process; {limit}")
        print(f"results    : {self.out}/<task>/<algorithm>/<variant>/rep<k>.json (logs: {self.out / 'logs'})")
        print(f"outputs    : {self.tables / 'training_time'}.md|csv|tex, {self.figures / 'training_time'}.png|pdf")
        for n in self.notes:
            print("note       : " + n)


def make_config(tasks=None, algos=None, tempo_variants=TEMPO_VARIANTS, train_steps: int = TRAIN_STEPS,
                settle_steps: int | None = None, repeats: int = 1, smoke: bool = False, out_dir=None,
                timeout_min: float | None = None, rerun_outdated: bool = True, smoke_tasks: bool = True,
                bench_root=None) -> Config:
    """Resolve the parameters. ``tasks``: None / "all" = the 23 tasks of the comparison (``runlab.tasks()`` without
    :data:`LEFT_OUT`), "subset" = :data:`FAMILY_SUBSET`, or a list (which may name a left-out task).
    ``algos``: None = per task its baselines (``report.expected_pool``; on toy text its PPO / DQN references) + TEMPO; or names among
    SAC, TD3, CrossQ, PPO, DQN, TEMPO (an algorithm that cannot act in a task is skipped there; PPO / DQN on toy text run
    with the grid-search winners of benchmarks/tuned_toy_text.json, as in Baselines_test). ``tempo_variants``: TEMPO configurations (``benchmarks/variants.py`` names; "default"
    = ``TEMPO("MlpPolicy", env)``). ``smoke`` (``TEMPO_NOTEBOOK_SMOKE`` wins): tasks :data:`SMOKE_TASKS` (unless
    ``smoke_tasks=False``), :data:`SMOKE_TRAIN_STEPS` / :data:`SMOKE_SETTLE_STEPS`, one repeat, everything under the
    scratch folder. ``out_dir``: results root (default ``Comparison/timing``); tables / figures go to its ``tables/`` and
    ``figures/``. ``rerun_outdated``: measurements of other code (tempo_rl source, library versions) or other settings are
    moved to ``<out>/_outdated/<time>/`` and timed again. Creates no file."""
    rl = _rl()
    from benchmarks.variants import VARIANTS

    notes = []
    smoke = rl.smoke_flag(smoke)
    all_tasks = rl.tasks()
    if smoke and smoke_tasks:
        tasks = SMOKE_TASKS
    elif tasks is None or (isinstance(tasks, str) and tasks == "all"):
        tasks = tuple(t for t in all_tasks if t not in LEFT_OUT)
        notes.append(f"left out (timelab.LEFT_OUT, as in the comparison): {', '.join(LEFT_OUT)}; TASKS = [...] naming it times it")
    elif isinstance(tasks, str) and tasks == "subset":
        tasks = FAMILY_SUBSET
    elif isinstance(tasks, str):
        tasks = (tasks,)
    tasks = tuple(tasks)
    bad = [t for t in tasks if t not in all_tasks]
    if bad:
        raise ValueError(f"unknown task(s) {bad}; known: {all_tasks}")
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"TASKS lists a task twice: {list(tasks)}")
    tasks = tuple(t for t in all_tasks if t in tasks)
    if algos is not None:
        algos = (algos,) if isinstance(algos, str) else tuple(algos)
        bad = [a for a in algos if a not in (*BASELINES, "TEMPO")]
        if bad:
            raise ValueError(f"unknown algorithm(s) {bad}; known: {[*BASELINES, 'TEMPO']}")
    tempo_variants = (tempo_variants,) if isinstance(tempo_variants, str) else tuple(tempo_variants or ())
    bad = [v for v in tempo_variants if v not in VARIANTS]
    if bad:
        raise ValueError(f"unknown TEMPO variant(s) {bad}; known (benchmarks/variants.py): {sorted(VARIANTS)}")
    if smoke:
        train_steps, settle_steps, repeats = SMOKE_TRAIN_STEPS, SMOKE_SETTLE_STEPS, 1
        timeout = SMOKE_TIMEOUT_S                       # a smoke measurement takes seconds: TIMEOUT_MIN does not apply
    else:
        settle_steps = SETTLE_STEPS if settle_steps is None else settle_steps
        timeout = TIMEOUT_S if timeout_min is None else (None if timeout_min <= 0 else 60.0 * timeout_min)
    for name, v in (("TRAIN_STEPS", train_steps), ("SETTLE_STEPS", settle_steps), ("REPEATS", repeats)):
        if isinstance(v, bool) or int(v) != v or int(v) < (0 if name == "SETTLE_STEPS" else 1):
            raise ValueError(f"{name} must be a whole number >= {0 if name == 'SETTLE_STEPS' else 1}, not {v!r}")
    if out_dir is not None:
        out = pathlib.Path(out_dir).expanduser()
    elif smoke:
        out = rl.bl.scratch_root() / "Comparison_timing"
    else:
        out = HERE / "timing"
    if smoke and out.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"smoke mode never writes into the project ({out}); set TEMPO_NOTEBOOK_SCRATCH elsewhere")
    if smoke:
        notes.append(f"SMOKE: tasks {', '.join(tasks)}, {train_steps} timed steps, {settle_steps} settle steps, one repeat")
    return Config(tasks=tasks, algos=algos, tempo_variants=tempo_variants, train_steps=int(train_steps),
                  settle_steps=int(settle_steps), repeats=int(repeats), smoke=bool(smoke), out=out,
                  tables=out / "tables", figures=out / "figures", timeout=timeout, rerun_outdated=bool(rerun_outdated),
                  budget=int(rl.TEST_BUDGET), eval_freq=int(rl.TEST_EVAL_FREQ), n_eval=int(rl.N_EVAL),
                  seeds=tuple(int(s) for s in rl.TEST_SEEDS), bench_root=pathlib.Path(bench_root or ROOT), notes=notes)


# --------------------------------------------------------------------------- the measurements
@dataclass
class Item:
    """One planned measurement."""
    task: str
    algo: str
    variant: str
    label: str
    repeat: int
    job: dict
    path: pathlib.Path
    log: pathlib.Path
    time_limit: int | None
    learning_starts: int
    cycle: int
    window: tuple                       # (start, end) env steps of the timed window, TEMPO with its novelty bonus
                                        # (start, end, end of the bonus-off window) (plan estimate)
    est_s: float
    state: str = "to run"
    reason: str = ""

    def spec(self, cfg: Config, test_eval_freq: int | None = None) -> dict:
        s = dict(job=self.job, label=self.label, repeat=self.repeat, train_steps=cfg.train_steps,
                 settle_steps=cfg.settle_steps, time_limit=self.time_limit, out=str(self.path))
        if test_eval_freq:
            s["test_eval_freq"] = int(test_eval_freq)
        return s


@contextlib.contextmanager
def _real_protocol_env():
    """``runlab.make_lab`` reads TEMPO_NOTEBOOK_SMOKE; the timing jobs are ALWAYS the real-protocol jobs (smoke mode only
    shortens the timed window), so the variable is hidden while they are built."""
    saved = os.environ.pop("TEMPO_NOTEBOOK_SMOKE", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["TEMPO_NOTEBOOK_SMOKE"] = saved


def _schedule(job: dict) -> tuple[int, int]:
    """(learning_starts, update cycle) of a job, for the plan's estimate (a measurement reads both from the model)."""
    algo = job["algo"]
    if algo == "TEMPO":
        import inspect

        from benchmarks.variants import tempo_kwargs
        from tempo_rl.tempo import TEMPO

        d = {k: p.default for k, p in inspect.signature(TEMPO.__init__).parameters.items()}
        kw = tempo_kwargs(job.get("variant", "default"), job["env"])
        n_envs = int(kw.get("n_envs", d["n_envs"]))
        tf = kw.get("train_freq", d["train_freq"])
        if tf is None:
            tf = max(1, int(round(int(kw.get("batch_size", d["batch_size"]))
                                  / (int(kw.get("ref_batch", d["ref_batch"])) * max(1, n_envs)))))
        ls = kw.get("learning_starts", d["learning_starts"])
        if ls is None:
            ls = int(min(max(0.02 * job["steps"], 1000), 10_000))
        return int(ls), int(tf) * n_envs
    from benchmarks.zoo import zoo_hyperparams

    hp, _ = zoo_hyperparams(algo, job["env"])
    n_envs = int(hp.get("n_envs", 1))
    if algo == "PPO":
        return 0, int(hp.get("n_steps", SB3_DEFAULTS["PPO"]["n_steps"])) * n_envs
    return (int(hp.get("learning_starts", SB3_DEFAULTS[algo]["learning_starts"])),
            int(hp.get("train_freq", SB3_DEFAULTS[algo]["train_freq"])) * n_envs)


def _tempo_bonus(job: dict) -> bool:
    """Whether a TEMPO job trains with the novelty bonus (plan estimate; a measurement reads the model). Never on the pixel tasks
    (``plans.IMAGE``): TEMPO switches the bonus off for image-only observations (its novelty estimator reads vector observations)."""
    from benchmarks import plans

    if job["algo"] != "TEMPO" or job["env"] in plans.IMAGE:
        return False
    import inspect

    from benchmarks.variants import tempo_kwargs
    from tempo_rl.tempo import TEMPO

    d = inspect.signature(TEMPO.__init__).parameters["exploration_bonus"].default
    return float(tempo_kwargs(job.get("variant", "default"), job["env"]).get("exploration_bonus", d)) > 0


def _window(ls: int, cycle: int, settle: int, train: int, bonus: bool = False) -> tuple:
    """(a, b) of the timed window; TEMPO with its novelty bonus: (a, b, b2), b-b2 = the bonus-off window."""
    c = max(1, cycle)
    a = c * math.ceil((ls + max(1, settle)) / c)
    b = a + c * max(1, math.ceil(train / c))
    return (a, b, b + (b - a)) if bonus else (a, b)


def _live_code() -> dict:
    rl = _rl()
    v = _versions()
    return dict(tempo_source=rl.bl.source_hash(ROOT), torch=v.get("torch"), sb3=v.get("stable_baselines3"),
                sb3_contrib=v.get("sb3_contrib"), gymnasium=v.get("gymnasium"))


def _state(cfg: Config, it: Item, live: dict) -> tuple[str, str]:
    if not it.path.exists():
        return "to run", ""
    try:
        r = json.loads(it.path.read_text())
        m = r["measured"]
        float(m["train_rate"]), float(m["eval_s"])
    except Exception as e:                              # noqa: BLE001
        return "unreadable", f"{type(e).__name__}"
    if r.get("format") != FORMAT:
        return "other settings", f"format {r.get('format')!r}"
    want = dict(train_steps=cfg.train_steps, settle_steps=cfg.settle_steps)
    diff = [f"{k} {r.get(k)} (now {v})" for k, v in want.items() if r.get(k) != v]
    j, p = r.get("job", {}), r.get("protocol", {})
    for k in ("steps", "eval_freq", "n_eval", "device", "threads", "seed", "env_kwargs"):
        if j.get(k) != it.job.get(k):
            diff.append(f"job {k} {j.get(k)!r} (now {it.job.get(k)!r})")
    if diff:
        return "other settings", "; ".join(diff)
    if it.algo != "TEMPO":       # a baseline timed with other hyperparameters than benchmarks/zoo.py gives now (2026-11-04)
        hp = _rl()._hp_differences(it.algo, it.task, r.get("info"), int(it.job["steps"]))
        if hp:
            return "other settings", "hyperparameters: " + "; ".join(hp)
    vers = r.get("machine", {}).get("versions", {})
    code = [f"torch {vers.get('torch')} (now {live['torch']})"] if vers.get("torch") != live["torch"] else []
    if it.algo == "TEMPO":
        if r.get("tempo_source") != live["tempo_source"]:
            code.append(f"tempo_rl {r.get('tempo_source')} (now {live['tempo_source']})")
    else:
        for k, name in (("stable_baselines3", "sb3"), ("sb3_contrib", "sb3_contrib"), ("gymnasium", "gymnasium")):
            if vers.get(k) != live[name]:
                code.append(f"{k} {vers.get(k)} (now {live[name]})")
    if code:
        return "older code", "; ".join(code)
    return "done", ""


def measurements(cfg: Config) -> list[Item]:
    """The planned measurements: per task, algorithm / variant and repeat, the job of the test protocol built by
    ``runlab.make_lab`` / ``make_jobs`` (so devices, seeds, env_kwargs and budget are those of Baselines_test / TEMPO_test),
    with suite ``timing``. Each item's ``state``: done / to run / older code / other settings / unreadable."""
    rl = _rl()
    report = _report()
    items, notes = [], []
    live = _live_code()
    with _real_protocol_env(), tempfile.TemporaryDirectory(prefix="timelab_lab_") as tmp:
        for task in cfg.tasks:
            info = rl.bl.task_info(task)
            limit = info["time_limit"]                  # in agent steps (the frames of a frame-skipped pixel task / FRAME_SKIP)
            limit = int(limit) // FRAME_SKIP[task] if limit and task in FRAME_SKIP else limit
            if cfg.algos is None:
                base, tempo = list(info["baselines"]) or (list(rl.TOY_TEXT_ALGOS) if info.get("toy_text") else []), True
            else:
                from benchmarks import plans

                ok = DISCRETE_ALGOS if info["discrete"] else CONTINUOUS_ALGOS
                pixels = task in plans.IMAGE                # sb3-contrib CrossQ has no CNN policy (runlab refuses it)
                base = [a for a in BASELINES if a in cfg.algos and a in ok and not (pixels and a == "CrossQ")]
                skipped = [a for a in cfg.algos if a in BASELINES and a not in ok]
                if skipped:
                    notes.append(f"{task}: {', '.join(skipped)} cannot act here "
                                 f"({'discrete' if info['discrete'] else 'continuous'} actions): skipped")
                if pixels and "CrossQ" in cfg.algos and "CrossQ" in ok:
                    notes.append(f"{task}: CrossQ cannot run on pixel observations (sb3-contrib CrossQ has no CNN policy): "
                                 "skipped")
                tempo = "TEMPO" in cfg.algos
            jobs = []
            if base:
                lab = rl.make_lab("baselines", task, smoke=False, algos=tuple(base), out_dir=pathlib.Path(tmp) / task,
                                  show=False, export_env=False)
                jobs += rl.make_jobs(lab)
                notes += [f"{task}: {n}" for n in lab.notes]
            for v in (cfg.tempo_variants if tempo else ()):
                lab = rl.make_lab("tempo", task, smoke=False, variant=v, out_dir=pathlib.Path(tmp) / f"{task}_{v}",
                                  show=False, export_env=False)
                jobs += rl.make_jobs(lab)
                notes += [f"{task} TEMPO[{v}]: {n}" for n in lab.notes]
            by_key = {}
            for j in jobs:
                by_key.setdefault((j["algo"], j["variant"]), {})[int(j["seed"])] = j
            for (algo, variant), seeds in by_key.items():
                ls = cyc = None
                for rep in range(1, cfg.repeats + 1):
                    seed = cfg.seeds[(rep - 1) % len(cfg.seeds)]
                    job = dict(seeds[seed], suite=cfg.suite)
                    if ls is None:
                        ls, cyc = _schedule(job)
                    win = _window(ls, cyc, cfg.settle_steps, cfg.train_steps, _tempo_bonus(job))
                    b = win[-1]                                 # end of the last timed window
                    rate = rl.bl.job_rate(job)                  # runlab's planning rate (fully loaded machine)
                    est = (rl.bl.JOB_OVERHEAD_S + ls / WARMUP_RATE_GUESS + (b - ls) / rate
                           + int(job["n_eval"]) * (limit or EPISODE_STEPS_GUESS.get(task, 1000)) * EVAL_STEP_GUESS_S)
                    d = cfg.out / task / algo / variant
                    it = Item(task=task, algo=algo, variant=variant, label=report.label_of(algo, variant), repeat=rep,
                              job=job, path=d / f"rep{rep}.json",
                              log=cfg.out / "logs" / f"{task}__{algo}__{variant}__rep{rep}.log",
                              time_limit=limit, learning_starts=ls, cycle=cyc, window=win, est_s=est)
                    it.state, it.reason = _state(cfg, it, live)
                    items.append(it)
    for n in dict.fromkeys(notes):
        print("note: " + n)
    items.sort(key=lambda i: (rl.tasks().index(i.task), report.label_key(i.label), i.repeat))
    return items


def _fmt_dur(s: float) -> str:
    s = float(s)
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{int(s // 3600)} h {int(s % 3600 // 60):02d} min"


def _display(df) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import display

        if get_ipython() is not None:
            import pandas as pd

            with pd.option_context("display.max_rows", 500, "display.max_columns", 50, "display.width", 250):
                display(df)
            return
    except ImportError:
        pass
    print(df.to_string())


def plan(cfg: Config, show: bool = True) -> list[Item]:
    """The measurements with their state and a rough duration estimate; prints one row per measurement and the total."""
    import pandas as pd

    items = measurements(cfg)
    rows = [dict(task=i.task, algorithm=i.label, repeat=i.repeat, seed=i.job["seed"], device=i.job["device"],
                 **{"learning starts": i.learning_starts, "update cycle": i.cycle,
                    "timed window": f"{i.window[0]:,}-{i.window[1]:,}"
                                    + (f" + {i.window[1]:,}-{i.window[2]:,} (bonus 0)" if len(i.window) > 2 else ""),
                    "est. min": round(i.est_s / 60, 1)},
                 state=i.state + (f" ({i.reason})" if i.reason else "")) for i in items]
    todo = [i for i in items if i.state != "done"]
    if show:
        if rows:
            _display(pd.DataFrame(rows))
        print(f"{len(items)} measurements: {len(items) - len(todo)} done, {len(todo)} to run"
              + (f" ({sum(i.state != 'to run' for i in todo)} of them re-timed: older code / other settings / unreadable)"
                 if any(i.state != "to run" for i in todo) else ""))
        print(f"estimated duration: {_fmt_dur(sum(i.est_s for i in todo))} for the measurements to run "
              f"({_fmt_dur(sum(i.est_s for i in items))} for all {len(items)}). A rough UPPER estimate: runlab's planning "
              "rates describe a fully loaded machine (a measurement runs alone and is usually faster), warm-up steps at "
              f"{WARMUP_RATE_GUESS:,.0f}/s, evaluation steps at {EVAL_STEP_GUESS_S * 1e3:g} ms.")
        print("learning starts / update cycle: the settings the plan expects (the measurement reads them from the model); "
              "timed window = env steps (after learning starts + settle, whole cycles); TEMPO with its novelty bonus: "
              "+ a second window of the same length with the bonus at 0, its value after the anneal.")
    return items


# --------------------------------------------------------------------------- running
def busy_processes() -> list | None:
    """(pid, command) of running benchmark jobs / schedulers / other timings (Python processes whose command line holds
    one of :data:`BUSY_MARKS`; a shell or editor that merely shows the text does not count); None when the process
    table is unreadable."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=15).stdout
    except Exception:                                   # noqa: BLE001
        return None
    me, found = os.getpid(), []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        cmd = cmd.strip()
        exe = os.path.basename(cmd.split(" ", 1)[0]).lower()
        if pid.isdigit() and int(pid) != me and "python" in exe and any(m in cmd for m in BUSY_MARKS):
            found.append((int(pid), cmd))
    return found


def guard() -> tuple:
    """Refuse (RuntimeError) while benchmark processes or another timing run; else return the load averages."""
    procs = busy_processes()
    if procs is None:
        print("WARNING: cannot read the process table; cannot check that no benchmark runs alongside")
    elif procs:
        lines = "\n".join(f"  pid {p}: {c[:160]}" for p, c in procs[:12])
        raise RuntimeError(
            f"REFUSED: {len(procs)} benchmark / timing process(es) are running. A measurement next to them would time the "
            "machine's load, not the algorithm. Wait until they have finished (or stop them), then run this cell again: "
            f"finished measurements are kept and skipped.\n{lines}")
    return os.getloadavg()


def _kill(p: subprocess.Popen) -> None:
    if p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(15)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def _tail(path: pathlib.Path, n: int = 12) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return "(no log)"


def run_measurement(cfg: Config, it: Item, test_eval_freq: int | None = None, timeout=None) -> dict:
    """Run one measurement in a fresh subprocess (``python timelab.py _measure <spec>``, cwd = project root,
    ``OMP_NUM_THREADS`` = the job's threads as the scheduler sets it); returns the result. Raises on failure or timeout;
    a KeyboardInterrupt stops the subprocess first. The result file is written atomically by the subprocess."""
    spec = it.spec(cfg, test_eval_freq)
    it.log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": str(it.job.get("threads", 1)), "PYTHONUNBUFFERED": "1",
           "PYTHONDONTWRITEBYTECODE": "1",
           "TEMPO_BENCH_RESULTS": str(pathlib.Path(tempfile.gettempdir()) / "timelab_unused_results")}
    cmd = [sys.executable, str(HERE / "timelab.py"), "_measure", json.dumps(spec)]
    timeout = cfg.timeout if timeout is None else timeout
    with open(it.log, "w") as fh:
        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill(p)
            it.path.with_suffix(".tmp").unlink(missing_ok=True)
            raise TimeoutError(f"no result after {timeout / 60:.0f} min (TIMEOUT_MIN); stopped. Log: {it.log}") from None
        except BaseException:
            _kill(p)
            it.path.with_suffix(".tmp").unlink(missing_ok=True)
            raise
    if rc != 0 or not it.path.exists():
        raise RuntimeError(f"failed (exit code {rc}); last lines of {it.log}:\n{_tail(it.log)}")
    return json.loads(it.path.read_text())


def _move_outdated(cfg: Config, items: list[Item]) -> list:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    moved = []
    for it in items:
        if it.path.exists():
            dest = cfg.out / "_outdated" / stamp / it.path.relative_to(cfg.out)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(it.path), str(dest))
            moved.append(dest)
    if moved:
        print(f"moved {len(moved)} earlier measurement(s) of other code / settings to {cfg.out / '_outdated' / stamp} "
              "(never deleted); they are timed again")
    return moved


def run(cfg: Config, items: list[Item] | None = None, force: bool = False) -> dict:
    """Time every measurement that is not done, strictly one at a time (a fresh subprocess each), with :func:`guard`
    before each one. Progress: one line per measurement. Ctrl-C / Kernel > Interrupt stops the running measurement
    cleanly (no partial file); run again to continue. A failed measurement is reported and the next one starts.
    ``force``: time the given items again even when done (their files are moved to ``_outdated/``, never deleted)."""
    items = measurements(cfg) if items is None else items
    live = _live_code()
    for it in items:                                    # the plan may be old: re-check the files
        if it.path.exists() or it.state == "done":
            it.state, it.reason = _state(cfg, it, live)
        if force and it.state == "done":
            it.state, it.reason = "other settings", "forced"
    if force:
        cfg.rerun_outdated = True
    redo = [i for i in items if i.state in ("older code", "other settings", "unreadable")]
    if redo and cfg.rerun_outdated:
        _move_outdated(cfg, redo)
    todo = [i for i in items if i.state == "to run" or (cfg.rerun_outdated and i in redo)]
    kept = [i for i in redo if not cfg.rerun_outdated]
    print(f"{len(items)} measurements: {len(items) - len(todo) - len(kept)} done, {len(todo)} to run"
          + (f", {len(kept)} of other code / settings kept (RERUN_OUTDATED = False)" if kept else ""))
    if not todo:
        return dict(done=0, failed=[], interrupted=False)
    done, failed, t_all = 0, [], _now()
    try:
        for k, it in enumerate(todo, 1):
            load = guard()
            head = f"[{k}/{len(todo)}] {it.task} {it.label} rep{it.repeat} on {it.job['device']}"
            warn = f"  WARNING: machine busy (load {load[0]:.1f})" if load[0] > LOAD_WARN else ""
            t0 = _now()
            try:
                r = run_measurement(cfg, it)
            except (RuntimeError, TimeoutError) as e:
                failed.append(it)
                print(f"{head}: FAILED after {_fmt_dur(_now() - t0)} (load {load[0]:.1f}) - {e}", flush=True)
                continue
            m, d = r["measured"], r["derived"]
            it.state = "done"
            done += 1
            off = (f" ({m['train_rate_after_bonus']:,.1f} after the bonus anneal)"
                   if m.get("train_rate_after_bonus") else "")
            print(f"{head}: {m['train_rate']:,.1f} env-steps/s{off}, setup {m['setup_s']:.1f} s, one evaluation "
                  f"{m['eval_s']:.1f} s -> per run: train {d['train_h']:.2f} h + eval {d['eval_h']:.2f} h "
                  f"[{_fmt_dur(_now() - t0)}; load {load[0]:.1f}]{warn}", flush=True)
    except KeyboardInterrupt:
        print(f"INTERRUPTED: {done} measurement(s) finished in this call and are kept; the running one was stopped and "
              "left no file. Run this cell again to continue.")
        return dict(done=done, failed=failed, interrupted=True)
    print(f"finished {done} measurement(s) in {_fmt_dur(_now() - t_all)}"
          + (f"; FAILED {len(failed)}: " + ", ".join(f"{i.task} {i.label} rep{i.repeat}" for i in failed)
             + " (logs in " + str(cfg.out / "logs") + "; they are tried again at the next run)" if failed else ""))
    return dict(done=done, failed=failed, interrupted=False)


# --------------------------------------------------------------------------- results
@dataclass
class Results:
    """Loaded measurements (``df``: one row per measurement; ``summary``: median over repeats per task and algorithm),
    the in-benchmark wall-clock (``bench``) and the tables (:data:`TABLES`)."""
    cfg: Config
    df: object
    summary: object
    bench: object
    tables: dict
    notes: list

    def show(self, name: str) -> None:
        title, df, note = self.tables[name]
        print(title)
        if df is None or len(df) == 0:
            print("  pending: no finished measurement yet" if name != "benchmark" else
                  "  none: no finished run at the test protocol in Baselines_test / TEMPO_test yet")
        else:
            _display(df)
        if note:
            print(note)


def _load_rows(cfg: Config, items: list[Item]) -> tuple[list, list]:
    rows, notes = [], []
    for it in items:
        if it.state != "done":
            if it.state != "to run":
                notes.append(f"{it.task} {it.label} rep{it.repeat}: {it.state} ({it.reason}) - left out")
            continue
        r = json.loads(it.path.read_text())
        m = r["measured"]
        d = derive(m, r["protocol"], r.get("time_limit"))      # recomputed: one formula for every file
        rows.append(dict(task=it.task, label=it.label, algo=it.algo, variant=it.variant, repeat=it.repeat,
                         seed=r["seed"], device=r["device"], setup_s=m["setup_s"], learning_starts=m["learning_starts"],
                         cycle=m["cycle"], warmup_rate=m["warmup_rate"], train_rate=m["train_rate"],
                         train_rate_after_bonus=m.get("train_rate_after_bonus"), bonus_end_step=m.get("bonus_end_step"),
                         run_rate=d["run_rate"],
                         window_steps=m["window_steps"], window_s=m["window_s"], window_cpu_s=m["window_cpu_s"],
                         window_updates=m["window_updates"], evaluations_in_training=m["evaluations_in_training"],
                         eval_s=m["eval_s"], eval_steps=m["eval_steps"], eval_s_per_step=m["eval_s_per_step"],
                         steps=d["steps"], n_evals=d["n_evals"], train_h=d["train_h"], eval_h=d["eval_h"],
                         total_h=d["total_h"], eval_h_at_limit=d["eval_h_at_limit"],
                         peak_rss_mb=r["machine"].get("peak_rss_mb"), load_before=r["load_before"][0],
                         chip=r["machine"].get("chip"), finished=r["finished"], file=str(it.path)))
    return rows, notes


def benchmark_wallclock(cfg: Config):
    """Median ``wall_s`` (hours) over the finished runs at the test protocol in ``Baselines_test`` / ``TEMPO_test``
    (read only): includes the evaluations and whatever else ran on the machine at the time."""
    import numpy as np
    import pandas as pd

    report = _report()
    rows = []
    for kind, (folder, suite) in BENCH_FOLDERS.items():
        base = cfg.bench_root / folder / "results" / suite
        for task in cfg.tasks:
            for f in sorted((base / task).glob("*/*/seed*.json")):
                algo, variant = f.parent.parent.name, f.parent.name
                if (algo == "TEMPO") != (kind == "tempo"):
                    continue
                try:
                    r = json.loads(f.read_text())
                    job, wall = r["job"], float(r["wall_s"])
                except Exception:                       # noqa: BLE001 - unreadable: complab reports those
                    continue
                if (int(job.get("steps", -1)), int(job.get("eval_freq", -1)), int(job.get("n_eval", -1))) != \
                        (cfg.budget, cfg.eval_freq, cfg.n_eval_of(task)) or not np.isfinite(wall):
                    continue
                rows.append(dict(task=task, label=report.label_of(algo, variant), seed=int(job["seed"]),
                                 device=job.get("device", "cpu"), wall_h=wall / 3600.0))
    if not rows:
        return pd.DataFrame(columns=["task", "label", "n", "median_h", "device"])
    df = pd.DataFrame(rows)
    g = df.groupby(["task", "label"])
    return g.agg(n=("wall_h", "size"), median_h=("wall_h", "median"), min_h=("wall_h", "min"), max_h=("wall_h", "max"),
                 device=("device", lambda s: "+".join(sorted(set(s))))).reset_index()


def _h(v) -> str:
    return "–" if v is None or not math.isfinite(v) else (f"{v:.2f}" if v >= 0.1 else f"{v:.2g}")


def _rate(v) -> str:
    return f"{v:,.0f}" if v >= 100 else f"{v:.3g}"


def _matrix(cfg: Config, items: list[Item], S, value, fmt, dev_mark: bool = True, suffix=None):
    """Task x algorithm table of strings: ``fmt(value(row))`` (+ ``suffix(row)``, + " (mps)" for Apple-GPU runs),
    'pending' when planned but not measured, '–' when the algorithm does not run on the task."""
    import pandas as pd

    report = _report()
    labels = sorted({i.label for i in items}, key=report.label_key)
    planned = {(i.task, i.label) for i in items}
    have = {(r.task, r.label): r for r in S.itertuples()} if len(S) else {}
    out = []
    for task in cfg.tasks:
        row = {"Task": task}
        for lab in labels:
            if (task, lab) in have:
                r = have[(task, lab)]
                row[lab] = (fmt(value(r)) + (suffix(r) if suffix else "")
                            + (f" ({r.device})" if dev_mark and r.device != "cpu" else ""))
            else:
                row[lab] = "pending" if (task, lab) in planned else "–"
        out.append(row)
    return pd.DataFrame(out, columns=["Task", *labels])


def results(cfg: Config, items: list[Item] | None = None) -> Results:
    """Load every finished measurement of the plan and build the tables: ``train_h`` (1), ``rate`` (2), ``eval_h`` (3),
    ``speedup`` (4), ``benchmark`` (5). Values are medians over repeats."""
    import numpy as np
    import pandas as pd

    report = _report()
    items = measurements(cfg) if items is None else items
    live = _live_code()
    for it in items:
        if it.path.exists():
            it.state, it.reason = _state(cfg, it, live)
        elif it.state == "done":                        # moved away since the plan
            it.state, it.reason = "to run", ""
    rows, notes = _load_rows(cfg, items)
    df = pd.DataFrame(rows)
    if len(df):
        for col in ("warmup_rate", "train_rate_after_bonus", "bonus_end_step", "run_rate", "eval_s_per_step",
                    "eval_h_at_limit"):                 # optional numbers (None -> NaN, also when all are None)
            df[col] = pd.to_numeric(df[col], errors="coerce")
        S = df.groupby(["task", "label"]).agg(
            algo=("algo", "first"), variant=("variant", "first"), device=("device", "first"), repeats=("repeat", "size"),
            setup_s=("setup_s", "median"), learning_starts=("learning_starts", "first"), cycle=("cycle", "first"),
            warmup_rate=("warmup_rate", "median"), train_rate=("train_rate", "median"),
            train_rate_after_bonus=("train_rate_after_bonus", "median"), bonus_end_step=("bonus_end_step", "first"),
            run_rate=("run_rate", "median"), eval_s=("eval_s", "median"),
            eval_s_per_step=("eval_s_per_step", "median"), steps=("steps", "first"), n_evals=("n_evals", "first"),
            train_h=("train_h", "median"), train_h_min=("train_h", "min"), train_h_max=("train_h", "max"),
            eval_h=("eval_h", "median"), eval_h_at_limit=("eval_h_at_limit", "median"),
            total_h=("total_h", "median")).reset_index()
    else:
        S = pd.DataFrame(columns=["task", "label", "device", "train_h", "train_rate", "run_rate", "eval_h", "total_h"])
    bench = benchmark_wallclock(cfg)
    budget_k = f"{cfg.budget / 1e3:g}k"
    T = {}
    reps = f"median over {cfg.repeats} repeats; " if cfg.repeats > 1 else ""
    T["train_h"] = (f"(1) {TABLES['train_h']}: one {budget_k}-step run, without evaluations",
                    _matrix(cfg, items, S, lambda r: r.train_h, _h) if len(S) else None,
                    f"Setup + warm-up (steps before learning starts) + training at the measured rate (TEMPO: the rate "
                    "with its novelty bonus on until the bonus anneal ends, half the run, then the rate with the bonus off); "
                    f"{reps}"
                    "(mps) = Apple GPU, (cuda) = NVIDIA GPU, else one CPU thread; – = not run on this task; TEMPO = library default (fast recipe), "
                    "TEMPO[c1] = previous default.")

    def two_rates(r):                                   # TEMPO with its bonus: (bonus on / bonus off)
        off = r.train_rate_after_bonus
        return f" ({_rate(r.train_rate)} / {_rate(off)})" if off is not None and np.isfinite(off) else ""

    T["rate"] = (f"(2) {TABLES['rate']} during training (after learning starts)",
                 _matrix(cfg, items, S, lambda r: r.run_rate, _rate, suffix=two_rates) if len(S) else None,
                 "Env steps per second from learning start to the end of the run (= table 1's training part), measured "
                 "over timed windows of whole update cycles (GPU synchronised at both ends). TEMPO (x / y): its novelty "
                 "bonus is on until its anneal ends (half the run for TEMPO and TEMPO[c1]; rate x) and off after it "
                 "(rate y, timed in a second window with the bonus at 0); the first number is their average over the run.")

    def at_limit(r):                                    # the bound when it differs by more than 10%
        lim = r.eval_h_at_limit
        ok = lim is not None and np.isfinite(lim) and lim > 1.1 * r.eval_h
        return f" (max {_h(lim)})" if ok else ""

    tab = _matrix(cfg, items, S, lambda r: r.eval_h, _h, suffix=at_limit) if len(S) else None
    T["eval_h"] = (f"(3) {TABLES['eval_h']}: {cfg.budget // cfg.eval_freq} evaluations x {cfg.n_eval_text()}",
                   tab, "One evaluation timed with the policy of the timed window x the number of evaluations; (max x) = "
                   "if every episode ran to the time limit (a trained policy's episodes can be longer than this early "
                   "policy's).")
    sp_rows = []
    if len(S):
        th = {(r.task, r.label): r.train_h for r in S.itertuples()}
        tempo_labels = sorted({l for l in S.label if report.is_tempo(l)}, key=report.label_key)
        bases = [b for b in BASELINES if b in set(S.label)]
        for tl_ in tempo_labels:
            row = {"Algorithm": tl_}
            for b in bases:
                ratios = [th[(t, b)] / th[(t, tl_)] for t in cfg.tasks if (t, b) in th and (t, tl_) in th]
                row[b] = (f"{np.median(ratios):.2f}x ({len(ratios)} task{'s' if len(ratios) != 1 else ''})"
                          if ratios else "–")
            sp_rows.append(row)
    T["speedup"] = (f"(4) {TABLES['speedup']}: median over tasks of baseline hours / TEMPO hours",
                    pd.DataFrame(sp_rows) if sp_rows else None,
                    "> 1x: TEMPO trains faster than that baseline; < 1x: slower. Training time per run (table 1), on the "
                    "tasks where both were measured.")
    if len(bench):
        bt = bench
        labels = sorted(set(bt.label), key=report.label_key)
        brow = []
        for task in cfg.tasks:
            row = {"Task": task}
            for lab in labels:
                sel = bt[(bt.task == task) & (bt.label == lab)]
                if len(sel):
                    r = sel.iloc[0]
                    row[lab] = f"{_h(r.median_h)} (n={r.n})" + (f" ({r.device})" if r.device != "cpu" else "")
                else:
                    row[lab] = "–"
            if any(v != "–" for k, v in row.items() if k != "Task"):
                brow.append(row)
        btab = pd.DataFrame(brow, columns=["Task", *labels])
    else:
        btab = None
    T["benchmark"] = (f"(5) {TABLES['benchmark']}",
                      btab, f"Median wall_s over the finished seeds (n) of Baselines_test / TEMPO_test at the test protocol "
                            f"({cfg.budget:,} steps, evaluation every {cfg.eval_freq:,}): includes evaluations and the load "
                            "of the other runs that shared the machine, so it is NOT comparable across algorithms; shown "
                            "for reference only.")
    for n in notes:
        print("note: " + n)
    if len(df) and (df["evaluations_in_training"] > 0).any():
        raise RuntimeError("a measurement recorded evaluations during training: it is invalid")
    return Results(cfg=cfg, df=df, summary=S, bench=bench, tables=T, notes=notes)


# --------------------------------------------------------------------------- figure and files
def _fs():
    nb = ROOT / "notebooks"
    if str(nb) not in sys.path:
        sys.path.insert(0, str(nb))
    import figstyle

    return figstyle


def plot(cfg: Config, res: Results, show: bool = True) -> list:
    """Dot plot: one row per task, training hours per run (log scale) per algorithm (report.STYLE colours, one marker
    shape per algorithm, TEMPO variants in figstyle's variant colours). Saved as PDF + PNG when drawn; returns the paths."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    S = res.summary
    if not len(S):
        print("pending: no finished measurement yet (nothing drawn)")
        return []
    fs, report = _fs(), _report()
    tasks = [t for t in cfg.tasks if t in set(S.task)]
    labels = sorted(set(S.label), key=report.label_key)
    tempo_labels = [l for l in labels if report.is_tempo(l)]
    offs = [0.0] if len(labels) == 1 else [(-0.3 + 0.6 * k / (len(labels) - 1)) for k in range(len(labels))]
    with fs.paper():
        fig, ax = fs.subplots(width=fs.DOUBLE, height=0.24 * len(tasks) + 0.75)
        handles = []
        for k, lab in enumerate(labels):
            st = fs.tempo_style(lab, "TEMPO", tempo_labels)
            sel = S[S.label == lab]
            ys = [tasks.index(t) + offs[k] for t in sel.task]
            mk = MARKERS.get(lab.split("[")[0], "o")
            size = 26 if lab == "TEMPO" else 18
            ax.scatter(sel.train_h, ys, s=size, marker=mk, color=st["color"], zorder=3 + st.get("z", 2) / 10,
                       linewidths=0)
            if (sel.train_h_max > sel.train_h_min * 1.0001).any():        # repeats: their range as a thin bar
                for y, lo, hi in zip(ys, sel.train_h_min, sel.train_h_max):
                    ax.plot([lo, hi], [y, y], color=st["color"], lw=fs.ERRORBAR_WIDTH, zorder=2)
            handles.append(Line2D([], [], ls="none", marker=mk, ms=4.5 if lab == "TEMPO" else 4, color=st["color"],
                                  label=lab))
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(tasks)
        ax.set_ylim(len(tasks) - 0.5, -0.5)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x:g}"))   # 0.1, 1, 10 instead of 10^-1, ...
        ax.grid(True, axis="x", which="major")
        ax.set_xlabel(f"Training time per {cfg.budget / 1e3:g}k-step run (h, log scale)")
        fs.legend(fig, handles)
        paths = fs.save(fig, cfg.figures / "training_time")
    plt.close(fig)
    if show:
        fs.show(paths[1] if len(paths) > 1 else paths[0])
        print(fs.caption(
            "Figure. Training time of one run at the test protocol (setup + warm-up + training, no evaluations), each "
            "algorithm timed alone on this machine with the benchmark's device (one CPU thread, or the GPU for "
            "CrossQ and TEMPO); one marker per task and algorithm"
            + (", bars = range over repeats" if cfg.repeats > 1 else "") + "."
            + (" SMOKE: tiny timed windows, the numbers mean nothing." if cfg.smoke else "")))
    print("saved: " + ", ".join(str(p) for p in paths))
    return paths


def _md_table(df) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def _tex_escape(s: str) -> str:
    return (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("#", r"\#").replace("–", "--"))


def save(cfg: Config, res: Results) -> list:
    """Write ``training_time.md`` (all tables), ``.csv`` (one row per task and algorithm: every measured and derived
    number, plus the in-benchmark wall-clock) and ``.tex`` (table 1, booktabs) to ``cfg.tables``. Returns the paths."""
    import pandas as pd

    if not len(res.summary):
        print("nothing to save yet (no finished measurement)")
        return []
    cfg.tables.mkdir(parents=True, exist_ok=True)
    m0 = json.loads(pathlib.Path(res.df["file"].iloc[0]).read_text())["machine"]
    machine = (f"{m0.get('chip')}, {m0.get('cpu_cores')} cores, {m0.get('memory_gb')} GB; torch "
               f"{m0['versions'].get('torch')}, SB3 {m0['versions'].get('stable_baselines3')}, gymnasium "
               f"{m0['versions'].get('gymnasium')}")
    md = [f"# Training time per algorithm{' (SMOKE: the numbers mean nothing)' if cfg.smoke else ''}", "",
          f"Machine: {machine}. Protocol: {cfg.budget:,} steps per run, an evaluation every {cfg.eval_freq:,} steps "
          f"({cfg.n_eval_text()}). Timed window: {cfg.train_steps:,} env steps after learning starts + "
          f"{cfg.settle_steps:,}; {cfg.repeats} repeat(s). Generated by Comparison/timelab.py on "
          f"{time.strftime('%Y-%m-%d %H:%M')}.", ""]
    for name, (title, df, note) in res.tables.items():
        md += [f"## {title}", ""]
        md += [_md_table(df) if df is not None and len(df) else "(none yet)", ""]
        if note:
            md += [note, ""]
    paths = []
    p = cfg.tables / "training_time.md"
    p.write_text("\n".join(md))
    paths.append(p)
    S = res.summary.copy()
    if len(res.bench):
        S = S.merge(res.bench.rename(columns={"n": "benchmark_runs", "median_h": "benchmark_wall_h_median",
                                              "min_h": "benchmark_wall_h_min", "max_h": "benchmark_wall_h_max",
                                              "device": "benchmark_device"}), on=["task", "label"], how="left")
    p = cfg.tables / "training_time.csv"
    S.to_csv(p, index=False, float_format="%.6g")
    paths.append(p)
    title, df, note = res.tables["train_h"]
    cols = list(df.columns)
    tex = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{Training time per " + f"{cfg.budget / 1e3:g}k" + r"-step run (hours, without evaluations), each "
           r"algorithm timed alone on " + _tex_escape((m0.get("chip") or "this machine") + (f" with {m0['gpu']}" if m0.get("gpu") else ""))
           + r". (mps) = Apple GPU, (cuda) = NVIDIA GPU, otherwise one CPU thread.}", r"\label{tab:training-time}",
           r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\toprule",
           " & ".join(_tex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for r in df.itertuples(index=False):
        tex.append(" & ".join(_tex_escape(v) for v in r) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    p = cfg.tables / "training_time.tex"
    p.write_text("\n".join(tex))
    paths.append(p)
    figs = sorted(cfg.figures.glob("training_time.*")) if cfg.figures.is_dir() else []
    print("wrote: " + ", ".join(str(x) for x in paths) + ("; figure: " + ", ".join(str(x) for x in figs) if figs else ""))
    return paths


def show_machine() -> dict:
    """Print the machine and package versions the measurements will record (no torch import in the kernel)."""
    info = machine_info()
    v = info["versions"]
    print(f"machine    : {info['chip']}, {info['cpu_cores']} cores ({info['performance_cores']} performance + "
          f"{info['efficiency_cores']} efficiency), {info['memory_gb']} GB; {info['os']}")
    print("packages   : " + ", ".join(f"{k} {val}" for k, val in v.items()))
    load = os.getloadavg()
    procs = busy_processes()
    print(f"load       : {load[0]:.2f} / {load[1]:.2f} / {load[2]:.2f} (1 / 5 / 15 min); "
          + ("benchmark processes running: " + str(len(procs)) + " (the Run cell will refuse while they run)" if procs
             else "no benchmark process running"))
    return info


# =========================================================================== CLI
def _cli(argv: list[str]) -> int:
    import argparse

    if argv and argv[0] == "_measure":
        _child_main(argv[1])
        return 0
    ap = argparse.ArgumentParser(prog="timelab.py", description="Time one (task, algorithm, variant, repeat) in a fresh "
                                 "subprocess; result JSON under Comparison/timing/ (smoke: the scratch folder).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("time", help="time one measurement")
    t.add_argument("task")
    t.add_argument("algo", choices=[*BASELINES, "TEMPO"])
    t.add_argument("--variant", default="default", help="TEMPO variant (benchmarks/variants.py); baselines: default")
    t.add_argument("--repeat", type=int, default=1)
    t.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    t.add_argument("--settle-steps", type=int, default=SETTLE_STEPS)
    t.add_argument("--smoke", action="store_true", help="tiny window, scratch folder")
    t.add_argument("--force", action="store_true", help="time again even if the result exists (the old one is moved)")
    t.add_argument("--timeout-min", type=float, default=None)
    a = ap.parse_args(argv)
    if a.algo != "TEMPO" and a.variant != "default":
        ap.error("the baselines have no variants")
    cfg = make_config(tasks=[a.task], algos=[a.algo], tempo_variants=(a.variant,), train_steps=a.train_steps,
                      settle_steps=a.settle_steps, repeats=a.repeat, smoke=a.smoke, timeout_min=a.timeout_min,
                      smoke_tasks=False)
    items = [i for i in measurements(cfg) if i.repeat == a.repeat]
    if not items:
        print(f"{a.algo} does not run on {a.task}")
        return 2
    out = run(cfg, items, force=a.force)
    for i in items:
        if i.path.exists():
            print(f"result: {i.path}")
    return 1 if out["failed"] or out["interrupted"] else 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
