"""Hyperparameter search for the baselines that have no RL-Zoo entry: PPO and DQN on the toy-text tasks.

Since 2026-11-04 every baseline of the comparison is tuned. ``benchmarks/zoo.py`` gives every other baseline
an RL-Zoo setting; the zoo has no FrozenLake, CliffWalking, Blackjack or Taxi entry, so PPO and DQN are tuned here, with
the benchmark's own protocol and code, on development seeds that the evaluation never uses:

* candidates: the 2 x 2 x 2 grid :data:`FACTORS` around the library defaults; the first value of every factor is the
  default, so the first candidate IS the untuned baseline and tuning cannot select anything worse on the development seeds;
* every candidate is an ordinary ``benchmarks.run_one`` job of the test protocol (``runlab.TEST_BUDGET`` = 500,000 steps,
  an evaluation every ``runlab.TEST_EVAL_FREQ`` = 10,000 steps, ``runlab.n_eval_for`` episodes, ``plans.ENV_KWARGS``, one CPU
  thread) with seeds :data:`DEV_SEEDS` = (0, 1) instead of ``runlab.TEST_SEEDS`` (10-14); ``benchmarks/tune_one.py``
  hands ``run_one`` the candidate's hyperparameters instead of ``benchmarks/zoo.py``'s;
* selection (:func:`select`): the highest mean final score over the development seeds (final = mean of the last three
  evaluations at or before the budget, ``benchlab.final_of``, as everywhere in the comparison); ties go to the higher mean
  area under the curve, then to the earlier grid entry;
* ``select(write=True)`` writes the winners to ``benchmarks/tuned_toy_text.json``. ``benchmarks/zoo.py`` reads that file, so
  the Baselines_test notebooks of these tasks see their old runs as 'other protocol' and run them again with the tuned
  values at their next launch (moved to ``results/_outdated/``, never deleted).

Files: ``Baselines_test/tuning/results/<suite>/<env>/<algo>/<candidate>/seed<k>.json`` and ``Baselines_test/tuning/logs/``
(one log per job, the pool's plan / status / log). Nothing else is written, apart from ``benchmarks/tuned_toy_text.json`` by
``select(write=True)``; smoke mode writes to a temporary folder only. ``Baselines_test/tune_toy_text.ipynb`` drives this
module; the protocol is pre-registered in ``benchmarks/PROTOCOL.md`` (addendum 2026-11-04).
"""
from __future__ import annotations

import itertools
import json
import math
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks import plans  # noqa: E402

PY = sys.executable
FOLDER = ROOT / "Baselines_test" / "tuning"
SUITE = "toy_text_tuning"
TUNED_JSON = ROOT / "benchmarks" / "tuned_toy_text.json"
TASKS = tuple(plans.TOY_TEXT_ALL)
ALGOS = ("PPO", "DQN")
DEV_SEEDS = (0, 1)
#: smoke mode: a few thousand steps, a temporary folder (checks the machinery, never a result)
SMOKE_STEPS, SMOKE_EVAL_FREQ, SMOKE_N_EVAL = 3_000, 1_000, 2
#: processes that compete for the CPU: a search does not start while one runs (and Comparison/timelab.py waits for a search)
BUSY_MARKS = ("benchmarks.run_one", "benchmarks.scheduler", "timelab.py _measure", "benchmarks.tune_one", "tunelab.py _pool")

#: Factors of each grid: (hyperparameter, values); the FIRST value of every factor is the Stable-Baselines3 2.9.0 default.
#: PPO's "rollout" keeps 2048 samples per update: one environment x 2048 steps (default) or eight x 256.
FACTORS = {
    "DQN": [("learning_rate", (1e-4, 1e-3)),
            ("exploration_fraction", (0.1, 0.4)),
            ("target_update_interval", (10_000, 500))],
    "PPO": [("learning_rate", (3e-4, 1e-3)),
            ("ent_coef", (0.0, 0.01)),
            ("rollout", ({"n_envs": 1, "n_steps": 2048}, {"n_envs": 8, "n_steps": 256}))],
}
_SHORT = {"learning_rate": "lr", "exploration_fraction": "xf", "target_update_interval": "tu", "ent_coef": "ent"}


def grid(algo: str) -> list[tuple[str, dict]]:
    """``[(name, hp)]`` of ``algo``'s candidates, the library default first. ``hp`` holds only what differs from the default
    (so the first candidate is ``("default", {})``); names such as ``lr0.001_xf0.4_tu500`` or ``ent0.01_env8x256``."""
    out = []
    names, values = zip(*FACTORS[algo])
    for combo in itertools.product(*values):
        hp, parts = {}, []
        for name, v, default in zip(names, combo, (vals[0] for vals in values)):
            if v == default:
                continue
            if name == "rollout":
                hp.update(v)
                parts.append(f"env{v['n_envs']}x{v['n_steps']}")
            else:
                hp[name] = v
                parts.append(f"{_SHORT[name]}{v:g}")
        out.append(("_".join(parts) or "default", hp))
    return out


def show_grid() -> None:
    """Print the protocol and every candidate."""
    import runlab as rl
    print(f"tasks            : {', '.join(TASKS)}")
    print(f"protocol         : {rl.TEST_BUDGET:,} steps, an evaluation every {rl.TEST_EVAL_FREQ:,} steps "
          f"(runlab.n_eval_for episodes: Blackjack 1,000, else 10), one CPU thread, as the benchmark")
    print(f"development seeds: {list(DEV_SEEDS)} (evaluation seeds {list(rl.TEST_SEEDS)} are never used here)")
    print("selection        : highest mean final score over the development seeds (mean of the last three evaluations); "
          "ties: higher mean AUC, then the earlier candidate")
    for algo in ALGOS:
        print(f"\n{algo} ({len(grid(algo))} candidates; anything not listed is the Stable-Baselines3 2.9.0 default):")
        for name, hp in grid(algo):
            print(f"  {name:28s} {hp if hp else '{}  (library defaults = the untuned baseline)'}")
    n = len(TASKS) * sum(len(grid(a)) for a in ALGOS) * len(DEV_SEEDS)
    print(f"\n{n} runs in all ({len(TASKS)} tasks x {sum(len(grid(a)) for a in ALGOS)} candidates x {len(DEV_SEEDS)} seeds)")


def jobs(tasks=TASKS, algos=ALGOS, seeds=DEV_SEEDS, smoke: bool = False) -> list[dict]:
    """One ``benchmarks.run_one`` job per (task, algorithm, candidate, seed): the test protocol of ``runlab`` (smoke: tiny)
    with the candidate's hyperparameters in ``hp``; suite :data:`SUITE`, variant = candidate name."""
    import runlab as rl
    if set(seeds) & set(rl.TEST_SEEDS):
        raise ValueError(f"development seeds must not overlap the evaluation seeds {rl.TEST_SEEDS}")
    out = []
    for env in tasks:
        for algo in algos:
            for name, hp in grid(algo):
                for s in seeds:
                    out.append(dict(suite=SUITE, algo=algo, variant=name, env=env, seed=int(s),
                                    steps=SMOKE_STEPS if smoke else rl.TEST_BUDGET,
                                    eval_freq=SMOKE_EVAL_FREQ if smoke else rl.TEST_EVAL_FREQ,
                                    n_eval=SMOKE_N_EVAL if smoke else rl.n_eval_for(env),
                                    device="cpu", threads=1, cost=0, env_kwargs=plans.ENV_KWARGS.get(env), hp=hp))
    return out


def paths(smoke: bool = False) -> tuple[pathlib.Path, pathlib.Path]:
    """``(results, logs)`` folders: Baselines_test/tuning/ (smoke: a temporary folder)."""
    base = pathlib.Path(tempfile.gettempdir()) / "tunelab_smoke" if smoke else FOLDER
    return base / "results", base / "logs"


def result_path(job: dict, results: pathlib.Path) -> pathlib.Path:
    """Where ``benchmarks.run_one`` saves the job when ``TEMPO_BENCH_RESULTS`` = ``results`` (as ``run_one.result_path``)."""
    return results / job["suite"] / job["env"] / job["algo"] / job["variant"] / f"seed{job['seed']}.json"


def job_id(job: dict) -> str:
    return f"{job['env']}__{job['algo']}__{job['variant']}__s{job['seed']}"


def busy() -> list[tuple[int, str]]:
    """Running processes of :data:`BUSY_MARKS` (benchmark runs, timing measurements, a search), this process excluded."""
    ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    out = []
    for line in ps.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        exe = os.path.basename(cmd.split()[0]) if cmd.split() else ""
        if pid.isdigit() and int(pid) != os.getpid() and exe.startswith("python") and any(m in cmd for m in BUSY_MARKS):
            out.append((int(pid), cmd.strip()[:140]))
    return out


def _pool_pid(logs: pathlib.Path) -> int | None:
    """The pid of the search running for this logs folder, or None (a stale pid file is removed: the process is gone, a
    zombie, or another program that reuses the pid)."""
    f = logs / "pool.pid"
    try:
        pid = int(f.read_text())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return None
    out = subprocess.run(["ps", "-ww", "-o", "stat=,command=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    if out and not out.startswith("Z") and "tunelab.py _pool" in out and str(logs / "pool_plan.json") in out:
        return pid
    f.unlink(missing_ok=True)
    return None


def launch(workers: int = 12, tasks=TASKS, algos=ALGOS, seeds=DEV_SEEDS, smoke: bool = False, force: bool = False) -> None:
    """Start the search in the background (it keeps running if the notebook closes): at most ``workers`` runs at once, one
    CPU thread each; finished runs are skipped, so a second launch resumes. Refuses while a benchmark run, a timing
    measurement or another search is running (they would slow each other down), unless ``force``."""
    results, logs = paths(smoke)
    pid = _pool_pid(logs)
    if pid:
        print(f"A search is already running (pid {pid}); status() shows its progress, stop() ends it.")
        return
    others = busy()
    if others and not force:
        print("Not started: these processes would compete for the CPU (benchmark runs / timing measurements / a search):")
        for p, cmd in others[:8]:
            print(f"  pid {p}: {cmd}")
        print("Launch again when they have finished (or pass force=True).")
        return
    js = jobs(tasks, algos, seeds, smoke)
    todo = [j for j in js if not result_path(j, results).exists()]
    if not todo:
        print(f"All {len(js)} runs are finished: summary() / select().")
        return
    results.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    plan = logs / "pool_plan.json"
    plan.write_text(json.dumps(dict(jobs=js, workers=int(workers), results=str(results), logs=str(logs)), indent=1))
    with open(logs / "pool.log", "a") as out:
        subprocess.Popen([PY, str(pathlib.Path(__file__).resolve()), "_pool", str(plan)], cwd=ROOT, stdout=out,
                         stderr=subprocess.STDOUT, start_new_session=True)
    print(f"Started: {len(todo)} of {len(js)} runs to go, at most {workers} at once ({'SMOKE, ' if smoke else ''}results in "
          f"{results}). status() shows the progress.")


def stop(smoke: bool = False) -> None:
    """End a running search: its unfinished runs are stopped (a later launch starts them again from the beginning)."""
    pid = _pool_pid(paths(smoke)[1])
    if not pid:
        print("No search is running.")
        return
    os.kill(pid, signal.SIGTERM)
    print(f"Stopping the search (pid {pid}).")


def _write_status(logs: pathlib.Path, st: dict) -> None:
    tmp = logs / "pool_status.tmp"
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, logs / "pool_status.json")


def _pool(plan_path: str) -> None:
    """The background search: runs the plan's unfinished jobs, ``workers`` at a time (``benchmarks.tune_one``, one log each)."""
    plan = json.loads(pathlib.Path(plan_path).read_text())
    results, logs, workers = pathlib.Path(plan["results"]), pathlib.Path(plan["logs"]), int(plan["workers"])
    pidf = logs / "pool.pid"
    pidf.write_text(str(os.getpid()))
    env = dict(os.environ, TEMPO_BENCH_RESULTS=str(results), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    queue = [j for j in plan["jobs"] if not result_path(j, results).exists()]
    total, running, finished, failed = len(queue), {}, [], []
    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append(True))
    t_start = time.time()

    def report(state: str) -> None:
        _write_status(logs, dict(state=state, pid=os.getpid(), started=time.strftime("%Y-%m-%d %H:%M:%S",
                                 time.localtime(t_start)), updated=time.strftime("%Y-%m-%d %H:%M:%S"), workers=workers,
                                 to_run_at_start=total, finished=finished, failed=failed, running=sorted(running)))

    try:
        while (queue or running) and not stopping:
            while queue and len(running) < workers and not stopping:
                j = queue.pop(0)
                jid = job_id(j)
                lf = open(logs / f"{jid}.log", "w")
                p = subprocess.Popen([PY, "-m", "benchmarks.tune_one", json.dumps(j)], cwd=ROOT, env=env, stdout=lf,
                                     stderr=subprocess.STDOUT)
                running[jid] = (p, j, time.time(), lf)
            report("running")
            time.sleep(2.0)
            for jid in list(running):
                p, j, t0, lf = running[jid]
                rc = p.poll()
                if rc is None:
                    continue
                lf.close()
                del running[jid]
                rec = dict(job=jid, minutes=round((time.time() - t0) / 60, 2), returncode=rc)
                (finished if rc == 0 and result_path(j, results).exists() else failed).append(rec)
    finally:
        for p, j, t0, lf in running.values():
            p.terminate()
        for p, j, t0, lf in running.values():
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
            lf.close()
        report("stopped" if stopping else "finished")
        pidf.unlink(missing_ok=True)


def status(smoke: bool = False, show: bool = True):
    """Progress per (task, algorithm): finished / all runs, running, failed (with their logs); returns the table."""
    import pandas as pd
    results, logs = paths(smoke)
    plan = logs / "pool_plan.json"
    js = json.loads(plan.read_text())["jobs"] if plan.exists() else jobs(smoke=smoke)
    st = json.loads((logs / "pool_status.json").read_text()) if (logs / "pool_status.json").exists() else {}
    pid = _pool_pid(logs)
    running = set(st.get("running", [])) if pid else set()
    failed = {f["job"] for f in st.get("failed", [])}
    rows = []
    for j in js:
        jid = job_id(j)
        done = result_path(j, results).exists()
        rows.append(dict(task=j["env"], algorithm=j["algo"], state="finished" if done else "running" if jid in running
                         else "failed" if jid in failed else "to run"))
    df = pd.DataFrame(rows)
    table = df.pivot_table(index=["task", "algorithm"], columns="state", aggfunc="size", fill_value=0)
    if show:
        n_done = int((df.state == "finished").sum())
        mins = [f["minutes"] for f in st.get("finished", [])]
        eta = ""
        if pid and mins and n_done < len(df):
            left = (len(df) - n_done) * (sum(mins) / len(mins)) / max(1, int(st.get("workers", 1)))
            eta = f"; about {left:.0f} min left"
        print(f"search {'RUNNING (pid %d)' % pid if pid else 'not running'}: {n_done} of {len(df)} runs finished{eta}"
              + (" [SMOKE]" if smoke else ""))
        if failed - {job_id(j) for j in js if result_path(j, results).exists()}:
            print("failed runs (their logs are in " + str(logs) + "):",
                  ", ".join(sorted(failed - {job_id(j) for j in js if result_path(j, results).exists()})[:10]))
        print(table.to_string())
    return table


def _auc(path: pathlib.Path) -> float:
    """Mean evaluation return over the whole run (evaluations after the budget discarded)."""
    import numpy as np
    r = json.loads(pathlib.Path(path).read_text())
    keep = [i for i, s in enumerate(r["steps"]) if s <= r["job"]["steps"]]
    return float(np.mean([np.mean(r["returns"][i]) for i in keep])) if keep else math.nan


def summary(smoke: bool = False, tasks=TASKS, algos=ALGOS, seeds=DEV_SEEDS, show: bool = True):
    """Every candidate's mean final score (and AUC) over the development seeds, best first per (task, algorithm); the
    winner (:func:`select`'s rule) is marked. Returns the table."""
    import benchlab as bl
    import pandas as pd
    results, _ = paths(smoke)
    rows = []
    for env in tasks:
        for algo in algos:
            for order, (name, hp) in enumerate(grid(algo)):
                finals, aucs = [], []
                for s in seeds:
                    p = result_path(dict(suite=SUITE, env=env, algo=algo, variant=name, seed=s), results)
                    if p.exists():
                        finals.append(bl.final_of(p))
                        aucs.append(_auc(p))
                rows.append(dict(task=env, algorithm=algo, candidate=name, order=order, seeds=len(finals),
                                 final=float(sum(finals) / len(finals)) if finals else math.nan,
                                 per_seed=", ".join(f"{x:.4g}" for x in finals),
                                 auc=float(sum(aucs) / len(aucs)) if aucs else math.nan, hp=hp))
    df = pd.DataFrame(rows)
    # ties are compared on means rounded to 9 decimals: scores are multiples of 1/(episodes x seeds) far coarser than that,
    # so equal scores reached through different per-seed splits tie exactly instead of by the last bit of a float
    df["_f"], df["_a"] = df.final.round(9), df.auc.round(9)
    df["winner"] = False
    for (env, algo), g in df.groupby(["task", "algorithm"]):
        if (g.seeds == len(seeds)).all():
            best = g.sort_values(["_f", "_a", "order"], ascending=[False, False, True]).index[0]
            df.loc[best, "winner"] = True
    df = df.sort_values(["task", "algorithm", "_f", "_a", "order"],
                        ascending=[True, True, False, False, True]).drop(columns=["_f", "_a"])
    if show:
        for (env, algo), g in df.groupby(["task", "algorithm"], sort=False):
            done = (g.seeds == len(seeds)).all()
            print(f"\n{env} {algo}" + ("" if done else f"  (incomplete: {int((g.seeds == len(seeds)).sum())} of {len(g)} "
                                                       "candidates have every seed; no winner yet)"))
            for _, r in g.iterrows():
                mark = "WINNER " if r.winner else "       "
                print(f"  {mark}{r.candidate:26s} final {r.final:9.4g} (seeds: {r.per_seed or '-'})  AUC {r.auc:9.4g}")
    return df


def select(write: bool = False, smoke: bool = False, partial: bool = False) -> dict:
    """The winner of every (task, algorithm) whose candidates have all finished: ``{algo: {task: {...}}}``. With ``write``,
    merge them into ``benchmarks/tuned_toy_text.json`` (smoke: into the temporary folder), which ``benchmarks/zoo.py`` reads;
    only once every run of the search has finished, unless ``partial``."""
    results, logs = paths(smoke)
    js = jobs(smoke=smoke)
    n_done = sum(result_path(j, results).exists() for j in js)
    if n_done < len(js):
        print(f"THE SEARCH IS NOT FINISHED: {n_done} of {len(js)} runs"
              + (" (it is still running in the background)" if _pool_pid(logs) else " (it is not running: launch() resumes it)")
              + ". Run this cell again when status() shows every run finished."
              + (" Nothing is written until then." if write and not partial else ""))
        if write and not partial:
            write = False
    df = summary(smoke=smoke, show=False)
    out: dict = {}
    for _, r in df[df.winner].iterrows():
        g = df[(df.task == r.task) & (df.algorithm == r.algorithm)]
        out.setdefault(r.algorithm, {})[r.task] = dict(
            hp=r.hp, candidate=r.candidate, dev_seeds=list(DEV_SEEDS), mean_final=r.final, mean_auc=r.auc,
            default_final=float(g[g.candidate == "default"].final.iloc[0]), candidates=int(len(g)))
    missing = sorted({(t, a) for t in TASKS for a in ALGOS} - {(t, a) for a, d in out.items() for t in d})
    for algo, d in out.items():
        for t, w in d.items():
            print(f"{t:16s} {algo}: {w['candidate']:24s} final {w['mean_final']:.4g} (default {w['default_final']:.4g})")
    if missing:
        print("not selected yet (unfinished candidates): " + ", ".join(f"{t} {a}" for t, a in missing))
    if write and out:
        target = paths(True)[0].parent / "tuned_toy_text.json" if smoke else TUNED_JSON
        data = json.loads(target.read_text()) if target.exists() else {}
        data["_about"] = ("PPO / DQN hyperparameters of the toy-text tasks (no RL-Zoo entry exists), selected by the grid "
                          "search of notebooks/tunelab.py on development seeds 0-1: highest mean final score, ties to the "
                          "higher mean AUC. Read by benchmarks/zoo.py; written by tunelab.select(write=True).")
        for algo, d in out.items():
            for t, w in d.items():
                data.setdefault(algo, {})[t] = dict(w, selected=time.strftime("%Y-%m-%d %H:%M"))
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, target)
        print(f"\nwritten: {target}")
        if not smoke:
            print("Next: run the launch cells of Baselines_test 19_FrozenLake-v1, 20_CliffWalking-v1, 23_Blackjack-v1 and "
                  "24_Taxi-v4. Their PPO / DQN runs are now 'other protocol' (untuned) and run again with these values.")
    elif write:
        print("\nnothing written")
    return out


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "_pool":
        _pool(sys.argv[2])
    else:
        print("usage: python notebooks/tunelab.py _pool <plan.json>   (tunelab.launch() starts it)")
