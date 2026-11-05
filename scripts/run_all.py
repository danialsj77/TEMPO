"""Run the final benchmark from the command line: every task of both test folders (TEMPO in TEMPO_test/, the baselines in
Baselines_test/), seeds 10-14, at most ``--jobs`` runs at a time on one machine.

    bash run_all.sh                       # 10 runs at a time; JOBS=12 bash run_all.sh for 12
    python scripts/run_all.py --jobs 10 [--gpu-jobs N] [--kinds tempo,baselines] [--tasks Hopper-v5,Ant-v5]
                              [--carracing] [--allow-cpu] [--dry-run] [--smoke]

The runs are exactly those of the run notebooks: ``notebooks/runlab.py`` plans them (``make_lab`` + ``make_jobs``: the
protocol, the seeds, the hyperparameters of ``benchmarks/zoo.py`` and the device policy: TEMPO and CrossQ on the GPU, NVIDIA
CUDA when present, else the Apple GPU; SAC, TD3, PPO and DQN on one CPU thread each) and ``benchmarks/run_one.py`` trains each
one. Results, logs and lock files go where the notebooks put them (``<folder>/results``, ``<folder>/logs``), so the notebooks
and ``Comparison/`` read them unchanged. One queue serves both folders, longest runs first. Finished runs are skipped, so the
script resumes after an interruption (an interrupted run starts again from step 0). Result files that cannot be used (another
protocol or other hyperparameters, older TEMPO code, unreadable) are first moved to ``<folder>/results/_outdated/<time>/``,
never deleted, exactly as a notebook's launch cell does. CarRacing-v3 is left out unless ``--carracing`` (it is left out of
the comparison too and takes days on its own). Only one copy of the script runs at a time (``logs/run_all.lock``), and it
refuses to start without a GPU unless ``--allow-cpu`` (TEMPO and CrossQ would then run on the CPU: another device than the
protocol's and many times slower).
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import fcntl
import io
import json
import os
import pathlib
import platform
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True

LEFT_OUT = ("CarRacing-v3",)          # as Comparison/complab.py LEFT_OUT
TO_RUN = ("to run", "failed", "interrupted", "unreadable", "other protocol", "older code")   # runlab row states


def _os_name(platform_string: str | None) -> str:
    """'macOS', 'Linux', ... from a ``platform.platform()`` string."""
    return str(platform_string or "unknown").split("-")[0]


def plan(kinds, tasks, smoke: bool) -> tuple[list[dict], list, collections.Counter]:
    """(runs to start, [(lab, jobs)], finished results by the OS they were made on): every planned job of every (kind, task)
    whose state is one of ``TO_RUN`` (runlab's states), with the paths the notebooks use."""
    import runlab as rl
    import benchlab as bl

    items, labs, made_on = [], [], collections.Counter()
    for kind in kinds:
        for task in tasks:
            with contextlib.redirect_stdout(io.StringIO()):
                lab = rl.make_lab(kind, task, smoke=smoke, show=False, export_env=False)
                jobs = rl.make_jobs(lab)
                rows = rl._rows(lab, jobs)
            labs.append((lab, jobs))
            for r in rows:
                j = r["job"]
                if r["status"].startswith("done"):
                    try:
                        rec = json.loads(bl.result_path(j, lab.dirs.results).read_text())
                        made_on[_os_name((rec.get("versions") or {}).get("platform"))] += 1
                    except (OSError, ValueError, AttributeError):
                        made_on["unknown"] += 1
                if r["status"] not in TO_RUN:
                    continue                                          # done, running elsewhere, or kept older code
                jid = bl.job_id(j)
                gpu = str(j.get("device", "cpu")) != "cpu"
                items.append(dict(job=j, jid=jid, kind=kind, task=task, state=r["status"], results=lab.dirs.results,
                                  logs=lab.dirs.logs, result=bl.result_path(j, lab.dirs.results),
                                  lock=lab.dirs.logs / "locks" / f"{jid}.pid", gpu=gpu,
                                  lab_gpu_cap=int(lab.mps_workers),       # the notebook's GPU limit of this task (CarRacing: 2-3)
                                  cost=float(j.get("cost") or bl.job_seconds(j))))
    return items, labs, made_on


def move_unusable(labs) -> int:
    """Move unusable / older-code result files out of the way, as runlab.launch does (never deleted)."""
    import runlab as rl

    n, stamp = 0, time.strftime("%Y%m%d-%H%M%S")
    for lab, jobs in labs:
        with contextlib.redirect_stdout(io.StringIO()):
            rl._clean_stale_locks(lab)
            bad = rl._unusable_results(lab, jobs)
            stale = rl._stale_results(lab, jobs) if lab.kind == "tempo" and lab.rerun_outdated else []
        if bad or stale:
            moved = rl._move_unusable(lab, bad, stamp) + (rl.archive_outdated(lab, jobs, stamp=stamp) if stale else [])
            n += len(moved)
            print(f"{lab.env} ({lab.kind}): moved {len(moved)} unusable result file(s) to {lab.dirs.results / '_outdated' / stamp}")
    return n


def _cmdline(pid: int) -> str:
    """The command line of a process ('' if it is gone or cannot be read)."""
    try:
        return " ".join(a.decode(errors="replace") for a in pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if a)
    except OSError:
        pass
    try:
        return subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(pid)], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:                                                 # noqa: BLE001
        return ""


def _lock_owner(lock: pathlib.Path, jid: str) -> int | None:
    """The pid of the live process that holds ``lock`` and trains job ``jid`` (``benchmarks.run_one`` with that job, as
    started by this script or by a notebook's scheduler), else None: a lock left behind by a crash whose pid now belongs to
    another process, even another run, is not an owner."""
    import benchlab as bl
    import runlab as rl

    pid = bl._lock_pid(lock)
    if pid is None or not rl._alive(pid):
        return None
    cmd = _cmdline(pid)
    if "benchmarks.run_one" not in cmd:
        return None
    try:
        return pid if bl.job_id(json.loads(cmd[cmd.index("{"):cmd.rindex("}") + 1])) == jid else None
    except (ValueError, KeyError, TypeError):
        return pid                                                    # a run_one whose job cannot be read: assume it is this one


def _single_instance(path: pathlib.Path):
    """An exclusive lock on ``path`` for the life of this process (None if another run_all holds it)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.seek(0)
    fh.truncate()
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh


def run(items: list[dict], n_jobs: int, n_gpu: int, status: pathlib.Path, poll: float = 5.0) -> int:
    """Start the runs, longest first, keeping at most ``n_jobs`` running (at most ``n_gpu`` on the GPU, and per task at most
    its notebook's GPU limit); returns the number of failed or not started runs."""
    queue = sorted(items, key=lambda it: -it["cost"])
    running: dict[str, tuple] = {}
    done, failed, skipped, t_start = [], [], [], time.time()
    stop = []

    def _stop(*_):
        stop.append(True)

    def _write_status(state: str) -> None:
        tmp = status.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(updated=time.strftime("%Y-%m-%d %H:%M:%S"), state=state, running=sorted(running),
                                       pending=[it["jid"] for it in queue], done=done, failed=failed, not_started=skipped,
                                       hours=round((time.time() - t_start) / 3600, 2)), indent=1))
        os.replace(tmp, status)

    def _can_start(it) -> bool:
        if not it["gpu"]:
            return True
        on_gpu = [r for _, r, _, _ in running.values() if r["gpu"]]
        same = sum(1 for r in on_gpu if (r["kind"], r["task"]) == (it["kind"], it["task"]))
        return len(on_gpu) < n_gpu and same < max(1, it["lab_gpu_cap"])

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    status.parent.mkdir(parents=True, exist_ok=True)
    if n_gpu == 0 and any(it["gpu"] for it in queue):
        skipped = [it["jid"] for it in queue if it["gpu"]]
        queue = [it for it in queue if not it["gpu"]]
        print(f"[run_all] --gpu-jobs 0: {len(skipped)} GPU run(s) are not started", flush=True)
    print(f"[run_all] {len(queue)} runs to do, at most {n_jobs} at a time ({n_gpu} on the GPU)", flush=True)
    try:
        while (queue or running) and not stop:
            for jid, (p, it, t0, fh) in list(running.items()):
                if p.poll() is None:
                    continue
                fh.close()
                del running[jid]
                it["lock"].unlink(missing_ok=True)
                ok = p.returncode == 0 and it["result"].exists()
                (done if ok else failed).append(jid)
                print(f"[run_all] {'DONE' if ok else 'FAILED'} {jid} ({(time.time() - t0) / 60:.1f} min) | running "
                      f"{len(running)} pending {len(queue)} done {len(done)} failed {len(failed)}", flush=True)
            while queue and len(running) < n_jobs and not stop:
                nxt = next((it for it in queue if _can_start(it)), None)
                if nxt is None:
                    break
                queue.remove(nxt)
                if nxt["result"].exists():
                    print(f"[run_all] skipped {nxt['jid']}: its result appeared meanwhile", flush=True)
                    continue
                owner = _lock_owner(nxt["lock"], nxt["jid"])
                if owner is not None:
                    print(f"[run_all] skipped {nxt['jid']}: process {owner} is running it", flush=True)
                    continue
                j = nxt["job"]
                nxt["logs"].mkdir(parents=True, exist_ok=True)
                nxt["lock"].parent.mkdir(parents=True, exist_ok=True)
                fh = open(nxt["logs"] / f"{nxt['jid']}.log", "w")
                env = {**os.environ, "TEMPO_BENCH_RESULTS": str(nxt["results"]), "TEMPO_BENCH_LOGS": str(nxt["logs"]),
                       "OMP_NUM_THREADS": str(j.get("threads", 1)), "MKL_NUM_THREADS": str(j.get("threads", 1)),
                       "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
                p = subprocess.Popen([sys.executable, "-m", "benchmarks.run_one", json.dumps(j)], cwd=ROOT, env=env,
                                     stdout=fh, stderr=subprocess.STDOUT)
                nxt["lock"].write_text(f"{p.pid} {j.get('device', 'cpu')}")
                running[nxt["jid"]] = (p, nxt, time.time(), fh)
            if queue and not running and not stop:                   # nothing runs and nothing may start: give up
                skipped += [it["jid"] for it in queue]
                print(f"[run_all] {len(queue)} run(s) cannot start under the GPU limits: {', '.join(skipped[-3:])} ...",
                      flush=True)
                queue = []
            _write_status("running")
            if queue or running:
                time.sleep(poll)
    finally:
        if running:
            print(f"[run_all] stopping {len(running)} running run(s); they start again from step 0 next time", flush=True)
        for p, it, _, fh in running.values():
            p.terminate()
        for p, it, _, fh in running.values():
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
            fh.close()
            it["lock"].unlink(missing_ok=True)
        running.clear()
    _write_status("stopped" if stop else "finished")
    print(f"[run_all] {'stopped' if stop else 'finished'} after {(time.time() - t_start) / 3600:.1f} h: {len(done)} done, "
          f"{len(failed)} failed{': ' + ', '.join(failed) if failed else ''}"
          + (f", {len(skipped)} not started" if skipped else ""), flush=True)
    return len(failed) + len(skipped)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=10, help="runs at the same time (default 10)")
    ap.add_argument("--gpu-jobs", type=int, default=None, help="at most this many of them on the GPU (default: --jobs)")
    ap.add_argument("--kinds", default="tempo,baselines", help="tempo, baselines or both (default)")
    ap.add_argument("--tasks", default=None, help="comma-separated task ids (default: every task except CarRacing-v3)")
    ap.add_argument("--carracing", action="store_true", help="include CarRacing-v3 (days of runs; outside the comparison)")
    ap.add_argument("--allow-cpu", action="store_true", help="run even without a GPU (TEMPO and CrossQ then on the CPU)")
    ap.add_argument("--dry-run", action="store_true", help="only show what would run")
    ap.add_argument("--smoke", action="store_true", help="tiny pipeline check: first seed, 1,500 steps, scratch folder")
    a = ap.parse_args(argv)
    if a.smoke:
        os.environ["TEMPO_NOTEBOOK_SMOKE"] = "1"
    else:
        os.environ.pop("TEMPO_NOTEBOOK_SMOKE", None)
    import runlab as rl
    import benchlab as bl
    from benchmarks import plans

    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()]
    if not set(kinds) <= {"tempo", "baselines"}:
        ap.error("--kinds takes tempo and/or baselines")
    tasks = [t.strip() for t in a.tasks.split(",")] if a.tasks else [t for t in rl.tasks() if a.carracing or t not in LEFT_OUT]
    unknown = [t for t in tasks if t not in rl.tasks()]
    if unknown:
        ap.error(f"unknown task(s): {', '.join(unknown)}; known: {', '.join(rl.tasks())}")
    gpu = plans.gpu_device()
    print(f"[run_all] GPU: {gpu or 'none (GPU jobs run on the CPU)'}; seeds {list(rl.TEST_SEEDS)}; "
          f"{rl.TEST_BUDGET:,} steps per run; {len(tasks)} tasks; {', '.join(kinds)}" + ("; SMOKE" if a.smoke else ""), flush=True)
    if gpu is None and not (a.allow_cpu or a.smoke or a.dry_run):
        print("[run_all] not started: PyTorch sees no GPU (torch.cuda.is_available() is False), so TEMPO and CrossQ would run "
              "on the CPU, another device than the protocol's and many times slower. Install a CUDA build of PyTorch (README) "
              "or pass --allow-cpu.")
        return 2
    competing = [] if a.smoke else rl._competing_processes()
    if competing:
        print("[run_all] not started: a baseline hyperparameter search or a timing measurement is running (their load would "
              "end up in the runs' wall-clock times):\n" + "\n".join(f"  pid {p}: {c}" for p, c in competing))
        return 2
    lock = None
    if not a.dry_run:
        lock = _single_instance((bl.scratch_root() if a.smoke else ROOT / "logs") / "run_all.lock")
        if lock is None:
            print("[run_all] not started: another run_all is running (logs/run_all.lock); stop it first or wait for it")
            return 2
    items, labs, made_on = plan(kinds, tasks, a.smoke)
    by = collections.Counter((it["kind"], it["job"]["algo"], it["job"].get("device", "cpu"), it["state"]) for it in items)
    for (kind, algo, dev, state), n in sorted(by.items()):
        print(f"  {kind:9s} {algo:7s} on {dev:5s}: {n:3d} ({state})")
    here = _os_name(platform.platform())
    other = {k: n for k, n in made_on.items() if k != here}
    if other:
        print(f"[run_all] NOTE: {sum(other.values())} finished result(s) were made on another system "
              f"({', '.join(f'{n} on {k}' for k, n in sorted(other.items()))}) and are kept, not re-run. For a benchmark "
              f"that comes entirely from this machine, start from a fresh clone (or move those result files away).")
    if a.dry_run or not items:
        print(f"[run_all] {len(items)} runs to do" + (" (dry run: nothing started)" if a.dry_run else ""))
        return 0
    move_unusable(labs)
    status = (bl.scratch_root() if a.smoke else ROOT / "logs") / "run_all_status.json"
    print(f"[run_all] progress: {status}; logs: <folder>/logs/<run>.log", flush=True)
    try:
        return 1 if run(items, max(1, a.jobs), max(0, a.gpu_jobs if a.gpu_jobs is not None else a.jobs), status) else 0
    finally:
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
