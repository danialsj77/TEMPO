"""Run a list of benchmark jobs as subprocesses with per-device concurrency limits.

    python -m benchmarks.scheduler plan_name --cpu 10 --mps 4      # --mps: GPU jobs (Apple "mps" or NVIDIA "cuda")

Jobs whose result file exists are skipped, so the scheduler can be restarted at
any time. Logs go to benchmarks/logs/<job-id>.log; progress to benchmarks/logs/<plan>.status.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
#: logs root; override with TEMPO_BENCH_LOGS
LOGS = pathlib.Path(__import__("os").environ.get("TEMPO_BENCH_LOGS", ROOT / "benchmarks" / "logs"))
PY = sys.executable


def _pool(device) -> str:
    """Worker pool of a job's device: "cpu", or "mps" for any GPU (Apple "mps", NVIDIA "cuda" / "cuda:<i>"), limited by --mps."""
    return "cpu" if str(device or "cpu") == "cpu" else "mps"


def job_id(j: dict) -> str:
    return f"{j.get('suite', 'main')}__{j['env']}__{j['algo']}__{j.get('variant', 'default')}__s{j['seed']}"


def _alive(pid: int) -> bool:
    import os

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _lock(j: dict) -> pathlib.Path:
    return LOGS / "locks" / f"{job_id(j)}.pid"


def running_elsewhere(j: dict) -> bool:
    """True if another scheduler is currently running this job (restart safety)."""
    f = _lock(j)
    if not f.exists():
        return False
    return _lock_device(f) is not None      # lock files hold "pid" or "pid device"


def _lock_device(f: pathlib.Path) -> tuple[int, str] | None:
    """(pid, device) of a live job recorded in a lock file, else None."""
    try:
        parts = f.read_text().split()
        pid = int(parts[0])
    except (ValueError, IndexError, OSError):
        return None
    if not _alive(pid):
        return None
    if len(parts) > 1:
        return pid, parts[1]
    try:                                    # older lock files hold only the pid
        cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True,
                             text=True).stdout
        return pid, ("mps" if ('"device": "mps"' in cmd or '"device": "cuda' in cmd) else "cpu")
    except OSError:
        return pid, "cpu"


def external_running(own: set[str]) -> dict[str, int]:
    """Live jobs per device started by *other* scheduler processes (from lock files)."""
    counts = {"cpu": 0, "mps": 0}
    for f in (LOGS / "locks").glob("*.pid"):
        if f.stem in own:
            continue
        info = _lock_device(f)
        if info is not None:
            counts[_pool(info[1])] = counts.get(_pool(info[1]), 0) + 1
    return counts


def run(jobs: list[dict], plan: str, max_cpu: int, max_mps: int, poll: float = 5.0,
        shared_capacity: bool = True) -> None:
    """Run ``jobs`` with at most ``max_cpu`` / ``max_mps`` concurrent jobs per device.

    With ``shared_capacity`` (default) the limits also count jobs that other scheduler
    processes are running (seen through lock files), so restarting a scheduler or running
    several plans side by side never oversubscribes a device beyond the given limits.
    """
    from .run_one import result_path

    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "locks").mkdir(exist_ok=True)
    status = LOGS / f"{plan}.status"
    pending = [j for j in jobs if not result_path(j).exists() and not running_elsewhere(j)]
    pending.sort(key=lambda j: -j.get("cost", j["steps"]))
    running: dict[str, tuple[subprocess.Popen, dict, float]] = {}
    done, failed = [], []
    print(f"[{plan}] {len(jobs)} jobs, {len(jobs) - len(pending)} already done, {len(pending)} to run",
          flush=True)
    while pending or running:
        for jid, (p, j, t0) in list(running.items()):
            if p.poll() is not None:
                del running[jid]
                _lock(j).unlink(missing_ok=True)
                ok = p.returncode == 0 and result_path(j).exists()
                (done if ok else failed).append(jid)
                print(f"[{plan}] {'DONE' if ok else 'FAILED'} {jid} ({(time.time() - t0) / 60:.1f} min) "
                      f"| running {len(running)} pending {len(pending)} done {len(done)} failed {len(failed)}",
                      flush=True)
        ext = external_running(set(running)) if shared_capacity else {"cpu": 0, "mps": 0}
        for dev, cap in (("mps", max_mps), ("cpu", max_cpu)):
            n_dev = sum(1 for _, j, _ in running.values() if _pool(j.get("device")) == dev)
            n_dev += ext.get(dev, 0)
            while n_dev < cap:
                nxt = next((j for j in pending if _pool(j.get("device")) == dev), None)
                if nxt is None:
                    break
                pending.remove(nxt)
                if result_path(nxt).exists() or running_elsewhere(nxt):
                    continue
                jid = job_id(nxt)
                log = open(LOGS / f"{jid}.log", "w")
                p = subprocess.Popen([PY, "-m", "benchmarks.run_one", json.dumps(nxt)], cwd=ROOT,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     env={**__import__("os").environ, "OMP_NUM_THREADS": str(nxt.get("threads", 1)),
                                          "PYTHONUNBUFFERED": "1"})
                running[jid] = (p, nxt, time.time())
                _lock(nxt).write_text(f"{p.pid} {nxt.get('device', 'cpu')}")
                n_dev += 1
        status.write_text(json.dumps(dict(running=list(running), pending=[job_id(j) for j in pending],
                                          done=done, failed=failed, time=time.strftime("%H:%M:%S")),
                                     indent=1))
        time.sleep(poll)
    print(f"[{plan}] finished: {len(done)} done, {len(failed)} failed: {failed}", flush=True)


def status(plan: str) -> dict:
    """Progress of a plan started by :func:`run` (reads ``logs/<plan>.status``)."""
    f = LOGS / f"{plan}.status"
    return json.loads(f.read_text()) if f.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", help="a name in benchmarks.plans.PLANS, or any label with --jobs-file")
    ap.add_argument("--jobs-file", default=None, help="JSON list of job dicts (e.g. from a notebook)")
    ap.add_argument("--cpu", type=int, default=10)
    ap.add_argument("--mps", type=int, default=4, help="concurrent GPU jobs (Apple mps or NVIDIA cuda)")
    a = ap.parse_args()
    if a.jobs_file:
        jobs = json.loads(pathlib.Path(a.jobs_file).read_text())
    else:
        from . import plans

        jobs = plans.PLANS[a.plan]()
    run(jobs, a.plan, a.cpu, a.mps)


if __name__ == "__main__":
    main()
