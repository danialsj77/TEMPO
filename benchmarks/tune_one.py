"""Run one candidate of a baseline hyperparameter search: ``benchmarks.run_one.main`` with the job's own SB3
hyperparameters (``job["hp"]``) in place of ``benchmarks/zoo.py``.

    python -m benchmarks.tune_one '<job json>'     # TEMPO_BENCH_RESULTS = the search's results folder

``notebooks/tunelab.py`` builds the jobs (the test protocol of ``notebooks/runlab.py`` with development seeds) and runs
this module once per job. ``run_one`` itself is not modified: ``run_sb3`` imports ``zoo_hyperparams`` from
``benchmarks.zoo`` when it is called, so replacing that attribute in this process hands it the candidate. The result file
records the candidate twice: in ``job["hp"]`` and, as SB3 received it, in ``info["config"]``.
"""
import json
import sys
import traceback


def main(job: dict) -> None:
    if job.get("algo") == "TEMPO" or not isinstance(job.get("hp"), dict):
        raise ValueError("tune_one runs baseline candidates: a job with algo != 'TEMPO' and an 'hp' dict")
    from benchmarks import run_one, zoo

    def candidate(algo: str, env_id: str) -> tuple[dict, bool]:
        if (algo, env_id) != (job["algo"], job["env"]):
            raise RuntimeError(f"tune_one: asked for {algo} on {env_id}, but the job is {job['algo']} on {job['env']}")
        return json.loads(json.dumps(job["hp"])), False      # a fresh copy; False: a candidate, not a tuned setting

    zoo.zoo_hyperparams = candidate
    run_one.main(job)


if __name__ == "__main__":
    try:
        main(json.loads(sys.argv[1]))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
