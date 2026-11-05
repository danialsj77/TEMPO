"""Time every algorithm alone on every task (Comparison/timelab.py, the timing notebook's call sequence): the isolated
training times of the rank score's robustness check. Run it after the final run, with nothing else running (it refuses
otherwise). Finished measurements are skipped; measurements of changed settings are timed again.

    python scripts/time_all.py
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "Comparison"), str(ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True


def main() -> int:
    os.environ.pop("TEMPO_NOTEBOOK_SMOKE", None)
    import matplotlib

    matplotlib.use("Agg")
    import timelab as tl

    cfg = tl.make_config()
    plan = tl.plan(cfg)
    tl.run(cfg, plan)
    res = tl.results(cfg, plan)
    tl.plot(cfg, res)
    tl.save(cfg, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
