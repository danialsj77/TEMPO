"""Compare TEMPO with the baselines from the command line: the comparison notebook's whole call sequence
(Comparison/complab.py), reading Baselines_test/results and TEMPO_test/results and writing the figures and tables to
Comparison/figures and Comparison/tables (or below --out).

    python scripts/compare.py [--out DIR]
"""
import argparse
import contextlib
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "Comparison"), str(ROOT / "notebooks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.dont_write_bytecode = True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=None, help="output folder (figures/ and tables/ go below it); default Comparison/")
    ap.add_argument("--log", default=None, help="write the printed report to this file instead of the screen")
    a = ap.parse_args(argv)
    if a.out:
        os.environ["COMPARISON_OUT_DIR"] = str(pathlib.Path(a.out).resolve())
    os.environ.pop("TEMPO_NOTEBOOK_SMOKE", None)
    import matplotlib

    matplotlib.use("Agg")
    import complab as cl

    t = time.time()
    with (open(a.log, "w") if a.log else contextlib.nullcontext(sys.stdout)) as fh, contextlib.redirect_stdout(fh):
        cfg = cl.make_config()
        cfg.show()
        D = cl.load_data(cfg)
        cl.show_inventory(D)
        cl.plot_curve_grid(D)
        cl.plot_metric_groups(D, "final")
        cl.plot_metric_groups(D, "auc")
        cl.plot_auc_intervals(D)
        cl.plot_heatmap(D)
        cl.plot_family_overlays(D)
        cl.show_scores(D)
        cl.show_aggregates(D)
        cl.show_efficiency(D)
        cl.show_stability(D)
        cl.show_compute(D)
        cl.show_variants(D)
        cl.show_exports(D)
    print(f"comparison of {len(D.tasks)} tasks done in {time.time() - t:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
