"""Publication figures and tables for the benchmark suites.

    python -m benchmarks.report --suite final    # curves, score/AUC tables, aggregates, W/T/L, compute
    python -m benchmarks.report --screens        # development-screen tables (screen1, screen2, diag)
    python -m benchmarks.report --paper          # both, + README results block + paper figures

Inputs: ``<results>/<suite>/<env>/<algo>/<variant>/seed*.json`` as written by
``benchmarks.run_one``. Outputs go to ``benchmarks/figures`` and ``benchmarks/tables``
(``--results``, ``--figures`` and ``--tables`` redirect them, e.g. to a scratch
directory for testing). Nothing under ``--results`` is ever written.

Paper and README outputs (``--suite final``): ``paper_main_final.tex`` (every task x
algorithm), ``paper_aggregate.tex`` (TEMPO vs. each baseline on the tasks where TEMPO
has all five seeds), ``paper_crossq.tex`` (TEMPO vs. CrossQ with critic widths),
``final_status.json`` (which runs exist) and the figure ``final_tempo_curves``;
``--screens`` adds ``paper_screens.tex``. Missing runs are shown, never dropped:
``--`` / ``pending`` = planned but not finished, ``(n=k)`` = fewer than five seeds.
``--update-readme`` rewrites only the block between ``<!-- RESULTS:START -->`` and
``<!-- RESULTS:END -->`` of README.md; ``--paper-figures DIR`` copies the figures the
paper includes into DIR (``paper/figures``).

Per run (benchmarks/PROTOCOL.md): evaluations after the step budget are discarded
(PPO with many parallel environments overshoots the budget by up to one rollout);
``final`` = mean return of the last three evaluations (30 episodes); ``AUC`` = mean
return over all evaluations. Per task: mean over seeds and a 95% percentile-bootstrap
CI. Aggregates (Agarwal et al., 2021): per-task scores normalised as
``(x - random) / (best_baseline - random)``, where ``best_baseline`` is the highest
mean final score among the SB3 baselines on that task; IQM (25% trimmed mean over all
runs), mean and median (over per-task means) with stratified-bootstrap CIs; performance
profiles; probability of improvement. Pairwise tests: two-sided Welch t-test on the
per-seed final scores, alpha = 0.05.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import shutil
import warnings
import zlib
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .plans import ADDED, DISCRETE, FINAL_SEEDS, IMAGE, SUITE, TOY_TEXT, TOY_TEXT_ALL

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"
BASELINES = ("SAC", "TD3", "CrossQ", "PPO", "DQN")
ALGO_ORDER = ("TEMPO", "SAC", "TD3", "CrossQ", "PPO", "DQN")
BEST = "Best baseline"      # composite: the per-task best SB3 baseline (by mean final score)
ALPHA = 0.05
FAMILY_NAMES = {"classic-discrete": "Classic control, discrete",
                "classic-continuous": "Classic control, continuous",
                "box2d-discrete": "Box2D, discrete", "box2d-continuous": "Box2D, continuous",
                "mujoco": "MuJoCo", "toy-text": "Toy text (TEMPO only; no tuned baselines)",
                "other": "Other"}
TASK_INFO = {**SUITE, **TOY_TEXT}     # env: (budget, eval_freq, family) of the EARLIER benchmark (20 tasks)


def _family_order() -> list:
    """Every task of the test folders (the 20 of TASK_INFO + the 4 of plans.ADDED) in report order: each added task
    right after the last task of its family, so that families stay contiguous (the 20 keep their relative order)."""
    order = list(TASK_INFO)
    for env, fam in ADDED.items():
        at = max((i for i, e in enumerate(order) if (TASK_INFO[e][2] if e in TASK_INFO else ADDED[e]) == fam),
                 default=len(order) - 1)
        order.insert(at + 1, env)
    return order


#: report / comparison order of all 24 tasks (CarRacing after BipedalWalker, HumanoidStandup after Humanoid, Blackjack
#: and Taxi after CliffWalking); the notebook numbers 21-24 of the added tasks are runlab.tasks()' order instead
TASK_ORDER = _family_order()

# ------------------------------------------------------------------ style
# Categorical slots from the validated default palette (dataviz skill), ordered so the six
# series pass every hard gate all-pairs on a white surface (normal-vision dE >= 15.6).
# CVD separation of TD3/TEMPO sits in the 6-8 warn band, so every series also carries its
# own dash pattern (secondary encoding); yellow/aqua are sub-3:1, relieved by the legend and
# the tables. TEMPO: the strongest slot, thicker line, drawn on top.
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STYLE = {
    "TEMPO": dict(color="#e34948", ls="-", lw=2.0, z=10),
    "SAC": dict(color="#2a78d6", ls="-", lw=1.25, z=5),
    "TD3": dict(color="#1baf7a", ls=(0, (4, 1.5)), lw=1.25, z=4),
    "CrossQ": dict(color="#4a3aa7", ls=(0, (6, 1.5, 1.5, 1.5)), lw=1.25, z=6),
    "PPO": dict(color="#eda100", ls=(0, (1.2, 1.1)), lw=1.6, z=3),
    "DQN": dict(color="#008300", ls=(0, (3, 1, 1, 1, 1, 1)), lw=1.25, z=4),
    BEST: dict(color=INK2, ls=(0, (3, 2)), lw=1.25, z=2),
}
_EXTRA = [(INK2, (0, (2, 1))), (MUTED, (0, (5, 2))), ("#383835", (0, (1, 2))),
          (MUTED, (0, (4, 1, 1, 1)))]


def style(label: str) -> dict:
    if label in STYLE:
        return STYLE[label]
    i = zlib.crc32(label.encode()) % len(_EXTRA)
    return dict(color=_EXTRA[i][0], ls=_EXTRA[i][1], lw=1.25, z=2)


def label_of(algo: str, variant: str) -> str:
    return algo if variant == "default" else f"{algo}[{variant}]"


def label_key(label: str):
    base = label.split("[")[0]
    return (ALGO_ORDER.index(base) if base in ALGO_ORDER else len(ALGO_ORDER), label != base, label)


def task_key(env: str):
    order = TASK_ORDER
    return (order.index(env) if env in order else len(order), env)


def family(env: str) -> str:
    return TASK_INFO[env][2] if env in TASK_INFO else ADDED.get(env, "other")


def expected_pool(env: str) -> set[str]:
    """Algorithms the benchmark plan runs on a task (anything else is 'n/a', not 'pending'): TEMPO only on toy text
    (earlier and added); no CrossQ on the pixel tasks of plans.IMAGE (sb3-contrib CrossQ has no CNN policy)."""
    if env in TOY_TEXT_ALL:
        return {"TEMPO"}
    pool = {"DQN", "PPO"} if env in DISCRETE else {"SAC", "TD3", "CrossQ", "PPO"}
    return {"TEMPO"} | (pool - {"CrossQ"} if env in IMAGE else pool)


def fmt_steps(n: int) -> str:
    return f"{n / 1e6:g}M" if n >= 1_000_000 else f"{n / 1e3:g}k"


# ------------------------------------------------------------------ loading
@dataclass
class Run:
    env: str
    algo: str
    variant: str
    seed: int
    budget: int
    eval_freq: int
    n_eval: int
    steps: np.ndarray               # evaluation steps <= budget
    episodes: list                  # per evaluation, list of episode returns (<= budget)
    n_dropped: int                  # evaluations after the budget (discarded)
    raw_last_step: int
    wall_h: float
    device: str
    path: pathlib.Path
    info: dict = field(default_factory=dict)
    tempo_source: str = ""
    code_root: str = ""
    versions: dict = field(default_factory=dict)
    finished: str = ""

    @property
    def label(self) -> str:
        return label_of(self.algo, self.variant)

    @property
    def curve(self) -> np.ndarray:
        return np.array([np.mean(e) for e in self.episodes], float)

    @property
    def final(self) -> float:
        return float(np.mean(np.concatenate([np.asarray(e, float) for e in self.episodes[-3:]])))

    @property
    def auc(self) -> float:
        return float(self.curve.mean())


def load_suite(results: pathlib.Path, suite: str) -> dict[str, dict[str, list[Run]]]:
    """{env: {label: [Run sorted by seed]}}."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for f in sorted((results / suite).glob("*/*/*/seed*.json")):
        r = json.loads(f.read_text())
        j = r["job"]
        steps = np.asarray(r["steps"], int)
        keep = steps <= int(j["steps"])
        if not keep.any():
            continue
        run = Run(env=j["env"], algo=j["algo"], variant=j.get("variant", "default"), seed=int(j["seed"]),
                  budget=int(j["steps"]), eval_freq=int(j["eval_freq"]), n_eval=int(j["n_eval"]),
                  steps=steps[keep], episodes=[r["returns"][i] for i in np.flatnonzero(keep)],
                  n_dropped=int((~keep).sum()), raw_last_step=int(steps[-1]),
                  wall_h=float(r.get("wall_s", np.nan)) / 3600, device=str(j.get("device", "cpu")),
                  path=f, info=r.get("info") or {}, tempo_source=r.get("tempo_source", ""),
                  code_root=r.get("code_root", ""), versions=r.get("versions") or {},
                  finished=str(r.get("finished", "")))
        out[run.env][run.label].append(run)
    for env in out:
        for lab in out[env]:
            out[env][lab].sort(key=lambda x: x.seed)
    return {e: dict(sorted(out[e].items(), key=lambda kv: label_key(kv[0])))
            for e in sorted(out, key=task_key)}


def load_randoms(path: pathlib.Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


# ------------------------------------------------------------------ statistics
def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(map(str, key)).encode()))


def boot_mean_ci(x, key, n_boot: int) -> tuple[float, float, float]:
    """Mean and 95% percentile-bootstrap CI over seeds (CI is NaN for n < 2)."""
    x = np.asarray(x, float)
    if x.size == 0:
        return (math.nan,) * 3
    m = float(x.mean())
    if x.size < 2:
        return m, math.nan, math.nan
    b = x[_rng(*key).integers(0, x.size, size=(n_boot, x.size))].mean(1)
    return m, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def trim_mean(a: np.ndarray, axis: int = -1) -> np.ndarray:
    """25% trimmed mean (the IQM), same convention as scipy.stats.trim_mean / rliable."""
    a = np.sort(np.asarray(a, float), axis=axis)
    n = a.shape[axis]
    k = int(0.25 * n)
    return np.take(a, np.arange(k, n - k), axis=axis).mean(axis=axis)


def aggregate_point(scores: dict[str, np.ndarray]) -> dict[str, float]:
    means = np.array([v.mean() for v in scores.values()])
    return dict(IQM=float(trim_mean(np.concatenate(list(scores.values())))),
                mean=float(means.mean()), median=float(np.median(means)))


def aggregate_ci(scores: dict[str, np.ndarray], n_boot: int, key) -> dict[str, list[float]]:
    """IQM / mean / median with stratified bootstrap (runs resampled within each task)."""
    rng = _rng("agg", *key)
    point = aggregate_point(scores)
    res = [v[rng.integers(0, v.size, size=(n_boot, v.size))] for v in scores.values()]
    iqm_b = trim_mean(np.concatenate(res, axis=1), axis=1)
    task_means = np.stack([r.mean(1) for r in res], axis=1)
    boots = dict(IQM=iqm_b, mean=task_means.mean(1), median=np.median(task_means, axis=1))
    return {k: [point[k], float(np.percentile(boots[k], 2.5)), float(np.percentile(boots[k], 97.5))]
            for k in ("IQM", "median", "mean")}


def performance_profile(scores: dict[str, np.ndarray], taus: np.ndarray, n_boot: int, key):
    """Fraction of runs with normalised score > tau (tasks weighted equally) and a 95% band."""
    rng = _rng("prof", *key)
    point = np.mean([(v[:, None] > taus[None]).mean(0) for v in scores.values()], axis=0)
    acc = np.zeros((n_boot, taus.size))
    for v in scores.values():
        r = v[rng.integers(0, v.size, size=(n_boot, v.size))]
        acc += (r[:, :, None] > taus[None, None]).mean(1)
    acc /= len(scores)
    return point, np.percentile(acc, 2.5, axis=0), np.percentile(acc, 97.5, axis=0)


def _u_stat(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """P(X > Y) + 0.5 P(X = Y) along the last two axes (x: [..., n], y: [..., m])."""
    gt = (x[..., :, None] > y[..., None, :]).mean((-1, -2))
    eq = (x[..., :, None] == y[..., None, :]).mean((-1, -2))
    return gt + 0.5 * eq


def prob_improvement(xs: dict[str, np.ndarray], ys: dict[str, np.ndarray], n_boot: int, key):
    """Average over common tasks of P(X > Y); X and Y resampled independently within each task."""
    tasks = [t for t in xs if t in ys]
    if not tasks:
        return None
    rng = _rng("poi", *key)
    point = float(np.mean([_u_stat(xs[t], ys[t]) for t in tasks]))
    acc = np.zeros(n_boot)
    for t in tasks:
        x, y = xs[t], ys[t]
        acc += _u_stat(x[rng.integers(0, x.size, size=(n_boot, x.size))],
                       y[rng.integers(0, y.size, size=(n_boot, y.size))])
    acc /= len(tasks)
    return dict(p=point, lo=float(np.percentile(acc, 2.5)), hi=float(np.percentile(acc, 97.5)), tasks=tasks)


def welch(a, b) -> tuple[float, float]:
    """Two-sided Welch t-test -> (t, p). Zero variance in both samples: p = 1 if the means
    are equal, else 0 (a deterministic difference). NaN if either sample has < 2 runs."""
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return math.nan, math.nan
    if a.var(ddof=1) == 0 and b.var(ddof=1) == 0:
        d = a.mean() - b.mean()
        return (0.0, 1.0) if d == 0 else (math.copysign(math.inf, d), 0.0)
    with warnings.catch_warnings():      # near-constant samples (e.g. every seed at the return cap)
        warnings.simplefilter("ignore", RuntimeWarning)
        t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def outcome(a, b) -> tuple[str, float]:
    """'win' / 'tie' / 'loss' of sample a against sample b."""
    _, p = welch(a, b)
    if math.isnan(p):
        return "n/a", p
    if p < ALPHA:
        return ("win" if np.mean(a) > np.mean(b) else "loss"), p
    return "tie", p


# ------------------------------------------------------------------ per-task table
@dataclass
class Cell:
    n: int
    seeds: list
    finals: list
    aucs: list
    final: tuple
    auc: tuple
    wall_h: float
    devices: list
    budget: int


def per_task(data, n_boot: int) -> dict[str, dict[str, Cell]]:
    out = {}
    for env, rows in data.items():
        out[env] = {}
        for lab, runs in rows.items():
            finals, aucs = [r.final for r in runs], [r.auc for r in runs]
            out[env][lab] = Cell(n=len(runs), seeds=[r.seed for r in runs], finals=finals, aucs=aucs,
                                 final=boot_mean_ci(finals, ("final", env, lab), n_boot),
                                 auc=boot_mean_ci(aucs, ("auc", env, lab), n_boot),
                                 wall_h=float(np.nanmean([r.wall_h for r in runs])),
                                 devices=sorted({r.device for r in runs}), budget=runs[0].budget)
    return out


def normalisation(cells, randoms) -> dict[str, dict]:
    out = {}
    for env, row in cells.items():
        base = {b: row[b].final[0] for b in BASELINES if b in row}
        best_algo = max(base, key=base.get) if base else None
        rnd = randoms.get(env)
        best = base.get(best_algo) if best_algo else None
        ok = best is not None and rnd is not None and (best - rnd) > 1e-6 * max(1.0, abs(rnd))
        out[env] = dict(random=rnd, best_baseline=best, best_algo=best_algo, valid=bool(ok),
                        reason=None if ok else ("no baseline" if best is None else
                                                "no random score" if rnd is None else
                                                "best baseline not above random"))
    return out


def normalised_scores(cells, norm, label: str) -> dict[str, np.ndarray]:
    out = {}
    for env, row in cells.items():
        nz = norm[env]
        if not nz["valid"]:
            continue
        src = nz["best_algo"] if label == BEST else label
        if src in row:
            out[env] = (np.asarray(row[src].finals) - nz["random"]) / (nz["best_baseline"] - nz["random"])
    return out


def raw_finals(cells, norm, label: str) -> dict[str, np.ndarray]:
    out = {}
    for env, row in cells.items():
        src = norm[env]["best_algo"] if label == BEST else label
        if src and src in row:
            out[env] = np.asarray(row[src].finals, float)
    return out


# ------------------------------------------------------------------ formatting helpers
def _decimals(values) -> int:
    v = [abs(x) for x in values if x is not None and np.isfinite(x)]
    m = max(v) if v else 0
    return 0 if m >= 100 else 1 if m >= 10 else 2


def _num(x: float, d: int, tex: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    s = f"{x:.{d}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "$-$") if tex else s


def _overlaps(a: tuple, b: tuple) -> bool:
    lo_a, hi_a = (a[1], a[2]) if np.isfinite(a[1]) else (a[0], a[0])
    lo_b, hi_b = (b[1], b[2]) if np.isfinite(b[1]) else (b[0], b[0])
    return lo_a <= hi_b and lo_b <= hi_a


def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")


def md_table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    sep = ["---:" if a == "r" else ":---:" if a == "c" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ (2) main results tables
def score_tables(cells, labels, metric: str, n_seeds_expected: int):
    """Markdown and LaTeX tables of mean [95% CI] per algorithm per task, best in bold."""
    head_md = ["Environment", "Budget"] + labels + ["Seeds"]
    rows_md, body_tex = [], []
    prev_fam = None
    for env, row in cells.items():
        pool = expected_pool(env)
        stats = {l: getattr(row[l], metric) for l in labels if l in row}
        best = max(stats, key=lambda l: stats[l][0]) if stats else None
        bold = {l for l in stats if _overlaps(stats[l], stats[best])} if best else set()
        d = _decimals([s[0] for s in stats.values()])
        md, tex = [], []
        for l in labels:
            if l not in row:
                missing = "pending" if l.split("[")[0] in pool and l in ALGO_ORDER else "n/a"
                md.append(missing)
                tex.append(r"\textemdash" if missing == "n/a" else r"\textit{pending}")
                continue
            m, lo, hi = stats[l]
            ci_md = f"[{_num(lo, d)}, {_num(hi, d)}]" if np.isfinite(lo) else "[n=1]"
            ci_tex = f"[{_num(lo, d, True)}, {_num(hi, d, True)}]" if np.isfinite(lo) else "[n=1]"
            mm, mt = _num(m, d), _num(m, d, True)
            md.append(f"**{mm}** {ci_md}" if l in bold else f"{mm} {ci_md}")
            tex.append((rf"\textbf{{{mt}}}" if l in bold else mt) + rf" {{\scriptsize {ci_tex}}}")
        ns = [row[l].n if l in row else None for l in labels]
        present = [n for n in ns if n is not None]
        seeds = (str(present[0]) if present and len(set(present)) == 1
                 else "/".join("-" if n is None else str(n) for n in ns))
        budget = fmt_steps(TASK_INFO[env][0] if env in TASK_INFO else max(c.budget for c in row.values()))
        rows_md.append([env, budget] + md + [seeds])
        fam = family(env)
        if fam != prev_fam:
            if prev_fam is not None:
                body_tex.append(r"\midrule")
            body_tex.append(rf"\multicolumn{{{len(labels) + 3}}}{{@{{}}l}}{{\emph{{{FAMILY_NAMES[fam]}}}}} \\")
            prev_fam = fam
        body_tex.append(" & ".join([_tex_escape(env), budget] + tex + [seeds]) + r" \\")
    what = "Final return (mean of the last three evaluations, 30 episodes)" if metric == "final" \
        else "Area under the learning curve (mean return over all evaluations)"
    note = (f"{what}: mean over seeds [95% percentile-bootstrap CI]. Bold: best mean and every "
            f"algorithm whose CI overlaps the best one's. Seeds: runs per algorithm (a single number "
            f"when all columns agree, else per column in order, '-' = none); the protocol uses "
            f"{n_seeds_expected} seeds (10-14). 'pending' = planned, not finished; 'n/a' = not run on "
            f"this action space.")
    md = f"{md_table(head_md, rows_md)}\n{note}\n"
    cols = "@{}ll" + "c" * len(labels) + "c@{}"
    tex = "\n".join([
        "% Generated by `python -m benchmarks.report`; requires \\usepackage{booktabs,graphicx}.",
        r"\begin{table*}[t]", r"\centering",
        rf"\caption{{{_tex_escape(note)}}}", rf"\label{{tab:{metric}-scores}}",
        r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
        " & ".join(["Environment", "Budget"] + [_tex_escape(l) for l in labels] + ["Seeds"]) + r" \\",
        r"\midrule", *body_tex, r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])
    return md, tex


# ------------------------------------------------------------------ (4) win / tie / loss
def win_tie_loss(cells, norm, tempo="TEMPO"):
    per_task, counts = [], defaultdict(lambda: dict(win=0, tie=0, loss=0))
    for env, row in cells.items():
        if tempo not in row:
            continue
        rec = dict(env=env, tempo_mean=row[tempo].final[0], tempo_n=row[tempo].n, vs={})
        for b in list(BASELINES) + [BEST]:
            src = norm[env]["best_algo"] if b == BEST else b
            if not src or src not in row:
                continue
            res, p = outcome(row[tempo].finals, row[src].finals)
            rec["vs"][b] = dict(algo=src, mean=row[src].final[0], n=row[src].n, p=p, result=res)
            if res in ("win", "tie", "loss"):
                counts[b][res] += 1
        per_task.append(rec)
    return per_task, {k: dict(v) for k, v in counts.items()}


def wtl_outputs(per_task, counts):
    if not per_task:
        msg = "No TEMPO results in this suite yet: win/tie/loss not computed.\n"
        return msg, "% " + msg
    cols = [b for b in list(BASELINES) + [BEST] if b in counts]
    summ_rows = [[b if b != BEST else "best baseline per task",
                  str(counts[b]["win"]), str(counts[b]["tie"]), str(counts[b]["loss"]),
                  str(sum(counts[b].values()))] for b in cols]
    md = ["## TEMPO vs. baselines: win / tie / loss (final score)\n",
          f"Two-sided Welch t-test on the per-seed final scores, alpha = {ALPHA}; win/loss = significant "
          "difference in TEMPO's favour/against, tie = not significant. Tasks where either side has "
          "fewer than two seeds are not counted. The best baseline is the SB3 baseline with the "
          "highest mean final score on that task.\n",
          md_table(["Opponent", "Win", "Tie", "Loss", "Tasks"], summ_rows), "\n### Per task\n"]
    rows = []
    for rec in per_task:
        vb = rec["vs"].get(BEST)
        d = _decimals([rec["tempo_mean"]] + [v["mean"] for v in rec["vs"].values()])
        cells = [rec["env"], f"{_num(rec['tempo_mean'], d)} (n={rec['tempo_n']})"]
        if vb:
            cells += [f"{vb['algo']} {_num(vb['mean'], d)} (n={vb['n']})", _pfmt(vb["p"]), vb["result"]]
        else:
            cells += ["-", "-", "-"]
        cells += [rec["vs"][b]["result"] if b in rec["vs"] else "" for b in BASELINES]
        rows.append(cells)
    md.append(md_table(["Environment", "TEMPO", "Best baseline", "p", "vs best"] + list(BASELINES), rows,
                       align="lrrrc" + "c" * len(BASELINES)))
    tex_rows = [" & ".join([_tex_escape(r[0])] + [re.sub(r"(?<![\w.])-(?=\d)", "$-$", x.replace("_", r"\_"))
                                                 .replace("<", "$<$") for x in r[1:]])
                + r" \\" for r in rows]
    tex = "\n".join([
        "% Generated by `python -m benchmarks.report`; requires \\usepackage{booktabs}.",
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{TEMPO vs.\ the SB3 baselines, final score: win/tie/loss by a two-sided Welch "
        rf"$t$-test on per-seed final scores ($\alpha={ALPHA}$).}}", r"\label{tab:wtl}",
        r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule", r"Opponent & Win & Tie & Loss & Tasks \\", r"\midrule",
        *[" & ".join(r) + r" \\" for r in summ_rows], r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
        r"\begin{table*}[t]", r"\centering\small",
        r"\caption{Per-task comparison of TEMPO with each SB3 baseline (Welch $t$-test, "
        rf"$\alpha={ALPHA}$).}}", r"\label{tab:wtl-per-task}",
        r"\begin{tabular}{@{}lrrrc" + "c" * len(BASELINES) + "@{}}", r"\toprule",
        " & ".join(["Environment", "TEMPO", "Best baseline", "$p$", "vs best"] + list(BASELINES)) + r" \\",
        r"\midrule", *tex_rows, r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(md), tex


def _pfmt(p: float) -> str:
    return "n/a" if not np.isfinite(p) else "<0.001" if p < 1e-3 else f"{p:.3f}"


# ------------------------------------------------------------------ (5) compute table
def compute_tables(cells, labels):
    rows, tex_rows = [], []
    for env, row in cells.items():
        md, tex = [], []
        for l in labels:
            if l not in row:
                md.append("-")
                tex.append(r"\textemdash")
                continue
            c = row[l]
            mark = "" if c.devices == ["cpu"] else " (" + "+".join(c.devices) + ")"
            h = f"{c.wall_h:.2f}" if c.wall_h >= 0.1 else f"{c.wall_h:.2g}"
            md.append(f"{h}{mark}")
            tex.append(h + (r"$^\dagger$" if any(d != "cpu" for d in c.devices) else ""))
        rows.append([env, fmt_steps(TASK_INFO[env][0]) if env in TASK_INFO else ""] + md)
        tex_rows.append(" & ".join([_tex_escape(env), rows[-1][1]] + tex) + r" \\")
    note = ("Mean wall-clock hours per run (one process per run, 1 CPU thread). Runs shared one machine "
            "with other concurrent jobs, so times include contention and are comparable only roughly. "
            "Device other than CPU in parentheses (mps = Apple GPU, cuda = NVIDIA GPU; the neural networks "
            "ran there, the environments on CPU).")
    tex_note = (r"Mean wall-clock hours per run (one process per run, 1 CPU thread). Runs shared one machine "
                r"with other concurrent jobs, so times include contention and are comparable only roughly. "
                r"$^\dagger$Networks on the GPU (Apple MPS or NVIDIA CUDA), environments on CPU.")
    md = md_table(["Environment", "Budget"] + labels, rows) + "\n" + note + "\n"
    tex = "\n".join([
        "% Generated by `python -m benchmarks.report`; requires \\usepackage{booktabs}.",
        r"\begin{table}[t]", r"\centering\small",
        rf"\caption{{{tex_note}}}",
        r"\label{tab:compute}", r"\begin{tabular}{@{}ll" + "r" * len(labels) + "@{}}", r"\toprule",
        " & ".join(["Environment", "Budget"] + [_tex_escape(l) for l in labels]) + r" \\", r"\midrule",
        *tex_rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return md, tex


# ------------------------------------------------------------------ data checks
def data_checks(data, cells, norm, randoms, suite: str) -> list[tuple[str, str]]:
    """Heuristic scan for failed / diverged / suspicious runs. Returns [(severity, message)]."""
    out = []
    expected = set(FINAL_SEEDS) if suite == "final" else None
    for env, rows in data.items():
        nz = norm[env]
        scale = (nz["best_baseline"] - nz["random"]) if nz["valid"] else None
        rnd = randoms.get(env)
        for lab, runs in rows.items():
            c = cells[env][lab]
            seeds = {r.seed for r in runs}
            if expected and lab.split("[")[0] in ALGO_ORDER and seeds != expected:
                miss = sorted(expected - seeds)
                if miss:
                    out.append(("info", f"{env} {lab}: {len(runs)}/{len(expected)} seeds (missing {miss})"))
            for r in runs:
                exp_evals = r.budget // r.eval_freq
                if r.n_dropped:
                    out.append(("warn", f"{env} {lab} s{r.seed}: ran to {r.raw_last_step} steps for a budget "
                                        f"of {r.budget} ({r.n_dropped} evaluation(s) after the budget "
                                        "discarded by this report)"))
                if len(r.steps) != exp_evals or r.steps[-1] != r.budget:
                    out.append(("warn", f"{env} {lab} s{r.seed}: {len(r.steps)} evaluations up to step "
                                        f"{r.steps[-1]} (expected {exp_evals} up to {r.budget})"))
                n_ep = {len(e) for e in r.episodes}
                if n_ep != {r.n_eval}:
                    out.append(("warn", f"{env} {lab} s{r.seed}: episodes per evaluation {sorted(n_ep)} "
                                        f"(expected {r.n_eval})"))
                if not np.all(np.isfinite(np.concatenate([np.asarray(e, float) for e in r.episodes]))):
                    out.append(("error", f"{env} {lab} s{r.seed}: non-finite evaluation returns"))
                if rnd is not None and r.final < rnd:
                    out.append(("warn", f"{env} {lab} s{r.seed}: final {r.final:.4g} is below the random "
                                        f"policy ({rnd:.4g})"))
                if scale:
                    cur = r.curve
                    peak = float(cur.max())
                    if (peak - r.final) / scale > 0.5:
                        out.append(("warn", f"{env} {lab} s{r.seed}: late collapse - peak evaluation "
                                            f"{peak:.4g} at step {int(r.steps[cur.argmax()])}, final "
                                            f"{r.final:.4g}"))
            if scale and len(runs) >= 3:
                f = np.asarray(c.finals)
                med = np.median(f)
                for r, x in zip(runs, f):
                    if abs(x - med) / scale > 0.5:
                        out.append(("warn", f"{env} {lab} s{r.seed}: outlier seed, final {x:.4g} vs median "
                                            f"{med:.4g} of the {len(f)} seeds (seeds: "
                                            + ", ".join(f"{v:.4g}" for v in f) + ")"))
                ns = (c.final[0] - nz["random"]) / scale
                if ns < 0.1:
                    out.append(("warn", f"{env} {lab}: mean final {c.final[0]:.4g} = {ns:.2f} on the "
                                        f"random ({nz['random']:.4g}) -> best-baseline ({nz['best_baseline']:.4g}) "
                                        "scale: little or no learning within the budget"))
            curves = [tuple(np.round(r.curve, 6)) for r in runs]
            # A curve flat at one value (e.g. the task's maximum return at every evaluation, as
            # CrossQ on InvertedPendulum-v5) is the same for every seed that reaches it: not a
            # sign of duplicated data. Only curves that vary are compared.
            varying = [cv for cv in curves if len(set(cv)) > 1]
            if len(set(varying)) < len(varying):
                out.append(("error", f"{env} {lab}: two or more seeds have identical learning curves"))
            flat = sorted({cv[0] for cv in curves if len(set(cv)) == 1})
            for v in flat:
                k = sum(1 for cv in curves if len(set(cv)) == 1 and cv[0] == v)
                if k > 1:
                    out.append(("info", f"{env} {lab}: {k} seeds score {v:.4g} at every evaluation "
                                        "(identical flat curves)"))
            tuned = {r.info.get("zoo_tuned") for r in runs if "zoo_tuned" in r.info}
            if False in tuned:
                out.append(("info", f"{env} {lab}: no RL-Zoo entry for this task; SB3 default "
                                    "hyperparameters were used"))
            wall = np.array([r.wall_h for r in runs])
            if len(runs) >= 2 and np.nanmin(wall) > 0.05 and np.nanmax(wall) / np.nanmin(wall) > 1.6:
                out.append(("info", f"{env} {lab}: wall-clock varies {np.nanmin(wall):.2f}-"
                                    f"{np.nanmax(wall):.2f} h across seeds (machine contention)"))
            if lab.startswith("TEMPO"):
                src = {r.tempo_source for r in runs}
                roots = {pathlib.Path(r.code_root).name for r in runs}
                if len(src) > 1:
                    out.append(("warn", f"{env} {lab}: runs used {len(src)} different tempo_rl sources"))
                if suite == "final" and roots - {"final", ROOT.name}:
                    out.append(("warn", f"{env} {lab}: TEMPO code root(s) {sorted(roots)} (expected the "
                                        "frozen snapshot benchmarks/snapshots/final or the live library)"))
                elif suite == "final" and ROOT.name in roots:
                    # PROTOCOL.md, note of 2026-09-26: later final runs use the live
                    # library, whose default code path equals the archived snapshot
                    out.append(("info", f"{env} {lab}: "
                                        + ("mixes runs from the frozen snapshot and the live library"
                                           if "final" in roots else "runs from the live library")
                                        + " (PROTOCOL.md, 2026-09-26)"))
            vers = {(r.info.get("sb3_version"), r.info.get("sb3_contrib_version")) for r in runs
                    if "sb3_version" in r.info}
            if len(vers) > 1:
                out.append(("warn", f"{env} {lab}: mixed SB3 versions {sorted(vers)}"))
    for env, nz in norm.items():
        if not nz["valid"] and env in data and any(l in BASELINES for l in data[env]):
            out.append(("warn", f"{env}: excluded from the aggregates ({nz['reason']})"))
    order = {"error": 0, "warn": 1, "info": 2}
    return sorted(out, key=lambda x: order[x[0]])


# ------------------------------------------------------------------ figures
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7.5, "font.family": "sans-serif",
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK2, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "savefig.dpi": 220, "figure.dpi": 110, "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "butt",
    })
    return plt


def _steps_formatter():
    from matplotlib.ticker import FuncFormatter

    def f(x, _):
        if x == 0:
            return "0"
        return f"{x / 1e6:g}M" if abs(x) >= 1e6 else f"{x / 1e3:g}k"

    return FuncFormatter(f)


def curve_band(runs: list[Run], n_boot: int, key):
    grid = np.array(sorted(set().union(*[set(r.steps.tolist()) for r in runs])), float)
    Y = np.full((len(runs), grid.size), np.nan)
    for i, r in enumerate(runs):
        m = (grid >= r.steps[0]) & (grid <= r.steps[-1])
        Y[i, m] = np.interp(grid[m], r.steps, r.curve)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(Y, 0)
        n = (~np.isnan(Y)).sum(0)
        if len(runs) >= 2:
            boots = np.nanmean(Y[_rng("curve", *key).integers(0, len(runs), size=(n_boot, len(runs)))], axis=1)
            lo, hi = np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)
        else:
            lo = hi = np.full(grid.size, np.nan)
    lo[n < 2], hi[n < 2] = np.nan, np.nan
    return grid, mean, lo, hi


def _draw_task(ax, env, rows, n_boot):
    from matplotlib.ticker import MaxNLocator

    for lab, runs in rows.items():
        st = style(lab)
        x, m, lo, hi = curve_band(runs, n_boot, (env, lab))
        ok = np.isfinite(lo)
        if ok.any():
            ax.fill_between(x, lo, hi, where=ok, color=st["color"], alpha=0.14, lw=0, zorder=st["z"] - 0.5)
        ax.plot(x, m, color=st["color"], ls=st["ls"], lw=st["lw"], zorder=st["z"], label=lab)
    budget = TASK_INFO[env][0] if env in TASK_INFO else None
    if budget:
        ax.set_xlim(0, budget)
    ax.xaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 2.5, 5, 10]))
    ax.xaxis.set_major_formatter(_steps_formatter())
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_title(env, pad=3)


def _legend_handles(labels, rename=None):
    from matplotlib.lines import Line2D

    rename = rename or {}
    return [Line2D([], [], color=style(l)["color"], ls=style(l)["ls"], lw=style(l)["lw"] + 0.3,
                   label=rename.get(l, l)) for l in labels]


def _save(fig, path: pathlib.Path, written: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(path.with_suffix(ext), bbox_inches="tight", pad_inches=0.02,
                    facecolor="white")
        written.append(path.with_suffix(ext))


def plot_curves(data, figures: pathlib.Path, suite: str, n_boot: int, written: list):
    plt = _mpl()
    envs = list(data)
    if not envs:
        return
    labels = sorted({l for rows in data.values() for l in rows}, key=label_key)
    ncol = 4 if len(envs) > 6 else min(3, len(envs))
    nrow = math.ceil(len(envs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 1.55 * nrow + 0.35), squeeze=False)
    for ax, env in zip(axes.flat, envs):
        _draw_task(ax, env, data[env], n_boot)
    for ax in axes.flat[len(envs):]:
        ax.set_visible(False)
    for i in range(nrow):
        axes[i, 0].set_ylabel("Return")
    for ax in axes.flat[max(0, len(envs) - ncol):len(envs)]:
        ax.set_xlabel("Environment steps")
    fig.tight_layout(h_pad=0.8, w_pad=0.6, rect=(0, 0, 1, 1 - 0.32 / (1.55 * nrow + 0.35)))
    fig.legend(handles=_legend_handles(labels), loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.4)
    _save(fig, figures / f"{suite}_curves", written)
    plt.close(fig)
    for env in envs:
        fig, ax = plt.subplots(figsize=(3.3, 2.35))
        _draw_task(ax, env, data[env], n_boot)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel("Return")
        labs = list(data[env])
        ax.legend(handles=_legend_handles(labs), loc="lower center", bbox_to_anchor=(0.5, 1.08),
                  ncol=min(len(labs), 3), handlelength=2.4, columnspacing=1.0, borderaxespad=0)
        _save(fig, figures / "curves" / f"{suite}_{env}", written)
        plt.close(fig)


def _interval_rows(ax, items, xref, xlabel):
    """items: [(label, point, lo, hi)] top to bottom."""
    for i, (lab, p, lo, hi) in enumerate(items):
        st = style(lab)
        y = len(items) - 1 - i
        if np.isfinite(lo):
            ax.plot([lo, hi], [y, y], color=st["color"], lw=6, alpha=0.45, solid_capstyle="butt", zorder=2)
        ax.plot([p], [y], marker="|", ms=11, mew=2.0, color=st["color"], zorder=3)
    ax.axvline(xref, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.set_ylim(-0.6, len(items) - 0.4)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def plot_aggregate(agg, profiles, taus, poi, figures: pathlib.Path, written: list, tempo="TEMPO"):
    plt = _mpl()
    fig = plt.figure(figsize=(7.0, 5.1))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.15], hspace=0.55, wspace=0.12, top=0.86)
    labels = [l for l in agg]            # already ordered
    names = [f"{l if l != BEST else 'Best baseline*'} ({len(agg[l]['tasks'])})" for l in labels]
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for k, (ax, stat) in enumerate(zip(axes, ("IQM", "median", "mean"))):
        items = [(l,) + tuple(agg[l][stat]) for l in labels]
        _interval_rows(ax, items, 1.0, {"IQM": "IQM", "median": "Median", "mean": "Mean"}[stat]
                       + " normalised score")
        ax.set_yticks(range(len(labels))[::-1])
        ax.set_yticklabels(names if k == 0 else [])
    axes[0].set_title("Aggregate final score (95% stratified-bootstrap CI)", loc="left", pad=6, fontsize=8)
    axp = fig.add_subplot(gs[1, :2])
    for l in labels:
        st = style(l)
        p, lo, hi = profiles[l]
        axp.fill_between(taus, lo, hi, color=st["color"], alpha=0.12, lw=0, step="post", zorder=st["z"] - 0.5)
        axp.step(taus, p, where="post", color=st["color"], ls=st["ls"], lw=st["lw"], zorder=st["z"],
                 label=l if l != BEST else "Best baseline*")
    axp.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    axp.set_xlim(taus[0], taus[-1])
    axp.set_ylim(0, 1.02)
    axp.set_xlabel(r"Normalised score $\tau$")
    axp.set_ylabel(r"Fraction of runs with score $> \tau$")
    axp.set_title("Performance profile", loc="left", fontsize=8)
    fig.legend(handles=_legend_handles(labels, {BEST: "Best baseline*"}), loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=min(len(labels), 7), handlelength=2.6, columnspacing=1.3)
    axq = fig.add_subplot(gs[1, 2])
    if poi:
        items = [(b, v["p"], v["lo"], v["hi"]) for b, v in poi.items()]
        _interval_rows(axq, items, 0.5, f"P({tempo} > X)")
        axq.set_yticks(range(len(items))[::-1])
        axq.set_yticklabels([f"{b if b != BEST else 'Best baseline*'} ({len(poi[b]['tasks'])})" for b in poi])
        axq.yaxis.tick_right()
        axq.set_xlim(0, 1)
    else:
        axq.text(0.5, 0.5, f"no {tempo} runs yet", ha="center", va="center", color=INK2,
                 transform=axq.transAxes)
        axq.set_xticks([])
        axq.set_yticks([])
    axq.set_title("Probability of improvement", loc="left", fontsize=8)
    fig.text(0.0, -0.02, "Scores normalised per task as (x - random) / (best SB3 baseline mean - random); "
             "dashed line = the best SB3 baseline on each task.\n(k) = number of tasks. *Best baseline = "
             "the SB3 baseline with the highest mean final score on each task.", fontsize=6.5, color=INK2,
             ha="left", va="top")
    _save(fig, figures / "aggregate", written)
    plt.close(fig)


# ------------------------------------------------------------------ paper tables, README block
N_SEEDS = len(FINAL_SEEDS)
#: critic hidden layers of a run that recorded neither its resolved architecture
#: (``info["net_arch"]``, written by run_one since 2026-09-30) nor ``net_arch`` in its config
DEFAULT_QF = {"CrossQ": (1024, 1024),     # sb3-contrib 2.9.0 CrossQ with net_arch=None
              "TEMPO": (256, 256)}        # tempo_rl's DEFAULT_ARCH before 2026-09-30 (C1: 1024)
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
#: scale a wide table down to the text width, never up (graphicx provides \\width)
SHRINK_TO_WIDTH = r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%"
#: figures that paper/RL.tex includes (copied by --paper-figures)
PAPER_FIGURES = ("final_tempo_curves",)


def is_tempo(label: str) -> bool:
    return label.split("[")[0] == "TEMPO"


def critic_arch(run: Run) -> tuple:
    """Hidden widths of the run's critic: the resolved architecture recorded by run_one, else
    the recorded config, else the default of the algorithm at the time such runs were made."""
    resolved = run.info.get("net_arch")
    if isinstance(resolved, dict) and resolved.get("qf"):
        return tuple(resolved["qf"])
    pk = (run.info.get("config") or {}).get("policy_kwargs")
    na = pk.get("net_arch") if isinstance(pk, dict) else None
    if isinstance(na, dict):
        return tuple(na.get("qf", ()))
    if isinstance(na, (list, tuple)):
        return tuple(na)
    return DEFAULT_QF.get(run.algo, ())


def fmt_arch(arch, tex: bool = False) -> str:
    if not arch:
        return "?"
    x = r"{\times}" if tex else "x"
    s = f"{len(arch)}{x}{arch[0]}" if len(set(arch)) == 1 else x.join(map(str, arch))
    return f"${s}$" if tex else s


def label_arch(runs: list, tex: bool = False) -> str:
    archs = {critic_arch(r) for r in runs}
    return " / ".join(fmt_arch(a, tex) for a in sorted(archs)) if archs else "?"


def _join(items: list[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _seed_ranges(seeds) -> str:
    s = sorted(seeds)
    if len(s) > 1 and s == list(range(s[0], s[-1] + 1)):
        return f"seeds {s[0]}-{s[-1]}"
    return ("seed " if len(s) == 1 else "seeds ") + ", ".join(map(str, s))


def suite_status(data, labels) -> dict:
    """Which final-benchmark runs exist: TEMPO per task, missing baseline runs."""
    tasks = {}
    for env in TASK_INFO:
        rows = data.get(env, {})
        tasks[env] = {l: sorted({r.seed for r in rows.get(l, [])} & set(FINAL_SEEDS))
                      for l in labels if is_tempo(l)}
    tempo = {}
    for l in [l for l in labels if is_tempo(l)]:
        n = {env: len(v[l]) for env, v in tasks.items()}
        if l != "TEMPO" and not any(n.values()):
            continue
        tempo[l] = dict(
            planned=len(TASK_INFO) * N_SEEDS if l == "TEMPO" else None,
            finished=int(sum(n.values())),
            complete=[e for e, k in n.items() if k >= N_SEEDS],
            partial={e: k for e, k in n.items() if 0 < k < N_SEEDS},
            pending=[e for e, k in n.items() if k == 0])
    planned, missing = 0, []
    for env in SUITE:
        for algo in [a for a in ALGO_ORDER if a in expected_pool(env) and a != "TEMPO"]:
            have = {r.seed for r in data.get(env, {}).get(algo, [])}
            planned += N_SEEDS
            miss = sorted(set(FINAL_SEEDS) - have)
            if miss:
                missing.append(dict(algo=algo, env=env, seeds=miss))
    return dict(tempo=tempo, baselines=dict(planned=planned,
                                            finished=planned - sum(len(m["seeds"]) for m in missing),
                                            missing=missing),
                n_tasks=len(TASK_INFO), n_seeds=N_SEEDS, seeds=list(FINAL_SEEDS))


def _missing_baselines_text(missing: list) -> str:
    by_algo = defaultdict(list)
    for m in missing:
        by_algo[m["algo"]].append(f"{m['env']} ({_seed_ranges(m['seeds'])})")
    return "; ".join(f"{a} on {_join(v)}" for a, v in by_algo.items())


def _score_cell(c, d: int, tex: bool, bold: bool) -> str:
    """mean [CI] with an (n=k) mark below the protocol's seed count."""
    m, lo, hi = c.final
    ci = (f"[{_num(lo, d, tex)}, {_num(hi, d, tex)}]" if np.isfinite(lo) else "")
    nk = "" if c.n >= N_SEEDS else (rf"($n{{=}}{c.n}$)" if tex else f"(n={c.n})")
    mm = _num(m, d, tex)
    if tex:
        mm = rf"\textbf{{{mm}}}" if bold else mm
        small = " ".join(x for x in (ci, nk) if x)
        return mm + (rf" {{\scriptsize {small}}}" if small else "")
    mm = f"**{mm}**" if bold else mm
    return " ".join(x for x in (mm, ci, nk) if x)


def _bold_set(row: dict, labels) -> set:
    """Best mean among entries with all seeds, and every such entry whose CI overlaps it."""
    full = {l: row[l].final for l in labels if l in row and row[l].n >= N_SEEDS}
    if not full:
        return set()
    best = max(full, key=lambda l: full[l][0])
    return {l for l in full if _overlaps(full[l], full[best])}


def _missing_cell(env: str, label: str, tex: bool) -> str:
    planned = label in ALGO_ORDER and label in expected_pool(env)
    if tex:
        return "--" if planned else r"{\scriptsize n/a}"
    return "pending" if planned else "n/a"


def _vs(row: dict, a: str, b: str | None) -> tuple[str, float]:
    if b is None or a not in row or b not in row:
        return "", math.nan
    return outcome(row[a].finals, row[b].finals)


def paper_main_table(cells, norm, labels) -> str:
    """Every task of the plan (with or without runs) x every algorithm, final score."""
    body, prev = [], None
    for env in TASK_INFO:
        row = cells.get(env, {})
        bold = _bold_set(row, labels)
        d = _decimals([row[l].final[0] for l in labels if l in row])
        tex = [_score_cell(row[l], d, True, l in bold) if l in row else _missing_cell(env, l, True)
               for l in labels]
        best = norm.get(env, {}).get("best_algo")
        if "TEMPO" not in row:
            vs = "--" if "TEMPO" in expected_pool(env) else ""
        elif env in TOY_TEXT or best is None:
            vs = r"{\scriptsize no baseline}"
        else:
            res, _ = _vs(row, "TEMPO", best)
            vs = rf"{{\scriptsize untested ({best})}}" if res == "n/a" else res + rf" {{\scriptsize {best}}}"
        fam = family(env)
        if fam != prev:
            if prev is not None:
                body.append(r"\midrule")
            body.append(rf"\multicolumn{{{len(labels) + 3}}}{{@{{}}l}}{{\emph{{{FAMILY_NAMES[fam]}}}}} \\")
            prev = fam
        body.append(" & ".join([_tex_escape(env), fmt_steps(TASK_INFO[env][0])] + tex + [vs]) + r" \\")
    caption = (
        r"Final return on every task of the benchmark plan: mean of the last three evaluations "
        r"(30 episodes) per run, mean over seeds with a 95\,\% percentile-bootstrap CI. "
        rf"Entries have the protocol's {N_SEEDS} seeds ({FINAL_SEEDS[0]}--{FINAL_SEEDS[-1]}) unless marked "
        r"($n{=}k$); -- = planned but not run yet; n/a = not run on this action space. Bold: the best "
        rf"mean among entries with all {N_SEEDS} seeds and every such entry whose CI overlaps it. "
        r"Last column: TEMPO against the best finished SB3 baseline on the task (named), two-sided "
        rf"Welch $t$-test on per-seed final scores, $\alpha={ALPHA}$; untested = fewer than two TEMPO "
        r"seeds.")
    cols = "@{}ll" + "r" * len(labels) + "l@{}"
    return "\n".join([
        "% Generated by `python -m benchmarks.report --suite final`; requires booktabs, graphicx.",
        r"\begin{table}[tbp]", r"\centering", rf"\caption{{{caption}}}", r"\label{tab:main}",
        SHRINK_TO_WIDTH, rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
        " & ".join(["Task", "Budget"] + [_tex_escape(l) for l in labels] + ["TEMPO vs.\\ best"]) + r" \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""])


def complete_comparison(cells, norm, n_boot: int, tempo: str = "TEMPO") -> dict:
    """TEMPO vs. each baseline on the tasks where TEMPO has all seeds (and X at least two)."""
    complete = [e for e, row in cells.items() if tempo in row and row[tempo].n >= N_SEEDS
                and e in norm and norm[e]["valid"]]
    out = {}
    for b in list(BASELINES) + [BEST]:
        src, common = {}, []
        for env in complete:
            s = norm[env]["best_algo"] if b == BEST else b
            if s and s in cells[env] and cells[env][s].n >= 2:
                src[env] = s
                common.append(env)
        if not common:
            out[b] = dict(tasks=[])
            continue

        def ns(env, lab):
            nz = norm[env]
            return (np.asarray(cells[env][lab].finals) - nz["random"]) / (nz["best_baseline"] - nz["random"])

        per = {}
        for env in common:
            res, p = outcome(cells[env][tempo].finals, cells[env][src[env]].finals)
            per[env] = dict(algo=src[env], result=res, p=p, n=cells[env][src[env]].n)
        out[b] = dict(
            tasks=common, per_task=per,
            wtl={k: sum(v["result"] == k for v in per.values()) for k in ("win", "tie", "loss")},
            tempo=aggregate_ci({e: ns(e, tempo) for e in common}, n_boot, (tempo, "complete", b)),
            other=aggregate_ci({e: ns(e, src[e]) for e in common}, n_boot, (b, "complete", tempo)),
            poi=prob_improvement({e: np.asarray(cells[e][tempo].finals, float) for e in common},
                                 {e: np.asarray(cells[e][src[e]].finals, float) for e in common},
                                 n_boot, (tempo, "complete", b)))
    return dict(tempo=tempo, complete_tasks=complete, vs=out)


def _ci3(v, tex: bool = False) -> str:
    s = f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"
    return s.replace("-", "$-$") if tex else s


def paper_aggregate_table(comps: list[dict]) -> str:
    multi = len(comps) > 1           # a first column naming the TEMPO configuration
    rows = []
    for comp in comps:
        t = comp["tempo"]
        lead = [_tex_escape(t)] if multi else []
        for b, v in comp["vs"].items():
            name = "best baseline per task" if b == BEST else b
            if not v["tasks"]:
                rows.append(" & ".join(lead + [name, "0", rf"\multicolumn{{4}}{{l}}{{\scriptsize no task "
                                               rf"with {N_SEEDS} {_tex_escape(t)} seeds and a finished "
                                               rf"{name} run yet}}"]) + r" \\")
                continue
            w = v["wtl"]
            poi = v["poi"]
            rows.append(" & ".join(lead + [
                name, str(len(v["tasks"])), _ci3(v["tempo"]["IQM"], True),
                _ci3(v["other"]["IQM"], True), f"{poi['p']:.2f} [{poi['lo']:.2f}, {poi['hi']:.2f}]",
                f"{w['win']} / {w['tie']} / {w['loss']}"]) + r" \\")
    tasks = comps[0]["complete_tasks"] if comps else []
    caption = (
        rf"TEMPO against each baseline on the tasks where TEMPO has all {N_SEEDS} final seeds "
        rf"({_join([_tex_escape(e) for e in tasks]) or 'none yet'}) and the baseline at least two. "
        r"Scores are normalised per task as $(x - x_{\text{random}})/(x_{\text{best baseline}} - "
        r"x_{\text{random}})$; IQM = interquartile mean over runs with a 95\,\% stratified-bootstrap CI "
        r"(Agarwal et al., 2021); $P(\text{TEMPO} > X)$ = probability that a TEMPO run beats a run of $X$ "
        r"on a random common task, bootstrap CI; W/T/L = per-task two-sided Welch $t$-test, "
        rf"$\alpha={ALPHA}$. Tasks without all TEMPO seeds are not aggregated; Table~\ref{{tab:main}} "
        r"lists them.")
    head = (["Configuration"] if multi else []) + [
        r"Opponent $X$", "Tasks", "IQM TEMPO", r"IQM $X$", r"$P(\text{TEMPO} > X)$", "W / T / L"]
    return "\n".join([
        "% Generated by `python -m benchmarks.report --suite final`; requires booktabs.",
        r"\begin{table}[tbp]", r"\centering\small", rf"\caption{{{caption}}}", r"\label{tab:aggregate}",
        r"\begin{tabular}{@{}" + ("l" if multi else "") + "lrcccc@{}}", r"\toprule",
        " & ".join(head) + r" \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def crossq_rows(data, cells, randoms, tempo_labels):
    """Per task where CrossQ is planned: CrossQ vs. every TEMPO label."""
    out = []
    for env in SUITE:
        if "CrossQ" not in expected_pool(env):
            continue
        row = cells.get(env, {})
        rec = dict(env=env, crossq=row.get("CrossQ"),
                   crossq_arch=label_arch(data[env]["CrossQ"]) if "CrossQ" in row else "",
                   crossq_arch_tex=label_arch(data[env]["CrossQ"], True) if "CrossQ" in row else "",
                   tempo={})
        for t in tempo_labels:
            if t not in row:
                continue
            res, p = _vs(row, t, "CrossQ")
            rnd = randoms.get(env)
            rel = math.nan
            if "CrossQ" in row and rnd is not None and row["CrossQ"].final[0] - rnd > 1e-9:
                rel = (row[t].final[0] - rnd) / (row["CrossQ"].final[0] - rnd)
            rec["tempo"][t] = dict(cell=row[t], result=res, p=p, rel=rel)
        out.append(rec)
    return out


def paper_crossq_table(data, cells, randoms, tempo_labels) -> str:
    recs = crossq_rows(data, cells, randoms, tempo_labels)
    body = []
    counts = {t: dict(win=0, tie=0, loss=0) for t in tempo_labels}
    for rec in recs:
        cq = rec["crossq"]
        vals = [cq.final[0]] if cq else []
        vals += [v["cell"].final[0] for v in rec["tempo"].values()]
        d = _decimals(vals)
        cells_tex = [_tex_escape(rec["env"]), rec["crossq_arch_tex"] or "--",
                     _score_cell(cq, d, True, False) if cq else "--"]
        for t in tempo_labels:
            v = rec["tempo"].get(t)
            if v is None:
                cells_tex += ["--", "", "", ""]
                continue
            if v["result"] in counts[t]:
                counts[t][v["result"]] += 1
            cells_tex += [_score_cell(v["cell"], d, True, False),
                          "" if not np.isfinite(v["rel"]) else f"{v['rel']:.2f}",
                          "" if v["result"] in ("", "n/a") else _pfmt(v["p"]).replace("<", "$<$"),
                          {"": "", "n/a": r"{\scriptsize untested}"}.get(v["result"], v["result"])]
        body.append(" & ".join(cells_tex) + r" \\")
    tarch = {t: label_arch([r for rows in data.values() for r in rows.get(t, [])], True) for t in tempo_labels}
    head = ["Task", "CrossQ critic", "CrossQ"]
    sub = []
    for t in tempo_labels:
        head += [rf"\multicolumn{{4}}{{c}}{{{_tex_escape(t)} (critic {tarch[t]})}}"]
        sub += ["mean", "rel.", "$p$", "result"]
    summary = "; ".join(f"{_tex_escape(t)}: {c['win']} wins, {c['tie']} ties, {c['loss']} losses"
                        for t, c in counts.items())
    caption = (
        r"TEMPO against sb3-contrib CrossQ (RL-Zoo hyperparameters) on every continuous-action task. "
        r"CrossQ critic: hidden layers of the critic in the recorded run configuration (sb3-contrib "
        r"default when the Zoo sets none). Final return, mean over seeds with 95\,\% bootstrap CI; "
        rf"($n{{=}}k$) = fewer than {N_SEEDS} seeds; -- = not run yet. rel.\ = TEMPO's mean on the scale "
        r"random $=0$, CrossQ $=1$. $p$, result: two-sided Welch $t$-test on per-seed final scores, "
        rf"$\alpha={ALPHA}$. Summary over the tested tasks: {summary}.")
    return "\n".join([
        "% Generated by `python -m benchmarks.report --suite final`; requires booktabs, graphicx.",
        r"\begin{table}[tbp]", r"\centering", rf"\caption{{{caption}}}", r"\label{tab:crossq}",
        SHRINK_TO_WIDTH, r"\begin{tabular}{@{}llr" + "rrrl" * len(tempo_labels) + "@{}}",
        r"\toprule", " & ".join(head) + r" \\",
        "".join(rf"\cmidrule(l){{{4 + 4 * i}-{7 + 4 * i}}}" for i in range(len(tempo_labels))),
        " & ".join(["", "", ""] + sub) + r" \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}}",
        r"\end{table}", ""])


def plot_tempo_curves(data, figures: pathlib.Path, suite: str, n_boot: int, written: list):
    """Learning curves of the tasks where TEMPO has final runs (the paper's figure)."""
    envs = [e for e in data if any(is_tempo(l) for l in data[e])]
    if not envs:
        return
    plt = _mpl()
    labels = sorted({l for e in envs for l in data[e]}, key=label_key)
    ncol = min(4, len(envs))
    nrow = math.ceil(len(envs) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 1.7 * nrow + 0.35), squeeze=False)
    for ax, env in zip(axes.flat, envs):
        _draw_task(ax, env, data[env], n_boot)
        n = len(data[env].get("TEMPO", []))
        if 0 < n < N_SEEDS:
            ax.set_title(f"{env} (TEMPO: {n} seed{'s' if n > 1 else ''})", pad=3)
    for ax in axes.flat[len(envs):]:
        ax.set_visible(False)
    for i in range(nrow):
        axes[i, 0].set_ylabel("Return")
    for ax in axes.flat[max(0, len(envs) - ncol):len(envs)]:
        ax.set_xlabel("Environment steps")
    fig.tight_layout(h_pad=0.8, w_pad=0.6, rect=(0, 0, 1, 1 - 0.32 / (1.7 * nrow + 0.35)))
    fig.legend(handles=_legend_handles(labels), loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.4)
    _save(fig, figures / f"{suite}_tempo_curves", written)
    plt.close(fig)


def readme_block(data, cells, norm, labels, status, comps, crossq, figures_rel: str) -> str:
    """The generated Markdown between the RESULTS markers of README.md."""
    st, tb = status["tempo"].get("TEMPO", {}), status["baselines"]
    stamps = sorted(r.finished for rows in data.values() for runs in rows.values() for r in runs)
    lines = [
        "<!-- Generated by `python -m benchmarks.report --suite final --update-readme` from "
        "benchmarks/results/final. Do not edit by hand: the next run overwrites this block. -->", ""]
    # ---- status
    n_run = len(st.get("complete", [])) + len(st.get("partial", {}))
    part = [f"{k} of {N_SEEDS} seeds on {e}" for e, k in st.get("partial", {}).items()]
    s = (f"**Status.** Final TEMPO runs (library defaults, {_seed_ranges(FINAL_SEEDS)}) exist for "
         f"**{n_run} of {status['n_tasks']} tasks** ({st.get('finished', 0)} of {st.get('planned', 0)} "
         "planned runs)")
    have = ([f"all {N_SEEDS} seeds on {_join(st['complete'])}"] if st.get("complete") else []) + part
    s += (": " + "; ".join(have) + ".") if have else "."
    if st.get("pending"):
        s += f" **Not run yet:** {_join(st['pending'])}."
    s += (f" SB3 baselines: {tb['finished']} of {tb['planned']} planned runs"
          + (f"; missing {_missing_baselines_text(tb['missing'])}." if tb["missing"] else "."))
    for l, v in status["tempo"].items():
        if l != "TEMPO":
            s += f" {l}: {v['finished']} runs on {len(v['complete']) + len(v['partial'])} tasks."
    lines += [s, ""]
    # ---- summary on complete tasks
    for comp in comps:
        t = comp["tempo"]
        lines.append(f"**{t} vs. each baseline on the tasks where {t} has all {N_SEEDS} seeds** "
                     f"({_join(comp['complete_tasks']) or 'none yet'}). Win/tie/loss: two-sided Welch "
                     f"t-test on per-seed final scores, alpha = {ALPHA}. IQM: interquartile mean of scores "
                     "normalised per task as (x - random) / (best baseline - random), 95% stratified-"
                     "bootstrap CI. P(TEMPO > X): probability that a TEMPO run beats a run of X on a random "
                     "common task.\n")
        rows = []
        for b, v in comp["vs"].items():
            name = "best baseline per task" if b == BEST else b
            if not v["tasks"]:
                rows.append([name, "0", "-", "-", "-", "-", "-",
                             f"no task with {N_SEEDS} {t} seeds and a finished {name} run yet"])
                continue
            w = v["wtl"]
            losses = [e for e, x in v["per_task"].items() if x["result"] == "loss"]
            wins = [e for e, x in v["per_task"].items() if x["result"] == "win"]
            note = "; ".join(x for x in (("wins: " + ", ".join(wins)) if wins else "",
                                         ("losses: " + ", ".join(losses)) if losses else "") if x)
            rows.append([name, str(len(v["tasks"])), str(w["win"]), str(w["tie"]), str(w["loss"]),
                         _ci3(v["tempo"]["IQM"]), _ci3(v["other"]["IQM"]),
                         f"{v['poi']['p']:.2f} [{v['poi']['lo']:.2f}, {v['poi']['hi']:.2f}]"
                         + (f"; {note}" if note else "")])
        lines.append(md_table([f"{t} vs.", "Tasks", "Win", "Tie", "Loss", f"IQM {t}", "IQM opponent",
                               f"P({t} > X); tasks won / lost"], rows, align="lrrrrrrl"))
    # ---- per-task table
    lines.append("**Per task: final return** (mean of the last three evaluations = 30 episodes per run; "
                 f"mean over seeds [95% bootstrap CI]; {N_SEEDS} seeds unless marked (n=k); **bold**: best "
                 f"mean among entries with all {N_SEEDS} seeds and every such entry whose CI overlaps it; "
                 "pending = planned, not run yet; n/a = not run on this action space).\n")
    head = ["Task", "Budget"] + labels + ["TEMPO vs. best baseline", "TEMPO vs. CrossQ"]
    rows = []
    for env in TASK_INFO:
        row = cells.get(env, {})
        bold = _bold_set(row, labels)
        d = _decimals([row[l].final[0] for l in labels if l in row])
        r = [env, fmt_steps(TASK_INFO[env][0])]
        r += [_score_cell(row[l], d, False, l in bold) if l in row else _missing_cell(env, l, False)
              for l in labels]
        best = norm.get(env, {}).get("best_algo")
        for k, opp in enumerate((best, "CrossQ")):
            if k == 1 and "CrossQ" not in expected_pool(env):
                r.append("n/a")
            elif env in TOY_TEXT or opp is None:
                r.append("pending" if "TEMPO" not in row else "no baseline")
            elif "TEMPO" not in row or opp not in row:
                r.append("pending")
            else:
                res, p = _vs(row, "TEMPO", opp)
                pv = "p<0.001" if p < 1e-3 else f"p={p:.3f}" if np.isfinite(p) else ""
                r.append(f"not tested (TEMPO n={row['TEMPO'].n})" if res == "n/a" else
                         f"{res} ({pv}{', ' + opp if k == 0 else ''})")
        rows.append(r)
    lines.append(md_table(head, rows, align="lr" + "r" * len(labels) + "ll"))
    by_arch = defaultdict(list)
    for x in crossq:
        if x["crossq"] is not None:
            by_arch[x["crossq_arch"]].append(x["env"])
    common = max(by_arch, key=lambda a: len(by_arch[a])) if by_arch else None
    other = [f"{a} on {_join(v)}" for a, v in by_arch.items() if a != common]
    cq_text = (common + (f", except {_join(other)}" if other else "")) if common else "none yet"
    t_used = [l for l in labels if is_tempo(l) and any(l in c for c in cells.values())]
    t_text = ", ".join(("" if l == "TEMPO" and len(t_used) == 1 else l + " ")
                       + label_arch([r for rows in data.values() for r in rows.get(l, [])]) for l in t_used)
    lines.append(f"Critic hidden layers (from the recorded run configurations): CrossQ {cq_text}; "
                 f"TEMPO {t_text or 'n/a'}. Baselines use the RL-Baselines3-Zoo "
                 "hyperparameters (SB3 2.9.0 / sb3-contrib 2.9.0); TEMPO uses its defaults on every task.\n")
    lines.append(f"Learning curves: [`{figures_rel}/final_tempo_curves.png`]({figures_rel}/final_tempo_curves.png)"
                 f" (tasks with TEMPO runs), [`{figures_rel}/final_curves.png`]({figures_rel}/final_curves.png) "
                 "(all tasks). All tables, including AUC, aggregates, compute and automatic data checks: "
                 "[`benchmarks/tables/`](benchmarks/tables/). Newest result file: "
                 f"{stamps[-1] if stamps and stamps[-1] else 'unknown'}.\n")
    lines.append("Regenerate this section after new runs finish:\n\n```bash\n"
                 "python -m benchmarks.report --paper    # tables, figures, this README block, paper figures\n"
                 "```")
    return "\n".join(lines) + "\n"


def check_readme_markers(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text.count(README_START) != 1 or text.count(README_END) != 1 or \
            text.index(README_START) > text.index(README_END):
        raise SystemExit(f"{path}: expected exactly one '{README_START}' followed by one '{README_END}'")
    return text


def update_readme(path: pathlib.Path, block: str) -> bool:
    """Replace the text between the RESULTS markers of README.md (and nothing else)."""
    text = check_readme_markers(path)
    i, j = text.index(README_START) + len(README_START), text.index(README_END)
    new = text[:i] + "\n" + block.strip("\n") + "\n" + text[j:]
    if new == text:
        return False
    path.write_text(new, encoding="utf-8")
    return True


def copy_paper_figures(figures: pathlib.Path, dest: pathlib.Path, written: list) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for stem in PAPER_FIGURES:
        src = figures / f"{stem}.pdf"
        if not src.exists():
            print(f"warning: {src} does not exist (run --suite final first); not copied")
            continue
        shutil.copy2(src, dest / src.name)
        written.append(dest / src.name)


# ------------------------------------------------------------------ final-suite driver
def report_suite(results: pathlib.Path, suite: str, figures: pathlib.Path, tables: pathlib.Path,
                 randoms_path: pathlib.Path, n_boot: int, tempo: str = "TEMPO",
                 readme: pathlib.Path | None = None) -> list:
    written: list = []
    data = load_suite(results, suite)
    if not data:
        print(f"no results under {results / suite}")
        return written
    randoms = load_randoms(randoms_path)
    cells = per_task(data, n_boot)
    norm = normalisation(cells, randoms)
    labels = sorted({l for row in cells.values() for l in row} | {"TEMPO"}, key=label_key)
    if suite == "final":     # every planned algorithm gets a column, even while pending
        labels = sorted(set(labels) | set(ALGO_ORDER), key=label_key)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    def w(name, text):
        p = tables / name
        p.write_text(text)
        written.append(p)

    # (1) curves
    plot_curves(data, figures, suite, min(n_boot, 2000), written)
    # (2) score tables
    for metric, stem in (("final", "scores"), ("auc", "auc")):
        md, tex = score_tables(cells, labels, metric, len(FINAL_SEEDS))
        w(f"{suite}_{stem}.md", md)
        w(f"{suite}_{stem}.tex", tex)
    # (3) aggregates
    algos = [l for l in labels if any(l in row for row in cells.values())] + [BEST]
    agg, profiles = {}, {}
    all_scores = {l: normalised_scores(cells, norm, l) for l in algos}
    all_scores = {l: s for l, s in all_scores.items() if s}
    pooled = np.concatenate([v for s in all_scores.values() for v in s.values()]) if all_scores else np.ones(1)
    tau_max = float(np.clip(math.ceil(np.percentile(pooled, 97) * 4) / 4, 1.25, 3.0))
    taus = np.linspace(0.0, tau_max, 241)
    for l, s in all_scores.items():
        agg[l] = dict(tasks=sorted(s, key=task_key), n_runs=int(sum(v.size for v in s.values())),
                      **aggregate_ci(s, n_boot, (l,)))
        profiles[l] = performance_profile(s, taus, min(n_boot, 2000), (l,))
    raw = {l: raw_finals(cells, norm, l) for l in algos}
    poi, pairwise = {}, {}
    if raw.get(tempo):
        for b in list(BASELINES) + [BEST]:
            r = prob_improvement(raw[tempo], raw[b], n_boot, (tempo, b)) if raw.get(b) else None
            if r:
                poi[b] = r
            common = {t: v for t, v in all_scores.get(tempo, {}).items() if t in all_scores.get(b, {})}
            if common:
                pairwise[b] = dict(tasks=sorted(common, key=task_key),
                                   tempo=aggregate_ci(common, n_boot, (tempo, "common", b)),
                                   other=aggregate_ci({t: all_scores[b][t] for t in common}, n_boot,
                                                      (b, "common", tempo)))
    if agg:
        plot_aggregate(agg, profiles, taus, poi, figures, written, tempo)
    per_task_wtl, counts = win_tie_loss(cells, norm, tempo)
    summary = dict(
        suite=suite, n_boot=n_boot,
        definitions=dict(final="mean return of the last three evaluations (30 episodes), evaluations "
                               "after the step budget discarded",
                         auc="mean return over all evaluations up to the budget",
                         normalised="(x - random) / (best_baseline_mean - random)",
                         best_baseline=f"highest mean final score among {list(BASELINES)} on the task",
                         IQM="25% trimmed mean over all runs of the tasks the algorithm ran on",
                         mean_median="mean / median over tasks of the per-task mean normalised score",
                         CI="95% percentile stratified bootstrap (runs resampled within each task)",
                         poi="mean over common tasks of P(TEMPO > X) + 0.5 P(=), per-seed final scores",
                         wtl=f"two-sided Welch t-test on per-seed final scores, alpha={ALPHA}"),
        normalisation=norm,
        aggregates=agg,
        probability_of_improvement=poi,
        pairwise_on_common_tasks=pairwise,
        win_tie_loss=dict(counts=counts, per_task=per_task_wtl),
        per_task={env: {l: dict(n=c.n, seeds=c.seeds, finals=c.finals, aucs=c.aucs, final=list(c.final),
                                auc=list(c.auc), wall_h=c.wall_h, devices=c.devices)
                        for l, c in row.items()} for env, row in cells.items()},
    )
    w(f"{suite}_summary.json", json.dumps(summary, indent=1, default=_json_default))
    w("aggregate.json", json.dumps(dict(normalisation=norm, aggregates=agg,
                                        probability_of_improvement=poi,
                                        pairwise_on_common_tasks=pairwise,
                                        performance_profile=dict(
                                            taus=taus.tolist(),
                                            curves={l: [v.tolist() for v in p] for l, p in profiles.items()}),
                                        definitions=summary["definitions"]),
                                   indent=1, default=_json_default))
    w("aggregate.md", aggregate_md(agg, poi, pairwise, norm, cells, tempo))
    # (4) win / tie / loss
    md, tex = wtl_outputs(per_task_wtl, counts)
    w("win_tie_loss.md", md)
    w("win_tie_loss.tex", tex)
    # (5) compute
    md, tex = compute_tables(cells, [l for l in labels if any(l in row for row in cells.values())])
    w("compute.md", md)
    w("compute.tex", tex)
    # data checks
    checks = data_checks(data, cells, norm, randoms, suite)
    lines = [f"# Data checks: suite `{suite}`\n",
             "Automatic scan (benchmarks/report.py). error = broken data; warn = failed, diverged or "
             "suspicious run worth a look; info = bookkeeping.\n"]
    lines += [f"- **{sev}** {msg}" for sev, msg in checks] or ["- nothing flagged"]
    w(f"{suite}_data_checks.md", "\n".join(lines) + "\n")
    if suite != "final":
        return written
    # (6) paper tables, run status, the paper's figure and the README block
    tempo_labels = ["TEMPO"] + [l for l in labels if is_tempo(l) and l != "TEMPO"
                                and any(l in row for row in cells.values())]
    status = suite_status(data, labels)
    comps = [complete_comparison(cells, norm, n_boot, t) for t in tempo_labels]
    crossq = crossq_rows(data, cells, randoms, tempo_labels)
    def _gammas(runs):          # None = the library's default discount (0.99 in SB3 and TEMPO)
        return sorted({(r.info.get("config") or {}).get("gamma") for r in runs}, key=str)

    w("final_status.json", json.dumps(dict(
        status=status,
        run_configs={env: {l: dict(gamma=_gammas(runs), critic=label_arch(runs))
                           for l, runs in rows.items()} for env, rows in data.items()},
        complete_task_comparison={c["tempo"]: c for c in comps},
        crossq={r["env"]: dict(crossq_critic=r["crossq_arch"],
                               crossq=None if r["crossq"] is None else dict(
                                   n=r["crossq"].n, final=list(r["crossq"].final)),
                               tempo={t: dict(n=v["cell"].n, final=list(v["cell"].final), result=v["result"],
                                              p=v["p"], rel_to_crossq=v["rel"])
                                      for t, v in r["tempo"].items()})
                for r in crossq}), indent=1, default=_json_default))
    w("paper_main_final.tex", paper_main_table(cells, norm, labels))
    w("paper_aggregate.tex", paper_aggregate_table(comps))
    w("paper_crossq.tex", paper_crossq_table(data, cells, randoms, tempo_labels))
    plot_tempo_curves(data, figures, suite, min(n_boot, 2000), written)
    if readme is not None:
        try:
            rel = figures.resolve().relative_to(readme.resolve().parent).as_posix()
        except ValueError:
            rel = figures.as_posix()
        block = readme_block(data, cells, norm, labels, status, comps, crossq, rel)
        w("readme_results.md", block)
        if update_readme(readme, block):
            written.append(readme)
    return written


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def aggregate_md(agg, poi, pairwise, norm, cells, tempo) -> str:
    def ci(v):
        return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"

    out = ["# Aggregate statistics (final score)\n",
           "Per-task normalisation `(x - random) / (best_baseline - random)`; best_baseline = highest mean "
           f"final score among the SB3 baselines ({', '.join(BASELINES)}) on the task, so the best baseline "
           "scores exactly 1 on every task by construction (`Best baseline` row). 95% stratified-bootstrap "
           "CIs (Agarwal et al., 2021). Algorithms are aggregated over the tasks they ran on, which differ "
           "(e.g. DQN: discrete tasks only); the pairwise section compares on common tasks.\n"]
    rows = [[l if l != BEST else "Best baseline (per task)", str(len(v["tasks"])), str(v["n_runs"]),
             ci(v["IQM"]), ci(v["median"]), ci(v["mean"])] for l, v in agg.items()]
    out.append(md_table(["Algorithm", "Tasks", "Runs", "IQM", "Median", "Mean"], rows))
    if poi:
        out.append(f"\n## Probability of improvement P({tempo} > X)\n")
        out.append(md_table(["X", "Tasks", f"P({tempo} > X) [95% CI]"],
                            [[b if b != BEST else "Best baseline (per task)", str(len(v["tasks"])),
                              f"{v['p']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"] for b, v in poi.items()]))
    if pairwise:
        out.append(f"\n## {tempo} vs. X on the tasks both ran (normalised IQM)\n")
        out.append(md_table(["X", "Tasks", f"{tempo} IQM", "X IQM", f"{tempo} mean", "X mean"],
                            [[b if b != BEST else "Best baseline (per task)", str(len(v["tasks"])),
                              ci(v["tempo"]["IQM"]), ci(v["other"]["IQM"]), ci(v["tempo"]["mean"]),
                              ci(v["other"]["mean"])] for b, v in pairwise.items()]))
    elif not poi:
        out.append(f"\nNo {tempo} results yet: probability of improvement and pairwise comparisons "
                   "are not computed.\n")
    out.append("\n## Normalisation constants\n")
    nrows = []
    for env, nz in norm.items():
        nrows.append([env, "-" if nz["random"] is None else f"{nz['random']:.4g}",
                      "-" if nz["best_baseline"] is None else f"{nz['best_baseline']:.4g}",
                      nz["best_algo"] or "-", "yes" if nz["valid"] else f"no ({nz['reason']})"])
    out.append(md_table(["Environment", "Random", "Best baseline mean", "Best baseline", "In aggregate"],
                        nrows, align="lrrll"))
    return "\n".join(out)


# ------------------------------------------------------------------ screens
SCREEN_SUITES = ("screen1", "screen2", "diag")
SELECTED = "bn256_e05_hl"
VARIANT_ORDER = ["ln", "ln_phys", "bn256", "bn512", "bn512_n1", "bn256_e05", "bn256_hl", "bn256_e05_hl",
                 "bn256_e05_hl_lr", "bn256_d03", "bn256_dfix005", "bn256_n1", "bn256_d03_hl", "default"]
VARIANT_DOC = {
    "ln": "LayerNorm critics + Polyak target nets, lr 3e-4, MSE critic, entropy -0.3 -> -1.0 nats/dim",
    "ln_phys": "ln with entropy in physical action coordinates",
    "bn256": "Batch-Renorm actor/critic (CrossQ-style, no target net), critic 256x2, lr 1e-3, policy delay 3, "
             "MSE critic, entropy -0.3 -> -1.0",
    "bn512": "bn256 with 512x2 critics", "bn512_n1": "bn512 with n-step 1",
    "bn256_e05": "bn256, entropy 0.0 -> -0.5 nats/dim", "bn256_hl": "bn256, HL-Gauss critic",
    "bn256_e05_hl": "bn256, entropy 0.0 -> -0.5, HL-Gauss critic (selected recipe = library default)",
    "bn256_e05_hl_lr": "bn256_e05_hl + linear lr decay 1e-3 -> 1e-4",
    "bn256_d03": "bn256, discrete entropy 0.3 -> 0.05 of ln(n_actions)",
    "bn256_dfix005": "bn256, discrete entropy fixed at 0.05 of ln(n_actions)", "bn256_n1": "bn256 with n-step 1",
    "bn256_d03_hl": "bn256_d03 + HL-Gauss critic", "default": "library defaults",
}


def _vkey(v):
    return (VARIANT_ORDER.index(v) if v in VARIANT_ORDER else len(VARIANT_ORDER), v)


def _screen_runs(results: pathlib.Path, suites) -> dict:
    """{(env, budget): {variant: [(suite, seed, final)]}} for TEMPO runs."""
    out: dict = defaultdict(lambda: defaultdict(list))
    for s in suites:
        for env, rows in load_suite(results, s).items():
            for lab, runs in rows.items():
                for r in runs:
                    if r.algo == "TEMPO":
                        out[(env, r.budget)][r.variant].append((s, r.seed, r.final))
    return out


def _screen_matrix(runs: dict, randoms: dict, title: str):
    cols = sorted(runs, key=lambda k: (task_key(k[0]), k[1]))
    variants = sorted({v for c in cols for v in runs[c]}, key=_vkey)
    head = ["Variant"] + [f"{e} ({fmt_steps(b)})" for e, b in cols]
    md_rows, tex_rows = [], []
    col_means = {c: {u: float(np.mean([y[2] for y in xs])) for u, xs in runs[c].items()} for c in cols}
    col_dec = {c: _decimals(list(col_means[c].values())) for c in cols}
    for v in variants:
        md, tex = [], []
        for c in cols:
            xs = sorted(runs[c].get(v, []), key=lambda x: x[1])
            if not xs:
                md.append("")
                tex.append("")
                continue
            finals = [x[2] for x in xs]
            d = col_dec[c]
            per = ", ".join(f"s{x[1]}: {_num(x[2], d)}" for x in xs)
            best = max(col_means[c].values())
            m = _num(float(np.mean(finals)), d)
            is_best = np.isclose(np.mean(finals), best)
            md.append((f"**{m}**" if is_best else m) + (f" ({per})" if len(xs) > 1 else f" (s{xs[0][1]})"))
            seeds = ",".join(str(x[1]) for x in xs)
            mt = _num(float(np.mean(finals)), d, True)
            tex.append((rf"\textbf{{{mt}}}" if is_best else mt) + rf" {{\scriptsize({seeds})}}")
        name = f"**{v}** (selected)" if v == SELECTED else v
        md_rows.append([name] + md)
        tex_rows.append(" & ".join([rf"\textbf{{{_tex_escape(v)}}}" if v == SELECTED else _tex_escape(v)] + tex)
                        + r" \\")
    rnd = ["random"] + [_num(randoms[e], _decimals([randoms[e]])) if e in randoms else "" for e, _ in cols]
    md_rows.append(rnd)
    md = f"### {title}\n\n" + md_table(head, md_rows, align="l" + "r" * len(cols))
    tex = "\n".join([
        r"\begin{table*}[t]", r"\centering", rf"\caption{{Development screen: {_tex_escape(title)}. Final "
        r"return (mean of the last three evaluations) averaged over the seeds in parentheses; bold = best "
        r"variant per task.}", rf"\label{{tab:{title.split()[0].lower().replace('+', '')}}}",
        r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{@{}l" + "r" * len(cols) + "@{}}", r"\toprule",
        " & ".join(["Variant"] + [_tex_escape(e) + rf" ({fmt_steps(b)})" for e, b in cols]) + r" \\",
        r"\midrule", *tex_rows, r"\midrule",
        " & ".join([r"\textit{random}"] + [_num(randoms[e], _decimals([randoms[e]]), True) if e in randoms
                                           else "" for e, _ in cols]) + r" \\",
        r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""])
    return md, tex


#: rows and columns of the paper's development table (screen 1 + 2 pooled, as select_recipe pools them)
PAPER_SCREEN_ROWS = (("ln", "LN (unit entropy)"), ("ln_phys", "LN (physical entropy)"),
                     ("bn256", "BRN-256"), ("bn512", "BRN-512"), ("bn512_n1", "BRN-512, $n=1$"),
                     ("bn256_e05", "BRN-256 + e05"), ("bn256_hl", "BRN-256 + HL"),
                     ("bn256_e05_hl", "BRN-256 + e05 + HL"),
                     ("bn256_e05_hl_lr", "BRN-256 + e05 + HL + LR decay"))
PAPER_SCREEN_TASKS = ("Humanoid-v5", "Ant-v5", "Hopper-v5", "Walker2d-v5", "HalfCheetah-v5", "LunarLander-v3")


def paper_screens_table(runs: dict, randoms: dict) -> str:
    """Variant x task development table (final score, mean over the development seeds)."""
    cols = []
    for env in PAPER_SCREEN_TASKS:
        keys = [k for k in runs if k[0] == env]
        if keys:
            cols.append(max(keys, key=lambda k: sum(len(v) for v in runs[k].values())))
    body = []
    for v, name in PAPER_SCREEN_ROWS:
        cells = []
        for c in cols:
            xs = sorted(runs[c].get(v, []), key=lambda x: x[1])
            if not xs:
                cells.append("--")
                continue
            d = _decimals([float(np.mean([y[2] for y in u])) for u in runs[c].values()])
            m = _num(float(np.mean([x[2] for x in xs])), d, True)
            m = rf"\textbf{{{m}}}" if v == SELECTED else m
            cells.append(m + rf"\,{{\scriptsize({','.join(str(x[1]) for x in xs)})}}")
        body.append(" & ".join([rf"\textbf{{{name}}}" if v == SELECTED else name] + cells) + r" \\")
    rnd = [_num(randoms[e], _decimals([randoms[e]]), True) if e in randoms else "" for e, _ in cols]
    head = [f"{e.split('-v')[0]} ({fmt_steps(b)})" for e, b in cols]
    caption = (
        r"Development screens (screens 1 and 2 pooled, as in the selection script): final score = mean "
        r"of the last three evaluations, averaged over the development seeds in parentheses (per-seed "
        r"values in \texttt{benchmarks/tables/screens.md}). BRN = batch-renormalised critic without "
        r"target network, LN = LayerNorm critic with target network, HL = HL-Gauss categorical critic "
        r"with reward scaling, e05 = entropy schedule $0 \to -0.5$ nats/dim (otherwise $-0.3 \to -1$); "
        r"-- = not run. Bold: the selected configuration, TEMPO's default.")
    return "\n".join([
        "% Generated by `python -m benchmarks.report --screens`; requires booktabs, graphicx.",
        r"\begin{table}[tbp]", r"\centering", rf"\caption{{{caption}}}", r"\label{tab:screen}",
        SHRINK_TO_WIDTH, r"\begin{tabular}{@{}l" + "r" * len(cols) + "@{}}", r"\toprule",
        " & ".join(["Variant"] + head) + r" \\", r"\midrule", *body, r"\midrule",
        " & ".join([r"\textit{random policy}"] + rnd) + r" \\", r"\bottomrule", r"\end{tabular}}",
        r"\end{table}", ""])


def report_screens(results: pathlib.Path, tables: pathlib.Path, randoms_path: pathlib.Path) -> list:
    written = []
    randoms = load_randoms(randoms_path)
    tables.mkdir(parents=True, exist_ok=True)
    md = ["# Development screens (TEMPO variants, seeds 0-3)\n",
          "Final return = mean of the last three evaluations (30 episodes). Each cell: mean over the "
          "seeds listed (per-seed values in parentheses). Bold = best variant in the column. These seeds "
          "were used for design decisions and are not part of the final benchmark (seeds 10-14); see "
          "benchmarks/PROTOCOL.md and benchmarks/selection_screen2.txt.\n"]
    tex = ["% Generated by `python -m benchmarks.report --screens`; requires \\usepackage{booktabs,graphicx}."]
    js = {}
    sections = [(f"{s}", _screen_runs(results, [s])) for s in SCREEN_SUITES]
    sections.append(("screen1+2 pooled (as in select_recipe)", _screen_runs(results, ["screen1", "screen2"])))
    for title, runs in sections:
        if not runs:
            continue
        m, t = _screen_matrix(runs, randoms, title)
        md.append(m)
        tex.append(t)
        js[title] = {f"{e}@{b}": {v: [dict(suite=s, seed=sd, final=f) for s, sd, f in xs]
                                  for v, xs in sorted(vs.items(), key=lambda kv: _vkey(kv[0]))}
                     for (e, b), vs in sorted(runs.items(), key=lambda kv: task_key(kv[0][0]))}
    md.append("### Variants\n")
    used = sorted({v for sec in js.values() for task in sec.values() for v in task}, key=_vkey)
    md.append(md_table(["Variant", "Configuration"], [[v, VARIANT_DOC.get(v, "")] for v in used], align="ll"))
    paper = paper_screens_table(_screen_runs(results, ["screen1", "screen2"]), randoms)
    for name, text in (("screens.md", "\n".join(md)), ("screens.tex", "\n".join(tex)),
                       ("screens.json", json.dumps(js, indent=1)), ("paper_screens.tex", paper)):
        p = tables / name
        p.write_text(text)
        written.append(p)
    return written


# ------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default=None, help="suite to report (e.g. final)")
    ap.add_argument("--screens", action="store_true", help="development-screen tables (screen1, screen2, diag)")
    ap.add_argument("--results", default=str(BENCH / "results"), help="results root (read only)")
    ap.add_argument("--figures", default=str(BENCH / "figures"))
    ap.add_argument("--tables", default=str(BENCH / "tables"))
    ap.add_argument("--random-scores", default=str(BENCH / "random_scores.json"))
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--update-readme", action="store_true",
                    help="rewrite the block between <!-- RESULTS:START --> and <!-- RESULTS:END --> of "
                         "--readme from the final suite (nothing else in the file is touched)")
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    ap.add_argument("--paper-figures", default=None,
                    help="copy the figures paper/RL.tex includes into this directory (e.g. paper/figures)")
    ap.add_argument("--paper", action="store_true",
                    help="everything the paper and README use: --suite final --screens --update-readme "
                         "--paper-figures paper/figures")
    a = ap.parse_args(argv)
    if a.paper:
        a.suite, a.screens, a.update_readme = a.suite or "final", True, True
        a.paper_figures = a.paper_figures or str(ROOT / "paper" / "figures")
    if not a.screens and a.suite is None:
        a.suite = "final"
    if a.update_readme and a.suite != "final":
        raise SystemExit("--update-readme needs --suite final")
    results, figures, tables = pathlib.Path(a.results), pathlib.Path(a.figures), pathlib.Path(a.tables)
    readme = pathlib.Path(a.readme) if a.update_readme else None
    paper_figures = pathlib.Path(a.paper_figures) if a.paper_figures else None
    for out in (figures, tables, readme, paper_figures):
        if out is not None and out.resolve().is_relative_to(results.resolve()):
            raise SystemExit(f"refusing to write inside the results directory: {out}")
    if readme is not None:
        check_readme_markers(readme)          # fail before writing anything
    written = []
    if a.suite:
        written += report_suite(results, a.suite, figures, tables, pathlib.Path(a.random_scores), a.n_boot,
                                readme=readme)
    if a.screens:
        written += report_screens(results, tables, pathlib.Path(a.random_scores))
    if paper_figures is not None:
        copy_paper_figures(figures, paper_figures, written)
    for p in written:
        try:
            print(p.relative_to(ROOT))
        except ValueError:
            print(p)
    if a.suite:
        dc = tables / f"{a.suite}_data_checks.md"
        if dc.exists():
            n = dc.read_text().count("\n- **")
            print(f"{n} data-check item(s): {dc}")


if __name__ == "__main__":
    main()
