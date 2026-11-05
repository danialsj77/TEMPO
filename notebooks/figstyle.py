"""Academic figure style of the ``Baselines_test/``, ``TEMPO_test/`` and ``Comparison/`` figures (the single source of truth).

Every figure of the three folders is drawn with this module: ``notebooks/runlab.py`` (the learning-curve figure of every
per-task notebook) and ``Comparison/complab.py`` (all comparison figures) import it; no other file holds style code.
Algorithm colours, dash patterns and widths are NOT defined here: they are ``benchmarks.report.STYLE`` (validated palette,
white surface; TEMPO/TD3 sit in the colour-blind warn band, so the dash patterns are the required secondary encoding, and
aqua TD3 / yellow PPO are below 3:1 contrast, so a legend is always drawn). This module never edits ``report.py``.

Importing it changes no global matplotlib state, creates no file and imports matplotlib only when a function needs it.

Encoding of RETURN curves over training (one bold line + one shaded band per algorithm)
---------------------------------------------------------------------------------------
Per algorithm ONE bold line = the mean over the seeds at every common evaluation step, and ONE shaded band in the same
colour (no edge line; alpha :func:`band_alpha`: 0.2 with 1-2 banded curves in the panel, 0.14 with 3-4, 0.10 with 5 or
more, so overlapping bands stay attributable) = the spread of the seeds. The band is chosen by one parameter:

    ``band="minmax"`` (:data:`DEFAULT_BAND`)  range across seeds: lowest to highest seed at every evaluation
    ``band="std"``                           mean +- 1 sample standard deviation (ddof=1) across seeds
    ``band="ci95"``                          95% percentile-bootstrap CI of the mean over seeds, computed by
                                             ``benchmarks.report.curve_band`` (the previous figures' band; same seed
                                             of the bootstrap for the same ``key``)
    ``band=None``                            the default (``"minmax"``); ``band=False`` / ``"none"``: no band

Individual seed lines are off by default (``seeds=True`` adds them as thin faint lines). With one seed (or at an
evaluation reached by fewer than two seeds) there is no band, only the line. TEMPO is drawn slightly bolder
(:data:`TEMPO_LINE_WIDTH` vs :data:`LINE_WIDTH`) and on top of every other curve. Only evaluations at or before the budget
are used (``align(..., budget=)``; ``benchmarks.report.Run`` objects are already cut at their budget, as in report.py), and
the seeds are aligned on their common evaluation steps before the mean and the band are computed (:func:`align`).
The y axis is fitted to the mean lines and the reference lines (:func:`fit_ylim`); a band is shown in full unless one
seed lies far outside them (more than :data:`ROBUST_CAP` = half their span beyond them), then that band is clipped at the
axes and the caption says so (``band_caption(..., clipped=True)``), so one collapsed or outlying seed never squashes the
other curves.

Colours: ``benchmarks.report.STYLE`` for the six algorithms; the headline TEMPO (the TEMPO of the notebook or comparison)
is ``STYLE["TEMPO"]``; every other TEMPO variant (plain "TEMPO" too when it is not the headline) gets a slot of
:data:`VARIANT_STYLES` (brown / plum, then repeated with new dash patterns; never the reference greys) by :func:`tempo_style`, the
one rule of every folder.

Academic style (:func:`rc`, :func:`paper`, :func:`apply`)
---------------------------------------------------------
White background; serif text font (:data:`SERIF_FONTS`: Times New Roman if installed, else STIXGeneral, else DejaVu
Serif; resolved once by :func:`serif_font`) with mathtext ``stix``; print sizes (:data:`FONT_SIZE` = 8 pt ticks, legend
and cell text, :data:`LABEL_SIZE` = 9 pt axis labels and panel titles, at the SAVED size); thin axes
(:data:`AXIS_WIDTH` = 0.6 pt), no top/right spines, outward ticks, a recessive light-grey horizontal grid only; frameless
legends; white figure; ``pdf.fonttype = ps.fonttype = 42`` (TrueType embedded); constrained layout (:func:`subplots`). Widths for a paper:
:data:`SINGLE` = 3.5 in (one column), :data:`DOUBLE` = 7.2 in (full text width); grids are sized from these
(:func:`grid_figsize`), never squeezed. Saved by :func:`save` as a vector PDF (TrueType fonts) and a 300-dpi PNG with a
tight bounding box and a small pad (:data:`PAD_INCHES`); :func:`show` displays the saved PNG (or a figure rendered exactly
like the saved PNG) inline at a fixed on-screen scale, so the notebook shows what the files contain.

API
---
Constants
    ``SINGLE``, ``DOUBLE`` (in), ``SAVE_DPI`` (300), ``DISPLAY_PPI`` (on-screen pixels per inch of a saved figure),
    ``PAD_INCHES``; ``FONT_SIZE``, ``LABEL_SIZE``, ``AXIS_WIDTH``; ``LINE_WIDTH`` (bold mean line, 1.5 pt),
    ``TEMPO_LINE_WIDTH`` (2.0 pt), ``SEED_LINE_WIDTH``, ``SEED_ALPHA``, ``BAND_ALPHA`` / ``BAND_ALPHAS`` (band opacity by
    the number of bands in a panel), ``ROBUST_CAP`` (see ``fit_ylim``), ``VARIANT_STYLES``; ``BANDS``, ``DEFAULT_BAND``,
    ``N_BOOT`` (resamples of ``band="ci95"``); ``INK``, ``INK2``, ``MUTED``, ``GRID`` (= ``report``'s text / grid
    colours); ``REFERENCE`` {"random" | "solved" | "optimum": dict(color, ls, lw, label)}: reference lines in neutral grey
    with distinct dashes; ``STEPS_LABEL`` ("Environment steps"), ``RETURN_LABEL`` ("Episode return"),
    ``NORMALISED_LABEL`` ("Normalised score"), ``HOURS_LABEL`` ("Wall-clock time (h)"); ``SEQUENTIAL`` (single-hue blue
    ramp, light -> dark, from the validated palette).

Style
    ``rc(**overrides) -> dict``                 the rcParams of the style (``overrides`` use rcParams names)
    ``paper(**overrides)``                      context manager: ``with figstyle.paper(): ...`` (global state restored)
    ``apply(**overrides)``                      the same, globally (explicit opt-in; never done at import)
    ``serif_font() -> str``                     the serif family in use on this machine
    ``figsize(width=SINGLE, height=None, aspect=0.66) -> (w, h)``
    ``grid_figsize(nrows, ncols, width=DOUBLE, panel_aspect=0.75, extra=0.55) -> (w, h)``
    ``subplots(nrows=1, ncols=1, width=SINGLE, height=None, **kw) -> (fig, axes)``   constrained layout, sized as above
                                                (call it inside ``paper()``)

Curves
    ``tempo_style(label, headline="TEMPO", tempo_labels=()) -> dict``   style of a curve: ``STYLE["TEMPO"]`` for the
                                                headline, a :data:`VARIANT_STYLES` slot for any other TEMPO label (its
                                                position among the non-headline TEMPO labels of ``tempo_labels``, sorted by
                                                ``report.label_key``: pass every TEMPO label of the result set),
                                                ``report.style`` otherwise
    ``band_alpha(n_curves) -> float``           band opacity of a panel with ``n_curves`` banded curves (0.2 / 0.14 / 0.10)
    ``align(curves, budget=None) -> (steps, Y)``  seeds on common evaluation steps: ``curves`` = ``report.Run`` objects
                                                (anything with ``.steps`` / ``.curve``) or ``(steps, values)`` pairs;
                                                ``Y`` has shape (n_seeds, n_steps). Evaluations after ``budget`` are
                                                dropped. Identical schedules (the normal case) are used as they are;
                                                otherwise the steps every seed evaluated at; if fewer than two such
                                                steps, every seed is interpolated onto the union of the steps inside the
                                                range all seeds cover; if that range is empty, the union grid with NaN
                                                outside each seed's own range (as ``report.curve_band``).
    ``mean_band(steps, Y, band=None, key=None, n_boot=N_BOOT) -> (mean, lo, hi)``
                                                the numbers of the line and the band (``lo``/``hi`` NaN where fewer than
                                                two seeds); ``key`` seeds the bootstrap of ``"ci95"``
                                                (``report.curve_band(runs, n_boot, key)``, e.g. ``(env, label)``)
    ``curve(ax, steps, Y, label=None, *, style=None, color=None, ls=None, lw=None, tempo=False, band=None,
            alpha=BAND_ALPHA, seeds=False, key=None, n_boot=N_BOOT) -> Curve | None``
                                                draws one algorithm: bold mean line, shaded band, optional seed lines.
                                                ``style`` = a ``report.STYLE`` entry (colour, dashes, z order);
                                                ``tempo=True`` = bolder and on top. ``x`` may be anything (steps, share
                                                of the budget, hours). Returns a :class:`Curve` (``.line``, ``.band``,
                                                ``.seed_lines``, ``.steps``, ``.mean``, ``.lo``, ``.hi``, ``.n``, ``.Y``
                                                (the aligned seeds), ``.legend_handle``, ``.label``) or None when there
                                                is nothing to draw.
    ``check_band(band) -> str``                 normalises / validates a band name (ValueError otherwise)
    ``band_caption(band=None, n=None, seeds=False, ids=None, clipped=False) -> str``   e.g. "Lines: mean over 5 seeds
                                                (10-14); shaded: range (min-max) across seeds." (``n`` = number of seeds,
                                                None when it differs between curves: "n per curve in the legend"; ``ids`` =
                                                which seeds; ``clipped`` = fit_ylim's flag: adds "Bands clipped at the axes
                                                (one seed outside the shown range).")
    ``reference_line(ax, y, kind, label=None) -> Line2D``    horizontal reference line ("random", "solved", "optimum")
                                                in neutral grey with its own dashes, below the data, labelled for the
                                                legend (default label ``REFERENCE[kind]["label"]``)
    ``reference_handle(kind, label=None) -> Line2D``         a legend proxy of a reference line (shared legends)
    ``num(x, fmt=".4g") -> str``                a number for figure text with a true minus sign (e.g. legend values)

Axes and legend
    ``steps_axis(ax, budget=None, label=True, nbins=5)``     x axis of environment steps: limits 0..budget, ticks 0,
                                                100k, ..., 500k (or 1M), label :data:`STEPS_LABEL`
    ``fmt_steps(x) -> str``                     tick text of a step count ("0", "500", "2.5k", "100k", "1M")
    ``value_axis(ax, label=RETURN_LABEL, nbins=5, axis="y")``  plain numbers (no offset), at most ``nbins`` intervals
    ``fit_ylim(ax, *values, bands=(), pad=0.05, cap=ROBUST_CAP) -> bool``   y limits: ``values`` (mean lines,
                                                reference lines) always in full; ``bands`` (Curve objects) in full up to
                                                ``cap`` x the span of the values beyond them or their second most extreme
                                                seed, else clipped at the axes (returns True: say so in the caption); a
                                                pad, so a reference line never sits on the axis
    ``legend(fig, *groups, where="below", ncol=None) -> Legend``   ONE shared frameless legend outside the data area
                                                (below the panels, or ``where="top"``). Each group is a list of
                                                :class:`Curve` / ``Line2D`` / ``(handle, label)`` items and is its own
                                                centred block of rows with its own column widths (e.g. algorithms, then
                                                reference lines); entries read row by row; ``ncol`` per group fitted to
                                                the figure width and balanced when None (3 entries: one row, keys of
                                                markers / solid lines shortened to ``LEGEND_MIN_HANDLE`` em if needed,
                                                else one column; never 2 + 1). A curve with a band shows line + band in
                                                its legend key.
    ``caption(text, width=None) -> str``        a caption string wrapped for printing under a figure (explanations go
                                                in captions and notebook text, never in long panel titles)

Saving and showing
    ``save(fig, path, formats=("pdf", "png"), dpi=SAVE_DPI) -> [Path]``  ``path`` WITHOUT suffix (a suffix .pdf/.png/.svg
                                                is replaced; any other dot stays part of the name, e.g.
                                                ``Hopper-v5.partial``); creates the folder; PDF with TrueType fonts and
                                                no creation date, PNG at ``dpi``, white, tight box with PAD_INCHES.
    ``show(obj)``                               displays a saved PNG path (or a Figure, rendered exactly as ``save``
                                                renders the PNG) inline at DISPLAY_PPI; does nothing outside IPython.

Heatmaps and marks
    ``sequential_cmap() -> Colormap``           single-hue blue, light -> dark (never a rainbow); label the colourbar
    ``text_on(rgba) -> str``                    ink or white, whichever reads better on a cell colour
    ``MARKER_SIZE``, ``ERRORBAR_WIDTH``         thin marks of dot / interval plots
    ``LEGEND_MIN_HANDLE``                       shortest legend key (em) of a 3-entry group fitted into one row

Minimal use::

    import figstyle as fs
    with fs.paper():
        fig, ax = fs.subplots(width=fs.SINGLE)
        items = []
        alpha = fs.band_alpha(sum(len(runs) > 1 for runs in rows.values()))
        for label, runs in rows.items():
            steps, Y = fs.align(runs, budget=budget)
            c = fs.curve(ax, steps, Y, label, style=fs.tempo_style(label, "TEMPO", rows), tempo=label == "TEMPO",
                         alpha=alpha, key=(env, label))
            if c: items.append(c)
        refs = [fs.reference_line(ax, rnd, "random")]
        clipped = fs.fit_ylim(ax, *[c.mean for c in items], rnd, bands=items)
        fs.steps_axis(ax, budget); fs.value_axis(ax, fs.RETURN_LABEL); ax.set_title(env)
        fs.legend(fig, items, refs)
        paths = fs.save(fig, out_dir / env)
    plt.close(fig); fs.show(paths[1]); print(fs.caption(fs.band_caption(n=5, clipped=clipped)))
"""
from __future__ import annotations

import contextlib
import functools
import io
import math
import pathlib
import struct
import sys
import textwrap
import types
import warnings
from dataclasses import dataclass

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:                       # the project root: benchmarks/ importable
    sys.path.insert(0, str(_HERE.parent))

from benchmarks import report  # noqa: E402  (colours and the bootstrap band of the project's report)

__all__ = ["SINGLE", "DOUBLE", "SAVE_DPI", "DISPLAY_PPI", "PAD_INCHES", "FONT_SIZE", "LABEL_SIZE", "AXIS_WIDTH",
           "LINE_WIDTH", "TEMPO_LINE_WIDTH", "SEED_LINE_WIDTH", "SEED_ALPHA", "BAND_ALPHA", "BAND_ALPHAS", "ROBUST_CAP",
           "VARIANT_STYLES", "tempo_style", "band_alpha", "BANDS", "DEFAULT_BAND",
           "N_BOOT", "INK", "INK2", "MUTED", "GRID", "REFERENCE", "STEPS_LABEL", "RETURN_LABEL", "NORMALISED_LABEL",
           "HOURS_LABEL", "SEQUENTIAL", "MARKER_SIZE", "ERRORBAR_WIDTH", "LEGEND_MIN_HANDLE",
           "SERIF_FONTS", "Curve", "rc", "paper", "apply",
           "serif_font", "figsize", "grid_figsize", "subplots", "align", "mean_band", "curve", "check_band",
           "band_caption", "num", "reference_line", "reference_handle", "steps_axis", "fmt_steps", "fit_ylim",
           "value_axis", "legend", "caption", "save", "show", "sequential_cmap", "text_on"]

# ------------------------------------------------------------------------------------------------ constants
SINGLE = 3.5                    # in: one column of a two-column paper
DOUBLE = 7.2                    # in: full text width
SAVE_DPI = 300
DISPLAY_PPI = 140               # inline display: screen pixels per inch of the saved figure (same picture as the file)
PAD_INCHES = 0.03               # white margin around the tight bounding box

FONT_SIZE = 8.0                 # pt at the saved size: tick labels, legend, cell text, notes
LABEL_SIZE = 9.0                # pt: axis labels and panel titles
AXIS_WIDTH = 0.6                # pt: spines and ticks
GRID_WIDTH = 0.4                # pt: horizontal grid

LINE_WIDTH = 1.5                # pt: the bold mean line of an algorithm
TEMPO_LINE_WIDTH = 2.0          # pt: TEMPO, slightly bolder (and drawn on top)
SEED_LINE_WIDTH = 0.5           # pt: optional individual seeds
SEED_ALPHA = 0.35
BAND_ALPHA = 0.2                # the shaded band (no edge line) of a panel with 1-2 banded curves (see band_alpha)
#: band opacity by the number of banded curves in one panel (band_alpha): fainter as more bands overlap, so that every band
#: stays attributable to its colour where 3-4 of them overlap
BAND_ALPHAS = ((2, BAND_ALPHA), (4, 0.14), (None, 0.10))
#: y limits (fit_ylim): a band may reach this fraction of the span of the mean and reference lines beyond them before
#: it is clipped at the axes (its second most extreme seed is always shown)
ROBUST_CAP = 0.5
MARKER_SIZE = 4.0               # pt: dots of dot / interval plots
ERRORBAR_WIDTH = 1.0            # pt: interval bars of dot plots
#: em: the shortest legend key :func:`legend` uses to fit a 3-entry group of markers / solid lines into one row (keys with a
#: dash pattern always keep the full ``legend.handlelength``, the dashes being a required encoding)
LEGEND_MIN_HANDLE = 1.6

BANDS = ("minmax", "std", "ci95")
DEFAULT_BAND = "minmax"
N_BOOT = 2000                   # bootstrap resamples of band="ci95" (as the previous figures)
_BAND_TEXT = {"minmax": "range (min-max) across seeds",
              "std": "mean ± 1 standard deviation across seeds",
              "ci95": "95% bootstrap confidence interval of the mean"}

INK, INK2, MUTED, GRID = report.INK, report.INK2, report.MUTED, report.GRID
SURFACE = "#ffffff"

#: Reference lines: neutral grey, thin, each with its own dash pattern (distinct from every algorithm's), below the data.
REFERENCE = {
    "random": dict(color=MUTED, ls=(0, (1, 1.6)), lw=0.9, label="Random policy"),
    "solved": dict(color=MUTED, ls=(0, (5, 2.5)), lw=0.8, label="Solved threshold"),
    "optimum": dict(color=MUTED, ls=(0, (8, 2, 1.5, 2)), lw=0.8, label="Known optimum"),
}

#: Further TEMPO variants (every TEMPO label that is not the headline, plain "TEMPO" included when another variant is the
#: headline): brown and plum, each with its own dash pattern, never the neutral greys of the reference lines. Slot ``i``
#: goes to the ``i``-th of these labels in ``report.label_key`` order (:func:`tempo_style`); past the second the colours
#: repeat with the dash patterns of :data:`_VARIANT_DASHES_2`. Chosen with the dataviz validator (white surface, all pairs):
#: together with TEMPO's red and the five baselines of ``report.STYLE`` they pass every hard gate without lowering the
#: palette's worst pairs (CVD dE 6.9 = the existing TEMPO/TD3 pair, normal-vision dE 15.6); with TEMPO's red alone (the
#: TEMPO_test figures) CVD dE 11.7, normal 17.8. No third hue passes next to all eight, so identity past two variants is
#: carried by the dash pattern (secondary encoding), never by a new colour.
VARIANT_STYLES = (
    dict(color="#7d460b", ls=(0, (5, 1.5)), lw=1.6, z=8),           # brown
    dict(color="#8e5c92", ls=(0, (2, 1)), lw=1.6, z=8),             # plum
)
_VARIANT_DASHES_2 = ((0, (3, 1, 1, 1)), (0, (8, 1.5)), (0, (1, 1, 3, 1)), (0, (4, 2.5)))

STEPS_LABEL = "Environment steps"
RETURN_LABEL = "Episode return"
NORMALISED_LABEL = "Normalised score"
HOURS_LABEL = "Wall-clock time (h)"

#: Single-hue sequential ramp (blue, light -> dark): the validated palette's blue steps 100..700 after a near-white start.
SEQUENTIAL = ("#f3f7fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

SERIF_FONTS = ("Times New Roman", "STIXGeneral", "DejaVu Serif")

# z orders: grid 0.5 (axes.axisbelow) < reference lines < bands < seed lines < mean lines (TEMPO on top of all)
_Z_REF, _Z_BAND, _Z_SEED, _Z_LINE, _Z_TEMPO = 1.2, 1.5, 2.0, 3.0, 4.5


# ------------------------------------------------------------------------------------------------ style
@functools.lru_cache(maxsize=1)
def serif_font() -> str:
    """The first of :data:`SERIF_FONTS` installed for matplotlib (DejaVu Serif ships with matplotlib)."""
    from matplotlib import font_manager

    names = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in SERIF_FONTS if n in names), "DejaVu Serif")


def rc(**overrides) -> dict:
    """The rcParams of the academic style (a new dict; ``overrides`` win, rcParams names with ``.`` written as ``__``
    or passed as ``**{"axes.titlesize": 10}``)."""
    font = serif_font()
    p = {
        # surface and text
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "savefig.edgecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "font.family": [font, "DejaVu Serif"] if font != "DejaVu Serif" else ["DejaVu Serif"],
        "font.serif": [font, *[f for f in SERIF_FONTS if f != font]],
        "mathtext.fontset": "stix", "axes.unicode_minus": True,
        "font.size": FONT_SIZE, "axes.titlesize": LABEL_SIZE, "axes.labelsize": LABEL_SIZE,
        "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE,
        "legend.title_fontsize": FONT_SIZE, "figure.titlesize": LABEL_SIZE,
        "axes.titleweight": "normal", "axes.titlepad": 4.0, "axes.labelpad": 3.0,
        # axes: thin, no top/right spine, outward ticks, recessive horizontal grid
        "axes.linewidth": AXIS_WIDTH, "axes.edgecolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK, "xtick.major.width": AXIS_WIDTH,
        "ytick.major.width": AXIS_WIDTH, "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.minor.visible": False, "ytick.minor.visible": False, "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": GRID_WIDTH, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "axes.formatter.useoffset": False, "axes.formatter.use_mathtext": True,
        # lines and legend
        "lines.linewidth": LINE_WIDTH, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.dash_capstyle": "butt", "lines.dash_joinstyle": "round", "lines.scale_dashes": True,
        "patch.linewidth": 0.0, "legend.frameon": False, "legend.handlelength": 2.8, "legend.handleheight": 0.8,
        "legend.handletextpad": 0.5, "legend.columnspacing": 1.3, "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.2, "legend.borderpad": 0.2,
        # layout and output
        "figure.figsize": (SINGLE, round(SINGLE * 0.66, 3)), "figure.dpi": 100,
        "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
        "savefig.dpi": SAVE_DPI, "savefig.bbox": "tight", "savefig.pad_inches": PAD_INCHES,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "image.cmap": "Blues",
    }
    p.update({k.replace("__", "."): v for k, v in overrides.items()})
    return p


#: rcParams read when a figure is DRAWN (not when its artists are created): :func:`save` / :func:`show` apply them so a
#: figure built inside :func:`paper` is saved identically even outside the context.
_DRAW_KEYS = ("mathtext.fontset", "axes.unicode_minus", "pdf.fonttype", "ps.fonttype", "svg.fonttype",
              "axes.formatter.useoffset", "axes.formatter.use_mathtext", "font.serif", "savefig.facecolor",
              "savefig.edgecolor")


@contextlib.contextmanager
def paper(**overrides):
    """``with figstyle.paper(): ...``: the academic style inside the block, the caller's matplotlib state after it."""
    import matplotlib

    with matplotlib.rc_context(rc(**overrides)):
        yield


def apply(**overrides) -> None:
    """Make the academic style the global matplotlib default (explicit; importing this module never does it)."""
    import matplotlib

    matplotlib.rcParams.update(rc(**overrides))


def figsize(width: float = SINGLE, height: float | None = None, aspect: float = 0.66) -> tuple:
    """``(width, height)`` in inches: ``height`` or ``width * aspect`` (the legend below adds to the saved height)."""
    return (float(width), float(height) if height is not None else round(float(width) * aspect, 3))


def grid_figsize(nrows: int, ncols: int, width: float = DOUBLE, panel_aspect: float = 0.75, extra: float = 0.55) -> tuple:
    """Size of a grid of ``nrows x ncols`` panels at a paper width: each panel ``width / ncols`` wide and
    ``panel_aspect`` times as high, plus ``extra`` inches for the shared legend and the axis labels. Choose ``ncols`` so
    that a panel stays at least ~1.6 in wide (DOUBLE: at most 4 columns) instead of squeezing panels."""
    pw = float(width) / max(1, int(ncols))
    return (float(width), round(int(nrows) * pw * panel_aspect + extra, 3))


def subplots(nrows: int = 1, ncols: int = 1, width: float = SINGLE, height: float | None = None, **kw):
    """``plt.subplots`` with constrained layout and a paper size (call inside :func:`paper`). ``height`` default: one
    panel ``width * 0.66`` high, a grid :func:`grid_figsize`."""
    import matplotlib.pyplot as plt

    if height is None:
        size = figsize(width) if nrows == 1 and ncols == 1 else grid_figsize(nrows, ncols, width)
    else:
        size = (float(width), float(height))
    kw.setdefault("layout", "constrained")
    kw.setdefault("squeeze", nrows * ncols == 1)
    return plt.subplots(nrows, ncols, figsize=size, **kw)


# ------------------------------------------------------------------------------------------------ algorithm styles
def tempo_style(label: str, headline: str = "TEMPO", tempo_labels=()) -> dict:
    """Colour / dash / width / z of an algorithm's curve (a new dict), the same rule in every folder:

    * ``label == headline`` (the TEMPO of this notebook / comparison): ``report.STYLE["TEMPO"]`` (red, solid, bold);
    * any other TEMPO label (``report.is_tempo``; plain ``"TEMPO"`` too when another variant is the headline): slot ``i`` of
      :data:`VARIANT_STYLES`, ``i`` = its position among the TEMPO labels that are not the headline (``tempo_labels`` plus
      ``label``, sorted by ``report.label_key``, so the order of ``tempo_labels`` does not matter). Pass every TEMPO label
      of the whole result set (all tasks), so that a variant keeps its colour from task to task and folder to folder;
    * any other label: ``report.style(label)`` (``report.STYLE`` for the six algorithms).

    A TEMPO label never falls back to ``report.style`` (whose extra slots are the neutral greys of the reference lines)."""
    if label == headline:
        return dict(report.STYLE["TEMPO"])
    if not report.is_tempo(label):
        return dict(report.style(label))
    others = sorted({str(l) for l in tempo_labels if report.is_tempo(str(l)) and l != headline} | {label},
                    key=report.label_key)
    i = others.index(label)
    n = len(VARIANT_STYLES)
    st = dict(VARIANT_STYLES[i % n])
    if i >= n:                                              # colours repeat: a new dash pattern keeps the pair apart
        st["ls"] = _VARIANT_DASHES_2[(i // n - 1) % len(_VARIANT_DASHES_2)]
    return st


def band_alpha(n_curves: int) -> float:
    """Opacity of the shaded bands of a panel holding ``n_curves`` curves with a band (:data:`BAND_ALPHAS`): 0.2 for
    1-2, 0.14 for 3-4, 0.10 for 5 or more, so that overlapping bands stay attributable to their colours."""
    n = max(1, int(n_curves))
    for top, a in BAND_ALPHAS:
        if top is None or n <= top:
            return float(a)
    return float(BAND_ALPHAS[-1][1])


# ------------------------------------------------------------------------------------------------ curves
def _pairs(curves, budget):
    out = []
    for c in curves:
        if isinstance(c, (tuple, list)) and len(c) == 2:
            s, v = c
        else:
            s, v = c.steps, c.curve
        s, v = np.asarray(s, float).ravel(), np.asarray(v, float).ravel()
        if s.size != v.size:
            raise ValueError(f"steps ({s.size}) and values ({v.size}) differ in length")
        keep = np.isfinite(s) & (s <= budget) if budget is not None else np.isfinite(s)
        s, v = s[keep], v[keep]
        if s.size:
            order = np.argsort(s, kind="stable")
            out.append((s[order], v[order]))
    return out


def align(curves, budget=None) -> tuple:
    """Seeds on common evaluation steps: ``(steps, Y)``, ``Y`` of shape (n_seeds, n_steps) (NaN only in the last
    fallback). ``curves``: ``report.Run`` objects (``.steps`` / ``.curve``) or ``(steps, values)`` pairs; evaluations after
    ``budget`` are dropped (report.py's rule). Same schedule for every seed (the normal case): used as it is. Otherwise
    the steps every seed evaluated at (at least two); otherwise the union of the steps inside the range every seed covers,
    each seed linearly interpolated; if no range is common, the union grid with NaN outside each seed's own range."""
    pairs = _pairs(curves, budget)
    if not pairs:
        return np.zeros(0), np.zeros((0, 0))
    first = pairs[0][0]
    if all(s.size == first.size and np.array_equal(s, first) for s, _ in pairs):
        return first.copy(), np.vstack([v for _, v in pairs])
    common = functools.reduce(np.intersect1d, [s for s, _ in pairs])
    if common.size >= 2:
        return common, np.vstack([v[np.searchsorted(s, common)] for s, v in pairs])
    union = np.unique(np.concatenate([s for s, _ in pairs]))
    lo, hi = max(s[0] for s, _ in pairs), min(s[-1] for s, _ in pairs)
    if lo <= hi:
        grid = union[(union >= lo) & (union <= hi)]
        return grid, np.vstack([np.interp(grid, s, v) for s, v in pairs])
    Y = np.full((len(pairs), union.size), np.nan)
    for i, (s, v) in enumerate(pairs):
        m = (union >= s[0]) & (union <= s[-1])
        Y[i, m] = np.interp(union[m], s, v)
    return union, Y


def check_band(band) -> str:
    """``None`` -> :data:`DEFAULT_BAND`; ``False`` / ``"none"`` -> ``"none"``; else one of :data:`BANDS`."""
    if band is None:
        return DEFAULT_BAND
    if band is False or (isinstance(band, str) and band.lower() in ("none", "off", "")):
        return "none"
    b = str(band).lower().replace("-", "").replace("_", "").replace(" ", "")
    b = {"range": "minmax", "minmax": "minmax", "std": "std", "sd": "std", "ci": "ci95", "ci95": "ci95",
         "bootstrap": "ci95"}.get(b, b)
    if b not in BANDS:
        raise ValueError(f"band must be one of {BANDS}, None (= {DEFAULT_BAND!r}) or 'none', not {band!r}")
    return b


def mean_band(steps, Y, band=None, key=None, n_boot: int = N_BOOT) -> tuple:
    """``(mean, lo, hi)`` over the seeds (rows of ``Y``) at every step; ``lo`` / ``hi`` NaN where fewer than two seeds
    have a value (and everywhere for ``band="none"``). ``"ci95"`` is ``report.curve_band`` on the aligned seeds with the
    bootstrap seeded by ``key`` (default ``("figstyle",)``)."""
    b = check_band(band)
    steps = np.asarray(steps, float)
    Y = np.atleast_2d(np.asarray(Y, float))
    nan = np.full(steps.size, np.nan)
    if Y.size == 0 or steps.size == 0:
        return nan, nan.copy(), nan.copy()
    n = np.isfinite(Y).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.where(n > 0, np.nanmean(Y, 0), np.nan)
        if b == "minmax":
            lo, hi = np.nanmin(Y, 0), np.nanmax(Y, 0)
        elif b == "std":
            sd = np.nanstd(Y, 0, ddof=1)
            lo, hi = mean - sd, mean + sd
        elif b == "ci95":
            runs = []
            for row in Y:
                ok = np.isfinite(row)
                if ok.any():
                    runs.append(types.SimpleNamespace(steps=steps[ok], curve=row[ok]))
            lo, hi = nan.copy(), nan.copy()
            if len(runs) >= 2:
                g, _, blo, bhi = report.curve_band(runs, int(n_boot), tuple(key) if key is not None else ("figstyle",))
                idx = np.searchsorted(g, steps)
                hit = (idx < g.size) & (g[np.minimum(idx, g.size - 1)] == steps)
                lo[hit], hi[hit] = blo[idx[hit]], bhi[idx[hit]]
        else:
            lo, hi = nan.copy(), nan.copy()
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    lo[n < 2], hi[n < 2] = np.nan, np.nan
    return mean, lo, hi


@dataclass
class Curve:
    """What :func:`curve` drew: the mean line, the band (or None), the seed lines and the numbers."""
    line: object
    band: object
    seed_lines: list
    steps: np.ndarray
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    n: int
    ls: object = "-"
    alpha: float = BAND_ALPHA
    Y: np.ndarray | None = None                             # the aligned seeds (n_seeds x n_steps) behind the line

    @property
    def label(self) -> str:
        return self.line.get_label()

    @property
    def legend_handle(self):
        """Legend key: the line over a patch of the band (when a band was drawn), else the line alone."""
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        ln = self.line
        proxy = Line2D([], [], color=ln.get_color(), ls=self.ls, lw=ln.get_linewidth())
        if self.band is None:
            return proxy
        return (Patch(facecolor=ln.get_color(), alpha=self.alpha, lw=0, edgecolor="none"), proxy)


def curve(ax, steps, Y, label=None, *, style=None, color=None, ls=None, lw=None, tempo: bool = False, band=None,
          alpha: float = BAND_ALPHA, seeds: bool = False, key=None, n_boot: int = N_BOOT, z=None):
    """Draw one algorithm on ``ax``: ONE bold line = the mean over the seeds (rows of ``Y``) and ONE shaded band in the
    same colour (no edge) = their spread (``band``: "minmax" default, "std", "ci95", "none"); ``seeds=True`` adds every
    seed as a thin faint line. ``style`` = a ``report.STYLE`` entry (``color``, ``ls``, ``lw``, ``z``); explicit
    ``color`` / ``ls`` / ``lw`` win. Width: :data:`LINE_WIDTH` (or the style's, if larger); ``tempo=True``:
    :data:`TEMPO_LINE_WIDTH` and above every other curve. Returns a :class:`Curve`, or None when ``Y`` holds no value."""
    st = dict(style or {})
    steps = np.asarray(steps, float)
    Y = np.atleast_2d(np.asarray(Y, float))
    if steps.size == 0 or Y.size == 0 or not np.isfinite(Y).any():
        return None
    rows = [r for r in Y if np.isfinite(r).any()]
    Y = np.vstack(rows)
    color = color or st.get("color", INK)
    ls = ls if ls is not None else st.get("ls", "-")
    if lw is None:                                          # bold; TEMPO visibly bolder than every other curve
        lw = TEMPO_LINE_WIDTH if tempo else min(max(LINE_WIDTH, float(st.get("lw", LINE_WIDTH))), TEMPO_LINE_WIDTH - 0.3)
    zz = float(st.get("z", 2)) if z is None else float(z)
    z_line = _Z_TEMPO if tempo else _Z_LINE + zz / 20.0
    mean, lo, hi = mean_band(steps, Y, band, key=key, n_boot=n_boot)
    poly = None
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any():
        poly = ax.fill_between(steps, lo, hi, where=ok, interpolate=False, color=color, alpha=alpha, lw=0,
                               edgecolor="none", zorder=_Z_BAND + zz / 100.0 + (0.05 if tempo else 0.0))
    seed_lines = []
    if seeds:
        for row in Y:
            (sl,) = ax.plot(steps, row, color=color, lw=SEED_LINE_WIDTH, alpha=SEED_ALPHA, ls="-",
                            zorder=_Z_SEED + zz / 100.0)
            seed_lines.append(sl)
    good = np.isfinite(mean)
    (line,) = ax.plot(steps[good], mean[good], color=color, ls=ls, lw=lw, zorder=z_line,
                      label=label if label is not None else "_nolegend_")
    return Curve(line=line, band=poly, seed_lines=seed_lines, steps=steps, mean=mean, lo=lo, hi=hi, n=len(rows),
                 ls=ls, alpha=alpha, Y=Y)


def band_caption(band=None, n=None, seeds: bool = False, ids: str | None = None, clipped: bool = False) -> str:
    """One sentence for a caption: what the line and the shading are. ``n`` = seeds per curve (None: it differs between
    curves, the legend gives it); ``ids`` = which seeds, e.g. ``"10-14"``; ``clipped`` = what :func:`fit_ylim` returned
    (a band or seed line was cut at the axes: the caption says so)."""
    b = check_band(band)
    if n == 1:
        s = "Lines: a single seed (no band)."
    else:
        who = ((f"mean over {n} seeds" + (f" ({ids})" if ids else "")) if n
               else "mean over the seeds (n per curve in the legend)")
        s = f"Lines: {who}" + ("." if b == "none" else f"; shaded: {_BAND_TEXT[b]}.")
        if b != "none" and not n:
            s = s[:-1] + "; a curve of one seed has no band."
    if seeds:
        s += " Thin lines: individual seeds."
    if clipped:                                             # fit_ylim always shows the second most extreme seed
        s += (" Seed lines clipped at the axes (one seed outside the shown range)." if b == "none" else
              " Bands clipped at the axes (one seed outside the shown range)." if b == "minmax" else
              " Bands clipped at the axes (they reach beyond the shown range).")
    return s


def num(x, fmt: str = ".4g") -> str:
    """A number for figure text with a true minus sign (U+2212), e.g. ``num(-1253.2)`` -> "\u22121253"."""
    return format(float(x), fmt).replace("-", "\u2212")


def reference_line(ax, y, kind: str, label: str | None = None):
    """A horizontal reference line (``kind`` = "random", "solved" or "optimum"): neutral grey, its own dashes, below the
    data, with ``label`` (default ``REFERENCE[kind]["label"]``) for the legend. Returns the Line2D."""
    st = REFERENCE[kind]
    return ax.axhline(float(y), color=st["color"], ls=st["ls"], lw=st["lw"], zorder=_Z_REF,
                      label=st["label"] if label is None else label)


def reference_handle(kind: str, label: str | None = None):
    """Legend proxy of a reference line (for a shared legend of several panels)."""
    from matplotlib.lines import Line2D

    st = REFERENCE[kind]
    return Line2D([], [], color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"] if label is None else label)


# ------------------------------------------------------------------------------------------------ axes and legend
def fmt_steps(x, _pos=None) -> str:
    """Tick text of a step count: 0, 500, 2.5k, 100k, 1M, 1.5M."""
    if x == 0:
        return "0"
    a = abs(x)
    if a >= 1e6:
        return f"{x / 1e6:g}M"
    if a >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def steps_axis(ax, budget=None, label=True, nbins: int = 5) -> None:
    """x axis of environment steps: 0..``budget`` (when given), at most ``nbins`` intervals of 1/2/2.5/5 x 10^k (0, 100k,
    ..., 500k; 0, 200k, ..., 1M), labelled :data:`STEPS_LABEL` (``label=False``: no label, ``label="..."``: that text)."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    if budget:
        ax.set_xlim(0, float(budget))
    ax.xaxis.set_major_locator(MaxNLocator(nbins, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_steps))
    if label:
        ax.set_xlabel(STEPS_LABEL if label is True else str(label))


def _finite(values) -> np.ndarray:
    parts = [np.asarray(v, float).ravel() for v in values if v is not None]
    v = np.concatenate(parts) if parts else np.zeros(0)
    return v[np.isfinite(v)]


def _kth(Y, k: int, largest: bool) -> np.ndarray:
    """Per column the ``k``-th smallest (``largest``: largest) finite value of ``Y``; NaN where fewer than ``k + 1``
    finite values (so the most extreme seed can be left out only where at least two remain)."""
    Y = np.atleast_2d(np.asarray(Y, float))
    S = np.sort(np.where(np.isfinite(Y), -Y if largest else Y, np.inf), axis=0)
    n = np.isfinite(Y).sum(0)
    out = np.full(Y.shape[1], np.nan)
    ok = n >= max(3, k + 1)
    if Y.shape[0] > k:
        out[ok] = S[k, ok]
    return -out if largest else out


def _band_extent(item) -> tuple:
    """``(full_lo, full_hi, trim_lo, trim_hi, mean)`` of a band: a :class:`Curve` (its band and, when drawn, its seed
    lines; trimmed = the second lowest / highest seed at every step where at least three seeds exist) or a ``(lo, hi)`` /
    ``(lo, hi, Y)`` tuple."""
    if isinstance(item, Curve):
        lo, hi, Y, mean = item.lo, item.hi, item.Y, item.mean
        parts_lo, parts_hi = [lo], [hi]
        if item.seed_lines and Y is not None:
            parts_lo.append(Y)
            parts_hi.append(Y)
    else:
        lo, hi = item[0], item[1]
        Y = item[2] if len(item) > 2 else None
        mean = None
        parts_lo, parts_hi = [lo], [hi]
    flo, fhi = _finite(parts_lo), _finite(parts_hi)
    tlo = thi = None
    if Y is not None and np.size(Y):
        a, b = _finite([_kth(Y, 1, False)]), _finite([_kth(Y, 1, True)])
        tlo = float(a.min()) if a.size else None
        thi = float(b.max()) if b.size else None
    return (float(flo.min()) if flo.size else None, float(fhi.max()) if fhi.size else None, tlo, thi, mean)


def fit_ylim(ax, *values, bands=(), pad: float = 0.05, cap: float | None = ROBUST_CAP) -> bool:
    """Fit the y limits; returns True when a band (or a seed line) was clipped at the axes (pass it to
    :func:`band_caption` as ``clipped=``).

    ``values`` (numbers or arrays: the mean lines, the reference lines) are always shown in full. ``bands`` (the
    :class:`Curve` objects, or ``(lo, hi[, Y])`` tuples) are robust: a band is shown in full while it stays within
    ``cap`` x the span of ``values`` beyond them (default :data:`ROBUST_CAP` = half the span) or within its second most
    extreme seed (a Curve with at least three seeds); further out, e.g. one seed that collapses or one huge outlier, it
    is clipped at the axes instead of squashing every other curve. ``cap=None``: bands always in full. ``pad`` of the
    final span is added on both sides, so a reference line never sits on the axis. No value: the limits are left alone;
    a single value: +-0.5 x max(|value|, 1) around it. (Without ``bands`` this is the plain fit of every value given.)"""
    ext = [_band_extent(b) for b in (bands or ()) if b is not None]
    core = _finite(list(values) + [e[4] for e in ext])
    if not core.size or cap is None:                        # nothing to anchor the cap: every band in full
        core = _finite([core] + [np.array([e[0], e[1]], dtype=float) for e in ext if e[0] is not None])
        ext_used = []
    else:
        ext_used = ext
    if not core.size:
        return False
    clo, chi = float(core.min()), float(core.max())
    span = chi - clo if chi > clo else max(abs(clo), 1.0)
    lo, hi = clo, chi
    for flo, fhi, tlo, thi, _ in ext_used:
        if flo is not None:
            allow = clo - cap * span if tlo is None else min(clo - cap * span, tlo)
            lo = min(lo, max(flo, allow))
        if fhi is not None:
            allow = chi + cap * span if thi is None else max(chi + cap * span, thi)
            hi = max(hi, min(fhi, allow))
    if hi - lo <= 0:
        half = 0.5 * max(abs(lo), 1.0)
        lo, hi = lo - half, hi + half
    else:
        s = hi - lo
        lo, hi = lo - pad * s, hi + pad * s
    ax.set_ylim(lo, hi)
    return any((flo is not None and flo < lo) or (fhi is not None and fhi > hi) for flo, fhi, *_ in ext_used)


def value_axis(ax, label=RETURN_LABEL, nbins: int = 5, axis: str = "y") -> None:
    """A value axis: at most ``nbins`` intervals, plain numbers without offset; ``label`` (None/False: none)."""
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(MaxNLocator(nbins, steps=[1, 2, 2.5, 5, 10]))
    f = ScalarFormatter(useOffset=False, useMathText=True)
    f.set_scientific(True)
    f.set_powerlimits((-4, 6))
    a.set_major_formatter(f)
    if label:
        (ax.set_ylabel if axis == "y" else ax.set_xlabel)(label)


def _entry(item):
    """``(handle, label)`` of a legend item: a Curve, an artist with a label, or a ``(handle, label)`` pair."""
    if isinstance(item, Curve):
        return item.legend_handle, item.label
    if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], str):
        return item
    return item, item.get_label()


def _text_widths(fig, labels, fontsize: float) -> list:
    """Widths in points of ``labels`` at ``fontsize`` in the current text font (measured by the figure's renderer;
    ~0.5 em per character where no renderer is available)."""
    from matplotlib.font_manager import FontProperties

    prop = FontProperties(size=fontsize)
    try:
        r = fig.canvas.get_renderer()
        return [r.get_text_width_height_descent(s, prop, ismath=False)[0] * 72.0 / fig.dpi if s else 0.0 for s in labels]
    except Exception:                                       # noqa: BLE001 - a canvas without a renderer
        return [0.5 * fontsize * len(s) for s in labels]


def _legend_rows(groups, ncol: int, blank) -> list:
    rows = []
    for g in groups:
        for i in range(0, len(g), ncol):
            r = g[i:i + ncol]
            rows.append(r + [blank] * (ncol - len(r)))
    return rows


def _fit_ncol(fig, groups, fontsize: float, blank, handlelength: float | None = None, avail_in: float | None = None) -> int:
    """The largest number of columns whose row-by-row layout of ``groups`` (sharing the column widths) fits the figure
    width (``avail_in``: that many inches instead), then balanced (4 entries in 2 x 2 rather than 3 + 1); ``handlelength`` (em)
    replaces ``legend.handlelength``. :func:`legend` calls it for one group at a time (through :func:`_group_layout`)."""
    import matplotlib

    p = matplotlib.rcParams
    hl = p["legend.handlelength"] if handlelength is None else float(handlelength)
    key = (hl + p["legend.handletextpad"]) * fontsize
    gap, pad = p["legend.columnspacing"] * fontsize, 2 * p["legend.borderpad"] * fontsize
    width = {l: w for (_, l), w in zip([e for g in groups for e in g],
                                       _text_widths(fig, [l for g in groups for _, l in g], fontsize))}
    avail = (fig.get_figwidth() if avail_in is None else float(avail_in)) * 72.0 - 4.0
    best = 1
    for ncol in range(max(len(g) for g in groups), 0, -1):
        rows = _legend_rows(groups, ncol, blank)
        cols = [max((key + width.get(r[c][1], 0.0)) if r[c][1] else 0.0 for r in rows) for c in range(ncol)]
        if sum(cols) + gap * (ncol - 1) + pad <= avail:
            best = ncol
            break
    return max(math.ceil(len(g) / math.ceil(len(g) / best)) for g in groups)


def _dashed(handle) -> bool:
    """Whether a legend key (an artist or a tuple of artists) draws a dash pattern (markers, solid lines and patches
    do not)."""
    for h in (handle if isinstance(handle, tuple) else (handle,)):
        try:
            ls = h.get_linestyle()
        except AttributeError:
            continue
        for s in (ls if isinstance(ls, list) else [ls]):     # collections: [(offset, dashes or None), ...]
            if (bool(s[1]) if isinstance(s, tuple) else s not in ("-", "solid", "None", "none", "", " ")):
                return True
    return False


def _group_layout(fig, group, fontsize: float, blank, width: float | None = None) -> tuple:
    """``(ncol, Legend keyword arguments)`` of one legend group: the columns that fit the figure width, balanced
    (:func:`_fit_ncol`), except that a group of 3 entries is never laid out 2 + 1 (the lone entry of the second row would
    set the width of the first column: e.g. 'Seed' followed by a wide gap because '95% percentile-bootstrap CI' sits
    under it). Such a group goes in one row with keys shortened down to :data:`LEGEND_MIN_HANDLE` em when that fits and
    no key carries a dash pattern (dashes need the full key), else in one column."""
    import matplotlib

    n = _fit_ncol(fig, [group], fontsize, blank, avail_in=width)
    if len(group) != 3 or n != 2:
        return n, {}
    if not any(_dashed(h) for h, _ in group):
        hl = float(matplotlib.rcParams["legend.handlelength"])
        for k in range(1, int(round((hl - LEGEND_MIN_HANDLE) * 10)) + 1):
            if _fit_ncol(fig, [group], fontsize, blank, handlelength=round(hl - 0.1 * k, 2), avail_in=width) == 3:
                return 3, {"handlelength": round(hl - 0.1 * k, 2)}
    return 1, {}


def _row_order(group, ncol: int, blank) -> tuple:
    """Handles and texts of one group in matplotlib's column-first fill order, so that the entries read row by row."""
    rows = _legend_rows([group], ncol, blank)
    order = [rows[r][c] for c in range(ncol) for r in range(len(rows))]
    return [h for h, _ in order], [l for _, l in order]


def _stack_groups(fig, leg, parts, kw) -> None:
    """Put the entry blocks of further groups under the first group's block inside the one Legend ``leg``, each block
    with its own column widths, centred (a Legend has one shared column grid, so a long label of one group would widen
    a column of another). Uses matplotlib's Legend layout boxes (``_legend_handle_box`` / ``_legend_box``)."""
    from matplotlib.legend import Legend
    from matplotlib.offsetbox import VPacker

    boxes = [leg._legend_handle_box]
    for handles, texts, n, extra in parts:
        sub = Legend(fig, handles, texts, ncol=n, **kw, **extra)   # never added to the figure: only its block is used
        boxes.append(sub._legend_handle_box)
        leg.texts.extend(sub.texts)
        leg.legend_handles.extend(sub.legend_handles)
    size = leg._fontsize
    box = VPacker(pad=leg.borderpad * size, sep=leg.labelspacing * size, align="center",
                  children=[leg._legend_title_box, *boxes])
    box.set_figure(fig)
    box.axes = leg.axes
    box.set_offset(leg._findoffset)                         # placed like the box it replaces (Legend._set_loc)
    leg._legend_box = box
    leg.stale = True


def legend(fig, *groups, where: str = "below", ncol: int | None = None, fontsize: float | None = None, ax=None,
           width: float | None = None):
    """ONE shared legend for the whole figure, frameless, outside the data area: below the panels (``where="below"``)
    or above them (``"top"``); or centred in ``ax`` (an axes switched off, e.g. the empty cells of a grid's last row), its
    columns fitted to ``width`` inches (default: the axes' current width). Every group (a list of :class:`Curve` / labelled artists / ``(handle, label)`` pairs) is
    its own centred block of rows with its own column widths (algorithms, then reference lines, then marks), so a long
    label of one group never pushes the entries of another apart; entries read row by row; ``ncol`` per group (default:
    what fits the figure width, balanced: 5 entries in 3 + 2 rather than 4 + 1; 3 entries never 2 + 1 but in one row,
    with shorter keys for markers / solid lines when needed, else in one column: :func:`_group_layout`). Returns the
    Legend (None when there is no entry)."""
    import matplotlib
    from matplotlib.font_manager import FontProperties
    from matplotlib.layout_engine import ConstrainedLayoutEngine
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D

    groups = [[_entry(i) for i in g if i is not None] for g in groups]
    groups = [[(h, l) for h, l in g if l and not l.startswith("_")] for g in groups]
    groups = [g for g in groups if g]
    if not groups:
        return None
    fs_ = float(fontsize) if fontsize else FontProperties(size=matplotlib.rcParams["legend.fontsize"]).get_size_in_points()
    blank = (Line2D([], [], color="none", lw=0), "")
    if ax is not None and width is None:
        width = ax.get_position().width * fig.get_figwidth()
    if ncol is not None:
        layout = [(max(1, min(int(ncol), len(g))), {}) for g in groups]
    else:
        layout = [_group_layout(fig, g, fs_, blank, width=width) for g in groups]
    cols = [max(1, n) for n, _ in layout]
    extras = [x for _, x in layout]
    kw = dict(frameon=False, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, fontsize=fs_)
    constrained = isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine)
    if where == "top":
        place = dict(loc="outside upper center") if constrained else dict(loc="lower center", bbox_to_anchor=(0.5, 1.0))
    else:
        place = dict(loc="outside lower center") if constrained else dict(loc="upper center", bbox_to_anchor=(0.5, 0.0))
    handles, texts = _row_order(groups[0], cols[0], blank)
    if ax is not None:
        leg = ax.legend(handles, texts, ncol=cols[0], loc="center", **kw, **extras[0])
    else:
        leg = fig.legend(handles, texts, ncol=cols[0], **place, **kw, **extras[0])
    if len(groups) > 1:
        try:
            _stack_groups(fig, leg, [(*_row_order(g, n, blank), n, x) for g, n, x in zip(groups[1:], cols[1:], extras[1:])], kw)
        except (AttributeError, TypeError):                 # other matplotlib internals: one shared column grid
            leg.remove()
            n = max(1, int(ncol if ncol is not None else _fit_ncol(fig, groups, fs_, blank, avail_in=width)))
            rows = _legend_rows(groups, n, blank)
            order = [rows[r][c] for c in range(n) for r in range(len(rows))]
            leg = (ax.legend([h for h, _ in order], [l for _, l in order], ncol=n, loc="center", **kw) if ax is not None else
                   fig.legend([h for h, _ in order], [l for _, l in order], ncol=n, **place, **kw))
    return leg


def caption(text: str, width: int | None = 110) -> str:
    """A caption string (whitespace normalised, wrapped at ``width`` characters; None: one line) for printing under a
    figure in a notebook or a report."""
    t = " ".join(str(text).split())
    return textwrap.fill(t, width=width) if width else t


# ------------------------------------------------------------------------------------------------ saving and showing
def _draw_rc() -> dict:
    p = rc()
    return {k: p[k] for k in _DRAW_KEYS}


def _savefig(fig, target, fmt: str, dpi: int) -> None:
    import matplotlib

    meta = {"CreationDate": None} if fmt == "pdf" else None       # reproducible PDF bytes
    with matplotlib.rc_context(_draw_rc()):
        fig.savefig(target, format=fmt, dpi=dpi, bbox_inches="tight", pad_inches=PAD_INCHES, facecolor=SURFACE,
                    edgecolor=SURFACE, metadata=meta)


def save(fig, path, formats=("pdf", "png"), dpi: int = SAVE_DPI) -> list:
    """Save ``fig`` as ``<path>.<fmt>`` for every format (default vector PDF with embedded TrueType fonts + PNG at
    ``dpi``); ``path`` is given WITHOUT suffix (a final .pdf/.png/.svg is dropped; other dots stay, e.g.
    ``figures/Hopper-v5.partial``). Creates the folder. Returns the paths in ``formats`` order."""
    path = pathlib.Path(path)
    if path.suffix.lower() in (".pdf", ".png", ".svg"):
        path = path.with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for fmt in formats:
        p = path.with_name(f"{path.name}.{fmt}")
        _savefig(fig, p, fmt, dpi)
        out.append(p)
    return out


def _png_size(data: bytes) -> tuple:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    return struct.unpack(">II", data[16:24])


def show(obj, dpi: int = SAVE_DPI) -> None:
    """Display a saved PNG (path) or a Figure (rendered exactly as :func:`save` renders the PNG) inline, scaled to
    :data:`DISPLAY_PPI` screen pixels per inch of the figure. Does nothing outside IPython."""
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
    except ImportError:                                     # pragma: no cover - plain Python without IPython
        return
    if get_ipython() is None:
        return
    if isinstance(obj, (str, pathlib.Path)):
        data = pathlib.Path(obj).read_bytes()
        try:
            from PIL import Image as _PIL
            with _PIL.open(io.BytesIO(data)) as im:
                d = im.info.get("dpi", (dpi, dpi))[0] or dpi
        except Exception:                                   # noqa: BLE001 - no Pillow / no dpi: the save default
            d = dpi
    else:
        buf = io.BytesIO()
        _savefig(obj, buf, "png", dpi)
        data, d = buf.getvalue(), dpi
    w, _ = _png_size(data)
    display(Image(data=data, format="png", width=int(round(w / float(d) * DISPLAY_PPI))))


# ------------------------------------------------------------------------------------------------ heatmaps
def sequential_cmap(name: str = "figstyle_blue"):
    """Single-hue sequential colormap (blue, light -> dark: :data:`SEQUENTIAL`). Never a rainbow; label its colourbar."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, list(SEQUENTIAL))


def _lum(rgb) -> float:
    c = [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in rgb[:3]]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def text_on(rgba) -> str:
    """Cell-text colour on a fill ``rgba``: :data:`INK` or white, whichever has the higher contrast."""
    from matplotlib.colors import to_rgb

    L = _lum(to_rgb(rgba))
    ink = _lum(to_rgb(INK))
    c_ink = (max(L, ink) + 0.05) / (min(L, ink) + 0.05)
    c_white = 1.05 / (L + 0.05)
    return INK if c_ink >= c_white else "#ffffff"


def _selftest() -> None:                                    # pragma: no cover - quick numeric check, no figure
    s = np.array([1.0, 2.0, 3.0])
    g, Y = align([(s, [1, 2, 3]), (s, [3, 4, 5]), ((1, 2, 3, 4), (0, 0, 0, 9))], budget=3)
    assert np.array_equal(g, s) and Y.shape == (3, 3)
    m, lo, hi = mean_band(g, Y, "minmax")
    assert np.allclose(m, [4 / 3, 2, 8 / 3]) and np.allclose(lo, [0, 0, 0]) and np.allclose(hi, [3, 4, 5])
    m1, lo1, _ = mean_band(g, Y[:1], "minmax")
    assert np.allclose(m1, [1, 2, 3]) and np.isnan(lo1).all()
    assert fmt_steps(500_000) == "500k" and fmt_steps(1_000_000) == "1M" and fmt_steps(2500) == "2.5k"
    assert check_band(None) == "minmax" and check_band("CI95") == "ci95" and check_band(False) == "none"
    labs = ["TEMPO", "TEMPO[bn256]", "TEMPO[wide]", "SAC"]
    assert tempo_style("TEMPO[bn256]", "TEMPO[bn256]", labs) == dict(report.STYLE["TEMPO"])
    assert tempo_style("TEMPO", "TEMPO[bn256]", labs)["color"] == VARIANT_STYLES[0]["color"]
    assert tempo_style("TEMPO[wide]", "TEMPO[bn256]", labs[::-1])["color"] == VARIANT_STYLES[1]["color"]
    assert tempo_style("SAC", "TEMPO", labs) == dict(report.STYLE["SAC"])
    assert [band_alpha(n) for n in (0, 1, 2, 3, 4, 5, 9)] == [0.2, 0.2, 0.2, 0.14, 0.14, 0.1, 0.1]

    class _Ax:                                              # records set_ylim
        def set_ylim(self, lo, hi):
            self.lim = (lo, hi)

    ax = _Ax()
    Yo = np.array([[0, 1, 2], [0, 1, 2.2], [0, -20, 1.8]], float)   # one seed dips far below
    lo, hi = Yo.min(0), Yo.max(0)
    assert fit_ylim(ax, Yo.mean(0), bands=[(lo, hi, Yo)], pad=0) and ax.lim[0] > -20 and ax.lim[1] == 2.2
    assert not fit_ylim(ax, Yo.mean(0), bands=[(lo, hi, Yo)], pad=0, cap=None) and ax.lim[0] == -20
