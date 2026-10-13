#!/usr/bin/env python3
"""make_pluck_scaling_plots.py — pluck-internals scaling curves, one per family.

Where make_scaling_plots.py compares PPLs (inference time / error vs size, one
line per PPL), this looks *inside pluck*: for each scaling family it draws one
log-log plot of how pluck's internal representations and per-stage timings grow
with the scaling parameter (the n/d suffix of the variant). Six series on two
y-axes:

  counts (left y-axis, solid):
    - BDD node count        (BDD guard nodes: sum)
    - interned SPN size     (SPN nodes (DAG): all-worlds-unique)
    - uninterned SPN size   (SPN nodes (TREE): sum)
  times (right y-axis, dashed):
    - symbolic-execution    (time (s): symbolic-exec)
    - WMC                    (time (s): wmc)
    - posterior computation  (time (s): posterior)

These metrics are NOT in the results CSV; they live in pluck's per-program logs
(the `=== PLUCK_STATS ===` block emitted to stderr under PLUCK_STATS=1, captured
at <run_dir>/logs/pluck__rep<k>_<program>.log). This script parses those logs
directly — no rerun needed. Counts are deterministic across repeats; timings are
aggregated across repeats with the median (matching bench_common's convention).

Output: one <family>.pdf + <family>.png per family, plus an overview.png contact
sheet, under <run_dir>/analysis/pluck_scaling/.

Usage:
    uv run python make_pluck_scaling_plots.py [--run-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter, NullLocator

import bench_common as bc
import make_table as mt

# Family display titles, verbatim from make_table's TABLE2_GROUPS (as in
# make_scaling_plots.py) so every plot labels each model identically.
FAMILY_TITLE = {g["family"]: g["title"]
                for g in mt.TABLE2_GROUPS if g["kind"] == "family"}


def family_title(fam: str) -> str:
    return FAMILY_TITLE.get(fam, fam).replace("--", "–")


# ---- x-axis mapping, shared verbatim with make_scaling_plots.py --------------
AXIS_LABEL = {"n": "problem size $n$", "d": "tree depth $d$"}

# Families whose scaling parameter is the inverse of the suffix (king_ei's d is
# an inverse depth: the problem shrinks as d grows), plotted as x = 1/size. Kept
# in sync with make_scaling_plots.INVERSE_FAMILIES.
INVERSE_FAMILIES = {"king_ei"}


def x_value(fam: str, size: int) -> float:
    return 1.0 / size if fam in INVERSE_FAMILIES else float(size)


def x_ticklabel(fam: str, size: int) -> str:
    return f"1/{size}" if fam in INVERSE_FAMILIES else str(size)


def x_axislabel(fam: str, letter: str) -> str:
    if fam in INVERSE_FAMILIES:
        return f"$1/{letter}$"
    return AXIS_LABEL.get(letter, "problem size")


# ---- the six pluck-internal series, styled for a two-axis plot ---------------
# Counts (left axis) are solid; times (right axis) are dashed. Colours and order
# are fixed so the legend is stable across families.
COUNT_SERIES = [
    ("bdd",          {"color": "#1f77b4", "marker": "o", "label": "BDD nodes"}),
    ("spn_interned", {"color": "#2ca02c", "marker": "s", "label": "SPN (interned)"}),
    ("spn_tree",     {"color": "#9467bd", "marker": "^", "label": "SPN (tree)"}),
]
TIME_SERIES = [
    ("t_symbolic",   {"color": "#d62728", "marker": "D", "label": "symbolic-exec (s)"}),
    ("t_wmc",        {"color": "#ff7f0e", "marker": "v", "label": "WMC (s)"}),
    ("t_posterior",  {"color": "#8c564b", "marker": "P", "label": "posterior (s)"}),
]

# ---- PLUCK_STATS field extractors -------------------------------------------


def parse_log_file(path: str) -> list[dict]:
    """Parse one pluck log into the metrics these plots use, one dict per repeat
    (each summed across that repeat's PLUCK_STATS blocks). Repeats without a
    complete block (e.g. a run that timed out or OOM'd before printing) are skipped.

    A thin renaming over bench_common.parse_pluck_stats_log(), which owns the regexes.
    """
    # Same completeness bar as before: a block missing the BDD/SPN size lines is not
    # a usable data point here, even though its timings parsed.
    need = ("bdd_sum", "spn_tree_sum", "spn_dag_unique",
            "t_symbolic", "t_wmc", "t_posterior")
    return [{
        "bdd": stats["bdd_sum"],
        # optional, so logs predating the "BDD vars allocated" line still parse
        "n_vars": stats.get("bdd_vars", 0),
        "spn_tree": stats["spn_tree_sum"],
        "spn_interned": stats["spn_dag_unique"],
        "t_symbolic": stats["t_symbolic"],
        "t_wmc": stats["t_wmc"],
        "t_posterior": stats["t_posterior"],
    } for stats in bc.parse_pluck_stats_log(path) if all(k in stats for k in need)]


def program_metrics(run_dir: str, program: str) -> dict | None:
    """Median of each PLUCK_STATS metric across all repeats of `program` (every
    repeat in logs/pluck_<program>.log, or in per-repeat logs/pluck__rep*_<program>.log).
    Returns None if no repeat produced a complete block."""
    paths = bc.pluck_log_paths(run_dir, program)
    parsed = [p for path in paths for p in parse_log_file(path)]
    if not parsed:
        return None
    keys = ("bdd", "n_vars", "spn_tree", "spn_interned",
            "t_symbolic", "t_wmc", "t_posterior")
    return {k: statistics.median([p[k] for p in parsed]) for k in keys}


def family_series(run_dir: str, fam: str, info: dict):
    """{metric_key: [(size, value), ...]} sorted by size, for every sized variant
    of `fam` that has a parseable log."""
    series: dict[str, list[tuple[int, float]]] = {}
    for size, prog in sorted((s, p) for s, p in info["sizes"].items() if s is not None):
        m = program_metrics(run_dir, prog)
        if m is None:
            continue
        for key in m:
            v = m[key]
            if v is not None and v > 0:
                series.setdefault(key, []).append((size, v))
    return series


def _apply_size_ticks(ax, fam: str, sizes: set):
    """Plain integer (or 1/size) ticks at the actual data sizes, blanking labels
    that would collide on the log axis. Copied from make_scaling_plots.plot_family."""
    pairs = sorted((x_value(fam, s), s) for s in sizes)
    ticks = [xv for xv, _ in pairs]
    labels, kept_log, min_gap = [], None, 0.06
    for xv, s in pairs:
        lg = math.log10(xv)
        if kept_log is None or lg - kept_log >= min_gap:
            labels.append(x_ticklabel(fam, s))
            kept_log = lg
        else:
            labels.append("")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter(labels))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", labelsize=8)
    if len(ticks) > 1:
        ax.set_xlim(ticks[0] / 1.15, ticks[-1] * 1.15)


def plot_family(ax, run_dir: str, fam: str, info: dict) -> bool:
    """Draw one family's six pluck-internal series onto `ax` (counts) and a twin
    axis (times), both log-log. Returns False if there's no data to plot."""
    series = family_series(run_dir, fam, info)
    if not series:
        return False
    letter = info["letter"]
    ax2 = ax.twinx()
    sizes = set()

    for key, st in COUNT_SERIES:
        pts = series.get(key)
        if not pts:
            continue
        sizes.update(p[0] for p in pts)
        ax.plot([x_value(fam, s) for s, _ in pts], [v for _, v in pts],
                marker=st["marker"], color=st["color"], label=st["label"],
                linestyle="-", markersize=5, linewidth=1.5)
    for key, st in TIME_SERIES:
        pts = series.get(key)
        if not pts:
            continue
        sizes.update(p[0] for p in pts)
        ax2.plot([x_value(fam, s) for s, _ in pts], [v for _, v in pts],
                 marker=st["marker"], color=st["color"], label=st["label"],
                 linestyle="--", markersize=5, linewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax2.set_yscale("log")
    _apply_size_ticks(ax, fam, sizes)

    ax.set_xlabel(x_axislabel(fam, letter))
    ax.set_ylabel("node count")
    ax2.set_ylabel("time (s)")
    ax.set_title(family_title(fam))
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)

    # Single legend combining both axes (counts then times, fixed order).
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="benchmark run dir")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: <run_dir>/analysis/pluck_scaling)")
    args = ap.parse_args()

    run_dir = args.run_dir
    if not run_dir or not os.path.isdir(run_dir):
        sys.exit(f"error: run dir not found: {run_dir!r}")
    if not os.path.isdir(os.path.join(run_dir, "logs")):
        sys.exit(f"error: no logs/ dir under {run_dir} (pluck logs are required)")

    out_dir = args.out_dir or os.path.join(run_dir, "analysis", "pluck_scaling")
    os.makedirs(out_dir, exist_ok=True)

    # families() takes the program universe (union of every PPL's source dir).
    source = bc.source_program_sets()
    universe = set().union(*source.values()) if source else set()
    fams = bc.families(universe)

    drawn = []
    for fam in sorted(fams):
        fig, ax = plt.subplots(figsize=(4.8, 3.5))
        if plot_family(ax, run_dir, fam, fams[fam]):
            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(out_dir, f"{fam}.{ext}"), dpi=150)
            drawn.append(fam)
        plt.close(fig)

    if drawn:
        ncol = 3
        nrow = math.ceil(len(drawn) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 3.5 * nrow))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, fam in zip(axes, drawn):
            plot_family(ax, run_dir, fam, fams[fam])
        for ax in axes[len(drawn):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "overview.png"), dpi=130)
        plt.close(fig)

    skipped = sorted(set(fams) - set(drawn))
    print(f"run dir : {run_dir}")
    print(f"plotted : {drawn}")
    if skipped:
        print(f"no data : {skipped}")
    print(f"wrote   : {out_dir}/<family>.{{pdf,png}} + overview.png")


if __name__ == "__main__":
    main()
