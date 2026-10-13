#!/usr/bin/env python3
"""make_approx_plots.py — Figure 9: runtime vs error of approximate inference.

For each program, a log-log scatter of every (budget, repeat) run of Turing and
Siren: inference time against relative error |θ - θ̂| / |θ| versus Hybrid Pluck's
exact answer (the fixed copy in reference/exact_answers.csv). Hybrid Pluck's own runs sit on a separate strip at error 0 below a
broken axis. One point per repeat, so the spread across repeats of the same budget
is visible.

The paper's panels are (a) hmm_second_order_n14 and (b) king_ei_d20 (the caption
of Fig. 9b says 1/15, but the plotted run is the 1/20 dataset fraction).

Usage:
    python make_approx_plots.py --run-dir DIR [--out-dir DIR] [--program PROG ...]
Writes <out-dir>/<program>_approx_inf_comparison.pdf (default out-dir:
<run_dir>/analysis).
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import bench_common as bc  # noqa: E402
import compute_error as ce  # noqa: E402

PROGRAMS = ["hmm_second_order_n14", "king_ei_d20"]
APPROX_SYSTEMS = ["turing", "siren"]  # the systems shown in the paper's Fig. 9

STYLE = {
    "pluck": {"color": "#1f77b4", "marker": "o", "size": 60, "label": "Hybrid Pluck"},
    "turing": {"color": "#8c564b", "marker": "P", "size": 60, "label": "Turing"},
    "siren": {"color": "#17becf", "marker": "*", "size": 100, "label": "Siren"},
}
FONTSIZE = 20


def load_rows(run_dir: str, program: str) -> list[dict]:
    """Every ok repeat of `program` across all results_*.csv files in the run."""
    rows = []
    for path in glob.glob(os.path.join(run_dir, "results_*.csv")):
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                if row["program"] == program and row["status"] == "ok":
                    rows.append(row)
    return rows


def points(run_dir: str, program: str) -> dict[str, list[tuple[float, float]]]:
    """{system: [(infer_s, rel_err), ...]}; pluck's points have rel_err 0."""
    gt, _src = ce.ground_truth(program)
    out: dict[str, list[tuple[float, float]]] = {}
    for row in load_rows(run_dir, program):
        system = row["system"]
        t = bc.parse_cell(row["infer_s"])
        if t is None:
            continue
        if system == "pluck":
            out.setdefault(system, []).append((t, 0.0))
            continue
        v = bc.parse_cell(row["result"])
        if system not in APPROX_SYSTEMS or v is None or not gt:
            continue
        err = abs(ce._adjust_approx(system, program, v) - gt) / abs(gt)
        out.setdefault(system, []).append((t, err))
    return out


def plot(pts: dict[str, list[tuple[float, float]]], title: str, path: str) -> None:
    fig = plt.figure(figsize=(6, 4.5), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[15, 1])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_top.set_xscale("log")
    ax_top.set_yscale("log")

    def scatter(ax, system):
        st = STYLE[system]
        xs = [t for t, _ in pts[system]]
        ys = [e for _, e in pts[system]]
        ax.scatter(xs, ys, color=st["color"], marker=st["marker"], s=st["size"],
                   edgecolors="white", linewidths=0.8)

    for system in APPROX_SYSTEMS:
        if system in pts:
            # a zero error cannot sit on the log axis; such runs are exact hits
            pts[system] = [(t, max(e, 1e-12)) for t, e in pts[system]]
            scatter(ax_top, system)
    if "pluck" in pts:
        scatter(ax_bottom, "pluck")

    ax_bottom.set_ylim(-0.5, 0.5)
    ax_top.grid(True, which="both", ls="--", alpha=0.4)
    ax_top.spines["bottom"].set_visible(False)
    ax_bottom.spines["top"].set_visible(False)
    ax_top.tick_params(which="both", labelbottom=False, bottom=False)
    ax_bottom.set_yticks([0])
    ax_bottom.set_yticklabels(["0"])
    ax_bottom.grid(True, which="both", axis="x", ls="--", alpha=0.4)
    ax_top.set_ylabel(r"Relative Error $|\theta - \hat{\theta}| / |\theta|$", fontsize=FONTSIZE - 6)
    ax_bottom.set_xlabel("Inference Time (s)", fontsize=FONTSIZE)
    ax_top.set_title(title, fontsize=FONTSIZE - 6)

    # diagonal break marks between the two axes
    d = 0.015
    kw = dict(transform=ax_top.transAxes, color="black", clip_on=False, lw=1.2)
    ax_top.plot((-d, +d), (-d, +d), **kw)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kw)
    kw.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kw)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)

    handles = [
        mlines.Line2D([], [], color="none", marker=STYLE[s]["marker"],
                      markerfacecolor=STYLE[s]["color"], markeredgecolor="white",
                      markersize=STYLE[s]["size"] ** 0.5, linestyle="None",
                      label=STYLE[s]["label"])
        for s in ["pluck", *APPROX_SYSTEMS] if s in pts
    ]
    if handles:
        ax_top.legend(handles=handles, fontsize=FONTSIZE - 8)
    fig.savefig(path)
    plt.close(fig)


def title_for(program: str) -> str:
    fam = bc.parse_family(program)
    if fam and fam[0] == "king_ei":
        return f"Ecological Inference (frac. dataset = 1/{fam[2]})"
    if fam and fam[0] == "hmm_second_order":
        return f"Second-Order HMM (n = {fam[2]})"
    return program


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="benchmark run dir")
    ap.add_argument("--out-dir", default=None, help="output dir (default: <run_dir>/analysis)")
    ap.add_argument("--program", action="append", default=None,
                    help=f"program to plot (repeatable; default: {' '.join(PROGRAMS)})")
    args = ap.parse_args()
    out_dir = args.out_dir or os.path.join(args.run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    for program in args.program or PROGRAMS:
        pts = points(args.run_dir, program)
        n = {s: len(v) for s, v in pts.items()}
        if not any(s in pts for s in APPROX_SYSTEMS):
            print(f"skip    : {program} (no Turing/Siren results in this run)")
            continue
        path = os.path.join(out_dir, f"{program}_approx_inf_comparison.pdf")
        plot(pts, title_for(program), path)
        print(f"wrote   : {path}  points={n}")


if __name__ == "__main__":
    main()
