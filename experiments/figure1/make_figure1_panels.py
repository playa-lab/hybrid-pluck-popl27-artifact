#!/usr/bin/env python3
"""Generate the panels for the paper's Figure 1 ("gallery of exact inferences").

Runs each row's program through the pluck CLI with --json, converts the exact
posterior mixtures (Beta / Gaussian / Gamma components weighted by their exact
posterior probabilities) into densities and marginals, and renders one PDF per
panel into --out. Panels are fixed-size (no tight cropping) so they sit on a
uniform grid in the figure: data panels are 1.42in wide, the first posterior
panel 0.80in, the second 1.78in (SIR: 1.42 + 1.16), all 0.80in tall.

The programs are the listings in examples/fig_*_listing.pluck, which the paper
shows in its appendix. The SIR panels also need per-patient and per-rate
queries, so they are drawn from fig_sir.pluck here: the same listing followed
by those auxiliary queries.

Also writes fig1_metrics.tex with the per-row LoC (non-blank, non-comment
lines of the listing) and, with --time / --time-psi, the timing macros. Pluck
timing is the median over N runs of the listing of parse + symbolic-exec + WMC
+ posterior from PLUCK_STATS (the time to construct the posterior SPN, without
the flattening / display work only needed for visualization). PSI timing is the
wall-clock of the PSI port fig_<row>.psi, or t/o.

Usage (from the repository root):
    uv run --project experiments experiments/figure1/make_figure1_panels.py \
        --out results/fig1 [--time 5] [--time-psi]
"""

import argparse
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import beta as beta_dist
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]
FIG1 = REPO / "experiments" / "figure1"
PLUCK = REPO / "target" / "release" / "pluck"

# ── muted palette: gray data, one teal accent, desaturated categorical ──
INK = "#3F3F3F"       # observed data
TEAL = "#0F7680"      # primary accent: all inferred quantities
SLATE = "#7189A8"     # secondary series
SAND = "#B3986B"      # tertiary series
GRAY = "#9A9A9A"      # quaternary series / de-emphasis
LIGHT = "#C9C9C9"
BAND = "#F2F2F2"
CAT = [TEAL, SLATE, SAND, GRAY]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Linux Libertine O", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 6.5,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 1.8,
    "ytick.major.size": 1.8,
    "pdf.fonttype": 42,
})

# uniform slot geometry (inches)
H = 0.80
W_DATA, W_A, W_B = 1.42, 0.80, 1.78
W_SIRB = 1.16

ROWS = ["polyreg", "sir", "changepoint", "dawid_skene", "mixture"]
LISTINGS = {name: REPO / "examples" / f"fig_{name}_listing.pluck" for name in ROWS}
PROGRAMS = dict(LISTINGS, sir=FIG1 / "fig_sir.pluck")


def run_pluck(path, json_dir):
    """Run a program with --json, keep its output in json_dir, and return the
    parsed results."""
    out = subprocess.run([str(PLUCK), str(path), "--json"],
                         cwd=REPO, capture_output=True, text=True, check=True)
    (json_dir / f"{path.stem}.json").write_text(out.stdout)
    return json.loads(out.stdout)


def items_of(results, query):
    for r in results:
        if r["query"] == query:
            return r["kind"]["items"]
    raise KeyError(query)


def weights_of(items):
    w = np.array([np.exp(it["log_probability"]) for it in items])
    return w / w.sum()


def gauss_block(item, names):
    """Return (mean, cov) over `names`, assembling independent blocks
    block-diagonally when the variables span several of them."""
    k = len(names)
    mean = np.zeros(k)
    cov = np.zeros((k, k))
    found = set()
    for g in item["posteriors"]["gaussian"]:
        hit = [i for i, n in enumerate(names) if n in g["var_order"]]
        if not hit:
            continue
        n = g["mean"][1]
        gm = np.array(g["mean"][0])
        gc = np.array(g["cov"][0]).reshape(n, n)
        pos = [g["var_order"].index(names[i]) for i in hit]
        mean[hit] = gm[pos]
        cov[np.ix_(hit, hit)] = gc[np.ix_(pos, pos)]
        found.update(hit)
    if len(found) != k:
        missing = [names[i] for i in range(k) if i not in found]
        raise KeyError(f"no gaussian block for {missing}")
    return mean, cov


def betas_by_name(item):
    return {b["Beta"]["name"]: (b["Beta"]["a"], b["Beta"]["b"])
            for b in item["posteriors"]["beta"]}


def gammas_by_name(item):
    out = {}
    for g in item["posteriors"]["gamma"]:
        gg = g["gamma"]["Gamma"]
        out[gg["name"]] = (gg["shape"], gg["rate"])
    return out


def beta_mixture_pdf(grid, comps):
    """comps: list of (w, a, b) — exact Beta-mixture density on grid."""
    pdf = np.zeros_like(grid)
    for w, a, b in comps:
        pdf += w * beta_dist.pdf(grid, a, b)
    return pdf


def panel(w, h, title=None, l=0.17, r=0.93, b=0.21, t=0.85):
    """Fixed-size figure with one axes at explicit margins (no tight crop),
    so every panel PDF has exactly the slot dimensions."""
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([l, b, r - l, t - b])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=6, color=INK, pad=2.5)
    return fig, ax


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def parse_bool_list(s):
    return [t == "True" for t in re.findall(r"True|False", s)]


# ── row 1: polynomial regression with outliers ────────────────────────

POLY_XS = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
POLY_YS = np.array([1.7, 4.8, 7.5, 9.0, 9.7, 8.9, 7.4, 5.1, 14.1, 18.8])
DEGREES = ["Zero", "Constant", "Linear", "Quadratic", "Cubic", "Quartic", "Quintic"]
POLY_YLIM = (-2.5, 22)


def render_polyreg(results, out):
    items = items_of(results, "polyreg")
    w = weights_of(items)

    deg_probs = defaultdict(float)
    outlier_probs = np.zeros(len(POLY_XS))
    by_degree = defaultdict(list)
    for wi, it in zip(w, items):
        m = re.match(r'\(Result "(\w+)" \[([^\]]*)\] \[([^\]]*)\]', it["display"])
        deg, outl_s, curve_s = m.group(1), m.group(2), m.group(3)
        outl = parse_bool_list(outl_s)
        names = [c.strip() for c in curve_s.split(",")] if curve_s.strip() else []
        deg_probs[deg] += wi
        outlier_probs += wi * np.array(outl, dtype=float)
        by_degree[deg].append((wi, names, it))

    # data panel
    fig, ax = panel(W_DATA, H, "$(x_i, y_i)$ pairs")
    ax.scatter(POLY_XS, POLY_YS, s=7, color=INK, zorder=3)
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_ylim(*POLY_YLIM)
    ax.set_yticks([0, 10, 20])
    save(fig, out, "polyreg_data")

    # posterior over degree (vertical bars, numeric degrees)
    fig, ax = panel(W_A, H, "P(degree)", l=0.21)
    ps = [deg_probs.get(d, 0.0) for d in DEGREES[1:7]]
    ax.bar(range(6), ps, color=TEAL, width=0.62)
    ax.set_xticks(range(6))
    ax.set_xticklabels(["0", "1", "2", "3", "4", "5"])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    save(fig, out, "polyreg_degree")

    # the two leading modes: conditional curve bands, and each mode's own
    # outliers ringed in that mode's color ("this curve explains those away")
    grid = np.linspace(0.5, 10.5, 120)
    fig, ax = panel(W_B, H, "curves (by degree) + outliers", l=0.11)
    top2 = [d for d, _ in sorted(deg_probs.items(), key=lambda x: -x[1])[:2]]
    styles = [(TEAL, "-", 0.30, 0.92), (SLATE, (0, (3, 1.5)), 0.22, 0.76)]
    ringed = set()
    for deg, (col, ls, alpha, ty) in zip(top2, styles):
        comps = by_degree[deg]
        wd = sum(c[0] for c in comps)
        renorm = [(wi / wd, names, it) for wi, names, it in comps]
        qlo, qmid, qhi = curve_band(renorm, grid)
        ax.fill_between(grid, qlo, qhi, color=col, alpha=alpha, linewidth=0)
        ax.plot(grid, qmid, color=col, linewidth=0.9, linestyle=ls)
        ax.text(0.03, ty, f"deg. {DEGREES.index(deg) - 1}: {deg_probs[deg]:.2f}",
                transform=ax.transAxes, fontsize=5.2, color=col, va="top")
        cond_outl = np.zeros(len(POLY_XS))
        for wi, names, it in comps:
            m = re.match(r'\(Result "\w+" \[([^\]]*)\]', it["display"])
            cond_outl += (wi / wd) * np.array(parse_bool_list(m.group(1)), float)
        for i in np.where(cond_outl > 0.5)[0]:
            ax.scatter([POLY_XS[i]], [POLY_YS[i]], s=16, facecolor="none",
                       edgecolor=col, linewidth=0.9, zorder=4)
            ringed.add(i)
    ax.scatter(POLY_XS, POLY_YS, s=7, color=INK, zorder=3)
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_ylim(*POLY_YLIM)
    ax.set_yticks([0, 10, 20])
    save(fig, out, "polyreg_fit")

    return {"deg": {k: round(float(v), 3) for k, v in
                    sorted(deg_probs.items(), key=lambda x: -x[1])[:3]},
            "P(outlier)": outlier_probs.round(2).tolist()}


def curve_band(parsed, grid, lo=0.05, hi=0.95):
    """Exact quantile band of the posterior over y(x) (mixture of Gaussians)."""
    means, sds, ws = [], [], []
    for wi, names, it in parsed:
        if not names:
            mu_x = np.zeros_like(grid)
            sd_x = np.zeros_like(grid) + 1e-9
        else:
            mean, cov = gauss_block(it, names)
            A = np.vstack([grid ** i for i in range(len(names))]).T
            mu_x = A @ mean
            sd_x = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", A, cov, A), 1e-18))
        means.append(mu_x)
        sds.append(sd_x)
        ws.append(wi)
    means, sds, ws = np.array(means), np.array(sds), np.array(ws)

    def mix_quantile(q):
        lo_y = (means - 6 * sds).min(axis=0)
        hi_y = (means + 6 * sds).max(axis=0)
        for _ in range(40):
            mid = (lo_y + hi_y) / 2
            cdf = (ws[:, None] * norm.cdf((mid[None, :] - means) / sds)).sum(axis=0)
            lo_y = np.where(cdf < q, mid, lo_y)
            hi_y = np.where(cdf < q, hi_y, mid)
        return (lo_y + hi_y) / 2

    return mix_quantile(lo), mix_quantile(0.5), mix_quantile(hi)


# ── row 2: SIR epidemic ───────────────────────────────────────────────

SIR_HIST = [[0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
RATE_LABELS = ["recovery", "infection", "symptom (healthy)", "symptom (infected)"]
RATE_COLORS = [SLATE, TEAL, GRAY, SAND]


def render_sir(results, out):
    fig, ax = panel(W_DATA, H, "reported symptoms", l=0.25, r=0.97)
    draw_grid(ax, np.array(SIR_HIST, dtype=float), INK, kind="squares")
    style_sir_axes(ax)
    save(fig, out, "sir_data")

    probs = np.zeros((3, 10))
    for p in range(3):
        items = items_of(results, f"sir-patient-{p}")
        w = weights_of(items)
        for wi, it in zip(w, items):
            probs[p] += wi * np.array(parse_bool_list(it["display"]), dtype=float)
    fig, ax = panel(W_DATA, H, "P(infected)", l=0.25, r=0.97)
    draw_grid(ax, probs, TEAL, kind="heat")
    style_sir_axes(ax)
    save(fig, out, "sir_infected")

    items = items_of(results, "sir-rates")
    w = weights_of(items)
    name_lists = [re.findall(r"[^\[\],\s]+", it["display"]) for it in items]
    grid = np.linspace(1e-3, 1 - 1e-3, 400)
    fig, ax = panel(W_SIRB, H, "rate posteriors", l=0.12)
    for k, lab in enumerate(RATE_LABELS):
        comps = []
        for wi, names, it in zip(w, name_lists, items):
            a, b = betas_by_name(it)[names[k]]
            comps.append((wi, a, b))
        pdf = beta_mixture_pdf(grid, comps)
        ax.plot(grid, pdf, linewidth=0.9, color=RATE_COLORS[k], label=lab)
    ax.set_ylim(0, 7)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=4.6, handlelength=0.8,
              borderaxespad=0.05, labelspacing=0.25, loc="upper right")
    save(fig, out, "sir_rates")
    return {"patientB": probs[1].round(2).tolist()}


def draw_grid(ax, vals, color, kind):
    n_r, n_c = vals.shape
    for r in range(n_r):
        for c in range(n_c):
            v = vals[r, c]
            if kind == "squares":
                fc = color if v > 0.5 else "white"
                ec = color
            else:
                fc = matplotlib.colors.to_rgba(color, alpha=float(v))
                ec = LIGHT
            ax.add_patch(plt.Rectangle((c, n_r - 1 - r), 0.82, 0.82,
                                       facecolor=fc, edgecolor=ec, linewidth=0.5))
    ax.set_xlim(-0.2, n_c)
    ax.set_ylim(-0.3, n_r)


def style_sir_axes(ax):
    ax.set_yticks([2.4, 1.4, 0.4])
    ax.set_yticklabels(["pat. A", "pat. B", "pat. C"])
    ax.set_xticks([0.4, 4.4, 9.4])
    ax.set_xticklabels(["day 1", "5", "10"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, pad=1.5)


# ── row 3: coal-mining changepoints ───────────────────────────────────

COAL_YEARS = np.arange(1880, 1900)
COAL_COUNTS = [4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 0, 0, 1]


def render_changepoint(results, out):
    items = items_of(results, "changepoints")
    w = weights_of(items)
    k_probs = defaultdict(float)
    loc_probs = np.zeros(19)
    rate_mean = np.zeros(20)
    year_comps = [[] for _ in range(20)]
    for wi, it in zip(w, items):
        m = re.match(r"\(Result (\d+) \[([^\]]*)\] \[([^\]]*)\]\)", it["display"])
        k = int(m.group(1))
        locs = [int(x) for x in m.group(2).split(",") if x.strip()]
        rate_names = [s.strip() for s in m.group(3).split(",")]
        k_probs[k] += wi
        for l in locs:
            loc_probs[l] += wi
        gn = gammas_by_name(it)
        for y, n in enumerate(rate_names):
            year_comps[y].append((wi, *gn[n]))
        rate_mean += wi * np.array([gn[n][0] / gn[n][1] for n in rate_names])

    fig, ax = panel(W_DATA, H, "disasters per year", l=0.14)
    ax.bar(COAL_YEARS, COAL_COUNTS, color=INK, width=0.75)
    ax.set_xticks([1880, 1890, 1899])
    ax.set_yticks([0, 5])
    save(fig, out, "changepoint_data")

    fig, ax = panel(W_A, H, "P($k$ changes)", l=0.21)
    ks = [0, 1, 2, 3]
    ax.bar(ks, [k_probs.get(k, 0) for k in ks], color=TEAL, width=0.62)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    save(fig, out, "changepoint_k")

    # overlay: data + exact per-year rate band + strongest changepoint lines
    q05 = np.array([gamma_mix_quantile(c, 0.05) for c in year_comps])
    q95 = np.array([gamma_mix_quantile(c, 0.95) for c in year_comps])
    fig, ax = panel(W_B, H, "disaster rate + changepoint locations", l=0.10)
    ax.bar(COAL_YEARS, COAL_COUNTS, color=LIGHT, width=0.75)
    pmax = loc_probs.max()
    for i, p in enumerate(loc_probs):
        if p > 0.55 * pmax:
            ax.axvline(1880.5 + i, color=TEAL, linewidth=0.9,
                       alpha=float(0.8 * (p / pmax) ** 2), zorder=2)
    xs = np.append(COAL_YEARS, 1900) - 0.5
    ax.fill_between(xs, np.append(q05, q05[-1]), np.append(q95, q95[-1]),
                    step="post", color=TEAL, alpha=0.22, linewidth=0)
    ax.step(xs, np.append(rate_mean, rate_mean[-1]),
            where="post", color=TEAL, linewidth=1.1, zorder=3)
    ax.set_xticks([1880, 1890, 1899])
    ax.set_xlim(1879.2, 1900.2)
    ax.set_yticks([0, 5])
    save(fig, out, "changepoint_loc")
    return {"P(k)": {k: round(float(k_probs.get(k, 0)), 3) for k in ks},
            "E[rate] first,last": [round(float(rate_mean[0]), 2), round(float(rate_mean[-1]), 2)]}


def gamma_mix_quantile(comps, q):
    """Quantile of a mixture of Gammas [(w, shape, rate)] by bisection."""
    from scipy.stats import gamma as gamma_dist
    ws = np.array([c[0] for c in comps])
    ws = ws / ws.sum()
    lo, hi = 0.0, max(c[1] / c[2] for c in comps) * 6 + 1
    for _ in range(40):
        mid = (lo + hi) / 2
        cdf = sum(wi * gamma_dist.cdf(mid, a, scale=1 / r)
                  for wi, (_, a, r) in zip(ws, comps))
        lo, hi = (mid, hi) if cdf < q else (lo, mid)
    return (lo + hi) / 2


# ── row 4: Dawid–Skene crowd annotation ───────────────────────────────

DS_ANNOTS = [(0, 0, 1), (0, 1, 1), (0, 4, 0),
             (1, 0, 1), (1, 2, 1), (1, 4, 0),
             (2, 1, 1), (2, 3, 1), (2, 4, 0),
             (3, 0, 1), (3, 1, 1), (3, 2, 1),
             (4, 0, 0), (4, 2, 0), (4, 4, 1),
             (5, 1, 0), (5, 3, 0), (5, 4, 1),
             (6, 0, 0), (6, 1, 0), (6, 4, 1),
             (7, 3, 1), (7, 4, 0)]
N_ITEMS, N_ANNOT = 8, 5
ANNOT_NAMES = ["A", "B", "C", "D", "E"]


def render_dawid_skene(results, out):
    items = items_of(results, "dawid-skene")
    w = weights_of(items)
    label_probs = np.zeros(N_ITEMS)
    pair_names = None
    for wi, it in zip(w, items):
        m = re.match(r"\(Result \[([^\]]*)\] \[(.*)\]\)", it["display"])
        label_probs += wi * np.array(parse_bool_list(m.group(1)), dtype=float)
        if pair_names is None:
            pair_names = re.findall(r"\((\S+), (\S+)\)", m.group(2))

    # data panel: labels grid (✓ / ✗ / blank)
    fig, ax = panel(W_DATA, H, "labels by annotator", l=0.25, b=0.21, r=0.97)
    for (i, j, lab) in DS_ANNOTS:
        ax.text(j + 0.5, N_ITEMS - 1 - i + 0.38, "✓" if lab else "✗",
                ha="center", va="center", fontsize=5.5,
                color=INK if lab else GRAY, family="DejaVu Sans")
    for i in range(N_ITEMS):
        for j in range(N_ANNOT):
            ax.add_patch(plt.Rectangle((j + 0.06, N_ITEMS - 1 - i + 0.02), 0.88, 0.82,
                                       facecolor="none", edgecolor=BAND, linewidth=0.6))
    ax.set_xlim(0, N_ANNOT)
    ax.set_ylim(-0.2, N_ITEMS)
    ax.set_xticks([j + 0.5 for j in range(N_ANNOT)])
    ax.set_xticklabels(ANNOT_NAMES)
    ax.set_yticks([N_ITEMS - 1 - i + 0.4 for i in range(N_ITEMS)])
    ax.set_yticklabels([str(i + 1) for i in range(N_ITEMS)])
    ax.tick_params(length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, out, "ds_data")

    # per-item true-label posterior (rows aligned with the data grid)
    fig, ax = panel(W_A, H, "P(true label)", l=0.12)
    ax.barh([N_ITEMS - 1 - i for i in range(N_ITEMS)], label_probs,
            color=TEAL, height=0.6)
    ax.set_yticks([])
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, 1])
    save(fig, out, "ds_labels")

    # annotator reliability: exact Beta-mixture densities per annotator
    grid = np.linspace(1e-3, 1 - 1e-3, 300)
    fig = plt.figure(figsize=(W_B, H))
    axes = [fig.add_axes([0.10, 0.20 + (N_ANNOT - 1 - j) * 0.132, 0.84, 0.120])
            for j in range(N_ANNOT)]
    axes[0].set_title("sensitivity (teal) vs. false-pos. rate (gray)",
                      fontsize=6, color=INK, pad=2.5)
    for j, axj in enumerate(axes):
        comps_tp, comps_fp = [], []
        for wi, it in zip(w, items):
            bn = betas_by_name(it)
            tp, fp = pair_names[j]
            comps_tp.append((wi, *bn[tp]))
            comps_fp.append((wi, *bn[fp]))
        axj.fill_between(grid, beta_mixture_pdf(grid, comps_tp),
                         color=TEAL, alpha=0.65, linewidth=0)
        axj.fill_between(grid, beta_mixture_pdf(grid, comps_fp),
                         color=GRAY, alpha=0.55, linewidth=0)
        axj.set_yticks([])
        axj.set_xlim(0, 1)
        axj.set_ylabel(ANNOT_NAMES[j], rotation=0, ha="right", va="center",
                       fontsize=5.5)
        for s in axj.spines.values():
            s.set_visible(False)
        if j < N_ANNOT - 1:
            axj.set_xticks([])
        else:
            axj.set_xticks([0, 0.5, 1])
            axj.tick_params(length=1.8, pad=1.5)
    save(fig, out, "ds_annotators")
    return {"P(label)": label_probs.round(2).tolist()}


# ── row 5: unknown-K 2D mixture ───────────────────────────────────────

MIX_PTS = np.array([(-2.3, -1.0), (-1.7, -1.8), (-2.2, -1.5),
                    (1.2, 2.2), (1.8, 1.7),
                    (2.75, 0.75), (3.3, 0.25)])
N_PTS = len(MIX_PTS)


def render_mixture(results, out):
    items = items_of(results, "mixture")
    w = weights_of(items)
    k_probs = defaultdict(float)
    part_probs = defaultdict(float)
    co = np.zeros((N_PTS, N_PTS))
    best = {}
    for wi, it in zip(w, items):
        m = re.match(r"\(Result (\d+) \[([^\]]*)\] \[(.*)\]\)", it["display"])
        k = int(m.group(1))
        zs = tuple(int(x) for x in m.group(2).split(","))
        canon = canonical_partition(zs)
        k_probs[k] += wi
        part_probs[canon] += wi
        za = np.array(zs)
        co += wi * (za[:, None] == za[None, :])
        if canon not in best or wi > best[canon][0]:
            comps = re.findall(r"\(Comp (\S+) (\S+) [\d.]+\)", m.group(3))
            best[canon] = (wi, zs, comps, it)

    fig, ax = panel(W_DATA, H, "2D points", l=0.14)
    ax.scatter(MIX_PTS[:, 0], MIX_PTS[:, 1], s=7, color=INK)
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.set_aspect("equal")
    save(fig, out, "mixture_data")

    fig, ax = panel(W_A, H, "P($K$ clusters)", l=0.21)
    ks = [1, 2, 3, 4]
    ax.bar(ks, [k_probs.get(k, 0) for k in ks], color=TEAL, width=0.62)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 1])
    save(fig, out, "mixture_k")

    # P(same cluster) matrix + the two leading clusterings with mean posteriors
    top2 = sorted(part_probs.items(), key=lambda x: -x[1])[:2]
    fig = plt.figure(figsize=(W_B, H))
    axm = fig.add_axes([0.10, 0.21, 0.36, 0.64])
    axm.imshow(co, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "teal", ["white", TEAL]), vmin=0, vmax=1)
    axm.set_title("P(same cluster)", fontsize=6, color=INK, pad=2.5)
    axm.set_xticks(range(N_PTS))
    axm.set_xticklabels([str(i + 1) for i in range(N_PTS)], fontsize=4.5)
    axm.set_yticks(range(N_PTS))
    axm.set_yticklabels([str(i + 1) for i in range(N_PTS)], fontsize=4.5)
    axm.tick_params(length=0, pad=1)
    for s in axm.spines.values():
        s.set_visible(False)
    for col, (canon, p) in enumerate(top2):
        axc = fig.add_axes([0.56 + col * 0.23, 0.21, 0.20, 0.64])
        wi, zs, comps, it = best[canon]
        comp_order = sorted(set(zs))
        for ci, comp in enumerate(comp_order):
            idx = [i for i, z in enumerate(zs) if z == comp]
            axc.scatter(MIX_PTS[idx, 0], MIX_PTS[idx, 1], s=6, color=CAT[ci % 4])
        axc.set_title(f"{part_probs[canon] * 100:.0f} %", fontsize=5.5,
                      color=INK, pad=2)
        axc.set_xticks([])
        axc.set_yticks([])
        axc.set_xlim(-3.2, 4.2)
        axc.set_ylim(-2.7, 3.5)
        axc.set_aspect("equal")
        for s in axc.spines.values():
            s.set_color(LIGHT)
    save(fig, out, "mixture_clusters")
    return {"P(K)": {k: round(float(k_probs.get(k, 0)), 3) for k in ks},
            "top2": [round(float(p), 3) for _, p in top2]}


def canonical_partition(zs):
    relabel, out = {}, []
    for z in zs:
        if z not in relabel:
            relabel[z] = len(relabel)
        out.append(relabel[z])
    return tuple(out)


# ── metrics: LoC and timing ───────────────────────────────────────────

def loc_of(path):
    return sum(1 for line in path.read_text().splitlines()
               if line.strip() and not line.strip().startswith(";"))


STAGE_RE = re.compile(
    r"parse=([\d.]+)\s+symbolic-exec=([\d.]+)\s+wmc=([\d.]+)\s+posterior=([\d.]+)")


def time_program(path, reps):
    """Median time to construct the posterior SPN (parse + symbolic-exec + WMC +
    posterior), via PLUCK_STATS."""
    ts = []
    env = dict(os.environ, PLUCK_STATS="1")
    for _ in range(reps):
        out = subprocess.run([str(PLUCK), str(path)], cwd=REPO, env=env,
                             capture_output=True, text=True, check=True)
        m = STAGE_RE.search(out.stderr)
        ts.append(sum(float(g) for g in m.groups()))
    return float(np.median(ts))


def time_psi(name, timeout):
    """Wall-clock of the PSI port of a row's model (fig_<name>.psi), with the
    Table 3 timeout convention. Returns a formatted chip string."""
    prog = FIG1 / f"fig_{name}.psi"
    t0 = time.perf_counter()
    try:
        out = subprocess.run(["psi", str(prog)], capture_output=True,
                             text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "t/o"
    t = time.perf_counter() - t0
    if out.returncode != 0 or re.search(r"Pr\[error\] *= *1", out.stdout):
        return "err"
    return fmt_time(t)


def fmt_time(s):
    if s < 0.095:
        return f"{s * 1000:.0f}\\,ms"
    if s < 9.5:
        return f"{s:.1f}\\,s"
    return f"{s:.0f}\\,s"


MACRO_KEYS = {"polyreg": "Polyreg", "sir": "Sir", "changepoint": "Changepoint",
              "dawid_skene": "DawidSkene", "mixture": "Mixture"}
RENDER = {"polyreg": render_polyreg, "sir": render_sir,
          "changepoint": render_changepoint,
          "dawid_skene": render_dawid_skene, "mixture": render_mixture}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--time", type=int, default=0, metavar="N",
                    help="also measure median-of-N posterior-SPN time per program")
    ap.add_argument("--time-psi", action="store_true",
                    help="also time the PSI ports (fig_*.psi)")
    ap.add_argument("--psi-timeout", type=float, default=600, metavar="SECONDS",
                    help="timeout per PSI port (default: 600, as in the paper)")
    args = ap.parse_args()
    json_dir = args.out / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    for name in ROWS:
        info = RENDER[name](run_pluck(PROGRAMS[name], json_dir), args.out)
        print(f"{name}: {info}")

    lines = [f"\\newcommand{{\\FigOneLoC{MACRO_KEYS[name]}}}{{{loc_of(LISTINGS[name])}}}"
             for name in ROWS]
    if args.time:
        for name in ROWS:
            t = time_program(LISTINGS[name], args.time)
            print(f"time {name}: {t:.3f}s")
            lines.append(f"\\newcommand{{\\FigOneTime{MACRO_KEYS[name]}}}{{{fmt_time(t)}}}")
    if args.time_psi:
        for name in ROWS:
            t = time_psi(name, args.psi_timeout)
            print(f"psi {name}: {t}")
            lines.append(f"\\newcommand{{\\FigOnePsi{MACRO_KEYS[name]}}}{{{t}}}")
    (args.out / "fig1_metrics.tex").write_text(
        "% Generated by experiments/figure1/make_figure1_panels.py\n"
        + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
