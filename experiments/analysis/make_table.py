#!/usr/bin/env python3
"""make_table.py — LaTeX inference-timing tables across all PPLs.

Emits two standalone tables, each into its own .tex file under the output
directory (default <run_dir>/analysis/). In this script's comments "Table 1" and
"Table 2" refer to these files, which are Tables 2 and 3 of the paper:

  * inference_standard.tex    — paper Table 2 (tab:inference-standard)
  * inference_applied.tex     — paper Table 3 (tab:inference-applied)

The tables are:

  1. STANDARD models (tab:inference-standard) — the simple textbook programs,
     grouped into four labelled sections: Fully Discrete, Fully Continuous,
     Mixed Discrete/Continuous, and Not Expressible in Pluck. Times are shown
     in *seconds*. A trailing column reports HyBit's relative approximation error
     vs the exact answer on the approximate (non-discrete) models.

  2. APPLIED / SCALING models (tab:inference-applied) — the larger models from
     the literature. Each model is a Title Case header row with indented
     sub-rows for every scaled or alternative variant, labelled by a meaningful
     parameter name (teams=6, timesteps=8, ...). Times are shown in *seconds*.

Both tables use these cell outcomes (Table 1 bolds the fastest exact only; Table
2 additionally underlines the fastest approximate):
    <number>       the median inference time (bold = fastest exact system in that
                   row; underline = fastest approximate system, Table 2 only)
    \\textsc{t/o}   TIMEOUT — ran but exceeded the per-run budget
    $\\textsc{r/e}$    ran, but every sampler budget's relative error (vs the exact
                   answer) was above ERROR_THRESHOLD (approximate backends only)
    \\_\\_           not expressible in that PPL (no program, OR present but
                   errored/spurious) — rendered "__"
    \\textsc{n/r}   not run: the program exists but this run has no result for it
    (blank)        HyBit's error column on Table 1's discrete models, where HyBit
                   is exact

For the approximate backends (hybit/turing/slicstan/siren) a sampler-budget
sweep produces several budgets per system. Rather than tabulating one fixed
budget, each such cell shows the budget with the SHORTEST run time whose relative
error vs the exact answer (reference/exact_answers.csv) is below ERROR_THRESHOLD
(a script-level constant); if none is, the cell shows $\\textsc{r/e}$. Cells with
no computable error fall back to the default-loaded budget. See ERROR_THRESHOLD /
build_error_selection().

On an incomplete run, programs without a result show \\textsc{n/r}, and the
reconciliation warning (see reconcile()) lists them.

The generated .tex requires \\usepackage{booktabs}, \\usepackage{arydshln}
(dashed section rules in Table 1), \\usepackage[table]{xcolor} (the shaded
approximate-inference cells in Table 1), and \\usepackage{array} (Table 2's
right-aligned fixed-width columns).

Times are inference-only seconds by default (pluck: inference_seconds; hybit:
t.time; slicstan: HMC sampling step). PSI prints no inference-only figure, so
its wall-clock is used (footnoted).

Usage:
    uv run python make_table.py [--run-dir DIR] [--out-dir DIR] [--metric infer|wall]
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import bench_common as bc
import compute_error as ce

# Hack
TRUE_BACKSLASH = r"@backslash"

# --------------------------------------------------------------------------
# Display order of the PPL columns (matches the paper layout: pluck first,
# then PSI, HyBit, and finally the two discrete-only backends).
# --------------------------------------------------------------------------
# mappl is DISABLED (dropped from this list) — see bench_common.SYSTEMS for why.
DISPLAY_SYSTEMS = ["pluck", "psi", "hybit", "slicstan", "siren", "turing"]

# Table 1 shows only the exact-capable / discrete backends (MaPPL and SlicStan
# are dropped there); Table 2 uses the full DISPLAY_SYSTEMS list.
TABLE1_SYSTEMS = ["pluck", "psi", "hybit"]

# Systems that do exact inference (the rest are approximate). These must be a
# prefix of DISPLAY_SYSTEMS so the exact/approximate split is a single column
# boundary; Table 2 draws a rule and an "Exact | Approximate" group header there.
EXACT_SYSTEMS = {"pluck", "psi"}

SYS_LABEL = {
    "pluck": "Hybrid Pluck",
    "hybit": "HyBit",
    "psi": "PSI",
    "turing": "Turing",
    "mappl": "MaPPL",
    "slicstan": "SlicStan",
    "siren": "Siren",
}

# --------------------------------------------------------------------------
# TABLE 1 — standard textbook models. Each section is
#   (section_title, show_mappl_slicstan, [(display_name, program), ...]).
# Only the fully-discrete section shows MaPPL/SlicStan results; the others
# force "__" in those two columns (they don't solve those models exactly).
# display_name is emitted through latex_escape(), so plain snake_case is fine.
# --------------------------------------------------------------------------
TABLE1_SECTIONS = [
    (
        "Fully Discrete",
        True,
        [
            ("Burglary Alarm", "burglary_alarm"),
            ("Diamond Network", "diamond_network"),
            ("Typos", "typos"),
        ],
    ),
    (
        "Fully Continuous",
        False,
        [
            ("addFun_sum", "addFun_sum"),
            ("conjugate_gaussians", "conjugate_gaussians"),
            ("linear_regression", "linear_regression"),
        ],
    ),
    (
        "Mixed Discrete/Continuous",
        False,
        [
            ("clickGraph", "clickGraph"),
            ("clinicalTrial1", "clinicalTrial1"),
            ("clinicalTrial2", "clinicalTrial2"),
            ("coinBias", "coinBias"),
            ("GPA", "GPA"),
            ("normal_mixture", "normal_mixture"),
        ],
    ),
    (
        "Not Expressible in Hybrid Pluck",
        False,
        [
            ("addFun_max", "addFun_max"),
            ("spacex", "spacex"),
            ("trueskill", "trueskill"),
            ("tug_of_war", "tug_of_war"),
            ("weekend", "weekend"),
        ],
    ),
]

# --------------------------------------------------------------------------
# TABLE 2 — applied / scaling models, in listed order. Each group is one of:
#   {"kind": "single",   "title", "program"}
#   {"kind": "variants", "title", "rows": [(sublabel, program), ...]}
#   {"kind": "family",   "title", "family", "label_fmt", "sizes": [...],
#                        "prefix_rows": [(sublabel, program), ...]}   # optional
# label_fmt is a str.format template taking {n} (the integer size). Sub-labels
# and titles are emitted through latex_escape() (which leaves $ {} = / intact),
# so simple math like "$k{=}1$" is fine but avoid underscores in labels.
# An optional "cite" holds BibTeX key(s) rendered inline after the model name
# (\citep in Table 2);
# models without one get no citation.
# --------------------------------------------------------------------------
TABLE2_GROUPS = [
    # Short models (one or two rows) up top.
    {
        "kind": "single",
        "title": "Spike & Slab",
        "program": "spike_and_slab",
        "cite": "georgeVariableSelectionGibbs1993,haldStatisticalTheoryEngineering1952",
        "desc": "Sparse linear regression whose coefficients mix a Dirac delta `spike' at zero with "
        "a `slab' normal prior; infers which predictors are active and their weights.",
    },
    {
        "kind": "single",
        "title": "Polynomial Regression",
        "program": "polyreg",
        "desc": "Bayesian polynomial fit to noisy observations with a discrete outlier indicator; infers the polynomial "
        "degree, coefficients, and probability of each point being an outlier.",
    },
    {
        "kind": "single",
        "title": "Run-Walk HRM",
        "program": "run_walk_hrm",
        "desc": "Hidden-rate model switching between run and walk regimes; infers the "
        "latent regime sequence and its rates.",
    },
    {
        "kind": "family",
        "title": "SIR Model",
        "family": "sir_model",
        "cite": "kermackContributionMathematicalTheory1927,kingStatisticalInferencePartially2016",
        "label_fmt": "timesteps$" + TRUE_BACKSLASH + "times$patients={n}",
        "sizes": [12, 72],
        "desc": "Epidemic model whose latent infection status evolves as a hidden Markov "
        "chain over time, emitting a noisy symptom each step; infers the posterior "
        "infection rate from an observed symptom history.",
    },
    {
        "kind": "variants",
        "title": "Felsenstein DNA Evolution",
        "cite": "felsensteinEvolutionaryTreesDNA1981",
        "rows": [
            ("Rate parameter only", "f81_rate"),
            ("Full joint model", "f81_joint"),
        ],
        "desc": "Phylogenetic likelihood of aligned DNA under the F81 model; infers "
        "substitution rates and, jointly, ancestral sequences.",
    },
    {
        "kind": "family",
        "title": "PCFG",
        "family": "pcfg",
        "cite": "johnsonBayesianInferencePCFGs2007",
        "prefix_rows": [("Ambiguous grammar", "pcfg_ambiguous")],
        "label_fmt": "Strict grammar",
        "sizes": [20],
        "desc": "Probabilistic context-free grammar over strings; infers the "
        "distribution of parse trees and production choices.",
    },
    # Scaling families.
    {
        "kind": "family",
        "title": "Coal-Mining Disasters",
        "family": "changepoint_coal",
        "cite": "jarrettNoteIntervalsCoalmining1979,greenReversibleJumpMarkov1995",
        "prefix_rows": [("$k{=}1$ changepoint, years=20", "changepoint_coal_k1")],
        "label_fmt": "Variable $k$, years={n}",
        "sizes": [20, 50],
        "desc": "Change-point model of yearly disaster counts; infers the "
        "change-point year(s) and the rates on either side.",
    },
    {
        "kind": "family",
        "title": "Dawid--Skene Annotation",
        "family": "dawid_skene",
        "cite": "dawidMaximumLikelihoodEstimation1979",
        "label_fmt": "items={n}",
        "sizes": [8, 16],
        "desc": "Aggregates noisy crowd labels; infers each item's true label and "
        "every annotator's confusion rates.",
    },
    {
        "kind": "family",
        "title": "First-Order HMM",
        "family": "hmm_first_order",
        "label_fmt": "timesteps={n}",
        "sizes": [2, 8, 14],
        "desc": "First-order HMM with continuous parameters and emissions; infers the "
        "latent discrete state sequence behind the observations.",
    },
    {
        "kind": "family",
        "title": "Second-Order HMM",
        "family": "hmm_second_order",
        "label_fmt": "timesteps={n}",
        "sizes": [4, 8, 14],
        "desc": "HMM whose discrete state depends on the previous two, with continuous "
        "emissions; infers the latent discrete state sequence.",
    },
    {
        "kind": "family",
        "title": "Multi-Chain HMM",
        "family": "hmm_factorial",
        "label_fmt": "timesteps={n}",
        "sizes": [2, 4, 6],
        "desc": "Several parallel latent chains generating one observation stream; "
        "infers the joint hidden state across chains.",
    },
    # king_ei's dial is an inverse  fraction of full dataset, so listing d largest-first
    # (1/60, 1/40, 1/20) runs fastest -> slowest like the other families.
    {
        "kind": "family",
        "title": "Ecological Inference",
        "family": "king_ei",
        "cite": "kingBinomialBetaHierarchicalModels1999",
        "label_fmt": " fraction of dataset $=1/{n}$",
        "sizes": [60, 20],
        "desc": "King's model recovering precinct-level voting rates of various populations from aggregate margins; "
        "infers the latent per-precinct voting propensities for each group from a dataset of vote counts for Waterbury, CT including N=13399 voters.",
    },
    {
        "kind": "family",
        "title": "Robust Linear Regression",
        "family": "outliers",
        "label_fmt": "points={n}",
        "sizes": [8, 18],
        "desc": "Linear regression with a heavy-tailed outlier component; infers "
        "coefficients and outlier identities.",
    },
    {
        "kind": "family",
        "title": "Gaussian Mixture",
        "family": "soft_kmeans",
        "label_fmt": "points={n}",
        "sizes": [4, 12],
        "desc": "Soft clustering of points into Gaussian components; infers cluster "
        "assignments and component means.",
    },
    {
        "kind": "family",
        "title": "Secret Ballot Vote",
        "family": "secret_ballot",
        "cite": "hansenEstimatingBayesianDecision2013",
        "label_fmt": "voters={n}",
        "sizes": [4, 128],
        "desc": "Model of K secret ballots which each represent a subset of N voters. For each ballot, only the majority yes/no outcome is observed; infers each voter's latent probability of voting `yes'.",
    },
    {
        "kind": "family",
        "title": "Sports Analysis",
        "family": "soccer",
        "cite": "maherModellingAssociationFootball1982",
        "label_fmt": "teams={n}",
        "sizes": [3, 7],
        "desc": "Soccer scores driven by latent team strengths and identities (`attack', `defend', `balanced'); infers each team's "
        "latent identity.",
    },
]

# Every size of every model a system can express has its own program, so a cell is
# only ever "__" (no program for that system), a measured outcome, or "n/r" (the
# program exists but this run has no result for it, e.g. a partial run).

# (system, family) pairs forced to "not expressible" even when the run produced a
# result for them. Currently:
# SlicStan on the second-order HMM, whose emitted results do not reflect a correct
# encoding of the model, and on the factorial HMM, whose translation SlicStan cannot
# parse ("Could not parse"; the run records ERROR(slic)).
FAMILY_FORCE_ABSENT = {
    ("slicstan", "hmm_second_order"),
    ("slicstan", "hmm_factorial"),
}

# Programs where PSI exits ok (so the cell shows its wall time) but the printed
# posterior still contains UNEVALUATED integrals (∫dξ terms PSI could not reduce)
# rather than a closed form — verified by grepping every psi log of the paper run
# (identical across all 15 repeats). All four are the Gaussian-comparison models
# in Table 1's "Not Expressible in Hybrid Pluck" section; weekend is NOT here (its
# output has erf terms but no leftover integrals). The cell keeps the time,
# wrapped in parentheses and explained in the table's footnote legend.
PSI_INTEGRAL_RESULT = {"addFun_max", "spacex", "trueskill", "tug_of_war"}

# --------------------------------------------------------------------------
# Sampler-budget selection policy for the APPROXIMATE backends (hybit, turing,
# slicstan, siren). A sweep produces several budgets per system; rather than
# tabulating the largest budget, we show — per (system, program) cell — the
# budget with the SHORTEST run time whose relative error (vs the fixed exact answer
# in reference/exact_answers.csv, see compute_error.py) is below ERROR_THRESHOLD. If
# every budget that ran is above it, the cell shows MARK_OVER instead of a time.
# Cells with no computable error (no exact answer) fall back to the default-loaded
# budget and are never gated. Set ERROR_THRESHOLD = None to disable the whole policy
# (revert to the old largest-budget-per-system behaviour).
ERROR_THRESHOLD = (
    bc.ERROR_THRESHOLD
)  # relative error (0.05 = 5%); set None here to disable

# Cell glyphs for the non-numeric outcomes (LaTeX).
MARK_TIMEOUT = r"\textsc{t/o}"
MARK_OOM = r"\textsc{o/m}"  # ran out of memory before the deadline
MARK_ERROR = r"$\times$"  # present but errored
MARK_OVER = r"$\textsc{r/e}$"  # ran, but relative error above ERROR_THRESHOLD
# not expressible: a double-underscore raised to the vertical middle of the cell
# (bare "\_\_" sits on the baseline and looks stuck to the bottom). Tune the
# raise if your font/size shifts it off-centre.
MARK_ABSENT = r"\raisebox{0.5ex}{\_\_}"
# the program exists but this run has no result for it (a partial run).
MARK_NOT_RUN = r"\textsc{n/r}"
# a PSI time whose "result" is an unevaluated-integral expression is wrapped in
# parentheses (see PSI_INTEGRAL_RESULT); the footnote legend (not the caption)
# explains the mark as $(\cdot)$.
MARK_INTEGRAL = "(%s)"

# Background shade applied to the approximate-inference cells in Table 1
# (non-discrete HyBit programs). Any xcolor spec works; edit to taste.
APPROX_SHADE = "gray!25"

# Pluck-speedup colour bands (see _fmt_speedup_colored). $\infty$ means pluck wins
# by an unbounded margin (no competing system on that side); the numeric bands run
# green -> orange -> red as pluck's advantage shrinks. Any xcolor spec works.
COL_INF = "green!55!white"  # light green: $\infty$ (no competitor)
COL_GT1 = "green!55!black"  # dark green : pluck faster (\geqx)
COL_MID = "orange"  # orange     : pluck slower but within 100x (0.01-1x)
COL_LOW = "red"  # red        : pluck >100x slower (<0.01x)

# Table 2's Model/Variant label column has a fixed narrow width so long model
# names and variant labels wrap; the timing columns are natural right-aligned
# columns, giving the table its usual content-sized width.
MODEL_COL_WIDTH = 3.2  # cm, Model / Variant label column (wraps)

# Table 1's leading section column: each variable-type class (Fully Discrete, ...)
# is a \multirow spanning that section's rows, wrapping within this fixed width.
SECTION_COL_WIDTH = 1.9  # cm

def latex_escape(s: str) -> str:
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace(TRUE_BACKSLASH, "\\")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def fmt_time(v: float) -> str:
    """Time value as a 3-significant-figure scientific-notation LaTeX math expr,
    e.g. 2.59 -> "$2.59$", 0.0454 -> "$4.54{\\times}10^{-2}$" (the \\times10^{0}
    factor is suppressed for values already in [1, 10)). The \\times is wrapped
    in braces so it renders as an ordinary symbol, dropping the wide binary-
    operator spacing LaTeX otherwise puts on either side."""
    mant, exp = f"{v:.2e}".split("e")  # 2 decimals -> 3 sig figs; exp like "+00"
    exp = int(exp)
    if exp == 0:
        return r"$%s$" % mant
    return r"$%s{\times}10^{%d}$" % (mant, exp)


def fmt_speedup(x: float) -> str:
    """Format a speedup ratio with an 'x' unit (thousands separated for big ones)."""
    if x >= 10:
        s = f"{x:,.0f}"
    elif x >= 1:
        s = f"{x:.1f}"
    else:
        s = f"{x:.2g}"
    return s + r"x"


def fmt_relerr(e: float) -> str:
    """A relative error as a percentage with two decimals, e.g. 0.032 -> "3.20\\%".
    Used to fill Table 1's dedicated relative-error column."""
    return r"%.2f\%%" % (e * 100.0)


def fmt_abserr(e: float) -> str:
    """An ABSOLUTE error as a plain number (no percent sign), e.g. 0.0039.
    Used in Table 1's error column for models whose exact value is 0, where relative
    error is undefined (currently only addFun_sum, E[x+y] = 0)."""
    return r"%.2g" % e


def _speedup_color(ratio: float | None) -> str:
    """Colour name for a pluck speedup ratio (None = $\\infty$). See COL_* bands."""
    if ratio is None:
        return COL_INF
    if ratio > 1:
        return COL_GT1
    return COL_MID  # orange for anything <= 1x (no separate red band)


def _fmt_speedup_colored(ratio: float | None) -> str:
    """A pluck speedup rendered in its band colour: None -> light-green $\\infty$
    (no competitor on that side); otherwise the ratio, coloured green/orange/red
    as pluck's advantage shrinks (see COL_* bands)."""
    txt = r"$\infty$" if ratio is None else fmt_speedup(ratio)
    return r"\textcolor{%s}{%s}" % (_speedup_color(ratio), txt)


def _emphasize(cells, systems, underline_approx: bool = True) -> list[str]:
    """Render each (text, value) cell, bolding the fastest *exact* time and
    (when `underline_approx`) underlining the fastest *approximate* time in the
    row. `systems` is the parallel list of system keys used to split exact from
    approximate. Table 1 passes underline_approx=False (no approximate underline)."""
    exact_vals = [
        v for s, (_, v) in zip(systems, cells) if v is not None and s in EXACT_SYSTEMS
    ]
    approx_vals = [
        v
        for s, (_, v) in zip(systems, cells)
        if v is not None and s not in EXACT_SYSTEMS
    ]
    exact_min = min(exact_vals) if exact_vals else None
    approx_min = min(approx_vals) if approx_vals else None
    out = []
    for s, (t, v) in zip(systems, cells):
        # {\boldmath\textbf{}} so both the math value and its unit bold (plain
        # \textbf does not bold math-mode content).
        if v is not None and s in EXACT_SYSTEMS and v == exact_min:
            out.append(r"{\boldmath\textbf{%s}}" % t)
        elif (
            underline_approx
            and v is not None
            and s not in EXACT_SYSTEMS
            and v == approx_min
        ):
            out.append(r"\underline{%s}" % t)
        else:
            out.append(t)
    return out


def pluck_speedup(
    run: bc.RunData, program: str, metric: str, systems, error_sel: dict
) -> str:
    """Pluck's speedup over the next-fastest *other* system among `systems`.
    \\_\\_ if pluck has no result for the row (n/r if it was not run); blank if no
    other system does. Uses effective_time, so an over-threshold system is excluded
    from the denominator (it has no displayed time to be faster than)."""
    pt = effective_time(run, "pluck", program, metric, error_sel)
    if pt is None or pt <= 0:
        return _no_pluck_mark(run, program)
    others = [
        v
        for s in systems
        if s != "pluck"
        for v in (effective_time(run, s, program, metric, error_sel),)
        if v is not None and v > 0
    ]
    if not others:
        return r""
    return _fmt_speedup_colored(min(others) / pt)


def pluck_speedup_split(
    run: bc.RunData, program: str, metric: str, systems, error_sel: dict
) -> str:
    """Two '&'-separated cells (for the @{}r@{}r column group): pluck's speedup
    over the fastest *exact* system (PSI) and over the fastest *approximate*
    system. Each side is $\\infty$ if no system on that side has a displayed
    time, and is coloured by its band (see _fmt_speedup_colored). Collapses to a
    spanning \\_\\_ if pluck has no result for the row (n/r if it was not run).
    Uses effective_time, so an over-threshold approximate system is excluded from
    its denominator."""
    pt = effective_time(run, "pluck", program, metric, error_sel)
    if pt is None or pt <= 0:
        return r"\multicolumn{2}{c}{%s}" % _no_pluck_mark(run, program)

    def side(group) -> str:
        ts = [
            v
            for s in group
            for v in (effective_time(run, s, program, metric, error_sel),)
            if v is not None and v > 0
        ]
        return _fmt_speedup_colored(min(ts) / pt if ts else None)

    exact = [s for s in systems if s != "pluck" and s in EXACT_SYSTEMS]
    approx = [s for s in systems if s not in EXACT_SYSTEMS]
    return side(exact) + r" & " + side(approx)


def _no_pluck_mark(run: bc.RunData, program: str) -> str:
    """Speedup cell when pluck has no time: n/r if pluck was not run, else \\_\\_."""
    if status_kind(run.cell_status("pluck", program)) == "notrun":
        return MARK_NOT_RUN
    return MARK_ABSENT


def status_kind(st: str) -> str:
    """Collapse a cell_status string into one of ok/timeout/oom/error/absent/notrun."""
    if st == "ABSENT":
        return "absent"
    if st == "PENDING":
        return "notrun"
    if st == "TIMEOUT":
        return "timeout"
    if st == "OOM":
        # resource exhaustion within the time budget (run_bench: rc=137 well
        # before the deadline) -- did-not-complete like t/o, but rendered with
        # its own mark; distinct from a genuine crash (ERROR).
        return "oom"
    if st == "ok":
        return "ok"
    return "error"  # ERROR(...)/BAD/NODATA/NODRIVER/"?" -> present but bad


# --------------------------------------------------------------------------
# Error-thresholded sampler-budget selection (see ERROR_THRESHOLD above).
# --------------------------------------------------------------------------
def build_error_selection(
    run_dir: str, metric: str, select: dict[str, str], criterion: str | None = None
) -> dict[tuple[str, str], tuple]:
    """Per (system, program) budget choice under ERROR_THRESHOLD, for the swept
    approximate backends. Returns {(system, program): sel} where sel is:
        ("ok",   time, tag, err)   fastest budget with error < threshold
        ("over", best_err, best_tag)   ran ok at >=1 budget, all above threshold
    The error statistic is chosen by `criterion` (default bc.DEFAULT_CRITERION =
    'mean-result', the legacy error-of-the-mean policy). Pairs with no computable
    error are absent (the caller falls back to the default-loaded variant and never
    gates them). A system named in `select` is skipped — an explicit --select is an
    unconditional manual override. Returns {} when disabled (ERROR_THRESHOLD is None)."""
    if ERROR_THRESHOLD is None:
        return {}
    criterion = criterion or bc.DEFAULT_CRITERION
    variants = bc.discover_variants(run_dir)

    # (system, program) -> [(time, err_scalar, tag), ...] across every budget
    cand: dict[tuple[str, str], list[tuple[float, float, str]]] = {}
    for system in ce.APPROX:
        if system in select:
            continue
        for tag in variants.get(system, {}):
            r = bc.load_run(run_dir, select={system: tag})
            for (s, prog), res in r.results.items():
                if s != system or res.status != "ok" or res.result is None:
                    continue
                gt, _src = ce.ground_truth(prog)
                if gt is None:
                    continue
                # per-repeat estimates (fallback: the aggregate) under the criterion
                avs = [
                    ce._adjust_approx(system, prog, v)
                    for v in (res.results or [res.result])
                ]
                err = bc.error_scalar(avs, gt, criterion)
                t = r.time(system, prog, metric)
                if t is None:
                    continue
                cand.setdefault((system, prog), []).append((t, err, tag))

    sel: dict[tuple[str, str], tuple] = {}
    for key, lst in cand.items():
        chosen = bc.select_budget(lst, criterion, ERROR_THRESHOLD)
        if chosen is not None:
            sel[key] = chosen
    return sel


def effective_time(
    run: bc.RunData, system: str, program: str, metric: str, error_sel: dict
) -> float | None:
    """The time (seconds) the table will actually display for this cell, or None
    if the cell is not a number (absent/timeout/error/over-threshold/no result).
    This is the single source of truth shared by the cells and pluck_speedup, so
    the speedup ratio compares the same times the table shows."""
    sel = error_sel.get((system, program))
    if sel is not None:
        return sel[1] if sel[0] == "ok" else None
    if status_kind(run.cell_status(system, program)) != "ok":
        return None
    return run.time(system, program, metric)


# --------------------------------------------------------------------------
# Cell resolution
# --------------------------------------------------------------------------
def table1_cell(
    run: bc.RunData, system: str, program: str, metric: str, error_sel: dict
):
    """Table-1 cell, value in seconds."""
    sel = error_sel.get((system, program))
    if sel is not None:  # error-thresholded budget choice wins
        if sel[0] == "over":
            return MARK_OVER, None
        return fmt_time(sel[1]), sel[1]  # value in seconds for row-min bolding
    kind = status_kind(run.cell_status(system, program))
    if kind == "ok":
        v = run.time(system, program, metric)
        if v is None:
            return MARK_ERROR, None
        txt = fmt_time(v)  # unit hoisted into the header/legend — see _legend_note
        if system == "psi" and program in PSI_INTEGRAL_RESULT:
            txt = MARK_INTEGRAL % txt
        return txt, v  # value in seconds for row-min bolding
    if kind == "timeout":
        return MARK_TIMEOUT, None
    if kind == "oom":
        return MARK_OOM, None
    if kind == "error":
        return MARK_ERROR, None
    if kind == "notrun":
        return MARK_NOT_RUN, None
    return MARK_ABSENT, None


def table2_cell(
    run: bc.RunData,
    system: str,
    program: str,
    metric: str,
    family: str | None,
    error_sel: dict,
):
    """Table-2 cell (value in seconds). No program -> "__"; program but no result
    in this run -> n/r. A present-but-errored cell is also rendered "__" (not
    expressible)."""
    # (system, family) pairs we force to "not expressible" regardless of any result
    # the run may hold for them (e.g. SlicStan's spurious second-order HMM output).
    if family is not None and (system, family) in FAMILY_FORCE_ABSENT:
        return MARK_ABSENT, None
    sel = error_sel.get((system, program))
    if sel is not None:  # error-thresholded budget choice wins
        if sel[0] == "over":
            return MARK_OVER, None
        return fmt_time(sel[1]), sel[
            1
        ]  # unit hoisted out of the cell (see _legend_note)
    kind = status_kind(run.cell_status(system, program))
    if kind == "ok":
        v = run.time(system, program, metric)
        if v is None:
            return MARK_ABSENT, None
        return fmt_time(v), v  # unit hoisted out of the cell (see _legend_note)
    if kind == "timeout":
        return MARK_TIMEOUT, None
    if kind == "oom":
        return MARK_OOM, None
    if kind == "error":
        return MARK_ABSENT, None  # present-but-errored -> "not expressible"
    if kind == "notrun":
        return MARK_NOT_RUN, None
    return MARK_ABSENT, None


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def _thr_pct() -> str:
    """ERROR_THRESHOLD as a LaTeX percentage string, e.g. 0.05 -> '5\\%'."""
    return r"%g\%%" % (ERROR_THRESHOLD * 100)


def _legend_note(
    include_shade: bool,
    include_error: bool,
    unit: str | None = None,
    include_over: bool = True,
    include_abs_dagger: bool = False,
    include_psi_integral: bool = False,
    body: str = "",
) -> str:
    """A small footnote-style legend rendered just below a table. The o/m and n/r
    marks are explained only when they occur in the table `body`. `unit` (e.g.
    "ms"/"s") is stated once here since it has been hoisted out of every cell.
    `include_over` gates the over-threshold marker (off for Table 1, which does
    not apply the relative-error check). `include_abs_dagger` adds the marker
    for error cells showing an absolute rather than relative error (Table 1's
    ground-truth-0 fallback). `include_psi_integral` adds the marker for PSI
    cells whose output is an unevaluated-integral expression (Table 1 only;
    see PSI_INTEGRAL_RESULT)."""
    parts = []
    if unit:
        parts.append(r"all times in %s" % unit)
    if include_shade:
        parts.append(r"\colorbox{%s}{\strut Approx. Inference}" % APPROX_SHADE)
    parts.append(r"\textsc{t/o} = timeout")
    if MARK_OOM in body:
        parts.append(r"%s = out of memory" % MARK_OOM)
    parts.append(r"%s = not implemented" % MARK_ABSENT)
    if MARK_NOT_RUN in body:
        parts.append(r"%s = not run" % MARK_NOT_RUN)
    if include_over and ERROR_THRESHOLD is not None:
        parts.append(r"%s = rel.\ error $>$ %s" % (MARK_OVER, _thr_pct()))
    if include_error:
        parts.append(r"$\times$ = error")
    if include_abs_dagger:
        parts.append(r"$\dagger$ = absolute error")
    if include_psi_integral:
        parts.append(r"$(\cdot)$ = unsimplified integral expression")
    # \enskip (0.5em) between items instead of \quad so the legend fits one line.
    return r"  {\footnotesize " + r"\enskip ".join(parts) + r"}"


SPEEDUP_HEADER = r"\multicolumn{1}{l}{\shortstack[l]{Speedup}}"


def _table_open(
    caption: str,
    label: str,
    colspec: str,
    header_lines: list[str],
    fontsize: str = r"\footnotesize",
    tabcolsep: str = "4pt",
    arraystretch: str = "0.9",
    placement: str = "t",
) -> list[str]:
    head = [
        r"\begin{table}[%s]" % placement,
        r"  \centering",
        r"  %s" % fontsize,
        r"  \renewcommand{\arraystretch}{%s}" % arraystretch,  # pack rows tightly
        r"  \setlength{\tabcolsep}{%s}"
        % tabcolsep,  # narrow the inter-column gaps (default 6pt)
        r"  \caption{%s}" % caption,
        r"  \label{%s}" % label,
        r"  \begin{tabular}{%s}" % colspec,
        r"    \toprule",
    ]
    head.extend(header_lines)
    head.append(r"    \midrule")
    return head


def render_table1(run: bc.RunData, metric: str, run_dir: str, criterion: str) -> str:
    # Table 1 does NOT apply the relative-error check: it uses the default-loaded
    # budget for every cell (no fastest-under-threshold selection, no over-threshold
    # marks). Use an empty selection throughout.
    error_sel = {}
    systems = TABLE1_SYSTEMS
    hybit_col = systems.index("hybit")  # the approximate column

    def hybit_err(prog: str):
        """(error, is_absolute) for HyBit vs the ground truth on `prog` under `criterion`,
        or None when no comparison is possible. Normally the relative error; falls back to
        the ABSOLUTE error when the ground truth is 0 (relative error is undefined — e.g.
        addFun_sum, whose exact E[x+y] is 0)."""
        h = ce.program_errors(run, prog, run_dir).get("hybit")
        if h is None:
            return None
        gt, abs_e, avs = h[1], h[2], h[5]           # gt, aggregate abs err, per-repeat est.
        if gt == 0:
            return (abs_e, True)
        e = bc.error_scalar(avs, gt, criterion)
        return (e, False) if math.isfinite(e) else None

    caption = (
        r"\textbf{Standard Symbolic Inference Benchmarks.} Programs "
        r"from existing exact and symbolic PPL benchmarks. Inference time "
        r"in \emph{seconds} (excludes JIT compilation and runtime loading); \textbf{bold} = fastest exact, per row. The "
        r"Speedup column is Hybrid Pluck's speedup over the next-fastest system (blank when "
        r"no other system succeeds). Hybrid Pluck and PSI are exact throughout; HyBit is "
        r"\emph{approximate} on the non-discrete models (shaded); the final column "
        r"reports HyBit's relative approximation error vs Hybrid Pluck (absolute error for "
        r"\texttt{addFun\_sum}, exact value is~0). The models under \emph{Not Expressible in Hybrid Pluck} "
        r"condition a Gaussian on a comparison (an inequality or a $\max$), which "
        r"Hybrid Pluck does not support."
    )
    # Leading section column (variable-type class, one \multirow per section),
    # then Model, speedup, systems, and a trailing HyBit relative-error column.
    colspec = r"p{%gcm}" % SECTION_COL_WIDTH + "l" + "r" * (1 + len(systems)) + "r"
    sys_hdr = " & ".join(r"\multicolumn{1}{l}{%s} (s)" % SYS_LABEL[s] for s in systems)
    # Unit lives in the stub head (the row-label column's title) rather than in
    # every cell or the legend.
    err_hdr = r"\multicolumn{1}{l}{HyBit Rel.\ Err.}"
    header = (
        "     & Model & " + SPEEDUP_HEADER + " & " + sys_hdr + " & " + err_hdr + r" \\"
    )
    lines = _table_open(caption, "tab:inference-standard", colspec, [header])

    first = True
    for title, show_extra, rows in TABLE1_SECTIONS:
        if not first:
            lines.append(r"    \hdashline")  # dashed separator, tight
        first = False
        for j, (disp, prog) in enumerate(rows):
            cells = [table1_cell(run, s, prog, metric, error_sel) for s in systems]
            # no approximate underline in Table 1 (only bold the fastest exact)
            rendered = _emphasize(cells, systems, underline_approx=False)
            parts = []
            for i, txt in enumerate(rendered):
                # shade the approximate cells: HyBit on the non-discrete models.
                if not show_extra and i == hybit_col:
                    txt = r"\cellcolor{%s}%s" % (APPROX_SHADE, txt)
                parts.append(txt)
            # trailing relative-error column (HyBit approximate models only): the
            # error vs Pluck when computable, shaded to match the HyBit column.
            err_cell = ""
            if not show_extra:
                res = hybit_err(prog) if cells[hybit_col][1] is not None else None
                if res is None:
                    inner = ""
                else:
                    val, is_abs = res
                    inner = (
                        fmt_abserr(val) + r"$^{\dagger}$" if is_abs else fmt_relerr(val)
                    )
                err_cell = r"\cellcolor{%s}%s" % (APPROX_SHADE, inner)
            # section label: a \multirow spanning this section's rows on the first
            # row, empty on the rest. Explicit width (= the p-column width) wraps the
            # label and works on all multirow versions (the `=` shorthand is newer).
            sec = (
                (
                    r"\multirow{%d}{%gcm}{\raggedright\textbf{%s}}"
                    % (len(rows), SECTION_COL_WIDTH, latex_escape(title))
                )
                if j == 0
                else ""
            )
            label = latex_escape(disp)
            speed = pluck_speedup(run, prog, metric, systems, error_sel)
            lines.append(
                "    "
                + sec
                + " & "
                + label
                + " & "
                + speed
                + " & "
                + " & ".join(parts)
                + " & "
                + err_cell
                + r" \\"
            )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \par\vspace{2pt}",
        _legend_note(
            include_shade=True,
            include_error=False,
            include_over=False,
            include_abs_dagger=True,
            include_psi_integral=True,
            body="\n".join(lines),
        ),
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def render_table2(run: bc.RunData, metric: str, error_sel: dict) -> str:
    systems = DISPLAY_SYSTEMS
    n = len(systems)
    split = next((i for i, s in enumerate(systems) if s not in EXACT_SYSTEMS), n)
    fams = bc.families(run.universe)
    caption = (
        r"\textbf{Approximate and Exact Inference Performance on Novel Benchmarks.} "
        r"Inference time in \emph{seconds} for exact and approximate inference on models described in Table~\ref{tab:model-descriptions} (excludes JIT compilation and runtime loading); "
        r"\textbf{bold} = fastest exact, \underline{underline} = fastest "
        r"approximate, per row. The speedup column reports two ratios, "
        r"\emph{exact}\,/\,\emph{approx}: Hybrid Pluck's speedup over the fastest exact "
        r"system (PSI) and over the fastest approximate system "
        r"($\infty$ when no system succeeds). Speedups are coloured "
        r"by band: \textcolor{%s}{$\infty$}, \textcolor{%s}{$\geq 1\times$}, "
        r"\textcolor{%s}{$<1\times$}. "
        r"Model families are shown across representative sizes." % (COL_INF, COL_GT1, COL_MID)
    )

    # Model/Variant label column wraps (fixed narrow width); the Pluck-speedup
    # group is two right-aligned columns with a thin gap (@{}r@{\,}r) giving the
    # exact and approximate speedups; the system columns are natural right-aligned
    # columns, so the table takes its usual content-sized width. Model names head
    # each group; variant labels are indented sub-rows beneath.
    ncol = 3 + n  # Model + 2 speedup + n systems
    colspec = r"p{%gcm}" % MODEL_COL_WIDTH + r"@{}r@{\,}r" + "r" * n

    sys_first = 4  # first system col (1=Model, 2-3=speedup)
    exact_end = sys_first + split - 1
    approx_start = exact_end + 1
    approx_end = sys_first + n - 1
    group_row = (
        "    & \\multicolumn{2}{c}{} "
        "& \\multicolumn{%d}{l}{\\textbf{Exact}}"
        " & \\multicolumn{%d}{l}{\\textbf{Approximate}} \\\\" % (split, n - split)
    )
    # \cmidrule(lr) trims each rule at both ends, leaving a gap between the
    # Exact and Approximate underlines (plain \cline rules would abut).
    clines = r"    \cmidrule(lr){%d-%d}\cmidrule(lr){%d-%d}" % (
        sys_first,
        exact_end,
        approx_start,
        approx_end,
    )
    sys_hdr = " & ".join(r"\multicolumn{1}{l}{%s}" % SYS_LABEL[s] for s in systems)
    speedup_hdr = r"\multicolumn{2}{c}{\shortstack{Speedup}}"
    col_row = "    Model / Variant & " + speedup_hdr + " & " + sys_hdr + r" \\"
    lines = _table_open(
        caption,
        "tab:inference-applied",
        colspec,
        [group_row, clines, col_row],
        tabcolsep="3pt",
        placement="p",
    )
    # Sub-header row (below the header \midrule, in a smaller size): the two
    # speedup columns labelled exact/approximate. The time unit (seconds) is
    # stated in the caption rather than in a column head.
    sub = ["", r"{\scriptsize exact}", r"{\scriptsize approx.}"] + [""] * n
    lines.append("    " + " & ".join(sub) + r" \\")
    lines.append(r"    \addlinespace[3pt]")  # small gap before the data rows

    def cite_suffix(g: dict) -> str:
        # inline citation after the model name; models without a "cite" key in
        # TABLE2_GROUPS get no citation at all.
        return r"~\citep{%s}" % g["cite"] if g.get("cite") else ""

    def model_header(g: dict) -> str:
        # group header: the bold model name (multi-variant models) spanning the
        # whole row as a \multicolumn, with its citation after the name; no
        # timings on this row.
        return r"    \multicolumn{%d}{l}{\textbf{%s}%s} \\" % (
            ncol,
            latex_escape(g["title"]),
            cite_suffix(g),
        )

    def model_label(g: dict) -> str:
        # single-variant model: bold name (+ citation) in the Model column
        # of the data row.
        return r"\textbf{%s}%s" % (latex_escape(g["title"]), cite_suffix(g))

    def variant_label(lbl: str) -> str:
        # indented sub-row label under a model header.
        return r"\quad{\itshape %s}" % latex_escape(lbl) if lbl else ""

    def data_row(label_cell: str, prog: str, fam) -> str:
        cells = [table2_cell(run, s, prog, metric, fam, error_sel) for s in systems]
        rendered = _emphasize(
            cells, systems
        )  # bold fastest exact, underline fastest approx
        speed = pluck_speedup_split(run, prog, metric, systems, error_sel)
        return (
            "    " + label_cell + " & " + speed + " & " + " & ".join(rendered) + r" \\"
        )

    first = True
    for g in TABLE2_GROUPS:
        # if not first:
        #     lines.append(r"    \addlinespace[1pt]")  # gap between models
        first = False
        kind = g["kind"]

        if kind == "single":
            # one row: bold model name in the Model column + its timings.
            lines.append(data_row(model_label(g), g["program"], None))
            continue

        # multi-variant: bold model-name header row, then indented variant rows.
        lines.append(model_header(g))
        if kind == "variants":
            subrows: list[tuple[str, str, str | None]] = [
                (lbl, prog, None) for lbl, prog in g["rows"]
            ]
        elif kind == "family":
            fam = g["family"]
            subrows = [(lbl, prog, fam) for lbl, prog in g.get("prefix_rows", [])]
            finfo = fams.get(fam, {"sizes": {}})
            for size in g["sizes"]:
                prog = finfo["sizes"].get(size)
                if prog is None:
                    print(
                        f"warning: family '{fam}' has no variant of size {size} "
                        f"in the program universe — row skipped",
                        file=sys.stderr,
                    )
                    continue
                subrows.append((g["label_fmt"].format(n=size), prog, fam))

        for lbl, prog, fam in subrows:
            lines.append(data_row(variant_label(lbl), prog, fam))

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \par\vspace{2pt}",
        _legend_note(include_shade=False, include_error=False, body="\n".join(lines)),
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _preamble(requires: str) -> str:
    return "% Generated by experiments/make_table.py\n% Requires " + requires + "\n"


# Each table is written to its own .tex file (basename -> renderer + the LaTeX
# packages that file's table needs). The default output directory is
# <run_dir>/analysis/ (see main()).
def render_tables(
    run: bc.RunData, metric: str, error_sel: dict, run_dir: str, criterion: str
) -> dict[str, str]:
    """Return {filename: tex} — one entry per standalone table file."""
    return {
        "inference_standard.tex": (
            _preamble(
                "\\usepackage{booktabs}, \\usepackage{arydshln}, "
                "\\usepackage[table]{xcolor}, \\usepackage{multirow}"
            )
            + render_table1(run, metric, run_dir, criterion)
        ),
        "inference_applied.tex": (
            _preamble(
                "\\usepackage{booktabs}, \\usepackage[table]{xcolor}, "
                "\\usepackage{array}"
            )
            + render_table2(run, metric, error_sel)
        ),
    }


# --------------------------------------------------------------------------
# Source/result reconciliation — flag programs present in only one of the two
# places (a source file with no CSV row, or a CSV row with no source file).
# --------------------------------------------------------------------------
def reconcile(run: bc.RunData):
    """Return (detail_lines, summary_lines) describing per-system mismatches
    between the on-disk source set and the programs that have a CSV row."""
    detail: list[str] = []
    summary: list[str] = []
    # Only systems that produced a CSV can have a meaningful mismatch; a system
    # with no results at all (e.g. gen was not run) is "not run", not partial.
    for sys_ in sorted(run.done_systems):
        have_source = run.source.get(sys_, set())
        have_result = {p for (s, p) in run.results if s == sys_}
        src_only = sorted(have_source - have_result)
        res_only = sorted(have_result - have_source)
        if not (src_only or res_only):
            continue
        detail.append(f"[{sys_}] source/result mismatch:")
        if src_only:
            detail.append(
                f"    program but no result ({len(src_only)}): " + ", ".join(src_only)
            )
        if res_only:
            detail.append(
                f"    result but no program ({len(res_only)}): " + ", ".join(res_only)
            )
        summary.append(
            f"{sys_}: +{len(src_only)} source-only / +{len(res_only)} result-only"
        )
    return detail, summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", required=True,
        help="benchmark run dir",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output directory for the per-table .tex files "
        "(default: <run_dir>/analysis/)",
    )
    ap.add_argument(
        "--metric",
        choices=["infer", "wall"],
        default="infer",
        help="time metric (default: infer = inference-only)",
    )
    ap.add_argument(
        "--select",
        nargs="*",
        metavar="SYS=TAG",
        default=None,
        help="force a single sweep variant for a system (unconditional manual "
        "override, disabling the error-thresholded auto-selection for it), "
        "e.g. --select turing=turing_p64_i1000. By default each swept "
        "approximate backend shows the fastest budget under ERROR_THRESHOLD "
        "per cell (see the top of this script)",
    )
    ap.add_argument(
        "--criterion",
        default=bc.DEFAULT_CRITERION,
        choices=list(bc.ERROR_CRITERIA),
        help="error statistic gating the per-cell budget selection "
        f"(default: {bc.DEFAULT_CRITERION}, the legacy error-of-the-mean "
        "policy). *-err criteria threshold the per-repeat error distribution.",
    )
    args = ap.parse_args()

    run_dir = args.run_dir
    if not run_dir or not os.path.isdir(run_dir):
        sys.exit(f"error: run dir not found: {run_dir!r}")

    select = bc.parse_select(args.select)
    run = bc.load_run(run_dir, select=select)
    if not run.done_systems:
        sys.exit(f"error: no results_<sys>.csv found under {run_dir}")

    # Per-cell budget choice for the swept approximate backends: fastest budget
    # under ERROR_THRESHOLD, else the over-threshold marker (see build_error_selection).
    error_sel = build_error_selection(run_dir, args.metric, select, args.criterion)

    out_dir = args.out_dir or os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    tables = render_tables(run, args.metric, error_sel, run_dir, args.criterion)
    written = []
    for fname, tex in tables.items():
        path = os.path.join(out_dir, fname)
        with open(path, "w") as fh:
            fh.write(tex)
        written.append(path)
        n_not_run = tex.count(MARK_NOT_RUN)
        if n_not_run:
            print(f"warning: {fname} has {n_not_run} cell(s) marked n/r (program exists but "
                  "was not run); see 'program but no result' below", file=sys.stderr)

    detail, summary = reconcile(run)

    missing = [s for s in bc.SYSTEMS if s not in run.done_systems]
    print(f"run dir : {run_dir}")
    print(
        f"systems : done={sorted(run.done_systems)}"
        + (f"  pending={missing}" if missing else "")
    )
    if ERROR_THRESHOLD is not None:
        n_ok = sum(1 for v in error_sel.values() if v[0] == "ok")
        n_over = sum(1 for v in error_sel.values() if v[0] == "over")
        print(
            f"err-sel : criterion={args.criterion}  threshold={ERROR_THRESHOLD:g}  "
            f"under={n_ok} cells  over={n_over} cells "
            f"(remaining cells have no computable error -> default budget)"
        )
    else:
        print("err-sel : disabled (ERROR_THRESHOLD=None; default largest-budget)")
    if detail:
        print("mismatches (source vs. result):", file=sys.stderr)
        for line in detail:
            print("  " + line, file=sys.stderr)
        print("mismatch summary : " + "; ".join(summary))
    else:
        print("mismatch summary : none (every source has a result and vice versa)")
    print("wrote   : " + "\n          ".join(written))


if __name__ == "__main__":
    main()
