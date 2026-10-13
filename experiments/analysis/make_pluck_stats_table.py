#!/usr/bin/env python3
"""make_pluck_stats_table.py — LaTeX table of pluck's internal statistics per scaling variant.

A companion to make_table.py, but instead of comparing PPLs it tabulates the
pluck-internal metrics that make_pluck_scaling_plots.py plots — one row per
scaling variant (grouped by family, like Table 2's applied models), with the
columns split into three sections:

    BDD        — Nodes      (BDD guard nodes, sum)
                 Vars       (allocated BDD boolean variables V, "discrete bits")
    SPN size   — Interned   (SPN nodes DAG, all-worlds-unique)
                 Tree       (SPN nodes TREE, sum)
                 Compr.     (compression factor: Tree / Interned, a multiplicative
                            factor >= 1 for how much hash-consing shares; higher is
                            better, formatted like make_table's speedup column and
                            capped at 1,000,000)
    Time (s)   — Symbolic   (symbolic-execution / program->BDD compilation)
                 WMC        (weighted model counting)
                 Posterior  (posterior computation)

The numbers come from pluck's `=== PLUCK_STATS ===` log blocks (captured under
PLUCK_STATS=1 at <run_dir>/logs/pluck__rep*_<program>.log); parsing + per-repeat
median aggregation is shared with make_pluck_scaling_plots.program_metrics — no
rerun needed. Counts are deterministic across repeats; times are medians.

Family titles and variant sub-labels reuse make_table's TABLE2_GROUPS so the
labels match the paper's Table 2; every numeric scaling size present in the run
is listed (not just Table 2's representative subset).

Two tables are written:
  pluck_stats.tex       main-text table: the INCLUDE_FAMILIES scaling families
                        only (requires \\usepackage{booktabs}).
  pluck_stats_full.tex  appendix table: every program in the run that Hybrid
                        Pluck solves — scaling families at every size plus every
                        standalone program, alphabetically by display title — as
                        a multipage longtable (additionally requires
                        \\usepackage{longtable}). Programs pluck cannot express,
                        or that error / time out before recording statistics, are
                        omitted entirely (no marker rows).

Usage:
    uv run python make_pluck_stats_table.py [--run-dir DIR] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import sys

import bench_common as bc
import make_pluck_scaling_plots as ps
import make_table as mt

# Family title + variant label template, from make_table's TABLE2_GROUPS so the
# labels match the paper. label_fmt is a str.format template taking {n};
# prefix_rows are un-sized companion programs drawn above the sized variants
# (e.g. pcfg_ambiguous), used by the full table only.
FAM_INFO = {g["family"]: {"title": g["title"], "label_fmt": g.get("label_fmt"),
                          "prefix_rows": g.get("prefix_rows", [])}
            for g in mt.TABLE2_GROUPS if g["kind"] == "family"}

# Tables 4 and 5 label the SIR rows "timesteps $\times$ patients", with spaces,
# unlike make_table's Table 3 label, so override it here. The scaling dial N is
# the TOTAL observation count (rows = N/12).
FAM_INFO["sir_model"] = {"title": "SIR Model", "label_fmt": "timesteps $\\times$ patients={n}",
                         "prefix_rows": []}

# Multi-variant standalone groups (TABLE2 kind "variants", e.g. the two
# Felsenstein models): the full table draws these as a family-style group —
# bold title header, indented sub-rows — instead of two unrelated single rows.
VARIANT_GROUPS = [(g["title"], g["rows"])
                  for g in mt.TABLE2_GROUPS if g["kind"] == "variants"]

# Scaling families shown in the main-text table (the higher-order/factorial HMMs,
# robust linear regression "outliers", etc. are omitted). An ordered LIST, not a
# set: this IS the render order of the family blocks, and a list keeps it stable
# across runs (a set's iteration order is hash-seed-random).
INCLUDE_FAMILIES = ["sir_model", "hmm_first_order", "secret_ballot"]

# Individual (family, size) variants to drop from the table (e.g. the degenerate
# smallest HMM, timesteps=2; king_ei's smallest slice, the 1/80 fraction).
EXCLUDE_VARIANTS = {("hmm_first_order", 2), ("king_ei", 80)}

# Families whose unscaled base program (folded in under size None) is dropped from
# the FULL table's family group. sir_model's base is the hand-written single history
# that is already the n12 variant's first row, so it is redundant with — and less
# informative than — the sized sir_model_n* variants.
EXCLUDE_FULL_BASE = {"sir_model"}

# Cell marker for a variant with no complete PLUCK_STATS block (timed out / OOM'd
# before printing, or errored). t/o is distinguished from a bare dash via status.
MARK_TIMEOUT = r"\textsc{t/o}"
MARK_MISSING = r"--"

# The metric columns, in section order. The BDD and SPN sections each carry two
# raw counts plus a derived compression factor (formatted specially, see
# render_table), so they are plain header lists rather than (key, header) pairs.
# Only the time columns map straight to a metric key.
BDD_COLS = ["Nodes", "Vars"]
SPN_COLS = ["Interned", "Tree", "Compression"]
TIME_COLS = [("t_symbolic", "Symb. Exc."), ("t_wmc", "WMC"), ("t_posterior", "Posterior")]
NDATA = len(BDD_COLS) + len(SPN_COLS) + len(TIME_COLS)   # = 8


def family_title(fam: str) -> str:
    return FAM_INFO.get(fam, {}).get("title", fam).replace("--", "–")


def variant_label(fam: str, letter: str, size: int) -> str:
    """Indented sub-row label for one variant, from make_table's label_fmt when
    the family has one, else a plain '<letter>=<size>'. If the family's label_fmt
    carries no {n} placeholder (e.g. pcfg's constant 'Strict grammar'), the size
    is appended so the rows stay distinguishable."""
    fmt = FAM_INFO.get(fam, {}).get("label_fmt")
    if fmt is None:
        lbl = f"${letter}={size}$"
    elif "{n}" in fmt:
        lbl = fmt.format(n=size)
    else:
        lbl = f"{fmt}, ${letter}={size}$"
    return r"\quad{\itshape %s}" % lbl


# Counts at or above this display cap (2^32) are shown as a "$\geq 2^{32}$" bound
# rather than in full. This subsumes the genuine Rust saturation: spn_tree_size
# (stats.rs) uses saturating_add, so an exponentially large tree count arrives as
# usize::MAX (2^64-1), which is >= this cap and thus reads as the same bound.
_SIZE_CAP = 1 << 32


def fmt_times(vals: list[float]) -> list[str]:
    """Format a row's per-stage times, emphasising the two most expensive: the
    largest is set in bold, the second-largest underlined (ties broken by column
    order). Ranking is on the raw seconds; each cell is `mt.fmt_time`'d first so
    the emphasis wraps the rendered math expression. Bold uses \\boldmath (not
    \\textbf, which would leave the math-mode digits un-bolded)."""
    cells = [mt.fmt_time(v) for v in vals]
    order = sorted(range(len(vals)), key=lambda i: (-vals[i], i))
    if len(order) >= 1:
        cells[order[0]] = r"{\boldmath %s}" % cells[order[0]]
    if len(order) >= 2:
        cells[order[1]] = r"\underline{%s}" % cells[order[1]]
    return cells


def fmt_count(v: float) -> str:
    """A (deterministic, integer) count, thousands-separated (comma) in full — no
    scientific notation. At or above _SIZE_CAP the cell reads "$\\geq 2^{32}$"."""
    n = int(round(v))
    if n >= _SIZE_CAP:
        return r"${\geq}2^{32}$"          # SPN size at/over the display cap (or saturated)
    return f"{n:,}"


# SPN compression is reported as a multiplicative factor and maxes out here: a
# saturated or merely very large tree count would otherwise blow the column's
# scale, and the interesting range is the small factors near 1. At or above the
# cap the cell reads "$\geq$1,000,000x".
_SPN_COMPR_CAP = 1_000_000


def fmt_spn_compression(dag: float, tree: float) -> str:
    """SPN compression factor: how many times smaller the hash-consed DAG
    (interned) is than the unshared tree — tree/interned, a multiplicative factor
    >= 1 where higher means more sharing. Rendered like make_table's pluck-speedup
    column (mt.fmt_speedup: thousands-separated, 'x' unit). Capped at
    _SPN_COMPR_CAP, so a saturated or very large tree count reads as
    "$\\geq$<cap>x" rather than a runaway number."""
    if dag <= 0 or tree <= 0:
        return MARK_MISSING
    factor = tree / dag
    if factor >= _SPN_COMPR_CAP:
        return r"$\geq$" + mt.fmt_speedup(float(_SPN_COMPR_CAP))
    return mt.fmt_speedup(factor)


# Column spacing: columns *within* a metric group (BDD / SPN / Time) are set
# tight with this gap, while the wider default \tabcolsep is left at the group
# boundaries so the three sections read as visually distinct blocks.
_INTRA_GAP = r"@{\hspace{3pt}}"


def _group_cols(n: int) -> str:
    """`n` right-aligned columns joined by the tight intra-group gap (the gaps
    *between* groups keep the default \\tabcolsep, since concatenating two of
    these leaves a plain `r`-to-`r` boundary)."""
    return "r" + (_INTRA_GAP + "r") * (n - 1)


def _colspec(label_cm: float) -> str:
    """Model/Variant label column wraps at `label_cm`; the eight metric columns
    are natural right-aligned columns split into three \\cmidrule'd groups, tight
    within each group and default-spaced between groups (see _group_cols /
    _INTRA_GAP). The main table keeps make_table's narrow 3.2cm label column;
    the full-page longtable gets a wider one so variant labels don't wrap."""
    return (r"p{%gcm}" % label_cm + _group_cols(len(BDD_COLS))
            + _group_cols(len(SPN_COLS)) + _group_cols(len(TIME_COLS)))


def _header_rows() -> list[str]:
    """The shared two-row column header: a metric-group row (BDD / SPN size /
    Time) with \\cmidrules under each group, then the per-column header row."""
    c0 = 2                                   # first data column (1 = label column)
    c_bdd_end = c0 + len(BDD_COLS) - 1
    c_spn0 = c_bdd_end + 1
    c_spn_end = c_spn0 + len(SPN_COLS) - 1
    c_time0 = c_spn_end + 1
    c_time_end = c_time0 + len(TIME_COLS) - 1
    group_row = (r"    & \multicolumn{%d}{c}{\textbf{BDD}}"
                 r" & \multicolumn{%d}{c}{\textbf{SPN size}}"
                 r" & \multicolumn{%d}{c}{\textbf{Time (s)}} \\"
                 % (len(BDD_COLS), len(SPN_COLS), len(TIME_COLS)))
    clines = (r"    \cmidrule(lr){%d-%d}\cmidrule(lr){%d-%d}\cmidrule(lr){%d-%d}"
              % (c0, c_bdd_end, c_spn0, c_spn_end, c_time0, c_time_end))
    headers = BDD_COLS + SPN_COLS + [h for _, h in TIME_COLS]
    col_row = (r"    Model / Variant & " + " & ".join(headers) + r" \\")
    return [group_row, clines, col_row]


def data_cells(run: bc.RunData, run_dir: str, prog: str) -> str:
    """The NDATA metric cells for `prog`, " & "-joined. When no repeat produced a
    complete PLUCK_STATS block, a single spanning marker cell instead: t/o for a
    timeout, make_table's "not expressible" mark when pluck has no such program,
    and a bare dash otherwise (errored / OOM'd before printing)."""
    m = ps.program_metrics(run_dir, prog)
    if m is None:
        st = run.cell_status("pluck", prog)
        mark = (MARK_TIMEOUT if mt.status_kind(st) == "timeout"
                else mt.MARK_ABSENT if st == "ABSENT" else MARK_MISSING)
        return r"\multicolumn{%d}{c}{%s}" % (NDATA, mark)
    cells = [fmt_count(m["bdd"]), fmt_count(m["n_vars"])]
    cells += [fmt_count(m["spn_interned"]), fmt_count(m["spn_tree"]),
              fmt_spn_compression(m["spn_interned"], m["spn_tree"])]
    cells += fmt_times([m[k] for k, _ in TIME_COLS])
    return " & ".join(cells)


def render_table(run: bc.RunData, run_dir: str) -> str:
    caption = (
        r"\textbf{Hybrid Pluck internal statistics across model variants.} For each "
        r"model variant, the size of the compiled BDD and "
        r"the number of boolean variables referenced in the BDD; the evidence SPN "
        r"counted both as the hash-consed DAG (\emph{interned}) and as the unshared "
        r"\emph{tree}, with its \emph{compression} factor (tree$/$interned, the "
        r"multiplicative shrink from hash-consing); and the per-stage inference time (symbolic execution, weighted "
        r"model counting, posterior computation), in seconds. Counts are "
        r"deterministic; times are the median over ($N=15$) repeats."
    )

    lines = mt._table_open(caption, "tab:pluck-stats", _colspec(3.2),
                           _header_rows())

    fams = bc.families(run.universe)
    drawn_fams = []
    first = True
    for fam in INCLUDE_FAMILIES:
        if fam not in fams:
            continue
        info = fams[fam]
        # king_ei is sized by dataset fraction 1/n, so ascending n is *descending*
        # difficulty; reverse it so its rows read hardest-to-easiest like the rest.
        sizes = sorted((s for s in info["sizes"]
                        if s is not None and (fam, s) not in EXCLUDE_VARIANTS),
                       reverse=(fam == "king_ei"))
        if not sizes:
            continue
        if not first:
            lines.append(r"    \addlinespace[1pt]")
        first = False
        lines.append(r"    \multicolumn{%d}{l}{\textbf{%s}} \\"
                     % (NDATA + 1, mt.latex_escape(family_title(fam))))
        for size in sizes:
            prog = info["sizes"][size]
            label = variant_label(fam, info["letter"], size)
            lines.append("    " + label + " & "
                         + data_cells(run, run_dir, prog) + r" \\")
        drawn_fams.append(fam)

    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n", drawn_fams


# --------------------------------------------------------------------------
# Full (appendix) table: every program, every scaling size, as a longtable.
# --------------------------------------------------------------------------
def _prog_titles() -> dict[str, str]:
    """Display title per standalone program, from make_table's TABLE1_SECTIONS
    (display names) and TABLE2_GROUPS singles (paper titles). Programs in
    neither fall back to their escaped basename at the call site."""
    titles: dict[str, str] = {}
    for _sec, _flag, rows in mt.TABLE1_SECTIONS:
        for name, prog in rows:
            titles[prog] = name
    for g in mt.TABLE2_GROUPS:
        if g["kind"] == "single":
            titles[g["program"]] = g["title"]
    return titles


def _sub_label(lbl: str) -> str:
    """Indented italic sub-row label under a group header (matches variant_label)."""
    return r"\quad{\itshape %s}" % mt.latex_escape(lbl)


def _has_metrics(run_dir: str, prog: str) -> bool:
    """True when pluck produced a complete PLUCK_STATS block for `prog` (some
    repeat's log parsed). Programs without one — not expressible, errored, or
    timed out before printing — are dropped from the full table entirely rather
    than shown as a marker row."""
    return ps.program_metrics(run_dir, prog) is not None


def _family_subrows(run: bc.RunData, run_dir: str, fam: str,
                    info: dict) -> list[tuple[str, str]]:
    """(label, program) sub-rows for one family in the full table, KEEPING ONLY
    variants pluck has statistics for: TABLE2 prefix rows first (when the program
    exists in this run), then the unsuffixed base program (families() folds it in
    under size None, labelled by its basename), then EVERY sized variant —
    king_ei descending like the main table, so rows read hardest-to-easiest."""
    rows = [(_sub_label(lbl), prog)
            for lbl, prog in FAM_INFO.get(fam, {}).get("prefix_rows", [])
            if prog in run.universe]
    if None in info["sizes"] and fam not in EXCLUDE_FULL_BASE:
        base = info["sizes"][None]
        rows.append((_sub_label(base), base))
    sizes = sorted((s for s in info["sizes"] if s is not None),
                   reverse=(fam == "king_ei"))
    rows += [(variant_label(fam, info["letter"], s), info["sizes"][s])
             for s in sizes]
    return [(label, prog) for label, prog in rows if _has_metrics(run_dir, prog)]


def render_full_table(run: bc.RunData, run_dir: str):
    """The appendix variant: EVERY program in the run — each scaling family at
    every size (plus its prefix/base programs), TABLE2 "variants" groups (the
    Felsenstein pair) as family-style groups, and every remaining standalone
    program as its own bold row — ordered alphabetically by display title.
    Emitted as a longtable so it can break across pages, with the column header
    repeated on each page. Returns (tex, n_programs)."""
    ncols = NDATA + 1
    caption = (
        r"\textbf{Hybrid Pluck internal statistics across the full benchmark "
        r"suite.} As Table~\ref{tab:pluck-stats}, but covering every benchmark "
        r"program and scaling size that Hybrid Pluck solves, alphabetically by "
        r"model: the size of the compiled BDD and the number of boolean "
        r"variables referenced in the BDD; the evidence SPN counted both as the "
        r"hash-consed DAG (\emph{interned}) and as the unshared \emph{tree}, "
        r"with its \emph{compression} factor (tree$/$interned); and the "
        r"per-stage inference time (symbolic execution, weighted model "
        r"counting, posterior computation), in seconds. Counts are "
        r"deterministic; times are medians over repeats. Programs that are not "
        r"expressible in Hybrid Pluck, or that error or time out before "
        r"reporting statistics, are omitted."
    )
    header = _header_rows()
    # longtable preamble: caption + header on the first page, a bare "continued"
    # header on later pages, a "continued" footer everywhere but the last page.
    # Font/spacing tweaks are scoped by the \begingroup...\endgroup pair.
    lines = [
        r"\begingroup",
        r"  \footnotesize",
        r"  \renewcommand{\arraystretch}{0.9}",   # pack rows tightly
        r"  \setlength{\tabcolsep}{2pt}",         # narrow inter-column gaps
        r"  \setlength{\LTcapwidth}{\linewidth}", # full-width caption
        r"  \begin{longtable}{%s}" % _colspec(3.1),
        r"    \caption{%s}\label{tab:pluck-stats-full} \\" % caption,
        r"    \toprule",
        *header,
        r"    \midrule",
        r"  \endfirsthead",
        r"    \multicolumn{%d}{l}{{\itshape Table~\ref{tab:pluck-stats-full} "
        r"(continued)}} \\" % ncols,
        r"    \toprule",
        *header,
        r"    \midrule",
        r"  \endhead",
        r"    \midrule",
        r"    \multicolumn{%d}{r}{{\itshape continued on next page}} \\" % ncols,
        r"  \endfoot",
        r"    \bottomrule",
        r"  \endlastfoot",
    ]

    # Top-level entries: one per scaling family, one per TABLE2 "variants"
    # group, one per remaining standalone program; sorted by display title.
    # Sub-rows are filtered to variants pluck solves (_family_subrows /
    # _has_metrics), and a family/group whose every variant is filtered out —
    # or a standalone with no statistics — is dropped rather than shown as a
    # marker row. Programs belonging to a group are claimed so they aren't also
    # drawn standalone.
    fams = bc.families(run.universe)
    titles = _prog_titles()
    entries: list[tuple[str, tuple]] = []
    claimed: set[str] = set()
    for fam, info in fams.items():
        claimed |= {p for _, p in FAM_INFO.get(fam, {}).get("prefix_rows", [])}
        subrows = _family_subrows(run, run_dir, fam, info)
        if subrows:
            entries.append((family_title(fam), ("group", subrows)))
    for title, rows in VARIANT_GROUPS:
        claimed |= {p for _, p in rows}
        present = [(_sub_label(lbl), p) for lbl, p in rows
                   if p in run.universe and _has_metrics(run_dir, p)]
        if present:
            entries.append((title, ("group", present)))
    for prog in bc.standalone_programs(run.universe, fams):
        if prog not in claimed and _has_metrics(run_dir, prog):
            entries.append((titles.get(prog, prog), ("single", prog)))
    entries.sort(key=lambda e: e[0].lower())

    n_programs = 0
    for i, (title, ent) in enumerate(entries):
        if i:
            lines.append(r"    \addlinespace[1pt]")
        if ent[0] == "single":
            prog = ent[1]
            lines.append(r"    \textbf{%s} & " % mt.latex_escape(title)
                         + data_cells(run, run_dir, prog) + r" \\")
            n_programs += 1
            continue
        # \\* forbids a page break right after the group header, so a title is
        # never stranded as the last line of a page.
        lines.append(r"    \multicolumn{%d}{l}{\textbf{%s}} \\*"
                     % (ncols, mt.latex_escape(title)))
        for label, prog in ent[1]:
            lines.append("    " + label + " & "
                         + data_cells(run, run_dir, prog) + r" \\")
            n_programs += 1

    lines += [r"  \end{longtable}", r"\endgroup"]
    return "\n".join(lines) + "\n", n_programs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True,
                    help="benchmark run dir")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: <run_dir>/analysis)")
    args = ap.parse_args()

    run_dir = args.run_dir
    if not run_dir or not os.path.isdir(run_dir):
        sys.exit(f"error: run dir not found: {run_dir!r}")
    if not os.path.isdir(os.path.join(run_dir, "logs")):
        sys.exit(f"error: no logs/ dir under {run_dir} (pluck logs are required)")

    run = bc.load_run(run_dir)
    out_dir = args.out_dir or os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    tex, drawn = render_table(run, run_dir)
    tex = ("% Generated by experiments/make_pluck_stats_table.py\n"
           "% Requires \\usepackage{booktabs}\n" + tex)
    path = os.path.join(out_dir, "pluck_stats.tex")
    with open(path, "w") as fh:
        fh.write(tex)

    full_tex, n_programs = render_full_table(run, run_dir)
    full_tex = ("% Generated by experiments/make_pluck_stats_table.py\n"
                "% Requires \\usepackage{booktabs} and \\usepackage{longtable}\n"
                + full_tex)
    full_path = os.path.join(out_dir, "pluck_stats_full.tex")
    with open(full_path, "w") as fh:
        fh.write(full_tex)

    print(f"run dir : {run_dir}")
    print(f"families: {drawn}")
    print(f"wrote   : {path}")
    print(f"wrote   : {full_path} ({n_programs} program rows)")


if __name__ == "__main__":
    main()
