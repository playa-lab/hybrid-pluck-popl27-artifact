#!/usr/bin/env python3
"""compare_to_paper.py — compare a run's tables and figures with the paper's.

Parses the tables a run produced (RUN_DIR/analysis/*.tex, RUN_DIR/fig1/) with
extract_reference.py, joins them cell by cell with the paper's values in
reference/paper_reference.csv, and writes RUN_DIR/analysis/comparison.md:

  * the paper's claims, checked on this run. Each claim is evaluated on the rows
    this run produced, for both the paper and the run, so a partial run is
    compared like for like;
  * the Table 2 and 3 cells whose status differs from the paper's, split into
    cells that timed out here but finished in the paper, cells limited by this
    machine's memory (Pluck on the programs in lists/large.txt), approximate
    cells on different sides of the 5% error threshold, and other differences;
  * Tables 4 and 5's counts, which must match exactly, except the SPN sizes of the
    programs in lists/spn_varies.txt, which vary between runs and are listed apart;
  * timing ratios (this run / paper) per table and system;
  * Figure 1 and every Table 2 and 3 cell, side by side.

Usage:
    python compare_to_paper.py RUN_DIR
"""
from __future__ import annotations

import csv
import math
import os
import re
import statistics
import sys

import bench_common as bc
import extract_reference as er
import make_table as mt

EXP = bc.experiments_root()
APPROX = [s for s in mt.DISPLAY_SYSTEMS if s not in mt.EXACT_SYSTEMS]
STAGES = ["t_symbolic", "t_wmc", "t_posterior"]
COUNTS = ["bdd_nodes", "bdd_vars", "spn_interned", "spn_tree", "spn_compression"]
TIMED = {"table2": ["pluck", "psi", "hybit"], "table3": mt.DISPLAY_SYSTEMS}
FAILED = {"t/o", "o/m", "n/i", "error"}     # no result, for a reason other than n/r
MISSING = ("n/r", None)


# ------------------------------------------------------------------ loading ----
def load_paper():
    with open(os.path.join(EXP, "reference", "paper_reference.csv"), newline="") as fh:
        return {(r["table"], r["program"], r["metric"]):
                (r["status"], float(r["value"]) if r["value"] else None)
                for r in csv.DictReader(fh)}


def load_run(run_dir):
    rows = er.extract(os.path.join(run_dir, "analysis"))
    fig1 = os.path.join(run_dir, "fig1", "fig1_metrics.tex")
    if os.path.isfile(fig1):
        with open(fig1) as fh:
            tex = fh.read()
        rows += er.parse_fig1(tex)
        names = {"Polyreg": "polyreg", "Sir": "sir", "Changepoint": "changepoint",
                 "DawidSkene": "dawid_skene", "Mixture": "mixture"}
        for name, val in re.findall(r"\\FigOnePsi(\w+)\}\{([^}]*)\}", tex):
            rows.append({"table": "fig1", "program": f"fig_{names[name]}_listing",
                         "metric": "psi", "status": val if val in ("t/o", "err") else "ok",
                         "value": None})
    # Round as paper_reference.csv does, so equal cells compare equal.
    return {(r["table"], r["program"], r["metric"]):
            (r["status"], None if r["value"] is None else float(f"{r['value']:.6g}"))
            for r in rows}


def read_settings(run_dir):
    """suite.env as a dict, plus fig1's PSI timeout."""
    s = {}
    for name in ("suite.env", "fig1.out"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            with open(path) as fh:
                for line in fh:
                    for k, v in re.findall(r"(\w+)=(\S+)", line):
                        s.setdefault(k, v)
    return s


def timeout_of(settings, system):
    v = settings.get(f"TIMEOUT_{system.upper()}") or settings.get("TIMEOUT")
    return float(v) if v else None


def read_list(name):
    """The programs in lists/NAME.txt."""
    with open(os.path.join(EXP, "lists", f"{name}.txt")) as fh:
        return {line.split("#")[0].strip() for line in fh} - {""}


def must_match(key, varies):
    """Whether a Table 4/5 count must equal the paper's: all but the SPN sizes of the
    programs in lists/spn_varies.txt."""
    return not (key[2].startswith("spn_") and key[1] in varies)


def table3_models():
    """{program: model title} for every program behind a Table 3 row."""
    fams = bc.families(bc.pluck_benchmarks())
    out = {}
    for g in mt.TABLE2_GROUPS:
        if g["kind"] == "single":
            out[g["program"]] = g["title"]
            continue
        rows = g["rows"] if g["kind"] == "variants" else g.get("prefix_rows", [])
        out.update({prog: g["title"] for _lbl, prog in rows})
        if g["kind"] == "family":
            out.update({p: g["title"] for p in fams.get(g["family"], {"sizes": {}})["sizes"].values()})
    return out


# --------------------------------------------------------------- formatting ----
def fmt(v):
    if v is None:
        return "–"
    if v == int(v) and abs(v) < 1e15:                  # counts, in full
        return f"{int(v):,}"
    if 0.01 <= abs(v) < 10000:
        return f"{v:.3g}"
    return f"{v:.2e}"


def cell(c):
    status, value = c
    return fmt(value) if status in ("ok", "geq") else status


def tname(t):
    return t.replace("table", "Table ").replace("fig1", "Fig. 1")


def md_table(header, rows, empty="None."):
    if not rows:
        return [empty]
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return out


# ------------------------------------------------------------------- claims ----
def programs(paper, table, metric):
    return [p for (t, p, m) in paper if t == table and m == metric]


def covered(run, table, progs, metrics):
    """The programs of `progs` whose cells for `metrics` this run produced."""
    return [p for p in progs if all(run.get((table, p, m), MISSING)[0] != "n/r" for m in metrics)]


def psi_fails_models(cells, rows, models):
    """Table 3 models with a row where Pluck finishes and PSI does not."""
    return {models[p] for p in rows
            if cells[("table3", p, "pluck")][0] == "ok" and cells[("table3", p, "psi")][0] in FAILED}


def pluck_not_fastest(cells, rows):
    """Table 2 rows where another system is faster than Pluck."""
    return {p for p in rows if cells[("table2", p, "speedup")][0] == "ok"
            and cells[("table2", p, "speedup")][1] <= 1}


def exact_faster(cells, rows):
    """Table 3 rows where PSI or HyBit is faster than Pluck."""
    def t(p, s):
        c = cells[("table3", p, s)]
        return c[1] if c[0] == "ok" else None
    return {p for p in rows if t(p, "pluck") is not None
            and any(t(p, s) is not None and t(p, s) < t(p, "pluck") for s in ("psi", "hybit"))}


def names(progs):
    return ", ".join(sorted(progs)) or "none"


def fastest_approx(cells, rows):
    """{system: Table 3 rows where it is the fastest approximate system}."""
    wins = {}
    for p in rows:
        ts = {s: cells[("table3", p, s)][1] for s in APPROX if cells[("table3", p, s)][0] == "ok"}
        if ts:
            best = min(ts, key=ts.get)
            wins[best] = wins.get(best, 0) + 1
    return wins


def check_claims(paper, run, settings, error_rows, overview, varies):
    """[(claim, where, paper, this run, agrees)], agrees being yes / no / "–"."""
    out = []
    models = table3_models()
    t2 = programs(paper, "table2", "pluck")
    t3 = programs(paper, "table3", "pluck")

    rows = covered(run, "table3", t3, ["pluck", "psi"])
    a, b = psi_fails_models(paper, rows, models), psi_fails_models(run, rows, models)
    if a or b:
        # Agreement is judged on the rows this run has; the paper's count over all
        # its rows is shown too when skipped rows lower it.
        n_all = len(psi_fails_models(paper, t3, models))
        paper_txt = f"{len(a)} models" + (
            f" in the rows run here ({n_all} in all rows)" if len(a) != n_all else "")
        run_txt = f"{len(b)} models" + "".join(
            f" ({word} {names(s)})" for word, s in (("adds", b - a), ("misses", a - b)) if s)
        out.append(("Hybrid Pluck finishes on problems where the exact baseline (PSI) times out",
                    "Abstract (14 problems)", paper_txt, run_txt, "yes" if b >= a else "no"))
    rows = covered(run, "table2", t2, ["pluck", "psi", "hybit"])
    if rows:
        a, b = pluck_not_fastest(paper, rows), pluck_not_fastest(run, rows)
        out.append(("Pluck is the fastest deterministic method on almost all standard "
                    "benchmarks (rows where it is not)", "§6.2, Table 2", names(a), names(b),
                    "yes" if b <= a else "no"))
    rows = covered(run, "table3", t3, ["pluck", "psi", "hybit"])
    if rows:
        a, b = exact_faster(paper, rows), exact_faster(run, rows)
        out.append(("Pluck is faster than PSI and HyBit on every Table 3 row (rows where "
                    "it is not)", "§6.2, Table 3", names(a), names(b), "yes" if b <= a else "no"))
    rows = covered(run, "table3", t3, APPROX)
    if rows:
        a, b = fastest_approx(paper, rows), fastest_approx(run, rows)
        out.append(("Siren is consistently the best-performing approximate method (rows where it is fastest)",
                    "§6.3, Table 3", f"{a.get('siren', 0)} of {sum(a.values())}",
                    f"{b.get('siren', 0)} of {sum(b.values())}",
                    "–" if not b else "yes" if b.get("siren", 0) == max(b.values()) else "no"))
    rows = [p for p in covered(run, "table3", t3, ["turing"]) if p.startswith("king_ei")]
    if rows:
        def fails(c):
            return sum(c[("table3", p, "turing")][0] in ("r/e", "t/o") for p in rows)
        out.append(("Turing fails to converge on Ecological Inference (r/e or t/o)",
                    "§6.3, Fig. 9b", f"{fails(paper)} of {len(rows)} rows",
                    f"{fails(run)} of {len(rows)} rows",
                    "yes" if fails(run) == len(rows) else "no"))

    keys = [k for k in paper if k[0] in ("table4", "table5") and k[2] in COUNTS
            and must_match(k, varies) and run.get(k, MISSING)[0] in ("ok", "geq")]
    if keys:
        same = sum(run[k] == paper[k] for k in keys)
        out.append(("Tables 4 and 5's BDD and SPN sizes match exactly (except the SPN sizes "
                    "in lists/spn_varies.txt)", "Tables 4, 5",
                    f"{len(keys)} cells", f"{same} match", "yes" if same == len(keys) else "no"))

    figs = sorted(p for (t, p, m) in paper if t == "fig1" and m == "loc")
    if ("fig1", figs[0], "loc") in run:
        same = sum(run.get(("fig1", p, "loc")) == paper[("fig1", p, "loc")] for p in figs)
        out.append(("Figure 1's programs have the paper's lines of code", "Fig. 1",
                    f"{len(figs)} programs", f"{same} match", "yes" if same == len(figs) else "no"))
    if ("fig1", figs[0], "psi") in run:
        to = sum(run[("fig1", p, "psi")][0] == "t/o" for p in figs)
        out.append(("PSI times out on all five Figure 1 models",
                    "Fig. 1 caption", "t/o at 600 s",
                    f"{to} of {len(figs)} t/o at {settings.get('psi_timeout', '?')} s",
                    "yes" if to == len(figs) else "no"))

    if overview:
        want = {"Z": "4.9e-03", "P(s2)": "0.57", "iota": "0.143 -> 0.178", "rho": "0.333 -> 0.329"}
        got = {}
        m = re.search(r"Z = ([\d.e+-]+) \* eps\^2", overview)
        got["Z"] = f"{float(m.group(1)):.1e}" if m else "?"
        m = re.search(r"P\(s2 = infected\) = ([\d.]+)", overview)
        got["P(s2)"] = f"{float(m.group(1)):.2f}" if m else "?"
        for v in ("iota", "rho"):
            m = re.search(v + r"\s+prior mean ([\d.]+) -> post mean ([\d.]+)", overview)
            got[v] = f"{m.group(1)} -> {m.group(2)}" if m else "?"
        out.append(("The overview example's numbers", "§2.4",
                    "; ".join(f"{k} {v}" for k, v in want.items()),
                    "; ".join(f"{k} {v}" for k, v in got.items()),
                    "yes" if got == want else "no"))

    if error_rows:   # relative difference, or absolute where the exact answer is 0
        worst = max(float(r["abs_err"] if float(r["exact_value"]) == 0 else r["rel_err"])
                    for r in error_rows)
        out.append(("This run's Pluck answers equal the fixed exact answers (largest relative difference)",
                    "reference/exact_answers.csv", "0", f"{worst:.1e} over {len(error_rows)} programs",
                    "yes" if worst <= 1e-6 else "no"))
    return out


# ------------------------------------------------------------------- report ----
def compare_cells(paper, run, large):
    """Sort every Table 2/3 timing and error cell into a category."""
    cats = {k: [] for k in ("match", "not_run", "timeout_here", "machine", "threshold", "other")}
    for (table, prog, metric), pc in paper.items():
        if table not in TIMED or metric not in TIMED[table] + ["hybit_err"]:
            continue
        rc = run.get((table, prog, metric), MISSING)
        if metric == "hybit_err" and run.get((table, prog, "hybit"), MISSING)[0] != "ok":
            continue                                    # no HyBit result, so no error
        if pc[0] == rc[0] == "n/i":
            continue                                    # no program for this system
        item = (table, prog, metric, pc, rc)
        if rc[0] == "n/r":
            cats["not_run"].append(item)
        elif rc[0] == pc[0]:
            cats["match"].append(item)
        elif pc[0] in ("ok", "r/e") and rc[0] in ("t/o", "o/m"):
            cats["machine" if metric == "pluck" and prog in large else "timeout_here"].append(item)
        elif metric in APPROX and {pc[0], rc[0]} == {"ok", "r/e"}:
            cats["threshold"].append(item)
        else:
            cats["other"].append(item)
    return cats


def timing_ratios(paper, run):
    """[(table, system, cells, geometric mean, min, max)] of this run / paper."""
    groups = {}
    for (table, prog, metric), (ps, pv) in paper.items():
        timed = (metric in TIMED.get(table, []) or (table in ("table4", "table5") and metric in STAGES)
                 or (table == "fig1" and metric == "time_s"))
        rs, rv = run.get((table, prog, metric), MISSING)
        if timed and ps == rs == "ok" and pv and rv:
            groups.setdefault((table, metric), []).append(rv / pv)
    return [(t, m, len(rs), math.exp(statistics.fmean(math.log(r) for r in rs)), min(rs), max(rs))
            for (t, m), rs in sorted(groups.items())]


def report(run_dir):
    paper, run = load_paper(), load_run(run_dir)
    settings, large, varies = read_settings(run_dir), read_list("large"), read_list("spn_varies")
    error_rows = []
    path = os.path.join(run_dir, "analysis", "error.csv")
    if os.path.isfile(path):
        with open(path, newline="") as fh:
            error_rows = [r for r in csv.DictReader(fh) if r["backend"] == "pluck"]
    path = os.path.join(run_dir, "analysis", "overview.out")
    overview = open(path).read() if os.path.isfile(path) else ""

    claims = check_claims(paper, run, settings, error_rows, overview, varies)
    cats = compare_cells(paper, run, large)
    counts = [(k, paper[k], run.get(k, MISSING)) for k in paper
              if k[0] in ("table4", "table5") and k[2] in COUNTS]
    count_diff = [(k, pc, rc) for k, pc, rc in counts if rc[0] in ("ok", "geq") and rc != pc]
    count_varied = [c for c in count_diff if not must_match(c[0], varies)]
    count_diff = [c for c in count_diff if must_match(c[0], varies)]
    count_missing = sorted({k[1] for k, pc, rc in counts if rc[0] not in ("ok", "geq")})
    compared = sum(len(v) for k, v in cats.items() if k != "not_run")

    L = [f"# Comparison with the paper: `{os.path.basename(run_dir.rstrip('/'))}`", ""]
    if "arch" in settings:
        L += [f"Tier `{settings.get('tier', '–')}` on {settings['arch']} with "
              f"{settings.get('nproc', '?')} CPUs; timeouts: "
              + ", ".join(f"{s} {timeout_of(settings, s):g} s" for s in mt.DISPLAY_SYSTEMS
                          if timeout_of(settings, s)) + ".", ""]
    L += ["## Summary", ""]
    L.append(f"- Claims: {sum(c[4] == 'yes' for c in claims)} of {len(claims)} checked claims agree.")
    if compared:
        L.append(f"- Tables 2 and 3: {compared} cells ran; {len(cats['match'])} have the paper's "
                 f"status, {len(cats['timeout_here'])} timed out here but finished in the paper, "
                 f"{len(cats['machine'])} were limited by this machine's memory, "
                 f"{len(cats['threshold'])} are on the other side of the 5% error threshold, "
                 f"and {len(cats['other'])} differ otherwise. {len(cats['not_run'])} cells were not run.")
    if any(rc[0] in ("ok", "geq") for _k, _p, rc in counts):
        L.append(f"- Tables 4 and 5: {len(count_diff)} counts differ from the paper, and "
                 f"{len(count_varied)} SPN sizes that vary between runs; "
                 f"{len(count_missing)} programs have no counts in this run.")

    L += ["", "## The paper's claims", ""]
    L += md_table(["Claim", "Paper reference", "Paper", "This run", "Agrees"], claims,
                  "This run has no rows that bear on the paper's claims.")

    def cell_rows(items, note=lambda *a: ""):
        return [(tname(t), p, m, cell(pc), cell(rc), note(t, p, m, pc, rc))
                for t, p, m, pc, rc in items]

    def timeout_note(t, p, m, pc, rc):
        to = timeout_of(settings, m)
        if pc[0] == "r/e":
            return "the paper's run finished, with error above 5%"
        if to is None:
            return ""
        if pc[1] >= to:
            return f"the paper's time is above this run's {to:g} s timeout"
        return f"at least {to / pc[1]:.1f}× slower than the paper"

    header = ["Table", "Program", "System", "Paper (s)", "This run", "Note"]
    L += ["", "## Differences from the paper", ""]
    L += ["### Timed out here, finished in the paper", "",
          "Not a disagreement: this machine (or this tier's timeout) is slower than the "
          "paper's. The note gives the slowdown this implies.", ""]
    L += md_table(header, cell_rows(cats["timeout_here"], timeout_note))
    L += ["", "### Limited by this machine", "",
          "Pluck programs in `lists/large.txt`, which need more than 10 GB of memory; see "
          "README.md for each one's memory and time.", ""]
    L += md_table(header, cell_rows(cats["machine"]))
    L += ["", "### Approximate inference on the other side of the 5% error threshold", "",
          "Sampling is random, and smaller tiers sweep fewer budgets, so a cell near the "
          "threshold can land on either side.", ""]
    L += md_table(header, cell_rows(cats["threshold"]))
    L += ["", "### Other differences", ""]
    L += md_table(header, cell_rows(cats["other"]))
    L += ["", "### Tables 4 and 5: counts that differ", ""]
    L += md_table(["Table", "Program", "Count", "Paper", "This run"],
                  [(tname(t), p, m, cell(pc), cell(rc)) for (t, p, m), pc, rc in count_diff])
    if count_missing:
        L += ["", "Programs with no counts in this run: " + ", ".join(count_missing) + "."]
    L += ["", "### Tables 4 and 5: SPN sizes that vary between runs", "",
          "The programs in `lists/spn_varies.txt` build SPNs of different sizes on different "
          "runs or machines; their BDD counts and answers do not change. Not a disagreement.", ""]
    L += md_table(["Table", "Program", "Count", "Paper", "This run", "Difference"],
                  [(tname(t), p, m, cell(pc), cell(rc),
                    f"{(rc[1] - pc[1]) / pc[1]:+.2%}" if pc[1] else "")
                   for (t, p, m), pc, rc in count_varied])

    L += ["", "## Timing: this run / paper", "",
          "Geometric mean, smallest and largest ratio of this run's time to the paper's, over "
          "the cells where both have a time. Approximate systems' times are those of the "
          "budget each run selected.", ""]
    ratios = timing_ratios(paper, run)
    L += md_table(["Table", "Metric", "Cells", "Geometric mean", "Min", "Max"],
                  [(tname(t), m, n, f"{g:.2f}×", f"{lo:.2f}×", f"{hi:.2f}×")
                   for t, m, n, g, lo, hi in ratios], "No timing cells in common.")

    figs = sorted(p for (t, p, m) in paper if t == "fig1" and m == "loc")
    if any(("fig1", p, "loc") in run for p in figs):
        L += ["", "## Figure 1", "",
              f"Compare the panels in `fig1/` with the paper's in `experiments/reference/fig1/`. "
              "The fonts differ (the image has no Linux Libertine).", ""]
        L += md_table(["Program", "LoC paper", "LoC here", "Time paper (s)", "Time here (s)", "PSI here"],
                      [(p, cell(paper[("fig1", p, "loc")]), cell(run.get(("fig1", p, "loc"), MISSING)),
                        cell(paper[("fig1", p, "time_s")]), cell(run.get(("fig1", p, "time_s"), MISSING)),
                        cell(run.get(("fig1", p, "psi"), MISSING))) for p in figs])

    if compared:
        L += ["", "## Every cell of Tables 2 and 3", ""]
        order = {k: i for i, k in enumerate(paper)}
        items = sorted((it for v in cats.values() for it in v), key=lambda it: order[it[:3]])
        L += md_table(["Table", "Program", "System", "Paper", "This run", "This run / paper"],
                      [(tname(t), p, m, cell(pc), cell(rc),
                        f"{rc[1] / pc[1]:.2f}×" if pc[0] == rc[0] == "ok" and pc[1] else "")
                       for t, p, m, pc, rc in items])

    out = os.path.join(run_dir, "analysis", "comparison.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    summary = L[L.index("## Summary") + 2: L.index("## The paper's claims") - 1]
    print("\n".join(summary))
    print(f"comparison: {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        sys.exit(__doc__)
    report(sys.argv[1])
