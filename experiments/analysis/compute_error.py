#!/usr/bin/env python3
"""compute_error.py — approximation error of every approximate PPL vs the exact answer.

For a benchmark run directory it:
  * reads each approximate backend's posterior scalar from results_<sys>.csv's
    `result` column (populated by run_bench.sh from the programs' `RESULT:` line);
  * compares it with the program's fixed exact answer in reference/exact_answers.csv
    (written by make_exact_answers.py from Pluck's posteriors), so the error does not
    depend on which Pluck programs this run included or finished;
  * emits abs_err = |approx - exact| and rel_err = abs_err/|exact| per (program,
    backend, SETTING), to <run>/analysis/error.csv and a printed table.

A sampler-budget SWEEP writes several results_<sys>_<tag>.csv per system (e.g.
results_turing_p32_i400.csv, results_slicstan_s2000.csv). Each such budget is a
distinct `setting` and gets its OWN error row — so a program that ran ok at s2000
but errored at s8000 is still scored (at s2000), instead of being silently dropped
because the largest-budget variant happened to fail. A non-swept backend (one bare
results_<sys>.csv, e.g. hybit) reports setting "-". psi/pluck are not swept.

This run's own pluck and psi answers are reported as cross-checks against the fixed
answer when their logs reduce cleanly (expected error ~ 0). Usage:
    python compute_error.py RUN_DIR
    python compute_error.py RUN_DIR --select turing=turing_p32_i400 ...
        (restrict a system to a single budget; by default ALL budgets are scored)
Importable: program_errors(run, program, run_dir) -> {backend: (approx, exact, abs, rel)}
for the ONE variant loaded in `run` (used by make_tradeoff_plots.py per tag).
"""
import os
import re
import sys
import csv
import functools
import math
import statistics
import bench_common as bc

NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
# mappl DISABLED — see bench_common.SYSTEMS.
APPROX = ["hybit", "turing", "slicstan", "siren"]

# The fixed exact answer of every benchmark: written by make_exact_answers.py from
# Pluck's posteriors, plus scipy values for the 5 Table 2 models Pluck can't express.
EXACT_ANSWERS = os.path.join(bc.experiments_root(), "reference", "exact_answers.csv")


# ------------------------------------------------------------- pluck reduce ----
def _family_mean(line):
    """Posterior mean of a single conjugate family on `line`, or None."""
    m = re.search(r"~\s*Beta\((%s),\s*(%s)\)" % (NUM, NUM), line)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return a / (a + b) if (a + b) else None
    m = re.search(r"~\s*N\((%s),\s*(%s)\)" % (NUM, NUM), line)
    if m:
        return float(m.group(1))                       # Gaussian mean = mu
    m = re.search(r"~\s*Gamma\((%s),\s*(%s)\)" % (NUM, NUM), line)
    if m:
        sh, rt = float(m.group(1)), float(m.group(2))
        return sh / rt if rt else None                 # Gamma mean = shape/rate
    # Gamma displayed as a single collapsed mixture leaf: Mix[+w·Gamma(a,b) ...]
    mix = re.findall(r"([+-]?%s)\s*·\s*Gamma\((%s),\s*(%s)\)" % (NUM, NUM, NUM), line)
    if mix:
        num = den = 0.0
        for w, a, b in mix:
            w, a, b = float(w), float(a), float(b)
            if b:
                num += w * (a / b)
                den += w
        return num / den if den else None
    return None


def reduce_pluck(log_text):
    """Reduce a pluck posterior log to a single scalar (E[x] or P(event))."""
    # the posterior is printed after the final run's `inference_seconds:` line;
    # everything before that (PLUCK_STATS) is skipped.
    idx = log_text.rfind("inference_seconds:")
    body = log_text[idx:] if idx >= 0 else log_text

    # (0) a single projected Dirichlet component (e.g. pcfg's queried production prob
    #     `(vector_index (index 0 p) @0)`): pluck prints the marginal as
    #     `~ Dirichlet(a, b, ...)`; the first-component mean is a0/sum. Guarded to a
    #     LONE Dirichlet (no Beta/Gaussian/Gamma present) so multi-family posteriors
    #     (e.g. f81_joint's p~Beta + q~Dirichlet) fall through to the mixture scan.
    if "Dirichlet(" in body and not re.search(r"~\s*(?:Beta|N|Gamma)\(", body):
        m = re.search(r"~\s*Dirichlet\(([^)]*)\)", body)
        if m:
            ps = [float(x) for x in re.findall(NUM, m.group(1))]
            if ps and sum(ps):
                return ps[0] / sum(ps)

    # (1) continuous: weighted mean over mixture components. Component weights
    #     appear as `├ 30.41%` / `└ 69.59%`; a lone family line is weight 1.
    pairs, cur_w, saw_w = [], 1.0, False
    for line in body.splitlines():
        wm = re.search(r"(%s)\s*%%" % NUM, line)
        if wm and ("%" in line):
            cur_w, saw_w = float(wm.group(1)) / 100.0, True
            continue
        fm = _family_mean(line)
        if fm is not None:
            pairs.append((cur_w if saw_w else 1.0, fm))
            cur_w, saw_w = 1.0, False
    if pairs:
        tot = sum(w for w, _ in pairs)
        return sum(w * m for w, m in pairs) / tot if tot > 0 else None

    # (2) discrete: rows like "  True  0.997" or "  1  0.5" (value then prob).
    rows = []
    for line in body.splitlines():
        m = re.match(r"^\s+(\S+)\s+(%s)\s*$" % NUM, line)
        if m:
            rows.append((m.group(1), float(m.group(2))))
    if rows:
        d = {k.lower(): v for k, v in rows}
        if "true" in d or "false" in d:                # boolean -> P(true)
            return d.get("true", 0.0)
        num = den = 0.0                                 # integer -> E[value]
        for k, p in rows:
            try:
                v = float(k)
            except ValueError:
                return None
            num += v * p
            den += p
        return num / den if den else None
    return None


def _psi_scalar(log_text):
    """Best-effort scalar from a psi log: a Dirac δ(v) or a printed number."""
    m = re.search(r"δ\((%s)(?:/(%s))?\)" % (NUM, NUM), log_text)
    if m:
        return float(m.group(1)) / float(m.group(2)) if m.group(2) else float(m.group(1))
    return None


def log_scalar(run_dir, system, program):
    # tolerate the __rep<k> suffix that per-repeat SLURM jobs add to the log prefix
    # (ground-truth values are deterministic, so any repeat's log is representative).
    p = bc.resolve_log(run_dir, system, program)
    if not p:
        return None
    text = open(p).read()
    return _psi_scalar(text) if system == "psi" else reduce_pluck(text)


_HYBIT_TUPLE = re.compile(
    r"=\s*\(\s*%s\s*,\s*%s\s*,\s*(%s)\s*,\s*%s\s*\)" % (NUM, NUM, NUM, NUM))


def reduce_hybit_log(log_text):
    """HyBit's query scalar = the 3rd element of its summary tuple
    `@show (...) = (bits, pieces, p, t.time)`. The pluck-inexpressible models print this
    tuple but no `RESULT:` line, so the CSV `result` is `-`; this recovers the value from
    the log. Keys on the RHS 4-tuple, so the three LHS label shapes (p / p[1.0] /
    precision,num_pieces,...) don't matter. Last match wins; None if absent."""
    ms = _HYBIT_TUPLE.findall(log_text)
    return float(ms[-1]) if ms else None


def hybit_log_scalar(run_dir, program):
    p = bc.resolve_log(run_dir, "hybit", program)   # tolerates the __rep<k> suffix
    return reduce_hybit_log(open(p).read()) if p else None


@functools.cache
def _exact_answers():
    with open(EXACT_ANSWERS, newline="") as fh:
        rows = list(csv.DictReader(fh))
    dummy = [r["program"] for r in rows if r["source"] == "dummy"]
    if dummy:
        print(f"WARNING: {EXACT_ANSWERS} has dummy values for {' '.join(dummy)}; "
              "their errors are not computed", file=sys.stderr)
    return {r["program"]: (float(r["value"]), r["source"])
            for r in rows if r["source"] != "dummy"}


def ground_truth(program):
    """(value, source) of `program`'s fixed exact answer, source "pluck" or "scipy";
    (None, None) if reference/exact_answers.csv has no entry or a dummy one."""
    return _exact_answers().get(program, (None, None))


# ------------------------------------------------------------- error compute ----
def _adjust_approx(be, program, v):
    """Backend-specific value normalization before comparison.

    spike_and_slab's inclusion indicator is a SlicStan `int<2>` whose posterior
    mean can land on the {1,2} value scale rather than {0,1}. A probability must
    be <= 1, so shift down by 1 when it exceeds 1 (encoding-agnostic: this is a
    no-op when SlicStan already reports the {0,1} probability directly)."""
    if be == "slicstan" and program.startswith("spike_and_slab") and v is not None and v > 1.0:
        return v - 1.0
    return v


def program_errors(run, program, run_dir):
    """{backend: (approx, exact, abs_err, rel_err, rel_errs, approxs)} for one program.

    Indices 0-3 are unchanged (aggregate error-of-the-mean, so old callers reading
    info[3] keep working); index 4 is the list of PER-REPEAT relative errors; index 5
    is the list of adjusted PER-REPEAT estimates (feed bc.error_scalar to score under
    any criterion). Both lists fall back to the single aggregate for a legacy run."""
    gt, gt_source = ground_truth(program)
    if gt is None:
        return {}
    out = {}
    for be in APPROX:
        r = run.results.get((be, program))
        if r is None or r.status != "ok":
            continue
        reps, agg = list(r.results), r.result
        # HyBit writes its scalar to the LOG (CSV `result` is "-") for the scipy-answered
        # models; recover it from the log when the CSV carried no number.
        if be == "hybit" and not reps and agg is None:
            hv = hybit_log_scalar(run_dir, program)
            if hv is not None:
                reps, agg = [hv], hv
        if agg is None:
            continue
        av = _adjust_approx(be, program, agg)
        abs_e = abs(av - gt)
        rel_e = abs_e / abs(gt) if gt != 0 else math.inf
        avs = [_adjust_approx(be, program, v) for v in (reps or [agg])]
        rel_list = [bc._rel(a, gt) for a in avs]
        out[be] = (av, gt, abs_e, rel_e, rel_list, avs)
    # psi cross-check (exact vs exact -> ~0) only for Pluck-answered programs: for the
    # scipy-answered models _psi_scalar spuriously matches a delta index (0/1) in PSI's
    # symbolic logs, so skip it.
    if gt_source == "pluck":
        pv = log_scalar(run_dir, "psi", program)
        if pv is not None:
            rel = abs(pv - gt) / abs(gt) if gt else math.inf
            out["psi"] = (pv, gt, abs(pv - gt), rel, [rel], [pv])
    return out


def _short_tag(system, tag):
    """The budget-defining suffix of a tag ('turing_p32_i400' -> 'p32_i400'); a
    bare tag (== system, i.e. a non-swept single-budget run) -> '-'."""
    return tag[len(system) + 1:] if tag.startswith(system + "_") else "-"


def settings_to_score(run_dir, select):
    """{system: [tag, ...] sorted by budget} for the approximate backends.

    ALL result variants of each system are scored (one row per budget). An explicit
    --select SYS=TAG restricts that system to the single named tag."""
    variants = bc.discover_variants(run_dir)
    out = {}
    for system in APPROX:
        cands = variants.get(system, {})
        if not cands:
            continue
        if system in select and select[system] in cands:
            out[system] = [select[system]]
        else:
            out[system] = sorted(cands, key=bc._tag_budget)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="benchmark run dir")
    ap.add_argument("--select", nargs="*", metavar="SYS=TAG", default=None,
                    help="restrict a system to a single sweep budget, e.g. "
                         "--select turing=turing_p64_i800 (default: score ALL budgets)")
    ap.add_argument("--criterion", default=bc.DEFAULT_CRITERION,
                    choices=list(bc.ERROR_CRITERIA),
                    help="which error statistic the printed REL_ERR column reflects "
                         f"(default: {bc.DEFAULT_CRITERION}). error.csv always writes ALL "
                         "spread stats regardless.")
    args = ap.parse_args()
    run_dir = args.run_dir
    if not run_dir or not os.path.isdir(run_dir):
        sys.exit(f"no run dir: {run_dir}")

    select = bc.parse_select(args.select)
    settings = settings_to_score(run_dir, select)

    # rows carry the adjusted per-repeat estimates `avs` so any criterion / spread
    # statistic can be computed at write time via bc.ERROR_CRITERIA.
    # (program, backend, setting, budget, exact_value, approx_value, abs_err, rel_err, avs)
    rows = []
    for system, tags in settings.items():
        for tag in tags:
            run = bc.load_run(run_dir, select={system: tag})
            setting, budget = _short_tag(system, tag), bc._tag_budget(tag)
            for (s, prog), r in run.results.items():
                if s != system or r.status != "ok":
                    continue
                gt, _src = ground_truth(prog)
                if gt is None:
                    continue
                reps, agg = list(r.results), r.result
                # HyBit writes its scalar to the LOG (CSV `result` is "-") for the
                # scipy-answered models; recover it when the CSV carried no number.
                if system == "hybit" and not reps and agg is None:
                    hv = hybit_log_scalar(run_dir, prog)
                    if hv is not None:
                        reps, agg = [hv], hv
                if agg is None:
                    continue
                av = _adjust_approx(system, prog, agg)
                ae = abs(av - gt)
                rel = ae / abs(gt) if gt != 0 else math.inf
                avs = [_adjust_approx(system, prog, v) for v in (reps or [agg])]
                rows.append((prog, system, setting, budget, gt, av, ae, rel, avs))

    # exact cross-checks (not swept): this run's pluck and psi answers against the fixed
    # answer, one row per program whose log reduces. Only for Pluck-answered programs:
    # for the scipy-answered models _psi_scalar spuriously matches a delta index in
    # PSI's symbolic logs.
    for prog in sorted({p for (_, p) in bc.load_run(run_dir).results}):
        gt, src = ground_truth(prog)
        if src != "pluck":
            continue
        for system in ("pluck", "psi"):
            v = log_scalar(run_dir, system, prog)
            if v is not None:
                rel = abs(v - gt) / abs(gt) if gt != 0 else math.inf
                rows.append((prog, system, "-", 0, gt, v, abs(v - gt), rel, [v]))

    rows.sort(key=lambda x: (x[0], x[1], x[3], x[2]))  # program, backend, budget, setting

    def _spread(avs, gt):
        """(n, mean, median, p75, max) of the per-repeat relative errors."""
        rels = [bc._rel(v, gt) for v in avs]
        if not rels:
            return 0, math.nan, math.nan, math.nan, math.nan
        return (len(rels), statistics.fmean(rels), statistics.median(rels),
                bc._percentile(rels, 75), max(rels))

    os.makedirs(os.path.join(run_dir, "analysis"), exist_ok=True)
    out_csv = os.path.join(run_dir, "analysis", "error.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["program", "backend", "setting", "exact_value", "approx_value",
                    "abs_err", "rel_err", "n_repeats",
                    "rel_err_mean", "rel_err_median", "rel_err_p75", "rel_err_max"])
        for prog, be, setting, _b, gt, av, ae, re_, avs in rows:
            n, rm, rmed, rp75, rmax = _spread(avs, gt)
            w.writerow([prog, be, setting, f"{gt:.6g}", f"{av:.6g}", f"{ae:.6g}",
                        f"{re_:.6g}", n, f"{rm:.6g}", f"{rmed:.6g}",
                        f"{rp75:.6g}", f"{rmax:.6g}"])

    print(f"criterion: {args.criterion}  (REL_ERR below; error.csv has all spread stats)\n")
    print(f"{'PROGRAM':24} {'BACKEND':9} {'SETTING':11} "
          f"{'EXACT':>10} {'APPROX':>10} {'ABS_ERR':>10} {'REL_ERR':>9} {'N':>3}")
    print("-" * 92)
    for prog, be, setting, _b, gt, av, ae, re_, avs in rows:
        crit = bc.error_scalar(avs, gt, args.criterion)
        print(f"{prog:24} {be:9} {setting:11} "
              f"{gt:10.4g} {av:10.4g} {ae:10.4g} {crit:9.3g} {len(avs):3d}")
    print(f"\nwrote {out_csv}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
