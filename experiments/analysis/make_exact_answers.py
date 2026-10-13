#!/usr/bin/env python3
"""make_exact_answers.py — write reference/exact_answers.csv, the fixed exact answer
of every benchmark. compute_error.py, make_table.py and make_approx_plots.py measure
all approximation error against this file, so the error columns do not depend on
which Pluck programs a run included or finished.

  * A Pluck benchmark's answer is its posterior, reduced to the program's query
    scalar (compute_error.reduce_pluck), taken from the Pluck logs of benchmark runs
    (source "pluck"). A program uses the first RUN_DIR whose log reduces to a number.
    A program with no such log gets a dummy row (value nan, source "dummy"), which
    the error scripts treat as having no exact answer.
  * The five Table 2 models Pluck cannot express get exact values computed here with
    scipy (source "scipy"). Each is the exact answer of the model HyBit runs
    (benchmark_translations/hybit/<prog>.jl), so HyBit's error isolates its
    bit-blasting error. Julia's Normal(mu, s) takes the standard deviation, and
    HyBit's bitblast(lo, hi) converges to the normal truncated to [lo, hi].
    Never seed these from HyBit's own output: that zeroes its error by construction.

Usage:
    python make_exact_answers.py [--out FILE] RUN_DIR [RUN_DIR ...]

--out writes FILE instead of reference/exact_answers.csv.
"""
import csv
import os
import sys
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, dblquad, quad
from scipy.stats import norm, truncnorm

import bench_common as bc
import compute_error as ce


def _trunc_mean(mu, sd, lo, hi):
    a, b = (lo - mu) / sd, (hi - mu) / sd
    return truncnorm.mean(a, b, loc=mu, scale=sd)


def _trunc_pdf(mu, sd, lo, hi, x):
    a, b = (lo - mu) / sd, (hi - mu) / sd
    return truncnorm.pdf(x, a, b, loc=mu, scale=sd)


def addFun_max():
    """E[max(x, y)], x, y ~ N(0, 1) = 1/sqrt(pi). HyBit truncates to [-8, 8], which
    changes E[max] by less than float precision."""
    return 1.0 / np.sqrt(np.pi)


def spacex():
    """E[engines + first_stage + second_stage] for Normal(5, 1), Normal(10, 3) and
    Normal(15, 3) (standard deviations), each truncated to [0, 64]."""
    return _trunc_mean(5, 1, 0, 64) + _trunc_mean(10, 3, 0, 64) + _trunc_mean(15, 3, 0, 64)


def weekend():
    """P(isWeekend | hour = 6) with isWeekend ~ Bernoulli(2/7) and hour truncated to
    [0, 8] from Normal(5, 4) on weekends and Normal(2, 4) on weekdays."""
    pw = 2.0 / 7.0
    d_we, d_wd = _trunc_pdf(5, 4, 0, 8, 6.0), _trunc_pdf(2, 4, 0, 8, 6.0)
    return (pw * d_we) / (pw * d_we + (1 - pw) * d_wd)


def trueskill():
    """P(m3 | m1, !m2) = 1/2 by symmetry: with d = bob - alice ~ N(0, sqrt 2) and
    p(d) = Phi(d / sqrt 8), P(m3) = E[p^2 (1-p)] / E[p (1-p)], and d -> -d shows the
    denominator is twice the numerator. Checked numerically."""
    p = lambda d: norm.cdf(d / np.sqrt(8.0))
    sd = np.sqrt(2.0)
    num = quad(lambda d: p(d) ** 2 * (1 - p(d)) * norm.pdf(d, 0, sd), -20, 20)[0]
    den = quad(lambda d: p(d) * (1 - p(d)) * norm.pdf(d, 0, sd), -20, 20)[0]
    assert abs(num / den - 0.5) < 1e-9, num / den
    return 0.5


def tug_of_war():
    """P(m5 | m1, !m2, m3, !m4) = 1/2 by symmetry (swap alice and bob). alice, bob ~
    N(5, 1); each pulls at skill - 2 with probability 1/3. Checked numerically."""
    def p_match(a, b):
        return sum(wa * wb * ((a - 2 * la) < (b - 2 * lb))
                   for la, wa in ((0, 2 / 3), (1, 1 / 3))
                   for lb, wb in ((0, 2 / 3), (1, 1 / 3)))
    w = lambda a, b: norm.pdf(a, 5, 1) * norm.pdf(b, 5, 1)
    with warnings.catch_warnings():     # the indicator limits quadrature accuracy
        warnings.simplefilter("ignore", IntegrationWarning)
        num = dblquad(lambda a, b: p_match(a, b) ** 3 * (1 - p_match(a, b)) ** 2 * w(a, b),
                      0, 10, lambda a: 0, lambda a: 10, epsabs=1e-9)[0]
        den = dblquad(lambda a, b: p_match(a, b) ** 2 * (1 - p_match(a, b)) ** 2 * w(a, b),
                      0, 10, lambda a: 0, lambda a: 10, epsabs=1e-9)[0]
    assert abs(num / den - 0.5) < 1e-4, num / den
    return 0.5


SCIPY = {f.__name__: f for f in (addFun_max, spacex, trueskill, tug_of_war, weekend)}


def main():
    run_dirs, out = sys.argv[1:], ce.EXACT_ANSWERS
    if run_dirs[:1] == ["--out"] and len(run_dirs) >= 2:
        out, run_dirs = run_dirs[1], run_dirs[2:]
    if not run_dirs or not all(os.path.isdir(d) for d in run_dirs):
        sys.exit(__doc__)
    rows, missing = [], []
    for prog in sorted(bc.pluck_benchmarks()):
        v = next((v for d in run_dirs
                  if (v := ce.log_scalar(d, "pluck", prog)) is not None), None)
        if v is None:
            missing.append(prog)
        else:
            rows.append((prog, v, "pluck"))
    if missing:
        print(f"WARNING: no Pluck answer in {run_dirs} for: {' '.join(missing)}; "
              "writing dummy rows, which get no error computed", file=sys.stderr)
    rows += [(prog, np.nan, "dummy") for prog in missing]
    rows += [(prog, f(), "scipy") for prog, f in SCIPY.items()]

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["program", "value", "source"])
        w.writerows((p, repr(float(v)), s) for p, v, s in sorted(rows))
    print(f"wrote {out} ({len(rows)} programs)")


if __name__ == "__main__":
    main()
