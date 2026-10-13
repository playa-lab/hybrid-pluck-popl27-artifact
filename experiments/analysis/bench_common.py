"""Shared helpers for the cross-PPL benchmark analysis scripts.

Both make_table.py (LaTeX inference-time table) and make_scaling_plots.py
(log-log wall-clock-vs-size curves) read the same per-PPL results produced by
container/run_bench.sh:

    <run_dir>/results_<sys>.csv     ONE ROW PER REPEAT (long format):
        system,program,repeat,status,wall_s,infer_s,result

Each repeat's raw wall time, inference time and result scalar get their own row;
load_run() groups the rows of a (system, program) back into a single Result that
carries both the per-repeat lists (walls/infers/results) AND the derived aggregates
it computes on read (median wall/infer, mean result) — so downstream analysis can
recompute any metric while the RunData/Result interface stays unchanged.

Older run dirs use the previous one-row-per-program AGGREGATE schema
(system,program,status,runs,wall_s,infer_s,result with wall_s/infer_s = "median±sd");
load_run() detects the schema (presence of a `repeat` column) and still reads those.

A sampler-budget SWEEP (submit_bench.sh with TURING_SWEEP / SLICSTAN_SWEEP) instead
writes several files per system, tagged by budget — results_turing_p32_i500.csv,
results_turing_p64_i1000.csv, results_slicstan_s500.csv, … — all under one run dir.
discover_variants() maps system -> {tag: path}; load_run() loads one variant per system
(selectable via `select`, else the largest budget) so existing consumers are unchanged.

parse_cell() normalizes a single cell to a float or None, handling a bare number,
a legacy "median±sd" string (e.g. "0.0185±0"), ">300" (TIMEOUT sentinel), or "-".

A program that ran is classified by its `status` column (ok / TIMEOUT /
ERROR(...) / BAD / NODATA / NODRIVER). A program that is simply *absent* from a
PPL's source dir is "not expressible" in that PPL — a different thing from an
error, which is why we scan the source program dirs for the full universe.
"""

from __future__ import annotations

import csv
import glob
import math
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Callable

# PPLs in display order. pluck (the subject of the comparison) goes first.
SYSTEMS = ["pluck", "hybit", "turing", "psi", "slicstan", "siren"]

# How each PPL's programs are stored (relative to the repo root), so we can tell
# "not expressible" (no file) apart from "errored" (file exists, status is an
# ERROR/BAD). (dir, extension). The pluck dir also holds non-benchmark examples;
# only the programs listed in experiments/benchmarks.txt count (see
# source_program_sets).
PROGRAM_DIRS = {
    "pluck":    ("examples",                        "pluck"),
    "hybit":    ("benchmark_translations/hybit",    "jl"),
    "psi":      ("benchmark_translations/psi",      "psi"),
    "slicstan": ("benchmark_translations/slicstan", "slicstan"),
    "turing":   ("benchmark_translations/turing",   "jl"),
    "siren":    ("benchmark_translations/siren",    "si"),
}

# A scaling variant is "<family>_n<digits>" or "<family>_d<digits>" (king_ei
# uses d for tree depth; everything else uses n). The trailing integer is the
# problem size / scaling parameter.
_SIZE_RE = re.compile(r"^(?P<family>.+)_(?P<letter>[nd])(?P<size>\d+)$")

@dataclass
class Result:
    status: str           # ok | TIMEOUT | ERROR(...) | BAD | NODATA | NODRIVER
    runs: int             # number of successful (ok) repeats
    wall: float | None    # median wall-clock seconds (None if not a number)
    infer: float | None   # median inference-only seconds (None if "-" / n/a)
    result: float | None = None  # MEAN posterior scalar over repeats (None if "-")
    # Raw per-repeat values (populated only from the new long-format CSV; empty for
    # legacy aggregate CSVs). Index-aligned only within each list; a value missing on
    # a given repeat (e.g. infer "-") is dropped, so lengths may differ.
    walls: list[float] = field(default_factory=list)
    infers: list[float] = field(default_factory=list)
    results: list[float] = field(default_factory=list)


def parse_family(program: str):
    """('soccer', 'n', 6) for a scaling variant, else None for a standalone/base."""
    m = _SIZE_RE.match(program)
    if not m:
        return None
    return m.group("family"), m.group("letter"), int(m.group("size"))


def parse_cell(cell: str) -> float | None:
    """Normalize a wall_s/infer_s cell to its median float, or None.

    Handles "0.0185±0" (median±sd), "8.7149" (bare number), ">300" (timeout
    sentinel) and "-" (no figure)."""
    cell = (cell or "").strip()
    if not cell or cell == "-" or cell.startswith(">"):
        return None
    val = cell.split("±", 1)[0].strip()
    try:
        return float(val)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Selection criteria (shared by make_table, compute_error, make_tradeoff_plots).
#
# A "criterion" reduces a backend's per-repeat result estimates (already passed
# through compute_error._adjust_approx) plus the exact ground truth `gt` to ONE
# relative-error scalar, which is thresholded at ERROR_THRESHOLD. Two families:
#   *-result : aggregate the estimates first, then take one error (lets per-run
#              over/under-shoots cancel — the lenient legacy table policy);
#   *-err    : take each repeat's error, then aggregate (mean/median/p75/max).
# error_scalar()/select_budget() are the single place selection is implemented.
# --------------------------------------------------------------------------
ERROR_THRESHOLD = 0.05          # relative error below which a budget is "accurate"
DEFAULT_CRITERION = "p75-error" 


def _rel(v: float, gt: float) -> float:
    """Relative error |v - gt| / |gt| (inf if gt == 0)."""
    return abs(v - gt) / abs(gt) if gt else math.inf


def _percentile(xs: list[float], q: float) -> float:
    """Linear-interpolated q-th percentile (q in [0,100]) of xs."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return math.nan
    if n == 1:
        return s[0]
    idx = (q / 100.0) * (n - 1)
    lo = int(idx)
    if lo + 1 >= n:
        return s[-1]
    return s[lo] + (idx - lo) * (s[lo + 1] - s[lo])


ERROR_CRITERIA: dict[str, Callable[[list[float], float], float]] = {
    "mean-result":   lambda rs, gt: _rel(statistics.fmean(rs), gt),
    "median-result": lambda rs, gt: _rel(statistics.median(rs), gt),
    "mean-error":    lambda rs, gt: statistics.fmean([_rel(v, gt) for v in rs]),
    "median-error":  lambda rs, gt: statistics.median([_rel(v, gt) for v in rs]),
    "p75-error":     lambda rs, gt: _percentile([_rel(v, gt) for v in rs], 75),
    "max-error":     lambda rs, gt: max(_rel(v, gt) for v in rs),
}


def error_scalar(results: list[float], gt: float | None,
                 criterion: str = DEFAULT_CRITERION) -> float:
    """One relative-error scalar for a backend's per-repeat estimates vs gt under
    `criterion`. `results` should already be _adjust_approx'ed; pass the aggregate
    as a 1-element list for a legacy run. Returns inf if there is nothing to score."""
    if not results or gt is None:
        return math.inf
    return ERROR_CRITERIA[criterion](results, gt)


def select_budget(cands: list[tuple[float, float, str]],
                  criterion: str = DEFAULT_CRITERION,
                  threshold: float = ERROR_THRESHOLD):
    """Pick a budget from `cands` = [(time, err_scalar, tag), ...].
    Returns ("ok", time, tag, err) for the FASTEST budget under threshold, else
    ("over", best_err, best_tag) for the closest-to-threshold budget, or None if
    `cands` is empty. (`criterion` is accepted for symmetry; err_scalar is assumed
    already computed under it.)"""
    if not cands:
        return None
    under = [c for c in cands if c[1] < threshold]
    if under:
        t, e, tag = min(under, key=lambda c: c[0])
        return ("ok", t, tag, e)
    t, e, tag = min(cands, key=lambda c: c[1])
    return ("over", e, tag)


def experiments_root() -> str:
    """The experiments/ directory (parent of this analysis/ module)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_root() -> str:
    """The repository root (parent of experiments/)."""
    return os.path.dirname(experiments_root())


def pluck_benchmarks() -> set[str]:
    """The paper's Pluck benchmark programs, from experiments/benchmarks.txt."""
    with open(os.path.join(experiments_root(), "benchmarks.txt")) as fh:
        return {line.strip() for line in fh if line.strip()}


def source_program_sets(root: str | None = None) -> dict[str, set[str]]:
    """For each PPL, the set of program basenames present in its source dir.

    This is the universe of programs each PPL *attempts*; a program missing here
    is genuinely not expressed/implemented in that PPL (vs. one that errors)."""
    root = root or repo_root()
    out: dict[str, set[str]] = {}
    for sys, (subdir, ext) in PROGRAM_DIRS.items():
        names = set()
        for f in glob.glob(os.path.join(root, subdir, f"*.{ext}")):
            names.add(os.path.basename(f)[: -(len(ext) + 1)])
        out[sys] = names
    out["pluck"] &= pluck_benchmarks()
    return out


@dataclass
class RunData:
    run_dir: str
    results: dict[tuple[str, str], Result]  # (system, program) -> Result
    done_systems: set[str]                   # PPLs whose results_<tag>.csv exists
    source: dict[str, set[str]]              # PPL -> set of program basenames
    universe: set[str]                       # union of all program basenames
    selected: dict[str, str] | None = None   # PPL -> the result tag actually loaded
                                             # (== system, or a sweep tag like
                                             # "turing_p32_i500"); see load_run().

    # ---- classification used by both the table and the plots --------------
    def cell_status(self, system: str, program: str) -> str:
        """One of: ABSENT (not expressible), PENDING (job not finished / row
        missing), or the program's own status (ok / TIMEOUT / ERROR... / BAD)."""
        if program not in self.source.get(system, set()):
            return "ABSENT"
        if system not in self.done_systems:
            return "PENDING"
        r = self.results.get((system, program))
        if r is None:
            return "PENDING"
        return r.status

    def time(self, system: str, program: str, metric: str) -> float | None:
        """metric='infer' returns inference-only seconds (PSI has none -> wall;
        any other None infer -> wall fallback). metric='wall' returns wall."""
        r = self.results.get((system, program))
        if r is None or r.status != "ok":
            return None
        if metric == "wall":
            return r.wall
        # inference-only
        if system == "psi" or r.infer is None:
            return r.wall
        return r.infer


def _tag_budget(tag: str) -> int:
    """A crude sampler-budget for ordering a system's sweep tags (product of the
    integers in the tag): turing_p32_i500 -> 32*500 ; slicstan_s500 -> 500. Used
    only to pick the largest-budget (most accurate) variant as the default."""
    nums = [int(x) for x in re.findall(r"\d+", tag)]
    prod = 1
    for n in nums:
        prod *= n
    return prod if nums else 0


def discover_variants(run_dir: str) -> dict[str, dict[str, list[str]]]:
    """Map system -> {tag: [results_csv_paths]} for every results_<tag>.csv in run_dir.

    A run at a single fixed budget produces results_<sys>.csv (tag == system). A
    sampler-budget sweep (submit_bench.sh) produces several files per system, e.g.
    results_turing_p32_i500.csv / results_turing_p64_i1000.csv. When each repeat is
    launched as its own SLURM job the files carry a __rep<k> suffix
    (results_turing_p32_i500__rep0.csv, ...__rep1.csv, ...); those are collapsed back
    onto their base tag here, so a config's repeats appear as ONE variant whose rows
    load_run concatenates. Hence the value is a LIST of paths per tag. The owning
    system is read from each file's `system` column (robust to any tag shape)."""
    out: dict[str, dict[str, list[str]]] = {}
    prefix, suffix = "results_", ".csv"
    for path in sorted(glob.glob(os.path.join(run_dir, f"{prefix}*{suffix}"))):
        tag = os.path.basename(path)[len(prefix):-len(suffix)]
        base = re.sub(r"__rep\d+$", "", tag)     # collapse per-repeat job files onto their config
        try:
            with open(path, newline="") as fh:
                row = next(csv.DictReader(fh), None)
        except OSError:
            continue
        sysname = (row or {}).get("system", "").strip()
        if sysname:
            out.setdefault(sysname, {}).setdefault(base, []).append(path)
    return out


def resolve_log(run_dir: str, prefix: str, program: str) -> str | None:
    """Path to a per-program log, tolerating the __rep<k> suffix that per-repeat SLURM
    jobs add to the log prefix. Returns logs/<prefix>_<program>.log if present, else the
    first rep-suffixed logs/<prefix>__rep*_<program>.log, else None. Ground-truth values
    parsed from these logs are deterministic, so any repeat's log is an equally valid
    representative."""
    logdir = os.path.join(run_dir, "logs")
    exact = os.path.join(logdir, f"{prefix}_{program}.log")
    if os.path.isfile(exact):
        return exact
    reps = sorted(glob.glob(os.path.join(logdir, f"{prefix}__rep*_{program}.log")))
    return reps[0] if reps else None


def parse_select(items: list[str] | None) -> dict[str, str]:
    """Parse ['turing=turing_p64_i1000', 'slicstan=slicstan_s1000'] -> dict."""
    sel: dict[str, str] = {}
    for it in items or []:
        if "=" in it:
            k, v = it.split("=", 1)
            sel[k.strip()] = v.strip()
    return sel


def _choose_tag(system: str, cands: dict[str, list[str]], select: dict[str, str]) -> str:
    """Pick which of a system's result tags to load: an explicit --select wins;
    else the bare results_<sys>.csv (single-budget run); else, for a sweep, the
    largest-budget variant (the most accurate reference keeps default plots full)."""
    if system in select and select[system] in cands:
        return select[system]
    if system in cands:                      # bare results_<sys>.csv present
        return system
    return max(cands, key=_tag_budget)


def load_run(run_dir: str, root: str | None = None,
             select: dict[str, str] | None = None) -> RunData:
    """Read one results CSV per system under run_dir into a RunData.

    For a single-budget run this reads results_<sys>.csv exactly as before. For a
    sampler-budget sweep (several results_<sys>_<tag>.csv per system) it loads ONE
    variant per system into the (system, program) slot — chosen by `select`
    (e.g. {"turing": "turing_p64_i1000"}), else the largest budget — so every
    existing consumer keeps working unchanged. The chosen tag is recorded in
    RunData.selected; discover_variants() exposes the full set for sweep-aware code."""
    select = select or {}
    source = source_program_sets(root)
    universe = set().union(*source.values()) if source else set()
    variants = discover_variants(run_dir)
    results: dict[tuple[str, str], Result] = {}
    done: set[str] = set()
    selected: dict[str, str] = {}

    for sys in PROGRAM_DIRS:
        cands = variants.get(sys)
        if not cands:
            continue
        tag = _choose_tag(sys, cands, select)
        selected[sys] = tag
        done.add(sys)
        # cands[tag] is a LIST of files: one per per-repeat SLURM job (or a single
        # file for a run whose repeats shared a job). Concatenate the long-format
        # rows across ALL of them, then aggregate per program — so N rep-jobs collapse
        # into one Result exactly as N in-file repeat rows used to.
        grouped: dict[str, list[dict]] = {}
        for path in cands[tag]:
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh)
                if "repeat" in (reader.fieldnames or []):
                    # long format: one row per repeat — group by program across all files
                    for row in reader:
                        grouped.setdefault(row["program"], []).append(row)
                else:
                    # legacy aggregate format: one row per program (no per-repeat lists)
                    for row in reader:
                        prog = row["program"]
                        try:
                            runs = int(row["runs"])
                        except (ValueError, KeyError):
                            runs = 0
                        results[(sys, prog)] = Result(
                            status=row.get("status", "?"),
                            runs=runs,
                            wall=parse_cell(row.get("wall_s", "")),
                            infer=parse_cell(row.get("infer_s", "")),
                            result=parse_cell(row.get("result", "")),
                        )
        for prog, rows in grouped.items():
            results[(sys, prog)] = _aggregate_repeats(rows)
    return RunData(run_dir, results, done, source, universe, selected)


def _aggregate_repeats(rows: list[dict]) -> Result:
    """Collapse the per-repeat rows of one (system, program) into a Result: the raw
    ok-repeat lists plus the same aggregates run_bench.sh used to precompute (median
    wall/infer, MEAN result). status is 'ok' if any repeat succeeded, else the failing
    repeat's status (a failed program emits a single non-ok row)."""
    ok = [r for r in rows if (r.get("status") or "").strip() == "ok"]
    walls, infers, results_ = [], [], []
    for r in ok:
        w = parse_cell(r.get("wall_s", ""))
        i = parse_cell(r.get("infer_s", ""))
        v = parse_cell(r.get("result", ""))
        if w is not None:
            walls.append(w)
        if i is not None:
            infers.append(i)
        if v is not None:
            results_.append(v)
    status = "ok" if ok else ((rows[0].get("status") or "?").strip() if rows else "?")
    n = len(ok)
    # Aggregates match run_bench.sh's old precomputed cells: wall/infer are the
    # median, but infer/result collapse to None if ANY ok repeat lacked that figure
    # (run_bench wrote "-" in that case) — so mean-result/time selection stay stable.
    return Result(
        status=status,
        runs=n,
        wall=statistics.median(walls) if len(walls) == n and n else None,
        infer=statistics.median(infers) if len(infers) == n and n else None,
        result=statistics.fmean(results_) if len(results_) == n and n else None,
        walls=walls,
        infers=infers,
        results=results_,
    )


def families(universe: set[str]) -> dict[str, dict]:
    """Map family -> {'letter': 'n'|'d', 'sizes': {size: program_name}}.

    A family is any basename that has at least one _n/_d-suffixed variant. The
    unsuffixed base program (if present) is folded in under size key None."""
    fams: dict[str, dict] = {}
    for prog in universe:
        fs = parse_family(prog)
        if fs is None:
            continue
        fam, letter, size = fs
        f = fams.setdefault(fam, {"letter": letter, "sizes": {}})
        f["sizes"][size] = prog
    # fold in unsuffixed base programs that share a family name
    for prog in universe:
        if prog in fams and parse_family(prog) is None:
            fams[prog]["sizes"].setdefault(None, prog)
    return fams


def standalone_programs(universe: set[str], fams: dict[str, dict]) -> list[str]:
    """Programs that are neither a scaling variant nor a family base."""
    out = []
    for prog in universe:
        if parse_family(prog) is not None:
            continue          # a sized variant
        if prog in fams:
            continue          # a family base (folded into its family)
        out.append(prog)
    return sorted(out)


# ---------------------------------------------------------------------------
# PLUCK_STATS log parsing
# ---------------------------------------------------------------------------
# `pluck` prints one `=== PLUCK_STATS ===` block per distribution query when
# PLUCK_STATS is set (run_bench.sh always sets it), and run_one tees the whole
# stdout+stderr to logs/<tag>_<program>.log. These counters are what explain WHY a
# timing moved -- a bigger BDD, more worlds, more mixture components -- so both the
# scaling plots and the wandb logger read them from here rather than each carrying
# its own regexes.

_PS_INT = {
    "worlds": r"worlds \(value,BDD-guard pairs\)\s*:\s*(\d+)",
    "cont_latents": r"continuous latents [^:]*:\s*(\d+)",
    "bdd_vars": r"BDD vars allocated \(discrete bits\)\s*:\s*(\d+)",
    "bdd_calls": r"BDD recursive calls \(compile work\)\s*:\s*(\d+)",
    "bdd_sum": r"BDD guard nodes\s*:\s*sum=(\d+)",
    "bdd_max": r"BDD guard nodes\s*:\s*sum=\d+\s+max=(\d+)",
    "spn_tree_sum": r"SPN nodes \(TREE\)\s*:\s*sum=(\d+)",
    "spn_tree_max": r"SPN nodes \(TREE\)\s*:\s*sum=\d+\s+max=(\d+)",
    "spn_dag_max": r"SPN nodes \(DAG\)\s*:\s*per-world-max=(\d+)",
    "spn_dag_unique": r"SPN nodes \(DAG\)\s*:.*?all-worlds-unique=(\d+)",
    "spn_leaf_sum": r"SPN leaf nodes\s*:\s*sum=(\d+)",
    "components": r"posterior mixture components \(total\)\s*:\s*(\d+)",
}
_PS_FLOAT = {
    "t_parse": r"\bparse=([\d.eE+-]+)",
    "t_symbolic": r"\bsymbolic-exec=([\d.eE+-]+)",
    "t_wmc": r"\bwmc=([\d.eE+-]+)",
    "t_posterior": r"\bposterior=([\d.eE+-]+)",
    "t_flatten": r"\bflatten=([\d.eE+-]+)",
    "t_display": r"display/naming=([\d.eE+-]+)",
    "t_total": r"\btotal=([\d.eE+-]+)",
}
_PS_RE = {k: re.compile(v) for k, v in {**_PS_INT, **_PS_FLOAT}.items()}

PLUCK_STATS_KEYS = tuple(_PS_INT) + tuple(_PS_FLOAT)


# run_bench.sh appends every repeat of a program to one log, each preceded by a
# `===== <status>  wall=<s>s  rc=<n>  =====` header line.
_REPEAT_HDR = re.compile(r"^===== \S+  wall=", re.M)


def _split_repeats(text: str) -> list[str]:
    """Split a run_bench.sh log into per-repeat chunks (no headers -> one chunk)."""
    starts = [m.start() for m in _REPEAT_HDR.finditer(text)]
    if not starts:
        return [text]
    return [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]


def _parse_pluck_stats(text: str) -> dict | None:
    """Every PLUCK_STATS counter in one repeat's output, summed across its blocks.

    Summed, not last-wins: a multi-query program emits one block per query and the
    program's total cost is their sum. (run_bench.sh's own `inference_seconds:`
    scrape takes `tail -1` instead, so its infer_s is the LAST query for such a
    program -- the scaling families are single-query, where the two agree.)

    Returns None when there is no complete block, which is how a run that timed
    out or was OOM-killed before printing shows up.
    """
    if "=== PLUCK_STATS ===" not in text:
        return None
    out: dict[str, float] = {}
    for key, rx in _PS_RE.items():
        found = rx.findall(text)
        if found:
            out[key] = (sum(int(x) for x in found) if key in _PS_INT
                        else sum(float(x) for x in found))
    # A block with no timing line is a truncated log, not a usable measurement.
    return out if "t_symbolic" in out else None


def parse_pluck_stats_log(path: str) -> list[dict]:
    """The PLUCK_STATS counters of each complete repeat in one log (see
    _parse_pluck_stats); empty if the log is missing or has no complete repeat."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return []
    return [st for st in map(_parse_pluck_stats, _split_repeats(text)) if st is not None]


def pluck_log_paths(run_dir: str, program: str, tag: str = "pluck") -> list[str]:
    """Logs for one program: the per-repeat `<tag>__rep<k>_<prog>.log` files written
    under the job-array scheme, plus the un-suffixed name a single-job run writes."""
    logdir = os.path.join(run_dir, "logs")
    paths = sorted(glob.glob(os.path.join(logdir, f"{tag}__rep*_{program}.log")))
    exact = os.path.join(logdir, f"{tag}_{program}.log")
    if os.path.isfile(exact):
        paths.append(exact)
    return paths
