#!/usr/bin/env bash
#
# run_bench.sh — table-driven cross-PPL benchmark runner.
#
#   experiments/run_bench.sh [all|pluck|hybit|psi|turing|slicstan|siren]
#
# Runs every program of the chosen system(s) N times (a run that times out or
# errors is not repeated), recording per-repeat wall time, inference time and the
# posterior RESULT scalar. Usually invoked by run_suite.sh, which loops over
# systems, sampler budgets and repeats. Outputs go to $BENCH_OUT/$RUN_ID/:
#   results_<RESULT_TAG>.csv   one row per repeat: system,program,repeat,status,wall_s,infer_s,result
#   results_<RESULT_TAG>.txt   human-readable median ± sd table
#   logs/<tag>_<program>.log   full output of every repeat
#
# Env (defaults are the locations inside the Docker image):
#   BENCH_OUT  writable output dir    (default /results, else <repo>/results)
#   N          runs per program       (default 5)
#   TIMEOUT    seconds per single run (default 60)
#   PROGRAMS   program-name globs     (default '*'; set by a tier; see match_filter below)
#   FILTER     program-name globs     (default '*'; narrows PROGRAMS further)
#   RESULT_TAG output-file suffix      (default = the system arg; a sampler-budget sweep
#              sets a per-setting tag, e.g. turing_p32_i500, so several invocations for one
#              system sharing a RUN_ID write to distinct results_<tag>.csv)
#   TURING_PARTICLES / TURING_ITERS    Turing PG budget   (default 32 / 500)
#   SLICSTAN_SAMPLES / SLICSTAN_WARMUP HMC budget         (default 500 / 500)
#   SIREN_PARTICLES                    Siren SMC budget   (default 100)
#
set -uo pipefail

# ---------------------------------------------------------------- config ----
TIMEOUT="${TIMEOUT:-60}"                 # seconds allowed per single run
N="${N:-5}"                              # runs per program (when the 1st succeeds)
# Base repeat index stamped into the CSV `repeat` column. When each repeat is a separate
# invocation (run_suite.sh sets N=1 + REPEAT=<k>), this labels the row with the true
# repeat number. Defaults to 0, so an in-job N>1 run is unchanged (0..N-1).
REPEAT="${REPEAT:-0}"
WHICH="${1:-all}"                        # all | pluck | hybit | psi | turing | slicstan | siren

# RUN_ID groups this run's logs+results under BENCH_OUT/<RUN_ID>/. Defaults to a
# timestamp so repeated invocations never overwrite each other; pass it explicitly
# (e.g. --env RUN_ID=...) to make several single-system runs share one folder.
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

# RESULT_TAG suffixes the output files (results_<RESULT_TAG>.{csv,txt} and the per-program
# log prefix). It defaults to WHICH, so a normal per-system run is byte-identical to before.
# A sampler-budget SWEEP (run_suite.sh) sets it to a per-setting tag — e.g.
# RESULT_TAG=turing_p32_i500 — so several jobs for the SAME system, sharing one RUN_ID, write
# to distinct files instead of clobbering each other / racing on shared logs.
RESULT_TAG="${RESULT_TAG:-$WHICH}"
# Per-program log filename prefix. For a single-system run it's the tag (so sweep jobs get
# distinct logs); in `all` mode it stays empty and each run_<sys> falls back to its own
# system name (so pluck/hybit/... logs don't collide within one invocation).
LOG_PREFIX=""
[ "$WHICH" != all ] && LOG_PREFIX="$RESULT_TAG"

# PROGRAMS and FILTER each select programs by name: a space/comma-separated list of
# shell globs matched against each program's basename (no extension). A program runs
# only if it matches both. A tier sets PROGRAMS (the programs the tier covers); FILTER
# narrows that further. Both default to '*' = all. A pattern starting with '!'
# excludes; decisions are last-match-wins, so
#   FILTER='coinBias *_n02 king_ei_d15'   runs just those,
#   FILTER='* !*_big'                      runs everything except *_big.
# (If the list is only negations, '*' is included first.) '@NAME' stands for every
# program in lists/NAME.txt, e.g. PROGRAMS='@table2 @table3' or FILTER='!@table3'.
PROGRAMS="${PROGRAMS:-*}"
FILTER="${FILTER:-*}"

# expand_lists <patterns> -> the patterns, space-separated, with each '@NAME' replaced
# by the names in lists/NAME.txt ('!@NAME' by the same names prefixed with '!').
expand_lists() {
  local tok neg name f
  set -f
  for tok in ${1//,/ }; do
    neg=""; [ "${tok#!}" != "$tok" ] && { neg="!"; tok="${tok#!}"; }
    if [ "${tok#@}" = "$tok" ]; then printf '%s ' "$neg$tok"; continue; fi
    f="$(dirname "${BASH_SOURCE[0]}")/lists/${tok#@}.txt"
    [ -f "$f" ] || { echo "no program list $f" >&2; return 1; }
    for name in $(sed 's/#.*//' "$f"); do printf '%s ' "$neg$name"; done
  done
}
PROGRAMS_PATS=$(expand_lists "$PROGRAMS") || exit 2
FILTER_PATS=$(expand_lists "$FILTER") || exit 2

# match_filter <name> <patterns> -> 0 (include) / 1 (skip). Glob match via [[ == ]]; the
# LAST matching pattern wins. A leading '!' on a pattern marks it as an exclusion.
# Globbing is disabled while splitting the patterns so a literal '*' token isn't
# expanded to cwd filenames; [[ == ]] pattern matching is unaffected by `set -f`.
match_filter() {
  local name="$1" pat first=1 decided="" had_include=0 noglob_was_off=0
  case $- in *f*) : ;; *) noglob_was_off=1; set -f ;; esac
  for pat in $2; do
    if [ "$first" = 1 ]; then
      # if the very first token is a negation, seed with include-all.
      [ "${pat#!}" != "$pat" ] && decided="in"
      first=0
    fi
    if [ "${pat#!}" != "$pat" ]; then
      [[ "$name" == ${pat#!} ]] && decided="out"
    else
      had_include=1
      [[ "$name" == $pat ]] && decided="in"
    fi
  done
  [ "$noglob_was_off" = 1 ] && set +f
  # nothing matched: include only if no positive pattern was ever given.
  [ -z "$decided" ] && { [ "$had_include" = 0 ] && return 0 || return 1; }
  [ "$decided" = "in" ]
}

# selected <name> -> 0 if the program is in PROGRAMS and passes FILTER.
selected() { match_filter "$1" "$PROGRAMS_PATS" && match_filter "$1" "$FILTER_PATS"; }

# --- path resolution --------------------------------------------------------
# Programs are found relative to the repo; toolchain defaults are the Docker
# image's locations. An explicit value in the environment always wins.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"        # experiments/ -> repo root
TRANS_DIR="$REPO_DIR/benchmark_translations"
pick() { [ -e "$1" ] && printf '%s' "$1" || printf '%s' "$2"; }

PLUCK_BIN="${PLUCK_BIN:-$(pick /usr/local/bin/pluck "$REPO_DIR/target/release/pluck")}"
ALEA_DIR="${ALEA_DIR:-/opt/Alea.jl}"
JULIA="${JULIA:-julia}"

PLUCK_DIR="$REPO_DIR/examples"
PLUCK_LIST="$SCRIPT_DIR/benchmarks.txt"  # the paper's pluck benchmarks (examples/ also holds demos)
HYBIT_DIR="$TRANS_DIR/hybit"
PSI_DIR="$TRANS_DIR/psi"
TURING_DIR="$TRANS_DIR/turing"           # Turing.jl project + programs (its own Project.toml)
# Turing PG sampler budget. Each program reads these env vars (falling back to its
# baked defaults 32/500); exporting them here makes run_bench's values WIN over every
# per-program default, uniformly across the whole suite.
export TURING_PARTICLES="${TURING_PARTICLES:-32}"   # PG particles
export TURING_ITERS="${TURING_ITERS:-500}"          # PG Gibbs iterations

# --- Siren (semi-symbolic PPL: continuous latents symbolic, discrete via SMC) ---
# Single-stage like pluck/hybit/psi (compile+infer in one process), so the per-run
# TIMEOUT applies directly. run_siren.sh runs the siren CLI and re-emits its
# `===== Result =====` scalar + `===== Evaluation Time =====` as the harness's
# `RESULT:` / `inference_seconds:` lines.
SIREN_DIR="$TRANS_DIR/siren"
DRIVERS="$SCRIPT_DIR/drivers"
SIREN_BIN="${SIREN_BIN:-$(pick /opt/siren/venv/bin/siren siren)}"
export SIREN_METHOD="${SIREN_METHOD:-ssi}"        # ssi = semi-symbolic (the hybrid method)
export SIREN_HANDLER="${SIREN_HANDLER:-smc}"      # smc handler for the sampled (discrete) latents
export SIREN_PARTICLES="${SIREN_PARTICLES:-100}"  # SMC particle count

# --- SlicStan (compiles .slicstan -> Stan; CmdStan transpiles/builds/samples) ---
SLICSTAN_DIR="$TRANS_DIR/slicstan"
SLICSTAN_DATA="$SCRIPT_DIR/slicstan-data"
SLICSTAN_BIN="${SLICSTAN_BIN:-/usr/local/bin/slicstan}"   # SlicStan compiler (.slicstan -> Stan)
CMDSTAN="${CMDSTAN:-/opt/cmdstan}"                        # Stan backend (stanc/make/stansummary)
SLICSTAN_TIMEOUT="${SLICSTAN_TIMEOUT:-$TIMEOUT}"          # per-run wall limit for sampling
SLICSTAN_WARMUP="${SLICSTAN_WARMUP:-500}"                 # HMC warmup iterations
SLICSTAN_SAMPLES="${SLICSTAN_SAMPLES:-500}"               # HMC sampling iterations
SLICSTAN_SEED="${SLICSTAN_SEED:-0}"
# python for fix_stan.py (stdlib only): prefer the analysis venv.
SLICSTAN_PY="$(pick "$SCRIPT_DIR/.venv/bin/python" python3)"
# slicstan breakdown rows: "name|status|slic|stanc|build|sample|total|result"
SLICSTAN_ROWS=()

# Per-system "this exited 0 but did not actually infer" detector. When non-empty,
# time_once downgrades a successful run to BAD if this regex matches its log.
# Set per system in run_<sys>; default off. (PSI exits 0 yet prints
# "Pr[error] = 1" when the model collapses — e.g. a measure-zero continuous
# observation — which must NOT be reported as a fast, clean 'ok'.)
BAD_IF_MATCH=""

# Writable output: logs + scratch. In Docker, /results is a bind mount of the host
# directory (see README.md). Everything for this run lives under
# BENCH_OUT/<RUN_ID>/ so repeated invocations don't clobber each other.
BENCH_OUT="${BENCH_OUT:-$(pick /results "$REPO_DIR/results")}"
RUN_DIR="$BENCH_OUT/$RUN_ID"
LOGDIR="$RUN_DIR/logs"
SCRATCH="$RUN_DIR/scratch"              # hybit/psi cwd: holds benchmarks/<name>/
if ! mkdir -p "$LOGDIR" "$SCRATCH" 2>/dev/null; then
  echo "ERROR: cannot write to RUN_DIR=$RUN_DIR (BENCH_OUT=$BENCH_OUT)" >&2
  echo "       mount a writable dir, e.g. docker run -v \"\$PWD/results:/results\" ..." >&2
  exit 1
fi

now() { gdate +%s.%N 2>/dev/null || date +%s.%N; }   # gdate: GNU coreutils on macOS

# --- wall-time budget across multi-stage pipelines --------------------------
# slicstan runs several sequential stages (compile, then inference). The
# single-command PPLs (pluck/hybit/psi) already get a whole-process timeout via
# time_once; slicstan needs the per-program TIMEOUT to bound TOTAL wall-clock across
# ALL stages, not just the final inference stage. So each stage is wrapped in
# `timeout` with the budget LEFT after the earlier stages.
#   budget_left <limit> <spent>  -> remaining seconds, floored just above 0 (so
#                                   `timeout` never receives 0, which DISABLES it).
#   budget_done <limit> <spent>  -> exit 0 (true) once <spent> >= <limit>.
# Callers check budget_done BEFORE a stage (short-circuit to TIMEOUT) and treat a
# stage's own rc of 124/137 as a TIMEOUT too.
budget_left() { awk -v t="$1" -v s="$2" 'BEGIN{r=t-s; if(r<0.1)r=0.1; printf "%.3f", r}'; }
budget_done() { awk -v t="$1" -v s="$2" 'BEGIN{exit !(s>=t)}'; }

# ROWS: aggregate rows "system|name|status|runs|wall|infer|result" (feed the .txt table).
# CSV_ROWS: one row PER REPEAT "system|name|repeat|status|wall|infer|result" (feed the CSV,
# so downstream analysis can recompute any metric from the raw per-repeat values).
ROWS=()
CSV_ROWS=()

# median ± sample-stddev of the numbers passed as args -> "median±sd"
stats() {
  printf '%s\n' "$@" | awk '
    { v[NR]=$1; s+=$1 }
    END {
      n=NR
      if (n==0) { printf "-"; exit }
      for (i=1;i<=n;i++) for (j=i+1;j<=n;j++) if (v[j]<v[i]) { t=v[i]; v[i]=v[j]; v[j]=t }
      med = (n%2) ? v[(n+1)/2] : (v[n/2]+v[n/2+1])/2
      m = s/n; var = 0
      for (i=1;i<=n;i++) var += (v[i]-m)^2
      sd = (n>1) ? sqrt(var/(n-1)) : 0
      printf "%.4g±%.2g", med, sd
    }'
}

# time_once <workdir> <logfile> <cmd...>
# Runs the command once, APPENDING its full stdout+stderr (with a header) to
# <logfile> so every repeat is preserved on disk for later analysis — run_one
# truncates the file before the first run. Sets globals T_STATUS
# (ok|TIMEOUT|ERROR(rc=N)|BAD), T_WALL (seconds), T_INFER (inference seconds or
# "-"). Returns 0 iff the run succeeded.
time_once() {
  local workdir="$1" log="$2"; shift 2
  local start end rc out
  start=$(now)
  out=$( cd "$workdir" && exec timeout -k 5 "$TIMEOUT" "$@" 2>&1 )
  rc=$?
  end=$(now)
  T_WALL=$(awk "BEGIN{printf \"%.4f\", $end-$start}")

  if [ "$rc" -eq 124 ]; then
    T_STATUS="TIMEOUT"
  elif [ "$rc" -eq 137 ]; then
    # SIGKILL. `timeout -k 5` only KILLs at TIMEOUT+5s, so a 137 at wall >=
    # TIMEOUT is a genuine (TERM-ignoring) timeout; a 137 well before that is the
    # OOM-killer (the container hit its memory limit). Distinguish them so an
    # out-of-memory run isn't mislabeled as merely slow (it shows its real wall).
    if awk "BEGIN{exit !($T_WALL < $TIMEOUT)}"; then T_STATUS="OOM"; else T_STATUS="TIMEOUT"; fi
  elif [ "$rc" -ne 0 ]; then
    T_STATUS="ERROR(rc=$rc)"
  elif [ -n "$BAD_IF_MATCH" ] && grep -qE "$BAD_IF_MATCH" <<<"$out"; then
    # Exited 0 but produced a non-result (e.g. PSI "Pr[error] = 1": the model
    # collapsed, typically a measure-zero continuous observation). Flag it so a
    # collapsed run is never mistaken for a fast, clean inference.
    T_STATUS="BAD"
  else
    T_STATUS="ok"
  fi

  # preserve this run's full output — all N repeats accumulate in the log.
  { printf '===== %s  wall=%ss  rc=%s  =====\n' "$T_STATUS" "$T_WALL" "$rc"
    printf '%s\n' "$out"; } >> "$log"

  # inference time excluding startup: pluck prints `inference_seconds:` under
  # PLUCK_STATS=1; hybit prints a `t.time` figure. PSI prints neither, so its
  # comparable number is wall-clock (see the RESULTS footer).
  # NB: feed $out via a here-string, NOT `printf ... | grep`. Under `set -o
  # pipefail`, `grep -q` exits on the first match and closes the pipe; when $out
  # exceeds one pipe buffer (~64 KB) the upstream `printf` then dies of SIGPIPE
  # (141), which pipefail promotes to the pipeline's status — so the guard reads
  # false and the figure (present in $out) is silently dropped. This bit exactly
  # the large-posterior programs (big-N hmm/outliers, f81_joint). here-strings
  # write to a temp file, so there is no pipe and no early-close SIGPIPE.
  T_INFER="-"
  if grep -q 'inference_seconds:' <<<"$out"; then
    T_INFER=$(grep 'inference_seconds:' <<<"$out" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  elif grep -q 't\.time' <<<"$out"; then
    T_INFER=$(grep 't\.time' <<<"$out" | tail -1 | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  fi
  [ -z "$T_INFER" ] && T_INFER="-"

  # RESULT: <scalar> — posterior scalar for the cross-PPL error study (turing/hybit/
  # siren/slicstan print it; pluck/psi are reduced from their logs by the error layer).
  T_RESULT="-"
  if grep -q 'RESULT:' <<<"$out"; then
    T_RESULT=$(grep 'RESULT:' <<<"$out" | tail -1 | sed 's/.*RESULT:[[:space:]]*//' | awk '{print $1}')
  fi
  [ -z "$T_RESULT" ] && T_RESULT="-"

  [ "$T_STATUS" = "ok" ]
}

# run_one <system> <name> <workdir> <logfile> <cmd...>
run_one() {
  local system="$1" name="$2" workdir="$3" log="$4"; shift 4
  printf '  %-6s %-22s ... ' "$system" "$name"
  : > "$log"                       # fresh cumulative log; time_once appends each run

  # --- first run decides whether we repeat ---
  time_once "$workdir" "$log" "$@"
  local status="$T_STATUS" n=1
  local walls=("$T_WALL") infers=("$T_INFER") results=("$T_RESULT")

  # only repeat when the first run succeeded (a timeout/error is deterministic
  # and expensive — note it once and move on).
  if [ "$status" = "ok" ]; then
    local i
    for (( i = 2; i <= N; i++ )); do
      time_once "$workdir" "$log" "$@"
      if [ "$T_STATUS" = "ok" ]; then
        walls+=("$T_WALL"); infers+=("$T_INFER"); results+=("$T_RESULT"); n=$((n + 1))
      fi
    done
  fi

  # --- build display cells ---
  local wall_cell infer_cell
  if [ "$status" = "TIMEOUT" ]; then
    wall_cell=">${TIMEOUT}"; infer_cell="-"
  elif [ "$status" != "ok" ]; then
    wall_cell="$T_WALL"; infer_cell="-"        # errored: show when it died
  else
    wall_cell=$(stats "${walls[@]}")
    if printf '%s\n' "${infers[@]}" | grep -q '^-$'; then
      infer_cell="-"                           # some run had no inference figure
    else
      infer_cell=$(stats "${infers[@]}")
    fi
  fi

  # representative result scalar: mean over successful runs ("-" if any run
  # printed none, e.g. pluck/psi — their value is reduced from the log downstream).
  local result_cell
  if [ "$status" != "ok" ] || printf '%s\n' "${results[@]}" | grep -qvE '^-?[0-9.]+([eE][-+]?[0-9]+)?$'; then
    result_cell="-"
  else
    result_cell=$(printf '%s\n' "${results[@]}" | awk '{s+=$1; n++} END{if(n) printf "%.6g", s/n; else print "-"}')
  fi
  printf '%-13s n=%-2s wall=%s\n' "$status" "$n" "$wall_cell"
  ROWS+=("$system|$name|$status|$n|$wall_cell|$infer_cell|$result_cell")

  # one CSV row per successful repeat (walls/infers/results are index-aligned); a
  # program with no successful run emits a single row carrying its failure status.
  if [ "$status" = "ok" ]; then
    local r
    for (( r = 0; r < n; r++ )); do
      CSV_ROWS+=("$system|$name|$((REPEAT + r))|ok|${walls[r]}|${infers[r]}|${results[r]}")
    done
  else
    CSV_ROWS+=("$system|$name|$REPEAT|$status|$wall_cell|-|-")
  fi
}

# ====================================================================
# PPL registry — one function per system. Each iterates its program dir
# and dispatches the system's command to run_one. To add a PPL: add a
# run_<sys> function below + a case in dispatch() (see commented hooks).
# ====================================================================

# pluck: env PLUCK_STATS=1 makes the engine print its stage breakdown plus an
# `inference_seconds:` line (= sum of stages, startup excluded). The size walks
# PLUCK_STATS runs add wall-clock but are excluded from inference_seconds, so the
# parsed number stays clean. It is printed once per query, and time_once keeps the
# last (`tail -1`): for the multi-query benchmarks (typos, polyreg, f81_joint) the
# time is that of the last query, whose answer is the error scalar, as in the paper.
# Tables 4 and 5 add up the stages of all queries instead.
#
# cwd: programs `(include "examples/shared/...")` with paths relative to the repo
# root, so pluck runs from there. Only the programs in benchmarks.txt are run.
run_pluck() {
  if [ ! -x "$PLUCK_BIN" ]; then
    echo "    pluck binary not found/executable at $PLUCK_BIN — skipping." >&2; return 1
  fi
  echo "==> Running pluck programs (N=${N}, timeout ${TIMEOUT}s)"
  [ "$FILTER" != '*' ] && echo "    filter: $FILTER"
  BAD_IF_MATCH=""
  local name
  while read -r name; do
    [ -n "$name" ] || continue
    selected "$name" || continue
    # PLUCK_DISPLAY_PRECISION widens posterior params + mixture weights from the
    # default 4/2 dp; 6 dp here lowers the ground-truth precision floor for the
    # cross-PPL error study (compute_error.py reduces this printed output).
    run_one pluck "$name" "$REPO_DIR" "$LOGDIR/${LOG_PREFIX:-pluck}_$name.log" \
      env PLUCK_STATS=1 PLUCK_DISPLAY_PRECISION="${PLUCK_DISPLAY_PRECISION:-6}" "$PLUCK_BIN" "$PLUCK_DIR/$name.pluck"
  done < "$PLUCK_LIST"
}

# hybit: julia + Alea.jl. The .jl files use their baked-in default (bits,pieces),
# tuned per program for the paper.
run_hybit() {
  if ! { command -v "$JULIA" >/dev/null 2>&1 || [ -x "$JULIA" ]; }; then
    echo "    julia ('$JULIA') not found — skipping hybit. (set JULIA=... for a non-PATH install)" >&2; return 1
  fi
  # Use the prebuilt Dice sysimage when present: skips ~13s/run of package load
  # (PyCall/SymPy) + first-call JIT, and gives a clean t.time.
  BAD_IF_MATCH=""
  local sysimg=()
  if [ -n "${DICE_SYSIMAGE:-}" ] && [ -f "${DICE_SYSIMAGE:-}" ]; then
    sysimg=(--sysimage "$DICE_SYSIMAGE")
    echo "==> Running hybit programs (N=${N}, timeout ${TIMEOUT}s, Julia + Alea.jl + sysimage)"
  else
    echo "==> Running hybit programs (N=${N}, timeout ${TIMEOUT}s, Julia + Alea.jl)"
    echo "    note: no DICE_SYSIMAGE — wall-clock includes ~13s startup+load+JIT each run."
  fi
  [ "$FILTER" != '*' ] && echo "    filter: $FILTER"
  local f name
  for f in "$HYBIT_DIR"/*.jl; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .jl)
    selected "$name" || continue
    run_one hybit "$name" "$SCRATCH" "$LOGDIR/${LOG_PREFIX:-hybit}_$name.log" \
      "$JULIA" "${sysimg[@]}" --startup-file=no --project="$ALEA_DIR" "$f"
  done
}

# Turing.jl programs (pure Particle Gibbs). Parses
# `inference_seconds:` (the sampling loop only; JIT warmed internally). Per-run
# wall-clock also includes Julia startup + the (heavy) Turing load unless a
# sysimage is supplied via TURING_SYSIMAGE.
run_turing() {
  if ! { command -v "$JULIA" >/dev/null 2>&1 || [ -x "$JULIA" ]; }; then
    echo "    julia ('$JULIA') not found — skipping turing. (set JULIA=... for a non-PATH install)" >&2; return 1
  fi
  if [ ! -f "$TURING_DIR/Project.toml" ]; then
    echo "    no Turing project at $TURING_DIR/Project.toml — skipping turing. (run: julia --project=$TURING_DIR -e 'using Pkg; Pkg.instantiate()')" >&2; return 1
  fi
  BAD_IF_MATCH=""
  local sysimg=()
  if [ -n "${TURING_SYSIMAGE:-}" ] && [ -f "${TURING_SYSIMAGE:-}" ]; then
    sysimg=(--sysimage "$TURING_SYSIMAGE")
    echo "==> Running turing programs (N=${N}, PG particles=${TURING_PARTICLES}, iters=${TURING_ITERS}, timeout ${TIMEOUT}s, Julia + Turing.jl + sysimage)"
  else
    echo "==> Running turing programs (N=${N}, PG particles=${TURING_PARTICLES}, iters=${TURING_ITERS}, timeout ${TIMEOUT}s, Julia + Turing.jl)"
    echo "    note: no TURING_SYSIMAGE — wall-clock includes ~30-60s Julia startup + Turing load each run;"
    echo "          inference_seconds: (parsed) times the sampling loop only (JIT warmed internally)."
  fi
  [ "$FILTER" != '*' ] && echo "    filter: $FILTER"
  local f name
  for f in "$TURING_DIR"/*.jl; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .jl)
    selected "$name" || continue
    run_one turing "$name" "$TURING_DIR" "$LOGDIR/${LOG_PREFIX:-turing}_$name.log" \
      "$JULIA" ${sysimg[@]+"${sysimg[@]}"} --startup-file=no --project="$TURING_DIR" "$f"
  done
}

# psi: PSI prints no inference line, so WALL-CLOCK is the comparable number.
run_psi() {
  if ! command -v psi >/dev/null 2>&1; then
    echo "    psi not found on PATH — skipping psi." >&2; return 1
  fi
  echo "==> Running psi programs (N=${N}, timeout ${TIMEOUT}s)"
  echo "    note: PSI reports no inference-only time; WALL(s) is the comparable figure."
  echo "    note: a 'BAD' status = PSI exited 0 but printed 'Pr[error] = 1' (model"
  echo "          collapsed, e.g. a measure-zero continuous observation) — not a real result."
  # PSI exits 0 even when the posterior collapses; catch that as BAD.
  BAD_IF_MATCH='Pr\[error\] *= *1'
  [ "$FILTER" != '*' ] && echo "    filter: $FILTER"
  local f name
  for f in "$PSI_DIR"/*.psi; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .psi)
    selected "$name" || continue
    run_one psi "$name" "$SCRATCH" "$LOGDIR/${LOG_PREFIX:-psi}_$name.log" \
      psi "$f"
  done
}

# siren: a .si program is run by the siren CLI (semi-symbolic inference). Single
# process, so TIMEOUT applies directly. The wrapper (siren-drivers/run_siren.sh)
# re-emits siren's Result scalar + Evaluation Time as `RESULT:` /
# `inference_seconds:` for the shared parser. Continuous latents are symbolic,
# discrete latents are SMC samples (-p SIREN_PARTICLES). File paths are absolute
# so cwd is irrelevant (programs inline their data).
run_siren() {
  if [ ! -x "$SIREN_BIN" ]; then
    echo "    siren not found/executable at $SIREN_BIN — skipping siren." >&2; return 1
  fi
  echo "==> Running siren programs (N=${N}, method=${SIREN_METHOD}, handler=${SIREN_HANDLER}, particles=${SIREN_PARTICLES}, timeout ${TIMEOUT}s)"
  BAD_IF_MATCH=""
  [ "$FILTER" != '*' ] && echo "    filter: $FILTER"
  local f name
  for f in "$SIREN_DIR"/*.si; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .si)
    selected "$name" || continue
    run_one siren "$name" "$SIREN_DIR" "$LOGDIR/${LOG_PREFIX:-siren}_$name.log" \
      env SIREN_BIN="$SIREN_BIN" bash "$DRIVERS/run_siren.sh" "$f"
  done
}

# slicstan: a .slicstan program is COMPILED to Stan by SlicStan (auto-marginalizing
#   discrete int<K> latents), the generated Stan is modernized (fix_stan.py),
#   transpiled+built by CmdStan, and sampled with HMC/NUTS. Each pipeline component
#   is timed separately (slic, stanc, build, sample) plus the total. The result is
#   the posterior mean of the query parameter (slicstan-data/<name>.query) plus the
#   max R_hat as a convergence sanity flag. Statuses:
#     ok            ran end-to-end (RESULT = mean(param) + Rhat)
#     ERROR(slic)   .slicstan rejected by SlicStan (unsupported DSL feature)
#     ERROR(stanc)  compiled to Stan, but the generated Stan is invalid (codegen)
#     ERROR(build)  valid Stan, but the C++ build failed
#     ERROR(run)    built, but HMC failed (e.g. could not initialize)
#     NODATA        compiles but no slicstan-data/<name>.json is present
#     TIMEOUT       compile + sampling exceeded the SLICSTAN_TIMEOUT wall budget
#                   (usually SlicStan's compile-time discrete marginalization)
slicstan_result() {  # <out.csv> <query-file> -> "mean(param)=.. Rhat=.."
  local csv="$1" qf="$2" q mean rhat
  rhat=$("$CMDSTAN/bin/stansummary" "$csv" 2>/dev/null | awk '
    /^lp__/{s=1} s { v=$NF+0; if (v==v && v>m) m=v } END { if (m=="") print "?"; else printf "%.2f", m }')
  q=$(head -1 "$qf" 2>/dev/null | awk '{print $1}')
  if [ -n "$q" ]; then
    mean=$("$CMDSTAN/bin/stansummary" --sig_figs=4 "$csv" 2>/dev/null | awk -v p="$q" '$1==p {print $2; exit}')
    [ -n "$mean" ] && { echo "mean($q)=$mean Rhat=$rhat"; return; }
  fi
  echo "Rhat=$rhat"
}

# slic_stop <status> <wall>: record a SlicStan program that stopped before sampling
# (compile timeout or error), in both the text table and the CSV.
slic_stop() {
  ROWS+=("slicstan|$name|$1|0|$2|-")
  CSV_ROWS+=("slicstan|$name|$REPEAT|$1|$2|-|-")
}
run_slicstan() {
  if [ ! -x "$SLICSTAN_BIN" ]; then
    echo "    slicstan not found/executable at $SLICSTAN_BIN — skipping slicstan." >&2; return 1
  fi
  if [ ! -x "$CMDSTAN/bin/stanc" ]; then
    echo "    cmdstan not found at $CMDSTAN — skipping slicstan." >&2; return 1
  fi
  echo "==> Running slicstan programs (N=${N}, warmup=${SLICSTAN_WARMUP}, samples=${SLICSTAN_SAMPLES}, wall budget ${SLICSTAN_TIMEOUT}s/program)"
  echo "    pipeline per program: slic (SlicStan->Stan) | stanc | build (C++) | sample (HMC)."
  echo "    note: the ${SLICSTAN_TIMEOUT}s budget bounds TOTAL wall (compile + sampling); each"
  echo "          stage gets the time left after the earlier ones (SlicStan compile included)."
  BAD_IF_MATCH=""
  local f name W raw stan data qf clog t0 t1 rc status result wall i
  local t_slic t_stanc t_build sample_c sample_med infer_c total_c darg bud compile_s
  [ "$FILTER" != '*' ] && echo "    filter: $FILTER"
  for f in "$SLICSTAN_DIR"/*.slicstan; do
    [ -e "$f" ] || continue
    name=$(basename "$f" .slicstan)
    selected "$name" || continue
    # Namespace the work dir by RESULT_TAG: concurrent budget-sweep jobs share one
    # RUN_ID (hence $SCRATCH on NFS), so a bare per-name path lets a sibling job's
    # `rm -rf` yank this job's files mid-pipeline -> ESTALE ("Stale file handle")
    # -> spurious ERROR(slic)/ERROR(stanc)/ERROR(run). Tagging isolates each job.
    W="$SCRATCH/slicstan/${RESULT_TAG:+$RESULT_TAG/}$name"; rm -rf "$W"; mkdir -p "$W"
    raw="$W/raw.stan"; stan="$W/model.stan"
    data="$SLICSTAN_DATA/$name.json"; qf="$SLICSTAN_DATA/$name.query"
    clog="$LOGDIR/${LOG_PREFIX:-slicstan}_${name}.log"; : > "$clog"
    printf '  %-8s %-18s ... ' slicstan "$name"
    t_slic="-"; t_stanc="-"; t_build="-"; sample_c="-"

    # --- SLIC: strip comments (SlicStan has none), SlicStan -> Stan, modernize ---
    # Wrapped in `timeout` (full budget — first stage) so SlicStan's compile-time
    # discrete marginalization (which can run for thousands of seconds on large
    # models) is capped by the per-program TIMEOUT and recorded as TIMEOUT, instead
    # of running unbounded and reporting a multi-thousand-second "ok".
    sed 's://.*$::' "$f" > "$W/in.slicstan"
    # SlicStan aborts (SIGABRT) when it rejects a program; run it in a subshell with
    # core dumps disabled and the shell's "Aborted" job message swallowed.
    t0=$(now); ( ulimit -c 0 2>/dev/null; exec timeout -k 5 "$SLICSTAN_TIMEOUT" "$SLICSTAN_BIN" --from-file "$W/in.slicstan" >"$raw" 2>"$clog" ) 2>/dev/null; rc=$?; t1=$(now)
    t_slic=$(awk "BEGIN{printf \"%.4f\", $t1-$t0}")
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      echo "TIMEOUT"; SLICSTAN_ROWS+=("$name|TIMEOUT|$t_slic|-|-|-|$t_slic|SlicStan compile exceeded ${SLICSTAN_TIMEOUT}s")
      slic_stop "TIMEOUT" "$t_slic"; continue
    fi
    if [ "$rc" -ne 0 ] || [ ! -s "$raw" ]; then
      echo "ERROR(slic)"; SLICSTAN_ROWS+=("$name|ERROR(slic)|$t_slic|-|-|-|-|SlicStan rejected")
      slic_stop "ERROR(slic)" "-"; continue
    fi
    "$SLICSTAN_PY" "$DRIVERS/fix_stan.py" <"$raw" >"$stan" 2>>"$clog"

    # --- STANC: validate the generated Stan (budget = TIMEOUT minus slic) ---
    if budget_done "$SLICSTAN_TIMEOUT" "$t_slic"; then
      echo "TIMEOUT"; SLICSTAN_ROWS+=("$name|TIMEOUT|$t_slic|-|-|-|$t_slic|compile exceeded ${SLICSTAN_TIMEOUT}s")
      slic_stop "TIMEOUT" "$t_slic"; continue
    fi
    bud=$(budget_left "$SLICSTAN_TIMEOUT" "$t_slic")
    t0=$(now); ( exec timeout -k 5 "$bud" "$CMDSTAN/bin/stanc" "$stan" --o="$W/model.hpp" ) >>"$clog" 2>&1; rc=$?; t1=$(now)
    t_stanc=$(awk "BEGIN{printf \"%.4f\", $t1-$t0}")
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      compile_s=$(awk "BEGIN{printf \"%.4f\", $t_slic+$t_stanc}")
      echo "TIMEOUT"; SLICSTAN_ROWS+=("$name|TIMEOUT|$t_slic|$t_stanc|-|-|$compile_s|stanc exceeded ${SLICSTAN_TIMEOUT}s")
      slic_stop "TIMEOUT" "$compile_s"; continue
    fi
    if [ "$rc" -ne 0 ]; then
      echo "ERROR(stanc)"; SLICSTAN_ROWS+=("$name|ERROR(stanc)|$t_slic|$t_stanc|-|-|-|generated Stan invalid")
      slic_stop "ERROR(stanc)" "-"; continue
    fi

    # Stan reads data from JSON at sample time; required iff the model declares data.
    darg=""
    if grep -qE '^[[:space:]]*data\b' "$f"; then
      if [ ! -f "$data" ]; then
        echo "NODATA"; SLICSTAN_ROWS+=("$name|NODATA|$t_slic|$t_stanc|-|-|-|no $name.json")
        slic_stop "NODATA" "-"; continue
      fi
      darg="data file=$data"
    fi

    # --- BUILD: stanc + g++ -> native executable (budget = TIMEOUT minus slic+stanc) ---
    compile_s=$(awk "BEGIN{printf \"%.4f\", $t_slic+$t_stanc}")
    if budget_done "$SLICSTAN_TIMEOUT" "$compile_s"; then
      echo "TIMEOUT"; SLICSTAN_ROWS+=("$name|TIMEOUT|$t_slic|$t_stanc|-|-|$compile_s|compile exceeded ${SLICSTAN_TIMEOUT}s")
      slic_stop "TIMEOUT" "$compile_s"; continue
    fi
    bud=$(budget_left "$SLICSTAN_TIMEOUT" "$compile_s")
    t0=$(now); ( cd "$CMDSTAN" && exec timeout -k 5 "$bud" make "$W/model" ) >>"$clog" 2>&1; rc=$?; t1=$(now)
    t_build=$(awk "BEGIN{printf \"%.4f\", $t1-$t0}")
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
      compile_s=$(awk "BEGIN{printf \"%.4f\", $t_slic+$t_stanc+$t_build}")
      echo "TIMEOUT"; SLICSTAN_ROWS+=("$name|TIMEOUT|$t_slic|$t_stanc|$t_build|-|$compile_s|build exceeded ${SLICSTAN_TIMEOUT}s")
      slic_stop "TIMEOUT" "$compile_s"; continue
    fi
    if [ "$rc" -ne 0 ] || [ ! -x "$W/model" ]; then
      echo "ERROR(build)"; SLICSTAN_ROWS+=("$name|ERROR(build)|$t_slic|$t_stanc|$t_build|-|-|C++ build failed")
      slic_stop "ERROR(build)" "-"; continue
    fi

    # --- SAMPLE: HMC/NUTS, repeated N times if the first run succeeds ---
    # Budget for the HMC step = TIMEOUT minus the one-time compile (slic+stanc+build),
    # so compile + sampling together stay within the per-program wall budget.
    compile_s=$(awk "BEGIN{printf \"%.4f\", $t_slic+$t_stanc+$t_build}")
    local samples=() slic_results=()
    status="ok"
    for (( i = 1; i <= N; i++ )); do
      if budget_done "$SLICSTAN_TIMEOUT" "$compile_s"; then status="TIMEOUT"; wall="0.0000"; break; fi
      bud=$(budget_left "$SLICSTAN_TIMEOUT" "$compile_s")
      t0=$(now)
      # distinct seed per repeat (SLICSTAN_SEED + i) so HMC repeats are independent
      # draws — otherwise every repeat is identical and per-repeat spread is meaningless.
      ( cd "$W" && exec timeout -k 5 "$bud" ./model sample \
          num_warmup="$SLICSTAN_WARMUP" num_samples="$SLICSTAN_SAMPLES" $darg \
          random seed="$((SLICSTAN_SEED + i))" output file="$W/out.csv" ) >"$W/sample.log" 2>&1
      rc=$?; t1=$(now)
      wall=$(awk "BEGIN{printf \"%.4f\", $t1-$t0}")
      if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then status="TIMEOUT"; break; fi
      if [ "$rc" -ne 0 ]; then status="ERROR(run)"; break; fi
      samples+=("$wall")
      # capture THIS repeat's posterior mean now (out.csv is overwritten next iter)
      rr=$(slicstan_result "$W/out.csv" "$qf")
      rv=$(printf '%s' "$rr" | grep -oE 'mean\([^)]*\)=[-0-9.eE+]+' | sed 's/.*=//')
      slic_results+=("${rv:--}")
    done

    if [ "$status" = "ok" ]; then
      sample_c=$(stats "${samples[@]}")                       # pure HMC sample (median±sd)
      sample_med=$(printf '%s' "$sample_c" | grep -oE '^[0-9.]+')
      total_c=$(awk "BEGIN{printf \"%.4f\", $t_slic+$t_stanc+$t_build+$sample_med}")
      # CSV infer_s for slicstan = slic (SlicStan->Stan compile) + sample (HMC), so the
      # SlicStan compilation is INCLUDED; it still EXCLUDES the stanc validate + the
      # one-time CmdStan C++ build (g++), which are CmdStan toolchain overhead, not
      # SlicStan inference. (wall_s/TOTAL still cover the whole pipeline.)
      infer_c=$(printf '%s' "$sample_c" | awk -v slic="$t_slic" -v med="$sample_med" \
        '{ sub(/^[0-9.]+/, sprintf("%.4g", med + slic)); print }')
      result=$(slicstan_result "$W/out.csv" "$qf")
    else
      sample_c="-"; infer_c="-"; total_c=$(awk "BEGIN{printf \"%.4f\", $t_slic+$t_stanc+$t_build+${wall:-0}}")
      if [ "$status" = "TIMEOUT" ]; then result="exceeded ${SLICSTAN_TIMEOUT}s"
      else result=$(grep -m1 -oE 'Exception:[^(]*|Initialization failed[^"]*' "$W/sample.log" | cut -c1-44); fi
    fi
    echo "$status"
    [ -n "$result" ] && echo "         -> $result"
    SLICSTAN_ROWS+=("$name|$status|$t_slic|$t_stanc|$t_build|$sample_c|$total_c|$result")
    # aggregate result for the CSV/.txt = MEAN of the per-repeat posterior means
    if [ "$status" = ok ] && [ "${#slic_results[@]}" -gt 0 ]; then
      slic_res=$(printf '%s\n' "${slic_results[@]}" | awk '{if($1+0==$1){s+=$1;n++}} END{if(n)printf "%.6g",s/n; else print "-"}')
    else
      slic_res="-"
    fi
    ROWS+=("slicstan|$name|$status|$([ "$status" = ok ] && echo "${#samples[@]}" || echo 0)|$total_c|$infer_c|$slic_res")
    # per-repeat CSV rows: each repeat used a distinct seed, so its posterior mean
    # (slic_results[]) genuinely varies. Per-repeat wall = compile + this repeat's HMC
    # sample; infer = slic-compile + this repeat's HMC sample. The medians of these
    # reproduce the aggregate total_c / infer_c cells (compile is one-time).
    if [ "$status" = ok ] && [ "${#samples[@]}" -gt 0 ]; then
      for (( _ri = 0; _ri < ${#samples[@]}; _ri++ )); do
        w_i=$(awk "BEGIN{printf \"%.4f\", $compile_s + ${samples[_ri]}}")
        i_i=$(awk "BEGIN{printf \"%.4g\", $t_slic + ${samples[_ri]}}")
        CSV_ROWS+=("slicstan|$name|$((REPEAT + _ri))|ok|$w_i|$i_i|${slic_results[_ri]:--}")
      done
    else
      CSV_ROWS+=("slicstan|$name|$REPEAT|$status|$total_c|-|-")
    fi
  done

  # --- slicstan component-timing breakdown ---
  echo
  echo "----- SlicStan component timing (seconds; median±sd over successful runs) -----"
  printf '%-18s %-13s %8s %8s %9s %9s %9s  %s\n' PROGRAM STATUS SLIC STANC BUILD SAMPLE TOTAL RESULT
  local row sn ss s1 s2 s3 s4 s5 sr
  for row in "${SLICSTAN_ROWS[@]}"; do
    IFS='|' read -r sn ss s1 s2 s3 s4 s5 sr <<< "$row"
    printf '%-18s %-13s %8s %8s %9s %9s %9s  %s\n' "$sn" "$ss" "$s1" "$s2" "$s3" "$s4" "$s5" "$sr"
  done
  echo "    SLIC=SlicStan->Stan compile. STANC=Stan->C++ validate. BUILD=C++ compile."
  echo "    SAMPLE=HMC inference (NUTS, ${SLICSTAN_WARMUP}+${SLICSTAN_SAMPLES} iters). TOTAL=sum."
  echo "    results_slicstan.csv: WALL=TOTAL; INFER=SLIC+SAMPLE (SlicStan compile + HMC,"
  echo "    excludes STANC+BUILD CmdStan overhead)."
}

dispatch() {
  case "$1" in
    pluck)    run_pluck ;;
    hybit)    run_hybit ;;
    psi)      run_psi ;;
    turing)   run_turing ;;
    slicstan) run_slicstan ;;
    siren)    run_siren ;;
    *) return 2 ;;
  esac
}

# ------------------------------------------------------------------ main ----
echo "Benchmark run — N=${N} runs/program, per-run timeout = ${TIMEOUT}s"
echo "run id:  $RUN_ID"
[ "$RESULT_TAG" != "$WHICH" ] && echo "tag:     $RESULT_TAG  (results_${RESULT_TAG}.csv)"
echo "run dir: $RUN_DIR"
echo "logs:    $LOGDIR"
echo "scratch: $SCRATCH"
[ "$PROGRAMS" != '*' ] && echo "programs: $PROGRAMS"
[ "$FILTER" != '*' ] && echo "filter:  $FILTER"
echo

case "$WHICH" in
  pluck|hybit|psi|turing|slicstan|siren) dispatch "$WHICH" ;;
  all)   dispatch pluck; echo; dispatch hybit; echo; dispatch psi; echo; dispatch turing; echo; dispatch slicstan; echo; dispatch siren ;;
  *) echo "usage: $0 [all|pluck|hybit|psi|turing|slicstan|siren]" >&2; exit 2 ;;
esac

# --------------------------------------------------------------- summary ----
# Results live under the run-id dir, suffixed by RESULT_TAG (= WHICH unless a sampler-budget
# sweep set a per-setting tag) so separate invocations sharing a RUN_ID — including several
# for the SAME system at different budgets — don't clobber each other.
RESULTS_TXT="$RUN_DIR/results_${RESULT_TAG}.txt"
RESULTS_CSV="$RUN_DIR/results_${RESULT_TAG}.csv"

# machine-readable: one row PER REPEAT for later analysis (raw wall/infer/result of
# each run, so any metric can be recomputed downstream — bench_common.load_run groups
# these back into one Result per program). Full per-run logs remain in $LOGDIR/*.log.
{
  echo "system,program,repeat,status,wall_s,infer_s,result"
  for row in "${CSV_ROWS[@]}"; do
    IFS='|' read -r system name repeat status wall infer result <<< "$row"
    echo "$system,$name,$repeat,$status,${wall:--},${infer:--},${result:--}"
  done
} > "$RESULTS_CSV"

# human-readable table — printed to the terminal AND saved to results.txt.
{
echo
echo "======================================== RESULTS ========================================"
printf '%-7s %-22s %-13s %5s %-16s %-18s %-14s\n' "SYSTEM" "PROGRAM" "STATUS" "RUNS" "WALL(s)" "INFER(s)" "RESULT"
printf '%-7s %-22s %-13s %5s %-16s %-18s %-14s\n' "------" "-------" "------" "----" "-------" "--------" "------"
for row in "${ROWS[@]}"; do
  IFS='|' read -r system name status runs wall infer result <<< "$row"
  printf '%-7s %-22s %-13s %5s %-16s %-18s %-14s\n' "$system" "$name" "$status" "$runs" "$wall" "$infer" "${result:--}"
done
echo "========================================================================================="
echo "RUNS     = successful runs measured (1 if the first run timed out / errored / was BAD)."
echo "WALL(s)  = median ± sample-stddev of total process wall-clock over RUNS runs."
echo "INFER(s) = median ± sample-stddev of inference-only time."
echo "           pluck: inference_seconds (via PLUCK_STATS); hybit: reported t.time."
echo "           psi:   '-' (no inference line) — use WALL(s) as the comparable number."
echo "           slicstan: WALL(s)=total pipeline (slic+stanc+build+sample); INFER(s)=slic"
echo "           +sample (SlicStan->Stan compile + HMC; EXCLUDES the stanc + C++ build that"
echo "           are CmdStan overhead). mean(param)+Rhat and the per-component breakdown are"
echo "           in the SlicStan timing table above (also logs/slicstan_<name>.log)."
echo "STATUS   = ok | TIMEOUT | OOM | ERROR(rc=N) | BAD | ERROR(run)"
echo "           | ERROR(slic) | ERROR(stanc) | ERROR(build) | NODATA."
echo "           OOM = SIGKILLed before the timeout = out of memory (the container hit its"
echo "           memory limit); WALL is the actual time it ran before dying, not a real"
echo "           inference time. Give Docker more memory to measure these."
echo "           slicstan: ERROR(slic) = .slicstan uses a feature SlicStan's DSL lacks;"
echo "           ERROR(stanc)/ERROR(build) = its generated Stan is invalid / fails to"
echo "           compile (SlicStan codegen); ERROR(run) = HMC could not initialize;"
echo "           NODATA = compiles but no slicstan-data/<name>.json."
echo "           BAD = process exited 0 but did not actually infer. For PSI this is"
echo "           'Pr[error] = 1' (posterior collapsed, e.g. observing a continuous"
echo "           variable at an exact value — measure-zero in PSI's exact semantics)."
echo "           A BAD/ERROR/TIMEOUT row's WALL is the single failing run, NOT a real"
echo "           inference time — do not compare it against ok rows."
echo
echo "logs:    $LOGDIR/<system>_<program>.log   (every run preserved)"
echo "table:   $RESULTS_TXT"
echo "csv:     $RESULTS_CSV"
} | tee "$RESULTS_TXT"
