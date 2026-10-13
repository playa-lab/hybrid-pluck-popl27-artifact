#!/usr/bin/env bash
#
# run_suite.sh — run a set of systems (and their sampler-budget sweeps) into ONE
# run directory, ready for `artifact analyze`.
#
#   experiments/run_suite.sh [SYSTEM...]          (default: all six systems)
#   TIER=smoke experiments/run_suite.sh           (settings from tiers/smoke.env)
#
# Each (system, budget) pair is one run_bench.sh invocation writing
# results_<tag>.csv, where tag is the system name or, for the approximate
# backends, the budget (turing_p<P>_i<I>, slicstan_s<S>, siren_p<P>).
#
# Env (a tier file may set any of these; explicit env wins over the tier file):
#   RUN_ID          run directory name           (default <tier>_<timestamp>)
#   SUITE_SYSTEMS   systems run when none are given (default all six)
#   N, TIMEOUT, PROGRAMS, FILTER                 (passed through to run_bench.sh)
#   N_<SYS>, TIMEOUT_<SYS>, PROGRAMS_<SYS>, FILTER_<SYS>
#                   per-system overrides, e.g. TIMEOUT_PSI=120. Tiers choose programs
#                   with PROGRAMS_<SYS> (e.g. '@table3'), which leaves FILTER free for
#                   narrowing a run, e.g. TIER=paper FILTER=sir_model_n72.
#   TURING_SWEEP    particles:iters pairs        (default '32:500')
#   SLICSTAN_SWEEP  HMC sample counts            (default '500')
#   SIREN_SWEEP     SMC particle counts          (default '100')
#   JOBS            (system, budget) jobs run in parallel (default 1). Every job is
#                   single-threaded; JOBS>1 is faster but adds timing noise.
#
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- internal: run one "sys|tag|KEY=VAL ..." spec (called back through xargs) ---
if [ "${1:-}" = "--one" ]; then
  IFS='|' read -r sys tag envs <<< "$2"
  up=$(printf '%s' "$sys" | tr '[:lower:]' '[:upper:]')
  n_var="N_$up"; t_var="TIMEOUT_$up"; p_var="PROGRAMS_$up"; f_var="FILTER_$up"
  export N="${!n_var:-${N:-5}}" TIMEOUT="${!t_var:-${TIMEOUT:-60}}"
  export PROGRAMS="${!p_var:-${PROGRAMS:-*}}" FILTER="${!f_var:-${FILTER:-*}}"
  export RESULT_TAG="$tag"
  for kv in $envs; do export "$kv"; done
  mkdir -p "$BENCH_OUT/$RUN_ID"
  echo "[start] $tag (N=$N TIMEOUT=$TIMEOUT PROGRAMS='$PROGRAMS' FILTER='$FILTER')"
  bash "$SCRIPT_DIR/run_bench.sh" "$sys" > "$BENCH_OUT/$RUN_ID/suite_$tag.out" 2>&1
  echo "[done]  $tag (exit $?)  -> suite_$tag.out"
  exit 0
fi

# Explicit env overrides the tier file: remember what was set before sourcing it.
_explicit=$(env | grep -E '^(RUN_ID|N|TIMEOUT|PROGRAMS|FILTER|JOBS|[A-Z]+_SWEEP|(N|TIMEOUT|PROGRAMS|FILTER)_[A-Z]+)=' || true)
if [ -n "${TIER:-}" ]; then
  [ -f "$SCRIPT_DIR/tiers/$TIER.env" ] || { echo "unknown TIER '$TIER' (see $SCRIPT_DIR/tiers/)" >&2; exit 2; }
  set -a; . "$SCRIPT_DIR/tiers/$TIER.env"; set +a
  while IFS= read -r kv; do [ -n "$kv" ] && export "$kv"; done <<< "$_explicit"
fi

SYSTEMS=("$@")
[ ${#SYSTEMS[@]} -eq 0 ] && read -r -a SYSTEMS <<< "${SUITE_SYSTEMS:-pluck psi hybit turing slicstan siren}"

export RUN_ID="${RUN_ID:-${TIER:-run}_$(date +%Y%m%d_%H%M%S)}"
export BENCH_OUT="${BENCH_OUT:-$([ -d /results ] && echo /results || echo "$SCRIPT_DIR/../results")}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 JULIA_NUM_THREADS=1
RUN_DIR="$BENCH_OUT/$RUN_ID"
mkdir -p "$RUN_DIR" || exit 1

specs=()
for sys in "${SYSTEMS[@]}"; do
  case "$sys" in
    turing)   for pair in ${TURING_SWEEP:-32:500}; do
                p="${pair%%:*}"; i="${pair##*:}"
                specs+=("turing|turing_p${p}_i${i}|TURING_PARTICLES=$p TURING_ITERS=$i")
              done ;;
    slicstan) for s in ${SLICSTAN_SWEEP:-500}; do specs+=("slicstan|slicstan_s${s}|SLICSTAN_SAMPLES=$s"); done ;;
    siren)    for p in ${SIREN_SWEEP:-100}; do specs+=("siren|siren_p${p}|SIREN_PARTICLES=$p"); done ;;
    pluck|psi|hybit) specs+=("$sys|$sys|") ;;
    *) echo "unknown system '$sys'" >&2; exit 2 ;;
  esac
done

# Record the configuration next to the results.
{
  echo "tier=${TIER:-}"; echo "systems=${SYSTEMS[*]}"; echo "jobs=${JOBS:-1}"
  env | grep -E '^(N|TIMEOUT|PROGRAMS|FILTER|[A-Z]+_SWEEP|(N|TIMEOUT|PROGRAMS|FILTER)_[A-Z]+)=' | sort
  echo "arch=$(uname -m)"; echo "nproc=$(getconf _NPROCESSORS_ONLN)"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$RUN_DIR/suite.env"

echo "run dir: $RUN_DIR"
echo "jobs:    ${#specs[@]} (${JOBS:-1} at a time)"
printf '%s\n' "${specs[@]}" | tr '\n' '\0' \
  | xargs -0 -P "${JOBS:-1}" -I{} bash "$SCRIPT_DIR/run_suite.sh" --one {}
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RUN_DIR/suite.env"
echo "all jobs finished: $RUN_DIR"
